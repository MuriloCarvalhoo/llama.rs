//! Backend de decode 100% na GPU: todas as ativações e o KV-cache residentes em VRAM.
//! Só os logits finais voltam ao host. Cada op é 1 dispatch + 1 wait nesta fatia (1C);
//! a fusão em 1 command buffer/token é a Fase 1D.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::matmul::MatmulError;
use crate::pipeline::ComputePipeline;
use crate::tensor::GpuTensor;
use ash::vk;
use llama_model::{GpuAuxWeights, GpuRawWeights, LlamaConfig};
use std::cell::RefCell;

/// Linhas de saida processadas por cada workgroup do matvec Q8_0 (tunavel — varredura no Step 5).
pub(crate) const MATVEC_NUM_ROWS: u32 = 2;
/// local_size_x do matvec (wave64 no MI50).
pub(crate) const MATVEC_WG: u32 = 64;

/// Buffer Vulkan simples (device-local ou host-visible) com tamanho conhecido.
pub(crate) struct Buf {
    pub buffer: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Buf {
    /// Buffer device-local STORAGE | TRANSFER_SRC | TRANSFER_DST de `bytes`.
    fn device(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind, create_buf};
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind(ctx, phys, d, buffer, false)?;
        Ok(Self {
            buffer,
            mem,
            size: bytes,
        })
    }

    /// Buffer host-visible TRANSFER_SRC | TRANSFER_DST de `bytes` (upload/readback).
    fn host(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind, create_buf};
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind(ctx, phys, d, buffer, true)?;
        Ok(Self {
            buffer,
            mem,
            size: bytes,
        })
    }

    fn destroy(&self, d: &ash::Device) {
        // SAFETY: handles criados por nós; chamado no Drop, sem uso concorrente.
        unsafe {
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.mem, None);
        }
    }
}

/// Escalares de arquitetura necessários ao decode.
pub(crate) struct Cfg {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rope_dim: usize,
    pub kv_dim: usize,
    pub vocab: usize,
    pub ctx: usize,
    pub rms_eps: f32,
}

/// Pesos Q8_0 residentes de uma camada.
pub(crate) struct LayerQ {
    pub attn_q: GpuTensor,
    pub attn_k: GpuTensor,
    pub attn_v: GpuTensor,
    pub attn_output: GpuTensor,
    pub ffn_gate: GpuTensor,
    pub ffn_up: GpuTensor,
    pub ffn_down: GpuTensor,
}

/// Buffers f32 auxiliares residentes de uma camada.
pub(crate) struct LayerAux {
    pub attn_norm: Buf,
    pub ffn_norm: Buf,
    pub q_bias: Option<Buf>,
    pub k_bias: Option<Buf>,
    pub v_bias: Option<Buf>,
}

/// Identifica qual pipeline um dispatch usa (resolvido em `pipe_of`).
#[derive(Clone, Copy)]
pub(crate) enum PipeId {
    Matvec,
    Rmsnorm,
    Rope,
    Attention,
    Swiglu,
    Add,
}

/// Como obter os bytes de push-constant de um dispatch no momento da gravação.
pub(crate) enum PushSpec {
    /// Push totalmente conhecido na construção do plano.
    Static(Vec<u8>),
    /// RoPE: precisa de `pos` na gravação. `n_head` fixo.
    Rope { n_head: u32 },
    /// Attention: precisa de `total_len`. `kv_layer_off` fixo.
    Attention { kv_layer_off: u32 },
}

/// Uma op do token. `Dispatch` usa um descriptor set pré-escrito; as cópias não.
pub(crate) enum PlannedOp {
    Dispatch {
        pipe: PipeId,
        set: vk::DescriptorSet,
        groups: u32,
        push: PushSpec,
    },
    /// Embedding lookup: copia a linha `token` de `token_embd` para `b_x`.
    Embed,
    /// Append do K e do V da camada ao KV-cache na posição `pos`.
    KvAppend { layer: usize },
}

/// Todo o estado residente do modelo (pesos + aux + KV + ativações). `None` no
/// construtor de micro-teste `new_pipelines_only`; `Some` após `new`.
pub(crate) struct ResidentState {
    pub cfg: Cfg,
    pub qw: Vec<LayerQ>,
    pub output_w: GpuTensor,
    pub aux: Vec<LayerAux>,
    pub output_norm_buf: Buf,
    pub freq_buf: Buf,
    /// Tabela de embedding f32 no host. Manter em VRAM custaria vocab*n_embd*4
    /// (3.1 GB no 14B) para ler **uma** linha por token; a linha (n_embd f32,
    /// ~20 KB) sobe por `embd_stage` a cada passo, ao custo de poucos µs.
    pub token_embd: Vec<f32>,
    pub embd_stage: Buf,
    pub kcache: Buf,
    pub vcache: Buf,
    pub b_x: Buf,
    pub b_normed: Buf,
    pub b_q: Buf,
    pub b_k: Buf,
    pub b_v: Buf,
    pub b_attn: Buf,
    pub b_proj: Buf,
    pub b_gate: Buf,
    pub b_up: Buf,
    pub b_act: Buf,
    pub b_logits: Buf,
    pub len: RefCell<usize>,
    pub plan: Vec<PlannedOp>,
    pub token_cmd: vk::CommandBuffer,
    pub token_fence: vk::Fence,
}

/// Backend de decode GPU-resident (1 GPU). Construído via `ResidentForward::new`.
pub struct ResidentForward<'ctx> {
    pub(crate) ctx: &'ctx VulkanContext,
    pub(crate) phys_idx: usize,
    pub(crate) dev: VulkanDevice,
    // pipelines (preenchidos na Task 9; campos públicos ao crate para as tasks de teste)
    pub(crate) matvec: ComputePipeline,
    pub(crate) rmsnorm: ComputePipeline,
    pub(crate) rope: ComputePipeline,
    pub(crate) attention: ComputePipeline,
    pub(crate) swiglu: ComputePipeline,
    pub(crate) add: ComputePipeline,
    pub(crate) desc_pool: vk::DescriptorPool,
    pub(crate) state: Option<ResidentState>,
}

impl<'ctx> ResidentForward<'ctx> {
    pub(crate) fn phys(&self) -> &VulkanPhysicalDevice {
        &self.ctx.amd_compute_devices()[self.phys_idx]
    }

    /// Aloca um descriptor set do pool com o layout da pipeline dada.
    pub(crate) fn alloc_set(
        &self,
        pipe: &ComputePipeline,
    ) -> Result<vk::DescriptorSet, MatmulError> {
        let d = &self.dev.device;
        let info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &pipe.desc_set_layout,
            ..Default::default()
        };
        // SAFETY: d e pool válidos; layout vem da pipeline.
        Ok(unsafe { d.allocate_descriptor_sets(&info)? }[0])
    }

    /// Escreve `bindings` no `set` (na ordem 0..n) e despacha `pipe` com `push` bytes e
    /// `groups` workgroups em x. Faz 1 submit + 1 wait. Não lê de volta (dados ficam residentes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch1(
        &self,
        pipe: &ComputePipeline,
        set: vk::DescriptorSet,
        bindings: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)], // (buffer, offset, range)
        push: &[u8],
        groups: u32,
    ) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        let dev = &self.dev;

        let buf_infos: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .map(|&(buffer, offset, range)| vk::DescriptorBufferInfo {
                buffer,
                offset,
                range,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(b, bi)| vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: b as u32,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: bi,
                ..Default::default()
            })
            .collect();
        // SAFETY: d válido; writes apontam para buf_infos vivos; GPU ociosa (wait por op).
        unsafe { d.update_descriptor_sets(&writes, &[]) };

        let cb_info = vk::CommandBufferAllocateInfo {
            command_pool: dev.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: d e cmd_pool válidos.
        let cmd = unsafe { d.allocate_command_buffers(&cb_info)? }[0];
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            // SAFETY: cmd recém-alocado; pipe/set válidos; push tem o tamanho do range do layout.
            d.begin_command_buffer(cmd, &begin)?;
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                d.cmd_push_constants(cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            d.cmd_dispatch(cmd, groups, 1, 1);
            d.end_command_buffer(cmd)?;
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        unsafe {
            // SAFETY: queue/submit/cmd válidos.
            d.queue_submit(dev.queue, &[submit], vk::Fence::null())?;
            d.queue_wait_idle(dev.queue)?;
            d.free_command_buffers(dev.cmd_pool, &[cmd]);
        }
        Ok(())
    }

    /// Sobe `data` (f32) para o `dst` device-local via um staging host-visible próprio.
    pub(crate) fn upload_f32(&self, dst: &Buf, data: &[f32]) -> Result<(), MatmulError> {
        use crate::tensor::one_shot_copy;
        let d = &self.dev.device;
        let bytes = std::mem::size_of_val(data) as vk::DeviceSize;
        let staging = Buf::host(self.ctx, self.phys(), d, bytes)?;
        unsafe {
            // SAFETY: staging host-visible com `bytes`; ptr válido até unmap.
            let ptr = d.map_memory(staging.mem, 0, bytes, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len());
            d.unmap_memory(staging.mem);
        }
        one_shot_copy(
            d,
            self.dev.queue,
            self.dev.cmd_pool,
            staging.buffer,
            dst.buffer,
            bytes,
        )?;
        staging.destroy(d);
        Ok(())
    }

    /// Lê `len` f32 do `src` device-local de volta ao host.
    pub(crate) fn readback(&self, src: &Buf, len: usize) -> Result<Vec<f32>, MatmulError> {
        use crate::tensor::one_shot_copy;
        let d = &self.dev.device;
        let bytes = (len * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let host = Buf::host(self.ctx, self.phys(), d, bytes)?;
        one_shot_copy(
            d,
            self.dev.queue,
            self.dev.cmd_pool,
            src.buffer,
            host.buffer,
            bytes,
        )?;
        let out = unsafe {
            // SAFETY: host host-visible com `bytes`; ptr válido até unmap.
            let ptr = d.map_memory(host.mem, 0, bytes, vk::MemoryMapFlags::empty())?;
            let mut v = vec![0f32; len];
            std::ptr::copy_nonoverlapping(ptr as *const f32, v.as_mut_ptr(), len);
            d.unmap_memory(host.mem);
            v
        };
        host.destroy(d);
        Ok(out)
    }

    /// Constrói só device + pipelines + descriptor pool (sem pesos/buffers). Para micro-testes.
    pub fn new_pipelines_only(ctx: &'ctx VulkanContext) -> Result<Self, MatmulError> {
        let phys = ctx.amd_compute_devices();
        if phys.is_empty() {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED));
        }
        let dev = VulkanDevice::create(ctx, &phys[0])?;
        let d = &dev.device;
        let matvec = ComputePipeline::new(d)?;
        let rmsnorm = ComputePipeline::with(d, crate::RMSNORM_SPV, 3, 8, &[])?; // dim:u32 + eps:f32
        let rope = ComputePipeline::with(d, crate::ROPE_SPV, 2, 16, &[])?;
        let attention = ComputePipeline::with(d, crate::ATTENTION_SPV, 4, 24, &[])?;
        let swiglu = ComputePipeline::with(d, crate::SWIGLU_SPV, 3, 4, &[])?;
        let add = ComputePipeline::with(d, crate::ADD_SPV, 2, 4, &[])?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4096,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1024,
            pool_size_count: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        // SAFETY: d válido; pool_info aponta para dados vivos.
        let desc_pool = unsafe { d.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            ctx,
            phys_idx: 0,
            dev,
            matvec,
            rmsnorm,
            rope,
            attention,
            swiglu,
            add,
            desc_pool,
            state: None,
        })
    }

    /// Constrói o backend GPU-resident: sobe todos os pesos (Q8_0 + aux f32) e aloca
    /// as ativações e o KV-cache em VRAM. Após retornar, `raw`/`aux` podem ser descartados.
    pub fn new(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'_>,
    ) -> Result<Self, MatmulError> {
        if config.head_dim > 256 {
            // Shader de attention distribui head_dim entre 64 lanes com no máximo
            // MAX_DPL=4 dimensões por lane.
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        // O matvec faz tiling da dimensão K em janelas de MATVEC_MAX_BLOCKS blocos,
        // então n_in é livre. (Antes havia um limite de n_in <= MATVEC_MAX_BLOCKS*32.)
        let mut me = Self::new_pipelines_only(ctx)?;
        let kv_dim = config.n_head_kv * config.head_dim;

        // Bloco que constrói todo o estado residente emprestando `me` imutavelmente;
        // ao final o `state` é movido para fora (sem borrows de `me`), e só então
        // `me.state = Some(state)` é atribuído.
        let state = {
            let phys = me.phys();
            let dev_ref = &me.dev;
            let d = &dev_ref.device;

            let cfg = Cfg {
                n_embd: config.n_embd,
                n_layer: config.n_layer,
                n_head: config.n_head,
                n_head_kv: config.n_head_kv,
                head_dim: config.head_dim,
                n_ff: config.n_ff,
                rope_dim: config.rope_dim,
                kv_dim,
                vocab: config.vocab,
                ctx: config.ctx,
                rms_eps: config.rms_eps,
            };

            let up_q =
                |bytes: &[u8], n_in: usize, n_out: usize| -> Result<GpuTensor, MatmulError> {
                    GpuTensor::upload_q8_0(ctx, phys, dev_ref, bytes, n_in, n_out)
                        .map_err(MatmulError::from)
                };
            let mut qw = Vec::with_capacity(cfg.n_layer);
            for lw in &raw.layers {
                qw.push(LayerQ {
                    attn_q: up_q(&lw.attn_q, cfg.n_embd, cfg.n_embd)?,
                    attn_k: up_q(&lw.attn_k, cfg.n_embd, kv_dim)?,
                    attn_v: up_q(&lw.attn_v, cfg.n_embd, kv_dim)?,
                    attn_output: up_q(&lw.attn_output, cfg.n_embd, cfg.n_embd)?,
                    ffn_gate: up_q(&lw.ffn_gate, cfg.n_embd, cfg.n_ff)?,
                    ffn_up: up_q(&lw.ffn_up, cfg.n_embd, cfg.n_ff)?,
                    ffn_down: up_q(&lw.ffn_down, cfg.n_ff, cfg.n_embd)?,
                });
            }
            let output_w = up_q(&raw.output, cfg.n_embd, cfg.vocab)?;

            let mk = |data: &[f32]| -> Result<Buf, MatmulError> {
                let b = Buf::device(ctx, phys, d, std::mem::size_of_val(data) as vk::DeviceSize)?;
                let bytes_val = std::mem::size_of_val(data) as vk::DeviceSize;
                let staging = Buf::host(ctx, phys, d, bytes_val)?;
                unsafe {
                    let ptr =
                        d.map_memory(staging.mem, 0, bytes_val, vk::MemoryMapFlags::empty())?;
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        ptr as *mut u8,
                        bytes_val as usize,
                    );
                    d.unmap_memory(staging.mem);
                }
                use crate::tensor::one_shot_copy;
                let res = one_shot_copy(
                    d,
                    dev_ref.queue,
                    dev_ref.cmd_pool,
                    staging.buffer,
                    b.buffer,
                    bytes_val,
                );
                staging.destroy(d);
                res?;
                Ok(b)
            };
            let mk_opt = |o: &Option<Vec<f32>>| -> Result<Option<Buf>, MatmulError> {
                match o {
                    Some(v) => Ok(Some(mk(v)?)),
                    None => Ok(None),
                }
            };
            let mut aux_buf = Vec::with_capacity(cfg.n_layer);
            for al in &aux.layers {
                aux_buf.push(LayerAux {
                    attn_norm: mk(&al.attn_norm)?,
                    ffn_norm: mk(&al.ffn_norm)?,
                    q_bias: mk_opt(&al.q_bias)?,
                    k_bias: mk_opt(&al.k_bias)?,
                    v_bias: mk_opt(&al.v_bias)?,
                });
            }
            let output_norm_buf = mk(&aux.output_norm)?;
            let freq_buf = mk(&aux.freq_table)?;
            let embd_stage = Buf::host(ctx, phys, d, (config.n_embd * 4) as vk::DeviceSize)?;

            let kv_elems = (cfg.n_layer * cfg.ctx * kv_dim) as vk::DeviceSize;
            let kcache = Buf::device(ctx, phys, d, kv_elems * 4)?;
            let vcache = Buf::device(ctx, phys, d, kv_elems * 4)?;

            let nf = |n: usize| -> Result<Buf, MatmulError> {
                Buf::device(ctx, phys, d, (n * 4) as vk::DeviceSize)
            };

            let cb_info = vk::CommandBufferAllocateInfo {
                command_pool: dev_ref.cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            // SAFETY: device/pool válidos; pool tem RESET_COMMAND_BUFFER.
            let token_cmd = unsafe { dev_ref.device.allocate_command_buffers(&cb_info)? }[0];
            let token_fence = unsafe {
                dev_ref
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?
            };

            ResidentState {
                cfg,
                qw,
                output_w,
                aux: aux_buf,
                output_norm_buf,
                freq_buf,
                token_embd: aux.token_embd.to_vec(),
                embd_stage,
                kcache,
                vcache,
                b_x: nf(config.n_embd)?,
                b_normed: nf(config.n_embd)?,
                b_q: nf(config.n_embd)?,
                b_k: nf(kv_dim)?,
                b_v: nf(kv_dim)?,
                b_attn: nf(config.n_embd)?,
                b_proj: nf(config.n_embd)?,
                b_gate: nf(config.n_ff)?,
                b_up: nf(config.n_ff)?,
                b_act: nf(config.n_ff)?,
                b_logits: nf(config.vocab)?,
                len: RefCell::new(0),
                plan: Vec::new(),
                token_cmd,
                token_fence,
            }
        };

        me.state = Some(state);
        let plan = me.build_plan()?;
        if let Some(st) = me.state.as_mut() {
            st.plan = plan;
        }
        Ok(me)
    }

    /// Diagnóstico: roda o shader attention GQA sobre q/k_cache/v_cache e devolve o resultado.
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        total_len: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        if head_dim > 64 {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        #[repr(C)]
        struct P {
            n_head: u32,
            n_head_kv: u32,
            head_dim: u32,
            total_len: u32,
            kv_dim: u32,
            kv_layer_off: u32,
        }
        let d = &self.dev.device;
        let kv_dim = n_head_kv * head_dim;
        let qb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(q) as vk::DeviceSize,
        )?;
        let kb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(k_cache) as vk::DeviceSize,
        )?;
        let vb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(v_cache) as vk::DeviceSize,
        )?;
        let ob = Buf::device(
            self.ctx,
            self.phys(),
            d,
            (n_head * head_dim * 4) as vk::DeviceSize,
        )?;
        self.upload_f32(&qb, q)?;
        self.upload_f32(&kb, k_cache)?;
        self.upload_f32(&vb, v_cache)?;
        let set = self.alloc_set(&self.attention)?;
        let push = P {
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            head_dim: head_dim as u32,
            total_len: total_len as u32,
            kv_dim: kv_dim as u32,
            kv_layer_off: 0,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 24) };
        self.dispatch1(
            &self.attention,
            set,
            &[
                (qb.buffer, 0, qb.size),
                (kb.buffer, 0, kb.size),
                (vb.buffer, 0, vb.size),
                (ob.buffer, 0, ob.size),
            ],
            pb,
            n_head as u32, // 1 workgroup por head
        )?;
        let out = self.readback(&ob, n_head * head_dim)?;
        qb.destroy(d);
        kb.destroy(d);
        vb.destroy(d);
        ob.destroy(d);
        Ok(out)
    }

    /// nº de workgroups para cobrir `n` elementos com local_size_x=64.
    pub(crate) fn groups_for(n: usize) -> u32 {
        ((n + 63) / 64) as u32
    }

    pub fn dbg_swiglu(&self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P {
            n: u32,
        }
        let d = &self.dev.device;
        let n = g.len();
        let gb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(g) as vk::DeviceSize,
        )?;
        let ub = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(u) as vk::DeviceSize,
        )?;
        let ob = Buf::device(self.ctx, self.phys(), d, (n * 4) as vk::DeviceSize)?;
        self.upload_f32(&gb, g)?;
        self.upload_f32(&ub, u)?;
        let set = self.alloc_set(&self.swiglu)?;
        let push = P { n: n as u32 };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 4) };
        self.dispatch1(
            &self.swiglu,
            set,
            &[
                (gb.buffer, 0, gb.size),
                (ub.buffer, 0, ub.size),
                (ob.buffer, 0, ob.size),
            ],
            pb,
            Self::groups_for(n),
        )?;
        let out = self.readback(&ob, n)?;
        gb.destroy(d);
        ub.destroy(d);
        ob.destroy(d);
        Ok(out)
    }

    pub fn dbg_add(&self, dst: &[f32], src: &[f32]) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P {
            n: u32,
        }
        let d = &self.dev.device;
        let n = dst.len();
        let db = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(dst) as vk::DeviceSize,
        )?;
        let sb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(src) as vk::DeviceSize,
        )?;
        self.upload_f32(&db, dst)?;
        self.upload_f32(&sb, src)?;
        let set = self.alloc_set(&self.add)?;
        let push = P { n: n as u32 };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 4) };
        self.dispatch1(
            &self.add,
            set,
            &[(db.buffer, 0, db.size), (sb.buffer, 0, sb.size)],
            pb,
            Self::groups_for(n),
        )?;
        let out = self.readback(&db, n)?; // dst foi mutado in-place
        db.destroy(d);
        sb.destroy(d);
        Ok(out)
    }

    /// Diagnóstico: roda só o shader rmsnorm sobre `x`,`w` e devolve a saída ao host.
    pub fn dbg_rmsnorm(&self, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P {
            dim: u32,
            eps: f32,
        }
        let d = &self.dev.device;
        let dim = x.len();
        let xb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(x) as vk::DeviceSize,
        )?;
        let wb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(w) as vk::DeviceSize,
        )?;
        let ob = Buf::device(self.ctx, self.phys(), d, (dim * 4) as vk::DeviceSize)?;
        self.upload_f32(&xb, x)?;
        self.upload_f32(&wb, w)?;
        let set = self.alloc_set(&self.rmsnorm)?;
        let push = P {
            dim: dim as u32,
            eps,
        };
        let push_bytes = unsafe {
            std::slice::from_raw_parts(&push as *const P as *const u8, std::mem::size_of::<P>())
        };
        self.dispatch1(
            &self.rmsnorm,
            set,
            &[
                (xb.buffer, 0, xb.size),
                (wb.buffer, 0, wb.size),
                (ob.buffer, 0, ob.size),
            ],
            push_bytes,
            1, // 1 workgroup processa a linha
        )?;
        let out = self.readback(&ob, dim)?;
        xb.destroy(d);
        wb.destroy(d);
        ob.destroy(d);
        Ok(out)
    }

    /// Diagnóstico: roda o shader rope in-place sobre `x` e devolve o resultado ao host.
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_rope(
        &self,
        x: &mut [f32],
        n_head: usize,
        head_dim: usize,
        rope_dim: usize,
        freq: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P {
            n_head: u32,
            head_dim: u32,
            rope_dim: u32,
            pos: f32,
        }
        let d = &self.dev.device;
        let xb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(x) as vk::DeviceSize,
        )?;
        let fb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(freq) as vk::DeviceSize,
        )?;
        self.upload_f32(&xb, x)?;
        self.upload_f32(&fb, freq)?;
        let set = self.alloc_set(&self.rope)?;
        let push = P {
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            pos: pos as f32,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 16) };
        let pairs = n_head * (rope_dim / 2);
        self.dispatch1(
            &self.rope,
            set,
            &[(xb.buffer, 0, xb.size), (fb.buffer, 0, fb.size)],
            pb,
            Self::groups_for(pairs),
        )?;
        let out = self.readback(&xb, x.len())?;
        xb.destroy(d);
        fb.destroy(d);
        Ok(out)
    }

    /// Barreira de memória global entre dispatches/cópias do mesmo command buffer.
    fn full_barrier(&self, cmd: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            ..Default::default()
        };
        let stage = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        // SAFETY: cmd em gravação; barreira global sem buffer/image barriers.
        unsafe {
            self.dev.device.cmd_pipeline_barrier(
                cmd,
                stage,
                stage,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );
        }
    }

    /// Grava a stack inteira do token em `cmd` (já em `begin`). Push/offsets dependem de token/pos.
    fn record_token(&self, cmd: vk::CommandBuffer, token: u32, pos: usize) {
        let d = &self.dev.device;
        let st = self
            .state
            .as_ref()
            .expect("record_token requer state (new())");
        let c = &st.cfg;
        let total_len = (pos + 1) as u32;

        for op in &st.plan {
            match op {
                PlannedOp::Embed => {
                    // Copia a linha do token da tabela host para o staging e daí para b_x.
                    let row = token as usize * c.n_embd;
                    let bytes = (c.n_embd * 4) as vk::DeviceSize;
                    if let Some(src) = st.token_embd.get(row..row + c.n_embd) {
                        // SAFETY: embd_stage é host-visible/coherent com `bytes`;
                        // o ponteiro é válido até unmap e a cópia respeita n_embd floats.
                        unsafe {
                            if let Ok(ptr) = d.map_memory(
                                st.embd_stage.mem,
                                0,
                                bytes,
                                vk::MemoryMapFlags::empty(),
                            ) {
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr(),
                                    ptr as *mut f32,
                                    c.n_embd,
                                );
                                d.unmap_memory(st.embd_stage.mem);
                            }
                        }
                    }
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: bytes,
                    };
                    // SAFETY: cmd em gravação; buffers vivos; offsets dentro do tamanho.
                    unsafe {
                        d.cmd_copy_buffer(cmd, st.embd_stage.buffer, st.b_x.buffer, &[region]);
                    }
                    self.full_barrier(cmd);
                }
                PlannedOp::KvAppend { layer } => {
                    let off = ((layer * c.ctx + pos) * c.kv_dim * 4) as vk::DeviceSize;
                    let sz = (c.kv_dim * 4) as vk::DeviceSize;
                    let rk = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: off,
                        size: sz,
                    };
                    // SAFETY: idem.
                    unsafe {
                        d.cmd_copy_buffer(cmd, st.b_k.buffer, st.kcache.buffer, &[rk]);
                        d.cmd_copy_buffer(cmd, st.b_v.buffer, st.vcache.buffer, &[rk]);
                    }
                    self.full_barrier(cmd);
                }
                PlannedOp::Dispatch {
                    pipe,
                    set,
                    groups,
                    push,
                } => {
                    let p = self.pipe_of(*pipe);
                    let bytes: Vec<u8> = match push {
                        PushSpec::Static(b) => b.clone(),
                        PushSpec::Rope { n_head } => {
                            #[repr(C)]
                            struct P {
                                n_head: u32,
                                head_dim: u32,
                                rope_dim: u32,
                                pos: f32,
                            }
                            let pp = P {
                                n_head: *n_head,
                                head_dim: c.head_dim as u32,
                                rope_dim: c.rope_dim as u32,
                                pos: pos as f32,
                            };
                            unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 16) }
                                .to_vec()
                        }
                        PushSpec::Attention { kv_layer_off } => {
                            #[repr(C)]
                            struct P {
                                n_head: u32,
                                n_head_kv: u32,
                                head_dim: u32,
                                total_len: u32,
                                kv_dim: u32,
                                kv_layer_off: u32,
                            }
                            let pp = P {
                                n_head: c.n_head as u32,
                                n_head_kv: c.n_head_kv as u32,
                                head_dim: c.head_dim as u32,
                                total_len,
                                kv_dim: c.kv_dim as u32,
                                kv_layer_off: *kv_layer_off,
                            };
                            unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 24) }
                                .to_vec()
                        }
                    };
                    // SAFETY: cmd em gravação; pipeline/set válidos; bytes do tamanho do range.
                    unsafe {
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
                        d.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            p.layout,
                            0,
                            &[*set],
                            &[],
                        );
                        d.cmd_push_constants(
                            cmd,
                            p.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            &bytes,
                        );
                        d.cmd_dispatch(cmd, *groups, 1, 1);
                    }
                    self.full_barrier(cmd);
                }
            }
        }
    }

    pub(crate) fn pipe_of(&self, id: PipeId) -> &ComputePipeline {
        match id {
            PipeId::Matvec => &self.matvec,
            PipeId::Rmsnorm => &self.rmsnorm,
            PipeId::Rope => &self.rope,
            PipeId::Attention => &self.attention,
            PipeId::Swiglu => &self.swiglu,
            PipeId::Add => &self.add,
        }
    }

    /// Monta a lista de ops do token (ordem idêntica a `decode_step`) e pré-aloca/escreve
    /// um descriptor set por dispatch (bindings estáticos entre tokens).
    fn build_plan(&self) -> Result<Vec<PlannedOp>, MatmulError> {
        use crate::pipeline::PushConstants;
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let c = &st.cfg;
        let d = &self.dev.device;
        let mut plan = Vec::new();

        let rms_push = || -> Vec<u8> {
            #[repr(C)]
            struct P {
                dim: u32,
                eps: f32,
            }
            let p = P {
                dim: c.n_embd as u32,
                eps: c.rms_eps,
            };
            unsafe { std::slice::from_raw_parts(&p as *const P as *const u8, 8) }.to_vec()
        };
        let n_push = |n: usize| -> Vec<u8> {
            #[repr(C)]
            struct P {
                n: u32,
            }
            let p = P { n: n as u32 };
            unsafe { std::slice::from_raw_parts(&p as *const P as *const u8, 4) }.to_vec()
        };
        let mv_push = |n_in: usize, n_out: usize| -> Vec<u8> {
            let p = PushConstants {
                n_in: n_in as u32,
                n_out: n_out as u32,
                row_offset: 0,
            };
            unsafe {
                std::slice::from_raw_parts(
                    &p as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                )
            }
            .to_vec()
        };
        let mv_groups = |n_out: usize| -> u32 { (n_out as u32).div_ceil(MATVEC_NUM_ROWS) };

        let mk = |pipe: PipeId,
                  binds: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
                  groups: u32,
                  push: PushSpec|
         -> Result<PlannedOp, MatmulError> {
            let set = self.alloc_set(self.pipe_of(pipe))?;
            let buf_infos: Vec<vk::DescriptorBufferInfo> = binds
                .iter()
                .map(|&(buffer, offset, range)| vk::DescriptorBufferInfo {
                    buffer,
                    offset,
                    range,
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(b, bi)| vk::WriteDescriptorSet {
                    dst_set: set,
                    dst_binding: b as u32,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: bi,
                    ..Default::default()
                })
                .collect();
            // SAFETY: d válido; writes apontam para buf_infos vivos; set nunca em uso durante o build.
            unsafe { d.update_descriptor_sets(&writes, &[]) };
            Ok(PlannedOp::Dispatch {
                pipe,
                set,
                groups,
                push,
            })
        };

        let nb = |n: usize| (n * 4) as vk::DeviceSize;

        plan.push(PlannedOp::Embed);

        for l in 0..c.n_layer {
            let lq = &st.qw[l];
            let la = &st.aux[l];

            plan.push(mk(
                PipeId::Rmsnorm,
                &[
                    (st.b_x.buffer, 0, nb(c.n_embd)),
                    (la.attn_norm.buffer, 0, la.attn_norm.size),
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                ],
                1,
                PushSpec::Static(rms_push()),
            )?);
            plan.push(mk(
                PipeId::Matvec,
                &[
                    (lq.attn_q.buffer, 0, lq.attn_q.size_bytes),
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_q.buffer, 0, nb(c.n_embd)),
                ],
                mv_groups(c.n_embd),
                PushSpec::Static(mv_push(c.n_embd, c.n_embd)),
            )?);
            plan.push(mk(
                PipeId::Matvec,
                &[
                    (lq.attn_k.buffer, 0, lq.attn_k.size_bytes),
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_k.buffer, 0, nb(c.kv_dim)),
                ],
                mv_groups(c.kv_dim),
                PushSpec::Static(mv_push(c.n_embd, c.kv_dim)),
            )?);
            plan.push(mk(
                PipeId::Matvec,
                &[
                    (lq.attn_v.buffer, 0, lq.attn_v.size_bytes),
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_v.buffer, 0, nb(c.kv_dim)),
                ],
                mv_groups(c.kv_dim),
                PushSpec::Static(mv_push(c.n_embd, c.kv_dim)),
            )?);
            if let Some(b) = &la.q_bias {
                plan.push(mk(
                    PipeId::Add,
                    &[(st.b_q.buffer, 0, nb(c.n_embd)), (b.buffer, 0, b.size)],
                    Self::groups_for(c.n_embd),
                    PushSpec::Static(n_push(c.n_embd)),
                )?);
            }
            if let Some(b) = &la.k_bias {
                plan.push(mk(
                    PipeId::Add,
                    &[(st.b_k.buffer, 0, nb(c.kv_dim)), (b.buffer, 0, b.size)],
                    Self::groups_for(c.kv_dim),
                    PushSpec::Static(n_push(c.kv_dim)),
                )?);
            }
            if let Some(b) = &la.v_bias {
                plan.push(mk(
                    PipeId::Add,
                    &[(st.b_v.buffer, 0, nb(c.kv_dim)), (b.buffer, 0, b.size)],
                    Self::groups_for(c.kv_dim),
                    PushSpec::Static(n_push(c.kv_dim)),
                )?);
            }
            plan.push(mk(
                PipeId::Rope,
                &[
                    (st.b_q.buffer, 0, nb(c.n_embd)),
                    (st.freq_buf.buffer, 0, st.freq_buf.size),
                ],
                Self::groups_for(c.n_head * (c.rope_dim / 2)),
                PushSpec::Rope {
                    n_head: c.n_head as u32,
                },
            )?);
            plan.push(mk(
                PipeId::Rope,
                &[
                    (st.b_k.buffer, 0, nb(c.kv_dim)),
                    (st.freq_buf.buffer, 0, st.freq_buf.size),
                ],
                Self::groups_for(c.n_head_kv * (c.rope_dim / 2)),
                PushSpec::Rope {
                    n_head: c.n_head_kv as u32,
                },
            )?);
            plan.push(PlannedOp::KvAppend { layer: l });
            let layer_off = (l * c.ctx * c.kv_dim) as u32;
            plan.push(mk(
                PipeId::Attention,
                &[
                    (st.b_q.buffer, 0, nb(c.n_embd)),
                    (st.kcache.buffer, 0, st.kcache.size),
                    (st.vcache.buffer, 0, st.vcache.size),
                    (st.b_attn.buffer, 0, nb(c.n_embd)),
                ],
                c.n_head as u32,
                PushSpec::Attention {
                    kv_layer_off: layer_off,
                },
            )?);
            plan.push(mk(
                PipeId::Matvec,
                &[
                    (lq.attn_output.buffer, 0, lq.attn_output.size_bytes),
                    (st.b_attn.buffer, 0, nb(c.n_embd)),
                    (st.b_proj.buffer, 0, nb(c.n_embd)),
                ],
                mv_groups(c.n_embd),
                PushSpec::Static(mv_push(c.n_embd, c.n_embd)),
            )?);
            plan.push(mk(
                PipeId::Add,
                &[
                    (st.b_x.buffer, 0, nb(c.n_embd)),
                    (st.b_proj.buffer, 0, nb(c.n_embd)),
                ],
                Self::groups_for(c.n_embd),
                PushSpec::Static(n_push(c.n_embd)),
            )?);
            plan.push(mk(
                PipeId::Rmsnorm,
                &[
                    (st.b_x.buffer, 0, nb(c.n_embd)),
                    (la.ffn_norm.buffer, 0, la.ffn_norm.size),
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                ],
                1,
                PushSpec::Static(rms_push()),
            )?);
            plan.push(mk(
                PipeId::Matvec,
                &[
                    (lq.ffn_gate.buffer, 0, lq.ffn_gate.size_bytes),
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_gate.buffer, 0, nb(c.n_ff)),
                ],
                mv_groups(c.n_ff),
                PushSpec::Static(mv_push(c.n_embd, c.n_ff)),
            )?);
            plan.push(mk(
                PipeId::Matvec,
                &[
                    (lq.ffn_up.buffer, 0, lq.ffn_up.size_bytes),
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_up.buffer, 0, nb(c.n_ff)),
                ],
                mv_groups(c.n_ff),
                PushSpec::Static(mv_push(c.n_embd, c.n_ff)),
            )?);
            plan.push(mk(
                PipeId::Swiglu,
                &[
                    (st.b_gate.buffer, 0, nb(c.n_ff)),
                    (st.b_up.buffer, 0, nb(c.n_ff)),
                    (st.b_act.buffer, 0, nb(c.n_ff)),
                ],
                Self::groups_for(c.n_ff),
                PushSpec::Static(n_push(c.n_ff)),
            )?);
            plan.push(mk(
                PipeId::Matvec,
                &[
                    (lq.ffn_down.buffer, 0, lq.ffn_down.size_bytes),
                    (st.b_act.buffer, 0, nb(c.n_ff)),
                    (st.b_proj.buffer, 0, nb(c.n_embd)),
                ],
                mv_groups(c.n_embd),
                PushSpec::Static(mv_push(c.n_ff, c.n_embd)),
            )?);
            plan.push(mk(
                PipeId::Add,
                &[
                    (st.b_x.buffer, 0, nb(c.n_embd)),
                    (st.b_proj.buffer, 0, nb(c.n_embd)),
                ],
                Self::groups_for(c.n_embd),
                PushSpec::Static(n_push(c.n_embd)),
            )?);
        }

        plan.push(mk(
            PipeId::Rmsnorm,
            &[
                (st.b_x.buffer, 0, nb(c.n_embd)),
                (st.output_norm_buf.buffer, 0, st.output_norm_buf.size),
                (st.b_normed.buffer, 0, nb(c.n_embd)),
            ],
            1,
            PushSpec::Static(rms_push()),
        )?);
        plan.push(mk(
            PipeId::Matvec,
            &[
                (st.output_w.buffer, 0, st.output_w.size_bytes),
                (st.b_normed.buffer, 0, nb(c.n_embd)),
                (st.b_logits.buffer, 0, nb(c.vocab)),
            ],
            mv_groups(c.vocab),
            PushSpec::Static(mv_push(c.n_embd, c.vocab)),
        )?);

        Ok(plan)
    }

    /// Regrava o command buffer do token, submete uma vez, espera o fence, lê os logits.
    fn record_and_submit(&self, token: u32, pos: usize) -> Result<Vec<f32>, MatmulError> {
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let d = &self.dev.device;
        let dev = &self.dev;
        let cmd = st.token_cmd;

        // SAFETY: pool RESET_COMMAND_BUFFER; cmd não está em uso (fence do token anterior aguardado).
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            d.begin_command_buffer(cmd, &begin)?;
        }
        self.record_token(cmd, token, pos);
        // SAFETY: cmd em gravação.
        unsafe {
            d.end_command_buffer(cmd)?;
        }

        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        // SAFETY: fence resetado antes do submit; cmd válido.
        unsafe {
            d.reset_fences(&[st.token_fence])?;
            d.queue_submit(dev.queue, &[submit], st.token_fence)?;
            d.wait_for_fences(&[st.token_fence], true, u64::MAX)?;
        }
        self.readback(&st.b_logits, st.cfg.vocab)
    }
}

impl llama_model::GpuResidentDecode for ResidentForward<'_> {
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, llama_model::ModelError> {
        let logits = self
            .record_and_submit(token, pos)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
        if let Some(st) = self.state.as_ref() {
            *st.len.borrow_mut() = pos + 1;
        }
        Ok(logits)
    }
    fn reset(&self) {
        if let Some(st) = self.state.as_ref() {
            *st.len.borrow_mut() = 0;
        }
    }
}

impl Drop for ResidentForward<'_> {
    fn drop(&mut self) {
        let d = &self.dev.device;
        // SAFETY: GPU ociosa antes de liberar.
        unsafe {
            let _ = d.device_wait_idle();
        }

        if let Some(st) = self.state.take() {
            for lq in st.qw {
                lq.attn_q.destroy(d);
                lq.attn_k.destroy(d);
                lq.attn_v.destroy(d);
                lq.attn_output.destroy(d);
                lq.ffn_gate.destroy(d);
                lq.ffn_up.destroy(d);
                lq.ffn_down.destroy(d);
            }
            st.output_w.destroy(d);
            for la in st.aux {
                la.attn_norm.destroy(d);
                la.ffn_norm.destroy(d);
                if let Some(b) = la.q_bias {
                    b.destroy(d);
                }
                if let Some(b) = la.k_bias {
                    b.destroy(d);
                }
                if let Some(b) = la.v_bias {
                    b.destroy(d);
                }
            }
            for b in [
                &st.output_norm_buf,
                &st.freq_buf,
                &st.embd_stage,
                &st.kcache,
                &st.vcache,
                &st.b_x,
                &st.b_normed,
                &st.b_q,
                &st.b_k,
                &st.b_v,
                &st.b_attn,
                &st.b_proj,
                &st.b_gate,
                &st.b_up,
                &st.b_act,
                &st.b_logits,
            ] {
                b.destroy(d);
            }
            // SAFETY: token_cmd/token_fence criados por nós; GPU ociosa.
            unsafe {
                d.free_command_buffers(self.dev.cmd_pool, &[st.token_cmd]);
                d.destroy_fence(st.token_fence, None);
            }
        }

        // SAFETY: pool/pipelines criados por nós.
        unsafe {
            d.destroy_descriptor_pool(self.desc_pool, None);
        }
        for p in [
            &self.matvec,
            &self.rmsnorm,
            &self.rope,
            &self.attention,
            &self.swiglu,
            &self.add,
        ] {
            // SAFETY: handles criados por nós, ordem inversa.
            unsafe {
                d.destroy_pipeline(p.pipeline, None);
                d.destroy_pipeline_layout(p.layout, None);
                d.destroy_descriptor_set_layout(p.desc_set_layout, None);
            }
        }
    }
}
