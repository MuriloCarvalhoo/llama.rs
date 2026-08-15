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

    /// Buffer host-visible que a **CPU vai ler** (readback). A diferença para `host` é o
    /// bit HOST_CACHED — ver `tensor::alloc_and_bind_cached`.
    fn host_read(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind_cached, create_buf};
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind_cached(ctx, phys, d, buffer)?;
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

/// Faixa de camadas que uma GPU executa no layer-split, e o papel dela no pipeline.
///
/// Layer-split divide o modelo por **camadas** entre as GPUs: a GPU do primeiro shard roda
/// `0..split`, a do segundo roda `split..n_layer`. Entre elas passa apenas a stream residual
/// (`n_embd` floats), uma vez por token — contra as 96 sincronizacoes/token que o
/// tensor-parallel exigiria. É o que torna executáveis modelos que não cabem numa GPU.
#[derive(Clone, Copy, Debug)]
pub struct Shard {
    /// Índice do device Vulkan (posição em `amd_compute_devices`).
    pub device: usize,
    /// Primeira camada global deste shard.
    pub first_layer: usize,
    /// Uma além da última camada global deste shard.
    pub end_layer: usize,
    /// Total de camadas do modelo (para saber se este shard é o último).
    pub n_layer_total: usize,
}

impl Shard {
    /// Shard único cobrindo o modelo inteiro (caminho single-GPU).
    pub fn whole(device: usize, n_layer: usize) -> Self {
        Self {
            device,
            first_layer: 0,
            end_layer: n_layer,
            n_layer_total: n_layer,
        }
    }
    /// Faz o embedding lookup (só o primeiro shard).
    pub fn is_first(&self) -> bool {
        self.first_layer == 0
    }
    /// Faz a norma final e a projeção de logits (só o último shard).
    pub fn is_last(&self) -> bool {
        self.end_layer == self.n_layer_total
    }
    pub fn n_layers(&self) -> usize {
        self.end_layer - self.first_layer
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
    /// Faixa de camadas deste shard. `n_layer` acima é a contagem LOCAL.
    pub shard: Shard,
    /// `Some` nas arquiteturas híbridas (qwen35).
    pub delta_net: Option<llama_model::DeltaNetConfig>,
}

/// Pesos Q8_0 residentes de uma camada.
pub(crate) struct LayerQ {
    pub mixer: MixerQ,
    pub ffn_gate: QWeight,
    pub ffn_up: QWeight,
    pub ffn_down: QWeight,
}

/// Pesos quantizados do que fica entre as duas normas da camada, já na VRAM.
pub(crate) enum MixerQ {
    Attn {
        attn_q: QWeight,
        attn_k: QWeight,
        attn_v: QWeight,
        attn_output: QWeight,
    },
    /// Atenção linear do qwen35 — ver `docs/qwen35-arquitetura.md`.
    Delta {
        attn_qkv: QWeight,
        attn_gate: QWeight,
        ssm_out: QWeight,
    },
}

/// Peso residente em VRAM junto com o tipo, que decide qual shader de matvec usar.
pub(crate) struct QWeight {
    pub ty: gguf::GgmlType,
    pub gpu: GpuTensor,
}

/// Buffers f32 auxiliares residentes de uma camada.
pub(crate) struct LayerAux {
    pub attn_norm: Buf,
    pub ffn_norm: Buf,
    pub q_bias: Option<Buf>,
    pub k_bias: Option<Buf>,
    pub v_bias: Option<Buf>,
    /// QK-norm por cabeça (qwen35): [head_dim].
    pub q_norm: Option<Buf>,
    pub k_norm: Option<Buf>,
    /// Pesos e estado residente da atenção linear.
    pub delta: Option<DeltaBufs>,
}

/// Ativações intermediárias do caminho de atenção linear, compartilhadas por todas as
/// camadas do shard (uma camada por vez usa cada uma).
pub(crate) struct DnBufs {
    /// Projeção q|k|v antes e depois da convolução: [conv_dim].
    pub qkv: Buf,
    pub conv: Buf,
    /// `z`, o gate que fecha a norma da saída: [value_dim].
    pub z: Buf,
    /// (g, beta) por cabeça de valor: [n_v_heads * 2].
    pub gb: Buf,
    /// q e k já normalizados em L2: [key_dim] cada.
    pub qn: Buf,
    pub kn: Buf,
    /// Saída da recorrência e da norma gated: [value_dim].
    pub out: Buf,
    pub normed: Buf,
    /// Ativação quantizada própria do caminho linear. Separada da global para que a
    /// projeção de saída não dispute o buffer com o FFN da mesma camada.
    pub xq: Buf,
    pub xd: Buf,
}

/// Canais que a convolução causal cobre: q, k (com as cabeças de chave) e v.
pub(crate) fn conv_dim_de(dn: &llama_model::DeltaNetConfig) -> usize {
    dn.d_state * dn.n_k_heads * 2 + dn.head_v_dim() * dn.n_v_heads
}

/// Pesos f32 e **estado recorrente** de uma camada de atenção linear.
///
/// `estado` e `janela` são o que substitui o KV-cache: tamanho fixo, lidos e reescritos a
/// cada token. Ficam na mesma VRAM do shard, então acompanham o layer-split de graça.
pub(crate) struct DeltaBufs {
    pub conv1d: Buf,
    /// (ssm_a, dt_bias) por cabeça, empacotados como o `dn_gates` os consome.
    pub adt: Buf,
    pub alpha: Buf,
    pub beta: Buf,
    pub norm: Buf,
    /// [n_v_heads * d_state * d_state] — o histórico inteiro da sequência.
    pub estado: Buf,
    /// [conv_dim * (d_conv - 1)] — os tokens anteriores que a convolução ainda enxerga.
    pub janela: Buf,
}

/// Identifica qual pipeline um dispatch usa (resolvido em `pipe_of`).
#[derive(Clone, Copy)]
pub(crate) enum PipeId {
    Matvec,
    MatvecQ5K,
    MatvecQ6K,
    QuantizeX,
    Rmsnorm,
    Rope,
    Attention,
    Swiglu,
    Add,
    DeltaNet,
    DnConv,
    DnGates,
    DnNorm,
    GateMul,
}

impl PipeId {
    pub(crate) fn label(self) -> &'static str {
        match self {
            PipeId::Matvec => "matvec",
            PipeId::MatvecQ5K => "matvec_q5k",
            PipeId::MatvecQ6K => "matvec_q6k",
            PipeId::QuantizeX => "quantize_x",
            PipeId::Rmsnorm => "rmsnorm",
            PipeId::Rope => "rope",
            PipeId::Attention => "attention",
            PipeId::Swiglu => "swiglu",
            PipeId::Add => "add",
            PipeId::DeltaNet => "delta_net",
            PipeId::DnConv => "dn_conv",
            PipeId::DnGates => "dn_gates",
            PipeId::DnNorm => "dn_norm",
            PipeId::GateMul => "gate_mul",
        }
    }

    /// Índices dos bindings que o shader **lê** e dos que ele **escreve**. Um binding
    /// declarado `inout` no GLSL (sem `readonly`/`writeonly`) aparece nas duas listas.
    ///
    /// É o que permite decidir se dois dispatches vizinhos podem rodar concorrentes —
    /// ver `marcar_barreiras`. A tabela espelha os qualificadores dos `.comp`: trocar um
    /// `readonly` por `writeonly` lá obriga a mexer aqui, ou a barreira some e o resultado
    /// passa a depender do escalonamento.
    fn acessos(self) -> (&'static [usize], &'static [usize]) {
        match self {
            // weight, xq, xd → out
            PipeId::Matvec | PipeId::MatvecQ5K | PipeId::MatvecQ6K | PipeId::Attention => {
                (&[0, 1, 2], &[3])
            }
            PipeId::QuantizeX => (&[0], &[1, 2]),
            PipeId::Rmsnorm | PipeId::Swiglu => (&[0, 1], &[2]),
            // x é inout: o RoPE gira em cima do próprio buffer, o Add acumula nele.
            PipeId::Rope | PipeId::Add | PipeId::GateMul => (&[0, 1], &[0]),
            // o estado recorrente (binding 0) é lido e reescrito no mesmo dispatch.
            PipeId::DeltaNet => (&[0, 1, 2, 3, 4], &[0, 5]),
            PipeId::DnConv => (&[0, 1, 2], &[0, 3]),
            PipeId::DnGates => (&[0, 1, 2, 3], &[4]),
            PipeId::DnNorm => (&[0, 1, 2], &[3]),
        }
    }
}

/// Trecho de um buffer que uma op toca: (buffer, offset em bytes, bytes).
pub(crate) type Faixa = (vk::Buffer, vk::DeviceSize, vk::DeviceSize);

/// Duas faixas colidem quando são do mesmo buffer e os intervalos se cruzam.
fn sobrepoe(a: &Faixa, b: &Faixa) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

/// Como obter os bytes de push-constant de um dispatch no momento da gravação.
pub(crate) enum PushSpec {
    /// Push totalmente conhecido na construção do plano.
    Static(Vec<u8>),
    /// RoPE: precisa de `pos` na gravação. `n_head` e o passo entre cabeças são fixos.
    Rope { n_head: u32, stride: u32 },
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
        /// Faixas de memória lidas e escritas, derivadas dos bindings e de `PipeId::acessos`.
        /// Só servem a `marcar_barreiras`, no build.
        le: Vec<Faixa>,
        esc: Vec<Faixa>,
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
    pub output_w: QWeight,
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
    /// Saída do FFN. Separada de `b_proj` (saída do mixer) porque compartilhar as duas
    /// esconde de qual das duas veio um valor quando se lê o buffer para diagnóstico.
    pub b_ffn: Buf,
    /// Buffers do caminho de atenção linear (qwen35). `None` nas arquiteturas densas.
    pub dn: Option<DnBufs>,
    /// Ativacao quantizada em int8 (8 uints por bloco de 32) e escala por bloco,
    /// produzidas uma vez por matvec pelo dispatch `QuantizeX`.
    pub b_xq: Buf,
    pub b_xd: Buf,
    /// Staging host-visible dos logits, alocado uma vez. A copia entra no proprio
    /// command buffer do token: alocar 608 KB (`vkAllocateMemory`) e submeter um
    /// segundo command buffer a cada token custava mais que o readback em si.
    pub logits_host: Buf,
    /// Mapa persistente de `logits_host`. Mapear/desmapear por token custa dois ioctls no
    /// caminho crítico. Válido enquanto o `State` viver (unmap no Drop).
    pub logits_ptr: *mut std::ffi::c_void,
    pub len: RefCell<usize>,
    pub plan: Vec<PlannedOp>,
    /// Paralelo a `plan`: se a op precisa de uma barreira de memória **antes** dela.
    /// Calculado uma vez em `marcar_barreiras`.
    pub barreiras: Vec<bool>,
    pub token_cmd: vk::CommandBuffer,
    pub token_fence: vk::Fence,
    /// Perfilamento por op via timestamp queries. `Some` só com LLAMA_RS_PROFILE=1.
    pub prof: Option<Prof>,
}

/// Acumulador de tempo de GPU por operação do plano, medido com timestamp queries.
/// Ativado por `LLAMA_RS_PROFILE=1`; fora disso nada é gravado no command buffer.
pub(crate) struct Prof {
    pub pool: vk::QueryPool,
    /// Nanossegundos de host por fase: gravacao do command buffer, submit+fence, leitura.
    pub host: RefCell<[u64; 3]>,
    /// Nanossegundos por tick do timestamp (VkPhysicalDeviceLimits::timestampPeriod).
    pub period_ns: f32,
    /// Nanossegundos acumulados por índice de op do plano.
    pub accum: RefCell<Vec<u64>>,
    pub tokens: std::cell::Cell<usize>,
    /// Zonas absolutas para a timeline (`--trace`). Vazio quando não se pede trace.
    pub spans: RefCell<Vec<GpuSpan>>,
    /// Limite de tokens gravados, para o arquivo não crescer sem fim.
    pub max_trace_tokens: usize,
}

/// Uma operação de GPU posicionada no relógio da CPU, para a timeline.
/// Qual shader de atenção linear rodar em `ResidentForward::dbg_dn`.
#[derive(Clone, Copy)]
pub enum DnPipe {
    DeltaNet,
    Conv,
    Gates,
    Norm,
    /// Não é do delta net, mas entra aqui para poder ser testado com o mesmo helper.
    QuantizeX,
}

#[derive(Clone, Debug)]
pub struct GpuSpan {
    pub name: &'static str,
    pub start: std::time::Instant,
    pub end: std::time::Instant,
}

/// Backend de decode GPU-resident (1 GPU). Construído via `ResidentForward::new`.
pub struct ResidentForward<'ctx> {
    pub(crate) ctx: &'ctx VulkanContext,
    pub(crate) phys_idx: usize,
    pub(crate) dev: VulkanDevice,
    // pipelines (preenchidos na Task 9; campos públicos ao crate para as tasks de teste)
    pub(crate) matvec: ComputePipeline,
    pub(crate) quantize_x: ComputePipeline,
    pub(crate) matvec_q5k: ComputePipeline,
    pub(crate) matvec_q6k: ComputePipeline,
    pub(crate) rmsnorm: ComputePipeline,
    pub(crate) rope: ComputePipeline,
    pub(crate) attention: ComputePipeline,
    pub(crate) swiglu: ComputePipeline,
    pub(crate) add: ComputePipeline,
    // Camadas de atenção linear (qwen35).
    pub(crate) delta_net: ComputePipeline,
    pub(crate) dn_conv: ComputePipeline,
    pub(crate) dn_gates: ComputePipeline,
    pub(crate) dn_norm: ComputePipeline,
    pub(crate) gate_mul: ComputePipeline,
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
        let host = Buf::host_read(self.ctx, self.phys(), d, bytes)?;
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
    /// Índice do device com mais VRAM livre; `LLAMA_RS_GPU` força um valor.
    ///
    /// A GPU que roda o display já tem ~1.6 GB ocupados, e num modelo que quase enche a
    /// VRAM o driver realoca o excedente em GTT (memória do host, via PCIe): medimos
    /// 95 GB/s no matvec do 14B na GPU do display contra 714 GB/s na outra.
    pub fn pick_device(ctx: &VulkanContext) -> usize {
        let phys = ctx.amd_compute_devices();
        std::env::var("LLAMA_RS_GPU")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|i| *i < phys.len())
            .or_else(|| {
                phys.iter()
                    .enumerate()
                    .filter_map(|(i, p)| p.free_device_memory(ctx).map(|f| (i, f)))
                    .max_by_key(|&(_, f)| f)
                    .map(|(i, _)| i)
            })
            .unwrap_or(0)
    }

    pub fn new_pipelines_only(ctx: &'ctx VulkanContext) -> Result<Self, MatmulError> {
        Self::new_pipelines_only_on(ctx, Self::pick_device(ctx))
    }

    pub fn new_pipelines_only_on(
        ctx: &'ctx VulkanContext,
        idx: usize,
    ) -> Result<Self, MatmulError> {
        let phys = ctx.amd_compute_devices();
        if phys.is_empty() {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED));
        }
        let dev = VulkanDevice::create(ctx, &phys[idx])?;
        let d = &dev.device;
        let matvec = ComputePipeline::new(d)?;
        // 3 bindings (x, xq, xd) + push de 12 bytes (n_in + 2 pads de alinhamento).
        let quantize_x = ComputePipeline::with(d, crate::QUANTIZE_X_SPV, 3, 12, &[])?;
        // Matvecs K-quant: os mesmos 4 bindings do Q8_0 (pesos, xq, xd, saída) e o mesmo push.
        let push_mv = std::mem::size_of::<crate::pipeline::PushConstants>() as u32;
        let matvec_q5k = ComputePipeline::with(d, crate::Q5_K_MATVEC_SPV, 4, push_mv, &[])?;
        let matvec_q6k = ComputePipeline::with(d, crate::Q6_K_MATVEC_SPV, 4, push_mv, &[])?;
        let rmsnorm = ComputePipeline::with(d, crate::RMSNORM_SPV, 3, 8, &[])?; // dim:u32 + eps:f32
        let rope = ComputePipeline::with(d, crate::ROPE_SPV, 2, 20, &[])?;
        let attention = ComputePipeline::with(d, crate::ATTENTION_SPV, 4, 28, &[])?;
        let swiglu = ComputePipeline::with(d, crate::SWIGLU_SPV, 3, 4, &[])?;
        let add = ComputePipeline::with(d, crate::ADD_SPV, 2, 4, &[])?;
        // Atenção linear: (estado, q, k, v, g|beta, saída), (estado, x, w, saída),
        // (x, alpha, beta, a|dt, saída) e (x, w, z, saída).
        let delta_net = ComputePipeline::with(d, crate::DELTA_NET_SPV, 6, 12, &[])?;
        let dn_conv = ComputePipeline::with(d, crate::DN_CONV_SPV, 4, 12, &[])?;
        let dn_gates = ComputePipeline::with(d, crate::DN_GATES_SPV, 5, 12, &[])?;
        let dn_norm = ComputePipeline::with(d, crate::DN_NORM_SPV, 4, 20, &[])?;
        let gate_mul = ComputePipeline::with(d, crate::GATE_MUL_SPV, 2, 12, &[])?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 65536,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            // Um descriptor set por op do plano. O 14B tem ~1100 ops/token (48 camadas
            // x ~23 ops), e cada matvec usa 4 bindings — o pool anterior (1024 sets)
            // estourava com ERROR_OUT_OF_POOL_MEMORY.
            max_sets: 16384,
            pool_size_count: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        // SAFETY: d válido; pool_info aponta para dados vivos.
        let desc_pool = unsafe { d.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            ctx,
            phys_idx: idx,
            dev,
            matvec,
            quantize_x,
            matvec_q5k,
            matvec_q6k,
            rmsnorm,
            rope,
            attention,
            swiglu,
            add,
            delta_net,
            dn_conv,
            dn_gates,
            dn_norm,
            gate_mul,
            desc_pool,
            state: None,
        })
    }

    /// Constrói o backend GPU-resident: sobe todos os pesos (Q8_0 + aux f32) e aloca
    /// as ativações e o KV-cache em VRAM. Após retornar, `raw`/`aux` podem ser descartados.
    /// Backend cobrindo o modelo inteiro numa GPU (escolhida por VRAM livre).
    pub fn new(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'_>,
    ) -> Result<Self, MatmulError> {
        let dev = Self::pick_device(ctx);
        Self::new_shard(ctx, config, raw, aux, Shard::whole(dev, config.n_layer))
    }

    /// Backend cobrindo apenas `shard.first_layer..shard.end_layer`, no device do shard.
    /// Só o primeiro shard faz embedding; só o último faz a norma final e os logits.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "subir_pesos")
    )]
    pub fn new_shard(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'_>,
        shard: Shard,
    ) -> Result<Self, MatmulError> {
        if config.head_dim > 256 {
            // Shader de attention distribui head_dim entre 64 lanes com no máximo
            // MAX_DPL=4 dimensões por lane.
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        // O matvec faz tiling da dimensão K em janelas de MATVEC_MAX_BLOCKS blocos,
        // então n_in é livre. (Antes havia um limite de n_in <= MATVEC_MAX_BLOCKS*32.)
        let mut me = Self::new_pipelines_only_on(ctx, shard.device)?;
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
                n_layer: shard.n_layers(),
                n_head: config.n_head,
                n_head_kv: config.n_head_kv,
                head_dim: config.head_dim,
                n_ff: config.n_ff,
                rope_dim: config.rope_dim,
                kv_dim,
                vocab: config.vocab,
                ctx: config.ctx,
                rms_eps: config.rms_eps,
                shard,
                delta_net: config.delta_net.clone(),
            };

            let up_q = |t: &llama_model::QTensor<'_>,
                        n_in: usize,
                        n_out: usize|
             -> Result<QWeight, MatmulError> {
                let gpu = GpuTensor::upload_quant(ctx, phys, dev_ref, t.ty, t.bytes, n_in, n_out)
                    .map_err(MatmulError::from)?;
                Ok(QWeight { ty: t.ty, gpu })
            };
            let mut qw = Vec::with_capacity(cfg.n_layer);
            for lw in &raw.layers[shard.first_layer..shard.end_layer] {
                // Só o caminho denso por enquanto: as camadas de atenção linear do
                // qwen35 ainda não têm plano de decode (ver docs/qwen35-arquitetura.md).
                let mixer = match &lw.mixer {
                    llama_model::MixerRaw::Attn {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                    } => {
                        // No qwen35 a projeção de Q sai dobrada (query|gate) e a saída da
                        // atenção entra com head_dim × n_head, que não é n_embd.
                        let (q_out, o_in) = match config.delta_net.as_ref() {
                            Some(_) => (
                                config.head_dim * config.n_head * 2,
                                config.head_dim * config.n_head,
                            ),
                            None => (config.n_embd, config.n_embd),
                        };
                        MixerQ::Attn {
                            attn_q: up_q(attn_q, cfg.n_embd, q_out)?,
                            attn_k: up_q(attn_k, cfg.n_embd, kv_dim)?,
                            attn_v: up_q(attn_v, cfg.n_embd, kv_dim)?,
                            attn_output: up_q(attn_output, o_in, cfg.n_embd)?,
                        }
                    }
                    llama_model::MixerRaw::Delta {
                        attn_qkv,
                        attn_gate,
                        ssm_out,
                    } => {
                        let dn = config
                            .delta_net
                            .as_ref()
                            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;
                        let value_dim = dn.head_v_dim() * dn.n_v_heads;
                        MixerQ::Delta {
                            attn_qkv: up_q(attn_qkv, cfg.n_embd, conv_dim_de(dn))?,
                            attn_gate: up_q(attn_gate, cfg.n_embd, value_dim)?,
                            ssm_out: up_q(ssm_out, value_dim, cfg.n_embd)?,
                        }
                    }
                };
                qw.push(LayerQ {
                    mixer,
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
            for al in &aux.layers[shard.first_layer..shard.end_layer] {
                aux_buf.push(LayerAux {
                    attn_norm: mk(&al.attn_norm)?,
                    ffn_norm: mk(&al.ffn_norm)?,
                    q_bias: mk_opt(&al.q_bias)?,
                    k_bias: mk_opt(&al.k_bias)?,
                    v_bias: mk_opt(&al.v_bias)?,
                    q_norm: mk_opt(&al.q_norm)?,
                    k_norm: mk_opt(&al.k_norm)?,
                    delta: match (&al.delta, config.delta_net.as_ref()) {
                        (Some(da), Some(dn)) => {
                            // (ssm_a, dt_bias) intercalados: o `dn_gates` lê os dois de
                            // uma cabeça num vec2.
                            let adt: Vec<f32> = (0..dn.n_v_heads)
                                .flat_map(|h| [da.a[h], da.dt_bias[h]])
                                .collect();
                            Some(DeltaBufs {
                                conv1d: mk(&da.conv1d)?,
                                adt: mk(&adt)?,
                                alpha: mk(&da.alpha)?,
                                beta: mk(&da.beta)?,
                                norm: mk(&da.norm)?,
                                // Estado recorrente e janela da convolução começam
                                // zerados — é o "contexto vazio" desta arquitetura.
                                estado: mk(&vec![0f32; dn.state_len()])?,
                                janela: mk(&vec![0f32; conv_dim_de(dn) * (dn.d_conv - 1)])?,
                            })
                        }
                        _ => None,
                    },
                });
            }
            let output_norm_buf = mk(&aux.output_norm)?;
            let freq_buf = mk(&aux.freq_table)?;
            let embd_stage = Buf::host(ctx, phys, d, (config.n_embd * 4) as vk::DeviceSize)?;

            let kv_elems = (cfg.n_layer * cfg.ctx * kv_dim) as vk::DeviceSize;
            let kcache = Buf::device(ctx, phys, d, kv_elems * 4)?;
            let vcache = Buf::device(ctx, phys, d, kv_elems * 4)?;

            let attn_dim = if config.delta_net.is_some() {
                config.head_dim * config.n_head
            } else {
                config.n_embd
            };
            // Query e gate juntos quando há QK-norm/gate (qwen35).
            let q_dim = if config.delta_net.is_some() {
                attn_dim * 2
            } else {
                config.n_embd
            };
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

            // Saída deste shard: logits no último, stream residual nos demais. Mapeada uma
            // única vez — ver `logits_ptr`.
            let saida_floats = if shard.is_last() {
                config.vocab
            } else {
                config.n_embd
            };
            let logits_host = Buf::host_read(ctx, phys, d, (saida_floats * 4) as vk::DeviceSize)?;
            // SAFETY: memória host-visible recém-criada com esse tamanho, ainda não mapeada.
            let logits_ptr = unsafe {
                d.map_memory(
                    logits_host.mem,
                    0,
                    logits_host.size,
                    vk::MemoryMapFlags::empty(),
                )?
            };

            ResidentState {
                cfg,
                qw,
                output_w,
                aux: aux_buf,
                output_norm_buf,
                freq_buf,
                // Só o primeiro shard faz o embedding lookup; nos demais a tabela seria
                // 3.1 GB de RAM sem uso (14B) — o suficiente para matar o processo por OOM.
                token_embd: if shard.is_first() {
                    aux.token_embd.to_vec()
                } else {
                    Vec::new()
                },
                embd_stage,
                kcache,
                vcache,
                b_x: nf(config.n_embd)?,
                b_normed: nf(config.n_embd)?,
                // No qwen35 a projeção de Q sai com query **e** gate por cabeça, e o
                // conjunto de cabeças não tem exatamente n_embd (24 × 256 = 6144 contra
                // 5120), então os buffers da atenção seguem head_dim × n_head.
                b_q: nf(q_dim)?,
                b_k: nf(kv_dim)?,
                b_v: nf(kv_dim)?,
                b_attn: nf(attn_dim)?,
                b_proj: nf(config.n_embd)?,
                b_gate: nf(config.n_ff)?,
                b_up: nf(config.n_ff)?,
                b_act: nf(config.n_ff)?,
                b_logits: nf(config.vocab)?,
                b_ffn: nf(config.n_embd)?,
                b_xq: nf(config.n_embd.max(config.n_ff) / 32 * 8)?,
                b_xd: nf(config.n_embd.max(config.n_ff) / 32)?,
                dn: match config.delta_net.as_ref() {
                    Some(dn) => {
                        let key_dim = dn.d_state * dn.n_k_heads;
                        let value_dim = dn.head_v_dim() * dn.n_v_heads;
                        let cd = conv_dim_de(dn);
                        Some(DnBufs {
                            qkv: nf(cd)?,
                            conv: nf(cd)?,
                            z: nf(value_dim)?,
                            gb: nf(dn.n_v_heads * 2)?,
                            qn: nf(key_dim)?,
                            kn: nf(key_dim)?,
                            out: nf(value_dim)?,
                            normed: nf(value_dim)?,
                            xq: nf(value_dim / 32 * 8)?,
                            xd: nf(value_dim / 32)?,
                        })
                    }
                    None => None,
                },
                // Saída deste shard: logits no último, stream residual nos demais.
                logits_host,
                logits_ptr,
                len: RefCell::new(0),
                plan: Vec::new(),
                barreiras: Vec::new(),
                token_cmd,
                token_fence,
                prof: None,
            }
        };

        me.state = Some(state);
        let plan = me.build_plan()?;
        // Perfilamento opcional: 1 timestamp antes do plano + 1 depois de cada op, então
        // o pool é dimensionado pelo plano (o 14B tem ~1100 ops/token).
        let prof = if std::env::var("LLAMA_RS_PROFILE").is_ok_and(|v| v != "0") {
            let info = vk::QueryPoolCreateInfo {
                query_type: vk::QueryType::TIMESTAMP,
                query_count: u32::try_from(plan.len() + 1)
                    .map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?,
                ..Default::default()
            };
            // SAFETY: device válido; info preenchido nesta stack frame.
            let pool = unsafe { me.dev.device.create_query_pool(&info, None)? };
            // SAFETY: instance e handle válidos (enumerados por VulkanContext).
            let props = unsafe {
                ctx.instance
                    .get_physical_device_properties(ctx.amd_compute_devices()[me.phys_idx].handle)
            };
            Some(Prof {
                pool,
                host: RefCell::new([0; 3]),
                period_ns: props.limits.timestamp_period,
                accum: RefCell::new(Vec::new()),
                tokens: std::cell::Cell::new(0),
                spans: RefCell::new(Vec::new()),
                max_trace_tokens: std::env::var("LLAMA_RS_TRACE_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8),
            })
        } else {
            None
        };
        if let Some(st) = me.state.as_mut() {
            st.barreiras = Self::marcar_barreiras(&plan, st);
            if std::env::var("LLAMA_RS_PROFILE").is_ok_and(|v| v != "0") {
                let n = st.barreiras.iter().filter(|b| **b).count();
                eprintln!(
                    "[prof] {} ops/token, {n} barreiras ({:.0}% das ops rodam agrupadas)",
                    plan.len(),
                    100.0 * (1.0 - n as f64 / plan.len() as f64)
                );
            }
            st.plan = plan;
            st.prof = prof;
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
        // O shader distribui até MAX_DPL=4 dimensões por lane sobre 64 lanes.
        if head_dim > 256 {
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
            q_stride: u32,
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
            // Cabeças contíguas neste helper de teste.
            q_stride: head_dim as u32,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 28) };
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

    /// Emite as ops de uma camada de atenção linear (qwen35), deixando o resultado em
    /// `b_proj` — o mesmo lugar onde a camada de atenção deixa o dela, para que o fecho
    /// da camada (residual, norma, FFN) seja comum aos dois caminhos.
    ///
    /// A sequência segue `docs/qwen35-arquitetura.md`:
    /// `qkv = W·x`, `z = Wg·x`, gates, convolução causal, L2 em q/k, recorrência, norma
    /// gated e projeção de saída. A ativação já foi quantizada em int8 pelo `QuantizeX`
    /// do começo da camada, então os três matvecs a consomem direto.
    #[allow(clippy::too_many_arguments)]
    fn plano_delta(
        plan: &mut Vec<PlannedOp>,
        st: &ResidentState,
        la: &LayerAux,
        c: &Cfg,
        pesos: (&QWeight, &QWeight, &QWeight),
        mk: &dyn Fn(
            PipeId,
            &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
            u32,
            PushSpec,
        ) -> Result<PlannedOp, MatmulError>,
        mv: &dyn Fn(&QWeight, &Buf, usize, usize) -> Result<PlannedOp, MatmulError>,
        mv_com: &dyn Fn(
            &QWeight,
            &Buf,
            (&Buf, &Buf),
            usize,
            usize,
        ) -> Result<PlannedOp, MatmulError>,
    ) -> Result<(), MatmulError> {
        let (w_qkv, w_gate, w_out) = pesos;
        let dn_cfg = c
            .delta_net
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;
        let b = st
            .dn
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;
        let da = la
            .delta
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;

        let key_dim = dn_cfg.d_state * dn_cfg.n_k_heads;
        let value_dim = dn_cfg.head_v_dim() * dn_cfg.n_v_heads;
        let conv_dim = conv_dim_de(dn_cfg);
        let nb = |n: usize| (n * 4) as vk::DeviceSize;
        let push3 = |a: u32, b: u32, c: u32| {
            let mut v = Vec::with_capacity(12);
            v.extend_from_slice(&a.to_le_bytes());
            v.extend_from_slice(&b.to_le_bytes());
            v.extend_from_slice(&c.to_le_bytes());
            v
        };
        let push_norm = |dim: u32, n_heads: u32, modo: u32, eps: f32| {
            let mut v = Vec::with_capacity(20);
            v.extend_from_slice(&dim.to_le_bytes());
            v.extend_from_slice(&n_heads.to_le_bytes());
            v.extend_from_slice(&modo.to_le_bytes());
            v.extend_from_slice(&eps.to_le_bytes());
            v.extend_from_slice(&dim.to_le_bytes()); // stride: cabeças contíguas
            v
        };

        // Projeções da entrada já quantizada.
        plan.push(mv(w_qkv, &b.qkv, c.n_embd, conv_dim)?);
        plan.push(mv(w_gate, &b.z, c.n_embd, value_dim)?);

        // (g, beta) por cabeça — leem `b_normed` em f32, não a versão int8: são
        // projeções f32 pequenas e o erro da quantização entraria num expoente.
        plan.push(mk(
            PipeId::DnGates,
            &[
                (st.b_normed.buffer, 0, nb(c.n_embd)),
                (da.alpha.buffer, 0, da.alpha.size),
                (da.beta.buffer, 0, da.beta.size),
                (da.adt.buffer, 0, da.adt.size),
                (b.gb.buffer, 0, nb(dn_cfg.n_v_heads * 2)),
            ],
            u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
            PushSpec::Static(push3(
                u32::try_from(c.n_embd).unwrap_or(u32::MAX),
                u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
                0,
            )),
        )?);

        // Convolução causal com estado, já saindo com SiLU.
        plan.push(mk(
            PipeId::DnConv,
            &[
                (da.janela.buffer, 0, da.janela.size),
                (b.qkv.buffer, 0, nb(conv_dim)),
                (da.conv1d.buffer, 0, da.conv1d.size),
                (b.conv.buffer, 0, nb(conv_dim)),
            ],
            Self::groups_for(conv_dim),
            PushSpec::Static(push3(
                u32::try_from(conv_dim).unwrap_or(u32::MAX),
                u32::try_from(dn_cfg.d_conv).unwrap_or(u32::MAX),
                0,
            )),
        )?);

        // L2 por cabeça em q e k, que estão nos dois primeiros terços de `conv`.
        let eps = c.rms_eps;
        for (off, dst) in [(0usize, &b.qn), (key_dim, &b.kn)] {
            plan.push(mk(
                PipeId::DnNorm,
                &[
                    (b.conv.buffer, nb(off), nb(key_dim)),
                    (da.norm.buffer, 0, da.norm.size), // não usado no modo 0
                    (b.conv.buffer, nb(off), nb(key_dim)), // idem
                    (dst.buffer, 0, nb(key_dim)),
                ],
                u32::try_from(dn_cfg.n_k_heads).unwrap_or(u32::MAX),
                PushSpec::Static(push_norm(
                    u32::try_from(dn_cfg.d_state).unwrap_or(u32::MAX),
                    u32::try_from(dn_cfg.n_k_heads).unwrap_or(u32::MAX),
                    0,
                    eps,
                )),
            )?);
        }

        // Recorrência: lê e reescreve o estado da camada.
        plan.push(mk(
            PipeId::DeltaNet,
            &[
                (da.estado.buffer, 0, da.estado.size),
                (b.qn.buffer, 0, nb(key_dim)),
                (b.kn.buffer, 0, nb(key_dim)),
                (b.conv.buffer, nb(2 * key_dim), nb(value_dim)), // v
                (b.gb.buffer, 0, nb(dn_cfg.n_v_heads * 2)),
                (b.out.buffer, 0, nb(value_dim)),
            ],
            u32::try_from(dn_cfg.n_v_heads * dn_cfg.d_state / 4).unwrap_or(u32::MAX),
            PushSpec::Static(push3(
                u32::try_from(dn_cfg.d_state).unwrap_or(u32::MAX),
                u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
                u32::try_from(dn_cfg.n_v_heads / dn_cfg.n_k_heads).unwrap_or(1),
            )),
        )?);

        // Norma gated: rmsnorm por cabeça vezes silu(z).
        plan.push(mk(
            PipeId::DnNorm,
            &[
                (b.out.buffer, 0, nb(value_dim)),
                (da.norm.buffer, 0, da.norm.size),
                (b.z.buffer, 0, nb(value_dim)),
                (b.normed.buffer, 0, nb(value_dim)),
            ],
            u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
            PushSpec::Static(push_norm(
                u32::try_from(dn_cfg.head_v_dim()).unwrap_or(u32::MAX),
                u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
                1,
                eps,
            )),
        )?);

        // A saída da recorrência precisa ser requantizada antes do matvec final: o
        // `QuantizeX` do começo da camada quantizou `b_normed`, não isto.
        plan.push(mk(
            PipeId::QuantizeX,
            &[
                (b.normed.buffer, 0, nb(value_dim)),
                (b.xq.buffer, 0, b.xq.size),
                (b.xd.buffer, 0, b.xd.size),
            ],
            u32::try_from((value_dim / 32).div_ceil(64)).unwrap_or(u32::MAX),
            PushSpec::Static(push3(u32::try_from(value_dim).unwrap_or(u32::MAX), 0, 0)),
        )?);
        plan.push(mv_com(
            w_out,
            &st.b_proj,
            (&b.xq, &b.xd),
            value_dim,
            c.n_embd,
        )?);
        Ok(())
    }

    /// nº de workgroups para cobrir `n` elementos com local_size_x=64.
    pub(crate) fn groups_for(n: usize) -> u32 {
        ((n + 63) / 64) as u32
    }

    /// Roda um shader de atenção linear com buffers f32 e devolve o conteúdo final de
    /// todos eles, na ordem dos bindings.
    ///
    /// Serve para validar os shaders do qwen35 contra `llama_model::delta_net`: os
    /// buffers `inout` (o estado recorrente, a janela da convolução) voltam atualizados,
    /// e é justamente essa atualização que precisa bater com a referência de CPU.
    pub fn dbg_dn(
        &self,
        qual: DnPipe,
        bufs: &[Vec<f32>],
        push: &[u8],
        groups: u32,
    ) -> Result<Vec<Vec<f32>>, MatmulError> {
        let d = &self.dev.device;
        let pipe = match qual {
            DnPipe::DeltaNet => &self.delta_net,
            DnPipe::Conv => &self.dn_conv,
            DnPipe::Gates => &self.dn_gates,
            DnPipe::Norm => &self.dn_norm,
            DnPipe::QuantizeX => &self.quantize_x,
        };
        let mut gpu = Vec::with_capacity(bufs.len());
        for b in bufs {
            let buf = Buf::device(
                self.ctx,
                self.phys(),
                d,
                (b.len() * 4).max(4) as vk::DeviceSize,
            )?;
            self.upload_f32(&buf, b)?;
            gpu.push(buf);
        }
        let set = self.alloc_set(pipe)?;
        let bindings: Vec<_> = gpu.iter().map(|b| (b.buffer, 0, b.size)).collect();
        self.dispatch1(pipe, set, &bindings, push, groups)?;

        let mut out = Vec::with_capacity(bufs.len());
        for (buf, orig) in gpu.iter().zip(bufs) {
            out.push(self.readback(buf, orig.len())?);
        }
        for buf in gpu {
            buf.destroy(d);
        }
        Ok(out)
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
            stride: u32,
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
            // Cabeças contíguas neste helper de teste.
            stride: head_dim as u32,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 20) };
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

    /// Para cada op do plano, se ela precisa de uma barreira de memória **antes**.
    ///
    /// Ops entre duas barreiras podem rodar concorrentes na GPU. Uma barreira só faz falta
    /// quando a op conflita com o que o grupo corrente já fez: lê o que foi escrito (RAW),
    /// escreve o que foi lido (WAR) ou escreve o que já foi escrito (WAW).
    ///
    /// Antes emitíamos uma barreira depois de **todo** dispatch, e isso custa um "tail" por
    /// op: nenhum workgroup do próximo começa antes que o último do anterior termine, e o
    /// fim de um matvec ocupa poucas waves de 240 SIMDs. Numa camada densa do Qwen2.5 as
    /// projeções Q/K/V leem a mesma ativação e escrevem buffers distintos — assim como
    /// `ffn_gate`/`ffn_up` —, então não havia dependência nenhuma a respeitar entre elas.
    ///
    /// O critério é conservador em dois pontos, de propósito: compara as faixas **inteiras**
    /// dos bindings (não o que o shader de fato toca) e trata o KV-cache como um buffer só,
    /// já que o offset do append depende de `pos` e não é conhecido aqui.
    fn marcar_barreiras(plan: &[PlannedOp], st: &ResidentState) -> Vec<bool> {
        // `LLAMA_RS_NO_GROUP=1` volta ao comportamento antigo (uma barreira por op) para
        // poder medir o efeito do agrupamento no mesmo binário.
        if std::env::var("LLAMA_RS_NO_GROUP").is_ok_and(|v| v != "0") {
            return vec![true; plan.len()];
        }
        let tudo = |b: &Buf| -> Faixa { (b.buffer, 0, b.size) };
        let mut grupo_le: Vec<Faixa> = Vec::new();
        let mut grupo_esc: Vec<Faixa> = Vec::new();
        let mut out = Vec::with_capacity(plan.len());

        for op in plan {
            let (le, esc): (Vec<Faixa>, Vec<Faixa>) = match op {
                PlannedOp::Dispatch { le, esc, .. } => (le.clone(), esc.clone()),
                PlannedOp::Embed => (vec![tudo(&st.embd_stage)], vec![tudo(&st.b_x)]),
                PlannedOp::KvAppend { .. } => (
                    vec![tudo(&st.b_k), tudo(&st.b_v)],
                    vec![tudo(&st.kcache), tudo(&st.vcache)],
                ),
            };
            let raw = le.iter().any(|f| grupo_esc.iter().any(|g| sobrepoe(f, g)));
            let war_waw = esc.iter().any(|f| {
                grupo_esc.iter().any(|g| sobrepoe(f, g)) || grupo_le.iter().any(|g| sobrepoe(f, g))
            });
            let precisa = raw || war_waw;
            if precisa {
                grupo_le.clear();
                grupo_esc.clear();
            }
            out.push(precisa);
            grupo_le.extend(le);
            grupo_esc.extend(esc);
        }
        out
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
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "gravar_cmdbuf")
    )]
    fn record_token(&self, cmd: vk::CommandBuffer, token: u32, pos: usize, x_in: Option<&[f32]>) {
        let d = &self.dev.device;
        let st = self
            .state
            .as_ref()
            .expect("record_token requer state (new())");
        let c = &st.cfg;
        let total_len = (pos + 1) as u32;

        // Timestamp inicial (slot 0); cada op grava o seu em slot i+1.
        if let Some(pf) = &st.prof {
            let n = (st.plan.len() + 1) as u32;
            // SAFETY: cmd em gravação; pool criado com 1024 slots >= n.
            unsafe {
                d.cmd_reset_query_pool(cmd, pf.pool, 0, n);
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pf.pool, 0);
            }
        }

        for (op_idx, op) in st.plan.iter().enumerate() {
            // Com o perfil ligado, serializa tudo: sem barreira as ops de um mesmo grupo se
            // sobrepõem e os timestamps de fim passam a medir a soma, não cada op. O TOTAL
            // impresso fica então acima do tempo real de um token.
            if st.barreiras[op_idx] || st.prof.is_some() {
                self.full_barrier(cmd);
            }
            match op {
                PlannedOp::Embed => {
                    // Fonte: a linha do token (primeiro shard) ou a stream residual que
                    // veio da GPU anterior. Vai para `embd_stage` e daí para b_x.
                    let row = token as usize * c.n_embd;
                    let bytes = (c.n_embd * 4) as vk::DeviceSize;
                    let from_table = st.token_embd.get(row..row + c.n_embd);
                    if let Some(src) = x_in.or(from_table) {
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
                }
                PlannedOp::Dispatch {
                    pipe,
                    set,
                    groups,
                    push,
                    ..
                } => {
                    let p = self.pipe_of(*pipe);
                    let bytes: Vec<u8> = match push {
                        PushSpec::Static(b) => b.clone(),
                        PushSpec::Rope { n_head, stride } => {
                            #[repr(C)]
                            struct P {
                                n_head: u32,
                                head_dim: u32,
                                rope_dim: u32,
                                pos: f32,
                                stride: u32,
                            }
                            let pp = P {
                                n_head: *n_head,
                                head_dim: c.head_dim as u32,
                                rope_dim: c.rope_dim as u32,
                                pos: pos as f32,
                                stride: *stride,
                            };
                            unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 20) }
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
                                q_stride: u32,
                            }
                            let pp = P {
                                n_head: c.n_head as u32,
                                n_head_kv: c.n_head_kv as u32,
                                head_dim: c.head_dim as u32,
                                total_len,
                                kv_dim: c.kv_dim as u32,
                                kv_layer_off: *kv_layer_off,
                                // No qwen35 query e gate dividem a cabeça.
                                q_stride: if c.delta_net.is_some() {
                                    (c.head_dim * 2) as u32
                                } else {
                                    c.head_dim as u32
                                },
                            };
                            unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 28) }
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
                }
            }
            if let Some(pf) = &st.prof {
                // SAFETY: cmd em gravação; o pool foi dimensionado com plan.len()+1 slots.
                unsafe {
                    d.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        pf.pool,
                        (op_idx + 1) as u32,
                    );
                }
            }
        }

        // Fecha o último grupo de ops antes de ler o que ele escreveu.
        self.full_barrier(cmd);

        // Saída deste shard para o staging host-visible, no mesmo command buffer. O último
        // shard entrega logits; os demais entregam a stream residual, que segue para a
        // próxima GPU.
        let (src, n_out) = if c.shard.is_last() {
            (&st.b_logits, c.vocab)
        } else {
            (&st.b_x, c.n_embd)
        };
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (n_out * 4) as vk::DeviceSize,
        };
        // SAFETY: cmd em gravação; ambos os buffers vivem no state.
        unsafe {
            d.cmd_copy_buffer(cmd, src.buffer, st.logits_host.buffer, &[region]);
        }
    }

    pub(crate) fn pipe_of(&self, id: PipeId) -> &ComputePipeline {
        match id {
            PipeId::Matvec => &self.matvec,
            PipeId::MatvecQ5K => &self.matvec_q5k,
            PipeId::MatvecQ6K => &self.matvec_q6k,
            PipeId::DeltaNet => &self.delta_net,
            PipeId::DnConv => &self.dn_conv,
            PipeId::DnGates => &self.dn_gates,
            PipeId::DnNorm => &self.dn_norm,
            PipeId::GateMul => &self.gate_mul,
            PipeId::QuantizeX => &self.quantize_x,
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
        // Push do quantize: n_in + 2 pads (o range declarado no pipeline e de 12 bytes).
        let qx_push = |n_in: usize| -> Vec<u8> {
            let p: [u32; 3] = [n_in as u32, 0, 0];
            unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), 12) }.to_vec()
        };
        let qx_groups = |n_in: usize| -> u32 { ((n_in / 32) as u32).div_ceil(64) };

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
            let (le_idx, esc_idx) = pipe.acessos();
            let faixas = |idx: &[usize]| -> Vec<Faixa> {
                idx.iter().filter_map(|&i| binds.get(i).copied()).collect()
            };
            Ok(PlannedOp::Dispatch {
                pipe,
                set,
                groups,
                push,
                le: faixas(le_idx),
                esc: faixas(esc_idx),
            })
        };

        let nb = |n: usize| (n * 4) as vk::DeviceSize;

        // Emite o matvec com o shader do tipo do peso. Os três tipos consomem a mesma
        // ativação int8 (`b_xq`/`b_xd`, produzida pelo dispatch QuantizeX): os sub-blocos
        // dos K-quants têm 32 elementos, exatamente o bloco de quantização do Q8_0, então
        // as escalas casam sem nenhuma conversão.
        let mv =
            |w: &QWeight, dst: &Buf, n_in: usize, n_out: usize| -> Result<PlannedOp, MatmulError> {
                let comum = [
                    (w.gpu.buffer, 0, w.gpu.size_bytes),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ];
                let saida = (dst.buffer, 0, nb(n_out));
                match w.ty {
                    gguf::GgmlType::Q8_0 => mk(
                        PipeId::Matvec,
                        &[comum[0], comum[1], comum[2], saida],
                        mv_groups(n_out),
                        PushSpec::Static(mv_push(n_in, n_out)),
                    ),
                    ty => {
                        // Os dois shaders K-quant fazem 4 waves x 2 linhas por wave: a
                        // ativação é lida uma vez para as duas linhas.
                        let (pipe, rows_por_wg) = if ty == gguf::GgmlType::Q5_K {
                            (PipeId::MatvecQ5K, 8)
                        } else {
                            (PipeId::MatvecQ6K, 8)
                        };
                        mk(
                            pipe,
                            &[comum[0], comum[1], comum[2], saida],
                            u32::try_from(n_out.div_ceil(rows_por_wg)).unwrap_or(u32::MAX),
                            PushSpec::Static(mv_push(n_in, n_out)),
                        )
                    }
                }
            };

        // Como `mv`, mas com a ativação vinda de buffers escolhidos pelo chamador.
        let mv_com = |w: &QWeight,
                      dst: &Buf,
                      ativ: (&Buf, &Buf),
                      n_in: usize,
                      n_out: usize|
         -> Result<PlannedOp, MatmulError> {
            let (xq, xd) = ativ;
            let rows_por_wg = 8;
            let pipe = if w.ty == gguf::GgmlType::Q5_K {
                PipeId::MatvecQ5K
            } else {
                PipeId::MatvecQ6K
            };
            mk(
                pipe,
                &[
                    (w.gpu.buffer, 0, w.gpu.size_bytes),
                    (xq.buffer, 0, xq.size),
                    (xd.buffer, 0, xd.size),
                    (dst.buffer, 0, nb(n_out)),
                ],
                u32::try_from(n_out.div_ceil(rows_por_wg)).unwrap_or(u32::MAX),
                PushSpec::Static(mv_push(n_in, n_out)),
            )
        };

        // Presente em todos os shards: no primeiro copia a linha do token da tabela de
        // embedding; nos demais copia a stream residual vinda da GPU anterior. Nos dois
        // casos o host escreve em `embd_stage` e a cópia entra no command buffer do token.
        plan.push(PlannedOp::Embed);

        // Diagnóstico: `LLAMA_RS_STOP_LAYER=N` executa só as N primeiras camadas do
        // shard. Com N=0 o token sai do embedding direto para a projeção final, o que dá
        // uma linha de base para saber se as camadas estão de fato contribuindo.
        let parar_em: Option<usize> = std::env::var("LLAMA_RS_STOP_LAYER")
            .ok()
            .and_then(|v| v.parse().ok());
        for l in 0..c.n_layer {
            if parar_em.is_some_and(|n| l >= n) {
                break;
            }
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
                PipeId::QuantizeX,
                &[
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ],
                qx_groups(c.n_embd),
                PushSpec::Static(qx_push(c.n_embd)),
            )?);
            // Camada de atenção linear (qwen35): caminho próprio, que troca o KV-cache
            // por estado recorrente. Os dois caminhos deixam o resultado em `b_proj`, e o
            // fecho da camada (residual + norma + FFN) é comum.
            let eh_delta = matches!(&lq.mixer, MixerQ::Delta { .. });
            if let MixerQ::Delta {
                attn_qkv,
                attn_gate,
                ssm_out,
            } = &lq.mixer
            {
                Self::plano_delta(
                    &mut plan,
                    st,
                    la,
                    c,
                    (attn_qkv, attn_gate, ssm_out),
                    &mk,
                    &mv,
                    &mv_com,
                )?;
            }
            if !eh_delta {
                let (w_q, w_k, w_v, w_o) = match &lq.mixer {
                    MixerQ::Attn {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                    } => (attn_q, attn_k, attn_v, attn_output),
                    MixerQ::Delta { .. } => unreachable!("tratado acima"),
                };
                // No qwen35 a projeção de Q sai com query e gate por cabeça (2 × head_dim),
                // e o conjunto de cabeças (head_dim × n_head) não é n_embd.
                let hib = c.delta_net.is_some();
                let attn_dim = if hib { c.head_dim * c.n_head } else { c.n_embd };
                let q_out = if hib { attn_dim * 2 } else { c.n_embd };
                plan.push(mv(w_q, &st.b_q, c.n_embd, q_out)?);
                plan.push(mv(w_k, &st.b_k, c.n_embd, c.kv_dim)?);
                plan.push(mv(w_v, &st.b_v, c.n_embd, c.kv_dim)?);
                // QK-norm: RMSNorm por cabeça, in-place. No Q as cabeças estão espaçadas de
                // 2 × head_dim porque o gate mora ao lado da query.
                if let (Some(qn), Some(kn)) = (&la.q_norm, &la.k_norm) {
                    let push_qk = |n_heads: u32, stride: u32| {
                        let mut v = Vec::with_capacity(20);
                        v.extend_from_slice(&u32::try_from(c.head_dim).unwrap_or(0).to_le_bytes());
                        v.extend_from_slice(&n_heads.to_le_bytes());
                        v.extend_from_slice(&2u32.to_le_bytes()); // modo QK-norm
                        v.extend_from_slice(&c.rms_eps.to_le_bytes());
                        v.extend_from_slice(&stride.to_le_bytes());
                        v
                    };
                    plan.push(mk(
                        PipeId::DnNorm,
                        &[
                            (st.b_q.buffer, 0, nb(q_out)),
                            (qn.buffer, 0, qn.size),
                            (st.b_q.buffer, 0, nb(q_out)),
                            (st.b_q.buffer, 0, nb(q_out)),
                        ],
                        u32::try_from(c.n_head).unwrap_or(u32::MAX),
                        PushSpec::Static(push_qk(
                            u32::try_from(c.n_head).unwrap_or(u32::MAX),
                            u32::try_from(c.head_dim * 2).unwrap_or(u32::MAX),
                        )),
                    )?);
                    plan.push(mk(
                        PipeId::DnNorm,
                        &[
                            (st.b_k.buffer, 0, nb(c.kv_dim)),
                            (kn.buffer, 0, kn.size),
                            (st.b_k.buffer, 0, nb(c.kv_dim)),
                            (st.b_k.buffer, 0, nb(c.kv_dim)),
                        ],
                        u32::try_from(c.n_head_kv).unwrap_or(u32::MAX),
                        PushSpec::Static(push_qk(
                            u32::try_from(c.n_head_kv).unwrap_or(u32::MAX),
                            u32::try_from(c.head_dim).unwrap_or(u32::MAX),
                        )),
                    )?);
                }
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
                        (st.b_q.buffer, 0, nb(q_out)),
                        (st.freq_buf.buffer, 0, st.freq_buf.size),
                    ],
                    Self::groups_for(c.n_head * (c.rope_dim / 2)),
                    PushSpec::Rope {
                        n_head: c.n_head as u32,
                        stride: if hib {
                            (c.head_dim * 2) as u32
                        } else {
                            c.head_dim as u32
                        },
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
                        stride: c.head_dim as u32,
                    },
                )?);
                plan.push(PlannedOp::KvAppend { layer: l });
                let layer_off = (l * c.ctx * c.kv_dim) as u32;
                plan.push(mk(
                    PipeId::Attention,
                    &[
                        (st.b_q.buffer, 0, nb(q_out)),
                        (st.kcache.buffer, 0, st.kcache.size),
                        (st.vcache.buffer, 0, st.vcache.size),
                        (st.b_attn.buffer, 0, nb(attn_dim)),
                    ],
                    c.n_head as u32,
                    PushSpec::Attention {
                        kv_layer_off: layer_off,
                    },
                )?);
                if !hib {
                    plan.push(mk(
                        PipeId::QuantizeX,
                        &[
                            (st.b_attn.buffer, 0, nb(c.n_embd)),
                            (st.b_xq.buffer, 0, st.b_xq.size),
                            (st.b_xd.buffer, 0, st.b_xd.size),
                        ],
                        qx_groups(c.n_embd),
                        PushSpec::Static(qx_push(c.n_embd)),
                    )?);
                }
                if hib {
                    // Portão do qwen35: a saída da atenção passa por sigmoid(gate), com o
                    // gate vindo da segunda metade da própria projeção de Q.
                    let mut pg = Vec::with_capacity(12);
                    pg.extend_from_slice(&u32::try_from(attn_dim).unwrap_or(0).to_le_bytes());
                    pg.extend_from_slice(&u32::try_from(c.head_dim).unwrap_or(0).to_le_bytes());
                    pg.extend_from_slice(&0u32.to_le_bytes());
                    plan.push(mk(
                        PipeId::GateMul,
                        &[
                            (st.b_attn.buffer, 0, nb(attn_dim)),
                            (st.b_q.buffer, 0, nb(q_out)),
                        ],
                        Self::groups_for(attn_dim),
                        PushSpec::Static(pg),
                    )?);
                    plan.push(mk(
                        PipeId::QuantizeX,
                        &[
                            (st.b_attn.buffer, 0, nb(attn_dim)),
                            (st.b_xq.buffer, 0, st.b_xq.size),
                            (st.b_xd.buffer, 0, st.b_xd.size),
                        ],
                        qx_groups(attn_dim),
                        PushSpec::Static(qx_push(attn_dim)),
                    )?);
                }
                plan.push(mv(w_o, &st.b_proj, attn_dim, c.n_embd)?);
            }
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
                PipeId::QuantizeX,
                &[
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ],
                qx_groups(c.n_embd),
                PushSpec::Static(qx_push(c.n_embd)),
            )?);
            plan.push(mv(&lq.ffn_gate, &st.b_gate, c.n_embd, c.n_ff)?);
            plan.push(mv(&lq.ffn_up, &st.b_up, c.n_embd, c.n_ff)?);
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
                PipeId::QuantizeX,
                &[
                    (st.b_act.buffer, 0, nb(c.n_ff)),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ],
                qx_groups(c.n_ff),
                PushSpec::Static(qx_push(c.n_ff)),
            )?);
            plan.push(mv(&lq.ffn_down, &st.b_ffn, c.n_ff, c.n_embd)?);
            plan.push(mk(
                PipeId::Add,
                &[
                    (st.b_x.buffer, 0, nb(c.n_embd)),
                    (st.b_ffn.buffer, 0, nb(c.n_embd)),
                ],
                Self::groups_for(c.n_embd),
                PushSpec::Static(n_push(c.n_embd)),
            )?);
        }

        if c.shard.is_last() {
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
                PipeId::QuantizeX,
                &[
                    (st.b_normed.buffer, 0, nb(c.n_embd)),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ],
                qx_groups(c.n_embd),
                PushSpec::Static(qx_push(c.n_embd)),
            )?);
            plan.push(mv(&st.output_w, &st.b_logits, c.n_embd, c.vocab)?);
        }

        Ok(plan)
    }

    /// Lê os timestamps do token e acumula o tempo de GPU por op. No-op sem profiling.
    fn collect_prof(&self, st: &ResidentState) -> Result<(), MatmulError> {
        let Some(pf) = &st.prof else { return Ok(()) };
        let n = st.plan.len() + 1;
        let mut ticks = vec![0u64; n];
        // SAFETY: pool tem >= n slots, todos escritos neste command buffer já concluído.
        unsafe {
            self.dev.device.get_query_pool_results(
                pf.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let mut accum = pf.accum.borrow_mut();
        if accum.len() < n - 1 {
            accum.resize(n - 1, 0);
        }
        for i in 0..n - 1 {
            let delta = ticks[i + 1].saturating_sub(ticks[i]);
            #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
            let ns = (delta as f64 * f64::from(pf.period_ns)) as u64;
            accum[i] += ns;
        }
        pf.tokens.set(pf.tokens.get() + 1);

        // Zonas absolutas para a timeline. O fence acabou de ser aguardado, então `agora`
        // é o instante do ÚLTIMO timestamp — dá para ancorar o token inteiro sem submit
        // extra nem VK_EXT_calibrated_timestamps, e sem drift acumulado entre tokens.
        if pf.tokens.get() <= pf.max_trace_tokens {
            let now = std::time::Instant::now();
            let last = ticks[n - 1];
            let at = |tick: u64| -> std::time::Instant {
                #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
                let back = (last.saturating_sub(tick) as f64 * f64::from(pf.period_ns)) as u64;
                now - std::time::Duration::from_nanos(back)
            };
            let mut spans = pf.spans.borrow_mut();
            for i in 0..n - 1 {
                let name = match st.plan.get(i) {
                    Some(PlannedOp::Dispatch { pipe, .. }) => pipe.label(),
                    Some(PlannedOp::Embed) => "embed",
                    Some(PlannedOp::KvAppend { .. }) => "kv_append",
                    None => continue,
                };
                spans.push(GpuSpan {
                    name,
                    start: at(ticks[i]),
                    end: at(ticks[i + 1]),
                });
            }
        }
        Ok(())
    }

    /// Zonas de GPU coletadas, para a timeline. Vazio sem profiling.
    pub fn gpu_spans(&self) -> Vec<GpuSpan> {
        self.state
            .as_ref()
            .and_then(|st| st.prof.as_ref())
            .map(|pf| pf.spans.borrow().clone())
            .unwrap_or_default()
    }

    /// Nome do device deste backend, para rotular a trilha na timeline.
    pub fn device_name(&self) -> String {
        self.ctx.amd_compute_devices()[self.phys_idx]
            .name()
            .to_owned()
    }

    /// Imprime o perfil agregado por tipo de op (stderr). No-op sem profiling.
    pub fn print_profile(&self) {
        let Some(st) = self.state.as_ref() else {
            return;
        };
        let Some(pf) = &st.prof else { return };
        let tokens = pf.tokens.get().max(1);
        let accum = pf.accum.borrow();

        let mut por_tipo: std::collections::BTreeMap<&'static str, (u64, usize)> =
            std::collections::BTreeMap::new();
        for (i, &ns) in accum.iter().enumerate() {
            let label = match st.plan.get(i) {
                Some(PlannedOp::Dispatch { pipe, .. }) => pipe.label(),
                Some(PlannedOp::Embed) => "embed",
                Some(PlannedOp::KvAppend { .. }) => "kv_append",
                None => continue,
            };
            let e = por_tipo.entry(label).or_insert((0, 0));
            e.0 += ns;
            e.1 += 1;
        }

        let total: u64 = accum.iter().sum();
        let ms = |ns: u64| ns as f64 / 1e6 / tokens as f64;
        eprintln!(
            "\n=== PERFIL GPU ({tokens} tokens, {} ops/token) ===",
            accum.len()
        );
        eprintln!(
            "{:<12} {:>10} {:>8} {:>10}",
            "op", "ms/token", "%", "n/token"
        );
        let mut linhas: Vec<_> = por_tipo.iter().collect();
        linhas.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        for (label, (ns, n)) in linhas {
            eprintln!(
                "{label:<12} {:>10.3} {:>7.1}% {:>10}",
                ms(*ns),
                100.0 * *ns as f64 / total.max(1) as f64,
                n
            );
        }
        eprintln!("{:<12} {:>10.3} {:>7.1}%", "TOTAL GPU", ms(total), 100.0);
        let h = pf.host.borrow();
        eprintln!("\n--- host (ms/token) ---");
        eprintln!("{:<12} {:>10.3}", "gravacao", ms(h[0]));
        eprintln!("{:<12} {:>10.3}", "submit+fence", ms(h[1]));
        eprintln!("{:<12} {:>10.3}", "leitura", ms(h[2]));
    }

    /// Executa este shard para um token. `x_in` é a stream residual vinda do shard
    /// anterior (`None` no primeiro, que faz o embedding lookup). Retorna os logits no
    /// último shard, ou a stream residual a repassar nos demais.
    #[cfg_attr(feature = "profiling", tracing::instrument(skip_all, name = "shard"))]
    pub fn decode_shard(
        &self,
        token: u32,
        pos: usize,
        x_in: Option<&[f32]>,
    ) -> Result<Vec<f32>, MatmulError> {
        let out = self.record_and_submit(token, pos, x_in)?;
        if let Some(st) = self.state.as_ref() {
            *st.len.borrow_mut() = pos + 1;
        }
        Ok(out)
    }

    /// Rótulos das ops do plano, na ordem de execução — para conferir a montagem.
    pub fn dbg_plano(&self) -> Vec<&'static str> {
        self.state
            .as_ref()
            .map(|st| {
                st.plan
                    .iter()
                    .map(|op| match op {
                        PlannedOp::Dispatch { pipe, .. } => pipe.label(),
                        PlannedOp::Embed => "embed",
                        PlannedOp::KvAppend { .. } => "kv_append",
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Lê a stream residual (`b_x`) — o estado oculto depois das camadas executadas.
    ///
    /// Com `LLAMA_RS_STOP_LAYER=N` isso dá o hidden state após N camadas, que é o que se
    /// compara com o `l_out-N` do `llama-eval-callback` do llama.cpp.
    pub fn dbg_hidden(&self) -> Option<Vec<f32>> {
        let st = self.state.as_ref()?;
        self.readback(&st.b_x, st.cfg.n_embd).ok()
    }

    /// Lê um dos buffers intermediários do caminho de atenção linear, por nome.
    ///
    /// Existe para bissecar contra o dump do `llama-eval-callback`: `qkv`, `conv`, `qn`,
    /// `kn`, `gb`, `out` e `normed` correspondem, na ordem, aos tensores
    /// `linear_attn_qkv_mixed`, `conv_output_silu`, `q_conv_predelta`, `k_conv_predelta`,
    /// `gate`/`beta_sigmoid`, `attn_output` e `final_output` da referência.
    pub fn dbg_dn_buf(&self, nome: &str) -> Option<Vec<f32>> {
        let st = self.state.as_ref()?;
        let b = st.dn.as_ref()?;
        let buf = match nome {
            "qkv" => &b.qkv,
            "conv" => &b.conv,
            "z" => &b.z,
            "gb" => &b.gb,
            "qn" => &b.qn,
            "kn" => &b.kn,
            "out" => &b.out,
            "normed" => &b.normed,
            // Saída da camada e stream residual, para fechar a bissecção.
            "proj" => &st.b_proj,
            "ffn" => &st.b_ffn,
            "x" => &st.b_x,
            "xd" => &st.b_xd,
            _ => return None,
        };
        self.readback(buf, (buf.size / 4) as usize).ok()
    }

    /// Lê o estado recorrente da camada local `l`, se ela for de atenção linear.
    ///
    /// Diagnóstico: o sintoma de um estado que não persiste entre tokens é o modelo
    /// repetir o último token do prompt — sem memória, a camada linear vira função só do
    /// token atual.
    pub fn dbg_estado_delta(&self, l: usize) -> Option<Vec<f32>> {
        let st = self.state.as_ref()?;
        let dn = st.aux.get(l)?.delta.as_ref()?;
        let n = (dn.estado.size / 4) as usize;
        self.readback(&dn.estado, n).ok()
    }

    /// Faixa de camadas e papel deste backend.
    pub fn shard(&self) -> Shard {
        self.state
            .as_ref()
            .map_or(Shard::whole(0, 0), |st| st.cfg.shard)
    }

    /// Zera o comprimento do KV-cache (início de nova sequência).
    pub fn reset_len(&self) {
        if let Some(st) = self.state.as_ref() {
            *st.len.borrow_mut() = 0;
            // Nas camadas de atenção linear não há comprimento de KV-cache para zerar: o
            // histórico está no estado recorrente e na janela da convolução, que precisam
            // voltar a zero — é o que representa "contexto vazio" nesta arquitetura.
            let d = &self.dev.device;
            let zerar: Vec<vk::Buffer> = st
                .aux
                .iter()
                .filter_map(|la| la.delta.as_ref())
                .flat_map(|dn| [dn.estado.buffer, dn.janela.buffer])
                .collect();
            if zerar.is_empty() {
                return;
            }
            let tamanhos: Vec<vk::DeviceSize> = st
                .aux
                .iter()
                .filter_map(|la| la.delta.as_ref())
                .flat_map(|dn| [dn.estado.size, dn.janela.size])
                .collect();
            // SAFETY: GPU ociosa entre sequências; buffers criados por nós.
            unsafe {
                let _ = d.device_wait_idle();
                let cb_info = vk::CommandBufferAllocateInfo {
                    command_pool: self.dev.cmd_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                };
                let Ok(cbs) = d.allocate_command_buffers(&cb_info) else {
                    return;
                };
                let cmd = cbs[0];
                let begin = vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                };
                if d.begin_command_buffer(cmd, &begin).is_ok() {
                    for (buf, size) in zerar.iter().zip(&tamanhos) {
                        d.cmd_fill_buffer(cmd, *buf, 0, *size, 0);
                    }
                    let _ = d.end_command_buffer(cmd);
                    let submit = vk::SubmitInfo {
                        command_buffer_count: 1,
                        p_command_buffers: &cmd,
                        ..Default::default()
                    };
                    let _ = d.queue_submit(self.dev.queue, &[submit], vk::Fence::null());
                    let _ = d.queue_wait_idle(self.dev.queue);
                }
                d.free_command_buffers(self.dev.cmd_pool, &cbs);
            }
        }
    }

    /// Espera o fence do token **sondando**, sem dormir.
    ///
    /// Um `wait_for_fences` bloqueante entrega a thread ao escalonador pelos ~25 ms do
    /// shard. Nesse tempo o governor (schedutil) vê utilização baixa e derruba a
    /// frequência, e a CPU entra em C-state profundo — o wakeup pela IRQ da GPU passa a
    /// custar milissegundos. O sintoma era o tempo por token **bimodal**, 53 ou 57 ms com
    /// o tempo de GPU idêntico até o décimo de µs; qualquer busy-loop rodando em paralelo
    /// "consertava" o número, o que fecha o diagnóstico.
    ///
    /// Medido no Qwen2.5-32B em layer-split: 17.5-18.9 tok/s oscilando com o wait
    /// bloqueante contra **19.4 estável** sondando. Dormir 90% do tempo previsto e sondar
    /// o resto NÃO resolve — a latência já é paga ao acordar do sono longo, mesmo por
    /// timeout.
    ///
    /// O preço é um núcleo ocupado enquanto a GPU trabalha (~3.5% desta máquina de 28).
    /// É o mesmo compromisso que o llama.cpp faz no seu laço de espera.
    fn espera_fence(&self, st: &ResidentState) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        loop {
            // SAFETY: fence válido e submetido.
            if unsafe { d.get_fence_status(st.token_fence)? } {
                return Ok(());
            }
            std::hint::spin_loop();
        }
    }

    /// Regrava o command buffer do token, submete uma vez, espera o fence, lê os logits.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "submit+fence")
    )]
    fn record_and_submit(
        &self,
        token: u32,
        pos: usize,
        x_in: Option<&[f32]>,
    ) -> Result<Vec<f32>, MatmulError> {
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
        let t0 = std::time::Instant::now();
        self.record_token(cmd, token, pos, x_in);
        // SAFETY: cmd em gravação.
        unsafe {
            d.end_command_buffer(cmd)?;
        }
        let t_rec = t0.elapsed();
        let t1 = std::time::Instant::now();

        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        // SAFETY: fence resetado antes do submit; cmd válido.
        unsafe {
            d.reset_fences(&[st.token_fence])?;
            d.queue_submit(dev.queue, &[submit], st.token_fence)?;
        }
        self.espera_fence(st)?;
        let t_sub = t1.elapsed();
        let t2 = std::time::Instant::now();
        self.collect_prof(st)?;

        let len = if st.cfg.shard.is_last() {
            st.cfg.vocab
        } else {
            st.cfg.n_embd
        };
        // O mapa é persistente (feito uma vez na construção): `vkMapMemory`/`vkUnmapMemory`
        // a cada token são dois ioctls no caminho crítico, e o vocabulário do 32B faz esse
        // caminho rodar com 608 KB.
        //
        // `Vec::with_capacity` + `set_len` em vez de `vec![0.0; len]`: o segundo zera os
        // 608 KB (calloc) só para a cópia logo abaixo sobrescrever tudo.
        let out = unsafe {
            let mut v = Vec::<f32>::with_capacity(len);
            // SAFETY: logits_host é host-coherent e a cópia já terminou (fence aguardado
            // acima); `v` tem capacidade para `len` f32, escritos antes do set_len.
            std::ptr::copy_nonoverlapping(st.logits_ptr.cast::<f32>(), v.as_mut_ptr(), len);
            v.set_len(len);
            v
        };
        if let Some(pf) = &st.prof {
            let mut h = pf.host.borrow_mut();
            h[0] += t_rec.as_nanos() as u64;
            h[1] += t_sub.as_nanos() as u64;
            h[2] += t2.elapsed().as_nanos() as u64;
        }
        Ok(out)
    }
}

impl llama_model::GpuResidentDecode for ResidentForward<'_> {
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, llama_model::ModelError> {
        let logits = self
            .record_and_submit(token, pos, None)
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
            // SAFETY: mapeada uma vez na construção e nunca desmapeada até aqui.
            unsafe { d.unmap_memory(st.logits_host.mem) };
            if let Some(pf) = &st.prof {
                // SAFETY: pool criado por nós; GPU ociosa (device_wait_idle acima).
                unsafe { d.destroy_query_pool(pf.pool, None) };
            }
            for lq in st.qw {
                match lq.mixer {
                    MixerQ::Attn {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                    } => {
                        attn_q.gpu.destroy(d);
                        attn_k.gpu.destroy(d);
                        attn_v.gpu.destroy(d);
                        attn_output.gpu.destroy(d);
                    }
                    MixerQ::Delta {
                        attn_qkv,
                        attn_gate,
                        ssm_out,
                    } => {
                        attn_qkv.gpu.destroy(d);
                        attn_gate.gpu.destroy(d);
                        ssm_out.gpu.destroy(d);
                    }
                }
                lq.ffn_gate.gpu.destroy(d);
                lq.ffn_up.gpu.destroy(d);
                lq.ffn_down.gpu.destroy(d);
            }
            st.output_w.gpu.destroy(d);
            for la in st.aux {
                la.attn_norm.destroy(d);
                la.ffn_norm.destroy(d);
                for b in [&la.q_norm, &la.k_norm].into_iter().flatten() {
                    b.destroy(d);
                }
                if let Some(dn) = &la.delta {
                    for b in [
                        &dn.conv1d, &dn.adt, &dn.alpha, &dn.beta, &dn.norm, &dn.estado, &dn.janela,
                    ] {
                        b.destroy(d);
                    }
                }
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
                &st.b_q,
                &st.b_k,
                &st.b_v,
                &st.b_proj,
                &st.b_gate,
                &st.b_up,
                &st.b_logits,
                &st.b_ffn,
                &st.b_xq,
                &st.b_xd,
                &st.logits_host,
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
