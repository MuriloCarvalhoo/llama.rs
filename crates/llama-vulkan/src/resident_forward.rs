//! Backend de decode 100% na GPU: todas as ativações e o KV-cache residentes em VRAM.
//! Só os logits finais voltam ao host. Cada op é 1 dispatch + 1 wait nesta fatia (1C);
//! a fusão em 1 command buffer/token é a Fase 1D.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::matmul::MatmulError;
use crate::pipeline::ComputePipeline;
use ash::vk;

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

    /// Copia `len` bytes de `src[+src_off]` para `dst[+dst_off]` (regiões dentro de buffers residentes).
    pub(crate) fn copy_region(
        &self,
        src: vk::Buffer,
        src_off: vk::DeviceSize,
        dst: vk::Buffer,
        dst_off: vk::DeviceSize,
        len: vk::DeviceSize,
    ) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        let dev = &self.dev;
        let cb_info = vk::CommandBufferAllocateInfo {
            command_pool: dev.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: d/pool válidos.
        let cmd = unsafe { d.allocate_command_buffers(&cb_info)? }[0];
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        let region = vk::BufferCopy {
            src_offset: src_off,
            dst_offset: dst_off,
            size: len,
        };
        unsafe {
            // SAFETY: cmd válido; src/dst buffers vivos; offsets/len dentro dos tamanhos (garantido pelo caller).
            d.begin_command_buffer(cmd, &begin)?;
            d.cmd_copy_buffer(cmd, src, dst, &[region]);
            d.end_command_buffer(cmd)?;
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        unsafe {
            d.queue_submit(dev.queue, &[submit], vk::Fence::null())?;
            d.queue_wait_idle(dev.queue)?;
            d.free_command_buffers(dev.cmd_pool, &[cmd]);
        }
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
        let rmsnorm = ComputePipeline::with(d, crate::RMSNORM_SPV, 3, 8)?; // dim:u32 + eps:f32
        let rope = ComputePipeline::with(d, crate::ROPE_SPV, 2, 16)?;
        let attention = ComputePipeline::with(d, crate::ATTENTION_SPV, 4, 24)?;
        let swiglu = ComputePipeline::with(d, crate::SWIGLU_SPV, 3, 4)?;
        let add = ComputePipeline::with(d, crate::ADD_SPV, 2, 4)?;

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
        })
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
}

impl Drop for ResidentForward<'_> {
    fn drop(&mut self) {
        let d = &self.dev.device;
        // SAFETY: wait_idle garante GPU ociosa antes de liberar.
        unsafe {
            let _ = d.device_wait_idle();
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
            // SAFETY: handles criados por nós, destruímos em ordem inversa.
            unsafe {
                d.destroy_pipeline(p.pipeline, None);
                d.destroy_pipeline_layout(p.layout, None);
                d.destroy_descriptor_set_layout(p.desc_set_layout, None);
            }
        }
    }
}
