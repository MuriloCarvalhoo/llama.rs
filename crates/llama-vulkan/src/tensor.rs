//! GpuTensor: buffer device-local + upload Q8_0 via staging buffer.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use ash::vk;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

/// Buffer device-local que armazena pesos Q8_0 na VRAM.
///
/// O `ash::Device` nao e armazenado aqui pois nao implementa `Clone`.
/// O caller deve chamar `destroy(dev)` antes de dropar o tensor.
pub struct GpuTensor {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub size_bytes: vk::DeviceSize,
    pub n_out: usize,
    pub n_in: usize,
}

impl GpuTensor {
    /// Upload de bytes Q8_0 para VRAM via staging buffer.
    ///
    /// Fluxo:
    /// 1. Cria staging buffer host-visible + aloca memoria host-visible
    /// 2. map_memory -> copy bytes -> unmap
    /// 3. Cria device-local buffer + aloca memoria device-local
    /// 4. one_shot_copy(staging -> device-local)
    /// 5. Destroi staging buffer e sua memoria
    pub fn upload_q8_0(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        dev: &VulkanDevice,
        bytes: &[u8],
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, TensorError> {
        // Repack Q8_0: blocos de 34 bytes (2 scale f16 + 32 qs i8) -> 36 bytes
        // (scale[2] | pad[2] | qs[32]) para que o buffer seja lido como `uint`
        // alinhado no shader (9 uints/bloco). Os valores nao mudam, so a posicao.
        debug_assert_eq!(n_in % 32, 0, "Q8_0 exige n_in multiplo de 32");
        let n_blocks = n_in / 32;
        assert_eq!(
            bytes.len(),
            n_out * n_blocks * 34,
            "Q8_0: bytes ({}) != n_out*n_blocks*34 ({})",
            bytes.len(),
            n_out * n_blocks * 34,
        );
        let repacked = {
            let mut out = vec![0u8; n_out * n_blocks * 36];
            for i in 0..(n_out * n_blocks) {
                let src = i * 34;
                let dst = i * 36;
                out[dst..dst + 2].copy_from_slice(&bytes[src..src + 2]); // scale f16
                // out[dst+2..dst+4] = pad (ja zerado)
                out[dst + 4..dst + 36].copy_from_slice(&bytes[src + 2..src + 34]); // 32 qs
            }
            out
        };
        let size = repacked.len() as vk::DeviceSize;
        let d = &dev.device;

        // 1. Staging buffer (host-visible)
        let staging_buf = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let staging_mem = alloc_and_bind(ctx, phys, d, staging_buf, true)?;

        // 2. map -> copy -> unmap
        unsafe {
            // SAFETY: staging_mem foi alocada host-visible com tamanho `size`;
            // o ponteiro retornado e valido ate unmap_memory.
            let ptr = d.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(repacked.as_ptr(), ptr as *mut u8, repacked.len());
            d.unmap_memory(staging_mem);
        }

        // 3. Device-local buffer
        let device_buf = create_buf(
            d,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let device_mem = alloc_and_bind(ctx, phys, d, device_buf, false)?;

        // 4. Copia staging -> device-local
        one_shot_copy(d, dev.queue, dev.cmd_pool, staging_buf, device_buf, size)?;

        // 5. Destroi staging
        unsafe {
            // SAFETY: staging_buf e staging_mem foram criados por nos nesta funcao;
            // a copia ja foi concluida (one_shot_copy faz fence wait).
            d.destroy_buffer(staging_buf, None);
            d.free_memory(staging_mem, None);
        }

        Ok(Self {
            buffer: device_buf,
            memory: device_mem,
            size_bytes: size,
            n_out,
            n_in,
        })
    }

    /// Libera os recursos Vulkan. Deve ser chamado antes de dropar.
    pub fn destroy(self, dev: &ash::Device) {
        unsafe {
            // SAFETY: dev e valido; buffer e memory foram criados por este device.
            dev.destroy_buffer(self.buffer, None);
            dev.free_memory(self.memory, None);
        }
        // Impede que Drop emita warning: os recursos ja foram liberados acima.
        // SAFETY: nao ha outros recursos a liberar alem de buffer e memory, ja feitos.
        std::mem::forget(self);
    }
}

impl Drop for GpuTensor {
    fn drop(&mut self) {
        // Se chegou aqui sem destroy(), recursos foram leaked.
        // Nao ha como fazer cleanup sem &ash::Device.
        // Os handles u64 nao causam crash imediato, o OS recupera ao terminar.
        if self.buffer != vk::Buffer::null() || self.memory != vk::DeviceMemory::null() {
            eprintln!("GpuTensor::drop: recursos nao liberados (chame destroy() antes de dropar)");
        }
    }
}

// ---------------------------------------------------------------------------
// Funções auxiliares pub(crate) — reutilizadas por pipeline.rs e matmul.rs
// ---------------------------------------------------------------------------

/// Cria um VkBuffer com os usage flags especificados.
pub(crate) fn create_buf(
    dev: &ash::Device,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<vk::Buffer, vk::Result> {
    let info = vk::BufferCreateInfo {
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    // SAFETY: dev e valido; info foi construido com valores corretos.
    unsafe { dev.create_buffer(&info, None) }
}

/// Aloca e faz bind de memoria para um buffer.
///
/// `host_visible = true`  → HOST_VISIBLE | HOST_COHERENT (staging)
/// `host_visible = false` → DEVICE_LOCAL (vram)
pub(crate) fn alloc_and_bind(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &ash::Device,
    buf: vk::Buffer,
    host_visible: bool,
) -> Result<vk::DeviceMemory, vk::Result> {
    // SAFETY: ctx.instance e phys.handle sao validos.
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(phys.handle)
    };

    // SAFETY: dev e valido; buf foi criado com sucesso pelo caller.
    let mem_reqs = unsafe { dev.get_buffer_memory_requirements(buf) };

    let required_flags = if host_visible {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    } else {
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    };

    let mem_type_idx = (0..mem_props.memory_type_count)
        .find(|&i| {
            let type_bit = 1u32 << i;
            let type_supported = mem_reqs.memory_type_bits & type_bit != 0;
            let flags_ok = mem_props.memory_types[i as usize]
                .property_flags
                .contains(required_flags);
            type_supported && flags_ok
        })
        .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size: mem_reqs.size,
        memory_type_index: mem_type_idx,
        ..Default::default()
    };

    // SAFETY: dev e valido; alloc_info tem mem_type_idx verificado acima.
    let memory = unsafe { dev.allocate_memory(&alloc_info, None)? };

    // SAFETY: buf e memory foram criados pelo mesmo device; offset 0 e valido.
    unsafe { dev.bind_buffer_memory(buf, memory, 0)? };

    Ok(memory)
}

/// Copia `src` para `dst` via command buffer one-shot com fence wait.
pub(crate) fn one_shot_copy(
    dev: &ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<(), vk::Result> {
    // Aloca command buffer
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    // SAFETY: dev e valido; pool foi criado por este device.
    let cmd_bufs = unsafe { dev.allocate_command_buffers(&alloc_info)? };
    let cmd = cmd_bufs[0];

    // Grava copia
    let begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    unsafe {
        // SAFETY: cmd e valido e recém alocado.
        dev.begin_command_buffer(cmd, &begin_info)?;

        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        // SAFETY: src, dst e cmd sao handles validos.
        dev.cmd_copy_buffer(cmd, src, dst, &[copy_region]);

        // SAFETY: cmd foi gravado com sucesso.
        dev.end_command_buffer(cmd)?;
    }

    // Submete com fence para sincronizar
    let fence_info = vk::FenceCreateInfo::default();
    // SAFETY: dev e valido.
    let fence = unsafe { dev.create_fence(&fence_info, None)? };

    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        ..Default::default()
    };

    // SAFETY: queue, fence e submit_info sao validos.
    let submit_res = unsafe { dev.queue_submit(queue, &[submit_info], fence) };

    // So espera se o submit teve sucesso; caso contrario a fence nunca foi sinalizada.
    // SAFETY: fence foi criada por nos; timeout u64::MAX garante espera completa.
    let wait_res = if submit_res.is_ok() {
        unsafe { dev.wait_for_fences(&[fence], true, u64::MAX) }
    } else {
        Ok(())
    };

    // Cleanup garantido em todos os paths (sucesso ou erro).
    unsafe {
        // SAFETY: fence foi criada por nos nesta funcao e nao sera mais usada.
        dev.destroy_fence(fence, None);
        // SAFETY: cmd foi alocado de pool pelo mesmo device nesta funcao.
        dev.free_command_buffers(pool, &[cmd]);
    }

    submit_res?;
    wait_res?;
    Ok(())
}

/// Quantiza `x` em int8 simetrico por bloco de 32, no mesmo layout que
/// `shaders/quantize_x.comp` produz: 8 u32 por bloco (4 i8 empacotados em cada,
/// byte j = elemento j) + a escala do bloco.
///
/// Usado pelos caminhos que ainda montam o dispatch do matvec no host; o decode
/// residente faz isso na propria GPU, uma vez por matvec.
pub(crate) fn quantize_x_host(x: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let n_blocks = x.len() / 32;
    let mut xq = vec![0u32; n_blocks * 8];
    let mut xd = vec![0f32; n_blocks];
    for b in 0..n_blocks {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d_x = amax / 127.0;
        let inv = if d_x > 0.0 { 1.0 / d_x } else { 0.0 };
        xd[b] = d_x;
        for g in 0..8 {
            let mut packed = 0u32;
            for j in 0..4 {
                #[allow(clippy::cast_possible_truncation)]
                let q = (blk[g * 4 + j] * inv).round().clamp(-127.0, 127.0) as i32;
                packed |= ((q as u32) & 0xff) << (8 * j);
            }
            xq[b * 8 + g] = packed;
        }
    }
    (xq, xd)
}
