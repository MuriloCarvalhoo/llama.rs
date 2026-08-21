//! GpuTensor: buffer device-local + upload Q8_0 via staging buffer.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use ash::vk;
use rayon::prelude::*;
use thiserror::Error;

/// Blocos por fatia do repack paralelo. Fatia curta demais paga mais coordenação do rayon
/// do que trabalho; longa demais deixa uma thread terminando sozinha. 4096 blocos são
/// ~144 KB de saída no Q8_0 — cabe folgado no L2 de cada núcleo.
const BLOCOS_POR_FATIA: usize = 4096;

/// Aplica `f` bloco a bloco, de blocos de `src_bl` bytes para blocos de `dst_bl` bytes.
///
/// O repack dos quantizados é uma transformação **posicional pura**: o bloco `i` da saída
/// depende só do bloco `i` da entrada. Por isso o laço se divide em fatias independentes e
/// cada thread escreve numa faixa disjunta do destino — que na carga é o staging buffer,
/// escrito uma única vez e em ordem crescente (memória write-combining não perdoa
/// releitura nem escrita salteada).
fn por_bloco(
    src: &[u8],
    dst: &mut [u8],
    src_bl: usize,
    dst_bl: usize,
    f: impl Fn(&[u8], &mut [u8]) + Sync,
) {
    debug_assert_eq!(src.len() / src_bl, dst.len() / dst_bl, "blocos não batem");
    dst.par_chunks_mut(BLOCOS_POR_FATIA * dst_bl)
        .zip(src.par_chunks(BLOCOS_POR_FATIA * src_bl))
        .for_each(|(dfatia, sfatia)| {
            for (d, s) in dfatia
                .chunks_exact_mut(dst_bl)
                .zip(sfatia.chunks_exact(src_bl))
            {
                f(s, d);
            }
        });
}

/// Repack Q8_0: blocos de 34 B (escala f16 + 32 qs i8) viram 36 B
/// (`escala[2] | pad[2] | qs[32]`), para o shader ler o buffer como `uint` alinhado —
/// 9 uints por bloco. Os valores não mudam, só a posição.
pub(crate) fn repack_q8_0_into(src: &[u8], dst: &mut [u8]) {
    por_bloco(src, dst, 34, 36, |s, d| {
        d[..2].copy_from_slice(&s[..2]);
        // O pad é escrito de propósito: o destino pode ser um staging reusado.
        d[2..4].fill(0);
        d[4..36].copy_from_slice(&s[2..34]);
    });
}

/// Pad do Q6_K: superblocos de 210 B alinhados em 212, pelo mesmo motivo do Q8_0.
pub(crate) fn pad_q6_k_into(src: &[u8], dst: &mut [u8]) {
    por_bloco(src, dst, 210, 212, |s, d| {
        d[..210].copy_from_slice(s);
        d[210..212].fill(0);
    });
}

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
    /// Upload de um tensor quantizado para VRAM, no layout que o shader do tipo espera.
    ///
    /// - **Q8_0**: repack de blocos de 34 B para 36 B (ver `upload_q8_0`).
    /// - **Q5_K**: cru — o superbloco já tem 176 B = 44 uints alinhados.
    /// - **Q4_K**: cru — o superbloco tem 144 B = 36 uints alinhados (sem o `qh` do Q5_K).
    /// - **Q6_K**: superbloco de 210 B alinhado em 212, para as leituras de 32 bits do
    ///   shader ficarem alinhadas (mesmo motivo do repack do Q8_0).
    pub fn upload_quant(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        dev: &VulkanDevice,
        ty: gguf::GgmlType,
        bytes: &[u8],
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, TensorError> {
        match ty {
            gguf::GgmlType::Q8_0 => Self::upload_q8_0(ctx, phys, dev, bytes, n_in, n_out),
            gguf::GgmlType::Q5_K | gguf::GgmlType::Q4_K => {
                Self::upload_raw(ctx, phys, dev, bytes, n_in, n_out)
            }
            gguf::GgmlType::Q6_K => {
                let padded = {
                    let _fase = llama_model::perfil_carga::Fase::nova("repack+staging");
                    let mut padded = vec![0u8; bytes.len() / 210 * 212];
                    pad_q6_k_into(bytes, &mut padded);
                    padded
                };
                Self::upload_raw(ctx, phys, dev, &padded, n_in, n_out)
            }
            _ => Err(TensorError::Vulkan(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)),
        }
    }

    /// Upload sem transformação de layout.
    fn upload_raw(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        dev: &VulkanDevice,
        bytes: &[u8],
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, TensorError> {
        let d = &dev.device;
        let size = bytes.len() as vk::DeviceSize;
        let staging = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let staging_mem = alloc_and_bind_cached(ctx, phys, d, staging)?;
        {
            let _fase = llama_model::perfil_carga::Fase::nova("repack+staging");
            // SAFETY: staging_mem é host-visible com `size`; ptr válido até unmap.
            unsafe {
                let ptr = d.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
                d.unmap_memory(staging_mem);
            }
        }
        let buffer = create_buf(
            d,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let memory = alloc_and_bind(ctx, phys, d, buffer, false)?;
        {
            let _fase = llama_model::perfil_carga::Fase::nova("espera GPU (upload)");
            one_shot_copy(d, dev.queue, dev.cmd_pool, staging, buffer, size)?;
        }
        // SAFETY: staging já copiado; ambos criados por nós.
        unsafe {
            d.destroy_buffer(staging, None);
            d.free_memory(staging_mem, None);
        }
        Ok(Self {
            buffer,
            memory,
            size_bytes: size,
            n_in,
            n_out,
        })
    }

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
            let _fase = llama_model::perfil_carga::Fase::nova("repack+staging");
            let mut out = vec![0u8; n_out * n_blocks * 36];
            repack_q8_0_into(bytes, &mut out);
            out
        };
        let size = repacked.len() as vk::DeviceSize;
        let d = &dev.device;

        // 1. Staging buffer (host-visible)
        let staging_buf = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let staging_mem = alloc_and_bind_cached(ctx, phys, d, staging_buf)?;

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
    let required_flags = if host_visible {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    } else {
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    };
    alloc_with_flags(ctx, phys, dev, buf, required_flags)
}

/// Como `alloc_and_bind(.., true)`, mas exigindo **HOST_CACHED**.
///
/// Sem esse bit o driver entrega o primeiro tipo host-visible, que na MI50 é
/// write-combining: ótimo para a CPU escrever, péssimo para ela ler. Copiar os 608 KB de
/// logits do 32B custava **4.2 ms/token (145 MB/s)**; com memória cacheada a mesma cópia
/// some do perfil. Cai no tipo não-cacheado se o device não oferecer nenhum.
pub(crate) fn alloc_and_bind_cached(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &ash::Device,
    buf: vk::Buffer,
) -> Result<vk::DeviceMemory, vk::Result> {
    let cached = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    match alloc_with_flags(ctx, phys, dev, buf, cached) {
        Ok(mem) => Ok(mem),
        Err(_) => alloc_and_bind(ctx, phys, dev, buf, true),
    }
}

fn alloc_with_flags(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &ash::Device,
    buf: vk::Buffer,
    required_flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, vk::Result> {
    // SAFETY: ctx.instance e phys.handle sao validos.
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(phys.handle)
    };

    // SAFETY: dev e valido; buf foi criado com sucesso pelo caller.
    let mem_reqs = unsafe { dev.get_buffer_memory_requirements(buf) };

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
#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes determinísticos, para as duas versões verem exatamente a mesma entrada.
    fn bytes_pseudo(n: usize) -> Vec<u8> {
        let mut x: u32 = 987_654_321;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 24) as u8
            })
            .collect()
    }

    /// Referência serial do repack Q8_0, escrita do jeito mais óbvio possível.
    fn repack_serial(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 34;
        let mut out = vec![0u8; n * 36];
        for i in 0..n {
            out[i * 36..i * 36 + 2].copy_from_slice(&src[i * 34..i * 34 + 2]);
            out[i * 36 + 4..i * 36 + 36].copy_from_slice(&src[i * 34 + 2..(i + 1) * 34]);
        }
        out
    }

    /// Referência serial do pad Q6_K.
    fn pad_serial(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 210;
        let mut out = vec![0u8; n * 212];
        for i in 0..n {
            out[i * 212..i * 212 + 210].copy_from_slice(&src[i * 210..(i + 1) * 210]);
        }
        out
    }

    /// Mais de uma fatia do rayon **e** um resto incompleto: é onde um repack paralelo
    /// erra, se for errar. O destino começa sujo porque na carga ele é um staging reusado.
    #[test]
    fn repack_q8_0_paralelo_da_os_mesmos_bytes_do_serial() {
        for n in [1, 37, BLOCOS_POR_FATIA, BLOCOS_POR_FATIA * 2 + 37] {
            let src = bytes_pseudo(n * 34);
            let mut dst = vec![0xAAu8; n * 36];
            repack_q8_0_into(&src, &mut dst);
            assert_eq!(dst, repack_serial(&src), "n={n} blocos");
        }
    }

    #[test]
    fn pad_q6_k_paralelo_da_os_mesmos_bytes_do_serial() {
        for n in [1, 5, BLOCOS_POR_FATIA + 3] {
            let src = bytes_pseudo(n * 210);
            let mut dst = vec![0x5Bu8; n * 212];
            pad_q6_k_into(&src, &mut dst);
            assert_eq!(dst, pad_serial(&src), "n={n} superblocos");
        }
    }

    /// O pad tem de ser escrito, não herdado: o mesmo staging serve a vários tensores.
    #[test]
    fn o_pad_e_zerado_mesmo_com_destino_sujo() {
        let src = bytes_pseudo(3 * 34);
        let mut dst = vec![0xFFu8; 3 * 36];
        repack_q8_0_into(&src, &mut dst);
        for i in 0..3 {
            assert_eq!(&dst[i * 36 + 2..i * 36 + 4], &[0, 0], "pad do bloco {i}");
        }
        let src = bytes_pseudo(2 * 210);
        let mut dst = vec![0xFFu8; 2 * 212];
        pad_q6_k_into(&src, &mut dst);
        for i in 0..2 {
            assert_eq!(&dst[i * 212 + 210..(i + 1) * 212], &[0, 0], "pad do sb {i}");
        }
    }
}
