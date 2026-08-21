//! GpuTensor: buffer device-local + upload Q8_0 via staging buffer.

use crate::alloc::GpuAllocator;
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
    /// Nula quando a memoria veio de um chunk do [`crate::alloc::GpuAllocator`] — quem
    /// libera e o alocador, em `cleanup`. `vkFreeMemory` de handle nulo e no-op por
    /// especificacao, entao `destroy` continua valendo para os dois casos.
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
} // ---------------------------------------------------------------------------
// Uploader: memória sub-alocada, staging duplo e submits em lote
// ---------------------------------------------------------------------------

/// Bytes de cada um dos dois staging buffers. 256 MB dão dezenas de tensores por lote, e
/// o maior tensor do 27B (a projeção de vocabulário, ~1 GB) atravessa em quatro pedaços.
pub const STAGING_BYTES: vk::DeviceSize = 256 * 1024 * 1024;

/// `(bytes por bloco no GGUF, bytes por bloco na VRAM)` de cada tipo com shader.
///
/// - **Q8_0**: 34 → 36 (`escala | pad | 32 qs`), para o shader ler `uint` alinhado.
/// - **Q6_K**: 210 → 212, pelo mesmo motivo.
/// - **Q5_K / Q4_K**: crus — 176 B são 44 uints e 144 B são 36 uints, já alinhados.
fn layout_de(ty: gguf::GgmlType) -> Option<(usize, usize)> {
    match ty {
        gguf::GgmlType::Q8_0 => Some((34, 36)),
        gguf::GgmlType::Q6_K => Some((210, 212)),
        gguf::GgmlType::Q5_K => Some((176, 176)),
        gguf::GgmlType::Q4_K => Some((144, 144)),
        _ => None,
    }
}

/// Escreve os blocos de `src` em `dst` no layout que o shader do tipo espera.
fn transformar(ty: gguf::GgmlType, src: &[u8], dst: &mut [u8]) {
    match ty {
        gguf::GgmlType::Q8_0 => repack_q8_0_into(src, dst),
        gguf::GgmlType::Q6_K => pad_q6_k_into(src, dst),
        _ => dst.copy_from_slice(src),
    }
}

/// Um dos dois lotes: staging mapeado de forma persistente, o command buffer que acumula
/// as cópias dele e o fence que diz quando a GPU terminou.
struct Lote {
    staging: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Submetido e ainda não esperado.
    voando: bool,
    /// Tem cópia gravada desde o último submit.
    sujo: bool,
}

/// Sobe os pesos de um shard para a VRAM com staging duplo e **um fence por lote**.
///
/// O caminho por tensor pagava, para cada um dos ~600 tensores do modelo: uma
/// `vkAllocateMemory`, um staging buffer novo, um command buffer, um fence e uma espera de
/// `u64::MAX` — latência × N, com a GPU parada durante o repack e a CPU parada durante a
/// cópia.
///
/// Aqui a memória dos pesos sai de chunks do [`GpuAllocator`], os dois staging vivem
/// enquanto o `Uploader` viver e o repack escreve **direto** no staging: a CPU preenche um
/// lote enquanto a GPU copia o outro. É a forma do loader do llama.cpp
/// (`llama-model-loader.cpp`), que usa 4 staging pinned pelo mesmo motivo.
///
/// A memória do staging é write-combining: escrita sequencial voa, releitura é desastrosa
/// — por isso cada byte é escrito uma vez só, em ordem crescente, e nada é lido de volta.
pub struct Uploader<'d> {
    dev: &'d ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    mem: GpuAllocator,
    lotes: [Lote; 2],
    /// Lote que a CPU está preenchendo.
    atual: usize,
    /// Bytes já ocupados no staging do lote atual.
    usado: vk::DeviceSize,
    staging_bytes: vk::DeviceSize,
    nome_cpu: String,
    nome_gpu: String,
}

impl<'d> Uploader<'d> {
    /// `total_bytes` é a estimativa do que este shard vai ocupar na VRAM — dimensiona o
    /// último chunk do alocador. `rotulo` identifica o device nas fases do perfil.
    pub fn new(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        dev: &'d VulkanDevice,
        total_bytes: vk::DeviceSize,
        rotulo: &str,
    ) -> Result<Self, TensorError> {
        Self::com_staging(ctx, phys, dev, total_bytes, rotulo, STAGING_BYTES)
    }

    /// Como [`Self::new`], com o tamanho do staging escolhido — o teste usa um staging
    /// minúsculo para exercitar a virada de lote e o fatiamento com poucos KB.
    pub(crate) fn com_staging(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        dev: &'d VulkanDevice,
        total_bytes: vk::DeviceSize,
        rotulo: &str,
        staging_bytes: vk::DeviceSize,
    ) -> Result<Self, TensorError> {
        let d = &dev.device;
        let mem = GpuAllocator::new(ctx, phys, dev, total_bytes)
            .map_err(|_| TensorError::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))?;
        let info = vk::CommandBufferAllocateInfo {
            command_pool: dev.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 2,
            ..Default::default()
        };
        // SAFETY: device e pool válidos; o pool tem RESET_COMMAND_BUFFER.
        let cmds = unsafe { d.allocate_command_buffers(&info)? };
        let (Some(&c0), Some(&c1)) = (cmds.first(), cmds.get(1)) else {
            return Err(TensorError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED));
        };
        let mut up = Self {
            lotes: [
                Self::novo_lote(ctx, phys, d, c0, staging_bytes)?,
                Self::novo_lote(ctx, phys, d, c1, staging_bytes)?,
            ],
            dev: d,
            queue: dev.queue,
            pool: dev.cmd_pool,
            mem,
            atual: 0,
            usado: 0,
            staging_bytes,
            nome_cpu: format!("{rotulo} repack+staging"),
            nome_gpu: format!("{rotulo} espera GPU"),
        };
        up.preparar(0)?;
        Ok(up)
    }

    fn novo_lote(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        staging_bytes: vk::DeviceSize,
    ) -> Result<Lote, TensorError> {
        let staging = create_buf(d, staging_bytes, vk::BufferUsageFlags::TRANSFER_SRC)?;
        // Host-visible sem HOST_CACHED: aqui só se escreve, e write-combining é o tipo
        // certo para isso — o cacheado só se paga quando a CPU lê de volta.
        let mem = alloc_and_bind(ctx, phys, d, staging, true)?;
        // SAFETY: memória host-visible recém-criada com esse tamanho; o mapa vive até
        // `finalizar` e nunca é remapeado.
        let ptr = unsafe {
            d.map_memory(mem, 0, staging_bytes, vk::MemoryMapFlags::empty())?
                .cast::<u8>()
        };
        // SAFETY: device válido.
        let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok(Lote {
            staging,
            mem,
            ptr,
            cmd,
            fence,
            voando: false,
            sujo: false,
        })
    }

    /// Sobe um tensor quantizado e devolve o buffer device-local ligado ao chunk.
    ///
    /// Tensor maior que o staging (a projeção de vocabulário do 27B tem ~1 GB) é fatiado
    /// em blocos inteiros e atravessa vários lotes.
    pub fn tensor(
        &mut self,
        ty: gguf::GgmlType,
        bytes: &[u8],
        n_in: usize,
        n_out: usize,
    ) -> Result<GpuTensor, TensorError> {
        let (src_bl, dst_bl) =
            layout_de(ty).ok_or(TensorError::Vulkan(vk::Result::ERROR_FORMAT_NOT_SUPPORTED))?;
        if bytes.is_empty() || !bytes.len().is_multiple_of(src_bl) {
            return Err(TensorError::Vulkan(vk::Result::ERROR_FORMAT_NOT_SUPPORTED));
        }
        let n_blocos = bytes.len() / src_bl;
        let total = (n_blocos * dst_bl) as vk::DeviceSize;
        let buffer = self.buffer_de_pesos(total)?;

        let mut b0 = 0usize;
        while b0 < n_blocos {
            let cabem = ((self.staging_bytes - self.usado) / dst_bl as vk::DeviceSize) as usize;
            if cabem == 0 {
                self.virar()?;
                continue;
            }
            let n = cabem.min(n_blocos - b0);
            let off = self.usado;
            {
                let _fase = llama_model::perfil_carga::Fase::nova(self.nome_cpu.as_str());
                let dst = self.staging_mut(off, n * dst_bl);
                transformar(ty, &bytes[b0 * src_bl..(b0 + n) * src_bl], dst);
            }
            self.gravar_copia(buffer, off, (b0 * dst_bl) as vk::DeviceSize, n * dst_bl);
            self.usado += (n * dst_bl) as vk::DeviceSize;
            b0 += n;
        }
        Ok(GpuTensor {
            buffer,
            // A memória é do chunk; quem libera é o alocador. Ver `GpuTensor::memory`.
            memory: vk::DeviceMemory::null(),
            size_bytes: total,
            n_in,
            n_out,
        })
    }

    /// Copia `dados` para um buffer que já existe — os auxiliares f32 (normas, tabela de
    /// frequências, estado inicial) entram na mesma fila de lotes dos pesos.
    pub fn bytes_para(&mut self, dst: vk::Buffer, dados: &[u8]) -> Result<(), TensorError> {
        let mut feito = 0usize;
        while feito < dados.len() {
            let cabem = (self.staging_bytes - self.usado) as usize;
            if cabem == 0 {
                self.virar()?;
                continue;
            }
            let n = cabem.min(dados.len() - feito);
            let off = self.usado;
            {
                let _fase = llama_model::perfil_carga::Fase::nova(self.nome_cpu.as_str());
                self.staging_mut(off, n)
                    .copy_from_slice(&dados[feito..feito + n]);
            }
            self.gravar_copia(dst, off, feito as vk::DeviceSize, n);
            self.usado += n as vk::DeviceSize;
            feito += n;
        }
        Ok(())
    }

    /// Submete o que falta, espera os dois fences e devolve a memória dos pesos, que vive
    /// enquanto o backend viver. Os staging morrem aqui: 512 MB de RAM que não fazem falta
    /// depois da carga.
    pub fn finalizar(mut self) -> Result<GpuAllocator, TensorError> {
        self.submeter(self.atual)?;
        let d = self.dev;
        for i in 0..2 {
            let nome = self.nome_gpu.as_str();
            let l = &mut self.lotes[i];
            if l.voando {
                let _fase = llama_model::perfil_carga::Fase::nova(nome);
                // SAFETY: fence criada por nós, submetida com este cmd.
                unsafe { d.wait_for_fences(&[l.fence], true, u64::MAX)? };
                l.voando = false;
            }
        }
        // SAFETY: nada mais em voo; todos os handles foram criados por nós.
        unsafe {
            for l in &self.lotes {
                d.unmap_memory(l.mem);
                d.destroy_buffer(l.staging, None);
                d.free_memory(l.mem, None);
                d.destroy_fence(l.fence, None);
            }
            d.free_command_buffers(self.pool, &[self.lotes[0].cmd, self.lotes[1].cmd]);
        }
        Ok(self.mem)
    }

    /// Buffer device-local ligado a um pedaço de chunk.
    fn buffer_de_pesos(&mut self, total: vk::DeviceSize) -> Result<vk::Buffer, TensorError> {
        let d = self.dev;
        let buffer = create_buf(
            d,
            total,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        // SAFETY: buffer recém-criado por este device.
        let reqs = unsafe { d.get_buffer_memory_requirements(buffer) };
        // O alocador alinha os offsets em 256 e fixou o tipo de memória em `new`; se o
        // buffer pedir algo que isso não satisfaz, falhar é melhor que um bind inválido.
        if !256u64.is_multiple_of(reqs.alignment)
            || reqs.memory_type_bits & (1u32 << self.mem.mem_type_idx()) == 0
        {
            // SAFETY: buffer criado logo acima e ainda sem bind.
            unsafe { d.destroy_buffer(buffer, None) };
            return Err(TensorError::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        }
        let alloc = self
            .mem
            .alloc(d, reqs.size)
            .map_err(|_| TensorError::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))?;
        let memoria = self
            .mem
            .memoria(&alloc)
            .ok_or(TensorError::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))?;
        // SAFETY: buffer e memória do mesmo device; offset alinhado e dentro do chunk.
        unsafe { d.bind_buffer_memory(buffer, memoria, alloc.offset)? };
        Ok(buffer)
    }

    /// Fatia do staging do lote atual. O lote não está em voo — `preparar` esperou o fence
    /// dele —, então a CPU é a única dona desses bytes.
    fn staging_mut(&mut self, off: vk::DeviceSize, n: usize) -> &mut [u8] {
        let ptr = self.lotes[self.atual].ptr;
        // SAFETY: `off + n <= staging_bytes` porque quem chama só escreve o que reservou;
        // a memória está mapeada desde `new` e é exclusiva deste lote.
        unsafe { std::slice::from_raw_parts_mut(ptr.add(off as usize), n) }
    }

    /// Acrescenta uma cópia staging → destino ao command buffer do lote atual.
    fn gravar_copia(
        &mut self,
        dst: vk::Buffer,
        src_off: vk::DeviceSize,
        dst_off: vk::DeviceSize,
        n: usize,
    ) {
        let regiao = vk::BufferCopy {
            src_offset: src_off,
            dst_offset: dst_off,
            size: n as vk::DeviceSize,
        };
        let d = self.dev;
        let l = &mut self.lotes[self.atual];
        // SAFETY: cmd em gravação (aberto em `preparar`); buffers vivos; as faixas cabem
        // no tamanho de cada um.
        unsafe { d.cmd_copy_buffer(l.cmd, l.staging, dst, &[regiao]) };
        l.sujo = true;
    }

    /// Fecha o lote atual (submete, **sem** esperar) e passa a preencher o outro.
    fn virar(&mut self) -> Result<(), TensorError> {
        self.submeter(self.atual)?;
        self.atual ^= 1;
        self.preparar(self.atual)?;
        self.usado = 0;
        Ok(())
    }

    /// Submete o lote `i`, se ele tiver alguma cópia gravada.
    fn submeter(&mut self, i: usize) -> Result<(), TensorError> {
        let d = self.dev;
        let queue = self.queue;
        let l = &mut self.lotes[i];
        if !l.sujo {
            return Ok(());
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &l.cmd,
            ..Default::default()
        };
        // SAFETY: cmd gravado nesta thread e fechado agora; fence e fila deste device.
        unsafe {
            d.end_command_buffer(l.cmd)?;
            d.reset_fences(&[l.fence])?;
            d.queue_submit(queue, &[submit], l.fence)?;
        }
        l.voando = true;
        l.sujo = false;
        Ok(())
    }

    /// Espera o lote `i` sair de voo e reabre o command buffer dele para gravação.
    fn preparar(&mut self, i: usize) -> Result<(), TensorError> {
        let d = self.dev;
        let nome = self.nome_gpu.as_str();
        let l = &mut self.lotes[i];
        if l.voando {
            let _fase = llama_model::perfil_carga::Fase::nova(nome);
            // SAFETY: fence criada por nós e submetida com este cmd.
            unsafe { d.wait_for_fences(&[l.fence], true, u64::MAX)? };
            l.voando = false;
        }
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        // SAFETY: cmd fora de voo (fence esperada acima); pool com RESET_COMMAND_BUFFER.
        unsafe {
            d.reset_command_buffer(l.cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(l.cmd, &begin)?;
        }
        Ok(())
    }
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


/// Lê de volta `n` bytes de um buffer device-local. Só o teste faz isso: o staging da
/// carga é write-combining e nunca é lido.
fn ler_vram(
    dev: &VulkanDevice,
    buf: vk::Buffer,
    n: usize,
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
) -> Vec<u8> {
    let d = &dev.device;
    let size = n as vk::DeviceSize;
    let host = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_DST).unwrap();
    let mem = alloc_and_bind_cached(ctx, phys, d, host).unwrap();
    one_shot_copy(d, dev.queue, dev.cmd_pool, buf, host, size).unwrap();
    // SAFETY: memória host-visible cacheada, do tamanho pedido, sem uso concorrente.
    let out = unsafe {
        let ptr = d
            .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();
        let v = std::slice::from_raw_parts(ptr.cast::<u8>(), n).to_vec();
        d.unmap_memory(mem);
        d.destroy_buffer(host, None);
        d.free_memory(mem, None);
        v
    };
    out
}

/// O que a GPU recebe tem de ser byte a byte o repack de referência, inclusive quando
/// o lote vira no meio e quando um tensor é maior que o staging inteiro.
///
/// Roda com dados sintéticos: não precisa de modelo, só de uma GPU.
#[test]
fn uploader_em_lotes_entrega_os_mesmos_bytes() {
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("Vulkan indisponível — pulando");
        return;
    };
    let phys = ctx.amd_compute_devices();
    let Some(p0) = phys.first() else {
        eprintln!("nenhuma GPU AMD — pulando");
        return;
    };
    let dev = VulkanDevice::create(&ctx, p0).unwrap();

    // Staging minúsculo de propósito: com 64 KiB os tensores abaixo forçam várias
    // viradas de lote e um fatiamento, que é o que se quer exercitar.
    const STAGING: vk::DeviceSize = 64 * 1024;
    let mut upl = Uploader::com_staging(&ctx, p0, &dev, 8 * 1024 * 1024, "teste", STAGING).unwrap();

    // (tipo, blocos por linha, linhas): o terceiro Q8_0 sozinho passa dos 64 KiB.
    let casos: [(gguf::GgmlType, usize, usize); 5] = [
        (gguf::GgmlType::Q8_0, 2, 300),
        (gguf::GgmlType::Q6_K, 1, 100),
        (gguf::GgmlType::Q8_0, 4, 900),
        (gguf::GgmlType::Q5_K, 1, 120),
        (gguf::GgmlType::Q8_0, 1, 7),
    ];
    let mut subidos = Vec::new();
    for (ty, blocos, linhas) in casos {
        let (src_bl, dst_bl) = layout_de(ty).unwrap();
        let src = bytes_pseudo(blocos * linhas * src_bl);
        let mut esperado = vec![0u8; blocos * linhas * dst_bl];
        transformar(ty, &src, &mut esperado);
        let n_in = blocos * usize::try_from(ty.block_size()).unwrap();
        let t = upl.tensor(ty, &src, n_in, linhas).unwrap();
        assert_eq!(t.size_bytes as usize, esperado.len());
        subidos.push((t, esperado));
    }
    let mut mem = upl.finalizar().unwrap();

    for (i, (t, esperado)) in subidos.into_iter().enumerate() {
        let lido = ler_vram(&dev, t.buffer, esperado.len(), &ctx, p0);
        assert_eq!(lido, esperado, "tensor {i}");
        t.destroy(&dev.device);
    }
    mem.cleanup(&dev.device);
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
