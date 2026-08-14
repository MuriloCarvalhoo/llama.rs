//! Spike da Fase 8.0 — risco nº 1 da spec: latência de all-reduce MI50↔MI50.
//!
//! NÃO é teste de correção; é **medição**. Roda só com `--ignored` em hardware 2× MI50:
//! ```text
//! cargo test -p llama-vulkan --lib spike -- --ignored --nocapture
//! ```
//! A saída alimenta a decisão de mecanismo (host-staged vs peer-to-peer) da Fase 2.

#![cfg(test)]

use crate::device::{VulkanContext, VulkanDevice};
use crate::tensor::{alloc_and_bind, create_buf, one_shot_copy};
use ash::vk;
use std::time::Instant;

/// Payload de um all-reduce: a stream residual de 1 token. n_embd do Qwen2.5-14B.
const N_EMBD: usize = 5120;
const ITERS: u32 = 500;
/// Qwen2.5-14B: 48 camadas × 2 all-reduces (attn-out e ffn-down) = 96 por token.
const LAYERS: u32 = 48;
const ALLREDUCES_PER_LAYER: u32 = 2;

/// Buffer cru para o spike (não reusa `Buf` de resident_forward — cujos construtores
/// são privados ao módulo). Device-local ou host-visible, destruído explicitamente.
struct SpikeBuf {
    buffer: vk::Buffer,
    mem: vk::DeviceMemory,
}

impl SpikeBuf {
    fn new(
        ctx: &VulkanContext,
        phys: &crate::device::VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
        host_visible: bool,
    ) -> Self {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage).expect("create_buf");
        let mem = alloc_and_bind(ctx, phys, d, buffer, host_visible).expect("alloc_and_bind");
        Self { buffer, mem }
    }

    fn destroy(&self, d: &ash::Device) {
        // SAFETY: handles criados por nós; device ainda vivo, sem uso concorrente.
        unsafe {
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.mem, None);
        }
    }
}

/// Lê `bytes` de um buffer host-visible para `dst`.
fn map_read(d: &ash::Device, mem: vk::DeviceMemory, dst: &mut [f32]) {
    let bytes = std::mem::size_of_val(dst) as vk::DeviceSize;
    // SAFETY: `mem` é host-visible/coherent com ao menos `bytes`; ptr válido até unmap.
    unsafe {
        let ptr = d
            .map_memory(mem, 0, bytes, vk::MemoryMapFlags::empty())
            .expect("map");
        std::ptr::copy_nonoverlapping(ptr as *const f32, dst.as_mut_ptr(), dst.len());
        d.unmap_memory(mem);
    }
}

/// Escreve `src` num buffer host-visible.
fn map_write(d: &ash::Device, mem: vk::DeviceMemory, src: &[f32]) {
    let bytes = std::mem::size_of_val(src) as vk::DeviceSize;
    // SAFETY: `mem` é host-visible/coherent com ao menos `bytes`; ptr válido até unmap.
    unsafe {
        let ptr = d
            .map_memory(mem, 0, bytes, vk::MemoryMapFlags::empty())
            .expect("map");
        std::ptr::copy_nonoverlapping(src.as_ptr(), ptr as *mut f32, src.len());
        d.unmap_memory(mem);
    }
}

fn report(label: &str, elapsed: std::time::Duration, iters: u32, transfers_per_token: u32) {
    let per_us = elapsed.as_secs_f64() * 1e6 / f64::from(iters);
    let per_token_ms = per_us * f64::from(transfers_per_token) / 1000.0;
    let ceiling = 1000.0 / per_token_ms;
    println!(
        "{label:<38} {per_us:>8.2} µs/op  ->  {per_token_ms:>7.3} ms/token  ->  teto {ceiling:>8.1} tok/s"
    );
}

/// Contexto + 2 devices, ou `None` se o hardware não estiver disponível.
fn two_devices() -> Option<(VulkanContext, VulkanDevice, VulkanDevice)> {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan indisponível: {e} — pulando spike");
            return None;
        }
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("Menos de 2 GPUs AMD — pulando spike");
        return None;
    }
    let phys = ctx.amd_compute_devices();
    let dev0 = VulkanDevice::create(&ctx, &phys[0]).expect("dev0");
    let dev1 = VulkanDevice::create(&ctx, &phys[1]).expect("dev1");
    Some((ctx, dev0, dev1))
}

/// Piso da latência: só map/unmap de memória host-visible, sem cópia device-local.
/// Enquadra o limite inferior — nenhum mecanismo host-staged será mais rápido que isto.
#[test]
#[ignore = "requer 2x MI50 — cargo test -p llama-vulkan --lib spike -- --ignored --nocapture"]
fn spike_allreduce_host_bounce() {
    let Some((ctx, dev0, dev1)) = two_devices() else {
        return;
    };
    let phys = ctx.amd_compute_devices();
    let bytes = (N_EMBD * 4) as vk::DeviceSize;
    let (d0, d1) = (&dev0.device, &dev1.device);

    let src = SpikeBuf::new(&ctx, &phys[0], d0, bytes, true);
    let dst = SpikeBuf::new(&ctx, &phys[1], d1, bytes, true);
    let mut staging = vec![0.0f32; N_EMBD];

    for _ in 0..50 {
        map_read(d0, src.mem, &mut staging);
        map_write(d1, dst.mem, &staging);
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        map_read(d0, src.mem, &mut staging);
        map_write(d1, dst.mem, &staging);
    }
    let elapsed = t0.elapsed();

    println!("\n=== ALL-REDUCE SPIKE — payload {N_EMBD} f32 ({bytes} B) ===");
    report(
        "host-bounce (piso, só map)",
        elapsed,
        ITERS,
        LAYERS * ALLREDUCES_PER_LAYER,
    );

    src.destroy(d0);
    dst.destroy(d1);
}

/// Número honesto: payload em VRAM device-local dos dois lados, com as cópias reais
/// (device→host em GPU0, host→device em GPU1) e fence em cada submit.
#[test]
#[ignore = "requer 2x MI50 — cargo test -p llama-vulkan --lib spike -- --ignored --nocapture"]
fn spike_allreduce_device_local() {
    let Some((ctx, dev0, dev1)) = two_devices() else {
        return;
    };
    let phys = ctx.amd_compute_devices();
    let bytes = (N_EMBD * 4) as vk::DeviceSize;
    let (d0, d1) = (&dev0.device, &dev1.device);

    // GPU0: device-local (origem) + staging host. GPU1: staging host + device-local (destino).
    let src_dev = SpikeBuf::new(&ctx, &phys[0], d0, bytes, false);
    let stage0 = SpikeBuf::new(&ctx, &phys[0], d0, bytes, true);
    let stage1 = SpikeBuf::new(&ctx, &phys[1], d1, bytes, true);
    let dst_dev = SpikeBuf::new(&ctx, &phys[1], d1, bytes, false);
    let mut host = vec![0.0f32; N_EMBD];

    // Uma transferência unidirecional completa GPU0 -> GPU1.
    let one_way = |host: &mut Vec<f32>| {
        one_shot_copy(
            d0,
            dev0.queue,
            dev0.cmd_pool,
            src_dev.buffer,
            stage0.buffer,
            bytes,
        )
        .expect("copy gpu0->stage0");
        map_read(d0, stage0.mem, host);
        map_write(d1, stage1.mem, host);
        one_shot_copy(
            d1,
            dev1.queue,
            dev1.cmd_pool,
            stage1.buffer,
            dst_dev.buffer,
            bytes,
        )
        .expect("copy stage1->gpu1");
    };

    for _ in 0..20 {
        one_way(&mut host);
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        one_way(&mut host);
    }
    let elapsed = t0.elapsed();

    // 1 transferência unidirecional por all-reduce é o piso otimista; um all-reduce
    // real de 2 GPUs troca nos dois sentidos, por isso reportamos as duas leituras.
    report(
        "device-local, 1 via (otimista)",
        elapsed,
        ITERS,
        LAYERS * ALLREDUCES_PER_LAYER,
    );
    report(
        "device-local, 2 vias (all-reduce real)",
        elapsed,
        ITERS,
        LAYERS * ALLREDUCES_PER_LAYER * 2,
    );

    src_dev.destroy(d0);
    stage0.destroy(d0);
    stage1.destroy(d1);
    dst_dev.destroy(d1);
}

/// Submete um command buffer vazio e espera o fence. Isola o custo de sincronização.
fn empty_submit(dev: &VulkanDevice) {
    let d = &dev.device;
    let alloc = vk::CommandBufferAllocateInfo {
        command_pool: dev.cmd_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    // SAFETY: device/pool válidos; cmd é gravado e liberado nesta mesma função.
    unsafe {
        let cmd = d.allocate_command_buffers(&alloc).expect("alloc cmd")[0];
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        d.begin_command_buffer(cmd, &begin).expect("begin");
        d.end_command_buffer(cmd).expect("end");
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        d.queue_submit(dev.queue, &[submit], vk::Fence::null())
            .expect("submit");
        d.queue_wait_idle(dev.queue).expect("wait");
        d.free_command_buffers(dev.cmd_pool, &[cmd]);
    }
}

/// Decompõe os 247 µs do all-reduce: quanto é sincronização (submit+fence) e quanto
/// é transferência real? Decide se batching (1 command buffer/token) resolve o gargalo.
#[test]
#[ignore = "requer 2x MI50 — cargo test -p llama-vulkan --lib spike -- --ignored --nocapture"]
fn spike_sync_overhead_decomposition() {
    let Some((ctx, dev0, _dev1)) = two_devices() else {
        return;
    };
    let phys = ctx.amd_compute_devices();
    let d0 = &dev0.device;

    println!("\n=== DECOMPOSIÇÃO DO CUSTO (GPU0) ===");

    // 1) Submit + fence de um command buffer VAZIO — puro custo de sincronização.
    for _ in 0..20 {
        empty_submit(&dev0);
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        empty_submit(&dev0);
    }
    let sync_us = t.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
    println!("submit+fence vazio (sync puro)          {sync_us:>8.2} µs");

    // 2) Cópia device->host de tamanhos diferentes: separa latência de banda.
    for &n in &[1usize, N_EMBD, N_EMBD * 16] {
        let bytes = (n * 4) as vk::DeviceSize;
        let src = SpikeBuf::new(&ctx, &phys[0], d0, bytes, false);
        let dst = SpikeBuf::new(&ctx, &phys[0], d0, bytes, true);
        let copy = || {
            one_shot_copy(d0, dev0.queue, dev0.cmd_pool, src.buffer, dst.buffer, bytes)
                .expect("copy");
        };
        for _ in 0..20 {
            copy();
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            copy();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
        println!(
            "copy device->host {:>9} B            {us:>8.2} µs  (sync = {:.0}%)",
            bytes,
            100.0 * sync_us / us
        );
        src.destroy(d0);
        dst.destroy(d0);
    }

    // 3) map/unmap puro, sem GPU envolvida.
    let bytes = (N_EMBD * 4) as vk::DeviceSize;
    let hostbuf = SpikeBuf::new(&ctx, &phys[0], d0, bytes, true);
    let mut tmp = vec![0.0f32; N_EMBD];
    for _ in 0..20 {
        map_read(d0, hostbuf.mem, &mut tmp);
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        map_read(d0, hostbuf.mem, &mut tmp);
    }
    let map_us = t.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
    println!("map+copy+unmap {N_EMBD} f32 (sem GPU)     {map_us:>8.2} µs");
    hostbuf.destroy(d0);
}

/// O número que decide a Fase 2: se as 96 transferências de um token forem gravadas
/// num **único** command buffer (1 submit, 1 fence), o custo de sincronização é pago
/// uma vez em vez de 96. Compara com o caminho ingênuo (1 submit por transferência).
#[test]
#[ignore = "requer 2x MI50 — cargo test -p llama-vulkan --lib spike -- --ignored --nocapture"]
fn spike_batched_transfers_per_token() {
    let Some((ctx, dev0, _dev1)) = two_devices() else {
        return;
    };
    let phys = ctx.amd_compute_devices();
    let d = &dev0.device;
    let bytes = (N_EMBD * 4) as vk::DeviceSize;
    let n_xfer = LAYERS * ALLREDUCES_PER_LAYER; // 96

    let src = SpikeBuf::new(&ctx, &phys[0], d, bytes, false);
    let dst = SpikeBuf::new(&ctx, &phys[0], d, bytes, true);

    // Grava `n_xfer` cópias num único command buffer, submete uma vez, espera uma vez.
    let batched = || {
        let alloc = vk::CommandBufferAllocateInfo {
            command_pool: dev0.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: device/pool válidos; cmd gravado, submetido e liberado aqui.
        unsafe {
            let cmd = d.allocate_command_buffers(&alloc).expect("alloc")[0];
            let begin = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            d.begin_command_buffer(cmd, &begin).expect("begin");
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: bytes,
            };
            for _ in 0..n_xfer {
                d.cmd_copy_buffer(cmd, src.buffer, dst.buffer, &[region]);
            }
            d.end_command_buffer(cmd).expect("end");
            let submit = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };
            d.queue_submit(dev0.queue, &[submit], vk::Fence::null())
                .expect("submit");
            d.queue_wait_idle(dev0.queue).expect("wait");
            d.free_command_buffers(dev0.cmd_pool, &[cmd]);
        }
    };

    for _ in 0..10 {
        batched();
    }
    const B_ITERS: u32 = 100;
    let t = Instant::now();
    for _ in 0..B_ITERS {
        batched();
    }
    let per_token_ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(B_ITERS);

    println!("\n=== BATCHING: {n_xfer} transferências de um token ===");
    println!(
        "1 command buffer, 1 submit, 1 fence      {per_token_ms:>7.3} ms/token  ->  teto {:>8.1} tok/s",
        1000.0 / per_token_ms
    );
    println!(
        "(ingênuo, 1 submit por transferência)    {:>7.3} ms/token  ->  teto {:>8.1} tok/s",
        23.755, 42.1
    );

    src.destroy(d);
    dst.destroy(d);
}

/// Existe caminho peer-to-peer (sem bounce pelo host) entre as duas GPUs?
/// Decide se a Fase 2 pode evitar o host no all-reduce.
#[test]
#[ignore = "requer 2x MI50 — cargo test -p llama-vulkan --lib spike -- --ignored --nocapture"]
fn probe_peer_to_peer_caps() {
    let Some((ctx, _dev0, _dev1)) = two_devices() else {
        return;
    };
    println!("\n=== PROBE peer-to-peer ===");
    for (i, p) in ctx.amd_compute_devices().iter().take(2).enumerate() {
        // SAFETY: instance e handle são válidos (criados/enumerados por VulkanContext).
        let exts = unsafe {
            ctx.instance
                .enumerate_device_extension_properties(p.handle)
                .expect("enumerate exts")
        };
        let has = |name: &str| {
            exts.iter().any(|e| {
                // SAFETY: extension_name é nul-terminado pela spec Vulkan.
                let s = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
                s.to_string_lossy() == name
            })
        };
        println!(
            "GPU{i} ({}): external_memory={} external_memory_fd={} dma_buf={} \
             external_semaphore_fd={}",
            p.name(),
            has("VK_KHR_external_memory"),
            has("VK_KHR_external_memory_fd"),
            has("VK_EXT_external_memory_dma_buf"),
            has("VK_KHR_external_semaphore_fd"),
        );
    }
}

/// Cria um device logico com as extensoes de memoria/semaforo externos habilitadas.
/// `VulkanDevice::create` nao habilita extensao nenhuma, entao o spike monta o seu.
fn device_with_external(
    ctx: &VulkanContext,
    phys: &crate::device::VulkanPhysicalDevice,
) -> (ash::Device, vk::Queue, u32) {
    let qf = phys.queue_family;
    let prio = [1.0f32];
    let qinfo = vk::DeviceQueueCreateInfo {
        queue_family_index: qf,
        queue_count: 1,
        p_queue_priorities: prio.as_ptr(),
        ..Default::default()
    };
    let exts = [
        c"VK_KHR_external_memory_fd".as_ptr(),
        c"VK_KHR_external_semaphore_fd".as_ptr(),
    ];
    let info = vk::DeviceCreateInfo {
        queue_create_info_count: 1,
        p_queue_create_infos: &qinfo,
        enabled_extension_count: exts.len() as u32,
        pp_enabled_extension_names: exts.as_ptr(),
        ..Default::default()
    };
    // SAFETY: handle valido; info e qinfo vivem nesta stack frame durante a chamada.
    let dev = unsafe {
        ctx.instance
            .create_device(phys.handle, &info, None)
            .expect("device com extensoes externas")
    };
    // SAFETY: device recem-criado com essa queue family.
    let q = unsafe { dev.get_device_queue(qf, 0) };
    (dev, q, qf)
}

/// Encontra um memory type compativel com `bits`, preferindo DEVICE_LOCAL.
/// Retorna `(indice, flags)` — as flags dizem se a memoria importada de fato caiu em
/// VRAM do peer ou em memoria do host.
fn pick_memory_type(
    ctx: &VulkanContext,
    phys: &crate::device::VulkanPhysicalDevice,
    bits: u32,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    // SAFETY: handle valido.
    let mp = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(phys.handle)
    };
    let compat: Vec<u32> = (0..mp.memory_type_count)
        .filter(|&i| bits & (1 << i) != 0)
        .collect();
    let pick = compat
        .iter()
        .find(|&&i| {
            mp.memory_types[i as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .or_else(|| compat.first())?;
    Some((*pick, mp.memory_types[*pick as usize].property_flags))
}

fn device_local_type(
    ctx: &VulkanContext,
    phys: &crate::device::VulkanPhysicalDevice,
    bits: u32,
) -> u32 {
    pick_memory_type(ctx, phys, bits)
        .expect("memory type compativel")
        .0
}

/// A pergunta que decide o tensor-parallel: a GPU1 consegue LER a VRAM da GPU0
/// diretamente (peer-to-peer), e a que velocidade?
///
/// Se sim, o all-reduce por camada troca dados sem passar pelo host. Se nao, resta o
/// host-staged, que a 63 us por sincronizacao (medido em spike_sync_overhead) custaria
/// mais que o proprio ganho de dividir os pesos.
#[test]
#[ignore = "requer 2x MI50 — cargo test -p llama-vulkan --lib spike -- --ignored --nocapture"]
fn spike_p2p_import_memory() {
    for ht in [
        ("OPAQUE_FD", vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
        ("DMA_BUF", vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
    ] {
        p2p_try(ht.0, ht.1);
    }
}

fn p2p_try(ht_name: &str, ht: vk::ExternalMemoryHandleTypeFlags) {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan indisponivel: {e} — pulando");
            return;
        }
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("Menos de 2 GPUs AMD — pulando");
        return;
    }
    let phys = ctx.amd_compute_devices();
    let (d0, _q0, _qf0) = device_with_external(&ctx, &phys[0]);
    let (d1, q1, qf1) = device_with_external(&ctx, &phys[1]);

    const BYTES: vk::DeviceSize = 64 * 1024 * 1024; // 64 MB: mede banda, nao latencia

    // --- GPU0: buffer device-local com memoria EXPORTAVEL ---
    let ext_buf_info = vk::ExternalMemoryBufferCreateInfo {
        handle_types: ht,
        ..Default::default()
    };
    let buf_info = vk::BufferCreateInfo {
        p_next: std::ptr::from_ref(&ext_buf_info).cast(),
        size: BYTES,
        usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    // SAFETY: d0 valido; infos vivos nesta frame.
    let src_buf = unsafe { d0.create_buffer(&buf_info, None).expect("buffer src") };
    // SAFETY: buffer recem-criado.
    let req = unsafe { d0.get_buffer_memory_requirements(src_buf) };
    let export_info = vk::ExportMemoryAllocateInfo {
        handle_types: ht,
        ..Default::default()
    };
    let alloc = vk::MemoryAllocateInfo {
        p_next: std::ptr::from_ref(&export_info).cast(),
        allocation_size: req.size,
        memory_type_index: device_local_type(&ctx, &phys[0], req.memory_type_bits),
        ..Default::default()
    };
    // SAFETY: d0 valido; alloc vive nesta frame.
    let src_mem = unsafe { d0.allocate_memory(&alloc, None).expect("aloca exportavel") };
    // SAFETY: buffer e memoria do mesmo device, tamanhos compativeis.
    unsafe {
        d0.bind_buffer_memory(src_buf, src_mem, 0)
            .expect("bind src")
    };

    // --- exporta o fd ---
    let fd_api0 = ash::khr::external_memory_fd::Device::new(&ctx.instance, &d0);
    let get_fd = vk::MemoryGetFdInfoKHR {
        memory: src_mem,
        handle_type: ht,
        ..Default::default()
    };
    // SAFETY: memoria alocada com handle type OPAQUE_FD.
    let fd = match unsafe { fd_api0.get_memory_fd(&get_fd) } {
        Ok(fd) => fd,
        Err(e) => {
            println!("\n[{ht_name}] export FALHOU ({e:?})");
            return;
        }
    };
    println!("\n=== P2P entre as duas MI50 — handle {ht_name} ===");
    println!("export de memoria da GPU0: ok (fd={fd})");

    // --- GPU1: importa o fd ---
    let fd_api1 = ash::khr::external_memory_fd::Device::new(&ctx.instance, &d1);
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    // SAFETY: fd valido, obtido acima.
    let import_ok = unsafe { fd_api1.get_memory_fd_properties(ht, fd, &mut fd_props) };
    if import_ok.is_err() {
        println!("import na GPU1: FALHOU ({import_ok:?}) — sem P2P, all-reduce via host");
        return;
    }
    println!(
        "import na GPU1: ok (memory_type_bits={:#x})",
        fd_props.memory_type_bits
    );

    let imported_buf = unsafe { d1.create_buffer(&buf_info, None).expect("buffer importado") };
    // SAFETY: buffer recem-criado em d1.
    let req1 = unsafe { d1.get_buffer_memory_requirements(imported_buf) };
    let mut import_info = vk::ImportMemoryFdInfoKHR {
        handle_type: ht,
        fd,
        ..Default::default()
    };
    let Some((mt, flags)) = pick_memory_type(
        &ctx,
        &phys[1],
        req1.memory_type_bits & fd_props.memory_type_bits,
    ) else {
        println!("nenhum memory type compativel na GPU1 — sem P2P");
        return;
    };
    println!("memory type da importacao: indice {mt}, flags {flags:?}");
    let alloc1 = vk::MemoryAllocateInfo {
        p_next: std::ptr::from_mut(&mut import_info).cast(),
        allocation_size: req.size,
        memory_type_index: mt,
        ..Default::default()
    };
    // SAFETY: fd valido e ainda nao consumido por outra importacao.
    let imported_mem = match unsafe { d1.allocate_memory(&alloc1, None) } {
        Ok(m) => m,
        Err(e) => {
            println!("alocacao importada na GPU1: FALHOU ({e:?}) — sem P2P");
            return;
        }
    };
    // SAFETY: mesma memoria/tamanho.
    unsafe {
        d1.bind_buffer_memory(imported_buf, imported_mem, 0)
            .expect("bind importado");
    }
    println!("bind da memoria da GPU0 na GPU1: ok");

    // --- mede a banda de leitura P2P: copia importado(GPU0) -> local(GPU1) ---
    let dst_info = vk::BufferCreateInfo {
        size: BYTES,
        usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    // SAFETY: d1 valido.
    let dst_buf = unsafe { d1.create_buffer(&dst_info, None).expect("buffer dst") };
    // SAFETY: buffer recem-criado.
    let dreq = unsafe { d1.get_buffer_memory_requirements(dst_buf) };
    let dalloc = vk::MemoryAllocateInfo {
        allocation_size: dreq.size,
        memory_type_index: device_local_type(&ctx, &phys[1], dreq.memory_type_bits),
        ..Default::default()
    };
    // SAFETY: d1 valido.
    let dst_mem = unsafe { d1.allocate_memory(&dalloc, None).expect("aloca dst") };
    // SAFETY: mesmos device/tamanho.
    unsafe {
        d1.bind_buffer_memory(dst_buf, dst_mem, 0)
            .expect("bind dst")
    };

    let pool_info = vk::CommandPoolCreateInfo {
        queue_family_index: qf1,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    // SAFETY: d1 valido.
    let pool = unsafe { d1.create_command_pool(&pool_info, None).expect("pool") };

    let copy = || {
        one_shot_copy(&d1, q1, pool, imported_buf, dst_buf, BYTES).expect("copy p2p");
    };
    for _ in 0..3 {
        copy();
    }
    const N: u32 = 20;
    let t = Instant::now();
    for _ in 0..N {
        copy();
    }
    let per = t.elapsed().as_secs_f64() / f64::from(N);
    println!(
        "leitura P2P GPU0->GPU1: {:.2} ms para {} MB  =>  {:.1} GB/s",
        per * 1000.0,
        BYTES / 1024 / 1024,
        BYTES as f64 / per / 1e9
    );
    println!("(PCIe 4.0 x16 ~25 GB/s; HBM local ~717 GB/s medidos no matvec)");

    // SAFETY: GPU ociosa apos as copias; handles criados por nos.
    unsafe {
        let _ = d1.device_wait_idle();
        d1.destroy_command_pool(pool, None);
        d1.destroy_buffer(dst_buf, None);
        d1.free_memory(dst_mem, None);
        d1.destroy_buffer(imported_buf, None);
        d1.free_memory(imported_mem, None);
        d1.destroy_device(None);
        d0.destroy_buffer(src_buf, None);
        d0.free_memory(src_mem, None);
        d0.destroy_device(None);
    }
}

/// Ultima incognita do tensor-parallel: quanto custa uma sincronizacao GPU->GPU.
///
/// Se as GPUs so conseguirem se sincronizar via host (submit+fence de cada lado, ~63 us
/// medidos em `spike_sync_overhead_decomposition`), os 96 all-reduces de um token do 14B
/// custam ~12 ms — mais que os ~10.8 ms que dividir os pesos economizaria. Com semaforo
/// externo a espera acontece na propria GPU e o custo deveria cair para poucos us.
#[test]
#[ignore = "requer 2x MI50 — cargo test -p llama-vulkan --lib spike -- --ignored --nocapture"]
fn spike_cross_device_semaphore() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan indisponivel: {e} — pulando");
            return;
        }
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("Menos de 2 GPUs AMD — pulando");
        return;
    }
    let phys = ctx.amd_compute_devices();
    let (d0, q0, qf0) = device_with_external(&ctx, &phys[0]);
    let (d1, q1, qf1) = device_with_external(&ctx, &phys[1]);

    println!("\n=== Sincronizacao GPU0 -> GPU1 por semaforo externo ===");

    // Semaforo exportavel na GPU0.
    let export = vk::ExportSemaphoreCreateInfo {
        handle_types: vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
        ..Default::default()
    };
    let sem_info = vk::SemaphoreCreateInfo {
        p_next: std::ptr::from_ref(&export).cast(),
        ..Default::default()
    };
    // SAFETY: d0 valido; infos vivas nesta frame.
    let sem0 = unsafe { d0.create_semaphore(&sem_info, None).expect("semaforo") };

    let sem_api0 = ash::khr::external_semaphore_fd::Device::new(&ctx.instance, &d0);
    let get = vk::SemaphoreGetFdInfoKHR {
        semaphore: sem0,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
        ..Default::default()
    };
    // SAFETY: semaforo criado com handle type OPAQUE_FD.
    let fd = match unsafe { sem_api0.get_semaphore_fd(&get) } {
        Ok(fd) => fd,
        Err(e) => {
            println!("export do semaforo FALHOU ({e:?}) — sincronizacao so via host");
            return;
        }
    };
    println!("export do semaforo da GPU0: ok (fd={fd})");

    // Importa na GPU1.
    // SAFETY: d1 valido.
    let sem1 = unsafe {
        d1.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .expect("semaforo destino")
    };
    let sem_api1 = ash::khr::external_semaphore_fd::Device::new(&ctx.instance, &d1);
    let import = vk::ImportSemaphoreFdInfoKHR {
        semaphore: sem1,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
        fd,
        ..Default::default()
    };
    // SAFETY: fd valido, obtido acima.
    if let Err(e) = unsafe { sem_api1.import_semaphore_fd(&import) } {
        println!("import do semaforo na GPU1: FALHOU ({e:?}) — sincronizacao so via host");
        println!("=> tensor-parallel dependeria de 96 round-trips pelo host por token");
        return;
    }
    println!("import na GPU1: ok — as GPUs conseguem se sincronizar sem o host");

    // Ping: GPU0 sinaliza, GPU1 espera. Mede o custo de uma sincronizacao.
    let mk_pool = |d: &ash::Device, qf: u32| {
        let i = vk::CommandPoolCreateInfo {
            queue_family_index: qf,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        // SAFETY: device valido.
        unsafe { d.create_command_pool(&i, None).expect("pool") }
    };
    let p0 = mk_pool(&d0, qf0);
    let p1 = mk_pool(&d1, qf1);

    let empty_cmd = |d: &ash::Device, pool: vk::CommandPool| {
        let a = vk::CommandBufferAllocateInfo {
            command_pool: pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: device/pool validos; cmd gravado vazio.
        unsafe {
            let c = d.allocate_command_buffers(&a).expect("cmd")[0];
            d.begin_command_buffer(c, &vk::CommandBufferBeginInfo::default())
                .expect("begin");
            d.end_command_buffer(c).expect("end");
            c
        }
    };
    let c0 = empty_cmd(&d0, p0);
    let c1 = empty_cmd(&d1, p1);

    // SAFETY: d1 valido.
    let fence1 = unsafe {
        d1.create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("fence")
    };

    // Variante pipelinada: K pares dependentes em voo, um unico wait de host no fim.
    // Semaforo binario so pode ser sinalizado uma vez por espera, entao sao K pares.
    const K: usize = 64;
    let mut sems0 = Vec::with_capacity(K);
    let mut sems1 = Vec::with_capacity(K);
    for _ in 0..K {
        // SAFETY: devices validos; o fd exportado e importado uma unica vez cada.
        unsafe {
            let a = d0.create_semaphore(&sem_info, None).expect("sem a");
            let g = vk::SemaphoreGetFdInfoKHR {
                semaphore: a,
                handle_type: vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
                ..Default::default()
            };
            let f = sem_api0.get_semaphore_fd(&g).expect("fd");
            let b = d1
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .expect("sem b");
            let imp = vk::ImportSemaphoreFdInfoKHR {
                semaphore: b,
                handle_type: vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
                fd: f,
                ..Default::default()
            };
            sem_api1.import_semaphore_fd(&imp).expect("import");
            sems0.push(a);
            sems1.push(b);
        }
    }
    let ws = vk::PipelineStageFlags::ALL_COMMANDS;
    let t_pipe = Instant::now();
    // SAFETY: cmds gravados; cada semaforo e sinalizado e aguardado exatamente uma vez.
    unsafe {
        for i in 0..K {
            let s0 = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &c0,
                signal_semaphore_count: 1,
                p_signal_semaphores: &sems0[i],
                ..Default::default()
            };
            d0.queue_submit(q0, &[s0], vk::Fence::null())
                .expect("submit0");
            let s1 = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &sems1[i],
                p_wait_dst_stage_mask: &ws,
                command_buffer_count: 1,
                p_command_buffers: &c1,
                ..Default::default()
            };
            let fence = if i == K - 1 {
                fence1
            } else {
                vk::Fence::null()
            };
            if i == K - 1 {
                d1.reset_fences(&[fence1]).expect("reset");
            }
            d1.queue_submit(q1, &[s1], fence).expect("submit1");
        }
        d1.wait_for_fences(&[fence1], true, u64::MAX).expect("wait");
    }
    let pipe_us = t_pipe.elapsed().as_secs_f64() * 1e6 / K as f64;
    println!("pipelinado ({K} pares, 1 wait de host no fim): {pipe_us:.2} us por sincronizacao");
    println!(
        "  -> 96 all-reduces/token: {:.2} ms/token",
        pipe_us * 96.0 / 1000.0
    );
    // SAFETY: GPUs ociosas apos o fence.
    unsafe {
        for i in 0..K {
            d0.destroy_semaphore(sems0[i], None);
            d1.destroy_semaphore(sems1[i], None);
        }
    }

    const N: u32 = 200;
    let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
    let t = Instant::now();
    for _ in 0..N {
        // SAFETY: cmds gravados; semaforo sinalizado por d0 e aguardado por d1 uma vez.
        unsafe {
            let s0 = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &c0,
                signal_semaphore_count: 1,
                p_signal_semaphores: &sem0,
                ..Default::default()
            };
            d0.queue_submit(q0, &[s0], vk::Fence::null())
                .expect("submit0");
            let s1 = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &sem1,
                p_wait_dst_stage_mask: &wait_stage,
                command_buffer_count: 1,
                p_command_buffers: &c1,
                ..Default::default()
            };
            d1.reset_fences(&[fence1]).expect("reset");
            d1.queue_submit(q1, &[s1], fence1).expect("submit1");
            d1.wait_for_fences(&[fence1], true, u64::MAX).expect("wait");
        }
    }
    let per_us = t.elapsed().as_secs_f64() * 1e6 / f64::from(N);
    let per_token_ms = per_us * 96.0 * 2.0 / 1000.0;
    println!("sincronizacao GPU0->GPU1: {per_us:.2} us");
    println!(
        "96 all-reduces/token (2 sync cada): {per_token_ms:.2} ms/token \
         — comparar com os ~10.8 ms que dividir os pesos economizaria"
    );

    // SAFETY: GPUs ociosas; handles criados por nos.
    unsafe {
        let _ = d0.device_wait_idle();
        let _ = d1.device_wait_idle();
        d1.destroy_fence(fence1, None);
        d1.destroy_command_pool(p1, None);
        d1.destroy_semaphore(sem1, None);
        d1.destroy_device(None);
        d0.destroy_command_pool(p0, None);
        d0.destroy_semaphore(sem0, None);
        d0.destroy_device(None);
    }
}
