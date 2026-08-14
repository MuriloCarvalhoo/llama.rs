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
