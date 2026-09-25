//! Bancada de banda dos matvec K-quant: um dispatch isolado, repetido com barreira entre as
//! repetições (como no plano do token) e cronometrado por timestamps de GPU. Mede GB/s por
//! forma de matriz contra o teto de leitura da placa — 836 GB/s medidos com
//! `scripts/teto-mi50.sh`, não os 1024 nominais.
//!
//! Só testes `#[ignore]`, porque ocupam a GPU e não verificam nada:
//! `cargo test --release -p llama-vulkan --lib bancada -- --ignored --nocapture --test-threads=1`
//!
//! Roda na GPU **sem** monitor (margem de 500 MiB): o compositor da outra disputa a banda e
//! suja a medição.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::pipeline::{ComputePipeline, PushConstants};
use crate::tensor::{alloc_and_bind, create_buf, one_shot_copy};
use ash::vk;

/// Teto de leitura medido com `scripts/teto-mi50.sh` (2026-09-25).
const TETO_GBS: f64 = 836.0;
const REPS: u32 = 200;

/// Formas dos matvec do Qwen3.8-27B Q4_K_M, na ordem do que pesam no token.
const FORMAS: [(&str, usize, usize); 7] = [
    ("ffn_gate/up", 5120, 17408),
    ("ffn_down", 17408, 5120),
    ("attn_q", 5120, 12288),
    ("attn_qkv", 5120, 10240),
    ("attn_gate", 5120, 6144),
    ("attn_output", 6144, 5120),
    ("attn_k/v", 5120, 1024),
];

/// Gerador determinístico — os valores não importam para o tempo, só a forma.
struct Xs(u64);
impl Xs {
    fn prox(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Pesos Q4_K com `d`/`dmin` finitos e o resto aleatório: 144 B por superbloco.
pub(crate) fn pesos_q4k(n_in: usize, n_out: usize, semente: u64) -> Vec<u8> {
    let n_sb = n_in / 256 * n_out;
    let mut r = Xs(semente | 1);
    let mut w = vec![0u8; n_sb * 144];
    for sb in w.as_chunks_mut::<144>().0 {
        sb[0..2].copy_from_slice(&0x2C00u16.to_le_bytes()); // d = 0.0625
        sb[2..4].copy_from_slice(&0x2400u16.to_le_bytes()); // dmin = 0.015625
        for b in &mut sb[4..] {
            *b = r.prox() as u8;
        }
    }
    w
}

pub(crate) fn ativacao(n: usize, semente: u64) -> Vec<f32> {
    let mut r = Xs(semente | 1);
    (0..n)
        .map(|_| (r.prox() % 2001) as f32 / 1000.0 - 1.0)
        .collect()
}

/// O device sem monitor, ou o primeiro se não houver como saber.
pub(crate) fn device_sem_monitor(ctx: &VulkanContext) -> Option<&VulkanPhysicalDevice> {
    let devs = ctx.amd_compute_devices();
    devs.iter()
        .find(|p| p.margem_vram() < 1 << 30)
        .or(devs.first())
}

/// Sobe `bytes` para um STORAGE_BUFFER device-local via staging descartável.
fn subir(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    bytes: &[u8],
) -> (vk::Buffer, vk::DeviceMemory) {
    let d = &dev.device;
    let size = bytes.len() as vk::DeviceSize;
    let staging = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_SRC).unwrap();
    let staging_mem = alloc_and_bind(ctx, phys, d, staging, true).unwrap();
    // SAFETY: staging_mem é host-visible com `size`; ptr válido até unmap.
    unsafe {
        let ptr = d
            .map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
        d.unmap_memory(staging_mem);
    }
    let buf = create_buf(
        d,
        size,
        vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC,
    )
    .unwrap();
    let mem = alloc_and_bind(ctx, phys, d, buf, false).unwrap();
    one_shot_copy(d, dev.queue, dev.cmd_pool, staging, buf, size).unwrap();
    // SAFETY: cópia concluída (one_shot_copy espera a fila); handles criados aqui.
    unsafe {
        d.destroy_buffer(staging, None);
        d.free_memory(staging_mem, None);
    }
    (buf, mem)
}

/// Um shader de matvec K-quant na bancada: pipeline, bindings (pesos, xq, xd, y, bias) e
/// geometria do dispatch.
#[derive(Clone, Copy)]
pub(crate) struct Kernel<'a> {
    pub spv: &'a [u8],
    pub spec: &'a [(u32, u32)],
    pub rows_por_wg: u32,
}

/// Roda `k` sobre `w` (já no layout do shader) e devolve `(µs por dispatch, y)`.
// Contexto Vulkan (3) + kernel + dados (2) + dimensões (3): mesmo critério do
// `dispatch_k_matvec`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn medir(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    k: &Kernel<'_>,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    cols: usize,
) -> (f64, Vec<f32>) {
    let d = &dev.device;
    let (xq, xd) = crate::tensor::quantize_x_host(x);
    // SAFETY: Vec<u32>/Vec<f32> são POD contíguos; reinterpretar como bytes é válido.
    let xq_b = unsafe { std::slice::from_raw_parts(xq.as_ptr().cast::<u8>(), xq.len() * 4) };
    let xd_b = unsafe { std::slice::from_raw_parts(xd.as_ptr().cast::<u8>(), xd.len() * 4) };
    let bufs = [
        subir(ctx, phys, dev, w),
        subir(ctx, phys, dev, xq_b),
        subir(ctx, phys, dev, xd_b),
        subir(ctx, phys, dev, &vec![0u8; cols * n_out * 4]),
    ];
    let pipe =
        ComputePipeline::with(d, k.spv, 5, size_of::<PushConstants>() as u32, k.spec).unwrap();
    let push = PushConstants {
        n_in: n_in as u32,
        n_out: n_out as u32,
        row_offset: 0,
        tem_bias: 0,
    };
    let tamanhos = [
        w.len(),
        xq_b.len(),
        xd_b.len(),
        cols * n_out * 4,
        xd_b.len(),
    ];
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 5,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo {
        max_sets: 1,
        pool_size_count: 1,
        p_pool_sizes: pool_sizes.as_ptr(),
        ..Default::default()
    };
    let qp_info = vk::QueryPoolCreateInfo {
        query_type: vk::QueryType::TIMESTAMP,
        query_count: 2,
        ..Default::default()
    };
    // SAFETY: instance e handle válidos (enumerados por VulkanContext).
    let periodo_ns = unsafe {
        ctx.instance
            .get_physical_device_properties(phys.handle)
            .limits
            .timestamp_period
    };
    // SAFETY: todos os handles abaixo são criados, usados e destruídos nesta frame, com a
    // fila esperada (`queue_wait_idle`) antes de qualquer destruição.
    unsafe {
        let desc_pool = d.create_descriptor_pool(&pool_info, None).unwrap();
        let set = d
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: desc_pool,
                descriptor_set_count: 1,
                p_set_layouts: &pipe.desc_set_layout,
                ..Default::default()
            })
            .unwrap()[0];
        // Binding 4 (bias) recebe `xd` só para completar o layout; `tem_bias: 0` o ignora.
        let infos: Vec<vk::DescriptorBufferInfo> = [0usize, 1, 2, 3, 2]
            .iter()
            .zip(tamanhos)
            .map(|(&b, t)| vk::DescriptorBufferInfo {
                buffer: bufs[b].0,
                offset: 0,
                range: t as vk::DeviceSize,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(b, i)| vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: b as u32,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: i,
                ..Default::default()
            })
            .collect();
        d.update_descriptor_sets(&writes, &[]);
        let qp = d.create_query_pool(&qp_info, None).unwrap();

        let grupos = push.n_out.div_ceil(k.rows_por_wg);
        let mut us = 0.0;
        // Primeira submissão aquece (clocks, caches de pipeline); a segunda é a medida.
        for _ in 0..2 {
            let cmd = d
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    command_pool: dev.cmd_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                })
                .unwrap()[0];
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
            .unwrap();
            d.cmd_reset_query_pool(cmd, qp, 0, 2);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            d.cmd_push_constants(
                cmd,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    std::ptr::from_ref(&push).cast::<u8>(),
                    size_of::<PushConstants>(),
                ),
            );
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            let barreira = vk::MemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ..Default::default()
            };
            for _ in 0..REPS {
                d.cmd_dispatch(cmd, grupos, 1, 1);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barreira],
                    &[],
                    &[],
                );
            }
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
            d.end_command_buffer(cmd).unwrap();
            d.queue_submit(
                dev.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                }],
                vk::Fence::null(),
            )
            .unwrap();
            d.queue_wait_idle(dev.queue).unwrap();
            d.free_command_buffers(dev.cmd_pool, &[cmd]);
            let mut ts = [0u64; 2];
            d.get_query_pool_results(
                qp,
                0,
                &mut ts,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
            .unwrap();
            us = (ts[1] - ts[0]) as f64 * f64::from(periodo_ns) / 1e3 / f64::from(REPS);
        }

        // Readback de y, para comparar variantes do shader entre si.
        let y_bytes = (cols * n_out * 4) as u64;
        let read = create_buf(d, y_bytes, vk::BufferUsageFlags::TRANSFER_DST).unwrap();
        let read_mem = alloc_and_bind(ctx, phys, d, read, true).unwrap();
        one_shot_copy(d, dev.queue, dev.cmd_pool, bufs[3].0, read, y_bytes).unwrap();
        let ptr = d
            .map_memory(read_mem, 0, y_bytes, vk::MemoryMapFlags::empty())
            .unwrap();
        let y = std::slice::from_raw_parts(ptr.cast::<f32>(), cols * n_out).to_vec();
        d.unmap_memory(read_mem);
        d.destroy_buffer(read, None);
        d.free_memory(read_mem, None);

        d.destroy_query_pool(qp, None);
        d.destroy_descriptor_pool(desc_pool, None);
        pipe.destroy(d);
        for (b, m) in bufs {
            d.destroy_buffer(b, None);
            d.free_memory(m, None);
        }
        (us, y)
    }
}

fn linha(nome: &str, n_in: usize, n_out: usize, bytes: usize, us: f64) {
    let gbs = bytes as f64 / (us * 1e3);
    eprintln!(
        "{nome:<12} {n_in:>6}->{n_out:<6} {:>7.1} MB {us:>8.1} µs {gbs:>6.0} GB/s {:>4.0}% do teto",
        bytes as f64 / 1e6,
        100.0 * gbs / TETO_GBS
    );
}

#[test]
#[ignore = "bancada de desempenho: ocupa a GPU e só imprime"]
fn bancada_q4k() {
    let Ok(ctx) = VulkanContext::new() else {
        return;
    };
    let Some(phys) = device_sem_monitor(&ctx) else {
        return;
    };
    let dev = VulkanDevice::create(&ctx, phys).unwrap();
    let (wg, rows) = crate::resident_forward::matvec_geom();
    let spec = [
        (0, wg),
        (1, rows),
        (2, 1),
        (3, crate::resident_forward::matvec_lds_pad()),
    ];
    let k = Kernel {
        spv: crate::Q4_K_MATVEC_SPV,
        spec: &spec,
        rows_por_wg: wg / 64 * rows,
    };
    // Mesmo clock da produção (`pstate.rs`): sem ele a medida depende de quanto o DPM
    // automático subiu o clock até ali — 429 contra 684 GB/s na mesma matriz.
    let pstate = phys
        .pci
        .as_deref()
        .and_then(|p| crate::pstate::Pstate::novo(p, "bancada"));
    eprintln!(
        "Q4_K matvec, geometria {wg},{rows}, {REPS} reps, {}, pstate {}",
        phys.name(),
        if pstate.is_some() {
            "fixo"
        } else {
            "automático"
        }
    );
    for (nome, n_in, n_out) in FORMAS {
        let w = pesos_q4k(n_in, n_out, 7);
        let x = ativacao(n_in, 3);
        // Melhor de 3: a primeira medida de cada forma ainda paga o aquecimento.
        let us = (0..3)
            .map(|_| {
                if let Some(p) = &pstate {
                    p.em_uso(false);
                }
                medir(&ctx, phys, &dev, &k, &w, &x, n_in, n_out, 1).0
            })
            .fold(f64::MAX, f64::min);
        linha(nome, n_in, n_out, w.len(), us);
    }
}

/// GEMM Q4_K do prefill (`mul_mm.comp`) nas formas grandes, em TOPS de int8 contra os
/// 51 TOPS de `V_DOT4_I32_I8` medidos em `scripts/teto-mi50.sh`. `LLAMA_RS_BATCH` muda as
/// colunas, como no prefill de verdade.
#[test]
#[ignore = "bancada de desempenho: ocupa a GPU e só imprime"]
fn bancada_mul_mm() {
    let Ok(ctx) = VulkanContext::new() else {
        return;
    };
    let Some(phys) = device_sem_monitor(&ctx) else {
        return;
    };
    let dev = VulkanDevice::create(&ctx, phys).unwrap();
    let cols = crate::resident_forward::batch_size();
    let spec = [(0, cols as u32)];
    let k = Kernel {
        spv: crate::MUL_MM_SPV,
        spec: &spec,
        rows_por_wg: crate::resident_forward::GEMM_LINHAS_POR_WG,
    };
    let variantes = [("mul_mm", k)];
    let pstate = phys
        .pci
        .as_deref()
        .and_then(|p| crate::pstate::Pstate::novo(p, "bancada"));
    eprintln!("mul_mm Q4_K, {cols} colunas, {REPS} reps, pstate peak do prefill");
    for (nome, n_in, n_out) in &FORMAS[..6] {
        let (n_in, n_out) = (*n_in, *n_out);
        let w = pesos_q4k(n_in, n_out, 7);
        let x = ativacao(n_in * cols, 3);
        let mut ys = Vec::new();
        for (rot, kern) in &variantes {
            let mut y = Vec::new();
            let us = (0..3)
                .map(|_| {
                    if let Some(p) = &pstate {
                        p.em_uso(true);
                    }
                    let (t, yy) = medir(&ctx, phys, &dev, kern, &w, &x, n_in, n_out, cols);
                    y = yy;
                    t
                })
                .fold(f64::MAX, f64::min);
            ys.push(y);
            let tops = 2.0 * (n_in * n_out * cols) as f64 / (us * 1e6);
            eprintln!(
                "{nome:<12} {rot:<6} {n_in:>6}->{n_out:<6} {us:>8.1} µs {tops:>6.2} TOPS \
                 {:>4.1}% do pico int8",
                100.0 * tops / 51.0
            );
        }
        for (i, y) in ys.iter().enumerate().skip(1) {
            let dif = ys[0]
                .iter()
                .zip(y)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
                .fold(0f32, f32::max);
            assert!(
                dif < 1e-4,
                "{nome}: {} diverge do atual ({dif:e})",
                variantes[i].0
            );
        }
    }
}
