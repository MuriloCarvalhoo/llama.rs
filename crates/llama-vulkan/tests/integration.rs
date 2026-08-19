//! Testes de integracao Vulkan -- exigem duas MI50 reais.
//! Pulam automaticamente se nenhum device Vulkan AMD estiver disponivel.

use llama_vulkan::{GpuWeights, VulkanContext, VulkanDevice};

#[test]
fn detects_at_least_one_amd_device() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let devices = ctx.amd_compute_devices();
    if devices.is_empty() {
        eprintln!("Nenhum device AMD");
        return;
    }
    assert!(!devices.is_empty());
    for d in devices {
        eprintln!("  {} -- subgroupSize={}", d.name(), d.subgroup_size());
    }
}

#[test]
fn detects_two_mi50_for_dual_gpu() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let devices = ctx.amd_compute_devices();
    if devices.len() < 2 {
        eprintln!("Menos de 2 AMD -- pulando");
        return;
    }
    for d in devices {
        assert_eq!(d.subgroup_size(), 64, "MI50 deve ter wave64");
    }
}

#[test]
fn creates_logical_device_for_first_amd() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let devices = ctx.amd_compute_devices();
    if devices.is_empty() {
        eprintln!("Nenhum device AMD -- pulando");
        return;
    }
    let phys = &devices[0];
    let dev = VulkanDevice::create(&ctx, phys);
    assert!(dev.is_ok(), "Falha ao criar device logico: {:?}", dev.err());
    eprintln!("Device logico criado para {}", phys.name());
}

#[test]
fn sub_allocator_chunks_independentes_para_tensores_grandes() {
    use llama_vulkan::alloc::{GpuAllocator, MAX_CHUNK_BYTES};

    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("Nenhum device AMD -- pulando");
        return;
    }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    let mut alloc = GpuAllocator::new(&ctx, &phys[0], &dev, 3 * MAX_CHUNK_BYTES)
        .expect("GpuAllocator::new falhou");

    // 100MB e 200MB cabem no mesmo chunk de 1.5GB
    let a = alloc
        .alloc(dev.as_device(), 100_000_000)
        .expect("alloc 100MB falhou");
    let b = alloc
        .alloc(dev.as_device(), 200_000_000)
        .expect("alloc 200MB falhou");
    assert_eq!(
        a.chunk_idx, b.chunk_idx,
        "duas alocacoes pequenas devem estar no mesmo chunk"
    );
    assert_eq!(a.offset, 0, "primeira alocacao deve comecar no offset 0");
    assert!(b.offset > 0, "segunda alocacao deve ter offset positivo");

    eprintln!(
        "Alocacao A: chunk={} offset={} size={}",
        a.chunk_idx, a.offset, a.size
    );
    eprintln!(
        "Alocacao B: chunk={} offset={} size={}",
        b.chunk_idx, b.offset, b.size
    );

    alloc.cleanup(dev.as_device());
}

#[test]
fn upload_tensor_q8_0_para_vram() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("Nenhum device AMD -- pulando");
        return;
    }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    let n_out = 64usize;
    let n_in = 128usize;
    // Q8_0: cada bloco de 32 elementos = 2 bytes (scale f16) + 32 bytes (quants) = 34 bytes
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;
    let bytes: Vec<u8> = (0..n_out * row_bytes).map(|i| (i % 256) as u8).collect();

    use llama_vulkan::tensor::GpuTensor;
    let tensor =
        GpuTensor::upload_q8_0(&ctx, &phys[0], &dev, &bytes, n_in, n_out).expect("upload falhou");
    assert_eq!(tensor.n_out, n_out);
    assert_eq!(tensor.n_in, n_in);
    // Buffer GPU repackeado para blocos de 36 bytes (scale[2]|pad[2]|qs[32]).
    let gpu_bytes = (n_out * n_blocks * 36) as u64;
    assert_eq!(tensor.size_bytes, gpu_bytes);
    eprintln!(
        "Upload OK: {}x{} Q8_0 ({} bytes GPU)",
        n_out, n_in, gpu_bytes
    );
    tensor.destroy(dev.as_device());
}

#[test]
fn matmul_gpu_matches_cpu_reference() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        return;
    }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    // n_in=32, n_out=4, n_blocks=1
    // row 0: scale=1.0f16, qs[0]=1, resto 0 -> y[0] = 1.0 * 1 * x[0] = 5.0
    // row 1: scale=2.0f16, qs[0]=1, resto 0 -> y[1] = 2.0 * 1 * x[0] = 10.0
    let n_in = 32usize;
    let n_out = 4usize;
    let row_bytes = 1 * 34; // 1 bloco
    let mut w_bytes = vec![0u8; n_out * row_bytes];

    // Escala f16 em little-endian: 1.0 = 0x3C00
    let f16_bytes_1_0: [u8; 2] = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    let f16_bytes_2_0: [u8; 2] = half::f16::from_f32(2.0).to_bits().to_le_bytes();
    w_bytes[0..2].copy_from_slice(&f16_bytes_1_0);
    w_bytes[2] = 1; // qs[0] = 1
    w_bytes[row_bytes..row_bytes + 2].copy_from_slice(&f16_bytes_2_0);
    w_bytes[row_bytes + 2] = 1; // qs[0] = 1

    let x_f32 = vec![5.0f32; n_in];

    use llama_vulkan::matmul::dispatch_q8_0_matvec;
    let y = dispatch_q8_0_matvec(&ctx, &phys[0], &dev, &w_bytes, &x_f32, n_in, n_out)
        .expect("matmul GPU falhou");

    assert_eq!(y.len(), n_out);
    assert!((y[0] - 5.0).abs() < 0.1, "y[0] esperado ~5.0, got {}", y[0]);
    assert!(
        (y[1] - 10.0).abs() < 0.1,
        "y[1] esperado ~10.0, got {}",
        y[1]
    );
    assert!(y[2].abs() < 0.1, "y[2] esperado ~0.0, got {}", y[2]);
    assert!(y[3].abs() < 0.1, "y[3] esperado ~0.0, got {}", y[3]);
    eprintln!("GPU matmul: y={:?}", y);
}

#[test]
fn dual_gpu_row_split_matches_single_gpu() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let phys = ctx.amd_compute_devices();
    if phys.len() < 2 {
        eprintln!("Menos de 2 GPUs -- pulando");
        return;
    }

    let n_in = 896usize; // n_embd do Qwen2.5-0.5B
    let n_out = 896usize;
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;

    let w_bytes: Vec<u8> = (0..n_out * row_bytes)
        .map(|i| (i.wrapping_mul(31) % 255) as u8)
        .collect();
    let x_f32: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.001).collect();

    // Single GPU reference
    let dev0 = VulkanDevice::create(&ctx, &phys[0]).unwrap();
    use llama_vulkan::matmul::dispatch_q8_0_matvec;
    let y_single =
        dispatch_q8_0_matvec(&ctx, &phys[0], &dev0, &w_bytes, &x_f32, n_in, n_out).unwrap();

    // Dual GPU
    use llama_vulkan::DualGpuMatmul;
    let dual = DualGpuMatmul::new(&ctx).expect("DualGpuMatmul::new falhou");
    let y_dual = dual
        .matvec_q8_0(&w_bytes, &x_f32, n_in, n_out)
        .expect("dual falhou");

    assert_eq!(y_dual.len(), n_out);
    let max_diff = y_dual
        .iter()
        .zip(y_single.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 0.01,
        "max_diff={max_diff} excede tolerancia 0.01"
    );
    eprintln!("Dual GPU row-split OK -- {n_out} saidas corretas, max_diff={max_diff}");
}

#[test]
fn gpu_weights_upload_synthetic() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("Nenhum device AMD -- pulando");
        return;
    }

    let weights = GpuWeights::upload_synthetic(&ctx, 24, 896).expect("upload_synthetic falhou");

    assert_eq!(weights.n_layers_loaded, 24);
    assert!(weights.vram_bytes > 0, "deve ter alocado VRAM");
    eprintln!(
        "GpuWeights OK: {} layers, {} MB VRAM",
        weights.n_layers_loaded,
        weights.vram_bytes / 1024 / 1024
    );
}

#[test]
fn forward_gpu_real_matches_f32_cpu_reference() {
    use std::path::Path;
    let model_path = Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf");
    let Ok(bytes) = std::fs::read(model_path) else {
        eprintln!("qwen ausente — pulando");
        return;
    };
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan indisponível: {e} — pulando");
            return;
        }
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("Menos de 2 MI50 — pulando");
        return;
    }

    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    let model = llama_model::Model::load_with_config(&f, &bytes, cfg.clone()).unwrap();
    let w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let backend = llama_vulkan::DualGpuBackend::new(&ctx).expect("backend falhou");

    // Referência CPU do MESMO algoritmo da GPU, incluindo a quantização int8 da
    // ativação que o shader faz. A GPU (wave64 subgroupAdd) e esta referência (soma
    // sequencial) diferem só na ordem de soma f32 — o token deve ser idêntico.
    struct CpuInt8ActMatmul;
    impl llama_model::GpuMatmul for CpuInt8ActMatmul {
        fn matvec_q8_0(
            &self,
            w_bytes: &[u8],
            x: &[f32],
            n_in: usize,
            n_out: usize,
        ) -> Result<Vec<f32>, llama_model::ModelError> {
            Ok(cpu_ref_q8_0_int8act(w_bytes, x, n_in, n_out))
        }
    }

    let prompt = [cfg.bos_id];
    let gpu_tok = model.decode_one_gpu_owned(&prompt, &backend, &w).unwrap();
    let ref_tok = model
        .decode_one_gpu_owned(&prompt, &CpuInt8ActMatmul, &w)
        .unwrap();
    // Token do caminho CPU quantizado (ativações Q8_0) — informativo: pode diferir, pois
    // é uma aproximação distinta (mais perdas) que a via f32 da GPU.
    let cpu_quant_tok = model.decode_one_cpu_owned(&prompt).unwrap();

    eprintln!(
        "Forward GPU real: gpu_tok={gpu_tok} ref_int8_tok={ref_tok} cpu_quant_tok={cpu_quant_tok}"
    );
    assert_eq!(
        gpu_tok, ref_tok,
        "GPU deve igualar a referência CPU do mesmo algoritmo (ativação int8)"
    );
}

/// Referência CPU que espelha **o que o shader realmente computa**: a ativação é
/// quantizada para int8 simétrico por bloco de 32 (como em `q8_0_matvec.comp`), e o
/// produto interno é feito em inteiros. Difere de `cpu_ref_q8_0_f32act` (ativação f32)
/// por ~0.14% — diferença suficiente para virar o argmax em logits quase degenerados,
/// por isso os testes de igualdade de token usam esta referência, não a f32.
/// Reconstrói `x` como o shader o enxerga: quantizado em int8 por blocos de 32 e
/// multiplicado de volta pela escala. Os matvecs K-quant consomem a mesma ativação int8 do
/// caminho Q8_0, então a referência de CPU precisa partir dela — comparar contra o `x` em
/// f32 original mediria o erro da *quantização*, não o do shader.
fn quant_dequant_x(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for b in 0..x.len() / 32 {
        let blk = &x[b * 32..b * 32 + 32];
        let d_x = blk.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
        let inv = if d_x > 0.0 { 1.0 / d_x } else { 0.0 };
        for i in 0..32 {
            out[b * 32 + i] = (blk[i] * inv).round().clamp(-127.0, 127.0) * d_x;
        }
    }
    out
}

fn cpu_ref_q8_0_int8act(w: &[u8], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;

    // Quantização da ativação: uma vez por matvec, compartilhada por todas as linhas.
    let mut xq = vec![0i32; n_in];
    let mut xd = vec![0f32; n_blocks];
    for b in 0..n_blocks {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d_x = amax / 127.0;
        let inv = if d_x > 0.0 { 1.0 / d_x } else { 0.0 };
        xd[b] = d_x;
        for i in 0..32 {
            xq[b * 32 + i] = (blk[i] * inv).round().clamp(-127.0, 127.0) as i32;
        }
    }

    let mut y = vec![0f32; n_out];
    for row in 0..n_out {
        let mut acc = 0f32;
        for b in 0..n_blocks {
            let off = row * row_bytes + b * 34;
            let d_w = half::f16::from_le_bytes([w[off], w[off + 1]]).to_f32();
            let mut isum = 0i32;
            for i in 0..32 {
                isum += i32::from(w[off + 2 + i] as i8) * xq[b * 32 + i];
            }
            acc += d_w * xd[b] * isum as f32;
        }
        y[row] = acc;
    }
    y
}

fn cpu_ref_q8_0_f32act(w: &[u8], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;
    let mut y = vec![0f32; n_out];
    for row in 0..n_out {
        let mut acc = 0f32;
        for b in 0..n_blocks {
            let off = row * row_bytes + b * 34;
            let scale = half::f16::from_le_bytes([w[off], w[off + 1]]).to_f32();
            let mut dot = 0f32;
            for i in 0..32 {
                let q = w[off + 2 + i] as i8 as f32;
                dot += q * x[b * 32 + i];
            }
            acc += scale * dot;
        }
        y[row] = acc;
    }
    y
}

#[test]
fn gpu_matvec_large_n_out_matches_cpu_ref() {
    // Regressão do bug de row-split OOB: com n_out grande (vocab=151936) a GPU1
    // retornava 0 por indexar pesos/saída pelo offset global. Guarda contra reintrodução.
    use llama_model::GpuMatmul;
    use std::path::Path;
    let Ok(bytes) = std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")) else {
        eprintln!("qwen ausente — pulando");
        return;
    };
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    if ctx.amd_compute_devices().len() < 2 {
        return;
    }
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    let w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let backend = llama_vulkan::DualGpuBackend::new(&ctx).unwrap();

    // Probe 1: output projection — n_in=n_embd=896, n_out=vocab=151936
    let x1: Vec<f32> = (0..cfg.n_embd)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let gpu1 = backend
        .matvec_q8_0(w.output.bytes, &x1, cfg.n_embd, cfg.vocab)
        .unwrap();
    let cpu1 = cpu_ref_q8_0_f32act(w.output.bytes, &x1, cfg.n_embd, cfg.vocab);
    let mut maxdiff1 = 0f32;
    let mut argi1 = 0usize;
    for (i, (a, b)) in gpu1.iter().zip(cpu1.iter()).enumerate() {
        let d = (a - b).abs();
        if d > maxdiff1 {
            maxdiff1 = d;
            argi1 = i;
        }
    }
    eprintln!(
        "[P1 output {}x{}] maxdiff={maxdiff1:.4} @row {argi1} (gpu={} cpu={})",
        cfg.vocab, cfg.n_embd, gpu1[argi1], cpu1[argi1]
    );
    eprintln!(
        "[P1] gpu[0..3]={:?} cpu[0..3]={:?}",
        &gpu1[0..3],
        &cpu1[0..3]
    );
    assert!(
        maxdiff1 < 0.01,
        "regressão row-split OOB: GPU diverge da referência em n_out={} (maxdiff={maxdiff1} @row {argi1}, gpu={} cpu={})",
        cfg.vocab,
        gpu1[argi1],
        cpu1[argi1]
    );

    // Probe 2: ffn_down — n_in=n_ff=4864, n_out=n_embd=896
    let x2: Vec<f32> = (0..cfg.n_ff)
        .map(|i| ((i % 5) as f32) * 0.05 - 0.1)
        .collect();
    let gpu2 = backend
        .matvec_q8_0(w.layers[0].ffn_down.bytes, &x2, cfg.n_ff, cfg.n_embd)
        .unwrap();
    let cpu2 = cpu_ref_q8_0_f32act(w.layers[0].ffn_down.bytes, &x2, cfg.n_ff, cfg.n_embd);
    let mut maxdiff2 = 0f32;
    for (a, b) in gpu2.iter().zip(cpu2.iter()) {
        maxdiff2 = maxdiff2.max((a - b).abs());
    }
    eprintln!(
        "[P2 ffn_down {}x{}] maxdiff={maxdiff2:.4} gpu[0..3]={:?} cpu[0..3]={:?}",
        cfg.n_embd,
        cfg.n_ff,
        &gpu2[0..3],
        &cpu2[0..3]
    );
    assert!(
        maxdiff2 < 0.01,
        "GPU diverge da referência em ffn_down (n_in={}, maxdiff={maxdiff2})",
        cfg.n_ff
    );
}

// ─── Fase 8.1C Task 4: ResidentForward rmsnorm GPU == CPU ────────────────────

#[test]
fn resident_fwd_rmsnorm_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let dim = 896usize;
    let x: Vec<f32> = (0..dim).map(|i| ((i % 13) as f32) * 0.1 - 0.5).collect();
    let w: Vec<f32> = (0..dim).map(|i| 1.0 + ((i % 7) as f32) * 0.01).collect();
    let eps = 1e-6f32;

    let ss: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
    let cpu: Vec<f32> = x
        .iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect();

    let gpu = fwd.dbg_rmsnorm(&x, &w, eps).unwrap();
    assert_eq!(gpu.len(), dim);
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "rmsnorm[{i}]: cpu={a} gpu={b}");
    }
}

// ─── Fase 8.1A: ResidentGpu (single-GPU, pesos+pipeline residentes) ──────────

#[test]
fn resident_gpu_decode_matches_cpu_ref() {
    use std::path::Path;
    let Ok(bytes) = std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")) else {
        eprintln!("qwen ausente — pulando");
        return;
    };
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan indisponível: {e} — pulando");
            return;
        }
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }

    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    let model = llama_model::Model::load_with_config(&f, &bytes, cfg.clone()).unwrap();
    let w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let backend = llama_vulkan::ResidentGpu::new(&ctx).expect("ResidentGpu falhou");

    // Referência com a MESMA matemática do shader (ativação quantizada em int8).
    // Usar a referência de ativação f32 aqui compararia contra um kernel diferente
    // do que roda na GPU desde a quantização int8 da ativação.
    struct CpuInt8ActMatmul;
    impl llama_model::GpuMatmul for CpuInt8ActMatmul {
        fn matvec_q8_0(
            &self,
            w_bytes: &[u8],
            x: &[f32],
            n_in: usize,
            n_out: usize,
        ) -> Result<Vec<f32>, llama_model::ModelError> {
            Ok(cpu_ref_q8_0_int8act(w_bytes, x, n_in, n_out))
        }
    }

    let prompt = [cfg.bos_id];
    let gpu_tok = model.decode_one_gpu_owned(&prompt, &backend, &w).unwrap();
    let ref_tok = model
        .decode_one_gpu_owned(&prompt, &CpuInt8ActMatmul, &w)
        .unwrap();
    eprintln!("ResidentGpu decode: gpu_tok={gpu_tok} ref_int8_tok={ref_tok}");
    assert_eq!(
        gpu_tok, ref_tok,
        "ResidentGpu deve igualar a referência int8 (mesma matemática do shader)"
    );
}

#[test]
fn resident_gpu_nao_re_uploada_peso() {
    use llama_model::GpuMatmul;
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("sem Vulkan — pulando");
            return;
        }
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let backend = llama_vulkan::ResidentGpu::new(&ctx).unwrap();

    // Peso Q8_0 sintético 1 linha × 32 col: 34 bytes (2 scale f16 + 32 quants i8).
    let w = vec![0u8; 34];
    let x = vec![0f32; 32];
    backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(backend.resident_count(), 1, "primeiro uso = 1 upload");
    backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(
        backend.resident_count(),
        1,
        "mesmo ponteiro = cache-hit, sem novo upload"
    );
}

#[test]
fn resident_gpu_buffers_estabilizam_apos_warmup() {
    use llama_model::GpuMatmul;
    use llama_vulkan::{ResidentGpu, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let backend = ResidentGpu::new(&ctx).unwrap();

    // Peso Q8_0 sintético 1 linha × 32 col: 34 bytes (2 scale + 32 quants).
    let w = vec![0u8; 34];
    let x = vec![0f32; 32];

    // 1ª chamada: cresce lado X (1) e lado Y (1) = 2 grows.
    let _ = backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(backend.buffer_grows(), 2, "1ª chamada aloca X e Y");

    // 2ª chamada idêntica: cache-hit total, nenhum grow novo.
    let _ = backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(backend.buffer_grows(), 2, "mesmo tamanho => sem realloc");

    // Saída maior (n_out=2): só o lado Y cresce (+1).
    let w2 = vec![0u8; 2 * 34];
    let _ = backend.matvec_q8_0(&w2, &x, 32, 2).unwrap();
    assert_eq!(
        backend.buffer_grows(),
        3,
        "n_out maior => 1 grow só no lado Y"
    );
}

// ─── Fase 8.1C Task 5: swiglu + add GPU == CPU ───────────────────────────────

#[test]
fn resident_fwd_swiglu_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n = 4864usize;
    let g: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.2 - 1.0).collect();
    let u: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.3 + 0.1).collect();
    let cpu: Vec<f32> = g
        .iter()
        .zip(u.iter())
        .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
        .collect();

    let gpu = fwd.dbg_swiglu(&g, &u).unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "swiglu[{i}]: cpu={a} gpu={b}");
    }
}

#[test]
fn resident_fwd_add_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n = 896usize;
    let dst: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let src: Vec<f32> = (0..n).map(|i| i as f32 * -0.25 + 1.0).collect();
    let cpu: Vec<f32> = dst.iter().zip(src.iter()).map(|(&a, &b)| a + b).collect();

    let gpu = fwd.dbg_add(&dst, &src).unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "add[{i}]: cpu={a} gpu={b}");
    }
}

// ─── Fase 8.1C Task 6: rope GPU == CPU ───────────────────────────────────────

#[test]
fn resident_fwd_rope_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n_head = 14usize;
    let head_dim = 64usize;
    let rope_dim = 64usize;
    let pos = 5usize;
    let freq_base = 1_000_000.0f32;
    let freq: Vec<f32> = (0..rope_dim / 2)
        .map(|i| freq_base.powf(-2.0 * i as f32 / rope_dim as f32))
        .collect();
    let mut x: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i % 17) as f32) * 0.1 - 0.7)
        .collect();

    let mut cpu = x.clone();
    for h in 0..n_head {
        let base = h * head_dim;
        for i in 0..rope_dim / 2 {
            let theta = pos as f32 * freq[i];
            let (s, c) = theta.sin_cos();
            let a = cpu[base + 2 * i];
            let b = cpu[base + 2 * i + 1];
            cpu[base + 2 * i] = a * c - b * s;
            cpu[base + 2 * i + 1] = a * s + b * c;
        }
    }

    let gpu = fwd
        .dbg_rope(&mut x, n_head, head_dim, rope_dim, &freq, pos)
        .unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "rope[{i}]: cpu={a} gpu={b}");
    }
}

// ─── Fase 8.1C Task 7: attention GQA GPU == CPU ──────────────────────────────

#[test]
fn resident_fwd_attention_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    // As posições são divididas entre as 4 waves do workgroup, então os casos que
    // importam são: total_len < 4 (waves sem nenhuma posição), não múltiplo de 4, e
    // head_dim = 128 (duas dimensões por lane — a geometria do Qwen2.5-32B).
    for (n_head, n_head_kv, head_dim, total_len) in [
        (14, 2, 64, 7),
        (14, 2, 64, 2),
        (40, 8, 128, 9),
        (40, 8, 128, 1),
    ] {
        attention_caso(&fwd, n_head, n_head_kv, head_dim, total_len);
    }
}

fn attention_caso(
    fwd: &llama_vulkan::ResidentForward<'_>,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    total_len: usize,
) {
    let kv_dim = n_head_kv * head_dim;
    let n_rep = n_head / n_head_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i % 19) as f32) * 0.05 - 0.4)
        .collect();
    let kc: Vec<f32> = (0..total_len * kv_dim)
        .map(|i| ((i % 23) as f32) * 0.03 - 0.3)
        .collect();
    let vc: Vec<f32> = (0..total_len * kv_dim)
        .map(|i| ((i % 29) as f32) * 0.02 - 0.2)
        .collect();

    let mut cpu = vec![0f32; n_head * head_dim];
    for h in 0..n_head {
        let kv_h = h / n_rep;
        let qoff = h * head_dim;
        let mut scores = vec![0f32; total_len];
        for j in 0..total_len {
            let koff = j * kv_dim + kv_h * head_dim;
            let dot: f32 = (0..head_dim).map(|dd| q[qoff + dd] * kc[koff + dd]).sum();
            scores[j] = dot * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }
        for j in 0..total_len {
            let voff = j * kv_dim + kv_h * head_dim;
            for dd in 0..head_dim {
                cpu[qoff + dd] += scores[j] * vc[voff + dd];
            }
        }
    }

    let gpu = fwd
        .dbg_attention(&q, &kc, &vc, n_head, n_head_kv, head_dim, total_len)
        .unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "attn[{i}] (n_head={n_head} head_dim={head_dim} total_len={total_len}): cpu={a} gpu={b}"
        );
    }
}

// ─── Fase 8.1D Task 4: geração multi-token 1D == CPU ────────────────────────────

#[test]
fn resident_forward_gera_igual_cpu_multi_token() {
    use llama_tokenizer::Tokenizer;
    use llama_vulkan::{ResidentForward, VulkanContext};
    use rand::{SeedableRng, rngs::SmallRng};

    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let path = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let model = llama_model::Model::load(&f, &bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config).unwrap();
    let aux = model.gpu_aux_weights().unwrap();
    let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux).unwrap();
    let sampler = llama_sampling::Sampler::Greedy;

    // Teacher forcing: os dois caminhos recebem SEMPRE a sequência escolhida pela CPU, e
    // o que se compara são os logits de cada passo — não a string gerada.
    //
    // Comparar strings exigia paridade bit-a-bit do argmax, e isso deixou de ser
    // alcançável quando o `attention.comp` passou a dividir as posições entre 8 waves: a
    // soma do softmax é reassociada, o resultado muda na ordem de 1e-6 (medido em
    // `resident_fwd_attention_igual_cpu`, tolerância 1e-5) e num 0.5B isso basta para
    // virar um argmax empatado. A verificação abaixo é mais forte, não mais fraca —
    // cobre os 8 passos com o KV-cache crescendo, em vez de só o token vencedor.
    let mut seq: Vec<u32> = tok.encode("Hello", true);
    let mut max_rel_global = 0.0f32;
    for passo in 0..8 {
        let cpu = model.decode_one_cpu_logits(&seq).unwrap();
        let gpu = model
            .decode_one_gpu_resident_logits(&seq, &backend)
            .unwrap();
        assert_eq!(cpu.len(), gpu.len(), "tamanho dos logits no passo {passo}");

        let max_abs = cpu.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
        let max_rel = cpu
            .iter()
            .zip(&gpu)
            .fold(0.0f32, |m, (&c, &g)| m.max((c - g).abs() / max_abs));
        max_rel_global = max_rel_global.max(max_rel);
        assert!(
            max_rel < 0.05,
            "passo {passo}: erro relativo {max_rel} deve ser < 5%"
        );

        let next = u32::try_from(sampler.sample(&cpu, &mut SmallRng::seed_from_u64(0))).unwrap();
        seq.push(next);
    }
    eprintln!(
        "8 passos de decode com KV crescente: erro relativo máximo {max_rel_global:.2e}; \
         continuação = {:?}",
        tok.decode(&seq)
    );
}

// ─── Fase 8.1C Task 11: ResidentForward logits == CPU (gate de correctude E2E) ─

#[test]
fn resident_forward_logits_iguais_a_cpu_qwen() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let path = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let model = llama_model::Model::load(&f, &bytes).unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config).unwrap();
    let aux = model.gpu_aux_weights().unwrap();
    let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux).unwrap();

    let prompt: [u32; 2] = [model.config.bos_id, 9707];
    let cpu = model.decode_one_cpu_owned(&prompt).unwrap();
    let gpu = model
        .decode_one_gpu_resident_owned(&prompt, &backend)
        .unwrap();
    assert_eq!(cpu, gpu, "argmax do decode GPU-resident deve igualar CPU");
}

// ─── Fase 8.3 Task 5: ativação int8 + dotPacked4x8 no matvec (tolerância) ──────

#[test]
fn resident_matvec_int8_logits_dentro_da_tolerancia() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let path = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let model = llama_model::Model::load(&f, &bytes).unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config).unwrap();
    let aux = model.gpu_aux_weights().unwrap();
    let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux).unwrap();

    let prompt: [u32; 2] = [model.config.bos_id, 9707];
    let cpu = model.decode_one_cpu_logits(&prompt).unwrap();
    let gpu = model
        .decode_one_gpu_resident_logits(&prompt, &backend)
        .unwrap();

    assert_eq!(cpu.len(), gpu.len(), "tamanho dos logits deve casar");

    // Erro relativo máximo (escalado pela magnitude máxima do vetor CPU para
    // evitar divisão por valores próximos de zero).
    let max_abs = cpu.iter().fold(0.0_f32, |m, &v| m.max(v.abs())).max(1e-6);
    let mut max_rel = 0.0_f32;
    for (&c, &g) in cpu.iter().zip(gpu.iter()) {
        let rel = (c - g).abs() / max_abs;
        if rel > max_rel {
            max_rel = rel;
        }
    }

    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let argmax_cpu = argmax(&cpu);
    let argmax_gpu = argmax(&gpu);
    eprintln!("max_rel_err={max_rel:.6} argmax_cpu={argmax_cpu} argmax_gpu={argmax_gpu}");

    assert!(
        max_rel < 0.05,
        "erro relativo máximo {max_rel} deve ser < 0.05 (5%)"
    );
    assert_eq!(argmax_cpu, argmax_gpu, "argmax CPU == GPU");
}

#[test]
fn gpu_matvec_k_tiling_n_in_maior_que_uma_janela_lds() {
    // O shader cacheia a ativação quantizada em LDS numa janela de MAX_BLOCKS=160
    // blocos (n_in <= 5120) e faz tiling da dimensão K acima disso. Nenhum modelo de
    // teste local exercita n_in > 5120 (o 0.5B tem n_ff=4864), mas o 14B tem
    // n_ff=13824 (432 blocos = 3 janelas). Este teste cobre o caminho multi-janela.
    use llama_model::GpuMatmul;
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }
    let backend = llama_vulkan::ResidentGpu::new(&ctx).expect("ResidentGpu");

    // 1 janela (n_in=4864), exatamente na borda (5120) e 3 janelas (13824 = n_ff do 14B).
    // Os pesos de todos os casos são criados ANTES do loop e mantidos vivos: o
    // `ResidentGpu` indexa o cache de peso pelo endereço do slice, e Vecs criados e
    // dropados em sequência podem reusar o mesmo endereço (cache hit falso).
    let n_out = 64usize;
    let casos: Vec<(usize, Vec<u8>, Vec<f32>)> = [4864usize, 5120, 13824]
        .iter()
        .map(|&n_in| {
            let n_blocks = n_in / 32;
            let w: Vec<u8> = (0..n_out * n_blocks * 34)
                .map(|i| match i % 34 {
                    0 => 0x00, // escala f16 = 1.0 -> bytes [0x00, 0x3c]
                    1 => 0x3c,
                    _ => (i.wrapping_mul(37) % 251) as u8,
                })
                .collect();
            let x: Vec<f32> = (0..n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
            (n_in, w, x)
        })
        .collect();

    for (n_in, w, x) in &casos {
        let (n_in, n_blocks) = (*n_in, n_in / 32);
        let gpu = backend.matvec_q8_0(w, x, n_in, n_out).expect("matvec GPU");
        let cpu = cpu_ref_q8_0_int8act(w, x, n_in, n_out);

        let max_abs = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
        let max_rel = gpu
            .iter()
            .zip(&cpu)
            .fold(0f32, |m, (&a, &b)| m.max((a - b).abs() / max_abs));
        eprintln!(
            "k-tiling n_in={n_in} ({n_blocks} blocos, {} janelas): erro rel max = {max_rel:.6}",
            n_blocks.div_ceil(160)
        );
        // Só a ordem de soma f32 difere (lane-strided + subgroupAdd vs sequencial).
        assert!(
            max_rel < 1e-4,
            "n_in={n_in}: GPU divergiu da referência int8 (erro rel {max_rel})"
        );
    }
}

#[test]
fn q6_k_matvec_gpu_bate_com_a_referencia_de_cpu() {
    // Q6_K tem o mapeamento sub-bloco -> elemento mais intrincado dos K-quants (metades de
    // 128, quartos de 32, nibbles de ql e pares de bits de qh). Referência: o dequant de
    // CPU + matmul f32.
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }
    let dev = llama_vulkan::VulkanDevice::create(&ctx, &phys[0]).unwrap();

    let n_in = 512usize;
    let n_out = 16usize;
    let sb_per_row = n_in / 256;
    let mut w = vec![0u8; n_out * sb_per_row * 210];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i.wrapping_mul(89).wrapping_add(i / 5) % 251) as u8;
    }
    // `d` (bytes 208..209 de cada superbloco) num valor são, e escalas i8 moderadas.
    for sb in 0..n_out * sb_per_row {
        let o = sb * 210;
        for j in 0..16 {
            w[o + 192 + j] = ((j as i32 * 7 % 40) - 20) as i8 as u8;
        }
        w[o + 208..o + 210].copy_from_slice(&half::f16::from_f32(0.0091).to_le_bytes());
    }
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.027)
        .collect();

    let gpu =
        llama_vulkan::matmul::dispatch_q6_k_matvec(&ctx, &phys[0], &dev, &w, &x, n_in, n_out, 1)
            .expect("dispatch Q6_K");

    let row_bytes = sb_per_row * 210;
    let x_int8 = quant_dequant_x(&x);
    let mut cpu = vec![0f32; n_out];
    for (r, out) in cpu.iter_mut().enumerate() {
        let deq =
            ggml_cpu::dequant_to_f32(&w[r * row_bytes..(r + 1) * row_bytes], gguf::GgmlType::Q6_K)
                .expect("dequant");
        *out = deq.iter().zip(&x_int8).map(|(a, b)| a * b).sum();
    }

    let max_abs = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let max_rel = gpu
        .iter()
        .zip(&cpu)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs() / max_abs));
    eprintln!("Q6_K matvec: erro relativo maximo = {max_rel:.3e}");
    assert!(
        max_rel < 1e-5,
        "shader Q6_K divergiu da referencia de CPU (erro rel {max_rel})\ngpu={:?}\ncpu={:?}",
        &gpu[..4],
        &cpu[..4]
    );
}

#[test]
fn q5_k_matvec_gpu_bate_com_a_referencia_de_cpu() {
    // O shader Q5_K desempacota os superblocos direto na GPU. A referência é o
    // `ggml_cpu::dequant_to_f32` (validado contra o gguf-py do llama.cpp) seguido de um
    // matmul f32 ingênuo — se a manipulação de bits do shader divergir, aparece aqui.
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }
    let dev = llama_vulkan::VulkanDevice::create(&ctx, &phys[0]).unwrap();

    // Cada wave cobre 16 superblocos (4 lanes por superbloco). Os tamanhos varridos batem
    // nas três situações: menos de uma rodada, exatamente uma, e duas com sobra na última
    // (5120 = n_embd do Qwen2.5-32B → 20 superblocos, que é o caso real).
    for n_in in [512usize, 4096, 5120] {
        q5_k_matvec_caso(&ctx, &phys[0], &dev, n_in);
    }
}

fn q5_k_matvec_caso(
    ctx: &llama_vulkan::VulkanContext,
    phys: &llama_vulkan::VulkanPhysicalDevice,
    dev: &llama_vulkan::VulkanDevice,
    n_in: usize,
) {
    let n_out = 16usize;
    let sb_per_row = n_in / 256;
    // Bytes pseudoaleatórios mas determinísticos: cobre todos os nibbles, bits de qh e
    // escalas de 6 bits. As escalas f16 (bytes 0..3 de cada superbloco) são fixadas em
    // valores sãos para o resultado não estourar.
    let mut w = vec![0u8; n_out * sb_per_row * 176];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i.wrapping_mul(101).wrapping_add(i / 7) % 251) as u8;
    }
    for sb in 0..n_out * sb_per_row {
        let o = sb * 176;
        w[o..o + 2].copy_from_slice(&half::f16::from_f32(0.0123).to_le_bytes()); // d
        w[o + 2..o + 4].copy_from_slice(&half::f16::from_f32(0.0045).to_le_bytes()); // dmin
    }
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.031)
        .collect();

    let gpu = llama_vulkan::matmul::dispatch_q5_k_matvec(ctx, phys, dev, &w, &x, n_in, n_out, 1)
        .expect("dispatch Q5_K");

    // Referência: desquantiza cada linha e faz o produto interno em f32.
    let row_bytes = sb_per_row * 176;
    let x_int8 = quant_dequant_x(&x);
    let mut cpu = vec![0f32; n_out];
    for (r, out) in cpu.iter_mut().enumerate() {
        let deq =
            ggml_cpu::dequant_to_f32(&w[r * row_bytes..(r + 1) * row_bytes], gguf::GgmlType::Q5_K)
                .expect("dequant");
        *out = deq.iter().zip(&x_int8).map(|(a, b)| a * b).sum();
    }

    let max_abs = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let max_rel = gpu
        .iter()
        .zip(&cpu)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs() / max_abs));
    eprintln!("Q5_K matvec (n_in={n_in}): erro relativo maximo = {max_rel:.3e}");
    assert!(
        max_rel < 1e-5,
        "shader Q5_K divergiu da referencia de CPU (erro rel {max_rel})\ngpu={:?}\ncpu={:?}",
        &gpu[..4],
        &cpu[..4]
    );
}

#[test]
fn q4_k_matvec_gpu_bate_com_a_referencia_de_cpu() {
    // Mesmo esquema do teste Q5_K acima: `ggml_cpu::dequant_to_f32` + matmul f32 ingênuo
    // como referência. Q4_K é o Q5_K sem o 5º bit (`qh`) — mesmo bug class possível se o
    // shader novo (`q4_k_matvec.comp`) errar o offset de `qs` sem o `qh` no meio.
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }
    let dev = llama_vulkan::VulkanDevice::create(&ctx, &phys[0]).unwrap();

    for n_in in [512usize, 4096, 5120] {
        q4_k_matvec_caso(&ctx, &phys[0], &dev, n_in);
    }
}

fn q4_k_matvec_caso(
    ctx: &llama_vulkan::VulkanContext,
    phys: &llama_vulkan::VulkanPhysicalDevice,
    dev: &llama_vulkan::VulkanDevice,
    n_in: usize,
) {
    let n_out = 16usize;
    let sb_per_row = n_in / 256;
    let mut w = vec![0u8; n_out * sb_per_row * 144];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i.wrapping_mul(101).wrapping_add(i / 7) % 251) as u8;
    }
    for sb in 0..n_out * sb_per_row {
        let o = sb * 144;
        w[o..o + 2].copy_from_slice(&half::f16::from_f32(0.0123).to_le_bytes()); // d
        w[o + 2..o + 4].copy_from_slice(&half::f16::from_f32(0.0045).to_le_bytes()); // dmin
    }
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.031)
        .collect();

    let gpu = llama_vulkan::matmul::dispatch_q4_k_matvec(ctx, phys, dev, &w, &x, n_in, n_out, 1)
        .expect("dispatch Q4_K");

    let row_bytes = sb_per_row * 144;
    let x_int8 = quant_dequant_x(&x);
    let mut cpu = vec![0f32; n_out];
    for (r, out) in cpu.iter_mut().enumerate() {
        let deq =
            ggml_cpu::dequant_to_f32(&w[r * row_bytes..(r + 1) * row_bytes], gguf::GgmlType::Q4_K)
                .expect("dequant");
        *out = deq.iter().zip(&x_int8).map(|(a, b)| a * b).sum();
    }

    let max_abs = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let max_rel = gpu
        .iter()
        .zip(&cpu)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs() / max_abs));
    eprintln!("Q4_K matvec (n_in={n_in}): erro relativo maximo = {max_rel:.3e}");
    assert!(
        max_rel < 1e-5,
        "shader Q4_K divergiu da referencia de CPU (erro rel {max_rel})\ngpu={:?}\ncpu={:?}",
        &gpu[..4],
        &cpu[..4]
    );
}

/// Qual K-quant o caso de batch exercita. Os três têm o mesmo contrato de `cols`, mas
/// layouts de superbloco (e, no Q6_K, `constant_id`) diferentes.
#[derive(Clone, Copy)]
enum QuantK {
    Q4,
    Q5,
    Q6,
}

impl QuantK {
    fn bytes_por_sb(self) -> usize {
        match self {
            Self::Q4 => 144,
            Self::Q5 => 176,
            Self::Q6 => 210,
        }
    }

    fn nome(self) -> &'static str {
        match self {
            Self::Q4 => "Q4_K",
            Self::Q5 => "Q5_K",
            Self::Q6 => "Q6_K",
        }
    }

    /// Pesos determinísticos com cabeçalho são: escalas f16 moderadas para o resultado não
    /// estourar, e todo o resto pseudoaleatório para cobrir nibbles, qh e escalas de 6 bits.
    fn pesos(self, n_out: usize, sb_per_row: usize) -> Vec<u8> {
        let sbb = self.bytes_por_sb();
        let mut w = vec![0u8; n_out * sb_per_row * sbb];
        for (i, b) in w.iter_mut().enumerate() {
            *b = (i.wrapping_mul(101).wrapping_add(i / 7) % 251) as u8;
        }
        for sb in 0..n_out * sb_per_row {
            let o = sb * sbb;
            match self {
                Self::Q4 | Self::Q5 => {
                    w[o..o + 2].copy_from_slice(&half::f16::from_f32(0.0123).to_le_bytes());
                    w[o + 2..o + 4].copy_from_slice(&half::f16::from_f32(0.0045).to_le_bytes());
                }
                Self::Q6 => {
                    for j in 0..16 {
                        w[o + 192 + j] = ((j as i32 * 7 % 40) - 20) as i8 as u8;
                    }
                    w[o + 208..o + 210].copy_from_slice(&half::f16::from_f32(0.0091).to_le_bytes());
                }
            }
        }
        w
    }

    fn dispatch(
        self,
        ctx: &llama_vulkan::VulkanContext,
        phys: &llama_vulkan::VulkanPhysicalDevice,
        dev: &llama_vulkan::VulkanDevice,
        w: &[u8],
        x: &[f32],
        n_in: usize,
        n_out: usize,
        cols: usize,
    ) -> Vec<f32> {
        use llama_vulkan::matmul::{
            dispatch_q4_k_matvec, dispatch_q5_k_matvec, dispatch_q6_k_matvec,
        };
        let r = match self {
            Self::Q4 => dispatch_q4_k_matvec(ctx, phys, dev, w, x, n_in, n_out, cols),
            Self::Q5 => dispatch_q5_k_matvec(ctx, phys, dev, w, x, n_in, n_out, cols),
            Self::Q6 => dispatch_q6_k_matvec(ctx, phys, dev, w, x, n_in, n_out, cols),
        };
        r.unwrap_or_else(|e| panic!("dispatch {} cols={cols} falhou: {e:?}", self.nome()))
    }
}

#[test]
fn matvec_k_em_batch_bate_coluna_a_coluna() {
    // Etapa 1 do prefill em batch (docs/prefill-em-batch.md): com COLS>1 o shader lê cada
    // peso uma vez e acumula contra COLS ativações. O invariante é ser indistinguível de
    // COLS chamadas separadas — cada coluna faz os mesmos dots na mesma ordem, então a
    // igualdade aqui é exata, não aproximada. Um erro de indexação por coluna (ler sempre
    // a coluna 0, ou escrever a saída embaralhada) quebra isto e não quebra o teste de
    // coluna única contra a referência de CPU.
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }
    let dev = llama_vulkan::VulkanDevice::create(&ctx, &phys[0]).unwrap();

    let (n_in, n_out) = (1024usize, 16usize);
    let sb_per_row = n_in / 256;

    for q in [QuantK::Q4, QuantK::Q5, QuantK::Q6] {
        let w = q.pesos(n_out, sb_per_row);
        for cols in [2usize, 4, 8] {
            // Colunas propositalmente diferentes entre si: uma coluna repetida esconderia
            // um erro de indexação que sempre lesse a coluna 0.
            let xs: Vec<Vec<f32>> = (0..cols)
                .map(|c| {
                    (0..n_in)
                        .map(|i| ((i % (29 + c)) as f32 - 14.0) * (0.031 + c as f32 * 0.004))
                        .collect()
                })
                .collect();
            let plano: Vec<f32> = xs.concat();

            let batch = q.dispatch(&ctx, &phys[0], &dev, &w, &plano, n_in, n_out, cols);
            assert_eq!(batch.len(), cols * n_out, "{} cols={cols}", q.nome());

            for (c, x) in xs.iter().enumerate() {
                let sozinha = q.dispatch(&ctx, &phys[0], &dev, &w, x, n_in, n_out, 1);
                assert_eq!(
                    &batch[c * n_out..(c + 1) * n_out],
                    &sozinha[..],
                    "{}: coluna {c} de {cols} divergiu do matvec de coluna única",
                    q.nome()
                );
            }
            eprintln!(
                "{} batch COLS={cols}: {cols} colunas iguais ao caminho de coluna única",
                q.nome()
            );
        }
    }
}
