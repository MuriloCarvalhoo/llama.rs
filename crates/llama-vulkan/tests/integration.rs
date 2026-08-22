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
    let row_bytes = 34; // 1 bloco
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
    for (row, y_row) in y.iter_mut().enumerate() {
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
        *y_row = acc;
    }
    y
}

fn cpu_ref_q8_0_f32act(w: &[u8], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;
    let mut y = vec![0f32; n_out];
    for (row, y_row) in y.iter_mut().enumerate() {
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
        *y_row = acc;
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
    // O SwiGLU não é mais um dispatch próprio: ele saiu fundido com a quantização em
    // `swiglu_quant.comp` (um só dispatch por camada). O teste segue comparando
    // silu(g)*u com a CPU, agora no shader que o plano de fato executa; a metade da
    // quantização é conferida em `delta_net.rs`, contra o `quantize_x`.
    use llama_vulkan::{DnPipe, ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n = 4864usize; // múltiplo de 32, como todo `n_ff`
    let g: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.2 - 1.0).collect();
    let u: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.3 + 0.1).collect();
    let cpu: Vec<f32> = g
        .iter()
        .zip(u.iter())
        .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
        .collect();

    let saida = fwd
        .dbg_dn(
            DnPipe::SwigluQuant,
            &[
                g.clone(),
                u.clone(),
                vec![0f32; n],
                vec![0f32; n / 32 * 8],
                vec![0f32; n / 32],
            ],
            &u32::try_from(n).unwrap().to_le_bytes(),
            u32::try_from((n / 32).div_ceil(64)).unwrap(),
        )
        .expect("dispatch swiglu_quant");
    for (i, (a, b)) in cpu.iter().zip(saida[2].iter()).enumerate() {
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

/// A atenção com o KV fatiado entre workgroups tem de dar o **mesmo** resultado da
/// versão de um workgroup por cabeça — é a mesma álgebra do softmax online, só que a
/// combinação dos parciais passa a acontecer entre workgroups.
///
/// Os casos que importam: contexto longo (onde o split existe para ganhar), número de
/// posições que não divide pelo número de splits, e mais splits do que posições — aí
/// sobram fatias vazias, que não podem contaminar a soma.
#[test]
fn atencao_com_split_bate_com_a_de_um_workgroup() {
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

    for (n_head, n_head_kv, head_dim, total_len, n_split) in [
        (24, 4, 256, 3000, 16), // geometria do qwen35 com contexto longo
        (24, 4, 256, 1001, 7),  // não divide igual
        (14, 2, 64, 5, 8),      // mais splits do que posições: fatias vazias
        (40, 8, 128, 300, 4),
        // Geometria do Qwen2.5-0.5B, que é o modelo do teste no plano completo.
        (14, 2, 64, 1200, 8),
        (14, 2, 64, 1, 8),
        (14, 2, 64, 2, 8),
        (14, 2, 64, 8, 2),
        (14, 2, 64, 8, 4),
        (14, 2, 64, 40, 2),
        (14, 2, 64, 3, 2),
        (14, 2, 64, 5, 2),
        (14, 2, 64, 9, 2),
    ] {
        attention_caso_split(&fwd, n_head, n_head_kv, head_dim, total_len, n_split);
    }
}

fn attention_caso_split(
    fwd: &llama_vulkan::ResidentForward<'_>,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    total_len: usize,
    n_split: usize,
) {
    let kv_dim = n_head_kv * head_dim;
    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i % 19) as f32) * 0.05 - 0.4)
        .collect();
    let kc: Vec<f32> = (0..total_len * kv_dim)
        .map(|i| ((i % 23) as f32) * 0.03 - 0.3)
        .collect();
    let vc: Vec<f32> = (0..total_len * kv_dim)
        .map(|i| ((i % 29) as f32) * 0.02 - 0.2)
        .collect();

    let base = fwd
        .dbg_attention(&q, &kc, &vc, n_head, n_head_kv, head_dim, total_len, 1)
        .unwrap();
    let split = fwd
        .dbg_attention_split(
            &q, &kc, &vc, n_head, n_head_kv, head_dim, total_len, 1, n_split,
        )
        .unwrap();

    assert_eq!(base.len(), split.len());
    let pior = base
        .iter()
        .zip(&split)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        pior < 1e-5,
        "split={n_split} total_len={total_len} head_dim={head_dim}: pior diferença {pior:.2e}"
    );
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
        for (j, score) in scores.iter_mut().enumerate() {
            let koff = j * kv_dim + kv_h * head_dim;
            let dot: f32 = (0..head_dim).map(|dd| q[qoff + dd] * kc[koff + dd]).sum();
            *score = dot * scale;
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
        for (j, score) in scores.iter().enumerate() {
            let voff = j * kv_dim + kv_h * head_dim;
            for dd in 0..head_dim {
                cpu[qoff + dd] += score * vc[voff + dd];
            }
        }
    }

    let gpu = fwd
        .dbg_attention(&q, &kc, &vc, n_head, n_head_kv, head_dim, total_len, 1)
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

/// O GEMM com tiling em LDS (`mul_mm.comp`, experimental) contra o matvec Q4_K que ele
/// substituiria. A referência é o próprio matvec, não a CPU: os dois têm de calcular a
/// mesma coisa, e o matvec já está preso à referência de CPU por
/// `q4_k_matvec_gpu_bate_com_a_referencia_de_cpu`.
///
/// O que só quebra aqui: o tile de peso em LDS (128 linhas × 32 elementos, carregado por
/// duas threads por linha), o mapeamento de 256 threads para 4 linhas × COLS/8 colunas, e o
/// termo afim do Q4_K (`-dmin·m·soma(x)`), que sobrevive ao tiling porque `soma(x)` só
/// depende da coluna. Um `n_out` que não é múltiplo de 128 entra de propósito, para pegar o
/// workgroup incompleto do fim.
///
/// A igualdade é **relativa**: a ordem de acumulação difere (o matvec reduz no subgrupo, o
/// GEMM acumula por thread), então bit a bit não vale.
#[test]
fn gemm_em_lds_bate_com_o_matvec_q4k() {
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

    let n_in = 1024usize;
    let w = QuantK::Q4.pesos(300, n_in / 256);

    // 300 linhas: dois workgroups cheios (128 + 128) e um de 44, que exercita a guarda.
    for n_out in [128usize, 300] {
        for cols in [8usize, 16, 32] {
            let x: Vec<f32> = (0..cols * n_in)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.021)
                .collect();
            let gemm = llama_vulkan::matmul::dispatch_mul_mm_q4k(
                &ctx, &phys[0], &dev, &w, &x, n_in, n_out, cols,
            )
            .expect("dispatch mul_mm");
            let mv = QuantK::Q4.dispatch(&ctx, &phys[0], &dev, &w, &x, n_in, n_out, cols);

            assert_eq!(gemm.len(), mv.len());
            let escala = mv.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let (i_pior, pior) = gemm
                .iter()
                .zip(&mv)
                .enumerate()
                .map(|(i, (a, b))| (i, (a - b).abs() / escala))
                .fold((0, 0f32), |acc, x| if x.1 > acc.1 { x } else { acc });
            eprintln!("mul_mm n_out={n_out} cols={cols}: erro rel máx {pior:.3e}");
            assert!(
                pior < 1e-5,
                "GEMM divergiu do matvec em n_out={n_out} cols={cols}: \
                 índice {i_pior}, rel {pior} (gemm={} mv={})",
                gemm[i_pior],
                mv[i_pior]
            );
        }
    }
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

    #[allow(clippy::too_many_arguments)]
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
        // Até o teto de `batch_size()`. O Q6_K dimensionava os acumuladores por um
        // `MAX_COLS = 8` fixo em vez de por `COLS`, e era ele que prendia o batch em 8:
        // com 16 ou mais o shader escrevia fora do array. Daí os valores acima de 8 aqui.
        for cols in [2usize, 4, 8, 16, 24, 32] {
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

#[test]
fn quantize_x_em_batch_bate_coluna_a_coluna() {
    // O `quantize_x.comp` trata cada bloco de 32 de forma independente e indexa a saída
    // pelo bloco global (`xq[blk*8+g]`, `xd[blk]`). Com as colunas concatenadas, o bloco
    // global vira `c*n_blk + blk` -- exatamente o índice que o matvec com COLS>1 lê. Ou
    // seja: o shader já serve ao prefill em batch sem alteração nenhuma, bastando passar
    // `n_in = cols * n_in`. Este teste prende essa propriedade, que é uma coincidência de
    // layout e não um contrato escrito em lugar nenhum -- mexer no empacotamento do
    // `quantize_x` quebraria o batch silenciosamente.
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }
    let fwd = llama_vulkan::ResidentForward::new_pipelines_only(&ctx).unwrap();

    let qx = |x: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let n = x.len();
        let mut push = Vec::new();
        push.extend_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        let saida = fwd
            .dbg_dn(
                llama_vulkan::DnPipe::QuantizeX,
                &[x.to_vec(), vec![0f32; n / 32 * 8], vec![0f32; n / 32]],
                &push,
                u32::try_from((n / 32).div_ceil(64)).unwrap(),
            )
            .expect("dispatch quantize_x");
        (saida[1].clone(), saida[2].clone())
    };

    let n_in = 5120usize;
    let n_blk = n_in / 32;
    for cols in [2usize, 4, 8] {
        let xs: Vec<Vec<f32>> = (0..cols)
            .map(|c| {
                (0..n_in)
                    .map(|i| ((i % (37 + c)) as f32 - 18.0) * (1e-4 + c as f32 * 2e-5))
                    .collect()
            })
            .collect();

        let (bq, bd) = qx(&xs.concat());
        assert_eq!(bq.len(), cols * n_blk * 8);
        assert_eq!(bd.len(), cols * n_blk);

        for (c, x) in xs.iter().enumerate() {
            let (sq, sd) = qx(x);
            assert_eq!(
                &bq[c * n_blk * 8..(c + 1) * n_blk * 8],
                &sq[..],
                "coluna {c} de {cols}: xq do batch difere do de coluna única"
            );
            assert_eq!(
                &bd[c * n_blk..(c + 1) * n_blk],
                &sd[..],
                "coluna {c} de {cols}: xd do batch difere do de coluna única"
            );
        }
        eprintln!("quantize_x batch cols={cols}: layout por coluna confirmado");
    }
}

#[test]
fn attention_em_batch_respeita_a_mascara_causal() {
    // Prefill em batch: N tokens num dispatch só, cada um enxergando apenas as posições
    // até a sua. O invariante é que o token t do bloco produza exatamente o mesmo que
    // produziria sozinho com `total_len = pos0 + t + 1` -- que é como o decode o calcula
    // hoje. Se a máscara vazasse (todo token vendo o bloco inteiro), os tokens iniciais
    // divergiriam e os finais não: por isso a comparação é por token, e não agregada.
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

    // (n_head, n_head_kv, head_dim): geometria densa e a do Qwen2.5-32B (head_dim=128,
    // duas dimensões por lane). n_tokens cobre menos que as 8 waves, igual, e com sobra.
    for (n_head, n_head_kv, head_dim) in [(14usize, 2usize, 64usize), (40, 8, 128)] {
        for (pos0, n_tokens) in [(0usize, 4usize), (3, 8), (11, 5)] {
            let kv_dim = n_head_kv * head_dim;
            let total_len = pos0 + n_tokens; // posições vistas pelo último token do bloco

            // KV-cache com o bloco inteiro já escrito, que é o estado em que a atenção
            // roda no prefill: o kv_append do bloco vem antes dela no plano.
            let kc: Vec<f32> = (0..total_len * kv_dim)
                .map(|i| ((i % 23) as f32) * 0.03 - 0.3)
                .collect();
            let vc: Vec<f32> = (0..total_len * kv_dim)
                .map(|i| ((i % 29) as f32) * 0.02 - 0.2)
                .collect();
            // Query diferente por token: uma query repetida esconderia um erro de offset
            // que sempre lesse o token 0.
            let q: Vec<f32> = (0..n_tokens * n_head * head_dim)
                .map(|i| ((i % 19) as f32) * 0.05 - 0.4)
                .collect();

            let batch = fwd
                .dbg_attention(
                    &q, &kc, &vc, n_head, n_head_kv, head_dim, total_len, n_tokens,
                )
                .expect("attention em batch");
            assert_eq!(batch.len(), n_tokens * n_head * head_dim);

            let por_head = n_head * head_dim;
            for t in 0..n_tokens {
                let visiveis = pos0 + t + 1;
                let qt = &q[t * por_head..(t + 1) * por_head];
                let sozinho = fwd
                    .dbg_attention(
                        qt,
                        &kc[..visiveis * kv_dim],
                        &vc[..visiveis * kv_dim],
                        n_head,
                        n_head_kv,
                        head_dim,
                        visiveis,
                        1,
                    )
                    .expect("attention de token único");
                for (i, (b, s)) in batch[t * por_head..(t + 1) * por_head]
                    .iter()
                    .zip(&sozinho)
                    .enumerate()
                {
                    assert!(
                        (b - s).abs() < 1e-5,
                        "token {t}/{n_tokens} (pos0={pos0}, vê {visiveis}), elem {i}: \
                         batch={b} sozinho={s} — máscara causal errada"
                    );
                }
            }
            eprintln!(
                "attention batch n_head={n_head} head_dim={head_dim} pos0={pos0} \
                 n_tokens={n_tokens}: causal confirmada"
            );
        }
    }
}

/// A norma fundida em batch tem que dar, token a token, exatamente o mesmo que N chamadas
/// de um token só.
///
/// São dois dispatches encadeados (`NormFused` acumula as parciais, `NormP2` fecha a
/// redução e quantiza), e o batch os separa por token na dimensão Y — as parciais de um
/// token não podem vazar para o outro. O layout de saída de `xq`/`xd` também é conferido
/// aqui, porque é o que o matvec em batch lê: bloco `b` da coluna `t` em `t * n_blk + b`.
#[test]
fn norma_em_batch_bate_token_a_token() {
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        return;
    }
    let fwd = llama_vulkan::ResidentForward::new_pipelines_only(&ctx).unwrap();

    for (dim, n_tok) in [(5120usize, 2usize), (5120, 4), (2048, 3)] {
        let n_blk = dim / 32;
        // np1_wg do plano: um workgroup por 256 elementos, com teto.
        let np1 = (u32::try_from(dim.div_ceil(256)).unwrap()).clamp(1, 32);
        let eps = 1e-6f32;

        let x: Vec<f32> = (0..dim * n_tok)
            .map(|i| ((i % 61) as f32 - 30.0) * 1e-3)
            .collect();
        let r: Vec<f32> = (0..dim * n_tok)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-4)
            .collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();

        let p1_push = {
            let mut v = Vec::new();
            v.extend_from_slice(&u32::try_from(dim).unwrap().to_le_bytes());
            v.extend_from_slice(&1u32.to_le_bytes()); // tem_residual
            v
        };
        let p2_push = {
            let mut v = Vec::new();
            v.extend_from_slice(&u32::try_from(dim).unwrap().to_le_bytes());
            v.extend_from_slice(&eps.to_le_bytes());
            v.extend_from_slice(&np1.to_le_bytes());
            v
        };

        // Batch: um dispatch com Y = n_tok.
        let p1 = fwd
            .dbg_dn_xy(
                llama_vulkan::DnPipe::NormFused,
                &[x.clone(), r.clone(), vec![0f32; np1 as usize * n_tok]],
                &p1_push,
                np1,
                u32::try_from(n_tok).unwrap(),
            )
            .expect("norm_fused em batch");
        let p2 = fwd
            .dbg_dn_xy(
                llama_vulkan::DnPipe::NormP2,
                &[
                    p1[0].clone(),
                    w.clone(),
                    p1[2].clone(),
                    vec![0f32; dim * n_tok],
                    vec![0f32; n_blk * 8 * n_tok],
                    vec![0f32; n_blk * n_tok],
                ],
                &p2_push,
                u32::try_from(n_blk.div_ceil(64)).unwrap(),
                u32::try_from(n_tok).unwrap(),
            )
            .expect("norm_p2 em batch");

        // Referência: cada token sozinho, exatamente como o decode faz.
        for t in 0..n_tok {
            let xs = x[t * dim..(t + 1) * dim].to_vec();
            let rs = r[t * dim..(t + 1) * dim].to_vec();
            let s1 = fwd
                .dbg_dn(
                    llama_vulkan::DnPipe::NormFused,
                    &[xs, rs, vec![0f32; np1 as usize]],
                    &p1_push,
                    np1,
                )
                .expect("norm_fused sozinho");
            let s2 = fwd
                .dbg_dn(
                    llama_vulkan::DnPipe::NormP2,
                    &[
                        s1[0].clone(),
                        w.clone(),
                        s1[2].clone(),
                        vec![0f32; dim],
                        vec![0f32; n_blk * 8],
                        vec![0f32; n_blk],
                    ],
                    &p2_push,
                    u32::try_from(n_blk.div_ceil(64)).unwrap(),
                )
                .expect("norm_p2 sozinho");

            // Saída normalizada e escalas: bit a bit, é o mesmo cálculo na mesma ordem.
            assert_eq!(
                &p2[3][t * dim..(t + 1) * dim],
                &s2[3][..],
                "dim={dim} n_tok={n_tok}: normed do token {t} divergiu"
            );
            assert_eq!(
                &p2[5][t * n_blk..(t + 1) * n_blk],
                &s2[5][..],
                "dim={dim} n_tok={n_tok}: escalas do token {t} divergiram"
            );
            // `xq` é `int` no shader: comparar os **bits**, porque lido como f32 um
            // empacotamento de int8 pode formar NaN, e NaN != NaN falharia à toa.
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&p2[4][t * n_blk * 8..(t + 1) * n_blk * 8]),
                bits(&s2[4][..]),
                "dim={dim} n_tok={n_tok}: int8 do token {t} divergiu"
            );
        }
        eprintln!("norma batch dim={dim} n_tok={n_tok}: idêntica token a token");
    }
}

/// O rope em batch tem que girar cada token pelo seu próprio ângulo.
///
/// `pos` no push é a posição do **último** token do bloco, então o token `t` de um bloco de
/// `n_tok` gira por `pos - (n_tok - 1) + t`. Com n_tok=1 isso é exatamente `pos`, que é o
/// que o decode sempre fez.
#[test]
fn rope_em_batch_usa_a_posicao_de_cada_token() {
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        return;
    }
    let fwd = llama_vulkan::ResidentForward::new_pipelines_only(&ctx).unwrap();

    let (n_head, head_dim, rope_dim) = (8usize, 64usize, 64usize);
    let freq: Vec<f32> = (0..rope_dim / 2)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / rope_dim as f32))
        .collect();

    for (pos0, n_tok) in [(0usize, 2usize), (5, 4), (17, 3)] {
        let por_tok = n_head * head_dim;
        let pos_ultimo = pos0 + n_tok - 1;
        let x: Vec<f32> = (0..por_tok * n_tok)
            .map(|i| ((i % 53) as f32 - 26.0) * 1e-2)
            .collect();

        let mut xb = x.clone();
        let batch = fwd
            .dbg_rope_xy(
                &mut xb,
                n_head,
                head_dim,
                rope_dim,
                &freq,
                pos_ultimo,
                u32::try_from(n_tok).unwrap(),
            )
            .expect("rope em batch");

        for t in 0..n_tok {
            let mut xs = x[t * por_tok..(t + 1) * por_tok].to_vec();
            let um = fwd
                .dbg_rope(&mut xs, n_head, head_dim, rope_dim, &freq, pos0 + t)
                .expect("rope de um token");
            assert_eq!(
                &batch[t * por_tok..(t + 1) * por_tok],
                &um[..],
                "pos0={pos0} n_tok={n_tok}: token {t} girou pelo ângulo errado"
            );
        }
        eprintln!("rope batch pos0={pos0} n_tok={n_tok}: cada token no seu ângulo");
    }
}

// ─── Prefill em batch: o bloco tem que dar os mesmos logits que token a token ────

/// O prefill em batch é uma reorganização do mesmo cálculo: os N tokens do bloco veem as
/// mesmas posições e o mesmo KV-cache que veriam um a um. Se algum shader batchado errar o
/// offset do token, a divergência aparece aqui — nos logits do fim do prompt.
///
/// O prompt tem tamanho **não múltiplo** do batch de propósito: o resto cai no caminho
/// token a token, e é a emenda entre os dois que costuma quebrar (`pos` do RoPE, `pos0`
/// do append no KV-cache).
#[test]
fn prefill_em_batch_bate_com_token_a_token() {
    use llama_model::GpuResidentDecode;
    use llama_tokenizer::Tokenizer;
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
    let tok = Tokenizer::from_gguf(&f).unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config).unwrap();
    let aux = model.gpu_aux_weights().unwrap();
    let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux).unwrap();

    let nb = backend.batch_size();
    if nb < 2 {
        eprintln!("LLAMA_RS_BATCH=1 — nada a comparar");
        return;
    }
    let mut seq: Vec<u32> = Vec::new();
    while seq.len() < 2 * nb + 3 {
        seq.extend(tok.encode(
            "The quick brown fox jumps over the lazy dog.",
            seq.is_empty(),
        ));
    }
    seq.truncate(2 * nb + 3);

    backend.reset();
    let mut ref_logits = Vec::new();
    for (pos, &t) in seq.iter().enumerate() {
        ref_logits = backend.decode(t, pos).unwrap();
    }

    backend.reset();
    let mut logits = Vec::new();
    let mut pos = 0usize;
    while seq.len() - pos >= nb {
        logits = backend.decode_batch(&seq[pos..pos + nb], pos).unwrap();
        pos += nb;
    }
    assert!(pos > 0, "o prompt tinha que cobrir ao menos um bloco");
    for &t in &seq[pos..] {
        logits = backend.decode(t, pos).unwrap();
        pos += 1;
    }

    assert_eq!(ref_logits.len(), logits.len());
    let max_abs = ref_logits
        .iter()
        .fold(0.0f32, |m, &v| m.max(v.abs()))
        .max(1e-6);
    let max_rel = ref_logits
        .iter()
        .zip(&logits)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs() / max_abs));
    eprintln!(
        "prefill: {} tokens em blocos de {nb} (+{} avulsos), erro relativo máximo {max_rel:.2e}",
        seq.len(),
        seq.len() % nb
    );
    assert!(max_rel < 1e-3, "erro relativo {max_rel} deve ser < 1e-3");
    assert_eq!(
        argmax_u32(&ref_logits),
        argmax_u32(&logits),
        "o token escolhido tem que ser o mesmo"
    );
}

fn argmax_u32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
            if x > acc.1 { (i, x) } else { acc }
        })
        .0
}

/// O caminho fatiado dentro do **plano** (não só no harness isolado): com contexto longo
/// o decode não pode ficar mais longe da referência de CPU do que o caminho de um
/// workgroup por cabeça.
///
/// A comparação é contra a CPU, e não entre os dois caminhos, por um motivo medido: as
/// duas somas do softmax são reassociadas de formas diferentes e diferem ~1e-7 logo
/// depois da atenção — mas num 0.5B essa diferença **amplifica** ao longo das 24 camadas
/// e chega a ~1e-2 nos logits (com 3 camadas, `LLAMA_RS_STOP_LAYER=3`, o erro entre eles
/// é 1.3e-7). O erro de cada caminho contra a CPU é da mesma ordem, então o que
/// distingue implementação certa de errada aqui é **não ficar pior que o baseline**.
#[test]
fn decode_com_kv_fatiado_nao_fica_pior_que_o_caminho_curto() {
    use llama_vulkan::{ResidentForward, VulkanContext, forcar_splits};

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
    let gpu: &dyn llama_model::GpuResidentDecode = &backend;
    let tok = llama_tokenizer::Tokenizer::from_gguf(&f).unwrap();

    let mut seq: Vec<u32> = Vec::new();
    while seq.len() < 40 {
        seq.extend(tok.encode(
            "The quick brown fox jumps over the lazy dog. ",
            seq.is_empty(),
        ));
    }
    seq.truncate(40);

    let rodar = |splits: u32| -> Vec<f32> {
        forcar_splits(splits);
        gpu.reset();
        let mut logits = Vec::new();
        for (pos, &t) in seq.iter().enumerate() {
            logits = gpu.decode(t, pos).unwrap();
        }
        logits
    };

    let cpu = model.decode_one_cpu_logits(&seq).unwrap();
    let escala = cpu.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let erro = |g: &[f32]| -> f32 {
        g.iter()
            .zip(&cpu)
            .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs() / escala))
    };

    let e_curto = erro(&rodar(1));
    let mut piores = Vec::new();
    for n in [2u32, 4, 8, 16] {
        let e = erro(&rodar(n));
        eprintln!("{n} fatias: erro vs CPU {e:.2e} (caminho curto: {e_curto:.2e})");
        piores.push((n, e));
    }
    forcar_splits(0);

    for (n, e) in piores {
        assert!(
            e <= e_curto * 2.0 + 1e-3,
            "{n} fatias ficou pior que o caminho curto: {e:.2e} contra {e_curto:.2e}"
        );
    }
}

/// O ganho que motiva o split: com o KV longo, fatiar tem de ser **muito** mais rápido.
///
/// O limite é folgado de propósito (2×) — o que se protege aqui é a característica, não
/// um número: se um dia o kernel voltar a serializar o laço, o teste cai.
#[test]
fn atencao_fatiada_e_mais_rapida_com_kv_longo() {
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

    // Geometria do Qwen3.8-27B com um contexto de agente.
    let (n_head, n_head_kv, head_dim, total_len) = (24, 4, 256, 26472);
    let kv_dim = n_head_kv * head_dim;
    let q: Vec<f32> = (0..n_head * head_dim)
        .map(|i| ((i % 19) as f32) * 0.05 - 0.4)
        .collect();
    let kc: Vec<f32> = (0..total_len * kv_dim)
        .map(|i| ((i % 23) as f32) * 0.03 - 0.3)
        .collect();
    let vc: Vec<f32> = (0..total_len * kv_dim)
        .map(|i| ((i % 29) as f32) * 0.02 - 0.2)
        .collect();

    let curto = fwd
        .dbg_attention_bench(&q, &kc, &vc, n_head, n_head_kv, head_dim, total_len, 1, 5)
        .unwrap();
    let fatiado = fwd
        .dbg_attention_bench(&q, &kc, &vc, n_head, n_head_kv, head_dim, total_len, 16, 5)
        .unwrap();

    eprintln!(
        "atenção com {total_len} posições: 1 workgroup/cabeça {:.1} ms | 16 fatias {:.1} ms",
        curto * 1e3,
        fatiado * 1e3
    );
    assert!(
        fatiado * 2.0 < curto,
        "fatiar não acelerou: {:.1} ms contra {:.1} ms",
        fatiado * 1e3,
        curto * 1e3
    );
}
