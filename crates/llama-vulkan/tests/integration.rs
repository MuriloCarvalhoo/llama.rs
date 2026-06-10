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
    assert_eq!(tensor.size_bytes, bytes.len() as u64);
    eprintln!("Upload OK: {}x{} Q8_0 ({} bytes)", n_out, n_in, bytes.len());
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

    // Referência CPU do MESMO algoritmo da GPU (ativações f32, sem quantização).
    // Isola "a GPU computa o decode corretamente" de "quantização de ativação muda o
    // argmax". A GPU (wave64 subgroupAdd) e esta referência (soma sequencial) diferem
    // apenas na ordem de soma f32 (~1e-5) — o token decodificado deve ser idêntico.
    struct CpuF32ActMatmul;
    impl llama_model::GpuMatmul for CpuF32ActMatmul {
        fn matvec_q8_0(
            &self,
            w_bytes: &[u8],
            x: &[f32],
            n_in: usize,
            n_out: usize,
        ) -> Result<Vec<f32>, llama_model::ModelError> {
            Ok(cpu_ref_q8_0_f32act(w_bytes, x, n_in, n_out))
        }
    }

    let prompt = [cfg.bos_id];
    let gpu_tok = model.decode_one_gpu_owned(&prompt, &backend, &w).unwrap();
    let ref_tok = model
        .decode_one_gpu_owned(&prompt, &CpuF32ActMatmul, &w)
        .unwrap();
    // Token do caminho CPU quantizado (ativações Q8_0) — informativo: pode diferir, pois
    // é uma aproximação distinta (mais perdas) que a via f32 da GPU.
    let cpu_quant_tok = model.decode_one_cpu_owned(&prompt).unwrap();

    eprintln!(
        "Forward GPU real: gpu_tok={gpu_tok} ref_f32_tok={ref_tok} cpu_quant_tok={cpu_quant_tok}"
    );
    assert_eq!(
        gpu_tok, ref_tok,
        "GPU deve igualar a referência CPU do mesmo algoritmo (ativações f32)"
    );
}

// Referência CPU do matvec Q8_0 com ativações f32 (mesmo algoritmo do shader GPU).
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
        .matvec_q8_0(&w.output, &x1, cfg.n_embd, cfg.vocab)
        .unwrap();
    let cpu1 = cpu_ref_q8_0_f32act(&w.output, &x1, cfg.n_embd, cfg.vocab);
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
        .matvec_q8_0(&w.layers[0].ffn_down, &x2, cfg.n_ff, cfg.n_embd)
        .unwrap();
    let cpu2 = cpu_ref_q8_0_f32act(&w.layers[0].ffn_down, &x2, cfg.n_ff, cfg.n_embd);
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
