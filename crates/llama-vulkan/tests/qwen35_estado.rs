//! O estado recorrente do qwen35 tem que **acumular** entre tokens.
//!
//! Sem memória, cada camada linear vira função só do token atual e o modelo passa a
//! repetir o último token do prompt — que foi exatamente o sintoma observado.

const MODELO: &str = "../../models/Qwen3.8-27B-Q5_K_M.gguf";

#[allow(unsafe_code)]
fn mapear() -> Option<memmap2::Mmap> {
    let file = std::fs::File::open(MODELO).ok()?;
    unsafe { memmap2::Mmap::map(&file) }.ok()
}

#[test]
fn estado_recorrente_acumula_entre_tokens() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8 ausente — pulando");
        return;
    };
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("menos de 2 GPUs — pulando");
        return;
    }
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    cfg.ctx = 64;
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let backend = llama_vulkan::LayerSplitForward::new(&ctx, &cfg, &raw, &aux).unwrap();
    use llama_model::GpuResidentDecode;

    backend.reset();
    let zerado = backend.dbg_estado_delta(0, 0).expect("camada 0 é linear");
    let soma_zerado: f32 = zerado.iter().map(|v| v.abs()).sum();
    assert!(
        soma_zerado == 0.0,
        "reset deve zerar o estado, soma={soma_zerado}"
    );

    backend.decode(46, 0).unwrap();
    let t1 = backend.dbg_estado_delta(0, 0).unwrap();
    let soma1: f32 = t1.iter().map(|v| v.abs()).sum();
    assert!(soma1 > 0.0, "o primeiro token deve escrever no estado");

    backend.decode(341, 1).unwrap();
    let t2 = backend.dbg_estado_delta(0, 0).unwrap();
    let dif: f32 = t1
        .iter()
        .zip(&t2)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let soma2: f32 = t2.iter().map(|v| v.abs()).sum();
    eprintln!("estado: soma t1={soma1:.3e} t2={soma2:.3e} dif máx={dif:.3e}");
    assert!(
        dif > 1e-6,
        "o segundo token tem que alterar o estado (dif={dif})"
    );
}

/// O hidden state depois da camada 0 tem que bater com o `l_out-0` do llama.cpp.
/// Rode com `LLAMA_RS_STOP_LAYER=1`.
#[test]
fn hidden_apos_camada_0() {
    if std::env::var("LLAMA_RS_STOP_LAYER").as_deref() != Ok("1") {
        eprintln!("exige LLAMA_RS_STOP_LAYER=1 — pulando");
        return;
    }
    let Some(bytes) = mapear() else { return };
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().len() < 2 {
        return;
    }
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    cfg.ctx = 64;
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let backend = llama_vulkan::LayerSplitForward::new(&ctx, &cfg, &raw, &aux).unwrap();
    use llama_model::GpuResidentDecode;
    backend.reset();
    backend.decode(46, 0).unwrap();
    let h = backend.dbg_hidden(0).unwrap();
    eprintln!(
        "l_out-0 (llama-rs): [{:.4}, {:.4}, {:.4}, ..., {:.4}, {:.4}, {:.4}]",
        h[0],
        h[1],
        h[2],
        h[h.len() - 3],
        h[h.len() - 2],
        h[h.len() - 1]
    );
    eprintln!("   llama.cpp esperado: [-0.0689, 0.0472, -0.0453, ..., -0.0847, -0.0933, -0.0469]");
}

/// Bissecção dos buffers da camada 0 contra o dump do llama.cpp.
/// Rode com `LLAMA_RS_STOP_LAYER=1`.
#[test]
fn buffers_da_camada_0() {
    if std::env::var("LLAMA_RS_STOP_LAYER").as_deref() != Ok("1") {
        eprintln!("exige LLAMA_RS_STOP_LAYER=1 — pulando");
        return;
    }
    let Some(bytes) = mapear() else { return };
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().len() < 2 {
        return;
    }
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    cfg.ctx = 64;
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let backend = llama_vulkan::LayerSplitForward::new(&ctx, &cfg, &raw, &aux).unwrap();
    use llama_model::GpuResidentDecode;
    backend.reset();
    backend.decode(46, 0).unwrap();
        // O matvec do ssm_out, refeito na CPU a partir do `normed` que a própria GPU
    // produziu: se `proj` não sair disto, o erro está entre os dois.
    if let (Some(normed), Some(proj)) = (
        backend.dbg_dn_buf(0, "normed"),
        backend.dbg_dn_buf(0, "proj"),
    ) {
        let llama_model::MixerRaw::Delta { ssm_out, .. } = &raw.layers[0].mixer else {
            panic!()
        };
        let deq = ggml_cpu::dequant_to_f32(ssm_out.bytes, ssm_out.ty).unwrap();
        let n_in = normed.len();
        let esperado: Vec<f32> = (0..cfg.n_embd)
            .map(|r| {
                deq[r * n_in..(r + 1) * n_in]
                    .iter()
                    .zip(&normed)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        let esc = esperado.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-9);
        let rel = proj
            .iter()
            .zip(&esperado)
            .fold(0f32, |m, (&a, &b)| m.max((a - b).abs() / esc));
        eprintln!(
            "ssm_out sobre o normed da GPU: erro relativo {rel:.3e}\n  proj={:?}\n  esper={:?}",
            &proj[..3],
            &esperado[..3]
        );
        // O mesmo matvec pelo caminho de dispatch isolado, com a ativação real.
        let dev = llama_vulkan::VulkanDevice::create(&ctx, &ctx.amd_compute_devices()[0]).unwrap();
        let iso = llama_vulkan::matmul::dispatch_q6_k_matvec(
            &ctx,
            &ctx.amd_compute_devices()[0],
            &dev,
            ssm_out.bytes,
            &normed,
            n_in,
            cfg.n_embd,
            1,
        )
        .unwrap();
        eprintln!("  isolado={:?}", &iso[..3]);
    }
    for nome in ["qkv", "conv", "gb", "qn", "out", "normed", "proj", "x"] {
        if let Some(v) = backend.dbg_dn_buf(0, nome) {
            eprintln!(
                "{nome:7}: [{:.4}, {:.4}, {:.4}, ..., {:.4}, {:.4}]",
                v[0],
                v[1],
                v[2],
                v[v.len() - 2],
                v[v.len() - 1]
            );
        }
    }
}

/// O matvec do `ssm_out` (Q6_K, 6144 -> 5120) contra a referência de CPU.
///
/// Foi onde a bissecção parou: `final_output` batia exatamente com o llama.cpp, mas a
/// saída da camada não.
#[test]
fn matvec_ssm_out_bate_com_a_cpu() {
    let Some(bytes) = mapear() else { return };
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        return;
    }
    let dev = llama_vulkan::VulkanDevice::create(&ctx, &phys[0]).unwrap();
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    let dn = cfg.delta_net.clone().unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let llama_model::MixerRaw::Delta { ssm_out, .. } = &raw.layers[0].mixer else {
        panic!("camada 0 devia ser linear");
    };
    assert_eq!(ssm_out.ty, gguf::GgmlType::Q6_K);

    let n_in = dn.head_v_dim() * dn.n_v_heads; // 6144
    let n_out = cfg.n_embd; // 5120
    // Ativação com a mesma ordem de grandeza da real (~1e-3).
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-4)
        .collect();

    let gpu = llama_vulkan::matmul::dispatch_q6_k_matvec(
        &ctx,
        &phys[0],
        &dev,
        ssm_out.bytes,
        &x,
        n_in,
        n_out,
        1,
    )
    .expect("dispatch");

    // Referência: desquantiza e faz o produto interno contra a mesma ativação int8.
    let deq = ggml_cpu::dequant_to_f32(ssm_out.bytes, gguf::GgmlType::Q6_K).unwrap();
    assert_eq!(deq.len(), n_in * n_out);
    let mut xi8 = vec![0f32; n_in];
    for b in 0..n_in / 32 {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for i in 0..32 {
            xi8[b * 32 + i] = (blk[i] * inv).round().clamp(-127.0, 127.0) * d;
        }
    }
    let cpu: Vec<f32> = (0..n_out)
        .map(|r| {
            deq[r * n_in..(r + 1) * n_in]
                .iter()
                .zip(&xi8)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();

    let escala = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-9);
    let rel = gpu
        .iter()
        .zip(&cpu)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs() / escala));
    eprintln!(
        "ssm_out {n_in}->{n_out}: erro relativo {rel:.3e}\n  gpu[0..3]={:?}\n  cpu[0..3]={:?}",
        &gpu[..3],
        &cpu[..3]
    );
    assert!(rel < 1e-4, "matvec Q6_K divergiu: {rel}");
}

/// O `quantize_x` da GPU contra a versão de host, nos tamanhos que o qwen35 usa.
#[test]
fn quantize_x_nos_tamanhos_do_qwen35() {
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        return;
    }
    let fwd = llama_vulkan::ResidentForward::new_pipelines_only(&ctx).unwrap();
    for n in [5120usize, 6144, 10240] {
        // Magnitude parecida com a saída da norma gated (~1e-3).
        let x: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 1e-4).collect();
        let mut push = Vec::new();
        push.extend_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        let saida = fwd
            .dbg_dn(
                llama_vulkan::DnPipe::QuantizeX,
                &[x.clone(), vec![0f32; n / 32 * 8], vec![0f32; n / 32]],
                &push,
                u32::try_from((n / 32).div_ceil(64)).unwrap(),
            )
            .expect("dispatch quantize_x");
        let xd = &saida[2];
        let mut pior = 0f32;
        for b in 0..n / 32 {
            let amax = x[b * 32..b * 32 + 32]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
            let esperado = amax / 127.0;
            pior = pior.max((xd[b] - esperado).abs() / esperado.max(1e-12));
        }
        eprintln!("quantize_x n={n}: pior erro relativo nas escalas {pior:.3e}");
        assert!(pior < 1e-5, "escalas erradas com n={n}: {pior}");
    }
}
