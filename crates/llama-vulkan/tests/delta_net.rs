//! Shaders da atenção linear (qwen35) contra a referência f32 de `llama_model::delta_net`.
//!
//! São os kernels que substituem a atenção em 3 de cada 4 camadas do Qwen3.8. Cada teste
//! roda o shader e a referência sobre os mesmos dados e compara **tudo que o shader
//! escreve** — inclusive o estado recorrente e a janela da convolução, que são `inout`:
//! um erro na atualização do estado não aparece na saída do primeiro token, só alguns
//! tokens depois.

use llama_vulkan::{DnPipe, ResidentForward, VulkanContext};

fn ctx_fwd() -> Option<(VulkanContext, ())> {
    let ctx = VulkanContext::new().ok()?;
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem device AMD — pulando");
        return None;
    }
    Some((ctx, ()))
}

/// Bytes de um push constant qualquer.
fn push_bytes<T: Copy>(p: &T) -> Vec<u8> {
    let n = std::mem::size_of::<T>();
    // SAFETY: T é POD (struct de u32/f32 repr(C)); lemos exatamente o seu tamanho.
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(p).cast::<u8>(), n) }.to_vec()
}

fn pseudo(n: usize, semente: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i.wrapping_mul(2_654_435_761).wrapping_add(semente) % 1000) as f32;
            x / 500.0 - 1.0
        })
        .collect()
}

fn max_dif(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn delta_net_bate_com_a_referencia_de_cpu() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let d = 128usize; // d_state do Qwen3.8
    let n_heads = 6usize;
    let estado0 = pseudo(n_heads * d * d, 1);
    let q = pseudo(n_heads * d, 2);
    let k = pseudo(n_heads * d, 3);
    let v = pseudo(n_heads * d, 4);
    // (g, beta) por cabeça: g < 0 (decaimento), beta em (0,1).
    let gb: Vec<f32> = (0..n_heads)
        .flat_map(|h| {
            let g = -0.05 * (h as f32 + 1.0);
            let beta = 0.3 + 0.1 * h as f32;
            [g, beta]
        })
        .collect();

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct P {
        d: u32,
        n_heads: u32,
        rep: u32,
    }
    // rep = 1: uma cabeça de chave por cabeça de valor (o caso sem GQA).
    let push = push_bytes(&P {
        d: d as u32,
        n_heads: n_heads as u32,
        rep: 1,
    });

    let saida = fwd
        .dbg_dn(
            DnPipe::DeltaNet,
            &[
                estado0.clone(),
                q.clone(),
                k.clone(),
                v.clone(),
                gb.clone(),
                vec![0f32; n_heads * d],
            ],
            &push,
            // n_heads * (d / 4): quatro colunas do estado por workgroup.
            (n_heads * d / 4) as u32,
        )
        .expect("dispatch delta_net");
    let estado_gpu = &saida[0];
    let out_gpu = &saida[5];

    // Referência: um passo por cabeça, sobre o mesmo estado inicial.
    let mut estado_cpu = estado0.clone();
    let mut out_cpu = vec![0f32; n_heads * d];
    for h in 0..n_heads {
        llama_model::delta_net::delta_net_step(
            &mut estado_cpu[h * d * d..(h + 1) * d * d],
            &q[h * d..(h + 1) * d],
            &k[h * d..(h + 1) * d],
            &v[h * d..(h + 1) * d],
            gb[h * 2],
            gb[h * 2 + 1],
            &mut out_cpu[h * d..(h + 1) * d],
        );
    }

    let dif_estado = max_dif(estado_gpu, &estado_cpu);
    let dif_out = max_dif(out_gpu, &out_cpu);
    eprintln!("delta_net: dif estado={dif_estado:.3e} saída={dif_out:.3e}");
    assert!(
        dif_estado < 1e-4,
        "estado recorrente divergiu: {dif_estado}"
    );
    assert!(dif_out < 1e-4, "saída divergiu: {dif_out}");
}

#[test]
fn conv_causal_bate_com_a_referencia_e_desliza_a_janela() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let canais = 512usize;
    let d_conv = 4usize;
    let estado0 = pseudo(canais * (d_conv - 1), 11);
    let x = pseudo(canais, 12);
    let w = pseudo(canais * d_conv, 13);

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct P {
        canais: u32,
        d_conv: u32,
        pad: u32,
    }
    let push = push_bytes(&P {
        canais: canais as u32,
        d_conv: d_conv as u32,
        pad: 0,
    });

    let saida = fwd
        .dbg_dn(
            DnPipe::Conv,
            &[estado0.clone(), x.clone(), w.clone(), vec![0f32; canais]],
            &push,
            u32::try_from(canais.div_ceil(64)).unwrap(),
        )
        .expect("dispatch dn_conv");

    let mut estado_cpu = estado0.clone();
    let bruto = llama_model::delta_net::conv1d_step(&mut estado_cpu, &x, &w, d_conv);
    // O shader já aplica SiLU na saída (a camada faz isso logo depois).
    let esperado: Vec<f32> = bruto
        .iter()
        .map(|&v| llama_model::delta_net::silu(v))
        .collect();

    let dif_estado = max_dif(&saida[0], &estado_cpu);
    let dif_out = max_dif(&saida[3], &esperado);
    eprintln!("dn_conv: dif janela={dif_estado:.3e} saída={dif_out:.3e}");
    assert!(
        dif_estado < 1e-6,
        "janela da convolução divergiu: {dif_estado}"
    );
    assert!(dif_out < 1e-5, "saída divergiu: {dif_out}");
}

#[test]
fn gates_batem_com_a_referencia_de_cpu() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n_embd = 512usize;
    let n_heads = 12usize;
    let x = pseudo(n_embd, 21);
    let alpha = pseudo(n_heads * n_embd, 22);
    let beta = pseudo(n_heads * n_embd, 23);
    // (ssm_a, dt_bias) por cabeça; ssm_a é sempre negativo no modelo real.
    let adt: Vec<f32> = (0..n_heads)
        .flat_map(|h| [-(0.5 + 0.1 * h as f32), 0.2 * h as f32 - 0.5])
        .collect();

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct P {
        n_embd: u32,
        n_heads: u32,
        pad: u32,
    }
    let push = push_bytes(&P {
        n_embd: n_embd as u32,
        n_heads: n_heads as u32,
        pad: 0,
    });

    let saida = fwd
        .dbg_dn(
            DnPipe::Gates,
            &[
                x.clone(),
                alpha.clone(),
                beta.clone(),
                adt.clone(),
                vec![0f32; n_heads * 2],
            ],
            &push,
            n_heads as u32,
        )
        .expect("dispatch dn_gates");

    let mut esperado = vec![0f32; n_heads * 2];
    for h in 0..n_heads {
        let pa: f32 = (0..n_embd).map(|i| alpha[h * n_embd + i] * x[i]).sum();
        let pb: f32 = (0..n_embd).map(|i| beta[h * n_embd + i] * x[i]).sum();
        esperado[h * 2] = adt[h * 2] * llama_model::delta_net::softplus(pa + adt[h * 2 + 1]);
        esperado[h * 2 + 1] = llama_model::delta_net::sigmoid(pb);
    }
    // Erro **relativo**: o produto interno de n_embd termos é uma redução em árvore na
    // GPU e uma soma sequencial na CPU, então a diferença cresce com a magnitude do
    // resultado, não com a do erro de cada operação.
    let escala = esperado.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let dif = max_dif(&saida[4], &esperado) / escala;
    eprintln!("dn_gates: dif relativa={dif:.3e}");
    assert!(dif < 1e-5, "gates divergiram: {dif}");
}

#[test]
fn norm_l2_e_norm_gated_batem_com_a_referencia() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let dim = 128usize;
    let n_heads = 8usize;
    let eps = 1e-6f32;
    let x = pseudo(n_heads * dim, 31);
    let w = pseudo(dim, 32);
    let z = pseudo(n_heads * dim, 33);

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct P {
        dim: u32,
        n_heads: u32,
        modo: u32,
        eps: f32,
    }

    // Modo 0: L2 por cabeça.
    let saida = fwd
        .dbg_dn(
            DnPipe::Norm,
            &[x.clone(), w.clone(), z.clone(), vec![0f32; n_heads * dim]],
            &push_bytes(&P {
                dim: dim as u32,
                n_heads: n_heads as u32,
                modo: 0,
                eps,
            }),
            n_heads as u32,
        )
        .expect("dispatch dn_norm L2");
    let mut esperado = x.clone();
    llama_model::delta_net::l2_norm_rows(&mut esperado, dim, eps);
    let dif_l2 = max_dif(&saida[3], &esperado);

    // Modo 1: rmsnorm(w) * silu(z).
    let saida = fwd
        .dbg_dn(
            DnPipe::Norm,
            &[x.clone(), w.clone(), z.clone(), vec![0f32; n_heads * dim]],
            &push_bytes(&P {
                dim: dim as u32,
                n_heads: n_heads as u32,
                modo: 1,
                eps,
            }),
            n_heads as u32,
        )
        .expect("dispatch dn_norm gated");
    let mut esperado_g = vec![0f32; n_heads * dim];
    for h in 0..n_heads {
        let fatia = &x[h * dim..(h + 1) * dim];
        let ss: f32 = fatia.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let escala = 1.0 / (ss + eps).sqrt();
        for i in 0..dim {
            esperado_g[h * dim + i] =
                fatia[i] * escala * w[i] * llama_model::delta_net::silu(z[h * dim + i]);
        }
    }
    let dif_gated = max_dif(&saida[3], &esperado_g);

    eprintln!("dn_norm: dif L2={dif_l2:.3e} gated={dif_gated:.3e}");
    assert!(dif_l2 < 1e-6, "L2 divergiu: {dif_l2}");
    assert!(dif_gated < 1e-5, "norma gated divergiu: {dif_gated}");
}
