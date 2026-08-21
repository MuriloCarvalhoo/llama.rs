//! Compara valores intermediários da camada 0 do qwen35 com o dump do
//! `llama-eval-callback` do llama.cpp.
//!
//! Exemplo de diagnóstico: abortar na primeira falha é o comportamento desejado.
//! `indexing_slicing` entra na lista pelo mesmo motivo: um índice fora do dump é
//! sinal de que o modelo mudou de forma, e parar ali é o que se quer.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

#[allow(unsafe_code)]
fn main() {
    let path = "models/Qwen3.8-27B-Q5_K_M.gguf";
    let file = std::fs::File::open(path).unwrap();
    let bytes = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    let dn = cfg.delta_net.clone().unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();

    let tok = 46usize;
    let n = cfg.n_embd;
    let emb = &aux.token_embd[tok * n..(tok + 1) * n];
    let ss: f32 = emb.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let esc = 1.0 / (ss + cfg.rms_eps).sqrt();
    let w = &aux.layers[0].attn_norm;
    let x: Vec<f32> = emb.iter().zip(w).map(|(a, b)| a * esc * b).collect();

    let mostra = |nome: &str, v: &[f32]| {
        let m = v.len();
        println!(
            "{nome}: [{:.4}, {:.4}, {:.4}, ..., {:.4}] soma={:.6}",
            v[0],
            v[1],
            v[2],
            v[m - 1],
            v.iter().sum::<f32>()
        );
    };

    // qkv = W · x, com W quantizado row-major [n_out][n_in].
    let matvec = |t: &llama_model::QTensor<'_>, n_in: usize, n_out: usize, x: &[f32]| -> Vec<f32> {
        let deq = ggml_cpu::dequant_to_f32(t.bytes, t.ty).expect("dequant");
        assert_eq!(deq.len(), n_in * n_out, "shape do peso");
        (0..n_out)
            .map(|r| {
                deq[r * n_in..(r + 1) * n_in]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    };
    let conv_dim = dn.d_state * dn.n_k_heads * 2 + dn.head_v_dim() * dn.n_v_heads;
    let llama_model::MixerRaw::Delta { attn_qkv, .. } = &raw.layers[0].mixer else {
        panic!("camada 0 devia ser linear");
    };
    let qkv = matvec(attn_qkv, n, conv_dim, &x);
    mostra("qkv    ", &qkv);

    // gates: alpha e beta são projeções f32.
    let da = aux.layers[0].delta.as_ref().unwrap();
    let proj = |w: &[f32], nh: usize| -> Vec<f32> {
        (0..nh)
            .map(|h| {
                w[h * n..(h + 1) * n]
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    };
    let alpha: Vec<f32> = proj(&da.alpha, dn.n_v_heads);
    let g: Vec<f32> = alpha
        .iter()
        .enumerate()
        .map(|(h, a)| da.a[h] * llama_model::delta_net::softplus(a + da.dt_bias[h]))
        .collect();
    mostra("g      ", &g);
    let beta: Vec<f32> = proj(&da.beta, dn.n_v_heads)
        .iter()
        .map(|v| llama_model::delta_net::sigmoid(*v))
        .collect();
    mostra("beta   ", &beta);

    // Convolução causal com janela zerada (primeiro token) e SiLU.
    let mut janela = vec![0f32; conv_dim * (dn.d_conv - 1)];
    let conv_bruto = llama_model::delta_net::conv1d_step(&mut janela, &qkv, &da.conv1d, dn.d_conv);
    let conv: Vec<f32> = conv_bruto
        .iter()
        .map(|&v| llama_model::delta_net::silu(v))
        .collect();
    mostra("conv   ", &conv);

    // q e k normalizados em L2 por cabeça de chave; v direto.
    let key_dim = dn.d_state * dn.n_k_heads;
    let mut q: Vec<f32> = conv[..key_dim].to_vec();
    let mut k: Vec<f32> = conv[key_dim..2 * key_dim].to_vec();
    let v = &conv[2 * key_dim..];
    llama_model::delta_net::l2_norm_rows(&mut q, dn.d_state, cfg.rms_eps);
    llama_model::delta_net::l2_norm_rows(&mut k, dn.d_state, cfg.rms_eps);

    // Recorrência por cabeça de valor, com GQA (cada cabeça de chave serve `rep`).
    let d = dn.d_state;
    let rep = dn.n_v_heads / dn.n_k_heads;
    let mut estado = vec![0f32; dn.state_len()];
    let mut out = vec![0f32; dn.n_v_heads * d];
    for h in 0..dn.n_v_heads {
        let hk = h / rep;
        let (ini, fim) = (h * d * d, (h + 1) * d * d);
        let mut o = vec![0f32; d];
        llama_model::delta_net::delta_net_step(
            &mut estado[ini..fim],
            &q[hk * d..(hk + 1) * d],
            &k[hk * d..(hk + 1) * d],
            &v[h * d..(h + 1) * d],
            g[h],
            beta[h],
            &mut o,
        );
        out[h * d..(h + 1) * d].copy_from_slice(&o);
    }
    mostra("attnout", &out);
    let soma_abs: f32 = out.iter().map(|v| v.abs()).sum();
    println!("attnout soma_abs={soma_abs:.6}");

    // Norma gated: rmsnorm por cabeça × silu(z).
    let z = matvec(
        match &raw.layers[0].mixer {
            llama_model::MixerRaw::Delta { attn_gate, .. } => attn_gate,
            llama_model::MixerRaw::Attn { .. } => unreachable!(),
        },
        n,
        dn.head_v_dim() * dn.n_v_heads,
        &x,
    );
    let mut normed = vec![0f32; out.len()];
    for h in 0..dn.n_v_heads {
        let fatia = &out[h * d..(h + 1) * d];
        let ss: f32 = fatia.iter().map(|t| t * t).sum::<f32>() / d as f32;
        let e = 1.0 / (ss + cfg.rms_eps).sqrt();
        for i in 0..d {
            normed[h * d + i] =
                fatia[i] * e * da.norm[i] * llama_model::delta_net::silu(z[h * d + i]);
        }
    }
    mostra("final  ", &normed);
    let saida = matvec(
        match &raw.layers[0].mixer {
            llama_model::MixerRaw::Delta { ssm_out, .. } => ssm_out,
            llama_model::MixerRaw::Attn { .. } => unreachable!(),
        },
        dn.head_v_dim() * dn.n_v_heads,
        n,
        &normed,
    );
    mostra("saida  ", &saida);

    // Mesma projeção, mas com a ativação quantizada em int8 por blocos de 32 — que é o
    // que o shader consome. Se a diferença vier daqui, o problema é de precisão, não de
    // lógica.
    let q_dq = |x: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        for b in 0..x.len() / 32 {
            let blk = &x[b * 32..b * 32 + 32];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for i in 0..32 {
                out[b * 32 + i] = (blk[i] * inv).round().clamp(-127.0, 127.0) * d;
            }
        }
        out
    };
    let normed_i8 = q_dq(&normed);
    let saida_i8 = matvec(
        match &raw.layers[0].mixer {
            llama_model::MixerRaw::Delta { ssm_out, .. } => ssm_out,
            llama_model::MixerRaw::Attn { .. } => unreachable!(),
        },
        dn.head_v_dim() * dn.n_v_heads,
        n,
        &normed_i8,
    );
    mostra("saida_i8", &saida_i8);
}
