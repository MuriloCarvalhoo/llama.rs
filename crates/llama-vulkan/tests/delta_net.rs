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

/// Push constant do `delta_net.comp`. `n_tok` é o bloco processado dentro do kernel e
/// `v_stride` o passo entre tokens em `v` — que no plano é `conv_dim`, não `n_heads * d`,
/// porque `v` é a última fatia do buffer da convolução.
#[repr(C)]
#[derive(Clone, Copy)]
struct PDelta {
    d: u32,
    n_heads: u32,
    rep: u32,
    n_tok: u32,
    v_stride: u32,
}

/// Push constant do `dn_conv.comp`.
#[repr(C)]
#[derive(Clone, Copy)]
struct PConv {
    canais: u32,
    d_conv: u32,
    n_tok: u32,
}

/// Push constant do `dn_l2_qk.comp`. `stride` é o passo entre tokens na entrada.
#[repr(C)]
#[derive(Clone, Copy)]
struct PL2Qk {
    dim: u32,
    n_heads: u32,
    eps: f32,
    stride: u32,
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

    // rep = 1: uma cabeça de chave por cabeça de valor (o caso sem GQA).
    let push = push_bytes(&PDelta {
        d: d as u32,
        n_heads: n_heads as u32,
        rep: 1,
        n_tok: 1,
        v_stride: (n_heads * d) as u32,
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

/// GQA: q/k têm menos cabeças que v (`n_k_heads < n_v_heads`, `rep = n_v_heads/n_k_heads`
/// no Qwen3.8). O llama.cpp expande via `ggml_repeat`, cuja semântica é **módulo**
/// (`dest[h] = src[h % n_k_heads]`, entrelaçado) — não blocos (`h / rep`). Este teste
/// existe porque `delta_net_bate_com_a_referencia_de_cpu` usa `rep=1` e não exercita esse
/// mapeamento; sem ele, um shader com o agrupamento errado ainda passaria.
#[test]
fn delta_net_gqa_mapeia_cabeca_de_valor_para_chave_por_modulo() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let d = 64usize;
    let n_k_heads = 2usize;
    let rep = 3usize;
    let n_heads = n_k_heads * rep; // 6 cabeças de valor, como no teste acima

    let estado0 = pseudo(n_heads * d * d, 41);
    let q_kheads = pseudo(n_k_heads * d, 42);
    let k_kheads = pseudo(n_k_heads * d, 43);
    let v = pseudo(n_heads * d, 44);
    let gb: Vec<f32> = (0..n_heads)
        .flat_map(|h| [-0.05 * (h as f32 + 1.0), 0.3 + 0.05 * h as f32])
        .collect();

    // Buffers do shader já vêm no tamanho de n_k_heads (o shader indexa `h % n_k_heads`
    // internamente); a referência de CPU replica esse `%` explicitamente abaixo.
    let push = push_bytes(&PDelta {
        d: d as u32,
        n_heads: n_heads as u32,
        rep: rep as u32,
        n_tok: 1,
        v_stride: (n_heads * d) as u32,
    });

    let saida = fwd
        .dbg_dn(
            DnPipe::DeltaNet,
            &[
                estado0.clone(),
                q_kheads.clone(),
                k_kheads.clone(),
                v.clone(),
                gb.clone(),
                vec![0f32; n_heads * d],
            ],
            &push,
            (n_heads * d / 4) as u32,
        )
        .expect("dispatch delta_net (GQA)");
    let estado_gpu = &saida[0];
    let out_gpu = &saida[5];

    // Referência: cabeça de valor `h` lê a cabeça de chave `h % n_k_heads` (módulo,
    // entrelaçado) — a semântica de `ggml_repeat`, não `h / rep` (blocos).
    let mut estado_cpu = estado0.clone();
    let mut out_cpu = vec![0f32; n_heads * d];
    for h in 0..n_heads {
        let hk = h % n_k_heads;
        llama_model::delta_net::delta_net_step(
            &mut estado_cpu[h * d * d..(h + 1) * d * d],
            &q_kheads[hk * d..(hk + 1) * d],
            &k_kheads[hk * d..(hk + 1) * d],
            &v[h * d..(h + 1) * d],
            gb[h * 2],
            gb[h * 2 + 1],
            &mut out_cpu[h * d..(h + 1) * d],
        );
    }

    let dif_estado = max_dif(estado_gpu, &estado_cpu);
    let dif_out = max_dif(out_gpu, &out_cpu);
    eprintln!("delta_net GQA: dif estado={dif_estado:.3e} saída={dif_out:.3e}");
    assert!(
        dif_estado < 1e-4,
        "estado recorrente (GQA) divergiu: {dif_estado}"
    );
    assert!(dif_out < 1e-4, "saída (GQA) divergiu: {dif_out}");
}

/// O bloco de prefill num dispatch só: `n_tok` tokens aplicados **dentro** do kernel, com
/// o estado em registrador entre eles. O que este teste prende é a recorrência — o estado
/// depois dos 8 tokens e as 8 saídas têm de bater com a referência de CPU aplicada token a
/// token, na mesma ordem.
///
/// Um shader que processasse os tokens em paralelo (ou fora de ordem) ainda daria saídas
/// plausíveis para o token 0; a divergência aparece a partir do segundo, e no estado final.
/// Por isso as duas comparações, e por isso a saída é conferida **token a token**.
///
/// A geometria é a do plano: GQA com `rep = 3` e `v` como fatia do buffer da convolução,
/// com passo `conv_dim` entre tokens — é o `v_stride` que o push carrega.
#[test]
fn delta_net_em_bloco_bate_com_a_referencia_token_a_token() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let d = 128usize;
    let n_k_heads = 2usize;
    let rep = 3usize;
    let n_heads = n_k_heads * rep;
    let n_tok = 8usize;
    let key_dim = n_k_heads * d;
    let value_dim = n_heads * d;
    let conv_dim = 2 * key_dim + value_dim;

    let estado0 = pseudo(n_heads * d * d, 91);
    // q e k chegam ao delta net normalizados em L2 (é o que o `dn_l2_qk` faz logo antes).
    // Sem isso o produto interno contra o estado cresce com `d` e, ao longo dos 8 passos,
    // o estado sai da faixa onde f32 tem precisão para uma comparação absoluta útil —
    // seria o teste medindo arredondamento em vez da recorrência.
    let mut q = pseudo(n_tok * key_dim, 92);
    let mut k = pseudo(n_tok * key_dim, 93);
    llama_model::delta_net::l2_norm_rows(&mut q, d, 1e-6);
    llama_model::delta_net::l2_norm_rows(&mut k, d, 1e-6);
    // `v` como no plano: a última fatia de cada token do buffer da convolução. O binding
    // começa em `2 * key_dim`, então aqui basta montar o buffer já a partir dali.
    let conv = pseudo(n_tok * conv_dim, 94);
    let v_bind: Vec<f32> = conv
        .get(2 * key_dim..)
        .expect("conv cobre o bloco inteiro")
        .to_vec();
    let gb: Vec<f32> = (0..n_tok * n_heads)
        .flat_map(|i| [-0.03 * (i % 7) as f32 - 0.01, 0.2 + 0.05 * (i % 5) as f32])
        .collect();

    let push = push_bytes(&PDelta {
        d: d as u32,
        n_heads: n_heads as u32,
        rep: rep as u32,
        n_tok: n_tok as u32,
        v_stride: conv_dim as u32,
    });

    let saida = fwd
        .dbg_dn(
            DnPipe::DeltaNet,
            &[
                estado0.clone(),
                q.clone(),
                k.clone(),
                v_bind,
                gb.clone(),
                vec![0f32; n_tok * value_dim],
            ],
            &push,
            (n_heads * d / 4) as u32,
        )
        .expect("dispatch delta_net (bloco)");
    let estado_gpu = &saida[0];
    let out_gpu = &saida[5];

    // Referência: os mesmos tokens, um de cada vez, sobre o mesmo estado inicial.
    let mut estado_cpu = estado0.clone();
    let mut out_cpu = vec![0f32; n_tok * value_dim];
    for t in 0..n_tok {
        for h in 0..n_heads {
            let hk = h % n_k_heads;
            let qk0 = t * key_dim + hk * d;
            let v0 = t * conv_dim + 2 * key_dim + h * d;
            llama_model::delta_net::delta_net_step(
                &mut estado_cpu[h * d * d..(h + 1) * d * d],
                &q[qk0..qk0 + d],
                &k[qk0..qk0 + d],
                &conv[v0..v0 + d],
                gb[(t * n_heads + h) * 2],
                gb[(t * n_heads + h) * 2 + 1],
                &mut out_cpu[t * value_dim + h * d..t * value_dim + (h + 1) * d],
            );
        }
    }

    // Tolerância apertada de propósito: com q/k normalizados o estado fica em O(1) e a
    // diferença medida é ~2e-7 (épsilon de f32 acumulado nos 8 passos). Qualquer erro de
    // ordem ou de índice sai várias ordens de grandeza acima disto.
    let dif_estado = max_dif(estado_gpu, &estado_cpu);
    eprintln!("delta_net bloco: dif estado={dif_estado:.3e}");
    assert!(
        dif_estado < 1e-5,
        "estado depois de {n_tok} tokens divergiu: {dif_estado}"
    );
    for t in 0..n_tok {
        let faixa = t * value_dim..(t + 1) * value_dim;
        let dif = max_dif(&out_gpu[faixa.clone()], &out_cpu[faixa]);
        eprintln!("delta_net bloco: t={t} dif saída={dif:.3e}");
        assert!(dif < 1e-5, "saída do token {t} divergiu: {dif}");
    }
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

    let push = push_bytes(&PConv {
        canais: canais as u32,
        d_conv: d_conv as u32,
        n_tok: 1,
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

/// A convolução do bloco num dispatch só: a janela avança dentro do kernel, em registrador.
///
/// A ordem importa tanto quanto no delta net — a janela do token `t` é o resultado dos
/// `d_conv - 1` tokens anteriores do mesmo bloco. `d_conv = 4` faz o token 3 já depender
/// só de tokens do bloco, então um shader que lesse a janela original para todos passaria
/// nos três primeiros e falharia daí em diante; por isso a saída é conferida token a token.
#[test]
fn conv_causal_em_bloco_bate_com_a_referencia_token_a_token() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let canais = 512usize;
    let d_conv = 4usize;
    let n_tok = 8usize;
    let estado0 = pseudo(canais * (d_conv - 1), 101);
    let x = pseudo(n_tok * canais, 102);
    let w = pseudo(canais * d_conv, 103);

    let saida = fwd
        .dbg_dn(
            DnPipe::Conv,
            &[
                estado0.clone(),
                x.clone(),
                w.clone(),
                vec![0f32; n_tok * canais],
            ],
            &push_bytes(&PConv {
                canais: canais as u32,
                d_conv: d_conv as u32,
                n_tok: n_tok as u32,
            }),
            u32::try_from(canais.div_ceil(64)).unwrap(),
        )
        .expect("dispatch dn_conv (bloco)");

    let mut estado_cpu = estado0.clone();
    let mut esperado = vec![0f32; n_tok * canais];
    for t in 0..n_tok {
        let bruto = llama_model::delta_net::conv1d_step(
            &mut estado_cpu,
            &x[t * canais..(t + 1) * canais],
            &w,
            d_conv,
        );
        for (dst, &v) in esperado[t * canais..(t + 1) * canais]
            .iter_mut()
            .zip(&bruto)
        {
            *dst = llama_model::delta_net::silu(v);
        }
    }

    let dif_janela = max_dif(&saida[0], &estado_cpu);
    eprintln!("dn_conv bloco: dif janela={dif_janela:.3e}");
    assert!(
        dif_janela < 1e-6,
        "janela depois de {n_tok} tokens divergiu: {dif_janela}"
    );
    for t in 0..n_tok {
        let faixa = t * canais..(t + 1) * canais;
        let dif = max_dif(&saida[3][faixa.clone()], &esperado[faixa]);
        assert!(dif < 1e-5, "saída do token {t} divergiu: {dif}");
    }
}

/// `n_tok = 1` é o decode; acima disso o bloco entra pela dimensão Y do dispatch, um
/// workgroup por (cabeça, token). Sem recorrência aqui — o que o caso `n_tok > 1` prende é
/// o roteamento: `x` avança `n_embd` por token e a saída avança `n_heads` pares (g, beta),
/// enquanto `alpha`/`beta`/`adt` são os mesmos pesos para todos. Trocar um desses índices
/// dá números plausíveis com o token errado, e é por isso que cada token usa a sua fatia.
#[test]
fn gates_batem_com_a_referencia_de_cpu() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n_embd = 512usize;
    let n_heads = 12usize;
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

    for n_tok in [1usize, 4] {
        let x = pseudo(n_tok * n_embd, 21);
        let saida = fwd
            .dbg_dn_xy(
                DnPipe::Gates,
                &[
                    x.clone(),
                    alpha.clone(),
                    beta.clone(),
                    adt.clone(),
                    vec![0f32; n_tok * n_heads * 2],
                ],
                &push,
                n_heads as u32,
                u32::try_from(n_tok).unwrap(),
            )
            .expect("dispatch dn_gates");

        let mut esperado = vec![0f32; n_tok * n_heads * 2];
        for t in 0..n_tok {
            for h in 0..n_heads {
                let xt = &x[t * n_embd..(t + 1) * n_embd];
                let pa: f32 = (0..n_embd).map(|i| alpha[h * n_embd + i] * xt[i]).sum();
                let pb: f32 = (0..n_embd).map(|i| beta[h * n_embd + i] * xt[i]).sum();
                let o = (t * n_heads + h) * 2;
                esperado[o] = adt[h * 2] * llama_model::delta_net::softplus(pa + adt[h * 2 + 1]);
                esperado[o + 1] = llama_model::delta_net::sigmoid(pb);
            }
        }
        // Erro **relativo**: o produto interno de n_embd termos é uma redução em árvore na
        // GPU e uma soma sequencial na CPU, então a diferença cresce com a magnitude do
        // resultado, não com a do erro de cada operação.
        let escala = esperado.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let dif = max_dif(&saida[4], &esperado) / escala;
        eprintln!("dn_gates (n_tok={n_tok}): dif relativa={dif:.3e}");
        assert!(dif < 1e-5, "gates divergiram com n_tok={n_tok}: {dif}");
    }
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
        stride: u32,
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
                stride: dim as u32,
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
                stride: dim as u32,
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

/// `dn_l2_qk` funde as duas L2 (q e k) que eram dois dispatches. O teste compara com a
/// mesma referência do modo 0 do `dn_norm`, e o que ele de fato protege é o **roteamento**:
/// q e k entram contíguos num binding só e saem em dois buffers distintos, então um shader
/// que trocasse os destinos, ou que normalizasse as 2×n_heads cabeças como se fossem um
/// tensor só, ainda produziria números plausíveis. Por isso q e k usam sementes diferentes
/// e o teste confere buffer a buffer.
///
/// Com `n_tok > 1` entra o caso do bloco, e com ele a assimetria que o `stride` existe para
/// resolver: na **entrada** os tokens estão espaçados de `conv_dim` (q | k | v por token,
/// como a convolução os deixa), na **saída** de `n_heads * dim`. O `v` de cada token cai
/// exatamente entre as cabeças de k de um token e as de q do seguinte — um shader que
/// usasse o passo da saída também na entrada normalizaria pedaços de `v` como se fossem q.
#[test]
fn l2_de_q_e_k_no_mesmo_dispatch_bate_com_a_referencia() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let dim = 128usize; // d_state do Qwen3.8
    let n_heads = 6usize; // n_k_heads
    let eps = 1e-6f32;
    // Passo entre tokens na entrada: q e k (2 × n_heads cabeças) mais o `v`, que aqui só
    // ocupa espaço — é a geometria do buffer da convolução.
    let conv_dim = 2 * n_heads * dim + 3 * dim;

    for n_tok in [1usize, 4] {
        let qk = pseudo(n_tok * conv_dim, 51);
        let saida = fwd
            .dbg_dn_xy(
                DnPipe::L2Qk,
                &[
                    qk.clone(),
                    vec![0f32; n_tok * n_heads * dim],
                    vec![0f32; n_tok * n_heads * dim],
                ],
                &push_bytes(&PL2Qk {
                    dim: dim as u32,
                    n_heads: n_heads as u32,
                    eps,
                    stride: conv_dim as u32,
                }),
                (2 * n_heads) as u32,
                u32::try_from(n_tok).unwrap(),
            )
            .expect("dispatch dn_l2_qk");

        // Referência: a mesma L2 por linha da CPU, sobre as 2*n_heads cabeças de cada token.
        for t in 0..n_tok {
            let mut esperado = qk[t * conv_dim..t * conv_dim + 2 * n_heads * dim].to_vec();
            llama_model::delta_net::l2_norm_rows(&mut esperado, dim, eps);
            let saiu = t * n_heads * dim..(t + 1) * n_heads * dim;
            let dif_q = max_dif(&saida[1][saiu.clone()], &esperado[..n_heads * dim]);
            let dif_k = max_dif(&saida[2][saiu], &esperado[n_heads * dim..]);
            eprintln!("dn_l2_qk (n_tok={n_tok}, t={t}): dif q={dif_q:.3e} k={dif_k:.3e}");
            assert!(dif_q < 1e-6, "q divergiu no token {t}: {dif_q}");
            assert!(dif_k < 1e-6, "k divergiu no token {t}: {dif_k}");
        }
        // A metade de k não pode ter caído no buffer de q: sem isto, um shader que ignorasse
        // `eh_k` e escrevesse tudo em `qn` passaria no `dif_q` acima.
        assert_ne!(
            saida[1], saida[2],
            "q e k saíram iguais — o roteamento entre os dois destinos está errado"
        );
    }
}

/// `gate_quant` funde o portão sigmoide do qwen35 com a quantização em int8 que vinha logo
/// depois. São duas verificações independentes, porque as duas metades podem errar sozinhas:
///
/// 1. o portão contra a referência de CPU (`sigmoid` de `llama_model::delta_net`) — o índice
///    do gate é o que mais tem como errar, porque ele mora na **segunda** metade de cada
///    cabeça de `b_q`, entrelaçado com a query;
/// 2. `xq`/`xd` contra o `quantize_x.comp` rodado sobre a saída que o próprio shader
///    produziu — igualdade **exata**, já que a entrada do quantizador é a mesma nos dois
///    caminhos. É o que prende a fusão como "mesma matemática".
#[test]
fn gate_quant_bate_com_o_portao_da_cpu_e_com_o_quantize_x() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    // Geometria do Qwen3.8: head_dim 256, 24 cabeças. `n` é múltiplo de 32 (o bloco de
    // quantização), como o plano garante.
    let head_dim = 256usize;
    let n_head = 24usize;
    let n = head_dim * n_head;
    let attn = pseudo(n, 61);
    // `b_q`: por cabeça, query na primeira metade e portão na segunda.
    let q = pseudo(n_head * 2 * head_dim, 62);

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct P {
        n: u32,
        head_dim: u32,
    }

    let saida = fwd
        .dbg_dn(
            DnPipe::GateQuant,
            &[
                attn.clone(),
                q.clone(),
                vec![0f32; n / 32 * 8],
                vec![0f32; n / 32],
            ],
            &push_bytes(&P {
                n: n as u32,
                head_dim: head_dim as u32,
            }),
            u32::try_from((n / 32).div_ceil(64)).unwrap(),
        )
        .expect("dispatch gate_quant");

    // 1. O portão, contra a CPU.
    let esperado: Vec<f32> = (0..n)
        .map(|i| {
            let h = i / head_dim;
            let d = i % head_dim;
            attn[i] * llama_model::delta_net::sigmoid(q[h * 2 * head_dim + head_dim + d])
        })
        .collect();
    let dif = max_dif(&saida[0], &esperado);
    eprintln!("gate_quant: dif do portão={dif:.3e}");
    assert!(dif < 1e-6, "portão divergiu: {dif}");

    // 2. `xq`/`xd`, contra o `quantize_x` sobre a mesma entrada — exatos.
    let mut qx_push = Vec::with_capacity(12);
    qx_push.extend_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
    qx_push.extend_from_slice(&0u32.to_le_bytes());
    qx_push.extend_from_slice(&0u32.to_le_bytes());
    let sozinho = fwd
        .dbg_dn(
            DnPipe::QuantizeX,
            &[saida[0].clone(), vec![0f32; n / 32 * 8], vec![0f32; n / 32]],
            &qx_push,
            u32::try_from((n / 32).div_ceil(64)).unwrap(),
        )
        .expect("dispatch quantize_x");
    // `xq` carrega int8 empacotados; comparar pelos bits evita que um padrão de NaN
    // faça duas saídas idênticas parecerem diferentes.
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(
        bits(&saida[2]),
        bits(&sozinho[1]),
        "xq difere do quantize_x"
    );
    assert_eq!(
        bits(&saida[3]),
        bits(&sozinho[2]),
        "xd difere do quantize_x"
    );
}

/// `swiglu_quant` funde o SwiGLU com a quantização que vinha logo depois. O `silu(g)*u`
/// contra a CPU está em `resident_fwd_swiglu_igual_cpu`; aqui o que se prende é a outra
/// metade: `xq`/`xd` **exatos** contra o `quantize_x.comp` rodado sobre a saída que o
/// próprio shader escreveu em `b_act`. É o que sustenta a fusão como "mesma matemática".
#[test]
fn swiglu_quant_quantiza_igual_ao_quantize_x() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n = 4864usize; // múltiplo de 32, como todo `n_ff`
    let g = pseudo(n, 71);
    let u = pseudo(n, 72);
    let grupos = u32::try_from((n / 32).div_ceil(64)).unwrap();

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
            grupos,
        )
        .expect("dispatch swiglu_quant");

    let mut qx_push = Vec::with_capacity(12);
    qx_push.extend_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
    qx_push.extend_from_slice(&0u32.to_le_bytes());
    qx_push.extend_from_slice(&0u32.to_le_bytes());
    let sozinho = fwd
        .dbg_dn(
            DnPipe::QuantizeX,
            &[saida[2].clone(), vec![0f32; n / 32 * 8], vec![0f32; n / 32]],
            &qx_push,
            grupos,
        )
        .expect("dispatch quantize_x");

    let bits2 = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(
        bits2(&saida[3]),
        bits2(&sozinho[1]),
        "xq difere do quantize_x"
    );
    assert_eq!(
        bits2(&saida[4]),
        bits2(&sozinho[2]),
        "xd difere do quantize_x"
    );
}

/// f16 (bits) → f32, para conferir o que o `kv_pack` escreveu no cache.
fn f16_para_f32(h: u16) -> f32 {
    let sinal = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);
    let bits = match (exp, mant) {
        (0, 0) => sinal,
        // Subnormais não aparecem nos valores de teste; normaliza mesmo assim.
        (0, _) => {
            let shift = mant.leading_zeros() - 21;
            sinal | ((127 - 15 - shift + 1) << 23) | ((mant << (shift + 14)) & 0x7F_FFFF)
        }
        (0x1F, _) => sinal | 0x7F80_0000 | (mant << 13),
        _ => sinal | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// `kv_pack` converte K e V do passo para o cache em f16 empacotado (um u32 por par).
/// A referência é a própria entrada: desempacotar o slot tem de devolver K e V dentro do
/// arredondamento de f16, no offset certo, e **nada fora do slot** pode ser escrito —
/// `dst_par` é o único offset do plano que depende da posição.
#[test]
fn kv_pack_escreve_o_bloco_convertido_no_slot() {
    let Some((ctx, ())) = ctx_fwd() else { return };
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let kv_dim = 1024usize;
    for n_tok in [1usize, 3] {
        let slot_off_par = 7usize * kv_dim / 2; // (slot × ctx + pos0) × kv_dim / 2
        let k = pseudo(kv_dim * n_tok, 11 + n_tok);
        let v = pseudo(kv_dim * n_tok, 300 + n_tok);
        let n_pares = kv_dim * n_tok / 2;
        // Caches maiores que o bloco, para pegar escrita fora do slot. Os buffers do
        // helper são f32; o shader os escreve como u32 — mesmo tamanho, reinterpretado.
        let cache = vec![0f32; slot_off_par + n_pares + kv_dim];

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct P {
            n_pares: u32,
            dst_par: u32,
        }
        let saida = fwd
            .dbg_dn(
                DnPipe::KvPack,
                &[k.clone(), v.clone(), cache.clone(), cache.clone()],
                &push_bytes(&P {
                    n_pares: n_pares as u32,
                    dst_par: slot_off_par as u32,
                }),
                u32::try_from(n_pares.div_ceil(64)).unwrap(),
            )
            .expect("dispatch kv_pack");

        for (nome, orig, out) in [("K", &k, &saida[2]), ("V", &v, &saida[3])] {
            let mut dif = 0f32;
            for (i, &x) in orig.iter().enumerate() {
                let par = out[slot_off_par + i / 2].to_bits();
                let h = if i % 2 == 0 {
                    par as u16
                } else {
                    (par >> 16) as u16
                };
                dif = dif.max((f16_para_f32(h) - x).abs());
            }
            // Arredondamento de f16 em valores |x| < 1: passo máximo ~2^-11.
            assert!(
                dif < 1e-3,
                "{nome} difere além do f16 (n_tok={n_tok}): {dif}"
            );
            assert!(
                out[..slot_off_par].iter().all(|&b| b == 0.0)
                    && out[slot_off_par + n_pares..].iter().all(|&b| b == 0.0),
                "{nome}: escreveu fora do slot (n_tok={n_tok})"
            );
        }
        eprintln!("kv_pack n_tok={n_tok}: bloco no slot, dentro do arredondamento de f16");
    }
}
