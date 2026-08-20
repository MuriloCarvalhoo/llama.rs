//! Operações da atenção linear (gated delta net) das arquiteturas híbridas tipo `qwen35`.
//!
//! São a referência em f32 do que os shaders precisam reproduzir, e a fonte da verdade
//! dos testes. A matemática segue `ggml_compute_forward_gated_delta_net` e
//! `ggml_ssm_conv` do llama.cpp — ver `docs/qwen35-arquitetura.md`.
//!
//! O que distingue estas camadas da atenção comum: não há KV-cache. O histórico inteiro
//! da sequência vive num estado `d_state × d_state` por cabeça, de tamanho fixo, que cada
//! token lê e reescreve.

/// `softplus(x) = ln(1 + e^x)`, na forma numericamente estável usada pelo ggml.
#[must_use]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Sigmoide logística.
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU (swish): `x * sigmoid(x)`.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Normaliza cada linha de `dim` elementos pela própria norma L2.
///
/// É L2, não RMS: divide por `sqrt(sum(x²))`, sem o `1/dim`. O `eps` entra dentro da raiz,
/// como em `ggml_l2_norm`.
pub fn l2_norm_rows(x: &mut [f32], dim: usize, eps: f32) {
    for linha in x.chunks_exact_mut(dim) {
        let soma: f32 = linha.iter().map(|v| v * v).sum();
        let inv = 1.0 / (soma + eps).sqrt();
        for v in linha.iter_mut() {
            *v *= inv;
        }
    }
}

/// Convolução causal de largura `d_conv` sobre um único token, com estado.
///
/// `state` guarda os `d_conv - 1` valores anteriores de cada canal, em layout
/// `[canal][passo]`, e é deslocado no lugar: o valor novo entra no fim e o mais antigo sai.
/// `w` é o kernel `[d_conv][canais]` (layout do GGUF: `ne = [d_conv, canais]`, então o
/// passo `t` do canal `c` está em `w[c * d_conv + t]`).
///
/// Devolve um valor por canal, já com o bias implícito do formato (não há bias no qwen35).
pub fn conv1d_step(state: &mut [f32], entrada: &[f32], w: &[f32], d_conv: usize) -> Vec<f32> {
    let canais = entrada.len();
    let passos_estado = d_conv - 1;
    debug_assert_eq!(state.len(), canais * passos_estado);
    debug_assert_eq!(w.len(), canais * d_conv);

    let mut saida = vec![0f32; canais];
    for c in 0..canais {
        let janela = &mut state[c * passos_estado..(c + 1) * passos_estado];
        let kernel = &w[c * d_conv..(c + 1) * d_conv];
        // Os `d_conv - 1` valores antigos, depois o token atual.
        let mut acc = 0f32;
        for (t, &v) in janela.iter().enumerate() {
            acc += v * kernel[t];
        }
        acc += entrada[c] * kernel[d_conv - 1];
        saida[c] = acc;
        // Desliza a janela: descarta o mais antigo, guarda o atual.
        janela.rotate_left(1);
        janela[passos_estado - 1] = entrada[c];
    }
    saida
}

/// Um passo da recorrência do gated delta net, para **uma** cabeça.
///
/// `state` é a matriz `d × d` da cabeça, guardada **transposta** (`state[j * d + i]` é
/// `S[i][j]`), do mesmo jeito que o ggml a guarda — assim a linha `j` fica contígua e as
/// três operações do passo viram produtos internos e um `mad` sobre linhas.
///
/// - `g`: log do decaimento (escalar por cabeça). O estado é multiplicado por `exp(g)`.
/// - `beta`: força da correção do delta rule.
///
/// Escreve a saída da cabeça em `out` (`d` elementos) e atualiza `state` no lugar.
pub fn delta_net_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    out: &mut [f32],
) {
    let d = q.len();
    debug_assert_eq!(state.len(), d * d);
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);
    debug_assert_eq!(out.len(), d);

    // 1. Decaimento do estado inteiro.
    let decay = g.exp();
    for s in state.iter_mut() {
        *s *= decay;
    }

    // 2. delta[j] = (v[j] - <S[:,j], k>) * beta — o erro de predição do delta rule.
    let mut delta = vec![0f32; d];
    for j in 0..d {
        let linha = &state[j * d..(j + 1) * d];
        let pred: f32 = linha.iter().zip(k).map(|(s, kk)| s * kk).sum();
        delta[j] = (v[j] - pred) * beta;
    }

    // 3. Atualização de posto 1: S += k ⊗ delta.
    for j in 0..d {
        let dj = delta[j];
        let linha = &mut state[j * d..(j + 1) * d];
        for (s, kk) in linha.iter_mut().zip(k) {
            *s += dj * kk;
        }
    }

    // 4. Leitura: out[j] = <S[:,j], q> / sqrt(d).
    let escala = 1.0 / (d as f32).sqrt();
    for j in 0..d {
        let linha = &state[j * d..(j + 1) * d];
        let soma: f32 = linha.iter().zip(q).map(|(s, qq)| s * qq).sum();
        out[j] = soma * escala;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_e_estavel_nos_extremos() {
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        // Para x grande, softplus(x) -> x sem estourar o exp.
        assert!((softplus(50.0) - 50.0).abs() < 1e-6);
        assert!(softplus(-30.0) > 0.0 && softplus(-30.0) < 1e-12);
    }

    #[test]
    fn l2_norm_deixa_a_linha_com_norma_um() {
        let mut x = vec![3.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0, 0.0];
        l2_norm_rows(&mut x, 4, 0.0);
        let n0: f32 = x[..4].iter().map(|v| v * v).sum::<f32>().sqrt();
        let n1: f32 = x[4..].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-6, "n0={n0}");
        assert!((n1 - 1.0).abs() < 1e-6, "n1={n1}");
        assert!((x[0] - 0.6).abs() < 1e-6);
    }

    /// A convolução causal com estado, aplicada token a token, tem que dar o mesmo que
    /// convoluir a sequência inteira de uma vez.
    #[test]
    fn conv1d_com_estado_bate_com_a_convolucao_direta() {
        let canais = 3usize;
        let d_conv = 4usize;
        let n_tok = 6usize;
        let w: Vec<f32> = (0..canais * d_conv)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let seq: Vec<Vec<f32>> = (0..n_tok)
            .map(|t| {
                (0..canais)
                    .map(|c| ((t * canais + c) as f32) * 0.25 - 1.0)
                    .collect()
            })
            .collect();

        // Passo a passo, com estado começando em zero (como no início de uma sequência).
        let mut state = vec![0f32; canais * (d_conv - 1)];
        let mut passo_a_passo = Vec::new();
        for tok in &seq {
            passo_a_passo.push(conv1d_step(&mut state, tok, &w, d_conv));
        }

        // Direto: para cada t, soma sobre a janela causal, com zeros antes do início.
        for t in 0..n_tok {
            for c in 0..canais {
                let kernel = &w[c * d_conv..(c + 1) * d_conv];
                let mut esperado = 0f32;
                for (i, kv) in kernel.iter().enumerate() {
                    let pos = t.cast_signed() + i.cast_signed() - (d_conv.cast_signed() - 1);
                    if pos >= 0 {
                        esperado += seq[pos.cast_unsigned()][c] * kv;
                    }
                }
                let obtido = passo_a_passo[t][c];
                assert!(
                    (obtido - esperado).abs() < 1e-5,
                    "t={t} c={c}: {obtido} != {esperado}"
                );
            }
        }
    }

    /// Com beta = 1 e decaimento nulo, o delta rule é uma regressão exata: depois de
    /// escrever (k, v) com k unitário, ler com q = k devolve v.
    #[test]
    fn delta_net_escreve_e_le_o_par_kv() {
        let d = 4usize;
        let mut state = vec![0f32; d * d];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, -1.5, 2.0, 0.25];
        let mut out = vec![0f32; d];
        delta_net_step(&mut state, &k, &k, &v, 0.0, 1.0, &mut out);
        let escala = (d as f32).sqrt();
        for j in 0..d {
            assert!(
                (out[j] * escala - v[j]).abs() < 1e-6,
                "j={j}: {} != {}",
                out[j] * escala,
                v[j]
            );
        }
    }

    /// Escrever um segundo par com chave ortogonal não apaga o primeiro, e o decaimento
    /// `g` encolhe o que já estava no estado.
    #[test]
    fn delta_net_mantem_chaves_ortogonais_e_aplica_decaimento() {
        let d = 4usize;
        let mut state = vec![0f32; d * d];
        let k1 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let k2 = vec![0.0, 1.0, 0.0, 0.0];
        let v2 = vec![-1.0, 0.0, 1.0, 0.0];
        let mut out = vec![0f32; d];

        delta_net_step(&mut state, &k1, &k1, &v1, 0.0, 1.0, &mut out);
        // g = ln(0.5): tudo que estava no estado vale metade antes da escrita nova.
        let g = 0.5f32.ln();
        delta_net_step(&mut state, &k2, &k2, &v2, g, 1.0, &mut out);
        let escala = (d as f32).sqrt();
        for j in 0..d {
            assert!(
                (out[j] * escala - v2[j]).abs() < 1e-6,
                "leitura de k2 em j={j}"
            );
        }

        // Lendo com k1: o valor antigo sobrevive, reduzido pelo decaimento.
        delta_net_step(&mut state, &k1, &k1, &vec![0.0; d], 0.0, 0.0, &mut out);
        for j in 0..d {
            let esperado = v1[j] * 0.5;
            assert!(
                (out[j] * escala - esperado).abs() < 1e-5,
                "j={j}: {} != {esperado}",
                out[j] * escala
            );
        }
    }
}
