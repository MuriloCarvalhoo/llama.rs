//! Kernels f32 puros do forward Llama. Layout token-major: `x[t*dim + d]`.
#![allow(clippy::indexing_slicing)]

use crate::error::ModelError;

/// GET_ROWS: para cada token, copia a linha de `embd` ({vocab, n_embd}).
/// Saída token-major [n_tok * n_embd].
pub(crate) fn embedding_lookup(
    embd: &[f32],
    tokens: &[u32],
    n_embd: usize,
) -> Result<Vec<f32>, ModelError> {
    let mut out = Vec::with_capacity(tokens.len() * n_embd);
    for &tok in tokens {
        let t = usize::try_from(tok).map_err(|_| ModelError::Overflow)?;
        let start = t * n_embd;
        let row = embd
            .get(start..start + n_embd)
            .ok_or_else(|| ModelError::Config(format!("token {t} fora do vocab")))?;
        out.extend_from_slice(row);
    }
    Ok(out)
}

/// RMSNorm por linha (sem peso): `x / sqrt(mean(x^2) + eps)`.
pub(crate) fn rmsnorm(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ss: f32 = row_in.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
        for (o, &i) in row_out.iter_mut().zip(row_in.iter()) {
            *o = i * scale;
        }
    }
    out
}

/// Multiplicação elementwise por peso broadcast por dimensão (MUL com {dim}).
pub(crate) fn mul_rows(x: &[f32], weight: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        for (idx, (o, &i)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            *o = i * weight[idx];
        }
    }
    out
}

/// MUL_MAT: `W{in,out}` (out linhas de comprimento in) × `x` token-major [n_tok*in].
/// Saída token-major [n_tok*out]: `out[t*out+j] = Σ_i W[j*in+i] * x[t*in+i]`.
pub(crate) fn matmul(w: &[f32], x: &[f32], n_in: usize, n_out: usize, n_tok: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * n_out];
    for t in 0..n_tok {
        let xrow = &x[t * n_in..t * n_in + n_in];
        let orow = &mut out[t * n_out..t * n_out + n_out];
        for (j, o) in orow.iter_mut().enumerate() {
            let wrow = &w[j * n_in..j * n_in + n_in];
            *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
        }
    }
    out
}

/// RoPE NORM (arch llama): rotaciona pares (2i,2i+1) de cada head.
/// `θ_i = pos * freq_base^(-2i/rope_dim)`, para i em 0..rope_dim/2.
pub(crate) fn rope_norm(
    x: &mut [f32],
    n_tok: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    pos0: usize,
) {
    for t in 0..n_tok {
        let pos = (pos0 + t) as f32;
        for h in 0..n_head {
            let base = (t * n_head + h) * head_dim;
            for i in 0..rope_dim / 2 {
                let theta = pos * freq_base.powf(-2.0 * i as f32 / rope_dim as f32);
                let (s, c) = theta.sin_cos();
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                x[base + 2 * i] = a * c - b * s;
                x[base + 2 * i + 1] = a * s + b * c;
            }
        }
    }
}

/// SiLU: `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SWIGLU: `silu(gate) * up`, elementwise (mesmo comprimento).
pub(crate) fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect()
}

/// Softmax numericamente estável sobre um slice (in-place).
pub(crate) fn softmax(z: &mut [f32]) {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in z.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in z.iter_mut() {
            *v /= sum;
        }
    }
}

/// Índice do maior valor (greedy / argmax). Empate → menor índice.
pub(crate) fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_lookup_copies_rows() {
        // vocab=3, n_embd=2
        let embd = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = embedding_lookup(&embd, &[2, 0], 2).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn embedding_lookup_rejects_oob_token() {
        let embd = vec![0.0, 1.0];
        assert!(embedding_lookup(&embd, &[5], 2).is_err());
    }

    #[test]
    fn rmsnorm_unit_vector() {
        // x = [3,4], dim=2, eps=0 → mean(x^2)=12.5 → scale=1/sqrt(12.5)
        let out = rmsnorm(&[3.0, 4.0], 2, 0.0);
        let s = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * s).abs() < 1e-6);
        assert!((out[1] - 4.0 * s).abs() < 1e-6);
    }

    #[test]
    fn mul_rows_broadcasts_weight() {
        // 2 linhas dim=2, peso [10,100]
        let out = mul_rows(&[1.0, 2.0, 3.0, 4.0], &[10.0, 100.0], 2);
        assert_eq!(out, vec![10.0, 200.0, 30.0, 400.0]);
    }

    #[test]
    fn matmul_2x2_identity_and_general() {
        // W in=2,out=2: linha0=[1,0], linha1=[0,1] (identidade) → out=x
        let w_id = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![5.0, 7.0]; // 1 token
        assert_eq!(matmul(&w_id, &x, 2, 2, 1), vec![5.0, 7.0]);

        // W in=2,out=1: linha0=[2,3] → out[0]=2*5+3*7=31
        let w = vec![2.0, 3.0];
        assert_eq!(matmul(&w, &x, 2, 1, 1), vec![31.0]);
    }

    #[test]
    fn matmul_two_tokens() {
        // W in=2,out=2: linha0=[1,1], linha1=[1,-1]
        let w = vec![1.0, 1.0, 1.0, -1.0];
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
        // t0: [1+2, 1-2]=[3,-1]; t1:[3+4,3-4]=[7,-1]
        assert_eq!(matmul(&w, &x, 2, 2, 2), vec![3.0, -1.0, 7.0, -1.0]);
    }

    #[test]
    fn rope_norm_pos_zero_is_identity() {
        // pos=0 → θ=0 → cos=1,sin=0 → sem mudança
        let mut x = vec![1.0, 2.0, 3.0, 4.0]; // 1 tok, 1 head, head_dim=4, rope_dim=4
        rope_norm(&mut x, 1, 1, 4, 4, 10000.0, 0);
        assert!(
            x.iter()
                .zip([1.0, 2.0, 3.0, 4.0])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }

    #[test]
    fn rope_norm_pos_one_rotates_first_pair_by_one_radian() {
        // i=0 → θ = 1 * base^0 = 1 rad. par (x0,x1)=(1,0) → (cos1, sin1)
        let mut x = vec![1.0, 0.0, 0.0, 0.0];
        rope_norm(&mut x, 1, 1, 4, 4, 10000.0, 1);
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn swiglu_matches_manual() {
        // gate=[0,2], up=[1,3]; silu(0)=0, silu(2)=2*sigmoid(2)
        let out = swiglu(&[0.0, 2.0], &[1.0, 3.0]);
        assert!((out[0] - 0.0).abs() < 1e-6);
        let silu2 = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((out[1] - silu2 * 3.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut z = vec![1.0, 2.0, 3.0];
        softmax(&mut z);
        assert!((z.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(z[2] > z[1] && z[1] > z[0]);
    }

    #[test]
    fn argmax_picks_first_max() {
        assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.2]), 1);
    }
}
