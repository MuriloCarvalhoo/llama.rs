//! Kernels f32 puros do forward Llama. Layout token-major: `x[t*dim + d]`.
#![allow(clippy::indexing_slicing)]
// Kernels AVX2 requerem unsafe; permitido apenas neste módulo.
#![allow(unsafe_code)]

use crate::error::ModelError;
use rayon::prelude::*;

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
#[cfg_attr(not(test), allow(dead_code))]
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
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn mul_rows(x: &[f32], weight: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        for (idx, (o, &i)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            *o = i * weight[idx];
        }
    }
    out
}

/// Otimização 5 — RMSNorm + escala elementwise em um único passe sobre os dados.
/// Equivale a `mul_rows(rmsnorm(x, dim, eps), weight, dim)` sem alocar buffer intermediário.
pub(crate) fn rmsnorm_and_scale(x: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ss: f32 = row_in.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
        for (idx, (o, &i)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            *o = i * scale * weight[idx];
        }
    }
    out
}

/// MUL_MAT: `W{in,out}` (out linhas de comprimento in) × `x` token-major [n_tok*in].
/// Saída token-major [n_tok*out]: `out[t*out+j] = Σ_i W[j*in+i] * x[t*in+i]`.
/// Threshold below which rayon thread-pool overhead exceeds benefit.
const PAR_MIN_N_OUT: usize = 512;
/// Lower threshold for Q8_0 actq path (kernel é mais rápido, vale paralelizar mais cedo).
const PAR_MIN_N_OUT_Q8: usize = 256;
/// Linhas por tarefa rayon. Valor menor aumenta paralelismo para matrizes menores
/// (ex.: attn_q n_out=896 com 28 threads: 64→14 tasks (metade idle), 32→28 tasks (todos ativos)).
const PAR_CHUNK: usize = 32;

pub(crate) fn matmul(w: &[f32], x: &[f32], n_in: usize, n_out: usize, n_tok: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * n_out];
    for t in 0..n_tok {
        let xrow = &x[t * n_in..(t + 1) * n_in];
        let orow = &mut out[t * n_out..(t + 1) * n_out];
        if n_out >= PAR_MIN_N_OUT {
            orow.par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (k, o) in chunk.iter_mut().enumerate() {
                        let j = ci * PAR_CHUNK + k;
                        let wrow = &w[j * n_in..(j + 1) * n_in];
                        *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
                    }
                });
        } else {
            for (j, o) in orow.iter_mut().enumerate() {
                let wrow = &w[j * n_in..(j + 1) * n_in];
                *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
            }
        }
    }
    out
}

/// MUL_MAT direto em bytes Q8_0 — sem expandir para f32.
///
/// Layout W: `n_out` linhas × `(n_in/32)` blocos × 34 bytes/bloco.
/// Bloco: 2 bytes (f16 LE scale `d`) + 32 bytes (i8 quants).
/// `n_in` deve ser múltiplo de 32 (garantido pelo formato GGUF).
pub(crate) fn matmul_q8_0(
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_tok: usize,
) -> Vec<f32> {
    const Q: usize = 32;
    const B: usize = 34;
    debug_assert_eq!(n_in % Q, 0, "n_in deve ser múltiplo de 32");

    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let mut out = vec![0.0f32; n_tok * n_out];

    for t in 0..n_tok {
        let x_row = &x[t * n_in..(t + 1) * n_in];
        let o_row = &mut out[t * n_out..(t + 1) * n_out];

        if n_out >= PAR_MIN_N_OUT {
            o_row
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    for (k, o) in chunk.iter_mut().enumerate() {
                        let j = ci * PAR_CHUNK + k;
                        *o = q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
                    }
                });
        } else {
            for (j, o) in o_row.iter_mut().enumerate() {
                *o = q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
            }
        }
    }
    out
}

/// Kernel AVX2 explícito para Q8_0 × f32.
/// Carrega 32 i8 quants em 4 × _mm256_cvtepi8_epi32 e acumula com VFMADD231PS.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn q8_0_dot_avx2(w_row: &[u8], x_row: &[f32], n_blocks: usize) -> f32 {
    use std::arch::x86_64::*;
    const B: usize = 34;
    const Q: usize = 32;

    // SAFETY: intrinsics protegidos por #[target_feature(enable = "avx2,fma")];
    // caller garante AVX2+FMA disponíveis em runtime (via is_x86_feature_detected!).
    // Ponteiros gerados a partir de slices válidos com bounds verificados pelo loop.
    let mut acc = unsafe { _mm256_setzero_ps() };

    for b in 0..n_blocks {
        let bw = b * B;
        let d_bits = u16::from_le_bytes([w_row[bw], w_row[bw + 1]]);
        let d = unsafe { _mm256_set1_ps(half::f16::from_bits(d_bits).to_f32()) };

        let qs = unsafe { w_row.as_ptr().add(bw + 2) };
        let xb = unsafe { x_row.as_ptr().add(b * Q) };

        let q0 = unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs as *const __m128i)) };
        let q1 = unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(8) as *const __m128i)) };
        let q2 = unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(16) as *const __m128i)) };
        let q3 = unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(24) as *const __m128i)) };

        let x0 = unsafe { _mm256_loadu_ps(xb) };
        let x1 = unsafe { _mm256_loadu_ps(xb.add(8)) };
        let x2 = unsafe { _mm256_loadu_ps(xb.add(16)) };
        let x3 = unsafe { _mm256_loadu_ps(xb.add(24)) };

        let s0 = unsafe { _mm256_mul_ps(_mm256_cvtepi32_ps(q0), x0) };
        let s1 = unsafe { _mm256_mul_ps(_mm256_cvtepi32_ps(q1), x1) };
        let s2 = unsafe { _mm256_mul_ps(_mm256_cvtepi32_ps(q2), x2) };
        let s3 = unsafe { _mm256_mul_ps(_mm256_cvtepi32_ps(q3), x3) };

        let sum = unsafe { _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)) };
        acc = unsafe { _mm256_fmadd_ps(d, sum, acc) };
    }

    unsafe { hsum_f32_avx(acc) }
}

/// Kernel AVX2 explícito para Q8_0 × Q8_0.
/// Usa VPMADDUBSW (abs(qw) u8 × signed_qx i8 → i16) + VPMADDWD(×1→i32):
/// 4 instruções por bloco de 32 elementos vs 7 na abordagem cvtepi8_epi16.
/// Prova de não-saturação: max par = 128×127+128×127 = 32512 < 32767 (i16 max).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn q8_0_q8_0_dot_avx2(w_row: &[u8], x_q8_row: &[u8], n_blocks: usize) -> f32 {
    use std::arch::x86_64::*;
    const B: usize = 34;

    // SAFETY: intrinsics protegidos por #[target_feature]; slices têm n_blocks × 34 bytes.
    let mut acc = unsafe { _mm256_setzero_ps() };
    let ones = unsafe { _mm256_set1_epi16(1) };

    for b in 0..n_blocks {
        let boff = b * B;
        unsafe {
            let dw = half::f16::from_bits(u16::from_le_bytes([
                *w_row.as_ptr().add(boff),
                *w_row.as_ptr().add(boff + 1),
            ]))
            .to_f32();
            let dx = half::f16::from_bits(u16::from_le_bytes([
                *x_q8_row.as_ptr().add(boff),
                *x_q8_row.as_ptr().add(boff + 1),
            ]))
            .to_f32();

            let qw = _mm256_loadu_si256(w_row.as_ptr().add(boff + 2) as *const __m256i);
            let qx = _mm256_loadu_si256(x_q8_row.as_ptr().add(boff + 2) as *const __m256i);

            // abs(qw): i8 → u8; sign(qx, qw): negate qx where qw<0 → qw*qx = abs(qw)*signed_qx
            let abs_qw = _mm256_abs_epi8(qw);
            let signed_qx = _mm256_sign_epi8(qx, qw);
            // maddubs: 32 pares u8×i8 → 16 somas i16; madd(×1): 16 pares i16 → 8 somas i32
            let sum_i32 = _mm256_madd_epi16(_mm256_maddubs_epi16(abs_qw, signed_qx), ones);

            acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * dx), _mm256_cvtepi32_ps(sum_i32), acc);
        }
    }

    unsafe { hsum_f32_avx(acc) }
}

/// Redução horizontal: 8 floats AVX → 1 f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn hsum_f32_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: veja q8_0_dot_avx2.
    unsafe {
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
        _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
    }
}

/// Otimização 1 — produto escalar Q8_0 × f32 com dispatch AVX2/fallback scalar.
#[inline]
fn q8_0_dot(w_row: &[u8], x_row: &[f32], n_blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detectada em runtime; slices têm tamanho correto (n_blocks × 34 e n_blocks × 32).
            return unsafe { q8_0_dot_avx2(w_row, x_row, n_blocks) };
        }
    }
    q8_0_dot_scalar(w_row, x_row, n_blocks)
}

/// Fallback scalar para Q8_0 × f32.
#[inline]
fn q8_0_dot_scalar(w_row: &[u8], x_row: &[f32], n_blocks: usize) -> f32 {
    const Q: usize = 32;
    const B: usize = 34;
    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let blk = &w_row[b * B..(b + 1) * B];
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let qs = &blk[2..2 + Q];
        let xb = &x_row[b * Q..(b + 1) * Q];
        let mut dot = 0.0f32;
        for i in 0..Q {
            dot += (qs[i] as i8 as f32) * xb[i];
        }
        acc += d * dot;
    }
    acc
}

/// AVX2: quantiza 32 f32 por bloco via SIMD — ~10× mais rápido que scalar.
/// Usa packs saturante (i32→i16→i8) + permutevar8x32 para corrigir lane ordering.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_q8_0_avx2(x: &[f32]) -> Vec<u8> {
    use std::arch::x86_64::*;
    const Q: usize = 32;
    const B: usize = 34;
    let n_blocks = x.len() / Q;
    let capacity = n_blocks * B;
    let mut out = Vec::with_capacity(capacity);
    // abs mask: zera o bit de sinal sem branch
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFi32));
    // permutação para corrigir cross-lane ordering após packs: dwords [0,4,1,5,2,6,3,7]
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let base_ptr: *mut u8 = out.as_mut_ptr();
    for b in 0..n_blocks {
        let p = x.as_ptr().add(b * Q);
        let v0 = _mm256_loadu_ps(p);
        let v1 = _mm256_loadu_ps(p.add(8));
        let v2 = _mm256_loadu_ps(p.add(16));
        let v3 = _mm256_loadu_ps(p.add(24));
        // max abs via árvore de reduções
        let max01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
        let max23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
        let max4 = _mm256_max_ps(max01, max23);
        let hi = _mm256_extractf128_ps(max4, 1);
        let lo = _mm256_castps256_ps128(max4);
        let m = _mm_max_ps(hi, lo);
        let m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x4E)); // swap pairs [2,3,0,1]
        let m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x01)); // lane0 ← max(lane0, lane1)
        let max_abs = _mm_cvtss_f32(m);
        // escala
        let d = if max_abs > 0.0 {
            max_abs / 127.0_f32
        } else {
            0.0_f32
        };
        let d_inv = if d > 0.0 { 1.0_f32 / d } else { 0.0_f32 };
        let d_bits = half::f16::from_f32(d).to_bits().to_le_bytes();
        // quantiza: multiply + round-to-nearest (MXCSR default) + convert to i32
        let inv_v = _mm256_set1_ps(d_inv);
        let q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, inv_v));
        let q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, inv_v));
        let q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, inv_v));
        let q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, inv_v));
        // i32→i16→i8 com saturação; corrige cross-lane com permutevar
        let q01 = _mm256_packs_epi32(q0, q1);
        let q23 = _mm256_packs_epi32(q2, q3);
        let bytes32 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(q01, q23), perm);
        // escreve diretamente no buffer pré-alocado
        let dst = base_ptr.add(b * B);
        std::ptr::copy_nonoverlapping(d_bits.as_ptr(), dst, 2);
        _mm256_storeu_si256(dst.add(2) as *mut __m256i, bytes32);
    }
    // SAFETY: capacity bytes foram escritos acima
    out.set_len(capacity);
    out
}

/// Otimização 2a — quantiza um vetor f32 para Q8_0.
/// Saída: n_in/32 blocos × 34 bytes (2 bytes f16 scale + 32 bytes i8).
fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: feature AVX2 detectada em runtime
        return unsafe { quantize_q8_0_avx2(x) };
    }
    const Q: usize = 32;
    const B: usize = 34;
    let n_blocks = x.len() / Q;
    let mut out = Vec::with_capacity(n_blocks * B);
    for b in 0..n_blocks {
        let blk = &x[b * Q..(b + 1) * Q];
        let max_abs = blk.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
        let d_inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for &v in blk {
            let q = (v * d_inv).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

/// Otimização 2b — quantiza x token-major [n_tok × n_in] para Q8_0 (token a token).
pub(crate) fn quantize_q8_0_split(x: &[f32], n_in: usize, n_tok: usize) -> Vec<u8> {
    const Q: usize = 32;
    const B: usize = 34;
    let n_blocks = n_in / Q;
    let tok_bytes = n_blocks * B;
    let mut out = Vec::with_capacity(n_tok * tok_bytes);
    for t in 0..n_tok {
        let row = &x[t * n_in..(t + 1) * n_in];
        out.extend(quantize_q8_0(row));
    }
    out
}

/// Otimização 2c — produto escalar Q8_0 × Q8_0 → f32 com dispatch AVX2/fallback scalar.
#[inline]
fn q8_0_q8_0_dot(w_row: &[u8], x_q8_row: &[u8], n_blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detectada em runtime; slices têm n_blocks × 34 bytes.
            return unsafe { q8_0_q8_0_dot_avx2(w_row, x_q8_row, n_blocks) };
        }
    }
    q8_0_q8_0_dot_scalar(w_row, x_q8_row, n_blocks)
}

/// Kernel AVX2 de 4 linhas: processa 4 linhas de saída simultaneamente,
/// reutilizando qx_lo/qx_hi e dx em todos os 4 pesos — reduz recargas de x 4×.
///
/// SAFETY: caller deve garantir AVX2+FMA disponíveis; ponteiros w_ptr e x_ptr devem
/// apontar para slices válidos com ao menos 4*row_bytes e n_blocks*34 bytes respectivamente.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_0_q8_0_dot_4rows_avx2(
    w_ptr: *const u8,
    row_bytes: usize,
    x_ptr: *const u8,
    n_blocks: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;
    const B: usize = 34;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let w1 = w_ptr.add(row_bytes);
    let w2 = w_ptr.add(2 * row_bytes);
    let w3 = w_ptr.add(3 * row_bytes);

    let ones = _mm256_set1_epi16(1);
    // Pré-busca para esconder latência: o loop tem apenas 28 iterações (n_in=896),
    // o que é curto demais para o hardware prefetcher aprender o padrão a tempo.
    const PF: usize = 1; // distância de pré-busca em blocos
    const PF_BYTES: usize = PF * B;

    for b in 0..n_blocks {
        let boff = b * B;

        // Software prefetch — carrega dados de w_ptr..w3 no L2 para a iteração b+PF.
        if boff + PF_BYTES < n_blocks * B {
            let next = boff + PF_BYTES;
            _mm_prefetch(w_ptr.add(next) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w1.add(next) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w2.add(next) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w3.add(next) as *const i8, _MM_HINT_T1);
        }

        // Carrega bloco x uma única vez — dx e qx ficam em registradores para as 4 linhas.
        let dx = half::f16::from_bits(u16::from_le_bytes([*x_ptr.add(boff), *x_ptr.add(boff + 1)]))
            .to_f32();
        let qx = _mm256_loadu_si256(x_ptr.add(boff + 2) as *const __m256i);

        // Macro maddubs: abs(qw)×sign(qx,qw) → i16 pares → i32; 4 instr. vs 7 (cvtepi8×4+madd×2+add).
        macro_rules! dot_row {
            ($acc:ident, $wp:expr) => {{
                let dw =
                    half::f16::from_bits(u16::from_le_bytes([*$wp.add(boff), *$wp.add(boff + 1)]))
                        .to_f32();
                let qw = _mm256_loadu_si256($wp.add(boff + 2) as *const __m256i);
                let sum_i32 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_abs_epi8(qw), _mm256_sign_epi8(qx, qw)),
                    ones,
                );
                $acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * dx), _mm256_cvtepi32_ps(sum_i32), $acc);
            }};
        }

        dot_row!(acc0, w_ptr);
        dot_row!(acc1, w1);
        dot_row!(acc2, w2);
        dot_row!(acc3, w3);
    }

    [
        hsum_f32_avx(acc0),
        hsum_f32_avx(acc1),
        hsum_f32_avx(acc2),
        hsum_f32_avx(acc3),
    ]
}

/// Kernel 8 linhas com F16C: converte escala f16→f32 em 1 instrução (vs ~10 no software),
/// e amortiza o carregamento de x em 8 saídas em vez de 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_0_q8_0_dot_8rows_avx2_f16c(
    w_ptr: *const u8,
    row_bytes: usize,
    x_ptr: *const u8,
    n_blocks: usize,
) -> [f32; 8] {
    use std::arch::x86_64::*;
    const B: usize = 34;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();

    let w1 = w_ptr.add(row_bytes);
    let w2 = w_ptr.add(2 * row_bytes);
    let w3 = w_ptr.add(3 * row_bytes);
    let w4 = w_ptr.add(4 * row_bytes);
    let w5 = w_ptr.add(5 * row_bytes);
    let w6 = w_ptr.add(6 * row_bytes);
    let w7 = w_ptr.add(7 * row_bytes);

    let ones = _mm256_set1_epi16(1);
    const PF: usize = 2;
    const PF_BYTES: usize = PF * B;

    for b in 0..n_blocks {
        let boff = b * B;
        if boff + PF_BYTES < n_blocks * B {
            let nxt = boff + PF_BYTES;
            _mm_prefetch(w_ptr.add(nxt) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w1.add(nxt) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w2.add(nxt) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w3.add(nxt) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w4.add(nxt) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w5.add(nxt) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w6.add(nxt) as *const i8, _MM_HINT_T1);
            _mm_prefetch(w7.add(nxt) as *const i8, _MM_HINT_T1);
        }

        // F16C: carrega f16 em 1 instrução vcvtph2ps ao invés de ~10 instruções software.
        macro_rules! f16c {
            ($p:expr) => {{
                let bits = (*($p).add(boff) as i32) | ((*($p).add(boff + 1) as i32) << 8);
                _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)))
            }};
        }
        let dx = f16c!(x_ptr);
        let qx = _mm256_loadu_si256(x_ptr.add(boff + 2) as *const __m256i);

        macro_rules! dot_row {
            ($acc:ident, $wp:expr) => {{
                let dw = f16c!($wp);
                let qw = _mm256_loadu_si256($wp.add(boff + 2) as *const __m256i);
                let sum_i32 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_abs_epi8(qw), _mm256_sign_epi8(qx, qw)),
                    ones,
                );
                $acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * dx), _mm256_cvtepi32_ps(sum_i32), $acc);
            }};
        }

        dot_row!(acc0, w_ptr);
        dot_row!(acc1, w1);
        dot_row!(acc2, w2);
        dot_row!(acc3, w3);
        dot_row!(acc4, w4);
        dot_row!(acc5, w5);
        dot_row!(acc6, w6);
        dot_row!(acc7, w7);
    }

    [
        hsum_f32_avx(acc0),
        hsum_f32_avx(acc1),
        hsum_f32_avx(acc2),
        hsum_f32_avx(acc3),
        hsum_f32_avx(acc4),
        hsum_f32_avx(acc5),
        hsum_f32_avx(acc6),
        hsum_f32_avx(acc7),
    ]
}

/// Recompacta pesos Q8_0 de row-major para block_q8_0x8.
/// Por grupo de 8 linhas, por bloco b:
///   bytes 0..16   : 8 escalas f16 (d0..d7)
///   bytes 16..272 : 8 × 32 quants (qs0[32] ++ qs1[32] ++ ... ++ qs7[32])
///   total = 272 bytes por superbloco.
/// Exige n_out múltiplo de 8; n_in múltiplo de 32.
pub(crate) fn repack_q8_0_8rows(w: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    const Q: usize = 32;
    const B: usize = 34;
    const PB: usize = 272; // 16 + 256
    debug_assert_eq!(n_out % 8, 0);
    debug_assert_eq!(n_in % Q, 0);
    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let n_groups = n_out / 8;
    let mut out = vec![0u8; n_groups * n_blocks * PB];
    for g in 0..n_groups {
        let base_row = g * 8;
        for b in 0..n_blocks {
            let dst = (g * n_blocks + b) * PB;
            // 8 escalas f16
            for j in 0..8 {
                let src = (base_row + j) * row_bytes + b * B;
                out[dst + j * 2] = w[src];
                out[dst + j * 2 + 1] = w[src + 1];
            }
            // 8 × 32 quants
            for j in 0..8 {
                let src = (base_row + j) * row_bytes + b * B + 2;
                let d = dst + 16 + j * Q;
                out[d..d + Q].copy_from_slice(&w[src..src + Q]);
            }
        }
    }
    out
}

/// Kernel para pesos em block_q8_0x8 + ativações Q8_0 padrão.
/// packed_ptr aponta para um grupo de 8 linhas em formato repacked:
///   bloco b @ packed_ptr + b*272 : [d0..d7 (16B f16)] + [qs0[32]..qs7[32] (256B)]
/// x_ptr aponta para ativações Q8_0 padrão (n_blocks × 34 bytes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_0_q8_0_dot_8rows_repacked_f16c(
    packed_ptr: *const u8,
    x_ptr: *const u8,
    n_blocks: usize,
) -> [f32; 8] {
    use std::arch::x86_64::*;
    const XB: usize = 34;
    const PB: usize = 272;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);

    for b in 0..n_blocks {
        let boff = b * PB;
        let x_boff = b * XB;
        // Prefetch próximo bloco (272 bytes = 4-5 cache lines)
        if b + 2 < n_blocks {
            let nxt = (b + 2) * PB;
            _mm_prefetch(packed_ptr.add(nxt) as *const i8, _MM_HINT_T0);
            _mm_prefetch(packed_ptr.add(nxt + 64) as *const i8, _MM_HINT_T0);
            _mm_prefetch(packed_ptr.add(nxt + 128) as *const i8, _MM_HINT_T0);
            _mm_prefetch(packed_ptr.add(nxt + 192) as *const i8, _MM_HINT_T0);
        }
        // Escala x via F16C
        let dx_bits = (*x_ptr.add(x_boff) as i32) | ((*x_ptr.add(x_boff + 1) as i32) << 8);
        let dx = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dx_bits)));
        let qx = _mm256_loadu_si256(x_ptr.add(x_boff + 2) as *const __m256i);
        // Carrega 8 escalas f16 em 1 load de 128 bits → 2 × _mm_cvtph_ps
        let sc128 = _mm_loadu_si128(packed_ptr.add(boff) as *const __m128i);
        let dw_lo = _mm_cvtph_ps(sc128);
        let dw_hi = _mm_cvtph_ps(_mm_srli_si128(sc128, 8));
        let d0 = _mm_cvtss_f32(dw_lo);
        let d1 = _mm_cvtss_f32(_mm_shuffle_ps(dw_lo, dw_lo, 0x55));
        let d2 = _mm_cvtss_f32(_mm_shuffle_ps(dw_lo, dw_lo, 0xAA));
        let d3 = _mm_cvtss_f32(_mm_shuffle_ps(dw_lo, dw_lo, 0xFF));
        let d4 = _mm_cvtss_f32(dw_hi);
        let d5 = _mm_cvtss_f32(_mm_shuffle_ps(dw_hi, dw_hi, 0x55));
        let d6 = _mm_cvtss_f32(_mm_shuffle_ps(dw_hi, dw_hi, 0xAA));
        let d7 = _mm_cvtss_f32(_mm_shuffle_ps(dw_hi, dw_hi, 0xFF));
        macro_rules! dot_row {
            ($acc:ident, $j:expr, $d:expr) => {{
                let qw = _mm256_loadu_si256(packed_ptr.add(boff + 16 + $j * 32) as *const __m256i);
                let sum_i32 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_abs_epi8(qw), _mm256_sign_epi8(qx, qw)),
                    ones,
                );
                $acc = _mm256_fmadd_ps(_mm256_set1_ps($d * dx), _mm256_cvtepi32_ps(sum_i32), $acc);
            }};
        }
        dot_row!(acc0, 0, d0);
        dot_row!(acc1, 1, d1);
        dot_row!(acc2, 2, d2);
        dot_row!(acc3, 3, d3);
        dot_row!(acc4, 4, d4);
        dot_row!(acc5, 5, d5);
        dot_row!(acc6, 6, d6);
        dot_row!(acc7, 7, d7);
    }
    [
        hsum_f32_avx(acc0),
        hsum_f32_avx(acc1),
        hsum_f32_avx(acc2),
        hsum_f32_avx(acc3),
        hsum_f32_avx(acc4),
        hsum_f32_avx(acc5),
        hsum_f32_avx(acc6),
        hsum_f32_avx(acc7),
    ]
}

/// Fallback scalar para Q8_0 × Q8_0.
#[inline]
fn q8_0_q8_0_dot_scalar(w_row: &[u8], x_q8_row: &[u8], n_blocks: usize) -> f32 {
    const Q: usize = 32;
    const B: usize = 34;
    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let bw = b * B;
        let bx = b * B;
        let dw = half::f16::from_bits(u16::from_le_bytes([w_row[bw], w_row[bw + 1]])).to_f32();
        let dx =
            half::f16::from_bits(u16::from_le_bytes([x_q8_row[bx], x_q8_row[bx + 1]])).to_f32();
        let qsw = &w_row[bw + 2..bw + 2 + Q];
        let qsx = &x_q8_row[bx + 2..bx + 2 + Q];
        let mut dot = 0i32;
        for i in 0..Q {
            dot += (qsw[i] as i8 as i32) * (qsx[i] as i8 as i32);
        }
        acc += dw * dx * (dot as f32);
    }
    acc
}

/// Otimização 2d — MUL_MAT Q8_0 com ativações também quantizadas (i8×i8 mais rápido que f32×i8).
/// `x_q8`: saída de `quantize_q8_0_split(x, n_in, n_tok)`.
/// Usa kernel de 4 linhas AVX2 para reduzir recargas de x — feature detectada uma vez por chamada.
pub(crate) fn matmul_q8_0_actq(
    w: &[u8],
    x_q8: &[u8],
    n_in: usize,
    n_out: usize,
    n_tok: usize,
) -> Vec<f32> {
    const Q: usize = 32;
    const B: usize = 34;
    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let mut out = vec![0.0f32; n_tok * n_out];

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    #[cfg(target_arch = "x86_64")]
    let use_f16c = use_avx2 && is_x86_feature_detected!("f16c");
    #[cfg(not(target_arch = "x86_64"))]
    let use_f16c = false;

    // Static work division: one contiguous range per thread, like llamafile/tinyBLAS.
    // ceil(n_out / n_threads) rounded up to nearest 8 for the 8-row inner kernel.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = {
        let raw = (n_out + n_threads - 1) / n_threads;
        ((raw.max(8) + 7) / 8) * 8
    };

    for t in 0..n_tok {
        let x_row = &x_q8[t * row_bytes..(t + 1) * row_bytes];
        let o_row = &mut out[t * n_out..(t + 1) * n_out];

        if n_out >= PAR_MIN_N_OUT_Q8 {
            if use_f16c {
                // Kernel 8 linhas + F16C: metade dos recarregamentos de x, conversão f16 em 1 ciclo.
                o_row
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let base_j = ci * chunk_size;
                        let n_full8 = (chunk.len() / 8) * 8;
                        let mut k = 0usize;
                        while k < n_full8 {
                            let j = base_j + k;
                            // SAFETY: f16c+avx2 detectados; j+7 < n_out por construção.
                            let [r0, r1, r2, r3, r4, r5, r6, r7] = unsafe {
                                q8_0_q8_0_dot_8rows_avx2_f16c(
                                    w.as_ptr().add(j * row_bytes),
                                    row_bytes,
                                    x_row.as_ptr(),
                                    n_blocks,
                                )
                            };
                            chunk[k] = r0;
                            chunk[k + 1] = r1;
                            chunk[k + 2] = r2;
                            chunk[k + 3] = r3;
                            chunk[k + 4] = r4;
                            chunk[k + 5] = r5;
                            chunk[k + 6] = r6;
                            chunk[k + 7] = r7;
                            k += 8;
                        }
                        // restante em grupos de 4
                        let n_full4 = k + ((chunk.len() - k) / 4) * 4;
                        while k < n_full4 {
                            let j = base_j + k;
                            let [r0, r1, r2, r3] = unsafe {
                                q8_0_q8_0_dot_4rows_avx2(
                                    w.as_ptr().add(j * row_bytes),
                                    row_bytes,
                                    x_row.as_ptr(),
                                    n_blocks,
                                )
                            };
                            chunk[k] = r0;
                            chunk[k + 1] = r1;
                            chunk[k + 2] = r2;
                            chunk[k + 3] = r3;
                            k += 4;
                        }
                        for rem in k..chunk.len() {
                            let j = base_j + rem;
                            chunk[rem] = q8_0_q8_0_dot_scalar(
                                &w[j * row_bytes..(j + 1) * row_bytes],
                                x_row,
                                n_blocks,
                            );
                        }
                    });
            } else if use_avx2 {
                o_row
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let base_j = ci * chunk_size;
                        let n_full4 = (chunk.len() / 4) * 4;
                        let mut k = 0usize;
                        while k < n_full4 {
                            let j = base_j + k;
                            let [r0, r1, r2, r3] = unsafe {
                                q8_0_q8_0_dot_4rows_avx2(
                                    w.as_ptr().add(j * row_bytes),
                                    row_bytes,
                                    x_row.as_ptr(),
                                    n_blocks,
                                )
                            };
                            chunk[k] = r0;
                            chunk[k + 1] = r1;
                            chunk[k + 2] = r2;
                            chunk[k + 3] = r3;
                            k += 4;
                        }
                        for rem in k..chunk.len() {
                            let j = base_j + rem;
                            chunk[rem] = q8_0_q8_0_dot_scalar(
                                &w[j * row_bytes..(j + 1) * row_bytes],
                                x_row,
                                n_blocks,
                            );
                        }
                    });
            } else {
                o_row
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        for (k, o) in chunk.iter_mut().enumerate() {
                            let j = ci * chunk_size + k;
                            *o = q8_0_q8_0_dot_scalar(
                                &w[j * row_bytes..(j + 1) * row_bytes],
                                x_row,
                                n_blocks,
                            );
                        }
                    });
            }
        } else {
            for (j, o) in o_row.iter_mut().enumerate() {
                *o = q8_0_q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
            }
        }
    }
    out
}

/// Variante de matmul_q8_0_actq que usa pesos pré-empacotados em block_q8_0x8.
/// packed_w: saída de repack_q8_0_8rows (cobre as primeiras n_out_packed linhas, múltiplo de 8).
/// w_tail: bytes raw das linhas restantes (n_out - n_out_packed), pode ser vazio.
pub(crate) fn matmul_q8_0_actq_packed(
    packed_w: &[u8],
    w_tail: &[u8],
    x_q8: &[u8],
    n_in: usize,
    n_out: usize,
    n_out_packed: usize,
    n_tok: usize,
) -> Vec<f32> {
    const Q: usize = 32;
    const B: usize = 34;
    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    const PB: usize = 272;
    let mut out = vec![0.0f32; n_tok * n_out];

    #[cfg(target_arch = "x86_64")]
    let use_f16c = is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c");
    #[cfg(not(target_arch = "x86_64"))]
    let use_f16c = false;

    let n_groups = n_out_packed / 8;

    // Static work division: one contiguous range per thread (same as llamafile).
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = {
        let raw = (n_out_packed.max(1) + n_threads - 1) / n_threads;
        ((raw.max(8) + 7) / 8) * 8
    };
    let n_groups = n_out_packed / 8;

    for t in 0..n_tok {
        let x_row = &x_q8[t * row_bytes..(t + 1) * row_bytes];
        let o_row = &mut out[t * n_out..(t + 1) * n_out];
        let (o_packed, o_tail) = o_row.split_at_mut(n_out_packed);

        // Repacked groups of 8 rows — spin pool (overhead ~1-3 µs vs ~26 µs rayon)
        if use_f16c && n_groups > 0 {
            let n_chunks = (n_out_packed + chunk_size - 1) / chunk_size;
            let batch = crate::spin_pool::MatmulBatch {
                packed_w: packed_w.as_ptr(),
                x_row: x_row.as_ptr(),
                w_tail: w_tail.as_ptr(),
                out: o_packed.as_mut_ptr(),
                n_blocks,
                chunk_size,
                n_out_packed,
                row_bytes,
            };
            // SAFETY: batch vive até dispatch retornar; ponteiros dentro de packed_w/x_row válidos.
            unsafe {
                crate::spin_pool::dispatch(&batch, n_chunks);
            }
        }

        // Tail rows (not multiple of 8)
        for (j, o) in o_tail.iter_mut().enumerate() {
            *o = q8_0_q8_0_dot_scalar(&w_tail[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
        }
    }
    out
}

/// Otimização 3 — RoPE NORM com tabela de frequências pré-computada.
/// `freq_table[i] = freq_base^(-2i/rope_dim)` para i em 0..rope_dim/2.
/// Elimina `powf` no loop interno (era a operação mais cara da RoPE).
pub(crate) fn rope_norm(
    x: &mut [f32],
    n_tok: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_table: &[f32],
    pos0: usize,
) {
    for t in 0..n_tok {
        let pos = (pos0 + t) as f32;
        for h in 0..n_head {
            let base = (t * n_head + h) * head_dim;
            for i in 0..rope_dim / 2 {
                let theta = pos * freq_table[i];
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

// ──────────────────────────────────────────────────────────────────────────────
// Wrappers pub(crate) para uso em spin_pool
// ──────────────────────────────────────────────────────────────────────────────

/// SAFETY: AVX2+FMA+F16C devem estar disponíveis; ponteiros válidos por n_blocks blocos.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn q8_0_q8_0_dot_8rows_repacked_f16c_pub(
    packed_ptr: *const u8,
    x_ptr: *const u8,
    n_blocks: usize,
) -> [f32; 8] {
    q8_0_q8_0_dot_8rows_repacked_f16c(packed_ptr, x_ptr, n_blocks)
}

/// SAFETY: w_ptr e x_ptr devem apontar para pelo menos `row_bytes` bytes válidos.
pub(crate) unsafe fn q8_0_q8_0_dot_scalar_pub(
    w_ptr: *const u8,
    x_ptr: *const u8,
    n_blocks: usize,
    row_bytes: usize,
) -> f32 {
    let w_row = std::slice::from_raw_parts(w_ptr, row_bytes);
    let x_row = std::slice::from_raw_parts(x_ptr, row_bytes);
    q8_0_q8_0_dot_scalar(w_row, x_row, n_blocks)
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
    fn rmsnorm_and_scale_matches_rmsnorm_plus_mul_rows() {
        // Verifica equivalência matemática com dois passos separados
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens, dim=2
        let weight = vec![0.5f32, 2.0];
        let eps = 1e-5;
        let dim = 2;
        let expected = mul_rows(&rmsnorm(&x, dim, eps), &weight, dim);
        let got = rmsnorm_and_scale(&x, &weight, dim, eps);
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-6, "divergência: {a} vs {b}");
        }
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
        // freq_table para rope_dim=4: [base^0, base^(-2/4)] = [1.0, 1/100.0]
        let freq_table: Vec<f32> = (0..2)
            .map(|i| 10000.0f32.powf(-2.0 * i as f32 / 4.0))
            .collect();
        rope_norm(&mut x, 1, 1, 4, 4, &freq_table, 0);
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
        let freq_table: Vec<f32> = (0..2)
            .map(|i| 10000.0f32.powf(-2.0 * i as f32 / 4.0))
            .collect();
        rope_norm(&mut x, 1, 1, 4, 4, &freq_table, 1);
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

    #[test]
    fn matmul_q8_0_matches_manual() {
        // 2 rows × 1 block Q8_0 (n_in=32, n_out=2, n_tok=1)
        // row 0: d=1.0, qs=[1,2,0×30] → dot(x) = 1×1 + 2×2 = 5 → out=5.0
        // row 1: d=2.0, qs=[1,0×31]   → dot(x) = 1×1       = 1 → out=2.0
        fn f16_le(v: f32) -> [u8; 2] {
            half::f16::from_f32(v).to_bits().to_le_bytes()
        }
        let mut w = Vec::with_capacity(68);
        w.extend_from_slice(&f16_le(1.0));
        w.push(1u8);
        w.push(2u8);
        w.extend(std::iter::repeat(0u8).take(30));
        w.extend_from_slice(&f16_le(2.0));
        w.push(1u8);
        w.extend(std::iter::repeat(0u8).take(31));

        let x: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let out = matmul_q8_0(&w, &x, 32, 2, 1);

        assert!((out[0] - 5.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-4, "out[1]={}", out[1]);
    }

    #[test]
    fn matmul_q8_0_two_tokens() {
        // n_in=32, n_out=1, n_tok=2; d=1.0, qs=[1,0×31]
        fn f16_le(v: f32) -> [u8; 2] {
            half::f16::from_f32(v).to_bits().to_le_bytes()
        }
        let mut w = Vec::with_capacity(34);
        w.extend_from_slice(&f16_le(1.0));
        w.push(1u8);
        w.extend(std::iter::repeat(0u8).take(31));

        let mut x = vec![0.0f32; 64];
        x[0] = 3.0; // token 0, dim 0
        x[32] = 7.0; // token 1, dim 0

        let out = matmul_q8_0(&w, &x, 32, 1, 2);

        assert!((out[0] - 3.0).abs() < 1e-4, "t0={}", out[0]);
        assert!((out[1] - 7.0).abs() < 1e-4, "t1={}", out[1]);
    }

    #[test]
    fn matmul_q8_0_actq_matches_matmul_q8_0() {
        // Verifica que actq produz resultados próximos ao q8_0 original
        // (pequena diferença de quantização de ativações é esperada)
        fn f16_le(v: f32) -> [u8; 2] {
            half::f16::from_f32(v).to_bits().to_le_bytes()
        }
        let mut w = Vec::with_capacity(68);
        w.extend_from_slice(&f16_le(1.0));
        w.push(1u8);
        w.push(2u8);
        w.extend(std::iter::repeat(0u8).take(30));
        w.extend_from_slice(&f16_le(2.0));
        w.push(1u8);
        w.extend(std::iter::repeat(0u8).take(31));

        let x: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let expected = matmul_q8_0(&w, &x, 32, 2, 1);
        let x_q8 = quantize_q8_0_split(&x, 32, 1);
        let got = matmul_q8_0_actq(&w, &x_q8, 32, 2, 1);

        // Tolerância maior (1%) devido ao ruído de quantização das ativações
        assert!(
            (got[0] - expected[0]).abs() / expected[0].abs() < 0.01,
            "out[0]: got={} expected={}",
            got[0],
            expected[0]
        );
        assert!(
            (got[1] - expected[1]).abs() / expected[1].abs() < 0.01,
            "out[1]: got={} expected={}",
            got[1],
            expected[1]
        );
    }
}
