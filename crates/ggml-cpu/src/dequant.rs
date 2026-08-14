//! Dequantização de blocos GGML para f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgmlType;

use crate::error::DequantError;

/// Converte `bytes` brutos de um tensor para `Vec<f32>`.
/// Suporta F32, F16, Q8_0, Q4_0, Q4_K, Q6_K.
pub fn dequant_to_f32(bytes: &[u8], ty: GgmlType) -> Result<Vec<f32>, DequantError> {
    match ty {
        GgmlType::F32 => dequant_f32(bytes),
        GgmlType::F16 => dequant_f16(bytes),
        GgmlType::Q8_0 => dequant_q8_0(bytes),
        GgmlType::Q4_0 => dequant_q4_0(bytes),
        GgmlType::Q4_K => dequant_q4_k(bytes),
        GgmlType::Q5_K => dequant_q5_k(bytes),
        GgmlType::Q6_K => dequant_q6_k(bytes),
        other => Err(DequantError::UnsupportedType(format!("{other:?}"))),
    }
}

fn dequant_f32(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DequantError::BadSize {
            ty: "F32",
            block_bytes: 4,
            got: bytes.len(),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn dequant_f16(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(DequantError::BadSize {
            ty: "F16",
            block_bytes: 2,
            got: bytes.len(),
        });
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect())
}

fn dequant_q8_0(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 34; // 2 (f16) + 32 (i8)
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q8_0",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for b in bytes.chunks_exact(BLOCK) {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        for &q in &b[2..34] {
            out.push(q.cast_signed() as f32 * d);
        }
    }
    Ok(out)
}
fn dequant_q4_0(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 18; // 2 (f16) + 16 (nibbles)
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q4_0",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 32];
    for (bi, b) in bytes.chunks_exact(BLOCK).enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let base = bi * 32;
        for j in 0..16 {
            let q = b[2 + j];
            let x0 = i32::from(q & 0x0F) - 8;
            let x1 = i32::from(q >> 4) - 8;
            out[base + j] = x0 as f32 * d;
            out[base + j + 16] = x1 as f32 * d;
        }
    }
    Ok(out)
}
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 144; // 2+2+12+128
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q4_K",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.chunks_exact(BLOCK).enumerate() {
        let d_val = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let min_val = half::f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
        let scales = &b[4..16];
        let qs = &b[16..144];
        let base = bi * 256;
        let mut qs_off = 0usize;
        let mut is = 0usize;

        for j_step in [0usize, 64, 128, 192] {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d_val * f32::from(sc1);
            let m1f = min_val * f32::from(m1);
            let d2 = d_val * f32::from(sc2);
            let m2f = min_val * f32::from(m2);
            for l in 0..32 {
                let q = qs[qs_off + l];
                out[base + j_step + l] = d1 * f32::from(q & 0xF) - m1f;
                out[base + j_step + l + 32] = d2 * f32::from(q >> 4) - m2f;
            }
            qs_off += 32;
            is += 2;
        }
    }
    Ok(out)
}
/// Q5_K: superbloco de 256 elementos em 176 bytes —
/// `d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128]`.
///
/// Mesmo esquema de escalas do Q4_K (6 bits por sub-bloco de 32, via `get_scale_min_k4`),
/// acrescido de um 5º bit por elemento vindo de `qh`: o bit `l` de `qh[l]` pertence ao
/// primeiro grupo de 32 e o bit seguinte ao segundo, avançando 2 bits a cada 64 elementos.
fn dequant_q5_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 176; // 2+2+12+32+128
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q5_K",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.chunks_exact(BLOCK).enumerate() {
        let d_val = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let min_val = half::f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        let base = bi * 256;
        let mut qs_off = 0usize;
        let mut is = 0usize;
        // Máscaras do bit alto: avançam duas posições a cada grupo de 64 elementos.
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for j_step in [0usize, 64, 128, 192] {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d_val * f32::from(sc1);
            let m1f = min_val * f32::from(m1);
            let d2 = d_val * f32::from(sc2);
            let m2f = min_val * f32::from(m2);
            for l in 0..32 {
                let q = qs[qs_off + l];
                let hi1 = u8::from(qh[l] & u1 != 0) * 16;
                let hi2 = u8::from(qh[l] & u2 != 0) * 16;
                out[base + j_step + l] = d1 * f32::from((q & 0xF) + hi1) - m1f;
                out[base + j_step + l + 32] = d2 * f32::from((q >> 4) + hi2) - m2f;
            }
            qs_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

fn dequant_q6_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 210; // 128+64+16+2
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(DequantError::BadSize {
            ty: "Q6_K",
            block_bytes: BLOCK,
            got: bytes.len(),
        });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.chunks_exact(BLOCK).enumerate() {
        let ql_full = &b[0..128];
        let qh_full = &b[128..192];
        let sc_full = &b[192..208]; // [i8; 16] stored as u8
        let d_val = half::f16::from_bits(u16::from_le_bytes([b[208], b[209]])).to_f32();
        let base = bi * 256;

        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;

        for n in [0usize, 128] {
            let ql = &ql_full[ql_off..ql_off + 64];
            let qh = &qh_full[qh_off..qh_off + 32];
            let sc = &sc_full[sc_off..sc_off + 8];
            for l in 0..32usize {
                let is = l / 16;
                let q1 = i32::from(((ql[l] & 0xF) | ((qh[l] & 3) << 4)).cast_signed()) - 32;
                let q2 =
                    i32::from(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)).cast_signed()) - 32;
                let q3 = i32::from(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)).cast_signed()) - 32;
                let q4 =
                    i32::from(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)).cast_signed()) - 32;
                let s0 = f32::from(sc[is].cast_signed());
                let s2 = f32::from(sc[is + 2].cast_signed());
                let s4 = f32::from(sc[is + 4].cast_signed());
                let s6 = f32::from(sc[is + 6].cast_signed());
                #[allow(clippy::cast_precision_loss)]
                {
                    out[base + n + l] = d_val * s0 * q1 as f32;
                    out[base + n + l + 32] = d_val * s2 * q2 as f32;
                    out[base + n + l + 64] = d_val * s4 * q3 as f32;
                    out[base + n + l + 96] = d_val * s6 * q4 as f32;
                }
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    #[test]
    fn f32_passthrough() {
        let bytes: Vec<u8> = 1.5f32.to_le_bytes().to_vec();
        let out = dequant_to_f32(&bytes, GgmlType::F32).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.5).abs() < 1e-7);
    }

    #[test]
    fn f16_conversion() {
        let bits = f16_bytes(0.5);
        let out = dequant_to_f32(&bits, GgmlType::F16).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn unsupported_type_returns_error() {
        assert!(dequant_to_f32(&[], GgmlType::Q2_K).is_err());
    }

    fn make_q8_0_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(34);
        b.extend_from_slice(&f16_bytes(d));
        b.extend(qs.iter().map(|&q| q.cast_unsigned()));
        b
    }

    #[test]
    fn q8_0_single_block() {
        // d=1.0; qs=[1, -1, 0×30] → out=[1.0, -1.0, 0×30]
        let mut qs = [0i8; 32];
        qs[0] = 1;
        qs[1] = -1;
        let block = make_q8_0_block(1.0, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q8_0).unwrap();
        assert_eq!(out.len(), 32);
        assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - (-1.0)).abs() < 1e-5, "out[1]={}", out[1]);
        assert!(out[2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_scale_applied() {
        // d=2.0; qs=[3, 0×31] → out[0]=6.0
        let mut qs = [0i8; 32];
        qs[0] = 3;
        let block = make_q8_0_block(2.0, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q8_0).unwrap();
        assert!((out[0] - 6.0).abs() < 1e-4, "out[0]={}", out[0]);
    }

    #[test]
    fn q8_0_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 33], GgmlType::Q8_0).is_err());
    }

    fn make_q4_0_block(d: f32, qs: &[u8; 16]) -> Vec<u8> {
        let mut b = Vec::with_capacity(18);
        b.extend_from_slice(&f16_bytes(d));
        b.extend_from_slice(qs);
        b
    }

    #[test]
    fn q4_0_single_block() {
        // d=1.0; qs[0]=0x89 → lower=9,x0=1 → out[0]=1.0; upper=8,x1=0; resto 0x88→zeros
        let mut qs = [0x88u8; 16];
        qs[0] = 0x89;
        let block = make_q4_0_block(1.0, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q4_0).unwrap();
        assert_eq!(out.len(), 32);
        assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!(
            out[1..16].iter().all(|&v| v == 0.0),
            "out[1..16] deve ser zero"
        );
        assert!(
            out[16..].iter().all(|&v| v == 0.0),
            "out[16..] deve ser zero"
        );
    }

    #[test]
    fn q4_0_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 17], GgmlType::Q4_0).is_err());
    }

    fn make_q4_k_block(d: f32, dmin: f32, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(144);
        b.extend_from_slice(&f16_bytes(d));
        b.extend_from_slice(&f16_bytes(dmin));
        b.extend_from_slice(scales);
        b.extend_from_slice(qs);
        b
    }

    #[test]
    fn q4_k_two_active_sub_blocks() {
        // d=1.0, dmin=1.0
        // scales=[8, 4, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]
        //   is=0: get(0)→sc=8,mn=0; get(1)→sc=4,mn=0
        //   is=2...: →sc=0,mn=0 (all zero)
        // qs = [0x22 × 128]: nibbles both = 2
        // out[0..32]   = 8*2-0 = 16.0
        // out[32..64]  = 4*2-0 = 8.0
        // out[64..256] = 0.0
        let scales: [u8; 12] = [8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let qs = [0x22u8; 128];
        let block = make_q4_k_block(1.0, 1.0, &scales, &qs);
        let out = dequant_to_f32(&block, GgmlType::Q4_K).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            let expected = if i < 32 {
                16.0f32
            } else if i < 64 {
                8.0
            } else {
                0.0
            };
            assert!(
                (v - expected).abs() < 1e-4,
                "out[{i}]={v} esperado={expected}"
            );
        }
    }

    #[test]
    fn q4_k_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 143], GgmlType::Q4_K).is_err());
    }

    fn make_q6_k_block(d: f32, scales: &[i8; 16], ql: &[u8; 128], qh: &[u8; 64]) -> Vec<u8> {
        let mut b = Vec::with_capacity(210);
        b.extend_from_slice(ql);
        b.extend_from_slice(qh);
        b.extend(scales.iter().map(|&s| s.cast_unsigned()));
        b.extend_from_slice(&f16_bytes(d));
        b
    }

    #[test]
    fn q6_k_all_zero_ql_qh() {
        // ql=zeros, qh=zeros → all 6-bit quants = 0-32 = -32
        // d=1.0, scales=[1,0,1,0,1,0,1,0, 1,0,1,0,1,0,1,0]
        // For l=0..15 (is=0): out=sc[0]*(-32)=-32; sc[2]*(-32)=-32; sc[4]*(-32)=-32; sc[6]*(-32)=-32
        // For l=16..31(is=1): out=sc[1]*(-32)=0; sc[3]*(-32)=0; sc[5]*(-32)=0; sc[7]*(-32)=0
        let scales: [i8; 16] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let ql = [0u8; 128];
        let qh = [0u8; 64];
        let block = make_q6_k_block(1.0, &scales, &ql, &qh);
        let out = dequant_to_f32(&block, GgmlType::Q6_K).unwrap();
        assert_eq!(out.len(), 256);
        for chunk_base in [0usize, 128] {
            for l in 0..16usize {
                assert!(
                    (out[chunk_base + l] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=0"
                );
                assert!(
                    (out[chunk_base + l + 32] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=32"
                );
                assert!(
                    (out[chunk_base + l + 64] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=64"
                );
                assert!(
                    (out[chunk_base + l + 96] - (-32.0)).abs() < 1e-4,
                    "chunk={chunk_base} l={l} off=96"
                );
            }
            for l in 16..32usize {
                assert_eq!(
                    out[chunk_base + l],
                    0.0,
                    "chunk={chunk_base} l={l} off=0 deve=0"
                );
                assert_eq!(
                    out[chunk_base + l + 32],
                    0.0,
                    "chunk={chunk_base} l={l} off=32 deve=0"
                );
                assert_eq!(
                    out[chunk_base + l + 64],
                    0.0,
                    "chunk={chunk_base} l={l} off=64 deve=0"
                );
                assert_eq!(
                    out[chunk_base + l + 96],
                    0.0,
                    "chunk={chunk_base} l={l} off=96 deve=0"
                );
            }
        }
    }

    #[test]
    fn q6_k_bad_size_returns_error() {
        assert!(dequant_to_f32(&[0u8; 209], GgmlType::Q6_K).is_err());
    }
}

#[cfg(test)]
mod q5k_tests {
    use super::*;

    // Primeiro superbloco de `token_embd.weight` do Qwen2.5-32B-Instruct-Q5_K_M, e os
    // valores correspondentes produzidos pelo `gguf-py` do llama.cpp — o oráculo do
    // formato. Fixture pequeno de propósito: pega qualquer erro de manipulação de bits
    // (nibble trocado, bit alto no grupo errado, escala mal desempacotada) sem depender
    // do modelo estar presente na máquina.
    const RAW: [u8; 176] = [
        60, 6, 128, 24, 217, 160, 160, 90, 213, 88, 91, 81, 255, 251, 235, 6, 222, 227, 191, 124,
        150, 29, 213, 59, 219, 255, 241, 30, 63, 145, 57, 124, 125, 249, 241, 221, 55, 146, 83,
        162, 124, 217, 252, 247, 247, 222, 117, 127, 55, 187, 228, 65, 208, 180, 35, 129, 127, 108,
        196, 99, 69, 252, 126, 159, 119, 251, 254, 205, 146, 62, 9, 159, 125, 184, 201, 15, 42, 8,
        10, 17, 214, 240, 97, 144, 166, 101, 29, 95, 52, 147, 233, 198, 44, 143, 174, 246, 7, 131,
        232, 20, 231, 252, 88, 238, 184, 91, 170, 8, 236, 187, 169, 30, 231, 32, 16, 75, 214, 171,
        246, 8, 230, 5, 204, 181, 104, 109, 93, 170, 55, 57, 28, 217, 103, 218, 220, 78, 37, 248,
        31, 99, 249, 10, 40, 33, 195, 21, 143, 105, 11, 176, 65, 126, 82, 68, 105, 177, 190, 92,
        217, 240, 193, 134, 65, 251, 28, 47, 3, 13, 216, 38, 161, 1, 31, 2, 149, 170,
    ];

    const SPOTS: [(usize, f32); 24] = [
        (0, -2.9495001e-02),
        (1, 1.8069506e-02),
        (2, 1.4219284e-03),
        (3, -4.3764353e-02),
        (4, -4.6142578e-02),
        (5, 1.4219284e-03),
        (6, -9.5629692e-04),
        (7, -5.7127476e-03),
        (124, -2.7265549e-03),
        (125, 2.9427052e-02),
        (126, -1.2619972e-02),
        (127, 4.6935081e-03),
        (128, -5.8579445e-04),
        (129, -1.3842773e-01),
        (130, -4.2537689e-02),
        (131, 2.3386717e-02),
        (248, -7.9493523e-03),
        (249, 2.5148392e-03),
        (250, 1.9257545e-02),
        (251, -1.6708374e-03),
        (252, 4.2200088e-04),
        (253, -1.6708374e-03),
        (254, -1.6320705e-02),
        (255, -1.4227867e-02),
    ];

    /// Soma dos 256 valores, para detectar erro em posições fora das amostradas.
    const SOMA: f64 = 1.0147691e-01;

    #[test]
    fn q5_k_bate_com_o_oraculo_do_llama_cpp() {
        let got = dequant_to_f32(&RAW, GgmlType::Q5_K).expect("dequant Q5_K");
        assert_eq!(got.len(), 256, "um superbloco = 256 elementos");

        for (i, esperado) in SPOTS {
            let d = (got[i] - esperado).abs();
            assert!(
                d < 1e-6,
                "indice {i}: obtido {}, esperado {esperado}",
                got[i]
            );
        }

        let soma: f64 = got.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (soma - SOMA).abs() < 1e-5,
            "soma dos 256 valores: obtida {soma:.9e}, esperada {SOMA:.9e}"
        );
    }

    #[test]
    fn q5_k_rejeita_tamanho_invalido() {
        assert!(dequant_to_f32(&RAW[..175], GgmlType::Q5_K).is_err());
    }
}
