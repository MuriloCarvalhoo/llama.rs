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
fn dequant_q4_k(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    Err(DequantError::UnsupportedType(
        "Q4_K (stub — implementado na Task 3)".to_owned(),
    ))
}
fn dequant_q6_k(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    Err(DequantError::UnsupportedType(
        "Q6_K (stub — implementado na Task 4)".to_owned(),
    ))
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
}
