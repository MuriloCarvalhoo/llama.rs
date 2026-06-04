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

fn dequant_q8_0(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    Err(DequantError::UnsupportedType(
        "Q8_0 (stub — implementado na Task 1)".to_owned(),
    ))
}
fn dequant_q4_0(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    Err(DequantError::UnsupportedType(
        "Q4_0 (stub — implementado na Task 2)".to_owned(),
    ))
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
}
