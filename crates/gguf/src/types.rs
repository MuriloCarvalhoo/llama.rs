//! Tipos de tensor ggml e valores de metadados GGUF.

use crate::error::GgufError;

/// Tipo de dado de um tensor (subconjunto ativo de `enum ggml_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    F64,
}

impl GgmlType {
    /// Elementos por bloco.
    pub fn block_size(self) -> u64 {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1 => 32,
            GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K => 256,
        }
    }

    /// Bytes por bloco.
    pub fn type_size(self) -> u64 {
        match self {
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 | GgmlType::Bf16 | GgmlType::I16 => 2,
            GgmlType::I8 => 1,
            GgmlType::I64 | GgmlType::F64 => 8,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q8_1 => 36,
            GgmlType::Q2_K => 84,
            GgmlType::Q3_K => 110,
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            GgmlType::Q8_K => 292,
        }
    }
}

impl TryFrom<u32> for GgmlType {
    type Error = GgufError;
    fn try_from(id: u32) -> Result<Self, Self::Error> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            30 => GgmlType::Bf16,
            other => return Err(GgufError::UnknownTensorType(other)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_type_ids_map() {
        assert_eq!(GgmlType::try_from(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::try_from(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::try_from(14).unwrap(), GgmlType::Q6_K);
    }

    #[test]
    fn unknown_type_id_is_error() {
        assert!(GgmlType::try_from(9999).is_err());
    }

    #[test]
    fn block_layout_matches_ggml() {
        assert_eq!((GgmlType::F32.block_size(), GgmlType::F32.type_size()), (1, 4));
        assert_eq!((GgmlType::F16.block_size(), GgmlType::F16.type_size()), (1, 2));
        assert_eq!((GgmlType::Q8_0.block_size(), GgmlType::Q8_0.type_size()), (32, 34));
        assert_eq!((GgmlType::Q4_0.block_size(), GgmlType::Q4_0.type_size()), (32, 18));
        assert_eq!((GgmlType::Q4_K.block_size(), GgmlType::Q4_K.type_size()), (256, 144));
        assert_eq!((GgmlType::Q6_K.block_size(), GgmlType::Q6_K.type_size()), (256, 210));
    }
}
