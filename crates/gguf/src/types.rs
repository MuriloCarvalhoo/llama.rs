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

/// Valor de um par de metadados GGUF.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// Array homogêneo de metadados (sem aninhamento — llama.cpp não o produz).
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl MetadataArray {
    pub fn len(&self) -> usize {
        match self {
            MetadataArray::U8(v) => v.len(),
            MetadataArray::I8(v) => v.len(),
            MetadataArray::U16(v) => v.len(),
            MetadataArray::I16(v) => v.len(),
            MetadataArray::U32(v) => v.len(),
            MetadataArray::I32(v) => v.len(),
            MetadataArray::F32(v) => v.len(),
            MetadataArray::Bool(v) => v.len(),
            MetadataArray::String(v) => v.len(),
            MetadataArray::U64(v) => v.len(),
            MetadataArray::I64(v) => v.len(),
            MetadataArray::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl MetadataValue {
    pub fn as_u32(&self, key: &str) -> Result<u32, GgufError> {
        match self {
            MetadataValue::U32(v) => Ok(*v),
            _ => Err(GgufError::WrongType {
                key: key.into(),
                expected: "u32",
            }),
        }
    }

    pub fn as_f32(&self, key: &str) -> Result<f32, GgufError> {
        match self {
            MetadataValue::F32(v) => Ok(*v),
            _ => Err(GgufError::WrongType {
                key: key.into(),
                expected: "f32",
            }),
        }
    }

    pub fn as_str(&self, key: &str) -> Result<&str, GgufError> {
        match self {
            MetadataValue::String(s) => Ok(s.as_str()),
            _ => Err(GgufError::WrongType {
                key: key.into(),
                expected: "string",
            }),
        }
    }

    pub fn as_string_array(&self, key: &str) -> Result<&[String], GgufError> {
        match self {
            MetadataValue::Array(MetadataArray::String(v)) => Ok(v),
            _ => Err(GgufError::WrongType {
                key: key.into(),
                expected: "string[]",
            }),
        }
    }

    pub fn as_f32_array(&self, key: &str) -> Result<&[f32], GgufError> {
        match self {
            MetadataValue::Array(MetadataArray::F32(v)) => Ok(v),
            _ => Err(GgufError::WrongType {
                key: key.into(),
                expected: "f32[]",
            }),
        }
    }

    pub fn as_i32_array(&self, key: &str) -> Result<&[i32], GgufError> {
        match self {
            MetadataValue::Array(MetadataArray::I32(v)) => Ok(v),
            _ => Err(GgufError::WrongType {
                key: key.into(),
                expected: "i32[]",
            }),
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            MetadataValue::Array(a) => Some(a.len()),
            _ => None,
        }
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
    fn metadata_accessors() {
        let v = MetadataValue::U32(42);
        assert_eq!(v.as_u32("k").unwrap(), 42);
        assert!(v.as_str("k").is_err());

        let s = MetadataValue::String("llama".into());
        assert_eq!(s.as_str("k").unwrap(), "llama");

        let arr = MetadataValue::Array(MetadataArray::F32(vec![0.5, 1.0]));
        assert_eq!(arr.as_f32_array("k").unwrap(), &[0.5, 1.0]);
        assert_eq!(arr.array_len(), Some(2));
    }

    #[test]
    fn block_layout_matches_ggml() {
        assert_eq!(
            (GgmlType::F32.block_size(), GgmlType::F32.type_size()),
            (1, 4)
        );
        assert_eq!(
            (GgmlType::F16.block_size(), GgmlType::F16.type_size()),
            (1, 2)
        );
        assert_eq!(
            (GgmlType::Q8_0.block_size(), GgmlType::Q8_0.type_size()),
            (32, 34)
        );
        assert_eq!(
            (GgmlType::Q4_0.block_size(), GgmlType::Q4_0.type_size()),
            (32, 18)
        );
        assert_eq!(
            (GgmlType::Q4_K.block_size(), GgmlType::Q4_K.type_size()),
            (256, 144)
        );
        assert_eq!(
            (GgmlType::Q6_K.block_size(), GgmlType::Q6_K.type_size()),
            (256, 210)
        );
    }

    #[test]
    fn all_type_ids_roundtrip_and_sizes() {
        // (id, esperado block_size, esperado type_size) para todo o subconjunto ativo.
        let cases: &[(u32, u64, u64)] = &[
            (0, 1, 4),
            (1, 1, 2),
            (2, 32, 18),
            (3, 32, 20),
            (6, 32, 22),
            (7, 32, 24),
            (8, 32, 34),
            (9, 32, 36),
            (10, 256, 84),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (15, 256, 292),
            (24, 1, 1),
            (25, 1, 2),
            (26, 1, 4),
            (27, 1, 8),
            (28, 1, 8),
            (30, 1, 2),
        ];
        for &(id, bs, ts) in cases {
            let t = GgmlType::try_from(id).unwrap();
            assert_eq!(t.block_size(), bs, "block_size id {id}");
            assert_eq!(t.type_size(), ts, "type_size id {id}");
        }
    }

    #[test]
    fn all_metadata_accessors_and_array_len() {
        // Acessores de array tipados.
        let strs = MetadataValue::Array(MetadataArray::String(vec!["a".into(), "b".into()]));
        assert_eq!(strs.as_string_array("k").unwrap(), &["a", "b"]);
        assert!(strs.as_f32_array("k").is_err());

        let ints = MetadataValue::Array(MetadataArray::I32(vec![1, 2, 3]));
        assert_eq!(ints.as_i32_array("k").unwrap(), &[1, 2, 3]);
        assert!(ints.as_string_array("k").is_err());

        // f32 escalar + caminho de erro.
        let f = MetadataValue::F32(1.5);
        assert_eq!(f.as_f32("k").unwrap(), 1.5);
        assert!(f.as_u32("k").is_err());

        // array_len em não-array é None.
        assert_eq!(MetadataValue::U8(0).array_len(), None);

        // len / is_empty em cada variante de MetadataArray.
        assert_eq!(MetadataArray::U8(vec![1]).len(), 1);
        assert_eq!(MetadataArray::I8(vec![1]).len(), 1);
        assert_eq!(MetadataArray::U16(vec![1]).len(), 1);
        assert_eq!(MetadataArray::I16(vec![1]).len(), 1);
        assert_eq!(MetadataArray::U32(vec![1]).len(), 1);
        assert_eq!(MetadataArray::I32(vec![1]).len(), 1);
        assert_eq!(MetadataArray::F32(vec![1.0]).len(), 1);
        assert_eq!(MetadataArray::Bool(vec![true]).len(), 1);
        assert_eq!(MetadataArray::String(vec!["x".into()]).len(), 1);
        assert_eq!(MetadataArray::U64(vec![1]).len(), 1);
        assert_eq!(MetadataArray::I64(vec![1]).len(), 1);
        assert_eq!(MetadataArray::F64(vec![1.0]).len(), 1);
        assert!(MetadataArray::U8(vec![]).is_empty());
        assert!(!MetadataArray::U8(vec![1]).is_empty());
    }
}
