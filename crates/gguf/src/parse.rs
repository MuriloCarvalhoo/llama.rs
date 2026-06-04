//! Parsing do header, dos pares de metadados e dos tensor infos.

use std::collections::BTreeMap;

use crate::error::GgufError;
use crate::reader::Reader;
use crate::types::{GgmlType, MetadataArray, MetadataValue};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSION: u32 = 3;

/// Lê um valor de metadado a partir do `value_type` GGUF.
pub(crate) fn read_value(r: &mut Reader, value_type: u32) -> Result<MetadataValue, GgufError> {
    Ok(match value_type {
        0 => MetadataValue::U8(r.u8()?),
        1 => MetadataValue::I8(r.i8()?),
        2 => MetadataValue::U16(r.u16()?),
        3 => MetadataValue::I16(r.i16()?),
        4 => MetadataValue::U32(r.u32()?),
        5 => MetadataValue::I32(r.i32()?),
        6 => MetadataValue::F32(r.f32()?),
        7 => MetadataValue::Bool(r.bool()?),
        8 => MetadataValue::String(r.gguf_string()?),
        9 => MetadataValue::Array(read_array(r)?),
        10 => MetadataValue::U64(r.u64()?),
        11 => MetadataValue::I64(r.i64()?),
        12 => MetadataValue::F64(r.f64()?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

fn read_array(r: &mut Reader) -> Result<MetadataArray, GgufError> {
    let elem_type = r.u32()?;
    let count = usize::try_from(r.u64()?).map_err(|_| GgufError::Overflow)?;
    // NÃO pré-alocar `count` (pode ser malicioso); push incremental — se os
    // bytes acabarem, o `read` falha e o loop aborta com erro.
    macro_rules! collect {
        ($variant:ident, $method:ident) => {{
            let mut v = Vec::new();
            for _ in 0..count {
                v.push(r.$method()?);
            }
            MetadataArray::$variant(v)
        }};
    }
    Ok(match elem_type {
        0 => collect!(U8, u8),
        1 => collect!(I8, i8),
        2 => collect!(U16, u16),
        3 => collect!(I16, i16),
        4 => collect!(U32, u32),
        5 => collect!(I32, i32),
        6 => collect!(F32, f32),
        7 => collect!(Bool, bool),
        8 => collect!(String, gguf_string),
        10 => collect!(U64, u64),
        11 => collect!(I64, i64),
        12 => collect!(F64, f64),
        9 => return Err(GgufError::NestedArray),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

/// Resultado intermediário do parsing (consumido por `file.rs`).
pub(crate) struct Parsed {
    pub version: u32,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<crate::file::TensorInfo>,
    pub data_offset: usize,
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Parsed, GgufError> {
    let mut r = Reader::new(bytes);

    let magic = r.array::<4>()?;
    if &magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic(magic));
    }
    let version = r.u32()?;
    if version != SUPPORTED_VERSION {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.gguf_string()?;
        let value_type = r.u32()?;
        let value = read_value(&mut r, value_type)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = r.gguf_string()?;
        let n_dims = r.u32()?;
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ggml_type = GgmlType::try_from(r.u32()?)?;
        let offset = r.u64()?;
        tensors.push(crate::file::TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    let alignment = match metadata.get("general.alignment") {
        Some(v) => {
            usize::try_from(v.as_u32("general.alignment")?).map_err(|_| GgufError::Overflow)?
        }
        None => 32,
    };
    let pos = r.position();
    let data_offset = align_up(pos, alignment)?;

    Ok(Parsed {
        version,
        metadata,
        tensors,
        data_offset,
    })
}

fn align_up(pos: usize, alignment: usize) -> Result<usize, GgufError> {
    if alignment == 0 {
        return Ok(pos);
    }
    let rem = pos % alignment;
    if rem == 0 {
        Ok(pos)
    } else {
        pos.checked_add(alignment - rem).ok_or(GgufError::Overflow)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing)]
    use crate::file::GgufFile;
    use crate::test_support::GgufBuilder;
    use crate::types::GgmlType;

    fn sample() -> Vec<u8> {
        GgufBuilder::new()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.block_count", 5)
            .kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_str_array("tokenizer.ggml.tokens", &["<unk>", "<s>", "</s>"])
            .tensor("token_embd.weight", &[64, 512], 0, 0)
            .build_meta_only()
    }

    #[test]
    fn parses_header_and_metadata() {
        let bytes = sample();
        let f = GgufFile::parse(&bytes).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(
            f.metadata
                .get("general.architecture")
                .unwrap()
                .as_str("k")
                .unwrap(),
            "llama"
        );
        assert_eq!(
            f.metadata
                .get("llama.block_count")
                .unwrap()
                .as_u32("k")
                .unwrap(),
            5
        );
        assert_eq!(
            f.metadata.get("tokenizer.ggml.tokens").unwrap().array_len(),
            Some(3)
        );
    }

    #[test]
    fn parses_tensor_info() {
        let bytes = sample();
        let f = GgufFile::parse(&bytes).unwrap();
        assert_eq!(f.tensors.len(), 1);
        let t = &f.tensors[0];
        assert_eq!(t.name, "token_embd.weight");
        assert_eq!(t.dims, vec![64, 512]);
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.offset, 0);
    }

    #[test]
    fn bad_magic_is_error() {
        let mut bytes = sample();
        bytes[0] = b'X';
        assert!(GgufFile::parse(&bytes).is_err());
    }

    #[test]
    fn unsupported_version_is_error() {
        let mut bytes = sample();
        bytes[4] = 2; // version = 2
        assert!(GgufFile::parse(&bytes).is_err());
    }
}
