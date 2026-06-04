//! Estrutura de alto nível do arquivo GGUF parseado.

use std::collections::BTreeMap;

use crate::error::GgufError;
use crate::types::{GgmlType, MetadataValue};

/// Descritor de um tensor (sem os dados).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relativo ao início da seção de dados.
    pub offset: u64,
}

/// Arquivo GGUF parseado: metadados + tensor infos. Não retém os bytes.
#[derive(Clone, Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub(crate) data_offset: usize,
}

impl GgufFile {
    /// Parseia metadados + tensor infos do conteúdo completo do arquivo.
    pub fn parse(bytes: &[u8]) -> Result<GgufFile, GgufError> {
        let p = crate::parse::parse(bytes)?;
        Ok(GgufFile {
            version: p.version,
            metadata: p.metadata,
            tensors: p.tensors,
            data_offset: p.data_offset,
        })
    }

    /// Atalho tipado para um KV obrigatório.
    pub fn get(&self, key: &str) -> Result<&MetadataValue, GgufError> {
        self.metadata.get(key).ok_or_else(|| GgufError::MissingKey(key.into()))
    }

    /// Bytes raw de um tensor (sem dequant). Slice sobre a seção de dados.
    pub fn tensor_data<'a>(
        &self,
        bytes: &'a [u8],
        t: &TensorInfo,
    ) -> Result<&'a [u8], GgufError> {
        let mut n_elements: u64 = 1;
        for &d in &t.dims {
            n_elements = n_elements.checked_mul(d).ok_or(GgufError::Overflow)?;
        }
        let block_size = t.ggml_type.block_size();
        let type_size = t.ggml_type.type_size();
        // n_elements deve ser múltiplo do block_size.
        if block_size == 0 || n_elements % block_size != 0 {
            return Err(GgufError::TensorOutOfBounds { name: t.name.clone() });
        }
        let n_blocks = n_elements / block_size;
        let n_bytes = n_blocks.checked_mul(type_size).ok_or(GgufError::Overflow)?;

        let start = self
            .data_offset
            .checked_add(usize::try_from(t.offset).map_err(|_| GgufError::Overflow)?)
            .ok_or(GgufError::Overflow)?;
        let n_bytes = usize::try_from(n_bytes).map_err(|_| GgufError::Overflow)?;
        let end = start.checked_add(n_bytes).ok_or(GgufError::Overflow)?;

        bytes
            .get(start..end)
            .ok_or_else(|| GgufError::TensorOutOfBounds { name: t.name.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::GgufBuilder;

    #[test]
    fn tensor_data_slices_correctly() {
        // 1 tensor F32 com 4 elementos = 16 bytes, offset 0, alignment 32.
        let data: Vec<u8> = (0..16u8).collect();
        let bytes = GgufBuilder::new()
            .tensor("t", &[4], 0, 0)
            .build_with_data(32, &data);
        let f = GgufFile::parse(&bytes).unwrap();
        let slice = f.tensor_data(&bytes, &f.tensors[0]).unwrap();
        assert_eq!(slice, &data[..]);
    }

    #[test]
    fn tensor_data_out_of_bounds_is_error() {
        let bytes = GgufBuilder::new()
            .tensor("t", &[4], 0, 0)
            .build_with_data(32, &[0u8; 4]); // só 4 bytes, precisa de 16
        let f = GgufFile::parse(&bytes).unwrap();
        assert!(f.tensor_data(&bytes, &f.tensors[0]).is_err());
    }

    #[test]
    fn tensor_data_q8_0_size() {
        // Q8_0 (id 8): block_size 32, type_size 34. 32 elementos = 1 bloco = 34 bytes.
        let bytes = GgufBuilder::new()
            .tensor("q", &[32], 8, 0)
            .build_with_data(32, &[7u8; 34]);
        let f = GgufFile::parse(&bytes).unwrap();
        assert_eq!(f.tensor_data(&bytes, &f.tensors[0]).unwrap().len(), 34);
    }
}
