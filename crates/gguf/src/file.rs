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
}
