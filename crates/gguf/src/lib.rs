#![forbid(unsafe_code)]
//! Parser do formato GGUF v3 (little-endian) sobre slice emprestado.
//!
//! Contexto delimitado (DDD): formato de arquivo GGUF, isolado de qualquer noção de
//! arquitetura de modelo.
//! - Aggregate root: [`GgufFile`] — metadados + tensor infos parseados de um `&[u8]`.
//! - Value objects: [`TensorInfo`], [`MetadataValue`], [`MetadataArray`], [`GgmlType`]
//!   (todos imutáveis, comparáveis por valor, sem identidade própria).
//! - Infraestrutura: `Reader` (cursor com bounds-check) e `parse` (parsing puro,
//!   privados — não fazem parte do domínio público).

mod error;
mod file;
mod parse;
mod reader;
mod types;

#[cfg(test)]
mod test_support;

pub use error::GgufError;
pub use file::{GgufFile, TensorInfo};
pub use types::{GgmlType, MetadataArray, MetadataValue};
