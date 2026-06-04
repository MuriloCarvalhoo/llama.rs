#![forbid(unsafe_code)]
//! Parser do formato GGUF v3 (little-endian) sobre slice emprestado.

mod error;
mod reader;
mod types;

#[cfg(test)]
mod test_support;

pub use error::GgufError;
pub use types::{GgmlType, MetadataArray, MetadataValue};
