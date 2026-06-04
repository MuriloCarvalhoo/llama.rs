#![forbid(unsafe_code)]
//! Parser do formato GGUF v3 (little-endian) sobre slice emprestado.

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
