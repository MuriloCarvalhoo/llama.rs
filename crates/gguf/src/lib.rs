#![forbid(unsafe_code)]
//! Parser do formato GGUF v3 (little-endian) sobre slice emprestado.

mod error;
mod reader;

pub use error::GgufError;
