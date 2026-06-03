#![forbid(unsafe_code)]
//! Harness diferencial: executa o llama.cpp C++ (oráculo) e captura
//! tokens, texto greedy e dumps de tensors como referência.

mod error;
mod parse;

pub use error::OracleError;
pub use parse::parse_token_ids;
