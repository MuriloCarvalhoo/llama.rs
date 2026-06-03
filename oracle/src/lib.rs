#![forbid(unsafe_code)]
//! Harness diferencial: executa o llama.cpp C++ (oráculo) e captura
//! tokens, texto greedy e dumps de tensors como referência.

mod error;
mod parse;
mod runner;

pub use error::OracleError;
pub use parse::parse_token_ids;
pub use runner::Oracle;
