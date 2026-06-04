#![forbid(unsafe_code)]
//! Tokenizer SPM (Llama) — encode/decode bit-exact vs llama.cpp.

mod error;
mod vocab;

pub use error::TokenizerError;
pub use vocab::Vocab;
