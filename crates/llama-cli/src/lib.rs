#![forbid(unsafe_code)]
//! Biblioteca auxiliar do `llama-cli`.

pub mod args;
mod runner;

pub use runner::Timing;
pub use runner::generate_text;
pub use runner::run_generate;
