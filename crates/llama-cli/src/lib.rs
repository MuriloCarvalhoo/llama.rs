#![allow(unsafe_code)]
//! Biblioteca auxiliar do `llama-cli`.
//!
//! Camada de aplicação (DDD): orquestra os contextos `gguf`, `llama-tokenizer`,
//! `llama-sampling`, `llama-model` e `llama-vulkan` para a CLI. Não é domínio em si
//! — [`args::Args`] é a entrada (comando do usuário) e `runner::run_generate` é o
//! serviço de aplicação que carrega o modelo e conduz a geração.

pub mod args;
mod runner;
#[cfg(feature = "profiling")]
pub mod trace;

pub use runner::Timing;
pub use runner::generate_text;
pub use runner::run_generate;
