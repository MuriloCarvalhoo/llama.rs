#![forbid(unsafe_code)]
//! Inferência forward (CPU, f32) da arquitetura Llama. Escopo: stories260K.

mod attention;
mod config;
mod error;
mod generate;
mod model;
mod ops;
mod weights;

pub use config::LlamaConfig;
pub use error::ModelError;
