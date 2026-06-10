#![deny(unsafe_code)]
//! Inferência forward (CPU, f32) da arquitetura Llama. Escopo: stories260K.

mod attention;
mod config;
mod error;
mod generate;
#[cfg(feature = "gpu")]
mod gpu;
mod model;
mod ops;
pub(crate) mod spin_pool;
mod weights;

pub use config::LlamaConfig;
pub use error::ModelError;
#[cfg(feature = "gpu")]
pub use gpu::{GpuLayerRaw, GpuMatmul, GpuRawWeights};
pub use model::Model;

/// Inicializa o spin pool com `n_workers` threads em background, pinados aos `cpus` fornecidos.
pub fn init_spin_pool(n_workers: usize, cpus: Vec<usize>) {
    spin_pool::init(n_workers, cpus);
}
