#![deny(unsafe_code)]
//! Inferência forward (CPU, f32) da arquitetura Llama. Escopo: stories260K.
//!
//! Contexto delimitado (DDD): definição e execução do modelo, independente do
//! backend (CPU aqui; Vulkan em `llama-vulkan`, atrás dos traits `GpuMatmul` /
//! `GpuResidentDecode`).
//! - Value object: [`LlamaConfig`]/[`DeltaNetConfig`] — hiperparâmetros lidos do GGUF
//!   (`config.rs`), incluindo a distinção MTP/NextN e o mixer híbrido do Qwen3.8.
//! - Entidade/aggregate root: [`Model`] — pesos + config carregados, com identidade
//!   própria (um `Model` por arquivo carregado) e estado mutável via `KvCache`.
//! - Serviços de domínio: `attention`, `ops`, `delta_net` — funções livres e sem
//!   estado (matemática pura de RoPE/RMSNorm/SwiGLU/matmul quantizado); mantidas como
//!   funções, não classes, por serem primitivas numéricas na hot-path de prefill/GPU.
//! - Camada `gpu` (feature `gpu`): ponte para o backend Vulkan residente.

mod attention;
mod config;
pub mod delta_net;
mod error;
mod generate;
#[cfg(feature = "gpu")]
mod gpu;
mod model;
#[cfg(feature = "gpu")]
mod mtp;
mod ops;
pub(crate) mod spin_pool;
mod weights;

pub use config::{DeltaNetConfig, LlamaConfig};
pub use error::ModelError;
#[cfg(feature = "gpu")]
pub use gpu::{
    AuxLayer, GpuAuxWeights, GpuLayerRaw, GpuMatmul, GpuRawWeights, GpuResidentDecode, MixerRaw,
    MtpAux, MtpRaw, QTensor, gerar_streaming_residente,
};
pub use model::Model;
#[cfg(feature = "gpu")]
pub use mtp::MtpHead;

/// Inicializa o spin pool com `n_workers` threads em background, pinados aos `cpus` fornecidos.
pub fn init_spin_pool(n_workers: usize, cpus: Vec<usize>) {
    spin_pool::init(n_workers, cpus);
}
