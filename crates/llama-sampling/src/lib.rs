#![forbid(unsafe_code)]
//! Estratégias de amostragem para inferência de LLMs.

mod sampler;
pub use sampler::Sampler;
