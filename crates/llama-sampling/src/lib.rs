#![forbid(unsafe_code)]
//! Estratégias de amostragem para inferência de LLMs.
//!
//! Contexto delimitado (DDD): amostragem, isolado do modelo e do tokenizer.
//! - Value object + serviço de domínio: [`Sampler`] — enum sem estado próprio;
//!   `sample()` implementa a estratégia (greedy/temperatura/top-k/top-p) como um
//!   método (pattern "strategy" via enum, idiomático em Rust — sem hierarquia de
//!   classes).

mod sampler;
pub use sampler::Sampler;
