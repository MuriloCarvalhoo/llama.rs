#![forbid(unsafe_code)]
//! Kernels CPU para dequantização de tensores GGML (safe Rust, sem SIMD por ora).

mod dequant;
mod error;

pub use dequant::dequant_to_f32;
pub use error::DequantError;
