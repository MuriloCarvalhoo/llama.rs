//! Erros do carregamento e da inferência do modelo Llama.

use ggml_cpu::DequantError;
use gguf::GgufError;
use llama_tokenizer::TokenizerError;

/// Falhas ao carregar config/pesos ou ao executar o forward.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("gguf: {0}")]
    Gguf(#[from] GgufError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("tensor ausente: {0}")]
    MissingTensor(String),
    #[error("dequantização: {0}")]
    Dequant(#[from] DequantError),
    #[error("config inconsistente: {0}")]
    Config(String),
    #[error("overflow de conversão numérica")]
    Overflow,
    #[error("contexto esgotado: posição {0} excede ctx={1}")]
    ContextOverflow(usize, usize),
    #[error("batch e caches com tamanhos diferentes: {0} vs {1}")]
    BatchMismatch(usize, usize),
}
