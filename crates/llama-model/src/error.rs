//! Erros do carregamento e da inferência do modelo Llama.

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
    #[error("bytes do tensor {0} não são múltiplos de 4 (f32)")]
    NotF32(String),
    #[error("config inconsistente: {0}")]
    Config(String),
    #[error("overflow de conversão numérica")]
    Overflow,
}
