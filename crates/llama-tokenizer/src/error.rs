//! Erros do tokenizer.

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("erro ao ler GGUF: {0}")]
    Gguf(#[from] gguf::GgufError),
    #[error("modelo de tokenizer não suportado: {0:?} (suportado: \"llama\"/SPM)")]
    UnsupportedModel(String),
    #[error("arrays do vocab com tamanhos inconsistentes: tokens={tokens}, scores={scores}, types={types}")]
    InconsistentVocab {
        tokens: usize,
        scores: usize,
        types: usize,
    },
}
