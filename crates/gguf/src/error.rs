//! Erros tipados do parser GGUF.

/// Falhas ao parsear um arquivo GGUF. GGUF é entrada não-confiável:
/// toda condição inesperada vira erro, nunca panic.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GgufError {
    #[error(
        "fim inesperado dos dados: precisava de {needed} bytes em offset {offset}, restam {available}"
    )]
    UnexpectedEof {
        offset: usize,
        needed: usize,
        available: usize,
    },
    #[error("magic GGUF inválido: {0:?}")]
    BadMagic([u8; 4]),
    #[error("versão GGUF não suportada: {0} (suportado: 3)")]
    UnsupportedVersion(u32),
    #[error("string não-UTF8 nos metadados")]
    InvalidUtf8,
    #[error("tipo de valor de metadado desconhecido: {0}")]
    UnknownValueType(u32),
    #[error("array aninhado não suportado")]
    NestedArray,
    #[error("tipo de tensor ggml desconhecido: {0}")]
    UnknownTensorType(u32),
    #[error("chave de metadado ausente: {0}")]
    MissingKey(String),
    #[error("tipo de metadado incorreto para a chave {key}: esperado {expected}")]
    WrongType { key: String, expected: &'static str },
    #[error("dados do tensor fora dos limites: tensor {name}")]
    TensorOutOfBounds { name: String },
    #[error("overflow aritmético ao calcular tamanho/alinhamento")]
    Overflow,
}
