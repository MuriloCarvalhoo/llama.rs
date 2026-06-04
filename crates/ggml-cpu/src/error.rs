/// Erro de dequantização.
#[derive(Debug, thiserror::Error)]
pub enum DequantError {
    #[error(
        "bytes insuficientes para tipo {ty}: esperado múltiplo de {block_bytes}, recebeu {got}"
    )]
    BadSize {
        ty: &'static str,
        block_bytes: usize,
        got: usize,
    },
    #[error("tipo {0} não suportado para dequantização")]
    UnsupportedType(String),
}
