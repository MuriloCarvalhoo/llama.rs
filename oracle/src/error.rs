#[derive(Debug, thiserror::Error)]
pub enum OracleError {
    #[error("falha ao executar {0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("{0} terminou com status {1}")]
    NonZero(String, i32),
    #[error("saída do oráculo não reconhecida: {0:?}")]
    Parse(String),
}
