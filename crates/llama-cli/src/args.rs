//! Argumentos de linha de comando para `llama-cli`.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "llama-cli", about = "Inferencia de LLMs em Rust (llama-rs)")]
pub struct Args {
    /// Caminho para o modelo GGUF
    #[arg(short, long)]
    pub model: PathBuf,

    /// Texto de prompt
    #[arg(short, long)]
    pub prompt: String,

    /// Número máximo de tokens a gerar
    #[arg(short = 'n', long, default_value = "128")]
    pub n_tokens: usize,

    /// Estratégia de amostragem: greedy, temperature, topk, topp
    #[arg(long, default_value = "greedy")]
    pub sampler: String,

    /// Temperatura (usada por temperature, topk, topp)
    #[arg(long, default_value = "0.8")]
    pub temp: f32,

    /// K para top-k sampling
    #[arg(long, default_value = "40")]
    pub top_k: usize,

    /// P para top-p (nucleus) sampling
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Semente RNG (para reprodutibilidade)
    #[arg(long, default_value = "42")]
    pub seed: u64,
}
