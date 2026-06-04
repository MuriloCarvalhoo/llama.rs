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
    #[arg(short, long, default_value = "")]
    pub prompt: String,

    /// Numero maximo de tokens a gerar
    #[arg(short = 'n', long, default_value_t = 128)]
    pub n_predict: usize,

    /// Temperatura de amostragem (0.0 = greedy deterministico)
    #[arg(long, default_value_t = 0.8)]
    pub temp: f32,

    /// Top-K -- manter K candidatos antes de amostrar (0 = desabilitado)
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Top-P / nucleus -- prob. acumulada minima (1.0 = desabilitado)
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,

    /// Semente aleatoria para amostragem reproduzivel
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Suprimir o prompt da saida
    #[arg(long)]
    pub no_display_prompt: bool,
}
