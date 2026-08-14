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

    /// Tamanho maximo do contexto (KV-cache). Limitado ao do modelo.
    /// O KV-cache residente custa n_layer*ctx*kv_dim*2*4 bytes de VRAM — usar o
    /// context_length do GGUF (ex.: 131072 no Qwen2.5-14B) pediria dezenas de GB.
    #[arg(long, default_value_t = 4096)]
    pub ctx: usize,

    /// Suprimir o prompt da saida
    #[arg(long)]
    pub no_display_prompt: bool,

    /// Imprimir tempo de geracao (tokens/seg) ao final para stderr
    #[arg(long)]
    pub timings: bool,

    /// Habilita backend Vulkan dual-GPU (requer duas AMD MI50 e feature "gpu").
    #[arg(long, default_value = "false")]
    pub gpu: bool,

    /// Backend Vulkan single-GPU com pesos residentes (Fase 8.1A). Requer feature "gpu".
    #[arg(long, default_value = "false")]
    pub gpu_single: bool,

    /// Backend Vulkan com decode 100% na GPU (Fase 1C). Requer feature "gpu".
    #[arg(long = "gpu-resident", default_value_t = false)]
    pub gpu_resident: bool,

    /// Layer-split: divide as camadas entre as GPUs proporcionalmente à VRAM livre.
    /// Para modelos que não cabem numa GPU só. Requer feature "gpu".
    #[arg(long = "gpu-layer-split", default_value_t = false)]
    pub gpu_layer_split: bool,

    /// Grava uma timeline cronológica de CPU e GPU no arquivo dado (formato Chrome Trace,
    /// abre em https://ui.perfetto.dev). Requer feature "profiling".
    #[arg(long = "trace", value_name = "ARQUIVO")]
    pub trace: Option<PathBuf>,
}
