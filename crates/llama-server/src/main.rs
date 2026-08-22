//! Servidor OpenAI-compatível para o backend residente.
//!
//! Uma requisição por vez, de propósito: os pesos são residentes e o decode ocupa as
//! duas GPUs inteiras: paralelizar requisições só disputaria a mesma banda de memória.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "llama-server",
    about = "Servidor de chat (API OpenAI) do llama-rs"
)]
struct Args {
    /// Caminho para o modelo GGUF
    #[arg(short, long)]
    model: PathBuf,

    /// Endereço de escuta
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Teto do contexto (KV-cache). Custa VRAM: ~136 KB por token no Qwen3.8-27B.
    #[arg(long, default_value_t = 32768)]
    ctx: usize,

    /// Nome do modelo exposto na API (o padrão é o nome do arquivo).
    #[arg(long)]
    nome: Option<String>,

    /// Divide as camadas entre as GPUs. Necessário para o que não cabe numa placa.
    #[arg(long = "gpu-layer-split", default_value_t = false)]
    gpu_layer_split: bool,

    /// Constrói o backend com a cabeça de multi-token prediction (`nextn`) e o plano de
    /// verify. Desligado por padrão: custa ~155 MB de snapshot mais 289 MB do bloco.
    ///
    /// O laço do `motor.rs` ainda decodifica um token por passo — quem o converte para
    /// propor→verificar é a frente que mexe em `Sessao`/TTFT.
    #[arg(long = "mtp", default_value_t = false)]
    mtp: bool,
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    match servir(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn servir(_args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    Err("compile com --features gpu: o servidor precisa do backend Vulkan".into())
}

#[cfg(feature = "gpu")]
fn servir(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use llama_server::motor::Motor;
    use llama_server::servidor::laco;
    use llama_vulkan::{LayerSplitForward, ResidentForward, VulkanContext};

    // SAFETY: mapeamento read-only de um arquivo tratado como imutável enquanto o
    // servidor roda — mesma premissa do mmap do llama.cpp e do llama-cli.
    #[allow(unsafe_code)]
    let bytes = unsafe { memmap2::Mmap::map(&std::fs::File::open(&args.model)?) }?;
    let f = gguf::GgufFile::parse(&bytes)?;
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f)?;
    cfg.ctx = cfg.ctx.min(args.ctx);
    let tokenizer = llama_tokenizer::Tokenizer::from_gguf(&f)?;
    let nome = args.nome.clone().unwrap_or_else(|| {
        args.model
            .file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_else(|| "modelo".to_owned())
    });

    eprintln!("[carga] lendo pesos de {}", args.model.display());
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg)?;
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg)?;

    let vk = VulkanContext::new()?;
    let n_gpus = vk.amd_compute_devices().len();
    if n_gpus == 0 {
        return Err("nenhuma GPU AMD encontrada".into());
    }

    let mtp = args.mtp && raw.mtp.is_some();
    if args.mtp && !mtp {
        eprintln!("[mtp] o modelo não traz bloco nextn — seguindo sem MTP");
    }

    if args.gpu_layer_split && n_gpus >= 2 {
        let backend =
            LayerSplitForward::new_com(&vk, &cfg, &raw, &aux, mtp).map_err(|e| e.to_string())?;
        let layout: Vec<String> = backend
            .layout()
            .iter()
            .map(|(d, a, b)| format!("GPU{d}: camadas {a}..{b}"))
            .collect();
        eprintln!("[gpu] layer-split — {}", layout.join(" | "));
        laco(
            &args.bind,
            &nome,
            Motor::novo(&tokenizer, &backend, cfg.ctx, cfg.eos_id),
        )
    } else {
        let backend =
            ResidentForward::new_com(&vk, &cfg, &raw, &aux, mtp).map_err(|e| e.to_string())?;
        eprintln!("[gpu] residente numa placa");
        laco(
            &args.bind,
            &nome,
            Motor::novo(&tokenizer, &backend, cfg.ctx, cfg.eos_id),
        )
    }
}
