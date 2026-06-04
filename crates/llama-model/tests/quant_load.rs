//! Integração: carrega qwen2.5-0.5b-q8_0 e verifica footprint de memória.
//! Skip automático se o modelo não estiver em models/.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use gguf::GgufFile;
use llama_model::{LlamaConfig, Model};

const QWEN_PATH: &str = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";

#[test]
fn qwen_q8_0_loads_without_error() {
    let Ok(bytes) = std::fs::read(Path::new(QWEN_PATH)) else {
        eprintln!("qwen ausente — pulando");
        return;
    };
    let f = GgufFile::parse(&bytes).expect("parse GGUF");

    let cfg = LlamaConfig::from_gguf(&f).expect("qwen2 config deve carregar");
    let model = Model::load_with_config(&f, &bytes, cfg).expect("qwen2 model deve carregar");

    let file_size = bytes.len();
    let mem = model.memory_bytes();

    assert!(
        mem <= file_size,
        "memory_bytes={mem} > file_size={file_size}"
    );

    let n_elem = model.weight_element_count();
    let ratio = mem as f64 / n_elem as f64;
    assert!(
        ratio < 2.0,
        "ratio bytes/elem={ratio:.3} ≥ 2.0 — suspeita de dequant precoce"
    );

    eprintln!(
        "qwen2.5-0.5b-q8_0: file={:.1}MB mem_raw={:.1}MB elem={n_elem} ratio={ratio:.3} bytes/elem",
        file_size as f64 / 1e6,
        mem as f64 / 1e6,
    );
}

// Inferência end-to-end requer tokenizador BPE (GPT2), ainda não suportado.
// O teste de forward com token raw fica em model.rs (pub(crate)).
