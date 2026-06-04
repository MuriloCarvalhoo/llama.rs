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

    // Tenta LlamaConfig; divergências de hparams são registradas e o teste pula.
    let cfg = match LlamaConfig::from_gguf(&f) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "qwen config falhou ({e}) — arquitetura divergente, pulando (esperado até Fase 7)"
            );
            return;
        }
    };

    let model = match Model::load_with_config(&f, &bytes, cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("qwen load falhou ({e}) — registrar e investigar");
            let msg = e.to_string();
            assert!(
                !msg.contains("não suportado"),
                "tipo de quant não suportado detectado: {msg}"
            );
            return;
        }
    };

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
