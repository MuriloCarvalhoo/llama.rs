//! Parseia o stories260K.gguf de verdade e confere fatos conhecidos
//! (do dump do loader do llama.cpp).
#![allow(clippy::expect_used, clippy::indexing_slicing)]
use std::path::Path;

fn load() -> Option<gguf::GgufFile> {
    // cwd nos testes de integração = raiz do crate (crates/gguf); modelo está
    // dois níveis acima.
    let path = Path::new("../../models/stories260K.gguf");
    let bytes = std::fs::read(path).ok()?;
    Some(gguf::GgufFile::parse(&bytes).expect("stories260K deve parsear"))
}

#[test]
fn stories260k_scalar_metadata() {
    let Some(f) = load() else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    assert_eq!(f.version, 3);
    assert_eq!(
        f.get("general.architecture").unwrap().as_str("k").unwrap(),
        "llama"
    );
    assert_eq!(
        f.get("tokenizer.ggml.model").unwrap().as_str("k").unwrap(),
        "llama"
    );
    assert_eq!(f.get("llama.block_count").unwrap().as_u32("k").unwrap(), 5);
    assert_eq!(
        f.get("llama.embedding_length")
            .unwrap()
            .as_u32("k")
            .unwrap(),
        64
    );
    assert_eq!(
        f.get("llama.attention.head_count")
            .unwrap()
            .as_u32("k")
            .unwrap(),
        8
    );
    assert_eq!(
        f.get("llama.attention.head_count_kv")
            .unwrap()
            .as_u32("k")
            .unwrap(),
        4
    );
    assert_eq!(
        f.get("tokenizer.ggml.bos_token_id")
            .unwrap()
            .as_u32("k")
            .unwrap(),
        1
    );
    assert_eq!(
        f.get("tokenizer.ggml.eos_token_id")
            .unwrap()
            .as_u32("k")
            .unwrap(),
        2
    );
}

#[test]
fn stories260k_arrays_and_tensors() {
    let Some(f) = load() else { return };
    assert_eq!(
        f.get("tokenizer.ggml.tokens").unwrap().array_len(),
        Some(512)
    );
    assert_eq!(
        f.get("tokenizer.ggml.scores").unwrap().array_len(),
        Some(512)
    );
    assert_eq!(
        f.get("tokenizer.ggml.token_type").unwrap().array_len(),
        Some(512)
    );
    // 48 tensores f32 (do dump do loader).
    assert_eq!(f.tensors.len(), 48);
    assert!(f.tensors.iter().all(|t| t.ggml_type == gguf::GgmlType::F32));
    // token_embd: tensor_data deve fatiar n_elements*4 bytes.
    let bytes = std::fs::read("../../models/stories260K.gguf").unwrap();
    let embd = f
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    let data = f.tensor_data(&bytes, embd).unwrap();
    let n: u64 = embd.dims.iter().product();
    assert_eq!(data.len() as u64, n * 4);
}
