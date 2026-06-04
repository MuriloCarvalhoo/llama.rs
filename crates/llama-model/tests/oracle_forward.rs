//! Teste diferencial contra o oráculo (auto-skip se modelo/refs ausentes).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use gguf::GgufFile;
use llama_model::Model;
use llama_tokenizer::Tokenizer;

#[test]
fn greedy_generation_matches_oracle_reference() {
    let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let Ok(reference) = std::fs::read_to_string(Path::new("../../refs/greedy.txt")) else {
        eprintln!("refs/greedy.txt ausente — pulando");
        return;
    };
    let f = GgufFile::parse(&bytes).unwrap();
    let model = Model::load(&f, &bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();

    let out = model.generate_greedy(&tok, "Once upon a time", 32).unwrap();

    eprintln!("got: {out:?}");
    eprintln!("ref: {reference:?}");

    // Gate duro: igualdade exata da sequência greedy decodificada.
    // Tolera trailing newline no arquivo de referência.
    let reference_trimmed = reference.trim_end_matches('\n');
    assert_eq!(
        out, reference_trimmed,
        "\n  got: {out:?}\n  ref: {reference_trimmed:?}"
    );
}
