//! Gate diferencial: saida greedy do llama-cli deve ser identica ao oraculo C++.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use llama_cli::{args::Args, generate_text, run_generate};

const MODEL: &str = "../../models/stories260K.gguf";
const REFS: &str = "../../refs/greedy.txt";
const PROMPT: &str = "Once upon a time";

#[test]
fn greedy_matches_oracle_reference() {
    if !Path::new(MODEL).exists() {
        eprintln!("modelo ausente -- pulando");
        return;
    }
    let Ok(reference) = std::fs::read_to_string(REFS) else {
        eprintln!("refs/greedy.txt ausente -- pulando");
        return;
    };

    let args = Args {
        model: MODEL.into(),
        prompt: PROMPT.to_owned(),
        n_predict: 32,
        seed: 42,
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        no_display_prompt: true,
        timings: false,
    };

    let output = generate_text(&args).expect("generate_text falhou");
    let reference_trimmed = reference.trim_end_matches('\n');

    eprintln!("got: {output:?}");
    eprintln!("ref: {reference_trimmed:?}");

    assert_eq!(
        output, reference_trimmed,
        "\n  got: {output:?}\n  ref: {reference_trimmed:?}"
    );
}

#[test]
fn topp_sampler_does_not_panic() {
    if !Path::new(MODEL).exists() {
        return;
    }
    let args = Args {
        model: MODEL.into(),
        prompt: "Once".to_owned(),
        n_predict: 2,
        seed: 1,
        temp: 0.5,
        top_k: 0,
        top_p: 0.8,
        no_display_prompt: true,
        timings: false,
    };
    generate_text(&args).expect("nao deve falhar com TopP");
}

#[test]
fn run_generate_streaming_does_not_panic() {
    if !Path::new(MODEL).exists() {
        eprintln!("modelo ausente — pulando");
        return;
    }
    let args = Args {
        model: MODEL.into(),
        prompt: PROMPT.to_owned(),
        n_predict: 4,
        seed: 42,
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        no_display_prompt: true,
        timings: false,
    };
    let mut pieces: Vec<String> = Vec::new();
    run_generate(&args, &mut |p| pieces.push(p.to_owned())).expect("run_generate nao deve falhar");
    assert!(!pieces.is_empty(), "esperado pelo menos um token");
}
