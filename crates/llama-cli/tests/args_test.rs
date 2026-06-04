//! Testes de parsing de argumentos.
#![allow(clippy::unwrap_used)]

use clap::Parser;
use llama_cli::args::Args;

#[test]
fn parses_required_args() {
    let args = Args::try_parse_from([
        "llama-cli",
        "--model",
        "/tmp/model.gguf",
        "--prompt",
        "hello world",
    ])
    .unwrap();
    assert_eq!(args.model.to_str().unwrap(), "/tmp/model.gguf");
    assert_eq!(args.prompt, "hello world");
    assert_eq!(args.n_tokens, 128);
    assert_eq!(args.sampler, "greedy");
}

#[test]
fn parses_all_args() {
    let args = Args::try_parse_from([
        "llama-cli",
        "--model",
        "/tmp/model.gguf",
        "--prompt",
        "test",
        "-n",
        "32",
        "--sampler",
        "temperature",
        "--temp",
        "0.7",
        "--top-k",
        "20",
        "--top-p",
        "0.95",
        "--seed",
        "123",
    ])
    .unwrap();
    assert_eq!(args.n_tokens, 32);
    assert_eq!(args.sampler, "temperature");
    assert!((args.temp - 0.7).abs() < 1e-6);
    assert_eq!(args.top_k, 20);
    assert!((args.top_p - 0.95).abs() < 1e-6);
    assert_eq!(args.seed, 123);
}

#[test]
fn rejects_missing_model() {
    let result = Args::try_parse_from(["llama-cli", "--prompt", "hello"]);
    assert!(result.is_err(), "should fail without --model");
}
