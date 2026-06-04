#![forbid(unsafe_code)]
//! Gera os artefatos de referência em refs/ a partir do oráculo C++.
//! Uso (na raiz do workspace): cargo run -p oracle
//! Env opcionais: ORACLE_BIN_DIR (default build-oracle/bin),
//!                ORACLE_MODEL  (default models/stories260K.gguf)

use std::fs;

use oracle::Oracle;

const PROMPT: &str = "Once upon a time";
const N_TOKENS: u32 = 32;
const CORPUS: &[&str] = &[
    "Once upon a time",
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Era uma vez uma menina",
    "  leading spaces",
    "trailing spaces   ",
    "multiple    internal    spaces",
    "café résumé naïve",
    "Tab\tand\nnewline",
    "MiXeD CaSe 123!?",
    ".",
    "123456789",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bin_dir = std::env::var("ORACLE_BIN_DIR").unwrap_or_else(|_| "build-oracle/bin".to_owned());
    let model =
        std::env::var("ORACLE_MODEL").unwrap_or_else(|_| "models/stories260K.gguf".to_owned());
    let oracle = Oracle::new(bin_dir, &model);

    fs::create_dir_all("refs")?;

    let tokens: Vec<serde_json::Value> = CORPUS
        .iter()
        .map(|text| Ok(serde_json::json!({ "text": text, "ids": oracle.tokenize(text)? })))
        .collect::<Result<_, oracle::OracleError>>()?;
    fs::write(
        "refs/tokens.json",
        serde_json::to_string_pretty(&serde_json::json!({ "model": model, "cases": tokens }))?,
    )?;

    fs::write("refs/greedy.txt", oracle.generate_greedy(PROMPT, N_TOKENS)?)?;
    fs::write("refs/tensors.txt", oracle.dump_tensors(PROMPT)?)?;

    println!("refs/ atualizadas (tokens.json, greedy.txt, tensors.txt)");
    Ok(())
}
