//! Critério de aceite da fase: encode bit-exact vs o corpus do oráculo.
use serde_json::Value;

#[test]
fn encode_matches_oracle_corpus() {
    let model_bytes = match std::fs::read("../../models/stories260K.gguf") {
        Ok(b) => b,
        Err(_) => {
            eprintln!("modelo ausente — pulando");
            return;
        }
    };
    let f = gguf::GgufFile::parse(&model_bytes).unwrap();
    let tok = llama_tokenizer::Tokenizer::from_gguf(&f).unwrap();

    let corpus: Value =
        serde_json::from_str(&std::fs::read_to_string("../../refs/tokens.json").unwrap()).unwrap();

    let mut failures = Vec::new();
    for case in corpus["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> =
            case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text, true);
        if got != expected {
            failures.push(format!("text={text:?}\n  esperado={expected:?}\n  obtido  ={got:?}"));
        }
    }
    assert!(failures.is_empty(), "divergências:\n{}", failures.join("\n"));
}
