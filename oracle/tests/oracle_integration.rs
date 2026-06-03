//! Testes que exigem o oráculo compilado (scripts/build-oracle.sh)
//! e o modelo baixado (scripts/get-model.sh). Rodar com:
//!     cargo test -p oracle -- --ignored
//! cargo test usa cwd = diretório do pacote (oracle/), por isso os
//! caminhos sobem um nível até a raiz do workspace.

use oracle::Oracle;

fn oracle_under_test() -> Oracle {
    Oracle::new("../build-oracle/bin", "../models/stories260K.gguf")
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn tokenize_returns_nonempty_ids() {
    let ids = oracle_under_test().tokenize("Once upon a time").unwrap();
    assert!(!ids.is_empty());
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn tokenize_is_deterministic() {
    let o = oracle_under_test();
    assert_eq!(o.tokenize("hello").unwrap(), o.tokenize("hello").unwrap());
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn greedy_generation_is_deterministic() {
    let o = oracle_under_test();
    let a = o.generate_greedy("Once upon a time", 16).unwrap();
    let b = o.generate_greedy("Once upon a time", 16).unwrap();
    assert!(!a.is_empty());
    assert_eq!(a, b);
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn missing_binary_is_reported_as_io_error() {
    let o = Oracle::new("caminho/inexistente", "../models/stories260K.gguf");
    assert!(matches!(
        o.tokenize("x"),
        Err(oracle::OracleError::Io(_, _))
    ));
}
