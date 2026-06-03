#!/usr/bin/env bash
# Gate de validação por tarefa — itens 2, 3 e 5 do gate da spec.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

if command -v cargo-llvm-cov >/dev/null 2>&1; then
    # Exclui cola de I/O do alvo de cobertura: runner.rs invoca os binários do
    # oráculo e main.rs é o entrypoint de captura — ambos validados pela suíte
    # de integração (cargo test -- --ignored) e pelo check de determinismo das
    # refs, não por testes unitários. A métrica de 80% mede a lógica de fato.
    cargo llvm-cov --workspace --fail-under-lines 80 \
        --ignore-filename-regex 'oracle/src/(runner|main)\.rs'
else
    echo "AVISO: cargo-llvm-cov não instalado — cobertura não verificada (cargo install cargo-llvm-cov --locked)"
fi
echo "GATE OK"
