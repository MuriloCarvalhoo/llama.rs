#!/usr/bin/env bash
# Gate de validação por tarefa — itens 2, 3 e 5 do gate da spec.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

if command -v cargo-llvm-cov >/dev/null 2>&1; then
    cargo llvm-cov --workspace --fail-under-lines 80
else
    echo "AVISO: cargo-llvm-cov não instalado — cobertura não verificada (cargo install cargo-llvm-cov --locked)"
fi
echo "GATE OK"
