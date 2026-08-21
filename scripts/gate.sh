#!/usr/bin/env bash
# Gate de validação por tarefa — itens 2, 3 e 5 do gate da spec.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
# --test-threads=1: dois testes de tests/qwen35_estado.rs carregam o modelo de
# ~19 GB e, em paralelo, o kernel mata o processo por falta de RAM (SIGKILL).
cargo test --workspace -- --test-threads=1

if command -v cargo-llvm-cov >/dev/null 2>&1; then
    # Exclui cola de I/O do alvo de cobertura: runner.rs invoca os binários do
    # oráculo e main.rs é o entrypoint de captura — ambos validados pela suíte
    # de integração (cargo test -- --ignored) e pelo check de determinismo das
    # refs, não por testes unitários. A métrica de 80% mede a lógica de fato.
    # spike.rs é `#![cfg(test)]` — benchmark de all-reduce que só roda com
    # `--ignored` em 2× MI50. Contá-lo como não coberto tiraria 6 pontos da
    # métrica sem que houvesse código de produção descoberto.
    # O `--test-threads=1` é pelo mesmo motivo do `cargo test` acima: o llvm-cov
    # roda a suíte por conta própria e não herda a flag dele.
    cargo llvm-cov --workspace --fail-under-lines 85 \
        --ignore-filename-regex 'oracle/src/(runner|main)\.rs|llama-vulkan/src/spike\.rs' \
        -- --test-threads=1
else
    echo "AVISO: cargo-llvm-cov não instalado — cobertura não verificada (cargo install cargo-llvm-cov --locked)"
fi
echo "GATE OK"
