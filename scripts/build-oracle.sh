#!/usr/bin/env bash
# Compila o llama.cpp upstream (somente leitura) out-of-tree como oráculo.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S llama.cpp -B build-oracle \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF
cmake --build build-oracle -j"$(nproc)" \
    --target llama-completion llama-tokenize llama-eval-callback
build-oracle/bin/llama-completion --version
