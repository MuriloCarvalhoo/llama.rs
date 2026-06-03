#!/usr/bin/env bash
# Baixa os modelos de teste:
#  - stories260K: arch llama minúscula (usada pelo CI do próprio llama.cpp) — debug camada a camada
#  - qwen2.5-0.5b q8_0: modelo realista para validação ponta a ponta
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

STORIES_URL="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"

[ -f models/stories260K.gguf ] || curl -fL --retry 3 -o models/stories260K.gguf "$STORIES_URL"
[ -f models/qwen2.5-0.5b-instruct-q8_0.gguf ] || curl -fL --retry 3 -o models/qwen2.5-0.5b-instruct-q8_0.gguf "$QWEN_URL"
ls -lh models/
