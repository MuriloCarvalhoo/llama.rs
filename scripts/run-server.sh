#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODELO="${1:-models/Qwen3.8-27B-Q4_K_M.gguf}"
CTX="${LLAMA_RS_CTX:-32768}"
BIND="${LLAMA_RS_BIND:-127.0.0.1:8080}"
NOME="${LLAMA_RS_NOME:-qwen3.8-27b}"

cargo build --release -p llama-server --features gpu

numa=()
if command -v numactl >/dev/null; then
    numa=(numactl --interleave=all)
else
    echo "AVISO: numactl não encontrado — sem ele o nó 0 estoura e a máquina trava (pacote: numactl)." >&2
fi

exec ${numa[@]+"${numa[@]}"} ./target/release/llama-server \
    -m "$MODELO" --gpu-layer-split --ctx "$CTX" --bind "$BIND" --nome "$NOME"
