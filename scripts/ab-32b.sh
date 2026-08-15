#!/usr/bin/env bash
# Mede o decode do 32B em layer-split com um protocolo estável o bastante para
# comparar duas versões do código.
#
# Duas fontes de ruído que este script neutraliza:
#   - o ponto de split muda com a VRAM livre  -> LLAMA_RS_SPLIT fixa a fronteira;
#   - o DPM das GPUs demora a subir o mclk    -> tokens suficientes para diluir, e
#     mediana de N repetições em vez de uma medição só.
#
# Uso: scripts/ab-32b.sh [repeticoes] [n_tokens]
set -euo pipefail

REPS="${1:-5}"
N="${2:-256}"
MODEL="${BENCH_MODEL:-models/qwen2.5-32b-instruct-q5_k_m.gguf}"
PROMPT="Explique o que e um transformador em uma frase."
export LLAMA_RS_SPLIT="${LLAMA_RS_SPLIT:-31}"

vals=()
for _ in $(seq "$REPS"); do
    out=$(./target/release/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N" \
        --temp 0 --seed 1 --ctx 4096 --no-display-prompt --timings \
        --gpu-layer-split 2>&1 | grep -oP '[0-9.]+(?= tok/s)')
    vals+=("$out")
    echo "  $out tok/s"
done

printf '%s\n' "${vals[@]}" | sort -n | awk -v r="$REPS" '
    { v[NR] = $1 }
    END {
        med = (r % 2) ? v[(r+1)/2] : (v[r/2] + v[r/2+1]) / 2
        printf "mediana %.2f  min %.2f  max %.2f  (n=%d)\n", med, v[1], v[r], r
    }'
