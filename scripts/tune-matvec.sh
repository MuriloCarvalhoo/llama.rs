#!/usr/bin/env bash
# Varre a geometria do matvec Q5_K (lanes por workgroup x linhas por wave) medindo, para
# cada par, a ocupância que o RADV alcança e o tok/s de ponta a ponta.
#
# Por que varrer: os dois parâmetros são specialization constants, então mudam a pressão de
# registrador do kernel. Na MI50 são 256 VGPRs por SIMD e as waves residentes são
# floor(256/VGPRs) — 40 VGPRs cabem 6 waves, 32 cabem 8, 25 cabem 10. Menos linhas por wave
# = menos acumuladores vivos = mais waves para esconder a latência da HBM, ao custo de reler
# a ativação. Qual dos dois vence não se deduz, se mede.
#
# O matvec Q5_K é ~72% do tempo de token no Qwen2.5-32B, então é o kernel que paga a varredura.
set -euo pipefail

MODEL="${BENCH_MODEL:-models/qwen2.5-32b-instruct-q5_k_m.gguf}"
REPS="${REPS:-3}"
N="${N:-128}"
export LLAMA_RS_SPLIT="${LLAMA_RS_SPLIT:-31}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

printf '%-10s %6s %6s %8s %8s\n' geom VGPRs waves mediana melhor

for geom in 64,1 64,2 128,1 128,2 256,1 256,2 256,3 256,4 512,1 512,2 1024,1; do
    export LLAMA_RS_MATVEC_GEOM="$geom"

    # Ocupância: o RADV imprime as stats na ordem em que as pipelines são criadas, e o
    # matvec Q5_K é a terceira (matvec q8_0, quantize_x, q5k, ...). Só imprime quando o
    # cache de shaders está desligado.
    MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaderstats \
        ./target/release/llama-cli -m "$MODEL" -p Oi -n 1 --temp 0 --ctx 512 \
        --no-display-prompt --gpu-layer-split >"$TMP/stats" 2>&1 || true
    read -r vgpr waves < <(awk '
        /SHADER STATS/ { n++ }
        n == 3 && /^VGPRs:/ { v = $2 }
        n == 3 && /Subgroups per SIMD:/ { w = $4; exit }
        END { print v+0, w+0 }' "$TMP/stats")

    vals=()
    for _ in $(seq "$REPS"); do
        vals+=("$(./target/release/llama-cli -m "$MODEL" -p "Explique o que e um transformador em uma frase." \
            -n "$N" --temp 0 --seed 1 --ctx 4096 --no-display-prompt --timings \
            --gpu-layer-split 2>&1 | grep -oP '[0-9.]+(?= tok/s)')")
    done
    read -r med best < <(printf '%s\n' "${vals[@]}" | sort -n | awk -v r="$REPS" '
        { v[NR] = $1 }
        END { printf "%.2f %.2f\n", (r % 2) ? v[(r+1)/2] : (v[r/2]+v[r/2+1])/2, v[r] }')

    printf '%-10s %6s %6s %8s %8s\n' "$geom" "$vgpr" "$waves" "$med" "$best"
done
