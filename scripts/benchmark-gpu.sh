#!/usr/bin/env bash
# Benchmark GPU AMD: llama.cpp (Vulkan) vs llama-rs (--gpu).
#
# Roda SOMENTE nas 2x AMD Instinct MI50 (Vega20 / driver RADV). A NVIDIA
# Tesla K80 da máquina é EXCLUÍDA explicitamente:
#   - llama.cpp : via GGML_VK_VISIBLE_DEVICES (índices Vulkan das AMD).
#   - llama-rs  : a própria impl filtra vendor AMD (0x1002) em llama-vulkan.
# O script aborta se detectar qualquer device NVIDIA/Tesla em uso.
#
# Backend Vulkan dos dois lados — isola a QUALIDADE DA IMPLEMENTAÇÃO
# (mesma API, mesmo driver RADV), não a diferença de backend.
#
# Uso:
#   ./scripts/benchmark-gpu.sh
#   BENCH_N=128 BENCH_PROMPT="The dragon said" ./scripts/benchmark-gpu.sh
#
# Variáveis de ambiente:
#   BENCH_N         tokens a gerar (decode)        (padrão: 64)
#   BENCH_PROMPT    prompt de entrada              (padrão: "Once upon a time")
#   BENCH_REPS      repetições do llama-bench      (padrão: 3)
#   BENCH_MODEL     caminho .gguf Q8_0             (padrão: qwen2.5-0.5b q8_0)
#   VK_AMD_DEVICES  índices Vulkan das 2 AMD       (padrão: "0,1")
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ──────────────────────────────────────────────────────────────────
PROMPT="${BENCH_PROMPT:-Once upon a time}"
N_TOKENS="${BENCH_N:-64}"
SEED=42
REPS="${BENCH_REPS:-3}"
MODEL="${BENCH_MODEL:-models/qwen2.5-0.5b-instruct-q8_0.gguf}"
VK_AMD_DEVICES="${VK_AMD_DEVICES:-0,1}"     # 0,1 = as duas MI50 na ordem Vulkan
VK_AMD_FIRST="${VK_AMD_DEVICES%%,*}"        # primeira AMD (para run single-GPU)

CPP_BENCH=build-vulkan/bin/llama-bench
RS_BIN=target/release/llama-cli
RESULTS_DIR=bench-results
mkdir -p "$RESULTS_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS="$RESULTS_DIR/gpu-$STAMP.md"

# ── Pré-requisitos ──────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
    echo "ERRO: modelo '$MODEL' ausente — rode ./scripts/get-model.sh" >&2
    exit 1
fi
if [[ ! -x "$CPP_BENCH" ]]; then
    echo "ERRO: $CPP_BENCH não encontrado. Compile o llama.cpp com Vulkan:" >&2
    echo "  INC=\$PWD/.deps/spirv-headers/include" >&2
    echo "  cmake -S llama.cpp -B build-vulkan -DCMAKE_BUILD_TYPE=Release \\" >&2
    echo "        -DGGML_VULKAN=ON -DLLAMA_CURL=OFF \\" >&2
    echo "        -DCMAKE_PREFIX_PATH=\$PWD/.deps/spirv-headers \\" >&2
    echo "        -DCMAKE_CXX_FLAGS=-I\$INC -DCMAKE_C_FLAGS=-I\$INC" >&2
    echo "  cmake --build build-vulkan -j\$(nproc) --target llama-bench llama-cli" >&2
    exit 1
fi

echo "Compilando llama-cli (release, feature gpu)..." >&2
cargo build --release -p llama-cli --features gpu -q

# ── Guardrail: nenhuma NVIDIA pode ser usada ─────────────────────────────────
assert_no_nvidia() {
    local logfile=$1 engine=$2
    if grep -qiE "nvidia|tesla|cuda" "$logfile"; then
        echo "" >&2
        echo "ABORTADO: $engine tocou em device NVIDIA. Linhas suspeitas:" >&2
        grep -iE "nvidia|tesla|cuda" "$logfile" >&2
        exit 2
    fi
}

# ── llama.cpp (Vulkan) — retorna 'avg ± stddev' de geração (tg) ──────────────
# $1 = índices Vulkan das GPUs (ex.: "0" ou "0,1"); $2 = arquivo de log.
run_cpp() {
    local devs=$1 log=$2 json
    json=$(GGML_VK_VISIBLE_DEVICES="$devs" "$CPP_BENCH" \
        -m "$MODEL" -ngl 99 -p 0 -n "$N_TOKENS" -r "$REPS" -o json 2>"$log")
    assert_no_nvidia "$log" "llama.cpp (Vulkan dev=$devs)"
    # Extrai o teste de geração (n_gen>0) e formata avg ± stddev.
    python3 - "$json" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
for row in data:
    if int(row.get("n_gen", 0)) > 0:
        print(f'{float(row["avg_ts"]):.2f} ± {float(row["stddev_ts"]):.2f}')
        break
PY
}

# ── Rust (--gpu, 2x MI50) — retorna tok/s de decode ──────────────────────────
run_rs() {
    local log=$1
    "$RS_BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" \
        --temp 0 --seed "$SEED" --no-display-prompt --timings --gpu \
        2>"$log" >/dev/null || true
    assert_no_nvidia "$log" "llama-rs (--gpu)"
    grep -oE "[0-9]+\.[0-9]+ tok/s" "$log" | grep -oE "^[0-9]+\.[0-9]+" | head -1
}

# ── Rust (--gpu-single, 1x MI50 resident) — retorna tok/s de decode ──────────
run_rs_single() {
    local log=$1
    "$RS_BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" \
        --temp 0 --seed "$SEED" --no-display-prompt --timings --gpu-single \
        2>"$log" >/dev/null || true
    assert_no_nvidia "$log" "llama-rs (--gpu-single)"
    grep -oE "[0-9]+\.[0-9]+ tok/s" "$log" | grep -oE "^[0-9]+\.[0-9]+" | head -1
}

# ── Execução ─────────────────────────────────────────────────────────────────
model_name="$(basename "$MODEL")"

{
echo "# Benchmark GPU AMD — llama.cpp (Vulkan) vs llama-rs (--gpu)"
echo ""
echo "- Data     : $STAMP"
echo "- Modelo   : \`$model_name\`"
echo "- Decode   : $N_TOKENS tokens | seed $SEED | greedy (temp=0)"
echo "- GPUs AMD : índices Vulkan \`$VK_AMD_DEVICES\` (2x Instinct MI50 / RADV)"
echo "- NVIDIA   : excluída (Tesla K80 ignorada)"
echo "- llama.cpp: $(git -C llama.cpp rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""
} | tee "$RESULTS"

echo "Rodando llama.cpp 1x MI50..." >&2
cpp1=$(run_cpp "$VK_AMD_FIRST"  /tmp/bench-cpp1.err)
echo "Rodando llama.cpp 2x MI50..." >&2
cpp2=$(run_cpp "$VK_AMD_DEVICES" /tmp/bench-cpp2.err)
echo "Rodando llama-rs 2x MI50..." >&2
rs=$(run_rs /tmp/bench-rs.err)
echo "Rodando llama-rs 1x MI50 (resident)..." >&2
rs1=$(run_rs_single /tmp/bench-rs1.err)

# Device names efetivamente usados (prova de AMD-only).
cpp_dev=$(grep -oiE "Radeon[^,)]*\)" /tmp/bench-cpp2.err | head -1 || true)
rs_dev=$(grep -oE "\[GPU\].*decode na GPU" /tmp/bench-rs.err | head -1 || true)

ratio="-"
if [[ -n "$rs" && -n "$cpp2" ]]; then
    cpp2_avg="${cpp2%% *}"
    ratio=$(awk "BEGIN { if ($cpp2_avg>0) printf \"%.3fx\", $rs/$cpp2_avg; else print \"-\" }")
fi

{
echo "## Resultados (tok/s de geração — maior é melhor)"
echo ""
printf "| %-28s | %-16s |\n" "Engine / GPUs" "decode tok/s"
printf "| %-28s | %-16s |\n" "$(printf -- '-%.0s' {1..28})" "$(printf -- '-%.0s' {1..16})"
printf "| %-28s | %-16s |\n" "llama.cpp — 1x MI50" "${cpp1:-erro}"
printf "| %-28s | %-16s |\n" "llama.cpp — 2x MI50" "${cpp2:-erro}"
printf "| %-28s | %-16s |\n" "llama-rs  — 2x MI50" "${rs:-erro}"
printf "| %-28s | %-16s |\n" "llama-rs  — 1x MI50 (resident)" "${rs1:-erro}"
echo ""
echo "**Razão llama-rs / llama.cpp (2x MI50): $ratio**"
echo ""
echo "### Devices confirmados (AMD-only)"
echo "- llama.cpp: ${cpp_dev:-?}"
echo "- llama-rs : ${rs_dev:-?}"
echo ""
echo "> llama.cpp: \`llama-bench -ngl 99 -p 0 -n $N_TOKENS -r $REPS\` (offload total, só decode)."
echo "> llama-rs : prefill na CPU, decode nas 2 MI50 (row-split), token-a-token sem batch."
} | tee -a "$RESULTS"

echo "" >&2
echo "Resultados salvos em: $RESULTS" >&2
