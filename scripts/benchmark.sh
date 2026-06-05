#!/usr/bin/env bash
# Benchmark llama.cpp vs llama-rs em dois modelos.
#
# Uso:
#   ./scripts/benchmark.sh
#   BENCH_N=128 BENCH_PROMPT="The dragon said" ./scripts/benchmark.sh
#
# Variáveis de ambiente:
#   BENCH_N      Número de tokens a gerar (padrão: 64)
#   BENCH_PROMPT Prompt de entrada        (padrão: "Once upon a time")
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────────────
PROMPT="${BENCH_PROMPT:-Once upon a time}"
N_TOKENS="${BENCH_N:-64}"
SEED=42

MODELS=(
    "stories260K        :models/stories260K.gguf"
    "qwen2.5-0.5b-q8_0  :models/qwen2.5-0.5b-instruct-q8_0.gguf"
)

CPP_BIN=build-oracle/bin/llama-completion
RS_BIN=target/release/llama-cli
LD_PATH=build-oracle/bin

# ── Pré-requisitos ────────────────────────────────────────────────────────────
if [[ ! -x "$CPP_BIN" ]]; then
    echo "ERRO: $CPP_BIN não encontrado — execute ./scripts/build-oracle.sh primeiro."
    exit 1
fi

echo "Compilando llama-cli (release)..."
cargo build --release -p llama-cli -q

# ── Funções de medição ────────────────────────────────────────────────────────

# Retorna tok/s de geração do llama.cpp (linha "eval time")
run_cpp() {
    local model=$1
    local out
    out=$(LD_LIBRARY_PATH=$LD_PATH "$CPP_BIN" \
        -m "$model" -p "$PROMPT" -n "$N_TOKENS" \
        --temp 0 --seed "$SEED" --no-display-prompt \
        --no-warmup --perf 2>&1 || true)

    # Formato: "eval time = XX ms / YY runs ( ZZ ms per token, WW,WW tokens per second)"
    # Exclui "prompt eval time" — queremos só o throughput de geração.
    # Decimal pode ser vírgula (locale pt_BR) ou ponto.
    echo "$out" \
        | grep -v "prompt eval" \
        | grep "eval time" \
        | grep -oE "[0-9][0-9,.]+ tokens per second" \
        | grep -oE "^[0-9][0-9,.]+" \
        | tr ',' '.' \
        | head -1
}

# Retorna tok/s do llama-rs (linha "N tokens, X.XX tok/s" no stderr)
run_rs() {
    local model=$1
    local err
    err=$("$RS_BIN" \
        -m "$model" -p "$PROMPT" -n "$N_TOKENS" \
        --temp 0 --seed "$SEED" --no-display-prompt \
        --timings 2>&1 >/dev/null || true)

    echo "$err" | grep -oE "[0-9]+\.[0-9]+ tok/s" | grep -oE "^[0-9]+\.[0-9]+"
}

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
echo ""
echo "Benchmark: llama.cpp vs llama-rs"
echo "  Prompt : \"$PROMPT\""
echo "  Tokens : $N_TOKENS  |  Seed: $SEED  |  Greedy (temp=0)"
echo ""
printf "%-22s | %17s | %16s | %11s\n" "Modelo" "llama.cpp (tok/s)" "llama-rs (tok/s)" "ratio rs/cpp"
printf "%s\n" "$(printf -- '-%.0s' {1..74})"

# ── Execução ──────────────────────────────────────────────────────────────────
for entry in "${MODELS[@]}"; do
    name="${entry%%:*}"
    model="${entry##*:}"
    model="${model# }"   # strip leading space

    if [[ ! -f "$model" ]]; then
        printf "%-22s | %17s | %16s | %11s\n" "$name" "modelo ausente" "-" "-"
        continue
    fi

    printf "%-22s | " "$name"

    cpp_tps=$(run_cpp "$model" || true)
    if [[ -z "$cpp_tps" ]]; then
        printf "%17s | " "erro"
    else
        LC_ALL=C printf "%17.1f | " "$cpp_tps"
    fi

    rs_tps=$(run_rs "$model" || true)
    if [[ -z "$rs_tps" ]]; then
        printf "%16s | %11s\n" "n/suportado" "-"
        continue
    fi
    LC_ALL=C printf "%16.1f | " "$rs_tps"

    if [[ -n "$cpp_tps" && "$cpp_tps" != "0" ]]; then
        ratio=$(awk "BEGIN { printf \"%.3f\", $rs_tps / $cpp_tps }")
        printf "%11sx\n" "$ratio"
    else
        printf "%11s\n" "-"
    fi
done

echo ""
echo "Nota: llama.cpp usa batch processing e KV-cache otimizado."
echo "      llama-rs roda token-a-token (sem batch) — comparação é de throughput bruto."
echo ""
