#!/usr/bin/env bash
# Uma configuração do benchmark de conversa longa: sobe o servidor, liga os guardas no PID
# dele, roda as profundidades pedidas e derruba tudo.
# Precisa de: servidor ouvindo em 127.0.0.1:<porta>, conversas.json calibrado (conversa.py calibrar).
# Uso: rodar_config.sh <rotulo> <motor: llamars|llamacpp> <porta> <pstate_externo: nenhum|standard> <profundidades> -- <comando do servidor...>
set -uo pipefail
cd /home/murilo/llama.rs
S="$(cd "$(dirname "$0")" && pwd)"
O="${BENCH_SAIDA:-/tmp/bench-conversa}"
mkdir -p "$O"
export BENCH_SAIDA="$O"
ROT=$1; MOTOR=$2; PORTA=$3; PST=$4; PROF=$5; shift 6
H=""
if [ "$PST" != nenhum ]; then
    python3 "$S/segura_pstate.py" "$PST" /dev/dri/renderD128 /dev/dri/renderD129 >"$O/pst-$ROT.log" 2>&1 &
    H=$!
    sleep 1
fi
scripts/rodar-limitado.sh "$@" >"$O/srv-$ROT.log" 2>&1 &
SP=$!
ALVO_PID=$SP "$S/guarda-vram.sh" "$O/gv-$ROT.log" & G=$!
ALVO_PID=$SP "$S/amostra-temp.sh" "$O/t-$ROT.txt" & T=$!
trap 'kill $SP $G $T $H 2>/dev/null' EXIT
for i in $(seq 180); do curl -sf "http://127.0.0.1:$PORTA/health" >/dev/null && break; sleep 1; done
echo "== $ROT ($(date +%T)) pstate externo: $PST"
# shellcheck disable=SC2086
if [ "$PROF" = frioquente ]; then
    python3 "$S/conversa.py" frioquente "$PORTA" "$ROT"
elif [ "$PROF" = crescer ]; then
    python3 "$S/conversa.py" crescer "$PORTA" "$MOTOR" "$ROT"
else
    python3 "$S/conversa.py" rodar "$PORTA" "$MOTOR" "$ROT" $PROF
fi
echo "VRAM/temp: $(cat "$O/gv-$ROT.log.min")"
grep -c MATANDO "$O/t-$ROT.txt" "$O/gv-$ROT.log" 2>/dev/null
[ "$MOTOR" = llamars ] && grep -E "^\[gen\]" "$O/srv-$ROT.log"
echo FIM
