#!/usr/bin/env bash
# Registra, a cada 0,5 s, o máximo por sensor e por placa: edge, junction, mem e PPT.
# Mata o processo $ALVO_PID (sem ele, o llama-cli) se a junction passar de 104 °C (regra do
# usuário: 105) ou a HBM de 90 °C (crit = 94 °C).
OUT="$1"
: >"$OUT"
while :; do
    sensors 2>/dev/null | awk '
        /^amdgpu-pci-/ { p = $1 }
        /^(edge|junction|mem):/ { gsub(/[+°C]/, "", $2); printf "%s %s %s\n", p, $1, $2 }
        /^PPT:/ { printf "%s PPT %s\n", p, $2 }' >>"$OUT"
    if awk '($2=="junction:" && $3>104) || ($2=="mem:" && $3>90) {f=1} END {exit !f}' <(tail -8 "$OUT"); then
        echo "MATANDO por temperatura" >>"$OUT"
        if [ -n "${ALVO_PID:-}" ]; then kill "$ALVO_PID"; else pkill -x llama-cli; fi
    fi
    sleep 0.5
done
