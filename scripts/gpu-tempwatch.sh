#!/usr/bin/env bash
# Watchdog de temperatura das MI50: mata a inferencia se qualquer sensor passar do limite.
#
# Limites: junction/edge 105 C (pedido do usuario), mem 94 C (o `crit` do proprio sensor
# nesta placa — o emerg da memoria e 99, entao 105 nunca dispararia para ela).
#
# Parse: ler SO o campo 2 de cada linha. Um grep solto casa o `crit = +105.0` e dispara
# com as placas frias.
LOG=/tmp/claude-1000/-home-murilo-llama-rs/e2f6862b-0a29-43ac-a4d3-b2f8a95779b6/scratchpad/tempwatch.log
LIM_JUNCTION=105
LIM_MEM=94

echo "[tempwatch] iniciado $(date +%T) — limites: junction/edge ${LIM_JUNCTION}C, mem ${LIM_MEM}C" > "$LOG"

while true; do
    read -r maxj maxm < <(sensors 2>/dev/null | awk '
        /^(junction|edge):/ { gsub(/[+°C]/,"",$2); if ($2+0 > j) j=$2+0 }
        /^mem:/             { gsub(/[+°C]/,"",$2); if ($2+0 > m) m=$2+0 }
        END { printf "%d %d\n", j+0, m+0 }')

    echo "$(date +%T) junction=${maxj}C mem=${maxm}C" >> "$LOG"

    if [ "$maxj" -ge "$LIM_JUNCTION" ] || [ "$maxm" -ge "$LIM_MEM" ]; then
        echo "$(date +%T) !!! LIMITE ATINGIDO (junction=${maxj} mem=${maxm}) — matando llama-cli" >> "$LOG"
        pkill -f '[l]lama-cli'
        sleep 5
        pkill -9 -f '[l]lama-cli'
        echo "$(date +%T) inferencia encerrada pelo watchdog" >> "$LOG"
        exit 1
    fi
    sleep 3
done
