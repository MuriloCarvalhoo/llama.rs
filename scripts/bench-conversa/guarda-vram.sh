#!/usr/bin/env bash
# Amostra a VRAM livre das duas MI50 a cada 0,2 s e registra o mínimo.
# Mata o processo $ALVO_PID (sem ele, o llama-cli) se a GPU do monitor ficar com < 2048 MiB,
# a outra com < 500 MiB, ou se alguma passar de 105 °C. Pelo PID, e não por `pkill -f`: o nome
# casaria o próprio shell, e `pkill -x llama-server` mataria o runner do Ollama.
LOG="${1:-/tmp/bench-conversa/guarda.log}"
mon=""
for s in /sys/class/drm/card*-*/status; do
    [ "$(cat "$s")" = connected ] && mon="$(basename "$(dirname "$s")" | cut -d- -f1)"
done
declare -A minimo
for c in card1 card2; do minimo[$c]=999999; done
echo "monitor=$mon" >"$LOG"
tmax=0
while :; do
    for c in card1 card2; do
        d=/sys/class/drm/$c/device
        livre=$(( ($(cat $d/mem_info_vram_total) - $(cat $d/mem_info_vram_used)) / 1048576 ))
        (( livre < minimo[$c] )) && minimo[$c]=$livre
        lim=500; [ "$c" = "$mon" ] && lim=2048
        if (( livre < lim )); then
            echo "$(date +%T) MATANDO: $c livre=${livre}MiB < ${lim}" >>"$LOG"
            if [ -n "${ALVO_PID:-}" ]; then kill "$ALVO_PID"; else pkill -x llama-cli; fi
        fi
    done
    t=$(sensors 2>/dev/null | awk '/^(junction|mem|edge):/ {gsub(/[+°C]/,"",$2); if ($2+0 > m) m=$2+0} END {print m+0}')
    if (( ${t%.*} > 105 )); then echo "$(date +%T) MATANDO: ${t}C" >>"$LOG"; if [ -n "${ALVO_PID:-}" ]; then kill "$ALVO_PID"; else pkill -x llama-cli; fi; fi
    (( ${t%.*} > tmax )) && tmax=${t%.*}
    printf 'min_livre card1=%s card2=%s temp_max=%s\n' "${minimo[card1]}" "${minimo[card2]}" "$tmax" >"$LOG.min"
    sleep 0.2
done
