#!/usr/bin/env bash
# Vigia as MI50 enquanto um processo roda e mata **esse processo** se ele passar dos limites
# de segurança da máquina. Ao fim imprime um resumo por GPU.
#
# Uso: scripts/monitor-gpu.sh <PID> [log]
#   exemplo: scripts/rodar-limitado.sh ./target/release/llama-cli ... & scripts/monitor-gpu.sh $!
#
# Limites (os mesmos das memórias do projeto):
#   - VRAM livre: 2048 MiB na GPU que dirige o monitor (conector `connected`), 500 MiB na outra;
#   - junction > 108 °C (regra do usuário: nunca passar de 110, o `temp2_emergency`) ou HBM
#     ("mem") > 90 °C (crit = 94 °C).
# Mata pelo PID, nunca por nome: `pkill -f` casaria o próprio shell que chamou o script, e
# `pkill -x llama-server` mataria o runner do Ollama, que tem o mesmo nome.
#
# O log tem uma linha por GPU e amostra (a cada 0,25 s):
#   hora card vram_livre_MiB junction_C mem_C ppt_W sclk_MHz mclk_MHz
# Saída 3 quando precisou matar o processo.
set -uo pipefail

PID="${1:?uso: $0 <PID> [log]}"
LOG="${2:-/tmp/monitor-gpu-$PID.log}"
PERIODO=0.25

# GPUs AMD com VRAM (cardN, não os conectores cardN-DP-M).
cards=()
for c in /sys/class/drm/card*; do
    n=$(basename "$c")
    [[ "$n" == *-* ]] && continue
    [ -f "$c/device/mem_info_vram_total" ] || continue
    [ "$(cat "$c/device/vendor" 2>/dev/null)" = 0x1002 ] || continue
    cards+=("$n")
done
[ ${#cards[@]} -gt 0 ] || { echo "nenhuma GPU AMD em /sys/class/drm" >&2; exit 1; }

declare -A margem pci sens_j sens_m pot vmin jmax mmax pmax hist_s hist_m
for n in "${cards[@]}"; do
    d=/sys/class/drm/$n/device
    pci[$n]=$(basename "$(readlink -f "$d")")
    margem[$n]=500
    for s in /sys/class/drm/"$n"-*/status; do
        [ "$(cat "$s" 2>/dev/null)" = connected ] && margem[$n]=2048
    done
    for l in "$d"/hwmon/hwmon*/temp*_label; do
        case "$(cat "$l")" in
            junction) sens_j[$n]=${l%_label}_input ;;
            mem) sens_m[$n]=${l%_label}_input ;;
        esac
    done
    pot[$n]=$(ls "$d"/hwmon/hwmon*/power1_input 2>/dev/null | head -1)
    vmin[$n]=999999; jmax[$n]=0; mmax[$n]=0; pmax[$n]=0
done
# Monitor em repouso: o DisplayPort passa a `disconnected`, e não dá para saber qual GPU o
# compositor vai usar ao acordar — a margem maior vale para todas.
dormindo=1
for n in "${cards[@]}"; do (( margem[$n] == 2048 )) && dormindo=0; done
if (( dormindo )); then
    for n in "${cards[@]}"; do margem[$n]=2048; done
fi

le() { local v; read -r v <"$1" 2>/dev/null && echo "$v" || echo 0; }
nivel() { awk '/\*/ {gsub(/[^0-9]/, "", $2); print $2}' "$1" 2>/dev/null; }

: >"$LOG"
matou=""
t0=$SECONDS
amostras=0
while kill -0 "$PID" 2>/dev/null; do
    amostras=$((amostras + 1))
    for n in "${cards[@]}"; do
        d=/sys/class/drm/$n/device
        livre=$(( ($(le "$d/mem_info_vram_total") - $(le "$d/mem_info_vram_used")) / 1048576 ))
        j=$(( $(le "${sens_j[$n]:-/dev/null}") / 1000 ))
        m=$(( $(le "${sens_m[$n]:-/dev/null}") / 1000 ))
        p=$(( $(le "${pot[$n]:-/dev/null}") / 1000000 ))
        s=$(nivel "$d/pp_dpm_sclk"); k=$(nivel "$d/pp_dpm_mclk")
        (( livre < vmin[$n] )) && vmin[$n]=$livre
        (( j > jmax[$n] )) && jmax[$n]=$j
        (( m > mmax[$n] )) && mmax[$n]=$m
        (( p > pmax[$n] )) && pmax[$n]=$p
        hist_s[$n:$s]=$(( ${hist_s[$n:$s]:-0} + 1 ))
        hist_m[$n:$k]=$(( ${hist_m[$n:$k]:-0} + 1 ))
        echo "$(date +%T.%N | cut -c1-12) $n $livre $j $m $p $s $k" >>"$LOG"

        motivo=""
        (( livre < margem[$n] )) && motivo="VRAM livre ${livre} MiB < ${margem[$n]}"
        (( j > 108 )) && motivo="junction ${j} °C > 108"
        (( m > 90 )) && motivo="mem ${m} °C > 90"
        if [ -n "$motivo" ] && [ -z "$matou" ]; then
            matou="$n: $motivo"
            echo "$(date +%T) MATANDO $PID — $matou" | tee -a "$LOG" >&2
            kill "$PID" 2>/dev/null
        fi
    done
    sleep "$PERIODO"
done

# Percentual de cada nível de clock, do mais frequente para o menos.
histograma() {
    local n=$1 tipo=$2 k total=0 linha=""
    declare -n h=$tipo
    for k in "${!h[@]}"; do [[ $k == "$n:"* ]] && total=$(( total + h[$k] )); done
    (( total > 0 )) || { echo "-"; return; }
    for k in "${!h[@]}"; do
        [[ $k == "$n:"* ]] || continue
        echo "$(( h[$k] * 100 / total )) ${k#"$n:"}"
    done | sort -rn | awk '{printf "%s%s MHz %s%%", (NR>1 ? " | " : ""), $2, $1}'
}

{
    echo "== monitor-gpu: PID $PID terminou após $(( SECONDS - t0 )) s, $amostras amostras (log: $LOG)"
    (( dormindo )) && echo "nenhum conector ativo (monitor em repouso?): margem de 2048 MiB em todas"
    for n in "${cards[@]}"; do
        papel="sem monitor"; (( margem[$n] == 2048 )) && papel="monitor"
        (( dormindo )) && papel="monitor desconhecido"
        echo "$n ($papel, ${pci[$n]}): VRAM livre mín ${vmin[$n]} MiB (margem ${margem[$n]}) | junction máx ${jmax[$n]} °C | mem máx ${mmax[$n]} °C | potência máx ${pmax[$n]} W"
        echo "  sclk: $(histograma "$n" hist_s)"
        echo "  mclk: $(histograma "$n" hist_m)"
    done
    [ -n "$matou" ] && echo "MATOU o processo — $matou"
} | tee -a "$LOG"
[ -z "$matou" ]  || exit 3
