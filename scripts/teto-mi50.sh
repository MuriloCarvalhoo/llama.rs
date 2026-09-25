#!/usr/bin/env bash
# Mede o teto real de cada MI50: banda de leitura da HBM2 e throughput por tipo numérico
# (fp64, fp32, fp16, dot2 f16, dot4 int8, dot8 int4). Ver o cabeçalho de teto-mi50.hip.
#
# Uso: scripts/teto-mi50.sh
#
# Aloca 2 GiB por GPU. Antes confere a margem de VRAM: a GPU que dirige o monitor tem de
# ficar com 2 GiB livres e a outra com 500 MiB — sem isso a sessão gráfica pode travar.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=target/teto-mi50
PRECISA=$((2048 + 64))   # MiB do buffer de leitura + folga

for c in /sys/class/drm/card*; do
    card=$(basename "$c")
    [[ "$card" == *-* ]] && continue   # conector (card2-DP-8), não GPU
    dev="$c/device"
    [ "$(cat "$dev/vendor" 2>/dev/null)" = 0x1002 ] || continue
    [ -f "$dev/mem_info_vram_total" ] || continue
    livre=$(( ($(cat "$dev/mem_info_vram_total") - $(cat "$dev/mem_info_vram_used")) / 1048576 ))
    margem=500
    for s in /sys/class/drm/"$card"-*/status; do
        [ "$(cat "$s")" = connected ] && margem=2048
    done
    if (( livre - PRECISA < margem )); then
        echo "$card: ${livre} MiB livres; faltaria margem de ${margem} MiB. Veja 'ollama ps'." >&2
        exit 1
    fi
done

mkdir -p target
if [ ! -x "$BIN" ] || [ scripts/teto-mi50.hip -nt "$BIN" ]; then
    /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 scripts/teto-mi50.hip -o "$BIN"
fi
exec "$BIN"
