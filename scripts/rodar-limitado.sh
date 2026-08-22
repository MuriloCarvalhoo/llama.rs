#!/usr/bin/env bash
# Roda um comando sob teto de memória (cgroup v2) + NUMA interleave.
# Se o comando estourar o teto, o kernel mata SÓ ele — o terminal e o
# resto do sistema sobrevivem (a máquina não tem swap: sem isso, um
# estouro congela o sistema inteiro antes do OOM killer agir).
#
# Uso: scripts/rodar-limitado.sh <comando...>
#   LIMITE_MEM  teto duro (default 40G)
#
# Só MemoryMax, sem MemoryHigh: sem swap, MemoryHigh vira throttling
# infinito de memória anônima (processo rasteja em vez de morrer).
# Ao bater no Max o kernel recupera page cache (o mmap do GGUF é
# reclaimável) e, se não bastar, OOM-kill restrito ao cgroup.
set -euo pipefail
MAX="${LIMITE_MEM:-40G}"
exec systemd-run --user --scope -q \
  -p MemoryMax="$MAX" -p MemorySwapMax=0 \
  numactl --interleave=all "$@"
