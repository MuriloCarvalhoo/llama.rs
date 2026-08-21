#!/usr/bin/env bash
# Roda todas as variantes de todos os experimentos, REPS vezes cada,
# um processo por variante (VmHWM é monotônico por processo).
# Saída bruta em resultados.csv; medianas no stderr ao final.
set -euo pipefail
cd "$(dirname "$0")"

REPS="${REPS:-3}"
OUT="${OUT:-resultados.csv}"

cargo build --release --quiet
cargo build --quiet

BIN_RELEASE=target/release/validacao-rust-praticas
BIN_DEV=target/debug/validacao-rust-praticas

experimentos=(
  "vec_crescimento push"
  "vec_crescimento with_capacity"
  "string_concat format_reconstroi"
  "string_concat push_str"
  "string_concat write_capacidade"
  "produto_escalar indexado"
  "produto_escalar zip"
  "ordenar sort"
  "ordenar sort_unstable"
  "mapa_insercao hashmap_novo"
  "mapa_insercao hashmap_capacidade"
  "mapa_insercao btreemap"
  "clone_em_laco clone"
  "clone_em_laco emprestimo"
  "io_escrita sem_buffer"
  "io_escrita buf_writer"
  "io_escrita write_all"
  "io_leitura sem_buffer"
  "io_leitura buf_reader"
  "collect_intermediario collect"
  "collect_intermediario fundido"
  "vec_inicializacao macro_vec"
  "vec_inicializacao push"
  "vec_inicializacao resize"
  "mapa_entry duas_buscas"
  "mapa_entry entry"
  "encolher unico"
)

: > "$OUT"
"$BIN_RELEASE" tamanhos unico >> "$OUT"

for rep in $(seq "$REPS"); do
  echo "== repetição $rep/$REPS ==" >&2
  for e in "${experimentos[@]}"; do
    echo "   $e" >&2
    # shellcheck disable=SC2086
    "$BIN_RELEASE" $e >> "$OUT"
  done
  echo "   soma_matmul (dev e release)" >&2
  "$BIN_DEV" soma_matmul auto >> "$OUT"
  "$BIN_RELEASE" soma_matmul auto >> "$OUT"
done

echo >&2
echo "medianas — experimento,variante,n,ms,pico_mb:" >&2
gawk -F, '
  $1 == "resultado" {
    chave = $2 "," $3 "," $4
    n[chave]++
    t[chave, n[chave]] = $5 + 0
    hwm[chave] = $6
  }
  END {
    for (chave in n) {
      m = n[chave]
      delete a
      for (i = 1; i <= m; i++) a[i] = t[chave, i]
      asort(a)
      printf "%s,%.1f,%s\n", chave, a[int((m + 1) / 2)], hwm[chave]
    }
  }
' "$OUT" | sort >&2
