#!/usr/bin/env bash
# Hook PostToolUse: formata com rustfmt e passa o clippy a cada .rs criado ou editado.
# Recebe o payload do Claude Code via stdin (JSON com .tool_name e .tool_input).
#
# Write/Edit trazem o caminho no payload. Bash nao traz, entao os alvos saem do
# git status — o que de quebra evita rodar o clippy atras de comando de leitura.
#
# O clippy roda SEM `-D warnings`: valem as severidades declaradas no Cargo.toml,
# e so os lints `deny` (unwrap, expect, panic, unsafe, casts) barram a edicao.
# `indexing_slicing` e' `warn` de proposito e tem ~100 ocorrencias no repo — com
# `-D warnings` o hook falharia em toda edicao. O portao duro segue em gate.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

payload=$(cat)
targets=()

case "$(jq -r '.tool_name // empty' <<<"$payload")" in
    Write|Edit)
        f=$(jq -r '.tool_input.file_path // empty' <<<"$payload")
        [ -f "$f" ] && [[ $f == *.rs ]] && targets=("$f")
        ;;
    Bash)
        while read -r f; do
            [ -f "$f" ] && targets+=("$f")
        done < <(git status --porcelain -- '*.rs' | awk '{print $NF}')
        ;;
esac

[ ${#targets[@]} -gt 0 ] || exit 0

rustfmt --edition 2024 "${targets[@]}"

if ! out=$(cargo clippy --workspace --all-targets --message-format short 2>&1); then
    # So os erros: os ~150 warnings herdados do workspace afogariam o retorno.
    errs=$(grep -E 'error(\[|:)' <<<"$out" | head -40 || true)
    echo "${errs:-$out}" >&2
    exit 2
fi
