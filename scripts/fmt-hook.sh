#!/usr/bin/env bash
# Hook PostToolUse: formata o arquivo .rs editado com rustfmt (edition 2024).
# Recebe o payload do Claude Code via stdin (JSON com .tool_input.file_path).
set -euo pipefail
f=$(jq -r '.tool_input.file_path // empty')
[ -n "$f" ] || exit 0
case "$f" in
    *.rs) rustfmt --edition 2024 "$f" ;;
esac
