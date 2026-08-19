#!/usr/bin/env bash
# Roda um modelo GGUF apontado na linha de comando.
#
# Uso:
#   scripts/run.sh <modelo> [flags do llama-cli...]
#   scripts/run.sh                         # lista os modelos encontrados
#   scripts/run.sh -h                      # todas as flags e variáveis
#
# <modelo> aceita um caminho .gguf ou um pedaço do nome, procurado em
# $LLAMA_RS_MODELS e em models/ (ex.: `scripts/run.sh 32b -p "Oi"`).
#
# Sem nenhuma flag --gpu* o script escolhe o backend pelo tamanho do arquivo
# contra a VRAM livre: --gpu-resident se couber numa GPU, --gpu-layer-split se
# não couber. Qualquer flag --gpu* passada por você desliga essa escolha.
#
# Variáveis de ambiente do script:
#   LLAMA_RS_MODELS  onde procurar os .gguf   (padrão: models/)
#   RUN_FEATURES     features do cargo        (padrão: gpu; use "" para CPU pura,
#                                              "gpu profiling" para habilitar --trace)
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="${LLAMA_RS_MODELS:-models}"
FEATURES="${RUN_FEATURES-gpu}"
BIN=target/release/llama-cli

# ── Ajuda ───────────────────────────────────────────────────────────────────
ajuda() {
    cat <<'FIM'
Uso: scripts/run.sh <modelo> [flags do llama-cli...]

Flags do llama-cli (repassadas como você escrever):
  -m, --model <ARQ>       modelo GGUF (o script preenche a partir de <modelo>)
  -p, --prompt <TEXTO>    prompt de entrada                       (padrão: vazio)
  -n, --n-predict <N>     tokens a gerar                          (padrão: 128)
      --temp <F>          temperatura; 0.0 = greedy determinístico (padrão: 0.8)
      --top-k <N>         manter K candidatos; 0 desliga           (padrão: 40)
      --top-p <F>         nucleus; 1.0 desliga                     (padrão: 0.9)
      --seed <N>          semente da amostragem                    (padrão: 42)
      --ctx <N>           teto do KV-cache; custa VRAM             (padrão: 4096)
      --no-display-prompt não repetir o prompt na saída
      --timings           tok/s no stderr ao final   (o script liga se você não ligar)
      --gpu-resident      decode 100% numa GPU — modelo tem de caber na VRAM
      --gpu-layer-split   divide as camadas entre as 2 MI50 — para o que não cabe
      --gpu-single        backend antigo single-GPU (pesos residentes, Fase 8.1A)
      --gpu               backend antigo dual-GPU
      --trace <ARQ>       timeline CPU+GPU em Chrome Trace (ui.perfetto.dev);
                          exige RUN_FEATURES="gpu profiling" e LLAMA_RS_PROFILE=1

Precedência dos backends no runner: layer-split > resident > single > gpu > CPU.

Variáveis de ambiente do llama-rs:
  LLAMA_RS_GPU=N          força o índice da GPU (padrão: a de mais VRAM livre —
                          a do display perde ~7x de banda por spill para GTT)
  LLAMA_RS_SPLIT=N        fixa a fronteira do layer-split na camada N, em vez de
                          derivá-la da VRAM livre (que varia entre execuções)
  LLAMA_RS_PROFILE=1      liga a coleta de timestamps de GPU (necessária ao --trace)
  LLAMA_RS_TRACE_TOKENS=N quantos tokens entram no --trace       (padrão: 8)
  LLAMA_RS_MATVEC_GEOM=wg,linhas  geometria do matvec; wg múltiplo de 64 em
                          64..1024, linhas em 1..4               (padrão: 256,2)
  LLAMA_RS_NO_GROUP=1     uma barreira por op (comparação A/B do agrupamento)
  LLAMA_RS_STOP_LAYER=N   executa só as N primeiras camadas do shard (diagnóstico)
  RAYON_NUM_THREADS=N     threads do caminho CPU (padrão: núcleos físicos do nó 0)

Variáveis do script: LLAMA_RS_MODELS, RUN_FEATURES.
FIM
}

# ── Modelos disponíveis ─────────────────────────────────────────────────────
# `-L` para seguir os symlinks de models/; realpath para não listar o mesmo
# arquivo duas vezes (models/ aponta para dentro de $MODELS_DIR). Partes de GGUF
# dividido ficam de fora: o carregador não as junta, e elas abafam a busca por nome.
listar() {
    find -L "$MODELS_DIR" models -maxdepth 2 -iname '*.gguf' 2>/dev/null |
        grep -viE 'mmproj|ggml-vocab|-[0-9]{5}-of-[0-9]{5}\.gguf$' |
        while read -r f; do realpath "$f"; done |
        sort -u
}

case "${1-}" in
-h | --help)
    ajuda
    exit 0
    ;;
"")
    echo "Modelos em $MODELS_DIR e models/:" >&2
    listar | while read -r f; do
        printf '  %8s  %s\n' "$(numfmt --to=iec "$(stat -Lc %s "$f")")" "$f"
    done
    echo >&2
    echo "Uso: scripts/run.sh <modelo> [flags]   (scripts/run.sh -h para as flags)" >&2
    exit 1
    ;;
esac

# ── Resolver o modelo ───────────────────────────────────────────────────────
alvo="$1"
shift
if [[ -f "$alvo" ]]; then
    MODELO="$alvo"
elif [[ -f "$MODELS_DIR/$alvo" ]]; then
    MODELO="$MODELS_DIR/$alvo"
else
    mapfile -t achados < <(listar | grep -iF -- "$alvo" || true)
    case ${#achados[@]} in
    0)
        echo "ERRO: nenhum .gguf casa com '$alvo' em $MODELS_DIR nem em models/" >&2
        exit 1
        ;;
    1) MODELO="${achados[0]}" ;;
    *)
        echo "ERRO: '$alvo' casa com ${#achados[@]} modelos — seja mais específico:" >&2
        printf '  %s\n' "${achados[@]}" >&2
        exit 1
        ;;
    esac
fi

# O carregador não junta GGUF em partes: um shard sozinho não tem os tensores todos.
if [[ "$MODELO" =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ ]]; then
    echo "ERRO: '$MODELO' é uma parte de um GGUF dividido, que o llama-rs não lê." >&2
    echo "      Junte com: llama-gguf-split --merge <parte-00001-of-N> <saida.gguf>" >&2
    exit 1
fi

TAM=$(stat -Lc %s "$MODELO")

# ── Backend ─────────────────────────────────────────────────────────────────
ARGS=("$@")

tem() { # tem <flag> — a flag já está nos argumentos do usuário?
    local f
    for f in ${ARGS[@]+"${ARGS[@]}"}; do [[ "$f" == "$1" ]] && return 0; done
    return 1
}

vram() { # vram <maior|soma> — VRAM livre das GPUs AMD, em bytes
    local d tot usado livre acc=0
    for d in /sys/class/drm/card*/device; do
        [[ -f "$d/mem_info_vram_total" && "$(cat "$d/vendor" 2>/dev/null)" == 0x1002 ]] || continue
        tot=$(cat "$d/mem_info_vram_total")
        usado=$(cat "$d/mem_info_vram_used")
        livre=$((tot - usado))
        if [[ "$1" == soma ]]; then
            acc=$((acc + livre))
        elif ((livre > acc)); then
            acc=$livre
        fi
    done
    echo "$acc"
}

backend=()
if [[ "$FEATURES" == *gpu* ]] &&
    ! tem --gpu-resident && ! tem --gpu-layer-split && ! tem --gpu-single && ! tem --gpu; then
    # Margem de 1 GiB sobre o arquivo: KV-cache, ativações e staging não estão no .gguf.
    precisa=$((TAM + 1073741824))
    if ((precisa <= $(vram maior))); then
        backend=(--gpu-resident)
    else
        backend=(--gpu-layer-split)
        if ((precisa > $(vram soma))); then
            echo "AVISO: $(numfmt --to=iec "$TAM") não cabe nem somando a VRAM livre das" >&2
            echo "       GPUs ($(numfmt --to=iec "$(vram soma)")) — deve falhar por falta de memória." >&2
        fi
    fi
fi
tem --timings || ARGS+=(--timings)

echo "[run] $(basename "$MODELO")  $(numfmt --to=iec "$TAM")  ${backend[*]:-CPU}" >&2

# ── Build + execução ────────────────────────────────────────────────────────
if [[ -n "$FEATURES" ]]; then
    cargo build --release -p llama-cli --features "$FEATURES"
else
    cargo build --release -p llama-cli
fi

# ── NUMA ────────────────────────────────────────────────────────────────────
# Sem isto a máquina *inteira* trava ao carregar um modelo grande, e o processo morre por
# OOM com dezenas de GB livres. A máquina tem dois nós (uma GPU em cada); as alocações caem
# todas no nó da CPU que roda o processo, esse nó estoura — o desktop já ocupa a maior parte
# dele — e o kernel não faz fallback para o outro. O `gpu_reclaim` do amdgpu, que reporta
# valores impossíveis (283 GB em 62 GB de RAM), faz o kswapd girar num shrinker que não
# devolve nada: o PSI mostra `memory full`, ou seja todas as tarefas bloqueadas ao mesmo
# tempo — daí o congelamento sem nada aparecer em 100%.
#
# `--interleave=all` espalha as alocações pelos dois nós. Medido: as três políticas NUMA
# (interleave, membind, preferred) dão o mesmo tok/s, porque no decode residente os pesos
# já estão na VRAM — então isto não distorce benchmark, só evita o travamento.
numa=()
if command -v numactl >/dev/null; then
    numa=(numactl --interleave=all)
else
    echo "AVISO: numactl não encontrado — com modelo grande o sistema pode travar por" >&2
    echo "       esgotar um único nó NUMA (pacote: numactl)." >&2
fi

exec ${numa[@]+"${numa[@]}"} "$BIN" -m "$MODELO" ${backend[@]+"${backend[@]}"} ${ARGS[@]+"${ARGS[@]}"}
