# Servidor de chat e uso no opencode

O `llama-server` expõe o backend residente na API de chat da OpenAI, que é o que o
opencode fala. Este documento registra como subir, o que está implementado e — mais
importante — **onde estão os limites**, para não serem redescobertos na marra.

## Subir

```bash
cargo build --release -p llama-server --features gpu

numactl --interleave=all ./target/release/llama-server \
    -m models/Qwen3.8-27B-Q4_K_M.gguf \
    --gpu-layer-split --ctx 32768 --nome qwen3.8-27b
```

`numactl --interleave=all` não é opcional: sem ele o nó 0 estoura e a máquina trava (ver
`scripts/run.sh`). O servidor escuta em `127.0.0.1:8080` por padrão.

> **Rodando a partir do Claude Code o processo morre com exit 137.** O shell do agente
> tem `oom_score_adj=200` e o servidor herda esse valor, virando o alvo do OOM killer
> quando os 19 GB de pesos sobem para a VRAM. Não é falta de RAM nem bug: subir num
> terminal comum resolve.

Rotas: `POST /v1/chat/completions` (com e sem `stream`), `GET /v1/models`, `GET /health`.

## Config do opencode

`~/.config/opencode/opencode.json`:

```json
"llamars": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "llama-rs (local)",
  "options": { "baseURL": "http://127.0.0.1:8080/v1" },
  "models": {
    "qwen3.8-27b": {
      "name": "Qwen3.8-27B (llama-rs)",
      "tool_call": true, "reasoning": true,
      "limit": { "context": 32768, "output": 8192 }
    }
  }
}
```

O `context` do config tem de caber no `--ctx` do servidor: prompt maior que o contexto é
recusado com erro, não truncado (truncar silenciosamente esconderia perda de instrução).

## O formato deste modelo

Duas coisas do Qwen3.8 que fogem do que se espera de um modelo OpenAI-like:

**Tool call é XML, não JSON.** O template ensina o formato abaixo, e é o que o parser em
`llama-server/src/saida.rs` lê:

```text
<tool_call>
<function=read>
<parameter=path>
src/main.rs
</parameter>
</function>
</tool_call>
```

O valor de cada parâmetro é texto puro. O tipo vem do schema que o cliente mandou em
`tools`; sem schema, o valor fica string — inventar tipo quebraria uma ferramenta que
espera `"10"`.

**A resposta começa dentro do raciocínio.** O prompt termina em `<|im_start|>assistant\n
<think>\n`, então tudo até `</think>` é `reasoning_content`, não conteúdo. Um servidor que
ignorasse isso entregaria o raciocínio como resposta.

O chat template não é interpretado em runtime: está reimplementado em `llama-chat` e
provado contra a saída do Jinja real, caso a caso, em `refs/chat_qwen38.json`
(regenerável por `scripts/gen-chat-refs.py`).

## Reuso do KV-cache entre turnos

O servidor mantém uma sessão só e reaproveita o prefixo comum entre o que está no cache e
o prompt novo (`llama_model::Sessao`). O caso do agente — histórico inteiro de volta mais
um turno — processa apenas o que cresceu.

**Quando o prefixo diverge no meio, o cache inteiro é descartado.** Não é preguiça: 48 das
65 camadas do qwen35 são delta-net, com estado recorrente. O KV de atenção poderia ser
truncado (basta recuar o comprimento), mas o estado recorrente é o produto de todos os
tokens processados, em ordem, e só voltaria atrás com um snapshot de 151 MB por ponto de
retorno. Pelo mesmo motivo não dá para reprocessar um token que já entrou.

Consequência prática, coberta por teste: **se o cliente não devolver o `reasoning_content`
no histórico**, o template reconstrói o turno do assistant diferente do que foi gerado, o
prefixo diverge e o prompt inteiro é reprocessado. Com o raciocínio de volta, o texto
re-renderizado bate byte a byte com o que o modelo gerou e o cache é aproveitado.

## Custo de contexto em VRAM

O KV-cache só existe nas camadas de atenção (16 das 65 no Qwen3.8-27B; as delta-net têm
estado de tamanho fixo). São **136 KB por token**:

| ctx | KV-cache |
|---:|---:|
| 4 096 | 0,5 GB |
| 32 768 | 4,4 GB |
| 65 536 | 8,7 GB |

Com os 16,3 GB de pesos, `--ctx 32768` ocupa ~21 GB das duas MI50. Antes de o cache ser
indexado por camada de atenção, o mesmo contexto pediria 17 GB só de cache e não caberia.

## Medido no modelo real

2× MI50, layer-split (camadas 0..30 | 30..64), `--ctx 32768`, greedy, esforço `low`:

| requisição | prompt | saída | tempo |
|---|---:|---:|---:|
| pergunta curta | 46 tok | 25 tok | 1,9 s |
| com bloco de tools, turno 1 | 315 tok | 49 tok | 6,5 s |
| **turno 2 do mesmo diálogo** | **399 tok** | 60 tok | **3,4 s** |

O turno 2 tem prompt **maior** e roda em metade do tempo: o prefixo comum já estava no
cache e só os ~84 tokens novos foram processados. É o item de reuso pagando a conta.

O tool call do turno 1 saiu como `finish_reason: tool_calls` com
`{"path":"src/main.rs"}` — o XML do modelo convertido para o formato da API.

## Uma requisição por vez

O laço de conexões é sequencial. Os pesos são residentes e um único decode já satura a
banda das duas placas; atender duas requisições em paralelo dividiria a mesma banda e
ainda embaralharia a sessão do KV-cache.

## O que não está implementado

- **Prefill mais rápido que 8 tokens por bloco.** O batch está limitado a 8
  (`LLAMA_RS_BATCH`); acima disso é preciso o GEMM com tiling em LDS
  (`docs/prefill-em-batch.md`, etapa 2). Um prompt inicial grande ainda custa minutos.
- **Snapshot do estado recorrente**, que tornaria o reuso de prefixo tolerante a
  divergência no meio.
- Imagens (o runtime não tem visão), `logprobs`, `n > 1`, penalidades de repetição.
