# MTP no llama-rs: o que está pronto e o que falta

Multi-token prediction usando a cabeça que o próprio Qwen3.8-27B traz
(`nextn_predict_layers = 1`). O objetivo é gerar 2 tokens por passo de forward, já que o
passo é dominado por **ler os pesos** e ler os mesmos 16,38 GB serve para 1 ou 2 tokens.

`tok/s = base × (1 + taxa de aceitação)` — a aceitação é o teto, e é o número que ainda
não conhecemos.

## Fluxo

```text
passo normal   h_t     = forward(t)                    → logits → amostra t+1
cabeça MTP     h_mtp   = eh_proj([enorm(emb(t+1)) ; hnorm(h_t)])   // 2*n_embd → n_embd
               h'      = camada_mtp(h_mtp)
               logits' = output(shared_head_norm(h'))  → propõe t+2
verificação    forward_batch([t+1, t+2])               → logits de t+2 e de t+3
               argmax(logits[t+1]) == t+2 ?  aceita os dois, segue de t+3
                                             senão, aceita só t+1
```

O ganho vem da verificação: `forward_batch` de 2 tokens lê os pesos **uma vez**, então
custa quase o mesmo que 1 token. É por isso que o batch é pré-requisito — sem ele o MTP só
adiciona trabalho.

## Fase A — carregar o bloco (PRONTA)

`MtpRaw` e `MtpAux` em `crates/llama-model/src/gpu.rs`, com
`crates/llama-model/tests/mtp_load.rs` validando as formas contra o GGUF.

**A pegadinha que o teste trava:** o bloco MTP é o `blk.64`, e `eh_linear(64)` responde
`true` porque `(64 + 1) % 4 != 0`. Seguir essa regra faria o carregador procurar
`ssm_out.weight`. O GGUF traz `attn_q/k/v/output` — **o bloco MTP é de atenção**, mesmo
numa posição que seria de atenção linear. Consequência boa: ele não tem estado recorrente
próprio.

Custo do bloco, medido: **289 MB → 0,40 ms** a 717 GB/s, ~0,9% de um passo de 44,8 ms.
Barato o bastante para não atrapalhar mesmo com aceitação baixa.

| tensor | forma | tipo |
|---|---|---|
| `nextn.eh_proj.weight` | 10240 → 5120 | Q8_0 |
| `nextn.enorm` / `hnorm` / `shared_head_norm` | 5120 | F32 |
| `attn_q` | 5120 → 12288 | Q4_K |
| `attn_k` / `attn_v` | 5120 → 1024 | Q4_K / Q6_K |
| `attn_output` | 6144 → 5120 | Q4_K |
| `ffn_gate` / `ffn_up` | 5120 → 17408 | Q4_K |
| `ffn_down` | 17408 → 5120 | Q6_K |

## Fase B — executar a cabeça (FALTA)

Montar um plano no `ResidentForward` com peças que **já existem**, mais uma que não:

| passo | op | existe? |
|---|---|---|
| `enorm(emb)` e `hnorm(h)` | `NormFused` + `NormP2` | sim |
| concatenar os dois em `2*n_embd` | escrita em offsets do mesmo buffer | **não** — mas é só binding com offset |
| `eh_proj` | matvec Q8_0 | sim |
| camada de decoder | mesmo plano de uma camada de atenção | sim |
| `shared_head_norm` | norma | sim |
| `output` | matvec Q6_K | sim |

Precisa ainda de **KV-cache próprio** para o bloco (~2,2 GiB no `ctx` cheio; o llama.cpp
reserva o mesmo).

**Risco conhecido:** mudar o número de bindings de um shader sem mudar o
`ComputePipeline::with(..., n_bindings, ...)` não dá erro nenhum — o modelo só passa a
gerar o último token do vocabulário, porque a saída vai para lugar nenhum. Já aconteceu
neste projeto. Validar cada op nova contra referência de CPU antes de encadear.

## Fase C — decode em batch (FALTA, é o pré-requisito do ganho)

`build_plan_batch`, irmão do `build_plan`, com `n_batch` colunas. Peças prontas:

- **matvec K-quant com `COLS`** — `q4_k/q5_k/q6_k_matvec.comp` já aceitam N colunas contra
  uma leitura de peso (commits `d93bb68`, `5a4d2fc`).
- **`quantize_x`** — não precisou de mudança: os blocos de 32 já são independentes.
- **`attention` causal em batch** — `gl_WorkGroupID.y` é o token do bloco (`bacc81f`).

Falta: `swiglu`/`add`/`gate_mul` (elementwise, triviais), `rope` (posição por token),
`norm_fused`/`norm_p2` (um workgroup por token), `delta_net` (recorrência serial — os
tokens do bloco têm de ser aplicados em ordem, mas as projeções em volta batcham), e os
buffers `n_batch×` mais o `build_plan_batch`.

## Fase D — rollback do estado recorrente (FALTA)

48 das 65 camadas são delta-net, **com estado recorrente**. Verificar `[t+1, t+2]` avança o
estado dois passos; rejeitar t+2 exige voltar um.

- Estado: `n_v_heads × d_state × head_v_dim × 4 B` = 3,1 MB por camada → **151 MB**.
- Salvar + restaurar a 717 GB/s: **~0,42 ms**, ~1% do passo. Tolerável.
- Alternativa sem cópia: aplicar a recorrência de t+2 só depois de confirmar a aceitação —
  troca memória por uma dependência a mais no plano.

O KV-cache é mais fácil: rejeitar é recuar o ponteiro de posição.

## Fase E — ligar e medir

Flag no `llama-cli` (desligado por padrão), e registrar em
`docs/decode-por-configuracao.md` os pares com/sem MTP na mesma build e com
`LLAMA_RS_SPLIT` fixo.

**Antes de investir nas fases C e D**, medir a **taxa de aceitação** com a Fase B pronta:
basta rodar a cabeça, guardar a proposta e comparar com o token que o passo seguinte
amostrar de verdade. Isso não precisa de batch nem de rollback, e é o número que decide se
o resto compensa. Referência a bater: o llama.cpp mediu **+3,4%** neste modelo
(`docs/mtp-e-k80.md`), o que implicaria aceitação baixa — mas mede a implementação dele
num híbrido SSM, não o teto.
