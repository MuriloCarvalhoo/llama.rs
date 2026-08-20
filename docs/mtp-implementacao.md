# MTP no llama-rs: o que está pronto e o que falta

Multi-token prediction usando a cabeça que o próprio Qwen3.8-27B traz
(`nextn_predict_layers = 1`). O objetivo é gerar 2 tokens por passo de forward, já que o
passo é dominado por **ler os pesos** e ler os mesmos 16,38 GB serve para 1 ou 2 tokens.

`tok/s = base × (1 + taxa de aceitação)`. A aceitação foi **medida: 60,9 %** (fator 1,61),
o que torna o alvo de 50 tok/s alcançável — ver `docs/decode-por-configuracao.md`.

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

## Fase B — executar a cabeça (PRONTA, na CPU)

`MtpHead::propor` em `crates/llama-model/src/mtp.rs` executa o bloco e devolve a proposta.
Roda **na CPU** de propósito: o que se queria era a taxa de aceitação, que não depende da
velocidade, e isso evitou montar um plano Vulkan antes de saber se o ganho existia.

Medido pelo `crates/llama-vulkan/tests/mtp_aceitacao.rs`, greedy, 23 propostas verificadas
contra o token que o modelo de fato produziu: **14 acertos, 60,9 %**.

Os 60,9 % também servem de prova de que a cabeça está correta — ela espelha a camada de
atenção do qwen35 incluindo dois detalhes fáceis de errar (queries com stride
`2 × head_dim`, porque o portão mora ao lado de cada uma; e o portão entrando como
`sigmoid` sobre a saída da atenção). Qualquer divergência derrubaria a aceitação para perto
de zero.

O matvec desquantiza **uma linha por vez** em vez de materializar o tensor: o
`output.weight` tem 248 320 linhas e custaria 5 GB de RAM só para o vocabulário.

**Para produção** a cabeça precisa migrar para a GPU, com estas peças — todas já existem:

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

## Fase C — decode em batch (é o pré-requisito do ganho)

O plano generaliza para `n_tok` em vez de ganhar um irmão duplicado: o token do bloco vem
sempre de `gl_WorkGroupID.y`, então com `n_tok=1` cada shader colapsa no caminho de decode
de antes — o que os 28 testes de integração existentes confirmam a cada mudança.

**Shaders prontos e testados:**

| shader | como | commit |
|---|---|---|
| `q4_k/q5_k/q6_k_matvec` | spec constant `COLS`, N colunas por leitura de peso | `d93bb68`, `5a4d2fc` |
| `attention` | `gl_WorkGroupID.y` é o token; máscara causal por laço curto | `bacc81f` |
| `norm_fused` / `norm_p2` | parciais e `xq`/`xd` separados por token | `f9434ca` |
| `rope` | `pos - (n_tok - 1) + t` por token | `f9434ca` |
| `quantize_x`, `gate_mul`, `add`, `swiglu` | **nenhuma mudança** — já são token-major | — |

`gate_mul` merece nota: com `n = attn_dim × n_tok`, o `h = i / head_dim` do shader avança
sozinho para `t * n_head + hh`, e `b_q` tem exatamente esse layout. Foi sorte da estrutura,
não projeto — mas está coberto pelo teste de ponta a ponta.

**`kv_append` também não precisa de shader novo:** as posições do bloco são consecutivas no
cache, então uma única `cmd_copy_buffer` de `n_tok × kv_dim` substitui as N cópias.

**O que falta:**

1. Buffers `n_batch×` (`b_x`, `b_q`, `b_k`, `b_v`, `b_attn`, `b_proj`, `b_gate`, `b_up`,
   `b_normed`, `b_xq`, `b_xd`, `b_parciais`) e `n_batch` no `Cfg`.
2. `decode_batch(tokens, pos0)` devolvendo os logits de cada token.
3. `plano_delta` em batch: os matvec das projeções levam `COLS = n_tok`, mas a
   **recorrência** (`delta_net`, `dn_conv`) roda `n_tok` dispatches em ordem, um por token.
   São 48 das 65 camadas, mas a parte serial custa só 1,35 ms de 42,44 (3,2 %) — o grosso
   dessas camadas são os matvec, que batcham.

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

A taxa de aceitação já está medida (60,9 %), então as fases C e D estão justificadas por
dado, não por projeção. O `docs/mtp-e-k80.md` registra **+3,4 %** para o MTP no llama.cpp
neste mesmo modelo — fator 1,03 contra os 1,61 medidos aqui. A explicação mais provável é
que a implementação dele não faz batch eficiente num híbrido SSM; a cabeça do modelo prevê
bem. Aquele número não deve ser usado como estimativa do potencial.

**Conta do alvo:** com a base de hoje (22,3) o MTP dá 35,9 tok/s. Para 50 é preciso a base
em ~31,1, o que exige também o matvec Q4_K perto dos 717 GB/s e a fusão das dispatches
pequenas.
