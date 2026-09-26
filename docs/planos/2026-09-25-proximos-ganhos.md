# Próximos ganhos — 2026-09-25

Lista priorizada do que ficou para depois da rodada de 2026-09-25 (pstate fixo, GEMM do prefill,
atenção de contexto longo, E01/E02/E05). Cada item traz a evidência que o justifica e uma
estimativa de esforço para uma pessoa familiarizada com o código. A comparação com o llama.cpp que
motiva a ordem está em [`../benchmark-conversa-longa-2026-09-25.md`](../benchmark-conversa-longa-2026-09-25.md).

## 0. Reuso de prefixo entre turnos — o maior ganho para o opencode

| # | O quê | Evidência | Esforço |
|---|---|---|---|
| 0.1 | Snapshot do estado recorrente **antes** do fim do prompt (na abertura `<|im_start|>assistant` da resposta) e mais de um checkpoint guardado | Numa conversa de 20 turnos o servidor mostrou `0 do cache` em todo turno: a sessão guarda um snapshot só, no fim exato do prompt (`sessao.rs:201`), e o template do turno seguinte re-renderiza esse fim. Com 30k tokens isso é **387 s até a primeira palavra**, contra ~11 s do llama.cpp, que guarda vários checkpoints e processa só os ~1,5k tokens novos. Ver `benchmark-conversa-longa-2026-09-25.md`. | 1–2 dias |

## 1. Prefill — a maior distância para o llama.cpp

O llama.cpp HIP faz 318 tok/s de prefill frio com 7,6k tokens; o llama.rs, 115 (2,8×). Com 30k o
llama.rs cai para 79 tok/s (a atenção do bloco cresce com o contexto).

| # | O quê | Evidência | Esforço |
|---|---|---|---|
| 1.1 | GEMM para Q5_K e Q6_K no prefill (hoje matvec-COLS) | Com bloco 32 o matvec-COLS deles dobra de 47 para 92 ms por bloco (pressão de registrador), e Q6_K+Q5_K são 34% dos bytes do modelo. Projeção: ~−1,3 ms/token de prefill. | ½–1 dia |
| 1.2 | Tile maior e BK=64 no `mul_mm.comp` | Ainda ~20% do pico INT8 (10 de 51 TOPS no ffn_gate). Cada passo de K tem só 96 dots por thread entre duas barreiras. | 1–2 dias |
| 1.3 | Split-K no GEMM para as formas de saída 5120 | ffn_down/attn_output têm 40 workgroups para 60 CUs: 9% do pico contra 20% das formas largas. | ½ dia |
| 1.4 | Estudar o MMQ do llama.cpp HIP para gfx906 | O HIP do llama.cpp faz o prefill ~2,8× mais rápido no mesmo hardware (ver o benchmark). | pesquisa, ½ dia |

## 2. Decode

| # | O quê | Evidência | Esforço |
|---|---|---|---|
| 2.0 | MTP adaptativo à profundidade | O MTP vale +29% com 1,4k de contexto e **perde 9% com 30k** (20,3 contra 22,2 tok/s): o verify de 3 tokens paga a atenção sobre o contexto inteiro 3 vezes. Desligar a especulação acima de ~20k, ou encurtar para 1 proposta, recupera o decode. | ½ dia |
| 2.1 | `V_DOT2_F32_F16` na atenção (q também em f16) | A atenção nova já usa `v_fma_mix_f32` (f16 na entrada, f32 na soma); o dot2 faria 2 MACs por instrução. Pesa em contexto longo. | ½ dia |
| 2.2 | Kernel Q4_K menos sensível ao clock do núcleo | No pstate `standard` (1316 MHz) o matvec Q4_K lê 617 GB/s (74% do teto) contra 681 no clock máximo: ainda gasta ALU por byte. | 1 dia |
| 2.3 | Fundir as ops pequenas do token | 1.126 ops e 900 barreiras por token; norms, gates e delta-net somam ~8 ms de ~41. | 1–2 dias |

## 3. Térmica e operação

| # | O quê | Evidência | Esforço |
|---|---|---|---|
| 3.1 | Refrigeração da card1 (PCI 05:00) | Roda ~12 °C acima da card2 em repouso; no prefill passa de 100 °C a ~200 W com o DPM automático (o llama.cpp chega a 104 °C). Opções: curva do ventilador (`pwm1`) ou teto de potência (`power1_cap`, ~160 W) — ambas pedem root. | decisão do usuário |
| 3.2 | Relatório de memória no `LLAMA_RS_PROFILE` | Não há VRAM por GPU nem pico de RAM (VmHWM) em nenhum modo de debug; hoje isso sai de scripts avulsos. | 30 min |
| 3.3 | Guardas de VRAM/temperatura/clock como ferramenta própria | Hoje estão em `scripts/bench-conversa/` (atrelados ao benchmark, matam por PID). Juntar num `scripts/monitor-gpu.sh` reutilizável por qualquer teste. | 1 h |

## 4. Correção e robustez (do `analise-repositorio-e-padroes-industria-2026-09-25.md`)

| # | O quê | Estado |
|---|---|---|
| 4.1 | Laço do `llama-cli` (`gerar_streaming_residente`) não confere o ctx antes do `passo_mtp`/`decode` | mesmo defeito do E01, só corrigido no servidor |
| 4.2 | Resto do E05: validar `temperature`/`top_p`, `n > 1`, `response_format`, `model` | só o 400 de contexto estourado foi feito |
| 4.3 | E03: timeouts e limite de cabeçalho no HTTP; `/health` durante geração | aberto |
| 4.4 | E06: limpeza de recursos Vulkan em falha de construção; o `Drop` do backend só destrói parte das pipelines | aberto |
| 4.5 | E09: README ainda diz que não há MTP | aberto |
