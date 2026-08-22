# Plano geral: Qwen3.8-27B acima de 50 tok/s, com carga e respostas rápidas

> **Para execução:** cada frente tem plano próprio nesta pasta. Executar uma frente por
> vez, na ordem abaixo, seguindo o protocolo de medição de
> `docs/decode-por-configuracao.md` (LLAMA_RS_SPLIT fixo, `TOTAL GPU` para comparar
> shader, `numactl --interleave=all` sempre).

**Meta:** decode do Qwen3.8-27B Q4_K_M ≥ 50 tok/s nas 2× MI50, funcionando **com e sem
MTP** (sem MTP continua sendo o caminho padrão), carga do modelo em segundos e não em
dezenas de segundos, e primeira resposta rápida mesmo com prompt grande.

**Estado em 2026-08-21:** o `qwen35` já roda de ponta a ponta na GPU — 22,3 tok/s no
protocolo padrão, 26,9 com prompt curto, contra 17–23 do llama.cpp no mesmo hardware.
MTP tem as fases A (carga do bloco) e B (cabeça na CPU) prontas, com **aceitação medida
de 60,9%**. O que falta está mapeado, não é pesquisa.

## Correções de premissa (verificadas em 2026-08-21)

1. **`~/llama.cpp-gfx906` não é um fork otimizado para gfx906.** É o fork do skyne98
   (dspark, checkpoint-db); zero commits de kernel gfx906 em 10.602 commits. Não há nada
   para portar de lá. O material real de gfx906 está no **upstream**: os 11 pontos de
   tuning `AMD_GCN` em `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (ver plano do decode).
2. **O llama.cpp HIP está compute-bound na MI50** (2,8× acima do roofline de banda no 27B
   denso — `~/llama.cpp/ORNITH-GFX906-NOTES.md:218-229`). O llama-rs vence porque os
   kernels dele sustentam 500–717 GB/s. Corolário: **qualquer otimização que troque banda
   por compute é suspeita** — as quatro que já falharam (nwarps, K80, etc.) falharam por
   isso.
3. `docs/qwen35-arquitetura.md` é o doc de projeto **pré-implementação**; a seção "o que
   falta" dele está obsoleta (tudo implementado). Ver housekeeping abaixo.

## A física do alvo

50 tok/s = 20,0 ms/token. Os matvec do Q4_K_M leem 16,38 GB/token; a 717 GB/s (pico
medido) custam 22,85 ms sozinhos. **Não existe 50 tok/s token-a-token neste modelo e
quantização** — o caminho é multiplicar tokens por passo com o MTP do próprio modelo:

```
tok/s = (1 + aceitação) / t_passo        aceitação medida (greedy, n=1): 60,9%
```

### A conta honesta do passo com MTP

A fórmula simples (`base × 1,61`) ignora o custo do próprio passo MTP. O passo real é:

| componente | ms | fonte |
|---|---:|---|
| forward de verify (batch 2) | GPU_base + ~1,7 | +1,35 da recorrência serial ×2, +0,3 da 2ª coluna de logits |
| cabeça MTP (289 MB) | 0,40 | medido, `docs/mtp-implementacao.md:37` |
| snapshot do estado (151 MB + janela 6 MB) | ~0,45 | estimado a 717 GB/s |
| host (gravação, submit, sampling) | ~2,4 | 44,8 ms wall − 42,4 GPU no baseline |

Alvo: `t_passo ≤ 1,609/50 = 32,2 ms` ⇒ **GPU_base ≤ ~27,5 ms** ⇒ base sem MTP ≈ 33
tok/s. Isso é mais duro que os 31,1 da conta antiga. Três alavancas fecham o gap, em
ordem de confiança:

1. **Base**: TOTAL GPU 41,35 → ≤30 ms (plano do decode). Chega a base ~31 e MTP ~46-48.
2. **Aceitação encadeada (n=2)**: se a cabeça, realimentada com a própria previsão,
   acertar ≥40% no 2º token, `tokens/passo = 1 + a₁ + a₁a₂ ≈ 1,9` e o orçamento de passo
   sobe para 38 ms — o alvo passa com folga. **Medir na CPU antes de implementar**, como
   a fase B fez (barato; o llama.cpp realimenta o `t_h_nextn` da própria cabeça —
   `common/speculative.cpp:1552-1701`). O fracasso do `n-max 2` no llama.cpp (−10%) mediu
   a implementação dele, não a cabeça: aqui o verify em batch custa quase o mesmo que 1
   token.
3. **Plano B — Q3_K_M**: ~13 GB/token ⇒ teto matvec 55 tok/s, base ~39. Exige shader
   Q3_K (que também destrava o Qwen3-14B) e validação de PPL. Só se 1+2 não bastarem.

### Observação fora de escopo deste plano

O caminho *garantido* para >50 tok/s neste hardware é MoE: o Qwen3.6-35B-A3B lê 1,98
GB/token (teto 103 tok/s com as outras ops) e o llama.cpp já faz 77,1 nele. Suportar
`qwen35moe` é um projeto maior (FFN MoE + tensor `ssm_ba` fundido) e não é o pedido —
fica registrado como alternativa se o 27B denso esgotar as alavancas.

## As cinco frentes, em ordem

| # | plano | entrega | pré-requisito |
|---|---|---|---|
| 0 | *(sem arquivo)* fechar o diff pendente de `resident_forward.rs` | geometria própria do matvec de prefill (8,9 ms vs 24,1 no bloco) + perfil dos dois planos + correção do dispatch parcial do Q5_K | — |
| 1 | [`2026-08-21-decode-base.md`](2026-08-21-decode-base.md) | base sem MTP: TOTAL GPU 41,35 → ≤30 ms (~31-33 tok/s) | 0 |
| 2 | [`2026-08-21-mtp-fases-c-e.md`](2026-08-21-mtp-fases-c-e.md) | MTP ligável por flag: verify em batch de 2, rollback do estado, cabeça na GPU → ≥46 tok/s, e ≥50 com a frente 1 completa ou n=2 | 0 (independe de 1) |
| 3 | [`2026-08-21-prefill-e-respostas.md`](2026-08-21-prefill-e-respostas.md) | prefill 73 → ≥180 tok/s (paridade HIP): delta-net multi-token num dispatch, batch >8, GEMM com LDS | 0 |
| 4 | [`2026-08-21-carga-rapida.md`](2026-08-21-carga-rapida.md) | carga do 27B: ~35 s → ≤6 s com page cache quente | — |

Frentes 1+2 são o alvo de 50 tok/s; frente 3 é "respostas rápidas" (o opencode manda
~27k tokens no primeiro turno — hoje 11 minutos); frente 4 é qualidade de vida e ciclo
de desenvolvimento. 2 e 3 compartilham infraestrutura (logits por token do bloco,
delta-net multi-token), por isso a ordem 2→3 reaproveita trabalho.

## Housekeeping (fazer junto com a frente 0)

- [x] `docs/qwen35-arquitetura.md`: marcar no topo como doc de projeto histórico e
      riscar a seção "O que falta no llama-rs" (tudo feito).
- [x] `crates/llama-vulkan/src/resident_forward.rs:1059-1060`: comentário obsoleto diz
      que delta-net não tem plano de decode; a linha 1084 trata `MixerRaw::Delta`.
      *(sumiu na reescrita do merge das frentes)*
- [x] Tokenizer: `tokenizer.ggml.pre` não é lido em lugar nenhum (grep confirma). O
      Qwen3.8 usa `pre = "qwen35"`, cuja regex difere da do Qwen2 que usamos: letras
      consomem combining marks (`[\p{L}\p{M}]+` — `~/llama.cpp/src/llama-vocab.cpp:382-388`).
      Português NFC quase não sofre, mas é divergência real de tokenização. Ler a chave em
      `crates/llama-tokenizer/src/vocab.rs:72-80` e escolher a variante da regex em
      `bpe.rs:54`; caso de teste com combining mark em `refs/tokens.json` via oracle.

## Referências que os planos citam

- Perfil e medições: `docs/decode-por-configuracao.md` (baseline 42,40 ms; matvec = 77%;
  q4k a 506 GB/s de 717).
- MTP: `docs/mtp-implementacao.md` (fases A/B prontas, C/D/E abertas) e
  `docs/mtp-e-k80.md` (lição do n-max).
- Implementação de referência: `~/llama.cpp/src/models/qwen35.cpp` (grafo),
  `~/llama.cpp/src/models/delta-net-base.cpp` (snapshots de estado),
  `~/llama.cpp/common/speculative.cpp:1281-1717` (driver MTP),
  `~/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp` (scan
  multi-token com estado em registrador).

## Resultado (2026-08-21, pós-merge das frentes)

Medido no protocolo padrão (`docs/decode-por-configuracao.md`), contexto curto:

| meta | alvo | medido |
|---|---|---|
| frente 1 — base sem MTP | ≤30 ms | 40,2 ms (knobs LDS/rope medidos e piores; sobrou a geometria do matvec) |
| frente 2 — MTP por flag | ≥46 tok/s | 31,4 tok/s com n=1 (+44 %); **34,5–36 com n=2 encadeado** |
| frente 3 — prefill | ≥180 tok/s | 92 tok/s (batch 24 + GEMM, −42 % vs batch 8) |
| frente 4 — carga warm | ≤6 s | **4,5–5,0 s** ✓ |
| aceitação encadeada n=2 | ≥40 % para valer | **41,7 %** no experimento; em geração real, 2,20 tok/passo |

Atualização 2026-08-22: o n=2 encadeado foi implementado (verify de 3 tokens, 56,2 ms
de GPU — 1,4 ms a mais que o de 2) e ligado também no motor do servidor: **36,2 tok/s
greedy ponta a ponta** (32,6 com temp 0,8). O que falta para 50: a base de 40 ms não
caiu (matvec ainda a 506 GB/s no Q4_K) — com ela em ≤30 ms, os mesmos 2,20 tok/passo
dariam ~48–52. Em contexto 9,3k o MTP rende 22,1 tok/s — a atenção longa segue sendo
o gargalo que nenhuma frente atacou.
