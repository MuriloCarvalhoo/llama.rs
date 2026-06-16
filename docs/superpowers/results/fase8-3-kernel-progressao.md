# Fase 8.3 — Progressão do kernel matvec Q8_0 (0.5B, 1× MI50)

**Data:** 2026-06-16 · **Hardware:** 2× AMD Instinct MI50 (gfx906/RADV) · **Modelo:** Qwen2.5-0.5B-Instruct Q8_0
**Bench canônico:** `bench-results/gpu-20260616-162536.md` (`benchmark-gpu.sh`, "Once upon a time", n=64, greedy, seed 42).

## Progressão das alavancas

| Etapa | Commit | tok/s | vs baseline | Resultado |
| --- | --- | --- | --- | --- |
| Baseline (Fase 1D) | — | ~80 | 1.0× | ponto de partida |
| Task 2 — NUM_ROWS=2 | `0b1eff5` | ~84 | 1.05× | bit-idêntico; ganho marginal |
| Task 3 — cache de x[] em LDS | (revertido) | ~52 | 0.65× | **rejeitado** (-38%: 20 KB LDS derruba ocupação; x é pequeno e já cacheado em L1/L2) |
| Task 4 — spike dotPacked4x8 | `d2ec5c0` | — | — | **gate GREEN**: `V_DOT4_I32_I8` (GL_EXT_integer_dot_product) compila e roda em RADV/gfx906 |
| Task 5 — ativação int8 + packed dot | `2649fe5` | ~86 | 1.07× | **correto, 0% de ganho** (tolerância 4.4%, argmax igual). Mantido: compõe com a Task 7. Prova que o kernel **não é ALU-bound** |
| Task 7 — pesos Q8_0 em blocos 36B alinhados | `cc50b5a` | ~85–91 | ~1.1× | loads de 32 bits alinhados (8 uint/bloco) em vez de 32 byte-loads; ganho ~5% |
| **Alvo: llama.cpp 1× MI50** | — | **305.6** | **3.8×** | — |

Bench final (1 run): `llama-rs res-fwd = 82.90` · `llama.cpp 1× MI50 = 305.60 ± 6.68` · `llama.cpp 2× MI50 = 152.63` (layer-split prejudica modelo pequeno, como a spec previu).

## Decisão de gate

**O gate da spec (§6 risco nº2 — "chegar perto dos ~314 tok/s no 0.5B antes de multi-GPU") NÃO foi atingido.** Estamos em ~85–91 tok/s (~0.28× do llama.cpp 1× MI50). A Fase 2 (row-split) **permanece bloqueada**.

## Conclusão técnica (o que aprendemos)

As 4 alavancas de kernel planejadas/derivadas deram ganhos marginais e **se esgotaram em ~91 tok/s**. As evidências cruzadas mostram que o gargalo **não** é o que o plano supôs:

- **Não é ALU-bound:** o packed dot `V_DOT4_I32_I8` (Task 5) não moveu nada.
- **Não é byte-load de peso isoladamente:** alinhar para loads de 32 bits (Task 7) deu só ~5%.
- **Não é banda saturada:** ~91 tok/s está a ~5% do teto de ~1 TB/s para os pesos do 0.5B.
- **Não é o overhead de submit:** o command-buffer único/token (Fase 1D) já eliminou isso.

O suspeito restante é **estrutural**: baixa ocupação / paralelização do matvec — 1 workgroup de 64 lanes por ~2 linhas de saída, com redução `subgroupAdd` por linha. Adjacent lanes ainda leem pesos com passo de bloco (não há coalescência verdadeira entre lanes), e a granularidade de paralelismo é pequena para o MI50 (120 CUs × 4 waves).

## Próximo passo recomendado (fora desta fase)

Antes de mais tweaks incrementais, **fazer profiling de hardware** (rocprof / contadores de ocupação e throughput de memória do matvec) para localizar o gargalo real, e então **rearquitetar a paralelização do matvec** (ex.: múltiplos subgroups por workgroup, particionamento da dimensão K com redução em árvore, ou o esquema de tiling do llama.cpp `mul_mat_vec`), em vez de continuar ajustando o kernel atual. Reavaliar com o usuário se a Fase 3 deve continuar com um redesenho ou se a estratégia muda.

## Observação colateral (pré-existente, fora de escopo)

Os testes `resident_gpu_decode_matches_cpu_ref` e `forward_gpu_real_matches_f32_cpu_reference` falham em master (token-lixo 89012) **antes** desta fase (verificado em `2649fe5` e no commit pai) — caminho `decode_one_gpu_owned`, não relacionado às mudanças da Fase 3. Merece investigação própria.
