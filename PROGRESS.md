# Progresso — llama-rs

**Última atualização:** 2026-08-13 (sessão Fase 0 + desbloqueios do 14B)

## Resumo em uma frase

CPU: pipeline completa e bit-exact contra o llama.cpp. GPU (Vulkan, 2× AMD MI50): decode residente em **1 GPU** é numericamente correto mas ~4× mais lento que o llama.cpp; a **Fase 0 foi concluída** (baseline do 14B + spike de all-reduce, risco nº1 resolvido) e vários bloqueios do 14B foram removidos, mas o **row-split real entre as 2 GPUs** — o objetivo central — ainda não foi implementado, e o 14B ainda não carrega por consumo de RAM no lado CPU.

---

## Sessão 2026-08-13 — o que mudou

### Fase 0 concluída (era o próximo passo nº1)

- **Baseline llama.cpp Vulkan no 14B** (`bench-results/fase8-0-baseline-14b-e-allreduce.md`):
  **1× MI50 = 40.59 tok/s** (o alvo), 2× layer-split = 27.34, 2× row-split = **não suportado**
  (`device Vulkan0 does not support split buffers`) — confirma a premissa fundadora do projeto.
  O 14B **cabe em 1 GPU**, e 1 GPU é mais rápida que 2 em layer-split.
- **Spike de all-reduce** (`crates/llama-vulkan/src/spike.rs`): o risco nº1 da spec era real no
  caminho ingênuo (247 µs/transferência → teto de 42 tok/s), mas **~63 µs de cada transferência é
  submit+fence**; com as 96 transferências de um token num único command buffer o custo cai para
  **0.121 ms/token (teto ~8250 tok/s)** — 196× melhor. P2P disponível
  (`external_memory_fd` + `dma_buf` + `external_semaphore_fd` nas duas GPUs).
  **Regra que decorre:** nunca um submit/fence por all-reduce.

### Os dois testes que falhavam: não eram bug

`resident_gpu_decode_matches_cpu_ref` e `forward_gpu_real_matches_f32_cpu_reference` falhavam com
"token-lixo 89012". Causa real: o commit `2649fe5` passou a **quantizar a ativação para int8** no
shader, mas os testes continuaram comparando contra uma referência de ativação **f32**. O kernel
diverge dessa referência em **0.1389%** (medido), o bastante para virar o argmax de um prompt de um
único BOS. Com uma referência que modela a matemática real do shader (`cpu_ref_q8_0_int8act`), a GPU
produz **exatamente o mesmo token**. Testes corrigidos — **22/22 verdes**.

### Bloqueios do 14B removidos

| Bloqueio | Correção |
|---|---|
| `n_ff=13824` rejeitado (LDS de 160 blocos) | **Tiling da dimensão K** no `q8_0_matvec.comp` — janelas de 160 blocos, `n_in` livre, LDS inalterado (5.6 KB). Validado com teste novo em 1, 1-na-borda e 3 janelas (erro 1e-6) |
| `head_dim=128` rejeitado (shader assumia ≤ 64) | `attention.comp` distribui até 4 dimensões por lane (head_dim ≤ 256); bit-idêntico para head_dim ≤ 64 |
| `fs::read` do modelo (15 GB de RAM anônima) | `memmap2` no `llama-cli` — vira page cache recuperável |
| KV-cache dimensionado pelo `context_length` do GGUF (131072 → **51 GB de VRAM**) | flag `--ctx` (padrão 4096), limitada ao do modelo |
| `MPOL_BIND` no nó NUMA 0 matando o processo | Só aplicado no caminho CPU — no caminho GPU restringia as alocações a metade da RAM |
| Tabela de embedding f32 na VRAM (3.1 GB para ler 1 linha/token) | Mantida no host; sobe só a linha do token (~20 KB, ~4 µs) |
| `GpuRawWeights` copiando todos os pesos (`raw.to_vec()`, 14.6 GB) | Passou a **emprestar** do mmap (`GpuRawWeights<'a>`) |

### O 14B ainda não roda — bloqueio restante (medido)

Com `--gpu-resident`, o processo ainda é morto por OOM. Causa dominante já localizada:
`Weights::from_gguf` chama `tensor_raw_repack` para **todo** peso, que faz `raw.to_vec()` (cópia
integral) e depois `repack_q8_0_8rows` (**segunda** cópia, layout `block_q8_0x8` de CPU) antes de
descartar a primeira. São **~14.6 GB de pesos de CPU que o caminho GPU-residente nunca usa**, mais
`token_embd` dequantizado para f32 (3.11 GB) e uma segunda cópia dele em `ResidentState` (3.11 GB).

**Próximo passo concreto:** tornar o repack de CPU preguiçoso (ou não construir os pesos de CPU no
caminho GPU) e parar de materializar `token_embd` em f32. Ver o documento de pesquisa
`docs/rust-memoria-e-desempenho.md` para as técnicas avaliadas.

---

## O que funciona hoje

- **Pipeline CPU completa**: parser GGUF v3, tokenizer SPM/BPE, forward f32 (RMSNorm/RoPE/GQA/SwiGLU/KV-cache), quantização Q8_0, sampling, `llama-cli`. Bit-exact contra o llama.cpp.
- **Decode residente em 1 GPU** (`ResidentForward`, `--gpu-resident`): pesos, KV-cache e ativações residentes em VRAM; 1 command buffer/token; bit-exact vs CPU no Qwen2.5-0.5B. Correto, mas lento: **~70–91 tok/s** vs **~305–334 tok/s** do llama.cpp na mesma MI50 (~0.25–0.3×).
- **`--gpu` (row-split ingênuo)**: `DualGpuMatmul` (`crates/llama-vulkan/src/dual_gpu.rs`) divide `n_out` ao meio via `rayon::join`, mas chama o caminho **não-residente** (`dispatch_inner`) — reenvia todos os pesos e recria a pipeline a cada matvec. Funciona, mas é ~145× mais lento que o llama.cpp 2×MI50 (**2.14 vs 143.66 tok/s**). É um protótipo de correção pré-Fase 8, não a arquitetura final.
- **Filtro de hardware**: só GPUs AMD (`vendor_id == 0x1002`) são enumeradas (`crates/llama-vulkan/src/device.rs`). NVIDIA/Tesla K80 é ignorada por decisão deliberada (ver seção própria abaixo).

## O que NÃO existe ainda

- **Tensor-parallel row-split "Megatron" (Fase 2 da spec ativa)** — column-parallel em q/k/v e ffn gate/up, row-parallel em attn-out e ffn-down, com 2 all-reduces por camada entre as MI50. Não implementado. O `DualGpuMatmul` atual não tem residência, nem persistência, nem all-reduce — é um split de linhas de um único matmul, síncrono.
- **Fase 0 (baseline 14B + spike de latência de all-reduce MI50↔MI50)**: planejada em detalhe (`docs/superpowers/plans/2026-06-16-fase8-0-baseline-allreduce-spike.md`) mas **nunca executada**. Não existe `crates/llama-vulkan/tests/allreduce_spike.rs` nem `bench-results/fase8-0-allreduce-decisao.md`. Sem NVLink entre as MI50, essa é a medição que decide se o all-reduce por token é sequer viável em latência — pré-requisito barato que ainda falta.
- **Correção de um bug conhecido**: os testes `resident_gpu_decode_matches_cpu_ref` e `forward_gpu_real_matches_f32_cpu_reference` (`crates/llama-vulkan/tests/integration.rs`) já geravam token divergente ("token-lixo 89012") no caminho `decode_one_gpu_owned` **antes** da Fase 8.3 (registrado, nunca investigado — a nota ficou num doc que foi removido do HEAD, ver abaixo).

## Tesla K80 — por que está fora de escopo

- `crates/llama-vulkan/src/device.rs:17,86-88` filtra por vendor AMD incondicionalmente; a K80 nunca é enumerada. Não é limitação de driver — é decisão de código.
- Motivo técnico: os shaders foram escritos para **wave64** (subgroup nativo do gfx906/MI50); a K80 (Kepler, sm_37) é **wave32** — precisaria de uma segunda família de kernels, num chip mais lento que uma única MI50.
- Mesmo se reativada, a K80 não poderia participar de **tensor-parallel simétrico** com as MI50 (sem peer-to-peer entre vendors distintos — o bounce teria que passar sempre pelo host). No máximo serviria como worker isolado de **layer-split**, uma arquitetura diferente da que está em desenvolvimento.
- O `README.md` até esta atualização ainda anunciava a K80 como alvo — corrigido junto com este documento.

---

## Benchmarks (Qwen2.5-0.5B-Instruct Q8_0, greedy, seed 42, 64 tokens)

Último resultado commitado (`bench-results/gpu-20260616-022226.md`):

| Engine / GPUs                         | tok/s            |
|----------------------------------------|------------------|
| llama.cpp — 1× MI50                    | 333.96 ± 6.98    |
| llama.cpp — 2× MI50 (layer-split)      | 143.66 ± 0.79    |
| llama-rs — 1× MI50 (resident, Fase 1B) | 12.39            |
| llama-rs — 1× MI50 (res-fwd, Fase 1D)  | 69.96            |
| llama-rs — 2× MI50 (row-split ingênuo) | 2.14             |

Após a Fase 8.3 (otimização de kernel, ver diagnóstico abaixo), o 1× MI50 residente chegou a **~85–91 tok/s** — número reportado em `docs/superpowers/results/fase8-3-kernel-progressao.md`, que foi removido do HEAD (recuperável via `git show 5452216~1:docs/superpowers/results/fase8-3-kernel-progressao.md`); o arquivo de benchmark bruto correspondente (`gpu-20260616-162536.md`) nunca chegou a ser commitado em `bench-results/`.

## Diagnóstico do gargalo single-GPU (Fase 8.3 — gate não atingido)

Meta: chegar perto de ~314 tok/s (1× MI50) antes de seguir para row-split (regra da spec: "se a Fase 1 não chegar perto do llama.cpp, o problema é kernel — resolver antes de multi-GPU"). Resultado: **~91 tok/s (~0.28×)**, gate não atingido, Fase 2 (row-split) permanece bloqueada pela própria regra que a antecede.

Alavancas testadas, eliminadas como causa uma a uma:

| Alavanca | Resultado |
|---|---|
| Specialization constants + NUM_ROWS=2 | ganho marginal (~5%) |
| Cache de `x[]` em shared memory (LDS) | **revertido**: -38%, derruba ocupação |
| Ativação int8 + `dotPacked4x8` (packed dot) | gate técnico OK, mas **0% de ganho** → não é ALU-bound |
| Pesos Q8_0 em blocos alinhados de 32 bits | ~5% → não é padrão de load isolado |

Conclusões cruzadas: não é ALU-bound, não é byte-load, não é banda saturada (~91 tok/s já está a ~5% do teto de ~1 TB/s para este modelo), não é overhead de submit (resolvido na Fase 1D). **Suspeito remanescente: baixa ocupação/paralelização estrutural do matvec** — 1 workgroup de 64 lanes cobrindo poucas linhas de saída, sem coalescência real entre lanes adjacentes.

Próximo passo recomendado no próprio documento (nunca executado): **profiling de hardware (`rocprof`)** para localizar o gargalo real, seguido de um **redesenho do particionamento do matvec** (múltiplos subgroups por workgroup, particionamento em árvore da dimensão K, ou tiling à la `mul_mat_vec` do llama.cpp) em vez de continuar ajustando o kernel atual.

---

## Próximos passos (ordem recomendada)

1. **Rodar a Fase 0 do row-split** — spike de latência de all-reduce MI50↔MI50 (host-bounce e device-local) + baseline do llama.cpp no 14B (layer-split e row-split, tomar o melhor). Decide o mecanismo (host-staged vs peer-to-peer) e se o teto de latência inviabiliza o design Megatron antes de escrever código de produção. Nenhum trabalho de Fase 2 deveria começar sem isso — é o pré-requisito mais barato e mais crítico que falta.
2. **Investigar o bug de `decode_one_gpu_owned`** (token divergente "89012") antes de construir mais em cima desse caminho — é uma regressão de correção conhecida e sem dono.
3. **Implementar a Fase 2 — tensor-parallel row-split real** (layout Megatron: §5 da spec ativa), substituindo o `DualGpuMatmul` ingênuo por um caminho residente com all-reduce, usando o mecanismo decidido no passo 1.
4. **Fase 3 — redesenho do kernel matvec** (multi-subgroup/workgroup, tiling), só depois de ter row-split funcionando — a otimização isolada de kernel (Fase 8.3) já mostrou retorno marginal sem mudar a arquitetura de paralelização.
5. **Manter README e este documento sincronizados a cada mudança de escopo** — a divergência entre o README e as specs mais recentes (K80 anunciada como alvo quando já estava excluída há duas fases) foi a causa da rodada de revisão que gerou este documento.

---

## Referências

- Spec ativa: `docs/superpowers/specs/2026-06-16-fase8-decode-gpu-persistente-rowsplit-design.md`
- Design geral / reorientação multi-GPU: `docs/superpowers/specs/2026-06-03-llama-rs-rewrite-design.md`
- Histórico de fases: `docs/superpowers/plans/2026-06-*.md`
- Diagnóstico completo da Fase 8.3 (fora do HEAD): `git show 5452216~1:docs/superpowers/results/fase8-3-kernel-progressao.md`
