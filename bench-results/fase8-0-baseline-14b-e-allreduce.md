# Fase 8.0 — Baseline 14B + decisão de all-reduce

**Data:** 2026-08-13 · **Hardware:** 2× AMD Instinct MI50 / Radeon Pro VII (gfx906, RADV), 16 GiB cada
**Modelo:** `qwen2.5-14b-instruct-q8_0` (14.62 GiB, 48 camadas, n_embd=5120, n_ff=13824)
**llama.cpp:** `d9da72a9b` (937), build Vulkan out-of-tree em `build-vulkan/`

---

## 1. Baseline — o número a bater

`llama-bench -ngl 99 -p 0 -n 32..64` (decode puro, batch-1):

| Engine / config | tok/s |
|---|---|
| **llama.cpp Vulkan — 1× MI50** | **40.59** ← **alvo** |
| llama.cpp Vulkan — 2× MI50 layer-split | 27.34 ± 0.43 |
| llama.cpp Vulkan — 2× MI50 row-split | **não suportado** |
| llama.cpp ROCm — 2× MI50 layer-split | 35.92 |
| llama.cpp ROCm — 2× MI50 row-split | falha ao carregar |

### Dois achados que reorientam a Fase 2

**(a) Row-split não carrega neste hardware — em nenhum backend.** Erro literal:

```
llama_model_load: error loading model: device Vulkan0 does not support split buffers
```

> **Correção (2026-08-14).** A leitura original disto — "o Vulkan do llama.cpp não implementa
> row-split, logo é espaço vazio a ocupar" — estava **errada** em duas frentes:
>
> 1. O `-sm row` existe, foi **deprecado** em favor de `-sm tensor`, e o tensor-parallel
>    backend-agnóstico foi mergeado em abril/2026 (PR #19378).
> 2. O erro não é falta de implementação: é o **hardware**. O mesmo `-sm row` falha no ROCm desta
>    máquina com `device ROCm0 does not support split buffers` / `NO_PEER_COPY=1`
>    (`/home/murilo/llama.cpp/ORNITH-GFX906-NOTES.md`). Medimos a causa depois: não há P2P de VRAM
>    entre estas MI50 (ver §2).
>
> O que de fato roda multi-GPU aqui é **layer-split**, e é ele que viabiliza os modelos de 20–28 GiB.

**(b) O 14B cabe em 1 MI50, e 1 GPU é MAIS RÁPIDA que 2 em layer-split** (40.59 vs 27.34).
Batch-1 com layer-split deixa só uma GPU ativa por vez e ainda paga sincronização — exatamente o
que a spec §3 previa. Consequência: a premissa da spec ("modelo grande que não cabe em 1 GPU")
**não vale para o 14B**; ele é campo de prova, e o alvo honesto é o número single-GPU (40.59).

### Teto físico (contexto para a meta)

> **Ressalva (2026-08-14):** "decode batch-1 é memory-bound" vale para **Q8_0** neste hardware
> (nosso matvec lê a 717 GB/s, perto do teto). **Não vale para K-quants**: o llama.cpp mede o
> Qwen3.6-27B denso a 42 ms/token contra um teto de banda de ~15 ms, ou seja **compute-bound** —
> a Radeon Pro VII tem HBM2 rápida e pouco compute (sem matrix cores), e K-quants custam mais ALU
> para desquantizar. Isso enfraquece a aposta em Q4_K (ver `docs/estrategia-inferencia-mi50.md`).

Decode batch-1 é memory-bandwidth bound. 14.62 GiB de pesos lidos por token:

- 1 GPU @ ~1 TB/s → ~15.7 ms/token → teto ~64 tok/s. llama.cpp entrega 40.59 = **~63% do teto**.
- 2 GPUs em row-split real → 7.3 GiB por GPU em paralelo → teto ~128 tok/s. Na mesma eficiência
  de 63%, ~80 tok/s. **É essa a janela: ~2× sobre o melhor llama.cpp.**

---

## 2. Spike de all-reduce MI50↔MI50 (risco nº 1 da spec)

Medido em `crates/llama-vulkan/src/spike.rs`, payload de 5120 f32 (20 KB = a stream residual
de 1 token do 14B), 96 all-reduces/token (48 camadas × 2).

### O risco era real — no caminho ingênuo

| Caminho | µs/transferência | ms/token | Teto |
|---|---|---|---|
| host-bounce (piso, só map/unmap) | 101.87 | 9.78 | 102 tok/s |
| device-local, 1 via (otimista) | 247.44 | 23.76 | **42 tok/s** |
| device-local, 2 vias (all-reduce real) | 247.44 | 47.51 | **21 tok/s** |

Com um submit+fence por transferência, o all-reduce sozinho **já ficaria abaixo do alvo de 40.59**
— o design morreria aqui.

### Decomposição: o custo é sincronização, não banda

| Medida | µs |
|---|---|
| submit + fence de command buffer **vazio** | **63.55** |
| copy device→host de 4 B | 57.65 (sync ≈ 100%) |
| copy device→host de 20 KB | 79.30 (sync ≈ 80%) |
| copy device→host de 320 KB | 148.54 (sync ≈ 43%) |
| map+copy+unmap de 20 KB (sem GPU) | 3.81 |

~63 µs de cada transferência é puro submit+fence. A transferência de 20 KB em si custa ~16 µs.

### O batching resolve

96 transferências de um token gravadas num **único** command buffer (1 submit, 1 fence):

| Caminho | ms/token | Teto |
|---|---|---|
| ingênuo (96 submits) | 23.755 | 42 tok/s |
| **batchado (1 submit)** | **0.121** | **8251 tok/s** |

**196× de diferença, toda em sincronização.** O volume de dados (~2 MB/token) é irrelevante,
como a spec §5 estimava.

### Capacidade peer-to-peer — disponível

```
GPU0 (RADV VEGA20): external_memory=true external_memory_fd=true dma_buf=true external_semaphore_fd=true
GPU1 (RADV VEGA20): external_memory=true external_memory_fd=true dma_buf=true external_semaphore_fd=true
```

`VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf` e `VK_KHR_external_semaphore_fd`
presentes nas duas GPUs → dá para importar memória da outra GPU e sincronizar entre devices
**sem round-trip pelo host**.

---

## 3. Decisão para a Fase 2

**Mecanismo: all-reduce gravado dentro do command buffer do token, com semáforos externos
(`VK_KHR_external_semaphore_fd`) entre os devices; memória compartilhada via
`VK_KHR_external_memory_fd`/dma_buf quando possível, host-staged como fallback.**

Regra que decorre da medição: **nunca um submit/fence por all-reduce.** O custo real do
tensor-parallel não é o volume de dados — é o número de sincronizações. O design tem de manter
1 submit por token por GPU, com as dependências expressas por barriers/semáforos.

Ressalva honesta sobre a medição de batching: ela foi feita intra-GPU (GPU0 device→host), então
prova que o *trabalho de transferência* é trivial e que o sync domina. A latência de
sincronização **cross-device** com semáforos externos ainda não foi medida — é o próximo spike,
mas as extensões necessárias existem.

---

## 4. Bloqueios do llama-rs para rodar o 14B (levantados nesta fase)

1. **GGUF split não suportado** — o 14B vem em 4 arquivos; `gguf::GgufFile::parse` recebe um só.
2. **`MATVEC_MAX_BLOCKS = 160`** (`resident_forward.rs:400`) rejeita `n_in > 5120`. O 14B tem
   `n_ff = 13824` → `ffn_down` seria recusado. Exige tiling da dimensão K no shader matvec.
3. **`std::fs::read` do modelo inteiro** (`runner.rs:152`) → 15 GB em RAM, mais a cópia de
   `GpuRawWeights` e o `token_embd` dequantizado para f32 (~3.1 GB).
