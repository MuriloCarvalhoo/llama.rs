> # ⚠️ SUPERADA EM 2026-08-14 — NÃO USAR COMO ROADMAP
>
> A tese central deste documento (**tensor-parallel row-split entre as 2 MI50**) foi **refutada por
> medição**. Mantido só como histórico da Fase 1, que foi implementada e continua válida.
>
> | Afirmação original | O que a medição mostrou |
> |---|---|
> | Row-split dá banda efetiva de ~2 TB/s, teto de ~2× | Não há **P2P de VRAM** entre estas placas: `OPAQUE_FD` falha no import, `DMA_BUF` importa como host-visible e lê a **10.2 GB/s** contra 717 GB/s locais |
> | "Comunicação é trivial: ~2 MB/token" | O volume é trivial; o custo é **latência**. 96 all-reduces × 59.3 µs = **5.69 ms/token**, contra 10.8 ms economizados → ~38 tok/s, abaixo do llama.cpp |
> | "O Vulkan do llama.cpp é fraco por não ter row-split" | `-sm row` falha **também no ROCm** (`NO_PEER_COPY=1`) — é limite do gfx906, não do backend |
> | "Modelo grande não cabe em 1 GPU → row-split" | O caminho correto para >16 GiB é **layer-split**: 1 sincronização por token (0.06 ms) em vez de 96 |
>
> **Roadmap atual:** [`docs/estrategia-inferencia-mi50.md`](../../estrategia-inferencia-mi50.md) §7
> e [`PROGRESS.md`](../../../PROGRESS.md).

# Fase 8 — Decode GPU persistente + tensor-parallel row-split (bater o llama.cpp multi-GPU)

- **Data:** 2026-06-16
- **Status:** design aprovado (brainstorming) → pronto para `writing-plans`
- **Hardware alvo:** 2× AMD Instinct MI50 (gfx906 / RADV), 16 GB HBM2 cada (~1 TB/s/GPU)
- **Predecessor:** Fase 7 (`2026-06-10-fase7-vulkan-mi50-forward-e2e`) — decode GPU validado bit-a-bit, porém ~145× mais lento que o llama.cpp.

---

## 1. Diagnóstico (estado atual)

Benchmark de referência (`bench-results/gpu-20260615-231932.md`, modelo `qwen2.5-0.5b-instruct-q8_0`):

| Engine / GPUs        | decode tok/s |
|----------------------|--------------|
| llama.cpp — 1× MI50  | 314.14 ± 7.17 |
| llama.cpp — 2× MI50  | 143.85 ± 0.28 |
| llama-rs  — 2× MI50  | **2.16**     |

Razão atual: **0.015×** (≈145× mais lento).

**Causa raiz** (`crates/llama-vulkan/src/matmul.rs::dispatch_inner`): o decode é um protótipo que validou *correção numérica*, mas nada é persistente. Em cada um dos **169 matvecs por token** (7 por camada × 24 camadas + `output`), a função faz do zero:

1. **Re-upload de todos os pesos Q8_0** para a VRAM (`GpuTensor::upload_q8_0`) — ~530 MB/token só nesse modelo pequeno;
2. **Recria a `ComputePipeline`** (recompila o shader) a cada matvec;
3. Descriptor pool/set + command buffer novos a cada chamada;
4. **`queue_wait_idle`** — sincronização total da GPU após *cada* matvec (zero overlap);
5. **Readback ao host** + ping-pong CPU↔GPU (RMSNorm/RoPE/attention/SwiGLU rodam na CPU em `model.rs`);
6. **Destrói todos os recursos** Vulkan no fim.

A infra de pesos persistentes (`crates/llama-vulkan/src/model_gpu.rs::GpuWeights`) **existe mas não está conectada** ao decode (comentário: "delega ao Model CPU"). O trabalho de correção está feito; falta a arquitetura de performance.

---

## 2. Objetivo e definição de sucesso

**Tornar o llama-rs mais rápido que o llama.cpp na configuração onde o multi-GPU é vantagem genuína: um modelo grande que não cabe em 1 GPU, decode batch-1, nas 2 MI50 com row-split.**

Critério de sucesso final: **decode tok/s do llama-rs (2× MI50) > melhor número do llama.cpp (2× MI50)** no mesmo modelo grande (14B Q8_0), com saída numericamente correta.

Por que não o 0.5B em 1 GPU: para um modelo pequeno em batch-1, dividir entre 2 GPUs *perde* (143 < 314 tok/s) — comunicação domina e cada GPU faz metade do trabalho memory-bound. Row-split só ganha quando o modelo grande força o uso das 2 GPUs. O 0.5B é usado apenas como **campo de prova local da Fase 1**.

---

## 3. Princípio físico (a tese)

Decode batch-1 é **memory-bandwidth bound**: tempo/token ≈ (bytes de peso lidos) / (banda efetiva) + overhead. Os FLOPs são irrelevantes (1 vetor de ativação).

- O llama.cpp, por padrão, faz **layer-split** (`-sm layer`): GPU0 = camadas `0..N/2`, GPU1 = `N/2..N`. Em batch-1, **só 1 GPU está ativa por vez** → banda efetiva = **1 TB/s**. A outra GPU ociosa.
- **Tensor-parallel row-split** faz as 2 GPUs lerem **suas metades de cada matriz em paralelo** → banda efetiva ≈ **2 TB/s** → teto ~2× sobre layer-split, *se* a comunicação for barata.

É exatamente aqui que está a janela para ganhar — e onde o README aponta que o Vulkan do llama.cpp é fraco (sem row-split eficiente).

---

## 4. Arquitetura alvo

Cinco mudanças, todas necessárias para sair dos 2 tok/s:

### 4.1 Residência persistente em VRAM (por GPU)
- Upload de cada matriz de peso **uma vez** no load (bytes Q8_0 raw), mantendo os handles `GpuTensor`. Base já existe: `GpuWeights` + sub-alocador VMA-style em chunks de 1.5 GB (`alloc.rs`).
- **KV-cache residente na VRAM**, crescendo por token.
- **Buffers de ativação residentes** (x, normed, q/k/v, attn_out, intermediários de FFN), alocados uma vez.

### 4.2 Pipelines/descriptors persistentes
- Cada `ComputePipeline` criado **uma vez** e reusado em todos os tokens. Descriptor sets pré-alocados (ou push-descriptors). Fim do churn de pool por dispatch.

### 4.3 Forward 100% na GPU (fim do ping-pong)
- Shaders para **RMSNorm, RoPE, attention GQA (com KV-cache), SwiGLU, residual add**. Um token inteiro fica residente na GPU. Só os **logits finais** (ou o argmax, no greedy) voltam ao host.
- Prefill pode permanecer na CPU na Fase 1 (o benchmark usa `-p 0`, então prefill não entra no número de decode).

### 4.4 Um command buffer por token, sem sync por-op
- Gravar a pilha de camadas inteira como **um command buffer** com **pipeline barriers** (dependências de memória) entre dispatches; submit único; **um fence wait por token**. Elimina ~169 submits + `wait_idle`.

### 4.5 Kernel matvec Q8_0 wave64
- 1 workgroup por linha de output, **64 lanes (wave64)** reduzindo o produto interno cooperativamente via subgroup reduction; dequant Q8_0 inline (scale f16 × int8), **sem materializar pesos f32**. É a vantagem reivindicada sobre a dequantização por-elemento do llama.cpp em gfx906.

---

## 5. Design do tensor-parallel row-split (o ponto que ganha)

Layout estilo **Megatron**, que limita a comunicação a **2 all-reduces por camada**. Cada GPU mantém a metade das linhas de cada matriz e a **stream residual replicada** (vetor completo `n_embd` em ambas).

Por camada do transformer:

| Op | Particionamento | Comunicação |
|----|-----------------|-------------|
| RMSNorm (pré-attn) | redundante na stream replicada | nenhuma |
| **q/k/v proj** | **column-parallel** (split por output) → cada GPU fica com suas heads | nenhuma |
| RoPE + attention (por head) | local: cada GPU computa attention das suas heads, com seu KV-cache | nenhuma |
| **attn output proj** | **row-parallel** (split por contração) → soma parcial | **all-reduce #1** |
| RMSNorm (pré-FFN) | redundante na stream replicada | nenhuma |
| **ffn gate/up** | **column-parallel** → cada GPU fica com sua fatia de `n_ff` | nenhuma |
| SwiGLU | local (na fatia de `n_ff`) | nenhuma |
| **ffn down** | **row-parallel** → soma parcial | **all-reduce #2** |

Após cada all-reduce, ambas as GPUs têm a stream residual completa → próxima RMSNorm é redundante e local.

**Volume de comunicação (14B):** `n_embd=5120`, 2 all-reduces × 48 camadas × 5120 × 4 B ≈ **~2 MB/token** cruzando entre GPUs, vs **~15 GB de pesos** lidos. Trivial — *desde que a latência por all-reduce seja baixa*.

O KV-cache é particionado por head junto com o split de q/k/v — cada GPU guarda só o KV das suas heads. Não há comunicação de KV.

---

## 6. Riscos

1. **All-reduce MI50↔MI50 (risco nº 1).** Sem NVLink. Caminhos possíveis: PCIe peer-to-peer (se RADV expuser `VK_EXT_external_memory` utilizável entre devices) ou via host bounce. O volume é pequeno, mas a **latência** a 100+ tok/s pode dominar (2 all-reduces × 48 camadas = 96 sincronizações/token). **Mitigação:** spike de medição na Fase 0; se peer-to-peer não existir, avaliar host-staged com double-buffering e sobreposição compute/transfer na Fase 3.
2. **Correção single-GPU primeiro.** Row-split exige que o decode single-GPU seja numericamente correto (bit-exact vs CPU) e funcional no 0.5B — esse é o pré-requisito da Fase 2, **não** uma meta de velocidade. A velocidade single-GPU não é objetivo do projeto; o ganho vem de paralelizar os tensores entre as GPUs (Fase 2) e otimizar o kernel com folga depois (Fase 3).
3. **VRAM apertada no 14B.** ~15.6 GB de pesos / 2 = ~7.8 GB por GPU + KV + ativações + overhead do driver. Folgado em 16 GB, mas medir. (No 32B Q8_0 estouraria → motivo da Fase 4 com Q4.)
4. **AMDVLK/RADV quirks** (limite de 2 GB por alocação já contornado pelo sub-alocador; revalidar com tensores grandes do 14B).

---

## 7. Fases (visão; detalhamento tarefa-a-tarefa vai para `plans/`)

### Fase 0 — Baseline & spikes de risco (sem código de produto)
- Obter `qwen2.5-14b-instruct-q8_0.gguf`.
- Medir llama.cpp 2× MI50 em **layer-split e row-split** (`-sm layer` / `-sm row`) → registrar o **melhor número** (o alvo honesto). Confirmar OOM em 1 GPU.
- Spike: bandwidth/latência de transferência MI50↔MI50 (peer-to-peer vs host). **Saída:** decisão do mecanismo de all-reduce.

### Fase 1 — Decode GPU-residente persistente, 1 GPU ✅ **concluída (correção bit-exact no 0.5B)**
- Conectar `GpuWeights` (upload único) ao decode; pipelines/descriptors persistentes.
- KV-cache + ativações residentes em VRAM.
- Shaders RMSNorm / RoPE / attention-GQA / SwiGLU; 1 command buffer/token; sem readback exceto logits.
- **Aceite:** bit-exact vs CPU no 0.5B (mesma sequência greedy). É só prova de correção do forward residente em 1 GPU — **não há meta de velocidade single-GPU**. O esqueleto validado aqui é reutilizado na Fase 2, distribuindo os matmuls entre as GPUs.

### Fase 2 — Tensor-parallel row-split, 2 GPUs, 14B Q8_0 ⬅️ **próximo passo (objetivo central: todas as GPUs a 100%)**
- Layout Megatron (§5) + all-reduce (mecanismo da Fase 0).
- **Aceite:** saída correta no 14B; **decode tok/s (2× MI50) > melhor número do llama.cpp (2× MI50)**.

### Fase 3 — Otimização de kernel para ganhar com folga
- Matvec Q8_0 wave64 com subgroup reduction; coalescência de memória; fusão de ops; double-buffer de tokens (sobrepõe compute com all-reduce).
- **Aceite:** margem ≥1.3× sobre o llama.cpp 2× MI50 no 14B.

### Fase 4 — Q4_K/Q5 → escalar a 32B+
- Shaders de dequant/matvec Q4_K e/ou Q5_K; re-benchmark no 32B (que não cabe em Q8_0 nas 2 GPUs).
- **Aceite:** rodar 32B Q4 nas 2 MI50 e bater o llama.cpp 2× MI50 no mesmo modelo/quant.

---

## 8. Metodologia de benchmark

- Reusar `scripts/benchmark-gpu.sh` (já parametrizável via `BENCH_MODEL`), estendendo o lado llama.cpp para reportar **layer-split e row-split** e tomar o melhor.
- Mesma API (Vulkan/RADV) dos dois lados → isola **qualidade da implementação**, não diferença de backend.
- NVIDIA K80 permanece excluída (guardrail já existente).
- Greedy (temp=0), seed fixa, comparação de igualdade de saída como gate de correção.

---

## 9. Fora de escopo (YAGNI)

- Batch > 1 / serving concorrente (métrica diferente; ver opção "throughput" descartada no brainstorming).
- Backend NVIDIA / Tesla K80.
- Quantizações além de Q8_0 até a Fase 4.
- Prefill na GPU até a Fase 1 estar validada (não afeta o número de decode).
- Refatorações não relacionadas ao caminho de decode.
