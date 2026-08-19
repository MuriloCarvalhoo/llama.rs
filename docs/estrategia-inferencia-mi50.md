# Estratégia de inferência nas 2× MI50 — avaliação crítica do rumo do projeto

**Data:** 2026-08-13 · **Hardware:** 2× AMD MI50 / Radeon Pro VII (gfx906, 16 GB HBM2, ~1 TB/s, PCIe, sem NVLink)

Documento de pesquisa com fontes. Ele **contradiz premissas** que este projeto vinha usando —
inclusive uma que eu afirmei nesta sessão. As correções estão marcadas.

---

## 0. Correção de duas premissas do projeto

### ❌ "O backend Vulkan do llama.cpp não suporta row-split"

Eu afirmei isso hoje com base no erro `device Vulkan0 does not support split buffers` que medimos.
**A conclusão estava errada.** O que a pesquisa mostra:

1. **`--split-mode row` funcionava no Vulkan** e quebrou numa regressão — issue
   [#25884](https://github.com/ggml-org/llama.cpp/issues/25884) reporta row-split rodando por meses
   em config híbrida AMD+Intel, quebrando no commit `74976e1`. É regressão, não ausência.
2. **`row` foi deprecado e substituído.** A [doc oficial](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
   marca `row` como *"Deprecated. Older row-split tensor-parallel path with comparatively poor
   performance"* e adiciona `--split-mode tensor` (experimental).
3. **Tensor-parallel backend-agnóstico foi mergeado em 09/04/2026** — PR
   [#19378](https://github.com/ggml-org/llama.cpp/pull/19378): um "meta backend" que infere o split
   pelo grafo e insere 2 AllReduce por camada. **Vulkan é suportado nominalmente**, com performance
   ruim em contexto curto e instabilidade em contexto longo.

**Consequência:** o projeto não está preenchendo um vazio. Está reconstruindo algo que existe, está
em desenvolvimento ativo, e cujos autores já documentaram onde dói.

### ❌ "Row-split dobra a banda efetiva → ~2×"

Medições públicas de TP=2 sobre PCIe, todas convergindo:

| Fonte | Config | Ganho em decode |
|---|---|---|
| PR #19378 (llama.cpp, PCIe 4.0 x16) | 2× RTX 4090, LLaMA 8B | **1.60–1.72×** |
| PR #19378 | 2× RTX 4090, Gemma 31B | 1.36–1.57× |
| [PremAI 2026](https://www.premai.io/blog/multi-gpu-llm-inference-tp-vs-pp-vs-ep-parallelism-guide-2026/) | TP=2 genérico | 1.7–1.9× (85–95% eficiência) |
| [ahmadosman.com](https://www.ahmadosman.com/blog/do-not-use-llama-cpp-or-ollama-on-multi-gpus-setups-use-vllm-or-exllamav2/) | AMD, PCIe 4.0 x16 | 1.5–1.7× |
| [arXiv 2504.17674](https://arxiv.org/pdf/2504.17674) | batch-1, 2 GPUs | latência −40% (=1.67×) |

**TP=2 sobre PCIe entrega 1.4–1.7×, não 2×.** E a PremAI recomenda explicitamente: *"Accept TP=2 as
maximum for PCIe"* e *"Stay single-GPU if model fits quantized"*.

Por quê: o gargalo **não é banda**. São 96 AllReduce/token (48 camadas × 2), ~2 MB/token total —
0.08 ms em PCIe 4.0, ~0.5% de um token. O custo é **latência × 96 sincronizações**. Isso bate com
o que medimos no nosso próprio spike (§ `bench-results/fase8-0-baseline-14b-e-allreduce.md`).

---

## 1. Quantização: o ganho é real, mas menor do que a conta sugere

> **Correção de 2026-08-14.** A versão original desta seção projetava Q4_K_M em ~71 tok/s assumindo
> `tok/s ∝ 1/bytes`. Benchmarks de referência do llama.cpp neste mesmo hardware (gfx906)
> mostram que **isso não vale para K-quants aqui**: o Qwen3.6-27B denso roda a 42 ms/token contra um
> teto de banda de ~15 ms — **compute-bound**, porque a Radeon Pro VII tem HBM2 rápida e pouco
> compute (sem matrix cores), e K-quants custam mais ALU para desquantizar que o Q8_0.
>
> Ou seja: trocar Q8_0 → Q4_K troca bytes por ALU exatamente onde esta placa é fraca. As linhas
> abaixo são o **limite superior otimista**, não previsão.

Nossa baseline: **40.59 tok/s** com Qwen2.5-14B **Q8_0** (15.7 GB) em 1 MI50.

| Quant | Tamanho | tok/s estimado¹ | Δ qualidade² |
|---|---|---|---|
| Q8_0 (atual) | 15.7 GB | **40.6 (medido)** | −0.06 pt |
| Q6_K | 12.1 GB | ~52 | −0.24 pt |
| Q5_K_M | 10.5 GB | ~60 | −0.11 pt |
| **Q4_K_M** | **8.9 GB** | **~71** | **−0.32 pt** |
| Q4_0 | 8.2 GB | ~77 | −1.49 pt ← evitar |

¹ **limite superior**: derivado de 1024 GB/s × 62% MBU ÷ tamanho, supondo memory-bound. Para
K-quants nesta placa a realidade fica abaixo disso (ver correção no topo da seção).
² [arXiv 2601.14277](https://arxiv.org/html/2601.14277v1), Llama-3.1-8B, média de 5 benchmarks.

**Note o degrau Q4_K_M (−0.32) vs Q4_0 (−1.49): K-quants importam.**

E há um ganho colateral grande: com Q8_0 sobra ~1.2 GB de VRAM → **~6k tokens de contexto**. Com
Q4_K_M sobram ~8 GB → **~40k tokens**. Ganha-se velocidade **e** contexto.

### A comparação (com o teto otimista)

| Caminho | tok/s | Esforço | Risco |
|---|---|---|---|
| Baseline atual (Q8_0, 1 GPU) | 40.6 | — | — |
| **Trocar para Q4_K_M** | **~71** | **1 comando** | −0.32 pt |
| Q4_K_M + speculative decoding | ~100–130 | dias | médio |
| **Escrever TP row-split Vulkan do zero** | **57–69** | **meses** | **alto** |

**Um comando bate o resultado final do projeto de meses.** Este é o argumento central e precisa ser
confrontado, não contornado.

---

## 2. Nossa eficiência já é boa (não há 3× escondido)

40.59 tok/s × 15.7 GB = 637 GB/s = **62% de utilização de banda (MBU)**. Comparação:

| Sistema | Modelo | MBU |
|---|---|---|
| **Nossa MI50 (llama.cpp Vulkan)** | Qwen2.5-14B Q8_0 | **62%** |
| MI50, llama.cpp Vulkan | llama-2-7B Q4_0 | 41–44% |
| RTX 5090, CUDA + graphs | gpt-oss-20b | ~80% (melhor público) |
| M2 Max | 7B q4 | 52–60% |

Duas leituras: (a) modelos maiores rendem MBU maior — já estamos no regime favorável; (b) o teto
prático é ~80%. **Fechar 62% → 80% em 1 GPU vale 40.6 → 52 tok/s**, comparável ao ganho líquido de
TP2, com uma fração do risco.

Teto absoluto de 1 GPU: 1024/15.7 = **65.2 tok/s**. TP2 perfeito seria 130; TP2 realista, **57–69**.

---

## 3. O risco que pode inviabilizar o TP: P2P no gfx906

- **[ROCm issue #4793](https://github.com/ROCm/ROCm/issues/4793)** — "MI50 32GB p2p not working":
  matriz de acesso mostra **0** entre as duas MI50 em ROCm 5.7.1 e 6.4.0. **Aberto, sem workaround.**
  A BAR region 0 é 16G num card de 32 GB — ou seja, a BAR não expõe a VRAM inteira, que é o
  requisito do P2PDMA.
- Comentário na thread do llama.cpp: *"Tensor parallelism with row mode crashes on HIP/ROCm because
  it requires P2P GPU access, which MI50 on PCIe doesn't support."*
- No lado Vulkan, o PR [#25051](https://github.com/ggml-org/llama.cpp/pull/25051) (AllReduce
  GPU-pipelined com timeline semaphores) registra: **"D2D transfers disabled due to GTT landing on
  AMD"**, regressão de 4.8× em contexto longo, e output lixo em modelos ≥35B.

> **Teste barato e decisivo:** nossos cards são de **16 GB**, e a BAR reportada na issue é 16G — num
> MI50 de 16 GB a BAR *cobriria* a VRAM inteira. Rodar `rocm-bandwidth-test` (com Above-4G decoding
> e `iommu=pt`) determina se o TP é viável ou se será host-bounce a 96 syncs/token.
> **Nosso spike já mostrou que as extensões P2P Vulkan estão presentes** (`external_memory_fd`,
> `dma_buf`, `external_semaphore_fd`), mas presença de extensão ≠ P2P funcional.

E há um sinal de alarme nos nossos próprios números: **layer-split cai 40.59 → 27.34 (−33%) com
UMA transferência de ativação por token.** Se um hop custa 33%, 96 AllReduces custam muito mais.
Investigar essa anomalia é o proxy mais barato para "quanto vai custar o TP".

---

## 4. O que fazer com a segunda GPU (a resposta boa)

**Speculative decoding com o draft na GPU2 e o target na GPU1.**

- Ganho medido: **1.4–1.85×** (Llama-3.1-8B com draft Llama-3.2-1B: **1.83×** com 5 draft tokens,
  [llama.cpp docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)).
- **Multiplica** com a quantização (§1) — são independentes.
- **Não exige P2P nem AllReduce.** Só troca alguns tokens entre os processos.
- É especialmente adequado aqui: prefill na MI50 é ~27× mais rápido que decode, então **verificar
  5–8 tokens de draft custa quase o mesmo que gerar 1**.
- Evita um bug conhecido: com os dois modelos na *mesma* GPU no Vulkan/RADV, o tempo do draft
  explode por haver uma única compute queue ([issue #23126](https://github.com/ggml-org/llama.cpp/issues/23126)).
  Duas GPUs resolvem isso naturalmente.

**Alternativa igualmente barata: trocar para um MoE.** Dado medido:
**Qwen3-Coder-30B-A3B Q4_K_M roda a 73.10 tok/s em 1× MI50**
([llama-bench](https://ahelpme.com/ai/llamacpp-ai/llama-bench-the-qwen3-coder-30b-a3b-and-amd-radeon-instinct-mi50-32gb/)).
Em batch-1 o MoE lê só os experts ativos → modelo grande **e** decode rápido, cabendo em layer-split
(onde a penalidade em MoE é pequena) sem TP nenhum.

---

## 5. Técnicas ordenadas por ganho em decode batch-1

| # | Técnica | Ganho | Esforço |
|---|---|---|---|
| 1 | **Quantização de pesos** (Q8_0 → Q4_K_M) | **1.3–1.8×** | zero |
| 2 | **Speculative decoding** | **1.4–1.85×** | médio |
| 3 | Reuso de command buffer / graph | 10–15% (medido em CUDA) | médio |
| 4 | Quantização de KV cache | libera VRAM (não acelera) | baixo |
| 5 | Flash Attention / Paged Attention | **~0 em batch-1 curto** | — |
| 6 | Medusa / EAGLE-3 | 2–4× no papel, 1.7–2.2 real | alto (treina draft head) |

Contexto: o paper [arXiv 2605.30571](https://arxiv.org/pdf/2605.30571) mostra que em batch-1 o
gargalo é **latência de memória + overhead de lançamento de kernel**, não saturação de banda. Isso
explica por que FA2 chegou a *piorar* decode (17.07 → 24.16 ms) em contexto curto.

Nota: o TP do llama.cpp **exige** FA habilitado e **proíbe** KV quantizado — se formos por TP,
perdemos a opção 4.

---

## 6. Panorama de engines para gfx906

| Engine | gfx906 | Vulkan | TP | Situação |
|---|---|---|---|---|
| **llama.cpp** | sim | sim, maduro | `-sm tensor` desde abr/2026, Vulkan instável | única opção viável hoje |
| vLLM | só forks | não | sim | [nlzy/vllm-gfx906](https://github.com/nlzy/vllm-gfx906) **arquivado em 20/02/2026** |
| SGLang / TensorRT-LLM / LMDeploy | não | não | sim | fora |
| ExLlamaV2/V3 | não | não | sim | "ROCm support" é item de to-do |
| MLC-LLM | talvez | **sim** | **sim** | único com TP+Vulkan por design; **sem medição pública em gfx906** |
| candle / burn / mistral.rs | — | mistral.rs tem Vulkan | não/imaturo | não são atalho |

Sinal relevante: existem ao menos 4–5 projetos "engine LLM em Rust + Vulkan do zero" recentes.
**Nenhum tem tensor-parallel.**

---

## 7. Veredito, revisado com os benchmarks da própria máquina

> Esta seção foi **reescrita em 2026-08-14** depois de ler os benchmarks de referência do
> llama.cpp (`ORNITH-GFX906-NOTES.md`, `BUILD-GFX906.md`, `benches/radeon-pro-vii-gfx906/`),
> que medem este hardware exato. Duas conclusões anteriores caíram.

### O que roda mais rápido nesta máquina (medido, não estimado)

| Modelo | VRAM | Multi-GPU | Decode |
|---|---|---|---|
| **Qwen3.6-35B-A3B (MoE) + `draft-dflash n-max 3`** | 28.64 GiB | **layer-split** | **77.1 tok/s** @128k |
| Ornith-1.0-35B MTP (MoE) | 28.75 GiB | layer-split | 69.6 tok/s @64k |
| Qwen3.6-27B **denso** | — | layer-split | **23.8 tok/s** |
| ThinkingCap-Qwen3.6-27B Q6_K | 20.88 GiB | layer-split (14.55 + 12.43) | 19.5 tok/s |

**Nada disso cabe em 16 GiB.** O layer-split não é uma otimização aqui — é o que torna esses
modelos executáveis.

### Correção 1 — layer-split é essencial, não inútil

O bench inicial deste projeto mediu layer-split no Qwen2.5-14B Q8_0 (que **cabe** em 1 GPU) e viu
27.34 contra 40.59 tok/s. A conclusão registrada — "layer-split serializa, multi-GPU não ajuda" —
é verdadeira só para esse caso e foi indevidamente generalizada.

O custo real do layer-split é **1 sincronização por token** (a fronteira entre as camadas de cada
GPU), contra **96** do tensor-parallel. A 59.3 µs medidos, são **0.06 ms/token** — ruído. É ~100×
mais barato que o TP e destrava a faixa de 20–28 GiB.

### Correção 2 — a aposta em Q4_K é mais fraca do que esta doc dizia

Ver a correção no topo da §1: o 27B denso já é **compute-bound** nesta placa (42 ms/token contra
~15 ms de teto de banda). Q4_K troca bytes por ALU onde o gfx906 é fraco. O ganho existe, mas não é
o ~1.8× que a conta de banda pura sugeria.

### O que continua valendo

- **Tensor-parallel está descartado** — e agora por três evidências independentes: sem P2P de VRAM
  (medido aqui, §3), `-sm row` falha também no ROCm (`NO_PEER_COPY=1`), e o balanço de 96 syncs
  × 59.3 µs come o ganho.
- **Speculative decoding só compensa em MoE esparso.** Medido nesta máquina: +23% no 35B A3B, e
  **net-negative no 27B denso** (23.77 → 21.95/16.57/15.67 conforme `n-max` sobe). A razão é a
  mesma da Correção 2: se batch-1 já satura o compute, verificar N tokens custa ~N×.

### Ordem recomendada (revisada)

1. **Layer-split entre as 2 GPUs.** Barato (1 sync/token), e é o único caminho para rodar qualquer
   coisa acima de 16 GiB — inclusive o Qwen3.6-27B, que era o alvo original do projeto.
2. **Quantização K-quant**, com expectativa calibrada (ganho sublinear, pode virar compute-bound).
3. Fusão de ops pequenas e redução de dispatches, que atacam o lado compute — o que de fato limita
   este hardware em K-quants.
4. ~~Tensor-parallel~~ — descartado por medição.
5. ~~Speculative decoding~~ — só se o alvo virar um MoE esparso.

### A ressalva que continua de pé

Se o objetivo é **aprender** GPU/Vulkan/inferência, construir do zero é ótima escolha e nada disso
se aplica. O que não funciona é misturar as justificativas: o alvo honesto para "bater o llama.cpp
no Qwen3.6-27B" é **23.8 tok/s**, não os 40.59 do Qwen2.5-14B Q8_0 que este documento usava como
referência — são modelos, quantizações e arquiteturas diferentes.

---

## 8. Lacunas de dados (não preencher com estimativa)

- Nenhuma medição pública de TP em 2× gfx906, em qualquer engine.
- Nenhuma medição pública de MLC-LLM TP sobre Vulkan em dGPU AMD.
- Nenhuma medição de latência de semáforo cross-device Vulkan em RADV/gfx906.
- Nenhum benchmark de banda alcançável (BabelStream) para MI50 — os MBU acima usam os 1024 GB/s de
  spec, logo são otimistas.
