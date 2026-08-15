# Progresso — llama-rs

**Última atualização:** 2026-08-15 (32B em layer-split passa o llama.cpp)

## Estado em números (2026-08-15)

| Config | llama-rs | llama.cpp | razão |
|---|---|---|---|
| Qwen2.5-32B Q5_K_M, 2× MI50 (layer-split) | **19.3 tok/s** | 18.02 ± 0.08 (ROCm) | **1.07×** |
| Qwen2.5-14B Q8_0, 1× MI50 | **28.4 tok/s** | 40.59 tok/s (Vulkan) | 0.70× |
| Qwen2.5-0.5B Q8_0, 1× MI50 | 123 tok/s | 334 tok/s (Vulkan) | 0.37× |

O 32B é o primeiro caso que **justifica** o layer-split: 22.2 GB não cabem nos 16.3 GiB
de uma MI50. Dividido, ocupa 10.4 GB na GPU0 e 12.2 GB na GPU1, com uma única
sincronização por token (a stream residual de `n_embd` floats). É também o primeiro
caso em que o llama-rs **passa o llama.cpp**: 19.17–19.40 tok/s em 5 medições contra
18.02 ± 0.08 do `llama-bench -n 128` no mesmo arquivo (protocolo idêntico: 128 tokens,
greedy, contexto pequeno).

O 14B saiu de **4.77 → 28.0 tok/s (5.9×)** nesta sessão. Perfil atual do 14B
(`LLAMA_RS_PROFILE=1`): GPU 27.8 ms/token, host 3.0 ms/token.

| op | ms/token | % |
|---|---|---|
| matvec | 21.57 | 77.7% |
| rmsnorm | 2.02 | 7.3% |
| attention | 1.85 | 6.6% |
| add | 0.89 | 3.2% |
| quantize_x | 0.77 | 2.8% |
| demais | 0.67 | 2.4% |

**O matvec lê 15.5 GiB a 717 GB/s — perto do máximo alcançável na MI50.** O ganho
restante não vem de mais ocupação; vem de **ler menos bytes**: o padding de 36 B por
bloco Q8_0 (contra 34 B do formato) custa 5.9%, e Q4_K quase metade. Ver
`docs/estrategia-inferencia-mi50.md`.

---

## Resumo em uma frase

CPU: pipeline completa e bit-exact contra o llama.cpp. GPU (Vulkan, 2× AMD MI50): decode residente correto, com Q8_0, Q5_K e Q6_K, e **1.07× o llama.cpp no Qwen2.5-32B em layer-split** — o caso que justifica dividir, porque 22.2 GB não cabem numa placa. Tensor-parallel row-split foi **descartado por medição** — sem P2P de VRAM neste hardware, o custo de sincronização supera o ganho.

---

## Sessão 2026-08-13 — o que mudou

### Fase 0 concluída (era o próximo passo nº1)

- **Baseline llama.cpp Vulkan no 14B** (`bench-results/fase8-0-baseline-14b-e-allreduce.md`):
  **1× MI50 = 40.59 tok/s**, 2× layer-split = 27.34, 2× row-split = falha ao carregar
  (`device Vulkan0 does not support split buffers` — é limitação do hardware, não do backend:
  o mesmo ocorre no ROCm com `NO_PEER_COPY=1`). O 14B **cabe em 1 GPU**, e por isso layer-split
  ali só atrapalha — o que **não** generaliza para modelos maiores (ver o gate revisado abaixo).
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

**Resolvido nesta sessão** (repack preguiçoso + `RawTensor<'a>` emprestando do mmap).

---

## Gate de decisão (Etapa 4 do plano)

O plano previa: *se o single-GPU chegar a ~1.5× do llama.cpp, o row-split passa a valer; se não,
row-split multiplica um número ruim.* Onde paramos: **0.69×** (28.0 vs 40.59).

A leitura honesta é que **o gate não foi atingido, mas o diagnóstico mudou**. O matvec não está
lento por arquitetura de paralelismo — está a 717 GB/s, perto do teto da placa. Isso significa:

- **Row-split não é o próximo passo.** Ele divide bytes entre 2 GPUs, mas cada GPU já lê perto da
  banda máxima; o ganho seria real (~1.4–1.7×, ver `docs/estrategia-inferencia-mi50.md`) porém
  aplicado sobre um número que ainda perde para o llama.cpp em 1 GPU.
- **O lever agora é ler menos bytes**, na ordem: (1) bloco Q8_0 de 36 B → 34 B, recuperando 5.9%
  de banda sem perda numérica; (2) suporte a **Q4_K**; (3) fusão das ops pequenas.
- Row-split não se justifica em nenhum cenário aqui (medido, ver abaixo).

### O row-split foi medido, não estimado (2026-08-14)

| Mecanismo | por sincronização | 96 all-reduces/token |
|---|---|---|
| host-mediado (fence de cada lado) | 101 µs | 19.4 ms |
| semáforo externo, pipelinado | **59.3 µs** | **5.69 ms** |

P2P de VRAM **não existe** entre estas MI50: `OPAQUE_FD` falha no import
(`ERROR_INVALID_EXTERNAL_HANDLE`), `DMA_BUF` importa mas o memory type é
`HOST_VISIBLE|HOST_COHERENT` e lê a **10.2 GB/s** contra 717 GB/s do HBM local
(issue ROCm #4793). Semáforo externo funciona, mas é por submit — 96 all-reduces
exigem 96 submits por GPU, e os 5.69 ms não descem.

Balanço no 14B: economiza 10.8 ms (matvec pela metade), custa 5.9 ms → **~38 tok/s**,
abaixo dos 40.59 do llama.cpp. **Tensor-parallel não é o caminho neste hardware.**

### Layer-split, por outro lado, é o caminho (revisão de 2026-08-14)

Os benchmarks da própria máquina (`/home/murilo/llama.cpp/benches/radeon-pro-vii-gfx906/` e
`ORNITH-GFX906-NOTES.md`) mostram que **tudo que roda rápido aqui usa layer-split entre as 2 GPUs**,
porque não cabe em 16 GiB:

| Modelo | VRAM | Decode |
|---|---|---|
| Qwen3.6-35B-A3B (MoE) + speculative | 28.64 GiB | **77.1 tok/s** |
| Qwen3.6-27B **denso** | — | **23.8 tok/s** ← alvo real do projeto |
| ThinkingCap-Qwen3.6-27B Q6_K | 20.88 GiB | 19.5 tok/s |

Custo do layer-split: **1 sincronização por token** (a fronteira entre as camadas de cada GPU) =
0.06 ms pelos 59.3 µs medidos. Contra 96 syncs / 5.69 ms do tensor-parallel — **~100× mais barato**.

E uma segunda correção: o 27B denso é **compute-bound** nesta placa (42 ms/token contra ~15 ms de
teto de banda; gfx906 não tem matrix cores). Isso enfraquece a aposta em Q4_K, que troca bytes por
ALU justamente onde a placa é fraca.

### Layer-split implementado (2026-08-14)

`--gpu-layer-split`. Divide as camadas entre as GPUs **proporcionalmente à VRAM livre** (mesma
política do llama.cpp), com uma cópia da stream residual na fronteira — 1 sincronização por token.
Saída idêntica ao caminho single-GPU.

| Modelo | 1 GPU | layer-split |
|---|---|---|
| Qwen2.5-0.5B Q8_0 | 111.2 | 95.1 tok/s |
| Qwen2.5-14B Q8_0 | 28.0 | 20.79 tok/s |

Mais lento nos dois porque **ambos cabem numa GPU** — dividir só acrescenta a fronteira. O custo
(−26% no 14B) é menor que o do próprio llama.cpp no mesmo modelo (−33%). O ganho é **capacidade**:
modelos de 20–28 GiB, a faixa do Qwen3.6-27B, não cabem nos 16 GiB de uma MI50.

**Ainda não validado no caso que justifica a feature**: não há modelo Q8_0 acima de 16 GiB
localmente (o 32B disponível é Q5_K_M, formato que o llama-rs não lê). Provar o ganho de capacidade
depende de suporte a K-quants ou de um Q8_0 maior.

### Próximo passo

Suporte a **K-quants** (Q4_K/Q5_K/Q6_K) — destrava tanto os modelos locais de 20–22 GiB quanto a
validação real do layer-split. Expectativa calibrada: o ganho de velocidade é sublinear porque o
gfx906 vira compute-bound em K-quants (ver acima).

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
| Ativação int8 + `dotPacked4x8` (packed dot) | **no Q8_0**: 0% de ganho → esse kernel não é ALU-bound. (Nos K-quants, em 2026-08-14, o mesmo dot deu **2×** — ali o desempacotamento byte a byte dominava.) |
| Pesos Q8_0 em blocos alinhados de 32 bits | ~5% → não é padrão de load isolado |

Conclusões cruzadas: não é ALU-bound, não é byte-load, não é banda saturada (~91 tok/s já está a ~5% do teto de ~1 TB/s para este modelo), não é overhead de submit (resolvido na Fase 1D). **Suspeito remanescente: baixa ocupação/paralelização estrutural do matvec** — 1 workgroup de 64 lanes cobrindo poucas linhas de saída, sem coalescência real entre lanes adjacentes.

Próximo passo recomendado no próprio documento (nunca executado): **profiling de hardware (`rocprof`)** para localizar o gargalo real, seguido de um **redesenho do particionamento do matvec** (múltiplos subgroups por workgroup, particionamento em árvore da dimensão K, ou tiling à la `mul_mat_vec` do llama.cpp) em vez de continuar ajustando o kernel atual.

---

## Sessão 2026-08-14 — K-quants no decode residente

O 32B Q5_K_M saiu de **2.81 → ~13.5 tok/s (4.8×)**. Progressão medida, cada passo isolado:

| passo | tok/s | matvec_q5k (ms/token, GPU0) |
|---|---|---|
| Q5_K/Q6_K desempacotando byte a byte | 2.81 | 120.0 |
| loads de 32 bits (4 elementos por load) | 5.47 | 62.2 |
| ativação lida em `vec4` | 6.15 | 56.1 |
| **ativação int8 + `dotPacked4x8`** | **12.75** | **17.5** |
| 4 waves por workgroup | ~13.5 | — |
| `soma(x)` pré-calculada no `quantize_x` | 12.24 | **revertido** |

O salto está no dot empacotado, e o que o torna barato nos K-quants é que **num uint de
`qs`/`ql` os 4 nibbles já estão em bytes separados**: `qsw & 0x0F0F0F0F` produz o operando
do `V_DOT4_I32_I8` em uma instrução. O sub-bloco de 32 elementos do Q5_K coincide com o
bloco de quantização da ativação, então as escalas casam sem conversão.

Duas medições que contrariam a intuição e valem como referência:

- **Pré-calcular `soma(x)`** (o termo constante do K-quant, que não depende dos pesos e era
  refeito para cada uma das n_out linhas) ficou **6% mais lento**: trocar ~8 instruções VALU
  por 2 loads é mau negócio. Depois do dot empacotado o kernel é limitado por memória.
- **4 waves por workgroup** rendeu só ~2% — a ocupância não era o gargalo que parecia.

Erro numérico validado contra a referência de CPU (`dequant_to_f32` + produto interno sobre
a mesma ativação int8): < 1e-5 nos dois shaders.

---

## Sessão 2026-08-15 — o 32B passa o llama.cpp (14.7 → 19.3 tok/s)

Alvo: bater o llama.cpp no Qwen2.5-32B Q5_K_M em layer-split. Baseline remedido no
mesmo arquivo e protocolo (`llama-bench -n 128`, ROCm): **18.02 ± 0.08 tok/s** — o
16.36 registrado antes estava desatualizado. Resultado: **19.17–19.40 tok/s**, estável.

Perfil de GPU por token, antes → depois (soma dos dois shards):

| op | antes | depois |
|---|---|---|
| matvec_q5k | 37.8 ms | 33.9 ms |
| matvec_q6k | 7.9 ms | 7.6 ms |
| rmsnorm | 3.3 ms | 1.3 ms |
| attention | 10.0 ms | 1.7 ms |
| **total GPU** | **59.8 ms** | **49.7 ms** |

### O que rendeu, em ordem de ganho

1. **Espera do fence por sondagem, não bloqueante** (+8%, e acaba com a oscilação).
   O tempo por token era bimodal — 53 ou 57 ms com o tempo de GPU idêntico até o
   décimo de µs. Dormir 25 ms num `wait_for_fences` faz o schedutil derrubar a
   frequência e a CPU entrar em C-state profundo; o wakeup pela IRQ da GPU passa a
   custar milissegundos. Qualquer busy-loop rodando em paralelo "consertava" o número,
   o que fecha o diagnóstico. Dormir 90% e sondar o resto **não** resolve: a latência
   já é paga ao acordar do sono longo. Custo: um núcleo ocupado enquanto a GPU
   trabalha (3.5% desta máquina).
2. **`attention` com 8 waves por head** (6× no kernel). O laço sobre as posições é uma
   cadeia serial (softmax online) com um `subgroupAdd` por passo, e 1 wave por head
   punha **40 waves numa GPU de 240 SIMDs**. O KV-cache era lido a 3.7 GB/s contra
   717 GB/s de pico — era serialização, não banda. Com 16 waves o kernel cai mais mas
   o total piora: workgroups de 1024 competem por recursos.
3. **spin pool fora do caminho GPU** (+3.6%). O pool de matmul de CPU era inicializado
   em todos os backends e seus workers ficavam em busy-wait permanente: **97.6% dos
   ciclos** do processo, disputando CPU com a thread que grava o command buffer.
4. **Readback dos logits em memória HOST_CACHED** (+5.5%). O primeiro tipo
   host-visible da MI50 é write-combining — ótimo para escrever, péssimo para ler. Os
   608 KB de logits custavam 4.2 ms/token (145 MB/s); agora custam 0.21 ms.
5. **Duas linhas de saída por wave nos dois matvecs K-quant** (−8% no q5k, −5% no q6k):
   a ativação e a soma dos int8 do bloco valem para as duas linhas, então os requests
   por linha caem de 8 para 6.5. Com **quatro** linhas piora: os acumuladores extras
   derrubam a ocupância.
6. **`q5_k_matvec` lendo em uvec4** (+7% no kernel): 176 B por superbloco são 11 uvec4
   alinhados, então 17 requests de 4 B por lane viram 5 de 16 B.
7. **`rmsnorm` em vec4** (2.5×): a redução roda num único workgroup — 1 CU dos 60 —
   então o que limita é a taxa de requests dessa CU, não a banda.

### O que foi testado e descartado por medição

- **Reduzir a redundância de leitura entre lanes do Q5_K** (par de sub-blocos por lane,
  4 lanes por superbloco em vez de 8): **4% mais lento**. A amplificação de 2.9× que o
  documento anterior apontava como "suspeita principal" é absorvida pelo cache; o que
  custa é deixar 16 das 64 lanes ociosas na última rodada de 20 superblocos.
- **Pré-calcular `soma(x)` no `quantize_x`**, empacotada em `vec2` junto da escala que o
  matvec já lia — sem load adicional, só 8 B em vez de 4: **3% mais lento**. Some-se à
  medição anterior (buffer próprio, +6%): o matvec é limitado por memória, não por ALU,
  e trocar ~8 instruções VALU por bytes lidos é sempre mau negócio aqui.
- **Pinar a thread principal ao nó NUMA das GPUs**: piora. As duas MI50 estão no nó 0,
  mas o nó 1 é consistentemente mais rápido. Não é disputa com a IRQ do amdgpu (rodar
  *na* CPU que a trata é rápido) — some junto com a bimodalidade quando a espera do
  fence deixa de bloquear.

### Onde ainda há espaço

- `matvec_q5k` é 68% do tempo e lê a ~490 GB/s de um pico de 717. O Q8_0 atinge 717
  com o mesmo dot empacotado, então a diferença é estrutural: 8 lanes por superbloco
  releem os mesmos 80 B para consumir 20 B úteis cada.
- Sobra ~3 ms/token de host fora do perfil. A gravação dos dois command buffers custa
  1.5 ms e é serial com a execução; dá para gravar o do shard seguinte enquanto o
  anterior roda, ou pré-gravar o plano inteiro passando `pos` por buffer em vez de
  push constant.
- As ~390 ops pequenas por token (add, quantize_x, rope, swiglu, kv_append) estão todas
  no piso de latência de lançamento (~4 µs). Fundir os 3 bias nos matvecs e o
  `quantize_x` no `rmsnorm` tiraria ~150 dispatches.

---

## Próximos passos (ordem recomendada)

1. **Q4_K.** É o formato dos modelos que interessam (o Qwen3.6-27B local está em Q4_K_M) e
   ainda não tem shader. O layout de escalas é o mesmo do Q5_K (`get_scale_min_k4`), então
   o kernel sai quase de graça a partir do `q5_k_matvec.comp` — sem `qh`, com metade dos
   requests de peso por lane.
2. **Tirar a gravação do command buffer do caminho crítico** (~1.5 ms/token de 52). Duas
   saídas: gravar o cmdbuf do shard seguinte enquanto o anterior executa, ou pré-gravar o
   plano inteiro passando `pos` por buffer em vez de push constant. A segunda também
   elimina o custo por token de vez.
3. **Fundir as ops pequenas.** ~390 dispatches/token estão no piso de latência de
   lançamento (~4 µs): os 3 bias cabem no matvec que os precede e o `quantize_x` cabe no
   `rmsnorm`/`swiglu` que o alimenta.
4. **`matvec_q5k` a ~490 GB/s de 717.** O Q8_0 atinge 717 com o mesmo dot empacotado, então
   a diferença é estrutural: 8 lanes por superbloco releem os mesmos 80 B para consumir
   20 B úteis cada. Reduzir a redundância por si só já foi medido e piora (ver a sessão de
   2026-08-15); o caminho seria um repack de `qh` no upload, transpondo o bit-plane para um
   uint por sub-bloco. Medir com `MESA_VK_TRACE=rgp` antes de reescrever.
4. **Tensor-parallel está descartado por medição, não por falta de tempo.** Sem P2P de VRAM
   neste hardware (`OPAQUE_FD` falha, `DMA_BUF` importa como host-visible a 10.2 GB/s contra
   717 GB/s local), os 96 all-reduces por token do layout Megatron custariam ~5.7 ms/token —
   mais do que economizariam. Layer-split é a resposta certa aqui, e é o que o llama.cpp faz.
5. **Manter README e este documento sincronizados a cada mudança de escopo.**

---

## Referências

- Spec ativa: `docs/superpowers/specs/2026-06-16-fase8-decode-gpu-persistente-rowsplit-design.md`
- Design geral / reorientação multi-GPU: `docs/superpowers/specs/2026-06-03-llama-rs-rewrite-design.md`
- Histórico de fases: `docs/superpowers/plans/2026-06-*.md`
- Diagnóstico completo da Fase 8.3 (fora do HEAD): `git show 5452216~1:docs/superpowers/results/fase8-3-kernel-progressao.md`
