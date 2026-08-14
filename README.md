# llama-rs

Reescrita do zero em Rust do runtime de inferência LLM, com foco no backend **Vulkan** para 2× GPUs AMD MI50.

O projeto constrói a pipeline completa em Rust (tokenizer → forward pass → sampling) com um backend Vulkan próprio, para ter controle total sobre os shaders SPIR-V e o agendamento de memória neste hardware.

> **Estado do backend Vulkan:** decode residente em 1 GPU funcionando e validado — **28 tok/s no Qwen2.5-14B Q8_0** contra 40.59 do llama.cpp Vulkan. Multi-GPU ainda não existe. Ver [`PROGRESS.md`](PROGRESS.md).

---

## Motivação

A premissa original era corrigir quatro fraquezas do llama.cpp em gfx906 (dequantização por elemento no Vulkan, Flash Attention incompatível com wave64, `WARP_SIZE` fixo em 32, e multi-GPU sem row-split).

**Duas dessas premissas não sobreviveram à medição** e ficam registradas para não serem refeitas:

- **"Multi-GPU sem row-split" não é uma falha do llama.cpp — é limite deste hardware.** Não há P2P de VRAM entre estas MI50 (medido em `crates/llama-vulkan/src/spike.rs`), e o `-sm row` falha igualmente no ROCm (`NO_PEER_COPY=1`). O que funciona multi-GPU aqui é **layer-split**.
- **Decode batch-1 nem sempre é memory-bound aqui.** Vale para Q8_0 (nosso matvec lê a 717 GB/s, perto do teto); **não vale para K-quants**, onde o gfx906 vira compute-bound por não ter matrix cores.

Ver [`docs/estrategia-inferencia-mi50.md`](docs/estrategia-inferencia-mi50.md) para as evidências.

---

## Estado atual

A pipeline **CPU** está funcional e bit-exact contra o llama.cpp:

- [x] Parser GGUF v3
- [x] Tokenizer SPM (Llama) e BPE (Qwen2/GPT-2)
- [x] Forward pass f32 completo: RMSNorm, RoPE, GQA, SwiGLU, KV-cache
- [x] Quantização Q8_0 — matmul direto no espaço inteiro (sem expansão f32)
- [x] Sampling: temperatura, top-p, greedy
- [x] CLI de geração com timings

O backend **Vulkan** em 1 GPU está correto e razoavelmente rápido; o multi-GPU ainda não existe:

- [x] Decode residente em 1 GPU, bit-exact vs CPU (`--gpu-resident`)
- [x] 1 command buffer/token, pipelines e descriptors persistentes
- [x] Seleção automática da GPU com mais VRAM livre (evita spill para GTT, que custava 7× de banda)
- [x] Perfil por operação (`LLAMA_RS_PROFILE=1`): ms/token por op + custo de host por fase

| Modelo | llama-rs | llama.cpp Vulkan | razão |
|---|---|---|---|
| Qwen2.5-14B Q8_0, 1× MI50 | **28.0 tok/s** | 40.59 | 0.69× |
| Qwen2.5-0.5B Q8_0, 1× MI50 | 123 tok/s | 334 | 0.37× |

- [ ] Layer-split entre as 2 MI50, para rodar modelos acima de 16 GiB (`--gpu` hoje é um row-split ingênuo e não-residente, ~18× mais lento que 1 GPU residente)

Detalhes, benchmarks e próximos passos em [`PROGRESS.md`](PROGRESS.md).

---

## Hardware alvo

| GPU | Arquitetura | VRAM | API |
|---|---|---|---|
| AMD MI50 (× 2) | GCN 5.1 / gfx906 | 16 GB HBM2 cada | Vulkan 1.2 |

Em token generation (batch-1) com **Q8_0** o limite é a banda de memória — nosso matvec lê a 717 GB/s, perto do teto da placa. Com **K-quants** o gfx906 vira **compute-bound** (sem matrix cores), o que muda quais otimizações valem a pena. Os 16 GiB por GPU são o outro limite duro: modelos de 20–28 GiB só rodam com layer-split entre as duas.

> **NVIDIA Tesla K80 — fora de escopo (decisão deliberada).** O backend enumera só devices AMD (`crates/llama-vulkan/src/device.rs`). A K80 (Kepler, wave32) é incompatível com os shaders wave64 escritos para o gfx906 e é mais lenta que uma única MI50; suportá-la exigiria uma segunda família de kernels sem ganho líquido. Ela também não poderia dividir tensores com as MI50 (sem P2P entre vendors diferentes) — no máximo serviria como worker isolado de layer-split, fora do design atual.

---

## Estrutura do workspace

```
crates/
├── gguf/              # Parser do formato GGUF v3 (zero-copy sobre slice)
├── llama-tokenizer/   # Tokenizer SPM (Llama) e BPE (Qwen2/GPT-2)
├── llama-model/       # Forward pass: attention, RMSNorm, RoPE, SwiGLU, matmul
├── ggml-cpu/          # Operações GGML de baixo nível no CPU
├── llama-sampling/    # Estratégias de sampling
└── llama-cli/         # CLI de geração de texto
```

---

## Uso (CPU)

```bash
# Build release
cargo build --release -p llama-cli

# Geração simples
./target/release/llama-cli \
    -m models/stories260K.gguf \
    -p "Once upon a time" \
    -n 128 \
    --timings

# Benchmark vs llama.cpp
./scripts/benchmark.sh
```

Variáveis de ambiente do benchmark:

```bash
BENCH_N=128 BENCH_PROMPT="The dragon said" ./scripts/benchmark.sh
```

---

## Benchmark atual (CPU)

Medido em token generation (greedy, temp=0, seed=42):

| Modelo | llama.cpp | llama-rs | ratio |
|---|---|---|---|
| stories260K (f32) | ~1000 tok/s | ~1045 tok/s | 1.04× |
| qwen2.5-0.5b-q8_0 | ~12 tok/s | ~4.3 tok/s | 0.36× |

O gap no Qwen2 é esperado: llama.cpp usa kernels AVX2 vetorizados para Q8\_0×Q8\_0 e thread pool persistente. O foco desta implementação é o backend Vulkan, não otimização CPU.

---

## Roadmap Vulkan

1. [x] **Sub-alocador de memória (VMA)** — evitar falhas de alocação em drivers conservadores (AMDVLK com limite de 2 GB por alocação)
2. [x] **Dequantização packed-int** — ativação int8 + `dotPacked4x8` (gate validado em RADV/gfx906; ganho medido ~0%, kernel não é ALU-bound)
3. [x] **Shaders wave64** — subgroup ops corretas para gfx906 (MI50), corrigindo o bug de WARP\_SIZE=32 upstream
4. [x] **Contextos de grafo persistentes** — 1 command buffer/token, pipelines e descriptors persistentes (Fase 8.1)
5. [x] **Kernel matvec** — o gargalo não era ocupação nem ALU, como se supunha: era **spill para GTT** na GPU do display (95 GB/s contra 714 GB/s na GPU livre). Com a colocação correta, quantização da ativação em dispatch próprio e rmsnorm paralelo: 4.77 → 28 tok/s no 14B. O matvec agora lê a 717 GB/s, perto do teto — o resto vem de ler menos bytes, não de mais paralelismo
6. [ ] **Layer-split entre as 2 MI50** — camadas `0..N/2` numa GPU e `N/2..N` na outra, com **1 sincronização por token** na fronteira (0.06 ms pelos 59.3 µs medidos). É o que torna executáveis os modelos de 20–28 GiB que não cabem em 16 GiB — a faixa do Qwen3.6-27B
7. ~~**Row-split / tensor-parallel**~~ — **descartado por medição**: não há P2P de VRAM entre estas placas (`OPAQUE_FD` falha no import; `DMA_BUF` importa como host-visible e lê a 10.2 GB/s contra 717 GB/s locais), e os 96 all-reduces por token custam 5.69 ms contra os 10.8 ms economizados. O mesmo limite aparece no ROCm (`NO_PEER_COPY=1`). Ver `docs/estrategia-inferencia-mi50.md` §3 e §7
8. **NVIDIA Tesla K80 — fora de escopo**, ver seção "Hardware alvo"

Estado detalhado, benchmarks e ordem recomendada dos próximos passos: [`PROGRESS.md`](PROGRESS.md).

---

## Modelos testados

| Modelo | Formato | Arquitetura |
|---|---|---|
| stories260K | f32 GGUF | Llama |
| Qwen2.5-0.5B-Instruct | Q8\_0 GGUF | Qwen2 |

---

## Requisitos

- Rust 1.87+ (ver `rust-toolchain.toml`)
- Modelos no formato GGUF v3 (compatível com llama.cpp)
- Para Vulkan (GPU, opcional): 2× GPU AMD (RADV/AMDVLK) com suporte a Vulkan 1.2+ e subgroup ops wave64 (gfx906 validado)

---

## Licença

MIT
