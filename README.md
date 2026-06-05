# llama-rs

Reescrita do zero em Rust do runtime de inferência LLM, com foco no backend **Vulkan** para GPUs AMD MI50 e NVIDIA Tesla K80.

O projeto resolve limitações arquiteturais conhecidas do llama.cpp nessas GPUs: dequantização elemento-a-elemento ineficiente no Vulkan, incompatibilidade com wave64 (AMD), ausência de row-split em multi-GPU e overhead de CPU redundante por token gerado.

---

## Motivação

O llama.cpp tem quatro problemas confirmados que afetam diretamente MI50 e K80:

| Problema | Impacto no hardware alvo |
|---|---|
| Dequantização Vulkan por elemento | Overhead algorítmico puro em cargas memory-bound |
| Flash Attention incompatível com wave64 | 2× mais lento em AMD RDNA/GCN com FA ativo |
| `WARP_SIZE` fixo em 32 | Output corrompido em gfx906 (MI50) para modelos MoE |
| Multi-GPU sem row-split | 3× mais lento vs ROCm no mesmo hardware |

Este projeto constrói a pipeline completa em Rust (tokenizer → forward pass → sampling) como base para um backend Vulkan que corrija esses problemas com controle total sobre os shaders SPIR-V e o agendamento de memória.

---

## Estado atual

A pipeline **CPU** está funcional e bit-exact contra o llama.cpp:

- [x] Parser GGUF v3
- [x] Tokenizer SPM (Llama) e BPE (Qwen2/GPT-2)
- [x] Forward pass f32 completo: RMSNorm, RoPE, GQA, SwiGLU, KV-cache
- [x] Quantização Q8_0 — matmul direto no espaço inteiro (sem expansão f32)
- [x] Sampling: temperatura, top-p, greedy
- [x] CLI de geração com timings
- [ ] Backend Vulkan (em desenvolvimento)

---

## Hardware alvo

| GPU | Arquitetura | VRAM | API |
|---|---|---|---|
| NVIDIA Tesla K80 | Kepler (sm_37) | 24 GB GDDR5 (2× 12 GB) | Vulkan 1.1 |
| AMD MI50 (× 2) | GCN 5.1 / gfx906 | 16 GB HBM2 cada | Vulkan 1.2 |

As três GPUs são **memory-bandwidth bound** em token generation (batch-1). O teto de performance é determinado pela largura de banda, não pelos FLOPS — isso torna a dequantização eficiente e o row-split multi-GPU críticos.

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

1. **Sub-alocador de memória (VMA)** — evitar falhas de alocação em drivers conservadores (AMDVLK com limite de 2 GB por alocação)
2. **Dequantização packed-int** — mantém valores inteiros até após o dot product (elimina overhead por elemento)
3. **Shaders wave64** — subgroup ops corretas para gfx906 (MI50), corrigindo o bug de WARP\_SIZE=32 upstream
4. **Contextos de grafo persistentes** — grafo SPIR-V construído uma vez, re-executado por token sem reconstrução
5. **Row-split multi-GPU** — dividir o matmul por linhas entre as duas MI50 em vez de só por camada
6. **Suporte K80** — validar shaders no sm\_37 via Vulkan 1.1 sem extensões modernas

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
- Para Vulkan (futuro): driver com suporte a Vulkan 1.1+, `VK_KHR_storage_buffer_storage_class`

---

## Licença

MIT
