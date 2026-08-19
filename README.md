# llama-rs

Runtime de inferência para LLMs escrito do zero em Rust, com um backend **Vulkan** próprio
(shaders SPIR-V escritos e ajustados à mão) voltado para GPUs AMD de datacenter mais antigas
(gfx906 — MI50/MI60/Radeon Pro VII) que os backends mainstream atendem mal.

Pipeline completa em Rust — parser GGUF, tokenizer, forward pass, sampling — sem depender do
ggml/llama.cpp em tempo de execução. O objetivo não é reimplementar o llama.cpp; é ter controle
total sobre os shaders e o agendamento de memória neste hardware específico, e usar esse controle
para bater os backends genéricos onde eles deixam banda na mesa.

## Resultado em destaque

Em 2× AMD MI50 (Vulkan), rodando **Qwen3.8-27B** (arquitetura híbrida atenção + gated
delta-net) dividido entre as duas placas por falta de VRAM numa só:

| | tok/s |
|---|---:|
| **llama-rs** (Q4_K_M) | **26.9** |
| llama.cpp, backend Vulkan (mesmo hardware) | 17.0 – 20.6 |
| llama.cpp, backend ROCm/HIP (mesmo hardware, backend mais rápido do llama.cpp aqui) | 19.5 – 23.1 |

Metodologia completa e os outros benchmarks (Qwen2.5-32B, Qwen2.5-14B) em
[`docs/benchmarks.md`](docs/benchmarks.md).

## Por quê

O llama.cpp é o runtime de referência, mas seu backend Vulkan trata GPUs AMD antigas como
NVIDIA-com-sotaque: dequantização genérica, `WARP_SIZE` fixo em 32 (o gfx906 é wave64), sem
aproveitar bem o padrão de acesso dos K-quants nessa arquitetura. Nada disso é bug — é o preço de
um backend que precisa rodar em dezenas de GPUs diferentes. Escrevendo os shaders à mão para um
hardware específico, dá para fechar boa parte dessa distância, e em alguns casos passar na frente.

Duas premissas iniciais do projeto não sobreviveram à medição, e ficam registradas para não serem
refeitas — ver [`docs/estrategia-inferencia-mi50.md`](docs/estrategia-inferencia-mi50.md):

- **Tensor-parallel / row-split não compensa neste hardware.** Sem P2P de VRAM entre as duas
  placas, a sincronização por camada custa mais do que a banda que economizaria. O que funciona é
  **layer-split** — não por velocidade, por capacidade: é o que permite rodar modelos que não cabem
  numa GPU só.
- **Decode batch-1 é limitado por banda de memória**, inclusive nos K-quants — não por ALU, como a
  hipótese original assumia.

## Uso

```bash
cargo build --release -p llama-cli --features gpu

./target/release/llama-cli \
    -m models/seu-modelo.gguf \
    --gpu-layer-split \
    -p "Once upon a time" -n 128 --timings
```

Sem GPU, ou para modelos pequenos, o caminho CPU funciona sem a feature `gpu`:

```bash
cargo build --release -p llama-cli
./target/release/llama-cli -m models/stories260K.gguf -p "Once upon a time" -n 128
```

`scripts/run.sh <pedaço-do-nome-do-modelo>` resolve o arquivo em `models/` (ou
`$LLAMA_RS_MODELS`) e escolhe `--gpu-resident` vs `--gpu-layer-split` pelo tamanho do modelo
contra a VRAM livre — ver `scripts/run.sh -h`.

## Estado

| | |
|---|---|
| Parser GGUF v3, tokenizer (SPM + BPE) | ✅ bit-exact vs llama.cpp |
| Forward CPU f32 (RMSNorm, RoPE, GQA, SwiGLU) | ✅ |
| Backend Vulkan, 1 GPU residente | ✅ Q8_0, Q5_K, Q6_K, Q4_K |
| Layer-split, 2 GPUs | ✅ |
| Qwen2 / Qwen2.5 (denso) | ✅ |
| Qwen3.5 / 3.8 (híbrido atenção + gated delta-net) | ✅ |
| MTP / speculative decoding | ⏳ não implementado — ver [`docs/mtp-e-k80.md`](docs/mtp-e-k80.md) |

## Estrutura do workspace

```
crates/
├── gguf/              # Parser do formato GGUF v3 (zero-copy sobre slice)
├── llama-tokenizer/   # Tokenizer SPM (Llama) e BPE (Qwen2/GPT-2)
├── llama-model/       # Forward pass: attention, RMSNorm, RoPE, SwiGLU, delta-net
├── ggml-cpu/          # Dequantização e operações GGML de baixo nível no CPU
├── llama-vulkan/      # Backend Vulkan: shaders SPIR-V, decode residente, layer-split
├── llama-sampling/    # Estratégias de sampling
└── llama-cli/         # CLI de geração de texto
```

## Hardware suportado

Vulkan 1.1+ com subgroup ops; validado em AMD gfx906 (MI50/MI60/Radeon Pro VII), wave64. O
backend enumera apenas devices AMD — ver [`docs/hardware.md`](docs/hardware.md) para o porquê e
os limites de VRAM.

## Documentação

- [`docs/benchmarks.md`](docs/benchmarks.md) — metodologia e todos os números medidos
- [`docs/qwen35-arquitetura.md`](docs/qwen35-arquitetura.md) — a arquitetura híbrida do Qwen3.5/3.8
- [`docs/estrategia-inferencia-mi50.md`](docs/estrategia-inferencia-mi50.md) — o que foi tentado,
  medido e descartado
- [`docs/hardware.md`](docs/hardware.md) — requisitos e limites de hardware
- [`docs/debugging.md`](docs/debugging.md) — profiling por operação e timeline CPU+GPU

## Requisitos

- Rust 1.96+ (ver `rust-toolchain.toml`)
- Modelos no formato GGUF v3 (compatível com llama.cpp)
- Para o backend Vulkan (opcional, feature `gpu`): GPU AMD com Vulkan 1.1+ e subgroup ops wave64

## Licença

MIT
