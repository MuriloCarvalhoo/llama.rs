# Benchmarks

Todos os números abaixo são token generation (decode), não prompt processing — greedy ou
sampling padrão, medidos com `--timings` (mede da emissão do primeiro token em diante, exclui o
tempo de prefill do prompt). Hardware: 2× AMD MI50 (Radeon Pro VII, gfx906, 16 GB HBM2 cada).

## Qwen3.8-27B (híbrido atenção + gated delta-net), 2× MI50 layer-split

| Config | tok/s |
|---|---:|
| llama-rs, Q5_K_M | 25.5 |
| llama-rs, Q4_K_M | **26.9** |
| llama.cpp, Vulkan, baseline (sem MTP) | 17.0 – 20.6 |
| llama.cpp, ROCm/HIP, baseline (sem MTP) | 19.5 |
| llama.cpp, ROCm/HIP, com MTP bem configurado | até 23.1 |

O intervalo do llama.cpp reflete profundidade de contexto — os dois extremos (prompt curto vs
~8K tokens de contexto acumulado) foram medidos separadamente. Os números do llama-rs foram
medidos com prompt curto, mesma classe de condição do extremo baixo do llama.cpp.

Essa é a arquitetura mais recente e mais complexa suportada: 64 camadas, 3 em cada 4 usam
**gated delta-net** (atenção linear com estado recorrente por cabeça, sem KV-cache) em vez de
atenção — ver [`qwen35-arquitetura.md`](qwen35-arquitetura.md). O layer-split não é só sobre
velocidade aqui: o modelo (15–20 GiB dependendo da quantização) não cabe numa única MI50 de 16 GB.

## Qwen2.5-32B Q5_K_M, 2× MI50 layer-split

| | tok/s | razão |
|---|---:|---:|
| **llama-rs** | 19.3 | **1.07×** |
| llama.cpp (ROCm) | 18.02 | — |

O caso que justifica dividir entre as duas GPUs: 22.2 GB não cabem numa placa só.

## Qwen2.5-14B Q8_0, 1× MI50

| | tok/s | razão |
|---|---:|---:|
| llama-rs | 28.0 | 0.69× |
| llama.cpp (Vulkan) | 40.59 | — |

Modelo que cabe inteiro numa GPU — aqui o llama.cpp ainda ganha. 40.59 tok/s corresponde a ~62%
de utilização de banda de memória (MBU); o teto físico de uma MI50 fica perto de 65 tok/s para
este tamanho de modelo, então não há uma vitória fácil escondida aqui.

## Qwen2.5-0.5B Q8_0, 1× MI50

| | tok/s | razão |
|---|---:|---:|
| llama-rs | 123 | 0.37× |
| llama.cpp (Vulkan) | 334 | — |

Modelo pequeno demais para o overhead por-dispatch do llama-rs se amortizar — o llama.cpp tem
kernels mais especializados para esse regime.

## Backend CPU (referência, sem GPU)

| Modelo | llama.cpp | llama-rs | razão |
|---|---:|---:|---:|
| stories260K (f32) | ~1000 tok/s | ~1045 tok/s | 1.04× |
| Qwen2.5-0.5B Q8_0 | ~12 tok/s | ~4.3 tok/s | 0.36× |

O foco do projeto é o backend Vulkan; o caminho CPU existe para desenvolvimento e para modelos
pequenos, não é otimizado com os mesmos kernels vetorizados que o llama.cpp usa em AVX2.

## O que já foi tentado e descartado

- **Tensor-parallel / row-split entre as 2 GPUs** — sem P2P de VRAM neste hardware, o custo de
  sincronização por camada (96 all-reduces/token num modelo de 14B) supera a banda que
  economizaria. Ver [`estrategia-inferencia-mi50.md`](estrategia-inferencia-mi50.md).
- **MTP / speculative decoding "ingênuo"** — o gargalo do decode aqui é banda de memória, não
  latência: verificar 2 tokens ainda lê quase os mesmos bytes de pesos que verificar 1. Medido no
  llama.cpp de referência neste mesmo hardware/modelo: só **+3–5%**, não uma multiplicação. Ver
  [`mtp-e-k80.md`](mtp-e-k80.md).
