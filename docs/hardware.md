# Hardware

## Alvo validado

| GPU | Arquitetura | VRAM | API |
|---|---|---|---|
| AMD MI50 / MI60 / Radeon Pro VII | GCN 5.1 / gfx906 | 16 GB HBM2 | Vulkan 1.1+ |

O backend Vulkan usa subgroup ops de tamanho 64 (wave64) — necessário para o gfx906 e para
qualquer outra GPU AMD GCN/CDNA com a mesma largura de wavefront. GPUs com wave32 (a maioria das
NVIDIA, e AMD RDNA) não são suportadas hoje.

## Por que só AMD

O backend enumera apenas physical devices com vendor ID AMD
(`crates/llama-vulkan/src/device.rs`). Não é uma limitação técnica do Vulkan — é escopo: os
shaders foram escritos e ajustados para o padrão de acesso e a largura de wavefront do gfx906, e
suportar outro vendor exigiria uma segunda família de kernels sem o mesmo nível de ajuste fino.

GPUs Kepler mais antigas (ex.: Tesla K80) foram avaliadas e descartadas deliberadamente: são
wave32, incompatíveis com os shaders wave64 escritos para o gfx906, mais lentas que uma única
MI50, e não conseguiriam dividir tensores com as MI50 mesmo que os kernels existissem (sem P2P
entre vendors diferentes). No máximo serviriam como worker isolado de layer-split — fora do
desenho atual, e sem ganho líquido óbvio para justificar uma segunda família de kernels.

## Limites de VRAM e o motivo do layer-split

16 GB por GPU é o teto: modelos de 20–28 GiB (a faixa de vários modelos de ~27–32B em quantização
K) não cabem numa placa só. O layer-split entre 2 GPUs existe por isso — não é sobre velocidade
(não há troca de dados suficiente entre as GPUs para valer a pena tentar paralelizar o cálculo em
si), é sobre destravar a capacidade.

Não há P2P de transferência de VRAM entre duas MI50 neste tipo de setup (sem NVLink/Infinity
Fabric equivalente) — a única coisa que atravessa a fronteira entre as GPUs a cada token é o
vetor de estado residual (alguns KB), uma vez por token. Ver
[`docs/estrategia-inferencia-mi50.md`](estrategia-inferencia-mi50.md) para a medição que
descartou tensor-parallel/row-split neste hardware.

## Múltiplas GPUs, uma delas com monitor ligado

Se uma das GPUs também dirige o display, o driver AMD pode realocar em GTT (memória do host, via
PCIe) o excedente do que não coube na VRAM dela — silenciosamente, sem erro. O sintoma é banda
efetiva muito abaixo do esperado, não uma falha de alocação. O backend seleciona a GPU com mais
VRAM livre via `VK_EXT_memory_budget` (não pelo índice do device) e reserva uma margem fixa antes
de calcular o layer-split, para não deixar a GPU do display sem VRAM livre nenhuma.

`LLAMA_RS_GPU=N` força um índice específico se a seleção automática não for o que se quer.
