# MTP no Qwen3.8-27B, e por que o draft não vai para a Tesla K80

Medições de 2026-08-15 no llama.cpp `49ac92eb0` (b950, ROCm), Qwen3.8-27B Q5_K_M,
2× MI50 em layer-split, 128 tokens, greedy, `llama-cli` com o mesmo prompt.

## O MTP só ajuda com `--spec-draft-n-max 1`

| configuração | tok/s |
|---|---|
| sem MTP | 20.8 |
| **MTP, `--spec-draft-n-max 1`** | **21.5** |
| MTP, `--spec-draft-n-max 2` | 18.7 |
| MTP, `--spec-draft-n-max 3` (**default**) | 14.9 |
| MTP, `n-max 1` + `--spec-draft-p-min 0.6` | 1.8 |

O default do llama.cpp (`n-max 3`) deixa o modelo **28% mais lento** que não usar MTP
nenhum. A razão está no próprio modelo: `qwen35.nextn_predict_layers = 1`, ou seja, a
cabeça MTP foi treinada para prever **um** token. Pedir três obriga a rodá-la em cadeia,
realimentando a própria previsão, e a qualidade da proposta desaba — cada rejeição custa
um forward inteiro do modelo principal.

Com `n-max 1` o ganho sobre não usar MTP é de **+3.4%**. É pouco perto do que
speculative decoding costuma render porque o gargalo aqui é banda, não latência: o
forward de verificação lê os mesmos 19.5 GB de pesos para 1 ou 2 tokens, então o teto do
ganho é a taxa de aceitação, e o custo do draft (324 MB, ver abaixo) sai inteiro do
orçamento.

`--spec-draft-p-min 0.6` colapsa para 1.8 tok/s — um caminho degenerado, não uma
degradação suave. Ficar no default (0.0).

## A cabeça MTP é 1.66% do modelo

| | bytes |
|---|---|
| modelo principal (64 camadas) | 19.50 GB |
| bloco MTP (`blk.64`, NextN) | **324 MB** |

O bloco é uma camada de decoder completa (attn 43+4+4+22 MB, FFN 61+61+73 MB) mais a
`eh_proj` de 56 MB, que combina o hidden state do último layer com o embedding do token
amostrado. Ele reserva ainda ~2250 MiB de contexto próprio na segunda GPU.

## Rodar o MTP na K80 e o modelo nas MI50: não compensa

A ideia é atraente — a cabeça é pequena, a K80 está parada, e o llama.cpp até tem
`--spec-draft-device` para escolher onde o draft roda. Não compensa por três motivos, em
ordem de importância:

**1. O draft é sequencial com o modelo principal — não há o que sobrepor.** O MTP precisa
do hidden state do token atual para propor o próximo, e a verificação precisa da proposta.
Um passo é `principal → draft → principal`. Colocar o draft noutra GPU não ganha tempo de
parede; só muda onde os 324 MB são lidos.

**2. A K80 é ~4× mais lenta para ler esses 324 MB.**

| | banda | tempo do draft |
|---|---|---|
| MI50 (medido) | 717 GB/s | **0.45 ms** |
| K80 (GDDR5, ~180 GB/s efetivos) | 240 GB/s de pico | **1.8 ms** |

Some a transferência do hidden state (`n_embd` = 5120 floats = 20 KB) entre vendors
diferentes: sem P2P, o caminho é MI50 → host → K80 → host → MI50, e medimos 59 µs por
sincronização host-mediada neste hardware (`crates/llama-vulkan/src/spike.rs`) — mais
0.12–0.24 ms por passo. O draft sairia de 0.45 ms para ~2.0 ms: **+1.55 ms por passo**,
contra os 0.45 ms que se economiza nas MI50. Perda líquida de ~4%.

**3. O backend nem compila para a K80.** O build atual só tem `libggml-hip.so`, e
`--list-devices` mostra apenas as duas MI50. Para a K80 (Kepler, sm_37) apareceria:

- o driver instalado é o 470.256.02, que para em CUDA 11.4;
- o `ggml/src/ggml-cuda/CMakeLists.txt` documenta `50 == Maxwell, lowest CUDA 12
  standard` — sm_37 está abaixo do mínimo;
- o dot int8 por byte (`__dp4a`), que é o que torna os K-quants viáveis, exige sm_61.

Ou seja, seria preciso portar kernels para uma arquitetura de 2014 para ganhar acesso a
uma GPU 4× mais lenta, num trecho que representa 1.66% do trabalho e não pode ser
sobreposto. É a mesma conclusão a que o projeto já tinha chegado por outro caminho
(`README.md`, seção "Hardware alvo"): a K80 só faria sentido como worker isolado de
layer-split, e nem isso paga o custo de uma segunda família de kernels.

**Onde a K80 poderia ajudar, em tese:** servindo requisições independentes em paralelo
(throughput, não latência de um stream). Isso é outro produto, não este.

## O que isso significa para o llama-rs

O alvo do Qwen3.8-27B em 2× MI50 passa a ser **21.5 tok/s** (llama.cpp com MTP bem
configurado). Sem MTP, 20.8. Como o llama-rs ainda não implementa speculative decoding, a
comparação justa no primeiro momento é contra os 20.8 — e o MTP fica como trabalho
posterior, com a lição já registrada: propor **um** token, não três.
