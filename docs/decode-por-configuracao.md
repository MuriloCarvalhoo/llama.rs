# Decode: tok/s por configuração

Registro vivo das medições de velocidade de decode, para não repetir experimento nem
comparar números medidos em condições diferentes. **Toda linha diz como foi obtida.**

## Como medir sem se enganar

Três armadilhas já custaram conclusões erradas neste projeto:

1. **A fronteira do layer-split varia entre execuções.** Ela é derivada da VRAM livre, e já
   vimos 139/149 dispatches virarem 134/154 entre duas rodadas do mesmo binário — sozinho
   isso move o resultado ~2%. Fixar com `LLAMA_RS_SPLIT=N`.
2. **tok/s é ruidoso; `TOTAL GPU` não é.** Na mesma configuração o tok/s oscilou de 19,8 a
   22,8 (15%), enquanto o `TOTAL GPU` do `LLAMA_RS_PROFILE=1` (timestamp query) ficou em
   ±0,3%. Para avaliar mudança de shader, comparar `TOTAL GPU`; tok/s só para o número final.
3. **Sempre `numactl --interleave=all`.** Sem isso o nó 0 estoura e a máquina trava — ver
   `travamento NUMA` no `scripts/run.sh`.

Protocolo padrão das linhas abaixo, salvo indicação em contrário:

```bash
LLAMA_RS_SPLIT=31 numactl --interleave=all target/release/llama-cli \
  -m models/Qwen3.8-27B-Q4_K_M.gguf --gpu-layer-split \
  -p "Explique em três frases o que é um transformer." -n 64 --no-display-prompt --timings
```

## llama-rs — Qwen3.8-27B Q4_K_M, 2× MI50 layer-split

| configuração | tok/s | TOTAL GPU (ms) | como |
|---|---:|---:|---|
| baseline (commit `31eef61`) | **22,3** | 42,40 | 6 execuções: 21,5 21,8 21,9 22,1 22,3 22,5 |
| `dn_gates` com NWAVE+vec4 (`5a4d2fc`) | — | **41,35** | 3 execuções: 41,44 41,56 41,04 |
| greedy (`--temp 0`) | 22,35 | — | 3 execuções; a amostragem custa ~1,3 ms/token |
| amostragem completa (temp 0.8, top-k 40) | 21,74 | — | 3 execuções, bem estável |
| MTP ligado | *pendente* | | falta o decode em batch |
| MTP desligado (mesma build) | *pendente* | | controle do anterior |

### Taxa de aceitação do MTP — **60,9 % medido**

`cargo test -p llama-vulkan --test mtp_aceitacao -- --nocapture`, greedy, 23 propostas
verificadas contra o token que o modelo de fato produziu: **14 acertos, 60,9 %**.

Isso é o teto do ganho: `tok/s = base × (1 + aceitação) = base × 1,61`.

| base | com MTP a 60,9 % |
|---:|---:|
| 22,3 (hoje) | 35,9 |
| 28 (banda parcial) | 45,1 |
| **31,1 (banda + fusão de ops)** | **50,1** |

**Contradiz o +3,4 % do llama.cpp** na tabela abaixo. A explicação mais provável é que a
implementação dele não faz batch eficiente num híbrido SSM — a cabeça do modelo prevê bem.
Medido com greedy; com amostragem a temperatura alta a aceitação cai.

### Varredura de geometria do matvec — sem efeito

`LLAMA_RS_MATVEC_GEOM=wg,linhas`, 3 execuções cada. Diferença de 1,2%, dentro do ruído:
geometria **não** é o gargalo.

| 256,2 | 256,3 | 512,2 | 512,3 |
|---:|---:|---:|---:|
| 21,98 | 22,12 | 22,24 | 22,25 |

### Onde vai o tempo (soma das 2 GPUs, ms/token)

Perfil de `LLAMA_RS_PROFILE=1` no baseline. Os três matvec são 77%.

| op | ms | % | banda |
|---|---:|---:|---:|
| matvec_q4k | 22,49 | 53,0 | 506 GB/s |
| matvec_q6k | 7,91 | 18,6 | 573 GB/s |
| matvec_q5k | 2,26 | 5,3 | 460 GB/s |
| `dn_gates` (antes da correção) | 2,87 | 6,8 | **33 GB/s** |
| as outras 13 ops | 6,91 | 16,3 | |
| **total** | **42,44** | | 501 GB/s agregado |

## Decode e prefill por comprimento de contexto — o gargalo é a atenção

Medido pelo `llama-server` (2026-08-21), 2× MI50 layer-split, `--ctx 32768`, greedy. O
log separa prefill de decode, que é o que torna o efeito visível:

| KV no momento do decode | prefill | decode |
|---:|---:|---:|
| ~1,5k tok | 73,2 tok/s | **18,5 tok/s** (54 ms/tok) |
| ~2,3k → 4,1k tok | 71,1 tok/s | **14,5 tok/s** (69 ms/tok) |
| ~26,5k tok | 39,9 tok/s | **4,3 tok/s** (233 ms/tok) |

O decode com contexto curto bate com o benchmark (22,3 tok/s medido com prompt de poucas
dezenas de tokens; aqui já há 1,5k no cache). Mas **a 26k tokens o decode custa 5,5× mais**,
e a conta diz de onde vem:

- KV lido por token a 26 472 de contexto: `26472 × 1024 × 2 (K,V) × 4 B × 16 camadas` =
  **3,47 GB**.
- Tempo a mais em relação ao decode de contexto curto: 233 − 42 = 191 ms.
- Banda efetiva da atenção: **18 GB/s** — contra os ~500 GB/s que os matvec sustentam.

A atenção dispara um workgroup por cabeça (24) e cada um varre o KV inteiro em série: em
contexto curto isso não aparecia (era 1,5% do passo), em 26k domina o token.

### O split-K resolveu — 9× no kernel

`attention_split.comp` fatia o KV entre os workgroups da dimensão Z e publica (m, l, acc)
parciais; `attn_reduce.comp` combina pela mesma álgebra do softmax online, que é
associativa nesses três. É o mesmo desenho do `flash_attn_split_k_reduce` do llama.cpp.

Medido com a geometria do Qwen3.8-27B e 26 472 posições (só o dispatch, sem o upload do
KV — `atencao_fatiada_e_mais_rapida_com_kv_longo`):

| caminho | por camada de atenção |
|---|---:|
| 1 workgroup por cabeça | 15,3 ms |
| **16 fatias** | **1,7 ms** |

São 16 camadas de atenção no modelo: 245 ms/token de atenção viram 27 ms.

A escolha é por token, em `splits_do_kv`: uma fatia a cada 512 posições, teto de 16. Com
contexto curto o caminho antigo continua sendo o gravado — a redução não se pagaria.
`LLAMA_RS_ATTN_SPLIT=N` força o número de fatias para comparar os dois no mesmo binário.

Ponta a ponta pelo servidor, **mesmo prompt de 9 110 tokens**, mesma máquina, só o kernel
mudando:

| | prefill | decode |
|---|---:|---:|
| 1 workgroup por cabeça | 155,1 s (58,7 tok/s) | **9,2 tok/s** |
| KV fatiado | 147,4 s (61,8 tok/s) | **19,6 tok/s** |

O decode **dobra**; o prefill quase não muda, porque lá a atenção divide o passo com os
matvec em batch — e é ele o próximo gargalo (11 min para os ~27k tokens que o opencode
manda no primeiro turno).

A amostragem, suspeita óbvia, **não** é o problema: 0,4–1,4 ms/token medidos.

O prefill em batch de 8 rende ~3× sobre o decode (73 contra 22 tok/s), não 8×: as 48
camadas delta-net são recorrentes e rodam um dispatch por token, e a atenção em batch tem
o mesmo problema de banda acima.

## llama.cpp no mesmo hardware — referência

Do `docs/mtp-e-k80.md` (2026-08-15, ROCm, **2× MI50**, Q5_K_M — não Q4_K_M):

| configuração | tok/s |
|---|---:|
| Qwen3.8-27B, sem MTP | 20,8 |
| Qwen3.8-27B, **MTP `--spec-draft-n-max 1`** | **21,5** (+3,4%) |
| Qwen3.8-27B, MTP `n-max 2` | 18,7 |
| Qwen3.8-27B, MTP `n-max 3` (default) | 14,9 (−28%) |
| Qwen3.6-35B-A3B **MoE** + speculative | **77,1** |
| Qwen3.6-27B denso | 23,8 |

O ganho de +3,4% do MTP mede a implementação do llama.cpp num modelo híbrido SSM, não o
teto teórico. A build local atualmente dá **segfault** com este modelo, então o número não
pôde ser reconfirmado.

## Tetos por banda — o que é fisicamente possível

Bytes lidos por token × 717 GB/s (banda de matvec medida neste projeto). O teto considera
**só** os matvec; some ~7 ms para as demais ops e o host.

| modelo | GB/token | teto matvec | com +7 ms |
|---|---:|---:|---:|
| Qwen3.8-27B Q4_K_M | 16,38 | 43,8 tok/s | 33,5 |
| Qwen3.8-27B Q3_K_M (estimado) | ~13,0 | 55 | 39,8 |
| Qwen3.8-27B Q2_K (estimado) | ~10,2 | 70 | 47 |
| Qwen3-14B Q3_K_M | ~7,0 | 102 | 52 |
| **Qwen3.6-35B-A3B MoE** | **1,98** | **363** | **103** |

**50 tok/s = 20,00 ms/token.** No 27B Q4_K_M os matvec sozinhos, com 100% de eficiência,
já custam 22,85 ms — o alvo não cabe sem reduzir bytes ou multiplicar tokens por passo.
Daí o MTP: `tok/s = base × (1 + taxa de aceitação)`.

## Modelos disponíveis e seus aceleradores

| arquivo | arch | acelerador | suportado pelo llama-rs |
|---|---|---|---|
| Qwen3.8-27B-Q4_K_M | `qwen35` | MTP (`nextn_predict_layers=1`) | sim (MTP a implementar) |
| Qwen3.8-27B-Cold-Fusion-…-MTP | `qwen35` | **idêntico ao acima** — "MTP" no nome é rótulo | sim |
| Qwen3.6-27B-Fable-…-MTP | `qwen35` | **idêntico** | sim |
| Qwen3.6-35B-A3B | `qwen35moe` | MoE 8/256 experts | **não** |
| Ornith-1.0-35B-MTP-APEX | `qwen35moe` | **MoE + MTP** | **não** |
| Qwen3.6-35B-A3B-DFlash | `dflash` | draft de 6 camadas, 0,42 GB | **não** |
| Qwen3-14B-Q3_K_M | `qwen3` | — | **não** (falta shader Q3_K) |
