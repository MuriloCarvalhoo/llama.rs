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
| MTP | *pendente* | | a implementar |
| MTP desligado (mesma build) | *pendente* | | controle do anterior |

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
