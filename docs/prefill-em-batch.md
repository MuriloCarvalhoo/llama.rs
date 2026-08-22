# Prefill em batch: por que falta, e o desenho

## O problema

`gerar_streaming_residente` (`crates/llama-model/src/gpu.rs`) processa o prompt um token
por vez:

```rust
for (pos, &t) in prompt_ids.iter().enumerate() {
    logits = gpu.decode(t, pos)?;
}
```

Cada token dispara um forward completo, que lê **os 19,5 GB de pesos inteiros**. Um prompt
de 500 tokens custa 500 × 19,5 GB = 9,75 TB, ou ~25 s a 717 GB/s divididos entre as duas
GPUs. É a diferença entre um protótipo e algo usável com contexto real, e não aparece em
nenhum número de tok/s de decode — que medimos sempre com prompt curto.

O llama.cpp tem dois caminhos para a mesma multiplicação:

| | shader | quando |
|---|---|---|
| decode | `mul_mat_vec*.comp` | 1 token — memory-bound, lê o peso por token |
| prefill | `mul_mm.comp` | N tokens — GEMM com tiling em LDS, lê o peso **uma vez** para os N |

Com N tokens em batch o kernel passa a ser compute-bound, e a MI50 tem folga de sobra ali:
~52 TOPS int8 com `V_DOT4_I32_I8` contra os 717 GB/s de banda.

## O desenho, em duas etapas

A segunda é a que dá o ganho grande, mas a primeira já entrega a maior parte por uma
fração do trabalho, e é pré-requisito da outra.

### Etapa 1 — matvec de N colunas (o `num_cols` do llama.cpp)

O kernel continua sendo o matvec que já temos; muda só o laço interno, que passa a
acumular N resultados contra o mesmo peso já em registrador:

```glsl
layout(constant_id = 2) const uint COLS = 1u;   // nova specialization constant
...
for (uint c = 0u; c < COLS; c++) {
    acc[r][c] += dot(peso_ja_lido, ativacao[c]);
}
```

Com N = 8, os pesos são lidos 500/8 = 62 vezes em vez de 500 — **8× menos tráfego**. O
custo é registrador: `ROWS_PER_WAVE × COLS` acumuladores vivos, então na prática a
combinação útil é `ROWS_PER_WAVE = 1` com `COLS = 8`. É a mesma tensão de ocupância que a
varredura de `scripts/tune-matvec.sh` mede, e por isso as duas constantes moram juntas.

O llama.cpp pré-compila as variantes 1..8 exatamente assim (`{wg_size, rm_kq, i+1}`).

O teto de N foi 8 por um bom tempo, e não por medição: o `q6_k_matvec.comp` era o único
que dimensionava os acumuladores por uma constante fixa (`MAX_COLS = 8`) em vez de por
`COLS`, e com 16 colunas escrevia fora do array. Com isso resolvido o teto de
`LLAMA_RS_BATCH` é 32, e **onde a curva vira é empírico**: `ROWS_PER_WAVE × COLS`
acumuladores vivos por lane, e em algum ponto a ocupância cai mais do que o reuso do peso
rende. O padrão continua 8, que é o valor medido.

### Etapa 2 — GEMM com tiling em LDS

Implementado em `shaders/mul_mm.comp`, **desligado por padrão**
(`LLAMA_RS_PREFILL_GEMM=1` liga). Só Q4_K, que é 53% do tempo de matvec no Qwen3.8-27B; os
demais tipos seguem na etapa 1.

O que muda em relação ao matvec-COLS: o reuso sai do registrador e vai para a LDS. Tile de
**128 linhas × COLS colunas × 32 elementos de K**, workgroup de 256 threads como grade
32×8, cada thread com 4 linhas × COLS/8 colunas — intensidade `TM·TN/(TM+TN)` = 2 com
COLS=32. `dotPacked4x8AccSatEXT` no laço interno, que é onde a MI50 tem folga no prefill
(~52 TOPS de int8 contra 717 GB/s).

`BK = 32` não é escolha livre: é o sub-bloco de quantização do Q4_K **e** o bloco do
`quantize_x`, então cada passo de K tem exatamente uma escala de cada lado. O termo afim do
Q4_K (`-dmin·m·soma(x)`) sobrevive ao tiling porque `soma(x)` só depende da coluna — sai
uma vez por (coluna, passo de K) para a LDS.

**Medido (2026-08-21, prompt de ~3,2k tokens, `TOTAL GPU` das duas GPUs, ms por token de
prefill):**

| config | ms/token | parede (carga+prefill+8 tok) |
|---|---:|---:|
| batch 8, matvec-COLS (padrão anterior) | 18,7 | 66,1 s |
| batch 16, matvec-COLS | 18,4 | 66,0 s |
| batch 32, matvec-COLS | 27,4 | 94,6 s |
| batch 8 + GEMM | 21,8 | 76,7 s |
| batch 16 + GEMM | 14,6 | — |
| **batch 24 + GEMM** | **10,8** | — |
| batch 32 + GEMM | 13,2 | 49,4 s |

Duas curvas que se cruzam: o matvec-COLS satura em 16 e despenca em 32 (pressão de
registrador, como previsto acima); o GEMM piora em bloco pequeno (o tile de 128×COLS não
enche), tem o ótimo em 24 e volta a cair em 32. −42 % contra o padrão antigo — o critério
dos 20 % foi batido: **adotado como padrão** `LLAMA_RS_BATCH=24` + GEMM (o knob agora
desliga com `LLAMA_RS_PREFILL_GEMM=0`).

## O que mais precisa virar batch

O matvec não é o único: **todo o resto do plano assume um vetor**. Os que precisam de uma
dimensão de batch, em ordem de dificuldade:

| op | mudança |
|---|---|
| `quantize_x` | N vetores em vez de 1 — só multiplicar o índice |
| `rmsnorm` | um workgroup por token em vez de um total |
| `swiglu_quant`, `add`, `gate_quant` | elementwise: só o tamanho muda |
| `rope` | posição por token, não uma só |
| `kv_append` | N posições contíguas — uma cópia só |
| `attention` | **máscara causal**: o token i só vê 0..i |
| `delta_net`, `dn_conv` (qwen35) | recorrência: o laço sobre os tokens vai **para dentro** do kernel |

A atenção é a única com trabalho real de algoritmo. O `delta_net` e a convolução causal são
os únicos com recorrência de verdade — o token i depende do estado depois do i-1 —, e a
saída não é rodá-los em sequência de dispatches: é pôr o laço sobre os tokens do bloco
dentro do kernel, com o estado em registrador entre eles. Assim o estado toca a memória
global uma vez na entrada e uma na saída do dispatch, em vez de 2×N vezes com uma barreira
de pipeline entre cada par. É o desenho do `gated_delta_net.comp` do llama.cpp, e é o que
torna o batch grande viável: em batch 32, o desenho de um dispatch por token daria 4 × 32
dispatches por camada linear × 48 camadas = 6 mil por bloco só de delta-net.

`dn_gates` e `dn_l2_qk` não têm recorrência nenhuma e batcham por `gl_WorkGroupID.y`, como
as ops de atenção. O `dn_l2_qk` precisa de um `stride` no push porque a entrada dele é o
buffer da convolução, que traz `q | k | v` por token: o passo entre tokens na entrada é
`conv_dim`, na saída é `n_k_heads × d_state`. O `delta_net` precisa do mesmo pelo `v`.

## O que não precisa

O prefill não amostra: só o **último** token do prompt precisa de logits. A projeção final
(`output`, 0,63 GB em Q6_K) roda uma vez, não N.

## Onde isso entra no código

Um segundo plano, irmão de `build_plan`, dimensionado para `n_batch` tokens, com os mesmos
pesos residentes e buffers de ativação `n_batch` vezes maiores. O prompt é consumido em
blocos de `n_batch`; o resto (`< n_batch` tokens) cai no caminho token-a-token que já
existe, que continua sendo o caminho do decode.
