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

### Etapa 2 — GEMM com tiling em LDS

Só compensa acima de ~32 tokens por batch, quando o peso lido cabe num tile reusado por
muitas colunas. É o `mul_mm.comp` deles. Fica para depois de a etapa 1 estar medida.

## O que mais precisa virar batch

O matvec não é o único: **todo o resto do plano assume um vetor**. Os que precisam de uma
dimensão de batch, em ordem de dificuldade:

| op | mudança |
|---|---|
| `quantize_x` | N vetores em vez de 1 — só multiplicar o índice |
| `rmsnorm` | um workgroup por token em vez de um total |
| `swiglu`, `add`, `gate_mul` | elementwise: só o tamanho muda |
| `rope` | posição por token, não uma só |
| `kv_append` | N posições contíguas — uma cópia só |
| `attention` | **máscara causal**: o token i só vê 0..i |
| `delta_net` (qwen35) | recorrência: é serial por construção, não paraleliza em batch |

A atenção é a única com trabalho real de algoritmo, e o `delta_net` é o único que **não**
tem versão em batch — a recorrência do token i depende do estado depois do token i-1. Para
o qwen35 o prefill teria de rodar as camadas lineares em sequência mesmo em batch, o que
ainda vale a pena porque as projeções (que são a maior parte dos bytes) continuam em batch.

## O que não precisa

O prefill não amostra: só o **último** token do prompt precisa de logits. A projeção final
(`output`, 0,63 GB em Q6_K) roda uma vez, não N.

## Onde isso entra no código

Um segundo plano, irmão de `build_plan`, dimensionado para `n_batch` tokens, com os mesmos
pesos residentes e buffers de ativação `n_batch` vezes maiores. O prompt é consumido em
blocos de `n_batch`; o resto (`< n_batch` tokens) cai no caminho token-a-token que já
existe, que continua sendo o caminho do decode.
