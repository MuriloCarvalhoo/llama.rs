# Análise: Como o llama.cpp trata o Qwen2 e por que é mais rápido

**Data:** 2026-06-04  
**Benchmark:** stories260K: llama-rs 1.08x | Qwen2.5-0.5b-q8_0: llama-rs **0.356x** (2.8x mais lento)

---

## 1. Arquitetura Qwen2.5-0.5B (do GGUF)

| Parâmetro | Valor |
|-----------|-------|
| `n_embd` | 896 |
| `n_layer` | 24 |
| `n_head` | 14 |
| `n_head_kv` | 2 |
| `head_dim` | 64 |
| `kv_dim` | 128 (2 × 64) |
| `n_ff` | 4864 |
| `rope_dim` | 64 |
| `freq_base` | 1,000,000 |
| Diferença vs Llama | **biases** em Q/K/V, GQA extremo (14 heads → 2 KV heads) |

---

## 2. O que o llama.cpp faz de especial para Qwen2

### 2.1 Registro de arquitetura dedicado

`llama.cpp/src/llama-arch.cpp:34`
```c
{ LLM_ARCH_QWEN2, "qwen2" }
```

O llama.cpp mantém um registro separado de arquiteturas (`LLM_ARCH_QWEN2`) que define exatamente quais tensores existem, quais operadores usar, e como mapear os campos do GGUF. Não há runtime fallback — tudo é resolvido em compile-time.

### 2.2 Graph-based execution (não serial como llama-rs)

O llama.cpp compila um **grafo de computação** (`ggml_cgraph`) e o entrega ao backend para execução. O grafo é:

1. Construído UMA VEZ na inicialização do contexto
2. Executado pelo backend (CPU thread pool ou GPU)
3. Re-executado com novos dados de entrada sem reconstrução

**Diferença crítica:** llama-rs chama funções sequencialmente no forward pass. llama.cpp agenda todas as operações como nós do grafo e o thread pool as executa com mínimo overhead de dispatch.

### 2.3 Thread pool persistente (não rayon por chamada)

`llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp:100-135`

```c
struct ggml_cpu_context {
    int n_threads;
    ggml_threadpool_t threadpool;  // threads dormentes, acordadas por cond var
};
```

- Threads ficam **dormentes** entre tokens, acordadas por `pthread_cond_signal`
- Zero overhead de criação/destruição de threads
- rayon, por outro lado, tem overhead de fork/join por chamada (~1-5µs)
- Para 168 matmuls por forward pass: 168 × 5µs = ~840µs de overhead só de rayon

### 2.4 Kernel AVX2 para Q8_0×Q8_0

`llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c`

```c
// 32 bytes processados em ~2-3 instruções AVX2
__m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
__m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
const __m256 q = mul_sum_i8_pairs_float(qx, qy);  // VPMADDUBSW + VPMADDWD
acc = _mm256_fmadd_ps(d, q, acc);                  // FMA
```

**Nossa implementação atual** (`ops.rs:q8_0_q8_0_dot`):
```rust
for i in 0..Q {  // loop escalar de 32 iterações
    dot += (qsw[i] as i8 as i32) * (qsx[i] as i8 as i32);
}
```

O LLVM pode auto-vectorizar, mas `mul_sum_i8_pairs_float` usa `_mm256_maddubs_epi16` (VPMADDUBSW) que faz 16 produtos i8×i8 + soma em pares em 1 ciclo, seguido de `_mm256_madd_epi16` (VPMADDWD) — equivalente a 16 FMAs em 1 ciclo AVX2.

### 2.5 Flash Attention (opcional)

`llama.cpp/src/llama-context.cpp:174`
```c
cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
```

Flash Attention faz o attention causal em tiles, evitando alocar a matriz de scores completa. Para GQA com n_head=14 e n_head_kv=2, a matriz de scores é `14 × 2048` por token — relativamente pequena, mas Flash Attention elimina a alocação.

### 2.6 Batch processing (n_batch > 1)

llama.cpp suporta processar múltiplos tokens em batch no prefill. Nossa implementação já tem `forward_batch` mas não tem infraestrutura de scheduler. O benchmark compara decode (1 token), então isso não é o gargalo atual.

---

## 3. Gap Analysis: onde llama-rs perde para llama.cpp no Qwen

### Gap 1 (CRÍTICO): Quantização redundante de ativações

No `model.rs::forward`, por layer:
```rust
// attn_in é quantizada 3 VEZES separadamente:
let q = lw.attn_q.matmul_into(&attn_in, ...);  // quantiza attn_in internamente
let k = lw.attn_k.matmul_into(&attn_in, ...);  // quantiza attn_in de NOVO
let v = lw.attn_v.matmul_into(&attn_in, ...);  // quantiza attn_in de NOVO (3x!)
// ...
let gate = lw.ffn_gate.matmul_into(&ffn_in, ...);  // quantiza ffn_in
let up   = lw.ffn_up.matmul_into(&ffn_in, ...);    // quantiza ffn_in de NOVO (2x!)
```

**Custo:** 5 quantizações × 896 elementos × 24 layers = 107,520 elementos desperdiçados por forward pass.

### Gap 2 (CRÍTICO): Overhead de rayon por matmul

Cada `matmul_into` Q8_0 com n_out ≥ 64 (`PAR_MIN_N_OUT_Q8=64`) dispara rayon.
Para Qwen decode (n_tok=1):
- attn_q: n_out=896 → rayon ✓ (certo)
- attn_k: n_out=128 → 128 >= 64, rayon ✓ (overhead > ganho)
- attn_v: n_out=128 → 128 >= 64, rayon ✓ (overhead > ganho)
- attn_out: n_out=896 → rayon ✓
- ffn_gate: n_out=4864 → rayon ✓
- ffn_up: n_out=4864 → rayon ✓
- ffn_down: n_out=896 → rayon ✓

= 168 dispatches rayon por forward. A threshold de 64 está muito baixa para n_out=128.

### Gap 3 (ALTO): Kernel q8_0_q8_0_dot sem SIMD explícito

LLVM pode auto-vectorizar o loop de 32 i8s, mas:
- Sem `target_feature = "+avx2"` explícito no perfil de release
- Sem instruções VPMADDUBSW/VPMADDWD que llama.cpp usa

**Potencial melhoria:** 2-4x no kernel puro com AVX2 explícito.

### Gap 4 (MÉDIO): Q/K/V matmuls sequenciais

No decode (n_tok=1), Q, K e V são independentes. Poderiam ser computadas em paralelo:
```rust
// Atual: serial
let q = lw.attn_q.matmul_into(&attn_in, ...)?;
let k = lw.attn_k.matmul_into(&attn_in, ...)?;
let v = lw.attn_v.matmul_into(&attn_in, ...)?;

// Otimizado: paralelo
let (q, k, v) = rayon::join3(|| ..., || ..., || ...);
```

### Gap 5 (MÉDIO): gate/up FFN sequenciais

Mesma lógica: `ffn_gate` e `ffn_up` são independentes.

### Gap 6 (BAIXO): Thread count não otimizado para decode

rayon usa todos os núcleos por default. Para decode memory-bound, mais threads podem causar contention na memória.

---

## 4. Estimativa de impacto por otimização

| # | Otimização | Ganho esperado | Complexidade |
|---|-----------|----------------|--------------|
| 1 | Reuso de attn_in/ffn_in Q8 | 1.1-1.3x | Baixa |
| 2 | Raise PAR_MIN_N_OUT_Q8: 64→256 | 1.1-1.2x | Trivial |
| 3 | Q/K/V e gate/up em paralelo | 1.2-1.5x | Média |
| 4 | Kernel AVX2 explícito (VPMADDUBSW) | 1.5-3x | Alta |
| 5 | Flash Attention simplificado | 1.05-1.1x | Alta |

**Combinado (multiplicativo):** 1.3 × 1.2 × 1.4 × 2.0 ≈ **4.4x** → de 13.4 → ~59 tok/s teórico

---

## 5. O que stories260K tem que Qwen2 não tem (por isso llama-rs já é mais rápido)

| Fator | stories260K | Qwen2.5 |
|-------|-------------|---------|
| n_embd | 64 (tiny) | 896 (14x maior) |
| n_layer | 5 | 24 (5x mais) |
| n_ff | 172 | 4864 (28x maior) |
| Matmuls/token | ~35 | ~168 |
| Dados/token | ~200KB | ~35MB |

Para stories260K, todos os tensores cabem no L1/L2 cache → compute-bound, rayon funciona bem.
Para Qwen2.5, o modelo é **memory-bandwidth bound** → rayon overhead é proporcionalmente maior.

---

## 6. Conclusão: Caminho para superar o llama.cpp

Para ir de 13.4 → >37.8 tok/s (2.8x mínimo):

1. **Reuso Q8 de ativações** — implementar em `model.rs`, passar `x_q8` pré-computado
2. **Elevar PAR_MIN_N_OUT_Q8 para 256** — change 1 linha em `ops.rs`
3. **Paralelizar Q+K+V e gate+up** — usar `rayon::join` em `model.rs`
4. **Kernel AVX2 com VPMADDUBSW** — implementar em `ops.rs` com target_feature

A combinação 1+2+3 deve atingir ~20-25 tok/s.
A adição do item 4 deve atingir ~40-60 tok/s, superando o llama.cpp.
