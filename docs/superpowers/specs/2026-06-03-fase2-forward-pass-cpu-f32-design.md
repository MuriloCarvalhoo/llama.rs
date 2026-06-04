# Fase 2 — Forward pass CPU f32 (arquitetura Llama / stories260K)

Data: 2026-06-03
Status: aprovado (design)

## Objetivo

Implementar a inferência (forward pass) da arquitetura Llama em CPU, em f32 puro,
validada por teste diferencial contra o oráculo llama.cpp já compilado. Escopo
restrito ao modelo `models/stories260K.gguf` (todos os 48 tensores são F32 —
nenhuma dequantização necessária nesta fase).

## Decisões (confirmadas com o usuário)

1. **Critério de aceitação = greedy token-match.** O oráculo usa `FLASH_ATTN_EXT`
   com KV cache em f16; uma atenção CPU pura em f32 não é bit-exact a nível de
   tensor (arredondamento f16 + ordem de acumulação do softmax online diferem).
   O gate duro é a igualdade da sequência greedy decodificada vs `refs/greedy.txt`.
   Checagens aproximadas (por tolerância) dos tensores pré-atenção servem de
   sanidade por-op.
2. **Escopo = só stories260K (f32).** Dequant q8_0, qwen2.5, RoPE NeoX, batching,
   perf/SIMD ficam para fases futuras.
3. **Motor de ops = f32 hand-rolled, sem abstração de tensor genérica** (`Vec<f32>`
   + shapes explícitos, funções livres por op). KISS / YAGNI; casa com os lints
   estritos do workspace (`unsafe_code = deny`, `unwrap/expect/panic = deny`,
   casts lossy negados).

## Fatos do modelo (stories260K)

| Param | Valor | Origem |
|-------|-------|--------|
| `n_embd` | 64 | `llama.embedding_length` |
| `n_layer` | 5 | `llama.block_count` |
| `n_head` | 8 | `llama.attention.head_count` |
| `n_head_kv` | 4 | `llama.attention.head_count_kv` |
| `head_dim` | 8 | `n_embd / n_head` |
| `n_ff` | 172 | `llama.feed_forward_length` |
| `rope_dim` | 8 | `llama.rope.dimension_count` (= head_dim, rotação total) |
| `rms_eps` | 1e-5 | `llama.attention.layer_norm_rms_epsilon` |
| `freq_base` | 10000 | default (chave ausente no GGUF) |
| `vocab` | 512 | `tokenizer.ggml.tokens` len |
| `ctx` | 2048 | `llama.context_length` |
| `bos/eos` | 1 / 2 | metadata tokenizer |

Tensores (ggml dims = `{ne0, ne1, ...}`, ne0 = dim contígua):

- `token_embd.weight {64, 512}` — linha do token `t` = 64 f32 contíguos em `t*64`.
- `blk.L.attn_norm.weight {64}`, `blk.L.ffn_norm.weight {64}`, `output_norm.weight {64}`.
- `blk.L.attn_q.weight {64, 64}` (in=64, out=64).
- `blk.L.attn_k.weight {64, 32}`, `blk.L.attn_v.weight {64, 32}` (in=64, out=32 = head_dim·n_head_kv).
- `blk.L.attn_output.weight {64, 64}`.
- `blk.L.ffn_gate.weight {64, 172}`, `blk.L.ffn_up.weight {64, 172}` (in=64, out=172).
- `blk.L.ffn_down.weight {172, 64}` (in=172, out=64).
- `output.weight {64, 512}` (logits; embeddings NÃO atadas — output existe).

Convenção `MUL_MAT(W{ne00=in, ne01=out}, x{in, n}) -> {out, n}`:
`out[j] = Σ_i W[i + j*in] * x[i]` — `W` row-major por linha de saída.

## Grafo do forward (confirmado pelo dump do oráculo)

```
embd = GET_ROWS(token_embd, tokens)            # {n_embd, n_tok}
para cada camada L em 0..n_layer:
    cur   = RMSNORM(residual, eps) * attn_norm_w
    Q     = matmul(attn_q_w, cur)              # {n_embd, n_tok} -> reshape {head_dim, n_head, n_tok}
    K     = matmul(attn_k_w, cur)              # -> {head_dim, n_head_kv, n_tok}
    V     = matmul(attn_v_w, cur)
    Q,K   = rope_norm(Q,K, posições)           # rope_dim=head_dim
    (append K,V no KvCache da camada L)
    attn  = causal_gqa_attention(Q, Kcache, Vcache, scale=1/sqrt(head_dim))
    ao    = matmul(attn_output_w, attn)
    res2  = ao + residual                      # ffn_inp
    cur   = RMSNORM(res2, eps) * ffn_norm_w
    g     = matmul(ffn_gate_w, cur); u = matmul(ffn_up_w, cur)
    cur   = swiglu(g, u) = silu(g) * u
    fo    = matmul(ffn_down_w, cur)
    residual = fo + res2                        # l_out
cur    = RMSNORM(residual, eps) * output_norm_w
logits = matmul(output_w, cur[última posição]) # {vocab}
```

RoPE NORM (arch `llama`): para pares `(2i, 2i+1)`, `θ_i = pos · freq_base^(-2i/rope_dim)`,
`x'[2i] = x[2i]·cosθ − x[2i+1]·sinθ`, `x'[2i+1] = x[2i]·sinθ + x[2i+1]·cosθ`.

GQA: query head `h` usa kv head `h / (n_head/n_head_kv)` = `h/2`.

## Estrutura do crate `crates/llama-model`

- `config.rs` — `LlamaConfig`, lido do `GgufFile`, validado na fronteira.
- `weights.rs` — `Weights`: views `&[f32]` por tensor (reinterpretação LE segura, sem `unsafe`).
- `ops.rs` — `embedding_lookup`, `rmsnorm`, `mul`, `matmul`, `rope_norm`, `silu`, `swiglu`, `softmax`.
- `attention.rs` — `KvCache` + atenção causal GQA f32.
- `model.rs` — `Model` (Config+Weights), `forward(tokens, &mut KvCache) -> Vec<f32>` (logits do último token).
- `generate.rs` — `generate_greedy(prompt, n_tokens) -> String`.
- `error.rs` — `ModelError` (thiserror).

Dependências: `gguf` (path), `llama-tokenizer` (path), `thiserror`. Dev: nenhum extra
além do que já existe; testes leem `models/` e `refs/` por caminho relativo (pulam se ausentes).

## Estratégia de teste

1. **Unit por op** — valores pequenos conhecidos (ex.: `rmsnorm` de vetor simples,
   `matmul` 2×2, `rope_norm` num pos conhecido, `swiglu`).
2. **Sanidade por-op com tolerância** — parsear os `sum = ...` do `refs/tensors.txt`
   para o prompt "Once upon a time" (5 tokens) e comparar `embd`, `norm-0`,
   `attn_norm-0`, `Qcur-0` (pós-rope), etc., até a fronteira da atenção
   (erro relativo pequeno). Da atenção em diante a divergência flash-f16 é esperada.
3. **Gate duro (diferencial)** — `generate_greedy("Once upon a time", 32)` decodificado
   == conteúdo de `refs/greedy.txt`.
4. Gate de qualidade do projeto: `scripts/gate.sh` (fmt + clippy `-D warnings`).

## Fora de escopo

Dequant (q8_0 e demais), qwen2.5-0.5b, RoPE NeoX, atenção flash, KV cache f16,
batching multi-sequência, paralelismo/SIMD, performance.
