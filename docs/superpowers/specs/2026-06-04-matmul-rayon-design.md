# Design: matmul paralelo com rayon + target-cpu=native

**Data:** 2026-06-04  
**Escopo:** `crates/llama-model`, `.cargo/config.toml`  
**Objetivo:** reduzir latência de decode no Qwen2 0.5B paralelizando `matmul` com rayon e ativando auto-vetorização AVX2 via flag de compilação.

---

## Contexto

O benchmark mostrou que llama-rs é 23× mais lento que llama.cpp para Qwen2 0.5B (1.6 vs 37.8 tok/s). O gargalo é `matmul` em `ops.rs`: um triplo loop sequencial sem paralelismo de threads e sem SIMD explícito.

O workspace tem `#![forbid(unsafe_code)]` nos crates relevantes, então `std::arch` está fora de escopo. A solução adota rayon (paralelismo safe) + LLVM auto-vetorização via `-C target-cpu=native`.

---

## Componentes alterados

### 1. `.cargo/config.toml` (novo arquivo)

Ativa `target-cpu=native` apenas para o profile `release`. Builds de debug e testes não são afetados.

```toml
[profile.release]
rustflags = ["-C", "target-cpu=native"]
```

Efeito: todos os loops existentes (rmsnorm, swiglu, softmax, rope, matmul inner) ganham vetorização SIMD de graça, sem mudança de código.

### 2. `crates/llama-model/Cargo.toml`

Adicionar `rayon` ao bloco `[dependencies]`.

### 3. `crates/llama-model/src/ops.rs` — `matmul`

Reescrever o loop de `n_out` para `par_iter_mut`, mantendo assinatura pública inalterada.

**Antes:**
```rust
for t in 0..n_tok {
    let xrow = &x[t * n_in..t * n_in + n_in];
    let orow = &mut out[t * n_out..t * n_out + n_out];
    for (j, o) in orow.iter_mut().enumerate() {
        let wrow = &w[j * n_in..j * n_in + n_in];
        *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
    }
}
```

**Depois:**
```rust
for t in 0..n_tok {
    let xrow = &x[t * n_in..(t + 1) * n_in];
    let orow = &mut out[t * n_out..(t + 1) * n_out];
    orow.par_iter_mut().enumerate().for_each(|(j, o)| {
        let wrow = &w[j * n_in..(j + 1) * n_in];
        *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
    });
}
```

Durante decode (n_tok=1), os `n_out` dot products (896–4864 para Qwen2) são distribuídos entre os cores. O loop interno é auto-vetorizado pelo compilador com AVX2.

---

## Assinaturas

Nenhuma assinatura pública muda.

---

## Testes

Sem testes novos necessários — suite existente cobre:

- `matmul_2x2_identity_and_general`, `matmul_two_tokens` — corretude numérica
- `oracle_forward` — comparação token-a-token contra llama.cpp

Performance: `scripts/benchmark.sh` antes e depois.

---

## Riscos

| Risco | Prob | Mitigação |
|---|---|---|
| Overhead rayon supera ganho para n_out pequeno | Baixa | n_out mínimo relevante é 896; overhead rayon ~1µs é desprezível |
| Binário não portável com `target-cpu=native` | Intencional | Só afeta `release`; CI pode usar `target-cpu=generic` se necessário |
| Regressão numérica por reordenação de f32 | Muito baixa | Paralelismo é sobre `j` (dot products independentes), não dentro de um dot |
