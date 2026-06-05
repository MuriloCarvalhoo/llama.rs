# Plan: matmul Q8_0 direto — eliminar dequant-antes-de-matmul

**Complexidade:** Medium

## Resumo

O Qwen2 0.5B Q8_0 está em 11% da velocidade do llama.cpp porque os pesos (525 MB de i8) são expandidos para f32 (2.1 GB) antes de cada matmul — mesmo com o OnceCell a expansão fica residente em RAM. A correção é um kernel `matmul_q8_0` que lê os bytes Q8_0 diretamente no produto escalar, sem nunca expandir para f32. Ganho esperado: 2–4x no Qwen2 Q8_0; stories260K (F32) inalterado.

## Diagnóstico do gap

```
Q8_0 hoje:  bytes_i8 → OnceCell<Vec<f32>> (2.1 GB) → matmul f32   (lê 2.1 GB/forward)
Q8_0 novo:  bytes_i8 ─────────────────────────────→ matmul direto  (lê 525 MB/forward)
```

Cada bloco Q8_0 = 34 bytes: 2 bytes (f16 scale `d`) + 32 bytes (i8 quants).
dot(W_row_j, x_t) = Σ_b  d_b × Σ_{k=0}^{31} i8_w[k] as f32 × x_f32[k]

## Padrões a espelhar

| Categoria | Fonte | Padrão |
|---|---|---|
| Kernel matmul | `crates/llama-model/src/ops.rs:56-77` | `par_iter_mut` sobre `n_out`, fallback serial abaixo de `PAR_MIN_N_OUT` |
| Dequant Q8_0 | `crates/ggml-cpu/src/dequant.rs:55-65` | bloco de 34 bytes: `b[0..2]` = f16 scale, `b[2..34]` = i8 quants |
| Dispatch por tipo | `crates/llama-model/src/weights.rs:42-55` | `match self.ty { Q8_0 => fast_path, _ => dequant_path }` |
| Erro de modelo | `crates/llama-model/src/error.rs` | `ModelError::from(DequantError)` |
| Testes unitários | `crates/llama-model/src/ops.rs:160-212` | `mod tests`, AAA, valores numéricos verificados à mão |
| Testes de integração | `crates/llama-model/tests/oracle_forward.rs` | `#[ignore]`, compara com oráculo C++ |

## Arquivos a modificar

| Arquivo | Ação | Motivo |
|---|---|---|
| `crates/llama-model/src/ops.rs` | UPDATE | Adicionar `matmul_q8_0` e helper `q8_0_dot` |
| `crates/llama-model/src/weights.rs` | UPDATE | Adicionar `RawTensor::matmul_into` que despacha por tipo |
| `crates/llama-model/src/model.rs` | UPDATE | Substituir 8 pares `{dequant; matmul}` por `tensor.matmul_into` |

Nenhum arquivo novo. Nenhuma mudança em `ggml-cpu` (o kernel vai em `ops.rs` onde rayon já existe).

---

## Task 0: `matmul_q8_0` em `ops.rs` (TDD)

**Arquivo:** `crates/llama-model/src/ops.rs`

### Step 0.1 — Escrever o teste RED antes do kernel

Adicionar ao `mod tests` existente em `ops.rs`:

```rust
#[test]
fn matmul_q8_0_matches_manual() {
    // 1 bloco Q8_0: n_in=32, n_out=2, n_tok=1
    // row 0: d=1.0, qs=[1,2,0×30] → dot = 1×1 + 2×2 = 5 → out=5.0
    // row 1: d=2.0, qs=[1,0×31]   → dot = 1×1       = 1 → out=2.0
    fn f16_le(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }
    let mut w = Vec::with_capacity(68); // 2 rows × 34 bytes
    w.extend_from_slice(&f16_le(1.0));
    w.push(1u8); w.push(2u8);
    w.extend(std::iter::repeat(0u8).take(30));
    w.extend_from_slice(&f16_le(2.0));
    w.push(1u8);
    w.extend(std::iter::repeat(0u8).take(31));

    let x: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let out = matmul_q8_0(&w, &x, 32, 2, 1);

    assert!((out[0] - 5.0).abs() < 1e-4, "out[0]={}", out[0]);
    assert!((out[1] - 2.0).abs() < 1e-4, "out[1]={}", out[1]);
}

#[test]
fn matmul_q8_0_two_tokens() {
    // n_in=32, n_out=1, n_tok=2; d=1.0, qs=[1,0×31]
    fn f16_le(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }
    let mut w = Vec::with_capacity(34);
    w.extend_from_slice(&f16_le(1.0));
    w.push(1u8);
    w.extend(std::iter::repeat(0u8).take(31));

    let mut x = vec![0.0f32; 64];
    x[0] = 3.0;  // token 0, dim 0
    x[32] = 7.0; // token 1, dim 0

    let out = matmul_q8_0(&w, &x, 32, 1, 2);

    assert!((out[0] - 3.0).abs() < 1e-4, "t0={}", out[0]);
    assert!((out[1] - 7.0).abs() < 1e-4, "t1={}", out[1]);
}
```

Rodar: `cargo test -p llama-model ops::tests::matmul_q8_0 2>&1 | head -5`
Esperado: **FAIL** (função não existe).

### Step 0.2 — Implementar `matmul_q8_0`

Adicionar em `ops.rs`, imediatamente após a função `matmul` existente:

```rust
/// MUL_MAT direto em bytes Q8_0 — sem expandir para f32.
///
/// Layout W: `n_out` linhas × `(n_in/32)` blocos × 34 bytes/bloco.
/// Bloco: 2 bytes (f16 LE scale `d`) + 32 bytes (i8 quants).
/// `n_in` DEVE ser múltiplo de 32 (garantido pelo formato GGUF).
pub(crate) fn matmul_q8_0(
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_tok: usize,
) -> Vec<f32> {
    const Q: usize = 32; // elementos por bloco
    const B: usize = 34; // bytes por bloco
    debug_assert_eq!(n_in % Q, 0, "n_in deve ser múltiplo de 32");

    let n_blocks = n_in / Q;
    let row_bytes = n_blocks * B;
    let mut out = vec![0.0f32; n_tok * n_out];

    for t in 0..n_tok {
        let x_row = &x[t * n_in..(t + 1) * n_in];
        let o_row = &mut out[t * n_out..(t + 1) * n_out];

        if n_out >= PAR_MIN_N_OUT {
            o_row.par_iter_mut().enumerate().for_each(|(j, o)| {
                *o = q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
            });
        } else {
            for (j, o) in o_row.iter_mut().enumerate() {
                *o = q8_0_dot(&w[j * row_bytes..(j + 1) * row_bytes], x_row, n_blocks);
            }
        }
    }
    out
}

#[inline]
fn q8_0_dot(w_row: &[u8], x_row: &[f32], n_blocks: usize) -> f32 {
    const Q: usize = 32;
    const B: usize = 34;
    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let blk = &w_row[b * B..(b + 1) * B];
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let qs = &blk[2..34];
        let x_blk = &x_row[b * Q..(b + 1) * Q];
        let dot: f32 = qs
            .iter()
            .zip(x_blk.iter())
            .map(|(&q, &xv)| q.cast_signed() as f32 * xv)
            .sum();
        acc += d * dot;
    }
    acc
}
```

Se `half` não estiver em `crates/llama-model/Cargo.toml`, adicionar:
```toml
half.workspace = true
```
(já está no workspace via `ggml-cpu`; verificar se já consta no workspace `[dependencies]`).

### Step 0.3 — Rodar e ver passar

```bash
cargo test -p llama-model ops::tests::matmul_q8_0 -- --nocapture
```

Esperado: **PASS** (2 testes).

### Step 0.4 — Commit

```bash
git add crates/llama-model/src/ops.rs
git commit -m "feat(ops): matmul_q8_0 — produto escalar direto em bytes Q8_0 (TDD)"
```

---

## Task 1: `RawTensor::matmul_into` — dispatch por tipo

**Arquivo:** `crates/llama-model/src/weights.rs`

### Step 1.1 — Adicionar imports em `weights.rs`

```rust
use crate::ops::{matmul, matmul_q8_0};
```

### Step 1.2 — Implementar `matmul_into`

Adicionar ao `impl RawTensor`, após `dequant_to_f32`:

```rust
/// Matmul otimizado por tipo de quantização.
/// - Q8_0: produto escalar direto em i8 (sem alocar buffer f32)
/// - outros: dequant → f32 → matmul
pub(crate) fn matmul_into(
    &self,
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_tok: usize,
) -> Result<Vec<f32>, ModelError> {
    if self.ty == gguf::GgmlType::Q8_0 {
        Ok(matmul_q8_0(&self.bytes, x, n_in, n_out, n_tok))
    } else {
        Ok(matmul(self.dequant_to_f32()?, x, n_in, n_out, n_tok))
    }
}
```

### Step 1.3 — Rodar testes de weights

```bash
cargo test -p llama-model weights::tests -- --nocapture
```

Esperado: PASS (todos os testes existentes — `matmul_into` é apenas um novo método, não quebra nada).

### Step 1.4 — Commit

```bash
git add crates/llama-model/src/weights.rs
git commit -m "feat(weights): RawTensor::matmul_into dispatch Q8_0 direto / fallback dequant"
```

---

## Task 2: Substituir os 8 pares `{dequant; matmul}` em `model.rs`

**Arquivo:** `crates/llama-model/src/model.rs`

### Step 2.1 — Substituição mecânica (seção de atenção por camada)

**Antes:**
```rust
let attn_norm = lw.attn_norm.dequant_to_f32()?;
let attn_q_w  = lw.attn_q.dequant_to_f32()?;
let attn_k_w  = lw.attn_k.dequant_to_f32()?;
let attn_v_w  = lw.attn_v.dequant_to_f32()?;
let attn_out_w = lw.attn_output.dequant_to_f32()?;
// ...
let mut q    = matmul(attn_q_w,  &attn_in, c.n_embd, c.n_embd, n_tok);
let mut k    = matmul(attn_k_w,  &attn_in, c.n_embd, kv_dim,   n_tok);
let mut v    = matmul(attn_v_w,  &attn_in, c.n_embd, kv_dim,   n_tok);
// ...
let attn_out = matmul(attn_out_w, &attn,   c.n_embd, c.n_embd, n_tok);
```

**Depois:**
```rust
let attn_norm = lw.attn_norm.dequant_to_f32()?;
// (linhas de dequant dos pesos de atenção removidas)
let mut q    = lw.attn_q.matmul_into(&attn_in, c.n_embd, c.n_embd, n_tok)?;
let mut k    = lw.attn_k.matmul_into(&attn_in, c.n_embd, kv_dim,   n_tok)?;
let mut v    = lw.attn_v.matmul_into(&attn_in, c.n_embd, kv_dim,   n_tok)?;
// ...
let attn_out = lw.attn_output.matmul_into(&attn, c.n_embd, c.n_embd, n_tok)?;
```

**Antes (seção FFN):**
```rust
let ffn_norm  = lw.ffn_norm.dequant_to_f32()?;
let ffn_gate_w = lw.ffn_gate.dequant_to_f32()?;
let ffn_up_w   = lw.ffn_up.dequant_to_f32()?;
let ffn_down_w = lw.ffn_down.dequant_to_f32()?;
// ...
let gate    = matmul(ffn_gate_w, &ffn_in, c.n_embd, c.n_ff,   n_tok);
let up      = matmul(ffn_up_w,   &ffn_in, c.n_embd, c.n_ff,   n_tok);
let ffn_out = matmul(ffn_down_w, &act,    c.n_ff,   c.n_embd, n_tok);
```

**Depois:**
```rust
let ffn_norm = lw.ffn_norm.dequant_to_f32()?;
let gate    = lw.ffn_gate.matmul_into(&ffn_in, c.n_embd, c.n_ff,   n_tok)?;
let up      = lw.ffn_up.matmul_into(  &ffn_in, c.n_embd, c.n_ff,   n_tok)?;
let ffn_out = lw.ffn_down.matmul_into(&act,    c.n_ff,   c.n_embd, n_tok)?;
```

**Antes (logits finais):**
```rust
let output_norm = self.weights.output_norm.dequant_to_f32()?;
let output_w    = self.weights.output.dequant_to_f32()?;
// ...
let logits = matmul(output_w, last, c.n_embd, c.vocab, 1);
```

**Depois:**
```rust
let output_norm = self.weights.output_norm.dequant_to_f32()?;
// ...
let logits = self.weights.output.matmul_into(last, c.n_embd, c.vocab, 1)?;
```

Remover o `use crate::ops::matmul` no topo de `model.rs` se não for mais referenciado diretamente.

### Step 2.2 — Compilar

```bash
cargo check -p llama-model
```

Esperado: zero erros.

### Step 2.3 — Gate diferencial (crítico)

```bash
cargo test -p llama-model --test oracle_forward -- --nocapture
```

Esperado: PASS — sequência greedy bit-a-bit idêntica ao oráculo C++.

### Step 2.4 — Todos os testes do crate

```bash
cargo test -p llama-model
```

Esperado: PASS.

### Step 2.5 — Commit

```bash
git add crates/llama-model/src/model.rs
git commit -m "perf(model): substituir dequant+matmul por matmul_into (Q8_0 sem expansão f32)"
```

---

## Task 3: Gate completo + benchmark

### Step 3.1 — fmt + clippy

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Issues comuns:
- `cast_signed()` — mesmo padrão de `dequant.rs`, sem lint novo.
- `unused_import` de `matmul` em `model.rs` se removido — ajustar `use`.
- `dead_code` em `q8_0_dot` — não ocorre; é chamada por `matmul_q8_0`.

### Step 3.2 — Workspace completo

```bash
cargo test --workspace
```

Esperado: PASS.

### Step 3.3 — Benchmark

```bash
bash scripts/benchmark.sh
```

Esperado:
- `stories260K`: sem regressão (F32, path inalterado, ~900–1000 tok/s)
- `qwen2.5-0.5b-q8_0`: melhora vs 4.1 tok/s; alvo conservador ≥ 8 tok/s (2x), potencial 12–16 tok/s se bandwidth-bound

### Step 3.4 — Commit com resultado

```bash
git commit --allow-empty -m "perf: benchmark matmul_q8_0 — <stories_tps> tok/s stories, <qwen_tps> tok/s qwen2"
```

---

## Riscos

| Risco | Likelihood | Mitigação |
|---|---|---|
| `n_in % 32 != 0` em algum tensor | Baixo | GGUF garante alinhamento; `debug_assert` captura em test |
| `half` não disponível em `llama-model` | Médio | Adicionar `half.workspace = true` se necessário |
| Oracle diverge após mudança | Baixo | Step 2.3 verifica antes do commit; matemática é equivalente |
| Speedup menor que esperado (qwen) | Médio | Se L1/L2 miss dominar, ganho pode ser 1.5–2x; ainda útil |

## Acceptance

- [ ] 2 testes unitários de `matmul_q8_0` PASS (valores verificados à mão)
- [ ] `oracle_forward` PASS após Task 2
- [ ] `cargo test --workspace` PASS
- [ ] `cargo clippy -D warnings` PASS
- [ ] Benchmark: qwen2 Q8_0 tok/s > 8 (2x do baseline 4.1)
- [ ] stories260K sem regressão (> 900 tok/s)
