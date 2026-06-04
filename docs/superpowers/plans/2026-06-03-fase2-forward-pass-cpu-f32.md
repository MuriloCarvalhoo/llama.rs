# Fase 2 — Forward pass CPU f32 (Llama/stories260K) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar inferência forward (prefill + decode greedy) da arquitetura Llama em CPU/f32 puro, validada por teste diferencial contra o oráculo llama.cpp, no modelo `models/stories260K.gguf`.

**Architecture:** Novo crate `crates/llama-model`. Ops f32 hand-rolled (sem abstração de tensor genérica): ativações em `Vec<f32>` token-major (stride `n_embd`, idêntico ao layout `{n_embd, n_tok}` do ggml). `LlamaConfig` lê escalares do GGUF; `Weights` materializa buffers `Vec<f32>` por tensor (reinterpretação LE segura, sem `unsafe`); `Model::forward` segue o grafo do oráculo com `KvCache` f32; `generate_greedy` faz argmax loop.

**Tech Stack:** Rust edition 2024, deps path `gguf` + `llama-tokenizer`, `thiserror`. Lints do workspace: `unsafe_code=deny`, `unwrap/expect/panic=deny`, casts lossy negados. Gate: `scripts/gate.sh` (fmt + clippy `-D warnings` + test + cobertura ≥80%).

**Convenções fixas (NÃO reinterpretar):**
- Ativações token-major: elemento `(dim d, token t)` em `x[t*dim + d]`.
- `matmul(W, x)`: `W{in,out}` armazenado como `out` linhas de comprimento `in` → `W[j*in + i]`. Saída `out[t*out + j] = Σ_i W[j*in+i] * x[t*in+i]`.
- Dentro de um token, head `h` ocupa `[h*head_dim .. h*head_dim+head_dim]`.
- RoPE NORM (arch `llama`): pares `(2i, 2i+1)`, `θ_i = pos * freq_base^(-2i/rope_dim)`.
- GQA: query head `h` → kv head `h / (n_head/n_head_kv)`.
- Fatos do modelo: `n_embd=64, n_layer=5, n_head=8, n_head_kv=4, head_dim=8, n_ff=172, rope_dim=8, rms_eps=1e-5, freq_base=10000, vocab=512, bos=1, eos=2`.
- Prompt de validação "Once upon a time" → ids `[1,403,407,261,378]`; `embd sum=-3.354056`; `Qcur-0` pós-rope `sum=148.969818`; saída greedy 32 tokens == `refs/greedy.txt`.

---

## File Structure

- Create `crates/llama-model/Cargo.toml` — manifesto do crate.
- Create `crates/llama-model/src/lib.rs` — módulos + re-exports.
- Create `crates/llama-model/src/error.rs` — `ModelError` (thiserror).
- Create `crates/llama-model/src/config.rs` — `LlamaConfig::from_gguf`.
- Create `crates/llama-model/src/weights.rs` — `Weights::from_gguf` (buffers f32).
- Create `crates/llama-model/src/ops.rs` — ops f32 puras + unit tests.
- Create `crates/llama-model/src/attention.rs` — `KvCache` + `attention` + tests.
- Create `crates/llama-model/src/model.rs` — `Model::load` + `forward`.
- Create `crates/llama-model/src/generate.rs` — `generate_greedy`.
- Create `crates/llama-model/tests/oracle_forward.rs` — gate diferencial greedy.
- Modify `Cargo.toml` (workspace) — adicionar membro + dep path.

---

## Task 0: Scaffold do crate + wiring no workspace

**Files:**
- Modify: `Cargo.toml` (raiz, workspace)
- Create: `crates/llama-model/Cargo.toml`
- Create: `crates/llama-model/src/lib.rs`
- Create: `crates/llama-model/src/error.rs`

- [ ] **Step 1: Adicionar o crate ao workspace**

Em `Cargo.toml` (raiz), atualizar `members` e expor a dep path. Substituir a linha `members = [...]`:

```toml
members = ["oracle", "crates/gguf", "crates/llama-tokenizer", "crates/llama-model"]
```

E em `[workspace.dependencies]`, após a linha `gguf = { path = "crates/gguf" }`, adicionar:

```toml
llama-tokenizer = { path = "crates/llama-tokenizer" }
```

- [ ] **Step 2: Criar o manifesto do crate**

`crates/llama-model/Cargo.toml`:

```toml
[package]
name = "llama-model"
version = "0.1.0"
edition.workspace = true

[dependencies]
thiserror.workspace = true
gguf.workspace = true
llama-tokenizer.workspace = true

[lints]
workspace = true
```

- [ ] **Step 3: Criar `error.rs`**

`crates/llama-model/src/error.rs`:

```rust
//! Erros do carregamento e da inferência do modelo Llama.

use gguf::GgufError;
use llama_tokenizer::TokenizerError;

/// Falhas ao carregar config/pesos ou ao executar o forward.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("gguf: {0}")]
    Gguf(#[from] GgufError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("tensor ausente: {0}")]
    MissingTensor(String),
    #[error("bytes do tensor {0} não são múltiplos de 4 (f32)")]
    NotF32(String),
    #[error("config inconsistente: {0}")]
    Config(String),
    #[error("overflow de conversão numérica")]
    Overflow,
}
```

- [ ] **Step 4: Criar `lib.rs` + stubs de módulo**

`crates/llama-model/src/lib.rs`:

```rust
#![forbid(unsafe_code)]
//! Inferência forward (CPU, f32) da arquitetura Llama. Escopo: stories260K.

mod attention;
mod config;
mod error;
mod generate;
mod model;
mod ops;
mod weights;

pub use config::LlamaConfig;
pub use error::ModelError;
pub use model::Model;
```

Criar arquivos placeholder para os módulos ainda não preenchidos, cada um só com um comentário (serão substituídos nas tasks seguintes):
- `crates/llama-model/src/config.rs` → `//! stub`
- `crates/llama-model/src/weights.rs` → `//! stub`
- `crates/llama-model/src/ops.rs` → `//! stub`
- `crates/llama-model/src/attention.rs` → `//! stub`
- `crates/llama-model/src/model.rs` → `//! stub`
- `crates/llama-model/src/generate.rs` → `//! stub`

(Como `lib.rs` referencia `config::LlamaConfig`, `model::Model` e `error::ModelError`, estes três precisam dos tipos já na Task 0 OU comentar temporariamente os `pub use config`/`pub use model` até as Tasks 1 e 8. Para manter o build verde a cada task: na Task 0 deixar em `lib.rs` apenas `pub use error::ModelError;` e adicionar `pub use config::LlamaConfig;` na Task 1 e `pub use model::Model;` na Task 8.)

`lib.rs` da Task 0 (versão que compila com stubs):

```rust
#![forbid(unsafe_code)]
//! Inferência forward (CPU, f32) da arquitetura Llama. Escopo: stories260K.

mod attention;
mod config;
mod error;
mod generate;
mod model;
mod ops;
mod weights;

pub use error::ModelError;
```

- [ ] **Step 5: Verificar que compila (esqueleto)**

Run: `cargo build -p llama-model`
Expected: PASS (warnings de `dead_code` aceitáveis; o gate só roda na Task 10).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/llama-model
git commit -m "chore(llama-model): scaffold do crate de inferência (Fase 2 Task 0)"
```

---

## Task 1: `LlamaConfig::from_gguf`

**Files:**
- Modify: `crates/llama-model/src/config.rs`
- Modify: `crates/llama-model/src/lib.rs` (adicionar `pub use config::LlamaConfig;`)

- [ ] **Step 1: Escrever impl + teste**

`crates/llama-model/src/config.rs` (substituir o stub):

```rust
//! Hiperparâmetros da arquitetura Llama, lidos do GGUF.

use gguf::{GgufFile, MetadataValue};

use crate::error::ModelError;

/// Hiperparâmetros do modelo Llama necessários ao forward f32.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rope_dim: usize,
    pub rms_eps: f32,
    pub freq_base: f32,
    pub vocab: usize,
    pub ctx: usize,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl LlamaConfig {
    /// Lê e valida os escalares do GGUF (arquitetura `llama`).
    pub fn from_gguf(f: &GgufFile) -> Result<Self, ModelError> {
        let u = |k: &str| -> Result<usize, ModelError> {
            let v = f.get(k)?.as_u32(k)?;
            usize::try_from(v).map_err(|_| ModelError::Overflow)
        };
        let n_embd = u("llama.embedding_length")?;
        let n_head = u("llama.attention.head_count")?;
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(ModelError::Config(
                "n_head inválido ou não divide n_embd".into(),
            ));
        }
        let head_dim = n_embd / n_head;
        let vocab = f
            .get("tokenizer.ggml.tokens")?
            .array_len()
            .ok_or_else(|| ModelError::Config("tokens não é array".into()))?;
        // freq_base é opcional no GGUF; default 10000.
        let freq_base = match f.metadata.get("llama.rope.freq_base") {
            Some(MetadataValue::F32(v)) => *v,
            _ => 10000.0,
        };
        Ok(Self {
            n_embd,
            n_layer: u("llama.block_count")?,
            n_head,
            n_head_kv: u("llama.attention.head_count_kv")?,
            head_dim,
            n_ff: u("llama.feed_forward_length")?,
            rope_dim: u("llama.rope.dimension_count")?,
            rms_eps: f
                .get("llama.attention.layer_norm_rms_epsilon")?
                .as_f32("rms")?,
            freq_base,
            vocab,
            ctx: u("llama.context_length")?,
            bos_id: f.get("tokenizer.ggml.bos_token_id")?.as_u32("bos")?,
            eos_id: f.get("tokenizer.ggml.eos_token_id")?.as_u32("eos")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load() -> Option<GgufFile> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        GgufFile::parse(&bytes).ok()
    }

    #[test]
    fn reads_stories260k_config() {
        let Some(f) = load() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let c = LlamaConfig::from_gguf(&f).unwrap();
        assert_eq!(c.n_embd, 64);
        assert_eq!(c.n_layer, 5);
        assert_eq!(c.n_head, 8);
        assert_eq!(c.n_head_kv, 4);
        assert_eq!(c.head_dim, 8);
        assert_eq!(c.n_ff, 172);
        assert_eq!(c.rope_dim, 8);
        assert_eq!(c.vocab, 512);
        assert_eq!(c.bos_id, 1);
        assert_eq!(c.eos_id, 2);
        assert!((c.rms_eps - 1e-5).abs() < 1e-9);
        assert!((c.freq_base - 10000.0).abs() < 1e-3);
    }
}
```

Adicionar em `lib.rs`: `pub use config::LlamaConfig;`

- [ ] **Step 2: Rodar o teste**

Run: `cargo test -p llama-model config::tests`
Expected: PASS (modelo presente). Se o build falhar por `array_len`/`metadata` (campo público), conferir a API real em `crates/gguf/src/{file,types}.rs` — `GgufFile.metadata` é `pub` e `MetadataValue::array_len() -> Option<usize>` existe.

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/config.rs crates/llama-model/src/lib.rs
git commit -m "feat(llama-model): LlamaConfig::from_gguf (Fase 2 Task 1)"
```

---

## Task 2: `Weights::from_gguf` (buffers f32)

**Files:**
- Modify: `crates/llama-model/src/weights.rs`

- [ ] **Step 1: Escrever impl + teste**

`crates/llama-model/src/weights.rs` (substituir o stub):

```rust
//! Materialização dos pesos f32 do GGUF em buffers próprios (sem `unsafe`).

use gguf::{GgmlType, GgufFile, TensorInfo};

use crate::config::LlamaConfig;
use crate::error::ModelError;

/// Pesos de uma camada transformer.
pub(crate) struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

/// Todos os pesos do modelo, em f32.
pub(crate) struct Weights {
    pub token_embd: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: Vec<f32>,
}

fn tensor_f32(f: &GgufFile, bytes: &[u8], name: &str) -> Result<Vec<f32>, ModelError> {
    let info: &TensorInfo = f
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(ModelError::NotF32(name.to_owned()));
    }
    let raw = f.tensor_data(bytes, info)?;
    raw.chunks_exact(4)
        .map(|c| {
            <[u8; 4]>::try_from(c)
                .map(f32::from_le_bytes)
                .map_err(|_| ModelError::NotF32(name.to_owned()))
        })
        .collect()
}

impl Weights {
    /// Lê todos os tensores f32 necessários. `bytes` é o arquivo GGUF inteiro.
    pub fn from_gguf(f: &GgufFile, bytes: &[u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |suffix: &str| format!("blk.{l}.{suffix}");
            layers.push(LayerWeights {
                attn_norm: tensor_f32(f, bytes, &p("attn_norm.weight"))?,
                attn_q: tensor_f32(f, bytes, &p("attn_q.weight"))?,
                attn_k: tensor_f32(f, bytes, &p("attn_k.weight"))?,
                attn_v: tensor_f32(f, bytes, &p("attn_v.weight"))?,
                attn_output: tensor_f32(f, bytes, &p("attn_output.weight"))?,
                ffn_norm: tensor_f32(f, bytes, &p("ffn_norm.weight"))?,
                ffn_gate: tensor_f32(f, bytes, &p("ffn_gate.weight"))?,
                ffn_up: tensor_f32(f, bytes, &p("ffn_up.weight"))?,
                ffn_down: tensor_f32(f, bytes, &p("ffn_down.weight"))?,
            });
        }
        Ok(Self {
            token_embd: tensor_f32(f, bytes, "token_embd.weight")?,
            layers,
            output_norm: tensor_f32(f, bytes, "output_norm.weight")?,
            output: tensor_f32(f, bytes, "output.weight")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn loads_all_weights_with_expected_sizes() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let f = GgufFile::parse(&bytes).unwrap();
        let cfg = LlamaConfig::from_gguf(&f).unwrap();
        let w = Weights::from_gguf(&f, &bytes, &cfg).unwrap();
        assert_eq!(w.token_embd.len(), cfg.vocab * cfg.n_embd); // 512*64
        assert_eq!(w.output.len(), cfg.vocab * cfg.n_embd);
        assert_eq!(w.output_norm.len(), cfg.n_embd);
        assert_eq!(w.layers.len(), cfg.n_layer);
        let l0 = &w.layers[0];
        assert_eq!(l0.attn_q.len(), cfg.n_embd * cfg.n_embd); // 64*64
        assert_eq!(l0.attn_k.len(), cfg.n_embd * cfg.n_head_kv * cfg.head_dim); // 64*32
        assert_eq!(l0.ffn_gate.len(), cfg.n_embd * cfg.n_ff); // 64*172
        assert_eq!(l0.ffn_down.len(), cfg.n_ff * cfg.n_embd); // 172*64
    }
}
```

(O `#[allow(clippy::indexing_slicing)]` não é necessário em `weights.rs` fora dos testes; os testes já têm `unwrap` liberado via `clippy.toml`. Se clippy reclamar de indexing nos testes, adicionar `#![allow(clippy::indexing_slicing)]` no `mod tests`.)

- [ ] **Step 2: Rodar o teste**

Run: `cargo test -p llama-model weights::tests`
Expected: PASS. Tamanhos batem com os shapes da spec.

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/weights.rs
git commit -m "feat(llama-model): Weights::from_gguf materializa pesos f32 (Fase 2 Task 2)"
```

---

## Task 3: `ops.rs` — embedding, rmsnorm, mul_rows

**Files:**
- Modify: `crates/llama-model/src/ops.rs`

- [ ] **Step 1: Escrever impl + testes (parte 1)**

`crates/llama-model/src/ops.rs` (substituir o stub). O `#![allow(clippy::indexing_slicing)]` é deliberado: kernels numéricos com índices validados pelos shapes da config.

```rust
//! Kernels f32 puros do forward Llama. Layout token-major: `x[t*dim + d]`.
#![allow(clippy::indexing_slicing)]

use crate::error::ModelError;

/// GET_ROWS: para cada token, copia a linha de `embd` ({vocab, n_embd}).
/// Saída token-major [n_tok * n_embd].
pub(crate) fn embedding_lookup(
    embd: &[f32],
    tokens: &[u32],
    n_embd: usize,
) -> Result<Vec<f32>, ModelError> {
    let mut out = Vec::with_capacity(tokens.len() * n_embd);
    for &tok in tokens {
        let t = usize::try_from(tok).map_err(|_| ModelError::Overflow)?;
        let start = t * n_embd;
        let row = embd
            .get(start..start + n_embd)
            .ok_or_else(|| ModelError::Config(format!("token {t} fora do vocab")))?;
        out.extend_from_slice(row);
    }
    Ok(out)
}

/// RMSNorm por linha (sem peso): `x / sqrt(mean(x^2) + eps)`.
pub(crate) fn rmsnorm(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ss: f32 = row_in.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
        for (o, &i) in row_out.iter_mut().zip(row_in.iter()) {
            *o = i * scale;
        }
    }
    out
}

/// Multiplicação elementwise por peso broadcast por dimensão (MUL com {dim}).
pub(crate) fn mul_rows(x: &[f32], weight: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        for (idx, (o, &i)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            *o = i * weight[idx];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_lookup_copies_rows() {
        // vocab=3, n_embd=2
        let embd = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = embedding_lookup(&embd, &[2, 0], 2).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn embedding_lookup_rejects_oob_token() {
        let embd = vec![0.0, 1.0];
        assert!(embedding_lookup(&embd, &[5], 2).is_err());
    }

    #[test]
    fn rmsnorm_unit_vector() {
        // x = [3,4], dim=2, eps=0 → mean(x^2)=12.5 → scale=1/sqrt(12.5)
        let out = rmsnorm(&[3.0, 4.0], 2, 0.0);
        let s = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * s).abs() < 1e-6);
        assert!((out[1] - 4.0 * s).abs() < 1e-6);
    }

    #[test]
    fn mul_rows_broadcasts_weight() {
        // 2 linhas dim=2, peso [10,100]
        let out = mul_rows(&[1.0, 2.0, 3.0, 4.0], &[10.0, 100.0], 2);
        assert_eq!(out, vec![10.0, 200.0, 30.0, 400.0]);
    }
}
```

- [ ] **Step 2: Rodar os testes**

Run: `cargo test -p llama-model ops::tests`
Expected: PASS (4 testes desta parte).

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/ops.rs
git commit -m "feat(llama-model): ops embedding/rmsnorm/mul_rows (Fase 2 Task 3)"
```

---

## Task 4: `ops.rs` — matmul

**Files:**
- Modify: `crates/llama-model/src/ops.rs`

- [ ] **Step 1: Adicionar `matmul` + teste**

Inserir em `ops.rs` (antes do `#[cfg(test)]`):

```rust
/// MUL_MAT: `W{in,out}` (out linhas de comprimento in) × `x` token-major [n_tok*in].
/// Saída token-major [n_tok*out]: `out[t*out+j] = Σ_i W[j*in+i] * x[t*in+i]`.
pub(crate) fn matmul(w: &[f32], x: &[f32], n_in: usize, n_out: usize, n_tok: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * n_out];
    for t in 0..n_tok {
        let xrow = &x[t * n_in..t * n_in + n_in];
        let orow = &mut out[t * n_out..t * n_out + n_out];
        for (j, o) in orow.iter_mut().enumerate() {
            let wrow = &w[j * n_in..j * n_in + n_in];
            *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
        }
    }
    out
}
```

Adicionar dentro de `mod tests`:

```rust
#[test]
fn matmul_2x2_identity_and_general() {
    // W in=2,out=2: linha0=[1,0], linha1=[0,1] (identidade) → out=x
    let w_id = vec![1.0, 0.0, 0.0, 1.0];
    let x = vec![5.0, 7.0]; // 1 token
    assert_eq!(matmul(&w_id, &x, 2, 2, 1), vec![5.0, 7.0]);

    // W in=2,out=1: linha0=[2,3] → out[0]=2*5+3*7=31
    let w = vec![2.0, 3.0];
    assert_eq!(matmul(&w, &x, 2, 1, 1), vec![31.0]);
}

#[test]
fn matmul_two_tokens() {
    // W in=2,out=2: linha0=[1,1], linha1=[1,-1]
    let w = vec![1.0, 1.0, 1.0, -1.0];
    let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
    // t0: [1+2, 1-2]=[3,-1]; t1:[3+4,3-4]=[7,-1]
    assert_eq!(matmul(&w, &x, 2, 2, 2), vec![3.0, -1.0, 7.0, -1.0]);
}
```

- [ ] **Step 2: Rodar os testes**

Run: `cargo test -p llama-model ops::tests::matmul`
Expected: PASS (2 testes).

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/ops.rs
git commit -m "feat(llama-model): matmul f32 (Fase 2 Task 4)"
```

---

## Task 5: `ops.rs` — RoPE NORM

**Files:**
- Modify: `crates/llama-model/src/ops.rs`

- [ ] **Step 1: Adicionar `rope_norm` + teste**

Inserir em `ops.rs` (antes do `#[cfg(test)]`). `x` é token-major [n_tok * n_head * head_dim]; rotaciona in-place os pares de cada head para a posição absoluta `pos0 + t`.

```rust
/// RoPE NORM (arch llama): rotaciona pares (2i,2i+1) de cada head.
/// `θ_i = pos * freq_base^(-2i/rope_dim)`, para i em 0..rope_dim/2.
pub(crate) fn rope_norm(
    x: &mut [f32],
    n_tok: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    pos0: usize,
) {
    for t in 0..n_tok {
        let pos = (pos0 + t) as f32;
        for h in 0..n_head {
            let base = (t * n_head + h) * head_dim;
            for i in 0..rope_dim / 2 {
                let theta = pos * freq_base.powf(-2.0 * i as f32 / rope_dim as f32);
                let (s, c) = theta.sin_cos();
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                x[base + 2 * i] = a * c - b * s;
                x[base + 2 * i + 1] = a * s + b * c;
            }
        }
    }
}
```

Adicionar em `mod tests`:

```rust
#[test]
fn rope_norm_pos_zero_is_identity() {
    // pos=0 → θ=0 → cos=1,sin=0 → sem mudança
    let mut x = vec![1.0, 2.0, 3.0, 4.0]; // 1 tok, 1 head, head_dim=4, rope_dim=4
    rope_norm(&mut x, 1, 1, 4, 4, 10000.0, 0);
    assert!(
        x.iter()
            .zip([1.0, 2.0, 3.0, 4.0])
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
}

#[test]
fn rope_norm_pos_one_rotates_first_pair_by_one_radian() {
    // i=0 → θ = 1 * base^0 = 1 rad. par (x0,x1)=(1,0) → (cos1, sin1)
    let mut x = vec![1.0, 0.0, 0.0, 0.0];
    rope_norm(&mut x, 1, 1, 4, 4, 10000.0, 1);
    assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
    assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);
}
```

- [ ] **Step 2: Rodar os testes**

Run: `cargo test -p llama-model ops::tests::rope`
Expected: PASS (2 testes).

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/ops.rs
git commit -m "feat(llama-model): rope_norm (Fase 2 Task 5)"
```

---

## Task 6: `ops.rs` — silu/swiglu/softmax/argmax

**Files:**
- Modify: `crates/llama-model/src/ops.rs`

- [ ] **Step 1: Adicionar funções + testes**

Inserir em `ops.rs` (antes do `#[cfg(test)]`):

```rust
/// SiLU: `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SWIGLU: `silu(gate) * up`, elementwise (mesmo comprimento).
pub(crate) fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect()
}

/// Softmax numericamente estável sobre um slice (in-place).
pub(crate) fn softmax(z: &mut [f32]) {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in z.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in z.iter_mut() {
            *v /= sum;
        }
    }
}

/// Índice do maior valor (greedy / argmax). Empate → menor índice.
pub(crate) fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
```

Adicionar em `mod tests`:

```rust
#[test]
fn swiglu_matches_manual() {
    // gate=[0,2], up=[1,3]; silu(0)=0, silu(2)=2*sigmoid(2)
    let out = swiglu(&[0.0, 2.0], &[1.0, 3.0]);
    assert!((out[0] - 0.0).abs() < 1e-6);
    let silu2 = 2.0 / (1.0 + (-2.0f32).exp());
    assert!((out[1] - silu2 * 3.0).abs() < 1e-5);
}

#[test]
fn softmax_sums_to_one() {
    let mut z = vec![1.0, 2.0, 3.0];
    softmax(&mut z);
    assert!((z.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(z[2] > z[1] && z[1] > z[0]);
}

#[test]
fn argmax_picks_first_max() {
    assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.2]), 1);
}
```

- [ ] **Step 2: Rodar os testes**

Run: `cargo test -p llama-model ops::tests`
Expected: PASS (todos os testes de ops).

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/ops.rs
git commit -m "feat(llama-model): silu/swiglu/softmax/argmax (Fase 2 Task 6)"
```

---

## Task 7: `attention.rs` — KvCache + atenção causal GQA

**Files:**
- Modify: `crates/llama-model/src/attention.rs`

- [ ] **Step 1: Escrever impl + testes**

`crates/llama-model/src/attention.rs` (substituir o stub):

```rust
//! KV cache f32 e atenção causal GQA. Layout K/V: [pos * (n_head_kv*head_dim)].
#![allow(clippy::indexing_slicing)]

use crate::ops::softmax;

/// KV cache f32, uma entrada por camada. `len` = posições já armazenadas.
pub(crate) struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    len: usize,
}

impl KvCache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_layer],
            v: vec![Vec::new(); n_layer],
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Anexa K/V (token-major [n_tok*kv_dim]) à camada `l`.
    pub fn append(&mut self, l: usize, k: &[f32], v: &[f32]) {
        self.k[l].extend_from_slice(k);
        self.v[l].extend_from_slice(v);
    }

    /// Avança o relógio do cache após processar todas as camadas.
    pub fn advance(&mut self, n_tok: usize) {
        self.len += n_tok;
    }
}

/// Atenção causal GQA em f32. `q` pós-rope token-major [n_tok*n_head*head_dim];
/// `k_cache`/`v_cache` são os buffers COMPLETOS da camada (já com os n_tok novos),
/// [(pos0+n_tok)*kv_dim]. Retorna [n_tok*n_head*head_dim].
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_tok: usize,
    pos0: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_embd = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let n_rep = n_head / n_head_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_tok * n_embd];

    for t in 0..n_tok {
        let abs_pos = pos0 + t;
        for h in 0..n_head {
            let kv_h = h / n_rep;
            let q_off = (t * n_head + h) * head_dim;
            let qv = &q[q_off..q_off + head_dim];

            // scores causais sobre posições 0..=abs_pos
            let mut scores = vec![0.0f32; abs_pos + 1];
            for (j, s) in scores.iter_mut().enumerate() {
                let k_off = j * kv_dim + kv_h * head_dim;
                let kv = &k_cache[k_off..k_off + head_dim];
                let dot: f32 = qv.iter().zip(kv.iter()).map(|(&a, &b)| a * b).sum();
                *s = dot * scale;
            }
            softmax(&mut scores);

            // saída ponderada por V
            let o_off = (t * n_head + h) * head_dim;
            let ov = &mut out[o_off..o_off + head_dim];
            for (j, &p) in scores.iter().enumerate() {
                let v_off = j * kv_dim + kv_h * head_dim;
                let vv = &v_cache[v_off..v_off + head_dim];
                for (o, &vval) in ov.iter_mut().zip(vv.iter()) {
                    *o += p * vval;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_token_attends_only_itself() {
        // n_head=1, n_head_kv=1, head_dim=2, 1 token, pos0=0.
        // Só uma posição → softmax trivial → out == V.
        let q = vec![1.0, 0.0];
        let k = vec![0.5, 0.5];
        let v = vec![9.0, 7.0];
        let out = attention(&q, &k, &v, 1, 0, 1, 1, 2);
        assert!((out[0] - 9.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn gqa_maps_query_heads_to_kv_heads() {
        // n_head=2, n_head_kv=1 → ambos query heads usam kv head 0. head_dim=1.
        // 1 token, 1 posição → out de cada head == V[0].
        let q = vec![1.0, 2.0]; // head0=1, head1=2
        let k = vec![3.0];
        let v = vec![5.0];
        let out = attention(&q, &k, &v, 1, 0, 2, 1, 1);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn causal_two_positions_first_token_ignores_future() {
        // n_head=1,n_head_kv=1,head_dim=1, 2 tokens prefill, pos0=0.
        // K=[k0,k1], V=[v0,v1]. token0 (pos0) só vê pos0 → out0=v0.
        let q = vec![1.0, 1.0];
        let k = vec![0.0, 100.0]; // k1 grande não afeta token0
        let v = vec![2.0, 8.0];
        let out = attention(&q, &k, &v, 2, 0, 1, 1, 1);
        assert!((out[0] - 2.0).abs() < 1e-6); // token0 → v0
    }

    #[test]
    fn kvcache_append_and_advance() {
        let mut c = KvCache::new(2);
        c.append(0, &[1.0, 2.0], &[3.0, 4.0]);
        c.advance(1);
        assert_eq!(c.len(), 1);
        assert_eq!(c.k[0], vec![1.0, 2.0]);
    }
}
```

- [ ] **Step 2: Rodar os testes**

Run: `cargo test -p llama-model attention::tests`
Expected: PASS (4 testes).

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/attention.rs
git commit -m "feat(llama-model): KvCache + atenção causal GQA f32 (Fase 2 Task 7)"
```

---

## Task 8: `model.rs` — Model::load + forward + sanidade por-op

**Files:**
- Modify: `crates/llama-model/src/model.rs`
- Modify: `crates/llama-model/src/lib.rs` (adicionar `pub use model::Model;`)

- [ ] **Step 1: Escrever `Model::load` + `forward` + `forward_argmax`**

`crates/llama-model/src/model.rs` (substituir o stub):

```rust
//! Modelo Llama: carrega config+pesos e executa o forward f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgufFile;

use crate::attention::{attention, KvCache};
use crate::config::LlamaConfig;
use crate::error::ModelError;
use crate::ops::{argmax, embedding_lookup, matmul, mul_rows, rmsnorm, rope_norm, swiglu};
use crate::weights::Weights;

/// Modelo carregado: config + pesos f32.
pub struct Model {
    pub config: LlamaConfig,
    pub(crate) weights: Weights,
}

impl Model {
    /// Carrega de um GGUF já parseado + bytes do arquivo.
    pub fn load(f: &GgufFile, bytes: &[u8]) -> Result<Self, ModelError> {
        let config = LlamaConfig::from_gguf(f)?;
        let weights = Weights::from_gguf(f, bytes, &config)?;
        Ok(Self { config, weights })
    }

    pub(crate) fn new_cache(&self) -> KvCache {
        KvCache::new(self.config.n_layer)
    }

    /// Processa `tokens` (prefill ou 1 token de decode) e devolve os logits
    /// (tamanho `vocab`) do ÚLTIMO token. Atualiza o `cache`.
    pub(crate) fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let n_tok = tokens.len();
        let pos0 = cache.len();
        let kv_dim = c.n_head_kv * c.head_dim;

        let mut x = embedding_lookup(&self.weights.token_embd, tokens, c.n_embd)?;

        for (l, lw) in self.weights.layers.iter().enumerate() {
            // --- bloco de atenção ---
            let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
            let attn_in = mul_rows(&normed, &lw.attn_norm, c.n_embd);

            let mut q = matmul(&lw.attn_q, &attn_in, c.n_embd, c.n_embd, n_tok);
            let mut k = matmul(&lw.attn_k, &attn_in, c.n_embd, kv_dim, n_tok);
            let v = matmul(&lw.attn_v, &attn_in, c.n_embd, kv_dim, n_tok);

            rope_norm(&mut q, n_tok, c.n_head, c.head_dim, c.rope_dim, c.freq_base, pos0);
            rope_norm(&mut k, n_tok, c.n_head_kv, c.head_dim, c.rope_dim, c.freq_base, pos0);

            cache.append(l, &k, &v);
            let attn = attention(
                &q,
                &cache.k[l],
                &cache.v[l],
                n_tok,
                pos0,
                c.n_head,
                c.n_head_kv,
                c.head_dim,
            );
            let attn_out = matmul(&lw.attn_output, &attn, c.n_embd, c.n_embd, n_tok);

            // residual 1
            for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }

            // --- bloco FFN ---
            let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
            let ffn_in = mul_rows(&normed, &lw.ffn_norm, c.n_embd);
            let gate = matmul(&lw.ffn_gate, &ffn_in, c.n_embd, c.n_ff, n_tok);
            let up = matmul(&lw.ffn_up, &ffn_in, c.n_embd, c.n_ff, n_tok);
            let act = swiglu(&gate, &up);
            let ffn_out = matmul(&lw.ffn_down, &act, c.n_ff, c.n_embd, n_tok);

            // residual 2
            for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
        }

        cache.advance(n_tok);

        // norma final + projeção de saída só do último token
        let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
        let final_x = mul_rows(&normed, &self.weights.output_norm, c.n_embd);
        let last = &final_x[(n_tok - 1) * c.n_embd..n_tok * c.n_embd];
        let logits = matmul(&self.weights.output, last, c.n_embd, c.vocab, 1);
        Ok(logits)
    }

    /// Atalho: argmax dos logits do último token.
    pub(crate) fn forward_argmax(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<u32, ModelError> {
        let logits = self.forward(tokens, cache)?;
        u32::try_from(argmax(&logits)).map_err(|_| ModelError::Overflow)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::ops::{embedding_lookup, matmul, mul_rows, rmsnorm, rope_norm};
    use std::path::Path;

    fn load_model() -> Option<(Model, Vec<u8>)> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        let m = Model::load(&f, &bytes).ok()?;
        Some((m, bytes))
    }

    #[test]
    fn embd_and_qcur_sums_match_oracle() {
        let Some((m, _)) = load_model() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let c = &m.config;
        let tokens = [1u32, 403, 407, 261, 378]; // "Once upon a time"
        let n_tok = tokens.len();

        // embd sum == -3.354056
        let x = embedding_lookup(&m.weights.token_embd, &tokens, c.n_embd).unwrap();
        let embd_sum: f32 = x.iter().sum();
        assert!((embd_sum - (-3.354056)).abs() < 1e-2, "embd_sum={embd_sum}");

        // Qcur-0 pós-rope sum == 148.969818
        let lw = &m.weights.layers[0];
        let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
        let attn_in = mul_rows(&normed, &lw.attn_norm, c.n_embd);
        let mut q = matmul(&lw.attn_q, &attn_in, c.n_embd, c.n_embd, n_tok);
        rope_norm(&mut q, n_tok, c.n_head, c.head_dim, c.rope_dim, c.freq_base, 0);
        let q_sum: f32 = q.iter().sum();
        assert!((q_sum - 148.969818).abs() < 1e-1, "q_sum={q_sum}");
    }
}
```

Adicionar em `lib.rs`: `pub use model::Model;`

- [ ] **Step 2: Rodar os testes**

Run: `cargo test -p llama-model model::tests`
Expected: PASS (sums batem dentro da tolerância). Se `embd_sum` divergir → conferir ordem LE em `tensor_f32`. Se `q_sum` divergir → suspeitar do tipo de RoPE (ver Riscos) — usar superpowers:systematic-debugging.

- [ ] **Step 3: Commit**

```bash
git add crates/llama-model/src/model.rs crates/llama-model/src/lib.rs
git commit -m "feat(llama-model): Model::forward + sanidade por-op vs oráculo (Fase 2 Task 8)"
```

---

## Task 9: `generate.rs` — generate_greedy + gate diferencial

**Files:**
- Modify: `crates/llama-model/src/generate.rs`
- Create: `crates/llama-model/tests/oracle_forward.rs`

- [ ] **Step 1: Implementar `generate_greedy` como método de `Model`**

`crates/llama-model/src/generate.rs` (substituir o stub):

```rust
//! Geração greedy (argmax, temp 0) com KV cache.

use llama_tokenizer::Tokenizer;

use crate::error::ModelError;
use crate::model::Model;

impl Model {
    /// Gera até `n_tokens` por argmax a partir de `prompt` (com BOS). Para em EOS.
    /// Retorna o texto decodificado dos tokens GERADOS (sem o prompt), espelhando
    /// `--no-display-prompt` do oráculo.
    pub fn generate_greedy(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        n_tokens: usize,
    ) -> Result<String, ModelError> {
        let prompt_ids = tokenizer.encode(prompt, true);
        let mut cache = self.new_cache();

        let mut next = self.forward_argmax(&prompt_ids, &mut cache)?;
        let mut generated = Vec::with_capacity(n_tokens);
        generated.push(next);

        while generated.len() < n_tokens {
            if next == self.config.eos_id {
                break;
            }
            next = self.forward_argmax(&[next], &mut cache)?;
            generated.push(next);
        }

        Ok(tokenizer.decode(&generated))
    }
}
```

- [ ] **Step 2: Garantir build**

Run: `cargo build -p llama-model`
Expected: PASS. (`Tokenizer::encode`/`decode` e `Tokenizer::from_gguf` são públicos — ver `crates/llama-tokenizer/src/lib.rs`.)

- [ ] **Step 3: Escrever o gate diferencial greedy**

`crates/llama-model/tests/oracle_forward.rs` (NOVO):

```rust
//! Teste diferencial contra o oráculo (auto-skip se modelo/refs ausentes).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use gguf::GgufFile;
use llama_model::Model;
use llama_tokenizer::Tokenizer;

#[test]
fn greedy_generation_matches_oracle_reference() {
    let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let Ok(reference) = std::fs::read_to_string(Path::new("../../refs/greedy.txt")) else {
        eprintln!("refs/greedy.txt ausente — pulando");
        return;
    };
    let f = GgufFile::parse(&bytes).unwrap();
    let model = Model::load(&f, &bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();

    let out = model.generate_greedy(&tok, "Once upon a time", 32).unwrap();

    // Gate duro: igualdade exata da sequência greedy decodificada.
    assert_eq!(out, reference, "\n  got: {out:?}\n  ref: {reference:?}");
}
```

- [ ] **Step 4: Rodar o gate diferencial**

Run: `cargo test -p llama-model --test oracle_forward -- --nocapture`
Expected: PASS — `out == refs/greedy.txt`.

Se FALHAR: usar superpowers:systematic-debugging. Ordem provável de investigação:
1. Tokenização do prompt (já bit-exact; conferir len==5, ids `[1,403,407,261,378]`).
2. Tipo de RoPE — confirmar NORM vs NeoX lendo `llama.cpp/src/llama-model.cpp` (`rope_type` para `LLM_ARCH_LLAMA`). Se NeoX, ajustar `rope_norm` para rotacionar `(i, i+rope_dim/2)`.
3. Trailing newline em `refs/greedy.txt` — se a única diferença for `\n` final, comparar `out == reference.trim_end_matches('\n')` (ajustar a asserção, NÃO o texto gerado).
4. Divergência f32 vs flash-attn/f16 na cauda: comparar token-a-token; replicar arredondamento f16 do KV cache está FORA DO ESCOPO — registrar e discutir com o usuário antes de expandir.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-model/src/generate.rs crates/llama-model/tests/oracle_forward.rs
git commit -m "feat(llama-model): generate_greedy + gate diferencial vs refs/greedy.txt (Fase 2 Task 9)"
```

---

## Task 10: Gate de qualidade completo + cobertura

**Files:** nenhum novo (ajustes pontuais se clippy reclamar).

- [ ] **Step 1: fmt**

Run: `cargo fmt --all` então `cargo fmt --all --check`
Expected: sem diferenças.

- [ ] **Step 2: clippy estrito**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS. Se aparecer `clippy::indexing_slicing` em `ops.rs`/`attention.rs`/`model.rs`, confirmar o `#![allow(clippy::indexing_slicing)]` no topo do módulo. `cast_precision_loss` NÃO está na lista negada (ignorar). Se aparecer `cast_possible_truncation`, trocar `as` por `try_from`.

- [ ] **Step 3: testes do workspace**

Run: `cargo test --workspace`
Expected: PASS (todos os crates; testes do llama-model rodam com o modelo presente).

- [ ] **Step 4: gate completo (inclui cobertura ≥80%)**

Run: `./scripts/gate.sh`
Expected: termina com `GATE OK`. A cobertura do `llama-model` vem dos unit tests das ops + sanidade + gate diferencial (que exercita forward/generate ponta-a-ponta). Se < 80%, adicionar teste do ramo de erro descoberto (ex.: `LlamaConfig::from_gguf` com chave faltando via `GgufBuilder` de `gguf::test_support`, ou `Weights::from_gguf` em tensor não-f32).

- [ ] **Step 5: Commit final (se houve ajustes de lint/fmt)**

```bash
git add -A
git commit -m "chore(llama-model): gate verde (fmt + clippy + cobertura) (Fase 2 Task 10)"
```

---

## Riscos conhecidos

1. **Tipo de RoPE.** A spec assume NORM (arch `llama`). Validado cedo pelo `Qcur-0 sum` (Task 8) — se divergir, é o primeiro suspeito (alternar para NeoX: rotacionar `(i, i+rope_dim/2)`). Verificação mais barata de correção do RoPE.
2. **Bit-exactness vs flash-attn/f16.** O gate é token-match, não tensor-exact. Se a sequência greedy divergir só na cauda por ruído f32↔f16, expandir escopo (cache f16) exige aprovação do usuário — não fazer silenciosamente.
3. **Cobertura depende do modelo.** `forward`/`generate` só são cobertos com `models/stories260K.gguf` presente (está, localmente). Mesma premissa dos testes de Fase 1.

## Self-Review (preenchido)

- **Cobertura da spec:** config (T1), weights (T2), todas as ops do grafo — embedding/rmsnorm/mul/matmul/rope/swiglu/softmax (T3-T6), atenção GQA + KvCache (T7), forward completo + sanidade por-op (T8), generate greedy + gate duro (T9), gate de qualidade (T10). Sem lacunas.
- **Placeholders:** nenhum TODO/TBD; todo passo tem código/comando concreto.
- **Consistência de tipos:** assinaturas usadas de forma idêntica entre tasks — `matmul(w,x,n_in,n_out,n_tok)`, `rope_norm(x,n_tok,n_head,head_dim,rope_dim,freq_base,pos0)`, `attention(q,k_cache,v_cache,n_tok,pos0,n_head,n_head_kv,head_dim)`, `KvCache::{new,len,append,advance}`, `Model::{load,new_cache,forward,forward_argmax,generate_greedy}`, `Weights::from_gguf(f,bytes,cfg)`, `LlamaConfig::from_gguf(f)`.
- **Build incremental:** `lib.rs` cresce os `pub use` por task (error→T0, config→T1, model→T8) para manter build verde a cada commit.
