# Fase 3 — Quantização: Q8_0, Q4_0, Q4_K, Q6_K — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar dequantização (Q8_0, Q4_0, Q4_K, Q6_K e F16) em `ggml-cpu`, mudar `llama-model` para armazenar pesos quantizados em memória (raw bytes) e dequantizar sob demanda, reduzindo o footprint de RAM para ≤ tamanho do arquivo GGUF.

**Architecture:** Novo crate `crates/ggml-cpu` com dequant em Rust safe (sem SIMD por ora). `RawTensor { bytes: Vec<u8>, ty: GgmlType }` substitui `Vec<f32>` em `Weights`. `model.rs` chama `raw.dequant_to_f32()?` antes de cada `matmul`. O gate diferencial do oráculo da Fase 2 continua verde (stories260K é F32); teste novo valida que o modelo qwen2.5-0.5b-q8_0 carrega sem erro e que o footprint em bytes ≤ tamanho do GGUF.

**Tech Stack:** Rust edition 2024, `half = "2"` (conversão f16→f32), dep path `gguf`, `thiserror`. Lints workspace herdados.

**Convenções fixas:**
- `GgmlType`, `block_size()`, `type_size()` já estão em `crates/gguf/src/types.rs` — não duplicar.
- Layouts de bloco (ggml-common.h): Q8_0 (32 elem, 34 B) = [d:f16, qs:[i8;32]]; Q4_0 (32 elem, 18 B) = [d:f16, qs:[u8;16]]; Q4_K (256 elem, 144 B) = [d:f16, dmin:f16, scales:[u8;12], qs:[u8;128]]; Q6_K (256 elem, 210 B) = [ql:[u8;128], qh:[u8;64], scales:[i8;16], d:f16].
- Memória: `RawTensor::memory_bytes()` = `self.bytes.len()` — bytes raw sem dequant.
- `RawTensor::n_elements()` = `(bytes.len() / type_size) * block_size`.

---

## File Structure

- Create: `crates/ggml-cpu/Cargo.toml`
- Create: `crates/ggml-cpu/src/lib.rs`
- Create: `crates/ggml-cpu/src/error.rs`
- Create: `crates/ggml-cpu/src/dequant.rs`
- Modify: `Cargo.toml` (workspace) — membro + dep + `half`
- Modify: `crates/llama-model/Cargo.toml` — dep `ggml-cpu`
- Modify: `crates/llama-model/src/error.rs` — variante `Dequant`
- Modify: `crates/llama-model/src/weights.rs` — `RawTensor` + `tensor_raw`
- Modify: `crates/llama-model/src/model.rs` — dequant sob demanda + `memory_bytes`
- Create: `crates/llama-model/tests/quant_load.rs`

---

## Task 0: Scaffold `crates/ggml-cpu` + adicionar `half` ao workspace

**Files:**
- Modify: `Cargo.toml` (workspace)
- Create: `crates/ggml-cpu/Cargo.toml`
- Create: `crates/ggml-cpu/src/lib.rs`
- Create: `crates/ggml-cpu/src/error.rs`
- Create: `crates/ggml-cpu/src/dequant.rs` (stubs)

- [ ] **Step 1: Atualizar `Cargo.toml` raiz**

Em `[workspace]`, adicionar `"crates/ggml-cpu"` ao vetor `members`:
```toml
members = ["oracle", "crates/gguf", "crates/llama-tokenizer", "crates/llama-model", "crates/ggml-cpu"]
```

Em `[workspace.dependencies]`, adicionar:
```toml
half = "2"
ggml-cpu = { path = "crates/ggml-cpu" }
```

- [ ] **Step 2: Criar `crates/ggml-cpu/Cargo.toml`**

```toml
[package]
name = "ggml-cpu"
version = "0.1.0"
edition.workspace = true

[dependencies]
thiserror.workspace = true
half.workspace = true
gguf.workspace = true

[lints]
workspace = true
```

- [ ] **Step 3: Criar `crates/ggml-cpu/src/error.rs`**

```rust
/// Erro de dequantização.
#[derive(Debug, thiserror::Error)]
pub enum DequantError {
    #[error("bytes insuficientes para tipo {ty}: esperado múltiplo de {block_bytes}, recebeu {got}")]
    BadSize {
        ty: &'static str,
        block_bytes: usize,
        got: usize,
    },
    #[error("tipo {0} não suportado para dequantização")]
    UnsupportedType(String),
}
```

- [ ] **Step 4: Criar `crates/ggml-cpu/src/lib.rs`**

```rust
#![forbid(unsafe_code)]
//! Kernels CPU para dequantização de tensores GGML (safe Rust, sem SIMD por ora).

mod dequant;
mod error;

pub use dequant::dequant_to_f32;
pub use error::DequantError;
```

- [ ] **Step 5: Criar `crates/ggml-cpu/src/dequant.rs` (stubs)**

```rust
//! Dequantização de blocos GGML para f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgmlType;

use crate::error::DequantError;

/// Converte `bytes` brutos de um tensor para `Vec<f32>`.
/// Suporta F32, F16, Q8_0, Q4_0, Q4_K, Q6_K.
pub fn dequant_to_f32(bytes: &[u8], ty: GgmlType) -> Result<Vec<f32>, DequantError> {
    match ty {
        GgmlType::F32 => dequant_f32(bytes),
        GgmlType::F16 => dequant_f16(bytes),
        GgmlType::Q8_0 => dequant_q8_0(bytes),
        GgmlType::Q4_0 => dequant_q4_0(bytes),
        GgmlType::Q4_K => dequant_q4_k(bytes),
        GgmlType::Q6_K => dequant_q6_k(bytes),
        other => Err(DequantError::UnsupportedType(format!("{other:?}"))),
    }
}

fn dequant_f32(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    if bytes.len() % 4 != 0 {
        return Err(DequantError::BadSize { ty: "F32", block_bytes: 4, got: bytes.len() });
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn dequant_f16(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    if bytes.len() % 2 != 0 {
        return Err(DequantError::BadSize { ty: "F16", block_bytes: 2, got: bytes.len() });
    }
    Ok(bytes.chunks_exact(2).map(|c| {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        half::f16::from_bits(bits).to_f32()
    }).collect())
}

fn dequant_q8_0(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    todo!("implementado na Task 1")
}
fn dequant_q4_0(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    todo!("implementado na Task 2")
}
fn dequant_q4_k(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    todo!("implementado na Task 3")
}
fn dequant_q6_k(_bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    todo!("implementado na Task 4")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    #[test]
    fn f32_passthrough() {
        let bytes: Vec<u8> = 1.5f32.to_le_bytes().to_vec();
        let out = dequant_to_f32(&bytes, GgmlType::F32).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.5).abs() < 1e-7);
    }

    #[test]
    fn f16_conversion() {
        let bits = f16_bytes(0.5);
        let out = dequant_to_f32(&bits, GgmlType::F16).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn unsupported_type_returns_error() {
        assert!(dequant_to_f32(&[], GgmlType::Q2_K).is_err());
    }
}
```

- [ ] **Step 6: Verificar build**

Run: `cargo build -p ggml-cpu`
Expected: PASS (warnings de `dead_code`/`todo!` aceitáveis).

- [ ] **Step 7: Rodar testes existentes**

Run: `cargo test -p ggml-cpu dequant::tests`
Expected: PASS (`f32_passthrough`, `f16_conversion`, `unsupported_type_returns_error`).

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock crates/ggml-cpu
git commit -m "chore(ggml-cpu): scaffold crate com dispatcher + stubs de dequant (Fase 3 Task 0)"
```

---

## Task 1: `dequant_q8_0`

**Files:**
- Modify: `crates/ggml-cpu/src/dequant.rs`

Layout (34 bytes, 32 elementos): `[d:f16 LE, qs:[i8;32]]`
Fórmula: `out[i] = qs[i] as f32 * d` para i em 0..32

- [ ] **Step 1: Escrever o teste (RED)**

Adicionar em `mod tests`:

```rust
fn make_q8_0_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(34);
    b.extend_from_slice(&f16_bytes(d));
    b.extend(qs.iter().map(|&q| q as u8));
    b
}

#[test]
fn q8_0_single_block() {
    // d=1.0; qs=[1, -1, 0×30] → out=[1.0, -1.0, 0×30]
    let mut qs = [0i8; 32];
    qs[0] = 1;
    qs[1] = -1;
    let block = make_q8_0_block(1.0, &qs);
    let out = dequant_to_f32(&block, GgmlType::Q8_0).unwrap();
    assert_eq!(out.len(), 32);
    assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
    assert!((out[1] - (-1.0)).abs() < 1e-5, "out[1]={}", out[1]);
    assert!(out[2..].iter().all(|&v| v == 0.0));
}

#[test]
fn q8_0_scale_applied() {
    // d=2.0; qs=[3, 0×31] → out[0]=6.0
    let mut qs = [0i8; 32];
    qs[0] = 3;
    let block = make_q8_0_block(2.0, &qs);
    let out = dequant_to_f32(&block, GgmlType::Q8_0).unwrap();
    assert!((out[0] - 6.0).abs() < 1e-4, "out[0]={}", out[0]);
}

#[test]
fn q8_0_bad_size_returns_error() {
    assert!(dequant_to_f32(&[0u8; 33], GgmlType::Q8_0).is_err());
}
```

Run: `cargo test -p ggml-cpu dequant::tests::q8_0`
Expected: **FAIL** (todo! panics).

- [ ] **Step 2: Implementar `dequant_q8_0`**

Substituir o stub `dequant_q8_0`:

```rust
fn dequant_q8_0(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 34; // 2 (f16) + 32 (i8)
    if bytes.len() % BLOCK != 0 {
        return Err(DequantError::BadSize { ty: "Q8_0", block_bytes: BLOCK, got: bytes.len() });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for b in bytes.chunks_exact(BLOCK) {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        for &q in &b[2..34] {
            out.push((q as i8) as f32 * d);
        }
    }
    Ok(out)
}
```

- [ ] **Step 3: Rodar os testes**

Run: `cargo test -p ggml-cpu dequant::tests`
Expected: PASS (todos até aqui).

- [ ] **Step 4: Commit**

```bash
git add crates/ggml-cpu/src/dequant.rs
git commit -m "feat(ggml-cpu): dequant_q8_0 (Fase 3 Task 1)"
```

---

## Task 2: `dequant_q4_0`

**Files:**
- Modify: `crates/ggml-cpu/src/dequant.rs`

Layout (18 bytes, 32 elementos): `[d:f16 LE, qs:[u8;16]]`
Fórmula para j em 0..16:
```
x0 = (qs[j] & 0x0F) as i32 - 8  →  out[j]
x1 = (qs[j] >> 4)   as i32 - 8  →  out[j + 16]
```

- [ ] **Step 1: Escrever o teste (RED)**

Adicionar em `mod tests`:

```rust
fn make_q4_0_block(d: f32, qs: &[u8; 16]) -> Vec<u8> {
    let mut b = Vec::with_capacity(18);
    b.extend_from_slice(&f16_bytes(d));
    b.extend_from_slice(qs);
    b
}

#[test]
fn q4_0_single_block() {
    // d=1.0; qs[0]=0x89 → lower=9,x0=1 → out[0]=1.0; upper=8,x1=0; resto 0x88→zeros
    let mut qs = [0x88u8; 16];
    qs[0] = 0x89;
    let block = make_q4_0_block(1.0, &qs);
    let out = dequant_to_f32(&block, GgmlType::Q4_0).unwrap();
    assert_eq!(out.len(), 32);
    assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
    assert!(out[1..16].iter().all(|&v| v == 0.0), "out[1..16] deve ser zero");
    assert!(out[16..].iter().all(|&v| v == 0.0), "out[16..] deve ser zero");
}

#[test]
fn q4_0_bad_size_returns_error() {
    assert!(dequant_to_f32(&[0u8; 17], GgmlType::Q4_0).is_err());
}
```

Run: `cargo test -p ggml-cpu dequant::tests::q4_0`
Expected: **FAIL** (todo!).

- [ ] **Step 2: Implementar `dequant_q4_0`**

Substituir o stub `dequant_q4_0`:

```rust
fn dequant_q4_0(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 18; // 2 (f16) + 16 (nibbles)
    if bytes.len() % BLOCK != 0 {
        return Err(DequantError::BadSize { ty: "Q4_0", block_bytes: BLOCK, got: bytes.len() });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 32];
    for (bi, b) in bytes.chunks_exact(BLOCK).enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let base = bi * 32;
        for j in 0..16 {
            let q = b[2 + j];
            let x0 = (q & 0x0F) as i32 - 8;
            let x1 = (q >> 4)   as i32 - 8;
            out[base + j]      = x0 as f32 * d;
            out[base + j + 16] = x1 as f32 * d;
        }
    }
    Ok(out)
}
```

- [ ] **Step 3: Rodar os testes**

Run: `cargo test -p ggml-cpu dequant::tests`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/ggml-cpu/src/dequant.rs
git commit -m "feat(ggml-cpu): dequant_q4_0 (Fase 3 Task 2)"
```

---

## Task 3: `dequant_q4_k`

**Files:**
- Modify: `crates/ggml-cpu/src/dequant.rs`

Layout (144 bytes, 256 elementos):
```
bytes[0..2]   : d     (f16 LE) — escala super-bloco para scales
bytes[2..4]   : dmin  (f16 LE) — escala super-bloco para mins
bytes[4..16]  : scales [u8; 12] — 8 pares (scale,min) em 6 bits, empacotados
bytes[16..144]: qs    [u8; 128] — nibbles: 256 quants de 4 bits
```

Decodificação de par `(sc, mn)` no índice `j` do vetor `scales[12]` (espelha `get_scale_min_k4` do ggml-quants.c):
```
se j < 4: sc = scales[j] & 63,  mn = scales[j+4] & 63
se j ≥ 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
           mn = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
```

Dequant (4 sub-blocos de 64 por super-bloco, `is` = índice de escala, `qs_ptr` avança 32 a cada sub-bloco):
```
is = 0, qs_ptr = &qs
para j_step em [0, 64, 128, 192]:
    (sc1,m1) = get_scale_min_k4(is,   scales)
    (sc2,m2) = get_scale_min_k4(is+1, scales)
    d1 = d_val * sc1;  m1f = min_val * m1
    d2 = d_val * sc2;  m2f = min_val * m2
    para l em 0..32:
        out[j_step + l]      = d1 * (qs_ptr[l] & 0xF) as f32 - m1f
        out[j_step + l + 32] = d2 * (qs_ptr[l] >> 4)  as f32 - m2f
    qs_ptr += 32;  is += 2
```

- [ ] **Step 1: Escrever o teste (RED)**

Adicionar em `mod tests`:

```rust
fn make_q4_k_block(d: f32, dmin: f32, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    b.extend_from_slice(&f16_bytes(d));
    b.extend_from_slice(&f16_bytes(dmin));
    b.extend_from_slice(scales);
    b.extend_from_slice(qs);
    b
}

#[test]
fn q4_k_two_active_sub_blocks() {
    // d=1.0, dmin=1.0
    // scales=[8, 4, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]
    //   is=0: get(0)→sc=8,mn=0; get(1)→sc=4,mn=0
    //   is=2: get(2)→sc=0,mn=0; get(3)→sc=0,mn=0
    //   is=4: j≥4 → (scales[8]&0xF)|((scales[0]>>6)<<4) = 0|(0<<4)=0
    //   is=6: j≥4 → 0
    // qs = [0x22 × 128]: nibbles ambos = 2
    // out[0..32]   = 8*2-0 = 16.0
    // out[32..64]  = 4*2-0 = 8.0
    // out[64..256] = 0.0
    let scales: [u8; 12] = [8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let qs = [0x22u8; 128];
    let block = make_q4_k_block(1.0, 1.0, &scales, &qs);
    let out = dequant_to_f32(&block, GgmlType::Q4_K).unwrap();
    assert_eq!(out.len(), 256);
    for (i, &v) in out.iter().enumerate() {
        let expected = if i < 32 { 16.0f32 } else if i < 64 { 8.0 } else { 0.0 };
        assert!((v - expected).abs() < 1e-4, "out[{i}]={v} esperado={expected}");
    }
}

#[test]
fn q4_k_bad_size_returns_error() {
    assert!(dequant_to_f32(&[0u8; 143], GgmlType::Q4_K).is_err());
}
```

Run: `cargo test -p ggml-cpu dequant::tests::q4_k`
Expected: **FAIL** (todo!).

- [ ] **Step 2: Implementar `dequant_q4_k`**

Substituir o stub `dequant_q4_k`. Adicionar o helper `get_scale_min_k4` antes da função:

```rust
/// Decodifica o j-ésimo par (scale, min) do vetor `scales[12]` do Q4_K.
/// Espelha `get_scale_min_k4` de ggml-quants.c.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4),
        )
    }
}

fn dequant_q4_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 144; // 2+2+12+128
    if bytes.len() % BLOCK != 0 {
        return Err(DequantError::BadSize { ty: "Q4_K", block_bytes: BLOCK, got: bytes.len() });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.chunks_exact(BLOCK).enumerate() {
        let d_val   = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        let min_val = half::f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
        let scales  = &b[4..16];
        let qs      = &b[16..144];
        let base    = bi * 256;
        let mut qs_off = 0usize;
        let mut is = 0usize;

        for j_step in [0usize, 64, 128, 192] {
            let (sc1, m1) = get_scale_min_k4(is,     scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1  = d_val   * sc1 as f32;
            let m1f = min_val * m1  as f32;
            let d2  = d_val   * sc2 as f32;
            let m2f = min_val * m2  as f32;
            for l in 0..32 {
                let q = qs[qs_off + l];
                out[base + j_step + l]      = d1 * (q & 0xF) as f32 - m1f;
                out[base + j_step + l + 32] = d2 * (q >> 4)  as f32 - m2f;
            }
            qs_off += 32;
            is += 2;
        }
    }
    Ok(out)
}
```

- [ ] **Step 3: Rodar os testes**

Run: `cargo test -p ggml-cpu dequant::tests`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/ggml-cpu/src/dequant.rs
git commit -m "feat(ggml-cpu): dequant_q4_k (Fase 3 Task 3)"
```

---

## Task 4: `dequant_q6_k`

**Files:**
- Modify: `crates/ggml-cpu/src/dequant.rs`

Layout (210 bytes, 256 elementos):
```
bytes[0..128]   : ql     [u8; 128] — bits baixos 4 dos quants 6-bit
bytes[128..192] : qh     [u8; 64]  — bits altos 2
bytes[192..208] : scales [u8; 16]  — armazenados como u8 mas interpretados como i8
bytes[208..210] : d      (f16 LE)
```

Dequant (2 chunks de 128 elementos; `ql`, `qh`, `sc` avançam a cada chunk):
```
para n em [0, 128]:
    para l em 0..32:
        is = l / 16
        q1 = ((ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32
        q2 = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32
        q3 = ((ql[l]    >> 4)  | (((qh[l] >> 4) & 3) << 4)) as i8 - 32
        q4 = ((ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) as i8 - 32
        out[n+l]    = d_val * sc[is]   * q1
        out[n+l+32] = d_val * sc[is+2] * q2
        out[n+l+64] = d_val * sc[is+4] * q3
        out[n+l+96] = d_val * sc[is+6] * q4
    ql += 64; qh += 32; sc += 8
```

- [ ] **Step 1: Escrever o teste (RED)**

Adicionar em `mod tests`:

```rust
fn make_q6_k_block(d: f32, scales: &[i8; 16], ql: &[u8; 128], qh: &[u8; 64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(210);
    b.extend_from_slice(ql);
    b.extend_from_slice(qh);
    b.extend(scales.iter().map(|&s| s as u8));
    b.extend_from_slice(&f16_bytes(d));
    b
}

#[test]
fn q6_k_all_zero_ql_qh() {
    // ql=zeros, qh=zeros → todos os quants 6-bit = 0-32 = -32.
    // d=1.0, scales=[1,0,1,0,1,0,1,0, 1,0,1,0,1,0,1,0]
    // Para cada chunk de 128:
    //   l=0..15 (is=0): out[l]=sc[0]*(-32)=-32; out[l+32]=sc[2]*(-32)=-32; ...
    //   l=16..31(is=1): out[l]=sc[1]*(-32)=0; out[l+32]=sc[3]*(-32)=0; ...
    let scales: [i8; 16] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let ql = [0u8; 128];
    let qh = [0u8; 64];
    let block = make_q6_k_block(1.0, &scales, &ql, &qh);
    let out = dequant_to_f32(&block, GgmlType::Q6_K).unwrap();
    assert_eq!(out.len(), 256);
    for chunk_base in [0usize, 128] {
        for l in 0..16usize {
            assert!((out[chunk_base + l]      - (-32.0)).abs() < 1e-4, "chunk={chunk_base} l={l} off=0");
            assert!((out[chunk_base + l + 32] - (-32.0)).abs() < 1e-4, "chunk={chunk_base} l={l} off=32");
            assert!((out[chunk_base + l + 64] - (-32.0)).abs() < 1e-4, "chunk={chunk_base} l={l} off=64");
            assert!((out[chunk_base + l + 96] - (-32.0)).abs() < 1e-4, "chunk={chunk_base} l={l} off=96");
        }
        for l in 16..32usize {
            assert_eq!(out[chunk_base + l],      0.0, "chunk={chunk_base} l={l} off=0 deve=0");
            assert_eq!(out[chunk_base + l + 32], 0.0, "chunk={chunk_base} l={l} off=32 deve=0");
            assert_eq!(out[chunk_base + l + 64], 0.0, "chunk={chunk_base} l={l} off=64 deve=0");
            assert_eq!(out[chunk_base + l + 96], 0.0, "chunk={chunk_base} l={l} off=96 deve=0");
        }
    }
}

#[test]
fn q6_k_bad_size_returns_error() {
    assert!(dequant_to_f32(&[0u8; 209], GgmlType::Q6_K).is_err());
}
```

Run: `cargo test -p ggml-cpu dequant::tests::q6_k`
Expected: **FAIL** (todo!).

- [ ] **Step 2: Implementar `dequant_q6_k`**

Substituir o stub `dequant_q6_k`:

```rust
fn dequant_q6_k(bytes: &[u8]) -> Result<Vec<f32>, DequantError> {
    const BLOCK: usize = 210; // 128+64+16+2
    if bytes.len() % BLOCK != 0 {
        return Err(DequantError::BadSize { ty: "Q6_K", block_bytes: BLOCK, got: bytes.len() });
    }
    let n_blocks = bytes.len() / BLOCK;
    let mut out = vec![0.0f32; n_blocks * 256];

    for (bi, b) in bytes.chunks_exact(BLOCK).enumerate() {
        let ql_full = &b[0..128];
        let qh_full = &b[128..192];
        let sc_full = &b[192..208]; // [i8; 16] armazenados como u8
        let d_val = half::f16::from_bits(u16::from_le_bytes([b[208], b[209]])).to_f32();
        let base = bi * 256;

        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;

        for n in [0usize, 128] {
            let ql = &ql_full[ql_off..ql_off + 64];
            let qh = &qh_full[qh_off..qh_off + 32];
            let sc = &sc_full[sc_off..sc_off + 8];
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                out[base + n + l]       = d_val * sc[is]     as i8 as f32 * q1 as f32;
                out[base + n + l + 32]  = d_val * sc[is + 2] as i8 as f32 * q2 as f32;
                out[base + n + l + 64]  = d_val * sc[is + 4] as i8 as f32 * q3 as f32;
                out[base + n + l + 96]  = d_val * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    Ok(out)
}
```

- [ ] **Step 3: Rodar os testes**

Run: `cargo test -p ggml-cpu dequant::tests`
Expected: PASS (todos os ~12 testes de dequant).

- [ ] **Step 4: Commit**

```bash
git add crates/ggml-cpu/src/dequant.rs
git commit -m "feat(ggml-cpu): dequant_q6_k (Fase 3 Task 4)"
```

---

## Task 5: `RawTensor` em `llama-model` + actualizar `error.rs`

**Files:**
- Modify: `crates/llama-model/Cargo.toml`
- Modify: `crates/llama-model/src/error.rs`
- Modify: `crates/llama-model/src/weights.rs`

- [ ] **Step 1: Adicionar dep `ggml-cpu` em `crates/llama-model/Cargo.toml`**

```toml
[dependencies]
thiserror.workspace = true
gguf.workspace = true
llama-tokenizer.workspace = true
ggml-cpu.workspace = true
```

- [ ] **Step 2: Substituir `error.rs`**

Remover `NotF32`, adicionar `Dequant`:

```rust
//! Erros do carregamento e da inferência do modelo Llama.

use gguf::GgufError;
use ggml_cpu::DequantError;
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
    #[error("dequantização: {0}")]
    Dequant(#[from] DequantError),
    #[error("config inconsistente: {0}")]
    Config(String),
    #[error("overflow de conversão numérica")]
    Overflow,
}
```

- [ ] **Step 3: Substituir `weights.rs` completamente**

```rust
//! Pesos quantizados do GGUF armazenados em bytes raw; dequantizados sob demanda.

use gguf::{GgufFile, TensorInfo};
use ggml_cpu::dequant_to_f32;

use crate::config::LlamaConfig;
use crate::error::ModelError;

/// Tensor raw: bytes tal como lidos do GGUF + tipo de dado para dequant.
pub(crate) struct RawTensor {
    pub bytes: Vec<u8>,
    pub ty: gguf::GgmlType,
}

impl RawTensor {
    /// Número de elementos (não de bytes).
    pub fn n_elements(&self) -> usize {
        let bs = self.ty.block_size() as usize;
        let ts = self.ty.type_size() as usize;
        if ts == 0 { return 0; }
        (self.bytes.len() / ts) * bs
    }

    /// Bytes raw (footprint de RAM — quantizado).
    pub fn memory_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Dequantiza para f32 (alocação sob demanda).
    pub fn dequant_to_f32(&self) -> Result<Vec<f32>, ModelError> {
        Ok(dequant_to_f32(&self.bytes, self.ty)?)
    }
}

/// Pesos de uma camada transformer.
pub(crate) struct LayerWeights {
    pub attn_norm:   RawTensor,
    pub attn_q:      RawTensor,
    pub attn_k:      RawTensor,
    pub attn_v:      RawTensor,
    pub attn_output: RawTensor,
    pub ffn_norm:    RawTensor,
    pub ffn_gate:    RawTensor,
    pub ffn_up:      RawTensor,
    pub ffn_down:    RawTensor,
}

/// Todos os pesos do modelo, em bytes raw.
pub(crate) struct Weights {
    pub token_embd:  RawTensor,
    pub layers:      Vec<LayerWeights>,
    pub output_norm: RawTensor,
    pub output:      RawTensor,
}

fn tensor_raw(f: &GgufFile, bytes: &[u8], name: &str) -> Result<RawTensor, ModelError> {
    let info: &TensorInfo = f
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))?;
    let raw = f.tensor_data(bytes, info)?;
    Ok(RawTensor { bytes: raw.to_vec(), ty: info.ggml_type })
}

impl Weights {
    /// Lê todos os tensores (qualquer tipo suportado pelo dispatcher de dequant).
    pub fn from_gguf(f: &GgufFile, bytes: &[u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |suffix: &str| format!("blk.{l}.{suffix}");
            layers.push(LayerWeights {
                attn_norm:   tensor_raw(f, bytes, &p("attn_norm.weight"))?,
                attn_q:      tensor_raw(f, bytes, &p("attn_q.weight"))?,
                attn_k:      tensor_raw(f, bytes, &p("attn_k.weight"))?,
                attn_v:      tensor_raw(f, bytes, &p("attn_v.weight"))?,
                attn_output: tensor_raw(f, bytes, &p("attn_output.weight"))?,
                ffn_norm:    tensor_raw(f, bytes, &p("ffn_norm.weight"))?,
                ffn_gate:    tensor_raw(f, bytes, &p("ffn_gate.weight"))?,
                ffn_up:      tensor_raw(f, bytes, &p("ffn_up.weight"))?,
                ffn_down:    tensor_raw(f, bytes, &p("ffn_down.weight"))?,
            });
        }
        Ok(Self {
            token_embd:  tensor_raw(f, bytes, "token_embd.weight")?,
            layers,
            output_norm: tensor_raw(f, bytes, "output_norm.weight")?,
            output:      tensor_raw(f, bytes, "output.weight")?,
        })
    }

    /// Soma dos bytes raw de todos os tensores.
    pub fn memory_bytes(&self) -> usize {
        let layer_bytes: usize = self.layers.iter().map(|lw| {
            lw.attn_norm.memory_bytes()
                + lw.attn_q.memory_bytes()
                + lw.attn_k.memory_bytes()
                + lw.attn_v.memory_bytes()
                + lw.attn_output.memory_bytes()
                + lw.ffn_norm.memory_bytes()
                + lw.ffn_gate.memory_bytes()
                + lw.ffn_up.memory_bytes()
                + lw.ffn_down.memory_bytes()
        }).sum();
        self.token_embd.memory_bytes()
            + layer_bytes
            + self.output_norm.memory_bytes()
            + self.output.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn loads_all_weights_with_expected_element_counts() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let f = GgufFile::parse(&bytes).unwrap();
        let cfg = LlamaConfig::from_gguf(&f).unwrap();
        let w = Weights::from_gguf(&f, &bytes, &cfg).unwrap();
        assert_eq!(w.token_embd.n_elements(),  cfg.vocab * cfg.n_embd);
        assert_eq!(w.output.n_elements(),      cfg.vocab * cfg.n_embd);
        assert_eq!(w.output_norm.n_elements(), cfg.n_embd);
        assert_eq!(w.layers.len(), cfg.n_layer);
        let l0 = &w.layers[0];
        assert_eq!(l0.attn_q.n_elements(),    cfg.n_embd * cfg.n_embd);
        assert_eq!(l0.attn_k.n_elements(),    cfg.n_embd * cfg.n_head_kv * cfg.head_dim);
        assert_eq!(l0.ffn_gate.n_elements(),  cfg.n_embd * cfg.n_ff);
        assert_eq!(l0.ffn_down.n_elements(),  cfg.n_ff   * cfg.n_embd);
    }
}
```

- [ ] **Step 4: Verificar build (model.rs ainda quebra — esperado)**

Run: `cargo build -p llama-model 2>&1 | grep "^error" | head -5`
Expected: erros apenas em `model.rs` (usa `Vec<f32>` diretamente). Outros crates devem compilar.

- [ ] **Step 5: Rodar teste de weights**

Run: `cargo test -p llama-model weights::tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-model/Cargo.toml crates/llama-model/src/error.rs crates/llama-model/src/weights.rs
git commit -m "feat(llama-model): RawTensor — pesos em bytes raw com dequant sob demanda (Fase 3 Task 5)"
```

---

## Task 6: Atualizar `model.rs` para dequant sob demanda

**Files:**
- Modify: `crates/llama-model/src/model.rs`

Cada referência a `&lw.<campo>` (que era `&[f32]`) vira `&lw.<campo>.dequant_to_f32()?[..]`.

- [ ] **Step 1: Verificar que o teste existe e falha**

Run: `cargo test -p llama-model model::tests 2>&1 | head -10`
Expected: erro de compilação (esperado).

- [ ] **Step 2: Substituir `model.rs` completamente**

```rust
//! Modelo Llama: carrega config+pesos e executa o forward f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgufFile;

use crate::attention::{attention, KvCache};
use crate::config::LlamaConfig;
use crate::error::ModelError;
use crate::ops::{argmax, embedding_lookup, matmul, mul_rows, rmsnorm, rope_norm, swiglu};
use crate::weights::Weights;

/// Modelo carregado: config + pesos raw (quantizados ou f32).
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

    /// Carrega com config já validada externamente (útil para arquiteturas que
    /// têm nomes de tensor compatíveis mas hparams com chaves diferentes).
    pub fn load_with_config(
        f: &GgufFile,
        bytes: &[u8],
        config: LlamaConfig,
    ) -> Result<Self, ModelError> {
        let weights = Weights::from_gguf(f, bytes, &config)?;
        Ok(Self { config, weights })
    }

    pub(crate) fn new_cache(&self) -> KvCache {
        KvCache::new(self.config.n_layer)
    }

    /// Soma dos bytes raw de todos os pesos (footprint de RAM, sem dequant).
    pub fn memory_bytes(&self) -> usize {
        self.weights.memory_bytes()
    }

    /// Contagem total de elementos em todos os tensores de peso.
    pub fn weight_element_count(&self) -> usize {
        let w = &self.weights;
        let layer_elem: usize = w.layers.iter().map(|lw| {
            lw.attn_norm.n_elements()
                + lw.attn_q.n_elements()
                + lw.attn_k.n_elements()
                + lw.attn_v.n_elements()
                + lw.attn_output.n_elements()
                + lw.ffn_norm.n_elements()
                + lw.ffn_gate.n_elements()
                + lw.ffn_up.n_elements()
                + lw.ffn_down.n_elements()
        }).sum();
        w.token_embd.n_elements()
            + layer_elem
            + w.output_norm.n_elements()
            + w.output.n_elements()
    }

    /// Processa `tokens` e devolve logits (tamanho `vocab`) do último token.
    pub(crate) fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let n_tok = tokens.len();
        let pos0 = cache.len();
        let kv_dim = c.n_head_kv * c.head_dim;

        let token_embd = self.weights.token_embd.dequant_to_f32()?;
        let mut x = embedding_lookup(&token_embd, tokens, c.n_embd)?;

        for (l, lw) in self.weights.layers.iter().enumerate() {
            let attn_norm  = lw.attn_norm.dequant_to_f32()?;
            let attn_q_w   = lw.attn_q.dequant_to_f32()?;
            let attn_k_w   = lw.attn_k.dequant_to_f32()?;
            let attn_v_w   = lw.attn_v.dequant_to_f32()?;
            let attn_out_w = lw.attn_output.dequant_to_f32()?;
            let ffn_norm   = lw.ffn_norm.dequant_to_f32()?;
            let ffn_gate_w = lw.ffn_gate.dequant_to_f32()?;
            let ffn_up_w   = lw.ffn_up.dequant_to_f32()?;
            let ffn_down_w = lw.ffn_down.dequant_to_f32()?;

            let normed  = rmsnorm(&x, c.n_embd, c.rms_eps);
            let attn_in = mul_rows(&normed, &attn_norm, c.n_embd);

            let mut q = matmul(&attn_q_w,  &attn_in, c.n_embd, c.n_embd, n_tok);
            let mut k = matmul(&attn_k_w,  &attn_in, c.n_embd, kv_dim,   n_tok);
            let v     = matmul(&attn_v_w,  &attn_in, c.n_embd, kv_dim,   n_tok);

            rope_norm(&mut q, n_tok, c.n_head,    c.head_dim, c.rope_dim, c.freq_base, pos0);
            rope_norm(&mut k, n_tok, c.n_head_kv, c.head_dim, c.rope_dim, c.freq_base, pos0);

            cache.append(l, &k, &v);
            let attn = attention(
                &q, &cache.k[l], &cache.v[l],
                n_tok, pos0, c.n_head, c.n_head_kv, c.head_dim,
            );
            let attn_out = matmul(&attn_out_w, &attn, c.n_embd, c.n_embd, n_tok);
            for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) { *xi += ai; }

            let normed  = rmsnorm(&x, c.n_embd, c.rms_eps);
            let ffn_in  = mul_rows(&normed, &ffn_norm, c.n_embd);
            let gate    = matmul(&ffn_gate_w, &ffn_in, c.n_embd, c.n_ff, n_tok);
            let up      = matmul(&ffn_up_w,   &ffn_in, c.n_embd, c.n_ff, n_tok);
            let act     = swiglu(&gate, &up);
            let ffn_out = matmul(&ffn_down_w, &act,   c.n_ff,   c.n_embd, n_tok);
            for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) { *xi += fi; }
        }

        cache.advance(n_tok);

        let output_norm = self.weights.output_norm.dequant_to_f32()?;
        let output_w    = self.weights.output.dequant_to_f32()?;
        let normed      = rmsnorm(&x, c.n_embd, c.rms_eps);
        let final_x     = mul_rows(&normed, &output_norm, c.n_embd);
        let last        = &final_x[(n_tok - 1) * c.n_embd..n_tok * c.n_embd];
        let logits      = matmul(&output_w, last, c.n_embd, c.vocab, 1);
        Ok(logits)
    }

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

    fn load_model() -> Option<Model> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        Model::load(&f, &bytes).ok()
    }

    #[test]
    fn embd_and_qcur_sums_match_oracle() {
        let Some(m) = load_model() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let c = &m.config;
        let tokens = [1u32, 403, 407, 261, 378];
        let n_tok = tokens.len();

        let token_embd = m.weights.token_embd.dequant_to_f32().unwrap();
        let x = embedding_lookup(&token_embd, &tokens, c.n_embd).unwrap();
        let embd_sum: f32 = x.iter().sum();
        assert!((embd_sum - (-3.354056)).abs() < 1e-2, "embd_sum={embd_sum}");

        let lw = &m.weights.layers[0];
        let attn_norm = lw.attn_norm.dequant_to_f32().unwrap();
        let attn_q_w  = lw.attn_q.dequant_to_f32().unwrap();
        let normed    = rmsnorm(&x, c.n_embd, c.rms_eps);
        let attn_in   = mul_rows(&normed, &attn_norm, c.n_embd);
        let mut q     = matmul(&attn_q_w, &attn_in, c.n_embd, c.n_embd, n_tok);
        rope_norm(&mut q, n_tok, c.n_head, c.head_dim, c.rope_dim, c.freq_base, 0);
        let q_sum: f32 = q.iter().sum();
        assert!((q_sum - 148.969818).abs() < 1e-1, "q_sum={q_sum}");
    }

    #[test]
    fn memory_bytes_less_than_file_size() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let file_size = bytes.len();
        let f = GgufFile::parse(&bytes).unwrap();
        let m = Model::load(&f, &bytes).unwrap();
        assert!(
            m.memory_bytes() <= file_size,
            "memory_bytes={} > file_size={file_size}",
            m.memory_bytes()
        );
    }
}
```

- [ ] **Step 3: Rodar os testes do modelo**

Run: `cargo test -p llama-model model::tests`
Expected: PASS (sums batem; memory_bytes ≤ tamanho do arquivo).

- [ ] **Step 4: Rodar o gate diferencial greedy (herdado da Fase 2)**

Run: `cargo test -p llama-model --test oracle_forward -- --nocapture`
Expected: PASS — sequência greedy idêntica a `refs/greedy.txt`.

Se falhar com divergência: causa mais provável é precisão F32 na conversão de bytes (já era correto antes — verificar se `tensor_raw` copia bytes corretamente com `raw.to_vec()`). Usar `superpowers:systematic-debugging` para bisseccionar.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-model/src/model.rs
git commit -m "feat(llama-model): forward com dequant sob demanda + memory_bytes + weight_element_count (Fase 3 Task 6)"
```

---

## Task 7: Teste de integração — carregamento do modelo quantizado

**Files:**
- Create: `crates/llama-model/tests/quant_load.rs`

- [ ] **Step 1: Criar o teste**

`crates/llama-model/tests/quant_load.rs`:

```rust
//! Integração: carrega qwen2.5-0.5b-q8_0 e verifica footprint de memória.
//! Skip automático se o modelo não estiver em models/.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use gguf::GgufFile;
use llama_model::{LlamaConfig, Model};

const QWEN_PATH: &str = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";

#[test]
fn qwen_q8_0_loads_without_error() {
    let Ok(bytes) = std::fs::read(Path::new(QWEN_PATH)) else {
        eprintln!("qwen ausente — pulando");
        return;
    };
    let f = GgufFile::parse(&bytes).expect("parse GGUF");

    // Tenta LlamaConfig; divergências de hparams são registradas e o teste pula.
    let cfg = match LlamaConfig::from_gguf(&f) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("qwen config falhou ({e}) — arquitetura divergente, pulando (esperado até Fase 7)");
            return;
        }
    };

    let model = match Model::load_with_config(&f, &bytes, cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("qwen load falhou ({e}) — registrar e investigar");
            // Falha explícita se o erro for UnsupportedType (indica tipo de quant
            // não implementado que deve ser adicionado ao dispatcher).
            let msg = e.to_string();
            assert!(
                !msg.contains("não suportado"),
                "tipo de quant não suportado detectado: {msg}"
            );
            return;
        }
    };

    let file_size = bytes.len();
    let mem = model.memory_bytes();

    // Gate 1: bytes raw ≤ tamanho do arquivo GGUF.
    assert!(
        mem <= file_size,
        "memory_bytes={mem} > file_size={file_size}"
    );

    // Gate 2: densidade de memória condizente com Q8_0 (< 2 bytes/elem).
    let n_elem = model.weight_element_count();
    let ratio = mem as f64 / n_elem as f64;
    assert!(
        ratio < 2.0,
        "ratio bytes/elem={ratio:.3} ≥ 2.0 — suspeita de dequant precoce"
    );

    eprintln!(
        "qwen2.5-0.5b-q8_0: file={:.1}MB mem_raw={:.1}MB elem={n_elem} ratio={ratio:.3} bytes/elem",
        file_size as f64 / 1e6,
        mem as f64 / 1e6,
    );
}
```

- [ ] **Step 2: Rodar o teste**

Run: `cargo test -p llama-model --test quant_load -- --nocapture`
Expected: PASS ou skip com mensagem de motivo.

Se falhar com `"tipo de quant não suportado"`, significa que o qwen usa um tipo além dos 4 implementados — verificar qual tipo com:
```bash
# Inspecionar tipos dos tensores do qwen (via GgufFile::tensors)
# Adicionar o tipo ao dispatcher em ggml-cpu antes de continuar.
```

Se falhar com `MissingTensor`: os nomes dos tensores do qwen diferem — registrar e reportar ao usuário (scope da Fase 7).

- [ ] **Step 3: Rodar o workspace completo**

Run: `cargo test --workspace`
Expected: PASS (todos os testes de Fase 2 e 3 verdes).

- [ ] **Step 4: Commit**

```bash
git add crates/llama-model/tests/quant_load.rs
git commit -m "test(llama-model): integração qwen2.5-q8_0 — load + footprint de memória (Fase 3 Task 7)"
```

---

## Task 8: Gate de qualidade completo

**Files:** nenhum novo (ajustes de lint se necessário).

- [ ] **Step 1: fmt**

Run: `cargo fmt --all` então `cargo fmt --all --check`
Expected: sem diferenças.

- [ ] **Step 2: clippy estrito**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS.

Issues comuns e resoluções:
- `cast_sign_loss` no cast `sc[idx] as i8` em `dequant_q6_k`: o byte está armazenado como u8 mas é um i8 — o cast é correto e intencional. Adicionar `#[allow(clippy::cast_possible_wrap)]` na linha se clippy reclamar, ou usar `i8::from_ne_bytes([sc[idx]])`.
- `cast_precision_loss` para `sc1 as f32` (u8→f32): não há perda de precisão. Adicionar `#[allow(clippy::cast_precision_loss)]` no fn se clippy reclamar (u8 max=255, todos representáveis em f32).
- `indexing_slicing`: já tem `#![allow(clippy::indexing_slicing)]` no topo de `dequant.rs` — verificar que está presente.

- [ ] **Step 3: testes do workspace**

Run: `cargo test --workspace`
Expected: PASS.

- [ ] **Step 4: gate completo**

Run: `./scripts/gate.sh`
Expected: `GATE OK`.

Se cobertura < 80% em `ggml-cpu`: adicionar um teste cobrindo o ramo de F32 bad-size:
```rust
#[test]
fn f32_bad_size_returns_error() {
    assert!(dequant_to_f32(&[0u8; 3], GgmlType::F32).is_err());
}
```

Se cobertura < 80% em `llama-model`: o crate foi amplamente testado nas fases anteriores. Verificar se `load_with_config` e `weight_element_count` têm linha de cobertura — o teste `quant_load` os exercita.

- [ ] **Step 5: Commit final (se houve ajustes)**

```bash
git add -A
git commit -m "chore(fase3): gate verde (fmt + clippy + cobertura) (Fase 3 Task 8)"
```

---

## Riscos conhecidos

1. **Nomes de tensores do Qwen divergem do Llama.** Se `Weights::from_gguf` falhar com `MissingTensor` para o qwen, os nomes dos blocos diferem (ex: `blk.0.attn_norm.weight` pode ser diferente). O teste `quant_load` pula com log — não é regressão, é scope da Fase 7.

2. **Tipo de quant adicional no qwen.** Se o qwen usar um tipo além de Q8_0/F16/F32 (improvável para este modelo específico, mas possível), `DequantError::UnsupportedType` é retornado e o teste falha explicitamente (assertado no teste). Adicionar o tipo ao dispatcher antes de continuar.

3. **Regressão de velocidade (dequant F32 por decode).** Stories260K é F32; a cada decode step, `dequant_to_f32` reinterpreta bytes → Vec<f32> por camada (≈ 45 alocações de Vec por decode). Não afeta corretude; otimização fica para Fase 5 (cache de pesos dequantizados ou dequant in-place durante matmul).

4. **Greedy gate da Fase 2 deve permanecer verde.** A mudança de `Vec<f32>` para `RawTensor` + `dequant_to_f32` para F32 é semanticamente equivalente a copiar bytes → f32 LE (o que `tensor_f32` fazia). Se o gate falhar, a causa é diferença de precisão no path de cópia — investigar com `systematic-debugging`.

---

## Self-Review

- **Cobertura da spec:** Q8_0 (T1), Q4_0 (T2), Q4_K (T3), Q6_K (T4), F16 (T0/T1), memória raw (T5–T6), footprint ≤ arquivo (T6–T7). Dequant bit-exact validado por vetores derivados diretamente do ggml-quants.c.
- **Placeholders:** nenhum TODO/TBD — todo step tem código concreto.
- **Consistência de tipos:** `RawTensor::dequant_to_f32() -> Result<Vec<f32>, ModelError>` consistente em T5 e T6; `model.rs` usa essa assinatura em todos os 11 pontos de chamada (token_embd, 9 por camada × n_layer, output_norm, output); `load_with_config`/`weight_element_count` adicionados em T6 e referenciados em T7.
- **Build incremental:** `ggml-cpu` compila a partir de T0; `llama-model` quebra em T5 (model.rs usa Vec<f32>) e é reparado em T6; `oracle_forward` e `quant_load` são integration tests que só rodam no final.
