# Fase 1 — Parser GGUF + Tokenizer SPM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar os crates `gguf` (parser GGUF v3 zero-unsafe sobre `&[u8]`) e `llama-tokenizer` (SPM/Llama), validados bit-exact contra o oráculo C++.

**Architecture:** `gguf` parseia o conteúdo completo do arquivo como slice emprestado, expondo metadados (`BTreeMap`), tensor infos e acesso raw aos bytes de cada tensor — sem mmap (adiado p/ Fase 2) e sem dequant (Fase 3). `llama-tokenizer` lê o vocab dos metadados GGUF e implementa o algoritmo SPM (merge-by-score + byte-fallback) como réplica fiel do `llm_tokenizer_spm` do upstream.

**Tech Stack:** Rust edition 2024, `thiserror`, `proptest` (dev), `cargo-llvm-cov`. Ambos os crates `#![forbid(unsafe_code)]`.

**Spec:** `docs/superpowers/specs/2026-06-03-fase-1-gguf-tokenizer-design.md`
**Branch:** `fase-1-gguf-tokenizer`

**Fonte da verdade (upstream, somente leitura):**
- GGUF value types (`enum gguf_type`): `ggml/include/gguf.h:53` → UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12.
- ggml type ids: `ggml/include/ggml.h:389` (F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9, Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, BF16=30, I8=24, I16=25, I32=26, I64=27, F64=28).
- SPM: `src/llama-vocab.cpp:96-232` (algoritmo), `:3806` (`byte_to_token`), `:3239` (`llama_escape_whitespace`), `:3290-3318` (prefixo de espaço).

**Convenção de commit:** `feat:`/`test:`/`chore:` ; sem trailer de atribuição (desabilitado globalmente).

**Gotcha de ambiente:** GateGuard (hook ECC) pode exigir "fatos" antes do 1º Bash e antes de Edit/Write — apresentar e repetir a operação. Hook PostToolUse roda `rustfmt --edition 2024` em `.rs` automaticamente.

---

## Task 0: Scaffolding do workspace

**Files:**
- Modify: `Cargo.toml` (workspace members + deps)
- Create: `crates/gguf/Cargo.toml`, `crates/gguf/src/lib.rs`
- Create: `crates/llama-tokenizer/Cargo.toml`, `crates/llama-tokenizer/src/lib.rs`

- [ ] **Step 1: Adicionar membros e deps ao workspace**

Editar `Cargo.toml` (raiz):

```toml
[workspace]
resolver = "3"
members = ["oracle", "crates/gguf", "crates/llama-tokenizer"]

[workspace.package]
edition = "2024"

[workspace.dependencies]
thiserror = "2"
serde_json = "1"
proptest = "1"
gguf = { path = "crates/gguf" }
```

(Manter o bloco `[workspace.lints]` existente inalterado.)

- [ ] **Step 2: Criar `crates/gguf/Cargo.toml`**

```toml
[package]
name = "gguf"
version = "0.1.0"
edition.workspace = true

[dependencies]
thiserror.workspace = true

[dev-dependencies]
proptest.workspace = true
serde_json.workspace = true

[lints]
workspace = true
```

- [ ] **Step 3: Criar `crates/gguf/src/lib.rs` (stub)**

```rust
#![forbid(unsafe_code)]
//! Parser do formato GGUF v3 (little-endian) sobre slice emprestado.
```

- [ ] **Step 4: Criar `crates/llama-tokenizer/Cargo.toml`**

```toml
[package]
name = "llama-tokenizer"
version = "0.1.0"
edition.workspace = true

[dependencies]
thiserror.workspace = true
gguf.workspace = true

[dev-dependencies]
serde_json.workspace = true

[lints]
workspace = true
```

- [ ] **Step 5: Criar `crates/llama-tokenizer/src/lib.rs` (stub)**

```rust
#![forbid(unsafe_code)]
//! Tokenizer SPM (Llama) — encode/decode bit-exact vs llama.cpp.
```

- [ ] **Step 6: Verificar build do workspace**

Run: `cargo build --workspace`
Expected: compila sem erros (3 crates).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock crates/
git commit -m "chore: scaffolding dos crates gguf e llama-tokenizer"
```

---

## Task 1: `gguf` — erros + reader com bounds-check

**Files:**
- Create: `crates/gguf/src/error.rs`
- Create: `crates/gguf/src/reader.rs`
- Modify: `crates/gguf/src/lib.rs`

- [ ] **Step 1: Escrever o teste que falha** (`crates/gguf/src/reader.rs`, ao final)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_scalars_little_endian() {
        let bytes = [0x01, 0x02, 0x03, 0x04, b'h', b'i'];
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u32().unwrap(), 0x0403_0201);
        assert_eq!(r.read_bytes(2).unwrap(), b"hi");
    }

    #[test]
    fn out_of_bounds_is_error_not_panic() {
        let bytes = [0x00, 0x01];
        let mut r = Reader::new(&bytes);
        assert!(r.u32().is_err());
    }

    #[test]
    fn gguf_string_roundtrip() {
        // u64 len = 3, "abc"
        let mut bytes = vec![3, 0, 0, 0, 0, 0, 0, 0];
        bytes.extend_from_slice(b"abc");
        let mut r = Reader::new(&bytes);
        assert_eq!(r.gguf_string().unwrap(), "abc");
    }

    #[test]
    fn string_length_overflow_is_error() {
        // len = u64::MAX → não pode ler, erro (sem alocar)
        let bytes = [0xFF; 8];
        let mut r = Reader::new(&bytes);
        assert!(r.gguf_string().is_err());
    }
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p gguf reader`
Expected: FAIL (`Reader` não existe).

- [ ] **Step 3: Implementar `crates/gguf/src/error.rs`**

```rust
//! Erros tipados do parser GGUF.

/// Falhas ao parsear um arquivo GGUF. GGUF é entrada não-confiável:
/// toda condição inesperada vira erro, nunca panic.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GgufError {
    #[error("fim inesperado dos dados: precisava de {needed} bytes em offset {offset}, restam {available}")]
    UnexpectedEof {
        offset: usize,
        needed: usize,
        available: usize,
    },
    #[error("magic GGUF inválido: {0:?}")]
    BadMagic([u8; 4]),
    #[error("versão GGUF não suportada: {0} (suportado: 3)")]
    UnsupportedVersion(u32),
    #[error("string não-UTF8 nos metadados")]
    InvalidUtf8,
    #[error("tipo de valor de metadado desconhecido: {0}")]
    UnknownValueType(u32),
    #[error("array aninhado não suportado")]
    NestedArray,
    #[error("tipo de tensor ggml desconhecido: {0}")]
    UnknownTensorType(u32),
    #[error("chave de metadado ausente: {0}")]
    MissingKey(String),
    #[error("tipo de metadado incorreto para a chave {key}: esperado {expected}")]
    WrongType { key: String, expected: &'static str },
    #[error("dados do tensor fora dos limites: tensor {name}")]
    TensorOutOfBounds { name: String },
    #[error("overflow aritmético ao calcular tamanho/alinhamento")]
    Overflow,
}
```

- [ ] **Step 4: Implementar `crates/gguf/src/reader.rs`**

```rust
//! Cursor com bounds-check sobre um slice `&[u8]`. Todo método retorna
//! `Result` em vez de panicar — GGUF é entrada não-confiável.

use crate::error::GgufError;

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Overflow)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(GgufError::UnexpectedEof {
                offset: self.pos,
                needed: n,
                available: self.bytes.len().saturating_sub(self.pos),
            })?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let slice = self.read_bytes(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.array::<1>()?[0])
    }
    pub(crate) fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(i8::from_le_bytes(self.array()?))
    }
    pub(crate) fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    pub(crate) fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.array()?))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    pub(crate) fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.array()?))
    }
    pub(crate) fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.array()?))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    pub(crate) fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.array()?))
    }
    pub(crate) fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.array()?))
    }
    pub(crate) fn bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.u8()? != 0)
    }

    /// String GGUF: `u64` de comprimento + bytes UTF-8.
    pub(crate) fn gguf_string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::Overflow)?;
        let bytes = self.read_bytes(len)?;
        core::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| GgufError::InvalidUtf8)
    }
}
```

- [ ] **Step 5: Declarar módulos em `crates/gguf/src/lib.rs`**

```rust
#![forbid(unsafe_code)]
//! Parser do formato GGUF v3 (little-endian) sobre slice emprestado.

mod error;
mod reader;

pub use error::GgufError;
```

- [ ] **Step 6: Rodar e ver passar**

Run: `cargo test -p gguf reader`
Expected: PASS (4 testes).

- [ ] **Step 7: Commit**

```bash
git add crates/gguf/src/
git commit -m "feat(gguf): erros tipados + reader com bounds-check"
```

---

## Task 2: `gguf` — `GgmlType` + tabela de blocos

**Files:**
- Create: `crates/gguf/src/types.rs`
- Modify: `crates/gguf/src/lib.rs`

- [ ] **Step 1: Escrever o teste que falha** (`crates/gguf/src/types.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_type_ids_map() {
        assert_eq!(GgmlType::try_from(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::try_from(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::try_from(14).unwrap(), GgmlType::Q6_K);
    }

    #[test]
    fn unknown_type_id_is_error() {
        assert!(GgmlType::try_from(9999).is_err());
    }

    #[test]
    fn block_layout_matches_ggml() {
        assert_eq!((GgmlType::F32.block_size(), GgmlType::F32.type_size()), (1, 4));
        assert_eq!((GgmlType::F16.block_size(), GgmlType::F16.type_size()), (1, 2));
        assert_eq!((GgmlType::Q8_0.block_size(), GgmlType::Q8_0.type_size()), (32, 34));
        assert_eq!((GgmlType::Q4_0.block_size(), GgmlType::Q4_0.type_size()), (32, 18));
        assert_eq!((GgmlType::Q4_K.block_size(), GgmlType::Q4_K.type_size()), (256, 144));
        assert_eq!((GgmlType::Q6_K.block_size(), GgmlType::Q6_K.type_size()), (256, 210));
    }
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p gguf types`
Expected: FAIL (`GgmlType` não existe).

- [ ] **Step 3: Implementar `crates/gguf/src/types.rs`** (parte 1: `GgmlType`)

> Constantes `block_size`/`type_size` conferidas contra `ggml/src/ggml.c`
> (`type_traits`): K-quants têm `block_size` 256; type_size = `sizeof(block_*)`.

```rust
//! Tipos de tensor ggml e valores de metadados GGUF.

use crate::error::GgufError;

/// Tipo de dado de um tensor (subconjunto ativo de `enum ggml_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    F64,
}

impl GgmlType {
    /// Elementos por bloco.
    pub fn block_size(self) -> u64 {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1 => 32,
            GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K => 256,
        }
    }

    /// Bytes por bloco.
    pub fn type_size(self) -> u64 {
        match self {
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 | GgmlType::Bf16 | GgmlType::I16 => 2,
            GgmlType::I8 => 1,
            GgmlType::I64 | GgmlType::F64 => 8,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q8_1 => 36,
            GgmlType::Q2_K => 84,
            GgmlType::Q3_K => 110,
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            GgmlType::Q8_K => 292,
        }
    }
}

impl TryFrom<u32> for GgmlType {
    type Error = GgufError;
    fn try_from(id: u32) -> Result<Self, Self::Error> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            30 => GgmlType::Bf16,
            other => return Err(GgufError::UnknownTensorType(other)),
        })
    }
}
```

- [ ] **Step 4: Declarar `mod types;` em `lib.rs`**

Adicionar a `crates/gguf/src/lib.rs`:

```rust
mod types;

pub use types::GgmlType;
```

- [ ] **Step 5: Rodar e ver passar**

Run: `cargo test -p gguf types`
Expected: PASS (3 testes).

- [ ] **Step 6: Commit**

```bash
git add crates/gguf/src/
git commit -m "feat(gguf): GgmlType com tabela de block_size/type_size"
```

---

## Task 3: `gguf` — `MetadataValue` + acessores

**Files:**
- Modify: `crates/gguf/src/types.rs`
- Modify: `crates/gguf/src/lib.rs`

- [ ] **Step 1: Escrever o teste que falha** (adicionar ao `mod tests` de `types.rs`)

```rust
#[test]
fn metadata_accessors() {
    let v = MetadataValue::U32(42);
    assert_eq!(v.as_u32("k").unwrap(), 42);
    assert!(v.as_str("k").is_err());

    let s = MetadataValue::String("llama".into());
    assert_eq!(s.as_str("k").unwrap(), "llama");

    let arr = MetadataValue::Array(MetadataArray::F32(vec![0.5, 1.0]));
    assert_eq!(arr.as_f32_array("k").unwrap(), &[0.5, 1.0]);
    assert_eq!(arr.array_len(), Some(2));
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p gguf metadata_accessors`
Expected: FAIL (`MetadataValue` não existe).

- [ ] **Step 3: Implementar (adicionar a `crates/gguf/src/types.rs`)**

```rust
/// Valor de um par de metadados GGUF.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// Array homogêneo de metadados (sem aninhamento — llama.cpp não o produz).
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl MetadataArray {
    pub fn len(&self) -> usize {
        match self {
            MetadataArray::U8(v) => v.len(),
            MetadataArray::I8(v) => v.len(),
            MetadataArray::U16(v) => v.len(),
            MetadataArray::I16(v) => v.len(),
            MetadataArray::U32(v) => v.len(),
            MetadataArray::I32(v) => v.len(),
            MetadataArray::F32(v) => v.len(),
            MetadataArray::Bool(v) => v.len(),
            MetadataArray::String(v) => v.len(),
            MetadataArray::U64(v) => v.len(),
            MetadataArray::I64(v) => v.len(),
            MetadataArray::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl MetadataValue {
    pub fn as_u32(&self, key: &str) -> Result<u32, GgufError> {
        match self {
            MetadataValue::U32(v) => Ok(*v),
            _ => Err(GgufError::WrongType { key: key.into(), expected: "u32" }),
        }
    }

    pub fn as_f32(&self, key: &str) -> Result<f32, GgufError> {
        match self {
            MetadataValue::F32(v) => Ok(*v),
            _ => Err(GgufError::WrongType { key: key.into(), expected: "f32" }),
        }
    }

    pub fn as_str(&self, key: &str) -> Result<&str, GgufError> {
        match self {
            MetadataValue::String(s) => Ok(s.as_str()),
            _ => Err(GgufError::WrongType { key: key.into(), expected: "string" }),
        }
    }

    pub fn as_string_array(&self, key: &str) -> Result<&[String], GgufError> {
        match self {
            MetadataValue::Array(MetadataArray::String(v)) => Ok(v),
            _ => Err(GgufError::WrongType { key: key.into(), expected: "string[]" }),
        }
    }

    pub fn as_f32_array(&self, key: &str) -> Result<&[f32], GgufError> {
        match self {
            MetadataValue::Array(MetadataArray::F32(v)) => Ok(v),
            _ => Err(GgufError::WrongType { key: key.into(), expected: "f32[]" }),
        }
    }

    pub fn as_i32_array(&self, key: &str) -> Result<&[i32], GgufError> {
        match self {
            MetadataValue::Array(MetadataArray::I32(v)) => Ok(v),
            _ => Err(GgufError::WrongType { key: key.into(), expected: "i32[]" }),
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            MetadataValue::Array(a) => Some(a.len()),
            _ => None,
        }
    }
}
```

- [ ] **Step 4: Re-exportar em `lib.rs`**

```rust
pub use types::{GgmlType, MetadataArray, MetadataValue};
```

- [ ] **Step 5: Rodar e ver passar**

Run: `cargo test -p gguf metadata_accessors`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/gguf/src/
git commit -m "feat(gguf): MetadataValue/MetadataArray com acessores tipados"
```

---

## Task 4: `gguf` — helper de construção GGUF para testes

**Files:**
- Create: `crates/gguf/src/test_support.rs`
- Modify: `crates/gguf/src/lib.rs`

> Construir bytes GGUF válidos à mão é tedioso e repetido em vários testes.
> Este builder (compilado só em teste) é a fonte DRY para os Tasks 5–7.

- [ ] **Step 1: Implementar `crates/gguf/src/test_support.rs`**

```rust
//! Builder de bytes GGUF para testes. Compilado apenas com `cfg(test)`.
#![cfg(test)]

/// Acumula bytes de um arquivo GGUF v3 little-endian.
pub(crate) struct GgufBuilder {
    kv: Vec<u8>,
    kv_count: u64,
    tensors: Vec<u8>,
    tensor_count: u64,
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self { kv: Vec::new(), kv_count: 0, tensors: Vec::new(), tensor_count: 0 }
    }

    fn push_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Adiciona KV escalar u32 (value_type = 4).
    pub fn kv_u32(mut self, key: &str, val: u32) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&4u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// Adiciona KV escalar f32 (value_type = 6).
    pub fn kv_f32(mut self, key: &str, val: f32) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&6u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// Adiciona KV string (value_type = 8).
    pub fn kv_string(mut self, key: &str, val: &str) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        Self::push_string(&mut self.kv, val);
        self.kv_count += 1;
        self
    }

    /// Adiciona KV array de strings (value_type = 9, elem_type = 8).
    pub fn kv_str_array(mut self, key: &str, vals: &[&str]) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            Self::push_string(&mut self.kv, v);
        }
        self.kv_count += 1;
        self
    }

    /// Adiciona KV array de f32 (value_type = 9, elem_type = 6).
    pub fn kv_f32_array(mut self, key: &str, vals: &[f32]) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&6u32.to_le_bytes());
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
        self
    }

    /// Adiciona KV array de i32 (value_type = 9, elem_type = 5).
    pub fn kv_i32_array(mut self, key: &str, vals: &[i32]) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&5u32.to_le_bytes());
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
        self
    }

    /// Adiciona um tensor info (sem dados). `ggml_type` é o id u32.
    pub fn tensor(mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> Self {
        Self::push_string(&mut self.tensors, name);
        self.tensors.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.tensors.extend_from_slice(&d.to_le_bytes());
        }
        self.tensors.extend_from_slice(&ggml_type.to_le_bytes());
        self.tensors.extend_from_slice(&offset.to_le_bytes());
        self.tensor_count += 1;
        self
    }

    /// Serializa header + KV + tensor infos. NÃO inclui padding nem dados.
    pub fn build_meta_only(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&self.tensor_count.to_le_bytes());
        out.extend_from_slice(&self.kv_count.to_le_bytes());
        out.extend_from_slice(&self.kv);
        out.extend_from_slice(&self.tensors);
        out
    }

    /// Serializa tudo + padding até `alignment` + `data`.
    pub fn build_with_data(&self, alignment: usize, data: &[u8]) -> Vec<u8> {
        let mut out = self.build_meta_only();
        while out.len() % alignment != 0 {
            out.push(0);
        }
        out.extend_from_slice(data);
        out
    }
}
```

- [ ] **Step 2: Declarar em `lib.rs`**

```rust
#[cfg(test)]
mod test_support;
```

- [ ] **Step 3: Verificar que compila em modo teste**

Run: `cargo test -p gguf --no-run`
Expected: compila sem erros.

- [ ] **Step 4: Commit**

```bash
git add crates/gguf/src/
git commit -m "test(gguf): builder de bytes GGUF para testes"
```

---

## Task 5: `gguf` — parser (header + KVs + tensor infos)

**Files:**
- Create: `crates/gguf/src/parse.rs`
- Create: `crates/gguf/src/file.rs`
- Modify: `crates/gguf/src/lib.rs`

- [ ] **Step 1: Escrever o teste que falha** (`crates/gguf/src/parse.rs`)

```rust
#[cfg(test)]
mod tests {
    use crate::file::GgufFile;
    use crate::test_support::GgufBuilder;
    use crate::types::GgmlType;

    fn sample() -> Vec<u8> {
        GgufBuilder::new()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.block_count", 5)
            .kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_str_array("tokenizer.ggml.tokens", &["<unk>", "<s>", "</s>"])
            .tensor("token_embd.weight", &[64, 512], 0, 0)
            .build_meta_only()
    }

    #[test]
    fn parses_header_and_metadata() {
        let bytes = sample();
        let f = GgufFile::parse(&bytes).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(
            f.metadata.get("general.architecture").unwrap().as_str("k").unwrap(),
            "llama"
        );
        assert_eq!(
            f.metadata.get("llama.block_count").unwrap().as_u32("k").unwrap(),
            5
        );
        assert_eq!(f.metadata.get("tokenizer.ggml.tokens").unwrap().array_len(), Some(3));
    }

    #[test]
    fn parses_tensor_info() {
        let bytes = sample();
        let f = GgufFile::parse(&bytes).unwrap();
        assert_eq!(f.tensors.len(), 1);
        let t = &f.tensors[0];
        assert_eq!(t.name, "token_embd.weight");
        assert_eq!(t.dims, vec![64, 512]);
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.offset, 0);
    }

    #[test]
    fn bad_magic_is_error() {
        let mut bytes = sample();
        bytes[0] = b'X';
        assert!(GgufFile::parse(&bytes).is_err());
    }

    #[test]
    fn unsupported_version_is_error() {
        let mut bytes = sample();
        bytes[4] = 2; // version = 2
        assert!(GgufFile::parse(&bytes).is_err());
    }
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p gguf parse`
Expected: FAIL (`GgufFile`/`parse` não existem).

- [ ] **Step 3: Implementar leitura de valores em `crates/gguf/src/parse.rs`**

```rust
//! Parsing do header, dos pares de metadados e dos tensor infos.

use std::collections::BTreeMap;

use crate::error::GgufError;
use crate::reader::Reader;
use crate::types::{GgmlType, MetadataArray, MetadataValue};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSION: u32 = 3;

/// Lê um valor de metadado a partir do `value_type` GGUF.
pub(crate) fn read_value(r: &mut Reader, value_type: u32) -> Result<MetadataValue, GgufError> {
    Ok(match value_type {
        0 => MetadataValue::U8(r.u8()?),
        1 => MetadataValue::I8(r.i8()?),
        2 => MetadataValue::U16(r.u16()?),
        3 => MetadataValue::I16(r.i16()?),
        4 => MetadataValue::U32(r.u32()?),
        5 => MetadataValue::I32(r.i32()?),
        6 => MetadataValue::F32(r.f32()?),
        7 => MetadataValue::Bool(r.bool()?),
        8 => MetadataValue::String(r.gguf_string()?),
        9 => MetadataValue::Array(read_array(r)?),
        10 => MetadataValue::U64(r.u64()?),
        11 => MetadataValue::I64(r.i64()?),
        12 => MetadataValue::F64(r.f64()?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

fn read_array(r: &mut Reader) -> Result<MetadataArray, GgufError> {
    let elem_type = r.u32()?;
    let count = usize::try_from(r.u64()?).map_err(|_| GgufError::Overflow)?;
    // NÃO pré-alocar `count` (pode ser malicioso); push incremental — se os
    // bytes acabarem, o `read` falha e o loop aborta com erro.
    macro_rules! collect {
        ($variant:ident, $method:ident) => {{
            let mut v = Vec::new();
            for _ in 0..count {
                v.push(r.$method()?);
            }
            MetadataArray::$variant(v)
        }};
    }
    Ok(match elem_type {
        0 => collect!(U8, u8),
        1 => collect!(I8, i8),
        2 => collect!(U16, u16),
        3 => collect!(I16, i16),
        4 => collect!(U32, u32),
        5 => collect!(I32, i32),
        6 => collect!(F32, f32),
        7 => collect!(Bool, bool),
        8 => collect!(String, gguf_string),
        10 => collect!(U64, u64),
        11 => collect!(I64, i64),
        12 => collect!(F64, f64),
        9 => return Err(GgufError::NestedArray),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

/// Resultado intermediário do parsing (consumido por `file.rs`).
pub(crate) struct Parsed {
    pub version: u32,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<crate::file::TensorInfo>,
    pub data_offset: usize,
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Parsed, GgufError> {
    let mut r = Reader::new(bytes);

    let magic = r.array::<4>()?;
    if &magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic(magic));
    }
    let version = r.u32()?;
    if version != SUPPORTED_VERSION {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.gguf_string()?;
        let value_type = r.u32()?;
        let value = read_value(&mut r, value_type)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = r.gguf_string()?;
        let n_dims = r.u32()?;
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ggml_type = GgmlType::try_from(r.u32()?)?;
        let offset = r.u64()?;
        tensors.push(crate::file::TensorInfo { name, dims, ggml_type, offset });
    }

    let alignment = match metadata.get("general.alignment") {
        Some(v) => v.as_u32("general.alignment")? as usize,
        None => 32,
    };
    let pos = r.position();
    let data_offset = align_up(pos, alignment)?;

    Ok(Parsed { version, metadata, tensors, data_offset })
}

fn align_up(pos: usize, alignment: usize) -> Result<usize, GgufError> {
    if alignment == 0 {
        return Ok(pos);
    }
    let rem = pos % alignment;
    if rem == 0 {
        Ok(pos)
    } else {
        pos.checked_add(alignment - rem).ok_or(GgufError::Overflow)
    }
}
```

- [ ] **Step 4: Implementar `crates/gguf/src/file.rs`**

```rust
//! Estrutura de alto nível do arquivo GGUF parseado.

use std::collections::BTreeMap;

use crate::error::GgufError;
use crate::types::{GgmlType, MetadataValue};

/// Descritor de um tensor (sem os dados).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relativo ao início da seção de dados.
    pub offset: u64,
}

/// Arquivo GGUF parseado: metadados + tensor infos. Não retém os bytes.
#[derive(Clone, Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub(crate) data_offset: usize,
}

impl GgufFile {
    /// Parseia metadados + tensor infos do conteúdo completo do arquivo.
    pub fn parse(bytes: &[u8]) -> Result<GgufFile, GgufError> {
        let p = crate::parse::parse(bytes)?;
        Ok(GgufFile {
            version: p.version,
            metadata: p.metadata,
            tensors: p.tensors,
            data_offset: p.data_offset,
        })
    }

    /// Atalho tipado para um KV obrigatório.
    pub fn get(&self, key: &str) -> Result<&MetadataValue, GgufError> {
        self.metadata.get(key).ok_or_else(|| GgufError::MissingKey(key.into()))
    }
}
```

- [ ] **Step 5: Declarar módulos em `lib.rs`**

```rust
mod file;
mod parse;

pub use file::{GgufFile, TensorInfo};
```

- [ ] **Step 6: Rodar e ver passar**

Run: `cargo test -p gguf parse`
Expected: PASS (4 testes).

- [ ] **Step 7: Commit**

```bash
git add crates/gguf/src/
git commit -m "feat(gguf): parser de header, metadados e tensor infos"
```

---

## Task 6: `gguf` — `tensor_data` (acesso raw)

**Files:**
- Modify: `crates/gguf/src/file.rs`

- [ ] **Step 1: Escrever o teste que falha** (adicionar ao final de `crates/gguf/src/file.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::GgufBuilder;

    #[test]
    fn tensor_data_slices_correctly() {
        // 1 tensor F32 com 4 elementos = 16 bytes, offset 0, alignment 32.
        let data: Vec<u8> = (0..16u8).collect();
        let bytes = GgufBuilder::new()
            .tensor("t", &[4], 0, 0)
            .build_with_data(32, &data);
        let f = GgufFile::parse(&bytes).unwrap();
        let slice = f.tensor_data(&bytes, &f.tensors[0]).unwrap();
        assert_eq!(slice, &data[..]);
    }

    #[test]
    fn tensor_data_out_of_bounds_is_error() {
        let bytes = GgufBuilder::new()
            .tensor("t", &[4], 0, 0)
            .build_with_data(32, &[0u8; 4]); // só 4 bytes, precisa de 16
        let f = GgufFile::parse(&bytes).unwrap();
        assert!(f.tensor_data(&bytes, &f.tensors[0]).is_err());
    }

    #[test]
    fn tensor_data_q8_0_size() {
        // Q8_0 (id 8): block_size 32, type_size 34. 32 elementos = 1 bloco = 34 bytes.
        let bytes = GgufBuilder::new()
            .tensor("q", &[32], 8, 0)
            .build_with_data(32, &[7u8; 34]);
        let f = GgufFile::parse(&bytes).unwrap();
        assert_eq!(f.tensor_data(&bytes, &f.tensors[0]).unwrap().len(), 34);
    }
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p gguf tensor_data`
Expected: FAIL (`tensor_data` não existe).

- [ ] **Step 3: Implementar `tensor_data` (adicionar ao `impl GgufFile`)**

```rust
    /// Bytes raw de um tensor (sem dequant). Slice sobre a seção de dados.
    pub fn tensor_data<'a>(
        &self,
        bytes: &'a [u8],
        t: &TensorInfo,
    ) -> Result<&'a [u8], GgufError> {
        let mut n_elements: u64 = 1;
        for &d in &t.dims {
            n_elements = n_elements.checked_mul(d).ok_or(GgufError::Overflow)?;
        }
        let block_size = t.ggml_type.block_size();
        let type_size = t.ggml_type.type_size();
        // n_elements deve ser múltiplo do block_size.
        if block_size == 0 || n_elements % block_size != 0 {
            return Err(GgufError::TensorOutOfBounds { name: t.name.clone() });
        }
        let n_blocks = n_elements / block_size;
        let n_bytes = n_blocks.checked_mul(type_size).ok_or(GgufError::Overflow)?;

        let start = self
            .data_offset
            .checked_add(usize::try_from(t.offset).map_err(|_| GgufError::Overflow)?)
            .ok_or(GgufError::Overflow)?;
        let n_bytes = usize::try_from(n_bytes).map_err(|_| GgufError::Overflow)?;
        let end = start.checked_add(n_bytes).ok_or(GgufError::Overflow)?;

        bytes
            .get(start..end)
            .ok_or_else(|| GgufError::TensorOutOfBounds { name: t.name.clone() })
    }
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test -p gguf tensor_data`
Expected: PASS (3 testes).

- [ ] **Step 5: Commit**

```bash
git add crates/gguf/src/
git commit -m "feat(gguf): tensor_data com cálculo de tamanho por bloco"
```

---

## Task 7: `gguf` — proptest (nunca panica) + teste no modelo real

**Files:**
- Create: `crates/gguf/tests/fuzz_parse.rs`
- Create: `crates/gguf/tests/real_model.rs`

- [ ] **Step 1: Escrever o proptest** (`crates/gguf/tests/fuzz_parse.rs`)

```rust
//! Garantia: `GgufFile::parse` nunca panica, qualquer que seja a entrada.
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn parse_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        let _ = gguf::GgufFile::parse(&bytes);
    }

    #[test]
    fn parse_never_panics_with_gguf_prefix(tail in proptest::collection::vec(any::<u8>(), 0..512)) {
        // Começa com magic + version válidos para entrar fundo no parser.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&tail);
        let _ = gguf::GgufFile::parse(&bytes);
    }
}
```

- [ ] **Step 2: Rodar e ver passar (deve passar de imediato; valida robustez)**

Run: `cargo test -p gguf --test fuzz_parse`
Expected: PASS. Se algum caso panicar, é bug no Task 1/5 — corrigir lá (sem `unwrap`/index direto).

- [ ] **Step 3: Escrever o teste no modelo real** (`crates/gguf/tests/real_model.rs`)

```rust
//! Parseia o stories260K.gguf de verdade e confere fatos conhecidos
//! (do dump do loader do llama.cpp).
use std::path::Path;

fn load() -> Option<gguf::GgufFile> {
    // cwd nos testes de integração = raiz do crate (crates/gguf); modelo está
    // dois níveis acima.
    let path = Path::new("../../models/stories260K.gguf");
    let bytes = std::fs::read(path).ok()?;
    Some(gguf::GgufFile::parse(&bytes).expect("stories260K deve parsear"))
}

#[test]
fn stories260k_scalar_metadata() {
    let Some(f) = load() else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    assert_eq!(f.version, 3);
    assert_eq!(f.get("general.architecture").unwrap().as_str("k").unwrap(), "llama");
    assert_eq!(f.get("tokenizer.ggml.model").unwrap().as_str("k").unwrap(), "llama");
    assert_eq!(f.get("llama.block_count").unwrap().as_u32("k").unwrap(), 5);
    assert_eq!(f.get("llama.embedding_length").unwrap().as_u32("k").unwrap(), 64);
    assert_eq!(f.get("llama.attention.head_count").unwrap().as_u32("k").unwrap(), 8);
    assert_eq!(f.get("llama.attention.head_count_kv").unwrap().as_u32("k").unwrap(), 4);
    assert_eq!(f.get("tokenizer.ggml.bos_token_id").unwrap().as_u32("k").unwrap(), 1);
    assert_eq!(f.get("tokenizer.ggml.eos_token_id").unwrap().as_u32("k").unwrap(), 2);
}

#[test]
fn stories260k_arrays_and_tensors() {
    let Some(f) = load() else { return };
    assert_eq!(f.get("tokenizer.ggml.tokens").unwrap().array_len(), Some(512));
    assert_eq!(f.get("tokenizer.ggml.scores").unwrap().array_len(), Some(512));
    assert_eq!(f.get("tokenizer.ggml.token_type").unwrap().array_len(), Some(512));
    // 48 tensores f32 (do dump do loader).
    assert_eq!(f.tensors.len(), 48);
    assert!(f.tensors.iter().all(|t| t.ggml_type == gguf::GgmlType::F32));
    // token_embd: tensor_data deve fatiar n_elements*4 bytes.
    let bytes = std::fs::read("../../models/stories260K.gguf").unwrap();
    let embd = f.tensors.iter().find(|t| t.name == "token_embd.weight").unwrap();
    let data = f.tensor_data(&bytes, embd).unwrap();
    let n: u64 = embd.dims.iter().product();
    assert_eq!(data.len() as u64, n * 4);
}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test -p gguf --test real_model`
Expected: PASS (2 testes). Se `tensors.len()` divergir, ajustar a asserção ao valor real do dump (48 é o esperado).

- [ ] **Step 5: Commit**

```bash
git add crates/gguf/tests/
git commit -m "test(gguf): proptest de robustez + parsing do modelo real"
```

---

## Task 8: `gguf` — snapshot de metadados vs oráculo

**Files:**
- Create: `refs/stories260k-meta.json`
- Create: `crates/gguf/tests/meta_snapshot.rs`

> Estratégia (do spec): escalares conferidos contra o dump do loader
> (`llama_model_loader: - kv ...`); arrays validados por tamanho (Task 7) e
> transitivamente pelo tokenizer (Task 13).

- [ ] **Step 1: Gerar o dump do loader e criar o snapshot**

Run (gera os KVs com valores completos para escalares):

```bash
LD_LIBRARY_PATH=build-oracle/bin build-oracle/bin/llama-tokenize \
  -m models/stories260K.gguf -p "x" 2>&1 | grep "llama_model_loader: - kv"
```

Criar `refs/stories260k-meta.json` com os escalares (conferidos contra o dump acima):

```json
{
  "model": "models/stories260K.gguf",
  "version": 3,
  "scalars": {
    "general.architecture": "llama",
    "general.name": "llama",
    "tokenizer.ggml.model": "llama",
    "tokenizer.ggml.unknown_token_id": 0,
    "tokenizer.ggml.bos_token_id": 1,
    "tokenizer.ggml.eos_token_id": 2,
    "llama.context_length": 2048,
    "llama.embedding_length": 64,
    "llama.feed_forward_length": 172,
    "llama.attention.head_count": 8,
    "llama.attention.head_count_kv": 4,
    "llama.block_count": 5,
    "llama.rope.dimension_count": 8,
    "llama.attention.layer_norm_rms_epsilon": 1.0e-5
  },
  "array_lengths": {
    "tokenizer.ggml.tokens": 512,
    "tokenizer.ggml.scores": 512,
    "tokenizer.ggml.token_type": 512
  },
  "tensor_count": 48
}
```

- [ ] **Step 2: Escrever o teste de snapshot** (`crates/gguf/tests/meta_snapshot.rs`)

```rust
//! Confere a saída parseada contra o snapshot revisado refs/stories260k-meta.json.
use serde_json::Value;

#[test]
fn matches_reviewed_snapshot() {
    let Ok(model) = std::fs::read("../../models/stories260K.gguf") else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&model).unwrap();
    let snap: Value =
        serde_json::from_str(&std::fs::read_to_string("../../refs/stories260k-meta.json").unwrap())
            .unwrap();

    assert_eq!(f.version as u64, snap["version"].as_u64().unwrap());

    for (key, expected) in snap["scalars"].as_object().unwrap() {
        let got = f.get(key).unwrap();
        match expected {
            Value::String(s) => assert_eq!(got.as_str(key).unwrap(), s, "{key}"),
            Value::Number(n) if n.is_f64() && n.as_u64().is_none() => {
                let e = n.as_f64().unwrap() as f32;
                assert!((got.as_f32(key).unwrap() - e).abs() < 1e-9, "{key}");
            }
            Value::Number(n) => {
                assert_eq!(got.as_u32(key).unwrap() as u64, n.as_u64().unwrap(), "{key}")
            }
            _ => panic!("tipo inesperado no snapshot para {key}"),
        }
    }

    for (key, len) in snap["array_lengths"].as_object().unwrap() {
        assert_eq!(f.get(key).unwrap().array_len(), Some(len.as_u64().unwrap() as usize), "{key}");
    }

    assert_eq!(f.tensors.len() as u64, snap["tensor_count"].as_u64().unwrap());
}
```

- [ ] **Step 3: Rodar e ver passar**

Run: `cargo test -p gguf --test meta_snapshot`
Expected: PASS. Ajustar valores do JSON se algum divergir do dump real.

- [ ] **Step 4: Commit**

```bash
git add refs/stories260k-meta.json crates/gguf/tests/meta_snapshot.rs
git commit -m "test(gguf): snapshot de metadados conferido contra o oráculo"
```

---

## Task 9: `llama-tokenizer` — `Vocab` + `from_gguf`

**Files:**
- Create: `crates/llama-tokenizer/src/error.rs`
- Create: `crates/llama-tokenizer/src/vocab.rs`
- Modify: `crates/llama-tokenizer/src/lib.rs`

- [ ] **Step 1: Escrever o teste que falha** (`crates/llama-tokenizer/src/vocab.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Vocab {
        // ids:      0       1      2      3        4        5
        let tokens = vec!["<unk>", "<s>", "</s>", "<0x41>", "ab", "abc"]
            .into_iter()
            .map(String::from)
            .collect();
        let scores = vec![0.0, 0.0, 0.0, 0.0, -1.0, -0.5];
        let token_types = vec![2, 3, 3, 6, 1, 1];
        Vocab::new(tokens, scores, token_types, 1, 2, 0)
    }

    #[test]
    fn text_to_token_lookup() {
        let v = tiny();
        assert_eq!(v.text_to_token("ab"), Some(4));
        assert_eq!(v.text_to_token("zzz"), None);
    }

    #[test]
    fn byte_to_token_uppercase_hex() {
        let v = tiny();
        // 0x41 = 'A' → token "<0x41>" = id 3
        assert_eq!(v.byte_to_token(0x41), Some(3));
    }

    #[test]
    fn score_lookup() {
        let v = tiny();
        assert_eq!(v.score(5), -0.5);
    }
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p llama-tokenizer vocab`
Expected: FAIL (`Vocab` não existe).

- [ ] **Step 3: Implementar `crates/llama-tokenizer/src/error.rs`**

```rust
//! Erros do tokenizer.

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("erro ao ler GGUF: {0}")]
    Gguf(#[from] gguf::GgufError),
    #[error("modelo de tokenizer não suportado: {0:?} (suportado: \"llama\"/SPM)")]
    UnsupportedModel(String),
    #[error("arrays do vocab com tamanhos inconsistentes: tokens={tokens}, scores={scores}, types={types}")]
    InconsistentVocab { tokens: usize, scores: usize, types: usize },
}
```

- [ ] **Step 4: Implementar `crates/llama-tokenizer/src/vocab.rs`**

```rust
//! Vocabulário SPM extraído dos metadados GGUF.

use std::collections::HashMap;

use gguf::GgufFile;

use crate::error::TokenizerError;

const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// Vocabulário: tokens, scores, tipos e ids especiais.
pub struct Vocab {
    tokens: Vec<String>,
    scores: Vec<f32>,
    #[allow(dead_code)]
    token_types: Vec<i32>,
    token_to_id: HashMap<String, u32>,
    pub(crate) bos_id: u32,
    pub(crate) eos_id: u32,
    #[allow(dead_code)]
    pub(crate) unk_id: u32,
}

impl Vocab {
    /// Construtor direto (usado em testes e por `from_gguf`).
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        bos_id: u32,
        eos_id: u32,
        unk_id: u32,
    ) -> Vocab {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            // Em colisão, o primeiro id vence (como o token_to_id do llama.cpp,
            // populado em ordem crescente sem sobrescrever).
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }
        Vocab { tokens, scores, token_types, token_to_id, bos_id, eos_id, unk_id }
    }

    /// Lê o vocab SPM dos metadados de um GGUF já parseado.
    pub fn from_gguf(f: &GgufFile) -> Result<Vocab, TokenizerError> {
        let model = f.get("tokenizer.ggml.model")?.as_str("tokenizer.ggml.model")?;
        if model != "llama" {
            return Err(TokenizerError::UnsupportedModel(model.to_owned()));
        }
        let tokens: Vec<String> = f
            .get("tokenizer.ggml.tokens")?
            .as_string_array("tokenizer.ggml.tokens")?
            .to_vec();
        let scores: Vec<f32> =
            f.get("tokenizer.ggml.scores")?.as_f32_array("tokenizer.ggml.scores")?.to_vec();
        let token_types: Vec<i32> =
            f.get("tokenizer.ggml.token_type")?.as_i32_array("tokenizer.ggml.token_type")?.to_vec();

        if tokens.len() != scores.len() || tokens.len() != token_types.len() {
            return Err(TokenizerError::InconsistentVocab {
                tokens: tokens.len(),
                scores: scores.len(),
                types: token_types.len(),
            });
        }

        let bos_id = f.get("tokenizer.ggml.bos_token_id")?.as_u32("bos")?;
        let eos_id = f.get("tokenizer.ggml.eos_token_id")?.as_u32("eos")?;
        let unk_id = f.get("tokenizer.ggml.unknown_token_id")?.as_u32("unk")?;

        Ok(Vocab::new(tokens, scores, token_types, bos_id, eos_id, unk_id))
    }

    pub(crate) fn text_to_token(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    pub(crate) fn score(&self, id: u32) -> f32 {
        self.scores.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY)
    }

    /// Byte → token. Tenta `<0xXX>` (hex maiúsculo), depois o byte como string
    /// de 1 caractere (espelha `llama_vocab::byte_to_token` para SPM).
    pub(crate) fn byte_to_token(&self, ch: u8) -> Option<u32> {
        let buf = [
            b'<',
            b'0',
            b'x',
            HEX[(ch >> 4) as usize],
            HEX[(ch & 0x0F) as usize],
            b'>',
        ];
        // `buf` é sempre ASCII válido.
        let key = core::str::from_utf8(&buf).unwrap_or("<0x00>");
        if let Some(id) = self.token_to_id.get(key).copied() {
            return Some(id);
        }
        let single = [ch];
        core::str::from_utf8(&single).ok().and_then(|s| self.token_to_id.get(s).copied())
    }

    pub(crate) fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }
}
```

> Nota: a linha `core::str::from_utf8(&buf).unwrap_or(...)` usa `unwrap_or`, não
> `unwrap` — o lint `unwrap_used` permanece satisfeito.

- [ ] **Step 5: Declarar em `lib.rs`**

```rust
#![forbid(unsafe_code)]
//! Tokenizer SPM (Llama) — encode/decode bit-exact vs llama.cpp.

mod error;
mod vocab;

pub use error::TokenizerError;
pub use vocab::Vocab;
```

- [ ] **Step 6: Rodar e ver passar**

Run: `cargo test -p llama-tokenizer vocab`
Expected: PASS (3 testes).

- [ ] **Step 7: Commit**

```bash
git add crates/llama-tokenizer/src/
git commit -m "feat(tokenizer): Vocab + from_gguf (SPM)"
```

---

## Task 10: `llama-tokenizer` — núcleo SPM (símbolos + merge + resegment)

**Files:**
- Create: `crates/llama-tokenizer/src/spm.rs`
- Modify: `crates/llama-tokenizer/src/lib.rs`

> Réplica de `llm_tokenizer_spm_session::tokenize` (`src/llama-vocab.cpp:117-201`).
> A entrada já vem normalizada (espaços → `▁`); BOS e prefixo são tratados no Task 11.

- [ ] **Step 1: Escrever o teste que falha** (`crates/llama-tokenizer/src/spm.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    fn tiny() -> Vocab {
        // "abc" tem score maior (-0.5) que "ab" (-1.0); merge deve preferir "abc".
        let tokens = vec!["<unk>", "<s>", "</s>", "<0x64>", "a", "b", "c", "ab", "abc"]
            .into_iter()
            .map(String::from)
            .collect();
        let scores = vec![0.0, 0.0, 0.0, 0.0, -3.0, -3.0, -3.0, -1.0, -0.5];
        let types = vec![2, 3, 3, 6, 1, 1, 1, 1, 1];
        Vocab::new(tokens, scores, types, 1, 2, 0)
    }

    #[test]
    fn merges_highest_score_first() {
        let v = tiny();
        // "abc" → deve resultar no único token id 8 ("abc"), não em "ab"+"c".
        assert_eq!(tokenize_spm(&v, "abc"), vec![8]);
    }

    #[test]
    fn byte_fallback_for_unknown_char() {
        let v = tiny();
        // 'd' (0x64) não está como char, mas "<0x64>" sim (id 3).
        assert_eq!(tokenize_spm(&v, "d"), vec![3]);
    }

    #[test]
    fn splits_when_no_merge() {
        let v = tiny();
        // "ba" não casa nenhum merge melhor → tokens "b","a" = [5,4].
        assert_eq!(tokenize_spm(&v, "ba"), vec![5, 4]);
    }
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p llama-tokenizer spm`
Expected: FAIL (`tokenize_spm` não existe).

- [ ] **Step 3: Implementar `crates/llama-tokenizer/src/spm.rs`**

```rust
//! Núcleo do algoritmo SPM (merge por score + byte-fallback).
//! Réplica fiel de `llm_tokenizer_spm_session` do llama.cpp.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::vocab::Vocab;

/// Símbolo na cadeia: fatia `[start, start+len)` dos bytes normalizados.
struct Symbol {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
}

/// Bigrama candidato a merge.
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bigram {
    /// "Maior" = maior score; empate → menor `left` (espelha o comparator do C++:
    /// `l < r` se `l.score < r.score || (== && l.left > r.left)`).
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.total_cmp(&o.score).then_with(|| o.left.cmp(&self.left))
    }
}

/// Comprimento em bytes de um char UTF-8 a partir do primeiro byte.
fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1, // continuação/inválido: trata como 1 byte (como o min() do C++)
    }
}

/// Tokeniza bytes já normalizados (espaços já viraram `▁`).
pub(crate) fn tokenize_spm(vocab: &Vocab, text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut symbols: Vec<Symbol> = Vec::new();

    // 1. Divide em símbolos UTF-8.
    let mut offs = 0usize;
    let mut index = 0i32;
    while offs < bytes.len() {
        let len = utf8_len(bytes[offs]).min(bytes.len() - offs);
        let next = if offs + len == bytes.len() { -1 } else { index + 1 };
        symbols.push(Symbol { start: offs, len, prev: index - 1, next });
        offs += len;
        index += 1;
    }

    let mut work: BinaryHeap<Bigram> = BinaryHeap::new();
    let mut rev_merge: HashMap<(usize, usize), (i32, i32)> = HashMap::new();

    let try_add_bigram =
        |work: &mut BinaryHeap<Bigram>,
         rev_merge: &mut HashMap<(usize, usize), (i32, i32)>,
         symbols: &[Symbol],
         left: i32,
         right: i32| {
            if left == -1 || right == -1 {
                return;
            }
            let l = &symbols[left as usize];
            let r = &symbols[right as usize];
            let start = l.start;
            let size = l.len + r.len;
            let Ok(text) = core::str::from_utf8(&bytes[start..start + size]) else {
                return;
            };
            let Some(id) = vocab.text_to_token(text) else {
                return;
            };
            work.push(Bigram { left, right, score: vocab.score(id), size });
            rev_merge.insert((start, size), (left, right));
        };

    // 2. Semeia bigramas adjacentes.
    for i in 1..symbols.len() as i32 {
        try_add_bigram(&mut work, &mut rev_merge, &symbols, i - 1, i);
    }

    // 3. Funde o par de maior score enquanto houver.
    while let Some(bigram) = work.pop() {
        let (ln, rn) = {
            let l = &symbols[bigram.left as usize];
            let r = &symbols[bigram.right as usize];
            (l.len, r.len)
        };
        if ln == 0 || rn == 0 || ln + rn != bigram.size {
            continue; // um dos símbolos já foi fundido
        }
        // funde right em left
        let right_next = symbols[bigram.right as usize].next;
        {
            let l = &mut symbols[bigram.left as usize];
            l.len += rn;
            l.next = right_next;
        }
        symbols[bigram.right as usize].len = 0;
        if right_next >= 0 {
            symbols[right_next as usize].prev = bigram.left;
        }
        let left_prev = symbols[bigram.left as usize].prev;
        let left_next = symbols[bigram.left as usize].next;
        try_add_bigram(&mut work, &mut rev_merge, &symbols, left_prev, bigram.left);
        try_add_bigram(&mut work, &mut rev_merge, &symbols, bigram.left, left_next);
    }

    // 4. Resegmenta a cadeia final.
    let mut output = Vec::new();
    let mut i = 0i32;
    while i != -1 && (i as usize) < symbols.len() {
        let (start, len) = {
            let s = &symbols[i as usize];
            (s.start, s.len)
        };
        resegment(vocab, bytes, &symbols, &rev_merge, start, len, &mut output);
        i = symbols[i as usize].next;
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn resegment(
    vocab: &Vocab,
    bytes: &[u8],
    symbols: &[Symbol],
    rev_merge: &HashMap<(usize, usize), (i32, i32)>,
    start: usize,
    len: usize,
    output: &mut Vec<u32>,
) {
    if let Ok(text) = core::str::from_utf8(&bytes[start..start + len]) {
        if let Some(id) = vocab.text_to_token(text) {
            output.push(id);
            return;
        }
    }
    match rev_merge.get(&(start, len)) {
        Some(&(left, right)) => {
            let l = &symbols[left as usize];
            let r = &symbols[right as usize];
            resegment(vocab, bytes, symbols, rev_merge, l.start, l.len, output);
            resegment(vocab, bytes, symbols, rev_merge, r.start, r.len, output);
        }
        None => {
            // byte-fallback: cada byte vira <0xXX> (ou byte cru).
            for j in 0..len {
                if let Some(id) = vocab.byte_to_token(bytes[start + j]) {
                    output.push(id);
                }
            }
        }
    }
}
```

> Diferença vs C++: o `rev_merge` do upstream é chaveado pela string; aqui uso
> `(start, len)` (posição na cadeia), que identifica unicamente o span e evita
> ambiguidade quando o mesmo texto aparece em posições diferentes. Em
> divergência, revisar este ponto com `systematic-debugging`.

- [ ] **Step 4: Declarar `mod spm;` em `lib.rs`** (privado por enquanto)

```rust
mod spm;
```

- [ ] **Step 5: Rodar e ver passar**

Run: `cargo test -p llama-tokenizer spm`
Expected: PASS (3 testes).

- [ ] **Step 6: Commit**

```bash
git add crates/llama-tokenizer/src/
git commit -m "feat(tokenizer): núcleo SPM (merge por score + byte-fallback)"
```

---

## Task 11: `llama-tokenizer` — `encode`/`decode` (normalização + BOS)

**Files:**
- Modify: `crates/llama-tokenizer/src/lib.rs`

> Pipeline de `encode` (`src/llama-vocab.cpp:3290-3318`): se `add_bos`, empurra
> BOS e marca `is_prev_special=true`; para o texto, se `add_space_prefix &&
> is_prev_special`, prefixa um espaço; depois `escape_whitespace` (' ' → `▁`,
> bytes `E2 96 81`); então SPM. `decode` reverte `▁` → ' '.

- [ ] **Step 1: Escrever o teste que falha** (`crates/llama-tokenizer/src/lib.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    fn tiny() -> Vocab {
        // inclui o token de espaço "▁" e algumas letras
        let tokens = vec!["<unk>", "<s>", "</s>", "\u{2581}", "hi"]
            .into_iter()
            .map(String::from)
            .collect();
        let scores = vec![0.0, 0.0, 0.0, -1.0, -0.5];
        let types = vec![2, 3, 3, 1, 1];
        Vocab::new(tokens, scores, types, 1, 2, 0)
    }

    #[test]
    fn encode_prepends_bos_and_space() {
        let t = Tokenizer::new(tiny());
        // "hi" com add_bos: BOS(1), depois "▁hi" → "▁"(3) + "hi"(4)
        assert_eq!(t.encode("hi", true), vec![1, 3, 4]);
    }

    #[test]
    fn encode_without_bos() {
        let t = Tokenizer::new(tiny());
        // sem add_bos → sem prefixo de espaço (is_prev_special começa false)
        assert_eq!(t.encode("hi", false), vec![4]);
    }

    #[test]
    fn decode_roundtrip_text() {
        let t = Tokenizer::new(tiny());
        let ids = t.encode("hi", true);
        assert_eq!(t.decode(&ids), "hi");
    }
}
```

> Atenção a `encode_without_bos`: confirme o comportamento real do upstream para
> este modelo durante o Task 12. Se divergir, este é o teste a ajustar (o
> critério de aceite usa `add_bos=true`).

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test -p llama-tokenizer --lib`
Expected: FAIL (`Tokenizer` não existe).

- [ ] **Step 3: Implementar (adicionar a `crates/llama-tokenizer/src/lib.rs`)**

```rust
use gguf::GgufFile;

const SPACE_ESCAPE: &str = "\u{2581}"; // ▁ = E2 96 81

/// Tokenizer SPM (Llama).
pub struct Tokenizer {
    vocab: Vocab,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    pub fn from_gguf(f: &GgufFile) -> Result<Self, TokenizerError> {
        Ok(Self { vocab: Vocab::from_gguf(f)? })
    }

    /// Codifica `text` em ids. Com `add_bos`, prefixa o token BOS e um espaço
    /// (add_space_prefix), espelhando o pipeline SPM do llama.cpp.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut output = Vec::new();
        let mut is_prev_special = false;
        if add_bos {
            output.push(self.vocab.bos_id);
            is_prev_special = true;
        }
        let mut buf = String::new();
        if is_prev_special {
            buf.push(' ');
        }
        buf.push_str(text);
        let normalized = buf.replace(' ', SPACE_ESCAPE);
        let ids = crate::spm::tokenize_spm(&self.vocab, &normalized);
        output.extend(ids);
        output
    }

    /// Decodifica ids em texto: concatena os textos dos tokens e reverte `▁`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == self.vocab.bos_id || id == self.vocab.eos_id {
                continue;
            }
            if let Some(t) = self.vocab.token_text(id) {
                out.push_str(t);
            }
        }
        let out = out.replace(SPACE_ESCAPE, " ");
        // remove o espaço de prefixo introduzido no encode
        out.strip_prefix(' ').map(String::from).unwrap_or(out)
    }
}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test -p llama-tokenizer --lib`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-tokenizer/src/
git commit -m "feat(tokenizer): encode/decode com BOS e normalização SPM"
```

---

## Task 12: `llama-tokenizer` — teste diferencial vs `refs/tokens.json`

**Files:**
- Create: `crates/llama-tokenizer/tests/oracle_corpus.rs`

- [ ] **Step 1: Escrever o teste diferencial** (`crates/llama-tokenizer/tests/oracle_corpus.rs`)

```rust
//! Critério de aceite da fase: encode bit-exact vs o corpus do oráculo.
use serde_json::Value;

#[test]
fn encode_matches_oracle_corpus() {
    let model_bytes = match std::fs::read("../../models/stories260K.gguf") {
        Ok(b) => b,
        Err(_) => {
            eprintln!("modelo ausente — pulando");
            return;
        }
    };
    let f = gguf::GgufFile::parse(&model_bytes).unwrap();
    let tok = llama_tokenizer::Tokenizer::from_gguf(&f).unwrap();

    let corpus: Value =
        serde_json::from_str(&std::fs::read_to_string("../../refs/tokens.json").unwrap()).unwrap();

    let mut failures = Vec::new();
    for case in corpus["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> =
            case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text, true);
        if got != expected {
            failures.push(format!("text={text:?}\n  esperado={expected:?}\n  obtido  ={got:?}"));
        }
    }
    assert!(failures.is_empty(), "divergências:\n{}", failures.join("\n"));
}
```

- [ ] **Step 2: Rodar**

Run: `cargo test -p llama-tokenizer --test oracle_corpus -- --nocapture`
Expected: PASS. Se falhar, NÃO ajustar o teste — depurar o SPM com
`superpowers:systematic-debugging` (bissecção por caso/símbolo) e ler o upstream
em `src/llama-vocab.cpp`. Ajustar `spm.rs`/`lib.rs` até bit-exact.

- [ ] **Step 3: Commit**

```bash
git add crates/llama-tokenizer/tests/
git commit -m "test(tokenizer): diferencial bit-exact vs refs/tokens.json"
```

---

## Task 13: Expandir o corpus via oráculo

**Files:**
- Modify: `oracle/` (fonte que define os casos) e/ou `scripts/`
- Modify: `refs/tokens.json`

> Decisão (c) do spec: ampliar o corpus com casos que estressam o tokenizer.

- [ ] **Step 1: Inspecionar como o oráculo gera tokens hoje**

Run: `sed -n '1,80p' oracle/src/runner.rs; echo ---; grep -rn "tokens.json\|llama-tokenize\|cases\|ids" oracle/src/ scripts/`
Expected: localizar onde os casos são definidos e como `refs/tokens.json` é escrito.

- [ ] **Step 2: Adicionar casos de estresse à lista do oráculo**

Adicionar estes textos à fonte que gera o corpus (onde hoje vivem os 4 casos):

```
"  leading spaces"
"trailing spaces   "
"multiple    internal    spaces"
"café résumé naïve"
"Tab\tand\nnewline"
"MiXeD CaSe 123!?"
"."
"123456789"
```

(Cada caso é tokenizado por `build-oracle/bin/llama-tokenize -m
models/stories260K.gguf -p "<texto>"` e o array de ids capturado, exatamente
como os 4 casos atuais.)

- [ ] **Step 3: Regenerar `refs/tokens.json`**

Run: `cargo run -p oracle` (ou o comando que a Fase 0 usa para gerar refs)
Expected: `refs/tokens.json` regenerado. Verificar `git diff refs/tokens.json`.

- [ ] **Step 4: Rodar o diferencial sobre o corpus ampliado**

Run: `cargo test -p llama-tokenizer --test oracle_corpus -- --nocapture`
Expected: PASS em todos os casos. Divergências → depurar `spm.rs` (não ajustar o teste).

- [ ] **Step 5: Commit**

```bash
git add refs/tokens.json scripts/ oracle/
git commit -m "test(tokenizer): corpus ampliado via oráculo (byte-fallback, espaços, unicode)"
```

---

## Task 14: Gate de qualidade — fmt, clippy, cobertura

**Files:**
- (possivelmente `scripts/gate.sh`, se ele rodar por crate)

- [ ] **Step 1: fmt + clippy do workspace**

Run:
```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```
Expected: ambos limpos. Corrigir o que aparecer (sem `#[allow]` injustificado).

- [ ] **Step 2: Suíte completa**

Run: `cargo test --workspace`
Expected: todos verdes (gguf, llama-tokenizer, oracle).

- [ ] **Step 3: Cobertura ≥80% nos crates novos**

Run:
```bash
cargo llvm-cov --package gguf --fail-under-lines 80
cargo llvm-cov --package llama-tokenizer --fail-under-lines 80
```
Expected: ambos ≥80%. Se faltar, adicionar testes unitários focados nas linhas
descobertas (ex.: acessores de erro, branches de `read_value`).

- [ ] **Step 4: Confirmar que `gate.sh` cobre os novos crates**

Run: `cat scripts/gate.sh`
Se o gate roda por crate específico, incluir `gguf` e `llama-tokenizer`. Se roda
`--workspace`, nada a fazer.

- [ ] **Step 5: Commit (se houve mudança)**

```bash
git add -A
git commit -m "chore: gate de qualidade da Fase 1 (fmt/clippy/cobertura ≥80%)"
```

---

## Task 15: Revisão de código e fechamento

- [ ] **Step 1: rust-review**

Invocar a skill `rust-review` sobre o diff da branch. Endereçar todo issue
CRITICAL/HIGH antes de fechar a fase.

- [ ] **Step 2: verification-before-completion**

Invocar `superpowers:verification-before-completion`: reexecutar
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
`cargo test --workspace` e os dois `cargo llvm-cov --fail-under-lines 80`,
colando a saída real como evidência.

- [ ] **Step 3: Fechar a fase**

Invocar `superpowers:finishing-a-development-branch` e abrir PR via `/pr` da
branch `fase-1-gguf-tokenizer` para `master`.

---

## Self-Review (preenchido pelo autor do plano)

**Cobertura do spec:**
- Parser GGUF v3 sobre `&[u8]`, zero-unsafe → Tasks 1,4,5,6 (`#![forbid(unsafe_code)]` no Step 3 do Task 0).
- `MetadataValue`/acessores → Task 3. `GgmlType`/tabela de blocos → Task 2.
- Acesso raw aos tensores (sem dequant) → Task 6.
- Tokenizer SPM (merge-by-score + byte-fallback) → Tasks 9,10,11.
- `encode` (critério) + `decode` → Task 11.
- Validação: tokens bit-exact → Tasks 12,13; metadados (snapshot + arrays + transitivo) → Tasks 7,8.
- proptest no parser → Task 7. Cobertura ≥80% → Task 14. rust-review → Task 15.
- Workspace `members += ...` → Task 0.

**Placeholders:** nenhum "TBD/TODO"; código completo em cada step. Os pontos de
risco (desempate SPM, `encode` sem BOS, contagem de tensores) têm instrução
explícita de depurar contra o upstream, não de relaxar o teste.

**Consistência de tipos:** `GgufFile::parse`, `GgufFile::get`, `tensor_data`,
`MetadataValue::as_*`/`array_len`, `Vocab::new`/`from_gguf`/`text_to_token`/
`byte_to_token`/`score`/`token_text`, `Tokenizer::new`/`from_gguf`/`encode`/
`decode`, `tokenize_spm` — assinaturas estáveis entre as tasks que as definem e
as que as consomem.

**Seam conhecido:** `GgmlType` em `gguf` (decisão (b)); mover para `ggml-core` é
trabalho da Fase 2.
