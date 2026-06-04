//! Pesos quantizados do GGUF armazenados em bytes raw; dequantizados sob demanda.

use std::cell::OnceCell;

use ggml_cpu::dequant_to_f32 as dequant_impl;
use gguf::{GgufFile, TensorInfo};

use crate::config::LlamaConfig;
use crate::error::ModelError;

/// Tensor raw: bytes tal como lidos do GGUF + tipo de dado para dequant.
/// Primeira chamada a `dequant_to_f32` dequantiza e cacheia em memória;
/// chamadas subsequentes retornam `&[f32]` sem realocar.
pub(crate) struct RawTensor {
    pub bytes: Vec<u8>,
    pub ty: gguf::GgmlType,
    f32_cache: OnceCell<Vec<f32>>,
}

impl RawTensor {
    pub(crate) fn new(bytes: Vec<u8>, ty: gguf::GgmlType) -> Self {
        Self {
            bytes,
            ty,
            f32_cache: OnceCell::new(),
        }
    }

    /// Número de elementos lógicos (não de bytes).
    pub fn n_elements(&self) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let bs = self.ty.block_size() as usize;
        #[allow(clippy::cast_possible_truncation)]
        let ts = self.ty.type_size() as usize;
        if ts == 0 {
            return 0;
        }
        (self.bytes.len() / ts) * bs
    }

    /// Bytes raw (footprint de RAM — quantizado, sem dequant).
    pub fn memory_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Dequantiza para f32 e cacheia. Primeira chamada: O(n). Subsequentes: O(1).
    pub fn dequant_to_f32(&self) -> Result<&[f32], ModelError> {
        if let Some(cached) = self.f32_cache.get() {
            return Ok(cached.as_slice());
        }
        let v = dequant_impl(&self.bytes, self.ty).map_err(ModelError::from)?;
        let _ = self.f32_cache.set(v);
        // SAFETY: we just set the value above; get() is guaranteed to return Some.
        Ok(self.f32_cache.get().map_or(&[], Vec::as_slice))
    }
}

/// Pesos de uma camada transformer.
pub(crate) struct LayerWeights {
    pub attn_norm: RawTensor,
    pub attn_q: RawTensor,
    pub attn_k: RawTensor,
    pub attn_v: RawTensor,
    pub attn_output: RawTensor,
    pub ffn_norm: RawTensor,
    pub ffn_gate: RawTensor,
    pub ffn_up: RawTensor,
    pub ffn_down: RawTensor,
}

/// Todos os pesos do modelo, em bytes raw.
pub(crate) struct Weights {
    pub token_embd: RawTensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: RawTensor,
    pub output: RawTensor,
}

fn tensor_raw(f: &GgufFile, bytes: &[u8], name: &str) -> Result<RawTensor, ModelError> {
    let info: &TensorInfo = f
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))?;
    let raw = f.tensor_data(bytes, info)?;
    Ok(RawTensor::new(raw.to_vec(), info.ggml_type))
}

impl Weights {
    /// Lê todos os tensores (qualquer tipo suportado pelo dispatcher de dequant).
    pub fn from_gguf(f: &GgufFile, bytes: &[u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |suffix: &str| format!("blk.{l}.{suffix}");
            layers.push(LayerWeights {
                attn_norm: tensor_raw(f, bytes, &p("attn_norm.weight"))?,
                attn_q: tensor_raw(f, bytes, &p("attn_q.weight"))?,
                attn_k: tensor_raw(f, bytes, &p("attn_k.weight"))?,
                attn_v: tensor_raw(f, bytes, &p("attn_v.weight"))?,
                attn_output: tensor_raw(f, bytes, &p("attn_output.weight"))?,
                ffn_norm: tensor_raw(f, bytes, &p("ffn_norm.weight"))?,
                ffn_gate: tensor_raw(f, bytes, &p("ffn_gate.weight"))?,
                ffn_up: tensor_raw(f, bytes, &p("ffn_up.weight"))?,
                ffn_down: tensor_raw(f, bytes, &p("ffn_down.weight"))?,
            });
        }
        Ok(Self {
            token_embd: tensor_raw(f, bytes, "token_embd.weight")?,
            layers,
            output_norm: tensor_raw(f, bytes, "output_norm.weight")?,
            output: tensor_raw(f, bytes, "output.weight")?,
        })
    }

    /// Soma dos bytes raw de todos os tensores.
    pub fn memory_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|lw| {
                lw.attn_norm.memory_bytes()
                    + lw.attn_q.memory_bytes()
                    + lw.attn_k.memory_bytes()
                    + lw.attn_v.memory_bytes()
                    + lw.attn_output.memory_bytes()
                    + lw.ffn_norm.memory_bytes()
                    + lw.ffn_gate.memory_bytes()
                    + lw.ffn_up.memory_bytes()
                    + lw.ffn_down.memory_bytes()
            })
            .sum();
        self.token_embd.memory_bytes()
            + layer_bytes
            + self.output_norm.memory_bytes()
            + self.output.memory_bytes()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn dequant_cache_second_call_returns_same_pointer() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let f = GgufFile::parse(&bytes).unwrap();
        let cfg = LlamaConfig::from_gguf(&f).unwrap();
        let w = Weights::from_gguf(&f, &bytes, &cfg).unwrap();
        let ptr1 = w.token_embd.dequant_to_f32().unwrap().as_ptr();
        let ptr2 = w.token_embd.dequant_to_f32().unwrap().as_ptr();
        assert_eq!(
            ptr1, ptr2,
            "segunda chamada deve reusar a cache (mesmo ponteiro)"
        );
    }

    #[test]
    fn loads_all_weights_with_expected_element_counts() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let f = GgufFile::parse(&bytes).unwrap();
        let cfg = LlamaConfig::from_gguf(&f).unwrap();
        let w = Weights::from_gguf(&f, &bytes, &cfg).unwrap();
        assert_eq!(w.token_embd.n_elements(), cfg.vocab * cfg.n_embd);
        assert_eq!(w.output.n_elements(), cfg.vocab * cfg.n_embd);
        assert_eq!(w.output_norm.n_elements(), cfg.n_embd);
        assert_eq!(w.layers.len(), cfg.n_layer);
        let l0 = &w.layers[0];
        assert_eq!(l0.attn_q.n_elements(), cfg.n_embd * cfg.n_embd);
        assert_eq!(
            l0.attn_k.n_elements(),
            cfg.n_embd * cfg.n_head_kv * cfg.head_dim
        );
        assert_eq!(l0.ffn_gate.n_elements(), cfg.n_embd * cfg.n_ff);
        assert_eq!(l0.ffn_down.n_elements(), cfg.n_ff * cfg.n_embd);
    }
}
