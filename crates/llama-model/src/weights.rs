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
    #![allow(clippy::indexing_slicing)]
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
