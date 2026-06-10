//! Backend GPU para o passo de decode. Os matmuls do decode são roteados para a GPU;
//! o forward em si (RMSNorm/RoPE/attention/SwiGLU) permanece em `model.rs`.

use crate::config::LlamaConfig;
use crate::error::ModelError;
use gguf::{GgmlType, GgufFile};

/// Pesos Q8_0 por camada, em bytes raw lidos do GGUF (cópia própria da GPU).
pub struct GpuLayerRaw {
    pub attn_q: Vec<u8>,
    pub attn_k: Vec<u8>,
    pub attn_v: Vec<u8>,
    pub attn_output: Vec<u8>,
    pub ffn_gate: Vec<u8>,
    pub ffn_up: Vec<u8>,
    pub ffn_down: Vec<u8>,
}

/// Todos os pesos Q8_0 que o decode envia à GPU.
pub struct GpuRawWeights {
    pub layers: Vec<GpuLayerRaw>,
    pub output: Vec<u8>,
}

impl GpuRawWeights {
    /// Lê e valida os pesos Q8_0 do GGUF. Erro se algum tensor não for Q8_0
    /// ou tiver `n_in % 32 != 0` (incompatível com o shader matvec wave64).
    pub fn from_gguf(f: &GgufFile, bytes: &[u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let kv_dim = cfg.n_head_kv * cfg.head_dim;

        let read = |name: &str, n_in: usize, n_out: usize| -> Result<Vec<u8>, ModelError> {
            let info = f
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| ModelError::Gpu(format!("tensor {name} ausente")))?;
            if info.ggml_type != GgmlType::Q8_0 {
                return Err(ModelError::Gpu(format!(
                    "tensor {name} não é Q8_0 (é {:?}) — GPU exige Q8_0",
                    info.ggml_type
                )));
            }
            if n_in % 32 != 0 {
                return Err(ModelError::Gpu(format!(
                    "tensor {name}: n_in={n_in} não é múltiplo de 32"
                )));
            }
            let raw = f
                .tensor_data(bytes, info)
                .map_err(|e| ModelError::Gpu(e.to_string()))?;
            let expected = n_out * (n_in / 32) * 34;
            if raw.len() != expected {
                return Err(ModelError::Gpu(format!(
                    "tensor {name}: {} bytes, esperado {expected}",
                    raw.len()
                )));
            }
            Ok(raw.to_vec())
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            layers.push(GpuLayerRaw {
                attn_q: read(&p("attn_q.weight"), cfg.n_embd, cfg.n_embd)?,
                attn_k: read(&p("attn_k.weight"), cfg.n_embd, kv_dim)?,
                attn_v: read(&p("attn_v.weight"), cfg.n_embd, kv_dim)?,
                attn_output: read(&p("attn_output.weight"), cfg.n_embd, cfg.n_embd)?,
                ffn_gate: read(&p("ffn_gate.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_up: read(&p("ffn_up.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_down: read(&p("ffn_down.weight"), cfg.n_ff, cfg.n_embd)?,
            });
        }
        let output = read("output.weight", cfg.n_embd, cfg.vocab)?;
        Ok(Self { layers, output })
    }
}

/// Multiplicação matriz-vetor Q8_0 executada na GPU.
///
/// `w_bytes`: pesos Q8_0 row-major, `n_out × (n_in/32 × 34)` bytes.
/// `x`: ativações f32 de tamanho `n_in`.
/// Retorna `y` de tamanho `n_out`.
pub trait GpuMatmul {
    fn matvec_q8_0(
        &self,
        w_bytes: &[u8],
        x: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, ModelError>;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;
    use crate::config::LlamaConfig;
    use gguf::GgufFile;
    use std::path::Path;

    fn load_qwen() -> Option<(Vec<u8>, GgufFile, LlamaConfig)> {
        let bytes =
            std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        let cfg = LlamaConfig::from_gguf(&f).ok()?;
        Some((bytes, f, cfg))
    }

    #[test]
    fn gpu_raw_weights_extrai_todas_as_camadas() {
        let Some((bytes, f, cfg)) = load_qwen() else {
            eprintln!("qwen ausente — pulando");
            return;
        };
        let w = GpuRawWeights::from_gguf(&f, &bytes, &cfg).expect("from_gguf falhou");
        assert_eq!(w.layers.len(), cfg.n_layer);
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let row_bytes_q = (cfg.n_embd / 32) * 34;
        assert_eq!(w.layers[0].attn_q.len(), cfg.n_embd * row_bytes_q);
        assert_eq!(w.layers[0].attn_k.len(), kv_dim * row_bytes_q);
        assert_eq!(w.output.len(), cfg.vocab * row_bytes_q);
        eprintln!("GpuRawWeights OK — {} camadas", w.layers.len());
    }
}
