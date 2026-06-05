//! Pesos quantizados do GGUF armazenados em bytes raw; dequantizados sob demanda.

use std::sync::OnceLock;

use ggml_cpu::dequant_to_f32 as dequant_impl;
use gguf::{GgufFile, TensorInfo};

use crate::ops::{
    matmul, matmul_q8_0, matmul_q8_0_actq, matmul_q8_0_actq_packed, quantize_q8_0_split,
    repack_q8_0_8rows,
};

use crate::config::LlamaConfig;
use crate::error::ModelError;

/// Tensor raw: bytes tal como lidos do GGUF + tipo de dado para dequant.
/// Primeira chamada a `dequant_to_f32` dequantiza e cacheia em memória;
/// chamadas subsequentes retornam `&[f32]` sem realocar.
pub(crate) struct RawTensor {
    pub bytes: Vec<u8>,
    pub ty: gguf::GgmlType,
    f32_cache: OnceLock<Vec<f32>>,
    /// Pesos Q8_0 reempacotados em block_q8_0x8 (272 bytes por grupo de 8 linhas por bloco).
    /// Some apenas quando ty==Q8_0, n_out%8==0, n_in%32==0.
    repacked: Option<Vec<u8>>,
    n_out_packed: usize,
}

impl RawTensor {
    pub(crate) fn new(bytes: Vec<u8>, ty: gguf::GgmlType) -> Self {
        Self {
            bytes,
            ty,
            f32_cache: OnceLock::new(),
            repacked: None,
            n_out_packed: 0,
        }
    }

    /// Constrói RawTensor com pesos Q8_0 reempacotados para melhor eficiência de cache.
    /// Executado uma vez no carregamento do modelo; n_in e n_out são dimensões da matriz de pesos.
    pub(crate) fn new_with_repack(
        bytes: Vec<u8>,
        ty: gguf::GgmlType,
        n_in: usize,
        n_out: usize,
    ) -> Self {
        if ty == gguf::GgmlType::Q8_0 && n_out % 8 == 0 && n_in % 32 == 0 {
            let packed = repack_q8_0_8rows(&bytes, n_in, n_out);
            // Descarta bytes originais: apenas o packed é usado em matmul, liberar reduz
            // pressão de memória e melhora eficiência do cache durante inferência.
            Self {
                bytes: Vec::new(),
                ty,
                f32_cache: OnceLock::new(),
                repacked: Some(packed),
                n_out_packed: n_out,
            }
        } else if ty == gguf::GgmlType::Q8_0 && n_in % 32 == 0 {
            // n_out não múltiplo de 8 — repack apenas as primeiras linhas completas
            let n_out_packed = (n_out / 8) * 8;
            if n_out_packed > 0 {
                let packed = repack_q8_0_8rows(&bytes, n_in, n_out_packed);
                Self {
                    bytes,
                    ty,
                    f32_cache: OnceLock::new(),
                    repacked: Some(packed),
                    n_out_packed,
                }
            } else {
                Self::new(bytes, ty)
            }
        } else {
            Self::new(bytes, ty)
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

    /// Matmul otimizado por tipo de quantização.
    /// - Q8_0 + n_in múltiplo de 32: quantiza ativações → produto escalar i8×i8
    /// - Q8_0 outros: produto escalar f32×i8 sem expandir pesos
    /// - outros tipos: dequant → f32 → matmul
    pub(crate) fn matmul_into(
        &self,
        x: &[f32],
        n_in: usize,
        n_out: usize,
        n_tok: usize,
    ) -> Result<Vec<f32>, ModelError> {
        if self.ty == gguf::GgmlType::Q8_0 && n_in % 32 == 0 {
            let x_q8 = quantize_q8_0_split(x, n_in, n_tok);
            Ok(self.matmul_actq_dispatch(&x_q8, n_in, n_out, n_tok))
        } else if self.ty == gguf::GgmlType::Q8_0 {
            Ok(matmul_q8_0(&self.bytes, x, n_in, n_out, n_tok))
        } else {
            Ok(matmul(self.dequant_to_f32()?, x, n_in, n_out, n_tok))
        }
    }

    /// Variante de matmul_into que aceita ativações pré-quantizadas para evitar
    /// re-quantização quando o mesmo vetor de entrada é usado em múltiplos matmuls.
    ///
    /// - `x_q8`: saída de `quantize_q8_0_split` — usada se Q8_0 e n_in%32==0
    /// - `x_f32`: ativações originais — usadas como fallback se n_in%32!=0 ou tipo não é Q8_0
    pub(crate) fn matmul_into_with_q8(
        &self,
        x_q8: &[u8],
        x_f32: &[f32],
        n_in: usize,
        n_out: usize,
        n_tok: usize,
    ) -> Result<Vec<f32>, ModelError> {
        if self.ty == gguf::GgmlType::Q8_0 && n_in % 32 == 0 {
            Ok(self.matmul_actq_dispatch(x_q8, n_in, n_out, n_tok))
        } else if self.ty == gguf::GgmlType::Q8_0 {
            Ok(matmul_q8_0(&self.bytes, x_f32, n_in, n_out, n_tok))
        } else {
            Ok(matmul(self.dequant_to_f32()?, x_f32, n_in, n_out, n_tok))
        }
    }

    /// Dispatcher interno: usa repacked quando disponível, fallback para actq padrão.
    fn matmul_actq_dispatch(
        &self,
        x_q8: &[u8],
        n_in: usize,
        n_out: usize,
        n_tok: usize,
    ) -> Vec<f32> {
        if let Some(packed) = &self.repacked {
            let n_packed = self.n_out_packed;
            let const_b: usize = 34;
            let row_bytes = (n_in / 32) * const_b;
            let tail_start = n_packed * row_bytes;
            // bytes pode ser vazio (descartado após repack completo) — tail só existe quando
            // n_out_packed < n_out (repack parcial), o que implica bytes ainda estão presentes.
            let w_tail = self.bytes.get(tail_start..).unwrap_or(&[]);
            matmul_q8_0_actq_packed(packed, w_tail, x_q8, n_in, n_out, n_packed, n_tok)
        } else {
            matmul_q8_0_actq(&self.bytes, x_q8, n_in, n_out, n_tok)
        }
    }
}

/// Pesos de uma camada transformer.
pub(crate) struct LayerWeights {
    pub attn_norm: RawTensor,
    pub attn_q: RawTensor,
    pub attn_k: RawTensor,
    pub attn_v: RawTensor,
    /// Bias de atenção Q/K/V — presente em Qwen2, ausente em Llama.
    pub attn_q_bias: Option<RawTensor>,
    pub attn_k_bias: Option<RawTensor>,
    pub attn_v_bias: Option<RawTensor>,
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

fn tensor_raw_repack(
    f: &GgufFile,
    bytes: &[u8],
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<RawTensor, ModelError> {
    let info: &TensorInfo = f
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))?;
    let raw = f.tensor_data(bytes, info)?;
    Ok(RawTensor::new_with_repack(
        raw.to_vec(),
        info.ggml_type,
        n_in,
        n_out,
    ))
}

fn tensor_raw_opt(f: &GgufFile, bytes: &[u8], name: &str) -> Result<Option<RawTensor>, ModelError> {
    match f.tensors.iter().find(|t| t.name == name) {
        Some(info) => {
            let raw = f.tensor_data(bytes, info)?;
            Ok(Some(RawTensor::new(raw.to_vec(), info.ggml_type)))
        }
        None => Ok(None),
    }
}

impl Weights {
    /// Lê todos os tensores (qualquer tipo suportado pelo dispatcher de dequant).
    pub fn from_gguf(f: &GgufFile, bytes: &[u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let n_kv_dim = cfg.n_head_kv * cfg.head_dim;
        let vocab = cfg.vocab;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |suffix: &str| format!("blk.{l}.{suffix}");
            layers.push(LayerWeights {
                attn_norm: tensor_raw(f, bytes, &p("attn_norm.weight"))?,
                attn_q: tensor_raw_repack(f, bytes, &p("attn_q.weight"), n_embd, n_embd)?,
                attn_k: tensor_raw_repack(f, bytes, &p("attn_k.weight"), n_embd, n_kv_dim)?,
                attn_v: tensor_raw_repack(f, bytes, &p("attn_v.weight"), n_embd, n_kv_dim)?,
                attn_q_bias: tensor_raw_opt(f, bytes, &p("attn_q.bias"))?,
                attn_k_bias: tensor_raw_opt(f, bytes, &p("attn_k.bias"))?,
                attn_v_bias: tensor_raw_opt(f, bytes, &p("attn_v.bias"))?,
                attn_output: tensor_raw_repack(f, bytes, &p("attn_output.weight"), n_embd, n_embd)?,
                ffn_norm: tensor_raw(f, bytes, &p("ffn_norm.weight"))?,
                ffn_gate: tensor_raw_repack(f, bytes, &p("ffn_gate.weight"), n_embd, n_ff)?,
                ffn_up: tensor_raw_repack(f, bytes, &p("ffn_up.weight"), n_embd, n_ff)?,
                ffn_down: tensor_raw_repack(f, bytes, &p("ffn_down.weight"), n_ff, n_embd)?,
            });
        }
        Ok(Self {
            token_embd: tensor_raw(f, bytes, "token_embd.weight")?,
            layers,
            output_norm: tensor_raw(f, bytes, "output_norm.weight")?,
            output: tensor_raw_repack(f, bytes, "output.weight", n_embd, vocab)?,
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
    fn qwen2_loads_attn_biases() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf"))
        else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let f = GgufFile::parse(&bytes).unwrap();
        let cfg = LlamaConfig::from_gguf(&f).unwrap();
        let w = Weights::from_gguf(&f, &bytes, &cfg).unwrap();
        let l0 = &w.layers[0];
        assert!(l0.attn_q_bias.is_some(), "Qwen2 deve ter attn_q_bias");
        assert!(l0.attn_k_bias.is_some(), "Qwen2 deve ter attn_k_bias");
        assert!(l0.attn_v_bias.is_some(), "Qwen2 deve ter attn_v_bias");
        assert_eq!(l0.attn_q_bias.as_ref().unwrap().n_elements(), cfg.n_embd);
        assert_eq!(
            l0.attn_k_bias.as_ref().unwrap().n_elements(),
            cfg.n_head_kv * cfg.head_dim
        );
    }

    #[test]
    fn stories260k_has_no_attn_biases() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let f = GgufFile::parse(&bytes).unwrap();
        let cfg = LlamaConfig::from_gguf(&f).unwrap();
        let w = Weights::from_gguf(&f, &bytes, &cfg).unwrap();
        assert!(w.layers[0].attn_q_bias.is_none());
        assert!(w.layers[0].attn_k_bias.is_none());
        assert!(w.layers[0].attn_v_bias.is_none());
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
