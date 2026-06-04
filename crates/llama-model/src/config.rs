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
