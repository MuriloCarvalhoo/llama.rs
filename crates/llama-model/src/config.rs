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
    /// Lê e valida os escalares do GGUF.
    /// Detecta o prefixo de arquitetura via `general.architecture` (ex: `llama`, `qwen2`).
    pub fn from_gguf(f: &GgufFile) -> Result<Self, ModelError> {
        let arch = match f.metadata.get("general.architecture") {
            Some(MetadataValue::String(s)) => s.clone(),
            _ => "llama".to_owned(),
        };
        let p = |suffix: &str| format!("{arch}.{suffix}");

        let u = |k: &str| -> Result<usize, ModelError> {
            let v = f.get(k)?.as_u32(k)?;
            usize::try_from(v).map_err(|_| ModelError::Overflow)
        };
        let n_embd = u(&p("embedding_length"))?;
        let n_head = u(&p("attention.head_count"))?;
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
        // freq_base é opcional; default 10000.
        let freq_base = match f.metadata.get(&p("rope.freq_base")) {
            Some(MetadataValue::F32(v)) => *v,
            _ => 10000.0,
        };
        // rope_dim é opcional; default head_dim quando ausente (ex: Qwen2).
        let rope_dim = match f.metadata.get(&p("rope.dimension_count")) {
            Some(v) => usize::try_from(v.as_u32("rope_dim")?).map_err(|_| ModelError::Overflow)?,
            None => head_dim,
        };
        Ok(Self {
            n_embd,
            n_layer: u(&p("block_count"))?,
            n_head,
            n_head_kv: u(&p("attention.head_count_kv"))?,
            head_dim,
            n_ff: u(&p("feed_forward_length"))?,
            rope_dim,
            rms_eps: f
                .get(&p("attention.layer_norm_rms_epsilon"))?
                .as_f32("rms")?,
            freq_base,
            vocab,
            ctx: u(&p("context_length"))?,
            bos_id: f.get("tokenizer.ggml.bos_token_id")?.as_u32("bos")?,
            eos_id: f.get("tokenizer.ggml.eos_token_id")?.as_u32("eos")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_stories() -> Option<GgufFile> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        GgufFile::parse(&bytes).ok()
    }

    fn load_qwen() -> Option<GgufFile> {
        let bytes =
            std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")).ok()?;
        GgufFile::parse(&bytes).ok()
    }

    #[test]
    fn reads_stories260k_config() {
        let Some(f) = load_stories() else {
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

    #[test]
    fn reads_qwen2_config() {
        let Some(f) = load_qwen() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let c = LlamaConfig::from_gguf(&f).unwrap();
        assert_eq!(c.n_embd, 896);
        assert_eq!(c.n_layer, 24);
        assert_eq!(c.n_head, 14);
        assert_eq!(c.n_head_kv, 2);
        assert_eq!(c.head_dim, 64); // 896 / 14
        assert_eq!(c.n_ff, 4864);
        assert_eq!(c.rope_dim, 64); // default head_dim (chave ausente no GGUF)
        assert!((c.freq_base - 1_000_000.0).abs() < 1.0);
        assert!(c.vocab > 0);
    }
}
