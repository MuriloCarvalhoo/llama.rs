//! Hiperparâmetros da arquitetura Llama, lidos do GGUF.

use gguf::{GgufFile, MetadataValue};

use crate::error::ModelError;

/// Parâmetros das camadas de atenção linear (gated delta net) das arquiteturas híbridas
/// tipo `qwen35`. Ver `docs/qwen35-arquitetura.md`.
///
/// Nessas camadas não há KV-cache: o histórico vive num estado `d_state × d_state` por
/// cabeça, de tamanho fixo. As camadas de atenção completa continuam usando o KV-cache
/// normal, uma a cada `full_attn_interval`.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaNetConfig {
    /// Tamanho do kernel da convolução causal aplicada a q|k|v (`ssm.conv_kernel`).
    pub d_conv: usize,
    /// Largura interna: `d_inner / n_v_heads` é a dimensão de cada cabeça de valor.
    pub d_inner: usize,
    /// Dimensão de cada cabeça de chave **e** de valor (`ssm.state_size`).
    pub d_state: usize,
    /// Cabeças de valor (`ssm.time_step_rank`).
    pub n_v_heads: usize,
    /// Cabeças de chave (`ssm.group_count`); divide `n_v_heads`, como em GQA.
    pub n_k_heads: usize,
    /// Uma camada em cada `full_attn_interval` é de atenção completa; as demais são
    /// lineares. Camada `il` é linear quando `(il + 1) % full_attn_interval != 0`.
    pub full_attn_interval: usize,
}

impl DeltaNetConfig {
    /// `true` se a camada `il` é de atenção linear (recorrente).
    #[must_use]
    pub fn eh_linear(&self, il: usize) -> bool {
        (il + 1) % self.full_attn_interval != 0
    }

    /// Dimensão de cada cabeça de valor.
    #[must_use]
    pub fn head_v_dim(&self) -> usize {
        self.d_inner / self.n_v_heads
    }

    /// Floats do estado recorrente de uma camada.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.d_state * self.d_state * self.n_v_heads
    }
}

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
    /// `Some` nas arquiteturas híbridas (`qwen35`); `None` no transformer denso.
    pub delta_net: Option<DeltaNetConfig>,
    /// Blocos MTP/NextN empilhados depois das `n_layer` camadas. Carregados pelo
    /// llama.cpp para speculative decoding e **ignorados** aqui: não participam do
    /// forward normal.
    pub n_layer_nextn: usize,
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
        if n_head == 0 {
            return Err(ModelError::Config("n_head é zero".into()));
        }
        // `attention.key_length` manda quando existe: no Qwen3.8-27B a cabeça tem 256
        // dimensões com n_embd=5120 e n_head=24, e `n_embd / n_head` nem sequer é inteiro.
        let head_dim = match f.metadata.get(&p("attention.key_length")) {
            Some(v) => {
                usize::try_from(v.as_u32("key_length")?).map_err(|_| ModelError::Overflow)?
            }
            None => {
                if n_embd % n_head != 0 {
                    return Err(ModelError::Config(
                        "n_head não divide n_embd e falta attention.key_length".into(),
                    ));
                }
                n_embd / n_head
            }
        };
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
        // Arquitetura híbrida: as chaves `ssm.*` só existem onde há atenção linear.
        let delta_net = if f.metadata.contains_key(&p("ssm.conv_kernel")) {
            let interval = match f.metadata.get(&p("full_attention_interval")) {
                Some(v) => usize::try_from(v.as_u32("full_attention_interval")?)
                    .map_err(|_| ModelError::Overflow)?,
                None => 4,
            };
            if interval == 0 {
                return Err(ModelError::Config("full_attention_interval é zero".into()));
            }
            Some(DeltaNetConfig {
                d_conv: u(&p("ssm.conv_kernel"))?,
                d_inner: u(&p("ssm.inner_size"))?,
                d_state: u(&p("ssm.state_size"))?,
                n_v_heads: u(&p("ssm.time_step_rank"))?,
                n_k_heads: u(&p("ssm.group_count"))?,
                full_attn_interval: interval,
            })
        } else {
            None
        };
        let n_layer_nextn = match f.metadata.get(&p("nextn_predict_layers")) {
            Some(v) => usize::try_from(v.as_u32("nextn_predict_layers")?)
                .map_err(|_| ModelError::Overflow)?,
            None => 0,
        };

        // `block_count` conta os blocos MTP/NextN empilhados no fim, que existem para
        // speculative decoding e não participam do forward normal — o llama.cpp faz a
        // mesma distinção entre `n_layer_all` e `n_layer()`.
        let n_layer = u(&p("block_count"))?
            .checked_sub(n_layer_nextn)
            .ok_or_else(|| ModelError::Config("nextn_predict_layers >= block_count".into()))?;

        Ok(Self {
            n_embd,
            n_layer,
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
            delta_net,
            n_layer_nextn,
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
