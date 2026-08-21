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
        !(il + 1).is_multiple_of(self.full_attn_interval)
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
    /// Se o prompt leva BOS. Vem de `tokenizer.ggml.add_bos_token`; quando a chave não
    /// existe o padrão segue o tipo de vocabulário — SPM adiciona, BPE não. Forçar BOS
    /// num modelo treinado sem ele degrada a geração.
    pub add_bos: bool,
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
        let add_bos = match f.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(MetadataValue::Bool(b)) => *b,
            _ => !matches!(
                f.metadata.get("tokenizer.ggml.model"),
                Some(MetadataValue::String(m)) if m == "gpt2"
            ),
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
            add_bos,
        })
    }
}

#[cfg(test)]
mod synthetic_tests {
    //! Testes de `LlamaConfig::from_gguf` sobre bytes GGUF sintéticos (sem ler nenhum
    //! `.gguf` do disco) — cobrem especificamente a distinção MTP/NextN e o mixer
    //! híbrido (`qwen35`/Qwen3.8) que os testes em `tests` (abaixo) só exercitam se
    //! houver um modelo real em `models/`.
    use super::*;

    struct MiniGguf {
        kv: Vec<u8>,
        kv_count: u64,
    }

    impl MiniGguf {
        fn new() -> Self {
            Self {
                kv: Vec::new(),
                kv_count: 0,
            }
        }

        fn push_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        fn str(mut self, key: &str, val: &str) -> Self {
            Self::push_str(&mut self.kv, key);
            self.kv.extend_from_slice(&8u32.to_le_bytes());
            Self::push_str(&mut self.kv, val);
            self.kv_count += 1;
            self
        }

        fn u32(mut self, key: &str, val: u32) -> Self {
            Self::push_str(&mut self.kv, key);
            self.kv.extend_from_slice(&4u32.to_le_bytes());
            self.kv.extend_from_slice(&val.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn f32(mut self, key: &str, val: f32) -> Self {
            Self::push_str(&mut self.kv, key);
            self.kv.extend_from_slice(&6u32.to_le_bytes());
            self.kv.extend_from_slice(&val.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn str_array(mut self, key: &str, vals: &[&str]) -> Self {
            Self::push_str(&mut self.kv, key);
            self.kv.extend_from_slice(&9u32.to_le_bytes());
            self.kv.extend_from_slice(&8u32.to_le_bytes());
            self.kv
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                Self::push_str(&mut self.kv, v);
            }
            self.kv_count += 1;
            self
        }

        fn build(&self) -> GgufFile {
            let mut out = Vec::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.kv);
            GgufFile::parse(&out).unwrap()
        }
    }

    /// Config densa mínima válida (arquitetura `llama`, 2 camadas, sem MTP/delta-net).
    fn dense_base() -> MiniGguf {
        MiniGguf::new()
            .str("general.architecture", "llama")
            .u32("llama.embedding_length", 8)
            .u32("llama.attention.head_count", 2)
            .u32("llama.attention.head_count_kv", 2)
            .u32("llama.feed_forward_length", 16)
            .u32("llama.context_length", 128)
            .u32("llama.block_count", 2)
            .f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .str_array("tokenizer.ggml.tokens", &["<unk>", "<s>", "</s>"])
            .u32("tokenizer.ggml.bos_token_id", 1)
            .u32("tokenizer.ggml.eos_token_id", 2)
    }

    #[test]
    fn dense_minimal_config_parses() {
        let f = dense_base().build();
        let c = LlamaConfig::from_gguf(&f).unwrap();
        assert_eq!(c.n_embd, 8);
        assert_eq!(c.n_layer, 2);
        assert_eq!(c.head_dim, 4); // n_embd / n_head, sem attention.key_length
        assert_eq!(c.n_layer_nextn, 0);
        assert!(c.delta_net.is_none());
    }

    #[test]
    fn nextn_predict_layers_removidos_de_n_layer() {
        // block_count=6, nextn_predict_layers=2 → n_layer efetivo = 4 (os 2 blocos MTP
        // no fim não entram no forward normal). Ver comentário em `from_gguf`.
        let f = dense_base()
            .u32("llama.block_count", 6)
            .u32("llama.nextn_predict_layers", 2)
            .build();
        let c = LlamaConfig::from_gguf(&f).unwrap();
        assert_eq!(c.n_layer, 4);
        assert_eq!(c.n_layer_nextn, 2);
    }

    #[test]
    fn nextn_maior_que_block_count_e_erro() {
        let f = dense_base()
            .u32("llama.block_count", 2)
            .u32("llama.nextn_predict_layers", 3)
            .build();
        assert!(LlamaConfig::from_gguf(&f).is_err());
    }

    #[test]
    fn hybrid_delta_net_detectado_via_ssm_conv_kernel() {
        // Presença de `ssm.conv_kernel` liga o mixer híbrido (qwen35/Qwen3.8): 3 em
        // cada 4 camadas viram atenção linear (delta net), sem KV-cache.
        let f = dense_base()
            .u32("llama.ssm.conv_kernel", 4)
            .u32("llama.ssm.inner_size", 8)
            .u32("llama.ssm.state_size", 2)
            .u32("llama.ssm.time_step_rank", 2)
            .u32("llama.ssm.group_count", 1)
            .build();
        let c = LlamaConfig::from_gguf(&f).unwrap();
        let dn = c.delta_net.expect("delta_net deveria estar presente");
        assert_eq!(dn.full_attn_interval, 4); // default quando a chave não existe
        assert!(dn.eh_linear(0)); // camada 0 é linear (delta)
        assert!(!dn.eh_linear(3)); // camada 3 (il+1 == 4) é atenção completa
    }

    #[test]
    fn n_head_zero_e_erro() {
        let f = dense_base().u32("llama.attention.head_count", 0).build();
        assert!(matches!(
            LlamaConfig::from_gguf(&f),
            Err(ModelError::Config(_))
        ));
    }

    #[test]
    fn attention_key_length_sobrepoe_head_dim_derivado() {
        // Qwen3.8-27B: head_dim=256 explícito não bate com n_embd/n_head.
        let f = dense_base().u32("llama.attention.key_length", 256).build();
        let c = LlamaConfig::from_gguf(&f).unwrap();
        assert_eq!(c.head_dim, 256);
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
