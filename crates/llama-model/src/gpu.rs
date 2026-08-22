//! Backend GPU para o passo de decode. Os matmuls do decode são roteados para a GPU;
//! o forward em si (RMSNorm/RoPE/attention/SwiGLU) permanece em `model.rs`.

use crate::config::LlamaConfig;
use crate::error::ModelError;
use gguf::{GgmlType, GgufFile};

/// Um tensor de peso quantizado, emprestado do buffer do GGUF (sem cópia).
///
/// O tipo viaja junto porque modelos `*_K_M` são **mistos**: no Qwen2.5-32B-Q5_K_M,
/// `attn_v` e `ffn_down` são Q6_K em algumas camadas e Q5_K em outras, então o backend
/// escolhe o shader por tensor, não por modelo.
#[derive(Clone, Copy)]
pub struct QTensor<'a> {
    pub ty: GgmlType,
    pub bytes: &'a [u8],
}

/// O que fica entre as duas normas de uma camada.
///
/// Nas arquiteturas densas é sempre `Attn`. No `qwen35` três em cada quatro camadas são
/// `Delta` — atenção linear, sem KV-cache. Ver `docs/qwen35-arquitetura.md`.
pub enum MixerRaw<'a> {
    Attn {
        /// No `qwen35` esta projeção sai com `2 * head_dim` por cabeça: query e gate.
        attn_q: QTensor<'a>,
        attn_k: QTensor<'a>,
        attn_v: QTensor<'a>,
        attn_output: QTensor<'a>,
    },
    Delta {
        /// `key_dim * 2 + value_dim`, fatiado em q|k|v depois da convolução.
        attn_qkv: QTensor<'a>,
        /// `value_dim` — o `z` que fecha a norma gated na saída.
        attn_gate: QTensor<'a>,
        /// `value_dim -> n_embd`.
        ssm_out: QTensor<'a>,
    },
}

/// Pesos quantizados de uma camada, emprestados do buffer do GGUF (sem cópia).
pub struct GpuLayerRaw<'a> {
    pub mixer: MixerRaw<'a>,
    pub ffn_gate: QTensor<'a>,
    pub ffn_up: QTensor<'a>,
    pub ffn_down: QTensor<'a>,
}

impl<'a> GpuLayerRaw<'a> {
    /// Bytes quantizados da camada, como estão no GGUF. Serve ao backend para dimensionar
    /// a alocação de VRAM antes de subir peso nenhum.
    #[must_use]
    pub fn bytes_totais(&self) -> usize {
        let mixer = match &self.mixer {
            MixerRaw::Attn {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
            } => {
                attn_q.bytes.len()
                    + attn_k.bytes.len()
                    + attn_v.bytes.len()
                    + attn_output.bytes.len()
            }
            MixerRaw::Delta {
                attn_qkv,
                attn_gate,
                ssm_out,
            } => attn_qkv.bytes.len() + attn_gate.bytes.len() + ssm_out.bytes.len(),
        };
        mixer + self.ffn_gate.bytes.len() + self.ffn_up.bytes.len() + self.ffn_down.bytes.len()
    }

    /// Atalho para as camadas densas, onde o mixer é sempre atenção.
    ///
    /// # Panics
    /// Se chamada numa camada de atenção linear.
    #[must_use]
    // O panic é o contrato documentado acima: chamar isto numa camada linear é erro de código.
    #[allow(clippy::panic)]
    pub fn attn(&self) -> (&QTensor<'a>, &QTensor<'a>, &QTensor<'a>, &QTensor<'a>) {
        match &self.mixer {
            MixerRaw::Attn {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
            } => (attn_q, attn_k, attn_v, attn_output),
            MixerRaw::Delta { .. } => panic!("camada de atenção linear não tem q/k/v"),
        }
    }
}

/// Todos os pesos Q8_0 que o decode envia à GPU.
///
/// Os slices apontam para o buffer do modelo (tipicamente um mmap): copiá-los
/// custaria uma segunda imagem inteira dos pesos em RAM — 14.6 GB no 14B Q8_0,
/// o suficiente para o processo ser morto por OOM antes de chegar à VRAM.
pub struct GpuRawWeights<'a> {
    pub layers: Vec<GpuLayerRaw<'a>>,
    pub output: QTensor<'a>,
    /// Bloco de multi-token prediction, quando o modelo traz um. `None` no resto.
    pub mtp: Option<MtpRaw<'a>>,
}

/// Pesos quantizados do bloco MTP (`blk.{n_layer}.*`).
///
/// O bloco propõe o token **seguinte** ao que acabou de ser amostrado, a partir do hidden
/// state do último layer e do embedding desse token:
///
/// ```text
/// h  = eh_proj( [enorm(embed(t+1)) ; hnorm(h_t)] )   // 2*n_embd -> n_embd
/// h' = camada_de_decoder(h)
/// logits(t+2) = output( shared_head_norm(h') )
/// ```
pub struct MtpRaw<'a> {
    /// Combina embedding e hidden state — entra com `2 * n_embd`, sai com `n_embd`.
    pub eh_proj: QTensor<'a>,
    /// A camada de decoder do bloco, com a mesma forma das camadas normais.
    pub layer: GpuLayerRaw<'a>,
}

impl<'a> GpuRawWeights<'a> {
    /// Lê e valida os pesos quantizados do GGUF. Aceita Q8_0, Q5_K, Q6_K e Q4_K — os tipos
    /// para os quais existe shader de matvec. Valida a granularidade que cada um exige:
    /// 32 elementos por bloco no Q8_0, 256 por superbloco nos K-quants.
    pub fn from_gguf(f: &GgufFile, bytes: &'a [u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let _fase = crate::perfil_carga::Fase::nova("pesos crus (mmap)");
        let kv_dim = cfg.n_head_kv * cfg.head_dim;

        let read = |name: &str, n_in: usize, n_out: usize| -> Result<QTensor<'a>, ModelError> {
            let info = f
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| ModelError::Gpu(format!("tensor {name} ausente")))?;
            let (elems_per_block, block_bytes) = match info.ggml_type {
                GgmlType::Q8_0 => (32usize, 34usize),
                GgmlType::Q5_K => (256, 176),
                GgmlType::Q6_K => (256, 210),
                GgmlType::Q4_K => (256, 144),
                other => {
                    return Err(ModelError::Gpu(format!(
                        "tensor {name} é {other:?} — a GPU só tem shader para Q8_0, Q5_K, Q6_K e Q4_K"
                    )));
                }
            };
            if !n_in.is_multiple_of(elems_per_block) {
                return Err(ModelError::Gpu(format!(
                    "tensor {name}: n_in={n_in} não é múltiplo de {elems_per_block} ({:?})",
                    info.ggml_type
                )));
            }
            let raw = f
                .tensor_data(bytes, info)
                .map_err(|e| ModelError::Gpu(e.to_string()))?;
            let expected = n_out * (n_in / elems_per_block) * block_bytes;
            if raw.len() != expected {
                return Err(ModelError::Gpu(format!(
                    "tensor {name}: {} bytes, esperado {expected}",
                    raw.len()
                )));
            }
            Ok(QTensor {
                ty: info.ggml_type,
                bytes: raw,
            })
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let linear = cfg.delta_net.as_ref().is_some_and(|d| d.eh_linear(l));
            let mixer = if let (true, Some(dn)) = (linear, cfg.delta_net.as_ref()) {
                let key_dim = dn.d_state * dn.n_k_heads;
                let value_dim = dn.head_v_dim() * dn.n_v_heads;
                MixerRaw::Delta {
                    attn_qkv: read(&p("attn_qkv.weight"), cfg.n_embd, key_dim * 2 + value_dim)?,
                    attn_gate: read(&p("attn_gate.weight"), cfg.n_embd, value_dim)?,
                    ssm_out: read(&p("ssm_out.weight"), value_dim, cfg.n_embd)?,
                }
            } else {
                // No qwen35 a projeção de Q sai com query e gate juntos, e a saída da
                // atenção entra com `head_dim * n_head`, que não é `n_embd`.
                let q_out = if cfg.delta_net.is_some() {
                    cfg.head_dim * cfg.n_head * 2
                } else {
                    cfg.n_embd
                };
                let o_in = cfg.head_dim * cfg.n_head;
                MixerRaw::Attn {
                    attn_q: read(&p("attn_q.weight"), cfg.n_embd, q_out)?,
                    attn_k: read(&p("attn_k.weight"), cfg.n_embd, kv_dim)?,
                    attn_v: read(&p("attn_v.weight"), cfg.n_embd, kv_dim)?,
                    attn_output: read(&p("attn_output.weight"), o_in, cfg.n_embd)?,
                }
            };
            layers.push(GpuLayerRaw {
                mixer,
                ffn_gate: read(&p("ffn_gate.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_up: read(&p("ffn_up.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_down: read(&p("ffn_down.weight"), cfg.n_ff, cfg.n_embd)?,
            });
        }
        let output = read("output.weight", cfg.n_embd, cfg.vocab)?;

        // Bloco MTP. `cfg.n_layer` já desconta os `nextn_predict_layers`, então esse índice
        // é exatamente o primeiro bloco MTP empilhado no fim.
        //
        // A regra `eh_linear` **não** vale aqui: no Qwen3.8-27B o bloco é o 64, que por
        // `(64 + 1) % 4 != 0` cairia como linear, mas o GGUF traz `attn_q/k/v/output` —
        // o bloco MTP é de atenção mesmo nas posições que seriam de atenção linear.
        let mtp = if cfg.n_layer_nextn > 0 {
            let l = cfg.n_layer;
            let p = |s: &str| format!("blk.{l}.{s}");
            let q_out = if cfg.delta_net.is_some() {
                cfg.head_dim * cfg.n_head * 2
            } else {
                cfg.n_embd
            };
            Some(MtpRaw {
                eh_proj: read(&p("nextn.eh_proj.weight"), cfg.n_embd * 2, cfg.n_embd)?,
                layer: GpuLayerRaw {
                    mixer: MixerRaw::Attn {
                        attn_q: read(&p("attn_q.weight"), cfg.n_embd, q_out)?,
                        attn_k: read(&p("attn_k.weight"), cfg.n_embd, kv_dim)?,
                        attn_v: read(&p("attn_v.weight"), cfg.n_embd, kv_dim)?,
                        attn_output: read(
                            &p("attn_output.weight"),
                            cfg.head_dim * cfg.n_head,
                            cfg.n_embd,
                        )?,
                    },
                    ffn_gate: read(&p("ffn_gate.weight"), cfg.n_embd, cfg.n_ff)?,
                    ffn_up: read(&p("ffn_up.weight"), cfg.n_embd, cfg.n_ff)?,
                    ffn_down: read(&p("ffn_down.weight"), cfg.n_ff, cfg.n_embd)?,
                },
            })
        } else {
            None
        };
        Ok(Self {
            layers,
            output,
            mtp,
        })
    }
}

/// Tabela de embedding mantida **quantizada**, do jeito que está no GGUF.
///
/// Dequantizá-la inteira custa `vocab × n_embd × 4` bytes — 5,1 GB no vocabulário de
/// 248 320 do Qwen3.8 — para que o passo de embedding leia **uma** linha por token (o
/// prefill lê `n_batch`). Aqui ficam os bytes emprestados do mmap, sem cópia, e a linha é
/// dequantizada no momento do uso: `n_embd` valores, microssegundos.
#[derive(Clone, Copy)]
pub struct TokenEmbd<'a> {
    ty: GgmlType,
    bytes: &'a [u8],
    n_embd: usize,
    /// Bytes de uma linha de `n_embd` elementos neste tipo.
    row_bytes: usize,
}

impl<'a> TokenEmbd<'a> {
    /// # Errors
    /// Se `n_embd` não fechar um número inteiro de blocos do tipo, ou se os bytes não
    /// cobrirem uma linha inteira.
    pub fn nova(ty: GgmlType, bytes: &'a [u8], n_embd: usize) -> Result<Self, ModelError> {
        let elems = usize::try_from(ty.block_size()).map_err(|_| ModelError::Overflow)?;
        let bloco = usize::try_from(ty.type_size()).map_err(|_| ModelError::Overflow)?;
        if elems == 0 || !n_embd.is_multiple_of(elems) {
            return Err(ModelError::Gpu(format!(
                "token_embd {ty:?}: n_embd={n_embd} não é múltiplo de {elems}"
            )));
        }
        let row_bytes = n_embd / elems * bloco;
        if bytes.len() < row_bytes {
            return Err(ModelError::Gpu(format!(
                "token_embd: {} bytes, menos que uma linha ({row_bytes})",
                bytes.len()
            )));
        }
        Ok(Self {
            ty,
            bytes,
            n_embd,
            row_bytes,
        })
    }

    /// Dequantiza a linha de `token`.
    ///
    /// # Errors
    /// Se o token estiver fora da tabela ou o tipo não tiver dequant na CPU.
    pub fn linha(&self, token: u32) -> Result<Vec<f32>, ModelError> {
        let ini = token as usize * self.row_bytes;
        let linha = self
            .bytes
            .get(ini..ini + self.row_bytes)
            .ok_or_else(|| ModelError::Gpu(format!("token {token} fora da tabela")))?;
        ggml_cpu::dequant_to_f32(linha, self.ty).map_err(ModelError::from)
    }

    /// As linhas de `tokens`, concatenadas — o bloco de prefill precisa de todas.
    ///
    /// # Errors
    /// Como [`Self::linha`].
    pub fn linhas(&self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let mut out = Vec::with_capacity(tokens.len() * self.n_embd);
        for &t in tokens {
            out.extend_from_slice(&self.linha(t)?);
        }
        Ok(out)
    }
}

/// Pesos auxiliares f32 que o decode GPU-resident precisa (norm/bias/freq/embd).
///
/// `token_embd` fica quantizado e emprestado do GGUF (ver [`TokenEmbd`]); o resto é sempre
/// copiado, porque é pequeno.
pub struct GpuAuxWeights<'a> {
    pub token_embd: TokenEmbd<'a>,
    pub layers: Vec<AuxLayer>,
    pub output_norm: Vec<f32>, // [n_embd]
    pub freq_table: Vec<f32>,  // [rope_dim/2]
    /// Auxiliares f32 do bloco MTP, quando existe. Ver [`MtpRaw`].
    pub mtp: Option<MtpAux>,
}

/// Pesos f32 do bloco de multi-token prediction.
pub struct MtpAux {
    /// RMSNorm aplicada ao embedding do token amostrado, antes do `eh_proj`.
    pub enorm: Vec<f32>, // [n_embd]
    /// RMSNorm aplicada ao hidden state do último layer, antes do `eh_proj`.
    pub hnorm: Vec<f32>, // [n_embd]
    /// RMSNorm final, antes da projeção de saída — faz o papel do `output_norm`.
    pub shared_head_norm: Vec<f32>, // [n_embd]
    pub layer: AuxLayer,
}

/// Pesos auxiliares f32 por camada.
pub struct AuxLayer {
    pub attn_norm: Vec<f32>,      // [n_embd]
    pub ffn_norm: Vec<f32>,       // [n_embd]
    pub q_bias: Option<Vec<f32>>, // [n_embd]
    pub k_bias: Option<Vec<f32>>, // [kv_dim]
    pub v_bias: Option<Vec<f32>>, // [kv_dim]
    /// QK-norm por cabeça — `Some` nas camadas de atenção do `qwen35`. [head_dim]
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    /// `Some` nas camadas de atenção linear do `qwen35`.
    pub delta: Option<DeltaAux>,
}

/// Pesos f32 de uma camada de atenção linear (gated delta net).
///
/// `alpha` e `beta` são matrizes de projeção que o GGUF guarda em f32 — cada uma tem
/// `n_embd × n_v_heads` (≈1 MB por camada no Qwen3.8-27B), pequenas o bastante para não
/// valer quantizar, mas grandes o bastante para precisarem de um matvec f32 na GPU.
pub struct DeltaAux {
    /// Kernel da convolução causal, `[conv_dim][d_conv]` (layout do GGUF).
    pub conv1d: Vec<f32>,
    /// `-exp(A_log)`, já aplicado no GGUF. [n_v_heads]
    pub a: Vec<f32>,
    /// Bias somado a `alpha` antes do softplus. [n_v_heads]
    pub dt_bias: Vec<f32>,
    /// Projeção do log-decaimento. [n_embd * n_v_heads]
    pub alpha: Vec<f32>,
    /// Projeção da força do delta rule. [n_embd * n_v_heads]
    pub beta: Vec<f32>,
    /// Norma aplicada à saída de cada cabeça. [head_v_dim]
    pub norm: Vec<f32>,
}

/// Decode 100% na GPU: a stack inteira do token roda na GPU, KV-cache residente.
/// Implementado por `llama_vulkan::ResidentForward`.
pub trait GpuResidentDecode {
    /// Decode de 1 token na posição absoluta `pos` (0-based). Retorna os logits ([vocab]).
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, ModelError>;
    /// Zera o comprimento do KV-cache interno (início de nova sequência).
    fn reset(&self);
    /// Tokens por bloco que `decode_batch` aceita. `1` significa sem prefill em batch.
    fn batch_size(&self) -> usize {
        1
    }
    /// Processa `tokens` nas posições `pos0..pos0 + tokens.len()` de uma vez, retornando
    /// os logits **do último** — os anteriores já ficaram no KV-cache, e é só isso que o
    /// prefill precisa. `tokens.len()` tem de ser exatamente `batch_size()`.
    fn decode_batch(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, ModelError> {
        let mut logits = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            logits = self.decode(t, pos0 + i)?;
        }
        Ok(logits)
    }
    /// Guarda o estado que não volta atrás sozinho (o recorrente das camadas lineares) e o
    /// comprimento do KV, para que uma divergência de prompt **depois** desta posição não
    /// custe reprocessar tudo. Um snapshot só: o novo sobrescreve o anterior.
    ///
    /// `false` quando o backend não guarda snapshot — e aí a sessão nunca pede recuo.
    fn marcar(&self) -> bool {
        false
    }
    /// Volta ao último [`GpuResidentDecode::marcar`]. `false` quando não há snapshot.
    fn restaurar(&self) -> bool {
        false
    }

    /// Se este backend foi construído com a cabeça de multi-token prediction.
    ///
    /// Só com ela os três métodos abaixo funcionam; sem ela o laço de geração segue o
    /// caminho normal, um token por passo.
    fn tem_mtp(&self) -> bool {
        false
    }

    /// Propõe o token seguinte a `token` com a cabeça MTP.
    ///
    /// `hidden_idx` diz de qual hidden partir: `0..VERIFY_TOK` indexa os hidden do tronco
    /// no último passo (0 depois de um decode simples ou de um verify rejeitado no 1º; o
    /// índice do último token aceito depois de um verify). [`HIDDEN_CABECA`] realimenta a
    /// cabeça com o hidden **do próprio bloco** da proposta anterior — é o encadeamento
    /// (n=2) medido em `aceitacao_encadeada_do_segundo_token`.
    fn propor_mtp(&self, token: u32, hidden_idx: usize) -> Result<u32, ModelError> {
        let _ = (token, hidden_idx);
        Err(ModelError::Gpu("backend sem cabeça MTP".into()))
    }

    /// Processa os [`VERIFY_TOK`] tokens de uma verificação nas posições `pos0..` e
    /// devolve os logits de **todos**, concatenados (`VERIFY_TOK × vocab`, o primeiro
    /// token primeiro).
    ///
    /// `tokens[0]` é o token já amostrado; `tokens[1]` e `tokens[2]` são a proposta da
    /// cabeça e a proposta encadeada. Ler os pesos uma vez serve aos três — é daí que vem
    /// o ganho do speculative decoding.
    fn decode_verify(
        &self,
        tokens: &[u32; VERIFY_TOK],
        pos0: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let _ = (tokens, pos0);
        Err(ModelError::Gpu("backend sem plano de verify".into()))
    }

    /// Desfaz os tokens rejeitados do último `decode_verify`, mantendo os `manter`
    /// primeiros (1 quando a 1ª proposta falhou, 2 quando só a encadeada falhou):
    /// restaura o snapshot do estado recorrente tirado depois do token `manter - 1` e
    /// recua o comprimento do KV em `VERIFY_TOK - manter`.
    fn rollback_verify(&self, manter: usize) -> Result<(), ModelError> {
        let _ = manter;
        Err(ModelError::Gpu("backend sem plano de verify".into()))
    }
}

/// Tokens por passo de verify do MTP: o token já amostrado, a proposta da cabeça e a
/// proposta encadeada (a cabeça realimentada com o próprio hidden). O valor fixa a
/// largura das pipelines de verify (`COLS` é specialization constant) e o número de
/// pontos de snapshot (`VERIFY_TOK - 1`).
pub const VERIFY_TOK: usize = 3;

/// Sentinela de `hidden_idx` em [`GpuResidentDecode::propor_mtp`]: alimentar a cabeça com
/// o hidden do próprio bloco MTP da proposta anterior, em vez de um hidden do tronco.
pub const HIDDEN_CABECA: usize = usize::MAX;

impl<'a> GpuAuxWeights<'a> {
    /// Lê os pesos auxiliares direto do GGUF, sem passar pelo `Model`.
    ///
    /// O caminho por `Model::gpu_aux_weights` exige a estrutura densa (`attn_q` em toda
    /// camada), que o `qwen35` não tem. Aqui cada camada é lida conforme a sua variante.
    pub fn from_gguf(f: &GgufFile, bytes: &'a [u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let _fase = crate::perfil_carga::Fase::nova("aux f32 (dequant)");
        let f32s = |name: &str, esperado: usize| -> Result<Vec<f32>, ModelError> {
            let info = f
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| ModelError::Gpu(format!("tensor {name} ausente")))?;
            let raw = f
                .tensor_data(bytes, info)
                .map_err(|e| ModelError::Gpu(e.to_string()))?;
            let v = ggml_cpu::dequant_to_f32(raw, info.ggml_type)
                .map_err(|e| ModelError::Gpu(format!("{name}: {e}")))?;
            if v.len() != esperado {
                return Err(ModelError::Gpu(format!(
                    "tensor {name}: {} floats, esperado {esperado}",
                    v.len()
                )));
            }
            Ok(v)
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let dn = cfg.delta_net.as_ref();
            let linear = dn.is_some_and(|d| d.eh_linear(l));
            // No qwen35 a segunda norma se chama `post_attention_norm`; nas arquiteturas
            // densas, `ffn_norm`. As duas ficam no mesmo lugar do esqueleto da camada.
            let segunda_norma = if dn.is_some() {
                p("post_attention_norm.weight")
            } else {
                p("ffn_norm.weight")
            };
            let (q_norm, k_norm) = if dn.is_some() && !linear {
                (
                    Some(f32s(&p("attn_q_norm.weight"), cfg.head_dim)?),
                    Some(f32s(&p("attn_k_norm.weight"), cfg.head_dim)?),
                )
            } else {
                (None, None)
            };
            let delta = match (linear, dn) {
                (true, Some(d)) => {
                    let conv_dim = d.d_state * d.n_k_heads * 2 + d.head_v_dim() * d.n_v_heads;
                    Some(DeltaAux {
                        conv1d: f32s(&p("ssm_conv1d.weight"), d.d_conv * conv_dim)?,
                        a: f32s(&p("ssm_a"), d.n_v_heads)?,
                        dt_bias: f32s(&p("ssm_dt.bias"), d.n_v_heads)?,
                        alpha: f32s(&p("ssm_alpha.weight"), cfg.n_embd * d.n_v_heads)?,
                        beta: f32s(&p("ssm_beta.weight"), cfg.n_embd * d.n_v_heads)?,
                        norm: f32s(&p("ssm_norm.weight"), d.head_v_dim())?,
                    })
                }
                _ => None,
            };
            layers.push(AuxLayer {
                attn_norm: f32s(&p("attn_norm.weight"), cfg.n_embd)?,
                ffn_norm: f32s(&segunda_norma, cfg.n_embd)?,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm,
                k_norm,
                delta,
            });
        }

        let meia = cfg.rope_dim / 2;
        let freq_table: Vec<f32> = (0..meia)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let expo = (2 * i) as f32 / cfg.rope_dim as f32;
                1.0 / cfg.freq_base.powf(expo)
            })
            .collect();

        // Auxiliares do bloco MTP — ver `MtpRaw` para o porquê de ele ser sempre de atenção.
        let mtp = if cfg.n_layer_nextn > 0 {
            let l = cfg.n_layer;
            let p = |s: &str| format!("blk.{l}.{s}");
            let dn = cfg.delta_net.as_ref();
            let segunda_norma = if dn.is_some() {
                p("post_attention_norm.weight")
            } else {
                p("ffn_norm.weight")
            };
            Some(MtpAux {
                enorm: f32s(&p("nextn.enorm.weight"), cfg.n_embd)?,
                hnorm: f32s(&p("nextn.hnorm.weight"), cfg.n_embd)?,
                shared_head_norm: f32s(&p("nextn.shared_head_norm.weight"), cfg.n_embd)?,
                layer: AuxLayer {
                    attn_norm: f32s(&p("attn_norm.weight"), cfg.n_embd)?,
                    ffn_norm: f32s(&segunda_norma, cfg.n_embd)?,
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    q_norm: dn
                        .map(|_| f32s(&p("attn_q_norm.weight"), cfg.head_dim))
                        .transpose()?,
                    k_norm: dn
                        .map(|_| f32s(&p("attn_k_norm.weight"), cfg.head_dim))
                        .transpose()?,
                    delta: None,
                },
            })
        } else {
            None
        };

        // A tabela de embedding fica quantizada: dequantizá-la inteira custaria 5,1 GB de
        // RAM no vocabulário do Qwen3.8 para que cada token leia uma linha. Ver `TokenEmbd`.
        let te = f
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or_else(|| ModelError::Gpu("tensor token_embd.weight ausente".into()))?;
        let te_bytes = f
            .tensor_data(bytes, te)
            .map_err(|e| ModelError::Gpu(e.to_string()))?;

        Ok(Self {
            token_embd: TokenEmbd::nova(te.ggml_type, te_bytes, cfg.n_embd)?,
            layers,
            output_norm: f32s("output_norm.weight", cfg.n_embd)?,
            freq_table,
            mtp,
        })
    }
}

/// Geração token a token com o decode inteiro na GPU, **sem depender do `Model`**.
///
/// As arquiteturas híbridas (`qwen35`) não têm a estrutura densa que `Model` exige — em 3
/// de cada 4 camadas não existe `attn_q` — mas o laço de geração só precisa da config, do
/// tokenizer e do backend.
#[allow(clippy::too_many_arguments)]
pub fn gerar_streaming_residente(
    config: &LlamaConfig,
    tokenizer: &llama_tokenizer::Tokenizer,
    prompt: &str,
    n_tokens: usize,
    sampler: &llama_sampling::Sampler,
    rng: &mut impl rand::Rng,
    gpu: &dyn GpuResidentDecode,
    on_token: &mut impl FnMut(&str),
) -> Result<(), ModelError> {
    // Aqui a carga terminou: os pesos já estão na VRAM e o backend está pronto. É o ponto
    // onde o perfil por fase (`LLAMA_RS_LOAD_PROFILE=1`) tem o que dizer.
    crate::perfil_carga::imprimir();
    let prompt_ids = tokenizer.encode(prompt, config.add_bos);
    if prompt_ids.is_empty() {
        return Err(ModelError::Gpu("prompt vazio".into()));
    }
    gpu.reset();
    let (mut logits, mut pos, mut hidden) = prefill_residente(gpu, &prompt_ids)?;
    let first_idx = sampler.sample(&logits, rng);
    let mut next = u32::try_from(first_idx).map_err(|_| ModelError::Overflow)?;
    let usar_mtp = gpu.tem_mtp();

    let mut count = 0usize;
    while count < n_tokens {
        if next == config.eos_id {
            break;
        }
        let piece = tokenizer.decode(&[next]);
        on_token(&piece);
        count += 1;

        if usar_mtp {
            let passo = passo_mtp(gpu, sampler, rng, next, hidden, pos, config.vocab)?;
            pos = passo.pos;
            hidden = passo.hidden;
            for aceito in passo.aceitos.into_iter().flatten() {
                if aceito == config.eos_id {
                    return Ok(());
                }
                if count >= n_tokens {
                    return Ok(());
                }
                on_token(&tokenizer.decode(&[aceito]));
                count += 1;
            }
            next = passo.seguinte;
            continue;
        }

        logits = gpu.decode(next, pos)?;
        pos += 1;
        hidden = 0;
        let idx = sampler.sample(&logits, rng);
        next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
    }
    Ok(())
}

/// Prefill de `ids`: blocos de `batch_size()` e o resto token a token.
///
/// Devolve os logits do último token, a posição seguinte e **o índice do token dentro do
/// último bloco** que produziu esses logits. O terceiro só interessa ao MTP: é qual dos
/// hidden que ficaram na GPU a cabeça deve consumir.
fn prefill_residente(
    gpu: &dyn GpuResidentDecode,
    ids: &[u32],
) -> Result<(Vec<f32>, usize, usize), ModelError> {
    // Em batch cada peso do modelo sai da VRAM uma vez para N tokens em vez de uma por
    // token — o decode é limitado por banda, e é aí que está o custo de um prompt longo.
    let nb = gpu.batch_size();
    let mut logits = Vec::new();
    let mut pos = 0usize;
    let mut hidden = 0usize;
    while nb > 1 && ids.len() - pos >= nb {
        let Some(bloco) = ids.get(pos..pos + nb) else {
            break;
        };
        logits = gpu.decode_batch(bloco, pos)?;
        pos += nb;
        hidden = nb - 1;
    }
    for &t in ids.get(pos..).unwrap_or(&[]) {
        logits = gpu.decode(t, pos)?;
        pos += 1;
        hidden = 0;
    }
    Ok((logits, pos, hidden))
}

/// O que um passo de speculative decoding produziu.
pub struct PassoMtp {
    /// Os tokens do passo além do primeiro, quando aceitos (0, 1 ou 2 deles).
    pub aceitos: [Option<u32>; VERIFY_TOK - 1],
    /// O token a emitir no próximo passo (ainda não passou pelo modelo).
    pub seguinte: u32,
    /// Posição seguinte no KV-cache: avança 1 + nº de aceitos.
    pub pos: usize,
    /// Qual hidden do passo alimenta a próxima proposta.
    pub hidden: usize,
    /// Logits do último token que ficou no cache — de onde `seguinte` foi amostrado.
    /// A sessão os guarda para responder de graça a um prompt repetido.
    pub logits_ultimo: Vec<f32>,
}

/// Um passo de speculative decoding: propor (encadeando) → verificar → aceitar ou desfazer.
///
/// `token` já foi amostrado e emitido; a cabeça propõe o **seguinte** a ele, é
/// realimentada com o próprio hidden para propor mais um, e o verify processa os três de
/// uma vez, lendo os pesos uma única vez. A rejeição não desperdiça o forward: os logits
/// do último token verificado **são** os do token correto seguinte, e por isso todo passo
/// entrega pelo menos um token novo além de `token`.
///
/// Com amostragem (temp > 0) uma proposta só é aceita se coincidir com o token que o
/// sampler tira dos logits do anterior — a distribuição continua exatamente a do caminho
/// sem MTP, e a taxa de aceitação cai na proporção da entropia.
pub fn passo_mtp(
    gpu: &dyn GpuResidentDecode,
    sampler: &llama_sampling::Sampler,
    rng: &mut impl rand::Rng,
    token: u32,
    hidden: usize,
    pos: usize,
    vocab: usize,
) -> Result<PassoMtp, ModelError> {
    let p1 = gpu.propor_mtp(token, hidden)?;
    let p2 = gpu.propor_mtp(p1, HIDDEN_CABECA)?;
    let logits = gpu.decode_verify(&[token, p1, p2], pos)?;
    let fatia = |i: usize| -> Result<&[f32], ModelError> {
        logits
            .get(i * vocab..(i + 1) * vocab)
            .ok_or_else(|| ModelError::Gpu("verify devolveu menos logits que o bloco".into()))
    };
    let amostra = |l: &[f32], rng: &mut _| -> Result<u32, ModelError> {
        u32::try_from(sampler.sample(l, rng)).map_err(|_| ModelError::Overflow)
    };
    let t1 = amostra(fatia(0)?, rng)?;
    if t1 != p1 {
        // Rejeitou a 1ª: o token certo já veio nos logits[0]. Desfaz os dois do bloco.
        gpu.rollback_verify(1)?;
        return Ok(PassoMtp {
            aceitos: [None; VERIFY_TOK - 1],
            seguinte: t1,
            pos: pos + 1,
            hidden: 0,
            logits_ultimo: fatia(0)?.to_vec(),
        });
    }
    let t2 = amostra(fatia(1)?, rng)?;
    if t2 != p2 {
        gpu.rollback_verify(2)?;
        return Ok(PassoMtp {
            aceitos: [Some(t1), None],
            seguinte: t2,
            pos: pos + 2,
            hidden: 1,
            logits_ultimo: fatia(1)?.to_vec(),
        });
    }
    let seguinte = amostra(fatia(2)?, rng)?;
    Ok(PassoMtp {
        aceitos: [Some(t1), Some(t2)],
        seguinte,
        // O hidden que produziu `seguinte` é o do **último** token do bloco.
        pos: pos + VERIFY_TOK,
        hidden: VERIFY_TOK - 1,
        logits_ultimo: fatia(2)?.to_vec(),
    })
}

#[cfg(feature = "gpu")]
use crate::attention::{KvCache, attention};
#[cfg(feature = "gpu")]
use crate::model::Model;
#[cfg(feature = "gpu")]
use crate::ops::{embedding_lookup, rmsnorm_and_scale, rope_norm, swiglu};

#[cfg(feature = "gpu")]
impl Model<'_> {
    /// Igual a `generate_streaming`, mas o **decode** roda na GPU via `gpu`.
    /// O **prefill** do prompt (n_tok>1) permanece na CPU (shader é matvec).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_gpu(
        &self,
        tokenizer: &llama_tokenizer::Tokenizer,
        prompt: &str,
        n_tokens: usize,
        sampler: &llama_sampling::Sampler,
        rng: &mut impl rand::Rng,
        gpu: &dyn GpuMatmul,
        w: &GpuRawWeights,
        on_token: &mut impl FnMut(&str),
    ) -> Result<(), ModelError> {
        let prompt_ids = tokenizer.encode(prompt, true);
        let mut cache = self.new_cache();

        // Prefill na CPU.
        let logits = self.forward(&prompt_ids, &mut cache)?;
        let first_idx = sampler.sample(&logits, rng);
        let mut next = u32::try_from(first_idx).map_err(|_| ModelError::Overflow)?;

        let mut count = 0usize;
        while count < n_tokens {
            if next == self.config.eos_id {
                break;
            }
            let piece = tokenizer.decode(&[next]);
            on_token(&piece);
            count += 1;
            // Decode na GPU.
            let logits = self.forward_gpu(&[next], &mut cache, gpu, w)?;
            let idx = sampler.sample(&logits, rng);
            next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
        }
        Ok(())
    }

    /// Decode de 1 passo na CPU (prefill incluso). Cache criado internamente.
    pub fn decode_one_cpu_owned(&self, prompt: &[u32]) -> Result<u32, ModelError> {
        let mut cache = self.new_cache();
        let logits = self.forward(prompt, &mut cache)?;
        u32::try_from(crate::ops::argmax(&logits)).map_err(|_| ModelError::Overflow)
    }

    /// Prefill na CPU + 1 decode na GPU. Cache criado internamente.
    pub fn decode_one_gpu_owned(
        &self,
        prompt: &[u32],
        gpu: &dyn GpuMatmul,
        w: &GpuRawWeights,
    ) -> Result<u32, ModelError> {
        let mut cache = self.new_cache();
        let _ = self.forward(prompt, &mut cache)?;
        let last = *prompt
            .last()
            .ok_or_else(|| ModelError::Gpu("prompt vazio".into()))?;
        let logits = self.forward_gpu(&[last], &mut cache, gpu, w)?;
        u32::try_from(crate::ops::argmax(&logits)).map_err(|_| ModelError::Overflow)
    }

    /// Coleta os pesos auxiliares f32 para o backend GPU-resident.
    pub fn gpu_aux_weights(&self) -> Result<GpuAuxWeights<'_>, ModelError> {
        let _fase = crate::perfil_carga::Fase::nova("aux f32 (dequant)");
        // Quantizada, emprestada do GGUF — o caminho de CPU tem o seu próprio cache f32,
        // mas o decode residente só precisa de uma linha por token. Ver `TokenEmbd`.
        let token_embd = TokenEmbd::nova(
            self.weights.token_embd.ty,
            self.weights.token_embd.bytes,
            self.config.n_embd,
        )?;
        let output_norm = self.output_norm_f32()?.to_vec();
        let freq_table = self.freq_table.clone();
        let mut layers = Vec::with_capacity(self.config.n_layer);
        for l in 0..self.config.n_layer {
            let (attn_norm, ffn_norm) = self.layer_norms_f32(l)?;
            let attn_norm = attn_norm.to_vec();
            let ffn_norm = ffn_norm.to_vec();
            let lw = self
                .weights
                .layers
                .get(l)
                .ok_or_else(|| ModelError::Gpu(format!("camada {l} ausente nos pesos")))?;
            let bias =
                |b: &Option<crate::weights::RawTensor>| -> Result<Option<Vec<f32>>, ModelError> {
                    match b {
                        Some(t) => Ok(Some(t.dequant_to_f32()?.to_vec())),
                        None => Ok(None),
                    }
                };
            layers.push(AuxLayer {
                attn_norm,
                ffn_norm,
                q_bias: bias(&lw.attn_q_bias)?,
                k_bias: bias(&lw.attn_k_bias)?,
                v_bias: bias(&lw.attn_v_bias)?,
                q_norm: None,
                k_norm: None,
                delta: None,
            });
        }
        Ok(GpuAuxWeights {
            token_embd,
            layers,
            output_norm,
            freq_table,
            // Este caminho parte do `Model` denso, que não carrega bloco MTP; quem lê o
            // bloco é `GpuAuxWeights::from_gguf`.
            mtp: None,
        })
    }

    /// Igual a `generate_streaming_gpu`, mas o token inteiro (prefill incluído) roda na
    /// GPU via `gpu` (KV-cache residente). Os logits só voltam ao host para o sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_gpu_resident(
        &self,
        tokenizer: &llama_tokenizer::Tokenizer,
        prompt: &str,
        n_tokens: usize,
        sampler: &llama_sampling::Sampler,
        rng: &mut impl rand::Rng,
        gpu: &dyn GpuResidentDecode,
        on_token: &mut impl FnMut(&str),
    ) -> Result<(), ModelError> {
        gerar_streaming_residente(
            &self.config,
            tokenizer,
            prompt,
            n_tokens,
            sampler,
            rng,
            gpu,
            on_token,
        )
    }

    /// Prefill + 1 decode 100% na GPU; argmax dos logits. Para teste de paridade vs CPU.
    pub fn decode_one_gpu_resident_owned(
        &self,
        prompt: &[u32],
        gpu: &dyn GpuResidentDecode,
    ) -> Result<u32, ModelError> {
        gpu.reset();
        let mut logits = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            logits = gpu.decode(t, pos)?;
        }
        u32::try_from(crate::ops::argmax(&logits)).map_err(|_| ModelError::Overflow)
    }

    /// Prefill + 1 decode na CPU; retorna o vetor de logits completo ([vocab]).
    /// Para teste de tolerância vs GPU.
    pub fn decode_one_cpu_logits(&self, prompt: &[u32]) -> Result<Vec<f32>, ModelError> {
        let mut cache = self.new_cache();
        self.forward(prompt, &mut cache)
    }

    /// Prefill + 1 decode 100% na GPU; retorna o vetor de logits completo ([vocab]).
    /// Para teste de tolerância vs CPU.
    pub fn decode_one_gpu_resident_logits(
        &self,
        prompt: &[u32],
        gpu: &dyn GpuResidentDecode,
    ) -> Result<Vec<f32>, ModelError> {
        gpu.reset();
        let mut logits = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            logits = gpu.decode(t, pos)?;
        }
        Ok(logits)
    }

    /// Forward de **decode** (n_tok=1) com os 8 matmuls na GPU.
    /// RMSNorm/RoPE/attention/SwiGLU/bias permanecem na CPU.
    pub(crate) fn forward_gpu(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        gpu: &dyn GpuMatmul,
        w: &GpuRawWeights,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        if tokens.len() != 1 {
            return Err(ModelError::Gpu(format!(
                "forward_gpu exige n_tok=1 (decode), recebeu {}",
                tokens.len()
            )));
        }
        let n_tok = 1usize;
        let pos0 = cache.len();
        let kv_dim = c.n_head_kv * c.head_dim;

        let token_embd = self.token_embd_f32()?;
        let mut x = embedding_lookup(token_embd, tokens, c.n_embd)?;

        for (l, gw) in w.layers.iter().enumerate() {
            let (attn_norm, ffn_norm) = self.layer_norms_f32(l)?;
            let attn_in = rmsnorm_and_scale(&x, attn_norm, c.n_embd, c.rms_eps);

            // Caminho denso (Q8_0): o mixer é sempre atenção.
            let (w_q, w_k, w_v, w_o) = gw.attn();
            let mut q = gpu.matvec_q8_0(w_q.bytes, &attn_in, c.n_embd, c.n_embd)?;
            let mut k = gpu.matvec_q8_0(w_k.bytes, &attn_in, c.n_embd, kv_dim)?;
            let mut v = gpu.matvec_q8_0(w_v.bytes, &attn_in, c.n_embd, kv_dim)?;

            self.add_layer_biases(l, &mut q, &mut k, &mut v, kv_dim, n_tok)?;

            rope_norm(
                &mut q,
                n_tok,
                c.n_head,
                c.head_dim,
                c.rope_dim,
                &self.freq_table,
                pos0,
            );
            rope_norm(
                &mut k,
                n_tok,
                c.n_head_kv,
                c.head_dim,
                c.rope_dim,
                &self.freq_table,
                pos0,
            );

            cache.append(l, &k, &v)?;
            let total_len = pos0 + n_tok;
            let attn = attention(
                &q,
                cache.k_slice(l, total_len),
                cache.v_slice(l, total_len),
                n_tok,
                pos0,
                c.n_head,
                c.n_head_kv,
                c.head_dim,
            );
            let attn_out = gpu.matvec_q8_0(w_o.bytes, &attn, c.n_embd, c.n_embd)?;
            for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }

            let ffn_in = rmsnorm_and_scale(&x, ffn_norm, c.n_embd, c.rms_eps);
            let gate = gpu.matvec_q8_0(gw.ffn_gate.bytes, &ffn_in, c.n_embd, c.n_ff)?;
            let up = gpu.matvec_q8_0(gw.ffn_up.bytes, &ffn_in, c.n_embd, c.n_ff)?;
            let act = swiglu(&gate, &up);
            let ffn_out = gpu.matvec_q8_0(gw.ffn_down.bytes, &act, c.n_ff, c.n_embd)?;
            for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
        }

        cache.advance(n_tok);

        let output_norm = self.output_norm_f32()?;
        let final_x = rmsnorm_and_scale(&x, output_norm, c.n_embd, c.rms_eps);
        let last = final_x.get(..c.n_embd).unwrap_or(&final_x);
        let logits = gpu.matvec_q8_0(w.output.bytes, last, c.n_embd, c.vocab)?;
        Ok(logits)
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

    /// GGUF sintético com um único tensor Q8_0 `token_embd.weight` de `vocab × n_embd`.
    /// Sem KV: o teste chega no tensor por `GgufFile::parse` + `tensor_data`, que é o
    /// caminho real da carga.
    fn gguf_token_embd(vocab: usize, n_embd: usize, dados: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // versão
        out.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        out.extend_from_slice(&0u64.to_le_bytes()); // 0 KVs
        let nome = "token_embd.weight";
        out.extend_from_slice(&(nome.len() as u64).to_le_bytes());
        out.extend_from_slice(nome.as_bytes());
        out.extend_from_slice(&2u32.to_le_bytes()); // 2 dimensões
        out.extend_from_slice(&(n_embd as u64).to_le_bytes());
        out.extend_from_slice(&(vocab as u64).to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
        out.extend_from_slice(&0u64.to_le_bytes()); // offset
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        out.extend_from_slice(dados);
        out
    }

    /// Bytes Q8_0 determinísticos: escala f16 = 1.0 (0x3C00) e quantizados de um LCG.
    /// Escala fixa e finita para que a comparação exata não esbarre em NaN.
    fn bytes_q8_0(n_blocos: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_blocos * 34);
        let mut x: u32 = 12345;
        for _ in 0..n_blocos {
            out.extend_from_slice(&[0x00, 0x3C]);
            for _ in 0..32 {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                out.push((x >> 24) as u8);
            }
        }
        out
    }

    /// O caminho preguiçoso (uma linha por vez) tem de dar **exatamente** os mesmos
    /// valores que dequantizar a tabela inteira — é o que autoriza trocar 5,1 GB de RAM
    /// por alguns microssegundos por token.
    #[test]
    fn token_embd_por_linha_e_identico_ao_dequant_completo() {
        let (vocab, n_embd) = (7usize, 64usize);
        let dados = bytes_q8_0(vocab * n_embd / 32);
        let bytes = gguf_token_embd(vocab, n_embd, &dados);
        let f = GgufFile::parse(&bytes).unwrap();
        let info = &f.tensors[0];
        let raw = f.tensor_data(&bytes, info).unwrap();

        let completo = ggml_cpu::dequant_to_f32(raw, info.ggml_type).unwrap();
        assert_eq!(completo.len(), vocab * n_embd);
        let te = TokenEmbd::nova(info.ggml_type, raw, n_embd).unwrap();

        for t in 0..vocab {
            assert_eq!(
                te.linha(u32::try_from(t).unwrap()).unwrap(),
                &completo[t * n_embd..(t + 1) * n_embd],
                "linha {t}"
            );
        }

        // O bloco de prefill é a concatenação das linhas, na ordem dos tokens.
        let tokens = [3u32, 0, 6, 1];
        let bloco = te.linhas(&tokens).unwrap();
        assert_eq!(bloco.len(), tokens.len() * n_embd);
        for (i, &t) in tokens.iter().enumerate() {
            let r = t as usize * n_embd;
            assert_eq!(
                &bloco[i * n_embd..(i + 1) * n_embd],
                &completo[r..r + n_embd]
            );
        }

        // Token fora da tabela é erro, não pânico nem lixo.
        assert!(te.linha(u32::try_from(vocab).unwrap()).is_err());
    }

    #[test]
    fn token_embd_recusa_n_embd_que_nao_fecha_bloco() {
        let dados = bytes_q8_0(2);
        assert!(TokenEmbd::nova(GgmlType::Q8_0, &dados, 33).is_err());
        assert!(TokenEmbd::nova(GgmlType::Q8_0, &dados, 64).is_ok());
        // Bytes menores que uma linha também não passam.
        assert!(TokenEmbd::nova(GgmlType::Q8_0, &dados[..34], 64).is_err());
    }

    fn load_qwen() -> Option<(Vec<u8>, GgufFile, LlamaConfig)> {
        let bytes =
            std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        let cfg = LlamaConfig::from_gguf(&f).ok()?;
        Some((bytes, f, cfg))
    }

    /// Mock que roda a matemática Q8_0 na CPU (mesmas funções quantize_q8_0_split +
    /// matmul_q8_0_actq). Note: o `forward` da CPU usa pesos *repacked* (block_q8_0x8),
    /// cuja ordem de soma difere ligeiramente — por isso a comparação usa tolerância
    /// (~1e-3), não igualdade exata.
    struct CpuMockMatmul;
    impl GpuMatmul for CpuMockMatmul {
        fn matvec_q8_0(
            &self,
            w_bytes: &[u8],
            x: &[f32],
            n_in: usize,
            n_out: usize,
        ) -> Result<Vec<f32>, ModelError> {
            let x_q8 = crate::ops::quantize_q8_0_split(x, n_in, 1);
            Ok(crate::ops::matmul_q8_0_actq(w_bytes, &x_q8, n_in, n_out, 1))
        }
    }

    #[test]
    fn forward_gpu_mock_identico_a_forward_cpu() {
        let Some((bytes, f, cfg)) = load_qwen() else {
            eprintln!("qwen ausente — pulando");
            return;
        };
        let model = crate::model::Model::load_with_config(&f, &bytes, cfg.clone()).unwrap();
        let w = GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
        let mock = CpuMockMatmul;

        let mut c_cpu = model.new_cache();
        let mut c_gpu = model.new_cache();
        let prompt = [cfg.bos_id, 9707u32];
        let _ = model.forward(&prompt, &mut c_cpu).unwrap();
        let _ = model.forward(&prompt, &mut c_gpu).unwrap();
        let next = cfg.bos_id;

        let logits_cpu = model.forward(&[next], &mut c_cpu).unwrap();
        let logits_gpu = model.forward_gpu(&[next], &mut c_gpu, &mock, &w).unwrap();

        assert_eq!(logits_cpu.len(), logits_gpu.len());
        for (i, (a, b)) in logits_cpu.iter().zip(logits_gpu.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "logit[{i}]: cpu={a} gpu={b}");
        }
        eprintln!(
            "forward_gpu(mock) == forward CPU — {} logits",
            logits_gpu.len()
        );
    }

    fn make_greedy_sampler() -> llama_sampling::Sampler {
        llama_sampling::Sampler::Greedy
    }

    #[test]
    fn generate_streaming_gpu_mock_igual_a_cpu() {
        use llama_tokenizer::Tokenizer;
        use rand::{SeedableRng, rngs::SmallRng};

        let Some((bytes, f, cfg)) = load_qwen() else {
            eprintln!("qwen ausente — pulando");
            return;
        };
        let model = crate::model::Model::load_with_config(&f, &bytes, cfg.clone()).unwrap();
        let tok = Tokenizer::from_gguf(&f).unwrap();
        let w = GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
        let sampler = make_greedy_sampler();

        let mut cpu_out = String::new();
        let mut r1 = SmallRng::seed_from_u64(0);
        model
            .generate_streaming(&tok, "Hello", 8, &sampler, &mut r1, &mut |p| {
                cpu_out.push_str(p)
            })
            .unwrap();

        let mut gpu_out = String::new();
        let mut r2 = SmallRng::seed_from_u64(0);
        model
            .generate_streaming_gpu(
                &tok,
                "Hello",
                8,
                &sampler,
                &mut r2,
                &CpuMockMatmul,
                &w,
                &mut |p| gpu_out.push_str(p),
            )
            .unwrap();

        assert_eq!(cpu_out, gpu_out, "saída GPU(mock) deve igualar CPU");
        eprintln!("generate_streaming_gpu(mock) == CPU: {gpu_out:?}");
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
        let (w_q, w_k, _, _) = w.layers[0].attn();
        assert_eq!(w_q.bytes.len(), cfg.n_embd * row_bytes_q);
        assert_eq!(w_k.bytes.len(), kv_dim * row_bytes_q);
        assert_eq!(w.output.bytes.len(), cfg.vocab * row_bytes_q);
        eprintln!("GpuRawWeights OK — {} camadas", w.layers.len());
    }
}
