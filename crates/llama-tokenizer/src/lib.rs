#![forbid(unsafe_code)]
//! Tokenizer SPM (Llama) e BPE (Qwen2/GPT-2) — encode/decode bit-exact vs llama.cpp.
//!
//! Contexto delimitado (DDD): tokenização, isolado do domínio do modelo.
//! - Value object: [`Vocab`] — tokens/scores/merges lidos do GGUF, imutável após
//!   construção.
//! - Serviço de domínio (facade): [`Tokenizer`] — decide a estratégia (SPM ou BPE)
//!   e expõe `encode`/`decode`; delega às estratégias privadas em `spm`/`bpe`.

mod bpe;
mod error;
mod spm;
mod vocab;

pub use error::TokenizerError;
pub use vocab::Vocab;

use std::collections::HashMap;

use gguf::GgufFile;
use regex::Regex;

const SPACE_ESCAPE: &str = "\u{2581}"; // ▁ = E2 96 81

/// Tokenizer SPM (Llama) ou BPE (Qwen2/GPT-2).
pub struct Tokenizer {
    vocab: Vocab,
    /// Mapa de rank de merge para BPE; vazio em tokenizers SPM.
    merge_ranks: HashMap<(u32, u32), u32>,
    /// Alternação dos tokens especiais, do mais longo para o mais curto.
    /// `None` quando o vocab não tem nenhum — ou quando a alternação não compila.
    especiais: Option<Regex>,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        let merge_ranks = vocab.merge_ranks();
        let alternacao = vocab
            .especiais()
            .iter()
            .map(|(t, _)| regex::escape(t))
            .collect::<Vec<_>>()
            .join("|");
        let especiais = if alternacao.is_empty() {
            None
        } else {
            Regex::new(&alternacao).ok()
        };
        Self {
            vocab,
            merge_ranks,
            especiais,
        }
    }

    pub fn from_gguf(f: &GgufFile) -> Result<Self, TokenizerError> {
        Ok(Self::new(Vocab::from_gguf(f)?))
    }

    /// Codifica `text` em ids.
    ///
    /// SPM: prefixa BOS + espaço (add_space_prefix) quando `add_bos=true`.
    /// BPE: prefixa apenas BOS; sem add_space_prefix (Qwen2 usa Ġ no vocab).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut output = Vec::new();
        if add_bos {
            output.push(self.vocab.bos_id);
        }

        if self.merge_ranks.is_empty() {
            // SPM path
            let mut buf = String::new();
            if add_bos {
                buf.push(' ');
            }
            buf.push_str(text);
            let normalized = buf.replace(' ', SPACE_ESCAPE);
            output.extend(crate::spm::tokenize_spm(&self.vocab, &normalized));
        } else {
            // BPE path
            output.extend(bpe::tokenize_bpe(&self.vocab, &self.merge_ranks, text));
        }

        output
    }

    /// Como `encode`, mas resolve os marcadores do chat template (`<|im_start|>`,
    /// `<tool_call>`, ...) para o id único de cada um, em vez de deixá-los virar
    /// texto comum. Não prefixa BOS: o template já traz tudo o que o modelo espera.
    pub fn encode_special(&self, text: &str) -> Vec<u32> {
        let Some(re) = self.especiais.as_ref() else {
            return self.encode(text, false);
        };
        let mut out = Vec::new();
        let mut ini = 0usize;
        for m in re.find_iter(text) {
            if let Some(antes) = text.get(ini..m.start()).filter(|s| !s.is_empty()) {
                out.extend(self.encode(antes, false));
            }
            if let Some(id) = self.vocab.text_to_token(m.as_str()) {
                out.push(id);
            }
            ini = m.end();
        }
        if let Some(resto) = text.get(ini..).filter(|s| !s.is_empty()) {
            out.extend(self.encode(resto, false));
        }
        out
    }

    /// Decodifica ids em texto, revertendo a codificação SPM ou BPE.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut encoded = String::new();
        for &id in ids {
            if id == self.vocab.bos_id || id == self.vocab.eos_id {
                continue;
            }
            if let Some(t) = self.vocab.token_text(id) {
                encoded.push_str(t);
            }
        }

        if self.merge_ranks.is_empty() {
            // SPM decode: reverte ▁ → espaço e remove espaço de prefixo
            let out = encoded.replace(SPACE_ESCAPE, " ");
            out.strip_prefix(' ').map(String::from).unwrap_or(out)
        } else {
            // BPE decode: reverte mapeamento byte-to-unicode
            bpe::decode_bpe(&encoded)
        }
    }
}

/// Bytes crus dos tokens, sem fechar caractere partido — o que o streaming precisa.
///
/// `decode` devolve texto e, num caractere dividido entre dois tokens, substitui o
/// pedaço por U+FFFD; aqui os bytes saem como estão e quem consome junta os pedaços.
impl Tokenizer {
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut encoded = String::new();
        for &id in ids {
            if id == self.vocab.bos_id || id == self.vocab.eos_id {
                continue;
            }
            if let Some(t) = self.vocab.token_text(id) {
                encoded.push_str(t);
            }
        }
        if self.merge_ranks.is_empty() {
            let out = encoded.replace(SPACE_ESCAPE, " ");
            let out = out.strip_prefix(' ').map_or(out.clone(), String::from);
            out.into_bytes()
        } else {
            bpe::decode_bpe_bytes(&encoded)
        }
    }
}

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
        Vocab::new(tokens, scores, types, 1, 2, 0, vec![])
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
        // sem add_bos → sem prefixo de espaço
        assert_eq!(t.encode("hi", false), vec![4]);
    }

    #[test]
    fn decode_roundtrip_text() {
        let t = Tokenizer::new(tiny());
        let ids = t.encode("hi", true);
        assert_eq!(t.decode(&ids), "hi");
    }

    /// Vocab BPE mínimo com tokens de controle (type 3) e user-defined (type 4).
    fn bpe_com_especiais() -> Vocab {
        // ids:      0       1                2               3             4    5    6     7
        let tokens = vec![
            "<unk>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "h",
            "i",
            "hi",
            "<tool_call>",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let scores = vec![0.0; 8];
        let types = vec![2, 3, 3, 3, 1, 1, 1, 4];
        // merge "h" + "i" -> "hi"
        Vocab::new(tokens, scores, types, 1, 3, 0, vec![(4, 5)])
    }

    #[test]
    fn encode_special_resolve_o_marcador_como_um_token_so() {
        let t = Tokenizer::new(bpe_com_especiais());
        assert_eq!(t.encode_special("<|im_start|>"), vec![2]);
    }

    #[test]
    fn encode_special_tokeniza_o_texto_entre_os_marcadores() {
        let t = Tokenizer::new(bpe_com_especiais());
        assert_eq!(t.encode_special("<|im_start|>hi<|im_end|>"), vec![2, 6, 3]);
    }

    #[test]
    fn encode_special_cobre_tambem_os_user_defined() {
        let t = Tokenizer::new(bpe_com_especiais());
        assert_eq!(t.encode_special("<tool_call>hi"), vec![7, 6]);
    }

    /// Em BPE byte-level um caractere multi-byte pode ficar partido entre dois tokens.
    /// `decode` fecha o buraco com U+FFFD; para streaming isso é perda irreversível,
    /// então quem monta o texto aos poucos precisa dos bytes crus.
    #[test]
    fn decode_bytes_nao_substitui_o_caractere_partido() {
        // "ç" = C3 A7. No vocab GPT-2, C3 → 'Ã' (U+00C3) e A7 → '§' (U+00A7).
        let tokens = vec!["<unk>", "\u{c3}", "\u{a7}", "a"]
            .into_iter()
            .map(String::from)
            .collect();
        let v = Vocab::new(
            tokens,
            vec![0.0; 4],
            vec![2, 1, 1, 1],
            0,
            0,
            0,
            vec![(1, 2)],
        );
        let t = Tokenizer::new(v);

        assert_eq!(t.decode_bytes(&[1]), vec![0xC3]);
        assert_eq!(t.decode_bytes(&[1, 2]), "ç".as_bytes());
        assert_eq!(t.decode(&[1]), "\u{fffd}", "o decode em texto perde o byte");
    }
}
