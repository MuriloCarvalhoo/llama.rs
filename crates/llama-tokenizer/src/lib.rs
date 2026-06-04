#![forbid(unsafe_code)]
//! Tokenizer SPM (Llama) e BPE (Qwen2/GPT-2) — encode/decode bit-exact vs llama.cpp.

mod bpe;
mod error;
mod spm;
mod vocab;

pub use error::TokenizerError;
pub use vocab::Vocab;

use std::collections::HashMap;

use gguf::GgufFile;

const SPACE_ESCAPE: &str = "\u{2581}"; // ▁ = E2 96 81

/// Tokenizer SPM (Llama) ou BPE (Qwen2/GPT-2).
pub struct Tokenizer {
    vocab: Vocab,
    /// Mapa de rank de merge para BPE; vazio em tokenizers SPM.
    merge_ranks: HashMap<(u32, u32), u32>,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        let merge_ranks = vocab.merge_ranks();
        Self { vocab, merge_ranks }
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
}
