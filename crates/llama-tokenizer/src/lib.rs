#![forbid(unsafe_code)]
//! Tokenizer SPM (Llama) — encode/decode bit-exact vs llama.cpp.

mod error;
mod spm;
mod vocab;

pub use error::TokenizerError;
pub use vocab::Vocab;

use gguf::GgufFile;

const SPACE_ESCAPE: &str = "\u{2581}"; // ▁ = E2 96 81

/// Tokenizer SPM (Llama).
pub struct Tokenizer {
    vocab: Vocab,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    pub fn from_gguf(f: &GgufFile) -> Result<Self, TokenizerError> {
        Ok(Self { vocab: Vocab::from_gguf(f)? })
    }

    /// Codifica `text` em ids. Com `add_bos`, prefixa o token BOS e um espaço
    /// (add_space_prefix), espelhando o pipeline SPM do llama.cpp.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut output = Vec::new();
        let mut is_prev_special = false;
        if add_bos {
            output.push(self.vocab.bos_id);
            is_prev_special = true;
        }
        let mut buf = String::new();
        if is_prev_special {
            buf.push(' ');
        }
        buf.push_str(text);
        let normalized = buf.replace(' ', SPACE_ESCAPE);
        let ids = crate::spm::tokenize_spm(&self.vocab, &normalized);
        output.extend(ids);
        output
    }

    /// Decodifica ids em texto: concatena os textos dos tokens e reverte `▁`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == self.vocab.bos_id || id == self.vocab.eos_id {
                continue;
            }
            if let Some(t) = self.vocab.token_text(id) {
                out.push_str(t);
            }
        }
        let out = out.replace(SPACE_ESCAPE, " ");
        // remove o espaço de prefixo introduzido no encode
        out.strip_prefix(' ').map(String::from).unwrap_or(out)
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
        Vocab::new(tokens, scores, types, 1, 2, 0)
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
        // sem add_bos → sem prefixo de espaço (is_prev_special começa false)
        assert_eq!(t.encode("hi", false), vec![4]);
    }

    #[test]
    fn decode_roundtrip_text() {
        let t = Tokenizer::new(tiny());
        let ids = t.encode("hi", true);
        assert_eq!(t.decode(&ids), "hi");
    }
}
