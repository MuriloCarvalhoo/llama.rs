//! Vocabulário SPM extraído dos metadados GGUF.

use std::collections::HashMap;

use gguf::GgufFile;

use crate::error::TokenizerError;

const HEX: &[u8; 16] = b"0123456789ABCDEF";

/// Vocabulário: tokens, scores, tipos e ids especiais.
pub struct Vocab {
    tokens: Vec<String>,
    scores: Vec<f32>,
    #[allow(dead_code)]
    token_types: Vec<i32>,
    token_to_id: HashMap<String, u32>,
    pub(crate) bos_id: u32,
    pub(crate) eos_id: u32,
    #[allow(dead_code)]
    pub(crate) unk_id: u32,
    /// Pares de merge para BPE (índice = rank). Vazio em vocabs SPM.
    pub(crate) merges: Vec<(u32, u32)>,
}

impl Vocab {
    /// Construtor direto (usado em testes e por `from_gguf`).
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        bos_id: u32,
        eos_id: u32,
        unk_id: u32,
        merges: Vec<(u32, u32)>,
    ) -> Vocab {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            // Em colisão, o último id vence — espelha `token_to_id[word] = i`
            // do llama.cpp (llama-vocab.cpp:2404). Vocabs SPM válidos não têm
            // duplicatas (upstream aborta via GGML_ASSERT), então é inerte na
            // prática, mas mantemos a semântica idêntica por fidelidade 1:1.
            if let Ok(id) = u32::try_from(i) {
                token_to_id.insert(t.clone(), id);
            }
        }
        Vocab {
            tokens,
            scores,
            token_types,
            token_to_id,
            bos_id,
            eos_id,
            unk_id,
            merges,
        }
    }

    /// Constrói HashMap de rank de merge para BPE: (id_a, id_b) → rank (índice no array).
    pub(crate) fn merge_ranks(&self) -> HashMap<(u32, u32), u32> {
        self.merges
            .iter()
            .enumerate()
            .map(|(rank, &pair)| {
                let rank = u32::try_from(rank).unwrap_or(u32::MAX);
                (pair, rank)
            })
            .collect()
    }

    /// Lê o vocab (SPM ou BPE) dos metadados de um GGUF já parseado.
    pub fn from_gguf(f: &GgufFile) -> Result<Vocab, TokenizerError> {
        let model = f
            .get("tokenizer.ggml.model")?
            .as_str("tokenizer.ggml.model")?;

        match model {
            "llama" | "gpt2" => {}
            other => return Err(TokenizerError::UnsupportedModel(other.to_owned())),
        }

        let tokens: Vec<String> = f
            .get("tokenizer.ggml.tokens")?
            .as_string_array("tokenizer.ggml.tokens")?
            .to_vec();

        // scores: obrigatório em SPM, ausente em BPE — default zeros
        let scores: Vec<f32> = match f.metadata.get("tokenizer.ggml.scores") {
            Some(v) => v.as_f32_array("tokenizer.ggml.scores")?.to_vec(),
            None => vec![0.0; tokens.len()],
        };

        // token_type: presente em ambos, mas defensivamente opcional
        let token_types: Vec<i32> = match f.metadata.get("tokenizer.ggml.token_type") {
            Some(v) => v.as_i32_array("tokenizer.ggml.token_type")?.to_vec(),
            None => vec![1; tokens.len()],
        };

        if tokens.len() != scores.len() || tokens.len() != token_types.len() {
            return Err(TokenizerError::InconsistentVocab {
                tokens: tokens.len(),
                scores: scores.len(),
                types: token_types.len(),
            });
        }

        let bos_id = f.get("tokenizer.ggml.bos_token_id")?.as_u32("bos")?;
        let eos_id = f.get("tokenizer.ggml.eos_token_id")?.as_u32("eos")?;
        // unk_id ausente em Qwen2 — default 0
        let unk_id = match f.metadata.get("tokenizer.ggml.unknown_token_id") {
            Some(v) => v.as_u32("unk")?,
            None => 0,
        };

        // Merges BPE: presente apenas em model="gpt2"
        let merges = if model == "gpt2" {
            let merge_strings = f
                .get("tokenizer.ggml.merges")?
                .as_string_array("tokenizer.ggml.merges")?;

            // token_to_id inline para resolver strings → IDs durante o parse
            let mut token_to_id: HashMap<&str, u32> = HashMap::with_capacity(tokens.len());
            for (i, t) in tokens.iter().enumerate() {
                if let Ok(id) = u32::try_from(i) {
                    token_to_id.insert(t.as_str(), id);
                }
            }

            let mut result = Vec::with_capacity(merge_strings.len());
            for s in merge_strings {
                // formato: "a b" (dois tokens separados por exatamente um espaço)
                if let Some(sp) = s.find(' ') {
                    let a = &s[..sp];
                    let b = &s[sp + 1..];
                    if let (Some(&ia), Some(&ib)) = (token_to_id.get(a), token_to_id.get(b)) {
                        result.push((ia, ib));
                    }
                }
            }
            result
        } else {
            Vec::new()
        };

        Ok(Vocab::new(
            tokens,
            scores,
            token_types,
            bos_id,
            eos_id,
            unk_id,
            merges,
        ))
    }

    pub(crate) fn text_to_token(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    pub(crate) fn score(&self, id: u32) -> f32 {
        self.scores
            .get(id as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY)
    }

    /// Byte → token. Tenta `<0xXX>` (hex maiúsculo), depois o byte como string
    /// de 1 caractere (espelha `llama_vocab::byte_to_token` para SPM).
    pub(crate) fn byte_to_token(&self, ch: u8) -> Option<u32> {
        let hi = HEX.get((ch >> 4) as usize).copied().unwrap_or(b'0');
        let lo = HEX.get((ch & 0x0F) as usize).copied().unwrap_or(b'0');
        let buf = [b'<', b'0', b'x', hi, lo, b'>'];
        // `buf` é sempre ASCII válido.
        let key = core::str::from_utf8(&buf).unwrap_or("<0x00>");
        if let Some(id) = self.token_to_id.get(key).copied() {
            return Some(id);
        }
        let single = [ch];
        core::str::from_utf8(&single)
            .ok()
            .and_then(|s| self.token_to_id.get(s).copied())
    }

    pub(crate) fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Vocab {
        // ids:      0       1      2      3        4        5
        let tokens = vec!["<unk>", "<s>", "</s>", "<0x41>", "ab", "abc"]
            .into_iter()
            .map(String::from)
            .collect();
        let scores = vec![0.0, 0.0, 0.0, 0.0, -1.0, -0.5];
        let token_types = vec![2, 3, 3, 6, 1, 1];
        Vocab::new(tokens, scores, token_types, 1, 2, 0, vec![])
    }

    #[test]
    fn text_to_token_lookup() {
        let v = tiny();
        assert_eq!(v.text_to_token("ab"), Some(4));
        assert_eq!(v.text_to_token("zzz"), None);
    }

    #[test]
    fn byte_to_token_uppercase_hex() {
        let v = tiny();
        // 0x41 = 'A' → token "<0x41>" = id 3
        assert_eq!(v.byte_to_token(0x41), Some(3));
    }

    #[test]
    fn score_lookup() {
        let v = tiny();
        assert_eq!(v.score(5), -0.5);
    }
}
