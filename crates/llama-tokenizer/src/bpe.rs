//! BPE tokenizer (GPT-2 / Qwen2 byte-level BPE).
//!
//! Pipeline por piece:
//!   pre-tokenize → byte→unicode → lookup IDs → merge loop

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use crate::vocab::Vocab;

// ── Byte-to-unicode table ─────────────────────────────────────────────────────

/// GPT-2 byte-to-unicode: mapeia cada byte (0-255) para um char único.
///
/// Bytes "limpos" (printável ASCII + Latin-1 parcial) ficam como-é;
/// os 68 bytes restantes mapeiam para U+0100…U+0143.
/// Em especial, byte 32 (espaço) → U+0120 'Ġ'.
fn byte_table() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = ['\0'; 256];
        let mut hi = 256u32;
        for b in 0u16..=255 {
            let b8 = b as u8;
            t[b as usize] = match b8 {
                33..=126 | 161..=172 | 174..=255 => b8 as char,
                _ => {
                    let c = char::from_u32(hi).unwrap_or('\u{FFFD}');
                    hi += 1;
                    c
                }
            };
        }
        t
    })
}

/// Inversa do byte_table: char GPT-2 → byte original.
fn unicode_to_byte() -> &'static HashMap<char, u8> {
    static MAP: OnceLock<HashMap<char, u8>> = OnceLock::new();
    MAP.get_or_init(|| {
        let bt = byte_table();
        bt.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect()
    })
}

// ── Pre-tokenizador ───────────────────────────────────────────────────────────

/// Regex do pre-tokenizador Qwen2 / GPT-4 (sem lookahead negativo).
fn pretok_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Derivado de: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
        //              ' '?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+
        // (?!\S) removido — irrelevante para prompts ASCII sem trailing whitespace.
        Regex::new(
            r"(?:'[sStTdDmM]|'re|'RE|'ve|'VE|'ll|'LL)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
        )
        .expect("regex BPE válida")
    })
}

// ── Tokenização ───────────────────────────────────────────────────────────────

/// Tokeniza `text` com BPE byte-level usando `merge_ranks`.
///
/// Retorna Vec de IDs de tokens (sem BOS/EOS).
pub(crate) fn tokenize_bpe(
    vocab: &Vocab,
    merge_ranks: &HashMap<(u32, u32), u32>,
    text: &str,
) -> Vec<u32> {
    let bt = byte_table();
    let mut output = Vec::new();
    let mut char_buf = [0u8; 4];

    for piece in pretok_re().find_iter(text) {
        let piece_str = piece.as_str();

        // Converte cada byte do piece para o char GPT-2 correspondente e busca o ID.
        let mut ids: Vec<u32> = piece_str
            .bytes()
            .filter_map(|b| {
                let c = bt[b as usize];
                let s = c.encode_utf8(&mut char_buf);
                vocab.text_to_token(s)
            })
            .collect();

        // Loop de merge: encontra o par de menor rank e funde.
        let mut merge_buf = String::with_capacity(32);
        loop {
            let Some((pos, _)) = ids
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| merge_ranks.get(&(w[0], w[1])).map(|&r| (i, r)))
                .min_by_key(|&(_, r)| r)
            else {
                break;
            };

            merge_buf.clear();
            if let Some(a) = vocab.token_text(ids[pos]) {
                merge_buf.push_str(a);
            }
            if let Some(b) = vocab.token_text(ids[pos + 1]) {
                merge_buf.push_str(b);
            }
            if let Some(merged_id) = vocab.text_to_token(&merge_buf) {
                ids[pos] = merged_id;
                ids.remove(pos + 1);
            } else {
                break;
            }
        }

        output.extend_from_slice(&ids);
    }

    output
}

// ── Decodificação ─────────────────────────────────────────────────────────────

/// Decodifica texto GPT-2 encoded (com 'Ġ' etc.) de volta para UTF-8 original.
pub(crate) fn decode_bpe(encoded: &str) -> String {
    let u2b = unicode_to_byte();
    let mut bytes = Vec::with_capacity(encoded.len());
    for c in encoded.chars() {
        if let Some(&b) = u2b.get(&c) {
            bytes.push(b);
        } else {
            // Char não está na tabela — escreve diretamente (ex: emojis no vocab).
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            bytes.extend_from_slice(s.as_bytes());
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

// ── Testes ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_space_is_gstroke() {
        // byte 32 (space) deve mapear para U+0120 'Ġ'
        assert_eq!(byte_table()[32], '\u{0120}');
    }

    #[test]
    fn byte_table_printable_ascii_identity() {
        // bytes 33-126 mapeiam para si mesmos
        assert_eq!(byte_table()[33], '!');
        assert_eq!(byte_table()[126], '~');
    }

    #[test]
    fn unicode_to_byte_roundtrip() {
        let u2b = unicode_to_byte();
        let bt = byte_table();
        for b in 0u8..=255 {
            let c = bt[b as usize];
            assert_eq!(u2b[&c], b, "roundtrip falhou para byte {b}");
        }
    }

    #[test]
    fn decode_bpe_reverses_space() {
        // 'Ġ' (U+0120) deve decodificar para espaço
        assert_eq!(decode_bpe("\u{0120}hello"), " hello");
    }

    #[test]
    fn pretokenize_simple_sentence() {
        let pieces: Vec<&str> = pretok_re()
            .find_iter("Once upon a time")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(pieces, vec!["Once", " upon", " a", " time"]);
    }
}
