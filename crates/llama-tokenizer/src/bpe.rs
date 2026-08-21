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
        // zip com o range de u8 no lugar de indexar: os 256 slots casam um a um.
        for (b, slot) in (0u8..=255).zip(t.iter_mut()) {
            *slot = match b {
                33..=126 | 161..=172 | 174..=255 => char::from(b),
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
        bt.iter().copied().zip(0u8..=255).collect()
    })
}

// ── Pre-tokenizador ───────────────────────────────────────────────────────────

/// Variante do pré-tokenizador, escolhida pela chave GGUF `tokenizer.ggml.pre`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Pretok {
    /// Qwen2 / GPT-4 — o que sempre usamos, e o destino de todo `pre` que não seja qwen35.
    #[default]
    Qwen2,
    /// Qwen3.5 (`pre = "qwen35"`): a marca combinante fica na run da letra.
    Qwen35,
}

impl Pretok {
    /// Mapeia o valor de `tokenizer.ggml.pre`. Só qwen35 diverge — os demais pres
    /// (e a ausência da chave) ficam no Qwen2.
    pub(crate) fn from_pre(pre: &str) -> Pretok {
        match pre {
            "qwen35" => Pretok::Qwen35,
            _ => Pretok::Qwen2,
        }
    }
}

/// Regex do pre-tokenizador Qwen2 / GPT-4 (sem lookahead negativo).
// O padrão é literal: se não compilar, é erro de digitação que o primeiro teste pega.
#[allow(clippy::expect_used)]
fn pretok_re(pre: Pretok) -> &'static Regex {
    static QWEN2: OnceLock<Regex> = OnceLock::new();
    static QWEN35: OnceLock<Regex> = OnceLock::new();
    match pre {
        // Derivado de: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
        //              ' '?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+
        // (?!\S) removido — irrelevante para prompts ASCII sem trailing whitespace.
        Pretok::Qwen2 => QWEN2.get_or_init(|| {
            Regex::new(
                r"(?:'[sStTdDmM]|'re|'RE|'ve|'VE|'ll|'LL)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
            )
            .expect("regex BPE válida")
        }),
        // Mesma alternação, com as duas divergências do qwen35 (llama-vocab.cpp:382-388):
        // a run de letras vira [\p{L}\p{M}]+ e a de não-letras passa a excluir \p{M} —
        // a marca combinante acompanha a letra, não a pontuação que veio antes.
        Pretok::Qwen35 => QWEN35.get_or_init(|| {
            Regex::new(
                r"(?:'[sStTdDmM]|'re|'RE|'ve|'VE|'ll|'LL)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
            )
            .expect("regex BPE qwen35 válida")
        }),
    }
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

    for piece in pretok_re(vocab.pre).find_iter(text) {
        let piece_str = piece.as_str();

        // Converte cada byte do piece para o char GPT-2 correspondente e busca o ID.
        let mut ids: Vec<u32> = piece_str
            .bytes()
            .filter_map(|b| {
                let c = *bt.get(usize::from(b))?;
                let s = c.encode_utf8(&mut char_buf);
                vocab.text_to_token(s)
            })
            .collect();

        // Loop de merge: encontra o par de menor rank e funde.
        let mut merge_buf = String::with_capacity(32);
        while let Some((pos, _)) = ids
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                let (a, b) = (*w.first()?, *w.get(1)?);
                merge_ranks.get(&(a, b)).map(|&r| (i, r))
            })
            .min_by_key(|&(_, r)| r)
        {
            merge_buf.clear();
            if let Some(a) = ids.get(pos).and_then(|&id| vocab.token_text(id)) {
                merge_buf.push_str(a);
            }
            if let Some(b) = ids.get(pos + 1).and_then(|&id| vocab.token_text(id)) {
                merge_buf.push_str(b);
            }
            if let Some(merged_id) = vocab.text_to_token(&merge_buf) {
                if let Some(slot) = ids.get_mut(pos) {
                    *slot = merged_id;
                }
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

/// Decodifica texto GPT-2 encoded (com 'Ġ' etc.) de volta para os bytes originais.
///
/// Pode terminar no meio de um caractere multi-byte — é o caso de um token que carrega
/// só o primeiro byte de um 'ç'. Quem monta texto incremental precisa desses bytes.
pub(crate) fn decode_bpe_bytes(encoded: &str) -> Vec<u8> {
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
    bytes
}

/// Como `decode_bpe_bytes`, mas fechando os buracos com U+FFFD.
pub(crate) fn decode_bpe(encoded: &str) -> String {
    String::from_utf8_lossy(&decode_bpe_bytes(encoded)).into_owned()
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
        for (b, &c) in (0u8..=255).zip(bt.iter()) {
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
        let pieces: Vec<&str> = pretok_re(Pretok::Qwen2)
            .find_iter("Once upon a time")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(pieces, vec!["Once", " upon", " a", " time"]);
    }

    fn pedacos(pre: Pretok, texto: &str) -> Vec<&str> {
        pretok_re(pre)
            .find_iter(texto)
            .map(|m| m.as_str())
            .collect()
    }

    /// A divergência que motiva a chave `pre`: com a marca combinante separada
    /// (NFD), o qwen35 mantém a palavra inteira numa run só; o qwen2 a parte em duas,
    /// porque `\p{L}+` para na letra e a marca vira o prefixo do pedaço seguinte.
    #[test]
    fn qwen35_absorve_a_marca_combinante_na_run_de_letras() {
        let texto = "e\u{301}xito"; // "êxito" em NFD: 'e' + acento agudo combinante

        assert_eq!(pedacos(Pretok::Qwen35, texto), vec![texto]);
        assert_eq!(pedacos(Pretok::Qwen2, texto), vec!["e", "\u{301}xito"]);
    }

    /// A outra metade da divergência: a classe de não-letras do qwen35 exclui `\p{M}`,
    /// então a marca não entra na run de pontuação. Precisa de **dois** sinais — com um
    /// só, o prefixo opcional `[^\r\n\p{L}\p{N}]?` o gruda na run de letras nas duas
    /// variantes, e a diferença não aparece.
    #[test]
    fn qwen35_nao_deixa_a_marca_combinante_na_run_de_pontuacao() {
        let texto = "!!\u{301}";

        assert_eq!(pedacos(Pretok::Qwen35, texto), vec!["!!", "\u{301}"]);
        assert_eq!(pedacos(Pretok::Qwen2, texto), vec![texto]);
    }

    /// Texto sem marca combinante tem de sair igual pelos dois pré-tokenizadores.
    #[test]
    fn as_duas_variantes_concordam_sem_marca_combinante() {
        let texto = "Once upon a time, 42 vezes.\n";

        assert_eq!(
            pedacos(Pretok::Qwen35, texto),
            pedacos(Pretok::Qwen2, texto)
        );
    }

    #[test]
    fn from_pre_so_reconhece_qwen35() {
        assert_eq!(Pretok::from_pre("qwen35"), Pretok::Qwen35);
        assert_eq!(Pretok::from_pre("qwen2"), Pretok::Qwen2);
        assert_eq!(Pretok::from_pre(""), Pretok::Qwen2);
        assert_eq!(
            Pretok::default(),
            Pretok::Qwen2,
            "sem a chave, é o de sempre"
        );
    }
}
