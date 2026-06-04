//! Núcleo do algoritmo SPM (merge por score + byte-fallback).
//! Réplica fiel de `llm_tokenizer_spm_session` do llama.cpp.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::vocab::Vocab;

/// Símbolo na cadeia: fatia `[start, start+len)` dos bytes normalizados.
struct Symbol {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
}

/// Bigrama candidato a merge.
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bigram {
    /// "Maior" = maior score; empate → menor `left` (espelha o comparator do C++:
    /// `l < r` se `l.score < r.score || (== && l.left > r.left)`).
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.total_cmp(&o.score).then_with(|| o.left.cmp(&self.left))
    }
}

/// Comprimento em bytes de um char UTF-8 a partir do primeiro byte.
fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1, // continuação/inválido: trata como 1 byte (como o min() do C++)
    }
}

/// Tokeniza bytes já normalizados (espaços já viraram `▁`).
pub(crate) fn tokenize_spm(vocab: &Vocab, text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut symbols: Vec<Symbol> = Vec::new();

    // 1. Divide em símbolos UTF-8.
    let mut offs = 0usize;
    let mut index = 0i32;
    while offs < bytes.len() {
        let len = utf8_len(bytes[offs]).min(bytes.len() - offs);
        let next = if offs + len == bytes.len() { -1 } else { index + 1 };
        symbols.push(Symbol { start: offs, len, prev: index - 1, next });
        offs += len;
        index += 1;
    }

    let mut work: BinaryHeap<Bigram> = BinaryHeap::new();
    let mut rev_merge: HashMap<(usize, usize), (i32, i32)> = HashMap::new();

    let try_add_bigram =
        |work: &mut BinaryHeap<Bigram>,
         rev_merge: &mut HashMap<(usize, usize), (i32, i32)>,
         symbols: &[Symbol],
         left: i32,
         right: i32| {
            if left == -1 || right == -1 {
                return;
            }
            let l = &symbols[left as usize];
            let r = &symbols[right as usize];
            let start = l.start;
            let size = l.len + r.len;
            let Ok(text) = core::str::from_utf8(&bytes[start..start + size]) else {
                return;
            };
            let Some(id) = vocab.text_to_token(text) else {
                return;
            };
            work.push(Bigram { left, right, score: vocab.score(id), size });
            rev_merge.insert((start, size), (left, right));
        };

    // 2. Semeia bigramas adjacentes.
    for i in 1..symbols.len() as i32 {
        try_add_bigram(&mut work, &mut rev_merge, &symbols, i - 1, i);
    }

    // 3. Funde o par de maior score enquanto houver.
    while let Some(bigram) = work.pop() {
        let (ln, rn) = {
            let l = &symbols[bigram.left as usize];
            let r = &symbols[bigram.right as usize];
            (l.len, r.len)
        };
        if ln == 0 || rn == 0 || ln + rn != bigram.size {
            continue; // um dos símbolos já foi fundido
        }
        // funde right em left
        let right_next = symbols[bigram.right as usize].next;
        {
            let l = &mut symbols[bigram.left as usize];
            l.len += rn;
            l.next = right_next;
        }
        symbols[bigram.right as usize].len = 0;
        if right_next >= 0 {
            symbols[right_next as usize].prev = bigram.left;
        }
        let left_prev = symbols[bigram.left as usize].prev;
        let left_next = symbols[bigram.left as usize].next;
        try_add_bigram(&mut work, &mut rev_merge, &symbols, left_prev, bigram.left);
        try_add_bigram(&mut work, &mut rev_merge, &symbols, bigram.left, left_next);
    }

    // 4. Resegmenta a cadeia final.
    let mut output = Vec::new();
    let mut i = 0i32;
    while i != -1 && (i as usize) < symbols.len() {
        let (start, len) = {
            let s = &symbols[i as usize];
            (s.start, s.len)
        };
        resegment(vocab, bytes, &symbols, &rev_merge, start, len, &mut output);
        i = symbols[i as usize].next;
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn resegment(
    vocab: &Vocab,
    bytes: &[u8],
    symbols: &[Symbol],
    rev_merge: &HashMap<(usize, usize), (i32, i32)>,
    start: usize,
    len: usize,
    output: &mut Vec<u32>,
) {
    if let Ok(text) = core::str::from_utf8(&bytes[start..start + len]) {
        if let Some(id) = vocab.text_to_token(text) {
            output.push(id);
            return;
        }
    }
    match rev_merge.get(&(start, len)) {
        Some(&(left, right)) => {
            let l = &symbols[left as usize];
            let r = &symbols[right as usize];
            resegment(vocab, bytes, symbols, rev_merge, l.start, l.len, output);
            resegment(vocab, bytes, symbols, rev_merge, r.start, r.len, output);
        }
        None => {
            // byte-fallback: cada byte vira <0xXX> (ou byte cru).
            for j in 0..len {
                if let Some(id) = vocab.byte_to_token(bytes[start + j]) {
                    output.push(id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    fn tiny() -> Vocab {
        // "abc" tem score maior (-0.5) que "ab" (-1.0); merge deve preferir "abc".
        let tokens = vec!["<unk>", "<s>", "</s>", "<0x64>", "a", "b", "c", "ab", "abc"]
            .into_iter()
            .map(String::from)
            .collect();
        let scores = vec![0.0, 0.0, 0.0, 0.0, -3.0, -3.0, -3.0, -1.0, -0.5];
        let types = vec![2, 3, 3, 6, 1, 1, 1, 1, 1];
        Vocab::new(tokens, scores, types, 1, 2, 0)
    }

    #[test]
    fn merges_highest_score_first() {
        let v = tiny();
        // "abc" → deve resultar no único token id 8 ("abc"), não em "ab"+"c".
        assert_eq!(tokenize_spm(&v, "abc"), vec![8]);
    }

    #[test]
    fn byte_fallback_for_unknown_char() {
        let v = tiny();
        // 'd' (0x64) não está como char, mas "<0x64>" sim (id 3).
        assert_eq!(tokenize_spm(&v, "d"), vec![3]);
    }

    #[test]
    fn splits_when_no_merge() {
        let v = tiny();
        // "ba" não casa nenhum merge melhor → tokens "b","a" = [5,4].
        assert_eq!(tokenize_spm(&v, "ba"), vec![5, 4]);
    }
}
