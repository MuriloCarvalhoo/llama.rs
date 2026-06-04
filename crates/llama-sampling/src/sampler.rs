//! Estratégias de amostragem: greedy, temperatura, top-k, top-p.
#![allow(clippy::indexing_slicing)]

use rand::Rng;

/// Estratégia de amostragem para selecionar o próximo token a partir de logits.
#[derive(Clone, Debug)]
pub enum Sampler {
    /// Argmax — determinístico, equivale a temperatura zero.
    Greedy,
    /// Multinomial com rescala de logits por `1/temp`. Se `temp == 0.0` → greedy.
    Temperature { temp: f32 },
    /// Mantém os `k` maiores logits antes de amostrar. Se `temp == 0.0` → greedy.
    TopK { k: usize, temp: f32 },
    /// Mantém o menor conjunto de tokens com prob. acumulada >= `p` antes de amostrar.
    TopP { p: f32, temp: f32 },
}

impl Sampler {
    /// Retorna o índice do token amostrado dado o vetor de logits.
    pub fn sample(&self, logits: &[f32], rng: &mut impl Rng) -> usize {
        match self {
            Sampler::Greedy => argmax(logits),
            Sampler::Temperature { temp } => {
                todo!("implementado na Task 1: temp={temp}")
            }
            Sampler::TopK { k, temp } => {
                todo!("implementado na Task 2: k={k} temp={temp}")
            }
            Sampler::TopP { p, temp } => {
                todo!("implementado na Task 2: p={p} temp={temp}")
            }
        }
    }
}

pub(crate) fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

pub(crate) fn sample_multinomial(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.random();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn greedy_returns_argmax() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
        assert_eq!(Sampler::Greedy.sample(&logits, &mut rng()), 3);
    }

    #[test]
    fn greedy_single_token() {
        assert_eq!(Sampler::Greedy.sample(&[1.0f32], &mut rng()), 0);
    }

    #[test]
    fn argmax_picks_max_index() {
        assert_eq!(argmax(&[0.0, 1.0, 0.5]), 1);
    }

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn softmax_with_negative_logits() {
        let probs = softmax(&[-1.0, -2.0, -3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
    }

    #[test]
    fn sample_multinomial_single_prob() {
        let mut r = SmallRng::seed_from_u64(1);
        assert_eq!(sample_multinomial(&[1.0], &mut r), 0);
    }
}
