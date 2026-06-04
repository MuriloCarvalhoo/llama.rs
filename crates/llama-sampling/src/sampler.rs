//! Estratégias de amostragem: greedy, temperatura, top-k, top-p.

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
        debug_assert!(!logits.is_empty(), "logits slice must not be empty");
        match self {
            Sampler::Greedy => argmax(logits),
            Sampler::Temperature { temp } => {
                if *temp == 0.0 {
                    return argmax(logits);
                }
                debug_assert!(*temp > 0.0, "temperature must be positive, got {temp}");
                let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
                let probs = softmax(&scaled);
                sample_multinomial(&probs, rng)
            }
            Sampler::TopK { k, temp } => {
                let indices = top_k_indices(logits, *k);
                let reduced: Vec<f32> = indices
                    .iter()
                    .filter_map(|&i| logits.get(i).copied())
                    .collect();
                let sampler = Sampler::Temperature { temp: *temp };
                let local_idx = sampler.sample(&reduced, rng);
                debug_assert!(
                    local_idx < indices.len(),
                    "local_idx must be in bounds — sample() returns index within reduced slice of length indices.len()"
                );
                indices.get(local_idx).copied().unwrap_or(0)
            }
            Sampler::TopP { p, temp } => {
                let probs_full = softmax(logits);
                let indices = top_p_indices(&probs_full, *p);
                let reduced: Vec<f32> = indices
                    .iter()
                    .filter_map(|&i| logits.get(i).copied())
                    .collect();
                let sampler = Sampler::Temperature { temp: *temp };
                let local_idx = sampler.sample(&reduced, rng);
                debug_assert!(
                    local_idx < indices.len(),
                    "local_idx must be in bounds — sample() returns index within reduced slice of length indices.len()"
                );
                indices.get(local_idx).copied().unwrap_or(0)
            }
        }
    }
}

/// Returns indices of the top-k logits (by value), unordered.
// SAFETY: k <= indexed.len() because of the .min(logits.len()) clamp below.
#[allow(clippy::indexing_slicing)]
fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(logits.len()).max(1);
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    // partial sort: move top-k to the front
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    indexed[..k].iter().map(|(i, _)| *i).collect()
}

/// Returns indices whose cumulative probability (sorted desc) covers at least `p`.
fn top_p_indices(probs: &[f32], p: f32) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let mut cumsum = 0.0_f32;
    let mut result = Vec::new();
    for (i, prob) in &indexed {
        result.push(*i);
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }
    result
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
    #[allow(clippy::indexing_slicing)]
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

    #[test]
    #[allow(clippy::indexing_slicing)]
    fn sample_multinomial_cumulative_sum() {
        // probs = [0.1, 0.6, 0.3] — index 1 has highest mass
        // With seed 42, r will hit index 1
        let mut r = SmallRng::seed_from_u64(42);
        let tok = sample_multinomial(&[0.1, 0.6, 0.3], &mut r);
        assert!(tok < 3, "index must be in range");
        // Verify the distribution roughly: run many samples and check index 1 wins most
        let mut r2 = SmallRng::seed_from_u64(99);
        let counts = (0..1000).fold([0usize; 3], |mut acc, _| {
            acc[sample_multinomial(&[0.1, 0.6, 0.3], &mut r2)] += 1;
            acc
        });
        assert!(
            counts[1] > counts[0] && counts[1] > counts[2],
            "index 1 (60%) should win most: {counts:?}"
        );
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![1.0_f32, 5.0, 2.0];
        let mut rng = SmallRng::seed_from_u64(42);
        let sampler = Sampler::Temperature { temp: 0.0 };
        assert_eq!(sampler.sample(&logits, &mut rng), 1);
    }

    #[test]
    fn temperature_skewed_picks_dominant() {
        // With very low temp, dominant logit (index 2 = 100.0) should win almost always
        let logits = vec![0.0_f32, 0.0, 100.0];
        let mut rng = SmallRng::seed_from_u64(0);
        let sampler = Sampler::Temperature { temp: 0.1 };
        let result = sampler.sample(&logits, &mut rng);
        assert_eq!(result, 2, "dominant logit should win at low temperature");
    }

    #[test]
    #[allow(clippy::indexing_slicing)]
    fn temperature_uniform_shows_variety() {
        // Equal logits + high temperature → all 3 indices appear in 300 samples
        let logits = vec![1.0_f32, 1.0, 1.0];
        let mut rng = SmallRng::seed_from_u64(0);
        let sampler = Sampler::Temperature { temp: 1.0 };
        let mut seen = [false; 3];
        for _ in 0..300 {
            seen[sampler.sample(&logits, &mut rng)] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "all indices should appear with uniform logits at temp=1.0"
        );
    }

    #[test]
    fn top_k_restricts_to_k_tokens() {
        // logits: index 0 = 10.0, index 1 = 9.0, index 2 = -100.0
        // top-k=2 → only indices 0 and 1 are eligible
        let logits = vec![10.0_f32, 9.0, -100.0];
        let mut rng = SmallRng::seed_from_u64(0);
        let sampler = Sampler::TopK { k: 2, temp: 1.0 };
        for _ in 0..50 {
            let result = sampler.sample(&logits, &mut rng);
            assert!(result == 0 || result == 1, "top-k=2 must not pick index 2");
        }
    }

    #[test]
    fn top_k_k1_is_greedy() {
        let logits = vec![1.0_f32, 5.0, 2.0];
        let mut rng = SmallRng::seed_from_u64(0);
        let sampler = Sampler::TopK { k: 1, temp: 1.0 };
        assert_eq!(sampler.sample(&logits, &mut rng), 1);
    }

    #[test]
    fn top_p_excludes_low_prob_tokens() {
        // logits: index 0 = 10.0 (dominant), index 1 = -100.0, index 2 = -100.0
        // After softmax, index 0 has ~1.0 prob → top-p=0.95 keeps only index 0
        let logits = vec![10.0_f32, -100.0, -100.0];
        let mut rng = SmallRng::seed_from_u64(0);
        let sampler = Sampler::TopP { p: 0.95, temp: 1.0 };
        for _ in 0..20 {
            assert_eq!(
                sampler.sample(&logits, &mut rng),
                0,
                "dominant token must always win"
            );
        }
    }

    #[test]
    #[allow(clippy::indexing_slicing)]
    fn top_p_uniform_allows_all() {
        // Equal logits + p=1.0 → all indices are eligible
        let logits = vec![1.0_f32, 1.0, 1.0];
        let mut rng = SmallRng::seed_from_u64(0);
        let sampler = Sampler::TopP { p: 1.0, temp: 1.0 };
        let mut seen = [false; 3];
        for _ in 0..300 {
            seen[sampler.sample(&logits, &mut rng)] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "all indices should be seen with p=1.0"
        );
    }
}
