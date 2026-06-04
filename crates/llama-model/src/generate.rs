//! Geração greedy (argmax, temp 0) e com Sampler arbitrário, com KV cache.

use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::Rng;

use crate::error::ModelError;
use crate::model::Model;

impl Model {
    /// Gera até `n_tokens` chamando `on_token` a cada token decodificado individualmente.
    /// Para em EOS ou quando `n_tokens` for atingido.
    pub fn generate_streaming(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        n_tokens: usize,
        sampler: &Sampler,
        rng: &mut impl Rng,
        on_token: &mut impl FnMut(&str),
    ) -> Result<(), ModelError> {
        let prompt_ids = tokenizer.encode(prompt, true);
        let mut cache = self.new_cache();

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
            let logits = self.forward(&[next], &mut cache)?;
            let idx = sampler.sample(&logits, rng);
            next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
        }

        Ok(())
    }

    /// Gera até `n_tokens` usando a estratégia `sampler`.
    /// Retorna o texto gerado (sem o prompt), parando em EOS.
    pub fn generate(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        n_tokens: usize,
        sampler: &Sampler,
        rng: &mut impl Rng,
    ) -> Result<String, ModelError> {
        let prompt_ids = tokenizer.encode(prompt, true);
        let mut cache = self.new_cache();

        let logits = self.forward(&prompt_ids, &mut cache)?;
        let first_idx = sampler.sample(&logits, rng);
        let mut next = u32::try_from(first_idx).map_err(|_| ModelError::Overflow)?;

        let mut generated = Vec::with_capacity(n_tokens);
        while generated.len() < n_tokens {
            if next == self.config.eos_id {
                break;
            }
            generated.push(next);
            let logits = self.forward(&[next], &mut cache)?;
            let idx = sampler.sample(&logits, rng);
            next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
        }

        Ok(tokenizer.decode(&generated))
    }

    /// Gera `n_tokens` para cada prompt em `prompts`, em batch.
    /// Cada sequência tem seu próprio cache; para quando atinge EOS ou `n_tokens`.
    /// Retorna um vetor de strings geradas (uma por prompt, sem o texto do prompt).
    pub fn generate_batch(
        &self,
        tokenizer: &Tokenizer,
        prompts: &[&str],
        n_tokens: usize,
        sampler: &Sampler,
        rng: &mut impl Rng,
    ) -> Result<Vec<String>, ModelError> {
        let n = prompts.len();
        let mut caches: Vec<_> = (0..n).map(|_| self.new_cache()).collect();

        let encoded: Vec<Vec<u32>> = prompts.iter().map(|p| tokenizer.encode(p, true)).collect();

        let batch_refs: Vec<&[u32]> = encoded.iter().map(|v| v.as_slice()).collect();
        let all_logits = self.forward_batch(&batch_refs, &mut caches)?;

        let mut current: Vec<u32> = all_logits
            .iter()
            .map(|logits| {
                let idx = sampler.sample(logits, rng);
                u32::try_from(idx).map_err(|_| ModelError::Overflow)
            })
            .collect::<Result<_, _>>()?;

        let mut generated: Vec<Vec<u32>> = vec![Vec::with_capacity(n_tokens); n];
        let mut done = vec![false; n];

        for _ in 0..n_tokens {
            if done.iter().all(|&d| d) {
                break;
            }
            for i in 0..n {
                if done[i] || current[i] == self.config.eos_id {
                    done[i] = true;
                    continue;
                }
                generated[i].push(current[i]);
                let logits = self.forward(&[current[i]], &mut caches[i])?;
                let idx = sampler.sample(&logits, rng);
                current[i] = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
            }
        }

        Ok(generated
            .into_iter()
            .map(|toks| tokenizer.decode(&toks))
            .collect())
    }

    /// Gera até `n_tokens` por argmax a partir de `prompt` (com BOS). Para em EOS.
    /// Retorna o texto decodificado dos tokens GERADOS (sem o prompt), espelhando
    /// `--no-display-prompt` do oráculo.
    pub fn generate_greedy(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        n_tokens: usize,
    ) -> Result<String, ModelError> {
        let prompt_ids = tokenizer.encode(prompt, true);
        let mut cache = self.new_cache();

        let mut next = self.forward_argmax(&prompt_ids, &mut cache)?;
        let mut generated = Vec::with_capacity(n_tokens);

        while generated.len() < n_tokens {
            if next == self.config.eos_id {
                break;
            }
            generated.push(next);
            next = self.forward_argmax(&[next], &mut cache)?;
        }

        Ok(tokenizer.decode(&generated))
    }
}

#[cfg(test)]
mod generate_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use llama_sampling::Sampler;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use std::path::Path;

    fn load() -> Option<(Model, llama_tokenizer::Tokenizer)> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        let f = gguf::GgufFile::parse(&bytes).ok()?;
        let model = Model::load(&f, &bytes).ok()?;
        let tok = llama_tokenizer::Tokenizer::from_gguf(&f).ok()?;
        Some((model, tok))
    }

    #[test]
    fn generate_streaming_calls_callback_for_each_token() {
        let Some((model, tok)) = load() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let mut pieces: Vec<String> = Vec::new();
        model
            .generate_streaming(
                &tok,
                "Once upon a time",
                8,
                &Sampler::Greedy,
                &mut rng,
                &mut |piece| pieces.push(piece.to_owned()),
            )
            .unwrap();
        assert!(
            !pieces.is_empty(),
            "callback deve ser chamado pelo menos uma vez"
        );
        assert!(pieces.len() <= 8, "no maximo 8 callbacks: {pieces:?}");
    }

    #[test]
    fn generate_streaming_zero_tokens_calls_no_callback() {
        let Some((model, tok)) = load() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let mut count = 0usize;
        model
            .generate_streaming(&tok, "Hello", 0, &Sampler::Greedy, &mut rng, &mut |_| {
                count += 1
            })
            .unwrap();
        assert_eq!(count, 0, "n_tokens=0 nao deve chamar o callback");
    }

    #[test]
    fn generate_with_greedy_sampler_matches_generate_greedy() {
        let Some((model, tok)) = load() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let out_gen = model
            .generate(&tok, "Once upon a time", 8, &Sampler::Greedy, &mut rng)
            .unwrap();
        let out_greedy = model.generate_greedy(&tok, "Once upon a time", 8).unwrap();
        assert_eq!(
            out_gen, out_greedy,
            "Sampler::Greedy deve produzir mesma saída que generate_greedy"
        );
    }

    #[test]
    fn generate_batch_matches_individual_generate() {
        let Some((model, tok)) = load() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let prompts = ["Once upon a time", "There was a little"];
        let n_tokens = 6;

        // Geração individual
        let mut rng_a = SmallRng::seed_from_u64(42);
        let out_a = model
            .generate(&tok, prompts[0], n_tokens, &Sampler::Greedy, &mut rng_a)
            .unwrap();
        let mut rng_b = SmallRng::seed_from_u64(42);
        let out_b = model
            .generate(&tok, prompts[1], n_tokens, &Sampler::Greedy, &mut rng_b)
            .unwrap();

        // Geração em batch com Greedy (rng não afeta o resultado)
        let mut rng_batch = SmallRng::seed_from_u64(42);
        let results = model
            .generate_batch(&tok, &prompts, n_tokens, &Sampler::Greedy, &mut rng_batch)
            .unwrap();

        assert_eq!(results[0], out_a, "batch[0] deve bater com individual");
        assert_eq!(results[1], out_b, "batch[1] deve bater com individual");
    }

    #[test]
    fn generate_batch_empty_prompts_returns_empty() {
        let Some((model, tok)) = load() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let results = model
            .generate_batch(&tok, &[], 8, &Sampler::Greedy, &mut rng)
            .unwrap();
        assert!(results.is_empty());
    }
}
