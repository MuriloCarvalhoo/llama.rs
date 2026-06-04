//! Geração greedy (argmax, temp 0) e com Sampler arbitrário, com KV cache.

use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::Rng;

use crate::error::ModelError;
use crate::model::Model;

impl Model {
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
}
