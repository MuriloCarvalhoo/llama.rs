//! Geração greedy (argmax, temp 0) com KV cache.

use llama_tokenizer::Tokenizer;

use crate::error::ModelError;
use crate::model::Model;

impl Model {
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
        generated.push(next);

        while generated.len() < n_tokens {
            if next == self.config.eos_id {
                break;
            }
            next = self.forward_argmax(&[next], &mut cache)?;
            generated.push(next);
        }

        Ok(tokenizer.decode(&generated))
    }
}
