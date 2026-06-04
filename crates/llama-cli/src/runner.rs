//! Logica de geracao reutilizavel — streaming e buffered.

use std::time::Instant;

use gguf::GgufFile;
use llama_model::Model;
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::args::Args;

/// Metricas coletadas durante a geracao.
pub struct Timing {
    pub n_tokens: usize,
    pub elapsed_secs: f64,
    pub tokens_per_sec: f64,
}

/// Carrega o modelo e chama `on_token` para cada token gerado. Retorna metricas de tempo.
pub fn run_generate(
    args: &Args,
    on_token: &mut impl FnMut(&str),
) -> Result<Timing, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&args.model)?;
    let f = GgufFile::parse(&bytes)?;
    let model = Model::load(&f, &bytes)?;
    let tokenizer = Tokenizer::from_gguf(&f)?;
    let sampler = choose_sampler(args);
    let mut rng = SmallRng::seed_from_u64(args.seed);

    let mut n_tokens = 0usize;
    let start = Instant::now();

    model.generate_streaming(
        &tokenizer,
        &args.prompt,
        args.n_predict,
        &sampler,
        &mut rng,
        &mut |piece| {
            on_token(piece);
            n_tokens += 1;
        },
    )?;

    let elapsed_secs = start.elapsed().as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    let tokens_per_sec = if elapsed_secs > 0.0 {
        n_tokens as f64 / elapsed_secs
    } else {
        0.0
    };

    Ok(Timing {
        n_tokens,
        elapsed_secs,
        tokens_per_sec,
    })
}

/// Carrega o modelo e retorna o texto completo como String (sem streaming).
/// Mantém compatibilidade com `greedy_gate.rs` e outros testes de integracao.
pub fn generate_text(args: &Args) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&args.model)?;
    let f = GgufFile::parse(&bytes)?;
    let model = Model::load(&f, &bytes)?;
    let tokenizer = Tokenizer::from_gguf(&f)?;
    let sampler = choose_sampler(args);
    let mut rng = SmallRng::seed_from_u64(args.seed);

    let text = model.generate(&tokenizer, &args.prompt, args.n_predict, &sampler, &mut rng)?;
    Ok(text)
}

#[allow(clippy::float_cmp)]
fn choose_sampler(args: &Args) -> Sampler {
    if args.temp == 0.0 {
        return Sampler::Greedy;
    }
    if args.top_k > 0 {
        return Sampler::TopK {
            k: args.top_k,
            temp: args.temp,
        };
    }
    if args.top_p < 1.0 {
        return Sampler::TopP {
            p: args.top_p,
            temp: args.temp,
        };
    }
    Sampler::Temperature { temp: args.temp }
}
