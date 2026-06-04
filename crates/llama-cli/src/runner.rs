//! Logica de geracao reutilizavel.

use gguf::GgufFile;
use llama_model::Model;
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::args::Args;

/// Carrega o modelo e gera texto conforme `args`. Retorna o texto gerado (sem o prompt).
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
