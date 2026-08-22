//! O laço de geração: prompt renderizado, prefill com reuso, amostragem, parada.

use llama_chat::{Opcoes, render};
use llama_model::{GpuResidentDecode, ModelError, Sessao};
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::api::{ChamadaPronta, Parada, Pedido};
use crate::detok::Detok;
use crate::saida::{Evento, Saida};

#[derive(Debug, thiserror::Error)]
pub enum MotorError {
    #[error("prompt inválido: {0}")]
    Chat(String),
    #[error("prompt de {0} tokens não cabe no contexto de {1}")]
    ContextoEstourado(usize, usize),
    #[error("backend: {0}")]
    Backend(String),
}

/// O que a geração produziu, já separado nos campos da resposta.
#[derive(Debug, Default, Clone)]
pub struct Resultado {
    pub conteudo: String,
    pub reasoning: String,
    pub chamadas: Vec<ChamadaPronta>,
    pub parada: Option<Parada>,
    pub tokens_prompt: usize,
    pub tokens_saida: usize,
    /// Tokens do prompt que **não** vieram do cache — o que o prefill de fato processou.
    pub tokens_prefill: usize,
    /// Tempos, para separar o que é prefill, decode e amostragem. Misturados numa taxa
    /// só, um prompt longo faz o decode parecer lento.
    pub ms_prefill: f64,
    pub ms_decode: f64,
    pub ms_amostragem: f64,
    /// Tempo até o primeiro byte de stream, contado da chegada do pedido — é o que o
    /// usuário sente antes de a resposta começar a aparecer, e o que o prefill decide.
    /// Inclui render do template, tokenização, prefill e os tokens que o detokenizador
    /// segurou até fechar o primeiro caractere. `None` quando nada foi emitido.
    pub ms_ttft: Option<f64>,
}

pub struct Motor<'a> {
    tokenizer: &'a Tokenizer,
    gpu: &'a dyn GpuResidentDecode,
    sessao: Sessao,
    ctx: usize,
    /// Tokens que encerram o turno: `<|im_end|>` e `<|endoftext|>`.
    fim: Vec<u32>,
}

impl<'a> Motor<'a> {
    pub fn novo(
        tokenizer: &'a Tokenizer,
        gpu: &'a dyn GpuResidentDecode,
        ctx: usize,
        eos: u32,
    ) -> Motor<'a> {
        let mut fim = vec![eos];
        for marcador in ["<|im_end|>", "<|endoftext|>"] {
            if let [id] = tokenizer.encode_special(marcador).as_slice()
                && !fim.contains(id)
            {
                fim.push(*id);
            }
        }
        Motor {
            tokenizer,
            gpu,
            sessao: Sessao::nova(gpu),
            ctx,
            fim,
        }
    }

    /// Gera a resposta do pedido, chamando `emitir` a cada evento (para o streaming).
    ///
    /// `emitir` devolve `false` para interromper a geração — é como o servidor desiste
    /// quando o cliente fecha a conexão, em vez de ocupar a GPU com tokens que ninguém
    /// vai ler.
    pub fn responder(
        &mut self,
        pedido: &Pedido,
        mut emitir: impl FnMut(&Evento) -> bool,
    ) -> Result<Resultado, MotorError> {
        // O relógio do TTFT começa aqui: o cliente espera o render e a tokenização junto
        // com o prefill.
        let t_req = std::time::Instant::now();
        let prompt = render(
            &pedido.mensagens,
            &Opcoes {
                ferramentas: pedido.ferramentas.clone(),
                add_generation_prompt: true,
                enable_thinking: pedido.pensar,
                esforco: pedido.esforco,
            },
        )
        .map_err(|e| MotorError::Chat(e.to_string()))?;

        let ids = self.tokenizer.encode_special(&prompt);
        if ids.len() >= self.ctx {
            return Err(MotorError::ContextoEstourado(ids.len(), self.ctx));
        }
        let ja_no_cache = prefixo_comum(self.sessao.tokens(), &ids, self.sessao.marca());
        let t0 = std::time::Instant::now();
        let mut logits = self
            .sessao
            .prefill(self.gpu, &ids)
            .map_err(erro_backend)?
            .to_vec();

        let mut res = Resultado {
            tokens_prompt: ids.len(),
            tokens_prefill: ids.len() - ja_no_cache,
            ms_prefill: t0.elapsed().as_secs_f64() * 1e3,
            ..Resultado::default()
        };
        let sampler = sampler_de(pedido);
        // Sem seed no pedido, o relógio serve: o que importa é não repetir a amostra
        // entre requisições, não ter aleatoriedade criptográfica.
        let semente = pedido.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| {
                    d.as_secs().wrapping_mul(1_000_000_000) + u64::from(d.subsec_nanos())
                })
        });
        let mut rng = SmallRng::seed_from_u64(semente);
        let teto = pedido
            .max_tokens
            .unwrap_or(usize::MAX)
            .min(self.ctx - ids.len());

        let mut detok = Detok::novo();
        let mut saida = Saida::nova(pedido.pensar, pedido.ferramentas.clone());
        let mut querem_mais = true;
        let t_decode = std::time::Instant::now();
        while res.tokens_saida < teto && querem_mais {
            let t_amostra = std::time::Instant::now();
            let escolhido = sampler.sample(&logits, &mut rng);
            res.ms_amostragem += t_amostra.elapsed().as_secs_f64() * 1e3;
            let token = u32::try_from(escolhido)
                .map_err(|_| MotorError::Backend("token fora da faixa de u32".to_owned()))?;
            if self.fim.contains(&token) {
                res.parada = Some(Parada::Fim);
                break;
            }
            res.tokens_saida += 1;

            let texto = detok.empurrar(&self.tokenizer.decode_bytes(&[token]));
            for evento in saida.empurrar(&texto) {
                if res.ms_ttft.is_none() {
                    res.ms_ttft = Some(t_req.elapsed().as_secs_f64() * 1e3);
                }
                querem_mais &= registrar(&mut res, &evento, &mut emitir);
            }
            if parou_em_sequencia(&res.conteudo, &pedido.stop) {
                res.parada = Some(Parada::Fim);
                break;
            }
            logits = self
                .sessao
                .decode(self.gpu, token)
                .map_err(erro_backend)?
                .to_vec();
        }

        res.ms_decode = t_decode.elapsed().as_secs_f64() * 1e3;

        let resto = detok.finalizar();
        if !resto.is_empty() {
            for evento in saida.empurrar(&resto) {
                registrar(&mut res, &evento, &mut emitir);
            }
        }
        for evento in saida.finalizar() {
            registrar(&mut res, &evento, &mut emitir);
        }
        if !querem_mais {
            res.parada = Some(Parada::Fim);
        }

        res.parada = Some(match (res.parada, res.chamadas.is_empty()) {
            (_, false) => Parada::Ferramenta,
            (Some(p), true) => p,
            (None, true) => Parada::Limite,
        });
        Ok(res)
    }

    /// Tokens que estão no KV-cache — o que a próxima requisição pode reaproveitar.
    pub fn tokens_no_cache(&self) -> usize {
        self.sessao.tokens().len()
    }
}

/// Quantos tokens iniciais de `ids` já estão no cache — o que o prefill vai poupar.
///
/// Espelha a decisão de `llama_model::planejar_reuso`: divergência no meio descarta o
/// cache, **exceto** o que estiver antes do snapshot de fronteira de turno.
fn prefixo_comum(cache: &[u32], ids: &[u32], marca: Option<usize>) -> usize {
    let comum = cache.iter().zip(ids).take_while(|(a, b)| a == b).count();
    if comum == cache.len() {
        return comum;
    }
    marca.filter(|&m| m <= comum).unwrap_or(0)
}

fn erro_backend(e: ModelError) -> MotorError {
    MotorError::Backend(e.to_string())
}

/// Acumula o evento no resultado e repassa. Devolve `false` quando o consumidor
/// desistiu do stream.
fn registrar(
    res: &mut Resultado,
    evento: &Evento,
    emitir: &mut impl FnMut(&Evento) -> bool,
) -> bool {
    match evento {
        Evento::Reasoning(t) => res.reasoning.push_str(t),
        Evento::Conteudo(t) => res.conteudo.push_str(t),
        Evento::Chamada { nome, argumentos } => res.chamadas.push(ChamadaPronta {
            id: format!("call_{}", res.chamadas.len()),
            nome: nome.clone(),
            argumentos: serde_json::to_string(argumentos).unwrap_or_else(|_| "{}".to_owned()),
        }),
    }
    emitir(evento)
}

fn parou_em_sequencia(conteudo: &str, stop: &[String]) -> bool {
    stop.iter()
        .any(|s| !s.is_empty() && conteudo.contains(s.as_str()))
}

fn sampler_de(pedido: &Pedido) -> Sampler {
    if pedido.temperatura <= 0.0 {
        return Sampler::Greedy;
    }
    Sampler::TopKP {
        // `top_k: 0` na API significa "sem corte".
        k: if pedido.top_k == 0 {
            usize::MAX
        } else {
            pedido.top_k
        },
        p: pedido.top_p,
        temp: pedido.temperatura,
    }
}
