//! Mede a **taxa de aceitação** da cabeça MTP: com que frequência o token que ela propõe
//! é o mesmo que o modelo de fato produz no passo seguinte.
//!
//! Esse número é o teto do ganho de speculative decoding — `tok/s = base × (1 + aceitação)`
//! — e é o que decide se vale construir o decode em batch e o rollback do estado
//! recorrente (ver `docs/mtp-implementacao.md`). A cabeça roda na CPU porque a aceitação
//! não depende da velocidade: a proposta é a mesma que a GPU daria.
//!
//! O segundo teste mede a aceitação **encadeada** (n=2): realimenta a cabeça com a própria
//! previsão e conta acertos do 2º token condicionados ao 1º ter acertado. É um experimento,
//! não produção — o runtime segue com propostas de um token, e este número é o que decide
//! se vale mudar isso.
//!
//! Rode com `--nocapture` para ver a sequência de acertos.

use std::path::Path;

const MODELO: &str = "../../models/Qwen3.8-27B-Q4_K_M.gguf";
/// Tokens gerados. Cada proposta custa ~1,3 GB de dequant na CPU, então mantém baixo.
const N_TOKENS: usize = 24;

#[allow(unsafe_code)]
fn mapear() -> Option<memmap2::Mmap> {
    let file = std::fs::File::open(Path::new(MODELO)).ok()?;
    unsafe { memmap2::Mmap::map(&file) }.ok()
}

fn argmax(v: &[f32]) -> u32 {
    let mut melhor = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[melhor] {
            melhor = i;
        }
    }
    u32::try_from(melhor).unwrap_or(0)
}

/// Tudo o que os dois testes precisam do modelo real, ou `None` para pular.
struct Cenario {
    bytes: memmap2::Mmap,
}

impl Cenario {
    fn abrir() -> Option<Cenario> {
        Some(Cenario { bytes: mapear()? })
    }
}

#[test]
fn mtp_propoe_o_token_que_o_modelo_produz() {
    let Some(cen) = Cenario::abrir() else {
        eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
        return;
    };
    let bytes = &cen.bytes;
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("menos de 2 GPUs — pulando");
        return;
    }
    let f = gguf::GgufFile::parse(bytes).unwrap();
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    cfg.ctx = 128;
    let raw = llama_model::GpuRawWeights::from_gguf(&f, bytes, &cfg).unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, bytes, &cfg).unwrap();

    let (Some(mtp_raw), Some(mtp_aux)) = (raw.mtp.as_ref(), aux.mtp.as_ref()) else {
        panic!("o modelo devia trazer bloco MTP");
    };

    let backend = llama_vulkan::LayerSplitForward::new(&ctx, &cfg, &raw, &aux).unwrap();
    use llama_model::GpuResidentDecode;
    backend.reset();

    let mut cabeca = llama_model::MtpHead::new(&cfg);
    // "O" — só precisa de um token de partida qualquer para a sequência andar.
    let mut token: u32 = 46;
    let mut proposta: Option<u32> = None;
    let (mut acertos, mut tentativas) = (0usize, 0usize);

    for pos in 0..N_TOKENS {
        let logits = backend.decode(token, pos).unwrap();
        let real = argmax(&logits);

        if let Some(p) = proposta {
            tentativas += 1;
            let ok = p == real;
            acertos += usize::from(ok);
            eprintln!(
                "pos {pos:3}: propôs {p:6}, veio {real:6}  {}",
                if ok { "ACERTOU" } else { "errou" }
            );
        }

        // Hidden do último shard: é o que a cabeça MTP combina com o embedding.
        let Some(h) = backend.dbg_hidden(1) else {
            panic!("dbg_hidden do último shard devia existir");
        };
        proposta = Some(
            cabeca
                .propor(
                    &cfg,
                    mtp_raw,
                    mtp_aux,
                    &raw.output,
                    &aux.token_embd,
                    &aux.freq_table,
                    &h,
                    real,
                )
                .unwrap(),
        );
        token = real;
    }

    #[allow(clippy::cast_precision_loss)]
    let taxa = acertos as f64 / tentativas.max(1) as f64;
    eprintln!(
        "\n=== aceitação: {acertos}/{tentativas} = {:.1}%",
        taxa * 100.0
    );
    eprintln!(
        "    tok/s projetado = base × {:.2}  (base medida hoje: 22,3 → {:.1})",
        1.0 + taxa,
        22.3 * (1.0 + taxa)
    );
    assert!(tentativas > 0, "nenhuma proposta foi verificada");
}

/// Aceitação **encadeada**: a cabeça realimentada com a própria previsão acerta o 2º token?
///
/// A cabeça do Qwen3.8 tem `nextn_predict_layers = 1`, ou seja um bloco só. Para propor
/// dois tokens de uma vez ela precisa rodar duas vezes, a segunda com o token que ela
/// mesma acabou de propor e com o **seu próprio** hidden — é o que o llama.cpp faz ao
/// realimentar o `t_h_nextn` em `common/speculative.cpp`.
///
/// O que decide se vale implementar n=2 em produção:
///
/// - `a₂ ≥ 40 %` → `tokens/passo = 1 + a₁ + a₁·a₂ ≥ 1,85`, e um `plan_verify` de três
///   tokens com dois pontos de snapshot vira a forma mais barata de chegar aos 50 tok/s;
/// - `a₂ < 40 %` → fica em n=1, e os 50 saem da frente do decode base.
///
/// O teste **não** falha por `a₂` baixo: ele existe para produzir o número, e a decisão
/// mora em `docs/planos/2026-08-21-mtp-fases-c-e.md`.
#[test]
fn aceitacao_encadeada_do_segundo_token() {
    let Some(cen) = Cenario::abrir() else {
        eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
        return;
    };
    let bytes = &cen.bytes;
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("menos de 2 GPUs — pulando");
        return;
    }
    let f = gguf::GgufFile::parse(bytes).unwrap();
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    cfg.ctx = 128;
    let raw = llama_model::GpuRawWeights::from_gguf(&f, bytes, &cfg).unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, bytes, &cfg).unwrap();
    let (Some(mtp_raw), Some(mtp_aux)) = (raw.mtp.as_ref(), aux.mtp.as_ref()) else {
        panic!("o modelo devia trazer bloco MTP");
    };

    let backend = llama_vulkan::LayerSplitForward::new(&ctx, &cfg, &raw, &aux).unwrap();
    use llama_model::GpuResidentDecode;
    backend.reset();

    let mut cabeca = llama_model::MtpHead::new(&cfg);
    let mut token: u32 = 46;
    // A proposta feita na posição `pos - 1`: (1º token, 2º token encadeado).
    let mut proposta: Option<(u32, u32)> = None;
    // O 1º token da proposta feita duas posições atrás, guardado para saber se o 2º pode
    // ser cobrado: só interessa a aceitação do 2º **condicionada** ao 1º ter acertado.
    let mut pendente: Option<u32> = None;
    let (mut acertos1, mut tentativas1) = (0usize, 0usize);
    let (mut acertos2, mut tentativas2) = (0usize, 0usize);
    let mut reais: Vec<u32> = Vec::with_capacity(N_TOKENS);

    for pos in 0..N_TOKENS {
        let logits = backend.decode(token, pos).unwrap();
        let real = argmax(&logits);
        reais.push(real);

        // O 2º token da proposta de duas posições atrás vence agora — mas só é cobrado se
        // o 1º daquela proposta tiver acertado.
        if let Some(p2) = pendente.take() {
            tentativas2 += 1;
            let ok = p2 == real;
            acertos2 += usize::from(ok);
            eprintln!(
                "pos {pos:3}: 2º token propôs {p2:6}, veio {real:6}  {}",
                if ok { "ACERTOU" } else { "errou" }
            );
        }

        if let Some((p1, p2)) = proposta.take() {
            tentativas1 += 1;
            let ok1 = p1 == real;
            acertos1 += usize::from(ok1);
            eprintln!(
                "pos {pos:3}: 1º token propôs {p1:6}, veio {real:6}  {}",
                if ok1 { "ACERTOU" } else { "errou" }
            );
            // Encadear só faz sentido quando o 1º acertou: com o 1º errado o passo é
            // rejeitado e o 2º token nem chega a ser verificado.
            if ok1 {
                pendente = Some(p2);
            }
        }

        let Some(h) = backend.dbg_hidden(1) else {
            panic!("dbg_hidden do último shard devia existir");
        };
        let propor = |cabeca: &mut llama_model::MtpHead, h: &[f32], t: u32| {
            cabeca
                .propor_com_hidden(
                    &cfg,
                    mtp_raw,
                    mtp_aux,
                    &raw.output,
                    &aux.token_embd,
                    &aux.freq_table,
                    h,
                    t,
                )
                .unwrap()
        };
        let (p1, h_mtp) = propor(&mut cabeca, &h, real);
        // Segunda passada: o token é a própria previsão, e o hidden é o do bloco MTP.
        let (p2, _) = propor(&mut cabeca, &h_mtp, p1);
        proposta = Some((p1, p2));
        token = real;
    }

    #[allow(clippy::cast_precision_loss)]
    let a1 = acertos1 as f64 / tentativas1.max(1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let a2 = acertos2 as f64 / tentativas2.max(1) as f64;
    eprintln!(
        "\n=== aceitação do 1º token: {acertos1}/{tentativas1} = {:.1}%",
        a1 * 100.0
    );
    eprintln!(
        "=== aceitação do 2º token (condicionada ao 1º): {acertos2}/{tentativas2} = {:.1}%",
        a2 * 100.0
    );
    eprintln!(
        "    n=1: {:.2} tokens/passo   |   n=2: {:.2} tokens/passo",
        1.0 + a1,
        1.0 + a1 + a1 * a2
    );
    eprintln!(
        "    critério do plano: a₂ >= 40% justifica plan_verify com n_tok=3 — medido {:.1}%",
        a2 * 100.0
    );
    assert!(tentativas1 > 0, "nenhuma proposta foi verificada");
}
