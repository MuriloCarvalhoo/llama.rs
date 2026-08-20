//! Mede a **taxa de aceitação** da cabeça MTP: com que frequência o token que ela propõe
//! é o mesmo que o modelo de fato produz no passo seguinte.
//!
//! Esse número é o teto do ganho de speculative decoding — `tok/s = base × (1 + aceitação)`
//! — e é o que decide se vale construir o decode em batch e o rollback do estado
//! recorrente (ver `docs/mtp-implementacao.md`). A cabeça roda na CPU porque a aceitação
//! não depende da velocidade: a proposta é a mesma que a GPU daria.
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

#[test]
fn mtp_propoe_o_token_que_o_modelo_produz() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
        return;
    };
    let Ok(ctx) = llama_vulkan::VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("menos de 2 GPUs — pulando");
        return;
    }
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let mut cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    cfg.ctx = 128;
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let aux = llama_model::GpuAuxWeights::from_gguf(&f, &bytes, &cfg).unwrap();

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
