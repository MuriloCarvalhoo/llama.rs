//! O bloco de multi-token prediction (`blk.{n_layer}.*`) tem que ser lido com as formas
//! certas — e **como camada de atenção**, que é a parte contraintuitiva.
//!
//! No Qwen3.8-27B o bloco MTP é o 64, e `eh_linear(64)` diz `true` porque
//! `(64 + 1) % 4 != 0`. Se o carregador seguisse essa regra procuraria `ssm_out.weight` e
//! falharia: o GGUF traz `attn_q/k/v/output` no bloco MTP.

use std::path::Path;

const MODELO: &str = "../../models/Qwen3.8-27B-Q4_K_M.gguf";

#[allow(unsafe_code)]
fn mapear() -> Option<memmap2::Mmap> {
    let file = std::fs::File::open(Path::new(MODELO)).ok()?;
    unsafe { memmap2::Mmap::map(&file) }.ok()
}

#[test]
fn bloco_mtp_carrega_com_as_formas_do_gguf() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();

    // 65 blocos no arquivo, 1 deles é MTP -> 64 camadas de verdade.
    assert_eq!(cfg.n_layer_nextn, 1, "o modelo devia declarar 1 bloco MTP");
    assert_eq!(cfg.n_layer, 64);

    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let mtp = raw.mtp.as_ref().expect("bloco MTP devia ter sido lido");

    // eh_proj combina embedding e hidden: entra com 2*n_embd, sai com n_embd.
    let sb = |n_in: usize, n_out: usize, bytes_por_sb: usize| n_out * (n_in / 256) * bytes_por_sb;
    assert_eq!(mtp.eh_proj.ty, gguf::GgmlType::Q8_0);
    assert_eq!(
        mtp.eh_proj.bytes.len(),
        cfg.n_embd * (cfg.n_embd * 2 / 32) * 34,
        "eh_proj devia ser [2*n_embd -> n_embd] em Q8_0"
    );

    // O mixer é atenção, não delta net — apesar de `eh_linear` dizer o contrário.
    let llama_model::MixerRaw::Attn {
        attn_q,
        attn_k,
        attn_v,
        attn_output,
    } = &mtp.layer.mixer
    else {
        panic!("o bloco MTP tem de ser de atenção");
    };
    let kv_dim = cfg.n_head_kv * cfg.head_dim;
    assert_eq!(
        attn_q.bytes.len(),
        sb(cfg.n_embd, cfg.head_dim * cfg.n_head * 2, 144)
    );
    assert_eq!(attn_k.bytes.len(), sb(cfg.n_embd, kv_dim, 144));
    assert_eq!(attn_v.bytes.len(), sb(cfg.n_embd, kv_dim, 210));
    assert_eq!(
        attn_output.bytes.len(),
        sb(cfg.head_dim * cfg.n_head, cfg.n_embd, 144)
    );
    assert_eq!(
        mtp.layer.ffn_gate.bytes.len(),
        sb(cfg.n_embd, cfg.n_ff, 144)
    );
    assert_eq!(
        mtp.layer.ffn_down.bytes.len(),
        sb(cfg.n_ff, cfg.n_embd, 210)
    );

    // O custo do bloco é o que decide se propor um token compensa: ~1% de um passo.
    let total: usize = mtp.eh_proj.bytes.len()
        + attn_q.bytes.len()
        + attn_k.bytes.len()
        + attn_v.bytes.len()
        + attn_output.bytes.len()
        + mtp.layer.ffn_gate.bytes.len()
        + mtp.layer.ffn_up.bytes.len()
        + mtp.layer.ffn_down.bytes.len();
    #[allow(clippy::cast_precision_loss)]
    let mb = total as f64 / 1e6;
    eprintln!(
        "bloco MTP: {mb:.0} MB -> {:.2} ms a 717 GB/s",
        mb / 717_000.0 * 1000.0
    );
    assert!(
        (250.0..450.0).contains(&mb),
        "bloco MTP com tamanho inesperado: {mb} MB"
    );
}
