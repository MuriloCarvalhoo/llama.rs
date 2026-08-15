//! Carga do Qwen3.8-27B (`qwen35`): config híbrida e pesos das duas variantes de camada.
//!
//! Pula sem o modelo. Usa mmap — não lê os 18.5 GB, só os cabeçalhos e os offsets.

use std::path::Path;

const MODELO: &str = "../../models/Qwen3.8-27B-Q5_K_M.gguf";

// mmap é a única forma sã de abrir 18.5 GB num teste; `fs::read` traria o arquivo
// inteiro para a RAM.
#[allow(unsafe_code)]
fn mapear() -> Option<memmap2::Mmap> {
    let file = std::fs::File::open(Path::new(MODELO)).ok()?;
    // SAFETY: arquivo tratado como imutável durante o teste.
    unsafe { memmap2::Mmap::map(&file) }.ok()
}

#[test]
fn config_hibrida_do_qwen35() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8-27B ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();

    assert_eq!(cfg.n_embd, 5120);
    assert_eq!(cfg.n_ff, 17408);
    assert_eq!(cfg.n_head, 24);
    assert_eq!(cfg.n_head_kv, 4);
    // Vem de `attention.key_length`: n_embd / n_head daria 213.33, que nem é inteiro.
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.rope_dim, 64, "só 64 das 256 dimensões são rotacionadas");
    assert_eq!(cfg.n_layer_nextn, 1, "um bloco MTP no fim da pilha");

    let dn = cfg.delta_net.as_ref().expect("qwen35 tem delta net");
    assert_eq!(dn.d_conv, 4);
    assert_eq!(dn.d_state, 128);
    assert_eq!(dn.d_inner, 6144);
    assert_eq!(dn.n_k_heads, 16);
    assert_eq!(dn.n_v_heads, 48);
    assert_eq!(dn.full_attn_interval, 4);
    assert_eq!(dn.head_v_dim(), 128);
    // Estado recorrente por camada: 128*128*48 floats = 3.1 MB. Não cresce com o contexto.
    assert_eq!(dn.state_len(), 128 * 128 * 48);

    // Camadas 0,1,2 lineares; a 3 é de atenção completa; e assim por diante.
    assert!(dn.eh_linear(0) && dn.eh_linear(1) && dn.eh_linear(2));
    assert!(!dn.eh_linear(3));
    assert!(!dn.eh_linear(63));
    // `block_count` é 65 no GGUF; o bloco MTP não entra no forward.
    assert_eq!(cfg.n_layer, 64);
    let lineares = (0..cfg.n_layer).filter(|&l| dn.eh_linear(l)).count();
    assert_eq!(lineares, 48, "48 lineares de {} camadas", cfg.n_layer);
}

#[test]
fn pesos_das_duas_variantes_de_camada() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8-27B ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();

    let w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).expect("carga dos pesos");
    assert_eq!(w.layers.len(), 64);

    let dn = cfg.delta_net.as_ref().unwrap();
    let key_dim = dn.d_state * dn.n_k_heads; // 2048
    let value_dim = dn.head_v_dim() * dn.n_v_heads; // 6144

    // Camada 0: atenção linear.
    match &w.layers[0].mixer {
        llama_model::MixerRaw::Delta {
            attn_qkv,
            attn_gate,
            ssm_out,
        } => {
            let sb = |t: &llama_model::QTensor<'_>, n_in: usize, n_out: usize| {
                let (elems, bloco) = match t.ty {
                    gguf::GgmlType::Q5_K => (256, 176),
                    gguf::GgmlType::Q6_K => (256, 210),
                    gguf::GgmlType::Q8_0 => (32, 34),
                    other => panic!("tipo inesperado {other:?}"),
                };
                assert_eq!(t.bytes.len(), n_out * (n_in / elems) * bloco);
            };
            sb(attn_qkv, cfg.n_embd, key_dim * 2 + value_dim);
            sb(attn_gate, cfg.n_embd, value_dim);
            sb(ssm_out, value_dim, cfg.n_embd);
        }
        llama_model::MixerRaw::Attn { .. } => panic!("camada 0 devia ser linear"),
    }

    // Camada 3: atenção completa, com Q saindo dobrado (query|gate).
    match &w.layers[3].mixer {
        llama_model::MixerRaw::Attn { attn_q, .. } => {
            let n_blocos = cfg.n_embd / 256;
            let esperado_q = cfg.head_dim * cfg.n_head * 2 * n_blocos * 176; // Q5_K
            assert_eq!(
                attn_q.bytes.len(),
                esperado_q,
                "attn_q da camada 3 deve ter 2*head_dim*n_head linhas"
            );
        }
        llama_model::MixerRaw::Delta { .. } => panic!("camada 3 devia ser de atenção"),
    }
}
