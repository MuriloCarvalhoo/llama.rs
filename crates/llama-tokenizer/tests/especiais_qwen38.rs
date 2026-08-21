//! Os marcadores do chat template têm de sair como **um** token cada no vocab real.
//!
//! O BPE byte-level não conhece marcador nenhum: `<|im_start|>` passaria pelo
//! pré-tokenizador e viraria `<`, `|`, `im`, `_start`, `|`, `>` — texto comum. O modelo
//! nunca veria o formato em que foi treinado, e a degradação é silenciosa.

use std::path::Path;

const MODELO: &str = "../../models/Qwen3.8-27B-Q4_K_M.gguf";

// ids lidos dos metadados do GGUF (`tokenizer.ggml.tokens`).
const IM_START: u32 = 248_045;
const IM_END: u32 = 248_046;
const TOOL_CALL: u32 = 248_058;
const THINK: u32 = 248_068;

#[allow(unsafe_code)]
fn mapear() -> Option<memmap2::Mmap> {
    let file = std::fs::File::open(Path::new(MODELO)).ok()?;
    unsafe { memmap2::Mmap::map(&file) }.ok()
}

#[test]
fn encode_special_resolve_os_marcadores_do_chat_template() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let tok = llama_tokenizer::Tokenizer::from_gguf(&f).unwrap();

    let ids = tok.encode_special("<|im_start|>user\nOi<|im_end|>\n<think>\n<tool_call>");

    assert_eq!(ids.first(), Some(&IM_START), "o texto começa no marcador");
    for esperado in [IM_END, THINK, TOOL_CALL] {
        assert!(ids.contains(&esperado), "faltou o marcador {esperado}");
    }
}

/// O contraste que justifica o método: `encode` trata o marcador como texto.
#[test]
fn encode_comum_nao_resolve_o_marcador() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let tok = llama_tokenizer::Tokenizer::from_gguf(&f).unwrap();

    let ids = tok.encode("<|im_start|>", false);

    assert!(
        ids.len() > 1,
        "sem parse de especiais o marcador se fragmenta"
    );
    assert!(!ids.contains(&IM_START));
}

/// Texto sem marcador nenhum tem de sair igual pelos dois caminhos.
#[test]
fn encode_special_nao_muda_texto_comum() {
    let Some(bytes) = mapear() else {
        eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let tok = llama_tokenizer::Tokenizer::from_gguf(&f).unwrap();

    let texto = "A raposa marrom pula sobre o cão preguiçoso.";

    assert_eq!(tok.encode_special(texto), tok.encode(texto, false));
}
