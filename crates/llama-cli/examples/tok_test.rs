fn main() {
    let bytes = std::fs::read("models/Qwen3.8-27B-Q5_K_M.gguf").unwrap();
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let t = llama_tokenizer::Tokenizer::from_gguf(&f).unwrap();
    for txt in ["Oceano", "O mar e azul."] {
        let ids = t.encode(txt, true);
        println!("{txt:?} -> {ids:?} -> {:?}", t.decode(&ids));
    }
}
