//! Garantia: `GgufFile::parse` nunca panica, qualquer que seja a entrada.
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn parse_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        let _ = gguf::GgufFile::parse(&bytes);
    }

    #[test]
    fn parse_never_panics_with_gguf_prefix(tail in proptest::collection::vec(any::<u8>(), 0..512)) {
        // Começa com magic + version válidos para entrar fundo no parser.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&tail);
        let _ = gguf::GgufFile::parse(&bytes);
    }
}
