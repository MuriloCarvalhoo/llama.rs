//! Validação empírica das recomendações da documentação oficial do Rust
//! (Perf Book, docs da std, lints do Clippy). Ver docs/rust-praticas-da-documentacao.md.
//!
//! Cada execução roda UMA variante de UM experimento num processo limpo,
//! porque VmHWM (pico de RSS) em /proc/self/status é monotônico por processo.
//!
//! Uso: validacao-rust-praticas <experimento> <variante> [n]
//! Saída (stdout): resultado,<experimento>,<variante>,<n>,<ms>,<vmhwm_mb>
//!                 info,<experimento>,<chave>,<valor>

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::io::{Read as _, Write as _};
use std::time::Instant;

fn vm_kb(campo: &str) -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(campo))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

/// xorshift64* — determinístico, sem crate externo.
struct Rng(u64);
impl Rng {
    fn nova() -> Self {
        Rng(0x9E37_79B9_7F4A_7C15)
    }
    fn proximo(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn dir_temporario() -> std::path::PathBuf {
    std::env::var_os("VALIDACAO_TMP")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

// ---------------------------------------------------------------------------
// Experimentos. Cada função devolve (n, tempo_ms) da região medida.
// ---------------------------------------------------------------------------

/// Vec::with_capacity vs push sem capacidade (Perf Book: Heap Allocations;
/// Vec: "Capacity and reallocation").
fn vec_crescimento(variante: &str, n: usize) -> (usize, f64) {
    let t = Instant::now();
    let mut v: Vec<u64> = match variante {
        "push" => Vec::new(),
        "with_capacity" => Vec::with_capacity(n),
        _ => panic!("variante desconhecida"),
    };
    for i in 0..n as u64 {
        v.push(i);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("info,vec_crescimento,capacidade_final,{}", v.capacity());
    black_box(&v);
    (n, ms)
}

/// Concatenação de String: reconstrução total a cada passo (O(n²)) vs
/// push_str vs write! com capacidade prévia (String: docs de capacidade e Add).
fn string_concat(variante: &str, n: usize) -> (usize, f64) {
    let t = Instant::now();
    let s = match variante {
        // MÁ PRÁTICA: format! recopia a string inteira a cada iteração.
        "format_reconstroi" => {
            let mut s = String::new();
            for i in 0..n {
                s = format!("{s}{i},");
            }
            s
        }
        // push_str linear, mas com uma String temporária por item.
        "push_str" => {
            let mut s = String::new();
            for i in 0..n {
                s.push_str(&i.to_string());
                s.push(',');
            }
            s
        }
        // write! não cria temporária; capacidade pré-alocada.
        "write_capacidade" => {
            let mut s = String::with_capacity(n * 8);
            for i in 0..n {
                let _ = write!(s, "{i},");
            }
            s
        }
        _ => panic!("variante desconhecida"),
    };
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("info,string_concat,bytes_finais,{}", s.len());
    black_box(&s);
    (n, ms)
}

/// Produto escalar u32→u64: indexação com bounds check no segundo slice vs
/// zip de iteradores (Perf Book: Bounds Checks, Iterators).
fn produto_escalar(variante: &str, n: usize) -> (usize, f64) {
    let a: Vec<u32> = (0..n).map(|i| (i & 0xFFFF) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i >> 3) & 0xFFFF) as u32).collect();
    let (a, b) = (black_box(a), black_box(b));
    let t = Instant::now();
    let acc: u64 = match variante {
        "indexado" => {
            let mut acc: u64 = 0;
            for i in 0..a.len() {
                acc += u64::from(a[i]) * u64::from(b[i]);
            }
            acc
        }
        "zip" => a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| u64::from(x) * u64::from(y))
            .sum(),
        _ => panic!("variante desconhecida"),
    };
    let ms = t.elapsed().as_secs_f64() * 1e3;
    black_box(acc);
    (n, ms)
}

/// sort (estável, aloca até n/2) vs sort_unstable (in-place) — docs de slice.
fn ordenar(variante: &str, n: usize) -> (usize, f64) {
    let mut rng = Rng::nova();
    let mut v: Vec<u64> = (0..n).map(|_| rng.proximo()).collect();
    let t = Instant::now();
    match variante {
        "sort" => v.sort(),
        "sort_unstable" => v.sort_unstable(),
        _ => panic!("variante desconhecida"),
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    black_box(&v);
    (n, ms)
}

/// Inserção em mapa: HashMap sem/com with_capacity vs BTreeMap
/// (Perf Book: Heap Allocations; std::collections: tabela de custos).
fn mapa_insercao(variante: &str, n: usize) -> (usize, f64) {
    let mut rng = Rng::nova();
    let chaves: Vec<u64> = (0..n).map(|_| rng.proximo()).collect();
    let chaves = black_box(chaves);
    let t = Instant::now();
    match variante {
        "hashmap_novo" => {
            let mut m: HashMap<u64, u64> = HashMap::new();
            for &k in &chaves {
                m.insert(k, k);
            }
            black_box(m.len());
        }
        "hashmap_capacidade" => {
            let mut m: HashMap<u64, u64> = HashMap::with_capacity(n);
            for &k in &chaves {
                m.insert(k, k);
            }
            black_box(m.len());
        }
        "btreemap" => {
            let mut m: BTreeMap<u64, u64> = BTreeMap::new();
            for &k in &chaves {
                m.insert(k, k);
            }
            black_box(m.len());
        }
        _ => panic!("variante desconhecida"),
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    (n, ms)
}

/// clone desnecessário em laço quente vs empréstimo
/// (Perf Book: Heap Allocations, "unnecessary clones"; Clippy: redundant_clone).
fn clone_em_laco(variante: &str, n: usize) -> (usize, f64) {
    let v: Vec<String> = (0..n).map(|i| format!("cadeia_de_teste_{i:016}")).collect();
    let v = black_box(v);
    const PASSADAS: usize = 5;
    let t = Instant::now();
    let mut total = 0usize;
    match variante {
        "clone" => {
            for _ in 0..PASSADAS {
                for s in &v {
                    total += black_box(s.clone()).len();
                }
            }
        }
        "emprestimo" => {
            for _ in 0..PASSADAS {
                for s in &v {
                    total += black_box(s).len();
                }
            }
        }
        _ => panic!("variante desconhecida"),
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    black_box(total);
    (n * PASSADAS, ms)
}

/// Escritas de 1 byte: File direto (1 syscall/byte) vs BufWriter vs write_all
/// (Perf Book: I/O; docs de BufWriter).
fn io_escrita(variante: &str, n: usize) -> (usize, f64) {
    let caminho = dir_temporario().join(format!("validacao_io_{}.bin", std::process::id()));
    let t = Instant::now();
    match variante {
        "sem_buffer" => {
            let mut f = fs::File::create(&caminho).unwrap();
            for i in 0..n {
                f.write_all(&[(i & 0xFF) as u8]).unwrap();
            }
            f.sync_data().ok();
        }
        "buf_writer" => {
            let f = fs::File::create(&caminho).unwrap();
            let mut w = std::io::BufWriter::new(f);
            for i in 0..n {
                w.write_all(&[(i & 0xFF) as u8]).unwrap();
            }
            w.flush().unwrap();
            w.into_inner().unwrap().sync_data().ok();
        }
        "write_all" => {
            let dados: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
            let mut f = fs::File::create(&caminho).unwrap();
            f.write_all(&dados).unwrap();
            f.sync_data().ok();
        }
        _ => panic!("variante desconhecida"),
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    fs::remove_file(&caminho).ok();
    (n, ms)
}

/// Leituras de 1 byte: File direto vs BufReader (docs de BufReader).
fn io_leitura(variante: &str, n: usize) -> (usize, f64) {
    let caminho = dir_temporario().join(format!("validacao_io_r_{}.bin", std::process::id()));
    let dados: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
    fs::write(&caminho, &dados).unwrap();
    let mut soma = 0u64;
    let mut byte = [0u8; 1];
    let t = Instant::now();
    match variante {
        "sem_buffer" => {
            let mut f = fs::File::open(&caminho).unwrap();
            while f.read(&mut byte).unwrap() == 1 {
                soma += u64::from(byte[0]);
            }
        }
        "buf_reader" => {
            let f = fs::File::open(&caminho).unwrap();
            let mut r = std::io::BufReader::new(f);
            while r.read(&mut byte).unwrap() == 1 {
                soma += u64::from(byte[0]);
            }
        }
        _ => panic!("variante desconhecida"),
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    fs::remove_file(&caminho).ok();
    black_box(soma);
    (n, ms)
}

/// collect intermediário vs cadeia fundida de iteradores (Perf Book: Iterators).
fn collect_intermediario(variante: &str, n: usize) -> (usize, f64) {
    let v: Vec<u32> = (0..n).map(|i| (i & 0xFFFF) as u32).collect();
    let v = black_box(v);
    let t = Instant::now();
    let soma: u64 = match variante {
        "collect" => {
            let intermediario: Vec<u64> = v.iter().map(|&x| u64::from(x) * 3).collect();
            intermediario.iter().filter(|&&x| x % 2 == 0).sum()
        }
        "fundido" => v
            .iter()
            .map(|&x| u64::from(x) * 3)
            .filter(|x| x % 2 == 0)
            .sum(),
        _ => panic!("variante desconhecida"),
    };
    let ms = t.elapsed().as_secs_f64() * 1e3;
    black_box(soma);
    (n, ms)
}

/// vec![0; n] vs push(0) em laço vs resize (Perf Book: Standard Library Types;
/// Clippy: slow_vector_initialization). Inclui uma passada de leitura para
/// forçar o toque das páginas em todas as variantes.
fn vec_inicializacao(variante: &str, n: usize) -> (usize, f64) {
    let t = Instant::now();
    let v: Vec<u8> = match variante {
        "macro_vec" => vec![0u8; n],
        "push" => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(0u8);
            }
            v
        }
        "resize" => {
            let mut v = Vec::new();
            v.resize(n, 0u8);
            v
        }
        _ => panic!("variante desconhecida"),
    };
    let soma: u64 = v.iter().map(|&b| u64::from(b)).sum();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    black_box((soma, &v));
    (n, ms)
}

/// contains_key + insert/get_mut (duas buscas) vs entry API (uma busca)
/// (Clippy: map_entry).
fn mapa_entry(variante: &str, n: usize) -> (usize, f64) {
    let mut rng = Rng::nova();
    let distintas = (n / 2) as u64;
    let chaves: Vec<u64> = (0..n).map(|_| rng.proximo() % distintas).collect();
    let chaves = black_box(chaves);
    let mut m: HashMap<u64, u64> = HashMap::new();
    let t = Instant::now();
    match variante {
        "duas_buscas" => {
            for &k in &chaves {
                if let Some(v) = m.get_mut(&k) {
                    *v += 1;
                } else {
                    m.insert(k, 1);
                }
            }
        }
        "entry" => {
            for &k in &chaves {
                *m.entry(k).or_insert(0) += 1;
            }
        }
        _ => panic!("variante desconhecida"),
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    black_box(m.len());
    (n, ms)
}

/// shrink_to_fit sobre um Vec crescido por push: capacidade e RSS antes/depois
/// (Vec: "Guarantees" — Vec nunca encolhe sozinho).
fn encolher(_variante: &str, n: usize) -> (usize, f64) {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..n as u64 {
        v.push(i);
    }
    let cap_antes_mb = v.capacity() * 8 / (1024 * 1024);
    let rss_antes_mb = vm_kb("VmRSS:") / 1024;
    let t = Instant::now();
    v.shrink_to_fit();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let cap_depois_mb = v.capacity() * 8 / (1024 * 1024);
    let rss_depois_mb = vm_kb("VmRSS:") / 1024;
    println!("info,encolher,capacidade_antes_mb,{cap_antes_mb}");
    println!("info,encolher,capacidade_depois_mb,{cap_depois_mb}");
    println!("info,encolher,rss_antes_mb,{rss_antes_mb}");
    println!("info,encolher,rss_depois_mb,{rss_depois_mb}");
    black_box(&v);
    (n, ms)
}

/// Tamanhos de tipos: valida a niche optimization garantida (std::option,
/// "Representation") e Vec vs Box<[T]> (Perf Book: Type Sizes). Sem tempo.
fn tamanhos() -> (usize, f64) {
    use std::mem::size_of;
    #[allow(dead_code)]
    enum EnumGrande {
        Pequena(u8),
        Grande([u8; 256]),
    }
    #[allow(dead_code)]
    enum EnumEncaixotada {
        Pequena(u8),
        Grande(Box<[u8; 256]>),
    }
    let linhas: &[(&str, usize)] = &[
        ("u64", size_of::<u64>()),
        ("Option<u64>", size_of::<Option<u64>>()),
        ("&u64", size_of::<&u64>()),
        ("Option<&u64>", size_of::<Option<&u64>>()),
        ("Box<u64>", size_of::<Box<u64>>()),
        ("Option<Box<u64>>", size_of::<Option<Box<u64>>>()),
        ("NonZeroU64", size_of::<std::num::NonZeroU64>()),
        (
            "Option<NonZeroU64>",
            size_of::<Option<std::num::NonZeroU64>>(),
        ),
        ("Vec<u64>", size_of::<Vec<u64>>()),
        ("Box<[u64]>", size_of::<Box<[u64]>>()),
        ("String", size_of::<String>()),
        ("Box<str>", size_of::<Box<str>>()),
        ("enum_variante_grande_inline", size_of::<EnumGrande>()),
        ("enum_variante_grande_em_box", size_of::<EnumEncaixotada>()),
    ];
    for (nome, bytes) in linhas {
        println!("info,tamanhos,{nome},{bytes}");
    }
    (0, 0.0)
}

/// Soma de 100M u32 + matmul ingênuo 384³ — mesma carga nos perfis dev e
/// release para validar o "10-100x" do Perf Book (Build Configuration).
fn soma_matmul(n: usize) -> (usize, f64) {
    let v: Vec<u32> = (0..n).map(|i| (i & 0xFFFF) as u32).collect();
    let v = black_box(v);
    const L: usize = 384;
    let a: Vec<f32> = (0..L * L).map(|i| (i % 17) as f32).collect();
    let b: Vec<f32> = (0..L * L).map(|i| (i % 13) as f32).collect();
    let (a, b) = (black_box(a), black_box(b));
    let t = Instant::now();
    let soma: u64 = v.iter().map(|&x| u64::from(x)).sum();
    let mut c = vec![0.0f32; L * L];
    for i in 0..L {
        for k in 0..L {
            let aik = a[i * L + k];
            for j in 0..L {
                c[i * L + j] += aik * b[k * L + j];
            }
        }
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    black_box((soma, &c));
    (n, ms)
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let exp = args.get(1).map(String::as_str).unwrap_or("");
    let mut variante = args.get(2).cloned().unwrap_or_default();
    let n_arg: Option<usize> = args.get(3).and_then(|s| s.parse().ok());
    let n = |padrao: usize| n_arg.unwrap_or(padrao);

    let (n_efetivo, ms) = match exp {
        "vec_crescimento" => vec_crescimento(&variante, n(100_000_000)),
        "string_concat" => string_concat(&variante, n(100_000)),
        "produto_escalar" => produto_escalar(&variante, n(100_000_000)),
        "ordenar" => ordenar(&variante, n(50_000_000)),
        "mapa_insercao" => mapa_insercao(&variante, n(10_000_000)),
        "clone_em_laco" => clone_em_laco(&variante, n(2_000_000)),
        "io_escrita" => io_escrita(&variante, n(2_000_000)),
        "io_leitura" => io_leitura(&variante, n(2_000_000)),
        "collect_intermediario" => collect_intermediario(&variante, n(100_000_000)),
        "vec_inicializacao" => vec_inicializacao(&variante, n(500_000_000)),
        "mapa_entry" => mapa_entry(&variante, n(10_000_000)),
        "encolher" => encolher(&variante, n(100_000_000)),
        "tamanhos" => tamanhos(),
        "soma_matmul" => {
            variante = if cfg!(debug_assertions) {
                "dev"
            } else {
                "release"
            }
            .to_string();
            soma_matmul(n(100_000_000))
        }
        _ => {
            eprintln!("experimento desconhecido: {exp:?}");
            std::process::exit(2);
        }
    };
    let hwm_mb = vm_kb("VmHWM:") / 1024;
    println!("resultado,{exp},{variante},{n_efetivo},{ms:.1},{hwm_mb}");
}
