//! Logica de geracao reutilizavel — streaming e buffered.

use std::time::Instant;

use gguf::GgufFile;
use llama_model::Model;
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::args::Args;

/// Lê a lista de CPUs de um nó NUMA (ex.: "0-13,28-41") e retorna Vec<usize>.
#[cfg(target_os = "linux")]
fn read_numa_node_cpus(node: usize) -> Vec<usize> {
    let path = format!("/sys/devices/system/node/node{node}/cpulist");
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    let mut cpus = Vec::new();
    for part in content.trim().split(',') {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (a.parse::<usize>(), b.parse::<usize>()) {
                cpus.extend(lo..=hi);
            }
        } else if let Ok(n) = part.trim().parse::<usize>() {
            cpus.push(n);
        }
    }
    cpus
}

/// Pina a thread atual ao conjunto de CPUs fornecido via sched_setaffinity.
/// cpu_set_t = 128 bytes (1024 bits) em Linux x86_64.
/// SAFETY: mask é array local de 128 bytes válido; pid=0 = thread atual.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn pin_thread_to_cpus(cpus: &[usize]) {
    let mut mask = [0u64; 16];
    for &cpu in cpus {
        if cpu < 1024 {
            mask[cpu / 64] |= 1u64 << (cpu % 64);
        }
    }
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") 203u64 => _,
            in("rdi") 0u64,
            in("rsi") 128u64,
            in("rdx") mask.as_ptr(),
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
}

/// Define política MPOL_BIND para NUMA node 0 via set_mempolicy.
/// Pesos carregados vão para a RAM local do socket 0 (acesso ~10 ns).
/// SAFETY: nodemask é u64 local válido; syscall 238 é set_mempolicy no x86_64.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn bind_memory_to_node0() {
    let nodemask: u64 = 1u64; // bit 0 = NUMA node 0
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") 238u64 => _,  // set_mempolicy
            in("rdi") 2u64,                  // MPOL_BIND
            in("rsi") &nodemask as *const u64,
            in("rdx") 64u64,                 // maxnode
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
}

/// Chamado ANTES de carregar o modelo: pina CPU ao node 0 e define MPOL_BIND
/// para que os ~500 MB de pesos fiquem na RAM local do socket 0.
fn init_numa_before_load() {
    #[cfg(target_os = "linux")]
    let numa0_cpus = read_numa_node_cpus(0);
    #[cfg(not(target_os = "linux"))]
    let numa0_cpus: Vec<usize> = Vec::new();

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        if !numa0_cpus.is_empty() {
            pin_thread_to_cpus(&numa0_cpus);
            bind_memory_to_node0();
        }
    }

    let _ = numa0_cpus;
}

/// Chamado APÓS carregar o modelo: cria o pool rayon com núcleos físicos do
/// node 0 (sem hyperthreads). Workers herdam a política MPOL_BIND e acessam
/// os pesos localmente.
fn init_rayon_after_model_load() {
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        #[cfg(target_os = "linux")]
        let numa0_cpus = read_numa_node_cpus(0);
        #[cfg(not(target_os = "linux"))]
        let numa0_cpus: Vec<usize> = Vec::new();

        let n = if numa0_cpus.is_empty() {
            (std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                / 2)
            .max(1)
        } else {
            // Núcleos físicos do nó 0 (sem hyperthreading).
            (numa0_cpus.len() / 2).max(1)
        };

        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            if !numa0_cpus.is_empty() {
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .start_handler(move |_tid| {
                        pin_thread_to_cpus(&numa0_cpus);
                    })
                    .build_global();
                return;
            }
        }

        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }
}

/// Metricas coletadas durante a geracao.
pub struct Timing {
    pub n_tokens: usize,
    pub elapsed_secs: f64,
    pub tokens_per_sec: f64,
}

/// Carrega o modelo e chama `on_token` para cada token gerado. Retorna metricas de tempo.
pub fn run_generate(
    args: &Args,
    on_token: &mut impl FnMut(&str),
) -> Result<Timing, Box<dyn std::error::Error>> {
    // Pinar CPU + política MPOL_BIND ANTES de ler o modelo: pesos ficam na RAM
    // local do node 0 (acesso ~10 ns para as threads do socket 0).
    init_numa_before_load();

    if args.gpu {
        #[cfg(feature = "gpu")]
        {
            use llama_vulkan::VulkanContext;
            match VulkanContext::new() {
                Ok(ctx) => {
                    let devs = ctx.amd_compute_devices();
                    if devs.len() >= 2 {
                        eprintln!(
                            "[GPU] {} + {} detectados -- backend Vulkan reconhecido",
                            devs[0].name(),
                            devs[1].name()
                        );
                        eprintln!(
                            "[GPU] AVISO: matmuls ainda em CPU -- substituicao GPU e fase futura"
                        );
                    } else {
                        eprintln!(
                            "[GPU] {} device(s) AMD detectado(s) (< 2) -- fallback CPU",
                            devs.len()
                        );
                    }
                }
                Err(e) => eprintln!("[GPU] Vulkan indisponivel ({e}) -- fallback CPU"),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!("[GPU] Build sem feature 'gpu' -- recompile com: cargo build --features gpu");
        }
    }

    let bytes = std::fs::read(&args.model)?;
    let f = GgufFile::parse(&bytes)?;
    let model = Model::load(&f, &bytes)?;
    // Criar pool rayon DEPOIS do load: workers herdam afinidade e política de memória.
    init_rayon_after_model_load();
    // Acordar todos os workers rayon com um broadcast noop — evita latência de
    // wakeup no primeiro matmul paralelo (workers ficam em spin-wait ~1ms após wakeup).
    rayon::broadcast(|_| {});
    // Inicializar spin pool com N-1 workers (N = rayon thread count).
    // Pinar workers ao node 0 (mesma afinidade dos rayon workers) para acesso local à RAM.
    {
        #[cfg(target_os = "linux")]
        let numa0_cpus = read_numa_node_cpus(0);
        #[cfg(not(target_os = "linux"))]
        let numa0_cpus: Vec<usize> = Vec::new();
        let n_workers = rayon::current_num_threads().saturating_sub(1);
        llama_model::init_spin_pool(n_workers, numa0_cpus);
    }
    let tokenizer = Tokenizer::from_gguf(&f)?;
    let sampler = choose_sampler(args);
    let mut rng = SmallRng::seed_from_u64(args.seed);

    // Timer inicia quando o primeiro token é emitido (após prefill do prompt),
    // equivalente ao "eval time" do llama.cpp que exclui prompt eval.
    let mut n_tokens = 0usize;
    let mut start: Option<Instant> = None;

    model.generate_streaming(
        &tokenizer,
        &args.prompt,
        args.n_predict,
        &sampler,
        &mut rng,
        &mut |piece| {
            if start.is_none() {
                start = Some(Instant::now());
            }
            on_token(piece);
            n_tokens += 1;
        },
    )?;

    let elapsed_secs = start.map_or(0.0, |t| t.elapsed().as_secs_f64());
    #[allow(clippy::cast_precision_loss)]
    let tokens_per_sec = if elapsed_secs > 0.0 {
        n_tokens as f64 / elapsed_secs
    } else {
        0.0
    };

    Ok(Timing {
        n_tokens,
        elapsed_secs,
        tokens_per_sec,
    })
}

/// Carrega o modelo e retorna o texto completo como String (sem streaming).
/// Mantém compatibilidade com `greedy_gate.rs` e outros testes de integracao.
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
