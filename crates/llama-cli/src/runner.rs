//! Logica de geracao reutilizavel — streaming e buffered.

use std::time::Instant;

use gguf::GgufFile;
use llama_model::{Model, ModelError};
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::args::Args;

/// Mapeia o `.gguf` em memória em vez de copiá-lo para a RAM. Um `fs::read` do
/// 14B Q8_0 aloca 15 GB de RAM anônima antes mesmo dos pesos serem extraídos;
/// com mmap as páginas ficam no page cache e são recuperáveis sob pressão.
fn map_model(path: &std::path::Path) -> Result<memmap2::Mmap, std::io::Error> {
    let file = std::fs::File::open(path)?;
    // SAFETY: mapeamento read-only de um arquivo tratado como imutável durante a
    // execução. Modificação externa do .gguf enquanto mapeado seria UB — mesma
    // premissa que o llama.cpp adota para o seu mmap de modelo.
    unsafe { memmap2::Mmap::map(&file) }
}

/// Config do modelo com o contexto limitado por `--ctx` (nunca acima do que o GGUF declara).
fn model_config(f: &GgufFile, args: &Args) -> Result<llama_model::LlamaConfig, ModelError> {
    let mut config = llama_model::LlamaConfig::from_gguf(f)?;
    config.ctx = config.ctx.min(args.ctx);
    Ok(config)
}

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
        // get_mut carrega a mesma garantia que o `cpu < 1024` fazia: 16 palavras de 64 bits.
        if let Some(palavra) = mask.get_mut(cpu / 64) {
            *palavra |= 1u64 << (cpu % 64);
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

/// Geração para arquiteturas híbridas (`qwen35`): sempre GPU, sempre layer-split se
/// houver duas placas. O caminho CPU não implementa atenção linear.
#[cfg(feature = "gpu")]
fn run_hibrido(
    args: &Args,
    f: &GgufFile,
    bytes: &[u8],
    cfg: llama_model::LlamaConfig,
    on_token: &mut impl FnMut(&str),
) -> Result<Timing, Box<dyn std::error::Error>> {
    use llama_vulkan::{LayerSplitForward, ResidentForward, VulkanContext};

    let tokenizer = Tokenizer::from_gguf(f)?;
    let sampler = choose_sampler(args);
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let raw = llama_model::GpuRawWeights::from_gguf(f, bytes, &cfg)?;
    let aux = llama_model::GpuAuxWeights::from_gguf(f, bytes, &cfg)?;

    let ctx = VulkanContext::new()?;
    let n_gpus = ctx.amd_compute_devices().len();
    if n_gpus == 0 {
        return Err("nenhuma GPU AMD — o qwen35 não tem caminho de CPU".into());
    }

    let mut n_tokens = 0usize;
    let mut start: Option<Instant> = None;
    let mut emitir = |piece: &str, on_token: &mut dyn FnMut(&str)| {
        if start.is_none() {
            start = Some(Instant::now());
        }
        on_token(piece);
        n_tokens += 1;
    };

    let usar_split = args.gpu_layer_split && n_gpus >= 2;
    if usar_split {
        let backend = LayerSplitForward::new(&ctx, &cfg, &raw, &aux)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
        let layout: Vec<String> = backend
            .layout()
            .iter()
            .map(|(d, a, b)| format!("GPU{d}: camadas {a}..{b}"))
            .collect();
        eprintln!("[GPU] layer-split — {}", layout.join(" | "));
        llama_model::gerar_streaming_residente(
            &cfg,
            &tokenizer,
            &args.prompt,
            args.n_predict,
            &sampler,
            &mut rng,
            &backend,
            &mut |p| emitir(p, on_token),
        )?;
        backend.print_profile();
    } else {
        let backend = ResidentForward::new(&ctx, &cfg, &raw, &aux)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
        llama_model::gerar_streaming_residente(
            &cfg,
            &tokenizer,
            &args.prompt,
            args.n_predict,
            &sampler,
            &mut rng,
            &backend,
            &mut |p| emitir(p, on_token),
        )?;
        backend.print_profile();
    }

    let elapsed_secs = start.map_or(0.0, |s| s.elapsed().as_secs_f64());
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
    // Só no caminho CPU: MPOL_BIND restringe as alocações à RAM de um único nó, e
    // nos backends GPU (onde os pesos vão para a VRAM) isso apenas corta pela metade
    // a memória disponível — um modelo de 15 GB é morto por OOM com RAM livre no nó 1.
    let usa_gpu = args.gpu || args.gpu_single || args.gpu_resident || args.gpu_layer_split;
    if !usa_gpu {
        init_numa_before_load();
    }

    #[cfg(feature = "profiling")]
    let trace_guard = args.trace.clone().map(crate::trace::init);

    let fase_mmap = llama_model::perfil_carga::Fase::nova("mmap+parse");
    let bytes = map_model(&args.model)?;
    let f = GgufFile::parse(&bytes)?;
    let cfg = model_config(&f, args)?;
    drop(fase_mmap);

    // Arquiteturas híbridas (`qwen35`) não têm a estrutura densa que o `Model` exige — em
    // 3 de cada 4 camadas não existe `attn_q` —, então o caminho de GPU delas monta os
    // pesos direto do GGUF e usa o laço de geração livre.
    #[cfg(feature = "gpu")]
    if cfg.delta_net.is_some() {
        return run_hibrido(args, &f, &bytes, cfg, on_token);
    }

    let model = {
        #[cfg(feature = "profiling")]
        let _s = tracing::info_span!("carregar_modelo").entered();
        Model::load_with_config(&f, &bytes, cfg)?
    };
    // Criar pool rayon DEPOIS do load: workers herdam afinidade e política de memória.
    init_rayon_after_model_load();
    // O spin pool serve só ao matmul Q8_0 de CPU. Nos backends GPU ele nunca é
    // despachado, mas os N-1 workers ficam em busy-wait permanente queimando todos os
    // núcleos: o `perf` de um decode do 32B em layer-split mostrou 97.6% dos ciclos em
    // `llama-spin`. A thread que grava o command buffer, copia os logits e faz o argmax
    // disputa CPU com eles, e o tempo por token oscilava 13.4–15.1 tok/s sem que o
    // tempo de GPU mudasse.
    if !usa_gpu {
        // Acordar todos os workers rayon com um broadcast noop — evita latência de
        // wakeup no primeiro matmul paralelo (workers ficam em spin-wait ~1ms após wakeup).
        rayon::broadcast(|_| {});
        // Inicializar spin pool com N-1 workers (N = rayon thread count).
        // Pinar workers ao node 0 (mesma afinidade dos rayon workers) para acesso local à RAM.
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

    #[cfg(feature = "gpu")]
    let used_gpu = if args.gpu_layer_split {
        use llama_vulkan::{LayerSplitForward, VulkanContext};
        match VulkanContext::new() {
            Ok(ctx) if ctx.amd_compute_devices().len() >= 2 => {
                let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config)?;
                let aux = model.gpu_aux_weights()?;
                let backend = LayerSplitForward::new(&ctx, &model.config, &raw, &aux)
                    .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
                let layout: Vec<String> = backend
                    .layout()
                    .iter()
                    .map(|(d, a, b)| format!("GPU{d}: camadas {a}..{b}"))
                    .collect();
                eprintln!("[GPU] layer-split — {}", layout.join(" | "));
                model.generate_streaming_gpu_resident(
                    &tokenizer,
                    &args.prompt,
                    args.n_predict,
                    &sampler,
                    &mut rng,
                    &backend,
                    &mut |piece| {
                        if start.is_none() {
                            start = Some(Instant::now());
                        }
                        on_token(piece);
                        n_tokens += 1;
                    },
                )?;
                // No-op sem LLAMA_RS_PROFILE=1.
                backend.print_profile();
                #[cfg(feature = "profiling")]
                if let Some(g) = trace_guard.as_ref() {
                    for (track, spans) in backend.gpu_spans_by_track() {
                        g.add_gpu_spans(&track, &spans);
                    }
                }
                true
            }
            Ok(_) => {
                eprintln!("[GPU] menos de 2 GPUs AMD — fallback CPU");
                false
            }
            Err(e) => {
                eprintln!("[GPU] Vulkan indisponivel ({e}) — fallback CPU");
                false
            }
        }
    } else if args.gpu_resident {
        use llama_vulkan::{ResidentForward, VulkanContext};
        match VulkanContext::new() {
            Ok(ctx) if !ctx.amd_compute_devices().is_empty() => {
                let dev0 = ctx.amd_compute_devices()[0].name().to_owned();
                eprintln!("[GPU] {dev0} — decode 100% na GPU (resident-fwd)");
                let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config)?;
                let aux = model.gpu_aux_weights()?;
                let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux)
                    .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
                model.generate_streaming_gpu_resident(
                    &tokenizer,
                    &args.prompt,
                    args.n_predict,
                    &sampler,
                    &mut rng,
                    &backend,
                    &mut |piece| {
                        if start.is_none() {
                            start = Some(Instant::now());
                        }
                        on_token(piece);
                        n_tokens += 1;
                    },
                )?;
                // No-op sem LLAMA_RS_PROFILE=1.
                backend.print_profile();
                #[cfg(feature = "profiling")]
                if let Some(g) = trace_guard.as_ref() {
                    g.add_gpu_spans(&backend.device_name(), &backend.gpu_spans());
                }
                true
            }
            Ok(_) => {
                eprintln!("[GPU] nenhum device AMD — fallback CPU");
                false
            }
            Err(e) => {
                eprintln!("[GPU] Vulkan indisponivel ({e}) — fallback CPU");
                false
            }
        }
    } else if args.gpu_single {
        use llama_vulkan::{ResidentGpu, VulkanContext};
        match VulkanContext::new() {
            Ok(ctx) if !ctx.amd_compute_devices().is_empty() => {
                let dev0 = ctx.amd_compute_devices()[0].name().to_owned();
                eprintln!("[GPU] {dev0} — decode na GPU (single, resident)");
                let gpu_w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config)?;
                let backend = ResidentGpu::new(&ctx)?;
                model.generate_streaming_gpu(
                    &tokenizer,
                    &args.prompt,
                    args.n_predict,
                    &sampler,
                    &mut rng,
                    &backend,
                    &gpu_w,
                    &mut |piece| {
                        if start.is_none() {
                            start = Some(Instant::now());
                        }
                        on_token(piece);
                        n_tokens += 1;
                    },
                )?;
                true
            }
            Ok(_) => {
                eprintln!("[GPU] nenhum device AMD — fallback CPU");
                false
            }
            Err(e) => {
                eprintln!("[GPU] Vulkan indisponivel ({e}) — fallback CPU");
                false
            }
        }
    } else if args.gpu {
        use llama_vulkan::{DualGpuBackend, VulkanContext};
        match VulkanContext::new() {
            Ok(ctx) if ctx.amd_compute_devices().len() >= 2 => {
                let devs = ctx.amd_compute_devices();
                eprintln!(
                    "[GPU] {} + {} — decode na GPU",
                    devs[0].name(),
                    devs[1].name()
                );
                let gpu_w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config)?;
                let backend = DualGpuBackend::new(&ctx)?;
                model.generate_streaming_gpu(
                    &tokenizer,
                    &args.prompt,
                    args.n_predict,
                    &sampler,
                    &mut rng,
                    &backend,
                    &gpu_w,
                    &mut |piece| {
                        if start.is_none() {
                            start = Some(Instant::now());
                        }
                        on_token(piece);
                        n_tokens += 1;
                    },
                )?;
                true
            }
            Ok(ctx) => {
                eprintln!(
                    "[GPU] {} device(s) AMD (<2) — fallback CPU",
                    ctx.amd_compute_devices().len()
                );
                false
            }
            Err(e) => {
                eprintln!("[GPU] Vulkan indisponivel ({e}) — fallback CPU");
                false
            }
        }
    } else {
        false
    };
    #[cfg(not(feature = "gpu"))]
    let used_gpu = {
        if args.gpu || args.gpu_single || args.gpu_resident || args.gpu_layer_split {
            eprintln!("[GPU] Build sem feature 'gpu' -- recompile com: cargo build --features gpu");
        }
        false
    };

    if !used_gpu {
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
    }

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
    let bytes = map_model(&args.model)?;
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

#[cfg(test)]
mod choose_sampler_tests {
    //! `choose_sampler` mapeia os args da CLI para uma estratégia — lógica pura, sem
    //! I/O nem GPU, então testável sem nenhum modelo.
    use super::*;
    use std::path::PathBuf;

    fn base_args() -> Args {
        Args {
            model: PathBuf::new(),
            prompt: String::new(),
            n_predict: 1,
            temp: 0.8,
            top_k: 40,
            top_p: 0.9,
            seed: 0,
            ctx: 4096,
            no_display_prompt: false,
            timings: false,
            gpu: false,
            gpu_single: false,
            gpu_resident: false,
            gpu_layer_split: false,
            trace: None,
        }
    }

    #[test]
    fn temp_zero_e_greedy() {
        let args = Args {
            temp: 0.0,
            ..base_args()
        };
        assert!(matches!(choose_sampler(&args), Sampler::Greedy));
    }

    #[test]
    fn top_k_positivo_prevalece() {
        let args = Args {
            top_k: 40,
            ..base_args()
        };
        assert!(matches!(choose_sampler(&args), Sampler::TopK { k: 40, .. }));
    }

    #[test]
    fn top_p_abaixo_de_um_quando_top_k_desabilitado() {
        let args = Args {
            top_k: 0,
            top_p: 0.9,
            ..base_args()
        };
        assert!(matches!(
            choose_sampler(&args),
            Sampler::TopP { p, .. } if p == 0.9
        ));
    }

    #[test]
    fn temperatura_pura_quando_top_k_e_top_p_desabilitados() {
        let args = Args {
            top_k: 0,
            top_p: 1.0,
            temp: 0.5,
            ..base_args()
        };
        assert!(matches!(
            choose_sampler(&args),
            Sampler::Temperature { temp } if temp == 0.5
        ));
    }
}
