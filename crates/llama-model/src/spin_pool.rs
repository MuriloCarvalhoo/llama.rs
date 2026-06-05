//! Spin-wait thread pool para dispatch de matmul com overhead mínimo.
//!
//! Rayon usa park/unpark de threads entre tokens (~26 µs por dispatch).
//! Este pool mantém N-1 workers em spin permanente: overhead ~1-3 µs.
//!
//! SAFETY: uso correto requer dispatch sequencial (uma batch por vez).
#![allow(unsafe_code)]

use std::hint::spin_loop;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering::*};

/// Parâmetros de uma batch de matmul empacotado.
/// Vida útil garantida pelo dispatcher: só retorna quando todos terminaram.
#[repr(C, align(64))]
pub(crate) struct MatmulBatch {
    pub packed_w: *const u8,
    pub x_row: *const u8,
    pub w_tail: *const u8,
    pub out: *mut f32,
    pub n_blocks: usize,
    pub chunk_size: usize,
    pub n_out_packed: usize,
    pub row_bytes: usize,
}
unsafe impl Send for MatmulBatch {}
unsafe impl Sync for MatmulBatch {}

// ──────────────────────────────────────────────────────────────────────────────
// Globais de sincronização
// ──────────────────────────────────────────────────────────────────────────────
static EPOCH: AtomicUsize = AtomicUsize::new(0); // incrementa por batch
static N_CHUNKS: AtomicUsize = AtomicUsize::new(0);
static NEXT_CHUNK: AtomicUsize = AtomicUsize::new(0);
static DONE: AtomicUsize = AtomicUsize::new(0);
static BATCH_PTR: AtomicUsize = AtomicUsize::new(0); // *const MatmulBatch
static SHUTDOWN: AtomicUsize = AtomicUsize::new(0); // 1 = encerrar
// Garante dispatch sequencial: 0 = livre, 1 = ocupado.
// rayon::join pode chamar dispatch de múltiplas threads — segunda thread processa localmente.
static DISPATCH_LOCK: AtomicUsize = AtomicUsize::new(0);

static N_SPIN_WORKERS: OnceLock<usize> = OnceLock::new();

// ──────────────────────────────────────────────────────────────────────────────
// Loop do worker
// ──────────────────────────────────────────────────────────────────────────────
fn worker_loop() {
    let mut seen_epoch = EPOCH.load(Acquire);
    loop {
        // Spin aguardando novo epoch
        let new_epoch = loop {
            if SHUTDOWN.load(Relaxed) != 0 {
                return;
            }
            let e = EPOCH.load(Acquire);
            if e != seen_epoch {
                break e;
            }
            spin_loop();
        };
        seen_epoch = new_epoch;

        // Carregar n_chunks — escritos antes de EPOCH ser incrementado (Release/Acquire)
        let nc = N_CHUNKS.load(Relaxed);

        // Reivindicar e processar chunks
        loop {
            let ci = NEXT_CHUNK.fetch_add(1, Relaxed);
            if ci >= nc {
                break;
            }
            // SAFETY: BATCH_PTR aponta para MatmulBatch na stack do dispatcher,
            // que só retorna quando DONE == nc → ponteiro é válido aqui.
            let batch = unsafe { &*(BATCH_PTR.load(Relaxed) as *const MatmulBatch) };
            process_chunk(batch, ci);
            DONE.fetch_add(1, Release);
        }
    }
}

/// Pina a thread atual ao conjunto de CPUs via sched_setaffinity.
/// cpu_set_t = 128 bytes (1024 bits) em Linux x86_64.
/// SAFETY: mask é array local de 128 bytes válido; pid=0 = thread atual.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn pin_to_cpus(cpus: &[usize]) {
    if cpus.is_empty() {
        return;
    }
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

/// Inicializa o spin pool com `n_workers` threads de background (além da thread principal).
/// `cpus`: lista de CPUs NUMA node 0 para pinar workers (evitar acesso remoto).
pub(crate) fn init(n_workers: usize, cpus: Vec<usize>) {
    N_SPIN_WORKERS.get_or_init(|| {
        SHUTDOWN.store(0, Relaxed);
        let cpus = std::sync::Arc::new(cpus);
        for _ in 0..n_workers {
            let cpus = cpus.clone();
            std::thread::Builder::new()
                .name("llama-spin".into())
                .spawn(move || {
                    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
                    pin_to_cpus(&cpus);
                    let _ = cpus;
                    worker_loop();
                })
                .expect("falha ao criar spin worker");
        }
        n_workers
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Processamento de chunk
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn process_chunk(batch: &MatmulBatch, ci: usize) {
    use crate::ops::{q8_0_q8_0_dot_8rows_repacked_f16c_pub, q8_0_q8_0_dot_scalar_pub};
    const PB: usize = 272;
    let base_j = ci * batch.chunk_size;
    let end_j = (base_j + batch.chunk_size).min(batch.n_out_packed);
    let chunk = unsafe { std::slice::from_raw_parts_mut(batch.out.add(base_j), end_j - base_j) };
    let n_full8 = (chunk.len() / 8) * 8;
    let mut k = 0usize;
    while k < n_full8 {
        let g = (base_j + k) / 8;
        let group_off = g * batch.n_blocks * PB;
        let [r0, r1, r2, r3, r4, r5, r6, r7] = unsafe {
            q8_0_q8_0_dot_8rows_repacked_f16c_pub(
                batch.packed_w.add(group_off),
                batch.x_row,
                batch.n_blocks,
            )
        };
        chunk[k] = r0;
        chunk[k + 1] = r1;
        chunk[k + 2] = r2;
        chunk[k + 3] = r3;
        chunk[k + 4] = r4;
        chunk[k + 5] = r5;
        chunk[k + 6] = r6;
        chunk[k + 7] = r7;
        k += 8;
    }
    for rem in k..chunk.len() {
        let j = base_j + rem;
        chunk[rem] = unsafe {
            q8_0_q8_0_dot_scalar_pub(
                batch.w_tail.add(j * batch.row_bytes),
                batch.x_row,
                batch.n_blocks,
                batch.row_bytes,
            )
        };
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn process_chunk(_batch: &MatmulBatch, _ci: usize) {
    unimplemented!("spin pool apenas em x86_64");
}

// ──────────────────────────────────────────────────────────────────────────────
// Dispatch
// ──────────────────────────────────────────────────────────────────────────────

/// Dispatcha `n_chunks` tarefas para o spin pool e aguarda conclusão.
/// Se outra thread já está despachando (rayon::join paralelo), processa localmente.
///
/// # Safety
/// `batch` deve permanecer válido até esta função retornar.
pub(crate) unsafe fn dispatch(batch: &MatmulBatch, n_chunks: usize) {
    if n_chunks == 0 {
        return;
    }

    // Tenta adquirir o lock de dispatch (CAS: 0 → 1).
    // Se falhar, outra thread já está despachando — processar localmente (sequencial).
    if DISPATCH_LOCK
        .compare_exchange(0, 1, Acquire, Relaxed)
        .is_err()
    {
        for ci in 0..n_chunks {
            process_chunk(batch, ci);
        }
        return;
    }

    let n_workers = *N_SPIN_WORKERS.get().unwrap_or(&0);

    if n_workers == 0 {
        for ci in 0..n_chunks {
            process_chunk(batch, ci);
        }
        DISPATCH_LOCK.store(0, Release);
        return;
    }

    // Publicar parâmetros antes de incrementar EPOCH (Release garante visibilidade)
    BATCH_PTR.store(batch as *const MatmulBatch as usize, Relaxed);
    N_CHUNKS.store(n_chunks, Relaxed);
    DONE.store(0, Relaxed);
    NEXT_CHUNK.store(0, Relaxed);
    EPOCH.fetch_add(1, Release); // sinaliza workers

    // Thread principal também processa
    loop {
        let ci = NEXT_CHUNK.fetch_add(1, Relaxed);
        if ci >= n_chunks {
            break;
        }
        process_chunk(batch, ci);
        DONE.fetch_add(1, Release);
    }

    // Aguarda todos os workers terminarem
    while DONE.load(Acquire) < n_chunks {
        spin_loop();
    }

    DISPATCH_LOCK.store(0, Release);
}
