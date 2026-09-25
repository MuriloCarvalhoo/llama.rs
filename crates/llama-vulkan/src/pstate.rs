//! Clocks fixos durante a geração: `AMDGPU_CTX_OP_SET_STABLE_PSTATE`, a mesma ioctl que o
//! RADV usa para o RGP. Não exige root.
//!
//! Por quê: no layer-split cada GPU fica ociosa enquanto a outra roda as camadas dela (~metade
//! do token), e o DPM automático da MI50 não sustenta o clock nesse regime. Medido em
//! 2026-09-25 num decode do Qwen3.8-27B: núcleo em 860 MHz em 75–88% das amostras e a HBM da
//! card1 em 350 MHz em 44%. O matvec Q4_K, que isolado e com clock sustentado lê 684 GB/s,
//! caía para ~450 no token. Com `standard` (núcleo 1316 MHz e HBM 1000 MHz fixos) o decode foi
//! de 21,65 para 26,16 tok/s em 2000 tokens; `peak` (núcleo 1700 MHz) deu 31,5 tok/s em
//! gerações curtas, mas levou a junction da card1 a 102 °C em 30 s — e com o vigia abaixo
//! segurando a temperatura rendeu 27,4 tok/s, nada acima do `standard`.
//!
//! O pstate é global ao device e vale enquanto o contexto amdgpu que o pediu existir: o kernel
//! o desfaz quando o fd fecha, inclusive se o processo morrer. Duas salvaguardas por cima:
//! - **ociosidade**: sem submit há [`OCIOSO`], o device volta ao automático — um servidor
//!   parado não fica esquentando com clock fixo. O próximo submit fixa de novo.
//! - **temperatura**: com a junction em [`QUENTE`] ou mais, desce um degrau (`peak` →
//!   `standard` → automático) por pelo menos [`ESFRIAR`] e até esfriar para [`FRIO`].
//!
//! O prefill é limitado por compute, não por banda, e roda em `peak` qualquer que seja o modo
//! (menos `auto`): o `standard` o deixava 17% mais lento.
//!
//! `LLAMA_RS_PSTATE`: `standard` (padrão), `peak`, ou `auto` para não mexer.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const OCIOSO: Duration = Duration::from_secs(2);
/// Junction em °C. O crit da MI50 é 105 e o watchdog do usuário mata em 105.
const QUENTE: i64 = 95;
const FRIO: i64 = 85;
const PERIODO: Duration = Duration::from_millis(250);

// uapi/drm/amdgpu_drm.h: DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union drm_amdgpu_ctx),
// com a union de 16 bytes.
const DRM_IOCTL_AMDGPU_CTX: libc::c_ulong = 0xC010_6442;
const AMDGPU_CTX_OP_ALLOC_CTX: u32 = 1;
const AMDGPU_CTX_OP_SET_STABLE_PSTATE: u32 = 6;
const PSTATE_NONE: u32 = 0;
const PSTATE_STANDARD: u32 = 1;
const PSTATE_PEAK: u32 = 4;

/// `union drm_amdgpu_ctx`: na ida `{op, flags, ctx_id, priority}`, na volta do ALLOC o
/// `ctx_id` no primeiro campo.
#[repr(C)]
struct CtxArgs([u32; 4]);

fn ctx_ioctl(fd: &File, args: &mut CtxArgs) -> std::io::Result<()> {
    // SAFETY: `fd` é um render node amdgpu aberto; `args` tem o tamanho e o layout da union
    // `drm_amdgpu_ctx` que o número da ioctl declara, e vive durante a chamada.
    let r = unsafe {
        libc::ioctl(
            fd.as_raw_fd(),
            DRM_IOCTL_AMDGPU_CTX as _,
            std::ptr::from_mut(args),
        )
    };
    if r == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn modo_do_ambiente() -> Option<u32> {
    match std::env::var("LLAMA_RS_PSTATE").as_deref() {
        Err(_) | Ok("standard") => Some(PSTATE_STANDARD),
        Ok("peak") => Some(PSTATE_PEAK),
        Ok(_) => None,
    }
}

/// `/sys/bus/pci/devices/<pci>/drm/renderD*` → `/dev/dri/renderD*`.
fn render_node(pci: &str) -> Option<PathBuf> {
    std::fs::read_dir(Path::new("/sys/bus/pci/devices").join(pci).join("drm"))
        .ok()?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .find(|n| n.starts_with("renderD"))
        .map(|n| Path::new("/dev/dri").join(n))
}

/// O `tempN_input` rotulado `junction` no hwmon do device, em milésimos de °C.
fn sensor_junction(pci: &str) -> Option<PathBuf> {
    let hwmon = Path::new("/sys/bus/pci/devices").join(pci).join("hwmon");
    for h in std::fs::read_dir(hwmon).ok()?.flatten() {
        for n in 1..=4 {
            let label = h.path().join(format!("temp{n}_label"));
            if std::fs::read_to_string(label).is_ok_and(|l| l.trim() == "junction") {
                return Some(h.path().join(format!("temp{n}_input")));
            }
        }
    }
    None
}

/// Um degrau abaixo, para quando a junction passa de [`QUENTE`]: `peak` cai para `standard`
/// e `standard` para o automático.
fn degrau_abaixo(modo: u32) -> u32 {
    if modo == PSTATE_PEAK {
        PSTATE_STANDARD
    } else {
        PSTATE_NONE
    }
}

/// Mínimo no degrau de baixo depois de esquentar: a junction esfria em segundos e, sem isso, o
/// clock alterna entre os degraus a cada leitura do sensor.
const ESFRIAR: Duration = Duration::from_secs(10);

struct Estado {
    /// O pstate pedido ao kernel agora.
    aplicado: u32,
    /// Até quando ficar no degrau de baixo, desde a última leitura em [`QUENTE`] ou mais.
    quente_ate: Option<Instant>,
    avisou: bool,
    ultimo_uso: Instant,
    /// O último submit foi de prefill (limitado por compute) ou de decode (por banda).
    prefill: bool,
}

struct Comum {
    fd: File,
    ctx_id: u32,
    modo: u32,
    nome: String,
    estado: Mutex<Estado>,
}

impl Comum {
    fn aplicar(&self, pstate: u32) -> bool {
        let mut a = CtxArgs([AMDGPU_CTX_OP_SET_STABLE_PSTATE, pstate, self.ctx_id, 0]);
        ctx_ioctl(&self.fd, &mut a).is_ok()
    }

    /// Leva o device ao pstate que o estado pede: automático se ocioso, um degrau abaixo se
    /// quente, o modo configurado no resto. Só faz a ioctl quando muda.
    fn ajustar(&self, e: &mut Estado) {
        // O prefill é limitado por compute e o `standard` trava o núcleo em 1316 MHz: medido
        // 12,35 ms/token contra 10,00 em `peak` e 10,56 no automático (1700 prompts de 24).
        let base = if e.prefill { PSTATE_PEAK } else { self.modo };
        let alvo = if e.ultimo_uso.elapsed() >= OCIOSO {
            PSTATE_NONE
        } else if e.quente_ate.is_some() {
            degrau_abaixo(base)
        } else {
            base
        };
        if alvo != e.aplicado && self.aplicar(alvo) {
            e.aplicado = alvo;
        }
    }
}

/// Clock fixo num device enquanto houver submits. `Drop` para o vigia e fecha o fd, e o
/// kernel devolve o device ao automático.
pub(crate) struct Pstate {
    comum: Arc<Comum>,
    parar: Option<Sender<()>>,
    vigia: Option<std::thread::JoinHandle<()>>,
}

impl Pstate {
    /// `None` com `LLAMA_RS_PSTATE=auto`, sem render node, ou se o kernel recusar (outro
    /// processo já segura o pstate do device: `EBUSY`).
    pub(crate) fn novo(pci: &str, nome: &str) -> Option<Self> {
        let modo = modo_do_ambiente()?;
        let fd = File::options()
            .read(true)
            .write(true)
            .open(render_node(pci)?)
            .ok()?;
        let mut a = CtxArgs([AMDGPU_CTX_OP_ALLOC_CTX, 0, 0, 0]);
        ctx_ioctl(&fd, &mut a).ok()?;
        let comum = Arc::new(Comum {
            fd,
            ctx_id: a.0[0],
            modo,
            nome: nome.to_owned(),
            estado: Mutex::new(Estado {
                aplicado: PSTATE_NONE,
                quente_ate: None,
                avisou: false,
                ultimo_uso: Instant::now(),
                prefill: false,
            }),
        });
        if !comum.aplicar(modo) {
            return None;
        }
        comum.estado.lock().ok()?.aplicado = modo;
        let (tx, rx) = channel();
        let c = Arc::clone(&comum);
        let sensor = sensor_junction(pci);
        let vigia = std::thread::Builder::new()
            .name("llama-pstate".into())
            .spawn(move || vigiar(&c, sensor.as_deref(), &rx))
            .ok()?;
        Some(Self {
            comum,
            parar: Some(tx),
            vigia: Some(vigia),
        })
    }

    /// Chamado antes de cada submit: marca o uso e a fase, e ajusta o clock se ela mudou ou se
    /// o device tinha voltado ao automático por ociosidade — umas poucas ioctls por
    /// requisição, não uma por token.
    pub(crate) fn em_uso(&self, prefill: bool) {
        let Ok(mut e) = self.comum.estado.lock() else {
            return;
        };
        e.ultimo_uso = Instant::now();
        e.prefill = prefill;
        self.comum.ajustar(&mut e);
    }
}

fn vigiar(c: &Comum, sensor: Option<&Path>, parar: &Receiver<()>) {
    while let Err(RecvTimeoutError::Timeout) = parar.recv_timeout(PERIODO) {
        let temp = sensor
            .and_then(|s| std::fs::read_to_string(s).ok())
            .and_then(|t| t.trim().parse::<i64>().ok())
            .map(|mc| mc / 1000);
        let Ok(mut e) = c.estado.lock() else {
            return;
        };
        match temp {
            Some(t) if t >= QUENTE => {
                e.quente_ate = Some(Instant::now() + ESFRIAR);
                if !e.avisou {
                    e.avisou = true;
                    eprintln!(
                        "[pstate] {}: junction {t} °C — clock um degrau abaixo até esfriar para \
                         {FRIO} °C (aviso único)",
                        c.nome
                    );
                }
            }
            Some(t) if t <= FRIO && e.quente_ate.is_some_and(|ate| Instant::now() >= ate) => {
                e.quente_ate = None;
            }
            _ => {}
        }
        c.ajustar(&mut e);
    }
}

impl Drop for Pstate {
    fn drop(&mut self) {
        // Soltar o `Sender` acorda o vigia com `Disconnected`.
        drop(self.parar.take());
        if let Some(v) = self.vigia.take() {
            let _ = v.join();
        }
        let _ = self.comum.aplicar(PSTATE_NONE);
    }
}
