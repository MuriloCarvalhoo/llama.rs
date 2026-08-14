//! Timeline cronológica de CPU + GPU no formato Chrome Trace Event, para abrir em
//! <https://ui.perfetto.dev>.
//!
//! Por que um coletor próprio em vez do `tracing-chrome`: ele carimba o tempo no momento
//! em que o evento acontece, e as zonas de GPU só ficam conhecidas **depois** (os
//! timestamp queries são lidos quando o fence libera). Precisamos posicionar eventos com
//! timestamps passados, o que exige escrever o JSON aqui.
//!
//! O motivo de existir as duas trilhas: um span de CPU em volta de `vkQueueSubmit` mede o
//! custo de *enfileirar*, não de *executar* — o `wait_for_fences` apareceria como uma barra
//! opaca de dezenas de ms. As zonas de GPU caem dentro dessa janela e mostram qual op pesa.

use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;
use tracing::span;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Uma fatia da timeline: nome, início/fim e a trilha em que aparece.
struct Slice {
    name: String,
    track: String,
    start_us: f64,
    dur_us: f64,
}

/// Estado compartilhado entre o `Layer` (que recebe os spans de CPU) e o writer.
struct Collector {
    epoch: Instant,
    slices: Mutex<Vec<Slice>>,
}

impl Collector {
    fn us(&self, t: Instant) -> f64 {
        // `saturating_duration_since` evita pânico se `t` for anterior à época (as zonas de
        // GPU são reconstruídas para trás a partir do fence e podem preceder o primeiro span).
        t.saturating_duration_since(self.epoch).as_secs_f64() * 1e6
    }

    fn push(&self, name: String, track: String, start: Instant, end: Instant) {
        let s = self.us(start);
        let e = self.us(end);
        if let Ok(mut v) = self.slices.lock() {
            v.push(Slice {
                name,
                track,
                start_us: s,
                dur_us: (e - s).max(0.0),
            });
        }
    }
}

/// `Layer` que grava cada span de CPU com início e fim reais.
struct CpuLayer(std::sync::Arc<Collector>);

/// Instante de entrada do span, guardado nas extensões do registry.
struct Entered(Instant);

impl<S> Layer<S> for CpuLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(sp) = ctx.span(id) {
            sp.extensions_mut().replace(Entered(Instant::now()));
        }
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let Some(sp) = ctx.span(&id) else { return };
        let started = sp.extensions().get::<Entered>().map(|e| e.0);
        if let Some(start) = started {
            self.0.push(
                sp.name().to_owned(),
                "CPU".to_owned(),
                start,
                Instant::now(),
            );
        }
    }
}

/// Guarda a instrumentação ativa; escreve o arquivo no `drop`.
pub struct TraceGuard {
    collector: std::sync::Arc<Collector>,
    path: std::path::PathBuf,
}

impl TraceGuard {
    /// Acrescenta as zonas de GPU de um backend à timeline, numa trilha própria.
    pub fn add_gpu_spans(&self, track: &str, spans: &[llama_vulkan::GpuSpan]) {
        for s in spans {
            self.collector
                .push(s.name.to_owned(), track.to_owned(), s.start, s.end);
        }
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        let Ok(slices) = self.collector.slices.lock() else {
            return;
        };
        // Uma "thread" do formato Chrome por trilha, para CPU e GPUs ficarem em faixas
        // separadas e alinhadas no mesmo eixo de tempo.
        let mut tracks: Vec<&str> = slices.iter().map(|s| s.track.as_str()).collect();
        tracks.sort_unstable();
        tracks.dedup();

        let mut out = String::from("{\"traceEvents\":[\n");
        for (tid, name) in tracks.iter().enumerate() {
            out.push_str(&format!(
                "{{\"ph\":\"M\",\"pid\":1,\"tid\":{tid},\"name\":\"thread_name\",\
                 \"args\":{{\"name\":\"{name}\"}}}},\n"
            ));
        }
        for s in slices.iter() {
            let tid = tracks.iter().position(|t| *t == s.track).unwrap_or(0);
            // "X" = evento completo (início + duração), o que o Perfetto desenha como fatia.
            out.push_str(&format!(
                "{{\"ph\":\"X\",\"pid\":1,\"tid\":{tid},\"name\":\"{}\",\"ts\":{:.3},\"dur\":{:.3}}},\n",
                s.name.replace('"', "'"),
                s.start_us,
                s.dur_us
            ));
        }
        if out.ends_with(",\n") {
            out.truncate(out.len() - 2);
        }
        out.push_str("\n]}\n");

        match std::fs::File::create(&self.path).and_then(|mut f| f.write_all(out.as_bytes())) {
            Ok(()) => eprintln!(
                "[trace] {} fatias em {} — abra em https://ui.perfetto.dev",
                slices.len(),
                self.path.display()
            ),
            Err(e) => eprintln!("[trace] falha ao escrever {}: {e}", self.path.display()),
        }
    }
}

/// Instala a instrumentação. O arquivo é escrito quando o guard sai de escopo.
pub fn init(path: std::path::PathBuf) -> TraceGuard {
    use tracing_subscriber::prelude::*;

    let collector = std::sync::Arc::new(Collector {
        epoch: Instant::now(),
        slices: Mutex::new(Vec::new()),
    });
    let layer = CpuLayer(std::sync::Arc::clone(&collector));
    // `try_init` para não entrar em pânico se algo já tiver instalado um subscriber.
    let _ = tracing_subscriber::registry().with(layer).try_init();
    TraceGuard { collector, path }
}
