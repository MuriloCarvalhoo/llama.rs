//! Cronômetro das fases da carga do modelo, atrás de `LLAMA_RS_LOAD_PROFILE=1`.
//!
//! Subir um modelo de 16 GB para as duas placas leva dezenas de segundos, e o custo se
//! espalha por três crates: mmap e parse no CLI, dequant dos auxiliares aqui, repack e
//! upload no backend Vulkan. Sem um acumulador comum não dá para dizer qual fase paga o
//! quê — e é isso que decide onde mexer.
//!
//! Fases de mesmo nome se somam: os ~600 uploads de um shard viram uma linha só.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// `LLAMA_RS_LOAD_PROFILE=1` liga o perfil. Lido uma vez por processo.
#[must_use]
pub fn ligado() -> bool {
    static LIGADO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LIGADO.get_or_init(|| std::env::var("LLAMA_RS_LOAD_PROFILE").is_ok_and(|v| v != "0"))
}

/// Fases na ordem em que apareceram, com o tempo acumulado de cada uma.
static FASES: Mutex<Vec<(String, Duration)>> = Mutex::new(Vec::new());
/// Instante do primeiro registro — dá o tempo de parede, que difere da soma quando os
/// shards carregam em paralelo.
static INICIO: Mutex<Option<Instant>> = Mutex::new(None);

/// Soma `d` ao total da fase `nome`.
pub fn registrar(nome: &str, d: Duration) {
    if !ligado() {
        return;
    }
    if let Ok(mut inicio) = INICIO.lock() {
        inicio.get_or_insert_with(|| Instant::now() - d);
    }
    if let Ok(mut fases) = FASES.lock() {
        match fases.iter_mut().find(|(n, _)| n == nome) {
            Some((_, acc)) => *acc += d,
            None => fases.push((nome.to_owned(), d)),
        }
    }
}

/// Cronômetro que registra o tempo decorrido ao sair de escopo. `None` com o perfil
/// desligado, então nada é medido nem alocado no caminho normal.
pub struct Fase {
    nome: String,
    t0: Instant,
}

impl Fase {
    /// Abre uma fase, ou `None` se o perfil está desligado.
    #[must_use]
    pub fn nova(nome: impl Into<String>) -> Option<Self> {
        ligado().then(|| Self {
            nome: nome.into(),
            t0: Instant::now(),
        })
    }
}

impl Drop for Fase {
    fn drop(&mut self) {
        registrar(&self.nome, self.t0.elapsed());
    }
}

/// Imprime a tabela no stderr e zera o acumulador. Chamada ao fim da carga.
pub fn imprimir() {
    if !ligado() {
        return;
    }
    let Ok(mut fases) = FASES.lock() else {
        return;
    };
    if fases.is_empty() {
        return;
    }
    let parede = INICIO
        .lock()
        .ok()
        .and_then(|mut i| i.take())
        .map(|t| t.elapsed());
    let soma: Duration = fases.iter().map(|(_, d)| *d).sum();
    let largura = fases.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (nome, d) in fases.iter() {
        eprintln!("[carga] {nome:<largura$}  {:7.2} s", d.as_secs_f64());
    }
    eprintln!(
        "[carga] {:<largura$}  {:7.2} s",
        "soma das fases",
        soma.as_secs_f64()
    );
    if let Some(p) = parede {
        eprintln!("[carga] {:<largura$}  {:7.2} s", "parede", p.as_secs_f64());
    }
    fases.clear();
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn sem_a_variavel_de_ambiente_nada_e_medido() {
        if ligado() {
            return; // suíte rodando com LLAMA_RS_LOAD_PROFILE=1
        }
        assert!(Fase::nova("x").is_none(), "guarda não deveria existir");
        registrar("x", Duration::from_secs(1));
        assert!(
            FASES.lock().unwrap().is_empty(),
            "tabela deveria ficar vazia"
        );
    }
}
