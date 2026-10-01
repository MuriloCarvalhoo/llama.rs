//! `cargo xtask sync`: transforma commits novos do upstream em tarefas de porte.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::git::{self, Change, Git, short};
use crate::map::{Action, SyncMap};
use crate::paths::Paths;
use crate::state::State;
use crate::task::{self, TaskMeta};

/// O que um commit pede para uma crate.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CratePlan {
    pub port: Vec<String>,
    pub vendor: Vec<String>,
    pub vendor_deleted: Vec<String>,
}

/// Agrupa as mudanças de um commit por crate; `ignore` some. Caminho sem regra é erro.
pub fn plan_commit(map: &SyncMap, changes: &[Change]) -> Result<BTreeMap<String, CratePlan>> {
    let mut plans: BTreeMap<String, CratePlan> = BTreeMap::new();
    for change in changes {
        let path = change.path();
        let Some(rule) = map.classify(path) else {
            bail!("sem regra em sync/map.toml: {path}");
        };
        let Some(krate) = &rule.krate else { continue };
        let plan = plans.entry(krate.clone()).or_default();
        match (rule.action, change) {
            (Action::Port, _) => plan.port.push(path.to_owned()),
            (Action::Vendor, Change::Upsert(_)) => plan.vendor.push(path.to_owned()),
            (Action::Vendor, Change::Deleted(_)) => plan.vendor_deleted.push(path.to_owned()),
            (Action::Ignore, _) => {}
        }
    }
    Ok(plans)
}

/// Caminho do upstream → arquivos Rust que o declaram com `//! upstream: <caminho>`.
pub fn rust_modules(crates_dir: &Path) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let mut index: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut stack = vec![crates_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if !dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(&dir).with_context(|| format!("lendo {}", dir.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("lendo {}", path.display()))?;
                for upstream in text
                    .lines()
                    .filter_map(|l| l.strip_prefix("//! upstream: "))
                {
                    index
                        .entry(upstream.trim().to_owned())
                        .or_default()
                        .push(path.clone());
                }
            }
        }
    }
    Ok(index)
}

/// Gera as tarefas de um commit para as crates registradas em `UPSTREAM.toml`.
struct Generator<'a> {
    git: &'a Git,
    map: &'a SyncMap,
    state: &'a State,
    modules: &'a BTreeMap<String, Vec<PathBuf>>,
    root: &'a Path,
    pending_dir: &'a Path,
}

impl Generator<'_> {
    /// Devolve quantas tarefas criou; `seq` avança uma por tarefa.
    fn commit(&self, sha: &str, seq: &mut u32) -> Result<usize> {
        let plans = plan_commit(self.map, &self.git.changes(sha)?)?;
        let subject = self.git.subject(sha)?;
        let mut files: Vec<(PathBuf, String)> = Vec::new();
        let mut next_seq = *seq;
        for (krate, plan) in plans {
            // Crate ainda não registrada: ela nasce com baseline posterior a este commit.
            if !self.state.crates.contains_key(&krate) {
                continue;
            }
            let diff = if plan.port.is_empty() {
                String::new()
            } else {
                let files: Vec<&str> = plan.port.iter().map(String::as_str).collect();
                self.git.diff(sha, &files)?
            };
            let meta = TaskMeta {
                seq: next_seq,
                sha: sha.to_owned(),
                krate,
                subject: subject.clone(),
                port: plan.port,
                vendor: plan.vendor,
                vendor_deleted: plan.vendor_deleted,
            };
            let body = self.body(&meta, &diff)?;
            files.push((
                self.pending_dir.join(meta.file_name()),
                task::render(&meta, &body)?,
            ));
            next_seq += 1;
        }
        // Tudo em memória antes de gravar; se uma gravação falhar, desfaz as do commit.
        let mut written: Vec<&Path> = Vec::new();
        for (path, contents) in &files {
            if let Err(e) = std::fs::write(path, contents) {
                for done in &written {
                    let _ = std::fs::remove_file(done);
                }
                return Err(e).with_context(|| format!("gravando {}", path.display()));
            }
            written.push(path);
        }
        *seq = next_seq;
        let created = files.len();
        Ok(created)
    }

    fn body(&self, meta: &TaskMeta, diff: &str) -> Result<String> {
        let sha7 = short(&meta.sha);
        let name = meta.file_name();
        let mut body = String::with_capacity(diff.len() + 1024);
        writeln!(body, "# {} (upstream {sha7})\n", meta.subject)?;
        writeln!(body, "Crate `{}` · commit `{}`\n", meta.krate, meta.sha)?;
        if !meta.port.is_empty() {
            writeln!(body, "## Portar\n")?;
            for path in &meta.port {
                match self.modules.get(path) {
                    Some(files) => {
                        for file in files {
                            let rel = file.strip_prefix(self.root).unwrap_or(file);
                            writeln!(body, "- `{path}` → `{}`", rel.display())?;
                        }
                    }
                    None => writeln!(
                        body,
                        "- `{path}` → **sem módulo Rust**: criar o módulo com `//! upstream: {path}`"
                    )?,
                }
            }
            writeln!(body)?;
        }
        if !meta.vendor.is_empty() || !meta.vendor_deleted.is_empty() {
            writeln!(
                body,
                "## Vendor\n\nRodar `cargo xtask task apply sync/pending/{name}`:\n"
            )?;
            for path in &meta.vendor {
                writeln!(body, "- `{path}`")?;
            }
            for path in &meta.vendor_deleted {
                writeln!(body, "- `{path}` (removido)")?;
            }
            writeln!(body)?;
        }
        writeln!(body, "## Fechar\n")?;
        writeln!(body, "1. `cargo xtask ci` verde")?;
        writeln!(body, "2. `cargo xtask task done sync/pending/{name}`")?;
        writeln!(
            body,
            "3. commit `sync({}): {} (upstream {sha7})`",
            meta.krate, meta.subject
        )?;
        if !diff.is_empty() {
            writeln!(
                body,
                "\n## Diff (só os arquivos a portar)\n\n~~~diff\n{diff}~~~"
            )?;
        }
        Ok(body)
    }
}

pub fn run(paths: &Paths, dry_run: bool) -> Result<()> {
    let mut state = State::load(&paths.upstream_toml())?;
    let map = SyncMap::load(&paths.map_toml())?;
    let git = git::open_upstream(&paths.upstream_clone(), &state.upstream.repo)?;
    git.fetch()?;
    let target = git.rev_parse("origin/master")?;
    let commits = git.commits_between(&state.upstream.cursor, &target)?;
    println!(
        "{} commits novos ({}..{})",
        commits.len(),
        short(&state.upstream.cursor),
        short(&target)
    );
    if dry_run {
        return report(&git, &map, &state, &commits);
    }

    let pending_dir = paths.pending_dir();
    std::fs::create_dir_all(&pending_dir)
        .with_context(|| format!("criando {}", pending_dir.display()))?;
    let modules = rust_modules(&paths.crates_dir())?;
    let mut seq = task::pending(&pending_dir)?
        .last()
        .map_or(1, |(meta, _)| meta.seq + 1);
    let mut created = 0;
    let mut outcome = Ok(());
    {
        let generator = Generator {
            git: &git,
            map: &map,
            state: &state,
            modules: &modules,
            root: paths.root(),
            pending_dir: &pending_dir,
        };
        let mut cursor = None;
        for sha in &commits {
            match generator.commit(sha, &mut seq) {
                Ok(n) => {
                    created += n;
                    cursor = Some(sha);
                }
                Err(e) => {
                    outcome = Err(e.context(format!("no commit {}", short(sha))));
                    break;
                }
            }
        }
        if let Some(sha) = cursor {
            state.upstream.cursor.clone_from(sha);
        }
    }
    // Grava o progresso mesmo se um commit falhou: a próxima execução continua dele.
    // Se o synced não puder ser recalculado, fica o valor anterior.
    let synced = task::pending(&pending_dir)
        .and_then(|pending| task::recompute_synced(&git, &state.upstream.cursor, &pending));
    let mut synced_err = None;
    match synced {
        Ok(sha) => state.upstream.synced = sha,
        Err(e) => synced_err = Some(e.context("recalculando o synced")),
    }
    let save_err = state.save(&paths.upstream_toml()).err();
    println!(
        "{created} tarefas novas em sync/pending/ · cursor {} · synced {}",
        short(&state.upstream.cursor),
        short(&state.upstream.synced)
    );
    // Prioridade do erro devolvido: o do commit, depois o da gravação, depois o do synced.
    let mut errors = outcome.err().into_iter().chain(save_err).chain(synced_err);
    let Some(first) = errors.next() else {
        return Ok(());
    };
    let others: Vec<String> = errors.map(|e| format!("{e:#}")).collect();
    if others.is_empty() {
        Err(first)
    } else {
        Err(first.context(format!("outras falhas: {}", others.join("; "))))
    }
}

/// `--dry-run`: o que cada commit pede, por crate, sem gravar nada.
fn report(git: &Git, map: &SyncMap, state: &State, commits: &[String]) -> Result<()> {
    let mut per_crate: BTreeMap<String, usize> = BTreeMap::new();
    let mut in_scope = 0;
    for sha in commits {
        let plans = plan_commit(map, &git.changes(sha)?)
            .with_context(|| format!("no commit {}", short(sha)))?;
        if plans.is_empty() {
            continue;
        }
        in_scope += 1;
        println!("{} {}", short(sha), git.subject(sha)?);
        for (krate, plan) in &plans {
            let note = if state.crates.contains_key(krate) {
                ""
            } else {
                "  (crate não registrada: sem tarefa)"
            };
            println!(
                "    {krate}: {} port, {} vendor{note}",
                plan.port.len(),
                plan.vendor.len() + plan.vendor_deleted.len()
            );
            *per_crate.entry(krate.clone()).or_default() += 1;
        }
    }
    println!("\n{in_scope} de {} commits tocam o escopo", commits.len());
    for (krate, n) in per_crate {
        println!("{n:>5}  {krate}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{CrateState, CrateStatus};
    use crate::testutil::{FIXTURE_MAP, Fixture};

    #[test]
    fn plan_agrupa_por_crate_e_descarta_ignore() {
        let map = SyncMap::parse(FIXTURE_MAP).unwrap();
        let changes = vec![
            Change::Upsert("keep/a.cpp".into()),
            Change::Upsert("shaders/n.comp".into()),
            Change::Deleted("shaders/s.comp".into()),
            Change::Upsert("skip/x.md".into()),
        ];
        let plans = plan_commit(&map, &changes).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(
            plans["k"],
            CratePlan {
                port: vec!["keep/a.cpp".into()],
                vendor: vec!["shaders/n.comp".into()],
                vendor_deleted: vec!["shaders/s.comp".into()],
            }
        );
    }

    #[test]
    fn plan_falha_em_caminho_sem_regra() {
        let map = SyncMap::parse(FIXTURE_MAP).unwrap();
        let Err(e) = plan_commit(&map, &[Change::Upsert("novo/x.c".into())]) else {
            panic!("devia falhar")
        };
        assert!(e.to_string().contains("novo/x.c"));
    }

    #[test]
    fn indice_de_modulos_le_os_marcadores() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("k/src/a.rs");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(
            &file,
            "//! upstream: keep/a.cpp\n//! upstream: keep/a.h\nfn x() {}\n",
        )
        .unwrap();
        let index = rust_modules(dir.path()).unwrap();
        assert_eq!(index["keep/a.cpp"], vec![file.clone()]);
        assert_eq!(index["keep/a.h"], vec![file]);
    }

    #[test]
    fn sync_gera_tarefa_so_para_crate_registrada() {
        let fx = Fixture::new();
        fx.upstream.write("keep/a.cpp", "int a = 2;\n");
        fx.upstream.write("shaders/t.comp", "void main() { }\n");
        fx.upstream.write("other/o.cpp", "int o;\n");
        let c1 = fx.upstream.commit("muda a, cria t e o");
        fx.upstream.write("skip/x.md", "y\n");
        let c2 = fx.upstream.commit("só docs");
        let paths = fx.paths();

        run(&paths, false).unwrap();

        let tasks = task::pending(&paths.pending_dir()).unwrap();
        assert_eq!(tasks.len(), 1);
        let (meta, file) = &tasks[0];
        assert_eq!(
            (meta.seq, meta.sha.as_str(), meta.krate.as_str()),
            (1, c1.as_str(), "k")
        );
        assert_eq!(meta.port, vec!["keep/a.cpp"]);
        assert_eq!(meta.vendor, vec!["shaders/t.comp"]);
        let text = std::fs::read_to_string(file).unwrap();
        assert!(text.contains("+int a = 2;"));
        assert!(text.contains("sem módulo Rust"));
        let state = State::load(&paths.upstream_toml()).unwrap();
        assert_eq!(state.upstream.cursor, c2);
        assert_eq!(state.upstream.synced, fx.base);
    }

    #[test]
    fn sync_para_no_commit_sem_regra_e_guarda_o_progresso() {
        let fx = Fixture::new();
        fx.upstream.write("keep/a.cpp", "int a = 3;\n");
        let c1 = fx.upstream.commit("ok");
        fx.upstream.write("novo/x.c", "x\n");
        fx.upstream.commit("caminho sem regra");
        let paths = fx.paths();

        let Err(e) = run(&paths, false) else {
            panic!("devia falhar")
        };
        assert!(format!("{e:#}").contains("novo/x.c"));
        assert_eq!(
            State::load(&paths.upstream_toml()).unwrap().upstream.cursor,
            c1
        );
    }

    #[test]
    fn sync_desfaz_tarefas_de_commit_que_falha_no_meio() {
        let fx = Fixture::new();
        let paths = fx.paths();
        let mut state = State::load(&paths.upstream_toml()).unwrap();
        state.crates.insert(
            "outra".into(),
            CrateState {
                baseline: fx.base.clone(),
                status: CrateStatus::Porting,
            },
        );
        state.save(&paths.upstream_toml()).unwrap();
        fx.upstream.write("keep/a.cpp", "int a = 4;\n");
        fx.upstream.write("other/o.cpp", "int o;\n");
        let c1 = fx.upstream.commit("toca duas crates");
        // Um diretório no lugar do segundo arquivo faz a gravação dele falhar.
        let pending_dir = paths.pending_dir();
        std::fs::create_dir_all(pending_dir.join(format!("00002-{}-outra.md", short(&c1))))
            .unwrap();

        assert!(run(&paths, false).is_err());

        assert!(task::pending(&pending_dir).unwrap().is_empty());
        assert!(
            !pending_dir
                .join(format!("00001-{}-k.md", short(&c1)))
                .exists()
        );
        let state = State::load(&paths.upstream_toml()).unwrap();
        assert_eq!(state.upstream.cursor, fx.base);
    }

    #[test]
    fn sync_grava_o_cursor_mesmo_se_o_synced_falha() {
        let fx = Fixture::new();
        let paths = fx.paths();
        // Tarefa válida de um commit que não existe: recalcular o synced falha.
        let ghost = TaskMeta {
            seq: 0,
            sha: "f".repeat(40),
            krate: "k".into(),
            subject: "fantasma".into(),
            port: vec![],
            vendor: vec![],
            vendor_deleted: vec![],
        };
        std::fs::create_dir_all(paths.pending_dir()).unwrap();
        std::fs::write(
            paths.pending_dir().join(ghost.file_name()),
            task::render(&ghost, "").unwrap(),
        )
        .unwrap();
        fx.upstream.write("keep/a.cpp", "int a = 5;\n");
        let c1 = fx.upstream.commit("bom");

        assert!(run(&paths, false).is_err());

        let state = State::load(&paths.upstream_toml()).unwrap();
        assert_eq!(state.upstream.cursor, c1);
        assert_eq!(state.upstream.synced, fx.base);
    }
}
