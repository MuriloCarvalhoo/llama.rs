//! Repositórios git temporários para os testes.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

use crate::paths::Paths;
use crate::state::{CrateState, CrateStatus, State, Upstream};

pub struct TempRepo {
    dir: TempDir,
}

impl TempRepo {
    pub fn init() -> Self {
        let repo = Self {
            dir: tempfile::tempdir().unwrap(),
        };
        repo.git(&["init", "-q", "-b", "master"]);
        repo
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    pub fn git(&self, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(self.path())
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args(["-c", "commit.gpgsign=false"])
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    pub fn write(&self, rel: &str, content: &str) {
        let path = self.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    pub fn remove(&self, rel: &str) {
        std::fs::remove_file(self.path().join(rel)).unwrap();
    }

    /// Commit de tudo; devolve o SHA.
    pub fn commit(&self, msg: &str) -> String {
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "-m", msg]);
        self.git(&["rev-parse", "HEAD"])
    }
}

pub const FIXTURE_MAP: &str = r#"
[[rule]]
path = "keep/*.cpp"
action = "port"
crate = "k"

[[rule]]
path = "shaders/**"
action = "vendor"
crate = "k"

[[rule]]
path = "other/*.cpp"
action = "port"
crate = "outra"

[[rule]]
path = "skip/**"
action = "ignore"
"#;

/// Um upstream falso e uma raiz do llama.rs com map, UPSTREAM.toml e a crate `k` registrada.
pub struct Fixture {
    pub upstream: TempRepo,
    pub root: TempDir,
    pub base: String,
}

impl Fixture {
    pub fn new() -> Self {
        let upstream = TempRepo::init();
        upstream.write("keep/a.cpp", "int a = 1;\n");
        upstream.write("shaders/s.comp", "void main() {}\n");
        upstream.write("skip/x.md", "x\n");
        let base = upstream.commit("base");

        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("sync")).unwrap();
        std::fs::write(root.path().join("sync/map.toml"), FIXTURE_MAP).unwrap();
        let mut crates = BTreeMap::new();
        crates.insert(
            "k".to_owned(),
            CrateState {
                baseline: base.clone(),
                status: CrateStatus::Porting,
            },
        );
        let state = State {
            upstream: Upstream {
                repo: upstream.path().to_str().unwrap().to_owned(),
                cursor: base.clone(),
                synced: base.clone(),
            },
            crates,
        };
        state.save(&root.path().join("UPSTREAM.toml")).unwrap();
        Self {
            upstream,
            root,
            base,
        }
    }

    pub fn paths(&self) -> Paths {
        Paths::new(self.root.path())
    }
}
