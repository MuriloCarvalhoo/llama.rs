//! Caminhos do repositório usados pelo xtask.

use std::path::{Path, PathBuf};

pub struct Paths {
    root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Raiz do workspace: o diretório pai de `xtask/`.
    pub fn from_manifest() -> Self {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        Self::new(manifest.parent().unwrap_or(manifest))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn upstream_toml(&self) -> PathBuf {
        self.root.join("UPSTREAM.toml")
    }

    pub fn map_toml(&self) -> PathBuf {
        self.root.join("sync/map.toml")
    }

    pub fn pending_dir(&self) -> PathBuf {
        self.root.join("sync/pending")
    }

    pub fn vendor_dir(&self) -> PathBuf {
        self.root.join("vendor/upstream")
    }

    pub fn crates_dir(&self) -> PathBuf {
        self.root.join("crates")
    }

    pub fn upstream_clone(&self) -> PathBuf {
        self.root.join(".upstream/llama.cpp")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raiz_e_o_workspace() {
        let root = Paths::from_manifest();
        assert!(root.root().join("rust-toolchain.toml").exists());
    }
}
