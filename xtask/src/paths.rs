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
