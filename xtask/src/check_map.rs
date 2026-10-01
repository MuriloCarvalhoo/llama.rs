//! `cargo xtask check-map`: todo arquivo do upstream precisa casar uma regra.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::git;
use crate::map::{Action, SyncMap};
use crate::paths::Paths;
use crate::state::State;

/// Arquivos sem regra.
pub fn unmatched<'f>(map: &SyncMap, files: &'f [String]) -> Vec<&'f str> {
    files
        .iter()
        .map(String::as_str)
        .filter(|f| map.classify(f).is_none())
        .collect()
}

/// Quantos arquivos caem em cada (ação, crate).
pub fn summarize<'m>(
    map: &'m SyncMap,
    files: &[String],
) -> BTreeMap<(Action, Option<&'m str>), usize> {
    let mut counts = BTreeMap::new();
    for rule in files.iter().filter_map(|f| map.classify(f)) {
        *counts
            .entry((rule.action, rule.krate.as_deref()))
            .or_insert(0) += 1;
    }
    counts
}

pub fn run(paths: &Paths, rev: Option<&str>) -> Result<()> {
    let state = State::load(&paths.upstream_toml())?;
    let map = SyncMap::load(&paths.map_toml())?;
    let git = git::open_upstream(&paths.upstream_clone(), &state.upstream.repo)?;
    let rev = rev.unwrap_or(&state.upstream.cursor);
    let files = git.ls_tree(rev)?;
    let missing = unmatched(&map, &files);
    if !missing.is_empty() {
        for f in &missing {
            eprintln!("sem regra: {f}");
        }
        bail!(
            "{} de {} arquivos sem regra em sync/map.toml",
            missing.len(),
            files.len()
        );
    }
    for ((action, krate), n) in summarize(&map, &files) {
        println!("{n:>5}  {action:?} {}", krate.unwrap_or(""));
    }
    println!(
        "ok: {} arquivos de {} classificados",
        files.len(),
        git::short(rev)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAP: &str = "[[rule]]\npath = \"src/*.cpp\"\naction = \"port\"\ncrate = \"llama\"\n\n[[rule]]\npath = \"*\"\naction = \"ignore\"\n";

    fn files(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn lista_so_os_sem_regra() {
        let map = SyncMap::parse(MAP).unwrap();
        let files = files(&["src/a.cpp", "README.md", "docs/x.md"]);
        assert_eq!(unmatched(&map, &files), vec!["docs/x.md"]);
    }

    #[test]
    fn conta_por_acao_e_crate() {
        let map = SyncMap::parse(MAP).unwrap();
        let files = files(&["src/a.cpp", "src/b.cpp", "README.md"]);
        let counts = summarize(&map, &files);
        assert_eq!(counts[&(Action::Port, Some("llama"))], 2);
        assert_eq!(counts[&(Action::Ignore, None)], 1);
    }
}
