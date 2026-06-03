use std::path::PathBuf;
use std::process::Command;

use crate::error::OracleError;
use crate::parse::parse_token_ids;

/// Executa os binários do llama.cpp compilado (o oráculo).
pub struct Oracle {
    bin_dir: PathBuf,
    model: PathBuf,
}

impl Oracle {
    pub fn new(bin_dir: impl Into<PathBuf>, model: impl Into<PathBuf>) -> Self {
        Self {
            bin_dir: bin_dir.into(),
            model: model.into(),
        }
    }

    /// Tokeniza `text` com o tokenizer do oráculo. Equivale a:
    /// `llama-tokenize -m <model> -p <text> --ids --log-disable`
    pub fn tokenize(&self, text: &str) -> Result<Vec<i64>, OracleError> {
        let out = self.run(
            "llama-tokenize",
            &[
                "-m",
                &self.model_arg(),
                "-p",
                text,
                "--ids",
                "--log-disable",
            ],
        )?;
        parse_token_ids(&out.stdout)
    }

    /// Gera `n_tokens` com sampling greedy determinístico; retorna o texto.
    /// Usa `llama-completion` (modo one-and-done com `-no-cnv`); no llama.cpp
    /// b9496 o `llama-cli` virou tool de chat interativo e não serve para isso.
    pub fn generate_greedy(&self, prompt: &str, n_tokens: u32) -> Result<String, OracleError> {
        let n = n_tokens.to_string();
        let out = self.run(
            "llama-completion",
            &[
                "-m",
                &self.model_arg(),
                "-p",
                prompt,
                "-n",
                &n,
                "--temp",
                "0",
                "--seed",
                "42",
                "-no-cnv",
                "--no-display-prompt",
                "--simple-io",
            ],
        )?;
        Ok(out.stdout)
    }

    /// Dump dos tensors intermediários do forward pass
    /// (saída do llama-eval-callback, stdout+stderr concatenados).
    pub fn dump_tensors(&self, prompt: &str) -> Result<String, OracleError> {
        let out = self.run(
            "llama-eval-callback",
            &["-m", &self.model_arg(), "-p", prompt, "-n", "1"],
        )?;
        let mut full = out.stdout;
        full.push_str(&out.stderr);
        Ok(full)
    }

    fn model_arg(&self) -> String {
        self.model.to_string_lossy().into_owned()
    }

    fn run(&self, bin: &str, args: &[&str]) -> Result<RunOutput, OracleError> {
        let path = self.bin_dir.join(bin);
        let out = Command::new(&path)
            .args(args)
            .output()
            .map_err(|e| OracleError::Io(bin.to_owned(), e))?;
        if !out.status.success() {
            return Err(OracleError::NonZero(
                bin.to_owned(),
                out.status.code().unwrap_or(-1),
            ));
        }
        Ok(RunOutput {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }
}

struct RunOutput {
    stdout: String,
    stderr: String,
}
