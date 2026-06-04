#![forbid(unsafe_code)]

use clap::Parser;
use llama_cli::args::Args;

fn main() {
    let args = Args::parse();
    if let Err(e) = llama_cli::runner::run(&args) {
        eprintln!("erro: {e}");
        std::process::exit(1);
    }
}
