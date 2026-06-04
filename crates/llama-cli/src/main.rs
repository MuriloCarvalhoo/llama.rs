#![forbid(unsafe_code)]

use clap::Parser;

use llama_cli::args::Args;
use llama_cli::generate_text;

fn main() {
    let args = Args::parse();

    if !args.no_display_prompt {
        print!("{}", args.prompt);
    }

    match generate_text(&args) {
        Ok(text) => print!("{text}"),
        Err(e) => {
            eprintln!("Erro: {e}");
            std::process::exit(1);
        }
    }
}
