use std::path::PathBuf;

fn main() {
    // (arquivo, env var). O matvec já existia; os demais entram na Fase 1C.
    let shaders = [
        ("q8_0_matvec.comp", "Q8_0_MATVEC_SPV"),
        ("quantize_x.comp", "QUANTIZE_X_SPV"),
        ("q5_k_matvec.comp", "Q5_K_MATVEC_SPV"),
        ("q6_k_matvec.comp", "Q6_K_MATVEC_SPV"),
        ("rmsnorm.comp", "RMSNORM_SPV"),
        ("rope.comp", "ROPE_SPV"),
        ("attention.comp", "ATTENTION_SPV"),
        ("swiglu.comp", "SWIGLU_SPV"),
        ("add.comp", "ADD_SPV"),
    ];

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let compiler = shaderc::Compiler::new().expect("shaderc init falhou");
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    opts.set_optimization_level(shaderc::OptimizationLevel::Performance);

    for (file, env) in shaders {
        let src_path = PathBuf::from("shaders").join(file);
        println!("cargo:rerun-if-changed=shaders/{file}");
        let src = std::fs::read_to_string(&src_path)
            .unwrap_or_else(|_| panic!("nao encontrou {}", src_path.display()));
        let artifact = compiler
            .compile_into_spirv(
                &src,
                shaderc::ShaderKind::Compute,
                file,
                "main",
                Some(&opts),
            )
            .unwrap_or_else(|e| panic!("Falha ao compilar {file}: {e}"));
        let spv_path = out_dir.join(format!("{file}.spv"));
        std::fs::write(&spv_path, artifact.as_binary_u8())
            .unwrap_or_else(|e| panic!("falha ao escrever {}: {e}", spv_path.display()));
        println!("cargo:rustc-env={env}={}", spv_path.display());
    }
}
