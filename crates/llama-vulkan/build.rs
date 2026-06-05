use std::path::PathBuf;

fn main() {
    let shader_src = PathBuf::from("shaders/q8_0_matvec.comp");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let spv_path = out_dir.join("q8_0_matvec.spv");

    println!("cargo:rerun-if-changed=shaders/q8_0_matvec.comp");

    let compiler = shaderc::Compiler::new().expect("shaderc init falhou");
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    opts.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let src = std::fs::read_to_string(&shader_src)
        .unwrap_or_else(|_| panic!("nao encontrou {}", shader_src.display()));

    let artifact = compiler
        .compile_into_spirv(
            &src,
            shaderc::ShaderKind::Compute,
            "q8_0_matvec.comp",
            "main",
            Some(&opts),
        )
        .unwrap_or_else(|e| panic!("Falha ao compilar shader: {e}"));

    std::fs::write(&spv_path, artifact.as_binary_u8()).unwrap();
    println!("cargo:rustc-env=Q8_0_MATVEC_SPV={}", spv_path.display());
}
