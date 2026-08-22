use std::path::PathBuf;

fn main() {
    // (arquivo, env var). O matvec já existia; os demais entram na Fase 1C.
    let shaders = [
        ("q8_0_matvec.comp", "Q8_0_MATVEC_SPV"),
        ("quantize_x.comp", "QUANTIZE_X_SPV"),
        ("q5_k_matvec.comp", "Q5_K_MATVEC_SPV"),
        ("q6_k_matvec.comp", "Q6_K_MATVEC_SPV"),
        ("q4_k_matvec.comp", "Q4_K_MATVEC_SPV"),
        ("rmsnorm.comp", "RMSNORM_SPV"),
        ("norm_fused.comp", "NORM_FUSED_SPV"),
        ("norm_p2.comp", "NORM_P2_SPV"),
        ("rope.comp", "ROPE_SPV"),
        ("rope_kv.comp", "ROPE_KV_SPV"),
        ("attention.comp", "ATTENTION_SPV"),
        // Atenção com o KV fatiado entre workgroups (contexto longo) + a redução.
        ("attention_split.comp", "ATTENTION_SPLIT_SPV"),
        ("attn_reduce.comp", "ATTN_REDUCE_SPV"),
        ("swiglu_quant.comp", "SWIGLU_QUANT_SPV"),
        ("add.comp", "ADD_SPV"),
        // Camadas de atenção linear (qwen35) — ver docs/qwen35-arquitetura.md.
        ("delta_net.comp", "DELTA_NET_SPV"),
        ("dn_conv.comp", "DN_CONV_SPV"),
        ("dn_gates.comp", "DN_GATES_SPV"),
        ("dn_norm.comp", "DN_NORM_SPV"),
        ("dn_l2_qk.comp", "DN_L2_QK_SPV"),
        ("gate_quant.comp", "GATE_QUANT_SPV"),
        // GEMM com tiling em LDS para o prefill (experimental, atrás de knob).
        ("mul_mm.comp", "MUL_MM_SPV"),
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
