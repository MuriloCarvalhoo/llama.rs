//! Testes de integracao Vulkan -- exigem duas MI50 reais.
//! Pulam automaticamente se nenhum device Vulkan AMD estiver disponivel.

use llama_vulkan::{VulkanContext, VulkanDevice};

#[test]
fn detects_at_least_one_amd_device() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let devices = ctx.amd_compute_devices();
    if devices.is_empty() {
        eprintln!("Nenhum device AMD");
        return;
    }
    assert!(!devices.is_empty());
    for d in devices {
        eprintln!("  {} -- subgroupSize={}", d.name(), d.subgroup_size());
    }
}

#[test]
fn detects_two_mi50_for_dual_gpu() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let devices = ctx.amd_compute_devices();
    if devices.len() < 2 {
        eprintln!("Menos de 2 AMD -- pulando");
        return;
    }
    for d in devices {
        assert_eq!(d.subgroup_size(), 64, "MI50 deve ter wave64");
    }
}

#[test]
fn creates_logical_device_for_first_amd() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let devices = ctx.amd_compute_devices();
    if devices.is_empty() {
        eprintln!("Nenhum device AMD -- pulando");
        return;
    }
    let phys = &devices[0];
    let dev = VulkanDevice::create(&ctx, phys);
    assert!(dev.is_ok(), "Falha ao criar device logico: {:?}", dev.err());
    eprintln!("Device logico criado para {}", phys.name());
}

#[test]
fn sub_allocator_chunks_independentes_para_tensores_grandes() {
    use llama_vulkan::alloc::{GpuAllocator, MAX_CHUNK_BYTES};

    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("Nenhum device AMD -- pulando");
        return;
    }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    let mut alloc = GpuAllocator::new(&ctx, &phys[0], &dev, 3 * MAX_CHUNK_BYTES)
        .expect("GpuAllocator::new falhou");

    // 100MB e 200MB cabem no mesmo chunk de 1.5GB
    let a = alloc
        .alloc(dev.as_device(), 100_000_000)
        .expect("alloc 100MB falhou");
    let b = alloc
        .alloc(dev.as_device(), 200_000_000)
        .expect("alloc 200MB falhou");
    assert_eq!(
        a.chunk_idx, b.chunk_idx,
        "duas alocacoes pequenas devem estar no mesmo chunk"
    );
    assert_eq!(a.offset, 0, "primeira alocacao deve comecar no offset 0");
    assert!(b.offset > 0, "segunda alocacao deve ter offset positivo");

    eprintln!(
        "Alocacao A: chunk={} offset={} size={}",
        a.chunk_idx, a.offset, a.size
    );
    eprintln!(
        "Alocacao B: chunk={} offset={} size={}",
        b.chunk_idx, b.offset, b.size
    );

    alloc.cleanup(dev.as_device());
}

#[test]
fn upload_tensor_q8_0_para_vram() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Vulkan nao disponivel: {e}");
            return;
        }
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        eprintln!("Nenhum device AMD -- pulando");
        return;
    }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    let n_out = 64usize;
    let n_in = 128usize;
    // Q8_0: cada bloco de 32 elementos = 2 bytes (scale f16) + 32 bytes (quants) = 34 bytes
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;
    let bytes: Vec<u8> = (0..n_out * row_bytes).map(|i| (i % 256) as u8).collect();

    use llama_vulkan::tensor::GpuTensor;
    let tensor =
        GpuTensor::upload_q8_0(&ctx, &phys[0], &dev, &bytes, n_in, n_out).expect("upload falhou");
    assert_eq!(tensor.n_out, n_out);
    assert_eq!(tensor.n_in, n_in);
    assert_eq!(tensor.size_bytes, bytes.len() as u64);
    eprintln!("Upload OK: {}x{} Q8_0 ({} bytes)", n_out, n_in, bytes.len());
    tensor.destroy(dev.as_device());
}

#[test]
fn matmul_gpu_matches_cpu_reference() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() {
        return;
    }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    // n_in=32, n_out=4, n_blocks=1
    // row 0: scale=1.0f16, qs[0]=1, resto 0 -> y[0] = 1.0 * 1 * x[0] = 5.0
    // row 1: scale=2.0f16, qs[0]=1, resto 0 -> y[1] = 2.0 * 1 * x[0] = 10.0
    let n_in = 32usize;
    let n_out = 4usize;
    let row_bytes = 1 * 34; // 1 bloco
    let mut w_bytes = vec![0u8; n_out * row_bytes];

    // Escala f16 em little-endian: 1.0 = 0x3C00
    let f16_bytes_1_0: [u8; 2] = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    let f16_bytes_2_0: [u8; 2] = half::f16::from_f32(2.0).to_bits().to_le_bytes();
    w_bytes[0..2].copy_from_slice(&f16_bytes_1_0);
    w_bytes[2] = 1; // qs[0] = 1
    w_bytes[row_bytes..row_bytes + 2].copy_from_slice(&f16_bytes_2_0);
    w_bytes[row_bytes + 2] = 1; // qs[0] = 1

    let x_f32 = vec![5.0f32; n_in];

    use llama_vulkan::matmul::dispatch_q8_0_matvec;
    let y = dispatch_q8_0_matvec(&ctx, &phys[0], &dev, &w_bytes, &x_f32, n_in, n_out)
        .expect("matmul GPU falhou");

    assert_eq!(y.len(), n_out);
    assert!((y[0] - 5.0).abs() < 0.1, "y[0] esperado ~5.0, got {}", y[0]);
    assert!(
        (y[1] - 10.0).abs() < 0.1,
        "y[1] esperado ~10.0, got {}",
        y[1]
    );
    assert!(y[2].abs() < 0.1, "y[2] esperado ~0.0, got {}", y[2]);
    assert!(y[3].abs() < 0.1, "y[3] esperado ~0.0, got {}", y[3]);
    eprintln!("GPU matmul: y={:?}", y);
}
