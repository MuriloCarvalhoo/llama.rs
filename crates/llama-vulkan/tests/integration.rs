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
