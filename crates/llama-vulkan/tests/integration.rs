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
