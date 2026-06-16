//! Spike Fase 8.3: GL_EXT_integer_dot_product roda em RADV/gfx906?
//! a = [1,2,3,4] (i8 packed LE), b = [5,6,7,8] → 1*5+2*6+3*7+4*8 = 70.
use llama_vulkan::{ResidentForward, VulkanContext};

#[test]
fn dot4_packed_roda_em_radv() {
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let a: i32 = i32::from_le_bytes([1, 2, 3, 4]);
    let b: i32 = i32::from_le_bytes([5, 6, 7, 8]);
    let r = ResidentForward::dbg_dot4_probe(&ctx, a, b)
        .expect("se a extensão não compilar/rodar, isto falha — resultado do spike");
    assert_eq!(r, 70, "dotPacked4x8 deve somar 70");
}
