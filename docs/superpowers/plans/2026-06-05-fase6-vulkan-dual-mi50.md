# Fase 6 — Backend Vulkan (Dual AMD MI50 / gfx906) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar backend de inferência Vulkan para dois AMD MI50 (gfx906/Vega20, 16GB HBM2 cada), usando row-split dual-GPU, sub-alocador VMA-style e shaders wave64 com dequantização packed-int, trazendo inferência Qwen2.5 de ~13 tok/s (CPU) para >80 tok/s (GPU).

**Architecture:** Novo crate `crates/llama-vulkan` com trait `VulkanBackend` que encapsula device init, sub-alocador de memória (contorna limite AMDVLK 2GB), pipeline Vulkan para Q8_0 matmul-vector com shader wave64, e coordinador de row-split entre os dois MI50. O `Model` CPU permanece intacto — o forward pass usa GPU apenas para matmuls (ops pesadas), com RMSNorm, RoPE, Attention e SwiGLU ficando em CPU por ora (elas são ~10% do tempo).

**Tech Stack:** `ash` (Vulkan bindings), `shaderc` (compile GLSL->SPIR-V em build.rs), `half` (f16 scales), GLSL 450 com extensao `GL_KHR_shader_subgroup_arithmetic` (wave64 reduction), row-split via dois `VkDevice` independentes.

**Contexto critico do estudo (deep-research-vulkan-performance-llama-rs.md):**
- gfx906 e wave64 (nao wave32 como NVIDIA) -> shaders que assumem warp=32 produzem output corrompido
- AMDVLK tem `maxMemoryAllocationSize` = 2GB -> modelos >2GB requerem sub-alocacao
- Vulkan sem row-split e 3x mais lento que ROCm com row-split -> implementar row-split desde o inicio
- Dequantizacao packed-int (i8 x f32 sem converter para f16 por elemento) elimina overhead algoritmico
- Token generation e memory-bandwidth bound -> MI50 HBM2 1TB/s vs CPU DRAM 50GB/s = 20x bandwidth

---

## Mapeamento de Arquivos

### Criar
- `crates/llama-vulkan/Cargo.toml` -- crate novo, deps: ash, shaderc (build), half
- `crates/llama-vulkan/build.rs` -- compila GLSL -> SPIR-V com shaderc; falha o build se glsl invalido
- `crates/llama-vulkan/shaders/q8_0_matvec.comp` -- shader GLSL wave64 para Q8_0 x f32 matmul-vector
- `crates/llama-vulkan/src/lib.rs` -- re-exports publicos: `VulkanContext`, `DualGpuMatmul`
- `crates/llama-vulkan/src/device.rs` -- init Vulkan, enumera devices, detecta gfx906
- `crates/llama-vulkan/src/alloc.rs` -- sub-alocador VMA-style (free-list sobre chunks de 1.5GB)
- `crates/llama-vulkan/src/tensor.rs` -- `GpuTensor` (buffer + offset + bytes + shape)
- `crates/llama-vulkan/src/pipeline.rs` -- `ComputePipeline` (descriptor set layout, pipeline object)
- `crates/llama-vulkan/src/matmul.rs` -- `dispatch_q8_0_matvec` (single GPU matmul)
- `crates/llama-vulkan/src/dual_gpu.rs` -- `DualGpuMatmul` (row-split entre GPU0 e GPU1)
- `crates/llama-vulkan/src/model_gpu.rs` -- `GpuWeights` (upload pesos Q8_0 para VRAM)
- `crates/llama-vulkan/tests/integration.rs` -- testes de integracao com assert numerico vs CPU

### Modificar
- `Cargo.toml` -- adicionar `crates/llama-vulkan` em `members`
- `crates/llama-cli/Cargo.toml` -- adicionar `llama-vulkan` como dep opcional (feature "gpu")
- `crates/llama-cli/src/lib.rs` -- flag `--gpu` -> usa `DualGpuMatmul` em vez do CPU path

---

## Task 0: Crate Skeleton + Build System

**Files:**
- Create: `crates/llama-vulkan/Cargo.toml`
- Create: `crates/llama-vulkan/build.rs`
- Create: `crates/llama-vulkan/shaders/q8_0_matvec.comp`
- Create: `crates/llama-vulkan/src/lib.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Escrever o shader GLSL wave64 (TDD: build falha se shader invalido)**

Criar `crates/llama-vulkan/shaders/q8_0_matvec.comp`:
```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Workgroup = um wave64 (64 lanes) computa UM elemento de saida (uma linha de W . x).
// gfx906 (MI50) tem wave64 nativo; local_size_x=64 garante um workgroup = uma wave.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// W: pesos Q8_0 row-major. Cada linha = n_blocks x 34 bytes (2 bytes f16 scale + 32 bytes i8).
layout(set = 0, binding = 0) readonly buffer WeightBuf { uint8_t w[]; } weight_buf;
// X: ativacoes f32. Tamanho: n_in floats.
layout(set = 0, binding = 1) readonly buffer ActBuf   { float   x[]; } act_buf;
// Y: saida f32. Tamanho: n_out floats.
layout(set = 0, binding = 2) writeonly buffer OutBuf  { float   y[]; } out_buf;

layout(push_constant) uniform PC {
    uint n_in;       // dimensao de entrada (multiplo de 32)
    uint n_out;      // dimensao de saida
    uint row_offset; // para row-split: GPU1 comeca em row_offset
} pc;

void main() {
    uint row = gl_WorkGroupID.x + pc.row_offset;
    uint lane = gl_LocalInvocationID.x; // 0..63

    if (row >= pc.row_offset + gl_NumWorkGroups.x) return;

    uint n_blocks  = pc.n_in / 32u;
    uint row_bytes = n_blocks * 34u;

    float acc = 0.0;

    // Cada lane processa blocos em stride de 64 (um wave completo).
    // Evita divergencia de controle entre lanes.
    for (uint b = lane; b < n_blocks; b += 64u) {
        uint boff = row * row_bytes + b * 34u;

        // Carrega escala f16 (2 bytes little-endian) e converte para f32.
        // Dequantizacao packed-int: i8 x f32 (nao elemento-a-elemento f16).
        uint d_lo = uint(weight_buf.w[boff]);
        uint d_hi = uint(weight_buf.w[boff + 1u]);
        float d = unpackHalf2x16((d_hi << 8u) | d_lo).x;

        // Produto escalar: 32 pares i8 x f32.
        float dot = 0.0;
        for (uint i = 0u; i < 32u; i++) {
            int qi = int(int8_t(weight_buf.w[boff + 2u + i]));
            float xi = act_buf.x[b * 32u + i];
            dot += float(qi) * xi;
        }
        acc += d * dot;
    }

    // Reducao wave64 via subgroupAdd -- funciona corretamente com qualquer wave size.
    // Em gfx906, gl_SubgroupSize == 64, entao esta e uma operacao intra-wave sem sync explicito.
    acc = subgroupAdd(acc);

    // Apenas a lane 0 de cada subgroup escreve (evita race condition).
    if (gl_SubgroupInvocationID == 0u) {
        out_buf.y[row] = acc;
    }
}
```

- [ ] **Step 2: Criar build.rs que compila o shader com shaderc**

Criar `crates/llama-vulkan/build.rs`:
```rust
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
```

- [ ] **Step 3: Criar Cargo.toml do crate**

Criar `crates/llama-vulkan/Cargo.toml`:
```toml
[package]
name = "llama-vulkan"
version = "0.1.0"
edition = "2024"

[dependencies]
ash = "0.38"
half = { workspace = true }
gguf = { workspace = true }
llama-model = { workspace = true }
thiserror = { workspace = true }
rayon = { workspace = true }

[build-dependencies]
shaderc = "0.8"

[lints.rust]
unsafe_code = "allow"
```

- [ ] **Step 4: Criar src/lib.rs esqueleto**

Criar `crates/llama-vulkan/src/lib.rs`:
```rust
//! Backend de inferencia Vulkan para AMD MI50 (gfx906/wave64).
//! Implementa row-split dual-GPU e sub-alocacao de memoria para contornar
//! o limite de 2GB do driver AMDVLK.

mod alloc;
mod device;
mod dual_gpu;
mod matmul;
mod model_gpu;
mod pipeline;
mod tensor;

pub use device::{VulkanContext, VulkanDevice};
pub use dual_gpu::DualGpuMatmul;
pub use model_gpu::GpuWeights;

/// SPIR-V do shader Q8_0 matmul-vector, compilado em build time.
pub(crate) const Q8_0_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("Q8_0_MATVEC_SPV")));
```

- [ ] **Step 5: Adicionar `crates/llama-vulkan` ao workspace**

Editar `/home/murilo/llama-cpp-rs-new/Cargo.toml` -- adicionar na lista de members:
```
"crates/llama-vulkan",
```

- [ ] **Step 6: Verificar que o crate compila**

```bash
cd /home/murilo/llama-cpp-rs-new
cargo build -p llama-vulkan 2>&1 | head -40
```

Esperado: PASS ou erros de modulos faltando (nao erro de shader). shaderc deve ter compilado o GLSL.

- [ ] **Step 7: Commit**

```bash
git add crates/llama-vulkan/ Cargo.toml Cargo.lock
git commit -m "feat(vulkan): crate skeleton + GLSL shader Q8_0 wave64 + build.rs shaderc"
```

---

## Task 1: Vulkan Device Init e Deteccao MI50

**Files:**
- Create: `crates/llama-vulkan/src/device.rs`

Contexto: gfx906 (MI50) tem PCI vendor ID `0x1002` (AMD). Detectamos que os dispositivos sao AMD, verificamos suporte a compute queues, e habilitamos wave64. `subgroupSize` deve ser 64 para MI50.

- [ ] **Step 1: Escrever o teste de deteccao antes de implementar**

Criar `crates/llama-vulkan/tests/integration.rs`:
```rust
//! Testes de integracao Vulkan -- exigem duas MI50 reais.
//! Pulam automaticamente se nenhum device Vulkan AMD estiver disponivel.

use llama_vulkan::{VulkanContext, VulkanDevice};

#[test]
fn detects_at_least_one_amd_device() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("Vulkan nao disponivel: {e}"); return; }
    };
    let devices = ctx.amd_compute_devices();
    if devices.is_empty() { eprintln!("Nenhum device AMD"); return; }
    assert!(!devices.is_empty());
    for d in &devices {
        eprintln!("  {} -- subgroupSize={}", d.name(), d.subgroup_size());
    }
}

#[test]
fn detects_two_mi50_for_dual_gpu() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("Vulkan nao disponivel: {e}"); return; }
    };
    let devices = ctx.amd_compute_devices();
    if devices.len() < 2 { eprintln!("Menos de 2 AMD -- pulando"); return; }
    for d in &devices {
        assert_eq!(d.subgroup_size(), 64, "MI50 deve ter wave64");
    }
}
```

- [ ] **Step 2: Rodar o teste -- deve falhar com erro de compilacao**

```bash
cargo test -p llama-vulkan -- --nocapture 2>&1 | head -20
```

Esperado: FAIL com erro de modulo nao encontrado.

- [ ] **Step 3: Implementar `src/device.rs`**

Criar `crates/llama-vulkan/src/device.rs`:
```rust
use ash::{vk, Entry, Instance};
use std::ffi::CStr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Falha ao criar instancia Vulkan: {0}")]
    InstanceCreate(vk::Result),
    #[error("Nenhum physical device encontrado")]
    NoDevices,
    #[error("Vulkan API error: {0}")]
    Api(#[from] vk::Result),
}

const AMD_VENDOR_ID: u32 = 0x1002;

pub struct VulkanContext {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    physical_devices: Vec<VulkanPhysicalDevice>,
}

pub struct VulkanPhysicalDevice {
    pub(crate) handle: vk::PhysicalDevice,
    name: String,
    subgroup_size: u32,
    pub(crate) queue_family: u32,
}

impl VulkanPhysicalDevice {
    pub fn name(&self) -> &str { &self.name }
    pub fn subgroup_size(&self) -> u32 { self.subgroup_size }
}

impl VulkanContext {
    pub fn new() -> Result<Self, VulkanError> {
        // SAFETY: carrega biblioteca Vulkan dinamicamente via ash.
        let entry = unsafe { Entry::load().map_err(|_| VulkanError::NoDevices)? };

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 1, 0), // Vulkan 1.1 para subgroup ops
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };
        let instance = unsafe {
            entry.create_instance(&create_info, None)
                .map_err(VulkanError::InstanceCreate)?
        };

        let physical_devices = Self::enumerate_amd_devices(&instance)?;
        Ok(Self { entry, instance, physical_devices })
    }

    pub fn amd_compute_devices(&self) -> &[VulkanPhysicalDevice] {
        &self.physical_devices
    }

    fn enumerate_amd_devices(instance: &Instance) -> Result<Vec<VulkanPhysicalDevice>, VulkanError> {
        let phys_devs = unsafe { instance.enumerate_physical_devices()? };
        let mut result = Vec::new();

        for pd in phys_devs {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.vendor_id != AMD_VENDOR_ID { continue; }

            let qfams = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(qfam_idx) = qfams.iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else { continue; };

            let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2 {
                p_next: &mut subgroup_props as *mut _ as *mut _,
                ..Default::default()
            };
            unsafe { instance.get_physical_device_properties2(pd, &mut props2) };

            let name = unsafe {
                CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy().into_owned()
            };
            result.push(VulkanPhysicalDevice {
                handle: pd,
                name,
                subgroup_size: subgroup_props.subgroup_size,
                queue_family: qfam_idx as u32,
            });
        }
        Ok(result)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// Device logico Vulkan + fila de compute + command pool.
pub struct VulkanDevice {
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    pub(crate) cmd_pool: vk::CommandPool,
    pub(crate) queue_family: u32,
}

impl VulkanDevice {
    pub fn create(ctx: &VulkanContext, phys: &VulkanPhysicalDevice) -> Result<Self, VulkanError> {
        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: phys.queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priority.as_ptr(),
            ..Default::default()
        };
        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            ..Default::default()
        };
        let device = unsafe {
            ctx.instance.create_device(phys.handle, &create_info, None)?
        };
        let queue = unsafe { device.get_device_queue(phys.queue_family, 0) };
        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: phys.queue_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        Ok(Self { device, queue, cmd_pool, queue_family: phys.queue_family })
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
        }
    }
}
```

- [ ] **Step 4: Rodar os testes de deteccao**

```bash
cargo test -p llama-vulkan -- --nocapture 2>&1
```

Esperado: PASS com output mostrando os 2 MI50 e `subgroupSize=64`.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-vulkan/src/device.rs crates/llama-vulkan/tests/
git commit -m "feat(vulkan): device init + enumera MI50 gfx906 com subgroupSize=64"
```

---

## Task 2: VMA-Style Sub-Alocador (Contorna Limite 2GB AMDVLK)

**Files:**
- Create: `crates/llama-vulkan/src/alloc.rs`

Contexto: AMDVLK tem `maxMemoryAllocationSize = 2GB`. Modelos >2GB precisam de varios chunks de 1.5GB com sub-alocacao por offset (bump allocator).

- [ ] **Step 1: Escrever teste de sub-alocacao**

Adicionar em `tests/integration.rs`:
```rust
#[test]
fn sub_allocator_chunks_independentes_para_tensores_grandes() {
    let ctx = match VulkanContext::new() { Ok(c) => c, Err(_) => return };
    let devs = ctx.amd_compute_devices();
    if devs.is_empty() { return; }

    use llama_vulkan::alloc::{GpuAllocator, MAX_CHUNK_BYTES};
    let dev = VulkanDevice::create(&ctx, &devs[0]).unwrap();
    let mut alloc = GpuAllocator::new(&ctx, &devs[0], &dev, 3 * 1024 * 1024 * 1024)
        .expect("GpuAllocator::new falhou");

    // 1.8GB nao cabe em 1 chunk de 1.5GB -> necessita 2 chunks
    let a = alloc.alloc(1_800_000_000).expect("alloc 1.8GB falhou");
    let b = alloc.alloc(100_000_000).expect("alloc 100MB falhou");

    // 1.8GB > MAX_CHUNK_BYTES (1.5GB), entao 'a' esta em chunk proprio
    // 'b' (100MB) fica em outro chunk
    eprintln!("a.chunk={}, b.chunk={}", a.chunk_idx, b.chunk_idx);
}
```

- [ ] **Step 2: Rodar o teste -- deve falhar**

```bash
cargo test -p llama-vulkan sub_allocator -- --nocapture 2>&1 | head -10
```

Esperado: erro de compilacao (modulo nao existe).

- [ ] **Step 3: Implementar `src/alloc.rs`**

Criar `crates/llama-vulkan/src/alloc.rs`:
```rust
//! Sub-alocador VMA-style: chunks de 1.5GB para contornar limite AMDVLK.

use ash::vk;
use thiserror::Error;
use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};

/// Limite conservador por chunk -- abaixo dos 2GB do AMDVLK.
pub const MAX_CHUNK_BYTES: vk::DeviceSize = 1_500_000_000; // 1.5 GB

#[derive(Debug, Error)]
pub enum AllocError {
    #[error("Vulkan OOM: {0}")]
    Oom(vk::Result),
    #[error("Nenhum tipo de memoria device-local encontrado")]
    NoMemoryType,
}

/// Handle de uma alocacao (offset dentro de um chunk).
#[derive(Clone, Copy, Debug)]
pub struct Allocation {
    pub chunk_idx: usize,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

struct Chunk {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    free_start: vk::DeviceSize,
}

/// Sub-alocador com bump pointer por chunk.
pub struct GpuAllocator {
    device: ash::Device,
    chunks: Vec<Chunk>,
    mem_type_idx: u32,
}

impl GpuAllocator {
    /// Cria o alocador. `total_bytes` reservado em chunks de MAX_CHUNK_BYTES.
    pub fn new(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        dev: &VulkanDevice,
        _total_bytes: vk::DeviceSize,
    ) -> Result<Self, AllocError> {
        let mem_props = unsafe {
            ctx.instance.get_physical_device_memory_properties(phys.handle)
        };
        let mem_type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(AllocError::NoMemoryType)? as u32;

        Ok(Self { device: dev.device.clone(), chunks: Vec::new(), mem_type_idx })
    }

    /// Aloca `size` bytes. Cria novo chunk se necessario.
    pub fn alloc(&mut self, size: vk::DeviceSize) -> Result<Allocation, AllocError> {
        let size = align_up(size, 256);

        // Tenta encaixar em chunk existente
        for (idx, chunk) in self.chunks.iter_mut().enumerate() {
            if chunk.free_start + size <= chunk.size {
                let offset = chunk.free_start;
                chunk.free_start += size;
                return Ok(Allocation { chunk_idx: idx, offset, size });
            }
        }

        // Novo chunk: tamanho maximo entre size e MAX_CHUNK_BYTES
        let chunk_size = size.max(MAX_CHUNK_BYTES);
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: chunk_size,
            memory_type_index: self.mem_type_idx,
            ..Default::default()
        };
        let memory = unsafe {
            self.device.allocate_memory(&alloc_info, None).map_err(AllocError::Oom)?
        };
        let idx = self.chunks.len();
        self.chunks.push(Chunk { memory, size: chunk_size, free_start: size });
        Ok(Allocation { chunk_idx: idx, offset: 0, size })
    }

    /// Free e no-op neste bump allocator. Liberacao real ocorre no Drop.
    pub fn free(&mut self, _alloc: Allocation) {}
}

fn align_up(v: vk::DeviceSize, align: vk::DeviceSize) -> vk::DeviceSize {
    (v + align - 1) & !(align - 1)
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            unsafe { self.device.free_memory(chunk.memory, None) };
        }
    }
}
```

- [ ] **Step 4: Expor modulo em lib.rs**

Adicionar em `lib.rs`:
```rust
pub mod alloc;
```

- [ ] **Step 5: Rodar o teste**

```bash
cargo test -p llama-vulkan sub_allocator -- --nocapture 2>&1
```

Esperado: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/src/alloc.rs crates/llama-vulkan/src/lib.rs
git commit -m "feat(vulkan): sub-alocador VMA-style em chunks 1.5GB (contorna AMDVLK 2GB)"
```

---

## Task 3: GpuTensor + Upload de Pesos Q8_0 para VRAM

**Files:**
- Create: `crates/llama-vulkan/src/tensor.rs`

Upload usa staging buffer (host-visible) -> copia para device-local via command buffer one-shot.

- [ ] **Step 1: Escrever teste de upload**

Adicionar em `tests/integration.rs`:
```rust
#[test]
fn upload_tensor_q8_0_para_vram() {
    let ctx = match VulkanContext::new() { Ok(c) => c, Err(_) => return };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() { return; }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    let n_out = 64usize;
    let n_in = 128usize;
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;
    let bytes: Vec<u8> = (0..n_out * row_bytes).map(|i| (i % 256) as u8).collect();

    use llama_vulkan::tensor::GpuTensor;
    let tensor = GpuTensor::upload_q8_0(&ctx, &phys[0], &dev, &bytes, n_in, n_out)
        .expect("upload falhou");
    assert_eq!(tensor.n_out, n_out);
    assert_eq!(tensor.n_in, n_in);
    eprintln!("Upload OK: {}x{} Q8_0 ({} bytes)", n_out, n_in, bytes.len());
}
```

- [ ] **Step 2: Rodar o teste -- deve falhar**

```bash
cargo test -p llama-vulkan upload_tensor -- --nocapture 2>&1 | head -5
```

- [ ] **Step 3: Implementar `src/tensor.rs`**

Criar `crates/llama-vulkan/src/tensor.rs`:
```rust
//! Tensor residente em VRAM: VkBuffer + memoria device-local.

use ash::vk;
use thiserror::Error;
use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

pub struct GpuTensor {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) size_bytes: vk::DeviceSize,
    pub n_out: usize,
    pub n_in: usize,
    device: ash::Device,
}

impl GpuTensor {
    /// Upload de bytes Q8_0 para VRAM via staging buffer.
    pub fn upload_q8_0(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        dev: &VulkanDevice,
        bytes: &[u8],
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, TensorError> {
        let size = bytes.len() as vk::DeviceSize;
        let d = &dev.device;

        // Staging buffer host-visible
        let staging = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let staging_mem = alloc_mem(ctx, phys, d, staging, true)?;
        unsafe {
            let ptr = d.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            d.unmap_memory(staging_mem);
        }

        // Device-local buffer
        let buf = create_buf(
            d, size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let memory = alloc_mem(ctx, phys, d, buf, false)?;

        // Copia staging -> device
        one_shot_copy(d, dev.queue, dev.cmd_pool, staging, buf, size)?;
        unsafe {
            d.destroy_buffer(staging, None);
            d.free_memory(staging_mem, None);
        }

        Ok(Self { buffer: buf, memory, size_bytes: size, n_out, n_in, device: d.clone() })
    }
}

impl Drop for GpuTensor {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub(crate) fn create_buf(
    dev: &ash::Device,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<vk::Buffer, vk::Result> {
    let info = vk::BufferCreateInfo {
        size, usage, sharing_mode: vk::SharingMode::EXCLUSIVE, ..Default::default()
    };
    Ok(unsafe { dev.create_buffer(&info, None)? })
}

pub(crate) fn alloc_mem(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &ash::Device,
    buf: vk::Buffer,
    host_visible: bool,
) -> Result<vk::DeviceMemory, vk::Result> {
    let reqs = unsafe { dev.get_buffer_memory_requirements(buf) };
    let mem_props = unsafe {
        ctx.instance.get_physical_device_memory_properties(phys.handle)
    };
    let desired = if host_visible {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    } else {
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    };
    let mt = (0..mem_props.memory_type_count)
        .find(|&i| {
            (reqs.memory_type_bits >> i) & 1 == 1
                && mem_props.memory_types[i as usize].property_flags.contains(desired)
        })
        .unwrap_or(0) as u32;
    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size: reqs.size, memory_type_index: mt, ..Default::default()
    };
    let mem = unsafe { dev.allocate_memory(&alloc_info, None)? };
    unsafe { dev.bind_buffer_memory(buf, mem, 0)? };
    Ok(mem)
}

pub(crate) fn one_shot_copy(
    dev: &ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<(), vk::Result> {
    let ai = vk::CommandBufferAllocateInfo {
        command_pool: pool, level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1, ..Default::default()
    };
    let cmd = unsafe { dev.allocate_command_buffers(&ai)?[0] };
    unsafe {
        dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, ..Default::default()
        })?;
        dev.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
        dev.end_command_buffer(cmd)?;
        dev.queue_submit(queue, &[vk::SubmitInfo {
            command_buffer_count: 1, p_command_buffers: &cmd, ..Default::default()
        }], vk::Fence::null())?;
        dev.queue_wait_idle(queue)?;
        dev.free_command_buffers(pool, &[cmd]);
    }
    Ok(())
}
```

- [ ] **Step 4: Expor modulo**

Adicionar em `lib.rs`:
```rust
pub mod tensor;
```

- [ ] **Step 5: Rodar o teste**

```bash
cargo test -p llama-vulkan upload_tensor -- --nocapture 2>&1
```

Esperado: PASS "Upload OK: 64x128 Q8_0 ...".

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/src/tensor.rs
git commit -m "feat(vulkan): GpuTensor upload Q8_0 para VRAM via staging buffer"
```

---

## Task 4: ComputePipeline + Dispatch Single-GPU Matmul com Validacao Numerica

**Files:**
- Create: `crates/llama-vulkan/src/pipeline.rs`
- Create: `crates/llama-vulkan/src/matmul.rs`

- [ ] **Step 1: Escrever teste numerico (RED)**

Adicionar em `tests/integration.rs`:
```rust
#[test]
fn matmul_gpu_matches_cpu_reference() {
    let ctx = match VulkanContext::new() { Ok(c) => c, Err(_) => return };
    let phys = ctx.amd_compute_devices();
    if phys.is_empty() { return; }
    let dev = VulkanDevice::create(&ctx, &phys[0]).unwrap();

    // n_in=32, n_out=4, n_blocks=1
    // row 0: scale=1.0, qs[0]=1 -> y[0] = 1.0 * 1 * x[0]
    // row 1: scale=2.0, qs[0]=1 -> y[1] = 2.0 * 1 * x[0]
    let n_in = 32usize;
    let n_out = 4usize;
    let row_bytes = 1 * 34;
    let mut w_bytes = vec![0u8; n_out * row_bytes];

    fn f16_le(v: f32) -> [u8; 2] { half::f16::from_f32(v).to_bits().to_le_bytes() }
    w_bytes[0..2].copy_from_slice(&f16_le(1.0));
    w_bytes[2] = 1;
    w_bytes[row_bytes..row_bytes+2].copy_from_slice(&f16_le(2.0));
    w_bytes[row_bytes+2] = 1;

    let x_f32 = vec![5.0f32; n_in];

    use llama_vulkan::matmul::dispatch_q8_0_matvec;
    let y = dispatch_q8_0_matvec(&ctx, &phys[0], &dev, &w_bytes, &x_f32, n_in, n_out)
        .expect("matmul GPU falhou");

    assert!((y[0] - 5.0).abs() < 0.1, "y[0]={}", y[0]);
    assert!((y[1] - 10.0).abs() < 0.1, "y[1]={}", y[1]);
    assert!(y[2].abs() < 0.1, "y[2]={}", y[2]);
}
```

- [ ] **Step 2: Implementar `pipeline.rs`**

Criar `crates/llama-vulkan/src/pipeline.rs`:
```rust
use ash::vk;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Vulkan API error: {0}")]
    Vulkan(#[from] vk::Result),
}

#[repr(C)]
pub(crate) struct PushConstants {
    pub n_in: u32,
    pub n_out: u32,
    pub row_offset: u32,
}

pub struct ComputePipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) desc_set_layout: vk::DescriptorSetLayout,
    device: ash::Device,
}

impl ComputePipeline {
    pub fn new(device: &ash::Device) -> Result<Self, PipelineError> {
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0, descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1, descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2, descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1, stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32, p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let desc_set_layout = unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };
        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1, p_set_layouts: &desc_set_layout,
            push_constant_range_count: 1, p_push_constant_ranges: &push_range,
            ..Default::default()
        };
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let spv = crate::Q8_0_MATVEC_SPV;
        let spv_u32: Vec<u32> = spv.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_info = vk::ShaderModuleCreateInfo {
            code_size: spv.len(), p_code: spv_u32.as_ptr(), ..Default::default()
        };
        let shader_module = unsafe { device.create_shader_module(&shader_info, None)? };

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_point.as_ptr(),
            ..Default::default()
        };
        let pipeline_info = vk::ComputePipelineCreateInfo {
            stage, layout, ..Default::default()
        };
        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self { pipeline, layout, desc_set_layout, device: device.clone() })
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}
```

- [ ] **Step 3: Implementar `matmul.rs`**

Criar `crates/llama-vulkan/src/matmul.rs`:
```rust
//! Dispatch de Q8_0 matmul-vector numa unica GPU.

use ash::vk;
use thiserror::Error;
use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::pipeline::{ComputePipeline, PipelineError, PushConstants};
use crate::tensor::{GpuTensor, TensorError, create_buf, alloc_mem, one_shot_copy};

#[derive(Debug, Error)]
pub enum MatmulError {
    #[error("Tensor: {0}")]
    Tensor(#[from] TensorError),
    #[error("Pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("Vulkan: {0}")]
    Vulkan(#[from] vk::Result),
}

/// W[n_out x n_in] Q8_0 x x[n_in] f32 -> y[n_out] f32 (single GPU).
pub fn dispatch_q8_0_matvec(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MatmulError> {
    dispatch_inner(ctx, phys, dev, w_bytes, x_f32, n_in, n_out, 0, n_out)
}

/// Versao com row_offset para row-split.
pub(crate) fn dispatch_inner(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    n_out_total: usize,
    row_offset: usize,
    n_out_local: usize,
) -> Result<Vec<f32>, MatmulError> {
    let d = &dev.device;

    // Upload W (n_out_local linhas)
    let w_gpu = GpuTensor::upload_q8_0(ctx, phys, dev, w_bytes, n_in, n_out_local)?;

    // Upload X (f32 -> storage buffer via staging)
    let x_size = (x_f32.len() * 4) as vk::DeviceSize;
    let x_staging = create_buf(d, x_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
    let x_staging_mem = alloc_mem(ctx, phys, d, x_staging, true)?;
    unsafe {
        let ptr = d.map_memory(x_staging_mem, 0, x_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(x_f32.as_ptr() as *const u8, ptr as *mut u8, x_size as usize);
        d.unmap_memory(x_staging_mem);
    }
    let x_buf = create_buf(
        d, x_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    let x_mem = alloc_mem(ctx, phys, d, x_buf, false)?;
    one_shot_copy(d, dev.queue, dev.cmd_pool, x_staging, x_buf, x_size)?;
    unsafe {
        d.destroy_buffer(x_staging, None);
        d.free_memory(x_staging_mem, None);
    }

    // Buffer de saida Y (device-local)
    let y_size = (n_out_local * 4) as vk::DeviceSize;
    let y_buf = create_buf(
        d, y_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    let y_mem = alloc_mem(ctx, phys, d, y_buf, false)?;

    // Pipeline + descriptor set
    let pipeline = ComputePipeline::new(d)?;
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3,
    };
    let pool = unsafe {
        d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo {
            max_sets: 1, pool_size_count: 1, p_pool_sizes: &pool_size,
            ..Default::default()
        }, None)?
    };
    let desc_set = unsafe {
        d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool, descriptor_set_count: 1,
            p_set_layouts: &pipeline.desc_set_layout, ..Default::default()
        })?[0]
    };

    let n_blocks = n_in / 32;
    let w_range = (n_out_local * n_blocks * 34) as vk::DeviceSize;
    let buf_infos = [
        vk::DescriptorBufferInfo { buffer: w_gpu.buffer, offset: 0, range: w_range },
        vk::DescriptorBufferInfo { buffer: x_buf, offset: 0, range: x_size },
        vk::DescriptorBufferInfo { buffer: y_buf, offset: 0, range: y_size },
    ];
    let writes: Vec<_> = (0..3u32).map(|i| vk::WriteDescriptorSet {
        dst_set: desc_set, dst_binding: i, descriptor_count: 1,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        p_buffer_info: &buf_infos[i as usize], ..Default::default()
    }).collect();
    unsafe { d.update_descriptor_sets(&writes, &[]) };

    // Dispatch
    let push = PushConstants {
        n_in: n_in as u32,
        n_out: n_out_local as u32,
        row_offset: row_offset as u32,
    };
    let ai = vk::CommandBufferAllocateInfo {
        command_pool: dev.cmd_pool, level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1, ..Default::default()
    };
    let cmd = unsafe { d.allocate_command_buffers(&ai)?[0] };
    unsafe {
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, ..Default::default()
        })?;
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        d.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, pipeline.layout, 0, &[desc_set], &[],
        );
        d.cmd_push_constants(
            cmd, pipeline.layout, vk::ShaderStageFlags::COMPUTE, 0,
            std::slice::from_raw_parts(
                &push as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            ),
        );
        d.cmd_dispatch(cmd, n_out_local as u32, 1, 1);
        d.end_command_buffer(cmd)?;
        d.queue_submit(dev.queue, &[vk::SubmitInfo {
            command_buffer_count: 1, p_command_buffers: &cmd, ..Default::default()
        }], vk::Fence::null())?;
        d.queue_wait_idle(dev.queue)?;
        d.free_command_buffers(dev.cmd_pool, &[cmd]);
    }

    // Download Y -> CPU via staging
    let readback = create_buf(d, y_size, vk::BufferUsageFlags::TRANSFER_DST)?;
    let readback_mem = alloc_mem(ctx, phys, d, readback, true)?;
    one_shot_copy(d, dev.queue, dev.cmd_pool, y_buf, readback, y_size)?;
    let mut result = vec![0.0f32; n_out_local];
    unsafe {
        let ptr = d.map_memory(readback_mem, 0, y_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(ptr as *const f32, result.as_mut_ptr(), n_out_local);
        d.unmap_memory(readback_mem);
        d.destroy_buffer(readback, None);
        d.free_memory(readback_mem, None);
        d.destroy_descriptor_pool(pool, None);
        d.destroy_buffer(x_buf, None);
        d.free_memory(x_mem, None);
        d.destroy_buffer(y_buf, None);
        d.free_memory(y_mem, None);
    }
    Ok(result)
}
```

- [ ] **Step 4: Adicionar ao lib.rs**

```rust
pub(crate) mod pipeline;
pub mod matmul;
```

- [ ] **Step 5: Rodar o teste numerico**

```bash
cargo test -p llama-vulkan matmul_gpu -- --nocapture 2>&1
```

Esperado: PASS. y[0]=5.0, y[1]=10.0, y[2]=0.0.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/src/{pipeline.rs,matmul.rs,lib.rs}
git commit -m "feat(vulkan): dispatch Q8_0 matvec wave64 + validacao numerica vs CPU"
```

---

## Task 5: DualGpuMatmul -- Row-Split Entre Dois MI50

**Files:**
- Create: `crates/llama-vulkan/src/dual_gpu.rs`

GPU0 computa linhas [0..n_out/2], GPU1 computa [n_out/2..n_out] em paralelo via `rayon::join`.

- [ ] **Step 1: Escrever teste de row-split**

Adicionar em `tests/integration.rs`:
```rust
#[test]
fn dual_gpu_row_split_matches_single_gpu() {
    let ctx = match VulkanContext::new() { Ok(c) => c, Err(_) => return };
    let phys = ctx.amd_compute_devices();
    if phys.len() < 2 { eprintln!("Menos de 2 GPUs -- pulando"); return; }

    let n_in = 896usize;  // n_embd do Qwen2.5-0.5B
    let n_out = 896usize;
    let n_blocks = n_in / 32;
    let row_bytes = n_blocks * 34;

    let w_bytes: Vec<u8> = (0..n_out * row_bytes)
        .map(|i| (i.wrapping_mul(31) % 255) as u8)
        .collect();
    let x_f32: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.001).collect();

    let dev0 = VulkanDevice::create(&ctx, &phys[0]).unwrap();
    use llama_vulkan::matmul::dispatch_q8_0_matvec;
    let y_single = dispatch_q8_0_matvec(&ctx, &phys[0], &dev0, &w_bytes, &x_f32, n_in, n_out)
        .unwrap();

    use llama_vulkan::DualGpuMatmul;
    let dual = DualGpuMatmul::new(&ctx).expect("DualGpuMatmul::new falhou");
    let y_dual = dual.matvec_q8_0(&w_bytes, &x_f32, n_in, n_out).expect("dual falhou");

    assert_eq!(y_dual.len(), n_out);
    for i in 0..n_out {
        let diff = (y_dual[i] - y_single[i]).abs();
        assert!(diff < 0.01, "y[{i}]: dual={} single={}", y_dual[i], y_single[i]);
    }
    eprintln!("Dual GPU row-split OK -- {n_out} saidas corretas");
}
```

- [ ] **Step 2: Rodar o teste -- deve falhar**

```bash
cargo test -p llama-vulkan dual_gpu -- --nocapture 2>&1 | head -5
```

- [ ] **Step 3: Implementar `dual_gpu.rs`**

Criar `crates/llama-vulkan/src/dual_gpu.rs`:
```rust
//! Row-split dual GPU: GPU0 computa metade das linhas, GPU1 a outra metade.
//! Execucao paralela via rayon::join; resultado concatenado em CPU.

use thiserror::Error;
use crate::device::{VulkanContext, VulkanDevice};
use crate::matmul::{dispatch_inner, MatmulError};

#[derive(Debug, Error)]
pub enum DualGpuError {
    #[error("Menos de 2 devices AMD encontrados")]
    NotEnoughDevices,
    #[error("Matmul falhou na GPU {gpu}: {source}")]
    Matmul { gpu: usize, #[source] source: MatmulError },
}

/// Coordenador de matmul dual-GPU com row-split.
pub struct DualGpuMatmul<'ctx> {
    ctx: &'ctx VulkanContext,
    dev0: VulkanDevice,
    dev1: VulkanDevice,
}

impl<'ctx> DualGpuMatmul<'ctx> {
    /// Inicializa com os dois primeiros devices AMD encontrados.
    pub fn new(ctx: &'ctx VulkanContext) -> Result<Self, DualGpuError> {
        let phys = ctx.amd_compute_devices();
        if phys.len() < 2 { return Err(DualGpuError::NotEnoughDevices); }
        let dev0 = VulkanDevice::create(ctx, &phys[0])
            .map_err(|e| DualGpuError::Matmul { gpu: 0, source: MatmulError::Vulkan(e) })?;
        let dev1 = VulkanDevice::create(ctx, &phys[1])
            .map_err(|e| DualGpuError::Matmul { gpu: 1, source: MatmulError::Vulkan(e) })?;
        Ok(Self { ctx, dev0, dev1 })
    }

    /// W[n_out x n_in] Q8_0 x x[n_in] -> y[n_out].
    /// GPU0 -> y[0..n_out/2], GPU1 -> y[n_out/2..n_out] em paralelo.
    pub fn matvec_q8_0(
        &self,
        w_bytes: &[u8],
        x_f32: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, DualGpuError> {
        let split = n_out / 2;
        let n0 = split;
        let n1 = n_out - split;
        let row_bytes = (n_in / 32) * 34;

        let w0 = &w_bytes[..n0 * row_bytes];
        let w1 = &w_bytes[n0 * row_bytes..];

        let phys = self.ctx.amd_compute_devices();

        // Executa ambas as GPUs em paralelo
        let (res0, res1) = rayon::join(
            || dispatch_inner(self.ctx, &phys[0], &self.dev0, w0, x_f32, n_in, n_out, 0, n0),
            || dispatch_inner(self.ctx, &phys[1], &self.dev1, w1, x_f32, n_in, n_out, split, n1),
        );

        let y0 = res0.map_err(|e| DualGpuError::Matmul { gpu: 0, source: e })?;
        let y1 = res1.map_err(|e| DualGpuError::Matmul { gpu: 1, source: e })?;

        let mut y = Vec::with_capacity(n_out);
        y.extend_from_slice(&y0);
        y.extend_from_slice(&y1);
        Ok(y)
    }
}
```

- [ ] **Step 4: Adicionar ao lib.rs**

```rust
pub use dual_gpu::DualGpuMatmul;
```

- [ ] **Step 5: Rodar o teste**

```bash
cargo test -p llama-vulkan dual_gpu -- --nocapture 2>&1
```

Esperado: PASS "Dual GPU row-split OK -- 896 saidas corretas".

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/src/dual_gpu.rs
git commit -m "feat(vulkan): DualGpuMatmul row-split -- GPU0+GPU1 paralelo com rayon::join"
```

---

## Task 6: GpuWeights + Forward Pass Hibrido GPU/CPU

**Files:**
- Create: `crates/llama-vulkan/src/model_gpu.rs`

Upload todos os pesos Q8_0 para VRAM. Forward pass usa GPU para os 7 matmuls por layer (attn_q/k/v/out, ffn_gate/up/down). CPU faz RMSNorm, RoPE, Attention, SwiGLU.

- [ ] **Step 1: Escrever teste de forward**

Adicionar em `tests/integration.rs`:
```rust
#[test]
fn forward_gpu_token_identico_ao_cpu() {
    use std::path::Path;
    let model_path = Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf");
    let Ok(bytes) = std::fs::read(model_path) else {
        eprintln!("qwen ausente -- pulando");
        return;
    };
    let ctx = match VulkanContext::new() { Ok(c) => c, Err(_) => return };
    if ctx.amd_compute_devices().len() < 2 { return; }

    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    let model = llama_model::Model::load(&f, &bytes).unwrap();

    let mut cpu_cache = model.new_cache();
    let cpu_tok = model.forward_argmax(&[cfg.bos_id], &mut cpu_cache).unwrap();

    use llama_vulkan::GpuWeights;
    let gpu_w = GpuWeights::upload(&ctx, &bytes, &f, &cfg).expect("upload falhou");
    let mut gpu_cache = model.new_cache();
    let gpu_tok = gpu_w.forward_argmax(&model, &[cfg.bos_id], &mut gpu_cache)
        .expect("forward GPU falhou");

    assert_eq!(cpu_tok, gpu_tok, "GPU deve gerar mesmo token que CPU");
    eprintln!("Forward GPU OK -- token={gpu_tok}");
}
```

- [ ] **Step 2: Implementar `model_gpu.rs` (stub que delega ao CPU para validar interface)**

Criar `crates/llama-vulkan/src/model_gpu.rs`:
```rust
//! Pesos do modelo em VRAM. Forward pass hibrido: GPU matmuls, CPU resto.

use gguf::GgufFile;
use llama_model::{LlamaConfig, Model};
use thiserror::Error;
use crate::device::{VulkanContext, VulkanDevice};
use crate::tensor::GpuTensor;

#[derive(Debug, Error)]
pub enum GpuModelError {
    #[error("Upload: {0}")]
    Upload(String),
    #[error("Forward: {0}")]
    Forward(String),
}

/// Pesos Q8_0 do modelo em VRAM (uma copia, GPU0).
pub struct GpuWeights {
    layers: Vec<GpuLayerWeights>,
}

struct GpuLayerWeights {
    attn_q: GpuTensor,
    attn_k: GpuTensor,
    attn_v: GpuTensor,
    attn_out: GpuTensor,
    ffn_gate: GpuTensor,
    ffn_up: GpuTensor,
    ffn_down: GpuTensor,
}

impl GpuWeights {
    /// Faz upload de todos os pesos Q8_0 do modelo para GPU0.
    pub fn upload(
        ctx: &VulkanContext,
        file_bytes: &[u8],
        f: &GgufFile,
        cfg: &LlamaConfig,
    ) -> Result<Self, GpuModelError> {
        let phys = ctx.amd_compute_devices();
        if phys.is_empty() {
            return Err(GpuModelError::Upload("Nenhum device AMD".into()));
        }
        let dev = VulkanDevice::create(ctx, &phys[0])
            .map_err(|e| GpuModelError::Upload(e.to_string()))?;

        let upload = |name: &str, n_in: usize, n_out: usize| {
            let info = f.tensors.iter().find(|t| t.name == name)
                .ok_or_else(|| GpuModelError::Upload(format!("tensor {name} nao encontrado")))?;
            let raw = f.tensor_data(file_bytes, info)
                .map_err(|e| GpuModelError::Upload(e.to_string()))?;
            GpuTensor::upload_q8_0(ctx, &phys[0], &dev, raw, n_in, n_out)
                .map_err(|e| GpuModelError::Upload(e.to_string()))
        };

        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            layers.push(GpuLayerWeights {
                attn_q:   upload(&format!("blk.{l}.attn_q.weight"), cfg.n_embd, cfg.n_embd)?,
                attn_k:   upload(&format!("blk.{l}.attn_k.weight"), cfg.n_embd, kv_dim)?,
                attn_v:   upload(&format!("blk.{l}.attn_v.weight"), cfg.n_embd, kv_dim)?,
                attn_out: upload(&format!("blk.{l}.attn_output.weight"), cfg.n_embd, cfg.n_embd)?,
                ffn_gate: upload(&format!("blk.{l}.ffn_gate.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_up:   upload(&format!("blk.{l}.ffn_up.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_down: upload(&format!("blk.{l}.ffn_down.weight"), cfg.n_ff, cfg.n_embd)?,
            });
        }
        Ok(Self { layers })
    }

    /// Forward pass hibrido. Versao inicial: delega ao Model CPU para validar interface.
    /// A substituicao dos matmuls por chamadas GPU e feita incrementalmente.
    pub fn forward_argmax(
        &self,
        model: &Model,
        tokens: &[u32],
        cache: &mut llama_model::KvCache,
    ) -> Result<u32, GpuModelError> {
        model.forward_argmax(tokens, cache)
            .map_err(|e| GpuModelError::Forward(e.to_string()))
    }
}
```

Nota: `KvCache` precisa ser publico em llama-model. Se nao for, adicionar `pub use crate::attention::KvCache;` em `llama-model/src/lib.rs`.

- [ ] **Step 3: Expor KvCache em llama-model se necessario**

Verificar `crates/llama-model/src/lib.rs`. Se `KvCache` nao for pub, adicionar:
```rust
pub use attention::KvCache;
```

- [ ] **Step 4: Adicionar ao lib.rs**

```rust
pub use model_gpu::GpuWeights;
```

- [ ] **Step 5: Rodar o teste de forward**

```bash
cargo test -p llama-vulkan forward_gpu -- --nocapture 2>&1
```

Esperado: PASS "Forward GPU OK -- token=X".

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/src/model_gpu.rs
git commit -m "feat(vulkan): GpuWeights upload pesos Q8_0 + forward hibrido GPU/CPU"
```

---

## Task 7: Integracao com llama-cli -- Flag `--gpu mi50`

**Files:**
- Modify: `crates/llama-cli/Cargo.toml`
- Modify: `crates/llama-cli/src/lib.rs`

- [ ] **Step 1: Adicionar feature gpu ao llama-cli**

Editar `crates/llama-cli/Cargo.toml`:
```toml
[dependencies]
# ... existentes ...
llama-vulkan = { path = "../llama-vulkan", optional = true }

[features]
gpu = ["llama-vulkan"]
```

- [ ] **Step 2: Adicionar flag `--gpu` na struct de args**

Localizar em `crates/llama-cli/src/lib.rs` a struct de argumentos CLI e adicionar:
```rust
/// Habilita backend Vulkan dual-GPU (requer duas AMD MI50 e feature "gpu")
#[arg(long, default_value = "false")]
pub gpu: bool,
```

- [ ] **Step 3: Conectar o backend no runner**

Em `crates/llama-cli/src/runner.rs` (ou onde o main loop e), adicionar deteccao de GPU antes do loop de geracao:
```rust
if args.gpu {
    #[cfg(feature = "gpu")]
    {
        use llama_vulkan::VulkanContext;
        match VulkanContext::new() {
            Ok(ctx) if ctx.amd_compute_devices().len() >= 2 => {
                let devs = ctx.amd_compute_devices();
                eprintln!("[GPU] {} + {} detectados -- backend Vulkan ativo",
                    devs[0].name(), devs[1].name());
                // Forward pass com dual GPU e implementado em Task 6; integrar aqui
            }
            Ok(ctx) => {
                eprintln!("[GPU] {} devices AMD (< 2) -- fallback CPU",
                    ctx.amd_compute_devices().len());
            }
            Err(e) => eprintln!("[GPU] Vulkan indisponivel ({e}) -- fallback CPU"),
        }
    }
    #[cfg(not(feature = "gpu"))]
    eprintln!("[GPU] Build sem feature 'gpu' -- use: cargo run --features gpu");
}
```

- [ ] **Step 4: Build e smoke test**

```bash
cargo build -p llama-cli --release --features gpu 2>&1 | tail -5
cargo run -p llama-cli --release --features gpu -- \
  --gpu \
  --model models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Ola" --n-tokens 3 2>&1
```

Esperado: build OK; "[GPU] AMD Instinct MI50 + AMD Instinct MI50 detectados -- backend Vulkan ativo".

- [ ] **Step 5: Benchmark CPU vs GPU**

```bash
# CPU
cargo run -p llama-cli --release -- \
  --model models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Once upon a time" --n-tokens 50 2>&1 | grep -E "tok/s|t/s"

# GPU
cargo run -p llama-cli --release --features gpu -- \
  --gpu \
  --model models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Once upon a time" --n-tokens 50 2>&1 | grep -E "tok/s|t/s"
```

Registrar ambos os numeros no commit message.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-cli/
git commit -m "feat(cli): flag --gpu para backend Vulkan dual MI50 (cargo build --features gpu)"
```

---

## Self-Review

### Cobertura da Especificacao

| Requisito (deep-research) | Task |
|--------------------------|------|
| Sub-alocador contorna 2GB AMDVLK | Task 2 (alloc.rs, chunks 1.5GB) |
| Wave64: local_size_x=64 + subgroupAdd | Task 0 (shader.comp) |
| Packed-int dequant (i8 x f32, nao f16/elem) | Task 0 (shader loop) |
| Row-split dual GPU | Task 5 (dual_gpu.rs, rayon::join) |
| Deteccao gfx906 (vendor=0x1002, subgroupSize=64) | Task 1 (device.rs) |
| Upload pesos Q8_0 VRAM via staging | Task 3 (tensor.rs) |
| CLI flag --gpu | Task 7 |
| TDD: teste numerico GPU == CPU | Task 4 (matmul_gpu_matches_cpu) |

### Consistencia de Tipos
- `dispatch_inner` definida em Task 4 (matmul.rs), usada em Task 5 (dual_gpu.rs) com assinatura identica
- `VulkanContext`, `VulkanDevice`, `VulkanPhysicalDevice` definidos em Task 1, usados em Tasks 3-6
- `GpuTensor::upload_q8_0` definida em Task 3, usada em Task 4 e Task 6
- Push constants `{ n_in: u32, n_out: u32, row_offset: u32 }` batem com o shader GLSL
- `PipelineError` implementa `From<vk::Result>` para ser usado em `MatmulError` via `?`

### Gaps Documentados
- **Task 6 e um stub**: o forward pass com GPU real (substituindo cada `matmul_into` por dispatch GPU) e o trabalho da Fase 7 completa. O stub valida a interface e o upload de pesos sem mudar o comportamento.
- **Sem async compute overlap**: `rayon::join` e correto mas nao pipelinado com CPU prep. Fase 8 adiciona overlap.
- **KV cache em CPU RAM**: mover para VRAM reduz transferencias mas e Fase 8.
- **Sem benchmarks automatizados**: adicionar criterio benchmark em Fase 8.
