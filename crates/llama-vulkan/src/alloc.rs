//! Sub-alocador VMA-style: chunks de 1.5GB para contornar limite AMDVLK.
//!
//! O driver AMDVLK tem `maxMemoryAllocationSize = 2GB`. Modelos grandes
//! (ex: 7B Q8_0 ≈ 7.7GB) requerem multiplas alocacoes menores. Este
//! sub-alocador usa bump allocation por offset dentro de chunks de 1.5GB.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use ash::vk;
use thiserror::Error;

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
///
/// O `ash::Device` nao implementa `Clone`, entao o device e recebido por
/// referencia em cada operacao que precisa dele. O caller e responsavel por
/// manter o device vivo e chamar `cleanup()` antes de dropar o alocador.
pub struct GpuAllocator {
    chunks: Vec<Chunk>,
    mem_type_idx: u32,
}

impl GpuAllocator {
    /// Cria o alocador, selecionando o tipo de memoria DEVICE_LOCAL.
    pub fn new(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        _dev: &VulkanDevice,
        _total_bytes: vk::DeviceSize,
    ) -> Result<Self, AllocError> {
        let mem_props = unsafe {
            // SAFETY: ctx.instance e phys.handle sao validos — criados e verificados
            // em VulkanContext::new antes de chegar aqui.
            ctx.instance
                .get_physical_device_memory_properties(phys.handle)
        };
        let mem_type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(AllocError::NoMemoryType)? as u32;

        Ok(Self {
            chunks: Vec::new(),
            mem_type_idx,
        })
    }

    /// Aloca `size` bytes alinhados a 256. Cria novo chunk se necessario.
    pub fn alloc(
        &mut self,
        dev: &ash::Device,
        size: vk::DeviceSize,
    ) -> Result<Allocation, AllocError> {
        let size = align_up(size, 256);

        // Tenta encaixar em chunk existente
        for (idx, chunk) in self.chunks.iter_mut().enumerate() {
            if chunk.free_start + size <= chunk.size {
                let offset = chunk.free_start;
                chunk.free_start += size;
                return Ok(Allocation {
                    chunk_idx: idx,
                    offset,
                    size,
                });
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
            // SAFETY: dev e valido; alloc_info foi construido com mem_type_idx
            // verificado em new() via get_physical_device_memory_properties.
            dev.allocate_memory(&alloc_info, None)
                .map_err(AllocError::Oom)?
        };
        let idx = self.chunks.len();
        self.chunks.push(Chunk {
            memory,
            size: chunk_size,
            free_start: size,
        });
        Ok(Allocation {
            chunk_idx: idx,
            offset: 0,
            size,
        })
    }

    /// Free e no-op neste bump allocator. Liberacao real ocorre em `cleanup()`.
    pub fn free(&mut self, _alloc: Allocation) {}

    /// Libera todos os chunks. Deve ser chamado antes de dropar o alocador.
    pub fn cleanup(&mut self, dev: &ash::Device) {
        for chunk in self.chunks.drain(..) {
            // SAFETY: dev e valido; chunk.memory foi alocado por este device em alloc()
            unsafe { dev.free_memory(chunk.memory, None) };
        }
    }
}

fn align_up(v: vk::DeviceSize, align: vk::DeviceSize) -> vk::DeviceSize {
    (v + align - 1) & !(align - 1)
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        // Nao podemos liberar sem o device — o caller deve chamar cleanup() antes.
        // Se chunks nao estiver vazio aqui, reporta mas nao faz panic (memory leak
        // intencional para evitar UB — o OS vai recuperar ao terminar o processo).
        if !self.chunks.is_empty() {
            eprintln!(
                "GpuAllocator::drop: {} chunks nao liberados (chame cleanup() antes)",
                self.chunks.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(512, 256), 512);
    }

    #[test]
    fn align_up_unaligned() {
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn align_up_zero() {
        assert_eq!(align_up(0, 256), 0);
    }
}
