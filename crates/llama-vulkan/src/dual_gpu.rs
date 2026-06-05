//! Row-split dual GPU: GPU0 computa metade das linhas, GPU1 a outra metade.
//! Execucao paralela via rayon::join; resultado concatenado em CPU.

use crate::device::{VulkanContext, VulkanDevice};
use crate::matmul::{DispatchArgs, MatmulError, dispatch_inner};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DualGpuError {
    #[error("Menos de 2 devices AMD encontrados")]
    NotEnoughDevices,
    #[error("Matmul falhou na GPU {gpu}: {source}")]
    Matmul {
        gpu: usize,
        #[source]
        source: MatmulError,
    },
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
        if phys.len() < 2 {
            return Err(DualGpuError::NotEnoughDevices);
        }
        let dev0 = VulkanDevice::create(ctx, &phys[0]).map_err(|e| DualGpuError::Matmul {
            gpu: 0,
            source: MatmulError::Vulkan(e),
        })?;
        let dev1 = VulkanDevice::create(ctx, &phys[1]).map_err(|e| DualGpuError::Matmul {
            gpu: 1,
            source: MatmulError::Vulkan(e),
        })?;
        Ok(Self { ctx, dev0, dev1 })
    }

    /// W[n_out x n_in] Q8_0 x x[n_in] -> y[n_out].
    /// GPU0 -> y[0..split], GPU1 -> y[split..n_out] em paralelo.
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
            || {
                dispatch_inner(DispatchArgs {
                    ctx: self.ctx,
                    phys: &phys[0],
                    dev: &self.dev0,
                    w_bytes: w0,
                    x_f32,
                    n_in,
                    row_offset: 0,
                    n_out_local: n0,
                })
            },
            || {
                dispatch_inner(DispatchArgs {
                    ctx: self.ctx,
                    phys: &phys[1],
                    dev: &self.dev1,
                    w_bytes: w1,
                    x_f32,
                    n_in,
                    row_offset: split,
                    n_out_local: n1,
                })
            },
        );

        let y0 = res0.map_err(|e| DualGpuError::Matmul { gpu: 0, source: e })?;
        let y1 = res1.map_err(|e| DualGpuError::Matmul { gpu: 1, source: e })?;

        let mut y = Vec::with_capacity(n_out);
        y.extend_from_slice(&y0);
        y.extend_from_slice(&y1);
        Ok(y)
    }
}
