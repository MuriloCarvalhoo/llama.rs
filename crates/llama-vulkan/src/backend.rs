//! Adaptador: expõe DualGpuMatmul como `llama_model::GpuMatmul`.

use crate::{DualGpuMatmul, VulkanContext};
use llama_model::{GpuMatmul, ModelError};

/// Backend dual-MI50 que satisfaz a trait do llama-model.
pub struct DualGpuBackend<'ctx> {
    inner: DualGpuMatmul<'ctx>,
}

impl<'ctx> DualGpuBackend<'ctx> {
    pub fn new(ctx: &'ctx VulkanContext) -> Result<Self, ModelError> {
        let inner = DualGpuMatmul::new(ctx).map_err(|e| ModelError::Gpu(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl GpuMatmul for DualGpuBackend<'_> {
    fn matvec_q8_0(
        &self,
        w_bytes: &[u8],
        x: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, ModelError> {
        self.inner
            .matvec_q8_0(w_bytes, x, n_in, n_out)
            .map_err(|e| ModelError::Gpu(e.to_string()))
    }
}
