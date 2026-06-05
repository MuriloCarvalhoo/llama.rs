pub mod alloc;
mod device;
mod dual_gpu;
mod matmul;
mod model_gpu;
mod pipeline;
mod tensor;

pub use device::{VulkanContext, VulkanDevice};
pub use dual_gpu::DualGpuMatmul;
pub use model_gpu::GpuWeights;

#[allow(dead_code)]
pub(crate) const Q8_0_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q8_0_MATVEC_SPV")));
