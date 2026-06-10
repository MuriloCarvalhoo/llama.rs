pub mod alloc;
mod backend;
mod device;
mod dual_gpu;
pub mod matmul;
mod model_gpu;
pub(crate) mod pipeline;
pub mod tensor;

pub use backend::DualGpuBackend;
pub use device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
pub use dual_gpu::DualGpuMatmul;
pub use model_gpu::GpuWeights;

#[allow(dead_code)]
pub(crate) const Q8_0_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q8_0_MATVEC_SPV")));
