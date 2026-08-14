pub mod alloc;
mod backend;
mod device;
mod dual_gpu;
pub mod matmul;
mod model_gpu;
pub(crate) mod pipeline;
mod resident;
mod resident_forward;
#[cfg(test)]
mod spike;
pub mod tensor;

pub use backend::DualGpuBackend;
pub use device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
pub use dual_gpu::DualGpuMatmul;
pub use model_gpu::GpuWeights;
pub use resident::ResidentGpu;
pub use resident_forward::ResidentForward;

pub(crate) const Q8_0_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q8_0_MATVEC_SPV")));
pub(crate) const QUANTIZE_X_SPV: &[u8] = include_bytes!(concat!(env!("QUANTIZE_X_SPV")));
#[allow(dead_code)]
pub(crate) const RMSNORM_SPV: &[u8] = include_bytes!(concat!(env!("RMSNORM_SPV")));
#[allow(dead_code)]
pub(crate) const ROPE_SPV: &[u8] = include_bytes!(concat!(env!("ROPE_SPV")));
#[allow(dead_code)]
pub(crate) const ATTENTION_SPV: &[u8] = include_bytes!(concat!(env!("ATTENTION_SPV")));
#[allow(dead_code)]
pub(crate) const SWIGLU_SPV: &[u8] = include_bytes!(concat!(env!("SWIGLU_SPV")));
#[allow(dead_code)]
pub(crate) const ADD_SPV: &[u8] = include_bytes!(concat!(env!("ADD_SPV")));
