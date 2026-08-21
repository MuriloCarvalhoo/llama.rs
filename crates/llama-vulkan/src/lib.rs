pub mod alloc;
mod backend;
mod device;
mod dual_gpu;
mod layer_split;
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
pub use layer_split::LayerSplitForward;
pub use model_gpu::GpuWeights;
pub use resident::ResidentGpu;
pub use resident_forward::{DnPipe, GpuSpan, ResidentForward, Shard, forcar_splits};

pub(crate) const Q8_0_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q8_0_MATVEC_SPV")));
pub(crate) const QUANTIZE_X_SPV: &[u8] = include_bytes!(concat!(env!("QUANTIZE_X_SPV")));
pub(crate) const Q5_K_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q5_K_MATVEC_SPV")));
pub(crate) const Q6_K_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q6_K_MATVEC_SPV")));
pub(crate) const Q4_K_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q4_K_MATVEC_SPV")));
#[allow(dead_code)]
pub(crate) const RMSNORM_SPV: &[u8] = include_bytes!(concat!(env!("RMSNORM_SPV")));
pub(crate) const NORM_FUSED_SPV: &[u8] = include_bytes!(concat!(env!("NORM_FUSED_SPV")));
pub(crate) const NORM_P2_SPV: &[u8] = include_bytes!(concat!(env!("NORM_P2_SPV")));
#[allow(dead_code)]
pub(crate) const ROPE_SPV: &[u8] = include_bytes!(concat!(env!("ROPE_SPV")));
#[allow(dead_code)]
pub(crate) const ATTENTION_SPV: &[u8] = include_bytes!(concat!(env!("ATTENTION_SPV")));
pub(crate) const ATTENTION_SPLIT_SPV: &[u8] = include_bytes!(concat!(env!("ATTENTION_SPLIT_SPV")));
pub(crate) const ATTN_REDUCE_SPV: &[u8] = include_bytes!(concat!(env!("ATTN_REDUCE_SPV")));
#[allow(dead_code)]
pub(crate) const SWIGLU_SPV: &[u8] = include_bytes!(concat!(env!("SWIGLU_SPV")));
#[allow(dead_code)]
pub(crate) const DELTA_NET_SPV: &[u8] = include_bytes!(concat!(env!("DELTA_NET_SPV")));
#[allow(dead_code)]
pub(crate) const DN_CONV_SPV: &[u8] = include_bytes!(concat!(env!("DN_CONV_SPV")));
#[allow(dead_code)]
pub(crate) const DN_GATES_SPV: &[u8] = include_bytes!(concat!(env!("DN_GATES_SPV")));
#[allow(dead_code)]
pub(crate) const DN_NORM_SPV: &[u8] = include_bytes!(concat!(env!("DN_NORM_SPV")));
#[allow(dead_code)]
pub(crate) const GATE_MUL_SPV: &[u8] = include_bytes!(concat!(env!("GATE_MUL_SPV")));
#[allow(dead_code)]
pub(crate) const ADD_SPV: &[u8] = include_bytes!(concat!(env!("ADD_SPV")));
