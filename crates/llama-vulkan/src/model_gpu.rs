//! Pesos do modelo em VRAM. Forward pass hibrido: atualmente delega ao Model CPU.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::tensor::GpuTensor;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuModelError {
    #[error("Upload falhou: {0}")]
    Upload(String),
    #[error("Forward falhou: {0}")]
    Forward(String),
}

/// Pesos do modelo em VRAM.
pub struct GpuWeights {
    /// Número de camadas com pesos carregados em VRAM.
    pub n_layers_loaded: usize,
    /// Bytes totais alocados em VRAM.
    pub vram_bytes: u64,
    // Os GpuTensors são destruídos manualmente — stored aqui para lifetime tracking.
    // Em fase futura, serão usados para dispatch Vulkan.
    // TODO: implementar cleanup em GpuWeights::Drop com &ash::Device
    #[allow(dead_code)]
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
    /// Upload sintético de validação: aloca tensors Q8_0 zerados na VRAM.
    ///
    /// Versão simplificada que não precisa dos bytes raw do GGUF — útil para
    /// validar que o pipeline de upload funciona corretamente antes de integrar
    /// com o parser GGUF.
    pub fn upload_synthetic(
        ctx: &VulkanContext,
        n_layers: usize,
        n_embd: usize,
    ) -> Result<Self, GpuModelError> {
        let phys = ctx.amd_compute_devices();
        if phys.is_empty() {
            return Err(GpuModelError::Upload("Nenhum device AMD".into()));
        }
        let dev = VulkanDevice::create(ctx, &phys[0])
            .map_err(|e| GpuModelError::Upload(e.to_string()))?;

        let n_in = n_embd;
        let n_ff = n_embd * 4; // estimativa feed-forward
        let kv_dim = n_embd / 8; // estimativa para GQA (ex: Qwen2.5-0.5B usa 64 heads KV)

        let mut layers = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let make_tensor = |n_out: usize,
                               n_in: usize,
                               phys: &VulkanPhysicalDevice|
             -> Result<GpuTensor, GpuModelError> {
                // Q8_0: cada bloco de 32 elementos = 2 bytes (scale f16) + 32 bytes (quants) = 34 bytes
                let n_blocks = n_in / 32;
                let row_bytes = n_blocks * 34;
                let bytes = vec![0u8; n_out * row_bytes];
                GpuTensor::upload_q8_0(ctx, phys, &dev, &bytes, n_in, n_out)
                    .map_err(|e| GpuModelError::Upload(e.to_string()))
            };

            layers.push(GpuLayerWeights {
                attn_q: make_tensor(n_embd, n_in, &phys[0])?,
                attn_k: make_tensor(kv_dim, n_in, &phys[0])?,
                attn_v: make_tensor(kv_dim, n_in, &phys[0])?,
                attn_out: make_tensor(n_embd, n_embd, &phys[0])?,
                ffn_gate: make_tensor(n_ff, n_in, &phys[0])?,
                ffn_up: make_tensor(n_ff, n_in, &phys[0])?,
                ffn_down: make_tensor(n_embd, n_ff, &phys[0])?,
            });
        }

        let vram_bytes = layers
            .iter()
            .map(|l| {
                [
                    &l.attn_q,
                    &l.attn_k,
                    &l.attn_v,
                    &l.attn_out,
                    &l.ffn_gate,
                    &l.ffn_up,
                    &l.ffn_down,
                ]
                .iter()
                .map(|t| t.size_bytes)
                .sum::<u64>()
            })
            .sum();

        Ok(Self {
            n_layers_loaded: n_layers,
            vram_bytes,
            layers,
        })
    }
}
