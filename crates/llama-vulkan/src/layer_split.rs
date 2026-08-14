//! Layer-split entre GPUs: cada uma executa uma faixa contígua de camadas.
//!
//! Entre um shard e o seguinte passa apenas a **stream residual** (`n_embd` floats,
//! ~20 KB), uma vez por token. É o que distingue esta abordagem do tensor-parallel, que
//! exigiria 2 all-reduces por camada — 96 sincronizações por token no Qwen2.5-14B, contra
//! 1 aqui. Medimos 59.3 µs por sincronização neste hardware (`spike.rs`), então o custo é
//! ~0.06 ms/token: ruído perto dos ~36 ms de um token do 14B.
//!
//! O ganho não é velocidade — é **capacidade**. Um modelo de 20–28 GiB não cabe nos 16 GiB
//! de uma MI50; dividido em duas, cabe. Os modelos que rodam rápido nesta máquina
//! (Qwen3.6-27B, 35B-A3B) estão todos nessa faixa.

use crate::device::VulkanContext;
use crate::matmul::MatmulError;
use crate::resident_forward::{ResidentForward, Shard};
use llama_model::{GpuAuxWeights, GpuRawWeights, LlamaConfig};

/// Decode dividido por camadas entre N GPUs.
pub struct LayerSplitForward<'ctx> {
    shards: Vec<ResidentForward<'ctx>>,
}

impl<'ctx> LayerSplitForward<'ctx> {
    /// Divide `config.n_layer` entre os devices **proporcionalmente à VRAM livre**, como o
    /// llama.cpp faz em `llama_model::load_tensors`. Divisão por contagem igual seria pior
    /// aqui: a GPU que roda o display tem ~1.6 GB a menos, e estourar a VRAM dela empurra
    /// pesos para GTT — medimos 95 GB/s contra 714 GB/s quando isso acontece.
    pub fn new(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'_>,
    ) -> Result<Self, MatmulError> {
        let devs = ctx.amd_compute_devices();
        if devs.len() < 2 {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        let free: Vec<u64> = devs
            .iter()
            .map(|p| p.free_device_memory(ctx).unwrap_or(1))
            .collect();
        let boundaries = Self::split_points(&free, config.n_layer);

        let mut shards = Vec::with_capacity(boundaries.len());
        for (i, &(first, end)) in boundaries.iter().enumerate() {
            if first == end {
                continue; // GPU sem camadas (VRAM livre desprezível)
            }
            shards.push(ResidentForward::new_shard(
                ctx,
                config,
                raw,
                aux,
                Shard {
                    device: i,
                    first_layer: first,
                    end_layer: end,
                    n_layer_total: config.n_layer,
                },
            )?);
        }
        Ok(Self { shards })
    }

    /// Fronteiras `[first, end)` por device, proporcionais a `free`.
    /// Extraído para poder ser testado sem GPU.
    pub(crate) fn split_points(free: &[u64], n_layer: usize) -> Vec<(usize, usize)> {
        let total: u128 = free.iter().map(|&f| u128::from(f)).sum::<u128>().max(1);
        let mut out = Vec::with_capacity(free.len());
        let mut acc: u128 = 0;
        let mut prev = 0usize;
        for (i, &f) in free.iter().enumerate() {
            acc += u128::from(f);
            // A última fronteira é fixada em n_layer para não perder camadas por arredondamento.
            let end = if i + 1 == free.len() {
                n_layer
            } else {
                usize::try_from(acc * n_layer as u128 / total).unwrap_or(n_layer)
            };
            let end = end.clamp(prev, n_layer);
            out.push((prev, end));
            prev = end;
        }
        out
    }

    /// Distribuição efetiva, para log: `(device, primeira, última+1)`.
    pub fn layout(&self) -> Vec<(usize, usize, usize)> {
        self.shards
            .iter()
            .map(|s| {
                let sh = s.shard();
                (sh.device, sh.first_layer, sh.end_layer)
            })
            .collect()
    }
}

impl llama_model::GpuResidentDecode for LayerSplitForward<'_> {
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, llama_model::ModelError> {
        let mut carry: Option<Vec<f32>> = None;
        for shard in &self.shards {
            let out = shard
                .decode_shard(token, pos, carry.as_deref())
                .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
            carry = Some(out);
        }
        carry.ok_or_else(|| llama_model::ModelError::Gpu("nenhum shard".into()))
    }

    fn reset(&self) {
        for shard in &self.shards {
            shard.reset_len();
        }
    }
}

use ash::vk;

#[cfg(test)]
mod tests {
    use super::LayerSplitForward as L;

    #[test]
    fn divide_proporcional_a_vram_livre() {
        // VRAM igual -> metade das camadas em cada.
        assert_eq!(L::split_points(&[16, 16], 48), vec![(0, 24), (24, 48)]);
        // A GPU do display tem menos livre -> recebe menos camadas.
        assert_eq!(L::split_points(&[14, 16], 48), vec![(0, 22), (22, 48)]);
        // Nenhuma camada se perde por arredondamento.
        for free in [[1u64, 7], [3, 5], [9, 1], [1, 1]] {
            let p = L::split_points(&free, 47);
            assert_eq!(p[0].0, 0);
            assert_eq!(p.last().unwrap().1, 47, "free={free:?}");
            assert_eq!(p[0].1, p[1].0, "faixas contiguas, free={free:?}");
        }
    }

    #[test]
    fn gpu_sem_vram_livre_nao_recebe_camadas() {
        let p = L::split_points(&[0, 16], 48);
        assert_eq!(p[0], (0, 0), "GPU sem VRAM livre fica sem camadas");
        assert_eq!(p[1], (0, 48));
    }

    #[test]
    fn tres_devices() {
        let p = L::split_points(&[16, 16, 16], 48);
        assert_eq!(p, vec![(0, 16), (16, 32), (32, 48)]);
    }
}
