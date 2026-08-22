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

/// Shard recém-construído, a caminho da thread que vai usá-lo.
///
/// O `ResidentForward` guarda o ponteiro da memória mapeada dos logits, e ponteiro cru não
/// é `Send`. Durante a carga o shard ainda não foi usado por ninguém — nenhum command
/// buffer gravado, nenhuma referência viva ao mapa —, então movê-lo da thread que o
/// construiu para a que vai usá-lo é seguro.
struct EnviaShard<'ctx>(ResidentForward<'ctx>);
// SAFETY: ver acima — a transferência acontece antes de qualquer uso concorrente, e o
// ponteiro mapeado só é lido pela thread dona depois disso.
unsafe impl Send for EnviaShard<'_> {}

impl<'ctx> LayerSplitForward<'ctx> {
    /// Divide `config.n_layer` entre os devices **proporcionalmente à VRAM livre**, como o
    /// llama.cpp faz em `llama_model::load_tensors`. Divisão por contagem igual seria pior
    /// aqui: a GPU que roda o display tem ~1.6 GB a menos, e estourar a VRAM dela empurra
    /// pesos para GTT — medimos 95 GB/s contra 714 GB/s quando isso acontece.
    pub fn new(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'ctx>,
    ) -> Result<Self, MatmulError> {
        Self::new_com(ctx, config, raw, aux, false)
    }

    /// Como [`Self::new`], com o multi-token prediction ligado em todos os shards.
    ///
    /// A cabeça em si só é montada no **último** shard, que é o que tem a norma final e
    /// portanto o hidden que ela combina com o embedding — mas os snapshots do estado
    /// recorrente são de cada shard, porque as camadas lineares estão divididas entre eles.
    pub fn new_com(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'ctx>,
        mtp: bool,
    ) -> Result<Self, MatmulError> {
        let devs = ctx.amd_compute_devices();
        if devs.len() < 2 {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        // `LLAMA_RS_SPLIT=N` fixa a fronteira entre as duas GPUs em vez de derivá-la da VRAM
        // livre. A VRAM livre varia com o que mais estiver no device, então o ponto de corte
        // muda entre execuções (30/34 numa, 31/33 na outra) e desloca o tempo por token em
        // alguns por cento — o bastante para afogar o efeito do que se está medindo.
        let boundaries = match std::env::var("LLAMA_RS_SPLIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0 && n < config.n_layer && devs.len() == 2)
        {
            Some(n) => vec![(0, n), (n, config.n_layer)],
            None => {
                // Reserva fixa por device antes de proporcionalizar: essas duas MI50 também
                // dirigem o display (uma tem um monitor 34" 3440x1440 ligado), e um device
                // sem VRAM livre nenhuma pode travar a sessão gráfica. 1.5 GiB é a margem
                // mais alta pedida (a GPU do monitor grande); aplicar nas duas é simples e
                // nunca half-fills a menor — não há como identificar de dentro do Vulkan
                // qual physical device é qual card do X sem VK_EXT_pci_bus_info.
                const MARGEM_VRAM: u64 = 1536 * 1024 * 1024;
                let free: Vec<u64> = devs
                    .iter()
                    .map(|p| {
                        p.free_device_memory(ctx)
                            .unwrap_or(1)
                            .saturating_sub(MARGEM_VRAM)
                            .max(1)
                    })
                    .collect();
                Self::split_points(&free, config.n_layer)
            }
        };

        // Uma thread por device: cada shard tem a sua fila, a sua command pool e o seu
        // staging, então as duas cargas não se cruzam em nada do Vulkan — e são vários
        // segundos de repack e cópia cada uma. Em série, uma GPU fica ociosa esperando a
        // outra terminar de subir os pesos dela.
        let shards = std::thread::scope(|escopo| {
            let threads: Vec<_> = boundaries
                .iter()
                .enumerate()
                // GPU sem camadas (VRAM livre desprezível) não abre thread.
                .filter(|&(_, &(first, end))| first != end)
                .map(|(i, &(first, end))| {
                    escopo.spawn(move || {
                        ResidentForward::new_shard_com(
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
                            mtp,
                        )
                        .map(EnviaShard)
                    })
                })
                .collect();
            let mut out = Vec::with_capacity(threads.len());
            for t in threads {
                // Pânico na thread de carga vira erro aqui: sem os pesos não há backend.
                let r = t
                    .join()
                    .map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
                out.push(r?.0);
            }
            Ok::<_, MatmulError>(out)
        })?;
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

    /// Zonas de GPU de cada shard, com o rótulo da trilha (device + faixa de camadas).
    pub fn gpu_spans_by_track(&self) -> Vec<(String, Vec<crate::GpuSpan>)> {
        self.shards
            .iter()
            .map(|s| {
                let sh = s.shard();
                (
                    format!("GPU{} ({}..{})", sh.device, sh.first_layer, sh.end_layer),
                    s.gpu_spans(),
                )
            })
            .collect()
    }

    /// Perfil agregado de cada shard (no-op sem `LLAMA_RS_PROFILE=1`).
    pub fn print_profile(&self) {
        for s in &self.shards {
            let sh = s.shard();
            eprintln!(
                "[prof] GPU{} camadas {}..{}",
                sh.device, sh.first_layer, sh.end_layer
            );
            s.print_profile();
        }
    }

    /// Rótulos das ops do plano do shard `s`.
    pub fn dbg_plano(&self, s: usize) -> Vec<&'static str> {
        self.shards
            .get(s)
            .map(|x| x.dbg_plano())
            .unwrap_or_default()
    }

    /// Stream residual do shard `s`, para diagnóstico.
    pub fn dbg_hidden(&self, s: usize) -> Option<Vec<f32>> {
        self.shards.get(s)?.dbg_hidden()
    }

    /// Buffer intermediário do caminho de atenção linear do shard `s`.
    pub fn dbg_dn_buf(&self, s: usize, nome: &str) -> Option<Vec<f32>> {
        self.shards.get(s)?.dbg_dn_buf(nome)
    }

    /// Estado recorrente da camada `l` do shard `s`, para diagnóstico.
    pub fn dbg_estado_delta(&self, s: usize, l: usize) -> Option<Vec<f32>> {
        self.shards.get(s)?.dbg_estado_delta(l)
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
        self.decode_batch(&[token], pos)
    }

    /// O batch atravessa os shards igual ao decode — a diferença é que a stream residual
    /// que passa entre as GPUs carrega os N tokens do bloco, não um só.
    fn decode_batch(
        &self,
        tokens: &[u32],
        pos0: usize,
    ) -> Result<Vec<f32>, llama_model::ModelError> {
        let mut carry: Option<Vec<f32>> = None;
        for shard in &self.shards {
            let out = shard
                .decode_shard_batch(tokens, pos0, carry.as_deref())
                .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
            carry = Some(out);
        }
        carry.ok_or_else(|| llama_model::ModelError::Gpu("nenhum shard".into()))
    }

    /// Todos os shards são construídos com o mesmo `batch_size()`.
    fn batch_size(&self) -> usize {
        self.shards
            .first()
            .map_or(1, llama_model::GpuResidentDecode::batch_size)
    }

    fn reset(&self) {
        for shard in &self.shards {
            shard.reset_len();
        }
    }

    /// O snapshot é por shard: cada GPU guarda o estado das camadas que ela executa. Só
    /// vale se **todos** aceitarem — meia sequência recuada é pior que reprocessar. Daí o
    /// `&` no lugar de `&&`: todos os shards têm de ser visitados de qualquer jeito.
    fn marcar(&self) -> bool {
        self.shards
            .iter()
            .fold(true, |ok, s| ok & s.marcar_snapshot())
    }
    fn restaurar(&self) -> bool {
        self.shards
            .iter()
            .fold(true, |ok, s| ok & s.restaurar_snapshot())
    }

    /// A cabeça mora no último shard — é dele o hidden que ela consome.
    fn tem_mtp(&self) -> bool {
        self.shards.last().is_some_and(ResidentForward::tem_mtp)
    }

    /// A tabela de embedding só existe no **primeiro** shard (nos demais seriam 5 GB de
    /// RAM sem uso), e a cabeça mora no último: a linha do token atravessa as duas GPUs
    /// pelo host, 20 KB por passo.
    fn propor_mtp(&self, token: u32, hidden_idx: usize) -> Result<u32, llama_model::ModelError> {
        let erro = |m: String| llama_model::ModelError::Gpu(m);
        let primeiro = self
            .shards
            .first()
            .ok_or_else(|| erro("nenhum shard".to_owned()))?;
        let ultimo = self
            .shards
            .last()
            .ok_or_else(|| erro("nenhum shard".to_owned()))?;
        let emb = primeiro
            .linha_embd(token)
            .ok_or_else(|| erro(format!("token {token} fora da tabela")))?;
        ultimo
            .propor_mtp(&emb, hidden_idx)
            .map_err(|e| erro(e.to_string()))
    }

    /// O verify atravessa os shards como o batch: a stream residual que passa entre as
    /// GPUs carrega os tokens do bloco, e só o último devolve logits.
    fn decode_verify(
        &self,
        tokens: &[u32; llama_model::VERIFY_TOK],
        pos0: usize,
    ) -> Result<Vec<f32>, llama_model::ModelError> {
        let mut carry: Option<Vec<f32>> = None;
        for shard in &self.shards {
            let out = shard
                .verify_shard(tokens, pos0, carry.as_deref())
                .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
            carry = Some(out);
        }
        carry.ok_or_else(|| llama_model::ModelError::Gpu("nenhum shard".into()))
    }

    /// O estado recorrente está dividido entre os shards: todos desfazem os rejeitados.
    fn rollback_verify(&self, manter: usize) -> Result<(), llama_model::ModelError> {
        for shard in &self.shards {
            shard
                .rollback_verify(manter)
                .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
        }
        Ok(())
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
