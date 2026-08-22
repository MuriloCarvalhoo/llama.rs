//! Backend de decode 100% na GPU: todas as ativações e o KV-cache residentes em VRAM.
//! Só os logits finais voltam ao host. O token inteiro é um command buffer, e dentro dele
//! as ops só são separadas por barreira quando há dependência de verdade — ver
//! `marcar_barreiras`.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::matmul::MatmulError;
use crate::pipeline::ComputePipeline;
use crate::tensor::GpuTensor;
use ash::vk;
use llama_model::{GpuAuxWeights, GpuRawWeights, LlamaConfig};
use std::cell::RefCell;

/// Linhas de saida processadas por cada workgroup do matvec Q8_0 (tunavel — varredura no Step 5).
pub(crate) const MATVEC_NUM_ROWS: u32 = 2;
/// local_size_x do matvec (wave64 no MI50).
pub(crate) const MATVEC_WG: u32 = 64;

/// Teto de workgroups do passo 1 da norma. Cada um produz uma soma parcial, e o passo 2
/// soma todas em cada lane — então subir muito custa mais no passo 2 do que rende no 1.
pub(crate) const NORM_P1_WG: u32 = 32;

/// Tokens por bloco do prefill em batch.
///
/// Vira a specialization constant `COLS` dos matvec, então é fixa na criação das
/// pipelines — daí ser uma constante de processo e não um parâmetro. Com N tokens o peso
/// sai da VRAM uma vez para N ativações, que é o ganho todo: o decode é limitado por
/// banda, não por ALU.
///
/// O teto era 8 pelo `MAX_COLS` do `q6_k_matvec.comp` — o único shader que dimensionava os
/// acumuladores por uma constante fixa em vez de por `COLS`. Com ele resolvido, o teto
/// passa a ser a pressão de registrador: são `ROWS_PER_WAVE * COLS` acumuladores vivos por
/// lane, e em algum ponto a ocupância cai mais do que o reuso do peso rende. **Onde fica
/// esse ponto é empírico**: medido em 2026-08-21 com o GEMM ligado, a curva faz
/// 8→21,8, 16→14,6, **24→10,8**, 32→13,2 ms por token de prefill — o padrão é 24
/// (ver `docs/prefill-em-batch.md`).
///
/// `LLAMA_RS_BATCH=n` sobrescreve; `1` desliga o batch (prefill volta a ser token a token).
pub(crate) fn batch_size() -> usize {
    std::env::var("LLAMA_RS_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(24)
        .clamp(1, 32)
}

/// Tokens do bloco de **verificação** do MTP: o token já amostrado, a proposta da cabeça
/// e a proposta encadeada (a cabeça realimentada com o próprio hidden — o experimento da
/// tarefa 6 mediu 41,7 % de aceitação condicionada no 2º, acima do limiar do plano).
///
/// O valor mora no `llama-model` porque o laço de geração e o trait compartilham a
/// largura; aqui ele fixa `COLS` das pipelines de verify e o nº de snapshots.
pub(crate) const VERIFY_TOK: usize = llama_model::VERIFY_TOK;

/// Qual dos três planos do shard executar.
///
/// Os três saem da mesma `build_plan`; o que muda é a largura do bloco e o que se faz no
/// fim: o prefill calcula logits só do último token, o verify calcula os dois.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Modo {
    /// Um token — o decode.
    Decode,
    /// `cfg.n_batch` tokens, logits só do último — o prefill.
    Batch,
    /// `VERIFY_TOK` tokens, logits de **todos**, com snapshot do estado recorrente entre
    /// cada par consecutivo.
    Verify,
}

impl Modo {
    /// Tokens do bloco deste plano.
    fn n_tok(self, c: &Cfg) -> usize {
        match self {
            Modo::Decode => 1,
            Modo::Batch => c.n_batch,
            Modo::Verify => VERIFY_TOK,
        }
    }
}

/// Geometria dos matvec do **bloco de prefill**, separada da do decode.
///
/// O que decide a ocupância é `ROWS_PER_WAVE * COLS` acumuladores vivos por lane: no
/// decode (COLS=1) cabem 2 linhas por wave, mas com COLS=8 seriam 16 acumuladores mais as
/// 8 ativações, e aí a pressão de registrador derruba as waves por SIMD — e o workgroup
/// grande piora, porque todas as suas waves precisam caber juntas na mesma CU.
///
/// Medido no Qwen3.8-27B (ms do `matvec_q4k_b` por bloco de 8, `LLAMA_RS_PROFILE=1`):
///
/// | wg,linhas | 64,1 | 128,1 | 256,1 | 256,2 | 512,1 | 1024,1 |
/// |---|---:|---:|---:|---:|---:|---:|
/// | ms | **8,9** | 14,7 | 24,1 | 31,2 | 48,2 | 151,5 |
///
/// Daí o padrão `(64, 1)`: um workgroup **é** uma wave, então o escalonador coloca quantas
/// couberem por VGPR, sem ter de alocar o grupo inteiro de uma vez. O decode segue com a
/// sua geometria, que foi tunada para COLS=1. `LLAMA_RS_MATVEC_GEOM_B=wg,linhas` sobrescreve.
pub(crate) fn matvec_geom_batch() -> (u32, u32) {
    std::env::var("LLAMA_RS_MATVEC_GEOM_B")
        .ok()
        .and_then(|v| {
            let (wg, rows) = v.split_once(',')?;
            Some((wg.trim().parse().ok()?, rows.trim().parse().ok()?))
        })
        .filter(|&(wg, rows): &(u32, u32)| {
            wg % 64 == 0 && (64..=1024).contains(&wg) && (1..=4).contains(&rows)
        })
        .unwrap_or((64, 1))
}

/// Linhas de saída que um workgroup do GEMM cobre — o `BM` do tile de `mul_mm.comp`.
pub(crate) const GEMM_LINHAS_POR_WG: u32 = 128;

/// Se o prefill usa o GEMM com tiling em LDS (`mul_mm.comp`) nos pesos Q4_K.
///
/// **Ligado por padrão** desde a medição de 2026-08-21: com batch 24, o GEMM faz o
/// prefill em 10,8 ms/token contra 18,7 do matvec-COLS com batch 8 (−42 %) — ver a
/// tabela em `docs/prefill-em-batch.md`. `LLAMA_RS_PREFILL_GEMM=0` desliga para o A/B.
///
/// Só vale para Q4_K (53% do tempo de matvec no Qwen3.8-27B) e só com `n_batch` múltiplo
/// de 8 — é a largura do tile. Os demais tipos e larguras seguem no matvec-COLS.
fn gemm_prefill() -> bool {
    !std::env::var("LLAMA_RS_PREFILL_GEMM").is_ok_and(|v| v == "0")
}

/// Larguras de bloco que o tile do `mul_mm.comp` cobre: múltiplas de 8 (a grade de threads
/// tem 8 grupos de coluna) e no máximo 64 (acima disso os acumuladores por thread estouram).
fn gemm_largura_ok(cols: usize) -> bool {
    (8..=64).contains(&cols) && cols.is_multiple_of(8)
}

/// Se este matvec vai pelo GEMM em vez do matvec-COLS.
fn gemm_para(cols: usize, ty: gguf::GgmlType) -> bool {
    gemm_prefill() && matches!(ty, gguf::GgmlType::Q4_K) && gemm_largura_ok(cols)
}

/// Slot de KV-cache de cada camada local e o total de slots.
///
/// Só camada de atenção tem cache. As delta-net do qwen35 (três de cada quatro) guardam
/// um estado recorrente de tamanho fixo e não escrevem nada aqui — numerar o cache pelo
/// índice da camada reservaria 4× a memória: 17 GB contra 4,4 GB num ctx de 32k.
pub(crate) fn slots_kv(eh_atencao: impl IntoIterator<Item = bool>) -> (Vec<Option<usize>>, usize) {
    let mut total = 0usize;
    let slots = eh_atencao
        .into_iter()
        .map(|attn| {
            attn.then(|| {
                let slot = total;
                total += 1;
                slot
            })
        })
        .collect();
    (slots, total)
}

/// Geometria dos matvec K-quant: (lanes por workgroup, linhas de saída por wave).
///
/// Vira specialization constant nos shaders, então o compilador desenrola os laços sobre as
/// linhas e o par escolhido muda a pressão de registrador — que na MI50 decide a ocupância:
/// são 256 VGPRs por SIMD, e o Q5_K com 40 VGPRs cabe 6 waves em vez de 10. Menos linhas por
/// wave = menos acumuladores vivos = mais waves, ao custo de reler a ativação.
///
/// `LLAMA_RS_MATVEC_GEOM=wg,linhas` sobrescreve, para a varredura de `scripts/tune-matvec.sh`.
pub(crate) fn matvec_geom() -> (u32, u32) {
    std::env::var("LLAMA_RS_MATVEC_GEOM")
        .ok()
        .and_then(|v| {
            let (wg, rows) = v.split_once(',')?;
            Some((wg.trim().parse().ok()?, rows.trim().parse().ok()?))
        })
        .filter(|&(wg, rows): &(u32, u32)| {
            wg % 64 == 0 && (64..=1024).contains(&wg) && (1..=4).contains(&rows)
        })
        .unwrap_or((256, 2))
}

/// KiB de LDS **morta** alocados por workgroup do matvec Q4_K do decode, para limitar a
/// ocupância de propósito (specialization constant `LDS_PAD_KIB`).
///
/// Precedente: o backend Vulkan do llama.cpp faz o mesmo em GCN
/// (`ggml-vulkan.cpp:3767-3777`, comentário "*too many subgroups ... thrashing the cache*").
/// A conta de requests por superbloco no topo de `q4_k_matvec.comp` mostra que o kernel já
/// lê o mínimo possível, então o que resta para explicar os 506 GB/s dele contra os 573 do
/// Q6_K é waves demais disputando o L1.
///
/// `0` (padrão) mantém o kernel exatamente como está. Vale só para a pipeline do **decode**:
/// o bloco de prefill tem outra geometria (`matvec_geom_batch`, 1 wave por workgroup) e a
/// aritmética da tabela no shader não se aplica a ela.
///
/// **A medir**: nenhum valor foi comparado ainda neste hardware.
pub(crate) fn matvec_lds_pad() -> u32 {
    std::env::var("LLAMA_RS_MATVEC_LDS_PAD")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
        // Acima disso a criação da pipeline falharia por `maxComputeSharedMemorySize`.
        .min(48)
}

/// Se o RoPE de K escreve direto no slot do KV-cache (`rope_kv.comp`), dispensando a cópia
/// de K do `kv_append`. V continua indo pela cópia — ele não passa por RoPE.
///
/// Desligado por padrão: medido em 2026-08-21 (TOTAL GPU, 3 execuções por lado), a fusão
/// custa ~0,4 ms/token (40,6 contra 40,2 ms) — a escrita espalhada no cache grande perde
/// mais do que a cópia de `b_k` custa. `LLAMA_RS_ROPE_KV=1` religa para comparar no mesmo
/// binário.
fn rope_no_kv() -> bool {
    std::env::var("LLAMA_RS_ROPE_KV").is_ok_and(|v| v == "1")
}

/// Buffer Vulkan simples (device-local ou host-visible) com tamanho conhecido.
pub(crate) struct Buf {
    pub buffer: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Buf {
    /// Buffer device-local STORAGE | TRANSFER_SRC | TRANSFER_DST de `bytes`.
    fn device(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind, create_buf};
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind(ctx, phys, d, buffer, false)?;
        Ok(Self {
            buffer,
            mem,
            size: bytes,
        })
    }

    /// Buffer host-visible TRANSFER_SRC | TRANSFER_DST de `bytes` (upload/readback).
    fn host(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind, create_buf};
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind(ctx, phys, d, buffer, true)?;
        Ok(Self {
            buffer,
            mem,
            size: bytes,
        })
    }

    /// Buffer host-visible que a **CPU vai ler** (readback). A diferença para `host` é o
    /// bit HOST_CACHED — ver `tensor::alloc_and_bind_cached`.
    fn host_read(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind_cached, create_buf};
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind_cached(ctx, phys, d, buffer)?;
        Ok(Self {
            buffer,
            mem,
            size: bytes,
        })
    }

    fn destroy(&self, d: &ash::Device) {
        // SAFETY: handles criados por nós; chamado no Drop, sem uso concorrente.
        unsafe {
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.mem, None);
        }
    }
}

/// Faixa de camadas que uma GPU executa no layer-split, e o papel dela no pipeline.
///
/// Layer-split divide o modelo por **camadas** entre as GPUs: a GPU do primeiro shard roda
/// `0..split`, a do segundo roda `split..n_layer`. Entre elas passa apenas a stream residual
/// (`n_embd` floats), uma vez por token — contra as 96 sincronizacoes/token que o
/// tensor-parallel exigiria. É o que torna executáveis modelos que não cabem numa GPU.
#[derive(Clone, Copy, Debug)]
pub struct Shard {
    /// Índice do device Vulkan (posição em `amd_compute_devices`).
    pub device: usize,
    /// Primeira camada global deste shard.
    pub first_layer: usize,
    /// Uma além da última camada global deste shard.
    pub end_layer: usize,
    /// Total de camadas do modelo (para saber se este shard é o último).
    pub n_layer_total: usize,
}

impl Shard {
    /// Shard único cobrindo o modelo inteiro (caminho single-GPU).
    pub fn whole(device: usize, n_layer: usize) -> Self {
        Self {
            device,
            first_layer: 0,
            end_layer: n_layer,
            n_layer_total: n_layer,
        }
    }
    /// Faz o embedding lookup (só o primeiro shard).
    pub fn is_first(&self) -> bool {
        self.first_layer == 0
    }
    /// Faz a norma final e a projeção de logits (só o último shard).
    pub fn is_last(&self) -> bool {
        self.end_layer == self.n_layer_total
    }
    pub fn n_layers(&self) -> usize {
        self.end_layer - self.first_layer
    }
}

/// Escalares de arquitetura necessários ao decode.
pub(crate) struct Cfg {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rope_dim: usize,
    pub kv_dim: usize,
    pub vocab: usize,
    pub ctx: usize,
    pub rms_eps: f32,
    /// Faixa de camadas deste shard. `n_layer` acima é a contagem LOCAL.
    pub shard: Shard,
    /// `Some` nas arquiteturas híbridas (qwen35).
    pub delta_net: Option<llama_model::DeltaNetConfig>,
    /// Tokens por bloco do prefill em batch (ver `batch_size`). Dimensiona as ativações.
    pub n_batch: usize,
}

/// Pesos Q8_0 residentes de uma camada.
pub(crate) struct LayerQ {
    pub mixer: MixerQ,
    pub ffn_gate: QWeight,
    pub ffn_up: QWeight,
    pub ffn_down: QWeight,
}

/// Pesos quantizados do que fica entre as duas normas da camada, já na VRAM.
pub(crate) enum MixerQ {
    Attn {
        attn_q: QWeight,
        attn_k: QWeight,
        attn_v: QWeight,
        attn_output: QWeight,
    },
    /// Atenção linear do qwen35 — ver `docs/qwen35-arquitetura.md`.
    Delta {
        attn_qkv: QWeight,
        attn_gate: QWeight,
        ssm_out: QWeight,
    },
}

/// Peso residente em VRAM junto com o tipo, que decide qual shader de matvec usar.
pub(crate) struct QWeight {
    pub ty: gguf::GgmlType,
    pub gpu: GpuTensor,
}

/// Buffers f32 auxiliares residentes de uma camada.
pub(crate) struct LayerAux {
    pub attn_norm: Buf,
    pub ffn_norm: Buf,
    pub q_bias: Option<Buf>,
    pub k_bias: Option<Buf>,
    pub v_bias: Option<Buf>,
    /// QK-norm por cabeça (qwen35): [head_dim].
    pub q_norm: Option<Buf>,
    pub k_norm: Option<Buf>,
    /// Pesos e estado residente da atenção linear.
    pub delta: Option<DeltaBufs>,
}

/// Ativações intermediárias do caminho de atenção linear, compartilhadas por todas as
/// camadas do shard (uma camada por vez usa cada uma).
pub(crate) struct DnBufs {
    /// Projeção q|k|v antes e depois da convolução: [conv_dim].
    pub qkv: Buf,
    pub conv: Buf,
    /// `z`, o gate que fecha a norma da saída: [value_dim].
    pub z: Buf,
    /// (g, beta) por cabeça de valor: [n_v_heads * 2].
    pub gb: Buf,
    /// q e k já normalizados em L2: [key_dim] cada.
    pub qn: Buf,
    pub kn: Buf,
    /// Saída da recorrência e da norma gated: [value_dim].
    pub out: Buf,
    pub normed: Buf,
    /// Ativação quantizada própria do caminho linear. Separada da global para que a
    /// projeção de saída não dispute o buffer com o FFN da mesma camada.
    pub xq: Buf,
    pub xd: Buf,
}

/// Canais que a convolução causal cobre: q, k (com as cabeças de chave) e v.
pub(crate) fn conv_dim_de(dn: &llama_model::DeltaNetConfig) -> usize {
    dn.d_state * dn.n_k_heads * 2 + dn.head_v_dim() * dn.n_v_heads
}

/// Pesos f32 e **estado recorrente** de uma camada de atenção linear.
///
/// `estado` e `janela` são o que substitui o KV-cache: tamanho fixo, lidos e reescritos a
/// cada token. Ficam na mesma VRAM do shard, então acompanham o layer-split de graça.
pub(crate) struct DeltaBufs {
    pub conv1d: Buf,
    /// (ssm_a, dt_bias) por cabeça, empacotados como o `dn_gates` os consome.
    pub adt: Buf,
    pub alpha: Buf,
    pub beta: Buf,
    pub norm: Buf,
    /// [n_v_heads * d_state * d_state] — o histórico inteiro da sequência.
    pub estado: Buf,
    /// [conv_dim * (d_conv - 1)] — os tokens anteriores que a convolução ainda enxerga.
    pub janela: Buf,
    /// Cópias de `estado` e `janela` depois de cada token do verify menos o último —
    /// `snap[i]` guarda o estado **depois do token `i`** do bloco.
    ///
    /// É o que o rollback restaura quando uma proposta da cabeça MTP é rejeitada: manter
    /// `m` tokens é restaurar `snap[m - 1]`. Vazios sem MTP — são 3,2 MB por camada e
    /// ponto (310 MB total no Qwen3.8-27B), que não faz sentido reservar no caminho
    /// padrão.
    pub estado_snap: Vec<Buf>,
    pub janela_snap: Vec<Buf>,
}

/// Identifica qual pipeline um dispatch usa (resolvido em `pipe_of`).
#[derive(Clone, Copy)]
pub(crate) enum PipeId {
    Matvec,
    MatvecQ5K,
    MatvecQ6K,
    MatvecQ4K,
    /// As mesmas quatro pipelines com `COLS = n_batch`: só o plano de prefill as usa.
    MatvecB,
    MatvecQ5KB,
    MatvecQ6KB,
    MatvecQ4KB,
    /// GEMM Q4_K com tiling em LDS — experimental, ver [`gemm_prefill`].
    MulMmQ4K,
    /// As mesmas quatro com `COLS = VERIFY_TOK`: só o plano de verify do MTP as usa.
    MatvecV,
    MatvecQ5KV,
    MatvecQ6KV,
    MatvecQ4KV,
    QuantizeX,
    /// Residual + parciais da RMSNorm — ver `norm_fused.comp`.
    NormFused,
    /// Escala + quantização, fechando a redução do `NormFused`.
    NormP2,
    Rope,
    /// RoPE de K escrevendo direto no slot do KV-cache — `rope_kv.comp`.
    RopeKv,
    Attention,
    AttentionSplit,
    AttnReduce,
    /// silu(gate) * up + quantização em int8 na mesma passada — `swiglu_quant.comp`.
    SwigluQuant,
    Add,
    DeltaNet,
    DnConv,
    DnGates,
    DnNorm,
    /// L2 de q e k num dispatch só — o modo 0 do `dn_norm`, fundido.
    DnL2Qk,
    /// Portão sigmoide + quantização em int8 na mesma passada — `gate_quant.comp`.
    GateQuant,
}

impl PipeId {
    pub(crate) fn label(self) -> &'static str {
        match self {
            PipeId::Matvec => "matvec",
            PipeId::MatvecQ5K => "matvec_q5k",
            PipeId::MatvecQ6K => "matvec_q6k",
            PipeId::MatvecQ4K => "matvec_q4k",
            PipeId::MatvecB => "matvec_b",
            PipeId::MatvecQ5KB => "matvec_q5k_b",
            PipeId::MatvecQ6KB => "matvec_q6k_b",
            PipeId::MatvecQ4KB => "matvec_q4k_b",
            PipeId::MulMmQ4K => "mul_mm_q4k",
            PipeId::MatvecV => "matvec_v",
            PipeId::MatvecQ5KV => "matvec_q5k_v",
            PipeId::MatvecQ6KV => "matvec_q6k_v",
            PipeId::MatvecQ4KV => "matvec_q4k_v",
            PipeId::QuantizeX => "quantize_x",
            PipeId::NormFused => "norm_fused",
            PipeId::NormP2 => "norm_p2",
            PipeId::Rope => "rope",
            PipeId::RopeKv => "rope_kv",
            PipeId::Attention => "attention",
            PipeId::AttentionSplit => "attention_split",
            PipeId::AttnReduce => "attn_reduce",
            PipeId::SwigluQuant => "swiglu_quant",
            PipeId::Add => "add",
            PipeId::DeltaNet => "delta_net",
            PipeId::DnConv => "dn_conv",
            PipeId::DnGates => "dn_gates",
            PipeId::DnNorm => "dn_norm",
            PipeId::DnL2Qk => "dn_l2_qk",
            PipeId::GateQuant => "gate_quant",
        }
    }

    /// Índices dos bindings que o shader **lê** e dos que ele **escreve**. Um binding
    /// declarado `inout` no GLSL (sem `readonly`/`writeonly`) aparece nas duas listas.
    ///
    /// É o que permite decidir se dois dispatches vizinhos podem rodar concorrentes —
    /// ver `marcar_barreiras`. A tabela espelha os qualificadores dos `.comp`: trocar um
    /// `readonly` por `writeonly` lá obriga a mexer aqui, ou a barreira some e o resultado
    /// passa a depender do escalonamento.
    fn acessos(self) -> (&'static [usize], &'static [usize]) {
        match self {
            // weight, xq, xd → out
            // weight, xq, xd, bias → out
            PipeId::Matvec
            | PipeId::MatvecQ5K
            | PipeId::MatvecQ6K
            | PipeId::MatvecQ4K
            | PipeId::MatvecB
            | PipeId::MatvecQ5KB
            | PipeId::MatvecQ6KB
            | PipeId::MatvecQ4KB
            | PipeId::MulMmQ4K
            | PipeId::MatvecV
            | PipeId::MatvecQ5KV
            | PipeId::MatvecQ6KV
            | PipeId::MatvecQ4KV => (&[0, 1, 2, 4], &[3]),
            PipeId::Attention | PipeId::AttentionSplit => (&[0, 1, 2], &[3]),
            PipeId::AttnReduce => (&[0], &[1]),
            PipeId::QuantizeX => (&[0], &[1, 2]),
            // a saída é inout: a segunda passada relê o que a primeira escreveu.
            PipeId::SwigluQuant => (&[0, 1, 2], &[2, 3, 4]),
            // x é inout (recebe o residual); sai a soma parcial por workgroup.
            PipeId::NormFused => (&[0, 1], &[0, 2]),
            PipeId::NormP2 => (&[0, 1, 2], &[3, 4, 5]),
            // x é inout: o RoPE gira em cima do próprio buffer, o Add acumula nele.
            PipeId::Rope | PipeId::Add => (&[0, 1], &[0]),
            // k, freq → KV-cache. O binding do cache é o buffer inteiro, porque o offset
            // do slot só existe na gravação — ver `marcar_barreiras`.
            PipeId::RopeKv => (&[0, 1], &[2]),
            // o estado recorrente (binding 0) é lido e reescrito no mesmo dispatch.
            PipeId::DeltaNet => (&[0, 1, 2, 3, 4], &[0, 5]),
            PipeId::DnConv => (&[0, 1, 2], &[0, 3]),
            PipeId::DnGates => (&[0, 1, 2, 3], &[4]),
            PipeId::DnNorm => (&[0, 1, 2], &[3]),
            // conv (q|k contíguos) → qn, kn.
            PipeId::DnL2Qk => (&[0], &[1, 2]),
            // dst é inout (recebe o portão) e sai também quantizado em xq/xd.
            PipeId::GateQuant => (&[0, 1], &[0, 2, 3]),
        }
    }
}

/// Trecho de um buffer que uma op toca: (buffer, offset em bytes, bytes).
pub(crate) type Faixa = (vk::Buffer, vk::DeviceSize, vk::DeviceSize);

/// Duas faixas colidem quando são do mesmo buffer e os intervalos se cruzam.
fn sobrepoe(a: &Faixa, b: &Faixa) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

/// Como obter os bytes de push-constant de um dispatch no momento da gravação.
pub(crate) enum PushSpec {
    /// Push totalmente conhecido na construção do plano.
    Static(Vec<u8>),
    /// RoPE: precisa de `pos` na gravação. `n_head` e o passo entre cabeças são fixos.
    Rope { n_head: u32, stride: u32 },
    /// RoPE de K direto no cache: além de `pos`, precisa do offset do slot, que sai da
    /// posição do primeiro token do bloco. `kv_layer_off` é a parte fixa (slot × ctx × kv_dim).
    RopeKv { n_head: u32, kv_layer_off: u32 },
    /// Attention: precisa de `total_len`. `kv_layer_off` fixo.
    Attention { kv_layer_off: u32 },
    /// Redução dos parciais do split: precisa do número de fatias, escolhido na gravação
    /// a partir de `total_len`.
    AttnReduce,
}

/// Fecha um dispatch genérico do plano: pipeline, bindings, grupos e push constants.
type MkDispatch<'a> = dyn Fn(
        PipeId,
        &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
        u32,
        PushSpec,
    ) -> Result<PlannedOp, MatmulError>
    + 'a;

/// Matvec simples: peso quantizado, entrada, `n_in`, `n_out`.
type MkMatvec<'a> = dyn Fn(&QWeight, &Buf, usize, usize) -> Result<PlannedOp, MatmulError> + 'a;

/// Matvec cuja saída vai para um par de buffers.
type MkMatvecCom<'a> =
    dyn Fn(&QWeight, &Buf, (&Buf, &Buf), usize, usize) -> Result<PlannedOp, MatmulError> + 'a;

/// Uma op do token. `Dispatch` usa um descriptor set pré-escrito; as cópias não.
pub(crate) enum PlannedOp {
    Dispatch {
        pipe: PipeId,
        set: vk::DescriptorSet,
        groups: u32,
        /// Dimensão Y do dispatch: o token do bloco nos shaders que batcham por ela
        /// (`attention`, `rope`, `norm_fused`, `norm_p2`). 1 no plano de decode.
        groups_y: u32,
        push: PushSpec,
        /// Faixas de memória lidas e escritas, derivadas dos bindings e de `PipeId::acessos`.
        /// Só servem a `marcar_barreiras`, no build.
        le: Vec<Faixa>,
        esc: Vec<Faixa>,
        /// Bytes lidos por dispatch, para a coluna `GB/s` do perfil. `0` = não anotado.
        /// Derivado das faixas de leitura — ver o cálculo em `build_plan`.
        bytes: u64,
    },
    /// Embedding lookup: copia a linha de cada token do bloco de `token_embd` para `b_x`.
    Embed,
    /// Cópia device-to-device de um trecho fixo: o snapshot do estado recorrente entre os
    /// dois tokens do verify, e o staging do embedding no plano da cabeça MTP.
    Copia {
        src: vk::Buffer,
        dst: vk::Buffer,
        bytes: vk::DeviceSize,
    },
    /// Append do K e do V da camada ao KV-cache a partir da posição do primeiro token do
    /// bloco. As posições do bloco são consecutivas, então é uma cópia só.
    ///
    /// `com_k` é falso quando o `rope_kv` já escreveu K no slot (ver [`rope_no_kv`]) e só
    /// resta copiar V.
    KvAppend { slot: usize, com_k: bool },
    /// Append de K e V no KV-cache **próprio da cabeça MTP**, na posição corrente dela.
    /// O bloco tem um cache só (não é dividido em slots por camada) e sempre copia os
    /// dois: a cabeça usa o `rope` in-place, não o `rope_kv`, para não depender do knob.
    KvAppendMtp,
    /// Copia o hidden de um dos tokens do último passo (`b_x`) para o buffer da cabeça.
    /// **Qual** token é escolhido na gravação: depois de um verify aceito é o segundo,
    /// depois de um rejeitado (ou de um decode simples) é o primeiro.
    CopiaHidden,
    /// Atenção com dois caminhos prontos: um workgroup por cabeça (contexto curto) ou o
    /// KV fatiado entre workgroups mais a redução dos parciais (contexto longo).
    ///
    /// A escolha depende de `total_len`, que só existe na gravação do command buffer —
    /// por isso os dois ficam no plano e só um é gravado por token.
    Atencao {
        curto: Box<PlannedOp>,
        split: Box<PlannedOp>,
        reduce: Box<PlannedOp>,
    },
}

impl PlannedOp {
    /// Rótulo da op — o mesmo em `dbg_plano`, na timeline e na tabela do perfil.
    fn label(&self) -> &'static str {
        match self {
            PlannedOp::Dispatch { pipe, .. } => pipe.label(),
            PlannedOp::Embed => "embed",
            PlannedOp::Copia { .. } | PlannedOp::CopiaHidden => "copia",
            PlannedOp::KvAppend { .. } | PlannedOp::KvAppendMtp => "kv_append",
            PlannedOp::Atencao { .. } => "attention",
        }
    }
}

/// Em quantas fatias dividir o KV. 1 mantém o caminho de um workgroup por cabeça.
///
/// Com o KV curto a cadeia serial do `attention.comp` já é curta e a redução não se
/// paga; a partir de alguns milhares de posições ela domina o token (medido: 18 GB/s
/// efetivos contra ~500 GB/s dos matvec) e fatiar recupera a ocupância.
pub(crate) fn splits_do_kv(total_len: u32) -> u32 {
    let forcado = SPLITS_FORCADOS.load(std::sync::atomic::Ordering::Relaxed);
    if forcado > 0 {
        return forcado.min(MAX_SPLITS as u32);
    }
    /// Posições por fatia. Abaixo disso o workgroup nasce sem trabalho suficiente para
    /// pagar a escrita do parcial.
    const POR_FATIA: u32 = 512;
    (total_len / POR_FATIA).clamp(1, MAX_SPLITS as u32)
}

/// Sobrescreve o número de fatias (0 = automático). `LLAMA_RS_ATTN_SPLIT=N` no ambiente,
/// ou [`forcar_splits`] em teste — é o que permite comparar os dois caminhos no mesmo
/// processo e medir a diferença sem trocar de binário.
static SPLITS_FORCADOS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn forcar_splits(n: u32) {
    SPLITS_FORCADOS.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// Lê `LLAMA_RS_ATTN_SPLIT` uma vez, na construção do backend.
fn splits_do_ambiente() {
    if let Some(n) = std::env::var("LLAMA_RS_ATTN_SPLIT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        forcar_splits(n);
    }
}

/// Teto de fatias do KV: o buffer de parciais é dimensionado por ele.
pub(crate) const MAX_SPLITS: usize = 16;

/// Pesos e buffers próprios do bloco de multi-token prediction (`blk.{n_layer}.*`).
///
/// O bloco **é de atenção**, mesmo ocupando uma posição que a regra `eh_linear` diria ser
/// linear — o GGUF traz `attn_q/k/v/output` e nenhum `ssm_*`. Consequência prática: ele
/// não tem estado recorrente, só um KV-cache próprio de `kv_dim` por posição (8 KB no
/// Qwen3.8-27B), e por isso a cabeça não entra no rollback.
///
/// Quase todas as ativações são emprestadas do plano principal (`b_normed`, `b_q`, `b_k`,
/// `b_v`, `b_attn`, `b_proj`, `b_gate`, `b_up`, `b_xq`, `b_xd`, `b_parciais`, `b_logits`):
/// a cabeça roda **entre** dois passos do modelo, com a GPU ociosa e os logits do passo
/// anterior já lidos. O que precisa ser próprio é o que sobrevive ao passo (`kcache`,
/// `vcache`) ou o que seria destruído antes de ser lido (`b_h`, `b_eh`, `b_x`).
pub(crate) struct MtpBufs {
    pub eh_proj: QWeight,
    pub attn_q: QWeight,
    pub attn_k: QWeight,
    pub attn_v: QWeight,
    pub attn_output: QWeight,
    pub ffn_gate: QWeight,
    pub ffn_up: QWeight,
    pub ffn_down: QWeight,
    pub enorm: Buf,
    pub hnorm: Buf,
    pub shared_head_norm: Buf,
    pub attn_norm: Buf,
    pub ffn_norm: Buf,
    pub q_norm: Option<Buf>,
    pub k_norm: Option<Buf>,
    /// Linha crua da tabela de embedding do token amostrado, vinda do host. [n_embd]
    pub emb_stage: Buf,
    pub b_emb: Buf,
    /// Cópia do hidden do passo anterior. O plano principal deixa **os dois** hidden do
    /// verify em `b_x`, e qual deles a cabeça consome depende de aceitação ou rejeição —
    /// por isso a cópia entra no command buffer com o offset escolhido na gravação.
    pub b_h: Buf,
    /// `[enorm(emb) ; hnorm(h)]` — a concatenação é escrita nos offsets 0 e n_embd do
    /// mesmo buffer, então não existe op de concatenar. [2 * n_embd]
    pub b_eh: Buf,
    /// Stream residual do bloco. Separada de `b_x` do modelo, que ainda guarda o hidden.
    pub b_x: Buf,
    pub b_ffn: Buf,
    pub kcache: Buf,
    pub vcache: Buf,
    /// Posições já escritas no cache do bloco. Anda 1 por proposta, como a referência de
    /// CPU (`MtpHead::propor`) — o cache da cabeça é dela, não acompanha o do modelo.
    pub len: RefCell<usize>,
}

/// Todo o estado residente do modelo (pesos + aux + KV + ativações). `None` no
/// construtor de micro-teste `new_pipelines_only`; `Some` após `new`.
pub(crate) struct ResidentState<'w> {
    pub cfg: Cfg,
    /// Memória device-local dos pesos, em chunks. Os `GpuTensor` de `qw`/`output_w`
    /// apontam para pedaços dela e não têm memória própria — ver `GpuTensor::memory`.
    pub pesos_mem: crate::alloc::GpuAllocator,
    pub qw: Vec<LayerQ>,
    pub output_w: QWeight,
    pub aux: Vec<LayerAux>,
    pub output_norm_buf: Buf,
    pub freq_buf: Buf,
    /// Tabela de embedding **quantizada**, emprestada do GGUF. Manter em VRAM custaria
    /// vocab*n_embd*4 (3.1 GB no 14B) para ler **uma** linha por token, e dequantizá-la
    /// no host custaria a mesma coisa em RAM; a linha do token é dequantizada no passo e
    /// sobe por `embd_stage`, ao custo de poucos µs. `None` fora do primeiro shard.
    pub token_embd: Option<llama_model::TokenEmbd<'w>>,
    pub embd_stage: Buf,
    pub kcache: Buf,
    pub vcache: Buf,
    pub b_x: Buf,
    pub b_normed: Buf,
    /// Somas parciais dos quadrados, uma por workgroup do `NormFused` (ver `norm_fused.comp`).
    pub b_parciais: Buf,
    pub b_q: Buf,
    pub b_k: Buf,
    pub b_v: Buf,
    pub b_attn: Buf,
    /// Parciais (m, l, acc) da atenção com o KV fatiado — ver `splits_do_kv`.
    pub b_attn_split: Buf,
    pub b_proj: Buf,
    pub b_gate: Buf,
    pub b_up: Buf,
    pub b_act: Buf,
    pub b_logits: Buf,
    /// Saída do FFN. Separada de `b_proj` (saída do mixer) porque compartilhar as duas
    /// esconde de qual das duas veio um valor quando se lê o buffer para diagnóstico.
    pub b_ffn: Buf,
    /// Buffers do caminho de atenção linear (qwen35). `None` nas arquiteturas densas.
    pub dn: Option<DnBufs>,
    /// Ativacao quantizada em int8 (8 uints por bloco de 32) e escala por bloco,
    /// produzidas uma vez por matvec pelo dispatch `QuantizeX`.
    pub b_xq: Buf,
    pub b_xd: Buf,
    /// Staging host-visible dos logits, alocado uma vez. A copia entra no proprio
    /// command buffer do token: alocar 608 KB (`vkAllocateMemory`) e submeter um
    /// segundo command buffer a cada token custava mais que o readback em si.
    pub logits_host: Buf,
    /// Mapa persistente de `logits_host`. Mapear/desmapear por token custa dois ioctls no
    /// caminho crítico. Válido enquanto o `State` viver (unmap no Drop).
    pub logits_ptr: *mut std::ffi::c_void,
    pub len: RefCell<usize>,
    /// Cópia do `estado` e da `janela` de cada camada linear numa fronteira de turno, para
    /// que uma divergência depois dela não custe reprocessar o prompt inteiro — ver
    /// `marcar_snapshot`. Alocada no primeiro snapshot, não na carga: são ~155 MB de VRAM
    /// que só quem serve várias requisições sobre a mesma sessão usa. Fica vazia nas
    /// arquiteturas densas, onde não há nada recorrente e o snapshot é só o comprimento
    /// do KV.
    pub snap: RefCell<Vec<(Buf, Buf)>>,
    /// Comprimento do KV-cache no snapshot. `None` = nenhum snapshot válido.
    pub snap_len: std::cell::Cell<Option<usize>>,
    pub plan: Vec<PlannedOp>,
    /// Paralelo a `plan`: se a op precisa de uma barreira de memória **antes** dela.
    /// Calculado uma vez em `marcar_barreiras`.
    pub barreiras: Vec<bool>,
    /// O mesmo par para o bloco de `cfg.n_batch` tokens do prefill. Vazio quando
    /// `n_batch == 1`, e aí o prefill usa o plano de decode token a token.
    pub plan_batch: Vec<PlannedOp>,
    pub barreiras_batch: Vec<bool>,
    /// O mesmo par para o bloco de dois tokens do verify. Vazio sem MTP.
    pub plan_verify: Vec<PlannedOp>,
    pub barreiras_verify: Vec<bool>,
    /// Cabeça de multi-token prediction residente, quando o backend foi construído com
    /// MTP ligado — ver [`ResidentForward::new_shard_com`].
    pub mtp: Option<MtpBufs>,
    pub mtp_plan: Vec<PlannedOp>,
    pub mtp_barreiras: Vec<bool>,
    /// Command buffer gravado **uma vez** com as cópias de volta dos snapshots. O
    /// conteúdo é estático (origem, destino e tamanho não mudam), então a rejeição custa
    /// só um submit. `None` sem MTP ou sem camada de atenção linear.
    pub rollback_cmds: Vec<vk::CommandBuffer>,
    pub token_cmd: vk::CommandBuffer,
    pub token_fence: vk::Fence,
    /// Perfilamento por op via timestamp queries. `Some` só com LLAMA_RS_PROFILE=1.
    pub prof: Option<Prof>,
}

/// Acumulador de tempo de GPU por operação do plano, medido com timestamp queries.
/// Ativado por `LLAMA_RS_PROFILE=1`; fora disso nada é gravado no command buffer.
pub(crate) struct Prof {
    pub pool: vk::QueryPool,
    /// Nanossegundos de host por fase: gravacao do command buffer, submit+fence, leitura.
    pub host: RefCell<[u64; 3]>,
    /// Nanossegundos por tick do timestamp (VkPhysicalDeviceLimits::timestampPeriod).
    pub period_ns: f32,
    /// Nanossegundos acumulados por índice de op do plano.
    pub accum: RefCell<Vec<u64>>,
    pub tokens: std::cell::Cell<usize>,
    /// O mesmo para o plano do bloco de prefill, que tem outra lista de ops.
    pub accum_batch: RefCell<Vec<u64>>,
    pub blocos: std::cell::Cell<usize>,
    /// E o mesmo para o bloco de verify do MTP — terceira lista de ops, terceiro acumulador.
    pub accum_verify: RefCell<Vec<u64>>,
    pub verifies: std::cell::Cell<usize>,
    /// Zonas absolutas para a timeline (`--trace`). Vazio quando não se pede trace.
    pub spans: RefCell<Vec<GpuSpan>>,
    /// Limite de tokens gravados, para o arquivo não crescer sem fim.
    pub max_trace_tokens: usize,
}

/// Uma operação de GPU posicionada no relógio da CPU, para a timeline.
/// Qual shader de atenção linear rodar em `ResidentForward::dbg_dn`.
#[derive(Clone, Copy)]
pub enum DnPipe {
    DeltaNet,
    Conv,
    Gates,
    Norm,
    /// L2 de q e k fundida — `dn_l2_qk.comp`.
    L2Qk,
    /// Portão sigmoide + quantização fundidos — `gate_quant.comp`.
    GateQuant,
    /// SwiGLU + quantização fundidos — `swiglu_quant.comp`.
    SwigluQuant,
    /// RoPE de K escrevendo direto no slot do KV-cache — `rope_kv.comp`.
    RopeKv,
    /// Não é do delta net, mas entra aqui para poder ser testado com o mesmo helper.
    QuantizeX,
    /// Idem — os dois passos da norma fundida, para testar o batch pela dimensão Y.
    NormFused,
    NormP2,
}

#[derive(Clone, Debug)]
pub struct GpuSpan {
    pub name: &'static str,
    pub start: std::time::Instant,
    pub end: std::time::Instant,
}

/// Backend de decode GPU-resident (1 GPU). Construído via `ResidentForward::new`.
pub struct ResidentForward<'ctx> {
    pub(crate) ctx: &'ctx VulkanContext,
    pub(crate) phys_idx: usize,
    pub(crate) dev: VulkanDevice,
    // pipelines (preenchidos na Task 9; campos públicos ao crate para as tasks de teste)
    pub(crate) matvec: ComputePipeline,
    pub(crate) quantize_x: ComputePipeline,
    pub(crate) matvec_q5k: ComputePipeline,
    pub(crate) matvec_q6k: ComputePipeline,
    pub(crate) matvec_q4k: ComputePipeline,
    // As mesmas quatro com COLS = batch_size(), para o plano de prefill.
    pub(crate) matvec_b: ComputePipeline,
    pub(crate) matvec_q5k_b: ComputePipeline,
    pub(crate) matvec_q6k_b: ComputePipeline,
    pub(crate) matvec_q4k_b: ComputePipeline,
    /// GEMM Q4_K com tiling em LDS, para o prefill. Criada sempre; só entra no plano com
    /// `LLAMA_RS_PREFILL_GEMM=1` — ver [`gemm_prefill`].
    pub(crate) mul_mm_q4k: ComputePipeline,
    // As mesmas quatro com COLS = VERIFY_TOK, para o plano de verify do MTP.
    pub(crate) matvec_v: ComputePipeline,
    pub(crate) matvec_q5k_v: ComputePipeline,
    pub(crate) matvec_q6k_v: ComputePipeline,
    pub(crate) matvec_q4k_v: ComputePipeline,
    pub(crate) rmsnorm: ComputePipeline,
    pub(crate) norm_fused: ComputePipeline,
    pub(crate) norm_p2: ComputePipeline,
    pub(crate) rope: ComputePipeline,
    pub(crate) rope_kv: ComputePipeline,
    pub(crate) attention: ComputePipeline,
    /// Atenção com o KV fatiado entre workgroups + a redução dos parciais. Só entram
    /// com contexto longo, onde a cadeia serial do `attention` domina o token.
    pub(crate) attention_split: ComputePipeline,
    pub(crate) attn_reduce: ComputePipeline,
    pub(crate) swiglu_quant: ComputePipeline,
    pub(crate) add: ComputePipeline,
    // Camadas de atenção linear (qwen35).
    pub(crate) delta_net: ComputePipeline,
    pub(crate) dn_conv: ComputePipeline,
    pub(crate) dn_gates: ComputePipeline,
    pub(crate) dn_norm: ComputePipeline,
    pub(crate) dn_l2_qk: ComputePipeline,
    pub(crate) gate_quant: ComputePipeline,
    pub(crate) desc_pool: vk::DescriptorPool,
    pub(crate) state: Option<ResidentState<'ctx>>,
}

impl<'ctx> ResidentForward<'ctx> {
    pub(crate) fn phys(&self) -> &VulkanPhysicalDevice {
        &self.ctx.amd_compute_devices()[self.phys_idx]
    }

    /// Aloca um descriptor set do pool com o layout da pipeline dada.
    pub(crate) fn alloc_set(
        &self,
        pipe: &ComputePipeline,
    ) -> Result<vk::DescriptorSet, MatmulError> {
        let d = &self.dev.device;
        let info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &pipe.desc_set_layout,
            ..Default::default()
        };
        // SAFETY: d e pool válidos; layout vem da pipeline.
        Ok(unsafe { d.allocate_descriptor_sets(&info)? }[0])
    }

    /// Escreve `bindings` no `set` (na ordem 0..n) e despacha `pipe` com `push` bytes e
    /// `groups` workgroups em x. Faz 1 submit + 1 wait. Não lê de volta (dados ficam residentes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch1(
        &self,
        pipe: &ComputePipeline,
        set: vk::DescriptorSet,
        bindings: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)], // (buffer, offset, range)
        push: &[u8],
        groups: u32,
    ) -> Result<(), MatmulError> {
        self.dispatch_xy(pipe, set, bindings, push, groups, 1)
    }

    /// Igual a `dispatch1`, com a dimensão Y do dispatch exposta. Só a atenção usa Y > 1,
    /// para o prefill em batch: um workgroup por (cabeça, token do bloco).
    pub(crate) fn dispatch_xy(
        &self,
        pipe: &ComputePipeline,
        set: vk::DescriptorSet,
        bindings: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
        push: &[u8],
        groups: u32,
        groups_y: u32,
    ) -> Result<(), MatmulError> {
        self.dispatch_xyz(pipe, set, bindings, push, groups, groups_y, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_xyz(
        &self,
        pipe: &ComputePipeline,
        set: vk::DescriptorSet,
        bindings: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)], // (buffer, offset, range)
        push: &[u8],
        groups: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        let dev = &self.dev;

        let buf_infos: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .map(|&(buffer, offset, range)| vk::DescriptorBufferInfo {
                buffer,
                offset,
                range,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(b, bi)| vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: b as u32,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: bi,
                ..Default::default()
            })
            .collect();
        // SAFETY: d válido; writes apontam para buf_infos vivos; GPU ociosa (wait por op).
        unsafe { d.update_descriptor_sets(&writes, &[]) };

        let cb_info = vk::CommandBufferAllocateInfo {
            command_pool: dev.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: d e cmd_pool válidos.
        let cmd = unsafe { d.allocate_command_buffers(&cb_info)? }[0];
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            // SAFETY: cmd recém-alocado; pipe/set válidos; push tem o tamanho do range do layout.
            d.begin_command_buffer(cmd, &begin)?;
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                d.cmd_push_constants(cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            d.cmd_dispatch(cmd, groups, groups_y, groups_z);
            d.end_command_buffer(cmd)?;
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        unsafe {
            // SAFETY: queue/submit/cmd válidos.
            d.queue_submit(dev.queue, &[submit], vk::Fence::null())?;
            d.queue_wait_idle(dev.queue)?;
            d.free_command_buffers(dev.cmd_pool, &[cmd]);
        }
        Ok(())
    }

    /// Sobe `data` (f32) para o `dst` device-local via um staging host-visible próprio.
    pub(crate) fn upload_f32(&self, dst: &Buf, data: &[f32]) -> Result<(), MatmulError> {
        use crate::tensor::one_shot_copy;
        let d = &self.dev.device;
        let bytes = std::mem::size_of_val(data) as vk::DeviceSize;
        let staging = Buf::host(self.ctx, self.phys(), d, bytes)?;
        unsafe {
            // SAFETY: staging host-visible com `bytes`; ptr válido até unmap.
            let ptr = d.map_memory(staging.mem, 0, bytes, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len());
            d.unmap_memory(staging.mem);
        }
        one_shot_copy(
            d,
            self.dev.queue,
            self.dev.cmd_pool,
            staging.buffer,
            dst.buffer,
            bytes,
        )?;
        staging.destroy(d);
        Ok(())
    }

    /// Lê `len` f32 do `src` device-local de volta ao host.
    pub(crate) fn readback(&self, src: &Buf, len: usize) -> Result<Vec<f32>, MatmulError> {
        use crate::tensor::one_shot_copy;
        let d = &self.dev.device;
        let bytes = (len * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let host = Buf::host_read(self.ctx, self.phys(), d, bytes)?;
        one_shot_copy(
            d,
            self.dev.queue,
            self.dev.cmd_pool,
            src.buffer,
            host.buffer,
            bytes,
        )?;
        let out = unsafe {
            // SAFETY: host host-visible com `bytes`; ptr válido até unmap.
            let ptr = d.map_memory(host.mem, 0, bytes, vk::MemoryMapFlags::empty())?;
            let mut v = vec![0f32; len];
            std::ptr::copy_nonoverlapping(ptr as *const f32, v.as_mut_ptr(), len);
            d.unmap_memory(host.mem);
            v
        };
        host.destroy(d);
        Ok(out)
    }

    /// Constrói só device + pipelines + descriptor pool (sem pesos/buffers). Para micro-testes.
    /// Índice do device com mais VRAM livre; `LLAMA_RS_GPU` força um valor.
    ///
    /// A GPU que roda o display já tem ~1.6 GB ocupados, e num modelo que quase enche a
    /// VRAM o driver realoca o excedente em GTT (memória do host, via PCIe): medimos
    /// 95 GB/s no matvec do 14B na GPU do display contra 714 GB/s na outra.
    pub fn pick_device(ctx: &VulkanContext) -> usize {
        let phys = ctx.amd_compute_devices();
        std::env::var("LLAMA_RS_GPU")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|i| *i < phys.len())
            .or_else(|| {
                phys.iter()
                    .enumerate()
                    .filter_map(|(i, p)| p.free_device_memory(ctx).map(|f| (i, f)))
                    .max_by_key(|&(_, f)| f)
                    .map(|(i, _)| i)
            })
            .unwrap_or(0)
    }

    pub fn new_pipelines_only(ctx: &'ctx VulkanContext) -> Result<Self, MatmulError> {
        Self::new_pipelines_only_on(ctx, Self::pick_device(ctx))
    }

    pub fn new_pipelines_only_on(
        ctx: &'ctx VulkanContext,
        idx: usize,
    ) -> Result<Self, MatmulError> {
        let phys = ctx.amd_compute_devices();
        if phys.is_empty() {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED));
        }
        let dev = VulkanDevice::create(ctx, &phys[idx])?;
        let d = &dev.device;
        let matvec = ComputePipeline::new(d)?;
        // 3 bindings (x, xq, xd) + push de 12 bytes (n_in + 2 pads de alinhamento).
        let quantize_x = ComputePipeline::with(d, crate::QUANTIZE_X_SPV, 3, 12, &[])?;
        // Matvecs K-quant: os mesmos 4 bindings do Q8_0 (pesos, xq, xd, saída) e o mesmo push.
        let push_mv = std::mem::size_of::<crate::pipeline::PushConstants>() as u32;
        let (mv_wg, mv_rows) = matvec_geom();
        let geom = [(0, mv_wg), (1, mv_rows)];
        let matvec_q5k = ComputePipeline::with(d, crate::Q5_K_MATVEC_SPV, 5, push_mv, &geom)?;
        let matvec_q6k = ComputePipeline::with(d, crate::Q6_K_MATVEC_SPV, 5, push_mv, &[])?;
        // Mesma geometria tunada do Q5_K: a estrutura de acesso é idêntica (só sem o `qh`).
        // Mais a LDS morta de `matvec_lds_pad` (0 = desligada), que só este shader declara.
        let geom_q4k = [(0, mv_wg), (1, mv_rows), (3, matvec_lds_pad())];
        let matvec_q4k = ComputePipeline::with(d, crate::Q4_K_MATVEC_SPV, 5, push_mv, &geom_q4k)?;
        // Variantes de prefill: `COLS` é specialization constant, então cada largura de
        // batch é uma pipeline própria. O Q6_K expõe COLS no id 0 (geometria fixa no shader).
        let cols = batch_size() as u32;
        let (mvb_wg, mvb_rows) = matvec_geom_batch();
        let geom_b = [(0, mvb_wg), (1, mvb_rows), (2, cols)];
        let matvec_b = ComputePipeline::with(
            d,
            crate::Q8_0_MATVEC_SPV,
            5,
            push_mv,
            &[(0, MATVEC_WG), (1, MATVEC_NUM_ROWS), (2, cols)],
        )?;
        let matvec_q5k_b = ComputePipeline::with(d, crate::Q5_K_MATVEC_SPV, 5, push_mv, &geom_b)?;
        let matvec_q6k_b =
            ComputePipeline::with(d, crate::Q6_K_MATVEC_SPV, 5, push_mv, &[(0, cols)])?;
        let matvec_q4k_b = ComputePipeline::with(d, crate::Q4_K_MATVEC_SPV, 5, push_mv, &geom_b)?;
        // `COLS` do GEMM é a largura do bloco quando ele está ligado; com o knob desligado
        // (ou a largura fora do tile) vale 8, a menor válida, só para a pipeline compilar.
        // O plano só emite `MulMmQ4K` quando `gemm_para` aceita — ver `mv_gen`.
        let gemm_cols = if gemm_prefill() && gemm_largura_ok(cols as usize) {
            cols
        } else {
            8
        };
        let mul_mm_q4k =
            ComputePipeline::with(d, crate::MUL_MM_SPV, 5, push_mv, &[(0, gemm_cols)])?;
        // Verify do MTP: `COLS = 2` fixo, com a geometria do **decode** e não a do prefill.
        // O que decide a ocupância é `ROWS_PER_WAVE * COLS` acumuladores vivos por lane, e
        // com duas colunas isso fica perto do decode (COLS=1) e longe das oito do bloco de
        // prefill — ver `matvec_geom_batch`. Pendente de medição no modelo real.
        let geom_v = [(0, mv_wg), (1, mv_rows), (2, VERIFY_TOK as u32)];
        let matvec_v = ComputePipeline::with(
            d,
            crate::Q8_0_MATVEC_SPV,
            5,
            push_mv,
            &[(0, MATVEC_WG), (1, MATVEC_NUM_ROWS), (2, VERIFY_TOK as u32)],
        )?;
        let matvec_q5k_v = ComputePipeline::with(d, crate::Q5_K_MATVEC_SPV, 5, push_mv, &geom_v)?;
        let matvec_q6k_v = ComputePipeline::with(
            d,
            crate::Q6_K_MATVEC_SPV,
            5,
            push_mv,
            &[(0, VERIFY_TOK as u32)],
        )?;
        // Mesma geometria do decode ⇒ mesma aritmética de LDS morta (ver `matvec_lds_pad`).
        let geom_q4k_v = [
            (0, mv_wg),
            (1, mv_rows),
            (2, VERIFY_TOK as u32),
            (3, matvec_lds_pad()),
        ];
        let matvec_q4k_v =
            ComputePipeline::with(d, crate::Q4_K_MATVEC_SPV, 5, push_mv, &geom_q4k_v)?;
        let rmsnorm = ComputePipeline::with(d, crate::RMSNORM_SPV, 3, 8, &[])?; // dim:u32 + eps:f32
        // dim:u32 + tem_residual:u32
        let norm_fused = ComputePipeline::with(d, crate::NORM_FUSED_SPV, 3, 8, &[])?;
        // dim:u32 + eps:f32 + n_parciais:u32
        let norm_p2 = ComputePipeline::with(d, crate::NORM_P2_SPV, 6, 12, &[])?;
        let rope = ComputePipeline::with(d, crate::ROPE_SPV, 2, 20, &[])?;
        // (k, freq, kcache) + n_head, head_dim, rope_dim, pos, kv_off.
        let rope_kv = ComputePipeline::with(d, crate::ROPE_KV_SPV, 3, 20, &[])?;
        let attention = ComputePipeline::with(d, crate::ATTENTION_SPV, 4, 28, &[])?;
        let attention_split = ComputePipeline::with(d, crate::ATTENTION_SPLIT_SPV, 4, 28, &[])?;
        let attn_reduce = ComputePipeline::with(d, crate::ATTN_REDUCE_SPV, 2, 12, &[])?;
        // (gate, up, act inout, xq, xd) + n.
        let swiglu_quant = ComputePipeline::with(d, crate::SWIGLU_QUANT_SPV, 5, 4, &[])?;
        let add = ComputePipeline::with(d, crate::ADD_SPV, 2, 4, &[])?;
        // Atenção linear: (estado, q, k, v, g|beta, saída), (estado, x, w, saída),
        // (x, alpha, beta, a|dt, saída) e (x, w, z, saída).
        // O delta net leva ainda `n_tok` e o passo de `v` entre tokens: o laço do bloco
        // roda dentro do kernel, com o estado em registrador.
        let delta_net = ComputePipeline::with(d, crate::DELTA_NET_SPV, 6, 20, &[])?;
        let dn_conv = ComputePipeline::with(d, crate::DN_CONV_SPV, 4, 12, &[])?;
        let dn_gates = ComputePipeline::with(d, crate::DN_GATES_SPV, 5, 12, &[])?;
        let dn_norm = ComputePipeline::with(d, crate::DN_NORM_SPV, 4, 20, &[])?;
        // (conv, qn, kn) + dim, n_heads, eps, stride.
        let dn_l2_qk = ComputePipeline::with(d, crate::DN_L2_QK_SPV, 3, 16, &[])?;
        // (dst inout, gate, xq, xd) + n, head_dim.
        let gate_quant = ComputePipeline::with(d, crate::GATE_QUANT_SPV, 4, 8, &[])?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 65536,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            // Um descriptor set por op do plano. O 14B tem ~1100 ops/token (48 camadas
            // x ~23 ops), e cada matvec usa 4 bindings — o pool anterior (1024 sets)
            // estourava com ERROR_OUT_OF_POOL_MEMORY.
            max_sets: 16384,
            pool_size_count: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        // SAFETY: d válido; pool_info aponta para dados vivos.
        let desc_pool = unsafe { d.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            ctx,
            phys_idx: idx,
            dev,
            matvec,
            quantize_x,
            matvec_q5k,
            matvec_q6k,
            matvec_q4k,
            matvec_b,
            matvec_q5k_b,
            matvec_q6k_b,
            matvec_q4k_b,
            mul_mm_q4k,
            matvec_v,
            matvec_q5k_v,
            matvec_q6k_v,
            matvec_q4k_v,
            rmsnorm,
            norm_fused,
            norm_p2,
            rope,
            rope_kv,
            attention,
            attention_split,
            attn_reduce,
            swiglu_quant,
            add,
            delta_net,
            dn_conv,
            dn_gates,
            dn_norm,
            dn_l2_qk,
            gate_quant,
            desc_pool,
            state: None,
        })
    }

    /// Constrói o backend GPU-resident: sobe todos os pesos (Q8_0 + aux f32) e aloca
    /// as ativações e o KV-cache em VRAM. Após retornar, `raw`/`aux` podem ser descartados.
    /// Backend cobrindo o modelo inteiro numa GPU (escolhida por VRAM livre).
    pub fn new(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'ctx>,
    ) -> Result<Self, MatmulError> {
        let dev = Self::pick_device(ctx);
        Self::new_shard(ctx, config, raw, aux, Shard::whole(dev, config.n_layer))
    }

    /// Como [`Self::new`], com o multi-token prediction ligado (ver [`Self::new_shard_com`]).
    pub fn new_com(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'ctx>,
        mtp: bool,
    ) -> Result<Self, MatmulError> {
        let dev = Self::pick_device(ctx);
        Self::new_shard_com(
            ctx,
            config,
            raw,
            aux,
            Shard::whole(dev, config.n_layer),
            mtp,
        )
    }

    /// Backend cobrindo apenas `shard.first_layer..shard.end_layer`, no device do shard.
    /// Só o primeiro shard faz embedding; só o último faz a norma final e os logits.
    pub fn new_shard(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'ctx>,
        shard: Shard,
    ) -> Result<Self, MatmulError> {
        Self::new_shard_com(ctx, config, raw, aux, shard, false)
    }

    /// Como [`Self::new_shard`], com o **multi-token prediction** opcionalmente ligado.
    ///
    /// Ligar custa VRAM que o caminho padrão não deve pagar: os snapshots do estado
    /// recorrente (3,2 MB por camada linear, ~155 MB no Qwen3.8-27B), o KV-cache próprio da
    /// cabeça (8 KB por posição) e os 289 MB de pesos do bloco. Daí ser um construtor
    /// separado, e não uma flag lida do ambiente: quem liga é a `--mtp` do CLI ou a config
    /// do servidor, e o resto do repositório continua construindo o backend como antes.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "subir_pesos")
    )]
    pub fn new_shard_com(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'ctx>,
        shard: Shard,
        mtp: bool,
    ) -> Result<Self, MatmulError> {
        if config.head_dim > 256 {
            // Shader de attention distribui head_dim entre 64 lanes com no máximo
            // MAX_DPL=4 dimensões por lane.
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        // O matvec faz tiling da dimensão K em janelas de MATVEC_MAX_BLOCKS blocos,
        // então n_in é livre. (Antes havia um limite de n_in <= MATVEC_MAX_BLOCKS*32.)
        let mut me = Self::new_pipelines_only_on(ctx, shard.device)?;
        let kv_dim = config.n_head_kv * config.head_dim;

        // Bloco que constrói todo o estado residente emprestando `me` imutavelmente;
        // ao final o `state` é movido para fora (sem borrows de `me`), e só então
        // `me.state = Some(state)` é atribuído.
        let state = {
            let phys = me.phys();
            let dev_ref = &me.dev;
            let d = &dev_ref.device;

            splits_do_ambiente();
            let cfg = Cfg {
                n_embd: config.n_embd,
                n_layer: shard.n_layers(),
                n_head: config.n_head,
                n_head_kv: config.n_head_kv,
                head_dim: config.head_dim,
                n_ff: config.n_ff,
                rope_dim: config.rope_dim,
                kv_dim,
                vocab: config.vocab,
                ctx: config.ctx,
                rms_eps: config.rms_eps,
                shard,
                delta_net: config.delta_net.clone(),
                n_batch: batch_size(),
            };
            let nbatch = cfg.n_batch;

            // Teto do que os pesos deste shard ocupam na VRAM: o repack Q8_0 é o que mais
            // cresce (34→36 B), então `cru × 36/34` cobre todos os tipos. É só a dica que
            // dimensiona o último chunk do alocador — errar para menos custa um chunk
            // extra do tamanho exato, não uma falha.
            let crus: usize = raw.layers[shard.first_layer..shard.end_layer]
                .iter()
                .map(llama_model::GpuLayerRaw::bytes_totais)
                .sum::<usize>()
                + raw.output.bytes.len();
            let vram_pesos = (crus as vk::DeviceSize).div_ceil(34) * 36;
            let mut upl = crate::tensor::Uploader::new(
                ctx,
                phys,
                dev_ref,
                vram_pesos,
                &format!("GPU{}", shard.device),
            )?;

            // O uploader é passado explicitamente porque `up_q` e `mk` o mutam: duas
            // closures capturando o mesmo `&mut` não coexistem.
            let up_q = |u: &mut crate::tensor::Uploader<'_>,
                        t: &llama_model::QTensor<'_>,
                        n_in: usize,
                        n_out: usize|
             -> Result<QWeight, MatmulError> {
                let gpu = u.tensor(t.ty, t.bytes, n_in, n_out)?;
                Ok(QWeight { ty: t.ty, gpu })
            };
            let mut qw = Vec::with_capacity(cfg.n_layer);
            for lw in &raw.layers[shard.first_layer..shard.end_layer] {
                let mixer = match &lw.mixer {
                    llama_model::MixerRaw::Attn {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                    } => {
                        // No qwen35 a projeção de Q sai dobrada (query|gate) e a saída da
                        // atenção entra com head_dim × n_head, que não é n_embd.
                        let (q_out, o_in) = match config.delta_net.as_ref() {
                            Some(_) => (
                                config.head_dim * config.n_head * 2,
                                config.head_dim * config.n_head,
                            ),
                            None => (config.n_embd, config.n_embd),
                        };
                        MixerQ::Attn {
                            attn_q: up_q(&mut upl, attn_q, cfg.n_embd, q_out)?,
                            attn_k: up_q(&mut upl, attn_k, cfg.n_embd, kv_dim)?,
                            attn_v: up_q(&mut upl, attn_v, cfg.n_embd, kv_dim)?,
                            attn_output: up_q(&mut upl, attn_output, o_in, cfg.n_embd)?,
                        }
                    }
                    llama_model::MixerRaw::Delta {
                        attn_qkv,
                        attn_gate,
                        ssm_out,
                    } => {
                        let dn = config
                            .delta_net
                            .as_ref()
                            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;
                        let value_dim = dn.head_v_dim() * dn.n_v_heads;
                        MixerQ::Delta {
                            attn_qkv: up_q(&mut upl, attn_qkv, cfg.n_embd, conv_dim_de(dn))?,
                            attn_gate: up_q(&mut upl, attn_gate, cfg.n_embd, value_dim)?,
                            ssm_out: up_q(&mut upl, ssm_out, value_dim, cfg.n_embd)?,
                        }
                    }
                };
                qw.push(LayerQ {
                    mixer,
                    ffn_gate: up_q(&mut upl, &lw.ffn_gate, cfg.n_embd, cfg.n_ff)?,
                    ffn_up: up_q(&mut upl, &lw.ffn_up, cfg.n_embd, cfg.n_ff)?,
                    ffn_down: up_q(&mut upl, &lw.ffn_down, cfg.n_ff, cfg.n_embd)?,
                });
            }
            let output_w = up_q(&mut upl, &raw.output, cfg.n_embd, cfg.vocab)?;

            // Auxiliares f32 (normas, tabela de frequências, estado inicial): buffer
            // device-local próprio, mas a cópia entra na mesma fila de lotes dos pesos —
            // eram ~450 fences de 20 KB cada, um por buffer.
            let mk =
                |u: &mut crate::tensor::Uploader<'_>, data: &[f32]| -> Result<Buf, MatmulError> {
                    let bytes_val = std::mem::size_of_val(data) as vk::DeviceSize;
                    let b = Buf::device(ctx, phys, d, bytes_val)?;
                    // SAFETY: `f32` não tem padding nem invariantes de bit; o slice de bytes
                    // vive só até o fim desta chamada, e `bytes_para` copia na hora.
                    let brutos = unsafe {
                        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), bytes_val as usize)
                    };
                    u.bytes_para(b.buffer, brutos)?;
                    Ok(b)
                };
            let mk_opt = |u: &mut crate::tensor::Uploader<'_>,
                          o: &Option<Vec<f32>>|
             -> Result<Option<Buf>, MatmulError> {
                match o {
                    Some(v) => Ok(Some(mk(u, v)?)),
                    None => Ok(None),
                }
            };
            let fase_aux = llama_model::perfil_carga::Fase::nova("aux → VRAM");
            let mut aux_buf = Vec::with_capacity(cfg.n_layer);
            for al in &aux.layers[shard.first_layer..shard.end_layer] {
                aux_buf.push(LayerAux {
                    attn_norm: mk(&mut upl, &al.attn_norm)?,
                    ffn_norm: mk(&mut upl, &al.ffn_norm)?,
                    q_bias: mk_opt(&mut upl, &al.q_bias)?,
                    k_bias: mk_opt(&mut upl, &al.k_bias)?,
                    v_bias: mk_opt(&mut upl, &al.v_bias)?,
                    q_norm: mk_opt(&mut upl, &al.q_norm)?,
                    k_norm: mk_opt(&mut upl, &al.k_norm)?,
                    delta: match (&al.delta, config.delta_net.as_ref()) {
                        (Some(da), Some(dn)) => {
                            // (ssm_a, dt_bias) intercalados: o `dn_gates` lê os dois de
                            // uma cabeça num vec2.
                            let adt: Vec<f32> = (0..dn.n_v_heads)
                                .flat_map(|h| [da.a[h], da.dt_bias[h]])
                                .collect();
                            let janela_len = conv_dim_de(dn) * (dn.d_conv - 1);
                            // Os snapshots só existem com MTP: 3,2 MB por camada e ponto
                            // que o caminho padrão nunca leria. Um ponto por token do
                            // verify menos o último.
                            let snap = |n: usize| -> Result<Vec<Buf>, MatmulError> {
                                if mtp {
                                    (0..VERIFY_TOK - 1)
                                        .map(|_| {
                                            Buf::device(ctx, phys, d, (n * 4) as vk::DeviceSize)
                                        })
                                        .collect()
                                } else {
                                    Ok(Vec::new())
                                }
                            };
                            Some(DeltaBufs {
                                conv1d: mk(&mut upl, &da.conv1d)?,
                                adt: mk(&mut upl, &adt)?,
                                alpha: mk(&mut upl, &da.alpha)?,
                                beta: mk(&mut upl, &da.beta)?,
                                norm: mk(&mut upl, &da.norm)?,
                                // Estado recorrente e janela da convolução começam
                                // zerados — é o "contexto vazio" desta arquitetura.
                                estado: mk(&mut upl, &vec![0f32; dn.state_len()])?,
                                janela: mk(&mut upl, &vec![0f32; janela_len])?,
                                estado_snap: snap(dn.state_len())?,
                                janela_snap: snap(janela_len)?,
                            })
                        }
                        _ => None,
                    },
                });
            }
            let output_norm_buf = mk(&mut upl, &aux.output_norm)?;
            let freq_buf = mk(&mut upl, &aux.freq_table)?;
            drop(fase_aux);
            // Largura máxima de bloco que as ativações precisam cobrir. Com MTP o plano de
            // verify tem dois tokens, e `LLAMA_RS_BATCH=1` deixaria os buffers apertados
            // para ele.
            let nblk = if mtp { nbatch.max(VERIFY_TOK) } else { nbatch };
            let embd_stage = Buf::host(ctx, phys, d, (config.n_embd * nblk * 4) as vk::DeviceSize)?;

            // Só as camadas de atenção têm KV-cache: no qwen35 as outras três de cada
            // quatro são delta-net, com estado recorrente de tamanho fixo. Reservar por
            // camada global custaria 4× — ver `slots_kv`.
            let (_, n_slots_kv) =
                slots_kv(qw.iter().map(|l| !matches!(l.mixer, MixerQ::Delta { .. })));
            let kv_elems = (n_slots_kv * cfg.ctx * kv_dim) as vk::DeviceSize;
            let kcache = Buf::device(ctx, phys, d, kv_elems * 4)?;
            let vcache = Buf::device(ctx, phys, d, kv_elems * 4)?;

            let attn_dim = if config.delta_net.is_some() {
                config.head_dim * config.n_head
            } else {
                config.n_embd
            };
            // Query e gate juntos quando há QK-norm/gate (qwen35).
            let q_dim = if config.delta_net.is_some() {
                attn_dim * 2
            } else {
                config.n_embd
            };
            // Ativações: `nf` para o que é por token (dimensionado para o bloco inteiro do
            // prefill) e `nf1` para o que é por sequência.
            let nf = |n: usize| -> Result<Buf, MatmulError> {
                Buf::device(ctx, phys, d, (n * nblk * 4) as vk::DeviceSize)
            };
            let nf1 = |n: usize| -> Result<Buf, MatmulError> {
                Buf::device(ctx, phys, d, (n * 4) as vk::DeviceSize)
            };
            // Colunas de logits: 1 no decode e no prefill, 2 no verify (os dois tokens).
            let cols_logits = if mtp { VERIFY_TOK } else { 1 };

            // Cabeça MTP: só no shard que tem a norma final, porque é dele o hidden que ela
            // combina com o embedding. Assim ela roda inteira numa GPU, sem tráfego extra.
            let mtp_bufs = match (mtp && shard.is_last(), raw.mtp.as_ref(), aux.mtp.as_ref()) {
                (true, Some(mraw), Some(maux)) => {
                    let llama_model::MixerRaw::Attn {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                    } = &mraw.layer.mixer
                    else {
                        // `MtpRaw` só é montado com mixer de atenção — ver `gpu.rs`.
                        return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
                    };
                    let q_out = if config.delta_net.is_some() {
                        attn_dim * 2
                    } else {
                        config.n_embd
                    };
                    let kv_pos = (config.ctx * kv_dim * 4) as vk::DeviceSize;
                    Some(MtpBufs {
                        eh_proj: up_q(&mut upl, &mraw.eh_proj, config.n_embd * 2, config.n_embd)?,
                        attn_q: up_q(&mut upl, attn_q, config.n_embd, q_out)?,
                        attn_k: up_q(&mut upl, attn_k, config.n_embd, kv_dim)?,
                        attn_v: up_q(&mut upl, attn_v, config.n_embd, kv_dim)?,
                        attn_output: up_q(&mut upl, attn_output, attn_dim, config.n_embd)?,
                        ffn_gate: up_q(&mut upl, &mraw.layer.ffn_gate, config.n_embd, config.n_ff)?,
                        ffn_up: up_q(&mut upl, &mraw.layer.ffn_up, config.n_embd, config.n_ff)?,
                        ffn_down: up_q(&mut upl, &mraw.layer.ffn_down, config.n_ff, config.n_embd)?,
                        enorm: mk(&mut upl, &maux.enorm)?,
                        hnorm: mk(&mut upl, &maux.hnorm)?,
                        shared_head_norm: mk(&mut upl, &maux.shared_head_norm)?,
                        attn_norm: mk(&mut upl, &maux.layer.attn_norm)?,
                        ffn_norm: mk(&mut upl, &maux.layer.ffn_norm)?,
                        q_norm: mk_opt(&mut upl, &maux.layer.q_norm)?,
                        k_norm: mk_opt(&mut upl, &maux.layer.k_norm)?,
                        emb_stage: Buf::host(ctx, phys, d, (config.n_embd * 4) as vk::DeviceSize)?,
                        b_emb: nf1(config.n_embd)?,
                        b_h: nf1(config.n_embd)?,
                        b_eh: nf1(config.n_embd * 2)?,
                        b_x: nf1(config.n_embd)?,
                        b_ffn: nf1(config.n_embd)?,
                        kcache: Buf::device(ctx, phys, d, kv_pos)?,
                        vcache: Buf::device(ctx, phys, d, kv_pos)?,
                        len: RefCell::new(0),
                    })
                }
                _ => None,
            };
            // Fecha o último lote e espera a GPU: daqui em diante todo peso está na VRAM
            // (inclusive os da cabeça MTP, que entram no mesmo lote). A memória dos chunks
            // passa a ser do estado, e os dois staging morrem aqui.
            let pesos_mem = upl.finalizar()?;

            let cb_info = vk::CommandBufferAllocateInfo {
                command_pool: dev_ref.cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            // SAFETY: device/pool válidos; pool tem RESET_COMMAND_BUFFER.
            let token_cmd = unsafe { dev_ref.device.allocate_command_buffers(&cb_info)? }[0];
            let token_fence = unsafe {
                dev_ref
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?
            };

            // Saída deste shard: logits no último, stream residual nos demais. Mapeada uma
            // única vez — ver `logits_ptr`.
            // Nos shards intermediários a stream residual sai com um vetor por token do
            // bloco; no último só os logits do último token do bloco interessam.
            let saida_floats = if shard.is_last() {
                config.vocab * cols_logits
            } else {
                config.n_embd * nblk
            };
            let logits_host = Buf::host_read(ctx, phys, d, (saida_floats * 4) as vk::DeviceSize)?;
            // SAFETY: memória host-visible recém-criada com esse tamanho, ainda não mapeada.
            let logits_ptr = unsafe {
                d.map_memory(
                    logits_host.mem,
                    0,
                    logits_host.size,
                    vk::MemoryMapFlags::empty(),
                )?
            };

            ResidentState {
                cfg,
                pesos_mem,
                qw,
                output_w,
                aux: aux_buf,
                output_norm_buf,
                freq_buf,
                // Só o primeiro shard faz o embedding lookup.
                token_embd: shard.is_first().then_some(aux.token_embd),
                embd_stage,
                kcache,
                vcache,
                b_x: nf(config.n_embd)?,
                b_normed: nf(config.n_embd)?,
                // NORM_P1_WG é o teto de workgroups do passo 1, então basta esse tanto de floats.
                b_parciais: nf(NORM_P1_WG as usize)?,
                // No qwen35 a projeção de Q sai com query **e** gate por cabeça, e o
                // conjunto de cabeças não tem exatamente n_embd (24 × 256 = 6144 contra
                // 5120), então os buffers da atenção seguem head_dim × n_head.
                b_q: nf(q_dim)?,
                b_k: nf(kv_dim)?,
                b_v: nf(kv_dim)?,
                b_attn: nf(attn_dim)?,
                // Parciais da atenção fatiada: por (token, cabeça, fatia) um registro
                // [m, l, acc[head_dim]]. Dimensionado pelo teto de `splits_do_kv`.
                b_attn_split: nf(config.n_head * MAX_SPLITS * (config.head_dim + 2))?,
                b_proj: nf(config.n_embd)?,
                b_gate: nf(config.n_ff)?,
                b_up: nf(config.n_ff)?,
                b_act: nf(config.n_ff)?,
                // Só o último token do bloco chega à projeção de logits — ver `build_plan`.
                // Com MTP são duas colunas: o verify precisa dos logits dos dois tokens.
                b_logits: nf1(config.vocab * cols_logits)?,
                b_ffn: nf(config.n_embd)?,
                b_xq: nf(config.n_embd.max(config.n_ff) / 32 * 8)?,
                b_xd: nf(config.n_embd.max(config.n_ff) / 32)?,
                dn: match config.delta_net.as_ref() {
                    Some(dn) => {
                        let key_dim = dn.d_state * dn.n_k_heads;
                        let value_dim = dn.head_v_dim() * dn.n_v_heads;
                        let cd = conv_dim_de(dn);
                        Some(DnBufs {
                            qkv: nf(cd)?,
                            conv: nf(cd)?,
                            z: nf(value_dim)?,
                            gb: nf(dn.n_v_heads * 2)?,
                            qn: nf(key_dim)?,
                            kn: nf(key_dim)?,
                            out: nf(value_dim)?,
                            normed: nf(value_dim)?,
                            xq: nf(value_dim / 32 * 8)?,
                            xd: nf(value_dim / 32)?,
                        })
                    }
                    None => None,
                },
                // Saída deste shard: logits no último, stream residual nos demais.
                logits_host,
                logits_ptr,
                len: RefCell::new(0),
                snap: RefCell::new(Vec::new()),
                snap_len: std::cell::Cell::new(None),
                plan: Vec::new(),
                barreiras: Vec::new(),
                plan_batch: Vec::new(),
                barreiras_batch: Vec::new(),
                plan_verify: Vec::new(),
                barreiras_verify: Vec::new(),
                mtp: mtp_bufs,
                mtp_plan: Vec::new(),
                mtp_barreiras: Vec::new(),
                rollback_cmds: Vec::new(),
                token_cmd,
                token_fence,
                prof: None,
            }
        };

        me.state = Some(state);
        let plan = me.build_plan(Modo::Decode)?;
        // Plano do bloco de prefill. Mesmo código, `n_tok` colunas: os matvec passam a ler
        // cada peso uma vez para os N tokens, que é o ganho do batch.
        let nbatch = batch_size();
        let plan_batch = if nbatch > 1 {
            me.build_plan(Modo::Batch)?
        } else {
            Vec::new()
        };
        // Plano da verificação do MTP: dois tokens, logits dos dois, snapshot do estado
        // recorrente entre eles.
        let plan_verify = if mtp {
            me.build_plan(Modo::Verify)?
        } else {
            Vec::new()
        };
        let plan_mtp = me.build_plan_mtp()?;
        // Perfilamento opcional: 1 timestamp antes do plano + 1 depois de cada op, então
        // o pool é dimensionado pelo **maior** dos planos — o do bloco de prefill tem
        // mais ops que o do decode, porque a recorrência do delta-net vira um dispatch por
        // token do bloco.
        let prof = if std::env::var("LLAMA_RS_PROFILE").is_ok_and(|v| v != "0") {
            let maior = plan.len().max(plan_batch.len()).max(plan_verify.len());
            let info = vk::QueryPoolCreateInfo {
                query_type: vk::QueryType::TIMESTAMP,
                query_count: u32::try_from(maior + 1)
                    .map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?,
                ..Default::default()
            };
            // SAFETY: device válido; info preenchido nesta stack frame.
            let pool = unsafe { me.dev.device.create_query_pool(&info, None)? };
            // SAFETY: instance e handle válidos (enumerados por VulkanContext).
            let props = unsafe {
                ctx.instance
                    .get_physical_device_properties(ctx.amd_compute_devices()[me.phys_idx].handle)
            };
            Some(Prof {
                pool,
                host: RefCell::new([0; 3]),
                period_ns: props.limits.timestamp_period,
                accum: RefCell::new(Vec::new()),
                tokens: std::cell::Cell::new(0),
                accum_batch: RefCell::new(Vec::new()),
                blocos: std::cell::Cell::new(0),
                accum_verify: RefCell::new(Vec::new()),
                verifies: std::cell::Cell::new(0),
                spans: RefCell::new(Vec::new()),
                max_trace_tokens: std::env::var("LLAMA_RS_TRACE_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8),
            })
        } else {
            None
        };
        if let Some(st) = me.state.as_mut() {
            st.barreiras = Self::marcar_barreiras(&plan, st);
            st.barreiras_batch = Self::marcar_barreiras(&plan_batch, st);
            st.plan_batch = plan_batch;
            st.barreiras_verify = Self::marcar_barreiras(&plan_verify, st);
            st.plan_verify = plan_verify;
            st.mtp_barreiras = Self::marcar_barreiras(&plan_mtp, st);
            st.mtp_plan = plan_mtp;
            if std::env::var("LLAMA_RS_PROFILE").is_ok_and(|v| v != "0") {
                let n = st.barreiras.iter().filter(|b| **b).count();
                eprintln!(
                    "[prof] {} ops/token, {n} barreiras ({:.0}% das ops rodam agrupadas)",
                    plan.len(),
                    100.0 * (1.0 - n as f64 / plan.len() as f64)
                );
            }
            st.plan = plan;
            st.prof = prof;
        }
        me.gravar_rollback()?;
        Ok(me)
    }

    /// Grava, uma vez, os command buffers que restauram os snapshots do estado
    /// recorrente — um por ponto de rollback (`cmds[i]` restaura o estado depois do
    /// token `i` do bloco, ou seja, mantém `i + 1` tokens).
    ///
    /// O conteúdo é estático — origem, destino e tamanho de cada cópia são conhecidos na
    /// construção —, então a rejeição de uma proposta custa só um submit, sem gravação.
    fn gravar_rollback(&mut self) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        let cmd_pool = self.dev.cmd_pool;
        let Some(st) = self.state.as_ref() else {
            return Ok(());
        };
        let mut cmds = Vec::with_capacity(VERIFY_TOK - 1);
        for ponto in 0..VERIFY_TOK - 1 {
            // As cópias existem só onde há snapshot, ou seja, com MTP e camada linear.
            let copias: Vec<(vk::Buffer, vk::Buffer, vk::DeviceSize)> = st
                .aux
                .iter()
                .filter_map(|la| la.delta.as_ref())
                .flat_map(|dn| {
                    dn.estado_snap
                        .get(ponto)
                        .map(|s| (s.buffer, dn.estado.buffer, dn.estado.size))
                        .into_iter()
                        .chain(
                            dn.janela_snap
                                .get(ponto)
                                .map(|s| (s.buffer, dn.janela.buffer, dn.janela.size)),
                        )
                })
                .collect();
            if copias.is_empty() {
                return Ok(());
            }
            let info = vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            // SAFETY: device e pool válidos; o buffer é gravado agora e resubmetido depois.
            let cmd = unsafe { d.allocate_command_buffers(&info)? }[0];
            let begin = vk::CommandBufferBeginInfo::default();
            // SAFETY: cmd recém-alocado; as cópias apontam para buffers vivos no state.
            unsafe {
                d.begin_command_buffer(cmd, &begin)?;
                for (src, dst, size) in &copias {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: *size,
                    };
                    d.cmd_copy_buffer(cmd, *src, *dst, &[region]);
                }
                d.end_command_buffer(cmd)?;
            }
            cmds.push(cmd);
        }
        if let Some(st) = self.state.as_mut() {
            st.rollback_cmds = cmds;
        }
        Ok(())
    }

    /// Diagnóstico: roda o shader attention GQA sobre q/k_cache/v_cache e devolve o resultado.
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        total_len: usize,
        // Tokens do bloco de prefill. 1 reproduz o decode; N processa N tokens, cada um
        // vendo `total_len - (N-1) + t` posições -- a máscara causal. `q` traz os N blocos
        // de query concatenados e a saída sai igual.
        n_tokens: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        // O shader distribui até MAX_DPL=4 dimensões por lane sobre 64 lanes.
        if head_dim > 256 {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        #[repr(C)]
        struct P {
            n_head: u32,
            n_head_kv: u32,
            head_dim: u32,
            total_len: u32,
            kv_dim: u32,
            kv_layer_off: u32,
            q_stride: u32,
        }
        let d = &self.dev.device;
        let kv_dim = n_head_kv * head_dim;
        let qb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(q) as vk::DeviceSize,
        )?;
        let kb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(k_cache) as vk::DeviceSize,
        )?;
        let vb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(v_cache) as vk::DeviceSize,
        )?;
        let ob = Buf::device(
            self.ctx,
            self.phys(),
            d,
            (n_tokens * n_head * head_dim * 4) as vk::DeviceSize,
        )?;
        self.upload_f32(&qb, q)?;
        self.upload_f32(&kb, k_cache)?;
        self.upload_f32(&vb, v_cache)?;
        let set = self.alloc_set(&self.attention)?;
        let push = P {
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            head_dim: head_dim as u32,
            total_len: total_len as u32,
            kv_dim: kv_dim as u32,
            kv_layer_off: 0,
            // Cabeças contíguas neste helper de teste.
            q_stride: head_dim as u32,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 28) };
        self.dispatch_xy(
            &self.attention,
            set,
            &[
                (qb.buffer, 0, qb.size),
                (kb.buffer, 0, kb.size),
                (vb.buffer, 0, vb.size),
                (ob.buffer, 0, ob.size),
            ],
            pb,
            n_head as u32,   // 1 workgroup por head...
            n_tokens as u32, // ...vezes um por token do bloco
        )?;
        let out = self.readback(&ob, n_tokens * n_head * head_dim)?;
        qb.destroy(d);
        kb.destroy(d);
        vb.destroy(d);
        ob.destroy(d);
        Ok(out)
    }

    /// Diagnóstico: a atenção com o KV fatiado em `n_split` workgroups, seguida da
    /// redução dos parciais. Mesma entrada e mesma saída do `dbg_attention`.
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_attention_split(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        total_len: usize,
        n_tokens: usize,
        n_split: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        if head_dim > 256 || n_split == 0 {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        #[repr(C)]
        struct P {
            n_head: u32,
            n_head_kv: u32,
            head_dim: u32,
            total_len: u32,
            kv_dim: u32,
            kv_layer_off: u32,
            q_stride: u32,
        }
        #[repr(C)]
        struct R {
            n_head: u32,
            head_dim: u32,
            n_split: u32,
        }
        let d = &self.dev.device;
        let kv_dim = n_head_kv * head_dim;
        let nb = |n: usize| (n * 4) as vk::DeviceSize;
        let qb = Buf::device(self.ctx, self.phys(), d, nb(q.len()))?;
        let kb = Buf::device(self.ctx, self.phys(), d, nb(k_cache.len()))?;
        let vb = Buf::device(self.ctx, self.phys(), d, nb(v_cache.len()))?;
        let parciais = Buf::device(
            self.ctx,
            self.phys(),
            d,
            nb(n_tokens * n_head * n_split * (head_dim + 2)),
        )?;
        let ob = Buf::device(self.ctx, self.phys(), d, nb(n_tokens * n_head * head_dim))?;
        self.upload_f32(&qb, q)?;
        self.upload_f32(&kb, k_cache)?;
        self.upload_f32(&vb, v_cache)?;

        let push = P {
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            head_dim: head_dim as u32,
            total_len: total_len as u32,
            kv_dim: kv_dim as u32,
            kv_layer_off: 0,
            q_stride: head_dim as u32,
        };
        // SAFETY: P é #[repr(C)] de 7 u32 contíguos; 28 bytes é o push range da pipeline.
        let pb = unsafe { std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), 28) };
        let set = self.alloc_set(&self.attention_split)?;
        self.dispatch_xyz(
            &self.attention_split,
            set,
            &[
                (qb.buffer, 0, qb.size),
                (kb.buffer, 0, kb.size),
                (vb.buffer, 0, vb.size),
                (parciais.buffer, 0, parciais.size),
            ],
            pb,
            n_head as u32,
            n_tokens as u32,
            n_split as u32,
        )?;

        let red = R {
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            n_split: n_split as u32,
        };
        // SAFETY: R é #[repr(C)] de 3 u32; 12 bytes é o push range da pipeline.
        let rb = unsafe { std::slice::from_raw_parts(std::ptr::from_ref(&red).cast::<u8>(), 12) };
        let set_r = self.alloc_set(&self.attn_reduce)?;
        self.dispatch_xy(
            &self.attn_reduce,
            set_r,
            &[(parciais.buffer, 0, parciais.size), (ob.buffer, 0, ob.size)],
            rb,
            n_head as u32,
            n_tokens as u32,
        )?;

        let out = self.readback(&ob, n_tokens * n_head * head_dim)?;
        qb.destroy(d);
        kb.destroy(d);
        vb.destroy(d);
        parciais.destroy(d);
        ob.destroy(d);
        Ok(out)
    }

    /// Bench: sobe o KV **uma vez** e cronometra `reps` dispatches da atenção, com o KV
    /// fatiado em `n_split` (1 = o caminho de um workgroup por cabeça).
    ///
    /// Existe porque medir pelo `dbg_attention*` mede o upload: com 26k posições o
    /// KV-cache tem 108 MB e a cópia domina o relógio.
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_attention_bench(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        total_len: usize,
        n_split: usize,
        reps: usize,
    ) -> Result<f64, MatmulError> {
        #[repr(C)]
        struct P {
            n_head: u32,
            n_head_kv: u32,
            head_dim: u32,
            total_len: u32,
            kv_dim: u32,
            kv_layer_off: u32,
            q_stride: u32,
        }
        let d = &self.dev.device;
        let kv_dim = n_head_kv * head_dim;
        let nb = |n: usize| (n * 4) as vk::DeviceSize;
        let qb = Buf::device(self.ctx, self.phys(), d, nb(q.len()))?;
        let kb = Buf::device(self.ctx, self.phys(), d, nb(k_cache.len()))?;
        let vb = Buf::device(self.ctx, self.phys(), d, nb(v_cache.len()))?;
        let parciais = Buf::device(
            self.ctx,
            self.phys(),
            d,
            nb(n_head * n_split.max(1) * (head_dim + 2)),
        )?;
        let ob = Buf::device(self.ctx, self.phys(), d, nb(n_head * head_dim))?;
        self.upload_f32(&qb, q)?;
        self.upload_f32(&kb, k_cache)?;
        self.upload_f32(&vb, v_cache)?;

        let push = P {
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            head_dim: head_dim as u32,
            total_len: total_len as u32,
            kv_dim: kv_dim as u32,
            kv_layer_off: 0,
            q_stride: head_dim as u32,
        };
        // SAFETY: P é #[repr(C)] de 7 u32; 28 bytes é o push range das duas pipelines.
        let pb = unsafe { std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), 28) };
        let usa_split = n_split > 1;
        let pipe = if usa_split {
            &self.attention_split
        } else {
            &self.attention
        };
        let saida = if usa_split { &parciais } else { &ob };
        let set = self.alloc_set(pipe)?;
        let binds = [
            (qb.buffer, 0, qb.size),
            (kb.buffer, 0, kb.size),
            (vb.buffer, 0, vb.size),
            (saida.buffer, 0, saida.size),
        ];

        // Aquecimento e medida.
        let z = if usa_split { n_split as u32 } else { 1 };
        self.dispatch_xyz(pipe, set, &binds, pb, n_head as u32, 1, z)?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps.max(1) {
            self.dispatch_xyz(pipe, set, &binds, pb, n_head as u32, 1, z)?;
        }
        #[allow(clippy::cast_precision_loss)]
        let media = t0.elapsed().as_secs_f64() / reps.max(1) as f64;

        qb.destroy(d);
        kb.destroy(d);
        vb.destroy(d);
        parciais.destroy(d);
        ob.destroy(d);
        Ok(media)
    }

    /// Emite as ops de uma camada de atenção linear (qwen35), deixando o resultado em
    /// `b_proj` — o mesmo lugar onde a camada de atenção deixa o dela, para que o fecho
    /// da camada (residual, norma, FFN) seja comum aos dois caminhos.
    ///
    /// A sequência segue `docs/qwen35-arquitetura.md`:
    /// `qkv = W·x`, `z = Wg·x`, gates, convolução causal, L2 em q/k, recorrência, norma
    /// gated e projeção de saída. A ativação já foi quantizada em int8 pelo `QuantizeX`
    /// do começo da camada, então os três matvecs a consomem direto.
    ///
    /// Com `n_tok > 1` **cada op vira um dispatch só**, como no resto do plano — inclusive
    /// as duas com estado. `dn_conv` e `delta_net` não perdem a recorrência por isso: o
    /// laço sobre os tokens do bloco mora dentro do kernel, com o estado em registrador
    /// entre eles (ver os comentários dos dois `.comp`). Antes eram `n_tok` dispatches por
    /// op, que `marcar_barreiras` serializava um a um — 4 × n_tok dispatches por camada
    /// linear, 48 camadas: em batch 32 seriam 6 mil só de delta-net por bloco.
    ///
    /// `dn_gates` e `dn_l2_qk` batcham por `gl_WorkGroupID.y`, como as ops de atenção.
    #[allow(clippy::too_many_arguments)]
    fn plano_delta(
        plan: &mut Vec<PlannedOp>,
        st: &ResidentState<'_>,
        la: &LayerAux,
        c: &Cfg,
        n_tok: usize,
        pesos: (&QWeight, &QWeight, &QWeight),
        mk: &MkDispatch<'_>,
        mv: &MkMatvec<'_>,
        mv_com: &MkMatvecCom<'_>,
    ) -> Result<(), MatmulError> {
        let (w_qkv, w_gate, w_out) = pesos;
        let dn_cfg = c
            .delta_net
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;
        let b = st
            .dn
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;
        let da = la
            .delta
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))?;
        // A janela da convolução vive em registrador dentro do `dn_conv.comp`, num array de
        // `MAX_PASSOS = 4`. Acima disso o shader calcularia com uma janela curta demais e o
        // erro só apareceria na qualidade da saída — falhar aqui é o que impede isso.
        if dn_cfg.d_conv > 5 {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }

        let key_dim = dn_cfg.d_state * dn_cfg.n_k_heads;
        let value_dim = dn_cfg.head_v_dim() * dn_cfg.n_v_heads;
        let conv_dim = conv_dim_de(dn_cfg);
        let nt = u32::try_from(n_tok).unwrap_or(1);
        let nb = |n: usize| (n * 4) as vk::DeviceSize;
        // Faixa de um buffer por token, cobrindo o bloco inteiro.
        let nbt = |n: usize| ((n * n_tok) * 4) as vk::DeviceSize;
        let push3 = |a: u32, b: u32, c: u32| {
            let mut v = Vec::with_capacity(12);
            v.extend_from_slice(&a.to_le_bytes());
            v.extend_from_slice(&b.to_le_bytes());
            v.extend_from_slice(&c.to_le_bytes());
            v
        };
        let push_norm = |dim: u32, n_heads: u32, modo: u32, eps: f32| {
            let mut v = Vec::with_capacity(20);
            v.extend_from_slice(&dim.to_le_bytes());
            v.extend_from_slice(&n_heads.to_le_bytes());
            v.extend_from_slice(&modo.to_le_bytes());
            v.extend_from_slice(&eps.to_le_bytes());
            v.extend_from_slice(&dim.to_le_bytes()); // stride: cabeças contíguas
            v
        };

        // Projeções da entrada já quantizada.
        plan.push(mv(w_qkv, &b.qkv, c.n_embd, conv_dim)?);
        plan.push(mv(w_gate, &b.z, c.n_embd, value_dim)?);

        // (g, beta) por cabeça — leem `b_normed` em f32, não a versão int8: são
        // projeções f32 pequenas e o erro da quantização entraria num expoente. Os tokens
        // do bloco entram pela dimensão Y (um workgroup por cabeça e token).
        plan.push(Self::com_y(
            mk(
                PipeId::DnGates,
                &[
                    (st.b_normed.buffer, 0, nbt(c.n_embd)),
                    (da.alpha.buffer, 0, da.alpha.size),
                    (da.beta.buffer, 0, da.beta.size),
                    (da.adt.buffer, 0, da.adt.size),
                    (b.gb.buffer, 0, nbt(dn_cfg.n_v_heads * 2)),
                ],
                u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
                PushSpec::Static(push3(
                    u32::try_from(c.n_embd).unwrap_or(u32::MAX),
                    u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
                    0,
                )),
            )?,
            nt,
        ));

        // Convolução causal com estado, já saindo com SiLU. A janela é o estado, e ela
        // avança token a token **dentro** do dispatch — ver `dn_conv.comp`.
        plan.push(mk(
            PipeId::DnConv,
            &[
                (da.janela.buffer, 0, da.janela.size),
                (b.qkv.buffer, 0, nbt(conv_dim)),
                (da.conv1d.buffer, 0, da.conv1d.size),
                (b.conv.buffer, 0, nbt(conv_dim)),
            ],
            Self::groups_for(conv_dim),
            PushSpec::Static(push3(
                u32::try_from(conv_dim).unwrap_or(u32::MAX),
                u32::try_from(dn_cfg.d_conv).unwrap_or(u32::MAX),
                nt,
            )),
        )?);

        // L2 por cabeça em q e k, que estão nos dois primeiros terços de `conv` — e por
        // isso não são contíguos entre tokens: daí o `stride` do push. Os dois tensores
        // saem no mesmo dispatch (`dn_l2_qk`): são cabeças contíguas em `conv`, então o
        // shader só precisa de `2 * n_k_heads` workgroups (× n_tok em Y) e dos destinos.
        let eps = c.rms_eps;
        plan.push(Self::com_y(
            mk(
                PipeId::DnL2Qk,
                &[
                    (b.conv.buffer, 0, nbt(conv_dim)),
                    (b.qn.buffer, 0, nbt(key_dim)),
                    (b.kn.buffer, 0, nbt(key_dim)),
                ],
                u32::try_from(dn_cfg.n_k_heads * 2).unwrap_or(u32::MAX),
                PushSpec::Static({
                    let mut v = Vec::with_capacity(16);
                    v.extend_from_slice(
                        &u32::try_from(dn_cfg.d_state)
                            .unwrap_or(u32::MAX)
                            .to_le_bytes(),
                    );
                    v.extend_from_slice(
                        &u32::try_from(dn_cfg.n_k_heads)
                            .unwrap_or(u32::MAX)
                            .to_le_bytes(),
                    );
                    v.extend_from_slice(&eps.to_le_bytes());
                    v.extend_from_slice(&u32::try_from(conv_dim).unwrap_or(u32::MAX).to_le_bytes());
                    v
                }),
            )?,
            nt,
        ));

        // Recorrência: lê o estado da camada, aplica os `n_tok` tokens em ordem dentro do
        // kernel e o reescreve uma vez. `v` é a última fatia de cada token em `conv`, daí
        // o binding começar em `2 * key_dim` e o passo entre tokens ser `conv_dim`.
        plan.push(mk(
            PipeId::DeltaNet,
            &[
                (da.estado.buffer, 0, da.estado.size),
                (b.qn.buffer, 0, nbt(key_dim)),
                (b.kn.buffer, 0, nbt(key_dim)),
                (
                    b.conv.buffer,
                    nb(2 * key_dim),
                    nb(conv_dim * n_tok - 2 * key_dim),
                ),
                (b.gb.buffer, 0, nbt(dn_cfg.n_v_heads * 2)),
                (b.out.buffer, 0, nbt(value_dim)),
            ],
            u32::try_from(dn_cfg.n_v_heads * dn_cfg.d_state / 4).unwrap_or(u32::MAX),
            PushSpec::Static({
                let mut v = push3(
                    u32::try_from(dn_cfg.d_state).unwrap_or(u32::MAX),
                    u32::try_from(dn_cfg.n_v_heads).unwrap_or(u32::MAX),
                    u32::try_from(dn_cfg.n_v_heads / dn_cfg.n_k_heads).unwrap_or(1),
                );
                v.extend_from_slice(&nt.to_le_bytes());
                v.extend_from_slice(&u32::try_from(conv_dim).unwrap_or(u32::MAX).to_le_bytes());
                v
            }),
        )?);

        // Norma gated: rmsnorm por cabeça vezes silu(z). As cabeças dos N tokens são
        // contíguas neste passo, então o batch é multiplicar a contagem (ver QK-norm).
        let heads_v = dn_cfg.n_v_heads * n_tok;
        plan.push(mk(
            PipeId::DnNorm,
            &[
                (b.out.buffer, 0, nb(value_dim * n_tok)),
                (da.norm.buffer, 0, da.norm.size),
                (b.z.buffer, 0, nb(value_dim * n_tok)),
                (b.normed.buffer, 0, nb(value_dim * n_tok)),
            ],
            u32::try_from(heads_v).unwrap_or(u32::MAX),
            PushSpec::Static(push_norm(
                u32::try_from(dn_cfg.head_v_dim()).unwrap_or(u32::MAX),
                u32::try_from(heads_v).unwrap_or(u32::MAX),
                1,
                eps,
            )),
        )?);

        // A saída da recorrência precisa ser requantizada antes do matvec final: o
        // `QuantizeX` do começo da camada quantizou `b_normed`, não isto.
        let vd_n = value_dim * n_tok;
        plan.push(mk(
            PipeId::QuantizeX,
            &[
                (b.normed.buffer, 0, nb(vd_n)),
                (b.xq.buffer, 0, b.xq.size),
                (b.xd.buffer, 0, b.xd.size),
            ],
            u32::try_from((vd_n / 32).div_ceil(64)).unwrap_or(u32::MAX),
            PushSpec::Static(push3(u32::try_from(vd_n).unwrap_or(u32::MAX), 0, 0)),
        )?);
        plan.push(mv_com(
            w_out,
            &st.b_proj,
            (&b.xq, &b.xd),
            value_dim,
            c.n_embd,
        )?);
        Ok(())
    }

    /// A mesma camada de atenção linear, para o bloco de **dois** tokens do verify.
    ///
    /// Duplica `plano_delta` de propósito, e não por descuido: o verify precisa que a
    /// recorrência seja exatamente `n_tok` dispatches em ordem, com um ponto de parada
    /// entre eles onde o estado do primeiro token pode ser copiado. O caminho de batch vai
    /// migrar para um kernel multi-token (uma passada só para os N tokens do bloco), e aí
    /// esse ponto de parada deixa de existir — o verify não pode herdar essa mudança sem
    /// perder o rollback.
    ///
    /// O snapshot entra em dois lugares, sempre **depois do token 0 e antes do token 1**:
    /// a janela da convolução (120 KB) e o estado recorrente (3,1 MB). Juntos são 3,2 MB
    /// por camada; nas 48 camadas lineares do Qwen3.8-27B, ~155 MB — 0,45 ms a 717 GB/s.
    #[allow(clippy::too_many_arguments)]
    fn plano_delta_verify(
        plan: &mut Vec<PlannedOp>,
        st: &ResidentState,
        la: &LayerAux,
        c: &Cfg,
        pesos: (&QWeight, &QWeight, &QWeight),
        mk: &MkDispatch<'_>,
        mv: &MkMatvec<'_>,
        mv_com: &MkMatvecCom<'_>,
    ) -> Result<(), MatmulError> {
        let (w_qkv, w_gate, w_out) = pesos;
        let n_tok = VERIFY_TOK;
        let faltando = || MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        let dn_cfg = c.delta_net.as_ref().ok_or_else(faltando)?;
        let b = st.dn.as_ref().ok_or_else(faltando)?;
        let da = la.delta.as_ref().ok_or_else(faltando)?;
        // Sem os buffers de snapshot não há rollback possível, e um verify sem rollback
        // corrompe o estado na primeira rejeição — melhor falhar na construção do plano.
        if da.estado_snap.len() != n_tok - 1 || da.janela_snap.len() != n_tok - 1 {
            return Err(faltando());
        }

        let key_dim = dn_cfg.d_state * dn_cfg.n_k_heads;
        let value_dim = dn_cfg.head_v_dim() * dn_cfg.n_v_heads;
        let conv_dim = conv_dim_de(dn_cfg);
        let nb = |n: usize| (n * 4) as vk::DeviceSize;
        let u = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let push3 = |a: u32, b: u32, c: u32| {
            let mut v = Vec::with_capacity(12);
            v.extend_from_slice(&a.to_le_bytes());
            v.extend_from_slice(&b.to_le_bytes());
            v.extend_from_slice(&c.to_le_bytes());
            v
        };
        let eps = c.rms_eps;

        // Projeções da entrada já quantizada — batcham por COLS como no resto do plano.
        plan.push(mv(w_qkv, &b.qkv, c.n_embd, conv_dim)?);
        plan.push(mv(w_gate, &b.z, c.n_embd, value_dim)?);

        // (g, beta) por cabeça: o peso é indexado pela cabeça, então um dispatch por token.
        for t in 0..n_tok {
            plan.push(mk(
                PipeId::DnGates,
                &[
                    (st.b_normed.buffer, nb(t * c.n_embd), nb(c.n_embd)),
                    (da.alpha.buffer, 0, da.alpha.size),
                    (da.beta.buffer, 0, da.beta.size),
                    (da.adt.buffer, 0, da.adt.size),
                    (
                        b.gb.buffer,
                        nb(t * dn_cfg.n_v_heads * 2),
                        nb(dn_cfg.n_v_heads * 2),
                    ),
                ],
                u(dn_cfg.n_v_heads),
                PushSpec::Static(push3(u(c.n_embd), u(dn_cfg.n_v_heads), 0)),
            )?);
        }

        // Convolução causal: a janela é estado, então os tokens entram em ordem e o
        // snapshot dela fica entre cada par consecutivo.
        for t in 0..n_tok {
            if t >= 1 {
                plan.push(PlannedOp::Copia {
                    src: da.janela.buffer,
                    dst: da.janela_snap[t - 1].buffer,
                    bytes: da.janela.size,
                });
            }
            plan.push(mk(
                PipeId::DnConv,
                &[
                    (da.janela.buffer, 0, da.janela.size),
                    (b.qkv.buffer, nb(t * conv_dim), nb(conv_dim)),
                    (da.conv1d.buffer, 0, da.conv1d.size),
                    (b.conv.buffer, nb(t * conv_dim), nb(conv_dim)),
                ],
                Self::groups_for(conv_dim),
                // n_tok = 1: aqui cada dispatch processa um token (o batch usa `nt`).
                PushSpec::Static(push3(u(conv_dim), u(dn_cfg.d_conv), 1)),
            )?);
        }

        // L2 por cabeça em q e k, que ocupam os dois primeiros terços de `conv` e por isso
        // não são contíguos entre tokens: um dispatch por token, os dois tensores juntos.
        for t in 0..n_tok {
            plan.push(mk(
                PipeId::DnL2Qk,
                &[
                    (b.conv.buffer, nb(t * conv_dim), nb(2 * key_dim)),
                    (b.qn.buffer, nb(t * key_dim), nb(key_dim)),
                    (b.kn.buffer, nb(t * key_dim), nb(key_dim)),
                ],
                u(dn_cfg.n_k_heads * 2),
                PushSpec::Static({
                    let mut v = Vec::with_capacity(16);
                    v.extend_from_slice(&u(dn_cfg.d_state).to_le_bytes());
                    v.extend_from_slice(&u(dn_cfg.n_k_heads).to_le_bytes());
                    v.extend_from_slice(&eps.to_le_bytes());
                    // stride entre tokens: sem uso aqui (Y = 1 token por dispatch).
                    v.extend_from_slice(&0u32.to_le_bytes());
                    v
                }),
            )?);
        }

        // Recorrência, um token de cada vez, com o snapshot do estado entre eles.
        for t in 0..n_tok {
            if t >= 1 {
                plan.push(PlannedOp::Copia {
                    src: da.estado.buffer,
                    dst: da.estado_snap[t - 1].buffer,
                    bytes: da.estado.size,
                });
            }
            plan.push(mk(
                PipeId::DeltaNet,
                &[
                    (da.estado.buffer, 0, da.estado.size),
                    (b.qn.buffer, nb(t * key_dim), nb(key_dim)),
                    (b.kn.buffer, nb(t * key_dim), nb(key_dim)),
                    (b.conv.buffer, nb(t * conv_dim + 2 * key_dim), nb(value_dim)), // v
                    (
                        b.gb.buffer,
                        nb(t * dn_cfg.n_v_heads * 2),
                        nb(dn_cfg.n_v_heads * 2),
                    ),
                    (b.out.buffer, nb(t * value_dim), nb(value_dim)),
                ],
                u(dn_cfg.n_v_heads * dn_cfg.d_state / 4),
                PushSpec::Static({
                    let mut v = push3(
                        u(dn_cfg.d_state),
                        u(dn_cfg.n_v_heads),
                        u32::try_from(dn_cfg.n_v_heads / dn_cfg.n_k_heads).unwrap_or(1),
                    );
                    // n_tok = 1: um token por dispatch; v_stride não é lido com t = 0,
                    // mas o layout do shader exige os 20 bytes.
                    v.extend_from_slice(&1u32.to_le_bytes());
                    v.extend_from_slice(&u(value_dim).to_le_bytes());
                    v
                }),
            )?);
        }

        // Norma gated: as cabeças dos dois tokens são contíguas, então o batch é só
        // multiplicar a contagem de cabeças.
        let heads_v = dn_cfg.n_v_heads * n_tok;
        plan.push(mk(
            PipeId::DnNorm,
            &[
                (b.out.buffer, 0, nb(value_dim * n_tok)),
                (da.norm.buffer, 0, da.norm.size),
                (b.z.buffer, 0, nb(value_dim * n_tok)),
                (b.normed.buffer, 0, nb(value_dim * n_tok)),
            ],
            u(heads_v),
            PushSpec::Static({
                let mut v = Vec::with_capacity(20);
                v.extend_from_slice(&u(dn_cfg.head_v_dim()).to_le_bytes());
                v.extend_from_slice(&u(heads_v).to_le_bytes());
                v.extend_from_slice(&1u32.to_le_bytes()); // modo: norma gated
                v.extend_from_slice(&eps.to_le_bytes());
                v.extend_from_slice(&u(dn_cfg.head_v_dim()).to_le_bytes()); // stride
                v
            }),
        )?);

        // A saída da recorrência precisa ser requantizada antes do matvec final.
        let vd_n = value_dim * n_tok;
        plan.push(mk(
            PipeId::QuantizeX,
            &[
                (b.normed.buffer, 0, nb(vd_n)),
                (b.xq.buffer, 0, b.xq.size),
                (b.xd.buffer, 0, b.xd.size),
            ],
            u((vd_n / 32).div_ceil(64)),
            PushSpec::Static(push3(u(vd_n), 0, 0)),
        )?);
        plan.push(mv_com(
            w_out,
            &st.b_proj,
            (&b.xq, &b.xd),
            value_dim,
            c.n_embd,
        )?);
        Ok(())
    }

    /// nº de workgroups para cobrir `n` elementos com local_size_x=64.
    pub(crate) fn groups_for(n: usize) -> u32 {
        n.div_ceil(64) as u32
    }

    /// Roda um shader de atenção linear com buffers f32 e devolve o conteúdo final de
    /// todos eles, na ordem dos bindings.
    ///
    /// Serve para validar os shaders do qwen35 contra `llama_model::delta_net`: os
    /// buffers `inout` (o estado recorrente, a janela da convolução) voltam atualizados,
    /// e é justamente essa atualização que precisa bater com a referência de CPU.
    pub fn dbg_dn(
        &self,
        qual: DnPipe,
        bufs: &[Vec<f32>],
        push: &[u8],
        groups: u32,
    ) -> Result<Vec<Vec<f32>>, MatmulError> {
        self.dbg_dn_xy(qual, bufs, push, groups, 1)
    }

    /// Igual a `dbg_dn`, com a dimensão Y do dispatch exposta — que é como os shaders
    /// recebem o token do bloco no batch (ver `attention.comp` e `norm_fused.comp`).
    pub fn dbg_dn_xy(
        &self,
        qual: DnPipe,
        bufs: &[Vec<f32>],
        push: &[u8],
        groups: u32,
        groups_y: u32,
    ) -> Result<Vec<Vec<f32>>, MatmulError> {
        let d = &self.dev.device;
        let pipe = match qual {
            DnPipe::DeltaNet => &self.delta_net,
            DnPipe::Conv => &self.dn_conv,
            DnPipe::Gates => &self.dn_gates,
            DnPipe::Norm => &self.dn_norm,
            DnPipe::L2Qk => &self.dn_l2_qk,
            DnPipe::GateQuant => &self.gate_quant,
            DnPipe::SwigluQuant => &self.swiglu_quant,
            DnPipe::RopeKv => &self.rope_kv,
            DnPipe::QuantizeX => &self.quantize_x,
            DnPipe::NormFused => &self.norm_fused,
            DnPipe::NormP2 => &self.norm_p2,
        };
        let mut gpu = Vec::with_capacity(bufs.len());
        for b in bufs {
            let buf = Buf::device(
                self.ctx,
                self.phys(),
                d,
                (b.len() * 4).max(4) as vk::DeviceSize,
            )?;
            self.upload_f32(&buf, b)?;
            gpu.push(buf);
        }
        let set = self.alloc_set(pipe)?;
        let bindings: Vec<_> = gpu.iter().map(|b| (b.buffer, 0, b.size)).collect();
        self.dispatch_xy(pipe, set, &bindings, push, groups, groups_y)?;

        let mut out = Vec::with_capacity(bufs.len());
        for (buf, orig) in gpu.iter().zip(bufs) {
            out.push(self.readback(buf, orig.len())?);
        }
        for buf in gpu {
            buf.destroy(d);
        }
        Ok(out)
    }

    pub fn dbg_add(&self, dst: &[f32], src: &[f32]) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P {
            n: u32,
        }
        let d = &self.dev.device;
        let n = dst.len();
        let db = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(dst) as vk::DeviceSize,
        )?;
        let sb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(src) as vk::DeviceSize,
        )?;
        self.upload_f32(&db, dst)?;
        self.upload_f32(&sb, src)?;
        let set = self.alloc_set(&self.add)?;
        let push = P { n: n as u32 };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 4) };
        self.dispatch1(
            &self.add,
            set,
            &[(db.buffer, 0, db.size), (sb.buffer, 0, sb.size)],
            pb,
            Self::groups_for(n),
        )?;
        let out = self.readback(&db, n)?; // dst foi mutado in-place
        db.destroy(d);
        sb.destroy(d);
        Ok(out)
    }

    /// Diagnóstico: roda só o shader rmsnorm sobre `x`,`w` e devolve a saída ao host.
    pub fn dbg_rmsnorm(&self, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P {
            dim: u32,
            eps: f32,
        }
        let d = &self.dev.device;
        let dim = x.len();
        let xb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(x) as vk::DeviceSize,
        )?;
        let wb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(w) as vk::DeviceSize,
        )?;
        let ob = Buf::device(self.ctx, self.phys(), d, (dim * 4) as vk::DeviceSize)?;
        self.upload_f32(&xb, x)?;
        self.upload_f32(&wb, w)?;
        let set = self.alloc_set(&self.rmsnorm)?;
        let push = P {
            dim: dim as u32,
            eps,
        };
        let push_bytes = unsafe {
            std::slice::from_raw_parts(&push as *const P as *const u8, std::mem::size_of::<P>())
        };
        self.dispatch1(
            &self.rmsnorm,
            set,
            &[
                (xb.buffer, 0, xb.size),
                (wb.buffer, 0, wb.size),
                (ob.buffer, 0, ob.size),
            ],
            push_bytes,
            1, // 1 workgroup processa a linha
        )?;
        let out = self.readback(&ob, dim)?;
        xb.destroy(d);
        wb.destroy(d);
        ob.destroy(d);
        Ok(out)
    }

    /// Diagnóstico: roda o shader rope in-place sobre `x` e devolve o resultado ao host.
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_rope(
        &self,
        x: &mut [f32],
        n_head: usize,
        head_dim: usize,
        rope_dim: usize,
        freq: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        self.dbg_rope_xy(x, n_head, head_dim, rope_dim, freq, pos, 1)
    }

    /// Igual a `dbg_rope`, com `n_tok` tokens concatenados em `x`. `pos` é a posição do
    /// **último** deles, que é o significado que o push já tinha com um token só.
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_rope_xy(
        &self,
        x: &mut [f32],
        n_head: usize,
        head_dim: usize,
        rope_dim: usize,
        freq: &[f32],
        pos: usize,
        n_tok: u32,
    ) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P {
            n_head: u32,
            head_dim: u32,
            rope_dim: u32,
            pos: f32,
            stride: u32,
        }
        let d = &self.dev.device;
        let xb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(x) as vk::DeviceSize,
        )?;
        let fb = Buf::device(
            self.ctx,
            self.phys(),
            d,
            std::mem::size_of_val(freq) as vk::DeviceSize,
        )?;
        self.upload_f32(&xb, x)?;
        self.upload_f32(&fb, freq)?;
        let set = self.alloc_set(&self.rope)?;
        let push = P {
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            pos: pos as f32,
            // Cabeças contíguas neste helper de teste.
            stride: head_dim as u32,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 20) };
        let pairs = n_head * (rope_dim / 2);
        self.dispatch_xy(
            &self.rope,
            set,
            &[(xb.buffer, 0, xb.size), (fb.buffer, 0, fb.size)],
            pb,
            Self::groups_for(pairs),
            n_tok,
        )?;
        let out = self.readback(&xb, x.len())?;
        xb.destroy(d);
        fb.destroy(d);
        Ok(out)
    }

    /// Para cada op do plano, se ela precisa de uma barreira de memória **antes**.
    ///
    /// Ops entre duas barreiras podem rodar concorrentes na GPU. Uma barreira só faz falta
    /// quando a op conflita com o que o grupo corrente já fez: lê o que foi escrito (RAW),
    /// escreve o que foi lido (WAR) ou escreve o que já foi escrito (WAW). O critério é
    /// conservador em dois pontos, de propósito: compara as faixas **inteiras** dos bindings
    /// (não o que o shader de fato toca) e trata o KV-cache como um buffer só, já que o
    /// offset do append depende de `pos` e não é conhecido aqui.
    ///
    /// `LLAMA_RS_NO_GROUP=1` volta a uma barreira por op, para comparar. Motivação e ganho
    /// medido em `docs/performance-tuning.md`.
    fn marcar_barreiras(plan: &[PlannedOp], st: &ResidentState<'_>) -> Vec<bool> {
        // `LLAMA_RS_NO_GROUP=1` volta ao comportamento antigo (uma barreira por op) para
        // poder medir o efeito do agrupamento no mesmo binário.
        if std::env::var("LLAMA_RS_NO_GROUP").is_ok_and(|v| v != "0") {
            return vec![true; plan.len()];
        }
        let tudo = |b: &Buf| -> Faixa { (b.buffer, 0, b.size) };
        let mut grupo_le: Vec<Faixa> = Vec::new();
        let mut grupo_esc: Vec<Faixa> = Vec::new();
        let mut out = Vec::with_capacity(plan.len());

        for op in plan {
            let (le, esc): (Vec<Faixa>, Vec<Faixa>) = match op {
                PlannedOp::Dispatch { le, esc, .. } => (le.clone(), esc.clone()),
                PlannedOp::Embed => (vec![tudo(&st.embd_stage)], vec![tudo(&st.b_x)]),
                PlannedOp::Copia { src, dst, bytes } => {
                    (vec![(*src, 0, *bytes)], vec![(*dst, 0, *bytes)])
                }
                // Sem K (o `rope_kv` já o escreveu) a op não toca `b_k` nem o `kcache`, e
                // aí ela cabe no mesmo grupo do RoPE em vez de exigir barreira.
                PlannedOp::KvAppend { com_k: true, .. } => (
                    vec![tudo(&st.b_k), tudo(&st.b_v)],
                    vec![tudo(&st.kcache), tudo(&st.vcache)],
                ),
                PlannedOp::KvAppend { com_k: false, .. } => {
                    (vec![tudo(&st.b_v)], vec![tudo(&st.vcache)])
                }
                // O cache da cabeça é outro buffer: sem declará-lo aqui, a atenção do
                // bloco leria o que a cópia ainda não terminou de escrever.
                PlannedOp::KvAppendMtp => match st.mtp.as_ref() {
                    Some(m) => (
                        vec![tudo(&st.b_k), tudo(&st.b_v)],
                        vec![tudo(&m.kcache), tudo(&m.vcache)],
                    ),
                    None => (Vec::new(), Vec::new()),
                },
                PlannedOp::CopiaHidden => match st.mtp.as_ref() {
                    Some(m) => (vec![tudo(&st.b_x)], vec![tudo(&m.b_h)]),
                    None => (Vec::new(), Vec::new()),
                },
                // A op declara as faixas dos **dois** caminhos, inclusive o buffer de
                // parciais: ele é reusado por todas as camadas de atenção, e sem
                // declará-lo o planejador deixa a fatia de uma camada sobrescrever o que
                // a redução da anterior ainda não leu (erro medido: 3e-2 nos logits).
                PlannedOp::Atencao {
                    curto,
                    split,
                    reduce,
                } => {
                    let mut le = Vec::new();
                    let mut esc = Vec::new();
                    for parte in [curto.as_ref(), split.as_ref(), reduce.as_ref()] {
                        if let PlannedOp::Dispatch { le: l, esc: e, .. } = parte {
                            le.extend(l.iter().copied());
                            esc.extend(e.iter().copied());
                        }
                    }
                    (le, esc)
                }
            };
            let raw = le.iter().any(|f| grupo_esc.iter().any(|g| sobrepoe(f, g)));
            let war_waw = esc.iter().any(|f| {
                grupo_esc.iter().any(|g| sobrepoe(f, g)) || grupo_le.iter().any(|g| sobrepoe(f, g))
            });
            let precisa = raw || war_waw;
            if precisa {
                grupo_le.clear();
                grupo_esc.clear();
            }
            out.push(precisa);
            grupo_le.extend(le);
            grupo_esc.extend(esc);
        }
        out
    }

    /// Grava um `PlannedOp::Dispatch` no command buffer.
    ///
    /// `groups_z` é a terceira dimensão (fatias do KV na atenção longa; 1 no resto) e
    /// `n_split` é o que a redução precisa saber para combinar os parciais.
    #[allow(clippy::too_many_arguments)]
    fn gravar_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        op: &PlannedOp,
        c: &Cfg,
        pos: usize,
        total_len: u32,
        groups_z: u32,
        n_split: u32,
    ) {
        let PlannedOp::Dispatch {
            pipe,
            set,
            groups,
            groups_y,
            push,
            ..
        } = op
        else {
            return;
        };
        let d = &self.dev.device;
        let p = self.pipe_of(*pipe);
        let bytes: Vec<u8> = match push {
            PushSpec::Static(b) => b.clone(),
            PushSpec::Rope { n_head, stride } => {
                #[repr(C)]
                struct P {
                    n_head: u32,
                    head_dim: u32,
                    rope_dim: u32,
                    pos: f32,
                    stride: u32,
                }
                let pp = P {
                    n_head: *n_head,
                    head_dim: c.head_dim as u32,
                    rope_dim: c.rope_dim as u32,
                    pos: pos as f32,
                    stride: *stride,
                };
                unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 20) }.to_vec()
            }
            PushSpec::RopeKv {
                n_head,
                kv_layer_off,
            } => {
                #[repr(C)]
                struct P {
                    n_head: u32,
                    head_dim: u32,
                    rope_dim: u32,
                    pos: f32,
                    kv_off: u32,
                }
                // O bloco ocupa posições consecutivas a partir de `pos0`, e `groups_y` é
                // quantos tokens ele tem — o mesmo `n_tok` que o `KvAppend` usa.
                let pos0 = total_len as usize - *groups_y as usize;
                let pp = P {
                    n_head: *n_head,
                    head_dim: c.head_dim as u32,
                    rope_dim: c.rope_dim as u32,
                    pos: pos as f32,
                    kv_off: kv_layer_off + (pos0 * c.kv_dim) as u32,
                };
                // SAFETY: P é #[repr(C)] de 5 palavras de 32 bits; 20 bytes é o push range.
                unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 20) }.to_vec()
            }
            PushSpec::Attention { kv_layer_off } => {
                #[repr(C)]
                struct P {
                    n_head: u32,
                    n_head_kv: u32,
                    head_dim: u32,
                    total_len: u32,
                    kv_dim: u32,
                    kv_layer_off: u32,
                    q_stride: u32,
                }
                let pp = P {
                    n_head: c.n_head as u32,
                    n_head_kv: c.n_head_kv as u32,
                    head_dim: c.head_dim as u32,
                    total_len,
                    kv_dim: c.kv_dim as u32,
                    kv_layer_off: *kv_layer_off,
                    // No qwen35 query e gate dividem a cabeça.
                    q_stride: if c.delta_net.is_some() {
                        (c.head_dim * 2) as u32
                    } else {
                        c.head_dim as u32
                    },
                };
                unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 28) }.to_vec()
            }
            PushSpec::AttnReduce => {
                #[repr(C)]
                struct R {
                    n_head: u32,
                    head_dim: u32,
                    n_split: u32,
                }
                let rr = R {
                    n_head: c.n_head as u32,
                    head_dim: c.head_dim as u32,
                    n_split,
                };
                // SAFETY: R é #[repr(C)] de 3 u32; 12 bytes é o push range da pipeline.
                unsafe { std::slice::from_raw_parts(&rr as *const R as *const u8, 12) }.to_vec()
            }
        };
        // SAFETY: cmd em gravação; pipeline/set válidos; bytes do tamanho do range.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                p.layout,
                0,
                &[*set],
                &[],
            );
            d.cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            d.cmd_dispatch(cmd, *groups, *groups_y, groups_z);
        }
    }

    /// Barreira de memória global entre dispatches/cópias do mesmo command buffer.
    fn full_barrier(&self, cmd: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            ..Default::default()
        };
        let stage = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        // SAFETY: cmd em gravação; barreira global sem buffer/image barriers.
        unsafe {
            self.dev.device.cmd_pipeline_barrier(
                cmd,
                stage,
                stage,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );
        }
    }

    /// Grava a stack inteira do bloco em `cmd` (já em `begin`). `pos` é a posição absoluta
    /// do **último** token de `tokens` — é dela que saem `total_len` e o RoPE, e os shaders
    /// derivam a posição de cada token do bloco por `pos - (n_tok - 1) + t`.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "gravar_cmdbuf")
    )]
    fn record_token(
        &self,
        cmd: vk::CommandBuffer,
        tokens: &[u32],
        pos: usize,
        x_in: Option<&[f32]>,
        modo: Modo,
    ) {
        let d = &self.dev.device;
        let st = self
            .state
            .as_ref()
            .expect("record_token requer state (new())");
        let c = &st.cfg;
        let n_tok = tokens.len();
        let total_len = (pos + 1) as u32;
        let (plan, barreiras) = match modo {
            Modo::Decode => (&st.plan, &st.barreiras),
            Modo::Batch => (&st.plan_batch, &st.barreiras_batch),
            Modo::Verify => (&st.plan_verify, &st.barreiras_verify),
        };
        // Os dois planos são medidos: cada um tem o seu acumulador, porque as listas de
        // ops não se correspondem (`collect_prof` escolhe pelo `n_tok`).
        let prof = st.prof.as_ref();

        // Timestamp inicial (slot 0); cada op grava o seu em slot i+1.
        if let Some(pf) = prof {
            let n = (plan.len() + 1) as u32;
            // SAFETY: cmd em gravação; pool criado com 1024 slots >= n.
            unsafe {
                d.cmd_reset_query_pool(cmd, pf.pool, 0, n);
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pf.pool, 0);
            }
        }

        for (op_idx, op) in plan.iter().enumerate() {
            // Com o perfil ligado, serializa tudo: sem barreira as ops de um mesmo grupo se
            // sobrepõem e os timestamps de fim passam a medir a soma, não cada op. O TOTAL
            // impresso fica então acima do tempo real de um token.
            if barreiras[op_idx] || prof.is_some() {
                self.full_barrier(cmd);
            }
            match op {
                PlannedOp::Embed => {
                    // Fonte: a linha de cada token do bloco (primeiro shard) ou a stream
                    // residual que veio da GPU anterior. Vai para `embd_stage` e daí para b_x.
                    let bytes = (c.n_embd * n_tok * 4) as vk::DeviceSize;
                    // Dequantiza só as linhas deste passo: uma no decode, `n_tok` no
                    // bloco de prefill. A tabela inteira em f32 custaria 5,1 GB de RAM.
                    let linhas: Option<Vec<f32>> = match (x_in, st.token_embd.as_ref()) {
                        (None, Some(te)) => te.linhas(tokens).ok(),
                        _ => None,
                    };
                    if let Some(src) = x_in.or(linhas.as_deref()) {
                        // SAFETY: embd_stage é host-visible/coherent com `bytes`;
                        // o ponteiro é válido até unmap e a cópia respeita n_embd floats.
                        unsafe {
                            if let Ok(ptr) = d.map_memory(
                                st.embd_stage.mem,
                                0,
                                bytes,
                                vk::MemoryMapFlags::empty(),
                            ) {
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr(),
                                    ptr as *mut f32,
                                    c.n_embd * n_tok,
                                );
                                d.unmap_memory(st.embd_stage.mem);
                            }
                        }
                    }
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: bytes,
                    };
                    // SAFETY: cmd em gravação; buffers vivos; offsets dentro do tamanho.
                    unsafe {
                        d.cmd_copy_buffer(cmd, st.embd_stage.buffer, st.b_x.buffer, &[region]);
                    }
                }
                PlannedOp::Copia { src, dst, bytes } => {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: *bytes,
                    };
                    // SAFETY: cmd em gravação; os dois buffers vivem no state, e `bytes`
                    // é o tamanho do menor deles (fixado na construção do plano).
                    unsafe { d.cmd_copy_buffer(cmd, *src, *dst, &[region]) };
                }
                PlannedOp::KvAppend { slot, com_k } => {
                    // As posições do bloco são consecutivas no cache e `b_k`/`b_v` estão
                    // token-major, então uma cópia cobre os N tokens.
                    let pos0 = pos + 1 - n_tok;
                    let off = ((slot * c.ctx + pos0) * c.kv_dim * 4) as vk::DeviceSize;
                    let sz = (c.kv_dim * n_tok * 4) as vk::DeviceSize;
                    let rk = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: off,
                        size: sz,
                    };
                    // SAFETY: idem.
                    unsafe {
                        // Com o `rope_kv` ligado, K já foi escrito no slot pelo próprio RoPE.
                        if *com_k {
                            d.cmd_copy_buffer(cmd, st.b_k.buffer, st.kcache.buffer, &[rk]);
                        }
                        d.cmd_copy_buffer(cmd, st.b_v.buffer, st.vcache.buffer, &[rk]);
                    }
                }
                // Só o plano da cabeça MTP as usa — ver `record_mtp`.
                PlannedOp::KvAppendMtp | PlannedOp::CopiaHidden => {}
                PlannedOp::Dispatch { .. } => {
                    self.gravar_dispatch(cmd, op, c, pos, total_len, 1, 1);
                }
                PlannedOp::Atencao {
                    curto,
                    split,
                    reduce,
                } => {
                    // Contexto curto: um workgroup por cabeça, como sempre foi. Longo: o
                    // KV fatiado entre `n_split` workgroups e a redução dos parciais.
                    let n_split = splits_do_kv(total_len);
                    if n_split <= 1 {
                        self.gravar_dispatch(cmd, curto, c, pos, total_len, 1, 1);
                    } else {
                        self.gravar_dispatch(cmd, split, c, pos, total_len, n_split, n_split);
                        // A redução lê o que as fatias acabaram de escrever.
                        self.full_barrier(cmd);
                        self.gravar_dispatch(cmd, reduce, c, pos, total_len, 1, n_split);
                    }
                }
            }
            if let Some(pf) = prof {
                // SAFETY: cmd em gravação; o pool foi dimensionado com plan.len()+1 slots.
                unsafe {
                    d.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        pf.pool,
                        (op_idx + 1) as u32,
                    );
                }
            }
        }

        // Fecha o último grupo de ops antes de ler o que ele escreveu.
        self.full_barrier(cmd);

        // Saída deste shard para o staging host-visible, no mesmo command buffer. O último
        // shard entrega logits; os demais entregam a stream residual, que segue para a
        // próxima GPU.
        // No último shard só o último token do bloco tem logits (ver `build_plan`); nos
        // demais a stream residual sai com um vetor por token.
        // No verify os dois tokens têm logits: `vocab × 2`, o do primeiro token primeiro.
        let (src, n_out) = if c.shard.is_last() {
            (
                &st.b_logits,
                if modo == Modo::Verify {
                    c.vocab * VERIFY_TOK
                } else {
                    c.vocab
                },
            )
        } else {
            (&st.b_x, c.n_embd * n_tok)
        };
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (n_out * 4) as vk::DeviceSize,
        };
        // SAFETY: cmd em gravação; ambos os buffers vivem no state.
        unsafe {
            d.cmd_copy_buffer(cmd, src.buffer, st.logits_host.buffer, &[region]);
        }
    }

    /// Grava o plano da cabeça MTP em `cmd` (já em `begin`).
    ///
    /// `hidden_idx` é qual dos hidden do último passo alimenta a cabeça: 0 depois de um
    /// decode simples ou de um verify rejeitado, 1 depois de um verify aceito — porque aí
    /// o token de que se parte veio dos logits do **segundo** token do bloco.
    fn record_mtp(&self, cmd: vk::CommandBuffer, hidden_idx: usize) {
        let d = &self.dev.device;
        let st = self.state.as_ref().expect("record_mtp requer state");
        let c = &st.cfg;
        let Some(m) = st.mtp.as_ref() else { return };
        let pos = *m.len.borrow();
        let total_len = (pos + 1) as u32;

        for (i, op) in st.mtp_plan.iter().enumerate() {
            if st.mtp_barreiras.get(i).copied().unwrap_or(true) {
                self.full_barrier(cmd);
            }
            match op {
                PlannedOp::Dispatch { .. } => {
                    self.gravar_dispatch(cmd, op, c, pos, total_len, 1, 1);
                }
                PlannedOp::Copia { src, dst, bytes } => {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: *bytes,
                    };
                    // SAFETY: cmd em gravação; buffers vivos no state.
                    unsafe { d.cmd_copy_buffer(cmd, *src, *dst, &[region]) };
                }
                PlannedOp::CopiaHidden => {
                    // Com `HIDDEN_CABECA` a origem é o residual do próprio bloco MTP da
                    // proposta anterior (`m.b_x`, ainda intacto: o `eh_proj` só o
                    // sobrescreve depois desta cópia) — é o encadeamento n=2. Senão, um
                    // dos hidden do tronco em `st.b_x`.
                    let (src, src_offset) = if hidden_idx == llama_model::HIDDEN_CABECA {
                        (m.b_x.buffer, 0)
                    } else {
                        (st.b_x.buffer, (hidden_idx * c.n_embd * 4) as vk::DeviceSize)
                    };
                    let region = vk::BufferCopy {
                        src_offset,
                        dst_offset: 0,
                        size: (c.n_embd * 4) as vk::DeviceSize,
                    };
                    // SAFETY: `b_x` cobre `n_embd * nblk` floats e `hidden_idx` é menor
                    // que o bloco do último passo (ou o sentinel, que lê `m.b_x` em 0).
                    unsafe { d.cmd_copy_buffer(cmd, src, m.b_h.buffer, &[region]) };
                }
                PlannedOp::KvAppendMtp => {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: (pos * c.kv_dim * 4) as vk::DeviceSize,
                        size: (c.kv_dim * 4) as vk::DeviceSize,
                    };
                    // SAFETY: o cache do bloco tem `ctx` posições e `pos < ctx` (conferido
                    // em `propor_mtp`).
                    unsafe {
                        d.cmd_copy_buffer(cmd, st.b_k.buffer, m.kcache.buffer, &[region]);
                        d.cmd_copy_buffer(cmd, st.b_v.buffer, m.vcache.buffer, &[region]);
                    }
                }
                PlannedOp::Atencao {
                    curto,
                    split,
                    reduce,
                } => {
                    let n_split = splits_do_kv(total_len);
                    if n_split <= 1 {
                        self.gravar_dispatch(cmd, curto, c, pos, total_len, 1, 1);
                    } else {
                        self.gravar_dispatch(cmd, split, c, pos, total_len, n_split, n_split);
                        self.full_barrier(cmd);
                        self.gravar_dispatch(cmd, reduce, c, pos, total_len, 1, n_split);
                    }
                }
                // O plano da cabeça não faz embedding lookup nem toca no KV do modelo.
                PlannedOp::Embed | PlannedOp::KvAppend { .. } => {}
            }
        }
        self.full_barrier(cmd);
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (c.vocab * 4) as vk::DeviceSize,
        };
        // SAFETY: cmd em gravação; `logits_host` tem pelo menos `vocab` floats.
        unsafe {
            d.cmd_copy_buffer(cmd, st.b_logits.buffer, st.logits_host.buffer, &[region]);
        }
    }

    /// A linha de embedding de `token`, quando este shard carrega a tabela (o primeiro).
    /// A tabela fica quantizada (emprestada do GGUF), então a linha sai dequantizada aqui.
    pub fn linha_embd(&self, token: u32) -> Option<Vec<f32>> {
        let st = self.state.as_ref()?;
        st.token_embd.as_ref()?.linha(token).ok()
    }

    /// Propõe o token seguinte com a cabeça MTP residente.
    ///
    /// `emb` é a linha crua da tabela de embedding do token já amostrado (o shard que tem
    /// a tabela é o primeiro, e a cabeça mora no último — daí ela vir de fora, 20 KB por
    /// passo). `hidden_idx` escolhe qual hidden do último passo alimenta a cabeça.
    ///
    /// O contador de posições do bloco anda **um por proposta**, como a referência de CPU
    /// `MtpHead::propor`: o KV-cache da cabeça é dela, não acompanha o do modelo.
    pub fn propor_mtp(&self, emb: &[f32], hidden_idx: usize) -> Result<u32, MatmulError> {
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let faltando = || MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        let m = st.mtp.as_ref().ok_or_else(faltando)?;
        if st.mtp_plan.is_empty() || emb.len() != st.cfg.n_embd || *m.len.borrow() >= st.cfg.ctx {
            return Err(faltando());
        }
        let d = &self.dev.device;
        let bytes = (st.cfg.n_embd * 4) as vk::DeviceSize;
        // SAFETY: `emb_stage` é host-visible/coherent com `n_embd` floats.
        unsafe {
            let ptr = d.map_memory(m.emb_stage.mem, 0, bytes, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(emb.as_ptr(), ptr.cast::<f32>(), st.cfg.n_embd);
            d.unmap_memory(m.emb_stage.mem);
        }

        let cmd = st.token_cmd;
        // SAFETY: pool com RESET_COMMAND_BUFFER; o fence do passo anterior já foi aguardado.
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            d.begin_command_buffer(cmd, &begin)?;
        }
        self.record_mtp(cmd, hidden_idx);
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        // SAFETY: cmd em gravação, fence resetado antes do submit.
        unsafe {
            d.end_command_buffer(cmd)?;
            d.reset_fences(&[st.token_fence])?;
            d.queue_submit(self.dev.queue, &[submit], st.token_fence)?;
        }
        self.espera_fence(st)?;
        *m.len.borrow_mut() += 1;

        // Argmax no host, sobre o mapa persistente de `logits_host`. A cabeça é greedy por
        // construção: ela propõe um candidato, e quem decide é a verificação.
        // SAFETY: host-coherent, a cópia terminou (fence aguardado) e o mapa cobre `vocab`.
        let logits =
            unsafe { std::slice::from_raw_parts(st.logits_ptr.cast::<f32>(), st.cfg.vocab) };
        let mut melhor = 0usize;
        let mut valor = f32::NEG_INFINITY;
        for (i, &x) in logits.iter().enumerate() {
            if x > valor {
                valor = x;
                melhor = i;
            }
        }
        u32::try_from(melhor).map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_UNKNOWN))
    }

    /// Se este shard carrega a cabeça MTP residente.
    pub fn tem_mtp(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|st| !st.mtp_plan.is_empty())
    }

    pub(crate) fn pipe_of(&self, id: PipeId) -> &ComputePipeline {
        match id {
            PipeId::Matvec => &self.matvec,
            PipeId::MatvecQ5K => &self.matvec_q5k,
            PipeId::MatvecQ6K => &self.matvec_q6k,
            PipeId::MatvecQ4K => &self.matvec_q4k,
            PipeId::MatvecB => &self.matvec_b,
            PipeId::MatvecQ5KB => &self.matvec_q5k_b,
            PipeId::MatvecQ6KB => &self.matvec_q6k_b,
            PipeId::MatvecQ4KB => &self.matvec_q4k_b,
            PipeId::MulMmQ4K => &self.mul_mm_q4k,
            PipeId::MatvecV => &self.matvec_v,
            PipeId::MatvecQ5KV => &self.matvec_q5k_v,
            PipeId::MatvecQ6KV => &self.matvec_q6k_v,
            PipeId::MatvecQ4KV => &self.matvec_q4k_v,
            PipeId::DeltaNet => &self.delta_net,
            PipeId::DnConv => &self.dn_conv,
            PipeId::DnGates => &self.dn_gates,
            PipeId::DnNorm => &self.dn_norm,
            PipeId::DnL2Qk => &self.dn_l2_qk,
            PipeId::GateQuant => &self.gate_quant,
            PipeId::QuantizeX => &self.quantize_x,
            PipeId::NormFused => &self.norm_fused,
            PipeId::NormP2 => &self.norm_p2,
            PipeId::Rope => &self.rope,
            PipeId::RopeKv => &self.rope_kv,
            PipeId::Attention => &self.attention,
            PipeId::AttentionSplit => &self.attention_split,
            PipeId::AttnReduce => &self.attn_reduce,
            PipeId::SwigluQuant => &self.swiglu_quant,
            PipeId::Add => &self.add,
        }
    }

    /// Ajusta a dimensão Y de um dispatch já montado — o token do bloco nos shaders que
    /// batcham por `gl_WorkGroupID.y`.
    fn com_y(op: PlannedOp, y: u32) -> PlannedOp {
        match op {
            PlannedOp::Dispatch {
                pipe,
                set,
                groups,
                push,
                le,
                esc,
                bytes,
                ..
            } => PlannedOp::Dispatch {
                pipe,
                set,
                groups,
                groups_y: y,
                push,
                le,
                esc,
                // Os bindings destes dispatches já cobrem o bloco inteiro (`nbt`), então a
                // dimensão Y não multiplica os bytes lidos.
                bytes,
            },
            outra => outra,
        }
    }

    /// Monta a lista de ops de um bloco de `n_tok` tokens (ordem idêntica a `decode_step`)
    /// e pré-aloca/escreve um descriptor set por dispatch (bindings estáticos entre blocos).
    ///
    /// `n_tok == 1` é o decode. Acima disso é o prefill em batch, e cada op vira batch de
    /// um destes três jeitos:
    ///
    /// - **`COLS`** nos matvec — o peso sai da VRAM uma vez para as N ativações;
    /// - **`gl_WorkGroupID.y`** em `attention`, `rope`, `norm_fused` e `norm_p2`;
    /// - **`n × N` elementos** no que já é token-major (`quantize_x`, `swiglu_quant`,
    ///   `add`, `gate_quant`) — nada muda no shader.
    ///
    /// O que sobra são as ops com estado (`dn_conv`, `delta_net`) e as que indexam peso
    /// pela cabeça (`dn_gates`): essas viram N dispatches com os bindings deslocados,
    /// executados em ordem.
    /// Fecha um dispatch do plano: aloca e escreve o descriptor set (os bindings são fixos
    /// entre passos) e deriva de `PipeId::acessos` as faixas que `marcar_barreiras` lê.
    fn mk_op(
        &self,
        pipe: PipeId,
        binds: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
        groups: u32,
        push: PushSpec,
    ) -> Result<PlannedOp, MatmulError> {
        let d = &self.dev.device;
        let set = self.alloc_set(self.pipe_of(pipe))?;
        let buf_infos: Vec<vk::DescriptorBufferInfo> = binds
            .iter()
            .map(|&(buffer, offset, range)| vk::DescriptorBufferInfo {
                buffer,
                offset,
                range,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(b, bi)| vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: b as u32,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: bi,
                ..Default::default()
            })
            .collect();
        // SAFETY: d válido; writes apontam para buf_infos vivos; set nunca em uso durante o build.
        unsafe { d.update_descriptor_sets(&writes, &[]) };
        let (le_idx, esc_idx) = pipe.acessos();
        let faixas = |idx: &[usize]| -> Vec<Faixa> {
            idx.iter().filter_map(|&i| binds.get(i).copied()).collect()
        };
        let le = faixas(le_idx);
        // Bytes lidos por dispatch, para a coluna `GB/s` do perfil: a soma das faixas de
        // leitura, descontando o mesmo binding ligado duas vezes (o matvec repete `xd`
        // nos bindings 2 e 4). É aproximação **por cima** — a faixa é o binding inteiro,
        // não o que o shader de fato toca —, e por isso a atenção fica de fora: ela liga
        // o KV-cache inteiro e lê só `total_len` posições, número que só existe na
        // gravação do command buffer. Nos matvec, que são 77% do token, o peso domina a
        // soma e o erro da ativação fica abaixo de 0,1%.
        let bytes = match pipe {
            PipeId::Attention | PipeId::AttentionSplit | PipeId::AttnReduce => 0,
            _ => {
                let mut vistas: Vec<Faixa> = Vec::with_capacity(le.len());
                for f in &le {
                    if !vistas.contains(f) {
                        vistas.push(*f);
                    }
                }
                vistas.iter().map(|f| f.2).sum()
            }
        };
        Ok(PlannedOp::Dispatch {
            pipe,
            set,
            groups,
            groups_y: 1,
            push,
            le,
            esc: faixas(esc_idx),
            bytes,
        })
    }

    fn build_plan(&self, modo: Modo) -> Result<Vec<PlannedOp>, MatmulError> {
        use crate::pipeline::PushConstants;
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let c = &st.cfg;
        let n_tok = modo.n_tok(c);
        let nt = u32::try_from(n_tok).unwrap_or(1);
        let mut plan = Vec::new();

        let n_push = |n: usize| -> Vec<u8> {
            #[repr(C)]
            struct P {
                n: u32,
            }
            let p = P { n: n as u32 };
            unsafe { std::slice::from_raw_parts(&p as *const P as *const u8, 4) }.to_vec()
        };
        let mv_push = |n_in: usize, n_out: usize| -> Vec<u8> {
            let p = PushConstants {
                n_in: n_in as u32,
                n_out: n_out as u32,
                row_offset: 0,
                // O bias das projeções entra por ops `Add` próprias, adiante no plano; o
                // binding 4 é preenchido só para completar o layout.
                tem_bias: 0,
            };
            unsafe {
                std::slice::from_raw_parts(
                    &p as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                )
            }
            .to_vec()
        };
        // Linhas que um workgroup do Q5_K cobre — tem de casar com as spec constants da
        // pipeline **deste plano**, senão sobram ou faltam workgroups. O bloco de prefill
        // tem geometria própria (`matvec_geom_batch`), e usar a do decode aqui deixava o
        // dispatch cobrindo uma fração das linhas: o kernel ficava mais rápido por não
        // fazer o trabalho (erro de 85% nos logits, pego pelo teste do prefill).
        let rows_q5k = {
            let (wg, rows) = if modo == Modo::Batch && n_tok > 1 {
                matvec_geom_batch()
            } else {
                matvec_geom()
            };
            (wg / 64 * rows) as usize
        };
        // Push do quantize: n_in + 2 pads (o range declarado no pipeline e de 12 bytes).
        let qx_push = |n_in: usize| -> Vec<u8> {
            let p: [u32; 3] = [n_in as u32, 0, 0];
            unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), 12) }.to_vec()
        };
        let qx_groups = |n_in: usize| -> u32 { ((n_in / 32) as u32).div_ceil(64) };
        // Workgroups do passo 1 da norma: um por 256 elementos, até o teto.
        let np1_wg = ((c.n_embd as u32).div_ceil(256)).clamp(1, NORM_P1_WG);
        // Push do passo 1: dim, tem_residual.
        let np1_push = |tem_residual: bool| -> Vec<u8> {
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&u32::try_from(c.n_embd).unwrap_or(0).to_le_bytes());
            v.extend_from_slice(&u32::from(tem_residual).to_le_bytes());
            v
        };
        // Push do passo 2: dim, eps, n_parciais.
        let np2_push = || -> Vec<u8> {
            let mut v = Vec::with_capacity(12);
            v.extend_from_slice(&u32::try_from(c.n_embd).unwrap_or(0).to_le_bytes());
            v.extend_from_slice(&c.rms_eps.to_le_bytes());
            v.extend_from_slice(&np1_wg.to_le_bytes());
            v
        };

        let mk = |pipe: PipeId,
                  binds: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
                  groups: u32,
                  push: PushSpec|
         -> Result<PlannedOp, MatmulError> {
            self.mk_op(pipe, binds, groups, push)
        };

        let nb = |n: usize| (n * 4) as vk::DeviceSize;
        // Faixa de um buffer por token, cobrindo o bloco inteiro.
        let nbt = |n: usize| ((n * n_tok) * 4) as vk::DeviceSize;

        // Emite o matvec com o shader do tipo do peso. Os três tipos consomem a mesma
        // ativação int8 (`b_xq`/`b_xd`, produzida pelo dispatch QuantizeX): os sub-blocos
        // dos K-quants têm 32 elementos, exatamente o bloco de quantização do Q8_0, então
        // as escalas casam sem nenhuma conversão.
        //
        // `cols` é quantas colunas (tokens) o shader acumula contra o mesmo peso já lido:
        // `n_tok` no bloco de prefill, 1 no decode e na projeção de logits (onde só o
        // último token do bloco interessa). O número de workgroups não muda com `cols` —
        // continua sendo uma partição das linhas de saída.
        let mv_gen = |w: &QWeight,
                      dst: &Buf,
                      ativ: (&Buf, &Buf),
                      cols: usize,
                      n_in: usize,
                      n_out: usize|
         -> Result<PlannedOp, MatmulError> {
            let (xq, xd) = ativ;
            // `COLS` é specialization constant, então cada largura é um trio de pipelines
            // próprio. A escolha é pelo **modo do plano**, não por `cols`: com
            // `LLAMA_RS_BATCH=2` as larguras do prefill e do verify coincidem mas as
            // geometrias não, e usar a pipeline errada faz o dispatch cobrir uma fração
            // das linhas de saída — em silêncio.
            let largura = match modo {
                _ if cols <= 1 => Modo::Decode,
                Modo::Verify => Modo::Verify,
                _ => Modo::Batch,
            };
            // A ativação é lida uma vez para todas as linhas do wave; quantas linhas cada
            // workgroup cobre sai de `matvec_geom` (Q5_K/Q4_K) ou da geometria fixa dos
            // shaders Q8_0 e Q6_K. O GEMM, quando ligado, só entra no bloco de prefill
            // (modo Batch) e cobre `BM` linhas por workgroup — o verify fica no matvec.
            let (pipe, rows_por_wg) = if matches!(largura, Modo::Batch) && gemm_para(cols, w.ty) {
                (PipeId::MulMmQ4K, GEMM_LINHAS_POR_WG as usize)
            } else {
                match w.ty {
                    gguf::GgmlType::Q8_0 => (
                        match largura {
                            Modo::Decode => PipeId::Matvec,
                            Modo::Batch => PipeId::MatvecB,
                            Modo::Verify => PipeId::MatvecV,
                        },
                        MATVEC_NUM_ROWS as usize,
                    ),
                    gguf::GgmlType::Q5_K => (
                        match largura {
                            Modo::Decode => PipeId::MatvecQ5K,
                            Modo::Batch => PipeId::MatvecQ5KB,
                            Modo::Verify => PipeId::MatvecQ5KV,
                        },
                        rows_q5k,
                    ),
                    gguf::GgmlType::Q4_K => (
                        match largura {
                            Modo::Decode => PipeId::MatvecQ4K,
                            Modo::Batch => PipeId::MatvecQ4KB,
                            Modo::Verify => PipeId::MatvecQ4KV,
                        },
                        rows_q5k,
                    ),
                    _ => (
                        match largura {
                            Modo::Decode => PipeId::MatvecQ6K,
                            Modo::Batch => PipeId::MatvecQ6KB,
                            Modo::Verify => PipeId::MatvecQ6KV,
                        },
                        8,
                    ),
                }
            };
            mk(
                pipe,
                &[
                    (w.gpu.buffer, 0, w.gpu.size_bytes),
                    (xq.buffer, 0, xq.size),
                    (xd.buffer, 0, xd.size),
                    (dst.buffer, 0, nb(n_out * cols)),
                    // Binding 4 = bias; sem bias fundido no matvec, repete `xd` (já lido no
                    // binding 2, então `marcar_barreiras` enxerga a mesma faixa).
                    (xd.buffer, 0, xd.size),
                ],
                u32::try_from(n_out.div_ceil(rows_por_wg)).unwrap_or(u32::MAX),
                PushSpec::Static(mv_push(n_in, n_out)),
            )
        };

        let mv =
            |w: &QWeight, dst: &Buf, n_in: usize, n_out: usize| -> Result<PlannedOp, MatmulError> {
                mv_gen(w, dst, (&st.b_xq, &st.b_xd), n_tok, n_in, n_out)
            };

        // Como `mv`, mas com a ativação vinda de buffers escolhidos pelo chamador.
        let mv_com = |w: &QWeight,
                      dst: &Buf,
                      ativ: (&Buf, &Buf),
                      n_in: usize,
                      n_out: usize|
         -> Result<PlannedOp, MatmulError> {
            mv_gen(w, dst, ativ, n_tok, n_in, n_out)
        };

        // As duas ops da norma, em sequência: `r` é o residual a somar (ignorado quando
        // `tem_residual` é falso) e `w` o peso da norma.
        // Os dois passos batcham pela dimensão Y: um workgroup por (fatia, token).
        let norma = |r: &Buf, w: &Buf, tem_residual: bool| -> Result<[PlannedOp; 2], MatmulError> {
            Ok([
                Self::com_y(
                    mk(
                        PipeId::NormFused,
                        &[
                            (st.b_x.buffer, 0, nbt(c.n_embd)),
                            (r.buffer, 0, nbt(c.n_embd)),
                            (st.b_parciais.buffer, 0, st.b_parciais.size),
                        ],
                        np1_wg,
                        PushSpec::Static(np1_push(tem_residual)),
                    )?,
                    nt,
                ),
                Self::com_y(
                    mk(
                        PipeId::NormP2,
                        &[
                            (st.b_x.buffer, 0, nbt(c.n_embd)),
                            (w.buffer, 0, w.size),
                            (st.b_parciais.buffer, 0, st.b_parciais.size),
                            (st.b_normed.buffer, 0, nbt(c.n_embd)),
                            (st.b_xq.buffer, 0, st.b_xq.size),
                            (st.b_xd.buffer, 0, st.b_xd.size),
                        ],
                        qx_groups(c.n_embd),
                        PushSpec::Static(np2_push()),
                    )?,
                    nt,
                ),
            ])
        };

        // Presente em todos os shards: no primeiro copia a linha do token da tabela de
        // embedding; nos demais copia a stream residual vinda da GPU anterior. Nos dois
        // casos o host escreve em `embd_stage` e a cópia entra no command buffer do token.
        plan.push(PlannedOp::Embed);

        // Diagnóstico: `LLAMA_RS_STOP_LAYER=N` executa só as N primeiras camadas do
        // shard. Com N=0 o token sai do embedding direto para a projeção final, o que dá
        // uma linha de base para saber se as camadas estão de fato contribuindo.
        let parar_em: Option<usize> = std::env::var("LLAMA_RS_STOP_LAYER")
            .ok()
            .and_then(|v| v.parse().ok());
        // Com `LLAMA_RS_STOP_LAYER=0` nenhuma camada roda, e aí não há residual do FFN a
        // somar: `b_ffn` guarda o que sobrou do token anterior.
        let rodou_camada = c.n_layer > 0 && parar_em != Some(0);
        // Slot no KV-cache de cada camada: as delta-net não têm nenhum.
        let (slot_kv, _) = slots_kv(
            st.qw
                .iter()
                .map(|l| !matches!(l.mixer, MixerQ::Delta { .. })),
        );
        for l in 0..c.n_layer {
            if parar_em.is_some_and(|n| l >= n) {
                break;
            }
            let lq = &st.qw[l];
            let la = &st.aux[l];

            // Norma de entrada da camada. Ela **absorve o residual do FFN da camada
            // anterior** (`b_ffn`), que antes era um `Add` próprio no fim do laço: com
            // `l == 0` não há o que somar, porque `b_x` acabou de vir do embedding (ou da
            // GPU anterior, nos shards seguintes).
            plan.extend(norma(&st.b_ffn, &la.attn_norm, l > 0)?);
            // Camada de atenção linear (qwen35): caminho próprio, que troca o KV-cache
            // por estado recorrente. Os dois caminhos deixam o resultado em `b_proj`, e o
            // fecho da camada (residual + norma + FFN) é comum.
            let eh_delta = matches!(&lq.mixer, MixerQ::Delta { .. });
            if let MixerQ::Delta {
                attn_qkv,
                attn_gate,
                ssm_out,
            } = &lq.mixer
            {
                let pesos = (attn_qkv, attn_gate, ssm_out);
                if modo == Modo::Verify {
                    Self::plano_delta_verify(&mut plan, st, la, c, pesos, &mk, &mv, &mv_com)?;
                } else {
                    Self::plano_delta(&mut plan, st, la, c, n_tok, pesos, &mk, &mv, &mv_com)?;
                }
            }
            if !eh_delta {
                let (w_q, w_k, w_v, w_o) = match &lq.mixer {
                    MixerQ::Attn {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                    } => (attn_q, attn_k, attn_v, attn_output),
                    MixerQ::Delta { .. } => unreachable!("tratado acima"),
                };
                // No qwen35 a projeção de Q sai com query e gate por cabeça (2 × head_dim),
                // e o conjunto de cabeças (head_dim × n_head) não é n_embd.
                let hib = c.delta_net.is_some();
                let attn_dim = if hib { c.head_dim * c.n_head } else { c.n_embd };
                let q_out = if hib { attn_dim * 2 } else { c.n_embd };
                plan.push(mv(w_q, &st.b_q, c.n_embd, q_out)?);
                plan.push(mv(w_k, &st.b_k, c.n_embd, c.kv_dim)?);
                plan.push(mv(w_v, &st.b_v, c.n_embd, c.kv_dim)?);
                // QK-norm: RMSNorm por cabeça, in-place. No Q as cabeças estão espaçadas de
                // 2 × head_dim porque o gate mora ao lado da query.
                if let (Some(qn), Some(kn)) = (&la.q_norm, &la.k_norm) {
                    let push_qk = |n_heads: u32, stride: u32| {
                        let mut v = Vec::with_capacity(20);
                        v.extend_from_slice(&u32::try_from(c.head_dim).unwrap_or(0).to_le_bytes());
                        v.extend_from_slice(&n_heads.to_le_bytes());
                        v.extend_from_slice(&2u32.to_le_bytes()); // modo QK-norm
                        v.extend_from_slice(&c.rms_eps.to_le_bytes());
                        v.extend_from_slice(&stride.to_le_bytes());
                        v
                    };
                    // O `dn_norm` indexa a cabeça por `h * stride`, e as cabeças dos N
                    // tokens são contíguas nesse mesmo passo — então o batch é só
                    // multiplicar a contagem de cabeças, sem tocar no shader.
                    let heads_q = c.n_head * n_tok;
                    let heads_k = c.n_head_kv * n_tok;
                    plan.push(mk(
                        PipeId::DnNorm,
                        &[
                            (st.b_q.buffer, 0, nbt(q_out)),
                            (qn.buffer, 0, qn.size),
                            (st.b_q.buffer, 0, nbt(q_out)),
                            (st.b_q.buffer, 0, nbt(q_out)),
                        ],
                        u32::try_from(heads_q).unwrap_or(u32::MAX),
                        PushSpec::Static(push_qk(
                            u32::try_from(heads_q).unwrap_or(u32::MAX),
                            u32::try_from(c.head_dim * 2).unwrap_or(u32::MAX),
                        )),
                    )?);
                    plan.push(mk(
                        PipeId::DnNorm,
                        &[
                            (st.b_k.buffer, 0, nbt(c.kv_dim)),
                            (kn.buffer, 0, kn.size),
                            (st.b_k.buffer, 0, nbt(c.kv_dim)),
                            (st.b_k.buffer, 0, nbt(c.kv_dim)),
                        ],
                        u32::try_from(heads_k).unwrap_or(u32::MAX),
                        PushSpec::Static(push_qk(
                            u32::try_from(heads_k).unwrap_or(u32::MAX),
                            u32::try_from(c.head_dim).unwrap_or(u32::MAX),
                        )),
                    )?);
                }
                // O bias é o mesmo vetor para todos os tokens, e o `add` não sabe repetir
                // a fonte — então é um dispatch por token, com o destino deslocado.
                for (bias, dim, dst) in [
                    (&la.q_bias, c.n_embd, &st.b_q),
                    (&la.k_bias, c.kv_dim, &st.b_k),
                    (&la.v_bias, c.kv_dim, &st.b_v),
                ] {
                    let Some(b) = bias else { continue };
                    for t in 0..n_tok {
                        plan.push(mk(
                            PipeId::Add,
                            &[(dst.buffer, nb(t * dim), nb(dim)), (b.buffer, 0, b.size)],
                            Self::groups_for(dim),
                            PushSpec::Static(n_push(dim)),
                        )?);
                    }
                }
                // RoPE e atenção batcham pela dimensão Y: cada token do bloco tem a sua
                // posição (`pos - (n_tok-1) + t`) e a sua máscara causal.
                plan.push(Self::com_y(
                    mk(
                        PipeId::Rope,
                        &[
                            (st.b_q.buffer, 0, nbt(q_out)),
                            (st.freq_buf.buffer, 0, st.freq_buf.size),
                        ],
                        Self::groups_for(c.n_head * (c.rope_dim / 2)),
                        PushSpec::Rope {
                            n_head: c.n_head as u32,
                            stride: if hib {
                                (c.head_dim * 2) as u32
                            } else {
                                c.head_dim as u32
                            },
                        },
                    )?,
                    nt,
                ));
                let slot = slot_kv.get(l).copied().flatten().unwrap_or(0);
                let layer_off = (slot * c.ctx * c.kv_dim) as u32;
                // K pode ir do RoPE direto para o slot do cache, dispensando a cópia do
                // `kv_append` — ver `rope_no_kv`. Nesse caminho o shader cobre `head_dim/2`
                // pares por cabeça em vez de `rope_dim/2`: o que não gira ainda precisa ser
                // copiado, coisa que o RoPE in-place ganhava de graça.
                let com_k = !rope_no_kv();
                plan.push(Self::com_y(
                    if com_k {
                        mk(
                            PipeId::Rope,
                            &[
                                (st.b_k.buffer, 0, nbt(c.kv_dim)),
                                (st.freq_buf.buffer, 0, st.freq_buf.size),
                            ],
                            Self::groups_for(c.n_head_kv * (c.rope_dim / 2)),
                            PushSpec::Rope {
                                n_head: c.n_head_kv as u32,
                                stride: c.head_dim as u32,
                            },
                        )?
                    } else {
                        mk(
                            PipeId::RopeKv,
                            &[
                                (st.b_k.buffer, 0, nbt(c.kv_dim)),
                                (st.freq_buf.buffer, 0, st.freq_buf.size),
                                (st.kcache.buffer, 0, st.kcache.size),
                            ],
                            Self::groups_for(c.n_head_kv * (c.head_dim / 2)),
                            PushSpec::RopeKv {
                                n_head: c.n_head_kv as u32,
                                kv_layer_off: layer_off,
                            },
                        )?
                    },
                    nt,
                ));
                plan.push(PlannedOp::KvAppend { slot, com_k });
                // Os dois caminhos da atenção ficam prontos; a gravação escolhe pelo
                // comprimento do KV (ver `splits_do_kv`).
                let attn_bind = [
                    (st.b_q.buffer, 0, nbt(q_out)),
                    (st.kcache.buffer, 0, st.kcache.size),
                    (st.vcache.buffer, 0, st.vcache.size),
                    (st.b_attn.buffer, 0, nbt(attn_dim)),
                ];
                let curto = Self::com_y(
                    mk(
                        PipeId::Attention,
                        &attn_bind,
                        c.n_head as u32,
                        PushSpec::Attention {
                            kv_layer_off: layer_off,
                        },
                    )?,
                    nt,
                );
                let split = Self::com_y(
                    mk(
                        PipeId::AttentionSplit,
                        &[
                            (st.b_q.buffer, 0, nbt(q_out)),
                            (st.kcache.buffer, 0, st.kcache.size),
                            (st.vcache.buffer, 0, st.vcache.size),
                            (st.b_attn_split.buffer, 0, st.b_attn_split.size),
                        ],
                        c.n_head as u32,
                        PushSpec::Attention {
                            kv_layer_off: layer_off,
                        },
                    )?,
                    nt,
                );
                let reduce = Self::com_y(
                    mk(
                        PipeId::AttnReduce,
                        &[
                            (st.b_attn_split.buffer, 0, st.b_attn_split.size),
                            (st.b_attn.buffer, 0, nbt(attn_dim)),
                        ],
                        c.n_head as u32,
                        PushSpec::AttnReduce,
                    )?,
                    nt,
                );
                plan.push(PlannedOp::Atencao {
                    curto: Box::new(curto),
                    split: Box::new(split),
                    reduce: Box::new(reduce),
                });
                if !hib {
                    plan.push(mk(
                        PipeId::QuantizeX,
                        &[
                            (st.b_attn.buffer, 0, nbt(c.n_embd)),
                            (st.b_xq.buffer, 0, st.b_xq.size),
                            (st.b_xd.buffer, 0, st.b_xd.size),
                        ],
                        qx_groups(c.n_embd * n_tok),
                        PushSpec::Static(qx_push(c.n_embd * n_tok)),
                    )?);
                }
                if hib {
                    // Portão do qwen35: a saída da atenção passa por sigmoid(gate), com o
                    // gate vindo da segunda metade da própria projeção de Q — e sai daqui já
                    // quantizada para o matvec de saída, numa passada só (`gate_quant`).
                    // O portão escrevia `b_attn` inteiro e o `quantize_x` relia o mesmo
                    // buffer logo em seguida, com barreira no meio.
                    let mut pg = Vec::with_capacity(8);
                    pg.extend_from_slice(
                        &u32::try_from(attn_dim * n_tok).unwrap_or(0).to_le_bytes(),
                    );
                    pg.extend_from_slice(&u32::try_from(c.head_dim).unwrap_or(0).to_le_bytes());
                    // `gate_quant` já é token-major: com `n = attn_dim × n_tok` o `h` do
                    // shader avança sozinho para `t * n_head + hh`, que é o layout de `b_q`.
                    plan.push(mk(
                        PipeId::GateQuant,
                        &[
                            (st.b_attn.buffer, 0, nbt(attn_dim)),
                            (st.b_q.buffer, 0, nbt(q_out)),
                            (st.b_xq.buffer, 0, st.b_xq.size),
                            (st.b_xd.buffer, 0, st.b_xd.size),
                        ],
                        qx_groups(attn_dim * n_tok),
                        PushSpec::Static(pg),
                    )?);
                }
                plan.push(mv(w_o, &st.b_proj, attn_dim, c.n_embd)?);
            }
            // Norma do FFN, absorvendo o residual do mixer (`b_proj`).
            plan.extend(norma(&st.b_proj, &la.ffn_norm, true)?);
            plan.push(mv(&lq.ffn_gate, &st.b_gate, c.n_embd, c.n_ff)?);
            plan.push(mv(&lq.ffn_up, &st.b_up, c.n_embd, c.n_ff)?);
            // silu(gate) * up já saindo quantizado para o `ffn_down`: eram dois dispatches
            // em todas as camadas, com o swiglu escrevendo `b_act` inteiro e o quantize
            // relendo o mesmo buffer logo depois.
            plan.push(mk(
                PipeId::SwigluQuant,
                &[
                    (st.b_gate.buffer, 0, nbt(c.n_ff)),
                    (st.b_up.buffer, 0, nbt(c.n_ff)),
                    (st.b_act.buffer, 0, nbt(c.n_ff)),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ],
                qx_groups(c.n_ff * n_tok),
                PushSpec::Static(n_push(c.n_ff * n_tok)),
            )?);
            plan.push(mv(&lq.ffn_down, &st.b_ffn, c.n_ff, c.n_embd)?);
            // Sem `Add` aqui: o residual do FFN é somado pela norma da camada seguinte, ou
            // pela norma final logo abaixo.
        }

        if c.shard.is_last() {
            // Norma final, absorvendo o residual do FFN da última camada.
            plan.extend(norma(&st.b_ffn, &st.output_norm_buf, rodou_camada)?);
            // A projeção de logits é o maior matvec do modelo (vocab × n_embd), e no
            // prefill só o **último** token do bloco tem logits que interessam: os outros
            // já estão no KV-cache. Então requantiza esse token no início de `b_xq`/`b_xd`
            // e roda a projeção com COLS=1 — vocab × (n_tok-1) linhas de saída a menos.
            //
            // O verify é o oposto: precisa dos logits dos **dois** tokens (do primeiro sai
            // a decisão de aceitar, do segundo o token seguinte). E não precisa
            // requantizar nada — o `norm_p2` já deixou os dois em `b_xq`/`b_xd` no layout
            // que o matvec de COLS=2 lê (`t * n_blk + b`). O custo é uma leitura do peso
            // Q6_K de 0,63 GB para as duas colunas, não duas leituras.
            if modo == Modo::Verify {
                plan.push(mv_gen(
                    &st.output_w,
                    &st.b_logits,
                    (&st.b_xq, &st.b_xd),
                    VERIFY_TOK,
                    c.n_embd,
                    c.vocab,
                )?);
                return Ok(plan);
            }
            if n_tok > 1 {
                plan.push(mk(
                    PipeId::QuantizeX,
                    &[
                        (st.b_normed.buffer, nb((n_tok - 1) * c.n_embd), nb(c.n_embd)),
                        (st.b_xq.buffer, 0, st.b_xq.size),
                        (st.b_xd.buffer, 0, st.b_xd.size),
                    ],
                    qx_groups(c.n_embd),
                    PushSpec::Static(qx_push(c.n_embd)),
                )?);
            }
            plan.push(mv_gen(
                &st.output_w,
                &st.b_logits,
                (&st.b_xq, &st.b_xd),
                1,
                c.n_embd,
                c.vocab,
            )?);
        } else if rodou_camada {
            // Shard intermediário: a stream residual que segue para a próxima GPU precisa
            // do último residual, e aqui não há norma para absorvê-lo.
            plan.push(mk(
                PipeId::Add,
                &[
                    (st.b_x.buffer, 0, nbt(c.n_embd)),
                    (st.b_ffn.buffer, 0, nbt(c.n_embd)),
                ],
                Self::groups_for(c.n_embd * n_tok),
                PushSpec::Static(n_push(c.n_embd * n_tok)),
            )?);
        }

        Ok(plan)
    }

    /// Plano da cabeça de multi-token prediction. Vazio quando o shard não tem o bloco.
    ///
    /// Não há op nova aqui: é o prólogo (`enorm`/`hnorm`/`eh_proj`) mais **uma camada de
    /// atenção do qwen35 inteira**, a norma final do bloco e a projeção de vocabulário
    /// compartilhada. A concatenação `[enorm(emb) ; hnorm(h)]` é só binding com offset —
    /// as duas normas escrevem nos offsets 0 e `n_embd` do mesmo buffer.
    ///
    /// Quase tudo é emprestado do plano principal, porque a cabeça roda **entre** dois
    /// passos: a GPU está ociosa e os logits do passo anterior já voltaram ao host.
    #[allow(clippy::too_many_lines)]
    fn build_plan_mtp(&self) -> Result<Vec<PlannedOp>, MatmulError> {
        use crate::pipeline::PushConstants;
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let Some(m) = st.mtp.as_ref() else {
            return Ok(Vec::new());
        };
        let c = &st.cfg;
        let hib = c.delta_net.is_some();
        let attn_dim = if hib { c.head_dim * c.n_head } else { c.n_embd };
        let q_out = if hib { attn_dim * 2 } else { c.n_embd };
        let nb = |n: usize| (n * 4) as vk::DeviceSize;
        let u = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let rows_kq = {
            let (wg, rows) = matvec_geom();
            (wg / 64 * rows) as usize
        };
        let mut plan = Vec::new();

        // Matvec de uma coluna, com o destino podendo cair no meio do buffer (é assim que
        // a concatenação do prólogo se resolve).
        let mv = |w: &QWeight,
                  ativ: (&Buf, &Buf),
                  dst: &Buf,
                  dst_off: usize,
                  n_in: usize,
                  n_out: usize|
         -> Result<PlannedOp, MatmulError> {
            let (xq, xd) = ativ;
            let (pipe, rows_por_wg) = match w.ty {
                gguf::GgmlType::Q8_0 => (PipeId::Matvec, MATVEC_NUM_ROWS as usize),
                gguf::GgmlType::Q5_K => (PipeId::MatvecQ5K, rows_kq),
                gguf::GgmlType::Q4_K => (PipeId::MatvecQ4K, rows_kq),
                _ => (PipeId::MatvecQ6K, 8),
            };
            let p = PushConstants {
                n_in: u(n_in),
                n_out: u(n_out),
                row_offset: 0,
                tem_bias: 0,
            };
            // SAFETY: PushConstants é #[repr(C)] de 4 u32; lemos exatamente o seu tamanho.
            let push = unsafe {
                std::slice::from_raw_parts(
                    std::ptr::from_ref(&p).cast::<u8>(),
                    std::mem::size_of::<PushConstants>(),
                )
            }
            .to_vec();
            self.mk_op(
                pipe,
                &[
                    (w.gpu.buffer, 0, w.gpu.size_bytes),
                    (xq.buffer, 0, xq.size),
                    (xd.buffer, 0, xd.size),
                    (dst.buffer, nb(dst_off), nb(n_out)),
                    (xd.buffer, 0, xd.size),
                ],
                u(n_out.div_ceil(rows_por_wg)),
                PushSpec::Static(push),
            )
        };

        let np1_wg = ((c.n_embd as u32).div_ceil(256)).clamp(1, NORM_P1_WG);
        // As duas ops da norma, com origem e destino escolhidos pelo chamador: no prólogo
        // a saída cai num offset de `b_eh`, no resto da camada é `b_normed` do modelo.
        let norma = |x: &Buf,
                     r: &Buf,
                     w: &Buf,
                     tem_residual: bool,
                     out: &Buf,
                     out_off: usize|
         -> Result<[PlannedOp; 2], MatmulError> {
            let mut p1 = Vec::with_capacity(8);
            p1.extend_from_slice(&u(c.n_embd).to_le_bytes());
            p1.extend_from_slice(&u32::from(tem_residual).to_le_bytes());
            let mut p2 = Vec::with_capacity(12);
            p2.extend_from_slice(&u(c.n_embd).to_le_bytes());
            p2.extend_from_slice(&c.rms_eps.to_le_bytes());
            p2.extend_from_slice(&np1_wg.to_le_bytes());
            Ok([
                self.mk_op(
                    PipeId::NormFused,
                    &[
                        (x.buffer, 0, nb(c.n_embd)),
                        (r.buffer, 0, nb(c.n_embd)),
                        (st.b_parciais.buffer, 0, st.b_parciais.size),
                    ],
                    np1_wg,
                    PushSpec::Static(p1),
                )?,
                self.mk_op(
                    PipeId::NormP2,
                    &[
                        (x.buffer, 0, nb(c.n_embd)),
                        (w.buffer, 0, w.size),
                        (st.b_parciais.buffer, 0, st.b_parciais.size),
                        (out.buffer, nb(out_off), nb(c.n_embd)),
                        (st.b_xq.buffer, 0, st.b_xq.size),
                        (st.b_xd.buffer, 0, st.b_xd.size),
                    ],
                    ((c.n_embd / 32) as u32).div_ceil(64),
                    PushSpec::Static(p2),
                )?,
            ])
        };
        let quantiza = |src: &Buf, n: usize| -> Result<PlannedOp, MatmulError> {
            let p: [u32; 3] = [u(n), 0, 0];
            // SAFETY: três u32 contíguos; 12 bytes é o push range da pipeline.
            let push = unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), 12) }.to_vec();
            self.mk_op(
                PipeId::QuantizeX,
                &[
                    (src.buffer, 0, nb(n)),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ],
                ((n / 32) as u32).div_ceil(64),
                PushSpec::Static(push),
            )
        };

        // 1. Prólogo: as duas normas escrevendo nas duas metades de `b_eh`.
        plan.push(PlannedOp::Copia {
            src: m.emb_stage.buffer,
            dst: m.b_emb.buffer,
            bytes: nb(c.n_embd),
        });
        plan.push(PlannedOp::CopiaHidden);
        plan.extend(norma(&m.b_emb, &m.b_emb, &m.enorm, false, &m.b_eh, 0)?);
        plan.extend(norma(&m.b_h, &m.b_h, &m.hnorm, false, &m.b_eh, c.n_embd)?);
        // A quantização do `norm_p2` valeu só para metade do vetor de cada vez; o
        // `eh_proj` consome os `2 * n_embd` de uma vez.
        plan.push(quantiza(&m.b_eh, c.n_embd * 2)?);
        plan.push(mv(
            &m.eh_proj,
            (&st.b_xq, &st.b_xd),
            &m.b_x,
            0,
            c.n_embd * 2,
            c.n_embd,
        )?);

        // 2. A camada de decoder do bloco — atenção, não delta-net (ver `MtpBufs`).
        plan.extend(norma(
            &m.b_x,
            &m.b_ffn,
            &m.attn_norm,
            false,
            &st.b_normed,
            0,
        )?);
        plan.push(mv(
            &m.attn_q,
            (&st.b_xq, &st.b_xd),
            &st.b_q,
            0,
            c.n_embd,
            q_out,
        )?);
        plan.push(mv(
            &m.attn_k,
            (&st.b_xq, &st.b_xd),
            &st.b_k,
            0,
            c.n_embd,
            c.kv_dim,
        )?);
        plan.push(mv(
            &m.attn_v,
            (&st.b_xq, &st.b_xd),
            &st.b_v,
            0,
            c.n_embd,
            c.kv_dim,
        )?);
        if let (Some(qn), Some(kn)) = (&m.q_norm, &m.k_norm) {
            let push_qk = |n_heads: usize, stride: usize| {
                let mut v = Vec::with_capacity(20);
                v.extend_from_slice(&u(c.head_dim).to_le_bytes());
                v.extend_from_slice(&u(n_heads).to_le_bytes());
                v.extend_from_slice(&2u32.to_le_bytes()); // modo QK-norm
                v.extend_from_slice(&c.rms_eps.to_le_bytes());
                v.extend_from_slice(&u(stride).to_le_bytes());
                v
            };
            // No Q as cabeças estão espaçadas de 2 × head_dim: o portão mora ao lado.
            plan.push(self.mk_op(
                PipeId::DnNorm,
                &[
                    (st.b_q.buffer, 0, nb(q_out)),
                    (qn.buffer, 0, qn.size),
                    (st.b_q.buffer, 0, nb(q_out)),
                    (st.b_q.buffer, 0, nb(q_out)),
                ],
                u(c.n_head),
                PushSpec::Static(push_qk(c.n_head, c.head_dim * 2)),
            )?);
            plan.push(self.mk_op(
                PipeId::DnNorm,
                &[
                    (st.b_k.buffer, 0, nb(c.kv_dim)),
                    (kn.buffer, 0, kn.size),
                    (st.b_k.buffer, 0, nb(c.kv_dim)),
                    (st.b_k.buffer, 0, nb(c.kv_dim)),
                ],
                u(c.n_head_kv),
                PushSpec::Static(push_qk(c.n_head_kv, c.head_dim)),
            )?);
        }
        // RoPE in-place nos dois, e a cópia para o cache do bloco logo depois. A cabeça
        // não usa o `rope_kv`: são 4 KB por proposta, e assim o plano não depende do knob.
        plan.push(self.mk_op(
            PipeId::Rope,
            &[
                (st.b_q.buffer, 0, nb(q_out)),
                (st.freq_buf.buffer, 0, st.freq_buf.size),
            ],
            Self::groups_for(c.n_head * (c.rope_dim / 2)),
            PushSpec::Rope {
                n_head: u(c.n_head),
                stride: u(if hib { c.head_dim * 2 } else { c.head_dim }),
            },
        )?);
        plan.push(self.mk_op(
            PipeId::Rope,
            &[
                (st.b_k.buffer, 0, nb(c.kv_dim)),
                (st.freq_buf.buffer, 0, st.freq_buf.size),
            ],
            Self::groups_for(c.n_head_kv * (c.rope_dim / 2)),
            PushSpec::Rope {
                n_head: u(c.n_head_kv),
                stride: u(c.head_dim),
            },
        )?);
        plan.push(PlannedOp::KvAppendMtp);
        let attn_push = PushSpec::Attention { kv_layer_off: 0 };
        plan.push(PlannedOp::Atencao {
            curto: Box::new(self.mk_op(
                PipeId::Attention,
                &[
                    (st.b_q.buffer, 0, nb(q_out)),
                    (m.kcache.buffer, 0, m.kcache.size),
                    (m.vcache.buffer, 0, m.vcache.size),
                    (st.b_attn.buffer, 0, nb(attn_dim)),
                ],
                u(c.n_head),
                attn_push,
            )?),
            split: Box::new(self.mk_op(
                PipeId::AttentionSplit,
                &[
                    (st.b_q.buffer, 0, nb(q_out)),
                    (m.kcache.buffer, 0, m.kcache.size),
                    (m.vcache.buffer, 0, m.vcache.size),
                    (st.b_attn_split.buffer, 0, st.b_attn_split.size),
                ],
                u(c.n_head),
                PushSpec::Attention { kv_layer_off: 0 },
            )?),
            reduce: Box::new(self.mk_op(
                PipeId::AttnReduce,
                &[
                    (st.b_attn_split.buffer, 0, st.b_attn_split.size),
                    (st.b_attn.buffer, 0, nb(attn_dim)),
                ],
                u(c.n_head),
                PushSpec::AttnReduce,
            )?),
        });
        if hib {
            let mut pg = Vec::with_capacity(8);
            pg.extend_from_slice(&u(attn_dim).to_le_bytes());
            pg.extend_from_slice(&u(c.head_dim).to_le_bytes());
            plan.push(self.mk_op(
                PipeId::GateQuant,
                &[
                    (st.b_attn.buffer, 0, nb(attn_dim)),
                    (st.b_q.buffer, 0, nb(q_out)),
                    (st.b_xq.buffer, 0, st.b_xq.size),
                    (st.b_xd.buffer, 0, st.b_xd.size),
                ],
                ((attn_dim / 32) as u32).div_ceil(64),
                PushSpec::Static(pg),
            )?);
        } else {
            plan.push(quantiza(&st.b_attn, attn_dim)?);
        }
        plan.push(mv(
            &m.attn_output,
            (&st.b_xq, &st.b_xd),
            &st.b_proj,
            0,
            attn_dim,
            c.n_embd,
        )?);

        // 3. FFN, com o residual do mixer absorvido pela norma (como no plano principal).
        plan.extend(norma(
            &m.b_x,
            &st.b_proj,
            &m.ffn_norm,
            true,
            &st.b_normed,
            0,
        )?);
        plan.push(mv(
            &m.ffn_gate,
            (&st.b_xq, &st.b_xd),
            &st.b_gate,
            0,
            c.n_embd,
            c.n_ff,
        )?);
        plan.push(mv(
            &m.ffn_up,
            (&st.b_xq, &st.b_xd),
            &st.b_up,
            0,
            c.n_embd,
            c.n_ff,
        )?);
        plan.push(self.mk_op(
            PipeId::SwigluQuant,
            &[
                (st.b_gate.buffer, 0, nb(c.n_ff)),
                (st.b_up.buffer, 0, nb(c.n_ff)),
                (st.b_act.buffer, 0, nb(c.n_ff)),
                (st.b_xq.buffer, 0, st.b_xq.size),
                (st.b_xd.buffer, 0, st.b_xd.size),
            ],
            ((c.n_ff / 32) as u32).div_ceil(64),
            PushSpec::Static(u(c.n_ff).to_le_bytes().to_vec()),
        )?);
        plan.push(mv(
            &m.ffn_down,
            (&st.b_xq, &st.b_xd),
            &m.b_ffn,
            0,
            c.n_ff,
            c.n_embd,
        )?);

        // 4. Norma final do bloco (faz o papel do `output_norm`) e projeção de vocabulário.
        //    O GGUF do 27B não traz `nextn.shared_head.head`: a projeção é a `output.weight`
        //    do modelo, compartilhada — é o mesmo fallback do llama.cpp.
        plan.extend(norma(
            &m.b_x,
            &m.b_ffn,
            &m.shared_head_norm,
            true,
            &st.b_normed,
            0,
        )?);
        plan.push(mv(
            &st.output_w,
            (&st.b_xq, &st.b_xd),
            &st.b_logits,
            0,
            c.n_embd,
            c.vocab,
        )?);
        Ok(plan)
    }

    /// Lê os timestamps do token e acumula o tempo de GPU por op. No-op sem profiling.
    fn collect_prof(&self, st: &ResidentState<'_>, modo: Modo) -> Result<(), MatmulError> {
        let Some(pf) = &st.prof else { return Ok(()) };
        let plano = match modo {
            Modo::Decode => &st.plan,
            Modo::Batch => &st.plan_batch,
            Modo::Verify => &st.plan_verify,
        };
        let n = plano.len() + 1;
        let mut ticks = vec![0u64; n];
        // SAFETY: pool tem >= n slots, todos escritos neste command buffer já concluído.
        unsafe {
            self.dev.device.get_query_pool_results(
                pf.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let mut accum = match modo {
            Modo::Decode => pf.accum.borrow_mut(),
            Modo::Batch => pf.accum_batch.borrow_mut(),
            Modo::Verify => pf.accum_verify.borrow_mut(),
        };
        if accum.len() < n - 1 {
            accum.resize(n - 1, 0);
        }
        for i in 0..n - 1 {
            let delta = ticks[i + 1].saturating_sub(ticks[i]);
            #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
            let ns = (delta as f64 * f64::from(pf.period_ns)) as u64;
            accum[i] += ns;
        }
        drop(accum);
        match modo {
            Modo::Decode => pf.tokens.set(pf.tokens.get() + 1),
            Modo::Batch => pf.blocos.set(pf.blocos.get() + 1),
            Modo::Verify => pf.verifies.set(pf.verifies.get() + 1),
        }

        // Zonas absolutas para a timeline. O fence acabou de ser aguardado, então `agora`
        // é o instante do ÚLTIMO timestamp — dá para ancorar o token inteiro sem submit
        // extra nem VK_EXT_calibrated_timestamps, e sem drift acumulado entre tokens.
        if modo == Modo::Decode && pf.tokens.get() <= pf.max_trace_tokens {
            let now = std::time::Instant::now();
            let last = ticks[n - 1];
            let at = |tick: u64| -> std::time::Instant {
                #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
                let back = (last.saturating_sub(tick) as f64 * f64::from(pf.period_ns)) as u64;
                now - std::time::Duration::from_nanos(back)
            };
            let mut spans = pf.spans.borrow_mut();
            for i in 0..n - 1 {
                let Some(name) = plano.get(i).map(PlannedOp::label) else {
                    continue;
                };
                spans.push(GpuSpan {
                    name,
                    start: at(ticks[i]),
                    end: at(ticks[i + 1]),
                });
            }
        }
        Ok(())
    }

    /// Zonas de GPU coletadas, para a timeline. Vazio sem profiling.
    pub fn gpu_spans(&self) -> Vec<GpuSpan> {
        self.state
            .as_ref()
            .and_then(|st| st.prof.as_ref())
            .map(|pf| pf.spans.borrow().clone())
            .unwrap_or_default()
    }

    /// Nome do device deste backend, para rotular a trilha na timeline.
    pub fn device_name(&self) -> String {
        self.ctx.amd_compute_devices()[self.phys_idx]
            .name()
            .to_owned()
    }

    /// Imprime o perfil agregado por tipo de op (stderr). No-op sem profiling.
    pub fn print_profile(&self) {
        let Some(st) = self.state.as_ref() else {
            return;
        };
        self.perfil_de(st, Modo::Decode);
        self.perfil_de(st, Modo::Batch);
        self.perfil_de(st, Modo::Verify);
    }

    /// Uma tabela do perfil, uma por plano. Não imprime nada se o plano nunca rodou.
    #[allow(clippy::cast_precision_loss)]
    fn perfil_de(&self, st: &ResidentState<'_>, modo: Modo) {
        let Some(pf) = &st.prof else { return };
        let (plano, accum, passos) = match modo {
            Modo::Decode => (&st.plan, pf.accum.borrow(), pf.tokens.get()),
            Modo::Batch => (&st.plan_batch, pf.accum_batch.borrow(), pf.blocos.get()),
            Modo::Verify => (&st.plan_verify, pf.accum_verify.borrow(), pf.verifies.get()),
        };
        if passos == 0 || accum.is_empty() {
            return;
        }
        let tokens = passos;

        // Por rótulo de op: (ns acumulados, dispatches, bytes lidos por passo).
        let mut por_tipo: std::collections::BTreeMap<&'static str, (u64, usize, u64)> =
            std::collections::BTreeMap::new();
        for (i, &ns) in accum.iter().enumerate() {
            let Some(op) = plano.get(i) else { continue };
            // As cópias e a atenção não trazem contagem de bytes — ver `bytes` em
            // `build_plan`.
            let bytes = match op {
                PlannedOp::Dispatch { bytes, .. } => *bytes,
                _ => 0,
            };
            let label = op.label();
            let e = por_tipo.entry(label).or_insert((0, 0, 0));
            e.0 += ns;
            e.1 += 1;
            e.2 += bytes;
        }

        let total: u64 = accum.iter().sum();
        let total_bytes: u64 = por_tipo.values().map(|v| v.2).sum();
        let ms = |ns: u64| ns as f64 / 1e6 / tokens as f64;
        // 1 byte/ns == 1 GB/s: os bytes são por passo e `ns` é a soma de `tokens` passos.
        let gbs = |bytes: u64, ns: u64| bytes as f64 * tokens as f64 / ns.max(1) as f64;
        let (unidade, por, nome) = match modo {
            Modo::Decode => ("tokens", "ms/token", "decode"),
            Modo::Batch => ("blocos", "ms/bloco", "prefill em batch"),
            Modo::Verify => ("verifies", "ms/verify", "verify do MTP (3 tokens)"),
        };
        let sh = st.cfg.shard;
        eprintln!(
            "\n=== PERFIL GPU{} {} — {nome} ({tokens} {unidade}, {} ops, camadas {}..{}) ===",
            sh.device,
            self.device_name(),
            accum.len(),
            sh.first_layer,
            sh.end_layer
        );
        eprintln!(
            "{:<16} {:>10} {:>8} {:>8} {:>9}",
            "op", por, "%", "n", "GB/s"
        );
        let mut linhas: Vec<_> = por_tipo.iter().collect();
        linhas.sort_by_key(|x| std::cmp::Reverse(x.1.0));
        let mut algum_sem_bytes = false;
        for (label, (ns, n, bytes)) in linhas {
            let banda = if *bytes == 0 {
                algum_sem_bytes = true;
                "        —".to_owned()
            } else {
                format!("{:>9.0}", gbs(*bytes, *ns))
            };
            eprintln!(
                "{label:<16} {:>10.3} {:>7.1}% {:>8} {banda}",
                ms(*ns),
                100.0 * *ns as f64 / total.max(1) as f64,
                n
            );
        }
        eprintln!(
            "{:<16} {:>10.3} {:>7.1}% {:>8} {:>9.0}",
            "TOTAL GPU",
            ms(total),
            100.0,
            "",
            gbs(total_bytes, total)
        );
        if algum_sem_bytes {
            eprintln!(
                "(— = bytes por dispatch não anotados; a atenção lê o KV pelo comprimento, \
                 conhecido só na gravação — o agregado exclui esses bytes e inclui o tempo)"
            );
        }
        if modo != Modo::Decode {
            return;
        }
        let h = pf.host.borrow();
        eprintln!("\n--- host (ms/token) ---");
        eprintln!("{:<12} {:>10.3}", "gravacao", ms(h[0]));
        eprintln!("{:<12} {:>10.3}", "submit+fence", ms(h[1]));
        eprintln!("{:<12} {:>10.3}", "leitura", ms(h[2]));
    }

    /// Executa este shard para um token. `x_in` é a stream residual vinda do shard
    /// anterior (`None` no primeiro, que faz o embedding lookup). Retorna os logits no
    /// último shard, ou a stream residual a repassar nos demais.
    #[cfg_attr(feature = "profiling", tracing::instrument(skip_all, name = "shard"))]
    pub fn decode_shard(
        &self,
        token: u32,
        pos: usize,
        x_in: Option<&[f32]>,
    ) -> Result<Vec<f32>, MatmulError> {
        self.decode_shard_batch(&[token], pos, x_in)
    }

    /// Verifica `VERIFY_TOK` tokens em posições consecutivas a partir de `pos0` e devolve
    /// os logits de **todos** (`VERIFY_TOK × vocab`, o primeiro token primeiro) no último
    /// shard, ou a stream residual do bloco nos demais.
    ///
    /// É um passo de speculative decoding: `tokens[0]` é o token já amostrado e
    /// `tokens[1]` a proposta da cabeça MTP. Ler os pesos uma vez serve aos dois, que é o
    /// ganho todo. Em caso de rejeição, [`Self::rollback_verify`] desfaz o segundo.
    pub fn verify_shard(
        &self,
        tokens: &[u32; VERIFY_TOK],
        pos0: usize,
        x_in: Option<&[f32]>,
    ) -> Result<Vec<f32>, MatmulError> {
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        if st.plan_verify.is_empty() || pos0 + VERIFY_TOK > st.cfg.ctx {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        let pos = pos0 + VERIFY_TOK - 1;
        let out = self.record_and_submit(tokens, pos, x_in, Modo::Verify)?;
        *st.len.borrow_mut() = pos + 1;
        Ok(out)
    }

    /// Desfaz os tokens rejeitados do último verify, mantendo os `manter` primeiros:
    /// restaura os snapshots do estado recorrente e da janela da convolução tirados
    /// depois do token `manter - 1`, e recua o comprimento do KV em
    /// `VERIFY_TOK - manter`.
    ///
    /// O KV-cache não precisa de snapshot. As posições do verify são consecutivas e o
    /// que o próximo passo escreve sobrescreve as rejeitadas — recuar é só o contador,
    /// que é escrituração: quem de fato decide a posição é o `pos` que o chamador passa
    /// adiante. Com o `rope_kv` ligado o K já entrou no slot girado, e isso também não
    /// muda nada.
    pub fn rollback_verify(&self, manter: usize) -> Result<(), MatmulError> {
        if manter == 0 || manter >= VERIFY_TOK {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let mut len = st.len.borrow_mut();
        *len = len.saturating_sub(VERIFY_TOK - manter);
        drop(len);
        let Some(&cmd) = st.rollback_cmds.get(manter - 1) else {
            // Modelo sem camada de atenção linear: não há estado recorrente a restaurar.
            return Ok(());
        };
        let d = &self.dev.device;
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        // SAFETY: o command buffer foi gravado uma vez na construção e não usa
        // ONE_TIME_SUBMIT; o fence do token já foi aguardado antes desta chamada.
        unsafe {
            d.reset_fences(&[st.token_fence])?;
            d.queue_submit(self.dev.queue, &[submit], st.token_fence)?;
        }
        self.espera_fence(st)
    }

    /// Como `decode_shard`, para um bloco de tokens em posições consecutivas a partir de
    /// `pos0`. `tokens.len()` tem de ser `cfg.n_batch` — é a largura para a qual o plano de
    /// prefill foi montado (`COLS` é specialization constant).
    pub fn decode_shard_batch(
        &self,
        tokens: &[u32],
        pos0: usize,
        x_in: Option<&[f32]>,
    ) -> Result<Vec<f32>, MatmulError> {
        let n_tok = tokens.len();
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        if n_tok == 0 || (n_tok > 1 && n_tok != st.cfg.n_batch) || pos0 + n_tok > st.cfg.ctx {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        let pos = pos0 + n_tok - 1;
        let modo = if n_tok > 1 { Modo::Batch } else { Modo::Decode };
        let out = self.record_and_submit(tokens, pos, x_in, modo)?;
        *st.len.borrow_mut() = pos + 1;
        Ok(out)
    }

    /// Rótulos das ops do plano, na ordem de execução — para conferir a montagem.
    pub fn dbg_plano(&self) -> Vec<&'static str> {
        self.state
            .as_ref()
            .map(|st| st.plan.iter().map(PlannedOp::label).collect())
            .unwrap_or_default()
    }

    /// O mesmo para o plano de verify do MTP — vazio quando o backend não tem MTP.
    pub fn dbg_plano_verify(&self) -> Vec<&'static str> {
        self.state
            .as_ref()
            .map(|st| st.plan_verify.iter().map(PlannedOp::label).collect())
            .unwrap_or_default()
    }

    /// Lê a stream residual (`b_x`) — o estado oculto depois das camadas executadas.
    ///
    /// Com `LLAMA_RS_STOP_LAYER=N` isso dá o hidden state após N camadas, que é o que se
    /// compara com o `l_out-N` do `llama-eval-callback` do llama.cpp.
    pub fn dbg_hidden(&self) -> Option<Vec<f32>> {
        let st = self.state.as_ref()?;
        self.readback(&st.b_x, st.cfg.n_embd).ok()
    }

    /// Lê um dos buffers intermediários do caminho de atenção linear, por nome.
    ///
    /// Existe para bissecar contra o dump do `llama-eval-callback`: `qkv`, `conv`, `qn`,
    /// `kn`, `gb`, `out` e `normed` correspondem, na ordem, aos tensores
    /// `linear_attn_qkv_mixed`, `conv_output_silu`, `q_conv_predelta`, `k_conv_predelta`,
    /// `gate`/`beta_sigmoid`, `attn_output` e `final_output` da referência.
    pub fn dbg_dn_buf(&self, nome: &str) -> Option<Vec<f32>> {
        let st = self.state.as_ref()?;
        let b = st.dn.as_ref()?;
        let buf = match nome {
            "qkv" => &b.qkv,
            "conv" => &b.conv,
            "z" => &b.z,
            "gb" => &b.gb,
            "qn" => &b.qn,
            "kn" => &b.kn,
            "out" => &b.out,
            "normed" => &b.normed,
            // Saída da camada e stream residual, para fechar a bissecção.
            "proj" => &st.b_proj,
            "ffn" => &st.b_ffn,
            "x" => &st.b_x,
            "xd" => &st.b_xd,
            _ => return None,
        };
        self.readback(buf, (buf.size / 4) as usize).ok()
    }

    /// Lê o estado recorrente da camada local `l`, se ela for de atenção linear.
    ///
    /// Diagnóstico: o sintoma de um estado que não persiste entre tokens é o modelo
    /// repetir o último token do prompt — sem memória, a camada linear vira função só do
    /// token atual.
    pub fn dbg_estado_delta(&self, l: usize) -> Option<Vec<f32>> {
        let st = self.state.as_ref()?;
        let dn = st.aux.get(l)?.delta.as_ref()?;
        let n = (dn.estado.size / 4) as usize;
        self.readback(&dn.estado, n).ok()
    }

    /// Faixa de camadas e papel deste backend.
    pub fn shard(&self) -> Shard {
        self.state
            .as_ref()
            .map_or(Shard::whole(0, 0), |st| st.cfg.shard)
    }

    /// Guarda estado recorrente, janela da convolução e comprimento do KV como estão agora.
    ///
    /// É o que torna barata a divergência de prompt entre dois turnos. O KV-cache de
    /// atenção volta atrás sozinho — basta recuar o comprimento e reescrever os slots —,
    /// mas o estado recorrente das camadas delta-net é o produto de todos os tokens em
    /// ordem, e sem uma cópia dele a divergência custa reprocessar o prompt inteiro.
    ///
    /// Um snapshot só: o novo sobrescreve o anterior. A fronteira que interessa é o fim do
    /// prefill de cada requisição, porque o que diverge no turno seguinte é o **re-render**
    /// da resposta que veio depois dela (bloco de raciocínio removido, chamada de
    /// ferramenta reformatada); guardar o fim da resposta deixaria a divergência antes do
    /// snapshot, que é justamente o caso que ele não cobre.
    pub fn marcar_snapshot(&self) -> bool {
        let Some(st) = self.state.as_ref() else {
            return false;
        };
        if !self.alocar_snapshot(st) {
            return false;
        }
        let snap = st.snap.borrow();
        let pares: Vec<(&Buf, &Buf)> = st
            .aux
            .iter()
            .filter_map(|la| la.delta.as_ref())
            .zip(snap.iter())
            .flat_map(|(dn, (e, j))| [(&dn.estado, e), (&dn.janela, j)])
            .collect();
        if !self.copiar_bufs(&pares) {
            return false;
        }
        st.snap_len.set(Some(*st.len.borrow()));
        true
    }

    /// Aloca os buffers do snapshot na primeira chamada. `false` se a VRAM não deu — e aí
    /// a sessão segue sem snapshot, reprocessando na divergência como antes.
    fn alocar_snapshot(&self, st: &ResidentState) -> bool {
        let mut snap = st.snap.borrow_mut();
        if !snap.is_empty() {
            return true;
        }
        let d = &self.dev.device;
        for la in &st.aux {
            let Some(dn) = la.delta.as_ref() else {
                continue;
            };
            let (Ok(e), Ok(j)) = (
                Buf::device(self.ctx, self.phys(), d, dn.estado.size),
                Buf::device(self.ctx, self.phys(), d, dn.janela.size),
            ) else {
                for (e, j) in snap.drain(..) {
                    e.destroy(d);
                    j.destroy(d);
                }
                return false;
            };
            snap.push((e, j));
        }
        true
    }

    /// Volta ao último [`marcar_snapshot`]. `false` quando não há snapshot válido — e aí o
    /// chamador não tem alternativa senão reprocessar do zero.
    pub fn restaurar_snapshot(&self) -> bool {
        let Some(st) = self.state.as_ref() else {
            return false;
        };
        let Some(len) = st.snap_len.get() else {
            return false;
        };
        let snap = st.snap.borrow();
        let pares: Vec<(&Buf, &Buf)> = st
            .aux
            .iter()
            .filter_map(|la| la.delta.as_ref())
            .zip(snap.iter())
            .flat_map(|(dn, (e, j))| [(e, &dn.estado), (j, &dn.janela)])
            .collect();
        if !self.copiar_bufs(&pares) {
            return false;
        }
        // O KV-cache não precisa ser limpo: as posições a partir daqui serão reescritas
        // pelos tokens novos, e a atenção só olha `total_len` posições.
        *st.len.borrow_mut() = len;
        true
    }

    /// Copia `(origem, destino)` em um command buffer só e espera. Acontece entre
    /// requisições, com a GPU ociosa — é o mesmo caminho de `reset_len`.
    fn copiar_bufs(&self, pares: &[(&Buf, &Buf)]) -> bool {
        if pares.is_empty() {
            return true;
        }
        let d = &self.dev.device;
        // SAFETY: GPU ociosa entre sequências; buffers criados por nós, com tamanhos iguais.
        unsafe {
            let _ = d.device_wait_idle();
            let cb_info = vk::CommandBufferAllocateInfo {
                command_pool: self.dev.cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let Ok(cbs) = d.allocate_command_buffers(&cb_info) else {
                return false;
            };
            let cmd = cbs[0];
            let begin = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            let mut ok = false;
            if d.begin_command_buffer(cmd, &begin).is_ok() {
                for (src, dst) in pares {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: src.size.min(dst.size),
                    };
                    d.cmd_copy_buffer(cmd, src.buffer, dst.buffer, &[region]);
                }
                ok = d.end_command_buffer(cmd).is_ok();
                let submit = vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                };
                ok &= d
                    .queue_submit(self.dev.queue, &[submit], vk::Fence::null())
                    .is_ok();
                ok &= d.queue_wait_idle(self.dev.queue).is_ok();
            }
            d.free_command_buffers(self.dev.cmd_pool, &cbs);
            ok
        }
    }

    /// Zera o comprimento do KV-cache (início de nova sequência).
    pub fn reset_len(&self) {
        if let Some(st) = self.state.as_ref() {
            *st.len.borrow_mut() = 0;
            // O snapshot pertence à sequência que acabou de ser descartada.
            st.snap_len.set(None);
            // O KV-cache da cabeça MTP é dela: o conteúdo não precisa ser zerado (o
            // `total_len` da atenção limita o que se lê), só o contador.
            if let Some(m) = st.mtp.as_ref() {
                *m.len.borrow_mut() = 0;
            }
            // Nas camadas de atenção linear não há comprimento de KV-cache para zerar: o
            // histórico está no estado recorrente e na janela da convolução, que precisam
            // voltar a zero — é o que representa "contexto vazio" nesta arquitetura.
            let d = &self.dev.device;
            let zerar: Vec<vk::Buffer> = st
                .aux
                .iter()
                .filter_map(|la| la.delta.as_ref())
                .flat_map(|dn| [dn.estado.buffer, dn.janela.buffer])
                .collect();
            if zerar.is_empty() {
                return;
            }
            let tamanhos: Vec<vk::DeviceSize> = st
                .aux
                .iter()
                .filter_map(|la| la.delta.as_ref())
                .flat_map(|dn| [dn.estado.size, dn.janela.size])
                .collect();
            // SAFETY: GPU ociosa entre sequências; buffers criados por nós.
            unsafe {
                let _ = d.device_wait_idle();
                let cb_info = vk::CommandBufferAllocateInfo {
                    command_pool: self.dev.cmd_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                };
                let Ok(cbs) = d.allocate_command_buffers(&cb_info) else {
                    return;
                };
                let cmd = cbs[0];
                let begin = vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                };
                if d.begin_command_buffer(cmd, &begin).is_ok() {
                    for (buf, size) in zerar.iter().zip(&tamanhos) {
                        d.cmd_fill_buffer(cmd, *buf, 0, *size, 0);
                    }
                    let _ = d.end_command_buffer(cmd);
                    let submit = vk::SubmitInfo {
                        command_buffer_count: 1,
                        p_command_buffers: &cmd,
                        ..Default::default()
                    };
                    let _ = d.queue_submit(self.dev.queue, &[submit], vk::Fence::null());
                    let _ = d.queue_wait_idle(self.dev.queue);
                }
                d.free_command_buffers(self.dev.cmd_pool, &cbs);
            }
        }
    }

    /// Espera o fence do token **sondando**, sem dormir.
    ///
    /// Um `wait_for_fences` bloqueante entrega a thread ao escalonador do SO, que reduz o
    /// clock da CPU por baixa utilização e a deixa entrar em C-state profundo — o wakeup
    /// pela IRQ da GPU passa a custar milissegundos em vez de microssegundos. O custo é um
    /// núcleo ocupado enquanto a GPU trabalha; é o mesmo compromisso que o llama.cpp faz no
    /// seu laço de espera. Diagnóstico e números medidos em `docs/performance-tuning.md`.
    fn espera_fence(&self, st: &ResidentState<'_>) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        loop {
            // SAFETY: fence válido e submetido.
            if unsafe { d.get_fence_status(st.token_fence)? } {
                return Ok(());
            }
            std::hint::spin_loop();
        }
    }

    /// Regrava o command buffer do token, submete uma vez, espera o fence, lê os logits.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "submit+fence")
    )]
    fn record_and_submit(
        &self,
        tokens: &[u32],
        pos: usize,
        x_in: Option<&[f32]>,
        modo: Modo,
    ) -> Result<Vec<f32>, MatmulError> {
        let st = self
            .state
            .as_ref()
            .ok_or(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let d = &self.dev.device;
        let dev = &self.dev;
        let cmd = st.token_cmd;

        // SAFETY: pool RESET_COMMAND_BUFFER; cmd não está em uso (fence do token anterior aguardado).
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            d.begin_command_buffer(cmd, &begin)?;
        }
        let t0 = std::time::Instant::now();
        self.record_token(cmd, tokens, pos, x_in, modo);
        // SAFETY: cmd em gravação.
        unsafe {
            d.end_command_buffer(cmd)?;
        }
        let t_rec = t0.elapsed();
        let t1 = std::time::Instant::now();

        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        // SAFETY: fence resetado antes do submit; cmd válido.
        unsafe {
            d.reset_fences(&[st.token_fence])?;
            d.queue_submit(dev.queue, &[submit], st.token_fence)?;
        }
        self.espera_fence(st)?;
        let t_sub = t1.elapsed();
        let t2 = std::time::Instant::now();
        self.collect_prof(st, modo)?;

        let len = if st.cfg.shard.is_last() {
            if modo == Modo::Verify {
                st.cfg.vocab * VERIFY_TOK
            } else {
                st.cfg.vocab
            }
        } else {
            st.cfg.n_embd * tokens.len()
        };
        // O mapa é persistente (feito uma vez na construção): `vkMapMemory`/`vkUnmapMemory`
        // a cada token são dois ioctls no caminho crítico, e o vocabulário do 32B faz esse
        // caminho rodar com 608 KB.
        //
        // `Vec::with_capacity` + `set_len` em vez de `vec![0.0; len]`: o segundo zera os
        // 608 KB (calloc) só para a cópia logo abaixo sobrescrever tudo.
        let out = unsafe {
            let mut v = Vec::<f32>::with_capacity(len);
            // SAFETY: logits_host é host-coherent e a cópia já terminou (fence aguardado
            // acima); `v` tem capacidade para `len` f32, escritos antes do set_len.
            std::ptr::copy_nonoverlapping(st.logits_ptr.cast::<f32>(), v.as_mut_ptr(), len);
            v.set_len(len);
            v
        };
        if let Some(pf) = &st.prof {
            let mut h = pf.host.borrow_mut();
            h[0] += t_rec.as_nanos() as u64;
            h[1] += t_sub.as_nanos() as u64;
            h[2] += t2.elapsed().as_nanos() as u64;
        }
        Ok(out)
    }
}

impl llama_model::GpuResidentDecode for ResidentForward<'_> {
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, llama_model::ModelError> {
        let logits = self
            .record_and_submit(&[token], pos, None, Modo::Decode)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
        if let Some(st) = self.state.as_ref() {
            *st.len.borrow_mut() = pos + 1;
        }
        Ok(logits)
    }
    fn batch_size(&self) -> usize {
        self.state.as_ref().map_or(1, |st| st.cfg.n_batch)
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        pos0: usize,
    ) -> Result<Vec<f32>, llama_model::ModelError> {
        self.decode_shard_batch(tokens, pos0, None)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))
    }
    fn tem_mtp(&self) -> bool {
        ResidentForward::tem_mtp(self)
    }
    fn propor_mtp(&self, token: u32, hidden_idx: usize) -> Result<u32, llama_model::ModelError> {
        // Shard único: a tabela de embedding e a cabeça moram na mesma GPU.
        let emb = self
            .linha_embd(token)
            .ok_or_else(|| llama_model::ModelError::Gpu(format!("token {token} fora da tabela")))?;
        ResidentForward::propor_mtp(self, &emb, hidden_idx)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))
    }
    fn decode_verify(
        &self,
        tokens: &[u32; VERIFY_TOK],
        pos0: usize,
    ) -> Result<Vec<f32>, llama_model::ModelError> {
        self.verify_shard(tokens, pos0, None)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))
    }
    fn rollback_verify(&self, manter: usize) -> Result<(), llama_model::ModelError> {
        ResidentForward::rollback_verify(self, manter)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))
    }
    /// Zera o comprimento do KV-cache **e** o estado recorrente das camadas lineares.
    ///
    /// Recuar só o comprimento bastava enquanto todas as camadas eram de atenção: os slots
    /// do cache são reescritos pelos tokens novos. No qwen35 o histórico das 48 camadas
    /// delta-net está num estado que ninguém sobrescreve, e uma sequência nova começando
    /// sobre o estado da anterior gera texto influenciado por ela.
    fn reset(&self) {
        self.reset_len();
    }
    fn marcar(&self) -> bool {
        self.marcar_snapshot()
    }
    fn restaurar(&self) -> bool {
        self.restaurar_snapshot()
    }
}

impl Drop for ResidentForward<'_> {
    fn drop(&mut self) {
        let d = &self.dev.device;
        // SAFETY: GPU ociosa antes de liberar.
        unsafe {
            let _ = d.device_wait_idle();
        }

        if let Some(st) = self.state.take() {
            // SAFETY: mapeada uma vez na construção e nunca desmapeada até aqui.
            unsafe { d.unmap_memory(st.logits_host.mem) };
            if let Some(pf) = &st.prof {
                // SAFETY: pool criado por nós; GPU ociosa (device_wait_idle acima).
                unsafe { d.destroy_query_pool(pf.pool, None) };
            }
            for lq in st.qw {
                match lq.mixer {
                    MixerQ::Attn {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                    } => {
                        attn_q.gpu.destroy(d);
                        attn_k.gpu.destroy(d);
                        attn_v.gpu.destroy(d);
                        attn_output.gpu.destroy(d);
                    }
                    MixerQ::Delta {
                        attn_qkv,
                        attn_gate,
                        ssm_out,
                    } => {
                        attn_qkv.gpu.destroy(d);
                        attn_gate.gpu.destroy(d);
                        ssm_out.gpu.destroy(d);
                    }
                }
                lq.ffn_gate.gpu.destroy(d);
                lq.ffn_up.gpu.destroy(d);
                lq.ffn_down.gpu.destroy(d);
            }
            st.output_w.gpu.destroy(d);
            for (estado, janela) in st.snap.borrow().iter() {
                estado.destroy(d);
                janela.destroy(d);
            }
            for la in st.aux {
                la.attn_norm.destroy(d);
                la.ffn_norm.destroy(d);
                for b in [&la.q_norm, &la.k_norm].into_iter().flatten() {
                    b.destroy(d);
                }
                if let Some(dn) = &la.delta {
                    for b in [
                        &dn.conv1d, &dn.adt, &dn.alpha, &dn.beta, &dn.norm, &dn.estado, &dn.janela,
                    ] {
                        b.destroy(d);
                    }
                    for b in dn.estado_snap.iter().chain(dn.janela_snap.iter()) {
                        b.destroy(d);
                    }
                }
                if let Some(b) = la.q_bias {
                    b.destroy(d);
                }
                if let Some(b) = la.k_bias {
                    b.destroy(d);
                }
                if let Some(b) = la.v_bias {
                    b.destroy(d);
                }
            }
            for b in [
                &st.output_norm_buf,
                &st.freq_buf,
                &st.embd_stage,
                &st.kcache,
                &st.vcache,
                &st.b_x,
                &st.b_q,
                &st.b_k,
                &st.b_v,
                &st.b_proj,
                &st.b_gate,
                &st.b_up,
                &st.b_logits,
                &st.b_ffn,
                &st.b_xq,
                &st.b_xd,
                &st.logits_host,
            ] {
                b.destroy(d);
            }
            if let Some(m) = st.mtp {
                for w in [
                    m.eh_proj,
                    m.attn_q,
                    m.attn_k,
                    m.attn_v,
                    m.attn_output,
                    m.ffn_gate,
                    m.ffn_up,
                    m.ffn_down,
                ] {
                    w.gpu.destroy(d);
                }
                for b in [
                    &m.enorm,
                    &m.hnorm,
                    &m.shared_head_norm,
                    &m.attn_norm,
                    &m.ffn_norm,
                    &m.emb_stage,
                    &m.b_emb,
                    &m.b_h,
                    &m.b_eh,
                    &m.b_x,
                    &m.b_ffn,
                    &m.kcache,
                    &m.vcache,
                ] {
                    b.destroy(d);
                }
                for b in m.q_norm.iter().chain(m.k_norm.iter()) {
                    b.destroy(d);
                }
            }
            // Os chunks dos pesos por último: os buffers acima apontavam para dentro deles.
            let mut pesos_mem = st.pesos_mem;
            pesos_mem.cleanup(d);
            // SAFETY: token_cmd/token_fence criados por nós; GPU ociosa.
            unsafe {
                d.free_command_buffers(self.dev.cmd_pool, &[st.token_cmd]);
                if !st.rollback_cmds.is_empty() {
                    d.free_command_buffers(self.dev.cmd_pool, &st.rollback_cmds);
                }
                d.destroy_fence(st.token_fence, None);
            }
        }

        // SAFETY: pool/pipelines criados por nós.
        unsafe {
            d.destroy_descriptor_pool(self.desc_pool, None);
        }
        for p in [
            &self.matvec,
            &self.rmsnorm,
            &self.norm_fused,
            &self.norm_p2,
            &self.rope,
            &self.attention,
            &self.swiglu_quant,
            &self.add,
        ] {
            // SAFETY: handles criados por nós, ordem inversa.
            unsafe {
                d.destroy_pipeline(p.pipeline, None);
                d.destroy_pipeline_layout(p.layout, None);
                d.destroy_descriptor_set_layout(p.desc_set_layout, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::slots_kv;

    /// No qwen35 três de cada quatro camadas são delta-net: elas guardam estado
    /// recorrente de tamanho fixo e **não** usam KV-cache. Numerar o cache por camada
    /// global reservaria 4× a memória necessária — 17 GB em vez de 4,4 GB num ctx de 32k.
    #[test]
    fn slots_kv_so_numera_as_camadas_de_atencao() {
        let padrao = [false, false, false, true, false, false, false, true];

        let (slots, total) = slots_kv(padrao);

        assert_eq!(total, 2, "8 camadas do qwen35 têm 2 de atenção");
        assert_eq!(slots.get(3), Some(&Some(0)));
        assert_eq!(slots.get(7), Some(&Some(1)));
        assert_eq!(slots.first(), Some(&None), "delta-net não ocupa slot");
    }

    /// Modelo denso (Qwen2/2.5): slot == camada, nada muda em relação ao layout antigo.
    #[test]
    fn slots_kv_no_modelo_denso_e_a_identidade() {
        let (slots, total) = slots_kv([true, true, true]);

        assert_eq!(total, 3);
        assert_eq!(slots, vec![Some(0), Some(1), Some(2)]);
    }

    /// A largura que o tile do GEMM cobre. O mesmo predicado decide o `COLS` com que a
    /// pipeline é compilada e se o plano emite `MulMmQ4K`; se os dois discordassem, o
    /// dispatch rodaria com um tile de outra largura. O decode (`cols = 1`) e a projeção de
    /// logits têm de ficar de fora sempre.
    #[test]
    fn largura_do_tile_do_gemm() {
        use super::gemm_largura_ok;
        for cols in [8usize, 16, 24, 32, 64] {
            assert!(
                gemm_largura_ok(cols),
                "{cols} é múltiplo de 8 e cabe no tile"
            );
        }
        for cols in [0usize, 1, 4, 12, 72] {
            assert!(!gemm_largura_ok(cols), "{cols} não cabe no tile");
        }
    }
}
