//! O passo de speculative decoding do MTP: verificação de dois tokens, rollback do estado
//! recorrente e a cabeça `nextn` rodando na GPU.
//!
//! **O invariante que vale por todos:** com greedy, speculative decoding é *lossless*. A
//! sequência gerada com MTP tem de ser idêntica, token a token, à gerada sem — porque a
//! proposta só é aceita quando coincide com o que a verificação diz, e quando não coincide
//! o token certo já veio nos logits do primeiro token do bloco.
//!
//! Quase tudo aqui precisa do Qwen3.8-27B: o bloco `nextn` só existe no GGUF real, e os
//! caminhos que se quer provar (recorrência do delta-net, KV fatiado, portão do qwen35) não
//! têm versão sintética. Sem o arquivo os testes dão skip limpo. O único que roda sempre é
//! o do snapshot, que usa os shaders da recorrência direto, com dados sintéticos.

use std::path::Path;

use llama_model::GpuResidentDecode;
use llama_vulkan::{DnPipe, ResidentForward, VulkanContext};

const MODELO: &str = "../../models/Qwen3.8-27B-Q4_K_M.gguf";

#[allow(unsafe_code)]
fn mapear() -> Option<memmap2::Mmap> {
    let file = std::fs::File::open(Path::new(MODELO)).ok()?;
    unsafe { memmap2::Mmap::map(&file) }.ok()
}

fn argmax(v: &[f32]) -> u32 {
    let mut melhor = 0usize;
    let mut valor = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > valor {
            valor = x;
            melhor = i;
        }
    }
    u32::try_from(melhor).unwrap_or(0)
}

fn pseudo(n: usize, semente: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i.wrapping_mul(2_654_435_761).wrapping_add(semente) % 1000) as f32;
            x / 500.0 - 1.0
        })
        .collect()
}

/// Bytes de um push constant qualquer.
fn push_bytes<T: Copy>(p: &T) -> Vec<u8> {
    let n = std::mem::size_of::<T>();
    // SAFETY: T é POD (struct de u32/f32 repr(C)); lemos exatamente o seu tamanho.
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(p).cast::<u8>(), n) }.to_vec()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PushDelta {
    d: u32,
    n_heads: u32,
    rep: u32,
    n_tok: u32,
    v_stride: u32,
}

/// Um passo da recorrência: devolve (estado depois, saída).
// Os argumentos são os bindings do shader; agrupá-los daria uma struct de uso único.
#[allow(clippy::too_many_arguments)]
fn passo_delta(
    fwd: &ResidentForward<'_>,
    estado: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gb: &[f32],
    push: &[u8],
    groups: u32,
) -> (Vec<f32>, Vec<f32>) {
    let out = vec![0f32; v.len()];
    let bufs = fwd
        .dbg_dn(
            DnPipe::DeltaNet,
            &[
                estado.to_vec(),
                q.to_vec(),
                k.to_vec(),
                v.to_vec(),
                gb.to_vec(),
                out,
            ],
            push,
            groups,
        )
        .expect("dispatch delta_net");
    (bufs[0].clone(), bufs[5].clone())
}

/// O snapshot que o `plan_verify` tira entre os dois tokens é **suficiente** para desfazer
/// o segundo: o estado recorrente é tudo o que a recorrência carrega de um token para o
/// seguinte.
///
/// Roda a sequência (A, B) guardando o estado depois de A, restaura esse estado e roda C.
/// O resultado tem de ser bit a bit o de rodar (A, C) direto. Se algum dia o
/// `delta_net.comp` passar a carregar estado em outro binding, o plano continuará copiando
/// só `estado`/`janela` e o rollback perderá essa parte em silêncio — é isso que este teste
/// prende.
#[test]
fn snapshot_do_estado_desfaz_exatamente_um_token() {
    let Ok(ctx) = VulkanContext::new() else {
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem device AMD — pulando");
        return;
    }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let d = 128usize;
    let n_heads = 6usize;
    let estado0 = pseudo(n_heads * d * d, 1);
    let gb: Vec<f32> = (0..n_heads)
        .flat_map(|h| [-0.05 * (h as f32 + 1.0), 0.3 + 0.1 * h as f32])
        .collect();
    let push = push_bytes(&PushDelta {
        d: d as u32,
        n_heads: n_heads as u32,
        rep: 1,
        n_tok: 1,
        v_stride: (n_heads * d) as u32,
    });
    let groups = (n_heads * d / 4) as u32;
    // Três tokens diferentes: um token repetido esconderia um estado que não avançou.
    let tok = |s: usize| {
        (
            pseudo(n_heads * d, s),
            pseudo(n_heads * d, s + 100),
            pseudo(n_heads * d, s + 200),
        )
    };
    let (qa, ka, va) = tok(2);
    let (qb, kb, vb) = tok(3);
    let (qc, kc, vc) = tok(4);

    // Referência: A e depois C, sem nunca ver B.
    let (estado_a, _) = passo_delta(&fwd, &estado0, &qa, &ka, &va, &gb, &push, groups);
    let (estado_ac, out_ac) = passo_delta(&fwd, &estado_a, &qc, &kc, &vc, &gb, &push, groups);

    // Caminho do verify: A, snapshot, B (rejeitado), restaura o snapshot, C.
    let (estado_a2, _) = passo_delta(&fwd, &estado0, &qa, &ka, &va, &gb, &push, groups);
    let snapshot = estado_a2.clone();
    let (estado_b, _) = passo_delta(&fwd, &estado_a2, &qb, &kb, &vb, &gb, &push, groups);
    assert_ne!(estado_b, snapshot, "o token B tinha de mexer no estado");
    let (estado_rc, out_rc) = passo_delta(&fwd, &snapshot, &qc, &kc, &vc, &gb, &push, groups);

    assert_eq!(estado_rc, estado_ac, "estado após o rollback divergiu");
    assert_eq!(out_rc, out_ac, "saída após o rollback divergiu");
    eprintln!(
        "snapshot do estado: (A,B,rollback,C) == (A,C), {} floats",
        estado_ac.len()
    );
}

/// Monta o backend com MTP ligado, ou sai do teste com skip limpo.
///
/// Precisa das duas GPUs: o 27B não cabe nos 16 GiB de uma MI50 só. `bytes`, `raw` e `aux`
/// ficam vivos no escopo do teste porque o backend empresta deles na construção.
macro_rules! backend_mtp {
    ($bytes:ident, $ctx:ident, $cfg:ident, $raw:ident, $aux:ident, $b:ident, $ctx_len:expr) => {
        let Some($bytes) = mapear() else {
            eprintln!("Qwen3.8-27B Q4_K_M ausente — pulando");
            return;
        };
        let Ok($ctx) = VulkanContext::new() else {
            return;
        };
        if $ctx.amd_compute_devices().len() < 2 {
            eprintln!("menos de 2 GPUs — pulando");
            return;
        }
        let f = gguf::GgufFile::parse(&$bytes).unwrap();
        let mut $cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
        $cfg.ctx = $ctx_len;
        let $raw = llama_model::GpuRawWeights::from_gguf(&f, &$bytes, &$cfg).unwrap();
        let $aux = llama_model::GpuAuxWeights::from_gguf(&f, &$bytes, &$cfg).unwrap();
        assert!($raw.mtp.is_some(), "o 27B devia trazer bloco nextn");
        let $b = llama_vulkan::LayerSplitForward::new_com(&$ctx, &$cfg, &$raw, &$aux, true)
            .expect("backend com MTP");
        assert!($b.tem_mtp(), "a cabeça MTP devia estar montada");
    };
}

/// Erro relativo máximo entre dois vetores de logits.
fn erro_rel(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs() / y.abs().max(1.0)))
}

/// `decode_verify([a, b])` tem de devolver, na primeira metade, os mesmos logits que
/// `decode(a)` daria, e na segunda os de `decode(b)` logo em seguida.
///
/// É o contrato da fase C inteira: se a segunda coluna do matvec de logits lesse a ativação
/// errada, ou se a máscara causal do bloco deixasse o token 0 enxergar o token 1, o erro
/// apareceria aqui — e em nenhum dos testes de decode existentes.
#[test]
fn decode_verify_bate_com_o_decode_token_a_token() {
    backend_mtp!(bytes, ctx, cfg, raw, aux, backend, 256);
    let prompt: [u32; 4] = [46, 3837, 101, 9707];
    let (a, b) = (785u32, 1749u32);

    // Caminho de referência: os dois tokens, um de cada vez.
    backend.reset();
    for (pos, &t) in prompt.iter().enumerate() {
        backend.decode(t, pos).unwrap();
    }
    let l_a = backend.decode(a, prompt.len()).unwrap();
    let l_b = backend.decode(b, prompt.len() + 1).unwrap();

    // Caminho do verify: os dois de uma vez, com os pesos lidos uma só vez.
    backend.reset();
    for (pos, &t) in prompt.iter().enumerate() {
        backend.decode(t, pos).unwrap();
    }
    let dois = backend.decode_verify(&[a, b], prompt.len()).unwrap();
    assert_eq!(dois.len(), cfg.vocab * 2, "o verify devolve os dois logits");
    let (v0, v1) = dois.split_at(cfg.vocab);

    assert_eq!(
        argmax(v0),
        argmax(&l_a),
        "argmax do primeiro token divergiu"
    );
    assert_eq!(argmax(v1), argmax(&l_b), "argmax do segundo token divergiu");
    let (e0, e1) = (erro_rel(v0, &l_a), erro_rel(v1, &l_b));
    assert!(e0 < 1e-2, "logits[0]: erro relativo {e0}");
    assert!(e1 < 1e-2, "logits[1]: erro relativo {e1}");
    eprintln!("decode_verify == decode token a token (erro rel {e0:.2e} / {e1:.2e})");
}

/// Depois de uma rejeição, o estado recorrente e o comprimento do KV têm de voltar ao que
/// eram com **um** token processado — senão a geração diverge a partir dali.
///
/// A proposta é errada de propósito, que é o caso raro (39% dos passos medidos) e o único
/// que exercita o `rollback_verify`.
#[test]
fn rollback_restaura_o_estado_de_um_token() {
    backend_mtp!(bytes, ctx, cfg, raw, aux, backend, 256);
    let prompt: [u32; 4] = [46, 3837, 101, 9707];
    let a = 785u32;
    // Um token qualquer que não vai ser o proposto — a rejeição é o ponto do teste.
    let proposta_errada = 12345u32;
    let seguinte = 1749u32;

    // Referência: só `a`, e depois `seguinte`.
    backend.reset();
    for (pos, &t) in prompt.iter().enumerate() {
        backend.decode(t, pos).unwrap();
    }
    backend.decode(a, prompt.len()).unwrap();
    let esperado = backend.decode(seguinte, prompt.len() + 1).unwrap();

    // Verify rejeitado + rollback, e então o mesmo `seguinte`.
    backend.reset();
    for (pos, &t) in prompt.iter().enumerate() {
        backend.decode(t, pos).unwrap();
    }
    let dois = backend
        .decode_verify(&[a, proposta_errada], prompt.len())
        .unwrap();
    assert_eq!(dois.len(), cfg.vocab * 2);
    backend.rollback_verify().unwrap();
    let obtido = backend.decode(seguinte, prompt.len() + 1).unwrap();

    assert_eq!(argmax(&obtido), argmax(&esperado), "argmax pós-rollback");
    let e = erro_rel(&obtido, &esperado);
    assert!(e < 1e-2, "logits pós-rollback: erro relativo {e}");
    eprintln!("rollback restaura o estado de um token (erro rel {e:.2e})");
}

/// A cabeça MTP da GPU tem de propor exatamente o que `MtpHead::propor` (CPU) propõe.
///
/// A referência de CPU é o oráculo da fase B — foi ela que mediu os 60,9% de aceitação, e
/// espelha a camada de atenção do qwen35 com os dois detalhes fáceis de errar (queries com
/// stride `2 × head_dim` e o portão entrando como sigmoid sobre a saída da atenção).
///
/// Este é também o teste que pega o modo de falha silencioso do projeto: um shader com
/// número de bindings diferente do declarado na pipeline não dá erro nenhum — a saída
/// simplesmente vai para lugar nenhum e o modelo passa a propor o último token do
/// vocabulário.
#[test]
fn cabeca_mtp_na_gpu_bate_com_a_referencia_de_cpu() {
    backend_mtp!(bytes, ctx, cfg, raw, aux, backend, 128);
    let (Some(mtp_raw), Some(mtp_aux)) = (raw.mtp.as_ref(), aux.mtp.as_ref()) else {
        panic!("o modelo devia trazer bloco MTP");
    };
    let ultimo = backend.layout().len() - 1;
    let mut cabeca = llama_model::MtpHead::new(&cfg);
    backend.reset();

    // Poucos passos: cada proposta da CPU desquantiza ~1,3 GB para varrer o vocabulário.
    let mut token = 46u32;
    for pos in 0..4usize {
        let logits = backend.decode(token, pos).unwrap();
        let real_tok = argmax(&logits);
        let h = backend.dbg_hidden(ultimo).expect("hidden do último shard");

        let na_cpu = cabeca
            .propor(
                &cfg,
                mtp_raw,
                mtp_aux,
                &raw.output,
                &aux.token_embd,
                &aux.freq_table,
                &h,
                real_tok,
            )
            .unwrap();
        let na_gpu = backend.propor_mtp(real_tok, 0).unwrap();

        assert_eq!(
            na_gpu, na_cpu,
            "pos {pos}: cabeça da GPU propôs {na_gpu}, a da CPU {na_cpu}"
        );
        eprintln!("pos {pos}: GPU e CPU propõem {na_gpu}");
        token = real_tok;
    }
}

/// **O teste de aceite da frente:** com greedy, a sequência gerada com MTP é idêntica à
/// gerada sem, token a token.
///
/// Os dois caminhos rodam no mesmo backend (o 27B não cabe duas vezes na VRAM), separados
/// por `reset()`. O caminho com MTP é o mesmo que `gerar_streaming_residente` executa:
/// propor → verificar → aceitar ou desfazer.
#[test]
fn greedy_com_mtp_gera_a_mesma_sequencia() {
    /// Tokens gerados por caminho. O critério de aceite da frente pede 256.
    const N: usize = 256;

    backend_mtp!(bytes, ctx, cfg, raw, aux, backend, 1024);
    let prompt: [u32; 4] = [46, 3837, 101, 9707];

    // Caminho sem MTP: um token por passo.
    backend.reset();
    let mut logits = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        logits = backend.decode(t, pos).unwrap();
    }
    let mut pos = prompt.len();
    let mut next = argmax(&logits);
    let mut sem_mtp = Vec::with_capacity(N);
    while sem_mtp.len() < N {
        sem_mtp.push(next);
        let l = backend.decode(next, pos).unwrap();
        pos += 1;
        next = argmax(&l);
    }

    // Caminho com MTP: propor, verificar, aceitar ou desfazer.
    backend.reset();
    let mut logits = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        logits = backend.decode(t, pos).unwrap();
    }
    let mut pos = prompt.len();
    let mut hidden = 0usize;
    let mut next = argmax(&logits);
    let mut com_mtp = Vec::with_capacity(N + 1);
    let (mut aceitos, mut passos) = (0usize, 0usize);
    while com_mtp.len() < N {
        com_mtp.push(next);
        let proposta = backend.propor_mtp(next, hidden).unwrap();
        let dois = backend.decode_verify(&[next, proposta], pos).unwrap();
        let (v0, v1) = dois.split_at(cfg.vocab);
        let t1 = argmax(v0);
        passos += 1;
        if t1 == proposta {
            aceitos += 1;
            pos += 2;
            hidden = 1;
            com_mtp.push(t1);
            next = argmax(v1);
        } else {
            backend.rollback_verify().unwrap();
            pos += 1;
            hidden = 0;
            next = t1;
        }
    }
    com_mtp.truncate(N);

    #[allow(clippy::cast_precision_loss)]
    let taxa = aceitos as f64 / passos.max(1) as f64;
    eprintln!(
        "aceitação em geração real: {aceitos}/{passos} = {:.1}% ({:.2} tokens/passo)",
        taxa * 100.0,
        1.0 + taxa
    );
    let divergiu = sem_mtp
        .iter()
        .zip(&com_mtp)
        .position(|(a, b)| a != b)
        .unwrap_or(usize::MAX);
    assert_eq!(
        divergiu,
        usize::MAX,
        "greedy com MTP divergiu no token {divergiu}: sem={:?} com={:?}",
        &sem_mtp[divergiu.saturating_sub(3)..(divergiu + 1).min(N)],
        &com_mtp[divergiu.saturating_sub(3)..(divergiu + 1).min(N)]
    );
    eprintln!("greedy lossless confirmado em {N} tokens");
}
