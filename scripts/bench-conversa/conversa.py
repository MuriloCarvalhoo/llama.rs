"""Conversa longa sintética para comparar llama.rs e llama.cpp na mesma carga.

Uso (normalmente via rodar_config.sh):
  python3 conversa.py calibrar <porta_llamacpp>     # grava conversas.json (usa /apply-template e /tokenize do llama.cpp)
  python3 conversa.py crescer <porta> <motor> <rot> # uma conversa só, turno a turno (llama.cpp)
  python3 conversa.py frioquente <porta> <rot>      # por profundidade: pedido frio + o mesmo pedido de novo (llama.rs)
  python3 conversa.py rodar <porta> <motor> <rot> [profundidades...]  # pedidos frios

Resultados e o conversas.json ficam em $BENCH_SAIDA (padrão /tmp/bench-conversa).

Cada profundidade é uma conversa independente (mensagem de sistema com um id próprio), para
que nenhum dos dois servidores reaproveite prefixo de outra: o prefill medido é sempre frio.
"""
import glob
import json
import os
import sys
import time
import urllib.request

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
# conversas.json e resultados vão para BENCH_SAIDA (padrão /tmp/bench-conversa).
AQUI = os.environ.get("BENCH_SAIDA", "/tmp/bench-conversa")
os.makedirs(AQUI, exist_ok=True)
ALVOS = [2000, 8000, 16000, 24000, 31000]
MAX_TOKENS = 256


def post(porta, rota, corpo, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{porta}{rota}", json.dumps(corpo).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def turnos():
    """Pares (usuário, assistente) com trechos dos docs do repositório."""
    texto = ""
    for p in sorted(glob.glob("/home/murilo/llama.rs/docs/*.md")) + sorted(glob.glob("/home/murilo/llama.rs/docs/planos/*.md")):
        texto += open(p).read() + "\n\n"
    pedacos = [texto[i:i + 4500] for i in range(0, len(texto), 4500)]
    out = []
    for n, p in enumerate(pedacos):
        frase = p.strip().split("\n")[0][:160]
        out.append((f"Trecho {n + 1} da documentação do projeto:\n\n{p}\n\nGuarde o conteúdo; vou perguntar no final.",
                    f"Entendido, trecho {n + 1} registrado. Ele começa com: {frase}"))
    return out


def mensagens(k, sid):
    msgs = [{"role": "system", "content": f"Sessão de teste {sid}. Você é um engenheiro de desempenho de GPUs, direto e preciso."}]
    for u, a in turnos()[:k]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": "Com base em todos os trechos, explique em detalhes quais são os gargalos de desempenho do projeto e proponha um plano priorizado."})
    return msgs


def n_tokens(porta, msgs):
    prompt = post(porta, "/apply-template", {"messages": msgs})["prompt"]
    return len(post(porta, "/tokenize", {"content": prompt})["tokens"])


def calibrar(porta):
    total = len(turnos())
    conversas = []
    k = 0
    for i, alvo in enumerate(ALVOS):
        while k < total and n_tokens(porta, mensagens(k + 1, i)) <= alvo:
            k += 1
        msgs = mensagens(k, i)
        conversas.append({"alvo": alvo, "turnos": k, "tokens_llamacpp": n_tokens(porta, msgs), "mensagens": msgs})
        print(f"alvo {alvo}: {k} turnos, {conversas[-1]['tokens_llamacpp']} tokens")
    json.dump(conversas, open(f"{AQUI}/conversas.json", "w"), ensure_ascii=False)


def esfriar(limite=65, teto_s=240):
    """Espera a junction mais quente das GPUs cair abaixo de `limite` °C antes de um pedido."""
    t0 = time.time()
    while time.time() - t0 < teto_s:
        temps = []
        for lab in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp*_label"):
            if open(lab).read().strip() == "junction":
                temps.append(int(open(lab.replace("_label", "_input")).read()) / 1000)
        if temps and max(temps) < limite:
            return max(temps)
        time.sleep(2)
    return max(temps) if temps else None


def rodar(porta, motor, rotulo, quais):
    conversas = json.load(open(f"{AQUI}/conversas.json"))
    res = []
    for c in conversas:
        if quais and str(c["alvo"]) not in quais:
            continue
        corpo = {"messages": c["mensagens"], "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False}
        if motor == "llamacpp":
            corpo["cache_prompt"] = False
        temp_ini = esfriar()
        t0 = time.time()
        j = post(porta, "/v1/chat/completions", corpo)
        parede = time.time() - t0
        r = {"alvo": c["alvo"], "parede_s": round(parede, 2), "temp_ini": temp_ini, "usage": j.get("usage")}
        if "timings" in j:
            t = j["timings"]
            r.update(prompt_n=t["prompt_n"], prompt_tps=round(t["prompt_per_second"], 2),
                     decode_n=t["predicted_n"], decode_tps=round(t["predicted_per_second"], 2),
                     draft_n=t.get("draft_n"), draft_aceitos=t.get("draft_n_accepted"))
        res.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{AQUI}/resultados-{rotulo}.json", "w"), ensure_ascii=False, indent=1)


PERGUNTA = "Com base em todos os trechos, explique em detalhes quais são os gargalos de desempenho do projeto e proponha um plano priorizado."


def crescer(porta, motor, rotulo):
    """Uma conversa só, turno a turno, como um cliente real: cada pedido estende o anterior e
    os dois servidores reaproveitam o prefixo. Nos marcos (os turnos das profundidades
    calibradas) a última mensagem vira a pergunta final e a resposta tem 256 tokens."""
    conversas = json.load(open(f"{AQUI}/conversas.json"))
    marcos = {c["turnos"]: c["alvo"] for c in conversas}
    pares = turnos()
    sistema = {"role": "system", "content": "Sessão de teste crescente. Você é um engenheiro de desempenho de GPUs, direto e preciso."}
    res = []
    for k in range(1, max(marcos) + 1):
        base = [sistema]
        for u, a in pares[:k - 1]:
            base += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        # Turno normal: só a mensagem nova, para o cache crescer em pedaços de ~1,5k tokens.
        corpo = {"messages": base + [{"role": "user", "content": pares[k - 1][0]}], "max_tokens": 1,
                 "temperature": 0, "stream": False}
        esfriar(75)
        # HOLD_PREFILL=standard: clock fixo só durante o turno de prefill (o llama.cpp não
        # tem controle térmico e leva a card1 a 101-104 °C em segundos no automático).
        segura = None
        if os.environ.get("HOLD_PREFILL"):
            import subprocess
            segura = subprocess.Popen(["python3", f"{SCRIPTS}/segura_pstate.py", os.environ["HOLD_PREFILL"],
                                       "/dev/dri/renderD128", "/dev/dri/renderD129"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        try:
            j = post(porta, "/v1/chat/completions", corpo)
        finally:
            if segura:
                segura.terminate()
                segura.wait()
        t = j.get("timings", {})
        print(json.dumps({"turno": k, "prompt_tokens": j["usage"]["prompt_tokens"], "prompt_n": t.get("prompt_n"),
                          "prompt_tps": t.get("prompt_per_second")}), flush=True)
        if k in marcos:
            u, a = pares[k - 1]
            msgs = base + [{"role": "user", "content": u}, {"role": "assistant", "content": a},
                           {"role": "user", "content": PERGUNTA}]
            corpo = {"messages": msgs, "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False}
            temp_ini = esfriar(65)
            t0 = time.time()
            j = post(porta, "/v1/chat/completions", corpo)
            r = {"marco": marcos[k], "turnos": k, "parede_s": round(time.time() - t0, 2), "temp_ini": temp_ini,
                 "prompt_tokens": j["usage"]["prompt_tokens"], "saida": j["usage"]["completion_tokens"]}
            if "timings" in j:
                t = j["timings"]
                r.update(prompt_n=t["prompt_n"], prompt_tps=round(t["prompt_per_second"], 1),
                         decode_tps=round(t["predicted_per_second"], 2),
                         draft_n=t.get("draft_n"), draft_aceitos=t.get("draft_n_accepted"))
            res.append(r)
            print("MARCO " + json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{AQUI}/crescer-{rotulo}.json", "w"), ensure_ascii=False, indent=1)


def frio_e_quente(porta, rotulo):
    """Para o llama.rs, que não reaproveita o prefixo entre turnos que o template reescreve:
    por profundidade, um pedido frio (mede o prefill) e, depois de esfriar, o **mesmo** pedido
    (acerta o snapshot do fim do prompt; mede o decode com a GPU fria, como nos marcos do
    llama.cpp)."""
    for c in json.load(open(f"{AQUI}/conversas.json")):
        corpo = {"messages": c["mensagens"], "max_tokens": 1, "temperature": 0, "stream": False}
        esfriar(65)
        t0 = time.time()
        post(porta, "/v1/chat/completions", corpo)
        print(json.dumps({"alvo": c["alvo"], "fase": "frio", "parede_s": round(time.time() - t0, 2)}), flush=True)
        corpo["max_tokens"] = MAX_TOKENS
        temp_ini = esfriar(65)
        t0 = time.time()
        j = post(porta, "/v1/chat/completions", corpo)
        print(json.dumps({"alvo": c["alvo"], "fase": "quente", "temp_ini": temp_ini, "parede_s": round(time.time() - t0, 2),
                          "usage": j["usage"]}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "frioquente":
        frio_e_quente(int(sys.argv[2]), sys.argv[3])
        sys.exit(0)
    if sys.argv[1] == "crescer":
        crescer(int(sys.argv[2]), sys.argv[3], sys.argv[4])
        sys.exit(0)
    if sys.argv[1] == "calibrar":
        calibrar(int(sys.argv[2]))
    else:
        rodar(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:])
