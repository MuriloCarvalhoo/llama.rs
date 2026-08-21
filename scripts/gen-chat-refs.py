#!/usr/bin/env python3
"""Gera refs/chat_qwen38.json: a saída do chat template **real** do GGUF.

O template mora nos metadados do modelo (`tokenizer.chat_template`) e é Jinja com
macros, namespace e slicing reverso. Em vez de embutir um motor Jinja no runtime, o
`llama-chat` reimplementa o formato em Rust — e é este arquivo que prova que a
reimplementação bate, caso a caso, com o que o modelo viu no treino.

Reproduz o ambiente do `transformers.apply_chat_template`: trim_blocks e
lstrip_blocks ligados, `tojson` sem escape HTML e com ensure_ascii desligado.

Uso: python3 scripts/gen-chat-refs.py [caminho-do-gguf]
"""

import json
import struct
import sys
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

RAIZ = Path(__file__).resolve().parent.parent
PADRAO = RAIZ / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
SAIDA = RAIZ / "refs" / "chat_qwen38.json"


def ler_metadados(caminho):
    """Lê só o cabeçalho GGUF v3 — o suficiente para o chat template."""
    f = open(caminho, "rb")
    f.read(4)  # magic
    struct.unpack("<I", f.read(4))  # versão
    struct.unpack("<Q", f.read(8))  # n_tensors
    (nkv,) = struct.unpack("<Q", f.read(8))

    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    escalares = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            if et == 8:
                return [rstr() for _ in range(n)]
            fmt = escalares[et]
            return list(struct.unpack("<" + fmt * n, f.read(struct.calcsize("<" + fmt) * n)))
        fmt = escalares[t]
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

    meta = {}
    for _ in range(nkv):
        k = rstr()
        (t,) = struct.unpack("<I", f.read(4))
        meta[k] = rval(t)
    return meta


def ambiente(template):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(template)


LER_ARQUIVO = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Lê um arquivo do disco",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "caminho"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}

CASOS = [
    {
        "nome": "system_e_user",
        "messages": [
            {"role": "system", "content": "Você é um assistente."},
            {"role": "user", "content": "Oi"},
        ],
        "add_generation_prompt": True,
    },
    {
        "nome": "sem_system",
        "messages": [{"role": "user", "content": "Oi"}],
        "add_generation_prompt": True,
    },
    {
        "nome": "dois_systems_iniciais",
        "messages": [
            {"role": "system", "content": "Regra 1"},
            {"role": "system", "content": "Regra 2"},
            {"role": "user", "content": "Oi"},
        ],
        "add_generation_prompt": True,
    },
    {
        "nome": "com_tools",
        "messages": [
            {"role": "system", "content": "Você é um agente de código."},
            {"role": "user", "content": "Leia o main.rs"},
        ],
        "tools": [LER_ARQUIVO],
        "add_generation_prompt": True,
    },
    {
        "nome": "tools_sem_system",
        "messages": [{"role": "user", "content": "Leia o main.rs"}],
        "tools": [LER_ARQUIVO],
        "add_generation_prompt": True,
    },
    {
        "nome": "multi_turno_com_reasoning",
        "messages": [
            {"role": "user", "content": "Quanto é 2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "somando dois e dois"},
            {"role": "user", "content": "E 3+3?"},
        ],
        "add_generation_prompt": True,
    },
    {
        "nome": "tool_call_e_resposta",
        "messages": [
            {"role": "user", "content": "Leia o main.rs"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "preciso ler o arquivo",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "read", "arguments": {"path": "src/main.rs", "limit": 10}},
                    }
                ],
            },
            {"role": "tool", "content": "fn main() {}"},
            {"role": "user", "content": "Resuma"},
        ],
        "tools": [LER_ARQUIVO],
        "add_generation_prompt": True,
    },
    {
        "nome": "duas_tool_calls_e_duas_respostas",
        "messages": [
            {"role": "user", "content": "Leia os dois arquivos"},
            {
                "role": "assistant",
                "content": "vou ler os dois",
                "tool_calls": [
                    {"type": "function", "function": {"name": "read", "arguments": {"path": "a.rs"}}},
                    {"type": "function", "function": {"name": "read", "arguments": {"path": "b.rs"}}},
                ],
            },
            {"role": "tool", "content": "conteudo de a"},
            {"role": "tool", "content": "conteudo de b"},
        ],
        "tools": [LER_ARQUIVO],
        "add_generation_prompt": True,
    },
    {
        "nome": "thinking_desligado",
        "messages": [
            {"role": "system", "content": "Seja breve."},
            {"role": "user", "content": "Oi"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "nome": "reasoning_effort_low",
        "messages": [{"role": "user", "content": "Oi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "low",
    },
    {
        "nome": "reasoning_effort_medium",
        "messages": [{"role": "user", "content": "Oi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "medium",
    },
    {
        "nome": "conteudo_em_partes",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "parte 1"}, {"type": "text", "text": "parte 2"}]},
        ],
        "add_generation_prompt": True,
    },
    {
        "nome": "sem_generation_prompt",
        "messages": [{"role": "user", "content": "Oi"}],
        "add_generation_prompt": False,
    },
]


def main():
    caminho = Path(sys.argv[1]) if len(sys.argv) > 1 else PADRAO
    meta = ler_metadados(caminho)
    tmpl = ambiente(meta["tokenizer.chat_template"])

    casos = []
    for caso in CASOS:
        kwargs = {k: v for k, v in caso.items() if k != "nome"}
        kwargs.setdefault("tools", None)
        casos.append({**caso, "esperado": tmpl.render(**kwargs)})

    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps({"modelo": caminho.name, "casos": casos}, ensure_ascii=False, indent=2))
    print(f"{len(casos)} casos → {SAIDA}")


if __name__ == "__main__":
    main()
