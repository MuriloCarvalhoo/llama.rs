"""Segura um pstate estável nas GPUs até ser morto (mesma ioctl de crates/llama-vulkan/src/pstate.rs).

Uso: python3 segura_pstate.py <standard|peak|min_sclk> /dev/dri/renderD128 [/dev/dri/renderD129 ...]
O kernel devolve o device ao automático quando o fd fecha (inclusive se este processo morrer).
"""
import fcntl
import os
import signal
import struct
import sys

DRM_IOCTL_AMDGPU_CTX = 0xC0106442
MODOS = {"standard": 1, "min_sclk": 2, "peak": 4}
modo = MODOS[sys.argv[1]]
fds = []
for no in sys.argv[2:]:
    fd = os.open(no, os.O_RDWR)
    a = bytearray(struct.pack("4I", 1, 0, 0, 0))          # ALLOC_CTX
    fcntl.ioctl(fd, DRM_IOCTL_AMDGPU_CTX, a)
    ctx = struct.unpack("4I", a)[0]
    b = bytearray(struct.pack("4I", 6, modo, ctx, 0))     # SET_STABLE_PSTATE
    fcntl.ioctl(fd, DRM_IOCTL_AMDGPU_CTX, b)
    fds.append(fd)
    print(f"{no}: pstate {sys.argv[1]} fixo (ctx {ctx})", flush=True)
signal.pause()
