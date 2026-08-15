"""Manual 2-process Gloo runner: replaces torchrun's elastic agent (which itself
holds a full torch import and pushes commit usage past this machine's pagefile).

Usage:
  python manual_gloo_runner.py <script.py> [--master-port N] [--script-args ...]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PORT = int(os.environ.get("MPORT", "29671"))


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    rest = sys.argv[2:]
    env = dict(os.environ)
    env.update({
        "USE_LIBUV": "0",
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTORCH_NO_CUDA": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(PORT),
    })
    procs = []
    for rank in (0, 1):
        e = dict(env)
        e["RANK"] = str(rank)
        e["LOCAL_RANK"] = str(rank)
        e["WORLD_SIZE"] = "2"
        procs.append(subprocess.Popen(
            [sys.executable, str(script), *rest],
            cwd=str(script.parent),
            env=e,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ))
    code = 0
    for rank, p in enumerate(procs):
        out = p.communicate(timeout=600)[0]
        text = out.decode("utf-8", errors="replace")
        sys.stdout.buffer.write(f"===== rank {rank} exit={p.returncode} =====\n".encode("utf-8", "replace"))
        sys.stdout.buffer.write(text[-2500:].encode("utf-8", "replace"))
        sys.stdout.buffer.flush()
        if p.returncode != 0:
            code = p.returncode
    sys.exit(code)


if __name__ == "__main__":
    main()
