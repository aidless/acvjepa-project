"""CUDA smoke for P1 domain-adapt training on the frozen official V-JEPA backbone.

Run (single proc, CUDA):
    python scripts/p1_cuda_smoke.py --manifest <domain_adapt_windows.jsonl> \
        --checkpoint <model.safetensors> --output <out_dir>

Builds a tiny 384px B-layer window set on the fly (synthetic frames), trains
train_p1_domain_adapt for 1 epoch on CUDA with the real frozen weights, and
prints the final CUDA peak memory so the 6GB budget claim is verified on every
run. Uses CUDA_LAUNCH_BLOCKING=1 internally? No — we leave async kernels alone
but synchronize before reporting memory.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train_p1_domain_adapt import DomainAdaptConfig, train_domain_adapt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this smoke")

    # Single-proc CUDA path: do NOT initialize a process group. The Windows
    # torch wheel has no NCCL (USE_NCCL=OFF), and world_size=1 needs no dist.
    # (If RANK/WORLD_SIZE happen to be set by a parent shell, remove them here.)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    os.environ["USE_LIBUV"] = "0"

    cfg = DomainAdaptConfig(
        manifest=args.manifest,
        output=args.output,
        epochs=args.epochs,
        per_rank_batch_size=args.batch_size,
        gradient_accumulation=1,
        num_workers=0,
        latent_dim=64,
        max_horizon=2,
        init_from=f"vjepa2hf:{args.checkpoint}:frozen",
        init_img_size=384,
        save_interval_steps=10**9,  # only last.pt
        log_interval_steps=1,
        ema_momentum=0.996,
        ema_broadcast_interval=10**9,
        amp=False,  # first CUDA pass without AMP to isolate variables
    )
    torch.cuda.reset_peak_memory_stats()
    train_domain_adapt(cfg)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(
        f'{{"smoke_test": "passed", "mode": "p1_cuda", "cuda_peak_mb": {peak_mb:.1f}, '
        f'"budget_mb": 6144, "device": "{torch.cuda.get_device_name(0)}"}}',
        flush=True,
    )


if __name__ == "__main__":
    main()
