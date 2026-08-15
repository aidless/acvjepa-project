"""CUDA smoke for P2 action-conditioned post-training (C-layer windows).

Run (single proc, CUDA):
    python scripts/p2_cuda_smoke.py --manifest <train_windows.jsonl> \
        --checkpoint <model.safetensors> --output <out_dir>

Trains `train_ac_vjepa_ddp.train` (action-conditioned loss with real events)
for 1 epoch on CUDA with the frozen official V-JEPA backbone and reports the
CUDA peak memory, verifying the P2 entry point (`--init-from vjepa2hf:`)
on the GPU path.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train_ac_vjepa_ddp import TrainConfig, train  # noqa: E402


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

    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    os.environ["USE_LIBUV"] = "0"

    cfg = TrainConfig(
        manifest=args.manifest,
        output=args.output,
        epochs=args.epochs,
        per_rank_batch_size=args.batch_size,
        gradient_accumulation=1,
        num_workers=0,
        latent_dim=64,
        max_horizon=3,
        init_from=f"vjepa2hf:{args.checkpoint}:frozen",
        init_img_size=384,
        save_interval_steps=10**9,
        log_interval_steps=1,
        ema_momentum=0.996,
        ema_broadcast_interval=10**9,
        amp=False,
    )
    torch.cuda.reset_peak_memory_stats()
    train(cfg)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(
        f'{{"smoke_test": "passed", "mode": "p2_cuda", "cuda_peak_mb": {peak_mb:.1f}, '
        f'"budget_mb": 6144, "device": "{torch.cuda.get_device_name(0)}"}}',
        flush=True,
    )


if __name__ == "__main__":
    main()
