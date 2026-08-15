"""Create a tiny local AC-V-JEPA window dataset for train_ac_vjepa_ddp.py smoke testing.

Supports --img-size for V-JEPA backbone runs (384px) and --root for a custom
output directory (default keeps the historical /home/ubuntu path).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--root", default="demo_ddp_data", help="output directory (portable default)")
    args = parser.parse_args()

    torch.manual_seed(42)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    entries = []
    for index in range(4):
        window = {
            "context_video": torch.randn(4, 3, args.img_size, args.img_size),
            "context_proprio": torch.randn(4, 8),
            "future_video": torch.randn(3, 3, args.img_size, args.img_size),
            "future_proprio": torch.randn(3, 8),
            "executed_actions": torch.randn(3, 20),
            "future_events": torch.randint(0, 2, (3, 4)).float(),
        }
        path = root / f"window_{index:03d}.pt"
        torch.save(window, path)
        entries.append({"path": str(path)})
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
