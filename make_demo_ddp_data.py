"""Create a tiny local AC-V-JEPA window dataset for train_ac_vjepa_ddp.py smoke testing."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def main() -> None:
    torch.manual_seed(42)
    root = Path("/home/ubuntu/lecun_analysis/demo_ddp_data")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    entries = []
    for index in range(4):
        window = {
            "context_video": torch.randn(4, 3, 32, 32),
            "context_proprio": torch.randn(4, 8),
            "future_video": torch.randn(3, 3, 32, 32),
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
