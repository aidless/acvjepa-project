"""Generate tiny synthetic B-layer clip frames (PNG) for pipeline smoke tests.

Usage: python scripts/make_synthetic_clips.py <output_dir>
Produces two 16-frame clips (clip_kitchen_a/b) as PNGs under output_dir.
These are NOT training data — they only exercise the video -> windows chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("verify_artifacts/b_layer_clips")
    for clip_id in ("clip_kitchen_a", "clip_kitchen_b"):
        directory = root / clip_id
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(hash(clip_id) % 2**32)
        for index in range(16):
            frame = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
            Image.fromarray(frame).save(directory / f"frame_{index:05d}.png")
    print(root)


if __name__ == "__main__":
    main()
