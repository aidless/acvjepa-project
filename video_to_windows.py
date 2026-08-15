"""Cut B-layer unlabeled videos into AC-VJEPA domain-adaptation windows (M2).

Purpose (P1 无动作 JEPA 域适配): the lightweight V-JEPA plan adapts the frozen
official encoder + a small predictor on *unlabeled* local kitchen/table videos
before any action-conditioned training. Those windows intentionally carry NO
executed actions and NO event targets:

    context_video:  [T_ctx, C, H, W]   (float32, [0,1])
    context_proprio:[T_ctx, P]         (zeros when unavailable)
    future_video:   [T_hor, C, H, W]
    future_proprio: [T_hor, P]
    executed_actions:[T_hor, A]        (zeros; NOT used for action training)
    future_events:  [T_hor, E]         (zeros)

The file layout mirrors `dataset_commit_to_acvjepa_windows.py` (one .pt per
window + a JSONL manifest), so the same downstream tooling and DDP trainer can
consume them. A provenance block marks these as `domain_adaptation_only`, which
downstream action-conditioned pipelines must NOT treat as action supervision.

Input formats:
  --video-dir : directory of frames as PNG/JPG, one subdir per clip:
                <video-dir>/<clip_id>/frame_00000.png, frame_00001.png ...
  --list      : JSONL [{path: <clip_dir>, clip_id, split, video_dir: optional}]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from train_ac_vjepa_ddp import REQUIRED_KEYS


def load_frames(clip_dir: Path, height: int, width: int) -> np.ndarray:
    files = sorted(clip_dir.glob("*.png")) or sorted(clip_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no frames in {clip_dir}")
    frames = []
    for path in files:
        image = Image.open(path).convert("RGB").resize((width, height))
        frames.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)  # [T, H, W, C]


def make_window(
    video: np.ndarray,
    start: int,
    *,
    context_steps: int,
    horizon: int,
    proprio_dim: int,
    action_dim: int,
    event_dim: int,
    clip_id: str,
    split: str,
    video_dir: str,
) -> Dict:
    future_start = start + context_steps
    future_end = future_start + horizon
    context = video[start:future_start]
    future = video[future_start:future_end]
    assert context.shape[0] == context_steps and future.shape[0] == horizon
    context_t = torch.from_numpy(context).permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
    future_t = torch.from_numpy(future).permute(0, 3, 1, 2).contiguous()
    return {
        "context_video": context_t,
        "context_proprio": torch.zeros(context_steps, proprio_dim, dtype=torch.float32),
        "future_video": future_t,
        "future_proprio": torch.zeros(horizon, proprio_dim, dtype=torch.float32),
        "executed_actions": torch.zeros(horizon, action_dim, dtype=torch.float32),
        "future_events": torch.zeros(horizon, event_dim, dtype=torch.float32),
        "provenance": {
            "domain_adaptation_only": True,
            "source_clip_id": clip_id,
            "source_split": split,
            "source_video_dir": video_dir,
            "contract": "unlabeled-domain-adaptation-v1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut B-layer unlabeled videos into domain-adaptation windows")
    parser.add_argument("--video-dir", help="root containing <clip_id>/frame_*.png subdirs")
    parser.add_argument("--list", help="JSONL of clips: {path, clip_id, split}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-steps", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--event-dim", type=int, default=4)
    args = parser.parse_args()
    if bool(args.video_dir) == bool(args.list):
        raise SystemExit("provide exactly one of --video-dir or --list")

    clips: List[Dict] = []
    if args.list:
        for line in Path(args.list).read_text(encoding="utf-8").splitlines():
            if line.strip():
                clips.append(json.loads(line))
    else:
        root = Path(args.video_dir)
        for clip_dir in sorted(root.iterdir()):
            if clip_dir.is_dir():
                clips.append({"path": str(clip_dir), "clip_id": clip_dir.name, "split": "train"})

    output = Path(args.output)
    windows_dir = output / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, str]] = []
    produced = 0
    for clip in clips:
        clip_dir = Path(clip["path"])
        clip_id = clip["clip_id"]
        split = clip.get("split", "train")
        video = load_frames(clip_dir, args.height, args.width)
        total = video.shape[0]
        for start in range(0, total - args.context_steps - args.horizon + 1):
            window = make_window(
                video, start,
                context_steps=args.context_steps,
                horizon=args.horizon,
                proprio_dim=args.proprio_dim,
                action_dim=args.action_dim,
                event_dim=args.event_dim,
                clip_id=clip_id,
                split=split,
                video_dir=str(clip_dir),
            )
            path = windows_dir / f"{clip_id}_t{start:04d}.pt"
            torch.save(window, path)
            manifest_entries.append({"path": str(path)})
            produced += 1
    if not manifest_entries:
        raise RuntimeError("no windows produced (clips shorter than context+horizon?)")
    manifest = output / "domain_adapt_windows.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in manifest_entries) + "\n", encoding="utf-8")
    print(json.dumps({
        "windows": produced,
        "clips": len(clips),
        "manifest": str(manifest),
        "height": args.height, "width": args.width,
        "context_steps": args.context_steps, "horizon": args.horizon,
        "domain_adaptation_only": True,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
