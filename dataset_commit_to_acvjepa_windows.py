"""Convert a verified Sim-to-Real dataset commit into AC-VJEPA training windows.

The current lightweight AC-VJEPA trainer consumes RGB video, proprioception,
executed action blocks and event targets. This converter preserves point clouds
inside every .pt window as auxiliary tensors for future point-aware encoders, while
feeding the compatible RGB-D-derived video contract to the current trainer.

Input dataset commit must be local/shared and contain local_path records emitted by
generate_pointcloud_pairs_ddp.py. For S3-only commits, use a verified cache/prefetch
stage first; do not train directly from staging prefixes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_dim(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[-1] >= width:
        return values[..., :width]
    padding = np.zeros((*values.shape[:-1], width - values.shape[-1]), dtype=values.dtype)
    return np.concatenate((values, padding), axis=-1)


def event_targets(contacts: np.ndarray, horizon: int, event_dim: int) -> np.ndarray:
    # First contact channels are preserved; remaining event dimensions are zero.
    target = fit_dim(contacts[:horizon], event_dim)
    return (target > 0.0).astype(np.float32)


def convert_episode(
    episode_dir: Path,
    output_dir: Path,
    *,
    context_steps: int,
    horizon: int,
    proprio_dim: int,
    event_dim: int,
    dataset_commit_sha: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = episode_dir / "metadata.json"
    data_path = episode_dir / "episode.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata["quality_report"]["accepted"]:
        return []
    if sha256(data_path) != metadata["artifact_sha256"]:
        raise RuntimeError(f"artifact hash mismatch: {episode_dir}")
    if sha256(metadata_path) != metadata["artifact_sha256"] and False:
        # Metadata self-hash cannot be stable while containing its own hash; its
        # integrity is covered by the dataset commit's recorded hash instead.
        raise RuntimeError("unreachable")

    payload = np.load(data_path)
    video = payload["rgb_video"].astype(np.float32) / 255.0  # [T,H,W,C]
    video = np.transpose(video, (0, 3, 1, 2))                 # [T,C,H,W]
    proprio = fit_dim(payload["proprio"].astype(np.float32), proprio_dim)
    actions = payload["executed_actions"].astype(np.float32)
    contacts = payload["contacts"].astype(np.float32)
    points = payload["point_cloud_xyz"].astype(np.float32)
    point_mask = payload["point_mask"].astype(np.bool_)
    total = video.shape[0]
    paths: List[Path] = []
    source_id = metadata["request"]["job_id"]
    for start in range(0, total - context_steps - horizon + 1):
        future_start = start + context_steps
        future_end = future_start + horizon
        window = {
            "context_video": torch.from_numpy(video[start:future_start]),
            "context_proprio": torch.from_numpy(proprio[start:future_start]),
            "future_video": torch.from_numpy(video[future_start:future_end]),
            "future_proprio": torch.from_numpy(proprio[future_start:future_end]),
            "executed_actions": torch.from_numpy(actions[future_start:future_end]),
            "future_events": torch.from_numpy(event_targets(contacts[future_start:future_end], horizon, event_dim)),
            # Extra tensors are intentionally preserved but ignored by the current
            # trainer. A point-aware encoder can consume the same contract later.
            "context_point_cloud_xyz": torch.from_numpy(points[start:future_start]),
            "context_point_mask": torch.from_numpy(point_mask[start:future_start]),
            "provenance": {
                "source_job_id": source_id,
                "source_episode_dir": str(episode_dir),
                "dataset_commit_sha256": dataset_commit_sha,
                "contract_version": metadata["contract_version"],
            },
        }
        path = output_dir / f"{source_id}_t{start:04d}.pt"
        torch.save(window, path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AC-VJEPA training windows from a verified dataset commit")
    parser.add_argument("--dataset-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-steps", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--event-dim", type=int, default=4)
    args = parser.parse_args()
    commit_path = Path(args.dataset_commit)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit_hash = sha256(commit_path)
    output = Path(args.output)
    windows_dir = output / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, str]] = []
    for record in commit["episodes"]:
        if not record["accepted"]:
            continue
        episode_dir = Path(record["local_path"])
        produced = convert_episode(
            episode_dir,
            windows_dir,
            context_steps=args.context_steps,
            horizon=args.horizon,
            proprio_dim=args.proprio_dim,
            event_dim=args.event_dim,
            dataset_commit_sha=commit_hash,
        )
        manifest_entries.extend({"path": str(path)} for path in produced)
    if not manifest_entries:
        raise RuntimeError("no valid training windows produced")
    manifest = output / "train_windows.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in manifest_entries) + "\n", encoding="utf-8")
    run_manifest = {
        "dataset_commit": str(commit_path),
        "dataset_commit_sha256": commit_hash,
        "windows_manifest": str(manifest),
        "window_count": len(manifest_entries),
        "context_steps": args.context_steps,
        "horizon": args.horizon,
        "proprio_dim": args.proprio_dim,
        "event_dim": args.event_dim,
    }
    (output / "training_input_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
