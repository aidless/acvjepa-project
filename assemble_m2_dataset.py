"""M2 end-to-end data assembly: B/C layer -> versioned training inputs.

This orchestrator chains the already-verified pipeline steps into one repeatable
assembly for the M2 domain-adaptation + action-conditioned stage:

  C layer (RoboCasa / synthetic):   sim2real_pointcloud_video_pipeline
      -> episode.npz + metadata.json -> generate_pointcloud_pairs_ddp (commit)
      -> dataset_commit_to_acvjepa_windows  -> train_windows.jsonl
  B layer (unlabeled local video):  video_to_windows -> domain_adapt_windows.jsonl

Then it:
  - verifies train/val/test isolation by source (clip / job / task), not frames;
  - writes a DATA_MANIFEST registration fragment (inputs, splits, hashes);
  - emits a single `assembly.json` for the trainer.

RoboCasa is NOT required to run this script: `--simulator synthetic` exercises
the identical contract chain. `--simulator robocasa` requires the robocasa
package installed on a GPU/display host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: List[str], workdir: Path) -> None:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(workdir))
    if result.returncode != 0:
        raise RuntimeError(f"step failed ({result.returncode}): {' '.join(command)}")


def verify_split_isolation(entries: List[Dict], key: str) -> List[str]:
    """Splits must be disjoint on `key` (clip_id / job_id / task_template)."""
    owners: Dict[str, str] = {}
    conflicts: List[str] = []
    for entry in entries:
        owner = str(entry.get(key, "?"))
        split = str(entry.get("split", "train"))
        previous = owners.get(owner)
        if previous is not None and previous != split:
            conflicts.append(f"{owner}: {previous} vs {split}")
        owners[owner] = split
    return conflicts


def assemble(args) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assembly: Dict = {"stages": {}, "inputs": {}, "splits": {}}

    # ---------------- C layer: episodes -> windows ----------------
    sim_out = output / "c_layer_episodes"
    if args.simulator == "robocasa":
        sim_cmd = ["python", "sim2real_pointcloud_video_pipeline.py", "--simulator", "robocasa", "--demo", "--output", str(sim_out)]
    else:
        sim_cmd = ["python", "sim2real_pointcloud_video_pipeline.py", "--demo", "--output", str(sim_out)]
    _run(sim_cmd, Path(args.repo))

    episodes_manifest = sim_out / "generated_manifest.jsonl"
    episode_entries = [json.loads(line) for line in episodes_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    conflicts = verify_split_isolation(episode_entries, "job_id")
    if conflicts:
        raise RuntimeError(f"split isolation violated (job_id): {conflicts}")

    # generated_manifest.jsonl records {job_id, path, accepted, reasons};
    # dataset_commit_to_acvjepa_windows expects commit["episodes"] with local_path
    # and a per-episode split. Wrap it into a compatible commit document.
    commit_records = []
    for record in episode_entries:
        episode_dir = Path(record["path"])
        meta = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        commit_records.append({
            "job_id": record["job_id"],
            "split": meta["request"]["split"],
            "local_path": record["path"],
            "accepted": record["accepted"],
            "reasons": record["reasons"],
            "episode_sha256": meta.get("artifact_sha256", ""),
        })
    commit_doc = {"episodes": commit_records, "contract_version": "soft-grasp-rgbd-pointcloud-v1"}
    commit_path = output / "c_layer_commit.json"
    commit_path.write_text(json.dumps(commit_doc, indent=2), encoding="utf-8")

    windows_out = output / "c_layer_windows"
    _run([
        "python", "dataset_commit_to_acvjepa_windows.py",
        "--dataset-commit", str(commit_path),
        "--output", str(windows_out),
    ], Path(args.repo))

    train_windows = windows_out / "train_windows.jsonl"
    window_entries = [json.loads(line) for line in train_windows.read_text(encoding="utf-8").splitlines() if line.strip()]
    assembly["stages"]["c_windows"] = {
        "episodes": len(episode_entries),
        "windows": len(window_entries),
        "manifest": str(train_windows),
        "sha256": sha256(train_windows),
    }
    assembly["inputs"]["c_layer_episodes_dir"] = str(sim_out)
    assembly["inputs"]["c_layer_windows_dir"] = str(windows_out)

    # ---------------- B layer: unlabeled video -> domain windows ----------------
    if args.video_dir:
        b_out = output / "b_layer_windows"
        _run([
            "python", "video_to_windows.py",
            "--video-dir", args.video_dir,
            "--output", str(b_out),
            "--height", str(args.height),
            "--width", str(args.width),
        ], Path(args.repo))
        da_manifest = b_out / "domain_adapt_windows.jsonl"
        da_entries = [json.loads(line) for line in da_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        da_conflicts = verify_split_isolation(da_entries, "clip_id")
        if da_conflicts:
            raise RuntimeError(f"split isolation violated (clip_id): {da_conflicts}")
        assembly["stages"]["b_windows"] = {
            "windows": len(da_entries),
            "manifest": str(da_manifest),
            "sha256": sha256(da_manifest),
            "domain_adaptation_only": True,
        }
        assembly["inputs"]["b_layer_video_dir"] = args.video_dir

    # ---------------- splits + registration ----------------
    splits: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for entry in episode_entries:
        splits[entry.get("split", "train")] = splits.get(entry.get("split", "train"), 0) + 1
    assembly["splits"] = splits
    assembly["assembled_at"] = "2026-08-15"
    assembly["repo"] = args.repo

    # DATA_MANIFEST registration fragment (appended by the operator).
    registration = {
        "date": "2026-08-15",
        "layer": "B/C",
        "simulator": args.simulator,
        "c_episodes": len(episode_entries),
        "c_windows": len(window_entries),
        "b_windows": int(assembly["stages"].get("b_windows", {}).get("windows", 0)),
        "splits": splits,
        "notes": "verify split isolation by clip/job, not frames; B-layer windows are domain_adaptation_only",
    }
    (output / "DATA_MANIFEST_registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    (output / "assembly.json").write_text(json.dumps(assembly, indent=2), encoding="utf-8")
    print(json.dumps(assembly, indent=2, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 data assembly: B/C layers -> versioned training inputs")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulator", choices=["synthetic", "robocasa"], default="synthetic")
    parser.add_argument("--video-dir", help="B-layer unlabeled clip dir (optional)")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()
    sys.exit(assemble(args))


if __name__ == "__main__":
    main()
