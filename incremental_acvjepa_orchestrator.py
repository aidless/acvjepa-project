"""Re-entrant incremental AC-VJEPA workflow orchestrator.

The orchestrator consumes ONLY an immutable, verified dataset commit. It writes a
versioned run state after each stage so retries skip completed stages with matching
inputs. It can execute the existing trainer, but evaluation commands are explicit:
no candidate becomes a shadow/canary release merely because training completed.

Example:
  python3 incremental_acvjepa_orchestrator.py \
    --dataset-commit /shared/dataset-commit.json --work-root /shared/runs/run-001 \
    --parent-checkpoint /models/baseline.pt --prepare-only

  # After configuring trusted evaluation commands and GPU runner:
  python3 incremental_acvjepa_orchestrator.py ... --execute-training \
    --offline-eval-cmd 'python3 evaluate_offline.py {checkpoint}' \
    --sim-eval-cmd 'python3 evaluate_sim.py {checkpoint}' \
    --edge-eval-cmd 'python3 evaluate_edge.py {checkpoint}'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


STAGES = (
    "VALIDATE_DATASET_COMMIT",
    "PREPARE_TRAINING_WINDOWS",
    "TRAIN_CANDIDATE",
    "OFFLINE_EVALUATION",
    "SIM_EVALUATION",
    "EDGE_EVALUATION",
    "SHADOW_CANDIDATE_READY",
)


@dataclass(frozen=True)
class RunInputs:
    dataset_commit: str
    dataset_commit_sha256: str
    parent_checkpoint: str
    parent_checkpoint_sha256: str
    code_contract_version: str = "ac-vjepa-incremental-run-v1"


class RunState:
    def __init__(self, path: Path, inputs: RunInputs):
        self.path = path
        self.inputs = inputs
        if path.exists():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
            old = self.payload["inputs"]
            if old != inputs.__dict__:
                raise RuntimeError("run state input mismatch; create a new run directory rather than mixing provenance")
        else:
            self.payload = {"inputs": inputs.__dict__, "stages": {}, "created_ns": time.time_ns()}
            self.save()

    def done(self, stage: str) -> bool:
        return self.payload["stages"].get(stage, {}).get("status") == "COMPLETE"

    def complete(self, stage: str, outputs: Dict) -> None:
        self.payload["stages"][stage] = {"status": "COMPLETE", "completed_ns": time.time_ns(), "outputs": outputs}
        self.save()

    def fail(self, stage: str, message: str) -> None:
        self.payload["stages"][stage] = {"status": "FAILED", "failed_ns": time.time_ns(), "message": message}
        self.save()

    def output(self, stage: str) -> Dict:
        return self.payload["stages"][stage]["outputs"]

    def save(self) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
        temp.replace(self.path)


def run_command(command: str, replacements: Dict[str, str], cwd: Path) -> None:
    rendered = command.format(**replacements)
    result = subprocess.run(rendered, shell=True, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {rendered}")


def validate_dataset_commit(path: Path) -> Dict:
    commit = json.loads(path.read_text(encoding="utf-8"))
    records = commit.get("episodes", [])
    accepted = [record for record in records if record.get("accepted")]
    if not accepted:
        raise RuntimeError("dataset commit has no accepted episodes")
    if len({record["job_id"] for record in accepted}) != len(accepted):
        raise RuntimeError("dataset commit contains duplicate accepted job IDs")
    for record in accepted:
        episode_dir = Path(record["local_path"])
        npz = episode_dir / "episode.npz"
        metadata = episode_dir / "metadata.json"
        if not (npz.is_file() and metadata.is_file()):
            raise RuntimeError(f"local/cache artifact missing for job {record['job_id']}")
        if sha256(npz) != record["episode_sha256"] or sha256(metadata) != record["metadata_sha256"]:
            raise RuntimeError(f"artifact hash mismatch for job {record['job_id']}")
    return {"accepted_episodes": len(accepted), "dataset_commit_sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental AC-VJEPA pipeline from immutable dataset commits")
    parser.add_argument("--dataset-commit", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--offline-eval-cmd")
    parser.add_argument("--sim-eval-cmd")
    parser.add_argument("--edge-eval-cmd")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.prepare_only and args.execute_training:
        raise SystemExit("choose at most one of --prepare-only and --execute-training")

    dataset_commit = Path(args.dataset_commit).resolve()
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    if not parent_checkpoint.is_file():
        raise SystemExit("parent checkpoint missing")
    work = Path(args.work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs = RunInputs(
        dataset_commit=str(dataset_commit),
        dataset_commit_sha256=sha256(dataset_commit),
        parent_checkpoint=str(parent_checkpoint),
        parent_checkpoint_sha256=sha256(parent_checkpoint),
    )
    state = RunState(work / "run_state.json", inputs)

    try:
        if not state.done("VALIDATE_DATASET_COMMIT"):
            state.complete("VALIDATE_DATASET_COMMIT", validate_dataset_commit(dataset_commit))

        training_input = work / "training_input"
        if not state.done("PREPARE_TRAINING_WINDOWS"):
            converter = Path(__file__).with_name("dataset_commit_to_acvjepa_windows.py")
            run_command(
                f"{sys.executable} {converter} --dataset-commit {{dataset_commit}} --output {{training_input}}",
                {"dataset_commit": str(dataset_commit), "training_input": str(training_input)},
                work,
            )
            input_manifest = training_input / "training_input_manifest.json"
            state.complete("PREPARE_TRAINING_WINDOWS", {"training_input_manifest": str(input_manifest), "sha256": sha256(input_manifest)})

        if args.prepare_only:
            print(json.dumps({"status": "TRAINING_INPUT_READY", "run_state": str(state.path)}, indent=2))
            return
        if not args.execute_training:
            print(json.dumps({"status": "AWAITING_TRAINING_AUTHORIZATION", "run_state": str(state.path)}, indent=2))
            return

        trainer_out = work / "candidate_training"
        if not state.done("TRAIN_CANDIDATE"):
            training_manifest = state.output("PREPARE_TRAINING_WINDOWS")["training_input_manifest"]
            data = json.loads(Path(training_manifest).read_text(encoding="utf-8"))
            trainer = Path(__file__).with_name("train_ac_vjepa_ddp.py")
            # Parent model initialization is explicit and strict. Optimizer/scaler
            # state are not inherited, so this remains a new versioned incremental run.
            command = (
                f"{sys.executable} {trainer} --manifest {data['windows_manifest']} "
                f"--output {trainer_out} --epochs {args.epochs} --init-from {parent_checkpoint}"
            )
            run_command(command, {}, work)
            checkpoint = trainer_out / "last.pt"
            if not checkpoint.is_file():
                raise RuntimeError("trainer completed without last.pt")
            state.complete("TRAIN_CANDIDATE", {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint), "parent_checkpoint": str(parent_checkpoint)})

        replacements = {"checkpoint": state.output("TRAIN_CANDIDATE")["checkpoint"], "work_root": str(work)}
        for stage, command in (
            ("OFFLINE_EVALUATION", args.offline_eval_cmd),
            ("SIM_EVALUATION", args.sim_eval_cmd),
            ("EDGE_EVALUATION", args.edge_eval_cmd),
        ):
            if state.done(stage):
                continue
            if not command:
                raise RuntimeError(f"{stage} command is required before candidate may become shadow-ready")
            run_command(command, replacements, work)
            state.complete(stage, {"command": command.format(**replacements)})

        if not state.done("SHADOW_CANDIDATE_READY"):
            candidate = state.output("TRAIN_CANDIDATE")
            shadow_release = {
                "candidate_checkpoint": candidate,
                "dataset_commit_sha256": inputs.dataset_commit_sha256,
                "parent_checkpoint_sha256": inputs.parent_checkpoint_sha256,
                "release_mode": "SHADOW_ONLY",
                "control_authority": False,
                "required_next_gate": "shadow_canary_gate.py",
            }
            release_path = work / "shadow_candidate.json"
            release_path.write_text(json.dumps(shadow_release, indent=2), encoding="utf-8")
            state.complete("SHADOW_CANDIDATE_READY", {"shadow_release": str(release_path)})
        print(json.dumps({"status": "SHADOW_CANDIDATE_READY", "run_state": str(state.path)}, indent=2))
    except Exception as exc:
        active_stage = next((stage for stage in STAGES if not state.done(stage)), "UNKNOWN")
        state.fail(active_stage, f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
