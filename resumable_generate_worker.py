"""Lease-based resumable worker for Sim-to-Real pointcloud/video generation.

Bootstrap once on durable shared storage, then start one or more workers (or use
one worker per torchrun rank). Node/process loss leaves leases to expire; a later
worker reclaims unfinished work. Completed jobs are identified by immutable job
keys and verified artifact/metadata hashes.

Examples:
  python3 resumable_generate_worker.py --bootstrap --manifest approved_jobs.jsonl \
      --ledger /shared/ledger.sqlite
  torchrun --standalone --nproc_per_node=4 resumable_generate_worker.py --run \
      --ledger /shared/ledger.sqlite --output /shared/pairs \
      --remote-uri file:///shared/object_store --backend contract
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict

from generate_pointcloud_pairs_ddp import ObjectStoreSynchronizer, adapter_for, sha256
from resumable_simjob_ledger import LeaseLedger, retry_with_backoff
from sim2real_pointcloud_video_pipeline import EpisodeWriter, EpisodeRequest, load_jobs, parse_request


def stable_job_key(request: EpisodeRequest) -> str:
    payload = {
        "job_id": request.job_id,
        "simulator": request.simulator,
        "simulator_version": request.simulator_version,
        "physics": request.physics,
        "visual_randomization": request.visual_randomization,
        "sensor_randomization": request.sensor_randomization,
        "action_perturbation": request.action_perturbation,
        "data_contract": "soft-grasp-rgbd-pointcloud-v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def request_payload(request: EpisodeRequest) -> Dict[str, Any]:
    # asdict avoided to keep payload schema explicit and durable.
    return {
        "job_id": request.job_id,
        "parent_episode_id": request.parent_episode_id,
        "split": request.split,
        "seed": request.seed,
        "simulator": request.simulator,
        "simulator_version": request.simulator_version,
        "object_class": request.object_class,
        "task_template": request.task_template,
        "physics": request.physics,
        "visual_randomization": request.visual_randomization,
        "sensor_randomization": request.sensor_randomization,
        "action_perturbation": request.action_perturbation,
        "source_references": request.source_references,
    }


def bootstrap(manifest: str, ledger_path: str) -> None:
    ledger = LeaseLedger(ledger_path)
    try:
        jobs = load_jobs(Path(manifest))
        for job in jobs:
            ledger.register(stable_job_key(job), request_payload(job))
        print(json.dumps({"registered": len(jobs), "summary": ledger.summary()}, indent=2))
    finally:
        ledger.conn.close()


def run_worker(args: argparse.Namespace) -> None:
    worker_id = args.worker_id or f"{socket.gethostname()}-pid{os.getpid()}-rank{os.environ.get('RANK', '0')}"
    ledger = LeaseLedger(args.ledger)
    writer = EpisodeWriter(Path(args.output), max_points=args.max_points)
    adapter = adapter_for(args.backend)
    synchronizer = ObjectStoreSynchronizer(args.remote_uri, args.release_id)
    completed = 0
    try:
        while completed < args.max_jobs:
            lease = ledger.acquire(worker_id, args.lease_seconds)
            if lease is None:
                break
            request = parse_request(lease.payload)
            try:
                # Production simulator adapters should invoke lease.heartbeat at
                # safe rollout checkpoints. This reference heartbeats before/after
                # the blocking rollout and uses a conservative lease duration.
                ledger.heartbeat(lease, args.lease_seconds)
                raw = adapter.rollout(request)
                lease = ledger.heartbeat(lease, args.lease_seconds)
                episode_dir, quality = writer.write(request, raw)
                if not quality.accepted:
                    ledger.retry(lease, f"quality_rejected:{','.join(quality.reasons)}", args.max_attempts)
                    continue

                def upload() -> str:
                    uri = synchronizer.upload_episode(episode_dir, request)
                    if uri is None:
                        raise RuntimeError("resumable completion requires remote URI/commit")
                    return uri

                remote_commit = retry_with_backoff(upload, attempts=args.upload_attempts, base_seconds=args.backoff_seconds)
                # Local source artifacts are re-hashed after upload; consumers will
                # also validate hashes recorded in the remote commit manifest.
                ledger.complete(
                    lease,
                    artifact_sha256=sha256(episode_dir / "episode.npz"),
                    metadata_sha256=sha256(episode_dir / "metadata.json"),
                    remote_commit_uri=remote_commit,
                )
                completed += 1
            except Exception as exc:
                ledger.retry(lease, f"{type(exc).__name__}:{exc}", args.max_attempts)
        print(json.dumps({"worker_id": worker_id, "completed": completed, "summary": ledger.summary()}, indent=2))
    finally:
        ledger.conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable lease-based SimJob worker")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output")
    parser.add_argument("--remote-uri")
    parser.add_argument("--release-id", default="resumable-soft-grasp")
    parser.add_argument("--backend", choices=("contract", "isaac_lab"), default="contract")
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--upload-attempts", type=int, default=5)
    parser.add_argument("--backoff-seconds", type=float, default=0.5)
    parser.add_argument("--max-jobs", type=int, default=1_000_000)
    parser.add_argument("--max-points", type=int, default=1024)
    args = parser.parse_args()
    if args.bootstrap == args.run:
        raise SystemExit("select exactly one of --bootstrap or --run")
    if args.bootstrap:
        if not args.manifest:
            raise SystemExit("--bootstrap requires --manifest")
        bootstrap(args.manifest, args.ledger)
    else:
        if not (args.output and args.remote_uri):
            raise SystemExit("--run requires --output and --remote-uri")
        run_worker(args)


if __name__ == "__main__":
    main()
