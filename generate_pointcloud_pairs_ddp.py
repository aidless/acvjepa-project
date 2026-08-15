"""Distributed generation and verified object-store sync for Sim-to-Real point-cloud/video pairs.

This script uses torch.distributed only for deterministic job sharding and barriers;
it is data parallel generation, not gradient training. It supports:
  - deterministic manifest sharding: sorted jobs[index % world_size == rank]
  - per-episode quality gate and SHA-256 provenance
  - local `file://` object-store emulation with atomic commits
  - S3-compatible `s3://` uploads through a preconfigured `aws` CLI, using a
    staging prefix and final commit manifest written last

It does NOT provide cloud credentials, invoke real robot hardware, or claim that
SyntheticDeformableBackend is physically valid training data. Use a pinned Isaac
Lab/RoboCasa adapter for real simulation.

Examples:
  # Contract test, two local workers and local object-store emulation
  torchrun --standalone --nproc_per_node=2 generate_pointcloud_pairs_ddp.py \
    --manifest demo_sim_jobs.jsonl --output /tmp/soft_pairs \
    --remote-uri file:///tmp/object_store --backend contract

  # Real multi-node invocation (requires shared output root and approved credentials)
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    generate_pointcloud_pairs_ddp.py --manifest approved_jobs.jsonl \
    --output /shared/soft_pairs --remote-uri s3://approved-bucket/soft-pairs \
    --backend isaac_lab
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import unquote, urlparse
from uuid import uuid4

import torch
import torch.distributed as dist

from sim2real_pointcloud_video_pipeline import (
    EpisodeRequest,
    EpisodeWriter,
    IsaacLabAdapter,
    SimulatorAdapter,
    SyntheticDeformableBackend,
    load_jobs,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    distributed: bool


@dataclass(frozen=True)
class EpisodeCommit:
    job_id: str
    parent_episode_id: str
    split: str
    rank: int
    local_path: str
    accepted: bool
    reasons: tuple[str, ...]
    metadata_sha256: str
    episode_sha256: str
    remote_commit_uri: Optional[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def init_distributed(backend: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        return DistributedContext(rank=0, world_size=1, distributed=False)
    selected = backend
    if backend == "auto":
        selected = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=selected, init_method="env://")
    return DistributedContext(rank=rank, world_size=world_size, distributed=True)


def finalize_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def shard_jobs(jobs: Sequence[EpisodeRequest], ctx: DistributedContext) -> List[EpisodeRequest]:
    ordered = sorted(jobs, key=lambda item: item.job_id)
    return [job for position, job in enumerate(ordered) if position % ctx.world_size == ctx.rank]


class ObjectStoreSynchronizer:
    """Uploads episode artifacts first and makes a commit manifest visible last.

    Downstream training readers must enumerate only `commits/*.json`; they should
    never treat objects in `.staging/` as training-ready data.
    """

    def __init__(self, remote_uri: Optional[str], release_id: str):
        self.remote_uri = remote_uri.rstrip("/") if remote_uri else None
        self.release_id = release_id

    @staticmethod
    def _read_metadata(episode_dir: Path) -> Dict:
        return json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))

    def upload_episode(self, episode_dir: Path, request: EpisodeRequest) -> Optional[str]:
        if self.remote_uri is None:
            return None
        metadata = self._read_metadata(episode_dir)
        if not metadata["quality_report"]["accepted"]:
            raise RuntimeError(f"refusing to upload rejected episode {request.job_id}")
        required = [episode_dir / "episode.npz", episode_dir / "metadata.json"]
        if any(not path.exists() for path in required):
            raise RuntimeError(f"episode artifacts incomplete: {episode_dir}")
        if self.remote_uri.startswith("file://"):
            return self._upload_file_store(episode_dir, request, metadata)
        if self.remote_uri.startswith("s3://"):
            return self._upload_s3(episode_dir, request, metadata)
        raise ValueError("remote URI must use file:// or s3://")

    def _commit_payload(self, episode_dir: Path, request: EpisodeRequest, metadata: Dict, artifact_prefix: str) -> Dict:
        return {
            "commit_version": "sim2real-object-commit-v1",
            "release_id": self.release_id,
            "job_id": request.job_id,
            "parent_episode_id": request.parent_episode_id,
            "split": request.split,
            "artifact_prefix": artifact_prefix,
            "episode_sha256": sha256(episode_dir / "episode.npz"),
            "metadata_sha256": sha256(episode_dir / "metadata.json"),
            "quality_report": metadata["quality_report"],
            "created_ns": time.time_ns(),
        }

    def _local_uri_root(self) -> Path:
        parsed = urlparse(self.remote_uri)
        path = unquote(parsed.path)
        # Windows: file:///F:/... -> /F:/... ; strip the leading slash so Path
        # does not interpret it as rooted on the current drive.
        if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path)

    def _upload_file_store(self, episode_dir: Path, request: EpisodeRequest, metadata: Dict) -> str:
        root = self._local_uri_root()
        staging = root / ".staging" / self.release_id / request.job_id
        final = root / "episodes" / request.split / request.job_id
        commit_dir = root / "commits" / self.release_id
        staging.parent.mkdir(parents=True, exist_ok=True)
        commit_dir.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(episode_dir, staging)
        # Re-check copy before making episode visible.
        if sha256(staging / "episode.npz") != sha256(episode_dir / "episode.npz"):
            raise RuntimeError("local object-store episode checksum mismatch")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
        payload = self._commit_payload(episode_dir, request, metadata, str(final))
        temporary_commit = commit_dir / f".{request.job_id}.tmp"
        final_commit = commit_dir / f"{request.job_id}.json"
        temporary_commit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary_commit, final_commit)
        return f"file://{final_commit}"

    @staticmethod
    def _run_aws(args: List[str]) -> None:
        result = subprocess.run(["aws", *args], text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"aws CLI failure: {result.stderr.strip()}")

    def _upload_s3(self, episode_dir: Path, request: EpisodeRequest, metadata: Dict) -> str:
        # S3 lacks a portable rename. Therefore uploads are written under staging,
        # validated by recorded hashes, and only the final small commit manifest
        # marks them visible to downstream consumers.
        prefix = f"{self.remote_uri}/.staging/{self.release_id}/{request.job_id}"
        for filename in ("episode.npz", "metadata.json"):
            self._run_aws(["s3", "cp", str(episode_dir / filename), f"{prefix}/{filename}"])
        payload = self._commit_payload(episode_dir, request, metadata, prefix)
        with tempfile.TemporaryDirectory() as directory:
            local_commit = Path(directory) / f"{request.job_id}.json"
            local_commit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary_uri = f"{self.remote_uri}/.committing/{self.release_id}/{request.job_id}.json"
            final_uri = f"{self.remote_uri}/commits/{self.release_id}/{request.job_id}.json"
            self._run_aws(["s3", "cp", str(local_commit), temporary_uri])
            # Last write is the only training-visible commit object. Object-store
            # consumers must validate the hashes in it before reading artifacts.
            self._run_aws(["s3", "cp", temporary_uri, final_uri])
        return final_uri

    def upload_dataset_commit(self, commit_path: Path) -> Optional[str]:
        if self.remote_uri is None:
            return None
        if self.remote_uri.startswith("file://"):
            root = self._local_uri_root()
            target_dir = root / "dataset_commits"
            target_dir.mkdir(parents=True, exist_ok=True)
            final = target_dir / f"{self.release_id}.json"
            temp = target_dir / f".{self.release_id}.tmp"
            shutil.copy2(commit_path, temp)
            os.replace(temp, final)
            return f"file://{final}"
        if self.remote_uri.startswith("s3://"):
            target = f"{self.remote_uri}/dataset_commits/{self.release_id}.json"
            self._run_aws(["s3", "cp", str(commit_path), target])
            return target
        raise ValueError("remote URI must use file:// or s3://")


def adapter_for(name: str) -> SimulatorAdapter:
    if name == "contract":
        return SyntheticDeformableBackend()
    if name == "isaac_lab":
        return IsaacLabAdapter()
    raise ValueError("backend must be contract or isaac_lab")


def write_rank_report(path: Path, commits: Iterable[EpisodeCommit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(item)) for item in commits) + "\n", encoding="utf-8")


def merge_rank_reports(report_dir: Path, world_size: int, release_id: str) -> Path:
    commits: List[Dict] = []
    for rank in range(world_size):
        report = report_dir / f"rank-{rank}.jsonl"
        if not report.exists():
            raise RuntimeError(f"missing rank report: {report}")
        commits.extend(json.loads(line) for line in report.read_text(encoding="utf-8").splitlines() if line)
    if len({item["job_id"] for item in commits}) != len(commits):
        raise RuntimeError("duplicate job IDs in distributed generation reports")
    manifest = {
        "dataset_commit_version": "sim2real-ddp-dataset-v1",
        "release_id": release_id,
        "created_ns": time.time_ns(),
        "episodes": sorted(commits, key=lambda item: item["job_id"]),
        "accepted_count": sum(item["accepted"] for item in commits),
        "rejected_count": sum(not item["accepted"] for item in commits),
    }
    target = report_dir / f"dataset-{release_id}.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP Sim-to-Real pointcloud/video generation with verified sync")
    parser.add_argument("--manifest", required=True, help="JSONL SimJob manifest generated by approved compiler")
    parser.add_argument("--output", required=True, help="shared local output directory")
    parser.add_argument("--remote-uri", help="optional file:// or s3:// prefix")
    parser.add_argument("--release-id", default=f"release-{uuid4()}")
    parser.add_argument("--backend", choices=("contract", "isaac_lab"), default="contract")
    parser.add_argument("--dist-backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--max-points", type=int, default=1024)
    args = parser.parse_args()

    ctx = init_distributed(args.dist_backend)
    try:
        jobs = load_jobs(Path(args.manifest))
        owned = shard_jobs(jobs, ctx)
        output_root = Path(args.output)
        report_dir = output_root / "rank_reports" / args.release_id
        writer = EpisodeWriter(output_root, max_points=args.max_points)
        synchronizer = ObjectStoreSynchronizer(args.remote_uri, args.release_id)
        adapter = adapter_for(args.backend)
        commits: List[EpisodeCommit] = []
        for job in owned:
            raw = adapter.rollout(job)
            episode_dir, quality = writer.write(job, raw)
            remote_commit = synchronizer.upload_episode(episode_dir, job) if quality.accepted else None
            commits.append(
                EpisodeCommit(
                    job_id=job.job_id,
                    parent_episode_id=job.parent_episode_id,
                    split=job.split,
                    rank=ctx.rank,
                    local_path=str(episode_dir),
                    accepted=quality.accepted,
                    reasons=quality.reasons,
                    metadata_sha256=sha256(episode_dir / "metadata.json"),
                    episode_sha256=sha256(episode_dir / "episode.npz"),
                    remote_commit_uri=remote_commit,
                )
            )
        write_rank_report(report_dir / f"rank-{ctx.rank}.jsonl", commits)
        if ctx.distributed:
            dist.barrier()
        if ctx.rank == 0:
            dataset_commit = merge_rank_reports(report_dir, ctx.world_size, args.release_id)
            remote_dataset_commit = synchronizer.upload_dataset_commit(dataset_commit)
            print(
                json.dumps(
                    {
                        "release_id": args.release_id,
                        "world_size": ctx.world_size,
                        "dataset_commit": str(dataset_commit),
                        "remote_dataset_commit": remote_dataset_commit,
                    },
                    indent=2,
                )
            )
        if ctx.distributed:
            dist.barrier()
    finally:
        finalize_distributed(ctx)


if __name__ == "__main__":
    main()
