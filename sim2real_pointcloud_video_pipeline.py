"""Generate traceable soft-object RGB-D / point-cloud / video-pair episodes.

The pipeline has two backends:
  1. SyntheticDeformableBackend: deterministic contract test only. It creates a
     simple deforming depth/RGB field; it is NOT a physical soft-body simulator.
  2. External simulator adapter contract: implement IsaacLabAdapter or
     RoboCasaAdapter to return RGB-D, camera calibration, actions and contacts
     from an approved simulator job manifest.

Every artifact carries seed, simulator/backend version, action trace, camera
calibration, physics parameters and source provenance. Generated data must pass
quality gates before training.

Run a contract demonstration:
  python3 sim2real_pointcloud_video_pipeline.py --demo --output /tmp/soft_pairs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # camera_to_robot as row-major 4x4 homogeneous transform
    camera_to_robot: tuple[float, ...]


@dataclass(frozen=True)
class EpisodeRequest:
    job_id: str
    parent_episode_id: str
    split: str
    seed: int
    simulator: str
    simulator_version: str
    object_class: str
    task_template: str
    physics: Dict[str, float]
    visual_randomization: Dict[str, Any]
    sensor_randomization: Dict[str, Any]
    action_perturbation: Dict[str, Any]
    source_references: Dict[str, str]


@dataclass
class RawSimulationEpisode:
    rgb: np.ndarray              # [T, H, W, 3], uint8
    depth_m: np.ndarray          # [T, H, W], float32, 0 for invalid pixels
    actions: np.ndarray          # [T, A], float32; actual executed simulated actions
    proprio: np.ndarray          # [T, P], float32
    contacts: np.ndarray         # [T, C], float32
    timestamps_ns: np.ndarray    # [T], int64
    calibration: CameraCalibration
    backend_metadata: Dict[str, Any]


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    reasons: tuple[str, ...]
    valid_depth_fraction: float
    finite_point_fraction: float
    temporal_alignment_ok: bool
    contacts_present: bool


class SimulatorAdapter(ABC):
    @abstractmethod
    def rollout(self, request: EpisodeRequest) -> RawSimulationEpisode:
        """Run a simulator episode and return synchronized, actual executed traces."""


class SyntheticDeformableBackend(SimulatorAdapter):
    """Contract-test backend, not a physics-valid soft-body simulation.

    It produces a bounded, seedable deforming Gaussian height field so the data
    writer and point-cloud/video alignment can be tested without Isaac Lab. Do
    not use its output for AC-VJEPA training or physical claims.
    """

    def __init__(self, frames: int = 12, height: int = 72, width: int = 96, action_dim: int = 20):
        self.frames = frames
        self.height = height
        self.width = width
        self.action_dim = action_dim

    def rollout(self, request: EpisodeRequest) -> RawSimulationEpisode:
        rng = np.random.default_rng(request.seed)
        h, w, t = self.height, self.width, self.frames
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rgb = np.zeros((t, h, w, 3), dtype=np.uint8)
        depth = np.zeros((t, h, w), dtype=np.float32)
        actions = rng.normal(0.0, 0.03, size=(t, self.action_dim)).astype(np.float32)
        proprio = rng.normal(0.0, 0.05, size=(t, 12)).astype(np.float32)
        contacts = np.zeros((t, 4), dtype=np.float32)

        stiffness = float(request.physics.get("young_modulus", 500.0))
        damping = float(request.physics.get("damping", 0.1))
        deformation_scale = 8.0 / max(math.sqrt(stiffness), 1.0)
        background = 1.1
        for frame in range(t):
            phase = frame / max(t - 1, 1)
            cx = w * (0.35 + 0.25 * phase)
            cy = h * (0.50 + 0.10 * math.sin(phase * math.pi))
            sigma_x = w * (0.14 + deformation_scale * phase * 0.01)
            sigma_y = h * (0.18 + deformation_scale * (1.0 - damping) * phase * 0.01)
            blob = np.exp(-(((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
            # Positive depths only; fabricated field is explicitly tagged test-only.
            frame_depth = background - 0.20 * blob + rng.normal(0.0, 0.002, size=(h, w))
            invalid = rng.random((h, w)) < 0.02
            frame_depth[invalid] = 0.0
            depth[frame] = frame_depth.astype(np.float32)
            color = np.stack((0.35 + 0.45 * blob, 0.20 + 0.25 * blob, 0.15 + 0.15 * blob), axis=-1)
            rgb[frame] = np.clip(color * 255.0, 0, 255).astype(np.uint8)
            contacts[frame, 0] = float(phase > 0.25)
            contacts[frame, 1] = float(phase > 0.70)

        calibration = CameraCalibration(
            width=w,
            height=h,
            fx=90.0,
            fy=90.0,
            cx=(w - 1) / 2.0,
            cy=(h - 1) / 2.0,
            camera_to_robot=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        return RawSimulationEpisode(
            rgb=rgb,
            depth_m=depth,
            actions=actions,
            proprio=proprio,
            contacts=contacts,
            timestamps_ns=np.arange(t, dtype=np.int64) * 33_333_333,
            calibration=calibration,
            backend_metadata={
                "backend": "SyntheticDeformableBackend",
                "physical_validity": "contract_test_only",
                "seed": request.seed,
            },
        )


class IsaacLabAdapter(SimulatorAdapter):
    """Integration contract for a real Isaac Lab backend.

    Implement this adapter in the Isaac Lab environment, where reset(), action
    replay, RGB-D camera capture, contact sensors and deformable-object physics
    are available. Keeping the adapter out of this portable script avoids claiming
    that a generic NumPy function simulates soft-body contact.
    """

    def rollout(self, request: EpisodeRequest) -> RawSimulationEpisode:
        raise NotImplementedError(
            "Implement in the pinned Isaac Lab/Isaac Sim runtime: reset approved scene/physics, "
            "replay bounded actions, record RGB-D/proprio/contact/time/calibration, and return RawSimulationEpisode."
        )


def backproject_depth(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    calibration: CameraCalibration,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one aligned RGB-D frame to fixed-size robot-frame point cloud.

    Returns xyz [N,3], rgb [N,3], valid_mask [N]. Invalid/padded locations are
    zeroed and identified by the mask, preserving a fixed shape for training.
    """
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    ys, xs = np.nonzero(valid)
    xyz_out = np.zeros((max_points, 3), dtype=np.float32)
    rgb_out = np.zeros((max_points, 3), dtype=np.uint8)
    mask = np.zeros((max_points,), dtype=bool)
    if len(xs) == 0:
        return xyz_out, rgb_out, mask

    if len(xs) > max_points:
        selected = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys = xs[selected], ys[selected]
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - calibration.cx) * z / calibration.fx
    y = (ys.astype(np.float32) - calibration.cy) * z / calibration.fy
    cam_xyz = np.stack((x, y, z, np.ones_like(z)), axis=1)
    transform = np.asarray(calibration.camera_to_robot, dtype=np.float32).reshape(4, 4)
    robot_xyz = (transform @ cam_xyz.T).T[:, :3]
    count = robot_xyz.shape[0]
    xyz_out[:count] = robot_xyz
    rgb_out[:count] = rgb[ys, xs]
    mask[:count] = True
    return xyz_out, rgb_out, mask


def quality_gate(raw: RawSimulationEpisode, point_mask: np.ndarray) -> QualityReport:
    reasons: List[str] = []
    if raw.rgb.ndim != 4 or raw.depth_m.ndim != 3 or raw.rgb.shape[:3] != raw.depth_m.shape:
        reasons.append("rgb_depth_shape_mismatch")
    if raw.actions.shape[0] != raw.rgb.shape[0] or raw.proprio.shape[0] != raw.rgb.shape[0]:
        reasons.append("action_or_proprio_not_time_aligned")
    temporal_alignment_ok = bool(np.all(np.diff(raw.timestamps_ns) > 0)) and raw.timestamps_ns.shape[0] == raw.rgb.shape[0]
    if not temporal_alignment_ok:
        reasons.append("timestamps_not_monotonic_or_aligned")
    valid_depth_fraction = float(np.mean(np.isfinite(raw.depth_m) & (raw.depth_m > 0.0)))
    finite_point_fraction = float(np.mean(np.isfinite(point_mask.astype(np.float32))))
    if valid_depth_fraction < 0.05:
        reasons.append("insufficient_valid_depth")
    contacts_present = bool(np.any(np.isfinite(raw.contacts))) and raw.contacts.shape[0] == raw.rgb.shape[0]
    if not contacts_present:
        reasons.append("missing_contact_trace")
    return QualityReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        valid_depth_fraction=valid_depth_fraction,
        finite_point_fraction=finite_point_fraction,
        temporal_alignment_ok=temporal_alignment_ok,
        contacts_present=contacts_present,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeWriter:
    def __init__(self, output_root: Path, max_points: int = 1024):
        self.output_root = output_root
        self.max_points = max_points

    def write(self, request: EpisodeRequest, raw: RawSimulationEpisode) -> tuple[Path, QualityReport]:
        rng = np.random.default_rng(request.seed + 10_000)
        t = raw.rgb.shape[0]
        point_xyz = np.zeros((t, self.max_points, 3), dtype=np.float32)
        point_rgb = np.zeros((t, self.max_points, 3), dtype=np.uint8)
        point_mask = np.zeros((t, self.max_points), dtype=bool)
        for index in range(t):
            xyz, color, mask = backproject_depth(
                raw.depth_m[index], raw.rgb[index], raw.calibration, self.max_points, rng
            )
            point_xyz[index], point_rgb[index], point_mask[index] = xyz, color, mask

        report = quality_gate(raw, point_mask)
        episode_dir = self.output_root / request.split / request.job_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        npz_path = episode_dir / "episode.npz"
        np.savez_compressed(
            npz_path,
            rgb_video=raw.rgb,
            depth_video_m=raw.depth_m,
            point_cloud_xyz=point_xyz,
            point_cloud_rgb=point_rgb,
            point_mask=point_mask,
            executed_actions=raw.actions,
            proprio=raw.proprio,
            contacts=raw.contacts,
            timestamps_ns=raw.timestamps_ns,
        )
        metadata = {
            "request": asdict(request),
            "camera_calibration": asdict(raw.calibration),
            "backend_metadata": raw.backend_metadata,
            "quality_report": asdict(report),
            "artifact": "episode.npz",
            "artifact_sha256": file_sha256(npz_path),
            "point_cloud_frame": "robot_base",
            "contract_version": "soft-grasp-rgbd-pointcloud-v1",
        }
        (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return episode_dir, report


def parse_request(raw: Mapping[str, Any]) -> EpisodeRequest:
    return EpisodeRequest(
        job_id=str(raw["job_id"]),
        parent_episode_id=str(raw["parent_episode_id"]),
        split=str(raw["split"]),
        seed=int(raw["seed"]),
        simulator=str(raw["simulator"]),
        simulator_version=str(raw["simulator_version"]),
        object_class=str(raw["object_class"]),
        task_template=str(raw["task_template"]),
        physics=dict(raw["physics"]),
        visual_randomization=dict(raw.get("visual_randomization", {})),
        sensor_randomization=dict(raw.get("sensor_randomization", {})),
        action_perturbation=dict(raw.get("action_perturbation", {})),
        source_references=dict(raw.get("source_references", {})),
    )


def load_jobs(path: Path) -> List[EpisodeRequest]:
    return [parse_request(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def demo_job() -> EpisodeRequest:
    return EpisodeRequest(
        job_id=str(uuid4()),
        parent_episode_id="approved-soft-cloth-hard-example-001",
        split="train",
        seed=2027,
        simulator="contract_test_backend",
        simulator_version="synthetic-deformable-v1",
        object_class="cloth",
        task_template="soft_object_reposition_low_speed",
        physics={"young_modulus": 800.0, "damping": 0.10, "friction": 0.35},
        visual_randomization={"lighting_profile": "warm_indoor"},
        sensor_randomization={"rgb_noise_profile": "nominal"},
        action_perturbation={"source": "approved_action_replay"},
        source_references={"prior_id": "cloth-prior-approved-v2"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RGB-D / point-cloud pairs from approved Sim-to-Real jobs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-manifest")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--max-points", type=int, default=1024)
    args = parser.parse_args()
    if bool(args.job_manifest) == bool(args.demo):
        raise SystemExit("provide exactly one of --job-manifest or --demo")

    # Use SyntheticDeformableBackend only to validate the artifact contract.
    # In a real run, select a pinned IsaacLabAdapter/RoboCasaAdapter based on the
    # job's simulator field and record the actual simulator runtime version.
    adapter: SimulatorAdapter = SyntheticDeformableBackend()
    jobs = [demo_job()] if args.demo else load_jobs(Path(args.job_manifest))
    writer = EpisodeWriter(Path(args.output), max_points=args.max_points)
    manifest: List[Dict[str, Any]] = []
    for job in jobs:
        raw = adapter.rollout(job)
        path, report = writer.write(job, raw)
        manifest.append({"job_id": job.job_id, "path": str(path), "accepted": report.accepted, "reasons": report.reasons})
    manifest_path = Path(args.output) / "generated_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(item) for item in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(manifest), "manifest": str(manifest_path), "results": manifest}, indent=2))


if __name__ == "__main__":
    main()
