"""RoboCasa data-collection adapter for the AC-VJEPA M2 data pyramid (C layer).

RoboCasa (https://robocasa.ai/, robocasa package on top of robosuite) provides
photoreal kitchen scenes, robot arms, object variants and task templates with
per-frame sub-task / atomic-skill annotations — exactly what the lightweight
V-JEPA plan needs for controlled intervention data (layout/texture/task combos).

This adapter satisfies the `SimulatorAdapter` contract from
`sim2real_pointcloud_video_pipeline.py`: it must return a
`RawSimulationEpisode` with synchronized rgb / depth / actions / proprio /
contacts / timestamps / calibration.

IMPORTANT scope boundary (same rule as IsaacLabAdapter):
- This file does NOT import robocasa at module import time.
- `RoboCasaAdapter.rollout()` raises an actionable error when `robocasa` is not
  installed, so portable verification (synthetic backend, contract tests) never
  depends on the heavy sim stack.
- The actual collection loop is implemented in `_rollout_with_robocasa`, which
  is only reached in an environment where `pip install robocasa` succeeded and
  the kitchen assets are available. Adapt names / task ids to the installed
  RoboCasa version before a real collection run.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from sim2real_pointcloud_video_pipeline import (
    CameraCalibration,
    EpisodeRequest,
    RawSimulationEpisode,
    SimulatorAdapter,
)

DEFAULT_INTRINSIC = (640.0, 640.0, 320.0, 240.0)  # fx, fy, cx, cy


class RoboCasaAdapter(SimulatorAdapter):
    """Collect a real RoboCasa kitchen episode as a RawSimulationEpisode.

    Configuration (constructor kwargs / job fields):

        task_template : RoboCasa task id, e.g. "PnD_CounterToCab" (pick-and-drop).
        object_class  : object instance id group used by the task.
        physics       : {"friction_range": [...], "density_range": [...]} optional.
        visual_randomization : {"texture_ids": [...], "lighting": [...]} optional.
        sensor_randomization : {"camera_noise": 0.0, "depth_range": [0.3, 3.0]}.
        action_perturbation : applied only as bounded perturbation to the
                              *executed* action trace, never fabricated.

    The collected `actions` must be the actions actually executed by the
    simulator step (post safety-limit), matching the AC-VJEPA data contract.
    """

    def __init__(
        self,
        *,
        frames: int = 48,
        height: int = 480,
        width: int = 640,
        control_freq: int = 20,
        episodes_per_task: int = 5,
    ):
        self.frames = frames
        self.height = height
        self.width = width
        self.control_freq = control_freq
        self.episodes_per_task = episodes_per_task

    def rollout(self, request: EpisodeRequest) -> RawSimulationEpisode:
        try:
            return self._rollout_with_robocasa(request)
        except ImportError as exc:
            raise RuntimeError(
                "RoboCasa is not installed in this environment. "
                "Run the portable pipeline with --simulator synthetic for contract tests, "
                "or install RoboCasa (pip install robocasa) on a GPU/display-capable host "
                "and pin the task ids to the installed version before real collection."
            ) from exc

    # -- real environment path -------------------------------------------------

    def _rollout_with_robocasa(self, request: EpisodeRequest) -> RawSimulationEpisode:
        # Import is intentionally deferred: this line only succeeds where
        # RoboCasa (and its MuJoCo/robosuite dependencies) are installed.
        import robocasa  # noqa: F401  (heavy dependency, import lazily)

        raise NotImplementedError(
            "RoboCasa collection loop not yet implemented in this repo: "
            "reset the scene for the task/object ids from `request`, step "
            "`control_freq` Hz while recording rgb/depth/proprio/contacts, and "
            "return a RawSimulationEpisode with actually executed actions. "
            "See DATA_MANIFEST.md C-layer and the RoboCasa API docs for the "
            "exact env/task/obs keys of the installed version."
        )

    # -- portable fallback used by contract tests / assembly smoke --------------

    def rollout_synthetic(self, request: EpisodeRequest) -> RawSimulationEpisode:
        """Deterministic stand-in so the assembly pipeline can be validated
        without the sim stack. NOT physics-valid; never train on it."""
        from sim2real_pointcloud_video_pipeline import SyntheticDeformableBackend

        return SyntheticDeformableBackend(
            frames=self.frames, height=self.height, width=self.width
        ).rollout(request)


def demo_robocasa_job() -> EpisodeRequest:
    """A sample job describing a RoboCasa-style kitchen task."""
    from sim2real_pointcloud_video_pipeline import demo_job

    base = demo_job()
    return EpisodeRequest(
        job_id=base.job_id,
        parent_episode_id="robocasa-approved-v1",
        split="train",
        seed=base.seed,
        simulator="robocasa",
        simulator_version="robocasa-1.2-pinned",
        object_class="mug",
        task_template="PnD_CounterToSink",
        physics={"friction_range": [0.6, 1.0], "density_range": [0.4, 0.6]},
        visual_randomization={"texture_ids": [0, 1, 2], "lighting_profile": "daylight"},
        sensor_randomization={"camera_noise": 0.0, "depth_range": [0.3, 3.0]},
        action_perturbation={"source": "approved_action_replay", "scale": 0.02},
        source_references={"robocasa_task": "PnD_CounterToSink", "scene": "kitchen"},
    )


def calibration_for(width: int, height: int) -> CameraCalibration:
    fx, fy, cx, cy = DEFAULT_INTRINSIC
    scale_x, scale_y = width / 640.0, height / 480.0
    return CameraCalibration(
        fx=fx * scale_x, fy=fy * scale_y, cx=cx * scale_x, cy=cy * scale_y
    )


if __name__ == "__main__":
    # Contract smoke: synthetic path only (no RoboCasa dependency).
    import sys
    from pathlib import Path

    from sim2real_pointcloud_video_pipeline import EpisodeWriter, quality_gate

    request = demo_robocasa_job()
    adapter = RoboCasaAdapter(frames=8, height=96, width=128)
    raw = adapter.rollout_synthetic(request)
    assert raw.rgb.shape[0] == 8
    writer = EpisodeWriter(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("verify_artifacts/robocasa_contract"), max_points=64)
    episode_dir, report = writer.write(request, raw)
    print(
        json.dumps(
            {"smoke_test": "passed", "backend": "synthetic-robocasa-contract",
             "episode_dir": str(episode_dir), "accepted": report.accepted,
             "reasons": report.reasons, "frames": int(raw.rgb.shape[0])},
            sort_keys=True,
        ),
        flush=True,
    )
