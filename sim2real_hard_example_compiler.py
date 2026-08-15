"""Compile approved AC-VJEPA hard examples into traceable Sim-to-Real job manifests.

This module intentionally does not claim to simulate deformable physics by itself.
It turns audited real-world difficulty signals into versioned, seedable job
specifications that an Isaac Lab, RoboCasa, MuJoCo, or other approved simulator
adapter can execute. The simulator adapter must report its physics/backend version
and preserve all generated data provenance.

Run a manifest-only demonstration:
    python3 sim2real_hard_example_compiler.py --demo --output /tmp/sim_jobs.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from uuid import uuid4


@dataclass(frozen=True)
class CandidateEpisode:
    """Audited edge-side record; raw media is referenced, not embedded."""

    episode_id: str
    source_model_version: str
    action_schema_version: str
    robot_calibration_id: str
    object_class: str
    task_template: str
    uncertainty_windows: int
    mean_uncertainty: float
    mean_prediction_residual: float
    novelty_score: float
    safety_events: tuple[str, ...]
    rgb_reference: str
    proprio_reference: str
    action_reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftObjectPrior:
    """Bounds are supplied by measured material characterization or expert review.

    Do not invent broad physical ranges from a single uncertain episode. The
    compiler only samples from versioned, approved priors.
    """

    prior_id: str
    density_range: tuple[float, float]
    friction_range: tuple[float, float]
    young_modulus_range: tuple[float, float]
    poisson_ratio_range: tuple[float, float]
    damping_range: tuple[float, float]
    contact_stiffness_range: tuple[float, float]
    geometry_variants: tuple[str, ...]
    approval_ticket: str


@dataclass(frozen=True)
class SimJob:
    job_id: str
    parent_episode_id: str
    split: str  # train | validation | stress
    seed: int
    simulator: str
    simulator_version: str
    robot_asset_id: str
    task_template: str
    object_class: str
    object_geometry_variant: str
    physics: Dict[str, float]
    visual_randomization: Dict[str, Any]
    sensor_randomization: Dict[str, Any]
    action_perturbation: Dict[str, Any]
    source_references: Dict[str, str]
    data_contract_version: str
    quality_requirements: Dict[str, Any]


class HardExampleSelector:
    """Filters candidate episodes before any active data-generation job is created."""

    def __init__(
        self,
        *,
        min_uncertainty_windows: int,
        min_novelty_score: float,
        max_safety_events_for_automation: int = 0,
    ):
        self.min_uncertainty_windows = min_uncertainty_windows
        self.min_novelty_score = min_novelty_score
        self.max_safety_events_for_automation = max_safety_events_for_automation

    def eligible(self, episode: CandidateEpisode) -> bool:
        # Any contact/people/hardware event must route to human review rather than
        # automatic synthetic expansion. Episode metadata can tag benign events
        # separately if a safety board explicitly permits it.
        return (
            episode.uncertainty_windows >= self.min_uncertainty_windows
            and episode.novelty_score >= self.min_novelty_score
            and len(episode.safety_events) <= self.max_safety_events_for_automation
        )


class ScenarioCompiler:
    def __init__(
        self,
        *,
        simulator: str,
        simulator_version: str,
        robot_asset_id: str,
        data_contract_version: str = "ac-vjepa-episode-v1",
    ):
        self.simulator = simulator
        self.simulator_version = simulator_version
        self.robot_asset_id = robot_asset_id
        self.data_contract_version = data_contract_version

    @staticmethod
    def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
        low, high = bounds
        if low > high:
            raise ValueError(f"invalid prior bounds: {bounds}")
        return rng.uniform(low, high)

    def compile(
        self,
        episode: CandidateEpisode,
        prior: SoftObjectPrior,
        *,
        seeds: Sequence[int],
        split: str,
    ) -> List[SimJob]:
        if not prior.approval_ticket:
            raise ValueError("soft-object prior requires an approval ticket")
        if split not in {"train", "validation", "stress"}:
            raise ValueError("split must be train, validation, or stress")

        jobs: List[SimJob] = []
        for seed in seeds:
            rng = random.Random(seed)
            physics = {
                "density": self._uniform(rng, prior.density_range),
                "friction": self._uniform(rng, prior.friction_range),
                "young_modulus": self._uniform(rng, prior.young_modulus_range),
                "poisson_ratio": self._uniform(rng, prior.poisson_ratio_range),
                "damping": self._uniform(rng, prior.damping_range),
                "contact_stiffness": self._uniform(rng, prior.contact_stiffness_range),
            }
            jobs.append(
                SimJob(
                    job_id=str(uuid4()),
                    parent_episode_id=episode.episode_id,
                    split=split,
                    seed=seed,
                    simulator=self.simulator,
                    simulator_version=self.simulator_version,
                    robot_asset_id=self.robot_asset_id,
                    task_template=episode.task_template,
                    object_class=episode.object_class,
                    object_geometry_variant=rng.choice(prior.geometry_variants),
                    physics=physics,
                    visual_randomization={
                        "camera_extrinsics_jitter": "approved_small_range",
                        "lighting_profile": rng.choice(["day", "warm_indoor", "low_contrast"]),
                        "material_texture_seed": rng.randint(0, 2**31 - 1),
                    },
                    sensor_randomization={
                        "rgb_noise_profile": rng.choice(["nominal", "low_light", "motion_blur"]),
                        "proprio_delay_profile": rng.choice(["nominal", "bounded_delay"]),
                    },
                    action_perturbation={
                        # Perturb only an approved, replayed action trace; this is
                        # not unconstrained exploration or arbitrary new control.
                        "source": "executed_action_replay",
                        "bounded_pose_noise": "approved_small_range",
                        "bounded_timing_jitter": "approved_small_range",
                    },
                    source_references={
                        "rgb": episode.rgb_reference,
                        "proprio": episode.proprio_reference,
                        "actions": episode.action_reference,
                        "prior_id": prior.prior_id,
                        "prior_approval_ticket": prior.approval_ticket,
                    },
                    data_contract_version=self.data_contract_version,
                    quality_requirements={
                        "must_reach_terminal_state": True,
                        "must_record_actual_actions": True,
                        "must_record_contacts": True,
                        "must_record_simulator_seed": True,
                        "must_pass_schema_validation": True,
                    },
                )
            )
        return jobs


def write_jsonl(records: Iterable[SimJob], destination: str) -> None:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def demo(output: str) -> None:
    episode = CandidateEpisode(
        episode_id="hard-soft-cloth-0001",
        source_model_version="ac-vjepa-edge-1.2.0",
        action_schema_version="action-block-v1",
        robot_calibration_id="kitchen-bimanual-cal-v3",
        object_class="cloth",
        task_template="fold_or_reposition_soft_object",
        uncertainty_windows=8,
        mean_uncertainty=2.3,
        mean_prediction_residual=1.1,
        novelty_score=0.88,
        safety_events=(),
        rgb_reference="s3://approved-edge-data/hard-soft-cloth-0001/rgb",
        proprio_reference="s3://approved-edge-data/hard-soft-cloth-0001/proprio",
        action_reference="s3://approved-edge-data/hard-soft-cloth-0001/actions",
    )
    prior = SoftObjectPrior(
        prior_id="cloth-prior-v2",
        density_range=(0.8, 1.2),
        friction_range=(0.2, 0.6),
        young_modulus_range=(100.0, 2000.0),
        poisson_ratio_range=(0.2, 0.45),
        damping_range=(0.01, 0.20),
        contact_stiffness_range=(100.0, 1000.0),
        geometry_variants=("towel_small", "towel_large", "cloth_folded"),
        approval_ticket="SIM-PRIOR-APPROVED-DEMO",
    )
    selector = HardExampleSelector(min_uncertainty_windows=3, min_novelty_score=0.5)
    if not selector.eligible(episode):
        raise RuntimeError("demo episode unexpectedly failed eligibility")
    compiler = ScenarioCompiler(
        simulator="isaac_lab_adapter",
        simulator_version="replace-with-pinned-runtime-version",
        robot_asset_id="bimanual_robot_usd",
    )
    jobs = compiler.compile(episode, prior, seeds=[101, 102, 103], split="train")
    write_jsonl(jobs, output)
    print(json.dumps({"jobs": len(jobs), "manifest": output}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile approved hard examples into simulator job manifests")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.demo:
        raise SystemExit("This reference script intentionally exposes only --demo; integrate real approved episode/prior stores explicitly.")
    demo(args.output)


if __name__ == "__main__":
    main()
