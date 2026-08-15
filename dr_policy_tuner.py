"""Turn shadow RCA hypotheses into auditable, bounded domain-randomization proposals.

This module NEVER expands approved numeric ranges. It changes only sampling
weights, scenario quotas and curriculum emphasis inside an approved configuration.
Any requested range expansion is represented as a review-required proposal rather
than an executable update.

Run:
    python3 dr_policy_tuner.py
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple

from shadow_degradation_rca import RCAReport, RootCause, Severity, ShadowDegradationAnalyzer, demo_records


@dataclass(frozen=True)
class ClosedRange:
    approved_low: float
    approved_high: float
    active_low: float
    active_high: float

    def validate(self) -> None:
        if not (self.approved_low <= self.active_low <= self.active_high <= self.approved_high):
            raise ValueError("active range must remain inside approved range")


@dataclass(frozen=True)
class ApprovedDRConfig:
    config_version: str
    approval_ticket: str
    visual_ranges: Dict[str, ClosedRange]
    physics_ranges: Dict[str, ClosedRange]
    action_ranges: Dict[str, ClosedRange]
    scenario_weights: Dict[str, float]
    # The config has no SDK objects; an Isaac Lab adapter can convert its fields to
    # versioned reset-event parameters after human/policy approval.


@dataclass(frozen=True)
class DRProposal:
    parent_config_version: str
    proposal_version: str
    approval_ticket: str
    based_on_causes: Tuple[str, ...]
    scenario_weight_updates: Dict[str, float]
    visual_focus: Dict[str, str]
    physics_focus: Dict[str, str]
    action_focus: Dict[str, str]
    requires_range_expansion_review: bool
    rationale: Tuple[str, ...]
    ci_cd_action: str


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("at least one scenario weight must be positive")
    return {key: max(0.0, value) / total for key, value in weights.items()}


class RootCauseDrivenDRPolicy:
    """Conservative mapping from evidence class to in-bounds sampling changes."""

    def propose(self, config: ApprovedDRConfig, report: RCAReport) -> DRProposal:
        for collection in (config.visual_ranges, config.physics_ranges, config.action_ranges):
            for item in collection.values():
                item.validate()
        if not config.approval_ticket:
            raise ValueError("approved DR config must carry an approval ticket")

        weights = dict(config.scenario_weights)
        visual_focus: Dict[str, str] = {}
        physics_focus: Dict[str, str] = {}
        action_focus: Dict[str, str] = {}
        rationale: List[str] = []
        causes: List[str] = []
        requires_review = False

        cause_set = {hypothesis.root_cause for hypothesis in report.hypotheses}
        blocking = any(hypothesis.severity == Severity.BLOCKING for hypothesis in report.hypotheses)

        if RootCause.VISUAL_DOMAIN_SHIFT in cause_set:
            causes.append(RootCause.VISUAL_DOMAIN_SHIFT.value)
            for profile in ("low_light", "warm_indoor", "camera_pose_jitter", "partial_occlusion"):
                weights[profile] = weights.get(profile, 0.0) + 1.0
            visual_focus.update(
                {
                    "lighting": "increase sampling of approved exposure/temperature profiles; keep numeric ranges unchanged",
                    "camera_pose": "increase reset-event sampling inside approved extrinsic jitter bounds",
                    "appearance": "increase approved texture/material and sensor-noise profile coverage",
                }
            )
            rationale.append("Visual cohorts show candidate-specific residual concentration; emphasize approved appearance/camera variants.")

        if RootCause.SENSOR_OR_CALIBRATION in cause_set:
            causes.append(RootCause.SENSOR_OR_CALIBRATION.value)
            # Do not train around an unresolved calibration failure. This only
            # schedules a separate robustness diagnostic profile, not a release fix.
            weights["sensor_contract_diagnostic"] = weights.get("sensor_contract_diagnostic", 0.0) + 0.5
            visual_focus["sensor_contract"] = "add bounded delay/noise diagnostics, but require sensor incident closure before model promotion"
            rationale.append("Sensor or timestamp evidence exists; isolate the data contract before treating it as domain drift.")

        if RootCause.SOFT_PHYSICS_GAP in cause_set:
            causes.append(RootCause.SOFT_PHYSICS_GAP.value)
            for profile in ("soft_contact", "soft_geometry_initial_state", "soft_material_prior", "low_speed_replay"):
                weights[profile] = weights.get(profile, 0.0) + 1.0
            physics_focus.update(
                {
                    "young_modulus": "reweight samples across the existing approved material range; do not expand it",
                    "damping_and_friction": "increase strata near residual-heavy prior bins, preserving approved bounds",
                    "contact_stiffness": "increase approved contact variants and record contact traces for every episode",
                    "geometry": "increase approved folded/wrinkled initial-state variants",
                }
            )
            action_focus["replay"] = "prioritize low-speed, low-force executed-action replay with approved bounded perturbations"
            rationale.append("Soft-object cohorts have elevated residuals; enrich approved material/contact/geometry coverage around audited hard examples.")

        if RootCause.UNCERTAINTY_MISCALIBRATION in cause_set:
            causes.append(RootCause.UNCERTAINTY_MISCALIBRATION.value)
            weights["uncertainty_stress_holdout"] = weights.get("uncertainty_stress_holdout", 0.0) + 1.0
            action_focus["data_split"] = "generate stress episodes for calibration evaluation; keep them isolated from training by default"
            rationale.append("Uncertainty evidence is unreliable; create held-out stress coverage before increasing autonomy or changing thresholds.")

        if RootCause.EDGE_RUNTIME in cause_set or RootCause.ENGINE_OR_QUANTIZATION in cause_set:
            causes.extend(
                cause.value
                for cause in (RootCause.EDGE_RUNTIME, RootCause.ENGINE_OR_QUANTIZATION)
                if cause in cause_set
            )
            rationale.append("Runtime/engine degradation is not solved by broader physics randomization; keep release paused and repair deployment first.")

        if not causes:
            rationale.append("No actionable visual/physics cause found; retain current approved randomization and collect more stratified evidence.")

        # Blocked deployment means this is only a candidate configuration; the
        # scheduler must not automatically enqueue release-grade data from it.
        ci_cd_action = (
            "create_dr_candidate_config_and_require_approval"
            if not blocking
            else "create_diagnostic_dr_config_but_keep_release_paused"
        )
        return DRProposal(
            parent_config_version=config.config_version,
            proposal_version=f"{config.config_version}-rca",
            approval_ticket=config.approval_ticket,
            based_on_causes=tuple(sorted(set(causes))),
            scenario_weight_updates=normalize_weights(weights),
            visual_focus=visual_focus,
            physics_focus=physics_focus,
            action_focus=action_focus,
            requires_range_expansion_review=requires_review,
            rationale=tuple(rationale),
            ci_cd_action=ci_cd_action,
        )


def demo_config() -> ApprovedDRConfig:
    return ApprovedDRConfig(
        config_version="soft-grasp-dr-v4",
        approval_ticket="DR-APPROVED-042",
        visual_ranges={
            "exposure": ClosedRange(-3.0, 2.0, -2.0, 1.0),
            "camera_x": ClosedRange(-0.03, 0.03, -0.02, 0.02),
            "camera_yaw": ClosedRange(-0.08, 0.08, -0.05, 0.05),
        },
        physics_ranges={
            "young_modulus": ClosedRange(100.0, 2000.0, 200.0, 1500.0),
            "damping": ClosedRange(0.01, 0.25, 0.03, 0.18),
            "friction": ClosedRange(0.15, 0.70, 0.20, 0.60),
            "contact_stiffness": ClosedRange(100.0, 1500.0, 200.0, 1200.0),
        },
        action_ranges={
            "pose_noise": ClosedRange(0.0, 0.02, 0.0, 0.01),
            "timing_jitter": ClosedRange(0.0, 0.10, 0.0, 0.05),
        },
        scenario_weights={"nominal": 1.0, "low_light": 0.2, "soft_contact": 0.5},
    )


def main() -> None:
    report = ShadowDegradationAnalyzer().analyze(demo_records())
    proposal = RootCauseDrivenDRPolicy().propose(demo_config(), report)
    print(json.dumps(asdict(proposal), indent=2))


if __name__ == "__main__":
    main()
