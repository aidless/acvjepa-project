"""Evidence-based root-cause analysis for AC-VJEPA shadow degradation.

The analyzer produces ranked hypotheses and conservative next actions. It does
not claim causal certainty, modify model weights, relax safety thresholds, or
control a robot. It is designed to turn shadow telemetry into auditable CI/CD
issues such as 'rebuild TensorRT engine' or 'collect verified soft-object data'.

Run:
    python3 shadow_degradation_rca.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from statistics import mean
from typing import Dict, Iterable, List, Sequence


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class RootCause(str, Enum):
    EDGE_RUNTIME = "edge_runtime_latency_or_thermal_regression"
    ENGINE_OR_QUANTIZATION = "engine_quantization_or_schema_regression"
    SENSOR_OR_CALIBRATION = "sensor_timestamp_or_calibration_drift"
    VISUAL_DOMAIN_SHIFT = "visual_domain_shift"
    SOFT_PHYSICS_GAP = "soft_object_physics_or_contact_gap"
    UNCERTAINTY_MISCALIBRATION = "uncertainty_miscalibration"
    MODEL_CAPACITY_OR_TRAINING = "model_capacity_or_incremental_training_regression"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ShadowTelemetry:
    correlation_id: str
    model_version: str
    baseline_version: str
    object_class: str
    task_template: str
    camera_id: str
    calibration_id: str
    preprocess_version: str
    engine_version: str
    layout_id: str
    # Per-window values computed only after the corresponding actual observation
    # arrives. This makes residual/hold analysis independent of model self-report.
    baseline_latency_ms: float
    candidate_latency_ms: float
    baseline_residual: float
    candidate_residual: float
    candidate_uncertainty: float
    candidate_invalid_output: bool
    candidate_requested_hold: bool
    observed_risk: bool
    timestamp_skew_ms: float
    image_quality_score: float  # normalized 0..1; lower is worse
    thermal_guard_active: bool


@dataclass(frozen=True)
class Hypothesis:
    root_cause: RootCause
    severity: Severity
    confidence: float  # evidence score, not causal probability
    evidence: Dict[str, float]
    recommendation: str
    ci_cd_action: str


@dataclass(frozen=True)
class RCAReport:
    records: int
    hypotheses: List[Hypothesis]
    global_metrics: Dict[str, float]


@dataclass(frozen=True)
class RCAThresholds:
    min_records: int = 30
    latency_regression_ratio: float = 1.10
    residual_regression_ratio: float = 1.15
    hold_miss_rate_limit: float = 0.0
    invalid_rate_limit: float = 0.0
    max_timestamp_skew_ms: float = 10.0
    min_image_quality_score: float = 0.65
    # Candidate should have greater uncertainty on higher-residual slices.
    min_residual_uncertainty_ratio: float = 1.20


def safe_ratio(numerator: float, denominator: float) -> float:
    return float("inf") if denominator <= 1e-9 and numerator > 0 else numerator / max(denominator, 1e-9)


def fraction(items: Sequence[bool]) -> float:
    return 0.0 if not items else sum(items) / len(items)


def cohort_mean(records: Sequence[ShadowTelemetry], key: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[ShadowTelemetry]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, key))].append(record)
    return {
        cohort: {
            "count": float(len(values)),
            "candidate_residual": mean(x.candidate_residual for x in values),
            "baseline_residual": mean(x.baseline_residual for x in values),
            "candidate_uncertainty": mean(x.candidate_uncertainty for x in values),
            "latency": mean(x.candidate_latency_ms for x in values),
            "timestamp_skew": mean(abs(x.timestamp_skew_ms) for x in values),
            "image_quality": mean(x.image_quality_score for x in values),
        }
        for cohort, values in grouped.items()
    }


class ShadowDegradationAnalyzer:
    def __init__(self, thresholds: RCAThresholds = RCAThresholds()):
        self.thresholds = thresholds

    def analyze(self, records: Sequence[ShadowTelemetry]) -> RCAReport:
        if len(records) < self.thresholds.min_records:
            return RCAReport(
                records=len(records),
                hypotheses=[
                    Hypothesis(
                        root_cause=RootCause.INSUFFICIENT_EVIDENCE,
                        severity=Severity.INFO,
                        confidence=1.0,
                        evidence={"records": float(len(records)), "required": float(self.thresholds.min_records)},
                        recommendation="Keep the candidate in shadow mode and collect coverage across object, camera, layout and task cohorts.",
                        ci_cd_action="keep_shadow_collect_evidence",
                    )
                ],
                global_metrics={"records": float(len(records))},
            )

        candidate_residual = mean(x.candidate_residual for x in records)
        baseline_residual = mean(x.baseline_residual for x in records)
        candidate_latency = mean(x.candidate_latency_ms for x in records)
        baseline_latency = mean(x.baseline_latency_ms for x in records)
        invalid_rate = fraction([x.candidate_invalid_output for x in records])
        hold_miss_rate = fraction(
            [x.observed_risk and not x.candidate_requested_hold for x in records]
        )
        thermal_rate = fraction([x.thermal_guard_active for x in records])
        skew_rate = fraction(
            [abs(x.timestamp_skew_ms) > self.thresholds.max_timestamp_skew_ms for x in records]
        )
        low_quality_rate = fraction(
            [x.image_quality_score < self.thresholds.min_image_quality_score for x in records]
        )
        hypotheses: List[Hypothesis] = []

        latency_ratio = safe_ratio(candidate_latency, baseline_latency)
        if latency_ratio > self.thresholds.latency_regression_ratio or thermal_rate > 0.0:
            severity = Severity.BLOCKING if thermal_rate > 0.0 else Severity.WARNING
            hypotheses.append(
                Hypothesis(
                    root_cause=RootCause.EDGE_RUNTIME,
                    severity=severity,
                    confidence=min(1.0, max(latency_ratio - 1.0, thermal_rate) + 0.35),
                    evidence={
                        "candidate_latency_ms": candidate_latency,
                        "baseline_latency_ms": baseline_latency,
                        "latency_ratio": latency_ratio,
                        "thermal_guard_rate": thermal_rate,
                    },
                    recommendation=(
                        "Profile edge p99 by preprocess, TensorRT/ORT subgraph and queue stage; verify engine cache/version, "
                        "thermal steady state and candidate batch/shape. Keep the candidate in shadow or roll back canary."
                    ),
                    ci_cd_action="open_edge_performance_ticket_and_pause_expansion",
                )
            )

        if invalid_rate > self.thresholds.invalid_rate_limit:
            hypotheses.append(
                Hypothesis(
                    root_cause=RootCause.ENGINE_OR_QUANTIZATION,
                    severity=Severity.BLOCKING,
                    confidence=1.0,
                    evidence={"invalid_output_rate": invalid_rate},
                    recommendation=(
                        "Quarantine the candidate engine. Compare FP32, FP16 and INT8 outputs for the same recorded windows; "
                        "verify ONNX I/O schema, calibration data coverage and JetPack/TensorRT/ORT version compatibility."
                    ),
                    ci_cd_action="quarantine_engine_rebuild_and_numerical_diff",
                )
            )

        if skew_rate > 0.0 or low_quality_rate > 0.0:
            hypotheses.append(
                Hypothesis(
                    root_cause=RootCause.SENSOR_OR_CALIBRATION,
                    severity=Severity.BLOCKING if skew_rate > 0.05 else Severity.WARNING,
                    confidence=min(1.0, max(skew_rate, low_quality_rate) + 0.25),
                    evidence={"timestamp_skew_rate": skew_rate, "low_image_quality_rate": low_quality_rate},
                    recommendation=(
                        "Validate camera/proprio clock alignment, calibration and preprocessing. Do not treat degraded sensor data as "
                        "a new soft-physics phenomenon; freeze model promotion until a clean replay is available."
                    ),
                    ci_cd_action="open_sensor_contract_incident_and_pause_release",
                )
            )

        residual_ratio = safe_ratio(candidate_residual, baseline_residual)
        by_object = cohort_mean(records, "object_class")
        soft_cohorts = [
            item for name, item in by_object.items()
            if name in {"cloth", "fabric", "sponge", "soft_packaging", "deformable"}
        ]
        soft_residual = mean([item["candidate_residual"] for item in soft_cohorts]) if soft_cohorts else 0.0
        nonsoft = [item["candidate_residual"] for name, item in by_object.items() if item not in soft_cohorts]
        nonsoft_residual = mean(nonsoft) if nonsoft else candidate_residual
        if residual_ratio > self.thresholds.residual_regression_ratio and soft_residual > nonsoft_residual:
            hypotheses.append(
                Hypothesis(
                    root_cause=RootCause.SOFT_PHYSICS_GAP,
                    severity=Severity.WARNING,
                    confidence=min(1.0, (residual_ratio - 1.0) + 0.35),
                    evidence={
                        "candidate_to_baseline_residual_ratio": residual_ratio,
                        "soft_object_candidate_residual": soft_residual,
                        "other_object_candidate_residual": nonsoft_residual,
                    },
                    recommendation=(
                        "Create a reviewed hard-example set for the affected material/task cohorts; fit or approve soft-physics priors; "
                        "generate bounded Sim-to-Real counterfactuals; incrementally fine-tune the action predictor/adapter with replay."
                    ),
                    ci_cd_action="open_soft_physics_data_ticket_and_schedule_sim2real",
                )
            )

        high_error = [x for x in records if x.candidate_residual > candidate_residual]
        low_error = [x for x in records if x.candidate_residual <= candidate_residual]
        if high_error and low_error:
            uncertainty_ratio = safe_ratio(
                mean(x.candidate_uncertainty for x in high_error),
                mean(x.candidate_uncertainty for x in low_error),
            )
            if uncertainty_ratio < self.thresholds.min_residual_uncertainty_ratio or hold_miss_rate > self.thresholds.hold_miss_rate_limit:
                hypotheses.append(
                    Hypothesis(
                        root_cause=RootCause.UNCERTAINTY_MISCALIBRATION,
                        severity=Severity.BLOCKING if hold_miss_rate > 0.0 else Severity.WARNING,
                        confidence=min(1.0, (1.0 - uncertainty_ratio) + hold_miss_rate + 0.3),
                        evidence={
                            "high_to_low_error_uncertainty_ratio": uncertainty_ratio,
                            "hold_miss_rate": hold_miss_rate,
                        },
                        recommendation=(
                            "Recalibrate uncertainty and hold thresholds using a held-out stress set; add error–uncertainty ranking loss; "
                            "do not expand canary until conservative-hold recall is restored."
                        ),
                        ci_cd_action="block_release_run_uncertainty_calibration_suite",
                    )
                )

        by_camera = cohort_mean(records, "camera_id")
        worst_camera_ratio = max(
            (safe_ratio(value["candidate_residual"], value["baseline_residual"]) for value in by_camera.values()),
            default=1.0,
        )
        if residual_ratio > self.thresholds.residual_regression_ratio and worst_camera_ratio > residual_ratio * 1.2:
            hypotheses.append(
                Hypothesis(
                    root_cause=RootCause.VISUAL_DOMAIN_SHIFT,
                    severity=Severity.WARNING,
                    confidence=min(1.0, worst_camera_ratio - 1.0),
                    evidence={"global_residual_ratio": residual_ratio, "worst_camera_residual_ratio": worst_camera_ratio},
                    recommendation=(
                        "Inspect camera-specific preprocessing, exposure and viewpoint cohorts; add reviewed visual domain randomization "
                        "and camera calibration checks before changing world-model capacity."
                    ),
                    ci_cd_action="open_visual_domain_shift_ticket",
                )
            )

        if residual_ratio > self.thresholds.residual_regression_ratio and not hypotheses:
            hypotheses.append(
                Hypothesis(
                    root_cause=RootCause.MODEL_CAPACITY_OR_TRAINING,
                    severity=Severity.WARNING,
                    confidence=min(1.0, residual_ratio - 1.0 + 0.25),
                    evidence={"candidate_to_baseline_residual_ratio": residual_ratio},
                    recommendation=(
                        "Compare checkpoints, optimizer/EMA state, action normalization and training manifests; run ablations on frozen "
                        "validation and replay sets before changing model capacity or compression level."
                    ),
                    ci_cd_action="open_training_regression_ticket",
                )
            )

        if not hypotheses:
            hypotheses.append(
                Hypothesis(
                    root_cause=RootCause.INSUFFICIENT_EVIDENCE,
                    severity=Severity.INFO,
                    confidence=0.5,
                    evidence={"candidate_to_baseline_residual_ratio": residual_ratio},
                    recommendation="No dominant degradation pattern found; increase cohort coverage and inspect paired replay samples.",
                    ci_cd_action="keep_shadow_collect_targeted_evidence",
                )
            )

        hypotheses.sort(key=lambda item: (item.severity == Severity.BLOCKING, item.confidence), reverse=True)
        return RCAReport(
            records=len(records),
            hypotheses=hypotheses,
            global_metrics={
                "candidate_residual": candidate_residual,
                "baseline_residual": baseline_residual,
                "residual_ratio": residual_ratio,
                "candidate_latency_ms": candidate_latency,
                "baseline_latency_ms": baseline_latency,
                "latency_ratio": latency_ratio,
                "invalid_rate": invalid_rate,
                "hold_miss_rate": hold_miss_rate,
                "thermal_guard_rate": thermal_rate,
            },
        )


def demo_records() -> List[ShadowTelemetry]:
    records: List[ShadowTelemetry] = []
    for index in range(36):
        soft = index % 2 == 0
        records.append(
            ShadowTelemetry(
                correlation_id=f"demo-{index}",
                model_version="candidate-int8-v4",
                baseline_version="baseline-fp16-v3",
                object_class="cloth" if soft else "rigid_cup",
                task_template="soft_object_regrasp_low_force" if soft else "carry_rigid_object",
                camera_id="wrist_cam" if index % 3 else "head_cam",
                calibration_id="cal-v3",
                preprocess_version="pre-v2",
                engine_version="trt-int8-v4",
                layout_id="kitchen-a",
                baseline_latency_ms=10.0,
                candidate_latency_ms=16.0,
                baseline_residual=0.4,
                candidate_residual=1.2 if soft else 0.45,
                candidate_uncertainty=0.35 if soft else 0.30,
                candidate_invalid_output=False,
                candidate_requested_hold=False,
                observed_risk=soft and index % 6 == 0,
                timestamp_skew_ms=1.0,
                image_quality_score=0.9,
                thermal_guard_active=index % 8 == 0,
            )
        )
    return records


def main() -> None:
    report = ShadowDegradationAnalyzer().analyze(demo_records())
    print(json.dumps(asdict(report), default=str, indent=2))


if __name__ == "__main__":
    main()
