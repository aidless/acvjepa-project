"""Shadow-mode and canary-release gate for AC-VJEPA research deployments.

This module decides only whether a candidate model may remain in SHADOW, enter a
limited CANARY stage, be promoted, paused, or rolled back. It deliberately does
NOT talk to robot drivers, trajectory queues, or safety controllers.

Run:
    python3 shadow_canary_gate.py
"""
from __future__ import annotations

import hashlib
import statistics
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4


class ReleaseStage(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    CANARY_LOW_RISK = "canary_low_risk"
    CANARY_EXPANDED = "canary_expanded"
    PRODUCTION = "production"
    PAUSED = "paused"
    ROLLED_BACK = "rolled_back"


class GateDecision(str, Enum):
    AWAIT_EVIDENCE = "await_evidence"
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class ModelOutcome:
    model_id: str
    correlation_id: str
    state_id: str
    latency_ms: float
    invalid_output: bool
    requested_local_hold: bool
    mean_uncertainty: float
    recommended_candidate_id: Optional[str]


@dataclass(frozen=True)
class ObservedOutcome:
    """Post-hoc facts collected without giving the model any control authority."""

    correlation_id: str
    hard_safety_event: bool
    environment_risk_observed: bool
    task_event_success: Optional[bool]


@dataclass(frozen=True)
class ShadowRecord:
    robot_id: str
    shift_id: str
    task_template: str
    baseline: ModelOutcome
    candidate: ModelOutcome
    observed: ObservedOutcome
    recorded_ns: int


@dataclass(frozen=True)
class GatePolicy:
    min_shadow_records: int = 100
    min_canary_records: int = 30
    max_candidate_p99_ms: float = 30.0
    max_latency_regression_ratio: float = 1.10
    max_invalid_rate: float = 0.0
    max_hold_miss_rate: float = 0.0
    max_hard_safety_events: int = 0
    # Candidate may be at most this much worse in recorded task-event success.
    max_task_success_regression: float = 0.02
    # First canary scope: identity-stable lower percentage and pre-registered skills.
    low_risk_canary_percent: float = 0.05
    expanded_canary_percent: float = 0.20
    allowed_low_risk_tasks: tuple[str, ...] = (
        "soft_object_observe",
        "soft_object_regrasp_low_force",
        "soft_object_reposition_low_speed",
    )


@dataclass(frozen=True)
class GateReport:
    stage: ReleaseStage
    decision: GateDecision
    reason: str
    records: int
    candidate_p99_ms: Optional[float]
    baseline_p99_ms: Optional[float]
    candidate_invalid_rate: float
    candidate_hold_miss_rate: float
    hard_safety_events: int
    candidate_success_rate: Optional[float]
    baseline_success_rate: Optional[float]


@dataclass(frozen=True)
class RouteDecision:
    """A model-selection decision only; the SafetyKernel still owns control admission."""

    selected_model_id: str
    stage: ReleaseStage
    reason: str


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


class IdentityStableRouter:
    """Routes a stable robot/shift/task identity to baseline or canary.

    Hash routing is intentionally deterministic. It prevents the same robot from
    flipping models every request, which would make outcome attribution and safe
    rollback difficult. It does not grant model control authority.
    """

    def __init__(self, baseline_model_id: str, candidate_model_id: str, policy: GatePolicy, salt: str):
        self.baseline_model_id = baseline_model_id
        self.candidate_model_id = candidate_model_id
        self.policy = policy
        self.salt = salt

    def choose(
        self,
        *,
        stage: ReleaseStage,
        robot_id: str,
        shift_id: str,
        task_template: str,
    ) -> RouteDecision:
        if stage in (ReleaseStage.CANDIDATE, ReleaseStage.SHADOW, ReleaseStage.PAUSED, ReleaseStage.ROLLED_BACK):
            return RouteDecision(self.baseline_model_id, stage, "candidate_has_no_control_authority")
        if task_template not in self.policy.allowed_low_risk_tasks:
            return RouteDecision(self.baseline_model_id, stage, "task_not_in_canary_allowlist")
        if stage == ReleaseStage.CANARY_LOW_RISK:
            percentage = self.policy.low_risk_canary_percent
        elif stage == ReleaseStage.CANARY_EXPANDED:
            percentage = self.policy.expanded_canary_percent
        elif stage == ReleaseStage.PRODUCTION:
            percentage = 1.0
        else:
            return RouteDecision(self.baseline_model_id, stage, "unknown_release_stage")

        identity = f"{self.salt}|{robot_id}|{shift_id}|{task_template}".encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
        if bucket < percentage:
            return RouteDecision(self.candidate_model_id, stage, "identity_in_canary_cohort")
        return RouteDecision(self.baseline_model_id, stage, "identity_outside_canary_cohort")


class ShadowCanaryGate:
    def __init__(
        self,
        *,
        baseline_model_id: str,
        candidate_model_id: str,
        policy: GatePolicy,
        max_records: int = 10_000,
    ):
        self.baseline_model_id = baseline_model_id
        self.candidate_model_id = candidate_model_id
        self.policy = policy
        self.stage = ReleaseStage.CANDIDATE
        self.records: Deque[ShadowRecord] = deque(maxlen=max_records)
        self.release_id = str(uuid4())

    def enter_shadow(self) -> None:
        if self.stage != ReleaseStage.CANDIDATE:
            raise RuntimeError(f"shadow entry allowed only from candidate, current={self.stage}")
        self.stage = ReleaseStage.SHADOW

    def record(self, item: ShadowRecord) -> None:
        if item.baseline.model_id != self.baseline_model_id:
            raise ValueError("baseline model id mismatch")
        if item.candidate.model_id != self.candidate_model_id:
            raise ValueError("candidate model id mismatch")
        if item.baseline.correlation_id != item.candidate.correlation_id:
            raise ValueError("baseline/candidate correlation id mismatch")
        if item.observed.correlation_id != item.candidate.correlation_id:
            raise ValueError("observed outcome correlation id mismatch")
        self.records.append(item)

    def _summary(self) -> Dict[str, object]:
        records = list(self.records)
        candidate_latencies = [record.candidate.latency_ms for record in records]
        baseline_latencies = [record.baseline.latency_ms for record in records]
        invalid_count = sum(record.candidate.invalid_output for record in records)
        # A missed hold is the safety-relevant false negative: environment risk was
        # observed after the window, but candidate did not request conservative hold.
        hold_misses = sum(
            record.observed.environment_risk_observed and not record.candidate.requested_local_hold
            for record in records
        )
        hard_events = sum(record.observed.hard_safety_event for record in records)
        candidate_success = [
            record.observed.task_event_success
            for record in records
            if record.observed.task_event_success is not None
        ]
        # Baseline/candidate task success must come from comparable shadow/canary
        # episodes. In pure shadow both see the same real action; here the field is
        # a placeholder for a paired counterfactual/low-risk canary evaluator.
        return {
            "records": len(records),
            "candidate_p99": percentile(candidate_latencies, 0.99),
            "baseline_p99": percentile(baseline_latencies, 0.99),
            "invalid_rate": rate(invalid_count, len(records)),
            "hold_miss_rate": rate(hold_misses, len(records)),
            "hard_events": hard_events,
            "candidate_success_rate": (
                rate(sum(value is True for value in candidate_success), len(candidate_success))
                if candidate_success
                else None
            ),
            # The caller can populate paired baseline results via an external
            # evaluator; do not infer baseline success from candidate-only actions.
            "baseline_success_rate": None,
        }

    def evaluate(self) -> GateReport:
        summary = self._summary()
        records = int(summary["records"])
        required = self.policy.min_shadow_records if self.stage == ReleaseStage.SHADOW else self.policy.min_canary_records
        candidate_p99 = summary["candidate_p99"]
        baseline_p99 = summary["baseline_p99"]
        invalid_rate = float(summary["invalid_rate"])
        hold_miss_rate = float(summary["hold_miss_rate"])
        hard_events = int(summary["hard_events"])

        def report(decision: GateDecision, reason: str) -> GateReport:
            return GateReport(
                stage=self.stage,
                decision=decision,
                reason=reason,
                records=records,
                candidate_p99_ms=candidate_p99 if isinstance(candidate_p99, float) else None,
                baseline_p99_ms=baseline_p99 if isinstance(baseline_p99, float) else None,
                candidate_invalid_rate=invalid_rate,
                candidate_hold_miss_rate=hold_miss_rate,
                hard_safety_events=hard_events,
                candidate_success_rate=summary["candidate_success_rate"],
                baseline_success_rate=summary["baseline_success_rate"],
            )

        # Hard conditions bypass sample-size requirements and immediately block
        # expansion. A physical safety event requires an external incident review.
        if hard_events > self.policy.max_hard_safety_events:
            return report(GateDecision.ROLLBACK, "hard_safety_event_observed")
        if invalid_rate > self.policy.max_invalid_rate:
            return report(GateDecision.ROLLBACK, "candidate_invalid_output_rate_exceeded")
        if hold_miss_rate > self.policy.max_hold_miss_rate:
            return report(GateDecision.ROLLBACK, "candidate_missed_conservative_hold")
        if candidate_p99 is not None and candidate_p99 > self.policy.max_candidate_p99_ms:
            return report(GateDecision.ROLLBACK, "candidate_p99_deadline_exceeded")
        if (
            candidate_p99 is not None
            and baseline_p99 is not None
            and candidate_p99 > baseline_p99 * self.policy.max_latency_regression_ratio
        ):
            return report(GateDecision.ROLLBACK, "candidate_latency_regression_exceeded")
        if records < required:
            return report(GateDecision.AWAIT_EVIDENCE, "insufficient_coverage_or_samples")
        return report(GateDecision.PROMOTE, "all_automatic_gates_passed")

    def apply(self, report: GateReport, authorized_by_release_policy: bool) -> ReleaseStage:
        """Apply a release transition. Human/policy authorization is explicit.

        Automatic rollback/hold are allowed. Promotion requires an explicit release
        policy authorization and never bypasses independent robot safety checks.
        """
        if report.decision == GateDecision.ROLLBACK:
            self.stage = ReleaseStage.ROLLED_BACK
            return self.stage
        if report.decision in (GateDecision.AWAIT_EVIDENCE, GateDecision.HOLD):
            self.stage = ReleaseStage.PAUSED if report.decision == GateDecision.HOLD else self.stage
            return self.stage
        if not authorized_by_release_policy:
            self.stage = ReleaseStage.PAUSED
            return self.stage

        transitions = {
            ReleaseStage.SHADOW: ReleaseStage.CANARY_LOW_RISK,
            ReleaseStage.CANARY_LOW_RISK: ReleaseStage.CANARY_EXPANDED,
            ReleaseStage.CANARY_EXPANDED: ReleaseStage.PRODUCTION,
        }
        if self.stage not in transitions:
            raise RuntimeError(f"cannot promote from stage {self.stage}")
        self.stage = transitions[self.stage]
        return self.stage


def make_demo_record(index: int, *, candidate_latency_ms: float = 10.5) -> ShadowRecord:
    correlation_id = f"demo-{index:03d}"
    baseline = ModelOutcome(
        model_id="baseline-v1",
        correlation_id=correlation_id,
        state_id=f"state-{index:03d}",
        latency_ms=10.0,
        invalid_output=False,
        requested_local_hold=False,
        mean_uncertainty=0.3,
        recommended_candidate_id="candidate-a",
    )
    candidate = ModelOutcome(
        model_id="candidate-v2",
        correlation_id=correlation_id,
        state_id=f"state-{index:03d}",
        latency_ms=candidate_latency_ms,
        invalid_output=False,
        requested_local_hold=False,
        mean_uncertainty=0.28,
        recommended_candidate_id="candidate-a",
    )
    observed = ObservedOutcome(
        correlation_id=correlation_id,
        hard_safety_event=False,
        environment_risk_observed=False,
        task_event_success=True,
    )
    return ShadowRecord(
        robot_id="robot-01",
        shift_id="shift-demo",
        task_template="soft_object_observe",
        baseline=baseline,
        candidate=candidate,
        observed=observed,
        recorded_ns=time.monotonic_ns(),
    )


def main() -> None:
    policy = GatePolicy(min_shadow_records=5, min_canary_records=3)
    gate = ShadowCanaryGate(
        baseline_model_id="baseline-v1", candidate_model_id="candidate-v2", policy=policy
    )
    gate.enter_shadow()
    for index in range(5):
        gate.record(make_demo_record(index))
    shadow_report = gate.evaluate()
    assert shadow_report.decision == GateDecision.PROMOTE, shadow_report
    assert gate.apply(shadow_report, authorized_by_release_policy=True) == ReleaseStage.CANARY_LOW_RISK

    router = IdentityStableRouter("baseline-v1", "candidate-v2", policy, salt="release-demo")
    route = router.choose(
        stage=gate.stage,
        robot_id="robot-01",
        shift_id="shift-demo",
        task_template="soft_object_observe",
    )
    print(
        {
            "shadow_report": shadow_report,
            "new_stage": gate.stage.value,
            "route": route,
            "note": "RouteDecision must still pass through an independent SafetyKernel.",
        }
    )


if __name__ == "__main__":
    main()
