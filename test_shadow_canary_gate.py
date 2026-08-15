"""Unit tests for shadow_canary_gate.py; no hardware or robot SDK is involved."""
from __future__ import annotations

import unittest

from shadow_canary_gate import (
    GateDecision,
    GatePolicy,
    IdentityStableRouter,
    ModelOutcome,
    ObservedOutcome,
    ReleaseStage,
    ShadowCanaryGate,
    ShadowRecord,
    make_demo_record,
)


class ShadowCanaryGateTests(unittest.TestCase):
    def make_gate(self) -> ShadowCanaryGate:
        return ShadowCanaryGate(
            baseline_model_id="baseline-v1",
            candidate_model_id="candidate-v2",
            policy=GatePolicy(min_shadow_records=3, min_canary_records=2),
        )

    def test_shadow_promotion_requires_explicit_authorization(self) -> None:
        gate = self.make_gate()
        gate.enter_shadow()
        for index in range(3):
            gate.record(make_demo_record(index))
        report = gate.evaluate()
        self.assertEqual(report.decision, GateDecision.PROMOTE)
        self.assertEqual(gate.apply(report, authorized_by_release_policy=False), ReleaseStage.PAUSED)

    def test_hard_safety_event_forces_rollback(self) -> None:
        gate = self.make_gate()
        gate.enter_shadow()
        record = make_demo_record(0)
        risk_record = ShadowRecord(
            robot_id=record.robot_id,
            shift_id=record.shift_id,
            task_template=record.task_template,
            baseline=record.baseline,
            candidate=record.candidate,
            observed=ObservedOutcome(
                correlation_id=record.observed.correlation_id,
                hard_safety_event=True,
                environment_risk_observed=True,
                task_event_success=False,
            ),
            recorded_ns=record.recorded_ns,
        )
        gate.record(risk_record)
        report = gate.evaluate()
        self.assertEqual(report.decision, GateDecision.ROLLBACK)
        self.assertEqual(gate.apply(report, authorized_by_release_policy=True), ReleaseStage.ROLLED_BACK)

    def test_missed_conservative_hold_forces_rollback(self) -> None:
        gate = self.make_gate()
        gate.enter_shadow()
        record = make_demo_record(0)
        candidate = ModelOutcome(
            model_id=record.candidate.model_id,
            correlation_id=record.candidate.correlation_id,
            state_id=record.candidate.state_id,
            latency_ms=record.candidate.latency_ms,
            invalid_output=False,
            requested_local_hold=False,
            mean_uncertainty=record.candidate.mean_uncertainty,
            recommended_candidate_id=record.candidate.recommended_candidate_id,
        )
        gate.record(
            ShadowRecord(
                robot_id=record.robot_id,
                shift_id=record.shift_id,
                task_template=record.task_template,
                baseline=record.baseline,
                candidate=candidate,
                observed=ObservedOutcome(
                    correlation_id=record.observed.correlation_id,
                    hard_safety_event=False,
                    environment_risk_observed=True,
                    task_event_success=False,
                ),
                recorded_ns=record.recorded_ns,
            )
        )
        self.assertEqual(gate.evaluate().decision, GateDecision.ROLLBACK)

    def test_shadow_and_disallowed_task_always_route_to_baseline(self) -> None:
        policy = GatePolicy()
        router = IdentityStableRouter("baseline-v1", "candidate-v2", policy, salt="test")
        shadow = router.choose(
            stage=ReleaseStage.SHADOW,
            robot_id="r1",
            shift_id="s1",
            task_template="soft_object_observe",
        )
        self.assertEqual(shadow.selected_model_id, "baseline-v1")
        disallowed = router.choose(
            stage=ReleaseStage.CANARY_LOW_RISK,
            robot_id="r1",
            shift_id="s1",
            task_template="unapproved_high_force_skill",
        )
        self.assertEqual(disallowed.selected_model_id, "baseline-v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
