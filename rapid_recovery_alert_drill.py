"""Rapid elastic-recovery and safe alert-response drill harness.

This script is deliberately non-destructive. It does not call Alertmanager,
Grafana, Prometheus HTTP APIs, torchrun, Kubernetes, a scheduler, SSH, RDMA,
firewall, switch or cloud APIs. It demonstrates:

1. Deterministic staged re-admission with bounded waves, exponential backoff and
   jitter to protect rendezvous/checkpoint services during mass preemption.
2. Prometheus metric emission through DistributedTrainingMetrics.
3. A catalog of Grafana-ready PromQL expressions.
4. A fail-closed Alertmanager-webhook *handler model* that permits only:
   freeze-new-plans, mark-SUSPECT, capture-existing-evidence and notify/escalate.

Production deployment must put the handler behind authenticated Alertmanager
webhook delivery and independently enforce job/environment allowlists,
idempotency, rate limits and durable audit storage. The handler must never gain
network or cluster mutation privileges.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Sequence

from distributed_training_observability import DistributedTrainingMetrics, TrainingMetricLabels


class AlertRejected(RuntimeError):
    pass


class ControlAction(str, Enum):
    FREEZE_NEW_PLANS = "freeze_new_plans"
    MARK_SUSPECT = "mark_suspect"
    CAPTURE_EVIDENCE = "capture_existing_evidence"
    NOTIFY_OWNER = "notify_owner"


@dataclass(frozen=True)
class RecoveryPolicy:
    max_nodes_per_wave: int = 4
    base_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0
    jitter_ratio: float = 0.2
    checkpoint_load_concurrency_per_wave: int = 4

    def validate(self) -> None:
        if self.max_nodes_per_wave < 1 or self.checkpoint_load_concurrency_per_wave < 1:
            raise ValueError("wave/load concurrency must be positive")
        if not (0 < self.base_backoff_seconds <= self.max_backoff_seconds and 0 <= self.jitter_ratio <= 1):
            raise ValueError("invalid recovery backoff policy")


@dataclass(frozen=True)
class RecoveryAdmission:
    node_id: str
    wave: int
    earliest_join_seconds: float
    checkpoint_load_slot: int


class RecoveryWavePlanner:
    """Deterministic wave and jitter planner, without controlling workers."""

    def __init__(self, policy: RecoveryPolicy) -> None:
        policy.validate()
        self.policy = policy

    @staticmethod
    def _unit_interval(seed: str) -> float:
        # Stable pseudo-randomness prevents synchronized join storms while keeping
        # a drill reproducible from its experiment/run ID.
        return int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def plan(self, *, run_id: str, restart_count: int, node_ids: Sequence[str]) -> tuple[RecoveryAdmission, ...]:
        if not run_id or restart_count < 0 or not node_ids or len(node_ids) != len(set(node_ids)):
            raise ValueError("run ID, non-negative restart count and unique node IDs are required")
        backoff = min(self.policy.max_backoff_seconds, self.policy.base_backoff_seconds * (2**restart_count))
        admissions: list[RecoveryAdmission] = []
        for index, node_id in enumerate(sorted(node_ids)):
            wave = index // self.policy.max_nodes_per_wave
            jitter = (self._unit_interval(f"{run_id}:{restart_count}:{node_id}") * 2 - 1) * self.policy.jitter_ratio * backoff
            # Wave separation protects rendezvous and checkpoint stores. A node
            # still must validate committed checkpoint/topology/cursor before join.
            earliest = max(0.0, wave * backoff + jitter)
            admissions.append(
                RecoveryAdmission(
                    node_id=node_id,
                    wave=wave,
                    earliest_join_seconds=round(earliest, 3),
                    checkpoint_load_slot=index % self.policy.checkpoint_load_concurrency_per_wave,
                )
            )
        return tuple(admissions)


PROMQL: Mapping[str, str] = {
    "plan_consensus": "min(acvjepa_training_plan_digest_consensus{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\"})",
    "two_to_one_plan": "acvjepa_training_plan_micro_batches{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\"}",
    "allreduce_p95": "job:acvjepa_training_allreduce_p95_seconds:5m{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\"}",
    "cursor_stuck": "acvjepa_training_cursor_reservations{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\"} > 0 and on(cluster,job,environment) increase(acvjepa_training_checkpoint_commits_total[10m]) == 0",
    "rebuild_rate": "sum by(cluster,job,environment) (increase(acvjepa_training_rendezvous_rebuilds_total[15m]))",
    "recovery_p95": "job:acvjepa_training_recovery_p95_seconds:30m{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\"}",
    "state_restore": "min(acvjepa_training_state_alignment_verified{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\",component=~\"model|ema|optimizer|rng\"})",
    "fp8_metadata": "acvjepa_training_fp8_metadata_verified{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\"}",
    "abort_ratio": "job:acvjepa_training_update_abort_ratio:5m{cluster=\"$cluster\",job=\"$job\",environment=\"$environment\"}",
}


@dataclass(frozen=True)
class SafeAlert:
    fingerprint: str
    alertname: str
    status: str  # firing | resolved
    cluster: str
    job: str
    environment: str
    severity: str


def alerts_from_alertmanager_payload(payload: Mapping[str, object]) -> tuple[SafeAlert, ...]:
    """Parse the small, allowlisted subset required from an Alertmanager webhook.

    This is intentionally a payload parser rather than an HTTP server. A
    production webhook adapter must authenticate the sender and enforce its own
    replay/idempotency controls before calling this function.
    """

    status = str(payload.get("status", ""))
    raw_alerts = payload.get("alerts")
    if status not in {"firing", "resolved"} or not isinstance(raw_alerts, list):
        raise AlertRejected("invalid Alertmanager payload envelope")
    parsed: list[SafeAlert] = []
    for raw in raw_alerts:
        if not isinstance(raw, Mapping):
            raise AlertRejected("invalid alert entry")
        labels = raw.get("labels")
        if not isinstance(labels, Mapping):
            raise AlertRejected("alert labels are missing")
        required = {name: labels.get(name) for name in ("alertname", "cluster", "job", "environment", "severity")}
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise AlertRejected("alert is missing a required bounded label")
        fingerprint = raw.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            # Fallback ID is deterministic and contains no unbounded user data.
            fingerprint = hashlib.sha256(json.dumps(required, sort_keys=True).encode()).hexdigest()
        parsed.append(
            SafeAlert(
                fingerprint=fingerprint,
                alertname=str(required["alertname"]),
                status=status,
                cluster=str(required["cluster"]),
                job=str(required["job"]),
                environment=str(required["environment"]),
                severity=str(required["severity"]),
            )
        )
    return tuple(parsed)


class SafeAlertControlPlane:
    """Idempotent, allowlisted, non-destructive control plane used by a webhook adapter."""

    ALLOWED = {
        "ACVJEPAUpdatePlanConsensusLost": (ControlAction.FREEZE_NEW_PLANS, ControlAction.MARK_SUSPECT, ControlAction.CAPTURE_EVIDENCE, ControlAction.NOTIFY_OWNER),
        "ACVJEPARendezvousRebuildStorm": (ControlAction.FREEZE_NEW_PLANS, ControlAction.MARK_SUSPECT, ControlAction.CAPTURE_EVIDENCE, ControlAction.NOTIFY_OWNER),
        "ACVJEPAExactStateRestoreFailed": (ControlAction.FREEZE_NEW_PLANS, ControlAction.MARK_SUSPECT, ControlAction.CAPTURE_EVIDENCE, ControlAction.NOTIFY_OWNER),
        "ACVJEPAFP8MetadataRestoreFailed": (ControlAction.FREEZE_NEW_PLANS, ControlAction.MARK_SUSPECT, ControlAction.CAPTURE_EVIDENCE, ControlAction.NOTIFY_OWNER),
        "ACVJEPAAllReduceTailLatencyHigh": (ControlAction.MARK_SUSPECT, ControlAction.CAPTURE_EVIDENCE, ControlAction.NOTIFY_OWNER),
        "ACVJEPAUpdateAbortRatioHigh": (ControlAction.MARK_SUSPECT, ControlAction.CAPTURE_EVIDENCE, ControlAction.NOTIFY_OWNER),
    }

    def __init__(self, *, allowed_environments: Iterable[str] = ("isolated-preproduction", "production")) -> None:
        self.allowed_environments = frozenset(allowed_environments)
        self.frozen_jobs: set[tuple[str, str, str]] = set()
        self.suspect_jobs: set[tuple[str, str, str]] = set()
        self.seen: set[str] = set()
        self.audit: list[dict[str, object]] = []

    def handle(self, alert: SafeAlert) -> tuple[ControlAction, ...]:
        if alert.status != "firing":
            return ()
        if alert.environment not in self.allowed_environments:
            raise AlertRejected("alert environment is outside the static allowlist")
        if alert.alertname not in self.ALLOWED:
            raise AlertRejected("alert is not in the non-destructive action allowlist")
        key = (alert.cluster, alert.job, alert.environment)
        if alert.fingerprint in self.seen:
            return ()
        self.seen.add(alert.fingerprint)
        actions = self.ALLOWED[alert.alertname]
        for action in actions:
            if action is ControlAction.FREEZE_NEW_PLANS:
                self.frozen_jobs.add(key)
            elif action is ControlAction.MARK_SUSPECT:
                self.suspect_jobs.add(key)
            # CAPTURE_EVIDENCE/NOTIFY_OWNER are audit intents only. A production
            # adapter writes an evidence request and sends an authenticated notice.
            self.audit.append({"alert": alert.alertname, "fingerprint": alert.fingerprint, "action": action.value, "target": key})
        return actions

    def may_arm_new_plan(self, labels: TrainingMetricLabels) -> bool:
        return (labels.cluster, labels.job, labels.environment) not in self.frozen_jobs


def simulate_drill() -> dict[str, object]:
    """Non-network drill: mass preemption planning + metrics + safe alert linkage."""

    labels = TrainingMetricLabels(cluster="chaos-cluster", job="acvjepa-recovery-drill", environment="isolated-preproduction")
    metrics = DistributedTrainingMetrics(labels)
    metrics.set_build(precision_mode="bf16", precision_backend="torch_amp")
    metrics.record_plan(global_samples=6, micro_batches_by_rank={0: 2, 1: 1}, digest_consensus=True)
    metrics.record_cursor(next_offset=120, prepared_reservations=1, checkpoint_age_seconds=4.0)
    metrics.record_rebuild(cause="mass_preemption", restart_count=2, recovery_seconds=17.2)
    planner = RecoveryWavePlanner(RecoveryPolicy(max_nodes_per_wave=3, base_backoff_seconds=2.0, max_backoff_seconds=30.0, jitter_ratio=0.15, checkpoint_load_concurrency_per_wave=2))
    admissions = planner.plan(run_id="drill-run-42", restart_count=2, node_ids=[f"node-{index}" for index in range(8)])
    assert len(admissions) == 8 and max(item.wave for item in admissions) == 2
    assert all(item.checkpoint_load_slot in {0, 1} for item in admissions)

    control = SafeAlertControlPlane()
    webhook_fixture = {
        "version": "4",
        "status": "firing",
        "alerts": [{
            "status": "firing",
            "fingerprint": "alert-fp-001",
            "labels": {
                "alertname": "ACVJEPARendezvousRebuildStorm",
                "cluster": labels.cluster,
                "job": labels.job,
                "environment": labels.environment,
                "severity": "critical",
            },
        }],
    }
    parsed_alert = alerts_from_alertmanager_payload(webhook_fixture)[0]
    actions = control.handle(parsed_alert)
    assert ControlAction.FREEZE_NEW_PLANS in actions and not control.may_arm_new_plan(labels)
    # Repeated webhook delivery is idempotent.
    assert control.handle(parsed_alert) == ()
    metrics.record_chaos(phase="recovery_frozen")
    return {
        "smoke_test": "passed",
        "admissions": [asdict(item) for item in admissions],
        "promql_queries": PROMQL,
        "safe_actions": [item.value for item in actions],
        "new_plans_frozen": not control.may_arm_new_plan(labels),
        "audit_events": len(control.audit),
        "metrics_contains_plan": "acvjepa_training_plan_micro_batches" in metrics.exposition(),
    }


def main() -> None:
    result = simulate_drill()
    assert result["metrics_contains_plan"] and result["new_plans_frozen"]
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
