"""Prometheus instrumentation for the AC-VJEPA elastic training control plane.

This exporter intentionally emits only low-cardinality operational labels. It
never exports work IDs, trajectory IDs, checkpoint URIs/hashes, raw error text,
or approvals. Those belong in the tamper-evident ledger/object store, linked by
a bounded incident/experiment ID outside metric labels.

Metrics are observations and alert inputs only. No method changes networking,
cluster membership, data cursor or model release state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


Outcome = Literal["committed", "aborted", "rejected"]


@dataclass(frozen=True)
class TrainingMetricLabels:
    cluster: str
    job: str
    environment: str  # bounded: isolated-preproduction | production

    def as_dict(self) -> dict[str, str]:
        if not all((self.cluster, self.job, self.environment)):
            raise ValueError("cluster, job and environment labels are required")
        return {"cluster": self.cluster, "job": self.job, "environment": self.environment}


class DistributedTrainingMetrics:
    """A process-local metric facade; scrape server wiring remains deployment-owned."""

    def __init__(self, labels: TrainingMetricLabels, registry: CollectorRegistry | None = None) -> None:
        self.labels = labels
        self.registry = registry or CollectorRegistry()
        common = ("cluster", "job", "environment")
        self.info = Gauge(
            "acvjepa_training_build_info",
            "Build/precision metadata; labels must be bounded release values.",
            (*common, "precision_mode", "precision_backend"),
            registry=self.registry,
        )
        self.update_attempts = Counter(
            "acvjepa_training_update_attempts_total",
            "Update attempts by bounded outcome.",
            (*common, "outcome"),
            registry=self.registry,
        )
        self.update_duration = Histogram(
            "acvjepa_training_update_duration_seconds",
            "End-to-end update duration including local accumulation.",
            common,
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.allreduce_duration = Histogram(
            "acvjepa_training_allreduce_duration_seconds",
            "Per-update final DDP synchronization duration.",
            (*common, "rail_class"),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
            registry=self.registry,
        )
        self.update_global_samples = Gauge(
            "acvjepa_training_update_global_samples",
            "Effective global samples in the currently armed UpdatePlan.",
            common,
            registry=self.registry,
        )
        self.plan_micro_batches = Gauge(
            "acvjepa_training_plan_micro_batches",
            "Planned local micro-batches by bounded rank slot.",
            (*common, "rank_slot"),
            registry=self.registry,
        )
        self.plan_digest_consensus = Gauge(
            "acvjepa_training_plan_digest_consensus",
            "1 when all ranks accepted the current plan digest, otherwise 0.",
            common,
            registry=self.registry,
        )
        self.cursor_next_offset = Gauge(
            "acvjepa_training_cursor_next_offset",
            "Next uncommitted global work-window offset from the durable ledger.",
            common,
            registry=self.registry,
        )
        self.cursor_reservations = Gauge(
            "acvjepa_training_cursor_reservations",
            "Number of prepared update reservations; normally zero or one.",
            common,
            registry=self.registry,
        )
        self.checkpoint_commits = Counter(
            "acvjepa_training_checkpoint_commits_total",
            "Atomically committed checkpoints.",
            common,
            registry=self.registry,
        )
        self.checkpoint_age = Gauge(
            "acvjepa_training_checkpoint_age_seconds",
            "Age of the latest verified committed checkpoint.",
            common,
            registry=self.registry,
        )
        self.rendezvous_rebuilds = Counter(
            "acvjepa_training_rendezvous_rebuilds_total",
            "Worker-group rebuilds by bounded cause.",
            (*common, "cause"),
            registry=self.registry,
        )
        self.restart_count = Gauge(
            "acvjepa_training_restart_count",
            "Current TORCHELASTIC restart count.",
            common,
            registry=self.registry,
        )
        self.recovery_duration = Histogram(
            "acvjepa_training_recovery_duration_seconds",
            "From failure marker to new-plan digest consensus.",
            common,
            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
            registry=self.registry,
        )
        self.state_alignment = Gauge(
            "acvjepa_training_state_alignment_verified",
            "1 only after exact committed-checkpoint restore validation.",
            (*common, "component"),
            registry=self.registry,
        )
        self.grad_scaler_scale = Gauge(
            "acvjepa_training_grad_scaler_scale",
            "Current GradScaler scale; absent/NaN semantics are represented by scaler_enabled.",
            common,
            registry=self.registry,
        )
        self.grad_scaler_enabled = Gauge(
            "acvjepa_training_grad_scaler_enabled",
            "1 for an enabled FP16 GradScaler, otherwise 0.",
            common,
            registry=self.registry,
        )
        self.precision_overflows = Counter(
            "acvjepa_training_precision_overflows_total",
            "Detected non-finite gradient/scale-overflow events by precision mode.",
            (*common, "precision_mode"),
            registry=self.registry,
        )
        self.fp8_metadata_verified = Gauge(
            "acvjepa_training_fp8_metadata_verified",
            "1 only when required FP8 metadata and runtime version matched checkpoint contract.",
            common,
            registry=self.registry,
        )
        self.chaos_phase = Gauge(
            "acvjepa_training_chaos_experiment_phase",
            "Current approved experiment phase; profile/experiment identity remain in audit logs.",
            (*common, "phase"),
            registry=self.registry,
        )
        self.chaos_guard_rejections = Counter(
            "acvjepa_training_chaos_guard_rejections_total",
            "Rejected chaos requests by bounded policy reason.",
            (*common, "reason"),
            registry=self.registry,
        )
        self.failpoint_triggers = Counter(
            "acvjepa_training_failpoint_triggers_total",
            "Offline or approved failpoint triggers by bounded class and terminal outcome.",
            (*common, "fault_class", "outcome"),
            registry=self.registry,
        )
        self.failpoint_stage_duration = Histogram(
            "acvjepa_training_failpoint_stage_duration_seconds",
            "Elapsed time of a bounded recovery phase, measured from local monotonic markers.",
            (*common, "fault_class", "phase"),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5, 15, 30, 60, 120, 300, 600, 1800),
            registry=self.registry,
        )
        self.failpoint_active = Gauge(
            "acvjepa_training_failpoint_active",
            "1 while a bounded failpoint recovery attempt is in progress, otherwise 0.",
            (*common, "fault_class"),
            registry=self.registry,
        )
        self.cache_fetches = Counter(
            "acvjepa_training_checkpoint_cache_fetches_total",
            "Verified checkpoint-cache fetches by bounded tier, component class and result.",
            (*common, "cache_tier", "component_class", "outcome"),
            registry=self.registry,
        )
        self.cache_bytes = Counter(
            "acvjepa_training_checkpoint_cache_bytes_total",
            "Verified checkpoint-cache bytes by bounded tier, component class and result.",
            (*common, "cache_tier", "component_class", "outcome"),
            registry=self.registry,
        )
        self.cache_fetch_duration = Histogram(
            "acvjepa_training_checkpoint_cache_fetch_duration_seconds",
            "Verified checkpoint-cache fetch duration by bounded tier, component class and result.",
            (*common, "cache_tier", "component_class", "outcome"),
            buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5, 15, 30, 60),
            registry=self.registry,
        )
        # RecoveryDeploymentEpoch is a bounded control-plane projection. Exact
        # Git revisions, hashes, cursor IDs and fencing tokens remain in the
        # durable arbitration ledger / audit log, never in Prometheus labels.
        self.recovery_deployment_state = Gauge(
            "acvjepa_training_recovery_deployment_state",
            "One-hot current RecoveryDeploymentEpoch state.",
            (*common, "state"),
            registry=self.registry,
        )
        self.recovery_deployment_generation = Gauge(
            "acvjepa_training_recovery_deployment_generation",
            "Monotonic RecoveryDeploymentEpoch generation; no token is exposed.",
            common,
            registry=self.registry,
        )
        self.recovery_deployment_state_age = Gauge(
            "acvjepa_training_recovery_deployment_state_age_seconds",
            "Age of the active RecoveryDeploymentEpoch state.",
            common,
            registry=self.registry,
        )
        self.recovery_deployment_inputs_valid = Gauge(
            "acvjepa_training_recovery_deployment_inputs_valid",
            "1 only when checkpoint, cursor, precision, topology and plan bindings pass the arbitration gate.",
            common,
            registry=self.registry,
        )
        self.recovery_git_revision_match = Gauge(
            "acvjepa_training_recovery_git_revision_match",
            "1 only when recovery binding matches the currently desired GitOps revision; revision itself is not labeled.",
            common,
            registry=self.registry,
        )
        self.recovery_deployment_fence_rejections = Counter(
            "acvjepa_training_recovery_deployment_fence_rejections_total",
            "Rejected stale recovery or deployment writes by bounded actor and reason.",
            (*common, "actor", "reason"),
            registry=self.registry,
        )
        self.gitops_sync_attempts = Counter(
            "acvjepa_training_gitops_sync_attempts_total",
            "GitOps sync attempts by bounded terminal result.",
            (*common, "outcome"),
            registry=self.registry,
        )
        self.gitops_pending_age = Gauge(
            "acvjepa_training_gitops_pending_age_seconds",
            "Age of desired GitOps change waiting for recovery/deployment arbitration.",
            common,
            registry=self.registry,
        )
        self._last_recovery_deployment_state: str | None = None

    @property
    def _labels(self) -> dict[str, str]:
        return self.labels.as_dict()

    def set_build(self, *, precision_mode: str, precision_backend: str) -> None:
        self.info.labels(**self._labels, precision_mode=precision_mode, precision_backend=precision_backend).set(1)

    def record_plan(self, *, global_samples: int, micro_batches_by_rank: dict[int, int], digest_consensus: bool) -> None:
        if global_samples <= 0 or any(value < 1 for value in micro_batches_by_rank.values()):
            raise ValueError("invalid plan metric values")
        self.update_global_samples.labels(**self._labels).set(global_samples)
        self.plan_digest_consensus.labels(**self._labels).set(int(digest_consensus))
        for rank, count in micro_batches_by_rank.items():
            self.plan_micro_batches.labels(**self._labels, rank_slot=str(rank)).set(count)

    def record_update(self, *, outcome: Outcome, duration_seconds: float, allreduce_seconds: float | None = None, rail_class: str = "unknown") -> None:
        if duration_seconds < 0 or allreduce_seconds is not None and allreduce_seconds < 0:
            raise ValueError("durations must be non-negative")
        self.update_attempts.labels(**self._labels, outcome=outcome).inc()
        self.update_duration.labels(**self._labels).observe(duration_seconds)
        if allreduce_seconds is not None:
            self.allreduce_duration.labels(**self._labels, rail_class=rail_class).observe(allreduce_seconds)

    def record_cursor(self, *, next_offset: int, prepared_reservations: int, checkpoint_age_seconds: float, committed: bool = False) -> None:
        if min(next_offset, prepared_reservations, checkpoint_age_seconds) < 0:
            raise ValueError("cursor/checkpoint metric values must be non-negative")
        self.cursor_next_offset.labels(**self._labels).set(next_offset)
        self.cursor_reservations.labels(**self._labels).set(prepared_reservations)
        self.checkpoint_age.labels(**self._labels).set(checkpoint_age_seconds)
        if committed:
            self.checkpoint_commits.labels(**self._labels).inc()

    def record_rebuild(self, *, cause: str, restart_count: int, recovery_seconds: float) -> None:
        if restart_count < 0 or recovery_seconds < 0:
            raise ValueError("restart/recovery values must be non-negative")
        self.rendezvous_rebuilds.labels(**self._labels, cause=cause).inc()
        self.restart_count.labels(**self._labels).set(restart_count)
        self.recovery_duration.labels(**self._labels).observe(recovery_seconds)

    def record_precision_restore(
        self,
        *,
        precision_mode: str,
        components_exact: dict[str, bool],
        scaler_enabled: bool,
        scaler_scale: float | None,
        fp8_metadata_exact: bool | None,
    ) -> None:
        for component, exact in components_exact.items():
            self.state_alignment.labels(**self._labels, component=component).set(int(exact))
        self.grad_scaler_enabled.labels(**self._labels).set(int(scaler_enabled))
        if scaler_scale is not None:
            self.grad_scaler_scale.labels(**self._labels).set(scaler_scale)
        if fp8_metadata_exact is not None:
            self.fp8_metadata_verified.labels(**self._labels).set(int(fp8_metadata_exact))
        if precision_mode not in {"bf16", "fp16", "fp8"}:
            raise ValueError("precision mode must be bf16, fp16 or fp8")

    def record_precision_overflow(self, precision_mode: str) -> None:
        self.precision_overflows.labels(**self._labels, precision_mode=precision_mode).inc()

    def record_chaos(self, *, phase: str, guard_rejection_reason: str | None = None) -> None:
        self.chaos_phase.labels(**self._labels, phase=phase).set(1)
        if guard_rejection_reason:
            self.chaos_guard_rejections.labels(**self._labels, reason=guard_rejection_reason).inc()

    def record_failpoint(self, *, fault_class: str, phase: str, duration_seconds: float, active: bool, outcome: str | None = None) -> None:
        valid_faults = {
            "node_loss_final_allreduce",
            "network_partition_after_prepare",
            "stale_plan_after_rendezvous",
            "plan_topology_mismatch",
            "cache_corruption_during_restore",
        }
        valid_phases = {"trigger_to_detect", "detect_to_freeze", "freeze_to_rendezvous", "rendezvous_to_cursor_replay", "cursor_replay_to_checkpoint_load", "checkpoint_load_to_state_verify", "state_verify_to_training_ready", "trigger_to_training_ready"}
        if fault_class not in valid_faults or phase not in valid_phases or duration_seconds < 0:
            raise ValueError("invalid bounded failpoint metric values")
        self.failpoint_stage_duration.labels(**self._labels, fault_class=fault_class, phase=phase).observe(duration_seconds)
        self.failpoint_active.labels(**self._labels, fault_class=fault_class).set(int(active))
        if outcome is not None:
            if outcome not in {"passed", "failed", "rejected"}:
                raise ValueError("invalid failpoint outcome")
            self.failpoint_triggers.labels(**self._labels, fault_class=fault_class, outcome=outcome).inc()

    def record_cache_fetch(self, *, cache_tier: str, component_class: str, outcome: str, byte_count: int, duration_seconds: float) -> None:
        valid_tiers = {"node_local", "rdma", "durable"}
        valid_components = {"model", "ema", "optimizer", "precision", "rng"}
        valid_outcomes = {"hit", "miss", "fallback", "negative_hit", "integrity_failed", "rejected"}
        if cache_tier not in valid_tiers or component_class not in valid_components or outcome not in valid_outcomes:
            raise ValueError("invalid bounded cache metric labels")
        if byte_count < 0 or duration_seconds < 0:
            raise ValueError("cache bytes and duration must be non-negative")
        labels = {**self._labels, "cache_tier": cache_tier, "component_class": component_class, "outcome": outcome}
        self.cache_fetches.labels(**labels).inc()
        self.cache_bytes.labels(**labels).inc(byte_count)
        self.cache_fetch_duration.labels(**labels).observe(duration_seconds)

    def record_recovery_deployment(
        self,
        *,
        state: str,
        generation: int,
        state_age_seconds: float,
        inputs_valid: bool,
        git_revision_matches: bool,
        gitops_pending_age_seconds: float,
        fence_rejection: tuple[str, str] | None = None,
        gitops_outcome: str | None = None,
    ) -> None:
        valid_states = {"IDLE", "RECOVERING", "RECOVERY_READY", "DEPLOYMENT_ARMED", "FROZEN"}
        valid_actors = {"training_worker", "recovery_controller", "gitops_controller", "admission"}
        valid_reasons = {"stale_generation", "lease_expired", "git_revision_mismatch", "input_binding_invalid", "state_not_armed", "cursor_checkpoint_mismatch"}
        valid_outcomes = {"applied", "deferred", "rejected", "failed"}
        if state not in valid_states or generation < 0 or min(state_age_seconds, gitops_pending_age_seconds) < 0:
            raise ValueError("invalid recovery/deployment state metric values")
        if self._last_recovery_deployment_state is not None and self._last_recovery_deployment_state != state:
            self.recovery_deployment_state.labels(**self._labels, state=self._last_recovery_deployment_state).set(0)
        self.recovery_deployment_state.labels(**self._labels, state=state).set(1)
        self._last_recovery_deployment_state = state
        self.recovery_deployment_generation.labels(**self._labels).set(generation)
        self.recovery_deployment_state_age.labels(**self._labels).set(state_age_seconds)
        self.recovery_deployment_inputs_valid.labels(**self._labels).set(int(inputs_valid))
        self.recovery_git_revision_match.labels(**self._labels).set(int(git_revision_matches))
        self.gitops_pending_age.labels(**self._labels).set(gitops_pending_age_seconds)
        if fence_rejection is not None:
            actor, reason = fence_rejection
            if actor not in valid_actors or reason not in valid_reasons:
                raise ValueError("invalid bounded fencing label")
            self.recovery_deployment_fence_rejections.labels(**self._labels, actor=actor, reason=reason).inc()
        if gitops_outcome is not None:
            if gitops_outcome not in valid_outcomes:
                raise ValueError("invalid bounded GitOps outcome")
            self.gitops_sync_attempts.labels(**self._labels, outcome=gitops_outcome).inc()

    def exposition(self) -> str:
        return generate_latest(self.registry).decode("utf-8")


def run_smoke_test() -> None:
    metrics = DistributedTrainingMetrics(TrainingMetricLabels(cluster="test-cluster", job="acvjepa-demo", environment="isolated-preproduction"))
    metrics.set_build(precision_mode="bf16", precision_backend="torch_amp")
    metrics.record_plan(global_samples=6, micro_batches_by_rank={0: 2, 1: 1}, digest_consensus=True)
    metrics.record_update(outcome="committed", duration_seconds=0.18, allreduce_seconds=0.01, rail_class="single_rail")
    metrics.record_cursor(next_offset=3, prepared_reservations=0, checkpoint_age_seconds=0.2, committed=True)
    metrics.record_rebuild(cause="node_failure", restart_count=1, recovery_seconds=11.0)
    metrics.record_precision_restore(
        precision_mode="bf16",
        components_exact={"model": True, "ema": True, "optimizer": True, "rng": True},
        scaler_enabled=False,
        scaler_scale=None,
        fp8_metadata_exact=None,
    )
    metrics.record_chaos(phase="rolled_back", guard_rejection_reason="production_scope")
    metrics.record_failpoint(fault_class="node_loss_final_allreduce", phase="trigger_to_training_ready", duration_seconds=0.12, active=False, outcome="passed")
    metrics.record_cache_fetch(cache_tier="rdma", component_class="optimizer", outcome="hit", byte_count=1024, duration_seconds=0.004)
    metrics.record_recovery_deployment(
        state="RECOVERY_READY", generation=4, state_age_seconds=1.5,
        inputs_valid=True, git_revision_matches=True, gitops_pending_age_seconds=0.0,
        fence_rejection=("gitops_controller", "stale_generation"), gitops_outcome="deferred",
    )
    text = metrics.exposition()
    for expected in (
        'acvjepa_training_update_global_samples{cluster="test-cluster",environment="isolated-preproduction",job="acvjepa-demo"} 6.0',
        'acvjepa_training_plan_micro_batches{cluster="test-cluster",environment="isolated-preproduction",job="acvjepa-demo",rank_slot="0"} 2.0',
        'acvjepa_training_state_alignment_verified{cluster="test-cluster",component="optimizer",environment="isolated-preproduction",job="acvjepa-demo"} 1.0',
        'acvjepa_training_failpoint_triggers_total{cluster="test-cluster",environment="isolated-preproduction",fault_class="node_loss_final_allreduce",job="acvjepa-demo",outcome="passed"} 1.0',
        'acvjepa_training_checkpoint_cache_fetches_total{cache_tier="rdma",cluster="test-cluster",component_class="optimizer",environment="isolated-preproduction",job="acvjepa-demo",outcome="hit"} 1.0',
        'acvjepa_training_recovery_deployment_state{cluster="test-cluster",environment="isolated-preproduction",job="acvjepa-demo",state="RECOVERY_READY"} 1.0',
        'acvjepa_training_recovery_git_revision_match{cluster="test-cluster",environment="isolated-preproduction",job="acvjepa-demo"} 1.0',
    ):
        assert expected in text, expected
    print('{"smoke_test":"passed","exported_metrics":true}', flush=True)


if __name__ == "__main__":
    run_smoke_test()
