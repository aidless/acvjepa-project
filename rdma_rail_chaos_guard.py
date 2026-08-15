"""Safe-by-default RDMA/rail chaos request guard.

This module deliberately DOES NOT contain shell, SSH, subprocess, firewall,
route, RDMA, switch, NIC, cloud or cluster-management commands. It validates a
signed experiment manifest and sends it only to an injected *trusted executor*
interface. The included executor is a local recording simulator used for tests.

A real executor must be deployed by infrastructure owners behind strong server-
side controls (independent authorization, mTLS/workload identity, test-resource
labels, TTL enforcement, idempotent rollback and immutable audit logging). The
client guard is defense-in-depth, never the only safety mechanism.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Mapping, Protocol, Sequence, Set
from uuid import uuid4


class GuardRejected(RuntimeError):
    """The requested experiment failed a fail-closed precondition."""


def _canonical(value: Mapping) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mac(secret: bytes, payload: Mapping) -> str:
    return hmac.new(secret, _canonical(payload), hashlib.sha256).hexdigest()


class ChaosMode(str, Enum):
    DRY_RUN = "dry_run"
    EXECUTE = "execute"


class ChaosPhase(str, Enum):
    PREPARED = "prepared"
    INJECTED = "injected"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class Approval:
    """Bounded approval signature; one training owner plus one infra owner."""

    role: str  # exactly: training_owner or infrastructure_owner
    principal: str
    issued_ns: int
    expires_ns: int
    signature: str

    @staticmethod
    def sign(*, role: str, principal: str, experiment_digest: str, issued_ns: int, expires_ns: int, secret: bytes) -> "Approval":
        payload = {
            "role": role,
            "principal": principal,
            "experiment_digest": experiment_digest,
            "issued_ns": issued_ns,
            "expires_ns": expires_ns,
        }
        return Approval(signature=_mac(secret, payload), **{key: payload[key] for key in ("role", "principal", "issued_ns", "expires_ns")})

    def verify(self, *, experiment_digest: str, secret: bytes, now_ns: int) -> None:
        if self.role not in {"training_owner", "infrastructure_owner"} or not self.principal:
            raise GuardRejected("approval role/principal is invalid")
        if not (self.issued_ns <= now_ns <= self.expires_ns):
            raise GuardRejected(f"approval for {self.role} is expired or not yet valid")
        expected = _mac(
            secret,
            {
                "role": self.role,
                "principal": self.principal,
                "experiment_digest": experiment_digest,
                "issued_ns": self.issued_ns,
                "expires_ns": self.expires_ns,
            },
        )
        if not hmac.compare_digest(expected, self.signature):
            raise GuardRejected(f"approval signature for {self.role} is invalid")


@dataclass(frozen=True)
class IsolationEvidence:
    """Attestation supplied by a separate test-environment inventory service."""

    environment: str
    dedicated_test_pool: bool
    shared_fabric: bool
    robot_control_active: bool
    production: bool
    target_group: str
    allowed_rails: tuple[str, ...]
    inventory_epoch: str


@dataclass(frozen=True)
class RuntimePreconditions:
    latest_checkpoint_committed: bool
    checkpoint_hash: str
    topology_epoch: str
    work_manifest_digest: str
    no_active_fault: bool
    gpu_health_green: bool
    nic_health_green: bool
    trace_sink_writable: bool


@dataclass(frozen=True)
class RailFaultExperiment:
    experiment_id: str
    mode: ChaosMode
    target_group: str
    rail_id: str
    fault_profile: str  # strictly a pre-registered executor-side profile identifier
    ttl_seconds: int
    max_restarts: int
    expected_topology_epoch: str
    expected_work_manifest_digest: str
    nonce: str

    def unsigned_payload(self) -> Dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "mode": self.mode.value,
            "target_group": self.target_group,
            "rail_id": self.rail_id,
            "fault_profile": self.fault_profile,
            "ttl_seconds": self.ttl_seconds,
            "max_restarts": self.max_restarts,
            "expected_topology_epoch": self.expected_topology_epoch,
            "expected_work_manifest_digest": self.expected_work_manifest_digest,
            "nonce": self.nonce,
        }

    @property
    def digest(self) -> str:
        return _sha256(_canonical(self.unsigned_payload()))


@dataclass(frozen=True)
class GuardPolicy:
    required_environment: str = "isolated-preproduction"
    allowed_target_groups: tuple[str, ...] = ("chaos-gpu-pool-a",)
    allowed_fault_profiles: tuple[str, ...] = ("rdma_rail_blackhole_test_profile", "rdma_rail_delay_test_profile")
    max_ttl_seconds: int = 120
    max_restarts: int = 2
    required_execution_phrase: str = "AUTHORIZED_ISOLATED_CHAOS_ONLY"


@dataclass(frozen=True)
class FaultReceipt:
    experiment_id: str
    phase: ChaosPhase
    executor_receipt_id: str
    expires_ns: int
    request_digest: str


class TrustedRailExecutor(Protocol):
    """Infrastructure-owned interface; not an implementation of network control."""

    def inject(self, experiment: RailFaultExperiment, approvals: Sequence[Approval]) -> FaultReceipt: ...

    def rollback(self, receipt: FaultReceipt, reason: str) -> FaultReceipt: ...


class RecordingExecutor:
    """Safe local mock that proves request/rollback orchestration without network I/O."""

    def __init__(self) -> None:
        self.receipts: Dict[str, FaultReceipt] = {}
        self.events: list[dict] = []

    def inject(self, experiment: RailFaultExperiment, approvals: Sequence[Approval]) -> FaultReceipt:
        receipt = FaultReceipt(
            experiment_id=experiment.experiment_id,
            phase=ChaosPhase.INJECTED,
            executor_receipt_id="mock-" + uuid4().hex,
            expires_ns=time.time_ns() + experiment.ttl_seconds * 1_000_000_000,
            request_digest=experiment.digest,
        )
        self.receipts[receipt.executor_receipt_id] = receipt
        self.events.append({"event": "mock_inject", "receipt": asdict(receipt), "approvals": [item.role for item in approvals]})
        return receipt

    def rollback(self, receipt: FaultReceipt, reason: str) -> FaultReceipt:
        if receipt.executor_receipt_id not in self.receipts:
            raise GuardRejected("executor does not recognize receipt")
        rolled_back = FaultReceipt(
            experiment_id=receipt.experiment_id,
            phase=ChaosPhase.ROLLED_BACK,
            executor_receipt_id=receipt.executor_receipt_id,
            expires_ns=receipt.expires_ns,
            request_digest=receipt.request_digest,
        )
        self.receipts[receipt.executor_receipt_id] = rolled_back
        self.events.append({"event": "mock_rollback", "receipt": asdict(rolled_back), "reason": reason})
        return rolled_back


class RailChaosGuard:
    """Validate every precondition before a trusted executor receives a request."""

    def __init__(self, policy: GuardPolicy, approval_secret: bytes) -> None:
        if len(approval_secret) < 16:
            raise ValueError("use a non-trivial dedicated test approval secret")
        self.policy = policy
        self.approval_secret = approval_secret

    def validate(
        self,
        *,
        experiment: RailFaultExperiment,
        approvals: Sequence[Approval],
        isolation: IsolationEvidence,
        runtime: RuntimePreconditions,
        now_ns: int | None = None,
    ) -> None:
        now_ns = time.time_ns() if now_ns is None else now_ns
        if experiment.mode is ChaosMode.EXECUTE and os.environ.get("CHAOS_EXECUTION_INTERLOCK") != self.policy.required_execution_phrase:
            raise GuardRejected("execution interlock is absent; only dry_run is allowed")
        if experiment.target_group not in self.policy.allowed_target_groups:
            raise GuardRejected("target group is not in the static test-pool allowlist")
        if experiment.fault_profile not in self.policy.allowed_fault_profiles:
            raise GuardRejected("fault profile is not pre-registered")
        if not (1 <= experiment.ttl_seconds <= self.policy.max_ttl_seconds):
            raise GuardRejected("TTL is outside the approved bounded range")
        if not (0 <= experiment.max_restarts <= self.policy.max_restarts):
            raise GuardRejected("restart budget is outside the policy")
        if (
            isolation.environment != self.policy.required_environment
            or isolation.production
            or not isolation.dedicated_test_pool
            or isolation.shared_fabric
            or isolation.robot_control_active
        ):
            raise GuardRejected("environment does not prove isolated non-production test scope")
        if isolation.target_group != experiment.target_group or experiment.rail_id not in isolation.allowed_rails:
            raise GuardRejected("target group or rail does not match signed isolation inventory")
        if not isolation.inventory_epoch:
            raise GuardRejected("isolation inventory epoch is missing")
        if (
            not runtime.latest_checkpoint_committed
            or not runtime.no_active_fault
            or not runtime.gpu_health_green
            or not runtime.nic_health_green
            or not runtime.trace_sink_writable
        ):
            raise GuardRejected("checkpoint, active-fault, health or evidence precondition failed")
        if runtime.topology_epoch != experiment.expected_topology_epoch:
            raise GuardRejected("topology epoch differs from the armed experiment")
        if runtime.work_manifest_digest != experiment.expected_work_manifest_digest:
            raise GuardRejected("work manifest digest differs from the armed experiment")
        if len(runtime.checkpoint_hash) != 64:
            raise GuardRejected("checkpoint hash is malformed")
        self._validate_dual_approvals(approvals, experiment.digest, now_ns)

    def inject(
        self,
        *,
        executor: TrustedRailExecutor,
        experiment: RailFaultExperiment,
        approvals: Sequence[Approval],
        isolation: IsolationEvidence,
        runtime: RuntimePreconditions,
    ) -> FaultReceipt:
        self.validate(experiment=experiment, approvals=approvals, isolation=isolation, runtime=runtime)
        # Even EXECUTE mode has no direct network capability here. The separate
        # executor must re-validate all fields before acting and issue a receipt.
        return executor.inject(experiment, approvals)

    def run_with_rollback(
        self,
        *,
        executor: TrustedRailExecutor,
        experiment: RailFaultExperiment,
        approvals: Sequence[Approval],
        isolation: IsolationEvidence,
        runtime: RuntimePreconditions,
        observe: callable,
    ) -> FaultReceipt:
        receipt = self.inject(
            executor=executor,
            experiment=experiment,
            approvals=approvals,
            isolation=isolation,
            runtime=runtime,
        )
        try:
            observe(receipt)
        finally:
            # Fail closed: successful observation, assertion error, timeout or
            # unexpected exception all request executor-side rollback.
            receipt = executor.rollback(receipt, reason="bounded_experiment_complete_or_exception")
        if receipt.phase is not ChaosPhase.ROLLED_BACK:
            raise GuardRejected("executor did not confirm rollback")
        return receipt

    def _validate_dual_approvals(self, approvals: Sequence[Approval], digest: str, now_ns: int) -> None:
        if len(approvals) != 2:
            raise GuardRejected("exactly two approvals are required")
        roles = {item.role for item in approvals}
        principals = {item.principal for item in approvals}
        if roles != {"training_owner", "infrastructure_owner"} or len(principals) != 2:
            raise GuardRejected("two distinct training/infrastructure approvers are required")
        for approval in approvals:
            approval.verify(experiment_digest=digest, secret=self.approval_secret, now_ns=now_ns)


def run_smoke_test() -> None:
    """Validate dry-run, fail-closed production rejection and rollback guarantee."""

    secret = b"only-for-isolated-chaos-tests"
    guard = RailChaosGuard(GuardPolicy(), secret)
    experiment = RailFaultExperiment(
        experiment_id="chaos-demo-001",
        mode=ChaosMode.DRY_RUN,
        target_group="chaos-gpu-pool-a",
        rail_id="rail-test-a",
        fault_profile="rdma_rail_delay_test_profile",
        ttl_seconds=30,
        max_restarts=1,
        expected_topology_epoch="epoch-abc",
        expected_work_manifest_digest="a" * 64,
        nonce="nonce-" + uuid4().hex,
    )
    now = time.time_ns()
    approvals = (
        Approval.sign(role="training_owner", principal="trainer.alice", experiment_digest=experiment.digest, issued_ns=now - 1, expires_ns=now + 60_000_000_000, secret=secret),
        Approval.sign(role="infrastructure_owner", principal="infra.bob", experiment_digest=experiment.digest, issued_ns=now - 1, expires_ns=now + 60_000_000_000, secret=secret),
    )
    isolation = IsolationEvidence(
        environment="isolated-preproduction",
        dedicated_test_pool=True,
        shared_fabric=False,
        robot_control_active=False,
        production=False,
        target_group="chaos-gpu-pool-a",
        allowed_rails=("rail-test-a",),
        inventory_epoch="inventory-v7",
    )
    runtime = RuntimePreconditions(
        latest_checkpoint_committed=True,
        checkpoint_hash="b" * 64,
        topology_epoch="epoch-abc",
        work_manifest_digest="a" * 64,
        no_active_fault=True,
        gpu_health_green=True,
        nic_health_green=True,
        trace_sink_writable=True,
    )
    executor = RecordingExecutor()
    receipt = guard.run_with_rollback(
        executor=executor,
        experiment=experiment,
        approvals=approvals,
        isolation=isolation,
        runtime=runtime,
        observe=lambda injected: None,
    )
    assert receipt.phase is ChaosPhase.ROLLED_BACK
    try:
        guard.validate(
            experiment=experiment,
            approvals=approvals,
            isolation=IsolationEvidence(**{**asdict(isolation), "production": True}),
            runtime=runtime,
            now_ns=now,
        )
    except GuardRejected as exc:
        assert "isolated" in str(exc)
    else:
        raise AssertionError("production scope must be rejected")
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "mode": experiment.mode.value,
                "events": [event["event"] for event in executor.events],
                "final_phase": receipt.phase.value,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run_smoke_test()
