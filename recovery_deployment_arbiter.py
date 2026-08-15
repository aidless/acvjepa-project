"""In-memory reference for rendezvous / GitOps concurrent-writer arbitration.

This module models control-plane correctness only. It does not call Kubernetes,
Git providers, object storage, NCCL, or any deployment tool. Production must
persist the record in a strongly-consistent store and use CAS/resourceVersion
or equivalent transactions; Kubernetes Lease provides leader liveness but does
not replace the bound state record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from threading import Lock
from typing import Literal


class ArbitrationError(RuntimeError):
    pass


class EpochState(str, Enum):
    IDLE = "IDLE"
    RECOVERING = "RECOVERING"
    RECOVERY_READY = "RECOVERY_READY"
    DEPLOYMENT_ARMED = "DEPLOYMENT_ARMED"
    FROZEN = "FROZEN"


Actor = Literal["recovery_controller", "gitops_controller", "admission"]


@dataclass(frozen=True)
class RecoveryInputs:
    checkpoint_hash: str
    cursor_commit_id: str
    precision_contract_hash: str
    topology_epoch: str
    plan_digest: str
    work_manifest_digest: str
    git_revision: str

    def validate(self) -> None:
        hashes = (self.checkpoint_hash, self.precision_contract_hash, self.plan_digest, self.work_manifest_digest)
        if not self.cursor_commit_id or not self.topology_epoch or not self.git_revision:
            raise ArbitrationError("cursor/topology/Git bindings are required")
        for item in hashes:
            if len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item):
                raise ArbitrationError("input hash must be lowercase SHA-256")

    @property
    def digest(self) -> str:
        self.validate()
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EpochRecord:
    generation: int
    state: EpochState
    lease_holder: str
    lease_expiry_monotonic: float
    inputs: RecoveryInputs
    inputs_digest: str
    reason: str | None = None


class RecoveryDeploymentArbiter:
    """Single-writer state machine with explicit generation fencing.

    A caller must present `expected_generation` for every transition. A newer
    recovery generation invalidates all old worker and GitOps transition calls,
    even if they retained valid checkpoint bytes. The record treats Git desired
    revision as a recovery input; a revision drift freezes the current epoch
    rather than applying old or new desired state opportunistically.
    """

    def __init__(self, initial_inputs: RecoveryInputs) -> None:
        initial_inputs.validate()
        self._lock = Lock()
        self._record = EpochRecord(
            generation=0,
            state=EpochState.IDLE,
            lease_holder="",
            lease_expiry_monotonic=0.0,
            inputs=initial_inputs,
            inputs_digest=initial_inputs.digest,
        )
        self.fence_rejections: dict[tuple[str, str], int] = {}

    def snapshot(self) -> EpochRecord:
        with self._lock:
            return self._record

    def begin_recovery(self, *, actor: Actor, expected_generation: int, holder: str, inputs: RecoveryInputs, now: float, lease_seconds: float) -> EpochRecord:
        if actor != "recovery_controller" or not holder or lease_seconds <= 0:
            raise ArbitrationError("only recovery controller may create a valid recovery epoch")
        inputs.validate()
        with self._lock:
            self._require_generation(expected_generation, actor)
            current = self._record
            active_lease = current.state in {EpochState.RECOVERING, EpochState.RECOVERY_READY, EpochState.DEPLOYMENT_ARMED} and now < current.lease_expiry_monotonic
            if active_lease:
                self._reject(actor, "lease_expired")
                raise ArbitrationError("current recovery lease is active; do not run concurrent recovery")
            self._record = EpochRecord(
                generation=current.generation + 1,
                state=EpochState.RECOVERING,
                lease_holder=holder,
                lease_expiry_monotonic=now + lease_seconds,
                inputs=inputs,
                inputs_digest=inputs.digest,
            )
            return self._record

    def mark_recovery_ready(self, *, actor: Actor, expected_generation: int, holder: str, inputs: RecoveryInputs, now: float) -> EpochRecord:
        if actor != "recovery_controller":
            raise ArbitrationError("only recovery controller may mark state ready")
        inputs.validate()
        with self._lock:
            self._require_generation(expected_generation, actor)
            current = self._record
            if current.state != EpochState.RECOVERING or current.lease_holder != holder or now >= current.lease_expiry_monotonic:
                self._reject(actor, "lease_expired")
                raise ArbitrationError("recovery lease expired or state is not RECOVERING")
            if current.inputs_digest != inputs.digest:
                self._reject(actor, "input_binding_invalid")
                raise ArbitrationError("checkpoint/cursor/precision/topology/plan/Git input binding changed")
            self._record = replace(current, state=EpochState.RECOVERY_READY)
            return self._record

    def arm_deployment(self, *, actor: Actor, expected_generation: int, inputs: RecoveryInputs, now: float) -> EpochRecord:
        if actor != "gitops_controller":
            raise ArbitrationError("only GitOps controller may arm deployment")
        inputs.validate()
        with self._lock:
            self._require_generation(expected_generation, actor)
            current = self._record
            if current.state != EpochState.RECOVERY_READY or now >= current.lease_expiry_monotonic:
                self._reject(actor, "state_not_armed")
                raise ArbitrationError("deployment may only arm an unexpired RECOVERY_READY epoch")
            if current.inputs.git_revision != inputs.git_revision:
                self._reject(actor, "git_revision_mismatch")
                raise ArbitrationError("Git desired revision drifted; freeze and create a new epoch")
            if current.inputs_digest != inputs.digest:
                self._reject(actor, "input_binding_invalid")
                raise ArbitrationError("GitOps inputs differ from verified recovery binding")
            self._record = replace(current, state=EpochState.DEPLOYMENT_ARMED)
            return self._record

    def freeze_for_revision_drift(self, *, actor: Actor, expected_generation: int, observed_git_revision: str) -> EpochRecord:
        if actor not in {"gitops_controller", "admission"} or not observed_git_revision:
            raise ArbitrationError("only GitOps/admission may freeze a non-empty revision drift")
        with self._lock:
            self._require_generation(expected_generation, actor)
            current = self._record
            if observed_git_revision == current.inputs.git_revision:
                raise ArbitrationError("revision has not drifted")
            self._record = replace(current, state=EpochState.FROZEN, reason="git_revision_drift")
            return self._record

    def _require_generation(self, expected_generation: int, actor: Actor) -> None:
        if expected_generation != self._record.generation:
            self._reject(actor, "stale_generation")
            raise ArbitrationError("stale generation is fenced")

    def _reject(self, actor: str, reason: str) -> None:
        key = (actor, reason)
        self.fence_rejections[key] = self.fence_rejections.get(key, 0) + 1


def _inputs(label: str, git_revision: str) -> RecoveryInputs:
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return RecoveryInputs(
        checkpoint_hash=digest("checkpoint-" + label),
        cursor_commit_id="cursor-" + label,
        precision_contract_hash=digest("precision-bf16"),
        topology_epoch="topology-" + label,
        plan_digest=digest("plan-" + label),
        work_manifest_digest=digest("work-" + label),
        git_revision=git_revision,
    )


def run_smoke_test() -> None:
    """Exercise Git revision drift while old rendezvous callbacks arrive late."""
    old = _inputs("old", "git-sha-old")
    arbiter = RecoveryDeploymentArbiter(old)
    epoch1 = arbiter.begin_recovery(actor="recovery_controller", expected_generation=0, holder="recovery-a", inputs=old, now=0.0, lease_seconds=30.0)
    assert epoch1.generation == 1 and epoch1.state == EpochState.RECOVERING
    frozen = arbiter.freeze_for_revision_drift(actor="gitops_controller", expected_generation=1, observed_git_revision="git-sha-new")
    assert frozen.state == EpochState.FROZEN
    new = _inputs("new", "git-sha-new")
    epoch2 = arbiter.begin_recovery(actor="recovery_controller", expected_generation=1, holder="recovery-b", inputs=new, now=1.0, lease_seconds=30.0)
    assert epoch2.generation == 2 and epoch2.state == EpochState.RECOVERING
    try:
        arbiter.mark_recovery_ready(actor="recovery_controller", expected_generation=1, holder="recovery-a", inputs=old, now=2.0)
    except ArbitrationError:
        late_old_worker_fenced = True
    else:
        late_old_worker_fenced = False
    assert late_old_worker_fenced
    ready = arbiter.mark_recovery_ready(actor="recovery_controller", expected_generation=2, holder="recovery-b", inputs=new, now=2.0)
    assert ready.state == EpochState.RECOVERY_READY
    try:
        arbiter.arm_deployment(actor="gitops_controller", expected_generation=1, inputs=old, now=3.0)
    except ArbitrationError:
        late_gitops_fenced = True
    else:
        late_gitops_fenced = False
    assert late_gitops_fenced
    armed = arbiter.arm_deployment(actor="gitops_controller", expected_generation=2, inputs=new, now=3.0)
    assert armed.state == EpochState.DEPLOYMENT_ARMED
    assert arbiter.fence_rejections[("recovery_controller", "stale_generation")] == 1
    assert arbiter.fence_rejections[("gitops_controller", "stale_generation")] == 1
    rendered_rejections = {f"{actor}/{reason}": count for (actor, reason), count in arbiter.fence_rejections.items()}
    print(json.dumps({"smoke_test": "passed", "generation": armed.generation, "state": armed.state.value, "fence_rejections": rendered_rejections}, sort_keys=True), flush=True)


if __name__ == "__main__":
    run_smoke_test()
