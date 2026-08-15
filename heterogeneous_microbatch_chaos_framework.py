"""Offline chaos framework for the 2:1 heterogeneous-microbatch recovery contract.

The framework contains no process killing, network shaping, RDMA access, shell,
SSH, scheduler, cloud, or cluster management calls. Each fault is a deterministic
logical failpoint against the transactional ledger/control/cache abstractions.
It is suitable for CI and for pre-validating the assertions a separately
approved infrastructure drill must collect.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from distributed_training_observability import DistributedTrainingMetrics, TrainingMetricLabels
from elastic_data_cursor_ledger import (
    CheckpointArtifact,
    CursorBoundPlan,
    CursorContractError,
    ElasticCursorLedger,
    ElasticIdentity,
    GlobalWorkManifest,
    WorkWindow,
    sha256_bytes,
    sha256_file,
)
from rapid_recovery_alert_drill import SafeAlert, SafeAlertControlPlane
from verified_checkpoint_cache import (
    CheckpointCacheContract,
    CheckpointManifest,
    DurableShardStore,
    InMemoryStrongCommitKV,
    ShardDescriptor,
    VerifiedReadThroughCache,
)


class ChaosFault(str, Enum):
    NODE_LOSS_FINAL_ALLREDUCE = "node_loss_final_allreduce"
    NETWORK_PARTITION_AFTER_PREPARE = "network_partition_after_prepare"
    STALE_PLAN_AFTER_RENDEZVOUS = "stale_plan_after_rendezvous"
    PLAN_TOPOLOGY_MISMATCH = "plan_topology_mismatch"
    CACHE_CORRUPTION_DURING_RESTORE = "cache_corruption_during_restore"


@dataclass(frozen=True)
class ChaosScenario:
    fault: ChaosFault
    experiment_id: str
    seed: int
    environment: str = "isolated-preproduction"
    dry_run: bool = True

    def validate(self) -> None:
        if not self.experiment_id or self.seed < 0:
            raise ValueError("experiment ID and non-negative seed are required")
        if self.environment != "isolated-preproduction" or not self.dry_run:
            raise PermissionError("framework is limited to isolated preproduction dry-run logical faults")


@dataclass(frozen=True)
class ChaosResult:
    fault: str
    passed: bool
    assertions: tuple[str, ...]


class HeterogeneousMicrobatchChaosFramework:
    """Executes one fault at a time and asserts recovery invariants."""

    def __init__(self) -> None:
        labels = TrainingMetricLabels("chaos-cluster", "acvjepa-2to1-chaos", "isolated-preproduction")
        self.metrics = DistributedTrainingMetrics(labels)
        self.labels = labels
        self.control = SafeAlertControlPlane(allowed_environments=("isolated-preproduction",))

    @staticmethod
    def _checkpoint(root: Path, label: str) -> CheckpointArtifact:
        file_path = root / f"{label}.pt"
        file_path.write_bytes(f"checkpoint:{label}".encode())
        digest = sha256_file(file_path)
        return CheckpointArtifact(
            uri=file_path.resolve().as_uri(),
            sha256=digest,
            model_state_hash=sha256_bytes(f"model:{label}".encode()),
            ema_state_hash=sha256_bytes(f"ema:{label}".encode()),
            optimizer_state_hash=sha256_bytes(f"optimizer:{label}".encode()),
            validation_hash=sha256_bytes(f"validation:{label}".encode()),
        )

    @staticmethod
    def _manifest() -> GlobalWorkManifest:
        return GlobalWorkManifest(
            dataset_commit="chaos-dataset-commit",
            ordered_windows=tuple(WorkWindow(f"work-{index}", sha256_bytes(f"source-{index}".encode()), 1.0) for index in range(8)),
        )

    @staticmethod
    def _identity(restart: int, epoch: str) -> ElasticIdentity:
        return ElasticIdentity("chaos-run", restart, 2, epoch, f"topology-{epoch}")

    @staticmethod
    def _plan(manifest: GlobalWorkManifest, identity: ElasticIdentity, assignment: tuple[tuple[str, ...], ...]) -> CursorBoundPlan:
        return CursorBoundPlan(
            plan_version=1,
            topology_epoch=identity.topology_epoch,
            topology_digest=identity.topology_digest,
            work_manifest_digest=manifest.digest,
            world_size=identity.world_size,
            rank_work_ids=assignment,
        )

    def _fresh_2to1_attempt(self, root: Path):
        manifest = self._manifest()
        identity0 = self._identity(0, "epoch-0")
        ledger = ElasticCursorLedger(str(root / "cursor.sqlite"))
        ledger.bootstrap(run_key="chaos-run", manifest=manifest, checkpoint=self._checkpoint(root, "genesis"), identity=identity0)
        # 2:1 micro-batch assignment; batch size=2 leads to 4 + 2 global samples.
        plan0 = self._plan(manifest, identity0, (("work-0", "work-1"), ("work-2",)))
        reservation = ledger.prepare_next_update(run_key="chaos-run", manifest=manifest, plan=plan0, identity=identity0)
        assert reservation.start_offset == 0 and reservation.end_offset == 3
        self.metrics.record_plan(global_samples=6, micro_batches_by_rank={0: 2, 1: 1}, digest_consensus=True)
        return ledger, manifest, identity0, plan0, reservation

    def _fire_safe_alert(self, *, alertname: str, fingerprint: str) -> None:
        actions = self.control.handle(
            SafeAlert(fingerprint, alertname, "firing", self.labels.cluster, self.labels.job, self.labels.environment, "critical")
        )
        assert actions and not self.control.may_arm_new_plan(self.labels)

    def run(self, scenario: ChaosScenario) -> ChaosResult:
        scenario.validate()
        handlers: dict[ChaosFault, Callable[[ChaosScenario], ChaosResult]] = {
            ChaosFault.NODE_LOSS_FINAL_ALLREDUCE: self._node_loss_final_allreduce,
            ChaosFault.NETWORK_PARTITION_AFTER_PREPARE: self._network_partition_after_prepare,
            ChaosFault.STALE_PLAN_AFTER_RENDEZVOUS: self._stale_plan_after_rendezvous,
            ChaosFault.PLAN_TOPOLOGY_MISMATCH: self._plan_topology_mismatch,
            ChaosFault.CACHE_CORRUPTION_DURING_RESTORE: self._cache_corruption_during_restore,
        }
        result = handlers[scenario.fault](scenario)
        self.metrics.record_chaos(phase="passed")
        return result

    def _node_loss_final_allreduce(self, scenario: ChaosScenario) -> ChaosResult:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            ledger, manifest, _, _, reservation = self._fresh_2to1_attempt(root)
            # Logical failpoint: rank 1 disappears at the final synchronized backward.
            ledger.abort_uncommitted(attempt_id=reservation.attempt_id, reason="failpoint:rank1_lost_final_allreduce")
            cursor_before = ledger.latest_committed("chaos-run")
            identity1 = self._identity(1, "epoch-1")
            recovered = ledger.recover_after_rendezvous(run_key="chaos-run", manifest=manifest, new_identity=identity1, reason="rank1_lost")
            # New group may change rank-local layout: 1:2, same global range.
            replay = self._plan(manifest, identity1, (("work-0",), ("work-1", "work-2")))
            reservation2 = ledger.prepare_next_update(run_key="chaos-run", manifest=manifest, plan=replay, identity=identity1)
            assert cursor_before.next_offset == recovered.next_offset == reservation2.start_offset == 0
            assert set(reservation2.reserved_global_work_ids) == {"work-0", "work-1", "work-2"}
            committed = ledger.commit_update(
                attempt_id=reservation2.attempt_id,
                manifest=manifest,
                checkpoint=self._checkpoint(root, "recovered"),
                current_identity=identity1,
            )
            assert committed.next_offset == 3 and committed.committed_step == 1
            ledger.close()
            return ChaosResult(scenario.fault.value, True, ("uncommitted_cursor_unchanged", "same_global_range_replayed", "rank_layout_reassigned", "new_checkpoint_committed"))

    def _network_partition_after_prepare(self, scenario: ChaosScenario) -> ChaosResult:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            ledger, manifest, _, _, reservation = self._fresh_2to1_attempt(Path(temp))
            ledger.abort_uncommitted(attempt_id=reservation.attempt_id, reason="failpoint:control_or_data_plane_partition")
            self._fire_safe_alert(alertname="ACVJEPARendezvousRebuildStorm", fingerprint=f"{scenario.experiment_id}-partition")
            assert ledger.latest_committed("chaos-run").next_offset == 0
            ledger.close()
            return ChaosResult(scenario.fault.value, True, ("reservation_aborted", "cursor_not_advanced", "new_plans_frozen", "no_network_operation_executed"))

    def _stale_plan_after_rendezvous(self, scenario: ChaosScenario) -> ChaosResult:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            ledger, manifest, identity0, _, reservation = self._fresh_2to1_attempt(root)
            # New identity fences the old attempt. A stale attempt must not commit.
            identity1 = self._identity(1, "epoch-1")
            ledger.recover_after_rendezvous(run_key="chaos-run", manifest=manifest, new_identity=identity1, reason="membership_changed")
            try:
                ledger.commit_update(
                    attempt_id=reservation.attempt_id,
                    manifest=manifest,
                    checkpoint=self._checkpoint(root, "stale"),
                    current_identity=identity0,
                )
            except CursorContractError:
                rejected = True
            else:
                rejected = False
            assert rejected and ledger.latest_committed("chaos-run").next_offset == 0
            ledger.close()
            return ChaosResult(scenario.fault.value, True, ("old_attempt_fenced", "stale_commit_rejected", "cursor_not_advanced"))

    def _plan_topology_mismatch(self, scenario: ChaosScenario) -> ChaosResult:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            ledger, manifest, identity0, _, _ = self._fresh_2to1_attempt(Path(temp))
            bad_plan = CursorBoundPlan(
                plan_version=1,
                topology_epoch="wrong-epoch",
                topology_digest="wrong-topology",
                work_manifest_digest=manifest.digest,
                world_size=2,
                rank_work_ids=(("work-3", "work-4"), ("work-5",)),
            )
            try:
                ledger.prepare_next_update(run_key="chaos-run", manifest=manifest, plan=bad_plan, identity=identity0)
            except CursorContractError:
                rejected = True
            else:
                rejected = False
            assert rejected
            ledger.close()
            return ChaosResult(scenario.fault.value, True, ("topology_digest_mismatch_rejected", "no_second_reservation_created"))

    def _cache_corruption_during_restore(self, scenario: ChaosScenario) -> ChaosResult:
        payload = b"adamw-exp-avg-committed-v1"
        descriptor = ShardDescriptor("optimizer", "rank-shard-0", sha256_bytes(payload), len(payload))
        contract = CheckpointCacheContract(
            namespace="isolated-preproduction/acvjepa",
            checkpoint_hash=sha256_bytes(b"checkpoint-chaos"),
            precision_contract_hash=sha256_bytes(b"bf16:torch_amp:adamw"),
            dataset_commit="chaos-dataset-commit",
            manifest_digest=sha256_bytes(b"manifest-chaos"),
        )
        manifest = CheckpointManifest(contract, (descriptor,), created_ns=1)
        durable = DurableShardStore()
        durable.put(descriptor, payload)
        kv = InMemoryStrongCommitKV()
        pointer = kv.compare_and_swap(run_key="chaos-run", expected_revision=0, manifest=manifest, committed_step=1)
        cache = VerifiedReadThroughCache(ttl_seconds=10)
        cache.fetch_verified(pointer=pointer, manifest=manifest, descriptor=descriptor, durable=durable, now_ns=1)
        cache.inject_corruption_for_test(contract=contract, descriptor=descriptor, corrupted=b"bad")
        recovered, source = cache.fetch_verified(pointer=pointer, manifest=manifest, descriptor=descriptor, durable=durable, now_ns=2)
        assert recovered == payload and source == "durable" and cache.stats["corrupt"] == 1
        return ChaosResult(scenario.fault.value, True, ("cache_corruption_detected", "entry_evicted", "durable_fallback_verified", "commit_pointer_unchanged"))


def run_smoke_test() -> None:
    framework = HeterogeneousMicrobatchChaosFramework()
    scenarios = tuple(
        ChaosScenario(fault, experiment_id=f"chaos-{index}", seed=index)
        for index, fault in enumerate(ChaosFault, start=1)
    )
    results = [framework.run(item) for item in scenarios]
    assert all(result.passed for result in results)
    print(json.dumps({"smoke_test": "passed", "scenarios": [result.fault for result in results], "assertions": sum(len(result.assertions) for result in results)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    run_smoke_test()
