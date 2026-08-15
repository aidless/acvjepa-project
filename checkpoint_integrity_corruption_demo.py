"""Offline Failpoint-5 checkpoint corruption walkthrough.

No sockets, RDMA, object storage, real KV, NCCL, GPU, or network controls are
used. The code demonstrates the recovery contract, not a production storage
implementation: corrupted cache data is untrusted acceleration, while the
committed pointer and durable content-addressed shard remain the source of
truth.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from checkpoint_cache_load_shedding_simulator import CacheAdmissionController, CheckpointLoadSheddingSimulator, SingleFlight
from distributed_training_observability import DistributedTrainingMetrics, TrainingMetricLabels
from elastic_data_cursor_ledger import CheckpointArtifact, ElasticCursorLedger, ElasticIdentity, GlobalWorkManifest, WorkWindow
from verified_checkpoint_cache import (
    CheckpointCacheContract,
    CheckpointManifest,
    DurableShardStore,
    InMemoryStrongCommitKV,
    ShardDescriptor,
    VerifiedReadThroughCache,
    sha256_bytes,
)


@dataclass(frozen=True)
class CorruptionDemoReport:
    committed_pointer_revision_before: int
    committed_pointer_revision_after: int
    cursor_next_offset_before: int
    cursor_next_offset_after: int
    cache_tier_state: str
    cache_corruption_events: int
    durable_fallback_reads: int
    coalesced_waiters: int
    follower_verified_cache_hits: int
    durable_admission_limit: int
    ttl_buckets: int


def _descriptor(component: str, shard_id: str, payload: bytes) -> ShardDescriptor:
    return ShardDescriptor(component=component, shard_id=shard_id, sha256=sha256_bytes(payload), byte_count=len(payload))


def run_demo(*, follower_count: int = 31) -> tuple[CorruptionDemoReport, str]:
    if follower_count < 1:
        raise ValueError("follower_count must be positive")
    started = time.monotonic()
    optimizer_bytes = b"adamw:exp_avg|exp_avg_sq|step=41|committed"
    descriptor = _descriptor("optimizer", "rank-shard-0", optimizer_bytes)
    contract = CheckpointCacheContract(
        namespace="isolated-preproduction/acvjepa",
        checkpoint_hash=sha256_bytes(b"checkpoint-step-41"),
        precision_contract_hash=sha256_bytes(b"bf16:torch_amp:adamw"),
        dataset_commit="dataset-demo",
        manifest_digest=sha256_bytes(b"global-work-manifest-demo"),
    )
    manifest = CheckpointManifest(contract=contract, shards=(descriptor,), created_ns=1)
    durable = DurableShardStore()
    durable.put(descriptor, optimizer_bytes)
    pointer_kv = InMemoryStrongCommitKV()
    pointer = pointer_kv.compare_and_swap(run_key="run-demo", expected_revision=0, manifest=manifest, committed_step=41)
    pointer_before = pointer_kv.get_linearizable("run-demo")

    # This separate ledger models the training cursor. No update reservation is
    # committed in this failpoint path, so the only legal cursor value is genesis.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint_file = root / "genesis.ckpt"
        checkpoint_file.write_bytes(b"genesis-checkpoint")
        artifact_digest = sha256_bytes(checkpoint_file.read_bytes())
        checkpoint = CheckpointArtifact(
            uri=checkpoint_file.resolve().as_uri(),
            sha256=artifact_digest,
            model_state_hash=sha256_bytes(b"model"),
            ema_state_hash=sha256_bytes(b"ema"),
            optimizer_state_hash=sha256_bytes(b"optimizer"),
            validation_hash=sha256_bytes(b"validation"),
        )
        work_manifest = GlobalWorkManifest(
            dataset_commit="dataset-demo",
            ordered_windows=tuple(WorkWindow(f"work-{idx}", sha256_bytes(f"source-{idx}".encode()), 1.0) for idx in range(3)),
        )
        identity = ElasticIdentity("run-demo", 0, 2, "epoch-0", "topology-0")
        ledger = ElasticCursorLedger(str(root / "cursor.sqlite"))
        ledger.bootstrap(run_key="run-demo", manifest=work_manifest, checkpoint=checkpoint, identity=identity)
        cursor_before = ledger.latest_committed("run-demo").next_offset

        metrics = DistributedTrainingMetrics(
            TrainingMetricLabels(cluster="local-compose", job="acvjepa-corruption-demo", environment="isolated-preproduction")
        )
        metrics.record_failpoint(
            fault_class="cache_corruption_during_restore",
            phase="trigger_to_detect",
            duration_seconds=0.0,
            active=True,
        )

        cache = VerifiedReadThroughCache(ttl_seconds=60)
        # Prime a verified acceleration copy, then corrupt only that cache entry.
        primed, primed_source = cache.fetch_verified(pointer=pointer, manifest=manifest, descriptor=descriptor, durable=durable, now_ns=10)
        assert primed == optimizer_bytes and primed_source == "durable"
        cache.inject_corruption_for_test(contract=contract, descriptor=descriptor, corrupted=b"TAMPERED:optimizer-state")

        # Open only the *cache tier* circuit. Durable storage remains available
        # behind bounded admission; pointer/cursor writes are not part of this path.
        cache_tier_state = "OPEN_SUSPECT_READS_BLOCKED"
        admission = CacheAdmissionController(durable_limit=2, per_node_limit=1)
        singleflight = SingleFlight()
        shard_key = descriptor.cache_key
        assert singleflight.acquire_or_join(shard_key), "first reader must become leader"
        assert admission.begin_durable("node-0"), "leader must obtain one durable token"

        # Followers arrive while the leader is still reading. They join the
        # immutable shard key rather than open duplicate durable reads.
        coalesced_waiters = sum(1 for _ in range(follower_count) if not singleflight.acquire_or_join(shard_key))
        assert coalesced_waiters == follower_count

        # The leader detects the hash mismatch, evicts the cache entry inside
        # fetch_verified(), reads durable bytes, verifies them, and only then rewarms cache.
        leader_payload, leader_source = cache.fetch_verified(
            pointer=pointer, manifest=manifest, descriptor=descriptor, durable=durable, now_ns=11
        )
        admission.finish_durable("node-0")
        singleflight.complete(shard_key)
        assert leader_payload == optimizer_bytes and leader_source == "durable"
        metrics.record_cache_fetch(
            cache_tier="node_local", component_class="optimizer", outcome="integrity_failed", byte_count=0, duration_seconds=0.0
        )
        metrics.record_cache_fetch(
            cache_tier="durable", component_class="optimizer", outcome="fallback", byte_count=len(optimizer_bytes), duration_seconds=0.0
        )

        # The waiting readers wake after the leader rewarms and independently
        # reverify the cache entry before consuming it.
        follower_hits = 0
        for index in range(follower_count):
            payload, source = cache.fetch_verified(pointer=pointer, manifest=manifest, descriptor=descriptor, durable=durable, now_ns=12 + index)
            assert payload == optimizer_bytes and source == "cache"
            follower_hits += 1
            metrics.record_cache_fetch(
                cache_tier="node_local", component_class="optimizer", outcome="hit", byte_count=len(payload), duration_seconds=0.0
            )

        # Deterministic jitter prevents unrelated shard expiries from forming a synchronized retry wave.
        jitter = {
            CheckpointLoadSheddingSimulator.ttl_with_deterministic_jitter(key=f"optimizer-shard-{idx}", base_ticks=100, jitter_ratio=0.2)
            for idx in range(64)
        }
        pointer_after = pointer_kv.get_linearizable("run-demo")
        cursor_after = ledger.latest_committed("run-demo").next_offset
        metrics.record_failpoint(
            fault_class="cache_corruption_during_restore",
            phase="trigger_to_training_ready",
            duration_seconds=time.monotonic() - started,
            active=False,
            outcome="passed",
        )

        assert cache.stats["corrupt"] == 1
        assert cache.stats["durable_fallback"] == 2  # initial warm + corruption recovery; only one recovery fallback.
        assert pointer_after == pointer_before
        assert cursor_after == cursor_before == 0
        assert admission.durable.inflight == 0 and admission.durable.limit == 2
        assert coalesced_waiters == follower_count and follower_hits == follower_count and len(jitter) > 1
        report = CorruptionDemoReport(
            committed_pointer_revision_before=pointer_before.revision,
            committed_pointer_revision_after=pointer_after.revision,
            cursor_next_offset_before=cursor_before,
            cursor_next_offset_after=cursor_after,
            cache_tier_state=cache_tier_state,
            cache_corruption_events=cache.stats["corrupt"],
            durable_fallback_reads=1,
            coalesced_waiters=coalesced_waiters,
            follower_verified_cache_hits=follower_hits,
            durable_admission_limit=admission.durable.limit,
            ttl_buckets=len(jitter),
        )
        ledger.close()
        return report, metrics.exposition()


def main() -> None:
    report, exposition = run_demo()
    payload = asdict(report)
    assert payload["committed_pointer_revision_before"] == payload["committed_pointer_revision_after"] == 1
    assert payload["cursor_next_offset_before"] == payload["cursor_next_offset_after"] == 0
    assert payload["cache_corruption_events"] == 1
    assert payload["durable_fallback_reads"] == 1
    assert payload["coalesced_waiters"] == payload["follower_verified_cache_hits"] == 31
    assert "acvjepa_training_checkpoint_cache_fetches_total" in exposition
    print(json.dumps({"smoke_test": "passed", "report": payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
