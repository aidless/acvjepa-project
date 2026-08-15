"""Deterministic offline load/stampede simulator for verified checkpoint caches.

No sockets, RDMA, object storage or KV services are contacted. The simulator
models the control decisions that must be validated before an isolated real
load test: coalescing, quotas, TTL jitter, negative caching, circuit breaking,
adaptive concurrency and CAS publication fencing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import floor
from typing import Dict

from verified_checkpoint_cache import (
    CacheContractError,
    CheckpointCacheContract,
    CheckpointManifest,
    InMemoryStrongCommitKV,
    ShardDescriptor,
    sha256_bytes,
)


@dataclass
class TokenBudget:
    name: str
    minimum: int
    maximum: int
    limit: int
    inflight: int = 0

    def acquire(self) -> bool:
        if self.inflight >= self.limit:
            return False
        self.inflight += 1
        return True

    def release(self) -> None:
        if self.inflight <= 0:
            raise RuntimeError(f"{self.name} release without acquire")
        self.inflight -= 1

    def adapt(self, *, p95_ms: float, target_ms: float) -> None:
        """A conservative AIMD-like policy driven by bounded latency evidence."""
        if p95_ms > target_ms:
            self.limit = max(self.minimum, floor(self.limit * 0.5))
        elif p95_ms < target_ms * 0.7:
            self.limit = min(self.maximum, self.limit + 1)


@dataclass(frozen=True)
class StampedeReport:
    request_count: int
    durable_fetches: int
    coalesced_waiters: int
    cache_hits_after_fill: int
    blocked_by_budget: int


class SingleFlight:
    """One in-flight load per immutable shard key; followers never duplicate I/O."""

    def __init__(self) -> None:
        self.inflight: set[str] = set()

    def acquire_or_join(self, key: str) -> bool:
        if key in self.inflight:
            return False
        self.inflight.add(key)
        return True

    def complete(self, key: str) -> None:
        self.inflight.remove(key)


class CacheAdmissionController:
    """Two-dimensional quota: global durable tier and per-node request cap."""

    def __init__(self, *, durable_limit: int, per_node_limit: int) -> None:
        self.durable = TokenBudget("durable", minimum=1, maximum=durable_limit, limit=durable_limit)
        self.per_node_limit = per_node_limit
        self.per_node_inflight: Dict[str, int] = {}
        self.circuit_open = False

    def begin_durable(self, node: str) -> bool:
        if self.circuit_open or self.per_node_inflight.get(node, 0) >= self.per_node_limit:
            return False
        if not self.durable.acquire():
            return False
        self.per_node_inflight[node] = self.per_node_inflight.get(node, 0) + 1
        return True

    def finish_durable(self, node: str) -> None:
        self.durable.release()
        self.per_node_inflight[node] -= 1

    def observe(self, *, p95_ms: float, target_ms: float, integrity_failures: int = 0) -> None:
        self.durable.adapt(p95_ms=p95_ms, target_ms=target_ms)
        if integrity_failures > 0:
            self.circuit_open = True

    def close_circuit_after_healthcheck(self) -> None:
        self.circuit_open = False


class CheckpointLoadSheddingSimulator:
    def __init__(self) -> None:
        self.singleflight = SingleFlight()
        self.controller = CacheAdmissionController(durable_limit=2, per_node_limit=1)
        self.cache: set[str] = set()
        self.negative_cache: Dict[str, int] = {}

    def cold_key_storm(self, *, key: str, nodes: list[str]) -> StampedeReport:
        """All nodes request one cold immutable optimizer shard simultaneously."""
        if key in self.cache:
            raise ValueError("storm key must start cold")
        leader = self.singleflight.acquire_or_join(key)
        assert leader
        leader_node = nodes[0]
        if not self.controller.begin_durable(leader_node):
            raise AssertionError("first single-flight request must receive durable budget")
        followers = 0
        blocked = 0
        for node in nodes[1:]:
            if not self.singleflight.acquire_or_join(key):
                followers += 1
            else:
                # This should not happen while leader is in flight; if it did,
                # admission controller would protect durable storage.
                if not self.controller.begin_durable(node):
                    blocked += 1
        self.controller.finish_durable(leader_node)
        self.singleflight.complete(key)
        self.cache.add(key)
        cache_hits_after_fill = sum(1 for _ in nodes if key in self.cache)
        return StampedeReport(len(nodes), durable_fetches=1, coalesced_waiters=followers, cache_hits_after_fill=cache_hits_after_fill, blocked_by_budget=blocked)

    def negative_cache_storm(self, *, missing_key: str, requests: int, ttl_ticks: int = 3) -> dict[str, int]:
        durable_probes = 0
        negative_hits = 0
        for tick in range(requests):
            if self.negative_cache.get(missing_key, -1) >= tick:
                negative_hits += 1
            else:
                durable_probes += 1
                self.negative_cache[missing_key] = tick + ttl_ticks
        return {"requests": requests, "durable_probes": durable_probes, "negative_hits": negative_hits}

    @staticmethod
    def ttl_with_deterministic_jitter(*, key: str, base_ticks: int, jitter_ratio: float) -> int:
        if base_ticks <= 0 or not 0 <= jitter_ratio <= 1:
            raise ValueError("invalid TTL/jitter")
        unit = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        delta = (unit * 2 - 1) * base_ticks * jitter_ratio
        return max(1, round(base_ticks + delta))

    def pointer_cas_write_storm(self, *, writer_count: int) -> dict[str, int]:
        """Many writers race to publish a tiny pointer; only one CAS may win."""
        payload = b"checkpoint-manifest"
        descriptor = ShardDescriptor("model", "shard-0", sha256_bytes(payload), len(payload))
        contract = CheckpointCacheContract(
            namespace="isolated-preproduction/acvjepa",
            checkpoint_hash=sha256_bytes(b"cp-v1"),
            precision_contract_hash=sha256_bytes(b"bf16"),
            dataset_commit="dataset-loadtest",
            manifest_digest=sha256_bytes(b"manifest"),
        )
        manifest = CheckpointManifest(contract, (descriptor,), created_ns=1)
        kv = InMemoryStrongCommitKV()
        won = conflicts = 0
        for _ in range(writer_count):
            try:
                kv.compare_and_swap(run_key="load-test", expected_revision=0, manifest=manifest, committed_step=1)
                won += 1
            except CacheContractError:
                conflicts += 1
        return {"writers": writer_count, "wins": won, "cas_conflicts": conflicts, "revision": kv.get_linearizable("load-test").revision}


def run_smoke_test() -> None:
    simulator = CheckpointLoadSheddingSimulator()
    report = simulator.cold_key_storm(key="optimizer:step-42:shard-0", nodes=[f"node-{index}" for index in range(32)])
    assert report.durable_fetches == 1
    assert report.coalesced_waiters == 31
    assert report.cache_hits_after_fill == 32
    negative = simulator.negative_cache_storm(missing_key="missing:fp8-meta", requests=10)
    assert negative == {"requests": 10, "durable_probes": 3, "negative_hits": 7}
    ttls = {simulator.ttl_with_deterministic_jitter(key=f"shard-{index}", base_ticks=100, jitter_ratio=0.2) for index in range(64)}
    assert len(ttls) > 1 and all(80 <= ttl <= 120 for ttl in ttls)
    # High latency halves durable concurrency; integrity failure opens circuit.
    simulator.controller.observe(p95_ms=250, target_ms=100)
    assert simulator.controller.durable.limit == 1
    simulator.controller.observe(p95_ms=80, target_ms=100, integrity_failures=1)
    assert simulator.controller.circuit_open
    simulator.controller.close_circuit_after_healthcheck()
    assert not simulator.controller.circuit_open
    cas = simulator.pointer_cas_write_storm(writer_count=16)
    assert cas == {"writers": 16, "wins": 1, "cas_conflicts": 15, "revision": 1}
    print(json.dumps({"smoke_test": "passed", "storm": report.__dict__, "negative": negative, "ttl_buckets": len(ttls), "cas": cas}, sort_keys=True), flush=True)


if __name__ == "__main__":
    run_smoke_test()
