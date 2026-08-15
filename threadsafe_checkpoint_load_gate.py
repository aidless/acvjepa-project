"""Thread-safe single-flight and load-shedding gate for immutable checkpoint shards.

This module is a same-process reference implementation. It does not connect to
RDMA, object storage, Kubernetes, a distributed KV, or NCCL. Across processes
or nodes, replace the in-memory flight map with a strongly-consistent lease and
fencing token, while retaining the same pointer/hash/precision/cursor gates.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition, Event, Lock
from typing import Callable, Literal


class LoadGateError(RuntimeError):
    pass


class LoadTimeout(LoadGateError):
    pass


class CacheCircuitOpen(LoadGateError):
    pass


Source = Literal["cache", "durable_leader", "singleflight_follower"]


@dataclass(frozen=True)
class LoadResult:
    payload: bytes
    source: Source


@dataclass
class GateStats:
    cache_hits: int = 0
    cache_corruptions: int = 0
    leaders: int = 0
    followers: int = 0
    durable_loads: int = 0
    durable_waits: int = 0
    failures: int = 0
    max_durable_inflight: int = 0


@dataclass
class _Flight:
    generation: int
    deadline: float
    done: bool = False
    payload: bytes | None = None
    error: BaseException | None = None


class ThreadSafeAdmission:
    """A dynamically adjustable, bounded durable-load budget.

    `acquire()` and `release()` share one Condition. The counter, limit and
    timeout check are evaluated while holding the same lock, so two threads
    cannot both observe a spare token and over-admit durable I/O.
    """

    def __init__(self, *, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self._condition = Condition(Lock())
        self._limit = limit
        self._inflight = 0
        self.max_inflight = 0

    def set_limit(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._condition:
            self._limit = limit
            self._condition.notify_all()

    def acquire(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._inflight >= self._limit:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LoadTimeout("durable admission timed out")
                self._condition.wait(remaining)
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)

    def release(self) -> None:
        with self._condition:
            if self._inflight <= 0:
                raise RuntimeError("durable admission release without acquire")
            self._inflight -= 1
            self._condition.notify_all()


class ThreadSafeVerifiedShardGate:
    """Per-key single-flight + verified cache + bounded durable admission.

    The lock protects cache map mutation, per-key flight ownership, circuit
    state and terminal result publication. Expensive durable I/O and SHA-256
    verification happen *outside* the lock so unrelated shard keys can still
    coordinate independently. A follower receives only the leader's already
    verified result or the same terminal error; it never re-runs `loader`.
    """

    def __init__(self, *, durable_limit: int) -> None:
        self._condition = Condition(Lock())
        self._cache: dict[str, bytes] = {}
        self._flights: dict[str, _Flight] = {}
        self._cache_reads_enabled = True
        self._next_generation = 0
        self.admission = ThreadSafeAdmission(limit=durable_limit)
        self.stats = GateStats()

    def warm_verified(self, *, key: str, payload: bytes, expected_sha256: str) -> None:
        self._verify(payload, expected_sha256)
        with self._condition:
            self._cache[key] = bytes(payload)

    def quarantine_cache_reads(self) -> None:
        """Disable cache as a source; durable verified reads remain admissible."""
        with self._condition:
            self._cache_reads_enabled = False

    def reopen_cache_reads_after_healthcheck(self) -> None:
        with self._condition:
            self._cache_reads_enabled = True
            self._condition.notify_all()

    def load(
        self,
        *,
        key: str,
        expected_sha256: str,
        loader: Callable[[], bytes],
        timeout_seconds: float = 5.0,
    ) -> LoadResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not key or len(expected_sha256) != 64:
            raise ValueError("key and SHA-256 are required")
        deadline = time.monotonic() + timeout_seconds

        with self._condition:
            cached = self._cache.get(key) if self._cache_reads_enabled else None
            if cached is not None:
                try:
                    self._verify(cached, expected_sha256)
                except LoadGateError:
                    # Conditional delete avoids deleting a newer re-warmed entry.
                    if self._cache.get(key) == cached:
                        del self._cache[key]
                    self.stats.cache_corruptions += 1
                else:
                    self.stats.cache_hits += 1
                    return LoadResult(cached, "cache")

            flight = self._flights.get(key)
            # A dead or stuck leader cannot own a key forever. Fence it under
            # the same lock before a new generation is admitted. The old leader
            # will later fail the identity check before it can publish cache data.
            if flight is not None and not flight.done and time.monotonic() >= flight.deadline:
                flight.error = LoadTimeout("single-flight leader lease expired")
                flight.done = True
                self._flights.pop(key, None)
                self._condition.notify_all()
                flight = None
            if flight is None:
                self._next_generation += 1
                flight = _Flight(generation=self._next_generation, deadline=deadline)
                self._flights[key] = flight
                self.stats.leaders += 1
                leader = True
            else:
                self.stats.followers += 1
                leader = False

            if not leader:
                while not flight.done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LoadTimeout("single-flight follower timed out")
                    self._condition.wait(remaining)
                if flight.error is not None:
                    raise LoadGateError("single-flight leader failed") from flight.error
                assert flight.payload is not None
                return LoadResult(flight.payload, "singleflight_follower")

        # Only this leader performs potentially slow durable I/O. The admission
        # token is released in finally even when loader or verification fails.
        try:
            before = time.monotonic()
            self.admission.acquire(max(0.001, deadline - before))
            try:
                waited = time.monotonic() - before > 0.001
                payload = bytes(loader())
                self._verify(payload, expected_sha256)
                if time.monotonic() > deadline:
                    raise LoadTimeout("single-flight leader exceeded its recovery deadline")
                with self._condition:
                    self.stats.durable_waits += int(waited)
                    self.stats.durable_loads += 1
                    self.stats.max_durable_inflight = max(self.stats.max_durable_inflight, self.admission.max_inflight)
            finally:
                self.admission.release()
        except BaseException as exc:
            with self._condition:
                flight.error = exc
                flight.done = True
                self._flights.pop(key, None)
                self.stats.failures += 1
                self._condition.notify_all()
            raise

        with self._condition:
            if self._flights.get(key) is not flight:
                # The lease expired and a newer flight was admitted. A stale
                # leader cannot overwrite the newer cache generation.
                raise LoadTimeout("single-flight leader fenced by newer generation")
            # Rewarm only verified payload. Pointer/CAS/cursor are intentionally
            # absent: this method is an acceleration layer, not a commit path.
            self._cache[key] = payload
            flight.payload = payload
            flight.done = True
            self._flights.pop(key, None)
            self._condition.notify_all()
        return LoadResult(payload, "durable_leader")

    @staticmethod
    def _verify(payload: bytes, expected_sha256: str) -> None:
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise LoadGateError("checkpoint shard integrity verification failed")


def run_smoke_test() -> None:
    """32 threads hit one corrupted cache key; only one durable load is allowed."""
    payload = b"immutable-optimizer-shard-v1"
    digest = hashlib.sha256(payload).hexdigest()
    gate = ThreadSafeVerifiedShardGate(durable_limit=1)
    gate.warm_verified(key="optimizer:shard-0", payload=payload, expected_sha256=digest)
    # Test-only corruption bypasses warm_verified to emulate an untrusted tier.
    gate._cache["optimizer:shard-0"] = b"corrupted"
    start = Event()
    loader_calls = 0
    loader_lock = Lock()

    def loader() -> bytes:
        nonlocal loader_calls
        with loader_lock:
            loader_calls += 1
        time.sleep(0.02)
        return payload

    def reader() -> LoadResult:
        start.wait(timeout=1)
        return gate.load(key="optimizer:shard-0", expected_sha256=digest, loader=loader, timeout_seconds=2)

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(reader) for _ in range(32)]
        start.set()
        results = [future.result(timeout=3) for future in futures]

    sources = [result.source for result in results]
    assert loader_calls == 1
    assert sources.count("durable_leader") == 1
    assert sources.count("singleflight_follower") == 31
    assert gate.stats.cache_corruptions == 1
    assert gate.stats.max_durable_inflight == 1
    assert all(result.payload == payload for result in results)
    # A later reader gets a verified cache hit; it does not touch durable I/O.
    final = gate.load(key="optimizer:shard-0", expected_sha256=digest, loader=loader)
    assert final.source == "cache" and loader_calls == 1

    # Single-flight is intentionally per immutable key. Different shards may
    # have different leaders, but the shared admission budget still serializes
    # durable I/O when its limit is one.
    key_a, payload_a = "optimizer:shard-a", b"optimizer-a"
    key_b, payload_b = "optimizer:shard-b", b"optimizer-b"
    digest_a = hashlib.sha256(payload_a).hexdigest()
    digest_b = hashlib.sha256(payload_b).hexdigest()
    start_two = Event()
    gate_two = ThreadSafeVerifiedShardGate(durable_limit=1)

    def read_distinct(key: str, expected: str, value: bytes) -> LoadResult:
        start_two.wait(timeout=1)
        return gate_two.load(key=key, expected_sha256=expected, loader=lambda: (time.sleep(0.02), value)[1], timeout_seconds=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(read_distinct, key_a, digest_a, payload_a)
        future_b = pool.submit(read_distinct, key_b, digest_b, payload_b)
        start_two.set()
        assert future_a.result(timeout=3).source == "durable_leader"
        assert future_b.result(timeout=3).source == "durable_leader"
    assert gate_two.stats.durable_loads == 2 and gate_two.stats.max_durable_inflight == 1

    print(
        {
            "smoke_test": "passed",
            "durable_loader_calls": loader_calls,
            "leader": sources.count("durable_leader"),
            "followers": sources.count("singleflight_follower"),
            "cache_corruptions": gate.stats.cache_corruptions,
            "max_durable_inflight": gate.stats.max_durable_inflight,
            "distinct_key_max_durable_inflight": gate_two.stats.max_durable_inflight,
        },
        flush=True,
    )


if __name__ == "__main__":
    run_smoke_test()
