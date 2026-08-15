"""Verified read-through checkpoint cache reference.

This is not an RDMA implementation and never opens a network connection. It
models the correctness contract needed before substituting an RDMA memory pool,
node-local NVMe cache, or distributed object cache for a durable checkpoint
read:

* A small, strongly-consistent commit pointer selects one immutable checkpoint
  manifest. It is not used to store bulk tensors.
* Every shard is content addressed and verified after cache or durable reads.
* Cache data is untrusted acceleration: a miss/corruption/expiry/mismatch
  falls back to durable storage; it never advances a cursor or replaces a
  COMMITTED ledger checkpoint.
* Namespace, precision contract and component/shard identity are all part of a
  cache key, preventing cross-job/precision contamination.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Sequence


class CacheContractError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value: Mapping[str, object]) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class CheckpointCacheContract:
    namespace: str
    checkpoint_hash: str
    precision_contract_hash: str
    dataset_commit: str
    manifest_digest: str

    def validate(self) -> None:
        if not self.namespace or not self.dataset_commit:
            raise CacheContractError("namespace and dataset commit are required")
        for value in (self.checkpoint_hash, self.precision_contract_hash, self.manifest_digest):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise CacheContractError("contract hashes must be lowercase SHA-256")


@dataclass(frozen=True)
class ShardDescriptor:
    component: str  # model | ema | optimizer | scaler | fp8_metadata | rng
    shard_id: str
    sha256: str
    byte_count: int

    @property
    def cache_key(self) -> str:
        return f"{self.component}:{self.shard_id}:{self.sha256}"

    def validate(self) -> None:
        if not self.component or not self.shard_id or self.byte_count < 0:
            raise CacheContractError("invalid shard descriptor")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise CacheContractError("invalid shard hash")


@dataclass(frozen=True)
class CheckpointManifest:
    contract: CheckpointCacheContract
    shards: tuple[ShardDescriptor, ...]
    created_ns: int

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_hash(
            {
                "contract": asdict(self.contract),
                "shards": [asdict(item) for item in self.shards],
                "created_ns": self.created_ns,
            }
        )

    def validate(self) -> None:
        self.contract.validate()
        if not self.shards:
            raise CacheContractError("checkpoint manifest requires at least one shard")
        for shard in self.shards:
            shard.validate()
        keys = [item.cache_key for item in self.shards]
        if len(keys) != len(set(keys)):
            raise CacheContractError("duplicate content-addressed shard")


@dataclass(frozen=True)
class CommittedPointer:
    revision: int
    checkpoint_hash: str
    manifest_digest: str
    committed_step: int


class InMemoryStrongCommitKV:
    """A CAS-only model for the tiny durable metadata plane.

    Production mapping: etcd/another strongly consistent KV holds a bounded
    pointer; full checkpoint bytes live in object/parallel storage, optionally
    warmed into a cache. This class intentionally has no watch or eventual-read
    path so tests do not confuse stale cache observation with committed truth.
    """

    def __init__(self) -> None:
        self._pointers: dict[str, CommittedPointer] = {}

    def get_linearizable(self, run_key: str) -> CommittedPointer:
        try:
            return self._pointers[run_key]
        except KeyError as exc:
            raise CacheContractError("no committed pointer for run") from exc

    def compare_and_swap(self, *, run_key: str, expected_revision: int, manifest: CheckpointManifest, committed_step: int) -> CommittedPointer:
        manifest.validate()
        if committed_step < 0:
            raise CacheContractError("committed step must be non-negative")
        old = self._pointers.get(run_key)
        actual = 0 if old is None else old.revision
        if actual != expected_revision:
            raise CacheContractError("commit pointer revision changed; refuse stale checkpoint publication")
        pointer = CommittedPointer(
            revision=actual + 1,
            checkpoint_hash=manifest.contract.checkpoint_hash,
            manifest_digest=manifest.digest,
            committed_step=committed_step,
        )
        self._pointers[run_key] = pointer
        return pointer


class DurableShardStore:
    """Content-addressed durable data store model; no filesystem or network I/O."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def put(self, descriptor: ShardDescriptor, payload: bytes) -> None:
        descriptor.validate()
        if len(payload) != descriptor.byte_count or sha256_bytes(payload) != descriptor.sha256:
            raise CacheContractError("durable shard payload does not match descriptor")
        self._data[descriptor.cache_key] = bytes(payload)

    def get(self, descriptor: ShardDescriptor) -> bytes:
        try:
            return self._data[descriptor.cache_key]
        except KeyError as exc:
            raise CacheContractError("durable shard missing") from exc


@dataclass
class CacheEntry:
    payload: bytes
    expires_ns: int


class VerifiedReadThroughCache:
    """Cache entries are verified on every hit and evicted on mismatch/expiry."""

    def __init__(self, *, ttl_seconds: float = 300.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("cache TTL must be positive")
        self.ttl_ns = int(ttl_seconds * 1_000_000_000)
        self._entries: dict[tuple[str, str], CacheEntry] = {}
        self.stats = {"hit": 0, "miss": 0, "corrupt": 0, "expired": 0, "durable_fallback": 0}

    @staticmethod
    def _key(contract: CheckpointCacheContract, descriptor: ShardDescriptor) -> tuple[str, str]:
        contract.validate()
        descriptor.validate()
        # Namespace and precision contract protect against cross-job / cross-mode
        # data reuse even when payload bytes happen to be equal.
        return (f"{contract.namespace}:{contract.precision_contract_hash}:{contract.checkpoint_hash}", descriptor.cache_key)

    def put_verified(self, *, contract: CheckpointCacheContract, descriptor: ShardDescriptor, payload: bytes, now_ns: int | None = None) -> None:
        self._verify_payload(descriptor, payload)
        now = time.time_ns() if now_ns is None else now_ns
        self._entries[self._key(contract, descriptor)] = CacheEntry(bytes(payload), now + self.ttl_ns)

    def fetch_verified(
        self,
        *,
        pointer: CommittedPointer,
        manifest: CheckpointManifest,
        descriptor: ShardDescriptor,
        durable: DurableShardStore,
        now_ns: int | None = None,
    ) -> tuple[bytes, str]:
        """Return payload and source (`cache` or `durable`) after every verification."""

        manifest.validate()
        if pointer.checkpoint_hash != manifest.contract.checkpoint_hash or pointer.manifest_digest != manifest.digest:
            raise CacheContractError("manifest is not the current committed pointer")
        if descriptor not in manifest.shards:
            raise CacheContractError("descriptor not declared by committed manifest")
        now = time.time_ns() if now_ns is None else now_ns
        key = self._key(manifest.contract, descriptor)
        entry = self._entries.get(key)
        if entry is not None:
            if entry.expires_ns < now:
                self.stats["expired"] += 1
                del self._entries[key]
            else:
                try:
                    self._verify_payload(descriptor, entry.payload)
                    self.stats["hit"] += 1
                    return entry.payload, "cache"
                except CacheContractError:
                    self.stats["corrupt"] += 1
                    del self._entries[key]
        self.stats["miss"] += 1
        payload = durable.get(descriptor)
        self._verify_payload(descriptor, payload)
        self.stats["durable_fallback"] += 1
        self.put_verified(contract=manifest.contract, descriptor=descriptor, payload=payload, now_ns=now)
        return payload, "durable"

    @staticmethod
    def _verify_payload(descriptor: ShardDescriptor, payload: bytes) -> None:
        if len(payload) != descriptor.byte_count or sha256_bytes(payload) != descriptor.sha256:
            raise CacheContractError("shard integrity verification failed")

    # Test-only helper: simulates memory/cache corruption without network I/O.
    def inject_corruption_for_test(self, *, contract: CheckpointCacheContract, descriptor: ShardDescriptor, corrupted: bytes) -> None:
        key = self._key(contract, descriptor)
        old = self._entries.get(key)
        if old is None:
            raise CacheContractError("cannot corrupt a cache entry that does not exist")
        self._entries[key] = CacheEntry(corrupted, old.expires_ns)


@dataclass(frozen=True)
class ParallelLoadAssignment:
    worker_slot: int
    descriptor: ShardDescriptor


def bounded_parallel_load_plan(manifest: CheckpointManifest, *, max_inflight: int) -> tuple[ParallelLoadAssignment, ...]:
    """Deterministically spread independent shard reads over bounded slots."""

    manifest.validate()
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    return tuple(ParallelLoadAssignment(index % max_inflight, shard) for index, shard in enumerate(manifest.shards))


def _descriptor(component: str, shard_id: str, payload: bytes) -> ShardDescriptor:
    return ShardDescriptor(component, shard_id, sha256_bytes(payload), len(payload))


def run_smoke_test() -> None:
    chunks = {
        ("model", "rank-shard-0"): b"model-state-v1",
        ("ema", "rank-shard-0"): b"ema-state-v1",
        ("optimizer", "rank-shard-0"): b"adamw-exp-avg-v1",
        ("rng", "global"): b"rng-v1",
    }
    descriptors = tuple(_descriptor(component, shard_id, payload) for (component, shard_id), payload in chunks.items())
    contract = CheckpointCacheContract(
        namespace="isolated-preproduction/acvjepa",
        checkpoint_hash=sha256_bytes(b"checkpoint-v1"),
        precision_contract_hash=sha256_bytes(b"bf16:torch_amp:adamw"),
        dataset_commit="dataset-demo",
        manifest_digest=sha256_bytes(b"work-manifest-demo"),
    )
    manifest = CheckpointManifest(contract, descriptors, created_ns=123)
    durable = DurableShardStore()
    for descriptor in descriptors:
        durable.put(descriptor, chunks[(descriptor.component, descriptor.shard_id)])
    kv = InMemoryStrongCommitKV()
    pointer = kv.compare_and_swap(run_key="run-demo", expected_revision=0, manifest=manifest, committed_step=9)
    cache = VerifiedReadThroughCache(ttl_seconds=10.0)
    model = descriptors[0]
    first, source_first = cache.fetch_verified(pointer=pointer, manifest=manifest, descriptor=model, durable=durable, now_ns=100)
    second, source_second = cache.fetch_verified(pointer=pointer, manifest=manifest, descriptor=model, durable=durable, now_ns=101)
    assert first == second == chunks[(model.component, model.shard_id)]
    assert source_first == "durable" and source_second == "cache"
    cache.inject_corruption_for_test(contract=contract, descriptor=model, corrupted=b"tampered")
    third, source_third = cache.fetch_verified(pointer=pointer, manifest=manifest, descriptor=model, durable=durable, now_ns=102)
    assert third == first and source_third == "durable" and cache.stats["corrupt"] == 1
    plan = bounded_parallel_load_plan(manifest, max_inflight=2)
    assert {assignment.worker_slot for assignment in plan} == {0, 1}
    try:
        kv.compare_and_swap(run_key="run-demo", expected_revision=0, manifest=manifest, committed_step=10)
    except CacheContractError:
        stale_pointer_rejected = True
    else:
        stale_pointer_rejected = False
    assert stale_pointer_rejected
    print(json.dumps({"smoke_test": "passed", "cache_stats": cache.stats, "parallel_slots": 2}, sort_keys=True), flush=True)


if __name__ == "__main__":
    run_smoke_test()
