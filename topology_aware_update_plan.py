"""Cross-node, topology-aware dynamic UpdatePlan for synchronous DDP/NCCL.

Design constraints
------------------
* All ranks gather a trusted *local* topology record only at an optimizer /
  checkpoint boundary. Rank 0 builds an immutable manifest and update plan.
* The complete manifest and plan are JSON encoded and transported with fixed-size
  tensors (not pickle). Every rank verifies a SHA-256 digest before backward.
* GPU/NIC topology changes influence only the next update's data-cost assignment
  and local micro-batch count. They never alter the active process group's
  collective order or let a survivor perform an independent optimizer step.
* A node failure, rendezvous restart or topology-epoch mismatch invalidates the
  plan. The caller must checkpoint, rebuild the group, re-gather topology and
  create a new plan.

Topology records must come from an authenticated node inventory or a privileged
read-only probe. This module intentionally does not change NIC selection,
routing, HCA configuration, GPU reset state, or cluster membership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.distributed as dist


MAX_CONTROL_BYTES = 128 * 1024


@dataclass(frozen=True)
class LocalTopologyRecord:
    """Trusted topology facts for the current global rank.

    `expected_collective_floor_ms` is a measured or inventory-derived lower
    bound for the current GPU-to-NIC/rail path. It is a planning guard, not a
    substitute for actual per-update NCCL telemetry.
    """

    rank: int
    node_id: str
    local_rank: int
    gpu_id: str
    numa_node: int
    nic_id: str
    rail_id: str
    gpu_nic_distance: int
    expected_collective_floor_ms: float
    inventory_epoch: str

    def validate(self) -> None:
        if self.rank < 0 or self.local_rank < 0 or self.numa_node < 0 or self.gpu_nic_distance < 0:
            raise ValueError("rank/local_rank/NUMA/distance must be non-negative")
        if not all(isinstance(value, str) and value for value in (self.node_id, self.gpu_id, self.nic_id, self.rail_id, self.inventory_epoch)):
            raise ValueError("topology identity fields must be non-empty strings")
        if not math.isfinite(self.expected_collective_floor_ms) or self.expected_collective_floor_ms < 0:
            raise ValueError("expected collective floor must be finite and non-negative")


@dataclass(frozen=True)
class TopologyManifest:
    """A stable, process-group-scoped topology snapshot."""

    topology_epoch: str
    world_size: int
    ranks: Tuple[LocalTopologyRecord, ...]

    def validate(self) -> None:
        if self.world_size <= 0 or len(self.ranks) != self.world_size:
            raise ValueError("manifest world size and rank record count must agree")
        if tuple(item.rank for item in self.ranks) != tuple(range(self.world_size)):
            raise ValueError("manifest ranks must be contiguous and ordered")
        if len({(item.node_id, item.local_rank) for item in self.ranks}) != self.world_size:
            raise ValueError("a node/local_rank pair cannot map to multiple global ranks")
        for item in self.ranks:
            item.validate()

    def canonical_payload(self) -> Dict[str, Any]:
        self.validate()
        return {"topology_epoch": self.topology_epoch, "world_size": self.world_size, "ranks": [asdict(item) for item in self.ranks]}

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.canonical_payload()))


@dataclass(frozen=True)
class RankTelemetry:
    rank: int
    samples_per_second: float
    p95_data_ms: float
    p95_allreduce_ms: float
    free_memory_gb: float
    healthy: bool = True

    def validate(self) -> None:
        if self.rank < 0 or self.samples_per_second <= 0 or self.p95_data_ms < 0 or self.p95_allreduce_ms < 0 or self.free_memory_gb < 0:
            raise ValueError("invalid rank telemetry")
        if not all(math.isfinite(value) for value in (self.samples_per_second, self.p95_data_ms, self.p95_allreduce_ms, self.free_memory_gb)):
            raise ValueError("telemetry must be finite")


@dataclass(frozen=True)
class WorkItem:
    """One already-batched training work unit for the upcoming update.

    `preferred_nodes` is a locality preference, never an authorization for the
    training process to alter placement. Work IDs must be immutable and unique
    within an update (for example: dataset-commit/window-id/batch-index).
    """

    work_id: str
    cost_units: float
    preferred_nodes: Tuple[str, ...] = ()
    provenance_hash: str = ""

    def validate(self) -> None:
        if not self.work_id or not math.isfinite(self.cost_units) or self.cost_units <= 0:
            raise ValueError("work item requires non-empty ID and positive finite cost")
        if len(set(self.preferred_nodes)) != len(self.preferred_nodes):
            raise ValueError("preferred_nodes must not contain duplicates")


@dataclass(frozen=True)
class TopologyRankUpdatePlan:
    rank: int
    node_id: str
    local_rank: int
    nic_id: str
    rail_id: str
    micro_batches: int
    samples_per_micro_batch: int
    local_samples: int
    loss_sum_scale: float
    compute_budget_ms: float
    network_guard_ms: float
    work_item_ids: Tuple[str, ...]


@dataclass(frozen=True)
class TopologyAwareUpdatePlan:
    plan_version: int
    topology_epoch: str
    topology_digest: str
    work_manifest_digest: str
    target_update_ms: float
    world_size: int
    global_samples: int
    ranks: Tuple[TopologyRankUpdatePlan, ...]

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "topology_epoch": self.topology_epoch,
            "topology_digest": self.topology_digest,
            "work_manifest_digest": self.work_manifest_digest,
            "target_update_ms": self.target_update_ms,
            "world_size": self.world_size,
            "global_samples": self.global_samples,
            "ranks": [asdict(item) for item in self.ranks],
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "TopologyAwareUpdatePlan":
        expected = {"plan_version", "topology_epoch", "topology_digest", "work_manifest_digest", "target_update_ms", "world_size", "global_samples", "ranks"}
        if set(value) != expected:
            raise ValueError(f"UpdatePlan schema mismatch: {sorted(value)}")
        return cls(
            plan_version=int(value["plan_version"]),
            topology_epoch=str(value["topology_epoch"]),
            topology_digest=str(value["topology_digest"]),
            work_manifest_digest=str(value["work_manifest_digest"]),
            target_update_ms=float(value["target_update_ms"]),
            world_size=int(value["world_size"]),
            global_samples=int(value["global_samples"]),
            ranks=tuple(
                TopologyRankUpdatePlan(
                    rank=int(item["rank"]),
                    node_id=str(item["node_id"]),
                    local_rank=int(item["local_rank"]),
                    nic_id=str(item["nic_id"]),
                    rail_id=str(item["rail_id"]),
                    micro_batches=int(item["micro_batches"]),
                    samples_per_micro_batch=int(item["samples_per_micro_batch"]),
                    local_samples=int(item["local_samples"]),
                    loss_sum_scale=float(item["loss_sum_scale"]),
                    compute_budget_ms=float(item["compute_budget_ms"]),
                    network_guard_ms=float(item["network_guard_ms"]),
                    work_item_ids=tuple(item["work_item_ids"]),
                )
                for item in value["ranks"]
            ),
        )


def initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if initialized() else 1


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bytes_tensor(raw: bytes, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(raw), dtype=torch.uint8, device=device)


def _all_gather_bytes(raw: bytes, device: torch.device, *, max_bytes: int = MAX_CONTROL_BYTES) -> List[bytes]:
    """All-gather canonical JSON bytes without object/pickle collectives."""

    if len(raw) == 0 or len(raw) > max_bytes:
        raise ValueError("control payload length is invalid")
    if not initialized():
        return [raw]
    length = torch.tensor([len(raw)], dtype=torch.int64, device=device)
    lengths = [torch.empty_like(length) for _ in range(world_size())]
    dist.all_gather(lengths, length)
    received_lengths = [int(item.item()) for item in lengths]
    if any(value <= 0 or value > max_bytes for value in received_lengths):
        raise RuntimeError("received an invalid control payload length")
    padded_size = max(received_lengths)
    padded = torch.zeros(padded_size, dtype=torch.uint8, device=device)
    padded[: len(raw)] = _bytes_tensor(raw, device)
    gathered = [torch.empty_like(padded) for _ in range(world_size())]
    dist.all_gather(gathered, padded)
    return [bytes(item[:size].cpu().tolist()) for item, size in zip(gathered, received_lengths)]


def _broadcast_bytes(root_bytes: bytes | None, device: torch.device, *, max_bytes: int = MAX_CONTROL_BYTES) -> bytes:
    """Broadcast bytes from rank 0. Callers parse and validate afterward."""

    if not initialized():
        if root_bytes is None:
            raise ValueError("single-process broadcast requires payload")
        return root_bytes
    if rank() == 0:
        if root_bytes is None or not (0 < len(root_bytes) <= max_bytes):
            raise ValueError("rank 0 control payload is invalid")
        length = torch.tensor([len(root_bytes)], dtype=torch.int64, device=device)
    else:
        length = torch.zeros(1, dtype=torch.int64, device=device)
    dist.broadcast(length, src=0)
    size = int(length.item())
    if not (0 < size <= max_bytes):
        raise RuntimeError("broadcast control length is invalid")
    wire = _bytes_tensor(root_bytes, device) if rank() == 0 else torch.empty(size, dtype=torch.uint8, device=device)
    dist.broadcast(wire, src=0)
    return bytes(wire.cpu().tolist())


def _assert_identical_bytes(raw: bytes, device: torch.device) -> None:
    digests = _all_gather_bytes(hashlib.sha256(raw).digest(), device, max_bytes=64)
    if any(item != digests[0] for item in digests):
        raise RuntimeError("control-plane digest diverged across ranks")


def _build_epoch(records: Sequence[LocalTopologyRecord]) -> str:
    """Include torch elastic epoch fields so a restart cannot reuse an old plan."""

    stable = {
        "torchelastic_run_id": os.environ.get("TORCHELASTIC_RUN_ID", "fixed-world"),
        "restart_count": os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        "world_size": len(records),
        "records": [asdict(item) for item in sorted(records, key=lambda item: item.rank)],
    }
    return sha256_hex(canonical_json(stable))


def gather_topology_manifest(local: LocalTopologyRecord, device: torch.device) -> TopologyManifest:
    """Gather local inventory records, build a rank-0 manifest, broadcast + verify it."""

    if local.rank != rank():
        raise ValueError("local topology rank must match distributed rank")
    local.validate()
    raw_records = _all_gather_bytes(canonical_json(asdict(local)), device)
    if rank() == 0:
        records = tuple(sorted((LocalTopologyRecord(**json.loads(raw)) for raw in raw_records), key=lambda item: item.rank))
        manifest = TopologyManifest(topology_epoch=_build_epoch(records), world_size=world_size(), ranks=records)
        manifest.validate()
        root_raw = canonical_json(manifest.canonical_payload())
    else:
        root_raw = None
    received = _broadcast_bytes(root_raw, device)
    payload = json.loads(received.decode("utf-8"))
    manifest = TopologyManifest(
        topology_epoch=payload["topology_epoch"],
        world_size=int(payload["world_size"]),
        ranks=tuple(LocalTopologyRecord(**item) for item in payload["ranks"]),
    )
    manifest.validate()
    if manifest.topology_epoch != _build_epoch(manifest.ranks):
        raise RuntimeError("topology epoch does not match current elastic run/records")
    _assert_identical_bytes(received, device)
    return manifest


class TopologyAwarePlanner:
    """Plans heterogeneous micro-batches and assigns one-cost-bucket work units.

    Planning favors locality and high-throughput ranks, but all assignments are
    bounded by a common update deadline. The final local micro-batch remains the
    only DDP-synchronized backward and is not encoded here; callers use
    `work_item_ids` as their deterministic input order.
    """

    def __init__(
        self,
        *,
        target_update_ms: float = 750.0,
        min_micro_batches: int = 1,
        max_micro_batches: int = 16,
        safety_jitter_ms: float = 5.0,
        remote_work_penalty_ms: float = 25.0,
        distance_penalty_ms: float = 0.5,
    ) -> None:
        if target_update_ms <= 0 or min_micro_batches < 1 or max_micro_batches < min_micro_batches:
            raise ValueError("planner bounds are invalid")
        self.target_update_ms = target_update_ms
        self.min_micro_batches = min_micro_batches
        self.max_micro_batches = max_micro_batches
        self.safety_jitter_ms = safety_jitter_ms
        self.remote_work_penalty_ms = remote_work_penalty_ms
        self.distance_penalty_ms = distance_penalty_ms
        self.version = 0

    def plan(
        self,
        *,
        manifest: TopologyManifest,
        telemetry: Iterable[RankTelemetry],
        samples_per_micro_batch: Mapping[int, int],
        work_items: Sequence[WorkItem],
    ) -> TopologyAwareUpdatePlan:
        manifest.validate()
        records = {item.rank: item for item in manifest.ranks}
        telemetries = {item.rank: item for item in telemetry}
        if set(telemetries) != set(records) or set(samples_per_micro_batch) != set(records):
            raise ValueError("every current rank needs topology, telemetry and a micro-batch size")
        if any(not telemetries[index].healthy for index in records):
            raise RuntimeError("unhealthy rank: checkpoint and rebuild membership before planning")
        if any(size <= 0 for size in samples_per_micro_batch.values()):
            raise ValueError("micro-batch sizes must be positive")
        for item in telemetries.values():
            item.validate()
        for item in work_items:
            item.validate()
        if len({item.work_id for item in work_items}) != len(work_items):
            raise ValueError("work IDs must be unique within an UpdatePlan")

        local_counts: Dict[int, int] = {}
        guards: Dict[int, float] = {}
        budgets: Dict[int, float] = {}
        for rank_id, topo in records.items():
            observed_network_ms = telemetries[rank_id].p95_allreduce_ms
            network_guard = max(observed_network_ms, topo.expected_collective_floor_ms)
            guard = telemetries[rank_id].p95_data_ms + network_guard + self.safety_jitter_ms
            compute_budget = self.target_update_ms - guard
            if compute_budget <= 0:
                raise RuntimeError(f"rank {rank_id} has no positive compute budget; investigate data/network tail latency")
            raw_micro_batches = telemetries[rank_id].samples_per_second * (compute_budget / 1000.0) / samples_per_micro_batch[rank_id]
            k = max(self.min_micro_batches, min(self.max_micro_batches, int(round(raw_micro_batches))))
            local_counts[rank_id] = k
            guards[rank_id] = network_guard
            budgets[rank_id] = compute_budget

        expected_item_count = sum(local_counts.values())
        if len(work_items) != expected_item_count:
            raise ValueError(
                f"work_items must contain one pre-batched item per planned micro-batch: "
                f"expected {expected_item_count}, got {len(work_items)}"
            )
        assignments = self._assign_work(
            records=records,
            telemetry=telemetries,
            local_counts=local_counts,
            work_items=work_items,
        )
        global_samples = sum(local_counts[item] * samples_per_micro_batch[item] for item in records)
        self.version += 1
        loss_scale = manifest.world_size / global_samples
        plan = TopologyAwareUpdatePlan(
            plan_version=self.version,
            topology_epoch=manifest.topology_epoch,
            topology_digest=manifest.digest,
            work_manifest_digest=work_manifest_digest(work_items),
            target_update_ms=self.target_update_ms,
            world_size=manifest.world_size,
            global_samples=global_samples,
            ranks=tuple(
                TopologyRankUpdatePlan(
                    rank=rank_id,
                    node_id=records[rank_id].node_id,
                    local_rank=records[rank_id].local_rank,
                    nic_id=records[rank_id].nic_id,
                    rail_id=records[rank_id].rail_id,
                    micro_batches=local_counts[rank_id],
                    samples_per_micro_batch=samples_per_micro_batch[rank_id],
                    local_samples=local_counts[rank_id] * samples_per_micro_batch[rank_id],
                    loss_sum_scale=loss_scale,
                    compute_budget_ms=budgets[rank_id],
                    network_guard_ms=guards[rank_id],
                    work_item_ids=tuple(assignments[rank_id]),
                )
                for rank_id in range(manifest.world_size)
            ),
        )
        validate_update_plan(plan, manifest, work_items)
        return plan

    def _assign_work(
        self,
        *,
        records: Mapping[int, LocalTopologyRecord],
        telemetry: Mapping[int, RankTelemetry],
        local_counts: Mapping[int, int],
        work_items: Sequence[WorkItem],
    ) -> Dict[int, List[str]]:
        """Greedy min-estimated-finish assignment with fixed per-rank slot caps."""

        assigned: Dict[int, List[WorkItem]] = {rank_id: [] for rank_id in records}
        assigned_cost: Dict[int, float] = {rank_id: 0.0 for rank_id in records}
        # Assign expensive work first; schedule expensive micro-batches first on a
        # rank so the final synchronized micro-batch is less likely to be the tail.
        for item in sorted(work_items, key=lambda value: (-value.cost_units, value.work_id)):
            candidates = [rank_id for rank_id in records if len(assigned[rank_id]) < local_counts[rank_id]]
            if not candidates:
                raise AssertionError("planner exhausted all slots before assigning work")

            def estimated_finish(rank_id: int) -> tuple[float, int]:
                topo = records[rank_id]
                remote_penalty = 0.0 if not item.preferred_nodes or topo.node_id in item.preferred_nodes else self.remote_work_penalty_ms
                topology_penalty = topo.gpu_nic_distance * self.distance_penalty_ms
                compute_ms = 1000.0 * (assigned_cost[rank_id] + item.cost_units) / telemetry[rank_id].samples_per_second
                return (compute_ms + remote_penalty + topology_penalty, rank_id)

            destination = min(candidates, key=estimated_finish)
            assigned[destination].append(item)
            assigned_cost[destination] += item.cost_units
        return {
            rank_id: [item.work_id for item in sorted(values, key=lambda value: (-value.cost_units, value.work_id))]
            for rank_id, values in assigned.items()
        }


def work_manifest_digest(work_items: Sequence[WorkItem]) -> str:
    """Hash all work attributes, not only work IDs, in a stable global order."""

    for item in work_items:
        item.validate()
    return sha256_hex(canonical_json({"work_items": [asdict(item) for item in sorted(work_items, key=lambda item: item.work_id)]}))


def validate_update_plan(plan: TopologyAwareUpdatePlan, manifest: TopologyManifest, work_items: Sequence[WorkItem]) -> None:
    manifest.validate()
    if plan.world_size != world_size() or plan.world_size != manifest.world_size:
        raise RuntimeError("plan/manfiest/process group world sizes disagree")
    if plan.topology_epoch != manifest.topology_epoch or plan.topology_digest != manifest.digest:
        raise RuntimeError("plan was built for a different topology epoch")
    if plan.work_manifest_digest != work_manifest_digest(work_items):
        raise RuntimeError("plan work manifest differs in IDs, costs, locality or provenance")
    if plan.plan_version < 1 or plan.global_samples <= 0 or not math.isfinite(plan.target_update_ms):
        raise ValueError("invalid plan metadata")
    if tuple(item.rank for item in plan.ranks) != tuple(range(plan.world_size)):
        raise ValueError("plan rank entries are invalid")
    expected_work = {item.work_id for item in work_items}
    planned_work = [work_id for item in plan.ranks for work_id in item.work_item_ids]
    if set(planned_work) != expected_work or len(planned_work) != len(expected_work):
        raise ValueError("plan work assignment omits, duplicates or invents work IDs")
    if plan.global_samples != sum(item.local_samples for item in plan.ranks):
        raise ValueError("global sample count does not equal local totals")
    expected_scale = plan.world_size / plan.global_samples
    for rank_plan, topo in zip(plan.ranks, manifest.ranks):
        if (rank_plan.node_id, rank_plan.local_rank, rank_plan.nic_id, rank_plan.rail_id) != (
            topo.node_id,
            topo.local_rank,
            topo.nic_id,
            topo.rail_id,
        ):
            raise ValueError("rank plan topology facts do not match its manifest")
        if rank_plan.micro_batches < 1 or len(rank_plan.work_item_ids) != rank_plan.micro_batches:
            raise ValueError("micro-batch count and work assignment disagree")
        if rank_plan.local_samples != rank_plan.micro_batches * rank_plan.samples_per_micro_batch:
            raise ValueError("local sample count is invalid")
        if not math.isclose(rank_plan.loss_sum_scale, expected_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("loss scaling does not match DDP rank-average semantics")


def as_legacy_ddp_plan(plan: TopologyAwareUpdatePlan):
    """Adapt a verified topology plan to the existing AC-VJEPA update-loop type.

    The topology fields and `work_item_ids` remain in the durable control-plane
    record. The returned object carries only the loss-scaling fields consumed by
    `acvjepa_dynamic_update`; the caller must build its rank-local iterator in
    exactly the `work_item_ids` order before invoking that loop.
    """

    from adaptive_ddp_accumulation import RankUpdatePlan, UpdatePlan

    return UpdatePlan(
        plan_version=plan.plan_version,
        target_update_ms=plan.target_update_ms,
        world_size=plan.world_size,
        global_samples=plan.global_samples,
        ranks=tuple(
            RankUpdatePlan(
                rank=item.rank,
                micro_batches=item.micro_batches,
                samples_per_micro_batch=item.samples_per_micro_batch,
                local_samples=item.local_samples,
                loss_sum_scale=item.loss_sum_scale,
            )
            for item in plan.ranks
        ),
    )


def broadcast_topology_aware_plan(
    root_plan: TopologyAwareUpdatePlan | None,
    *,
    manifest: TopologyManifest,
    work_items: Sequence[WorkItem],
    device: torch.device,
) -> TopologyAwareUpdatePlan:
    """Broadcast an immutable plan for the current topology epoch and verify it."""

    if rank() == 0:
        if root_plan is None:
            raise ValueError("rank 0 must provide a plan")
        validate_update_plan(root_plan, manifest, work_items)
        root_raw = canonical_json(root_plan.canonical_payload())
    else:
        root_raw = None
    received = _broadcast_bytes(root_raw, device)
    plan = TopologyAwareUpdatePlan.from_payload(json.loads(received.decode("utf-8")))
    validate_update_plan(plan, manifest, work_items)
    _assert_identical_bytes(received, device)
    return plan


def topology_aware_next_plan(
    *,
    local_topology: LocalTopologyRecord,
    local_telemetry: RankTelemetry,
    local_micro_batch_size: int,
    work_items: Sequence[WorkItem],
    planner: TopologyAwarePlanner | None,
    device: torch.device,
) -> Tuple[TopologyManifest, TopologyAwareUpdatePlan]:
    """Cross-node control-plane operation performed at a safe update boundary.

    Work items must already be deterministically materialized from the same
    dataset commit and their content/provenance hashes should be verified by the
    caller before planning. All ranks receive the same work list in this reference
    implementation; production can shard a signed global work manifest then
    retain the same `work_id`/digest checks.
    """

    manifest = gather_topology_manifest(local_topology, device)
    telemetry_raw = _all_gather_bytes(canonical_json(asdict(local_telemetry)), device)
    batch_raw = _all_gather_bytes(canonical_json({"rank": rank(), "batch_size": local_micro_batch_size}), device)
    telemetry = [RankTelemetry(**json.loads(raw)) for raw in telemetry_raw]
    sizes = {int(json.loads(raw)["rank"]): int(json.loads(raw)["batch_size"]) for raw in batch_raw}
    if rank() == 0:
        if planner is None:
            raise ValueError("rank 0 requires TopologyAwarePlanner")
        root_plan = planner.plan(
            manifest=manifest,
            telemetry=telemetry,
            samples_per_micro_batch=sizes,
            work_items=work_items,
        )
    else:
        root_plan = None
    return manifest, broadcast_topology_aware_plan(root_plan, manifest=manifest, work_items=work_items, device=device)


# ------------------------------- smoke test ---------------------------------
def _smoke_local_topology(rank_id: int) -> LocalTopologyRecord:
    # Two process hosts are simulated in CPU/Gloo: topology semantics are tested,
    # not real GPU/NIC discovery. Real jobs source these facts from trusted probe
    # output generated at worker-group initialization.
    return LocalTopologyRecord(
        rank=rank_id,
        node_id="node-a" if rank_id == 0 else "node-b",
        local_rank=0,
        gpu_id=f"GPU-{rank_id}",
        numa_node=0,
        nic_id="mlx5_0",
        rail_id="rail-0" if rank_id == 0 else "rail-1",
        gpu_nic_distance=1 if rank_id == 0 else 3,
        expected_collective_floor_ms=8.0 if rank_id == 0 else 20.0,
        inventory_epoch="inventory-demo-v1",
    )


def run_smoke_test() -> None:
    from datetime import timedelta

    if "RANK" in os.environ and not initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    if world_size() != 2:
        raise RuntimeError("smoke test requires exactly two ranks")
    local_rank = rank()
    work_items = [
        WorkItem("window-soft-000", 4.0, ("node-a",), "a" * 64),
        WorkItem("window-soft-001", 3.0, ("node-a",), "b" * 64),
        WorkItem("window-rigid-000", 1.0, ("node-b",), "c" * 64),
    ]
    planner = TopologyAwarePlanner(target_update_ms=120.0, min_micro_batches=1, max_micro_batches=3) if local_rank == 0 else None
    manifest, plan = topology_aware_next_plan(
        local_topology=_smoke_local_topology(local_rank),
        local_telemetry=RankTelemetry(
            rank=local_rank,
            samples_per_second=40.0 if local_rank == 0 else 17.0,
            p95_data_ms=12.0,
            p95_allreduce_ms=10.0 if local_rank == 0 else 18.0,
            free_memory_gb=16.0,
            healthy=True,
        ),
        local_micro_batch_size=2,
        work_items=work_items,
        planner=planner,
        device=torch.device("cpu"),
    )
    assert plan.ranks[0].micro_batches == 2 and plan.ranks[1].micro_batches == 1
    assert plan.global_samples == 6
    assert plan.topology_epoch == manifest.topology_epoch
    legacy = as_legacy_ddp_plan(plan)
    assert legacy.global_samples == plan.global_samples and legacy.ranks[0].micro_batches == 2
    if local_rank == 0:
        print(
            json.dumps(
                {
                    "smoke_test": "passed",
                    "topology_epoch": manifest.topology_epoch,
                    "topology_digest": manifest.digest,
                    "plan": plan.canonical_payload(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test:
        parser.error("Run with --smoke-test or import topology_aware_next_plan into a trainer.")
    run_smoke_test()


if __name__ == "__main__":
    main()
