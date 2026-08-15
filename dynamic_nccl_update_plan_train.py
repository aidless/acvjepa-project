"""Dynamic UpdatePlan broadcast for heterogeneous *synchronous* DDP training.

Safety and correctness contract
-------------------------------
- A rank can use a different number of local micro-batches, but every rank must
  execute exactly one final synchronized backward per optimizer update.
- UpdatePlan is immutable within an update. World-size/membership changes only
  occur after a durable checkpoint and a new process group rendezvous.
- Losses must be means over valid local samples. This module reweights them so
  DDP's rank-average AllReduce yields the global sample-average gradient.
- The implementation never performs robot control and does not promote models.

Use `torchrun ...` with NCCL for GPUs. The smoke test intentionally uses Gloo
when CUDA is absent; it verifies collective ordering and gradient semantics,
not NCCL hardware/network performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from adaptive_ddp_accumulation import (
    DynamicAccumulationPlanner,
    RankTelemetry,
    RankUpdatePlan,
    UpdatePlan,
    scaled_loss_from_mean,
)


@dataclass(frozen=True)
class DynamicStepConfig:
    """Configuration that remains constant during a given optimizer update."""

    amp: bool = False
    gradient_clip_norm: float = 1.0
    max_plan_wire_bytes: int = 64 * 1024


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if distributed() else 1


def canonical_json(payload: Mapping) -> bytes:
    """Canonical control-plane encoding; no pickle is accepted from the wire."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def plan_to_dict(plan: UpdatePlan) -> Dict:
    return {
        "plan_version": plan.plan_version,
        "target_update_ms": plan.target_update_ms,
        "world_size": plan.world_size,
        "global_samples": plan.global_samples,
        "ranks": [asdict(rank_plan) for rank_plan in plan.ranks],
    }


def plan_from_dict(payload: Mapping) -> UpdatePlan:
    required = {"plan_version", "target_update_ms", "world_size", "global_samples", "ranks"}
    if set(payload) != required:
        raise ValueError(f"UpdatePlan schema mismatch: keys={sorted(payload)}")
    rank_plans = tuple(RankUpdatePlan(**item) for item in payload["ranks"])
    plan = UpdatePlan(
        plan_version=int(payload["plan_version"]),
        target_update_ms=float(payload["target_update_ms"]),
        world_size=int(payload["world_size"]),
        global_samples=int(payload["global_samples"]),
        ranks=rank_plans,
    )
    validate_plan_structure(plan)
    return plan


def validate_plan_structure(plan: UpdatePlan) -> None:
    """Reject malformed plans before calling DDP backward/collectives."""

    if plan.plan_version < 1 or not math.isfinite(plan.target_update_ms) or plan.target_update_ms <= 0:
        raise ValueError("invalid UpdatePlan version or target duration")
    if plan.world_size != get_world_size():
        raise RuntimeError(
            f"plan world_size={plan.world_size} disagrees with process group={get_world_size()}; "
            "checkpoint and recreate the group before the next update"
        )
    if len(plan.ranks) != plan.world_size:
        raise ValueError("UpdatePlan must have exactly one rank entry per participant")
    if tuple(entry.rank for entry in plan.ranks) != tuple(range(plan.world_size)):
        raise ValueError("UpdatePlan ranks must be contiguous and ordered by rank")
    if plan.global_samples != sum(entry.local_samples for entry in plan.ranks) or plan.global_samples <= 0:
        raise ValueError("UpdatePlan global sample count is inconsistent")
    expected_scale = plan.world_size / plan.global_samples
    for entry in plan.ranks:
        if entry.micro_batches < 1 or entry.samples_per_micro_batch < 1:
            raise ValueError("each rank needs at least one non-empty micro-batch")
        if entry.local_samples != entry.micro_batches * entry.samples_per_micro_batch:
            raise ValueError("local_samples does not match planned micro-batches")
        if not math.isclose(entry.loss_sum_scale, expected_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("loss scaling is not compatible with DDP rank averaging")


def _wire_tensor(data: bytes, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.uint8, device=device)


def broadcast_update_plan(rank0_plan: UpdatePlan | None, device: torch.device, *, max_wire_bytes: int = 64 * 1024) -> UpdatePlan:
    """Broadcast a JSON-encoded UpdatePlan then verify a common digest.

    Unlike `broadcast_object_list`, this avoids pickling data received from the
    network. The process group remains a trusted control plane: encryption,
    network isolation and rank identity still belong to cluster operations.
    """

    if not distributed():
        if rank0_plan is None:
            raise ValueError("single-process execution still requires a plan")
        validate_plan_structure(rank0_plan)
        return rank0_plan

    rank = get_rank()
    if rank == 0:
        if rank0_plan is None:
            raise ValueError("rank 0 must create the UpdatePlan")
        validate_plan_structure(rank0_plan)
        raw = canonical_json(plan_to_dict(rank0_plan))
        if not (0 < len(raw) <= max_wire_bytes):
            raise ValueError("serialized UpdatePlan exceeds control-plane limit")
        size = torch.tensor([len(raw)], dtype=torch.int64, device=device)
    else:
        raw = b""
        size = torch.zeros(1, dtype=torch.int64, device=device)

    dist.broadcast(size, src=0)
    wire_size = int(size.item())
    if not (0 < wire_size <= max_wire_bytes):
        raise RuntimeError("received an invalid UpdatePlan length")
    wire = _wire_tensor(raw, device) if rank == 0 else torch.empty(wire_size, dtype=torch.uint8, device=device)
    dist.broadcast(wire, src=0)

    received_raw = bytes(wire.cpu().tolist())
    plan = plan_from_dict(json.loads(received_raw.decode("utf-8")))
    _verify_plan_consensus(received_raw, device)
    return plan


def _verify_plan_consensus(raw: bytes, device: torch.device) -> None:
    """Fail closed if any rank decoded or constructed a different plan."""

    if not distributed():
        return
    digest = _wire_tensor(hashlib.sha256(raw).digest(), device)
    all_digests = [torch.empty_like(digest) for _ in range(get_world_size())]
    dist.all_gather(all_digests, digest)
    if any(not torch.equal(digest, peer) for peer in all_digests):
        raise RuntimeError("UpdatePlan digest divergence: stop before the synchronized backward")


def all_gather_telemetry(local: RankTelemetry, device: torch.device) -> List[RankTelemetry]:
    """Gather a fixed, numeric telemetry contract without object collectives."""

    values = torch.tensor(
        [
            local.samples_per_second,
            local.p95_step_ms,
            local.p95_data_ms,
            local.p95_allreduce_ms,
            local.free_memory_gb,
            1.0 if local.healthy else 0.0,
        ],
        dtype=torch.float64,
        device=device,
    )
    if not distributed():
        return [local]
    gathered = [torch.empty_like(values) for _ in range(get_world_size())]
    dist.all_gather(gathered, values)
    return [
        RankTelemetry(
            rank=index,
            samples_per_second=float(item[0].item()),
            p95_step_ms=float(item[1].item()),
            p95_data_ms=float(item[2].item()),
            p95_allreduce_ms=float(item[3].item()),
            free_memory_gb=float(item[4].item()),
            healthy=bool(item[5].item() >= 0.5),
        )
        for index, item in enumerate(gathered)
    ]


def all_gather_micro_batch_sizes(local_size: int, device: torch.device) -> Dict[int, int]:
    if local_size <= 0:
        raise ValueError("local micro-batch size must be positive")
    value = torch.tensor([local_size], dtype=torch.int64, device=device)
    if not distributed():
        return {0: local_size}
    gathered = [torch.empty_like(value) for _ in range(get_world_size())]
    dist.all_gather(gathered, value)
    return {rank: int(item.item()) for rank, item in enumerate(gathered)}


def next_update_plan(
    *,
    planner: DynamicAccumulationPlanner | None,
    local_telemetry: RankTelemetry,
    local_micro_batch_size: int,
    device: torch.device,
    max_plan_wire_bytes: int = 64 * 1024,
) -> UpdatePlan:
    """Collect observations, let rank 0 plan, then broadcast one immutable plan."""

    if local_telemetry.rank != get_rank():
        raise ValueError("telemetry rank must match the current process rank")
    telemetry = all_gather_telemetry(local_telemetry, device)
    micro_batch_sizes = all_gather_micro_batch_sizes(local_micro_batch_size, device)
    if get_rank() == 0:
        if planner is None:
            raise ValueError("rank 0 requires a DynamicAccumulationPlanner")
        root_plan = planner.plan(telemetry, micro_batch_sizes)
    else:
        root_plan = None
    return broadcast_update_plan(root_plan, device, max_wire_bytes=max_plan_wire_bytes)


def plan_for_rank(plan: UpdatePlan) -> RankUpdatePlan:
    validate_plan_structure(plan)
    return plan.ranks[get_rank()]


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=(device.type == "cuda")) for name, value in batch.items()}


def _module(ddp_or_module: DDP | nn.Module) -> nn.Module:
    return ddp_or_module.module if isinstance(ddp_or_module, DDP) else ddp_or_module


def acvjepa_dynamic_update(
    *,
    ddp: DDP | nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    plan: UpdatePlan,
    micro_batches: Iterator[Mapping[str, torch.Tensor]],
    device: torch.device,
    config: DynamicStepConfig,
) -> Dict[str, float]:
    """Execute exactly one globally synchronous optimizer update.

    `micro_batches` must be a non-exhausting/cycling per-rank stream or have
    enough data for the supplied plan. All ranks must call this function in the
    same update order. For padded sequences, replace `actual_samples` by a loss
    reducer that computes a true valid-token sum; multiplying an already padded
    mean is not mathematically correct.
    """

    mine = plan_for_rank(plan)
    optimizer.zero_grad(set_to_none=True)
    latest_losses = None
    local_seen = 0

    for local_index in range(mine.micro_batches):
        try:
            raw_batch = next(micro_batches)
        except StopIteration as exc:
            raise RuntimeError(
                "local data stream exhausted inside an UpdatePlan; use a cycling stream or "
                "re-plan only at a completed optimizer/checkpoint boundary"
            ) from exc
        batch = _move_batch(raw_batch, device)
        actual_samples = int(batch["context_video"].shape[0])
        if actual_samples != mine.samples_per_micro_batch:
            raise RuntimeError(
                f"rank {get_rank()} received {actual_samples} samples, but UpdatePlan requires "
                f"{mine.samples_per_micro_batch}; abort before DDP synchronization"
            )
        local_seen += actual_samples
        final_local_micro_batch = local_index == mine.micro_batches - 1
        sync_context = nullcontext() if final_local_micro_batch or not isinstance(ddp, DDP) else ddp.no_sync()
        with sync_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(config.amp and device.type == "cuda"),
            ):
                # Calling ddp(...) rather than ddp.module(...) installs DDP hooks.
                prediction = ddp(
                    batch["context_video"],
                    batch["context_proprio"],
                    batch["executed_actions"],
                )
                module = _module(ddp)
                targets = module.target_latents(batch["future_video"], batch["future_proprio"])
                from ac_vjepa_core import action_conditioned_jepa_loss

                losses = action_conditioned_jepa_loss(prediction, targets, batch["future_events"])
                # Loss is a local *mean*. Convert it into a contribution compatible
                # with DDP's default rank-average gradient reduction:
                # (1/W) Σ_r [W/N Σ_i∈r ∇loss_i] = (1/N) Σ_all_i ∇loss_i.
                weighted_loss = scaled_loss_from_mean(losses.total, actual_samples, mine)
            scaler.scale(weighted_loss).backward()
            latest_losses = losses

    if local_seen != mine.local_samples:
        raise AssertionError("consumed samples do not match the immutable UpdatePlan")
    if latest_losses is None:
        raise AssertionError("a valid plan must produce at least one local micro-batch")

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(ddp.parameters(), config.gradient_clip_norm)
    scaler.step(optimizer)
    scaler.update()

    module = _module(ddp)
    if hasattr(module, "update_ema_target"):
        module.update_ema_target()

    # Diagnostics only; all ranks participate in this collective after the step.
    seen = torch.tensor([local_seen], dtype=torch.int64, device=device)
    if distributed():
        dist.all_reduce(seen, op=dist.ReduceOp.SUM)
    if int(seen.item()) != plan.global_samples:
        raise RuntimeError("actual global samples disagree with UpdatePlan; checkpoint and investigate")

    return {
        "plan_version": float(plan.plan_version),
        "global_samples": float(plan.global_samples),
        "local_samples": float(local_seen),
        "loss_total_last_micro_batch": float(latest_losses.total.detach().float().cpu()),
    }


# ---------------------------- CPU/Gloo smoke test ----------------------------
class ToyWorldModel(nn.Module):
    """Minimal model that exposes the AC-VJEPA training-loop contract."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.target_encoder = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.target_encoder.weight.copy_(self.encoder.weight)

    def forward(self, context_video: torch.Tensor, context_proprio: torch.Tensor, actions: torch.Tensor):
        # The smoke test uses a direct MSE loss path and does not call this class
        # through `acvjepa_dynamic_update`; it exists for interface documentation.
        return self.encoder(context_video)


def _toy_dynamic_update(
    ddp: DDP | nn.Module,
    optimizer: torch.optim.Optimizer,
    plan: UpdatePlan,
    stream: Iterator[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Small numerical test of the same scaling and no_sync pattern."""

    mine = plan_for_rank(plan)
    optimizer.zero_grad(set_to_none=True)
    for i in range(mine.micro_batches):
        x = next(stream).to(device)
        if x.shape[0] != mine.samples_per_micro_batch:
            raise RuntimeError("smoke batch size disagrees with plan")
        y = torch.full_like(x[:, :1], 2.0)
        sync_context = nullcontext() if i == mine.micro_batches - 1 or not isinstance(ddp, DDP) else ddp.no_sync()
        with sync_context:
            # Mean over local samples; weighting is identical to AC-VJEPA path.
            local_mean_loss = (ddp(x) - y).square().mean()
            weighted = scaled_loss_from_mean(local_mean_loss, x.shape[0], mine)
            weighted.backward()
    optimizer.step()
    return _module(ddp).weight.detach().clone()


def _cycling_toy_stream(batch_size: int, rank: int) -> Iterator[torch.Tensor]:
    # Rank-dependent data creates heterogeneous local gradients, so the test
    # verifies the reweighting rather than only matching identical replicas.
    generator = torch.Generator().manual_seed(1000 + rank)
    while True:
        yield torch.randn(batch_size, 4, generator=generator)


def run_smoke_test() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" in os.environ and not distributed():
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    device = torch.device("cpu")
    rank = get_rank()
    model = nn.Linear(4, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(0.25)
    ddp: DDP | nn.Module = DDP(model) if get_world_size() > 1 else model
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
    planner = DynamicAccumulationPlanner(target_update_ms=120.0, min_micro_batches=1, max_micro_batches=4) if rank == 0 else None

    # Rank 0 receives a larger plan than rank 1 due to reported throughput.
    local_telemetry = RankTelemetry(
        rank=rank,
        samples_per_second=40.0 if rank == 0 else 17.0,
        p95_step_ms=95.0 if rank == 0 else 115.0,
        p95_data_ms=10.0,
        p95_allreduce_ms=8.0,
        free_memory_gb=10.0,
        healthy=True,
    )
    plan = next_update_plan(
        planner=planner,
        local_telemetry=local_telemetry,
        local_micro_batch_size=2,
        device=device,
    )
    before = _module(ddp).weight.detach().clone()
    after = _toy_dynamic_update(ddp, optimizer, plan, _cycling_toy_stream(2, rank), device)

    if distributed():
        gathered = [torch.empty_like(after) for _ in range(get_world_size())]
        dist.all_gather(gathered, after)
        if any(not torch.allclose(after, peer, atol=1e-6) for peer in gathered):
            raise AssertionError("DDP replicas diverged after heterogeneous update")
        global_seen = torch.tensor([plan_for_rank(plan).local_samples], dtype=torch.int64)
        dist.all_reduce(global_seen, op=dist.ReduceOp.SUM)
        if int(global_seen.item()) != plan.global_samples:
            raise AssertionError("planned and observed global samples diverged")
    if rank == 0:
        print(
            json.dumps(
                {
                    "smoke_test": "passed",
                    "world_size": get_world_size(),
                    "plan": plan_to_dict(plan),
                    "weight_changed": not torch.allclose(before, after),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if distributed():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        parser.error("This reference module is run with --smoke-test or imported into the AC-VJEPA trainer.")


if __name__ == "__main__":
    main()
