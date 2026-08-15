"""Strong 2:1 heterogeneous-micro-batch correctness test.

Run on CPU CI:
  torchrun --standalone --nproc_per_node=2 test_dynamic_nccl_full_state_equivalence.py

This trusted test process uses Gloo and all_gather_object only to collect test
snapshots. It validates distributed-control semantics, not NCCL hardware/network
behavior. The production UpdatePlan control plane remains tensor/JSON based.

Assertions:
1. Rank 0 consumes two local micro-batches and rank 1 consumes one.
2. The entire AC-VJEPA state_dict (student + EMA target + all buffers) is equal
   across ranks after the synchronized optimizer step.
3. Every AdamW state entry and param-group field is equal across ranks.
4. Distributed 2:1 update equals a single-process reference update over the
   same six samples, modulo a documented floating-point tolerance.
"""
from __future__ import annotations

import copy
import json
from datetime import timedelta
from typing import Any, Dict, Iterator, List, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ac_vjepa_core import ActionConditionedVJEPA, action_conditioned_jepa_loss
from adaptive_ddp_accumulation import DynamicAccumulationPlanner, RankTelemetry
from dynamic_nccl_update_plan_train import (
    DynamicStepConfig,
    acvjepa_dynamic_update,
    get_rank,
    get_world_size,
    next_update_plan,
)


# Cross-rank replicas execute the same DDP-reduced update and should match far
# more tightly than a serial reference. The serial reference has a different
# floating-point reduction order (three local backward calls vs. DDP's two
# accumulated rank gradients), so it receives a separately documented budget.
REPLICA_ATOL = 1e-6
REPLICA_RTOL = 1e-6
# The serial reference and DDP follow different floating-point addition trees,
# but must follow identical loss scaling, clipping and optimizer semantics. A
# narrow 2e-5 one-step budget admits that reduction-order noise without masking
# meaningful state drift. Production thresholds still require calibration on a
# larger fixed-input reference corpus.
REFERENCE_ATOL = 2e-5
REFERENCE_RTOL = 2e-5


def make_module() -> ActionConditionedVJEPA:
    return ActionConditionedVJEPA(
        image_channels=3,
        proprio_dim=5,
        action_dim=4,
        latent_dim=16,
        event_dim=2,
        max_horizon=2,
        ema_momentum=0.99,
    )


def batch_stream(rank: int, batch_size: int = 2) -> Iterator[Dict[str, torch.Tensor]]:
    """Different, deterministic rank-local data prevents a weak identical-input test."""

    generator = torch.Generator().manual_seed(900 + rank)
    while True:
        yield {
            "context_video": torch.randn(batch_size, 1, 3, 16, 16, generator=generator),
            "context_proprio": torch.randn(batch_size, 1, 5, generator=generator),
            "executed_actions": torch.randn(batch_size, 2, 4, generator=generator),
            "future_video": torch.randn(batch_size, 2, 3, 16, 16, generator=generator),
            "future_proprio": torch.randn(batch_size, 2, 5, generator=generator),
            "future_events": torch.randint(0, 2, (batch_size, 2, 2), generator=generator),
        }


def to_cpu_tree(value: Any) -> Any:
    """Copy tensors so snapshot comparison cannot observe later in-place changes."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def assert_tree_close(
    reference: Any,
    candidate: Any,
    path: str = "root",
    *,
    atol: float = REPLICA_ATOL,
    rtol: float = REPLICA_RTOL,
) -> int:
    """Recursively compare every tensor and scalar, returning compared tensor count."""

    if isinstance(reference, torch.Tensor):
        assert isinstance(candidate, torch.Tensor), f"{path}: tensor/non-tensor mismatch"
        assert reference.shape == candidate.shape, f"{path}: shape {reference.shape} != {candidate.shape}"
        assert reference.dtype == candidate.dtype, f"{path}: dtype {reference.dtype} != {candidate.dtype}"
        if reference.is_floating_point() or reference.is_complex():
            max_error = float((reference - candidate).abs().max().item()) if reference.numel() else 0.0
            assert torch.allclose(reference, candidate, atol=atol, rtol=rtol), (
                f"{path}: max_abs_error={max_error}, atol={atol}, rtol={rtol}"
            )
        else:
            assert torch.equal(reference, candidate), f"{path}: integral/bool tensor differs"
        return 1
    if isinstance(reference, Mapping):
        assert isinstance(candidate, Mapping), f"{path}: mapping/non-mapping mismatch"
        assert set(reference) == set(candidate), f"{path}: mapping keys differ"
        return sum(
            assert_tree_close(reference[key], candidate[key], f"{path}.{key}", atol=atol, rtol=rtol)
            for key in reference
        )
    if isinstance(reference, (list, tuple)):
        assert isinstance(candidate, type(reference)), f"{path}: sequence type differs"
        assert len(reference) == len(candidate), f"{path}: sequence length differs"
        return sum(
            assert_tree_close(left, right, f"{path}[{index}]", atol=atol, rtol=rtol)
            for index, (left, right) in enumerate(zip(reference, candidate))
        )
    assert reference == candidate, f"{path}: {reference!r} != {candidate!r}"
    return 0


def reference_update(
    module: ActionConditionedVJEPA,
    optimizer: torch.optim.Optimizer,
    all_rank_batches: List[List[Dict[str, torch.Tensor]]],
    *,
    gradient_clip_norm: float,
) -> None:
    """One process computes the exact global six-sample objective and one AdamW step."""

    flat_batches = [batch for rank_batches in all_rank_batches for batch in rank_batches]
    sample_counts = [int(batch["context_video"].shape[0]) for batch in flat_batches]
    total_valid_samples = sum(sample_counts)
    assert total_valid_samples == 6, f"expected six samples, received {total_valid_samples}"
    optimizer.zero_grad(set_to_none=True)
    global_mean_loss = torch.zeros((), dtype=torch.float32)
    for batch, count in zip(flat_batches, sample_counts):
        prediction = module(batch["context_video"], batch["context_proprio"], batch["executed_actions"])
        targets = module.target_latents(batch["future_video"], batch["future_proprio"])
        losses = action_conditioned_jepa_loss(prediction, targets, batch["future_events"])
        global_mean_loss = global_mean_loss + losses.total * (count / total_valid_samples)
    global_mean_loss.backward()
    # Must match acvjepa_dynamic_update exactly. Omitting this operation caused
    # an apparent optimizer-state mismatch that was a test-oracle defect, not a
    # DDP replica divergence.
    torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_norm)
    optimizer.step()
    module.update_ema_target()


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = get_rank()
    world = get_world_size()
    assert world == 2, "this regression test explicitly targets a two-rank 2:1 plan"

    # Initial state is bit-identical across ranks before DDP. Each rank then gets
    # different deterministic input batches to exercise true gradient averaging.
    torch.manual_seed(12345)
    ddp = DDP(make_module())
    optimizer = torch.optim.AdamW([parameter for parameter in ddp.parameters() if parameter.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    planner = DynamicAccumulationPlanner(target_update_ms=120.0, min_micro_batches=1, max_micro_batches=3) if rank == 0 else None
    telemetry = RankTelemetry(
        rank=rank,
        samples_per_second=40.0 if rank == 0 else 17.0,
        p95_step_ms=100.0,
        p95_data_ms=12.0,
        p95_allreduce_ms=10.0,
        free_memory_gb=8.0,
        healthy=True,
    )
    plan = next_update_plan(
        planner=planner,
        local_telemetry=telemetry,
        local_micro_batch_size=2,
        device=torch.device("cpu"),
    )
    assert plan.ranks[0].micro_batches == 2 and plan.ranks[1].micro_batches == 1
    assert plan.global_samples == 6

    stream = batch_stream(rank)
    # Materialize exactly the batches used by the distributed update, so rank 0
    # can later replay the same global sample set in a single-process reference.
    local_batches = [next(stream) for _ in range(plan.ranks[rank].micro_batches)]
    metrics = acvjepa_dynamic_update(
        ddp=ddp,
        optimizer=optimizer,
        scaler=scaler,
        plan=plan,
        micro_batches=iter(local_batches),
        device=torch.device("cpu"),
        config=DynamicStepConfig(amp=False, gradient_clip_norm=1.0),
    )
    assert int(metrics["local_samples"]) == (4 if rank == 0 else 2)
    assert int(metrics["global_samples"]) == 6

    # all_gather_object is test-only and all participants are trusted local CI
    # ranks. It deliberately is not reused in the production control plane.
    all_batches: List[Any] = [None] * world
    dist.all_gather_object(all_batches, to_cpu_tree(local_batches))
    snapshot = {
        "model_state": to_cpu_tree(ddp.module.state_dict()),
        "optimizer_state": to_cpu_tree(optimizer.state_dict()),
    }
    all_snapshots: List[Any] = [None] * world
    dist.all_gather_object(all_snapshots, snapshot)

    if rank == 0:
        # Cross-rank comparison includes all student parameters, target EMA
        # parameters, any future buffers, Adam moments, step counters and groups.
        replica_tensors = 0
        for peer_rank in range(1, world):
            replica_tensors += assert_tree_close(snapshot, all_snapshots[peer_rank], f"rank0_vs_rank{peer_rank}")

        # Rebuild the exact initial module and replay the same rank0/rank1 batches
        # as one six-sample global objective. This validates the loss scaling,
        # not merely equality between two potentially identically-wrong replicas.
        torch.manual_seed(12345)
        reference_module = make_module()
        reference_optimizer = torch.optim.AdamW(
            [parameter for parameter in reference_module.parameters() if parameter.requires_grad], lr=1e-3
        )
        reference_update(
            reference_module,
            reference_optimizer,
            all_batches,
            gradient_clip_norm=1.0,
        )
        reference_tensors = assert_tree_close(
            to_cpu_tree(reference_module.state_dict()),
            snapshot["model_state"],
            "single_process_reference.model_state",
            atol=REFERENCE_ATOL,
            rtol=REFERENCE_RTOL,
        )
        reference_tensors += assert_tree_close(
            to_cpu_tree(reference_optimizer.state_dict()),
            snapshot["optimizer_state"],
            "single_process_reference.optimizer_state",
            atol=REFERENCE_ATOL,
            rtol=REFERENCE_RTOL,
        )
        print(
            json.dumps(
                {
                    "full_state_equivalence_test": "passed",
                    "global_samples": plan.global_samples,
                    "rank0_local_samples": plan.ranks[0].local_samples,
                    "rank1_local_samples": plan.ranks[1].local_samples,
                    "cross_rank_tensor_entries": replica_tensors,
                    "reference_tensor_entries": reference_tensors,
                    "replica_atol": REPLICA_ATOL,
                    "replica_rtol": REPLICA_RTOL,
                    "reference_atol": REFERENCE_ATOL,
                    "reference_rtol": REFERENCE_RTOL,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
