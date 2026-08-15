"""Two-process CPU/Gloo integration test for acvjepa_dynamic_update.

Run:
  torchrun --standalone --nproc_per_node=2 test_dynamic_nccl_acvjepa_integration.py

The production path selects NCCL when CUDA is available; Gloo only validates the
collective ordering and variable-local-accumulation semantics in this CPU CI test.
"""
from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import Dict, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ac_vjepa_core import ActionConditionedVJEPA
from adaptive_ddp_accumulation import DynamicAccumulationPlanner, RankTelemetry
from dynamic_nccl_update_plan_train import (
    DynamicStepConfig,
    acvjepa_dynamic_update,
    get_rank,
    get_world_size,
    next_update_plan,
)


def stream(rank: int, batch_size: int = 2) -> Iterator[Dict[str, torch.Tensor]]:
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


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = get_rank()
    torch.manual_seed(12345)  # All replicas start from the same state.
    module = ActionConditionedVJEPA(
        image_channels=3,
        proprio_dim=5,
        action_dim=4,
        latent_dim=16,
        event_dim=2,
        max_horizon=2,
        ema_momentum=0.99,
    )
    ddp = DDP(module)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
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
    metrics = acvjepa_dynamic_update(
        ddp=ddp,
        optimizer=optimizer,
        scaler=scaler,
        plan=plan,
        micro_batches=stream(rank),
        device=torch.device("cpu"),
        config=DynamicStepConfig(amp=False, gradient_clip_norm=1.0),
    )

    # One parameter tensor is enough to assert that rank replicas remain identical.
    final_weight = ddp.module.student_encoder.frame_encoder.net[0].weight.detach().clone()
    replicas = [torch.empty_like(final_weight) for _ in range(get_world_size())]
    dist.all_gather(replicas, final_weight)
    assert all(torch.allclose(final_weight, peer, atol=1e-6) for peer in replicas)
    assert int(metrics["global_samples"]) == 6
    if rank == 0:
        print(json.dumps({"integration_test": "passed", **metrics}, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
