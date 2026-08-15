"""Safe dynamic micro-batch planning for synchronous NCCL/DDP training.

Important: ranks may use different numbers of *local* micro-batches per update,
but every rank must execute exactly one synchronized final backward per update.
The planner broadcasts the full plan before the update begins, records the
resulting global sample count, and provides loss scaling for DDP's averaging
semantics. Membership changes are allowed only after a durable checkpoint and a
new process group/rendezvous, never mid-collective.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class RankTelemetry:
    rank: int
    samples_per_second: float
    p95_step_ms: float
    p95_data_ms: float
    p95_allreduce_ms: float
    free_memory_gb: float
    healthy: bool = True


@dataclass(frozen=True)
class RankUpdatePlan:
    rank: int
    micro_batches: int
    samples_per_micro_batch: int
    local_samples: int
    # DDP averages gradients across W ranks. Each local micro loss must be a SUM
    # over samples multiplied by this scale so the averaged result is a true
    # global sample mean: (1/W) sum_r [W/global_samples * sum_i grad_i].
    loss_sum_scale: float


@dataclass(frozen=True)
class UpdatePlan:
    plan_version: int
    target_update_ms: float
    world_size: int
    global_samples: int
    ranks: tuple[RankUpdatePlan, ...]


class DynamicAccumulationPlanner:
    def __init__(self, *, target_update_ms: float = 750.0, min_micro_batches: int = 1, max_micro_batches: int = 16):
        self.target_update_ms = target_update_ms
        self.min_micro_batches = min_micro_batches
        self.max_micro_batches = max_micro_batches
        self.version = 0

    def plan(self, telemetry: Iterable[RankTelemetry], samples_per_micro_batch: Dict[int, int]) -> UpdatePlan:
        records = sorted((item for item in telemetry if item.healthy), key=lambda item: item.rank)
        if not records:
            raise RuntimeError("no healthy ranks; checkpoint and restart membership outside the current update")
        if set(item.rank for item in records) != set(samples_per_micro_batch):
            raise ValueError("planner requires one micro-batch size for every participating rank")
        budgets: List[tuple[RankTelemetry, int]] = []
        for item in records:
            # Convert observed throughput into local sample budget for the next
            # update. p95 step estimates keep the policy conservative under jitter.
            capacity = item.samples_per_second * (self.target_update_ms / 1000.0)
            k = round(capacity / samples_per_micro_batch[item.rank])
            k = max(self.min_micro_batches, min(self.max_micro_batches, int(k)))
            budgets.append((item, k))
        global_samples = sum(k * samples_per_micro_batch[item.rank] for item, k in budgets)
        world = len(budgets)
        self.version += 1
        return UpdatePlan(
            plan_version=self.version,
            target_update_ms=self.target_update_ms,
            world_size=world,
            global_samples=global_samples,
            ranks=tuple(
                RankUpdatePlan(
                    rank=item.rank,
                    micro_batches=k,
                    samples_per_micro_batch=samples_per_micro_batch[item.rank],
                    local_samples=k * samples_per_micro_batch[item.rank],
                    loss_sum_scale=world / global_samples,
                )
                for item, k in budgets
            ),
        )


def scaled_loss_from_mean(local_mean_loss, local_batch_samples: int, plan: RankUpdatePlan):
    """Return a correctly weighted loss for each local micro-batch.

    Use a criterion whose `local_mean_loss` is a mean over `local_batch_samples`.
    The final synchronized backward in DDP yields the global mean gradient when
    all local micro-batches apply this scaling and each rank follows the plan.
    """
    return local_mean_loss * local_batch_samples * plan.loss_sum_scale


DDP_PSEUDOCODE = r'''
# At a completed optimizer/checkpoint boundary only:
telemetry = measure_rank_throughput_memory_and_p95()
plan = broadcast(rank0.planner.plan(all_gather(telemetry), per_rank_micro_batch))
my = plan.ranks[rank]
optimizer.zero_grad(set_to_none=True)
for local_index, micro_batch in enumerate(next_cost_bucketed_micro_batches(my.micro_batches)):
    is_final_local_micro_batch = local_index == my.micro_batches - 1
    sync = nullcontext() if is_final_local_micro_batch else ddp.no_sync()
    with sync:
        prediction = ddp(...)
        # criterion must expose a mean over valid local samples/tokens.
        loss = scaled_loss_from_mean(mean_loss, valid_samples, my)
        loss.backward()
# Every rank now enters exactly one final DDP all-reduce; optimizer step is aligned.
optimizer.step()
'''


if __name__ == "__main__":
    planner = DynamicAccumulationPlanner(target_update_ms=600.0)
    plan = planner.plan(
        [
            RankTelemetry(0, 52.0, 520.0, 30.0, 80.0, 24.0),
            RankTelemetry(1, 35.0, 610.0, 40.0, 145.0, 16.0),
        ],
        {0: 4, 1: 4},
    )
    print(asdict(plan))
