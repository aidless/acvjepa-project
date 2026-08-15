"""Optional DDP communication hooks for bandwidth-constrained AC-VJEPA training.

Use only after establishing a full-precision DDP baseline. Gradient compression
changes optimization dynamics and must be evaluated against the same fixed
validation, calibration and closed-loop safety suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class CommHookConfig:
    name: str = "none"  # none | fp16 | bf16 | powersgd
    power_sgd_rank: int = 1
    power_sgd_start_iter: int = 500
    power_sgd_min_compression_rate: float = 2.0
    power_sgd_error_feedback: bool = True
    power_sgd_warm_start: bool = True


def register_comm_hook(ddp_model: DDP, config: CommHookConfig) -> Optional[Any]:
    """Register exactly one PyTorch-supported DDP communication hook.

    Returns the hook state, which callers may retain for logging. This function
    intentionally does not provide lossy top-k/sparsity hooks: those require a
    carefully designed error-feedback/residual mechanism and should not be
    introduced into a safety-relevant world-model training pipeline as an
    unvalidated ad-hoc optimization.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return None

    name = config.name.lower()
    if name == "none":
        return None

    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook

    if name == "fp16":
        ddp_model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
        return {"name": "fp16"}

    if name == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 DDP compression hook requires a BF16-capable CUDA device")
        ddp_model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
        return {"name": "bf16"}

    if name == "powersgd":
        state = powerSGD_hook.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=config.power_sgd_rank,
            start_powerSGD_iter=config.power_sgd_start_iter,
            min_compression_rate=config.power_sgd_min_compression_rate,
            use_error_feedback=config.power_sgd_error_feedback,
            warm_start=config.power_sgd_warm_start,
        )
        ddp_model.register_comm_hook(state=state, hook=powerSGD_hook.powerSGD_hook)
        return state

    raise ValueError(f"unsupported DDP communication hook: {config.name}")
