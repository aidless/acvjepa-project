"""Distributed AC-V-JEPA training reference implementation.

Launch examples (run only after verifying network, NCCL and shared/replicated data):

  # One node, 4 GPUs
  torchrun --standalone --nproc_per_node=4 train_ac_vjepa_ddp.py \
      --manifest /data/manifest.jsonl --output /checkpoints/ac_vjepa

  # Two nodes, 4 GPUs each; run once per node with the correct node_rank
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR --master_port=29500 \
      train_ac_vjepa_ddp.py --manifest /data/manifest.jsonl --output /checkpoints/ac_vjepa

The manifest is JSONL with one object per line, e.g.:
  {"path": "/data/windows/episode_000001.pt"}

Each .pt file must contain CPU tensors with these exact keys:
  context_video:    [T_context, C, H, W]
  context_proprio:  [T_context, P]
  future_video:     [T_horizon, C, H, W]
  future_proprio:   [T_horizon, P]
  executed_actions: [T_horizon, A]
  future_events:    [T_horizon, E]

Safety/data semantics: `executed_actions` must be the actual, limit-clipped
ActionBlocks executed by the robot or simulator, not a high-level language plan.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ac_vjepa_core import ActionConditionedVJEPA, action_conditioned_jepa_loss
from ac_vjepa_comm_hooks import CommHookConfig, register_comm_hook


REQUIRED_KEYS = {
    "context_video",
    "context_proprio",
    "future_video",
    "future_proprio",
    "executed_actions",
    "future_events",
}


@dataclass
class TrainConfig:
    manifest: str
    output: str
    epochs: int = 10
    per_rank_batch_size: int = 4
    gradient_accumulation: int = 4
    num_workers: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    clip_grad_norm: float = 1.0
    image_channels: int = 3
    proprio_dim: int = 8
    action_dim: int = 20
    latent_dim: int = 128
    event_dim: int = 4
    max_horizon: int = 8
    ema_momentum: float = 0.996
    ema_broadcast_interval: int = 100
    save_interval_steps: int = 1000
    log_interval_steps: int = 50
    amp: bool = True
    static_graph: bool = False
    comm_hook: str = "none"
    power_sgd_rank: int = 1
    power_sgd_start_iter: int = 500
    seed: int = 2026
    init_from: Optional[str] = None
    init_lora_rank: int = 8
    init_unfreeze_last_k: int = 1
    init_img_size: int = 384


class WindowEpisodeDataset(Dataset[Dict[str, torch.Tensor]]):
    """Map-style, per-window dataset suitable for DistributedSampler.

    It deliberately loads an already validated window format. For high-throughput
    production, replace per-file torch.load with WebDataset/LMDB/MDS or local
    shard streaming while preserving the same tensor contract and no cross-rank
    sample duplication.
    """

    def __init__(self, manifest_path: str):
        self.paths: List[Path] = []
        manifest = Path(manifest_path)
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            path = Path(entry["path"])
            if not path.is_file():
                raise FileNotFoundError(f"manifest line {line_number}: missing episode file {path}")
            self.paths.append(path)
        if not self.paths:
            raise ValueError("manifest contains no valid episode windows")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # weights_only avoids unpickling arbitrary Python objects in modern PyTorch.
        episode = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        missing = REQUIRED_KEYS.difference(episode)
        if missing:
            raise KeyError(f"{self.paths[index]} missing tensor keys: {sorted(missing)}")
        return {key: episode[key].float().contiguous() for key in REQUIRED_KEYS}


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_primary() -> bool:
    return rank() == 0


def setup_distributed() -> tuple[int, torch.device]:
    """Initialise torchrun environment; NCCL for GPUs, Gloo for CPU smoke tests."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    using_cuda = torch.cuda.is_available()
    if using_cuda and dist.is_nccl_available():
        backend = "nccl"
    else:
        # Windows torch wheels are often built without NCCL (USE_NCCL=OFF); a
        # single-GPU or CPU run still works on Gloo. Multi-GPU NCCL training
        # requires a cluster build (see CLUSTER_VALIDATION_RUNBOOK B1).
        backend = "gloo"
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    if using_cuda:
        torch.cuda.set_device(local_rank)
        return local_rank, torch.device("cuda", local_rank)
    return local_rank, torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    # Rank offset avoids identical augmentation/random masking across ranks while
    # retaining reproducibility for a fixed world size and rank allocation.
    seed = seed + rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()}


def reduce_mean(value: torch.Tensor) -> float:
    result = value.detach().float().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= world_size()
    return float(result.cpu())


@torch.no_grad()
def broadcast_module(module: nn.Module, src: int = 0) -> None:
    """Avoid EMA drift after optimizer/all-reduce updates across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor, src=src)


def atomic_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_checkpoint(
    path: Path,
    ddp_model: DDP | ActionConditionedVJEPA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    epoch: int,
    global_step: int,
) -> None:
    if not is_primary():
        return
    module = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    payload = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "global_step": global_step,
        "world_size": world_size(),
        # These must be versioned alongside the model in a real deployment.
        "action_schema_version": "action-block-v1",
        "preprocess_version": "camera-proprio-v1",
        "init_from": config.init_from,
    }
    atomic_save(payload, path)


def build_loader(config: TrainConfig, dataset: WindowEpisodeDataset) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size(), rank=rank(), shuffle=True, drop_last=True)
        if world_size() > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config.per_rank_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )
    return loader, sampler


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="DDP training for lightweight Action-conditioned V-JEPA")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per-rank-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument(
        "--init-from",
        help=(
            "incremental parent checkpoint, or 'vjepa2:<path>[:mode]' (native-format "
            "V-JEPA 2 encoder: frozen|last_k|lora|finetune) or 'vjepa2hf:<path>[:mode]' "
            "(HuggingFace-format real V-JEPA 2.1 safetensors: frozen|finetune)"
        ),
    )
    parser.add_argument("--init-img-size", type=int, default=384)
    parser.add_argument("--init-lora-rank", type=int, default=8)
    parser.add_argument("--init-unfreeze-last-k", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--static-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comm-hook", choices=["none", "fp16", "bf16", "powersgd"], default="none")
    parser.add_argument("--power-sgd-rank", type=int, default=1)
    parser.add_argument("--power-sgd-start-iter", type=int, default=500)
    args = parser.parse_args()
    return TrainConfig(
        manifest=args.manifest,
        output=args.output,
        epochs=args.epochs,
        per_rank_batch_size=args.per_rank_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        init_from=args.init_from,
        amp=args.amp,
        static_graph=args.static_graph,
        comm_hook=args.comm_hook,
        power_sgd_rank=args.power_sgd_rank,
        power_sgd_start_iter=args.power_sgd_start_iter,
        init_lora_rank=args.init_lora_rank,
        init_unfreeze_last_k=args.init_unfreeze_last_k,
        init_img_size=args.init_img_size,
    )


def load_incremental_parent(module: ActionConditionedVJEPA, checkpoint_path: str) -> None:
    """Load only a compatible, local parent model before DDP wrapping.

    Optimizer/scaler state is intentionally not inherited: this is a new,
    versioned incremental run. Strict model loading rejects architecture/schema
    drift rather than silently producing a partially initialized candidate.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"incremental parent checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model" not in payload:
        raise KeyError("incremental parent checkpoint missing model state")
    module.load_state_dict(payload["model"], strict=True)


def init_from_vjepa2(
    module: ActionConditionedVJEPA,
    spec: str,
    *,
    strict: bool = True,
    lora_rank: int = 8,
    unfreeze_last_k: int = 1,
    latent_dim: int = 128,
    img_size: int = 384,
) -> None:
    """Install an official V-JEPA 2 encoder as the frozen/adapted backbone.

    `spec` forms:
      vjepa2:<path>[:mode]    native-format checkpoint (key-remapped structural twin)
      vjepa2hf:<path>[:mode]  HuggingFace-format safetensors via transformers
                              VJEPA2Model (real V-JEPA 2.1 weights)
    mode is one of `frozen` (default), `last_k`, `lora`, `finetune` (last_k/lora
    apply to the native twin; HF path supports frozen|finetune).
    """
    parts = spec.split(":", 1)
    kind = parts[0]
    if kind not in ("vjepa2", "vjepa2hf") or len(parts) < 2:
        raise ValueError(f"invalid vjepa2 spec: {spec!r} (expected 'vjepa2:<path>[:mode]' or 'vjepa2hf:<path>[:mode]')")
    remainder = parts[1]
    # mode is an optional trailing segment; paths may contain ':' (Windows drive)
    mode = "frozen"
    for candidate in ("frozen", "finetune", "last_k", "lora"):
        suffix = f":{candidate}"
        if remainder.endswith(suffix):
            mode = candidate
            remainder = remainder[: -len(suffix)]
            break
    checkpoint = remainder
    if not checkpoint:
        raise ValueError(f"empty checkpoint in spec {spec!r}")
    if kind == "vjepa2hf":
        if mode not in ("frozen", "finetune"):
            raise ValueError(f"vjepa2hf mode must be frozen|finetune, got {mode!r}")
        from vjepa_backbone import install_hf_vjepa2_encoder

        report = install_hf_vjepa2_encoder(
            module,
            latent_dim=latent_dim,
            ckpt_path=checkpoint,
            mode=mode,
            img_size=img_size,
        )
    else:
        if mode not in ("frozen", "last_k", "lora", "finetune"):
            raise ValueError(f"unknown vjepa2 mode {mode!r}")
        from vjepa_backbone import install_vjepa2_encoder

        report = install_vjepa2_encoder(
            module,
            latent_dim=latent_dim,
            checkpoint=checkpoint,
            mode=mode,
            lora_rank=lora_rank,
            unfreeze_last_k=unfreeze_last_k,
            strict=strict,
        )
    if is_primary():
        print(
            json.dumps(
                {
                    "init": kind,
                    "mode": mode,
                    "checkpoint": checkpoint,
                    "loaded_keys": report.loaded,
                    "skipped_keys": len(report.skipped_keys),
                    "strict_ok": report.strict_ok,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def train(config: TrainConfig) -> None:
    local_rank, device = setup_distributed()
    set_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = WindowEpisodeDataset(config.manifest)
    loader, sampler = build_loader(config, dataset)

    module = ActionConditionedVJEPA(
        image_channels=config.image_channels,
        proprio_dim=config.proprio_dim,
        action_dim=config.action_dim,
        latent_dim=config.latent_dim,
        event_dim=config.event_dim,
        max_horizon=config.max_horizon,
        ema_momentum=config.ema_momentum,
    ).to(device)
    if config.init_from:
        if config.init_from.startswith("vjepa2:") or config.init_from.startswith("vjepa2hf:"):
            init_from_vjepa2(
                module,
                config.init_from,
                strict=True,
                lora_rank=config.init_lora_rank,
                unfreeze_last_k=config.init_unfreeze_last_k,
                latent_dim=config.latent_dim,
                img_size=config.init_img_size,
            )
        else:
            load_incremental_parent(module, config.init_from)
        # Backbone installs happen after the initial .to(device); move the newly
        # attached encoder (HF model loads on CPU) to the training device.
        module = module.to(device)

    # DDP is preferable while the 80M-class backbone is frozen or lightly tuned:
    # all ranks keep a full replica, then all-reduce gradients efficiently.
    ddp: DDP | ActionConditionedVJEPA
    if world_size() > 1:
        ddp = DDP(
            module,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            static_graph=config.static_graph,
        )
    else:
        ddp = module

    comm_hook_state = None
    if isinstance(ddp, DDP):
        comm_hook_state = register_comm_hook(
            ddp,
            CommHookConfig(
                name=config.comm_hook,
                power_sgd_rank=config.power_sgd_rank,
                power_sgd_start_iter=config.power_sgd_start_iter,
            ),
        )
    if is_primary() and config.comm_hook != "none":
        print(
            json.dumps(
                {
                    "comm_hook": config.comm_hook,
                    "power_sgd_rank": config.power_sgd_rank,
                    "power_sgd_start_iter": config.power_sgd_start_iter,
                    "note": "Validate compressed training against the full-precision baseline.",
                }
            ),
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in ddp.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(config.amp and device.type == "cuda"))
    output_dir = Path(config.output)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            ddp.train()

            for batch_index, raw_batch in enumerate(loader):
                batch = move_batch(raw_batch, device)
                is_update_step = (batch_index + 1) % config.gradient_accumulation == 0
                sync_context = (
                    ddp.no_sync() if isinstance(ddp, DDP) and not is_update_step else nullcontext()
                )

                with sync_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=(config.amp and device.type == "cuda"),
                    ):
                        # Must call ddp(...) rather than ddp.module.predict(...), so
                        # DDP installs autograd hooks and all-reduces gradients.
                        prediction = ddp(
                            batch["context_video"],
                            batch["context_proprio"],
                            batch["executed_actions"],
                        )
                        module_ref = ddp.module if isinstance(ddp, DDP) else ddp
                        targets = module_ref.target_latents(
                            batch["future_video"], batch["future_proprio"]
                        )
                        losses = action_conditioned_jepa_loss(
                            prediction, targets, batch["future_events"]
                        )
                        scaled_loss = losses.total / config.gradient_accumulation
                    scaler.scale(scaled_loss).backward()

                if not is_update_step:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                module_ref = ddp.module if isinstance(ddp, DDP) else ddp
                module_ref.update_ema_target()
                global_step += 1

                # Student parameters are synchronized by DDP all-reduce. Periodic
                # target broadcast prevents subtle EMA numerical drift across ranks.
                if global_step % config.ema_broadcast_interval == 0:
                    broadcast_module(module_ref.target_encoder)

                if global_step % config.log_interval_steps == 0:
                    # Every rank must enter each all-reduce; only rank 0 prints.
                    metrics = {
                        "epoch": epoch,
                        "step": global_step,
                        "world_size": world_size(),
                        "loss_total": reduce_mean(losses.total),
                        "loss_latent_nll": reduce_mean(losses.latent_nll),
                        "loss_event": reduce_mean(losses.event_bce),
                        "loss_calibration": reduce_mean(losses.calibration),
                    }
                    if is_primary():
                        print(json.dumps(metrics), flush=True)

                if global_step % config.save_interval_steps == 0:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    save_checkpoint(
                        output_dir / f"step-{global_step:08d}.pt",
                        ddp,
                        optimizer,
                        scaler,
                        config,
                        epoch,
                        global_step,
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        save_checkpoint(output_dir / "last.pt", ddp, optimizer, scaler, config, config.epochs, global_step)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train(parse_args())
