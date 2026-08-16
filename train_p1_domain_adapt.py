"""P1 domain-adaptation training for the frozen V-JEPA encoder (PROJECT_PLAN M2).

Purpose
-------
P1 无动作 JEPA 域适配（轻量 V-JEPA 方案 §5.2）：在**无标签** B 层视频窗口
（video_to_windows.py 产出，executed_actions / future_events 恒为零）上，
微调轻量预测器 + 投影 head，使冻结的官方 V-JEPA 编码器适应厨房/台面领域。

Loss（域适配变体）——只保留潜在预测与不确定性校准，去掉动作条件与事件项：
    L = latent_nll + latent_cosine + calibration_weight * calibration
这与动作条件训练共用同一个预测器结构，但零动作 token 表示"无动作条件预测"。

入口复用
--------
- train_ac_vjepa_ddp.py 的 DDP 初始化 / EMA / checkpoint / loader 工具；
- vjepa_backbone.install_hf_vjepa2_encoder 安装真实 V-JEPA 2.1 权重（frozen）。

使用
----
    python train_p1_domain_adapt.py \
        --manifest <domain_adapt_windows.jsonl> \
        --output <p1_out> \
        --init-from vjepa2hf:<path>:frozen \
        --init-img-size 384

本机冒烟（CPU，2 进程）：
    python scripts/manual_gloo_runner.py train_p1_domain_adapt.py \
        --manifest ... --output ... --init-from vjepa2hf:<ckpt>:frozen --epochs 1
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ac_vjepa_core import ActionConditionedVJEPA, action_conditioned_jepa_loss
from train_ac_vjepa_ddp import (
    TrainConfig,
    WindowEpisodeDataset,
    broadcast_module,
    build_loader,
    cleanup_distributed,
    is_primary,
    move_batch,
    rank,
    reduce_mean,
    save_checkpoint,
    set_seed,
    setup_distributed,
    world_size,
)
from train_ac_vjepa_ddp import init_from_vjepa2


@dataclass
class DomainAdaptConfig(TrainConfig):
    calibration_weight: float = 0.05
    # All other fields inherited; executed_actions / future_events are zero in
    # B-layer windows and are deliberately excluded from the P1 loss.


def build_domain_adapt_config(args: argparse.Namespace) -> DomainAdaptConfig:
    # H-T4 sync arm: momentum 0.0 => target 每步硬拷贝 student head (无平滑)。
    # EMA 公式 target = m*target + (1-m)*source (ac_vjepa_core.EMAStateEncoder)，
    # 对应预注册 A/B 的 B 臂（同步目标）。
    effective_ema_momentum = 0.0 if args.ema_target == "sync" else args.ema_momentum
    return DomainAdaptConfig(
        manifest=args.manifest,
        output=args.output,
        epochs=args.epochs,
        per_rank_batch_size=args.per_rank_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        max_horizon=args.max_horizon,
        init_from=args.init_from,
        init_img_size=args.init_img_size,
        ema_momentum=effective_ema_momentum,
        ema_broadcast_interval=args.ema_broadcast_interval,
        save_interval_steps=args.save_interval_steps,
        log_interval_steps=args.log_interval_steps,
        amp=args.amp,
        static_graph=args.static_graph,
        comm_hook=args.comm_hook,
        seed=args.seed,
        calibration_weight=args.calibration_weight,
    )


def train_domain_adapt(config: DomainAdaptConfig) -> None:
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
                latent_dim=config.latent_dim,
                img_size=config.init_img_size,
            )
        else:
            from train_ac_vjepa_ddp import load_incremental_parent

            load_incremental_parent(module, config.init_from)
        # Backbone installs happen after the initial .to(device); move the newly
        # attached encoder (HF model loads on CPU) to the training device.
        module = module.to(device)

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

    optimizer = torch.optim.AdamW(
        [parameter for parameter in ddp.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(config.amp and device.type == "cuda"))
    output_dir = Path(config.output)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    if is_primary():
        print(
            json.dumps(
                {
                    "mode": "p1_domain_adapt",
                    "manifest": config.manifest,
                    "init_from": config.init_from,
                    "world_size": world_size(),
                    "epochs": config.epochs,
                    "windows": len(dataset),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    try:
        for epoch in range(config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            ddp.train()
            for batch_index, raw_batch in enumerate(loader):
                batch = move_batch(raw_batch, device)
                is_update_step = (batch_index + 1) % config.gradient_accumulation == 0
                if isinstance(ddp, DDP) and not is_update_step:
                    sync_context = ddp.no_sync()
                else:
                    sync_context = nullcontext()

                with sync_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=(config.amp and device.type == "cuda"),
                    ):
                        # B-layer windows have zero actions/events; the P1 loss
                        # ignores both (event_targets=None -> event term zero).
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
                            prediction,
                            targets,
                            event_targets=None,
                            event_weight=0.0,
                            calibration_weight=config.calibration_weight,
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

                if global_step % config.ema_broadcast_interval == 0:
                    broadcast_module(module_ref.target_encoder)

                if global_step % config.log_interval_steps == 0:
                    metrics = {
                        "mode": "p1_domain_adapt",
                        "epoch": epoch,
                        "step": global_step,
                        "world_size": world_size(),
                        "loss_total": reduce_mean(losses.total),
                        "loss_latent_nll": reduce_mean(losses.latent_nll),
                        "loss_cosine": reduce_mean(losses.latent_cosine),
                        "loss_calibration": reduce_mean(losses.calibration),
                        "event_term": 0.0,  # explicitly excluded in P1
                    }
                    if is_primary():
                        print(json.dumps(metrics), flush=True)

                if global_step % config.save_interval_steps == 0:
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    save_checkpoint(
                        output_dir / f"p1-step-{global_step:08d}.pt",
                        ddp, optimizer, scaler, config, epoch, global_step,
                    )
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.barrier()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        save_checkpoint(output_dir / "p1-last.pt", ddp, optimizer, scaler, config, config.epochs, global_step)
    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 domain adaptation on unlabeled B-layer windows")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per-rank-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--init-from", help="vjepa2:<path>[:mode] or vjepa2hf:<path>[:frozen|finetune]")
    parser.add_argument("--init-img-size", type=int, default=384)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument(
        "--ema-target",
        choices=["ema", "sync"],
        default="ema",
        help="H-T4 A/B (preregistered 2026-08-15): ema=EMA target (momentum 0.996 default), "
        "sync=hard-copy sync target (momentum 0.0, no smoothing)",
    )
    parser.add_argument("--ema-broadcast-interval", type=int, default=100)
    parser.add_argument("--save-interval-steps", type=int, default=1000)
    parser.add_argument("--log-interval-steps", type=int, default=50)
    parser.add_argument("--calibration-weight", type=float, default=0.05)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--static-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comm-hook", choices=["none", "fp16", "bf16", "powersgd"], default="none")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    train_domain_adapt(build_domain_adapt_config(parse_args()))
