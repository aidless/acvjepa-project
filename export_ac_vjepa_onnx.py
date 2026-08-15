"""Export the inference-only AC-VJEPA subgraph to a static-shape ONNX model.

The exported graph contains only:
  context_video + context_proprio + candidate ActionBlocks
    -> future_latents + log_variance + event_logits

It intentionally excludes EMA target encoding, training losses, LLM orchestration,
MPC, robot drivers, and all safety logic.

Example:
  python3 export_ac_vjepa_onnx.py \
      --checkpoint /checkpoints/ac_vjepa/last.pt \
      --output /artifacts/ac_vjepa_b1_t4_h3_224.onnx \
      --context-steps 4 --horizon 3 --height 224 --width 224
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from ac_vjepa_core import ActionConditionedVJEPA


class ACVJEPAInferenceWrapper(nn.Module):
    """ONNX-friendly wrapper that returns tensors rather than a Python dataclass."""

    def __init__(self, model: ActionConditionedVJEPA):
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        context_video: torch.Tensor,
        context_proprio: torch.Tensor,
        action_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self.model.predict(context_video, context_proprio, action_blocks)
        return prediction.future_latents, prediction.log_variance, prediction.event_logits


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export AC-VJEPA inference core to static ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--context-steps", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--opset", type=int, default=17)
    return p


def main() -> None:
    args = parser().parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]

    model = ActionConditionedVJEPA(
        image_channels=int(config.get("image_channels", 3)),
        proprio_dim=int(config["proprio_dim"]),
        action_dim=int(config["action_dim"]),
        latent_dim=int(config["latent_dim"]),
        event_dim=int(config["event_dim"]),
        max_horizon=int(config["max_horizon"]),
        ema_momentum=float(config.get("ema_momentum", 0.996)),
    )
    model.load_state_dict(payload["model"], strict=True)
    wrapper = ACVJEPAInferenceWrapper(model).eval()

    if args.horizon > int(config["max_horizon"]):
        raise ValueError("requested horizon exceeds the checkpoint's max_horizon")

    video = torch.zeros(
        args.batch_size,
        args.context_steps,
        int(config.get("image_channels", 3)),
        args.height,
        args.width,
        dtype=torch.float32,
    )
    proprio = torch.zeros(
        args.batch_size, args.context_steps, int(config["proprio_dim"]), dtype=torch.float32
    )
    actions = torch.zeros(
        args.batch_size, args.horizon, int(config["action_dim"]), dtype=torch.float32
    )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (video, proprio, actions),
        destination,
        input_names=["context_video", "context_proprio", "action_blocks"],
        output_names=["future_latents", "log_variance", "event_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        # Static shapes are intentional: on edge devices, fixed batch/window/
        # resolution avoids engine rebuilds and lowers p99 latency jitter.
        dynamic_axes=None,
    )

    metadata = {
        "checkpoint": str(args.checkpoint),
        "model_version": payload.get("config", {}).get("model_version", "unknown"),
        "action_schema_version": payload.get("action_schema_version"),
        "preprocess_version": payload.get("preprocess_version"),
        "static_input_shapes": {
            "context_video": list(video.shape),
            "context_proprio": list(proprio.shape),
            "action_blocks": list(actions.shape),
        },
        "outputs": ["future_latents", "log_variance", "event_logits"],
        "opset": args.opset,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"onnx": str(destination), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
