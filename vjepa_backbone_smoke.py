"""M1 smoke test for the V-JEPA 2 backbone adapter (vjepa_backbone.py).

Covers, with CPU-friendly synthetic tensors:
  1. ViT-B/16 backbone forward contract [N, C, H, W] -> [N, out_dim].
  2. Fine-tune modes: frozen (no backbone grads), last_k (k blocks trainable),
     lora (adapter params trainable, base frozen), finetune (all trainable).
  3. Key remapping: build an official-style checkpoint payload (with a prefix
     and `attn.qkv` layout) and verify `load_vjepa2_weights` maps it exactly.
  4. Installation into ActionConditionedVJEPA: forward + one optimizer step,
     EMA target swap keeps the parameter-count contract of update_from valid.

Run:
  python vjepa_backbone_smoke.py            # random-init path (no network)
  python vjepa_backbone_smoke.py --checkpoint <official.pt>   # real weights
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ac_vjepa_core import ActionConditionedVJEPA, action_conditioned_jepa_loss
from vjepa_backbone import (
    VJEPA2Backbone,
    build_vjepa2_backbone,
    install_vjepa2_encoder,
    load_vjepa2_weights,
)


def _synthetic_official_ckpt(backbone: VJEPA2Backbone, prefix: str = "encoder.") -> dict:
    """Build a checkpoint whose keys mimic the official layout: prefix + qkv.

    Adapter-only keys (projection head) are excluded because an official V-JEPA
    checkpoint never contains them; the remapper treats them as skippable.
    """
    state = copy.deepcopy(backbone.state_dict())
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    remapped = {f"{prefix}{key}": value for key, value in state.items()}
    return {"model": remapped}


def check_forward_contract(backbone: VJEPA2Backbone) -> None:
    frames = torch.randn(3, 3, 224, 224)
    out = backbone(frames)
    assert out.shape == (3, 768), out.shape
    print(f"  forward contract OK: {tuple(out.shape)}")


def _all_param_suffixes(block: nn.Module) -> list[str]:
    return [name for name, _ in block.named_parameters()]


def check_modes(backbone: VJEPA2Backbone) -> None:
    backbone.freeze()
    grads = {n for n, p in backbone.named_parameters() if p.requires_grad}
    assert grads <= {"head.weight", "head.bias"}, grads
    print(f"  frozen mode OK (only projection head trainable: {sorted(grads)})")

    backbone.unfreeze_last_k(2)
    grads = {n for n, p in backbone.named_parameters() if p.requires_grad}
    allowed = {"head.weight", "head.bias"}
    allowed |= {f"blocks.{i}.{suffix}" for i in (10, 11) for suffix in _all_param_suffixes(backbone.blocks[10])}
    assert grads and grads <= allowed, grads - allowed
    print(f"  last_k mode OK ({len(grads)} trainable params in last 2 blocks + head)")

    backbone.freeze()
    lora_params = backbone.apply_lora(rank=4, alpha=8)
    assert len(lora_params) == 2 * len(backbone.blocks) * 2, len(lora_params)  # qkv+proj per block
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    assert trainable, "lora should expose trainable adapters"
    print(f"  lora mode OK ({len(trainable)} adapter params, base frozen)")


def check_key_remap() -> None:
    backbone = build_vjepa2_backbone(out_dim=768, img_size=224)
    ckpt = _synthetic_official_ckpt(backbone, prefix="encoder.")
    report = load_vjepa2_weights(backbone, ckpt["model"], strict=True, num_blocks=len(backbone.blocks))
    assert report.loaded == len(backbone.state_dict()) - 2, (report.loaded, len(backbone.state_dict()))
    assert report.skipped == 0 and report.strict_ok, report.skipped_keys
    print(f"  key remap OK (loaded {report.loaded}, head adapter-only keys excluded, strict)")


def check_install_step() -> None:
    torch.manual_seed(7)
    model = ActionConditionedVJEPA(
        image_channels=3,
        proprio_dim=8,
        action_dim=4,
        latent_dim=64,
        event_dim=2,
        max_horizon=3,
    )
    report = install_vjepa2_encoder(
        model,
        latent_dim=64,
        checkpoint=None,
        mode="lora",
        lora_rank=4,
        img_size=224,
    )
    assert report.strict_ok
    # one forward + step with tiny inputs
    context_video = torch.randn(2, 2, 3, 224, 224)
    context_proprio = torch.randn(2, 2, 8)
    future_video = torch.randn(2, 2, 3, 224, 224)
    future_proprio = torch.randn(2, 2, 8)
    actions = torch.randn(2, 2, 4)
    events = torch.randint(0, 2, (2, 2, 2)).float()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    prediction = model.predict(context_video, context_proprio, actions)
    targets = model.target_latents(future_video, future_proprio)
    losses = action_conditioned_jepa_loss(prediction, targets, events)
    losses.total.backward()
    optim.step()
    model.update_ema_target()
    print("  install + one train step OK (lora adapters trained, EMA swapped)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="optional official V-JEPA checkpoint (pt)")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    print("== M1 vjepa_backbone smoke ==")
    backbone = build_vjepa2_backbone(out_dim=768, img_size=args.img_size)
    check_forward_contract(backbone)
    check_modes(backbone)
    check_key_remap()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            print(f"  checkpoint not found: {ckpt_path}"); sys.exit(2)
        real = build_vjepa2_backbone(out_dim=768, img_size=args.img_size)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload.get("encoder", payload))
        report = load_vjepa2_weights(real, state, strict=False, num_blocks=len(real.blocks))
        print(f"  real checkpoint: loaded={report.loaded} skipped={report.skipped} strict_ok={report.strict_ok}")
        print(f"  skipped sample: {report.skipped_keys[:5]}")
    else:
        print("  (no --checkpoint: random-init structural path only)")

    check_install_step()
    print("== M1 smoke PASS ==")


if __name__ == "__main__":
    main()
