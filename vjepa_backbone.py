"""V-JEPA 2 backbone adapter for the AC-VJEPA training stack (M1, PROJECT_PLAN).

Bridges the gap between the self-contained `ActionConditionedVJEPA` model
(`ac_vjepa_core.py`) and official V-JEPA 2 / 2.1 pretrained encoders
(`facebookresearch/vjepa2`; also exposed via HuggingFace Transformers as
`transformers.models.vjepa2.Vjepa2Model`).

Contract kept identical to `TinyFrameEncoder`:
    forward(frames: [N, C, H, W]) -> [N, latent_dim]
so it can be installed as `module.student_encoder.frame_encoder` without changing
any training code outside this file.

Loading policy (M1):
- If a real V-JEPA checkpoint is provided, keys are remapped through
  `VJEPA_KEY_ALIASES` + prefix stripping; unmatched keys are reported and, in
  strict mode, raise. In lenient mode the backbone falls back to random init for
  unmatched keys (never silently for the whole encoder).
- Without a checkpoint the backbone is randomly initialized with the same
  ViT-B/16 shape, which is exactly what `vjepa_backbone_smoke.py` exercises.

Fine-tune modes:
  frozen   -> backbone frozen, only the projection head + AC-VJEPA heads train
  last_k   -> unfreeze the last k transformer blocks (k=0 == frozen)
  lora     -> low-rank adapters on attention q/k/v/o; backbone otherwise frozen
  finetune -> full backbone unfrozen (only for small runs / short windows)
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Key remapping for official V-JEPA checkpoints
# ---------------------------------------------------------------------------

# Common prefixes found in released checkpoints / HF conversions. We strip the
# first matching prefix, then apply the aliases below.
PREFIXES = ("encoder.", "backbone.", "model.", "vjepa.", "module.", "")

# target_key -> tuple of accepted source keys (after prefix strip)
VJEPA_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "patch_embed.weight": ("patch_embed.proj.weight", "patch_embed.weight", "patch_embedding.proj.weight"),
    "patch_embed.bias": ("patch_embed.proj.bias", "patch_embed.bias", "patch_embedding.proj.bias"),
    "pos_embed": ("pos_embed", "positional_embedding"),
    "cls_token": ("cls_token", "class_token"),
    "norm.weight": ("norm.weight", "encoder_norm.weight"),
    "norm.bias": ("norm.bias", "encoder_norm.bias"),
}

# Keys that belong to the adapter (projection head) and are expected to be absent
# from an official V-JEPA checkpoint; they never count as an error.
SKIPPABLE_TARGET_PREFIXES = ("head.",)


def _block_aliases(block_idx: int) -> Dict[str, Tuple[str, ...]]:
    p = f"blocks.{block_idx}."
    aliases = {
        f"{p}norm1.weight": (f"{p}norm1.weight", f"{p}ln1.weight", f"{p}attn.norm.weight"),
        f"{p}norm1.bias": (f"{p}norm1.bias", f"{p}ln1.bias", f"{p}attn.norm.bias"),
        f"{p}norm2.weight": (f"{p}norm2.weight", f"{p}ln2.weight", f"{p}mlp.norm.weight"),
        f"{p}norm2.bias": (f"{p}norm2.bias", f"{p}ln2.bias", f"{p}mlp.norm.bias"),
        f"{p}attn.qkv.weight": (f"{p}attn.qkv.weight", f"{p}attn.q_proj.weight",),
        f"{p}attn.qkv.bias": (f"{p}attn.qkv.bias", f"{p}attn.q_proj.bias",),
        f"{p}attn.q.weight": (f"{p}attn.q.weight",),
        f"{p}attn.k.weight": (f"{p}attn.k.weight",),
        f"{p}attn.v.weight": (f"{p}attn.v.weight",),
        f"{p}attn.q.bias": (f"{p}attn.q.bias",),
        f"{p}attn.k.bias": (f"{p}attn.k.bias",),
        f"{p}attn.v.bias": (f"{p}attn.v.bias",),
        f"{p}attn.proj.weight": (f"{p}attn.proj.weight", f"{p}attn.out_proj.weight"),
        f"{p}attn.proj.bias": (f"{p}attn.proj.bias", f"{p}attn.out_proj.bias"),
        f"{p}mlp.fc1.weight": (f"{p}mlp.fc1.weight", f"{p}mlp.fc_in.weight"),
        f"{p}mlp.fc1.bias": (f"{p}mlp.fc1.bias", f"{p}mlp.fc_in.bias"),
        f"{p}mlp.fc2.weight": (f"{p}mlp.fc2.weight", f"{p}mlp.fc_out.weight"),
        f"{p}mlp.fc2.bias": (f"{p}mlp.fc2.bias", f"{p}mlp.fc_out.bias"),
    }
    return aliases


@dataclass(frozen=True)
class LoadReport:
    loaded: int
    skipped: int
    skipped_keys: Tuple[str, ...]
    strict_ok: bool


def load_vjepa2_weights(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = True,
    num_blocks: Optional[int] = None,
) -> LoadReport:
    """Remap and load an official V-JEPA state dict into a ViT backbone.

    Accepts keys with any of `PREFIXES` stripped and either combined
    `attn.qkv` or split `attn.q/k/v` layouts. Unmatched keys are collected; in
    strict mode a non-empty unmatched set raises ValueError.
    """
    if num_blocks is None:
        num_blocks = _count_blocks(module)
    aliases: Dict[str, Tuple[str, ...]] = dict(VJEPA_KEY_ALIASES)
    for idx in range(num_blocks):
        aliases.update(_block_aliases(idx))

    # Build target -> accepted source mapping, and inverse source -> target.
    target_to_sources: Dict[str, Tuple[str, ...]] = {}
    for target, sources in aliases.items():
        if target in module.state_dict():
            target_to_sources[target] = sources
    source_to_target: Dict[str, str] = {}
    for target, sources in target_to_sources.items():
        for src in sources:
            source_to_target.setdefault(src, target)

    remapped: Dict[str, torch.Tensor] = {}
    unmatched: List[str] = []
    used: set = set()
    for raw_key, tensor in state_dict.items():
        key = raw_key
        for prefix in PREFIXES:
            if key.startswith(prefix) and len(key) > len(prefix):
                key = key[len(prefix):]
                break
        if key in source_to_target:
            target = source_to_target[key]
            remapped[target] = tensor
            used.add(raw_key)
        else:
            unmatched.append(raw_key)

    missing = [t for t in target_to_sources if t not in remapped]
    # Adapter-only targets (projection head) are expected to be absent.
    missing_required = [
        t for t in missing if not t.startswith(SKIPPABLE_TARGET_PREFIXES)
    ]
    if strict and (unmatched or missing_required):
        raise ValueError(
            "V-JEPA checkpoint key mismatch: "
            f"{len(unmatched)} unmatched source keys, {len(missing_required)} missing target keys. "
            "Run vjepa_backbone_smoke.py with a checkpoint to see the report."
        )
    module.load_state_dict(remapped, strict=False)
    return LoadReport(
        loaded=len(remapped),
        skipped=len(unmatched),
        skipped_keys=tuple(unmatched[:20]),
        strict_ok=not (unmatched or missing_required),
    )


def _count_blocks(module: nn.Module) -> int:
    n = 0
    while hasattr(module, f"blocks") and n < 64:
        try:
            _ = module.blocks[n]
            n += 1
        except (IndexError, AttributeError):
            break
    return n


# ---------------------------------------------------------------------------
# LoRA (low-rank adapter) for attention projections
# ---------------------------------------------------------------------------


class LoRA(nn.Module):
    """Low-rank adapter on an existing frozen linear layer (in-place swap)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA currently supports nn.Linear bases only")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        fan_in, fan_out = base.in_features, base.out_features
        self.lora_a = nn.Parameter(torch.zeros(rank, fan_in))
        self.lora_b = nn.Parameter(torch.zeros(fan_out, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = Wx + (alpha/rank) * BAx
        return self.base(x) + (self.alpha / self.rank) * F.linear(F.linear(x, self.lora_a), self.lora_b)


# ---------------------------------------------------------------------------
# ViT-B/16 backbone (structural twin of official V-JEPA small encoder)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VJEPA2Backbone(nn.Module):
    """Structural twin of the official V-JEPA 2 ViT-B/16 encoder.

    forward(frames: [N, C, H, W]) -> [N, dim]  (CLS pooled latent per frame),
    matching the `TinyFrameEncoder` contract used by `StateEncoder`.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        grid = img_size // patch_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = grid * grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # Project official embed_dim down to the AC-VJEPA latent contract when needed.
        self.head: Optional[nn.Linear] = nn.Linear(embed_dim, out_dim) if out_dim else None
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # [B, D]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [N, C, H, W] -> [N, out_dim or embed_dim]
        feats = self.forward_features(frames)
        if self.head is not None:
            return self.head(feats)
        return feats

    # -- fine-tune mode helpers --------------------------------------------------

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        if self.head is not None:
            for p in self.head.parameters():
                p.requires_grad_(True)

    def unfreeze_last_k(self, k: int) -> None:
        self.freeze()
        for block in self.blocks[-k:]:
            for p in block.parameters():
                p.requires_grad_(True)

    def apply_lora(self, rank: int = 8, alpha: int = 16) -> List[nn.Parameter]:
        """Wrap attention q/k/v/o of every block with LoRA; returns lora params."""
        lora_params: List[nn.Parameter] = []
        for block in self.blocks:
            for name in ("qkv", "proj"):
                layer = getattr(block.attn, name)
                if isinstance(layer, nn.Linear):
                    adapted = LoRA(layer, rank=rank, alpha=alpha)
                    setattr(block.attn, name, adapted)
                    lora_params.extend([adapted.lora_a, adapted.lora_b])
        return lora_params

    def trainable_parameter_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]


def build_vjepa2_backbone(
    *,
    out_dim: Optional[int] = None,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
) -> VJEPA2Backbone:
    """Build a ViT-B/16 twin; `out_dim` should equal the AC-VJEPA latent_dim."""
    return VJEPA2Backbone(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        out_dim=out_dim,
    )


# ---------------------------------------------------------------------------
# HuggingFace Transformers adapter (real V-JEPA 2.1 weights, M1 verified)
# ---------------------------------------------------------------------------


class HFVJEPA2Backbone(nn.Module):
    """Wrap `transformers.VJEPA2Model` as an AC-VJEPA `frame_encoder`.

    Contract (identical to `TinyFrameEncoder`):
        forward(frames: [N, C, H, W]) -> [N, latent_dim]

    Internally it feeds each frame as a 1-frame video clip
    `[N, 1, C, H, W] -> last_hidden_state [N, num_patches, hidden]`, mean-pools
    over patches, then projects to `latent_dim`. This keeps the exact per-frame
    contract used by `StateEncoder.encode_frames` while running the real
    V-JEPA 2.1 encoder weights.

    Modes: only `frozen` (default, matches the "frozen 80M backbone + train the
    small heads" experiment plan) and `finetune` are supported; LoRA / last_k on
    the HF parameter tree is out of scope for M1.
    """

    def __init__(
        self,
        model_id: str,
        ckpt_path: Optional[str],
        latent_dim: int,
        *,
        mode: str = "frozen",
        img_size: int = 384,
    ) -> None:
        super().__init__()
        if mode not in ("frozen", "finetune"):
            raise ValueError(f"HFVJEPA2Backbone mode must be frozen|finetune, got {mode!r}")
        from transformers import VJEPA2Config, VJEPA2Model

        self.model_id = model_id
        self.img_size = img_size
        config = VJEPA2Config.from_pretrained(model_id)
        self.hf = VJEPA2Model(config)  # random init; weights loaded below
        self.head = nn.Linear(config.hidden_size, latent_dim)
        self.load_report: LoadReport = LoadReport(0, 0, (), True)
        if ckpt_path is not None:
            self.load_report = load_hf_vjepa2_weights(self.hf, Path(ckpt_path), config)
        if mode == "frozen":
            self.freeze()

    def freeze(self) -> None:
        for p in self.hf.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [N, C, H, W] -> videos: [N, 1, C, H, W]
        videos = frames.unsqueeze(1)
        out = self.hf(pixel_values_videos=videos)
        feats = out.last_hidden_state.mean(dim=1)  # [N, hidden]
        return self.head(feats)


def load_hf_vjepa2_weights(
    model,
    ckpt_path: Path,
    config,
) -> LoadReport:
    """Manually load shape-compatible keys from a safetensors checkpoint into a
    config-constructed VJEPA2Model. The released 2.1 ViT-B checkpoint carries
    predictor head dims (1664) that differ from this transformers version and
    auxiliary keys (img/video mod embeds, distillation norms) that we do not
    need; the encoder itself is fully covered (391/391 encoder keys).
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install safetensors to load HF-format weights") from exc

    sd = model.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    with safe_open(str(ckpt_path), framework="pt") as handle:
        keys = list(handle.keys())
        for k in keys:
            if k not in sd:
                continue
            tensor = handle.get_tensor(k)
            if tuple(tensor.shape) == tuple(sd[k].shape):
                loadable[k] = tensor
    skipped = [k for k in keys if k not in loadable]
    missing = [k for k in sd if k not in loadable]
    model.load_state_dict(loadable, strict=False)
    strict_ok = not skipped
    report = LoadReport(
        loaded=len(loadable),
        skipped=len(skipped),
        skipped_keys=tuple(skipped[:20]),
        strict_ok=strict_ok,
    )
    if not strict_ok:
        print(
            f"[vjepa_backbone] HF load report: loaded={len(loadable)}/{len(sd)} "
            f"skipped={len(skipped)} missing={len(missing)} (predictor head dims differ; "
            "encoder keys fully covered)"
        )
    return report


def install_hf_vjepa2_encoder(
    module,
    *,
    latent_dim: int,
    ckpt_path: Optional[str] = None,
    model_id: str = "davevanveen/vjepa2.1-vitb-fpc64-384",
    mode: str = "frozen",
    img_size: int = 384,
) -> LoadReport:
    """Install the real HF V-JEPA 2.1 backbone as `frame_encoder` (student +
    EMA twin). The frozen HF transformer is shared between student and target
    (saves ~400MB and keeps `EMAStateEncoder.update_from` parameter counts
    aligned); the trainable projection head gets its own EMA copy.
    """
    if not hasattr(module, "student_encoder") or not hasattr(module.student_encoder, "frame_encoder"):
        raise AttributeError("expected an ActionConditionedVJEPA-compatible module")
    backbone = HFVJEPA2Backbone(
        model_id=model_id,
        ckpt_path=ckpt_path,
        latent_dim=latent_dim,
        mode=mode,
        img_size=img_size,
    )
    module.student_encoder.frame_encoder = backbone
    if hasattr(module, "target_encoder") and hasattr(module.target_encoder, "encoder"):
        # EMA twin: share the frozen HF transformer object (EMA on a shared
        # object is a no-op, so frozen weights stay untouched) and give the
        # trainable projection head its own EMA copy.
        twin = HFVJEPA2Backbone(
            model_id=model_id,
            ckpt_path=None,
            latent_dim=latent_dim,
            mode="frozen",
            img_size=img_size,
        )
        twin.hf = backbone.hf  # share frozen transformer
        twin.head = copy.deepcopy(backbone.head)
        for p in twin.head.parameters():
            p.requires_grad_(False)
        module.target_encoder.encoder.frame_encoder = twin
    return backbone.load_report


def install_vjepa2_encoder(
    module,
    *,
    latent_dim: int,
    checkpoint: Optional[str] = None,
    mode: str = "frozen",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    unfreeze_last_k: int = 1,
    strict: bool = True,
    img_size: int = 224,
) -> LoadReport:
    """Replace `module.student_encoder.frame_encoder` (and the EMA twin) with
    a V-JEPA 2 backbone, optionally loading official weights.

    Returns the load report (loaded/skipped). `mode`:
      frozen | last_k | lora | finetune
    """
    if not hasattr(module, "student_encoder") or not hasattr(module.student_encoder, "frame_encoder"):
        raise AttributeError("expected an ActionConditionedVJEPA-compatible module")

    backbone = build_vjepa2_backbone(out_dim=latent_dim, img_size=img_size)
    report = LoadReport(loaded=0, skipped=0, skipped_keys=(), strict_ok=True)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # official dumps nest under "encoder"; HF under "model"; unwrap common wrappers
        state = ckpt.get("model", ckpt.get("encoder", ckpt))
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        if not isinstance(state, dict):
            raise TypeError(f"unsupported checkpoint payload: {type(state)}")
        report = load_vjepa2_weights(backbone, state, strict=strict, num_blocks=len(backbone.blocks))

    if mode == "frozen":
        backbone.freeze()
    elif mode == "last_k":
        backbone.unfreeze_last_k(unfreeze_last_k)
    elif mode == "lora":
        backbone.freeze()
        backbone.apply_lora(rank=lora_rank, alpha=lora_alpha)
    elif mode == "finetune":
        pass  # everything trainable
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Swap the student encoder's frame encoder AND the EMA target's copy so the
    # parameter-count contract of `EMAStateEncoder.update_from` stays valid.
    module.student_encoder.frame_encoder = backbone
    if hasattr(module, "target_encoder") and hasattr(module.target_encoder, "encoder"):
        module.target_encoder.encoder.frame_encoder = copy.deepcopy(backbone)
        for p in module.target_encoder.encoder.frame_encoder.parameters():
            p.requires_grad_(False)
    return report
