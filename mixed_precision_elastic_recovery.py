"""Exact committed-checkpoint restore contract for BF16/FP16/FP8 training.

"Exact" in this module means: immediately after loading a COMMITTED checkpoint
into the same software/precision backend and before the next forward pass, the
serialized model, EMA, AdamW state, scaler/FP8 metadata and CPU RNG bytes equal
the committed payload. It does NOT promise that an interrupted in-flight update
can be reconstructed, nor that subsequent floating-point trajectories remain
bitwise identical after world-size, kernel, hardware, precision-backend or
collective-tree changes.

This module is a CPU-testable reference. FP8 engine-specific state is passed as
an explicit mapping because different FP8 libraries/version may place metadata
in different state-dict keys (for example Transformer Engine `._extra_state`).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

import torch


class PrecisionRestoreError(RuntimeError):
    pass


class PrecisionMode(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"


@dataclass(frozen=True)
class PrecisionContract:
    mode: PrecisionMode
    torch_version: str
    precision_backend: str
    optimizer_type: str
    fp8_engine_version: str | None = None

    def validate(self) -> None:
        if not self.torch_version or not self.precision_backend or not self.optimizer_type:
            raise PrecisionRestoreError("precision contract is incomplete")
        if self.mode is PrecisionMode.FP8 and not self.fp8_engine_version:
            raise PrecisionRestoreError("FP8 recovery requires an explicit engine/version identity")
        if self.mode is not PrecisionMode.FP8 and self.fp8_engine_version is not None:
            raise PrecisionRestoreError("non-FP8 contract must not declare FP8 engine metadata")


@dataclass(frozen=True)
class ElasticCheckpointIdentity:
    committed_step: int
    checkpoint_hash: str
    dataset_commit: str
    manifest_digest: str
    topology_epoch: str
    world_size: int

    def validate(self) -> None:
        if self.committed_step < 0 or self.world_size <= 0:
            raise PrecisionRestoreError("invalid committed step/world size")
        if not self.dataset_commit or len(self.checkpoint_hash) != 64 or len(self.manifest_digest) != 64:
            raise PrecisionRestoreError("checkpoint identity is incomplete")


@dataclass(frozen=True)
class StateFingerprints:
    model: str
    ema: str
    optimizer: str
    scaler: str | None
    fp8_metadata: str | None
    rng_cpu: str


@dataclass
class PrecisionCheckpoint:
    identity: ElasticCheckpointIdentity
    contract: PrecisionContract
    model_state: Mapping[str, Any]
    ema_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    scaler_state: Mapping[str, Any] | None
    fp8_metadata_state: Mapping[str, Any] | None
    rng_cpu_state: torch.Tensor
    fingerprints: StateFingerprints


def _tree_hash(value: Any) -> str:
    """Stable type-aware SHA-256 over arbitrary nested state_dict-like objects."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
            return
        if item is None:
            digest.update(b"none\0")
            return
        digest.update(type(item).__qualname__.encode())
        digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _require_scaler_mode(contract: PrecisionContract, scaler: torch.amp.GradScaler | None) -> Mapping[str, Any] | None:
    if contract.mode is PrecisionMode.FP16:
        if scaler is None or not scaler.is_enabled():
            raise PrecisionRestoreError("FP16 contract requires enabled GradScaler state")
        return _clone_tree(scaler.state_dict())
    if scaler is not None and scaler.is_enabled():
        raise PrecisionRestoreError(f"{contract.mode.value} contract requires no enabled GradScaler")
    return None


def _require_fp8_metadata(contract: PrecisionContract, fp8_metadata_state: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if contract.mode is PrecisionMode.FP8:
        if not fp8_metadata_state:
            raise PrecisionRestoreError("FP8 contract requires committed scale/AMAX metadata state")
        return _clone_tree(fp8_metadata_state)
    if fp8_metadata_state is not None:
        raise PrecisionRestoreError("non-FP8 contract must not carry FP8 metadata state")
    return None


def snapshot_committed_precision_state(
    *,
    identity: ElasticCheckpointIdentity,
    contract: PrecisionContract,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    fp8_metadata_state: Mapping[str, Any] | None,
) -> PrecisionCheckpoint:
    """Capture all numerical state only at a verified COMMITTED boundary."""

    identity.validate()
    contract.validate()
    scaler_state = _require_scaler_mode(contract, scaler)
    fp8_state = _require_fp8_metadata(contract, fp8_metadata_state)
    model_state = _clone_tree(model.state_dict())
    ema_state = _clone_tree(ema_model.state_dict())
    optimizer_state = _clone_tree(optimizer.state_dict())
    rng_state = torch.get_rng_state().clone()
    fingerprints = StateFingerprints(
        model=_tree_hash(model_state),
        ema=_tree_hash(ema_state),
        optimizer=_tree_hash(optimizer_state),
        scaler=_tree_hash(scaler_state) if scaler_state is not None else None,
        fp8_metadata=_tree_hash(fp8_state) if fp8_state is not None else None,
        rng_cpu=_tree_hash(rng_state),
    )
    return PrecisionCheckpoint(
        identity=identity,
        contract=contract,
        model_state=model_state,
        ema_state=ema_state,
        optimizer_state=optimizer_state,
        scaler_state=scaler_state,
        fp8_metadata_state=fp8_state,
        rng_cpu_state=rng_state,
        fingerprints=fingerprints,
    )


def restore_and_verify_committed_precision_state(
    *,
    checkpoint: PrecisionCheckpoint,
    expected_contract: PrecisionContract,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    current_world_size: int,
    fp8_metadata_target: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """Load a committed snapshot and prove every required state fingerprint matches.

    A different `current_world_size` is permitted only as a new elastic group
    restarting from this checkpoint. It is explicitly reported, because the next
    update need not match an interrupted old-world update bit-for-bit.
    """

    checkpoint.identity.validate()
    checkpoint.contract.validate()
    expected_contract.validate()
    if checkpoint.contract != expected_contract:
        raise PrecisionRestoreError("precision/backend/optimizer version contract differs from checkpoint")
    if current_world_size <= 0:
        raise PrecisionRestoreError("current world size must be positive")
    model.load_state_dict(checkpoint.model_state, strict=True)
    ema_model.load_state_dict(checkpoint.ema_state, strict=True)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    if checkpoint.contract.mode is PrecisionMode.FP16:
        if scaler is None or not scaler.is_enabled() or checkpoint.scaler_state is None:
            raise PrecisionRestoreError("cannot restore FP16 checkpoint without enabled matching GradScaler")
        scaler.load_state_dict(dict(checkpoint.scaler_state))
    elif scaler is not None and scaler.is_enabled():
        raise PrecisionRestoreError("BF16/FP8 resume must not enable a GradScaler not present in checkpoint")
    if checkpoint.contract.mode is PrecisionMode.FP8:
        if fp8_metadata_target is None or checkpoint.fp8_metadata_state is None:
            raise PrecisionRestoreError("cannot restore FP8 checkpoint without explicit metadata target")
        # Caller owns engine-specific load. A mutable mapping is required so the
        # engine wrapper can receive every saved scale/AMAX entry exactly.
        if not hasattr(fp8_metadata_target, "clear") or not hasattr(fp8_metadata_target, "update"):
            raise PrecisionRestoreError("FP8 metadata target must be a mutable mapping")
        fp8_metadata_target.clear()  # type: ignore[attr-defined]
        fp8_metadata_target.update(_clone_tree(checkpoint.fp8_metadata_state))  # type: ignore[attr-defined]
    torch.set_rng_state(checkpoint.rng_cpu_state.clone())
    restored_scaler = _require_scaler_mode(checkpoint.contract, scaler)
    restored_fp8 = _require_fp8_metadata(checkpoint.contract, fp8_metadata_target)
    restored = StateFingerprints(
        model=_tree_hash(model.state_dict()),
        ema=_tree_hash(ema_model.state_dict()),
        optimizer=_tree_hash(optimizer.state_dict()),
        scaler=_tree_hash(restored_scaler) if restored_scaler is not None else None,
        fp8_metadata=_tree_hash(restored_fp8) if restored_fp8 is not None else None,
        rng_cpu=_tree_hash(torch.get_rng_state()),
    )
    result = {
        "model": restored.model == checkpoint.fingerprints.model,
        "ema": restored.ema == checkpoint.fingerprints.ema,
        "optimizer": restored.optimizer == checkpoint.fingerprints.optimizer,
        "scaler": restored.scaler == checkpoint.fingerprints.scaler,
        "fp8_metadata": restored.fp8_metadata == checkpoint.fingerprints.fp8_metadata,
        "rng_cpu": restored.rng_cpu == checkpoint.fingerprints.rng_cpu,
        "world_size_changed": current_world_size != checkpoint.identity.world_size,
    }
    if not all(value for key, value in result.items() if key != "world_size_changed"):
        failed = [key for key, value in result.items() if not value and key != "world_size_changed"]
        raise PrecisionRestoreError(f"committed checkpoint did not restore exactly: {failed}")
    return result


def _make_pair() -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Materialize AdamW exp_avg/exp_avg_sq; optimizer state is otherwise empty.
    loss = model(torch.randn(3, 4)).pow(2).mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.copy_(source)
    return model, ema, optimizer


def _identity(world_size: int) -> ElasticCheckpointIdentity:
    return ElasticCheckpointIdentity(
        committed_step=7,
        checkpoint_hash="a" * 64,
        dataset_commit="dataset-demo",
        manifest_digest="b" * 64,
        topology_epoch="epoch-demo",
        world_size=world_size,
    )


def run_smoke_test() -> None:
    torch.manual_seed(2026)
    # BF16: no enabled GradScaler. Exact contract applies to FP32 AdamW state,
    # model/EMA/RNG, regardless of autocast choice used by the outer trainer.
    model, ema, optimizer = _make_pair()
    bf16 = PrecisionContract(PrecisionMode.BF16, torch.__version__, "torch_amp", "AdamW")
    checkpoint = snapshot_committed_precision_state(
        identity=_identity(2), contract=bf16, model=model, ema_model=ema, optimizer=optimizer, scaler=None, fp8_metadata_state=None
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    result_bf16 = restore_and_verify_committed_precision_state(
        checkpoint=checkpoint,
        expected_contract=bf16,
        model=model,
        ema_model=ema,
        optimizer=optimizer,
        scaler=None,
        current_world_size=3,
        fp8_metadata_target=None,
    )
    assert result_bf16["optimizer"] and result_bf16["world_size_changed"]

    # FP16: GradScaler state is captured/restored alongside AdamW.
    model16, ema16, optimizer16 = _make_pair()
    scaler16 = torch.amp.GradScaler("cpu", enabled=True)
    fp16 = PrecisionContract(PrecisionMode.FP16, torch.__version__, "torch_amp", "AdamW")
    checkpoint16 = snapshot_committed_precision_state(
        identity=_identity(2), contract=fp16, model=model16, ema_model=ema16, optimizer=optimizer16, scaler=scaler16, fp8_metadata_state=None
    )
    result_fp16 = restore_and_verify_committed_precision_state(
        checkpoint=checkpoint16,
        expected_contract=fp16,
        model=model16,
        ema_model=ema16,
        optimizer=optimizer16,
        scaler=scaler16,
        current_world_size=2,
        fp8_metadata_target=None,
    )
    assert result_fp16["scaler"] and not result_fp16["world_size_changed"]

    # FP8: a library-owned scale/AMAX mapping is an explicit checkpoint field.
    model8, ema8, optimizer8 = _make_pair()
    fp8_meta = {"layer0._extra_state": {"scale": torch.tensor([1.0]), "amax_history": torch.tensor([[0.5, 0.25]])}}
    fp8 = PrecisionContract(PrecisionMode.FP8, torch.__version__, "transformer_engine", "AdamW", fp8_engine_version="test-2.x")
    checkpoint8 = snapshot_committed_precision_state(
        identity=_identity(2), contract=fp8, model=model8, ema_model=ema8, optimizer=optimizer8, scaler=None, fp8_metadata_state=fp8_meta
    )
    target_meta: dict[str, Any] = {}
    result_fp8 = restore_and_verify_committed_precision_state(
        checkpoint=checkpoint8,
        expected_contract=fp8,
        model=model8,
        ema_model=ema8,
        optimizer=optimizer8,
        scaler=None,
        current_world_size=2,
        fp8_metadata_target=target_meta,
    )
    assert result_fp8["fp8_metadata"] and "layer0._extra_state" in target_meta
    print('{"smoke_test":"passed","bf16":true,"fp16_scaler":true,"fp8_metadata":true}', flush=True)


if __name__ == "__main__":
    run_smoke_test()
