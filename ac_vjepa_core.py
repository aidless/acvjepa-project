"""Action-conditioned V-JEPA core module for research prototypes.

This file is intentionally NOT a robot driver. It predicts short-horizon latent
state transitions from video/proprioception/action blocks, monitors bounded
inference latency, and emits safe fallback events. A vendor-certified safety
controller must remain independent of this process.

Run a CPU smoke test:
    python3 ac_vjepa_core.py
"""
from __future__ import annotations

import copy
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field, replace
from enum import Enum
from threading import Lock
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Contracts: only typed, versioned messages cross the planning / control boundary
# -----------------------------------------------------------------------------


class RuntimeMode(str, Enum):
    NORMAL = "normal"
    CAUTIOUS_OBSERVE = "cautious_observe"
    LOCAL_HOLD = "local_hold"
    LLM_SUPERVISION = "llm_supervision"
    REPLAN_PENDING = "replan_pending"
    HARD_STOP = "hard_stop"


class FallbackReason(str, Enum):
    GPU_BUSY = "gpu_busy"
    INFERENCE_TIMEOUT = "inference_timeout"
    HIGH_UNCERTAINTY = "high_uncertainty"
    STALE_STATE = "stale_state"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    HARDWARE_HEALTH_FAULT = "hardware_health_fault"


class RecoveryChoice(str, Enum):
    """The only semantic choices a future LLM recovery layer may return."""

    OBSERVE = "observe"
    ASK_USER = "ask_user"
    SELECT_PREAPPROVED_SKILL = "select_preapproved_skill"
    RETRY_AFTER_HEALTH = "retry_after_health"
    END_TASK = "end_task"


@dataclass(frozen=True)
class RuntimeLimits:
    plan_deadline_ms: float = 150.0
    max_state_age_ms: float = 120.0
    max_mean_uncertainty: float = 1.50
    max_nan_fraction: float = 0.0
    model_version: str = "ac-vjepa-research-0.1"


@dataclass(frozen=True)
class StateEnvelope:
    state_id: str
    state_age_ms: float
    sensor_timestamp_ns: int
    robot_calibration_id: str
    facts: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FallbackEvent:
    event_id: str
    reason: FallbackReason
    state_id: str
    state_age_ms: float
    request_id: str
    model_version: str
    inference_latency_ms: Optional[float]
    mean_uncertainty: Optional[float]
    facts: Dict[str, Any]
    message: str


@dataclass(frozen=True)
class LLMSupervisionPacket:
    """Immutable facts sent to the LLM after LOCAL_HOLD is already confirmed."""

    event: FallbackEvent
    allowed_choices: Tuple[RecoveryChoice, ...] = (
        RecoveryChoice.OBSERVE,
        RecoveryChoice.ASK_USER,
        RecoveryChoice.SELECT_PREAPPROVED_SKILL,
        RecoveryChoice.RETRY_AFTER_HEALTH,
        RecoveryChoice.END_TASK,
    )


@dataclass(frozen=True)
class SafeRuntimeDecision:
    mode: RuntimeMode
    request_id: str
    prediction: Optional["Prediction"] = None
    fallback: Optional[FallbackEvent] = None
    supervision_packet: Optional[LLMSupervisionPacket] = None


# -----------------------------------------------------------------------------
# Lightweight action-conditioned JEPA model
# -----------------------------------------------------------------------------


class TinyFrameEncoder(nn.Module):
    """Self-contained stand-in for a frozen V-JEPA frame/video encoder.

    In a real experiment, replace this module with a V-JEPA checkpoint adapter
    that emits dense or pooled latent visual features. Its input/output contract
    must remain stable across training and deployment.
    """

    def __init__(self, image_channels: int, visual_dim: int):
        super().__init__()
        hidden = max(visual_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, visual_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, frames: Tensor) -> Tensor:
        # frames: [N, C, H, W] -> [N, visual_dim]
        return self.net(frames).flatten(1)


class StateEncoder(nn.Module):
    """Fuses visual frames with synchronized proprioception.

    `encode_context` consumes a short history. `encode_future` produces one
    latent target per future step. The same preprocessing/time conventions must
    be used in offline episodes and live robot streams.
    """

    def __init__(self, image_channels: int, proprio_dim: int, latent_dim: int):
        super().__init__()
        self.frame_encoder = TinyFrameEncoder(image_channels, latent_dim)
        self.proprio_projector = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, latent_dim),
        )
        self.out_norm = nn.LayerNorm(latent_dim)

    def encode_frames(self, video: Tensor, proprio: Tensor) -> Tensor:
        # video: [B, T, C, H, W], proprio: [B, T, P]
        batch, steps = video.shape[:2]
        visual = self.frame_encoder(video.flatten(0, 1)).view(batch, steps, -1)
        state = visual + self.proprio_projector(proprio)
        return self.out_norm(state)

    def encode_context(self, video: Tensor, proprio: Tensor) -> Tensor:
        # Mean pooling is intentionally simple. A production V-JEPA adapter may
        # use a temporal encoder/cache but must keep the same [B, D] contract.
        return self.encode_frames(video, proprio).mean(dim=1)

    def encode_future(self, video: Tensor, proprio: Tensor) -> Tensor:
        return self.encode_frames(video, proprio)


class EMAStateEncoder(nn.Module):
    """Non-gradient target encoder used to build JEPA future latent targets."""

    def __init__(self, source: StateEncoder, momentum: float = 0.996):
        super().__init__()
        self.encoder = copy.deepcopy(source)
        self.momentum = momentum
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_from(self, source: StateEncoder) -> None:
        for target_param, source_param in zip(
            self.encoder.parameters(), source.parameters()
        ):
            target_param.mul_(self.momentum).add_(source_param, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def encode_future(self, video: Tensor, proprio: Tensor) -> Tensor:
        return self.encoder.encode_future(video, proprio)


class ActionTokenizer(nn.Module):
    """Maps normalized, actually executed or candidate ActionBlocks to tokens.

    Action values should represent task/base-frame end-effector deltas, gripper
    mode and approved speed/contact profiles, NOT raw motor currents or unbounded
    LLM output.
    """

    def __init__(self, action_dim: int, latent_dim: int, max_horizon: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.time_embedding = nn.Embedding(max_horizon, latent_dim)

    def forward(self, actions: Tensor) -> Tensor:
        # actions: [B, H, A]
        horizon = actions.shape[1]
        positions = torch.arange(horizon, device=actions.device)
        return self.project(actions) + self.time_embedding(positions)[None, :, :]


@dataclass
class Prediction:
    future_latents: Tensor       # [B, H, D]
    log_variance: Tensor         # [B, H, D], heteroscedastic uncertainty
    event_logits: Tensor         # [B, H, E]

    def mean_uncertainty(self) -> Tensor:
        # Kept numerically bounded for a stable deployment threshold.
        return torch.exp(self.log_variance.clamp(min=-8.0, max=6.0)).mean(dim=(1, 2))

    def has_invalid_values(self) -> bool:
        tensors = (self.future_latents, self.log_variance, self.event_logits)
        return any(not torch.isfinite(t).all().item() for t in tensors)


class ActionConditionedVJEPA(nn.Module):
    """A compact latent world model for research and integration tests.

    The GRU rolls latent state forward under a sequence of ActionBlock tokens.
    Replacing it with a causal Transformer is possible without changing the
    contracts. The target encoder is only used during training.
    """

    def __init__(
        self,
        image_channels: int,
        proprio_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        event_dim: int = 4,
        max_horizon: int = 8,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.student_encoder = StateEncoder(image_channels, proprio_dim, latent_dim)
        self.target_encoder = EMAStateEncoder(self.student_encoder, ema_momentum)
        self.action_tokenizer = ActionTokenizer(action_dim, latent_dim, max_horizon)
        self.rollout = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.latent_delta = nn.Linear(latent_dim, latent_dim)
        self.log_variance_head = nn.Linear(latent_dim, latent_dim)
        self.event_head = nn.Linear(latent_dim, event_dim)

    def forward(
        self,
        context_video: Tensor,
        context_proprio: Tensor,
        action_blocks: Tensor,
    ) -> Prediction:
        """Standard forward path so DistributedDataParallel can install gradient hooks."""
        return self.predict(context_video, context_proprio, action_blocks)

    def predict(
        self,
        context_video: Tensor,
        context_proprio: Tensor,
        action_blocks: Tensor,
    ) -> Prediction:
        state0 = self.student_encoder.encode_context(context_video, context_proprio)
        action_tokens = self.action_tokenizer(action_blocks)
        hidden0 = state0.unsqueeze(0)  # GRU uses [layers, B, D]
        rollout_hidden, _ = self.rollout(action_tokens, hidden0)
        future_latents = state0[:, None, :] + self.latent_delta(rollout_hidden)
        return Prediction(
            future_latents=future_latents,
            log_variance=self.log_variance_head(rollout_hidden).clamp(-8.0, 6.0),
            event_logits=self.event_head(rollout_hidden),
        )

    def target_latents(self, future_video: Tensor, future_proprio: Tensor) -> Tensor:
        return self.target_encoder.encode_future(future_video, future_proprio)

    @torch.no_grad()
    def update_ema_target(self) -> None:
        self.target_encoder.update_from(self.student_encoder)


@dataclass
class LossBreakdown:
    total: Tensor
    latent_nll: Tensor
    latent_cosine: Tensor
    event_bce: Tensor
    calibration: Tensor


def action_conditioned_jepa_loss(
    prediction: Prediction,
    target_latents: Tensor,
    event_targets: Optional[Tensor] = None,
    event_weight: float = 0.20,
    calibration_weight: float = 0.05,
) -> LossBreakdown:
    """Latent JEPA alignment plus event and uncertainty calibration losses.

    The variance-aware term discourages a model from claiming low uncertainty on
    high-error states. It is not a full safety guarantee; deployment still needs
    OOD checks, geometric rules and measured thresholds.
    """

    squared_error = (prediction.future_latents - target_latents).square()
    inverse_variance = torch.exp(-prediction.log_variance)
    latent_nll = 0.5 * (inverse_variance * squared_error + prediction.log_variance).mean()

    pred_unit = F.normalize(prediction.future_latents, dim=-1)
    target_unit = F.normalize(target_latents, dim=-1)
    latent_cosine = (1.0 - (pred_unit * target_unit).sum(dim=-1)).mean()

    per_step_error = squared_error.detach().mean(dim=-1)
    predicted_variance = prediction.mean_uncertainty()[:, None]
    calibration = F.smooth_l1_loss(predicted_variance.expand_as(per_step_error), per_step_error)

    if event_targets is None:
        event_bce = torch.zeros((), device=target_latents.device)
    else:
        event_bce = F.binary_cross_entropy_with_logits(
            prediction.event_logits, event_targets.float()
        )

    total = latent_nll + latent_cosine + event_weight * event_bce + calibration_weight * calibration
    return LossBreakdown(total, latent_nll, latent_cosine, event_bce, calibration)


# -----------------------------------------------------------------------------
# Latency observability and single-flight inference
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencySample:
    request_id: str
    latency_ms: float
    outcome: str


class LatencyMonitor:
    def __init__(self, max_samples: int = 2048):
        self._samples: Deque[LatencySample] = deque(maxlen=max_samples)
        self._lock = Lock()

    def record(self, request_id: str, latency_ms: float, outcome: str) -> None:
        with self._lock:
            self._samples.append(LatencySample(request_id, latency_ms, outcome))

    def summary(self) -> Dict[str, float]:
        with self._lock:
            values = torch.tensor([s.latency_ms for s in self._samples], dtype=torch.float32)
        if values.numel() == 0:
            return {"count": 0.0, "p50_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan")}
        return {
            "count": float(values.numel()),
            "p50_ms": float(torch.quantile(values, 0.50)),
            "p95_ms": float(torch.quantile(values, 0.95)),
            "p99_ms": float(torch.quantile(values, 0.99)),
        }


class SingleFlightInferenceWorker:
    """Bounded, one-request worker for deadline-aware model inference.

    Important: Python cannot reliably cancel an already executing CUDA kernel.
    When a deadline is missed, the result is marked stale and never sent to the
    trajectory queue. New requests are rejected as GPU_BUSY until the old work
    completes and is discarded. Production systems should isolate this worker in
    a separate process with an OS/process watchdog.
    """

    def __init__(self, model: ActionConditionedVJEPA, device: torch.device, monitor: LatencyMonitor):
        self.model = model.eval().to(device)
        self.device = device
        self.monitor = monitor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ac-vjepa")
        self._inflight: Optional[Tuple[str, Future[Prediction]]] = None
        self._lock = Lock()

    def _run_model(self, inputs: Dict[str, Tensor]) -> Prediction:
        with torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            prediction = self.model.predict(**inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return prediction

    def infer_before_deadline(
        self, inputs: Dict[str, Tensor], deadline_ms: float
    ) -> Tuple[str, Optional[Prediction], Optional[FallbackReason], float]:
        request_id = str(uuid4())
        start = time.perf_counter()
        device_inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}

        with self._lock:
            # Finished late work is intentionally dropped. It must never become a
            # valid trajectory simply because it completed after a later state.
            if self._inflight is not None and self._inflight[1].done():
                self._inflight = None
            if self._inflight is not None:
                elapsed = (time.perf_counter() - start) * 1000.0
                self.monitor.record(request_id, elapsed, FallbackReason.GPU_BUSY.value)
                return request_id, None, FallbackReason.GPU_BUSY, elapsed
            future = self._executor.submit(self._run_model, device_inputs)
            self._inflight = (request_id, future)

        try:
            prediction = future.result(timeout=deadline_ms / 1000.0)
            elapsed = (time.perf_counter() - start) * 1000.0
            with self._lock:
                if self._inflight is not None and self._inflight[0] == request_id:
                    self._inflight = None
            if prediction.has_invalid_values():
                self.monitor.record(request_id, elapsed, FallbackReason.INVALID_MODEL_OUTPUT.value)
                return request_id, None, FallbackReason.INVALID_MODEL_OUTPUT, elapsed
            self.monitor.record(request_id, elapsed, "ok")
            return request_id, prediction, None, elapsed
        except TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000.0
            # Do NOT call Future.cancel() and assume CUDA stopped. The worker remains
            # single-flight/busy until the late future completes and is discarded.
            self.monitor.record(request_id, elapsed, FallbackReason.INFERENCE_TIMEOUT.value)
            return request_id, None, FallbackReason.INFERENCE_TIMEOUT, elapsed

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


# -----------------------------------------------------------------------------
# Safe handover coordinator: local hold first, LLM supervision second
# -----------------------------------------------------------------------------


class RecoveryGate:
    """Deterministically validates a restricted LLM recovery decision.

    The LLM never supplies actuator commands. It can only select an allowed
    semantic recovery action. Any motion still requires a fresh state, hardware
    health, a registered skill, new world-model planning and SafetyKernel approval.
    """

    def validate_choice(
        self,
        choice: RecoveryChoice,
        *,
        state_is_fresh: bool,
        hardware_healthy: bool,
        selected_skill_registered: bool = False,
    ) -> RuntimeMode:
        if choice == RecoveryChoice.END_TASK:
            return RuntimeMode.LOCAL_HOLD
        if choice in (RecoveryChoice.OBSERVE, RecoveryChoice.ASK_USER):
            return RuntimeMode.LLM_SUPERVISION
        if choice == RecoveryChoice.SELECT_PREAPPROVED_SKILL:
            if state_is_fresh and hardware_healthy and selected_skill_registered:
                return RuntimeMode.REPLAN_PENDING
            return RuntimeMode.LOCAL_HOLD
        if choice == RecoveryChoice.RETRY_AFTER_HEALTH:
            if state_is_fresh and hardware_healthy:
                return RuntimeMode.REPLAN_PENDING
            return RuntimeMode.LOCAL_HOLD
        return RuntimeMode.LOCAL_HOLD


class SafeHandoverCoordinator:
    def __init__(
        self,
        worker: SingleFlightInferenceWorker,
        limits: RuntimeLimits,
        recovery_gate: Optional[RecoveryGate] = None,
    ):
        self.worker = worker
        self.limits = limits
        self.recovery_gate = recovery_gate or RecoveryGate()

    def _fallback(
        self,
        reason: FallbackReason,
        state: StateEnvelope,
        request_id: str,
        latency_ms: Optional[float],
        uncertainty: Optional[float],
    ) -> SafeRuntimeDecision:
        # In production this must be sent atomically to the local safety process:
        #   1) block new trajectory admission,
        #   2) consume only an unexpired safe window,
        #   3) otherwise ramp to HOLD/RETREAT.
        # The LLM packet is prepared now but is NOT activated until the local
        # safety process confirms that the robot is already in LOCAL_HOLD.
        event = FallbackEvent(
            event_id=str(uuid4()),
            reason=reason,
            state_id=state.state_id,
            state_age_ms=state.state_age_ms,
            request_id=request_id,
            model_version=self.limits.model_version,
            inference_latency_ms=latency_ms,
            mean_uncertainty=uncertainty,
            facts=state.facts,
            message=(
                "Local hold requested. LLM may only choose a whitelisted recovery action; "
                "it cannot emit actuator commands or override safety gates."
            ),
        )
        return SafeRuntimeDecision(
            mode=RuntimeMode.LOCAL_HOLD,
            request_id=request_id,
            fallback=event,
            supervision_packet=LLMSupervisionPacket(event),
        )

    def confirm_local_hold(self, decision: SafeRuntimeDecision, hold_confirmed: bool) -> SafeRuntimeDecision:
        """Enter LLM supervision only after an independent local hold acknowledgement."""
        if not hold_confirmed or decision.mode != RuntimeMode.LOCAL_HOLD:
            return decision
        return replace(decision, mode=RuntimeMode.LLM_SUPERVISION)

    def _hard_stop(self, state: StateEnvelope, request_id: str = "no_request") -> SafeRuntimeDecision:
        event = FallbackEvent(
            event_id=str(uuid4()),
            reason=FallbackReason.HARDWARE_HEALTH_FAULT,
            state_id=state.state_id,
            state_age_ms=state.state_age_ms,
            request_id=request_id,
            model_version=self.limits.model_version,
            inference_latency_ms=None,
            mean_uncertainty=None,
            facts=state.facts,
            message="Hardware safety health is not confirmed. Require independent hard stop and human reset.",
        )
        return SafeRuntimeDecision(mode=RuntimeMode.HARD_STOP, request_id=request_id, fallback=event)

    def plan_or_handover(
        self,
        state: StateEnvelope,
        model_inputs: Dict[str, Tensor],
        hardware_healthy: bool,
    ) -> SafeRuntimeDecision:
        if not hardware_healthy:
            return self._hard_stop(state)
        if state.state_age_ms > self.limits.max_state_age_ms:
            return self._fallback(FallbackReason.STALE_STATE, state, "no_request", None, None)

        request_id, prediction, error, latency_ms = self.worker.infer_before_deadline(
            model_inputs, self.limits.plan_deadline_ms
        )
        if error is not None:
            return self._fallback(error, state, request_id, latency_ms, None)
        assert prediction is not None

        mean_uncertainty = float(prediction.mean_uncertainty().mean().detach().cpu())
        if mean_uncertainty > self.limits.max_mean_uncertainty:
            return self._fallback(
                FallbackReason.HIGH_UNCERTAINTY,
                state,
                request_id,
                latency_ms,
                mean_uncertainty,
            )

        return SafeRuntimeDecision(
            mode=RuntimeMode.NORMAL,
            request_id=request_id,
            prediction=prediction,
        )

    def apply_llm_recovery_choice(
        self,
        packet: LLMSupervisionPacket,
        choice: RecoveryChoice,
        *,
        state_is_fresh: bool,
        hardware_healthy: bool,
        selected_skill_registered: bool = False,
    ) -> RuntimeMode:
        if choice not in packet.allowed_choices:
            return RuntimeMode.LOCAL_HOLD
        return self.recovery_gate.validate_choice(
            choice,
            state_is_fresh=state_is_fresh,
            hardware_healthy=hardware_healthy,
            selected_skill_registered=selected_skill_registered,
        )


# -----------------------------------------------------------------------------
# Minimal smoke test: not a hardware integration test
# -----------------------------------------------------------------------------


def _smoke_test() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch, context_steps, horizon = 2, 4, 3
    channels, height, width = 3, 32, 32
    proprio_dim, action_dim, event_dim = 8, 20, 4

    model = ActionConditionedVJEPA(
        image_channels=channels,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        latent_dim=64,
        event_dim=event_dim,
        max_horizon=8,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    context_video = torch.randn(batch, context_steps, channels, height, width, device=device)
    context_proprio = torch.randn(batch, context_steps, proprio_dim, device=device)
    future_video = torch.randn(batch, horizon, channels, height, width, device=device)
    future_proprio = torch.randn(batch, horizon, proprio_dim, device=device)
    action_blocks = torch.randn(batch, horizon, action_dim, device=device)
    event_targets = torch.randint(0, 2, (batch, horizon, event_dim), device=device).float()

    model.train()
    prediction = model.predict(context_video, context_proprio, action_blocks)
    targets = model.target_latents(future_video, future_proprio)
    losses = action_conditioned_jepa_loss(prediction, targets, event_targets)
    optimizer.zero_grad(set_to_none=True)
    losses.total.backward()
    optimizer.step()
    model.update_ema_target()

    monitor = LatencyMonitor()
    worker = SingleFlightInferenceWorker(model, device, monitor)
    coordinator = SafeHandoverCoordinator(
        worker,
        RuntimeLimits(plan_deadline_ms=5_000.0, max_mean_uncertainty=10.0),
    )
    state = StateEnvelope(
        state_id="demo-state-001",
        state_age_ms=10.0,
        sensor_timestamp_ns=time.time_ns(),
        robot_calibration_id="demo-calibration",
        facts={"human_distance_m": 2.0, "protected_zone_clear": True},
    )
    inputs = {
        "context_video": context_video.detach().cpu(),
        "context_proprio": context_proprio.detach().cpu(),
        "action_blocks": action_blocks.detach().cpu(),
    }
    decision = coordinator.plan_or_handover(state, inputs, hardware_healthy=True)
    assert decision.mode == RuntimeMode.NORMAL, decision

    # Force a safe fallback to demonstrate the LLM supervision path.
    coordinator.limits = RuntimeLimits(plan_deadline_ms=5_000.0, max_mean_uncertainty=0.0)
    fallback = coordinator.plan_or_handover(state, inputs, hardware_healthy=True)
    assert fallback.mode == RuntimeMode.LOCAL_HOLD, fallback
    supervised = coordinator.confirm_local_hold(fallback, hold_confirmed=True)
    assert supervised.mode == RuntimeMode.LLM_SUPERVISION, supervised
    assert supervised.supervision_packet is not None
    next_mode = coordinator.apply_llm_recovery_choice(
        supervised.supervision_packet,
        RecoveryChoice.OBSERVE,
        state_is_fresh=True,
        hardware_healthy=True,
    )
    assert next_mode == RuntimeMode.LLM_SUPERVISION

    worker.close()
    print(
        {
            "device": str(device),
            "training_loss": round(float(losses.total.detach().cpu()), 5),
            "normal_mode": decision.mode.value,
            "fallback_mode": supervised.mode.value,
            "latency": monitor.summary(),
        }
    )


if __name__ == "__main__":
    _smoke_test()
