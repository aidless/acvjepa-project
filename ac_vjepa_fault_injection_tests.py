"""Offline fault-injection tests for ac_vjepa_core.

Safety boundary: this test module only mutates synthetic/replayed tensors and
state envelopes. It never imports a robot SDK or sends actuator commands.

Run:
    python3 -m unittest -v ac_vjepa_fault_injection_tests.py
"""
from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from typing import Dict

import torch

from ac_vjepa_core import (
    ActionConditionedVJEPA,
    FallbackReason,
    LatencyMonitor,
    RecoveryChoice,
    RuntimeLimits,
    RuntimeMode,
    SafeHandoverCoordinator,
    SingleFlightInferenceWorker,
    StateEnvelope,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class FaultProfile:
    gaussian_std: float = 0.0
    brightness_gain: float = 1.0
    frame_drop_probability: float = 0.0
    freeze_tail_frames: int = 0
    proprio_std: float = 0.0
    inject_nan: bool = False


class SensorFaultInjector:
    """Applies deterministic tensor faults to offline/replayed sensor windows."""

    def __init__(self, seed: int = 1234):
        self.generator = torch.Generator().manual_seed(seed)

    def apply(self, inputs: Dict[str, Tensor], profile: FaultProfile) -> Dict[str, Tensor]:
        output = {name: tensor.clone() for name, tensor in inputs.items()}
        video = output["context_video"]
        proprio = output["context_proprio"]

        if profile.gaussian_std > 0:
            noise = torch.randn(video.shape, generator=self.generator, dtype=video.dtype)
            output["context_video"] = video + profile.gaussian_std * noise
            video = output["context_video"]
        if profile.brightness_gain != 1.0:
            output["context_video"] = video * profile.brightness_gain
            video = output["context_video"]
        if profile.frame_drop_probability > 0:
            # Dropped frame means a zero frame; the runtime's state-age logic must
            # separately decide whether this makes the observation unsafe/stale.
            keep = torch.rand(video.shape[:2], generator=self.generator) >= profile.frame_drop_probability
            output["context_video"] = video * keep[:, :, None, None, None].to(video.dtype)
            video = output["context_video"]
        if profile.freeze_tail_frames > 0:
            count = min(profile.freeze_tail_frames, video.shape[1] - 1)
            output["context_video"][:, -count:] = video[:, -count - 1 : -count]
        if profile.proprio_std > 0:
            noise = torch.randn(proprio.shape, generator=self.generator, dtype=proprio.dtype)
            output["context_proprio"] = proprio + profile.proprio_std * noise
        if profile.inject_nan:
            output["context_video"][:, -1, :, 0, 0] = float("nan")
        return output


class SlowModel(ActionConditionedVJEPA):
    """Deliberately slow model used only to test timeout and single-flight logic."""

    def predict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(0.10)
        return super().predict(*args, **kwargs)


class ACVJEPAFaultInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.device = torch.device("cpu")
        cls.batch, cls.context_steps, cls.horizon = 2, 4, 3
        cls.channels, cls.height, cls.width = 3, 32, 32
        cls.proprio_dim, cls.action_dim, cls.event_dim = 8, 20, 4

    def make_model(self, slow: bool = False) -> ActionConditionedVJEPA:
        model_cls = SlowModel if slow else ActionConditionedVJEPA
        return model_cls(
            image_channels=self.channels,
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
            latent_dim=32,
            event_dim=self.event_dim,
            max_horizon=8,
        )

    def make_inputs(self) -> Dict[str, Tensor]:
        return {
            "context_video": torch.randn(
                self.batch, self.context_steps, self.channels, self.height, self.width
            ),
            "context_proprio": torch.randn(self.batch, self.context_steps, self.proprio_dim),
            "action_blocks": torch.randn(self.batch, self.horizon, self.action_dim),
        }

    def make_state(self, age_ms: float = 10.0) -> StateEnvelope:
        return StateEnvelope(
            state_id="test-state-v1",
            state_age_ms=age_ms,
            sensor_timestamp_ns=time.time_ns(),
            robot_calibration_id="test-calibration-v1",
            facts={"human_distance_m": 2.0, "protected_zone_clear": True},
        )

    def make_coordinator(
        self, model: ActionConditionedVJEPA, limits: RuntimeLimits
    ) -> tuple[SafeHandoverCoordinator, SingleFlightInferenceWorker]:
        worker = SingleFlightInferenceWorker(model, self.device, LatencyMonitor())
        return SafeHandoverCoordinator(worker, limits), worker

    def test_visual_noise_keeps_prediction_contract_finite(self) -> None:
        model = self.make_model().eval()
        inputs = self.make_inputs()
        corrupted = SensorFaultInjector().apply(
            inputs,
            FaultProfile(gaussian_std=0.08, brightness_gain=0.85, frame_drop_probability=0.25),
        )
        with torch.inference_mode():
            prediction = model.predict(**corrupted)
        self.assertEqual(
            tuple(prediction.future_latents.shape), (self.batch, self.horizon, 32)
        )
        self.assertFalse(prediction.has_invalid_values())

    def test_stale_sensor_state_enters_local_hold_before_any_model_call(self) -> None:
        coordinator, worker = self.make_coordinator(
            self.make_model(), RuntimeLimits(max_state_age_ms=5.0, plan_deadline_ms=1000.0)
        )
        try:
            decision = coordinator.plan_or_handover(
                self.make_state(age_ms=20.0), self.make_inputs(), hardware_healthy=True
            )
            self.assertEqual(decision.mode, RuntimeMode.LOCAL_HOLD)
            self.assertIsNotNone(decision.fallback)
            self.assertEqual(decision.fallback.reason, FallbackReason.STALE_STATE)
        finally:
            worker.close()

    def test_nan_sensor_data_is_rejected_to_local_hold(self) -> None:
        coordinator, worker = self.make_coordinator(
            self.make_model(), RuntimeLimits(plan_deadline_ms=1000.0)
        )
        try:
            corrupted = SensorFaultInjector().apply(self.make_inputs(), FaultProfile(inject_nan=True))
            decision = coordinator.plan_or_handover(
                self.make_state(), corrupted, hardware_healthy=True
            )
            self.assertEqual(decision.mode, RuntimeMode.LOCAL_HOLD)
            self.assertEqual(decision.fallback.reason, FallbackReason.INVALID_MODEL_OUTPUT)
        finally:
            worker.close()

    def test_high_uncertainty_requires_local_hold_then_llm_supervision(self) -> None:
        coordinator, worker = self.make_coordinator(
            self.make_model(),
            RuntimeLimits(plan_deadline_ms=1000.0, max_mean_uncertainty=0.0),
        )
        try:
            decision = coordinator.plan_or_handover(
                self.make_state(), self.make_inputs(), hardware_healthy=True
            )
            self.assertEqual(decision.mode, RuntimeMode.LOCAL_HOLD)
            self.assertEqual(decision.fallback.reason, FallbackReason.HIGH_UNCERTAINTY)

            supervised = coordinator.confirm_local_hold(decision, hold_confirmed=True)
            self.assertEqual(supervised.mode, RuntimeMode.LLM_SUPERVISION)
            self.assertIsNotNone(supervised.supervision_packet)

            # LLM cannot unilaterally resume. "observe" remains in supervision.
            mode = coordinator.apply_llm_recovery_choice(
                supervised.supervision_packet,
                RecoveryChoice.OBSERVE,
                state_is_fresh=True,
                hardware_healthy=True,
            )
            self.assertEqual(mode, RuntimeMode.LLM_SUPERVISION)

            # A registered recovery skill can only request REPLAN_PENDING.
            mode = coordinator.apply_llm_recovery_choice(
                supervised.supervision_packet,
                RecoveryChoice.SELECT_PREAPPROVED_SKILL,
                state_is_fresh=True,
                hardware_healthy=True,
                selected_skill_registered=True,
            )
            self.assertEqual(mode, RuntimeMode.REPLAN_PENDING)
        finally:
            worker.close()

    def test_deadline_timeout_then_single_flight_gpu_busy(self) -> None:
        coordinator, worker = self.make_coordinator(
            self.make_model(slow=True), RuntimeLimits(plan_deadline_ms=1.0)
        )
        try:
            first = coordinator.plan_or_handover(
                self.make_state(), self.make_inputs(), hardware_healthy=True
            )
            self.assertEqual(first.mode, RuntimeMode.LOCAL_HOLD)
            self.assertEqual(first.fallback.reason, FallbackReason.INFERENCE_TIMEOUT)

            # The slow request still executes. A second request must not queue behind
            # it and later apply to a newer physical state.
            second = coordinator.plan_or_handover(
                self.make_state(), self.make_inputs(), hardware_healthy=True
            )
            self.assertEqual(second.mode, RuntimeMode.LOCAL_HOLD)
            self.assertEqual(second.fallback.reason, FallbackReason.GPU_BUSY)
        finally:
            time.sleep(0.15)  # allow the deliberately slow CPU request to finish
            worker.close()

    def test_hardware_health_fault_is_hard_stop_not_llm_recovery(self) -> None:
        coordinator, worker = self.make_coordinator(
            self.make_model(), RuntimeLimits(plan_deadline_ms=1000.0)
        )
        try:
            decision = coordinator.plan_or_handover(
                self.make_state(), self.make_inputs(), hardware_healthy=False
            )
            self.assertEqual(decision.mode, RuntimeMode.HARD_STOP)
            self.assertIsNone(decision.supervision_packet)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
