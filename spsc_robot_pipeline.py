"""SPSC latest-only pipeline for an AC-VJEPA robot research prototype.

This is deliberately NOT a real robot controller. It has no vendor SDK, no motor
commands, no real-time scheduling guarantee, and no safety certification. It
shows how to keep sensor capture, state preparation, model inference and control
window consumption decoupled so a slow model cannot queue stale work.

Run:
    python3 spsc_robot_pipeline.py
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, List, Optional, TypeVar
from uuid import uuid4

import numpy as np

T = TypeVar("T")


class SPSCRing(Generic[T]):
    """Bounded single-producer/single-consumer ring with drop-oldest semantics.

    This simple implementation relies on CPython's GIL for the demonstration and
    assumes exactly one thread calls `push` and exactly one calls `pop/drain_latest`.
    Production robot software should use a validated lock-free native SPSC queue
    or process-isolated shared memory with explicit memory-order guarantees.
    """

    def __init__(self, capacity: int, name: str):
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        self.name = name
        self.capacity = capacity
        self._slots: List[Optional[T]] = [None] * capacity
        self._head = 0  # producer-owned logical write index
        self._tail = 0  # consumer-owned logical read index
        self.dropped = 0
        self.closed = False

    def push(self, item: T) -> bool:
        """Write newest item. If full, discard the oldest unread item."""
        if self.closed:
            return False
        if self._head - self._tail >= self.capacity:
            self._slots[self._tail % self.capacity] = None
            self._tail += 1
            self.dropped += 1
        self._slots[self._head % self.capacity] = item
        self._head += 1
        return True

    def pop(self) -> Optional[T]:
        if self._tail >= self._head:
            return None
        index = self._tail % self.capacity
        item = self._slots[index]
        self._slots[index] = None
        self._tail += 1
        return item

    def drain_latest(self) -> Optional[T]:
        """Consume all available elements and return only the freshest one."""
        newest: Optional[T] = None
        while True:
            item = self.pop()
            if item is None:
                return newest
            newest = item

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class SensorPacket:
    packet_id: str
    capture_ns: int
    rgb: np.ndarray
    proprio: np.ndarray


@dataclass(frozen=True)
class StateEstimate:
    state_id: str
    source_packet_id: str
    capture_ns: int
    prepared_ns: int
    features: np.ndarray
    state_age_ms: float


@dataclass(frozen=True)
class PredictionReport:
    state_id: str
    predicted_ns: int
    uncertainty: float
    planning_latency_ms: float
    safe_candidate_id: Optional[str]


@dataclass(frozen=True)
class ControlWindow:
    window_id: str
    source_state_id: str
    created_ns: int
    expires_ns: int
    candidate_id: str
    # In production this would reference a pre-validated trajectory payload.
    # It intentionally contains no motor current/force command in this prototype.


class SafetyMode(str, Enum):
    NORMAL = "normal"
    LOCAL_HOLD = "local_hold"
    HARD_STOP = "hard_stop"


class SafetyLatch:
    """Small thread-safe safety state used by all workers.

    A real safety kernel must be independent of Python and GPU execution. This
    latch only demonstrates the contract: when hold is active, the control loop
    refuses new windows and emits a non-actuating HOLD decision.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode = SafetyMode.NORMAL
        self._reason = ""

    def request_hold(self, reason: str) -> None:
        with self._lock:
            if self._mode != SafetyMode.HARD_STOP:
                self._mode = SafetyMode.LOCAL_HOLD
                self._reason = reason

    def hard_stop(self, reason: str) -> None:
        with self._lock:
            self._mode = SafetyMode.HARD_STOP
            self._reason = reason

    def snapshot(self) -> tuple[SafetyMode, str]:
        with self._lock:
            return self._mode, self._reason


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.values = {
            "captured": 0,
            "prepared": 0,
            "inferred": 0,
            "published_windows": 0,
            "control_accepted": 0,
            "control_holds": 0,
            "deadline_misses": 0,
            "high_uncertainty": 0,
            "stale_states": 0,
        }

    def inc(self, key: str) -> None:
        with self._lock:
            self.values[key] += 1

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.values)


def now_ns() -> int:
    return time.monotonic_ns()


def ms_since(timestamp_ns: int) -> float:
    return (now_ns() - timestamp_ns) / 1_000_000.0


class Pipeline:
    def __init__(
        self,
        *,
        sensor_hz: float = 30.0,
        state_hz: float = 30.0,
        inference_hz: float = 10.0,
        control_hz: float = 250.0,
        max_state_age_ms: float = 120.0,
        plan_deadline_ms: float = 30.0,
        max_uncertainty: float = 1.2,
        control_window_ms: float = 80.0,
        predict_fn: Optional[Callable[[StateEstimate], PredictionReport]] = None,
    ):
        self.sensor_period_s = 1.0 / sensor_hz
        self.state_period_s = 1.0 / state_hz
        self.inference_period_s = 1.0 / inference_hz
        self.control_period_s = 1.0 / control_hz
        self.max_state_age_ms = max_state_age_ms
        self.plan_deadline_ms = plan_deadline_ms
        self.max_uncertainty = max_uncertainty
        self.control_window_ns = int(control_window_ms * 1_000_000)

        # Every ring has exactly one producer and one consumer.
        self.sensor_ring: SPSCRing[SensorPacket] = SPSCRing(4, "sensor_to_state")
        self.state_ring: SPSCRing[StateEstimate] = SPSCRing(2, "state_to_inference")
        self.window_ring: SPSCRing[ControlWindow] = SPSCRing(2, "planner_to_control")
        self.safety = SafetyLatch()
        self.metrics = Metrics()
        self.stop_event = threading.Event()
        self.predict_fn = predict_fn or self._fake_predict
        self.threads: List[threading.Thread] = []

    def _fake_capture(self) -> SensorPacket:
        # Synthetic only. Replace with a timestamped camera/proprio source in a
        # sandboxed integration layer, never in the safety/servo loop.
        return SensorPacket(
            packet_id=str(uuid4()),
            capture_ns=now_ns(),
            rgb=np.random.rand(3, 64, 64).astype(np.float32),
            proprio=np.random.normal(0.0, 0.1, size=(8,)).astype(np.float32),
        )

    def _prepare_state(self, packet: SensorPacket) -> StateEstimate:
        # Fixed-shape placeholder for crop/normalize/timestamp alignment/feature prep.
        features = np.concatenate((packet.rgb.mean(axis=(1, 2)), packet.proprio))
        prepared_ns = now_ns()
        return StateEstimate(
            state_id=str(uuid4()),
            source_packet_id=packet.packet_id,
            capture_ns=packet.capture_ns,
            prepared_ns=prepared_ns,
            features=features,
            state_age_ms=ms_since(packet.capture_ns),
        )

    def _fake_predict(self, state: StateEstimate) -> PredictionReport:
        # Stand-in for a single-flight ORT/TensorRT worker. A real implementation
        # checks deadline from the caller and never allows a late result to enqueue.
        start_ns = now_ns()
        time.sleep(0.003)  # simulated GPU work, intentionally below the default deadline
        uncertainty = abs(float(np.sin(state.features.mean() * 5.0)))
        return PredictionReport(
            state_id=state.state_id,
            predicted_ns=now_ns(),
            uncertainty=uncertainty,
            planning_latency_ms=(now_ns() - start_ns) / 1_000_000.0,
            safe_candidate_id="registered_skill_candidate_0",
        )

    def _sensor_worker(self) -> None:
        next_deadline = time.monotonic()
        while not self.stop_event.is_set():
            packet = self._fake_capture()
            self.sensor_ring.push(packet)
            self.metrics.inc("captured")
            next_deadline += self.sensor_period_s
            time.sleep(max(0.0, next_deadline - time.monotonic()))

    def _state_worker(self) -> None:
        next_deadline = time.monotonic()
        while not self.stop_event.is_set():
            # latest-only prevents expensive state preparation of stale camera frames.
            packet = self.sensor_ring.drain_latest()
            if packet is not None:
                state = self._prepare_state(packet)
                self.state_ring.push(state)
                self.metrics.inc("prepared")
            next_deadline += self.state_period_s
            time.sleep(max(0.0, next_deadline - time.monotonic()))

    def _inference_worker(self) -> None:
        next_deadline = time.monotonic()
        while not self.stop_event.is_set():
            state = self.state_ring.drain_latest()
            if state is not None:
                age_ms = ms_since(state.capture_ns)
                if age_ms > self.max_state_age_ms:
                    self.metrics.inc("stale_states")
                    self.safety.request_hold("state_too_old")
                else:
                    report = self.predict_fn(state)
                    self.metrics.inc("inferred")
                    if report.planning_latency_ms > self.plan_deadline_ms:
                        self.metrics.inc("deadline_misses")
                        self.safety.request_hold("model_deadline_miss")
                    elif report.uncertainty > self.max_uncertainty:
                        self.metrics.inc("high_uncertainty")
                        self.safety.request_hold("model_high_uncertainty")
                    elif report.safe_candidate_id is None:
                        self.safety.request_hold("no_safe_candidate")
                    else:
                        # The planner is the only producer for the control ring.
                        self.window_ring.push(
                            ControlWindow(
                                window_id=str(uuid4()),
                                source_state_id=state.state_id,
                                created_ns=now_ns(),
                                expires_ns=now_ns() + self.control_window_ns,
                                candidate_id=report.safe_candidate_id,
                            )
                        )
                        self.metrics.inc("published_windows")
            next_deadline += self.inference_period_s
            time.sleep(max(0.0, next_deadline - time.monotonic()))

    def _control_worker(self) -> None:
        """Demonstrates local window validation; does not command hardware."""
        active_window: Optional[ControlWindow] = None
        next_deadline = time.monotonic()
        while not self.stop_event.is_set():
            newest = self.window_ring.drain_latest()
            if newest is not None:
                active_window = newest

            mode, _reason = self.safety.snapshot()
            valid_window = active_window is not None and now_ns() < active_window.expires_ns
            if mode == SafetyMode.NORMAL and valid_window:
                # Production: only here would a separately approved trajectory
                # reference be supplied to a vendor real-time bridge.
                self.metrics.inc("control_accepted")
            else:
                # Production: a separate local safety/servo controller holds or
                # retreats. This demo only records the non-actuating decision.
                self.metrics.inc("control_holds")
                active_window = None

            next_deadline += self.control_period_s
            time.sleep(max(0.0, next_deadline - time.monotonic()))

    def start(self) -> None:
        workers = (
            ("sensor-capture", self._sensor_worker),
            ("state-prep", self._state_worker),
            ("model-inference", self._inference_worker),
            ("control-window", self._control_worker),
        )
        self.threads = [threading.Thread(name=name, target=target, daemon=True) for name, target in workers]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for ring in (self.sensor_ring, self.state_ring, self.window_ring):
            ring.close()
        for thread in self.threads:
            thread.join(timeout=1.0)


def main() -> None:
    pipeline = Pipeline()
    pipeline.start()
    time.sleep(1.0)
    pipeline.stop()
    mode, reason = pipeline.safety.snapshot()
    print(
        {
            "metrics": pipeline.metrics.snapshot(),
            "ring_drops": {
                pipeline.sensor_ring.name: pipeline.sensor_ring.dropped,
                pipeline.state_ring.name: pipeline.state_ring.dropped,
                pipeline.window_ring.name: pipeline.window_ring.dropped,
            },
            "safety_mode": mode.value,
            "safety_reason": reason,
        }
    )


if __name__ == "__main__":
    main()
