"""ONNX Runtime + TensorRT EP wrapper for AC-VJEPA edge inference.

This module runs only the ONNX inference subgraph. It does NOT send robot control
commands. The caller must pass results to an independent local SafetyKernel,
trajectory TTL checker and real-time controller.

Run on a Jetson only after installing an ONNX Runtime build compatible with the
installed JetPack/CUDA/TensorRT stack.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # Allows static inspection on non-edge development machines.
    ort = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EdgeRuntimeConfig:
    onnx_path: str
    engine_cache_dir: str
    timing_cache_dir: str
    workspace_bytes: int = 2 * 1024 * 1024 * 1024
    fp16: bool = True
    int8: bool = False
    plan_deadline_ms: float = 100.0
    max_state_age_ms: float = 120.0
    device_id: int = 0


@dataclass(frozen=True)
class EdgeInferenceResult:
    request_id: str
    status: str  # ok | stale_state | deadline_miss | invalid_output | runtime_error
    latency_ms: float
    outputs: Optional[Dict[str, np.ndarray]]
    reason: Optional[str] = None


def build_ort_session(config: EdgeRuntimeConfig) -> Any:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed; use a JetPack-compatible ORT build on Jetson")

    engine_dir = Path(config.engine_cache_dir)
    timing_dir = Path(config.timing_cache_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Keep intra/inter op execution bounded. The actual best setting must be
    # measured on the Jetson alongside camera and control workloads.
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    tensorrt_options = {
        "device_id": config.device_id,
        "trt_max_workspace_size": config.workspace_bytes,
        "trt_fp16_enable": config.fp16,
        "trt_int8_enable": config.int8,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine_dir),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(timing_dir),
        # Static ONNX shapes are preferred for deterministic edge p99 latency.
        # CUDA EP remains a required fallback for unsupported TensorRT subgraphs.
    }
    cuda_options = {
        "device_id": config.device_id,
        "do_copy_in_default_stream": True,
    }
    return ort.InferenceSession(
        config.onnx_path,
        sess_options=session_options,
        providers=[
            ("TensorrtExecutionProvider", tensorrt_options),
            ("CUDAExecutionProvider", cuda_options),
        ],
    )


class EdgeACVJepaRuntime:
    def __init__(self, config: EdgeRuntimeConfig):
        self.config = config
        self.session = build_ort_session(config)
        self.input_names = {item.name for item in self.session.get_inputs()}
        expected = {"context_video", "context_proprio", "action_blocks"}
        if self.input_names != expected:
            raise ValueError(f"ONNX input contract mismatch: expected {expected}, got {self.input_names}")

    def infer(
        self,
        *,
        context_video: np.ndarray,
        context_proprio: np.ndarray,
        action_blocks: np.ndarray,
        state_age_ms: float,
    ) -> EdgeInferenceResult:
        request_id = str(uuid4())
        if state_age_ms > self.config.max_state_age_ms:
            return EdgeInferenceResult(
                request_id=request_id,
                status="stale_state",
                latency_ms=0.0,
                outputs=None,
                reason="state_age_exceeded",
            )

        inputs = {
            "context_video": np.ascontiguousarray(context_video, dtype=np.float32),
            "context_proprio": np.ascontiguousarray(context_proprio, dtype=np.float32),
            "action_blocks": np.ascontiguousarray(action_blocks, dtype=np.float32),
        }
        start = time.perf_counter()
        try:
            values = self.session.run(
                ["future_latents", "log_variance", "event_logits"], inputs
            )
        except Exception as exc:  # Runtime errors must cause local hold upstream.
            elapsed = (time.perf_counter() - start) * 1000.0
            return EdgeInferenceResult(request_id, "runtime_error", elapsed, None, repr(exc))

        elapsed = (time.perf_counter() - start) * 1000.0
        if elapsed > self.config.plan_deadline_ms:
            # Do not use a late output simply because it eventually arrived.
            return EdgeInferenceResult(request_id, "deadline_miss", elapsed, None, "plan_deadline_exceeded")

        outputs = dict(zip(["future_latents", "log_variance", "event_logits"], values))
        if any(not np.isfinite(value).all() for value in outputs.values()):
            return EdgeInferenceResult(request_id, "invalid_output", elapsed, None, "non_finite_onnx_output")
        return EdgeInferenceResult(request_id, "ok", elapsed, outputs)
