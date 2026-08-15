"""Offline CI harness for failpoint timing and cache observability.

This runner never invokes real torchrun/NCCL/RDMA/network/KV services. It runs
one logical failpoint at a time, emits Prometheus text exposition, and writes a
sanitized JSON report. Production code must emit the same bounded phase metrics
at actual recovery boundaries; this harness validates the metric contract and
end-to-end recovery assertions in CI.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from checkpoint_cache_load_shedding_simulator import CheckpointLoadSheddingSimulator
from distributed_training_observability import DistributedTrainingMetrics, TrainingMetricLabels
from heterogeneous_microbatch_chaos_framework import ChaosFault, ChaosScenario, HeterogeneousMicrobatchChaosFramework


@dataclass(frozen=True)
class DrillResult:
    fault_class: str
    passed: bool
    duration_seconds: float
    assertion_count: int


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_drill(*, report_path: Path, metrics_path: Path) -> dict[str, object]:
    labels = TrainingMetricLabels(cluster="ci-offline", job="acvjepa-failpoint-drill", environment="isolated-preproduction")
    metrics = DistributedTrainingMetrics(labels)
    framework = HeterogeneousMicrobatchChaosFramework()
    results: list[DrillResult] = []
    for index, fault in enumerate(ChaosFault, start=1):
        metrics.record_failpoint(fault_class=fault.value, phase="trigger_to_detect", duration_seconds=0.0, active=True)
        started = time.monotonic()
        result = framework.run(ChaosScenario(fault=fault, experiment_id=f"ci-{index}-{fault.value}", seed=index))
        elapsed = time.monotonic() - started
        metrics.record_failpoint(
            fault_class=fault.value,
            phase="trigger_to_training_ready",
            duration_seconds=elapsed,
            active=False,
            outcome="passed" if result.passed else "failed",
        )
        if not result.passed:
            raise AssertionError(f"failpoint scenario did not pass: {fault.value}")
        results.append(DrillResult(fault.value, result.passed, elapsed, len(result.assertions)))

    # Exercise low-cardinality hit/fallback metrics using the offline cache-stampede
    # model. This creates no storage or network traffic.
    simulator = CheckpointLoadSheddingSimulator()
    storm = simulator.cold_key_storm(key="optimizer:ci-shard-0", nodes=[f"node-{index}" for index in range(8)])
    metrics.record_cache_fetch(cache_tier="durable", component_class="optimizer", outcome="fallback", byte_count=0, duration_seconds=0.0)
    for _ in range(storm.cache_hits_after_fill):
        metrics.record_cache_fetch(cache_tier="node_local", component_class="optimizer", outcome="hit", byte_count=0, duration_seconds=0.0)

    report = {
        "schema_version": 1,
        "environment": "isolated-preproduction",
        "mode": "offline-logical-failpoints",
        "results": [asdict(item) for item in results],
        "cache_observation": {
            "request_count": storm.request_count,
            "durable_fetches": storm.durable_fetches,
            "coalesced_waiters": storm.coalesced_waiters,
            "cache_hits_after_fill": storm.cache_hits_after_fill,
        },
        "safety": "No real NCCL, RDMA, network, node, KV, object-store, deploy, or release operation was invoked.",
    }
    _write(report_path, json.dumps(report, sort_keys=True, indent=2) + "\n")
    _write(metrics_path, metrics.exposition())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline failpoint observability drill")
    parser.add_argument("--report", default="artifacts/failpoint_drill_report.json")
    parser.add_argument("--metrics", default="artifacts/failpoint_metrics.prom")
    args = parser.parse_args()
    report = run_drill(report_path=Path(args.report), metrics_path=Path(args.metrics))
    assert len(report["results"]) == 5
    assert all(item["passed"] for item in report["results"])
    assert report["cache_observation"]["durable_fetches"] == 1
    print(json.dumps({"smoke_test": "passed", "failpoints": len(report["results"]), "cache_hits": report["cache_observation"]["cache_hits_after_fill"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
