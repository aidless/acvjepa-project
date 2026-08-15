#!/usr/bin/env sh
# Offline logical contract only: no cluster, network, RDMA, KV, or deployment operation.
set -eu

ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/artifacts}"
mkdir -p "$ARTIFACT_DIR"

python -m unittest -v test_heterogeneous_microbatch_failpoints.py | tee "$ARTIFACT_DIR/failpoint_unit_tests.txt"
python heterogeneous_microbatch_chaos_framework.py | tee "$ARTIFACT_DIR/aggregate_chaos_framework.txt"
python threadsafe_checkpoint_load_gate.py | tee "$ARTIFACT_DIR/threadsafe_gate.txt"
python checkpoint_cache_load_shedding_simulator.py | tee "$ARTIFACT_DIR/cache_load_shedding.txt"
python checkpoint_integrity_corruption_demo.py | tee "$ARTIFACT_DIR/cache_corruption_demo.txt"
python recovery_deployment_arbiter.py | tee "$ARTIFACT_DIR/recovery_deployment_arbiter.txt"
python run_failpoint_observability_drill.py --report "$ARTIFACT_DIR/failpoint_drill_report.json" --metrics "$ARTIFACT_DIR/failpoint_metrics.prom" | tee "$ARTIFACT_DIR/failpoint_observability.txt"
python validate_monitoring_config.py | tee "$ARTIFACT_DIR/monitoring_config_validation.txt"
python validate_failpoint_ci_config.py | tee "$ARTIFACT_DIR/ci_config_validation.txt"
python validate_local_compose.py | tee "$ARTIFACT_DIR/compose_config_validation.txt"
python validate_kubernetes_chaos_lab.py | tee "$ARTIFACT_DIR/kubernetes_chaos_lab_validation.txt"
python validate_kubernetes_chaos_ci.py | tee "$ARTIFACT_DIR/kubernetes_chaos_ci_validation.txt"
