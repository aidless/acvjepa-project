#!/usr/bin/env sh
# Offline logical contract only: no cluster, network, RDMA, KV, or deployment operation.
# Each python command's own exit status is the contract status. A pipe into tee
# used to report success when python failed, because tee was the last stage.
set -eu

ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/artifacts}"
mkdir -p "$ARTIFACT_DIR"

run_py() {
  log=$1
  shift
  set +e
  python "$@" >"$log" 2>&1
  status=$?
  set -e
  cat "$log"
  if [ "$status" -ne 0 ]; then
    echo "contract command failed (exit $status): python $*" >&2
    exit "$status"
  fi
}

run_py "$ARTIFACT_DIR/failpoint_unit_tests.txt" -m unittest -v test_heterogeneous_microbatch_failpoints.py
run_py "$ARTIFACT_DIR/aggregate_chaos_framework.txt" heterogeneous_microbatch_chaos_framework.py
run_py "$ARTIFACT_DIR/threadsafe_gate.txt" threadsafe_checkpoint_load_gate.py
run_py "$ARTIFACT_DIR/cache_load_shedding.txt" checkpoint_cache_load_shedding_simulator.py
run_py "$ARTIFACT_DIR/cache_corruption_demo.txt" checkpoint_integrity_corruption_demo.py
run_py "$ARTIFACT_DIR/recovery_deployment_arbiter.txt" recovery_deployment_arbiter.py
run_py "$ARTIFACT_DIR/failpoint_observability.txt" run_failpoint_observability_drill.py --report "$ARTIFACT_DIR/failpoint_drill_report.json" --metrics "$ARTIFACT_DIR/failpoint_metrics.prom"
run_py "$ARTIFACT_DIR/monitoring_config_validation.txt" validate_monitoring_config.py
# CI workflow checks require the repo checkout (.github/workflows). The container
# image deliberately excludes .github via .dockerignore (no CI files shipped into
# the demo image), so skip these two inside the container and keep them strict
# when run from the repo checkout.
if [ -f .github/workflows/failpoint-chaos-ci.yml ]; then
  run_py "$ARTIFACT_DIR/ci_config_validation.txt" validate_failpoint_ci_config.py
else
  echo "SKIP validate_failpoint_ci_config.py (.github not shipped to container image)" | tee "$ARTIFACT_DIR/ci_config_validation.txt"
fi
run_py "$ARTIFACT_DIR/compose_config_validation.txt" validate_local_compose.py
run_py "$ARTIFACT_DIR/kubernetes_chaos_lab_validation.txt" validate_kubernetes_chaos_lab.py
if [ -f .github/workflows/kubernetes-chaos-contract.yml ]; then
  run_py "$ARTIFACT_DIR/kubernetes_chaos_ci_validation.txt" validate_kubernetes_chaos_ci.py
else
  echo "SKIP validate_kubernetes_chaos_ci.py (.github not shipped to container image)" | tee "$ARTIFACT_DIR/kubernetes_chaos_ci_validation.txt"
fi
# Stdlib-only unit tests that do not need torch, numpy, weights, or Docker.
run_py "$ARTIFACT_DIR/shadow_canary_unit_tests.txt" -m unittest -v test_shadow_canary_gate.py
