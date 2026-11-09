#!/bin/sh
# Fresh-clone verification for files that are actually in the repository.
# Does not download weights, start Docker, or require a GPU.
# Optional groups inside verify_all.ps1 are a separate host log; see VERIFY_RESULTS.md.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if ! command -v python >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    shim=$(mktemp -d)
    ln -s "$(command -v python3)" "$shim/python"
    PATH="$shim:$PATH"
    export PATH
  else
    echo "verify_committed: python is not on PATH" >&2
    exit 1
  fi
fi

python scripts/check_verification_drift.py

python -c "import yaml, prometheus_client" || {
  echo "verify_committed: install PyYAML and prometheus-client." >&2
  echo "The GitHub contract job pins PyYAML==6.0.2 and prometheus-client==0.20.0." >&2
  exit 1
}

ARTIFACT_DIR="${ARTIFACT_DIR:-verify_artifacts/offline_contract}"
export ARTIFACT_DIR
bash scripts/run_offline_chaos_contract.sh

echo "verify_committed: drift guard and offline contract passed."
echo "Not run: verify_all.ps1 groups that need torch, weights, CUDA, or Docker."
