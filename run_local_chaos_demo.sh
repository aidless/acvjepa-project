#!/usr/bin/env sh
# Local-only, offline synthetic demo. It does not contact a cluster or cloud service.
set -eu

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.local-chaos.yml}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts-compose}"

mkdir -p "$ARTIFACT_DIR"

docker compose -f "$COMPOSE_FILE" build metrics-demo chaos-ci
docker compose -f "$COMPOSE_FILE" up -d metrics-demo prometheus grafana

echo "Waiting for the local metrics service healthcheck..."
for attempt in $(seq 1 30); do
  status="$(docker compose -f "$COMPOSE_FILE" ps --format json metrics-demo 2>/dev/null || true)"
  if printf '%s' "$status" | grep -q 'healthy'; then
    break
  fi
  if [ "$attempt" -eq 30 ]; then
    echo "metrics-demo did not become healthy; inspect: docker compose -f $COMPOSE_FILE logs metrics-demo" >&2
    exit 1
  fi
  sleep 1
done

docker compose -f "$COMPOSE_FILE" --profile ci run --rm chaos-ci

echo ""
echo "Local offline demo is ready:"
echo "  metrics report: http://127.0.0.1:18000/report"
echo "  Prometheus:     http://127.0.0.1:19090"
echo "  Grafana:        http://127.0.0.1:13000  (local-demo / local-demo-not-for-production)"
echo "  CI artifacts:   $ARTIFACT_DIR/"
echo ""
echo "Stop and remove local containers: docker compose -f $COMPOSE_FILE down --volumes --remove-orphans"
