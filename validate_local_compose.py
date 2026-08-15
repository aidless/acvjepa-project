"""Static safety validation for the local offline Compose demo."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
COMPOSE = ROOT / "docker-compose.local-chaos.yml"


def main() -> None:
    raw = COMPOSE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert data["name"] == "acvjepa-local-chaos"
    assert set(data["services"]) == {"metrics-demo", "prometheus", "grafana", "chaos-ci"}
    assert data["networks"]["local-chaos"]["internal"] is True
    forbidden = ("privileged:", "network_mode: host", "pid: host", "docker.sock", "devices:", "cap_add:", "rdma", "nccl", "torchrun")
    assert not any(token in raw.lower() for token in forbidden)
    for name, service in data["services"].items():
        assert service["read_only"] is True, name
        assert service["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in service["security_opt"], name
        assert service["networks"] == ["local-chaos"], name
        for binding in service.get("ports", []):
            assert str(binding).startswith("127.0.0.1:"), (name, binding)
    assert data["services"]["metrics-demo"]["healthcheck"]["test"] == ["CMD", "python", "docker/healthcheck.py"]
    assert data["services"]["prometheus"]["depends_on"]["metrics-demo"]["condition"] == "service_healthy"
    assert data["services"]["chaos-ci"]["profiles"] == ["ci"]
    assert (ROOT / "docker/prometheus.local.yml").is_file()
    assert (ROOT / "docker/grafana/provisioning/datasources/local-prometheus.yml").is_file()
    assert (ROOT / "docker/grafana/provisioning/dashboards/local-dashboard.yml").is_file()
    print('{"smoke_test":"passed","compose_services":4,"network":"internal","published_ports":"loopback_only"}', flush=True)


if __name__ == "__main__":
    main()
