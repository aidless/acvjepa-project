"""Static validation for the supplied Prometheus rules and Grafana dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RULES = ROOT / "monitoring" / "prometheus_rules.yml"
DASHBOARD = ROOT / "monitoring" / "grafana_acvjepa_elastic_dashboard.json"


def main() -> None:
    rules = yaml.safe_load(RULES.read_text())
    assert isinstance(rules, dict) and len(rules.get("groups", [])) == 2
    entries = [rule for group in rules["groups"] for rule in group["rules"]]
    alerts = [rule for rule in entries if "alert" in rule]
    recordings = [rule for rule in entries if "record" in rule]
    assert len(alerts) == 18 and len(recordings) == 10
    assert all("expr" in rule and "labels" in rule for rule in alerts)
    assert all("action" in rule["labels"] for rule in alerts)
    assert any(rule["alert"] == "ACVJEPAExactStateRestoreFailed" for rule in alerts)
    assert any(rule["alert"] == "ACVJEPAFP8MetadataRestoreFailed" for rule in alerts)
    assert any(rule["alert"] == "ACVJEPAFailpointRecoveryTailHigh" for rule in alerts)
    assert any(rule["alert"] == "ACVJEPACacheIntegrityFailure" for rule in alerts)
    assert any(rule["alert"] == "ACVJEPACacheStampedeRisk" for rule in alerts)
    assert any(rule["alert"] == "ACVJEPAGitOpsRevisionDriftDuringRecovery" for rule in alerts)
    assert any(rule["alert"] == "ACVJEPARendezvousGitOpsFencingConflict" for rule in alerts)
    assert any(rule["record"] == "job:acvjepa_training_failpoint_recovery_p95_seconds:30m" for rule in recordings)
    assert any(rule["record"] == "job:acvjepa_training_checkpoint_cache_hit_ratio:5m" for rule in recordings)
    assert any(rule["record"] == "job:acvjepa_training_checkpoint_durable_fallback_p95_seconds:5m" for rule in recordings)
    assert any(rule["record"] == "job:acvjepa_training_recovery_deployment_fence_rejections:rate5m" for rule in recordings)

    dashboard = json.loads(DASHBOARD.read_text())
    assert dashboard["uid"] == "acvjepa-elastic-control-plane"
    assert len(dashboard["panels"]) == 19
    expressions = [target.get("expr", "") for panel in dashboard["panels"] for target in panel.get("targets", [])]
    assert any("plan_micro_batches" in expression for expression in expressions)
    assert any("state_alignment_verified" in expression for expression in expressions)
    assert any("fp8_metadata_verified" in expression for expression in expressions)
    assert any("failpoint_recovery_p95_seconds" in expression for expression in expressions)
    assert any("checkpoint_cache_hit_ratio" in expression for expression in expressions)
    assert any("recovery_deployment_state" in expression for expression in expressions)
    assert any("recovery_git_revision_match" in expression for expression in expressions)
    assert any("recovery_deployment_fence_rejections" in expression for expression in expressions)
    assert any("checkpoint_durable_fallback_p95_seconds" in expression for expression in expressions)
    print('{"smoke_test":"passed","alerts":18,"recording_rules":10,"dashboard_panels":19}', flush=True)


if __name__ == "__main__":
    main()
