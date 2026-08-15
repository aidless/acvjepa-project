"""Append production recovery/GitOps panels to the versioned Grafana dashboard."""
from __future__ import annotations

import json
from pathlib import Path


PATH = Path(__file__).resolve().parent / "monitoring/grafana_acvjepa_elastic_dashboard.json"


def panel(*, panel_id: int, title: str, x: int, y: int, targets: list[dict], unit: str = "none", thresholds: list[dict] | None = None) -> dict:
    defaults: dict[str, object] = {"unit": unit}
    if thresholds is not None:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
    return {
        "type": "timeseries",
        "title": title,
        "id": panel_id,
        "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}, "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def main() -> None:
    dashboard = json.loads(PATH.read_text(encoding="utf-8"))
    panels = [item for item in dashboard["panels"] if item.get("id") not in {15, 16, 17, 18, 19}]
    selector = 'cluster=~"$cluster",job=~"$job",environment=~"$environment"'
    panels.extend(
        [
            panel(
                panel_id=15,
                title="RecoveryDeploymentEpoch State and Generation",
                x=0,
                y=37,
                targets=[
                    {"expr": f'acvjepa_training_recovery_deployment_state{{{selector}}}', "legendFormat": "state {{state}}", "refId": "A"},
                    {"expr": f'acvjepa_training_recovery_deployment_generation{{{selector}}}', "legendFormat": "generation", "refId": "B"},
                    {"expr": f'acvjepa_training_recovery_deployment_state_age_seconds{{{selector}}}', "legendFormat": "state age (s)", "refId": "C"},
                ],
                unit="none",
            ),
            panel(
                panel_id=16,
                title="Recovery and GitOps Input Binding Gate",
                x=12,
                y=37,
                targets=[
                    {"expr": f'acvjepa_training_recovery_deployment_inputs_valid{{{selector}}}', "legendFormat": "all inputs valid", "refId": "A"},
                    {"expr": f'acvjepa_training_recovery_git_revision_match{{{selector}}}', "legendFormat": "Git revision matches", "refId": "B"},
                ],
                thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}],
            ),
            panel(
                panel_id=17,
                title="Fencing Rejections and GitOps Sync Outcomes",
                x=0,
                y=45,
                targets=[
                    {"expr": f'sum by (actor, reason) (increase(acvjepa_training_recovery_deployment_fence_rejections_total{{{selector}}}[5m]))', "legendFormat": "fenced {{actor}}/{{reason}}", "refId": "A"},
                    {"expr": f'sum by (outcome) (increase(acvjepa_training_gitops_sync_attempts_total{{{selector}}}[15m]))', "legendFormat": "GitOps {{outcome}}", "refId": "B"},
                    {"expr": f'acvjepa_training_gitops_pending_age_seconds{{{selector}}}', "legendFormat": "pending age (s)", "refId": "C"},
                ],
            ),
            panel(
                panel_id=18,
                title="Cache Stampede Protection: Verified Hit Ratio",
                x=12,
                y=45,
                targets=[
                    {"expr": f'job:acvjepa_training_checkpoint_cache_hit_ratio:5m{{{selector}}}', "legendFormat": "{{cache_tier}}/{{component_class}} hit ratio", "refId": "A"},
                ],
                unit="percentunit",
                thresholds=[{"color": "red", "value": None}, {"color": "yellow", "value": 0.85}, {"color": "green", "value": 0.95}],
            ),
            panel(
                panel_id=19,
                title="Cache Stampede Protection: Durable Fallback p95",
                x=0,
                y=53,
                targets=[
                    {"expr": f'job:acvjepa_training_checkpoint_durable_fallback_p95_seconds:5m{{{selector}}}', "legendFormat": "{{component_class}} durable fallback p95", "refId": "A"},
                    {"expr": f'sum by (cache_tier, component_class) (increase(acvjepa_training_checkpoint_cache_fetches_total{{{selector},outcome="integrity_failed"}}[5m]))', "legendFormat": "{{cache_tier}}/{{component_class}} integrity failures", "refId": "B"},
                ],
                unit="s",
                thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 2}, {"color": "red", "value": 5}],
            ),
        ]
    )
    dashboard["panels"] = panels
    dashboard["version"] = int(dashboard.get("version", 1)) + 1
    dashboard["tags"] = sorted(set(dashboard.get("tags", [])) | {"gitops", "recovery-epoch", "cache-slo"})
    PATH.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print('{"dashboard_update":"passed","panels":19}', flush=True)


if __name__ == "__main__":
    main()
