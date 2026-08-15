"""Static safety checks for Kubernetes chaos-contract artifact workflow."""
from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / ".github/workflows/kubernetes-chaos-contract.yml"
SHA_ACTION = re.compile(r"^[^@]+@[0-9a-f]{40}$")
FORBIDDEN = ("kubectl", "helm", "kubeconfig", "secrets", "id-token", "pull_request_target", "hostnetwork", "privileged", "terraform", "ssh ")


def main() -> None:
    raw = WORKFLOW.read_text(encoding="utf-8")
    assert not any(token in raw.lower() for token in FORBIDDEN)
    workflow = yaml.safe_load(raw)
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert set(jobs) == {"offline-contract", "render-approved-manifest"}
    offline = jobs["offline-contract"]
    render = jobs["render-approved-manifest"]
    assert offline["timeout-minutes"] <= 10 and render["timeout-minutes"] <= 5
    assert render["if"] == "github.event_name == 'workflow_dispatch'"
    assert render["needs"] == "offline-contract"
    assert render["environment"]["name"] == "chaos-lab-manifest-approval"
    all_steps = offline["steps"] + render["steps"]
    actions = [step["uses"] for step in all_steps if "uses" in step]
    assert actions and all(SHA_ACTION.match(item) for item in actions)
    render_text = "\n".join(str(step.get("run", "")) for step in render["steps"])
    assert "render_kubernetes_chaos_contract.py" in render_text
    assert '"$CHAOS_CONTRACT_IMAGE"' in render_text
    assert "scripts/run_offline_chaos_contract.sh" in "\n".join(str(step.get("run", "")) for step in offline["steps"])
    print('{"smoke_test":"passed","workflow":"artifact_only","permissions":"contents:read","cluster_operations":0}', flush=True)


if __name__ == "__main__":
    main()
