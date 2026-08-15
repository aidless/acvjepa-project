"""Static safety checks for the offline failpoint CI workflow."""
from __future__ import annotations

import re
from pathlib import Path

import yaml


FORBIDDEN = ("pull_request_target", "kubectl", "helm ", "terraform", "iptables", " tc ", "ssh ", "rdma", "torchrun", "nccl")
SHA_ACTION = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def run_check(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    # YAML 1.1 parser can coerce `on` to bool, so use raw text for trigger check.
    assert "pull_request_target" not in raw
    workflow = yaml.safe_load(raw)
    assert workflow["permissions"] == {"contents": "read"}
    assert "secrets" not in raw.lower()
    for forbidden in FORBIDDEN[1:]:
        assert forbidden not in raw.lower(), forbidden
    jobs = workflow["jobs"]
    job = jobs["offline-contract"]
    assert job["timeout-minutes"] <= 10
    steps = job["steps"]
    action_steps = [step for step in steps if "uses" in step]
    assert action_steps and all(SHA_ACTION.match(step["uses"]) for step in action_steps)
    uploads = [step for step in action_steps if step["uses"].startswith("actions/upload-artifact@")]
    assert len(uploads) == 1
    upload = uploads[0]["with"]
    assert upload["path"] == "artifacts/" and upload["include-hidden-files"] is False
    assert int(upload["retention-days"]) <= 14
    run_text = "\n".join(str(step.get("run", "")) for step in steps).lower()
    assert "scripts/run_offline_chaos_contract.sh" in run_text
    assert "threadsafe_checkpoint_load_gate.py" in run_text
    assert "validate_kubernetes_chaos_lab.py" in run_text
    print('{"smoke_test":"passed","ci_permissions":"contents:read","artifact_uploads":1}', flush=True)


if __name__ == "__main__":
    run_check(Path(__file__).parent / ".github/workflows/failpoint-chaos-ci.yml")
