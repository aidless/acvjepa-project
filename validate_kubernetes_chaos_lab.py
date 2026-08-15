"""Static safety checks for the isolated Kubernetes offline-chaos Job template."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "k8s/chaos-lab/offline-chaos-contract.yaml"


def main() -> None:
    raw = MANIFEST.read_text(encoding="utf-8")
    docs = list(yaml.safe_load_all(raw))
    namespace, service_account, network_policy, job = docs
    assert namespace["kind"] == "Namespace"
    assert namespace["metadata"]["name"] == "acvjepa-chaos-lab"
    labels = namespace["metadata"]["labels"]
    assert labels["pod-security.kubernetes.io/enforce"] == "restricted"
    assert service_account["kind"] == "ServiceAccount" and service_account["automountServiceAccountToken"] is False
    assert network_policy["kind"] == "NetworkPolicy" and network_policy["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert network_policy["spec"]["podSelector"] == {}
    assert job["kind"] == "Job" and job["metadata"]["generateName"] == "offline-chaos-contract-"
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never" and pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    container = pod["containers"][0]
    context = container["securityContext"]
    assert context["allowPrivilegeEscalation"] is False
    assert context["readOnlyRootFilesystem"] is True
    assert context["capabilities"]["drop"] == ["ALL"]
    assert "0000000000000000000000000000000000000000000000000000000000000000" in container["image"]
    forbidden = ("privileged:", "hostnetwork:", "hostpid:", "hostpath:", "nodeloss", "rdma", "nccl", "kubectl delete", "pods/exec", "secrets")
    assert not any(token in raw.lower() for token in forbidden)
    assert pod["containers"][0]["env"][0]["value"] == "offline-logical-failpoints-only"
    print('{"smoke_test":"passed","kubernetes_objects":4,"pod_security":"restricted","service_account_token":"disabled"}', flush=True)


if __name__ == "__main__":
    main()
