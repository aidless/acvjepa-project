"""Render a safe isolated-chaos Job from the checked-in template.

This is a CI artifact-generation step only. It does not load kubeconfig, run
kubectl, contact a registry, or apply a Kubernetes resource.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "k8s/chaos-lab/offline-chaos-contract.yaml"
SENTINEL = "registry.example.invalid/acvjepa-chaos-contract@sha256:" + "0" * 64
IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9./_-]*/acvjepa-chaos-contract@sha256:[0-9a-f]{64}$")


def render(*, image: str, output: Path) -> None:
    if not IMAGE_RE.fullmatch(image):
        raise ValueError("image must be an immutable acvjepa-chaos-contract @sha256 digest")
    raw = TEMPLATE.read_text(encoding="utf-8")
    if raw.count(SENTINEL) != 1:
        raise RuntimeError("template must contain exactly one image sentinel")
    rendered = raw.replace(SENTINEL, image)
    docs = list(yaml.safe_load_all(rendered))
    job = docs[-1]
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == image
    assert "offline-logical-failpoints-only" in rendered
    assert "privileged:" not in rendered and "hostNetwork:" not in rendered
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render isolated Kubernetes chaos contract manifest")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(image=args.image, output=args.output)
    print('{"render":"passed","cluster_operation":"none"}', flush=True)


if __name__ == "__main__":
    main()
