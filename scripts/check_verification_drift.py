#!/usr/bin/env python3
"""Fail when verification claims drift from the tree.

Counts come from the scripts and test sources, not from prose:

- ``verify_all.ps1`` ``Run-Check`` names and inline ``PASS`` names, by section
- ``python`` lines in ``scripts/run_offline_chaos_contract.sh``
- ``def test_`` methods in the repository
- absolute author-machine paths

``verification_manifest.json`` must match this computation. README phrases
must quote the computed counts. A pass fraction such as ``59/59`` is rejected
in the current-claim surfaces listed below.

Usage:
    python scripts/check_verification_drift.py
    python scripts/check_verification_drift.py --write
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "verification_manifest.json"
PS1 = ROOT / "verify_all.ps1"
CONTRACT = ROOT / "scripts" / "run_offline_chaos_contract.sh"

TEXT_SUFFIXES = {".py", ".md", ".sh", ".ps1", ".yml", ".yaml", ".json", ".txt", ".toml"}
SKIP_DIRS = {".git", "verify_artifacts", "__pycache__", ".venv", "venv"}

SECTION_TIER = {
    "A": "declared_requirements",
    "B": "declared_requirements",
    "C": "declared_requirements",
    "D": "declared_requirements",
    "E": "declared_requirements",
    "F": "optional_docker",
    "G": "committed_contract_wrapper",
    "H": "optional_weights",
    "I": "declared_requirements",
    "J": "optional_weights",
    "K": "optional_cuda_and_weights",
    "L": "optional_cuda_and_weights",
}

# Pieces are concatenated so this source file does not contain the path literals it forbids.
FORBIDDEN_PATH = (
    re.compile("/ho" + "me/"),
    re.compile("/Us" + "ers/"),
    re.compile("/ro" + "ot/"),
    re.compile("lecun_" + "analysis"),
    re.compile("file:///" + "[A-Za-z]:"),
    re.compile("[A-Za-z]:" + "\\\\"),
    re.compile(r"\bE: " + "zip" + r"\b"),
)

RUN_CHECK = re.compile(r"Run-Check '([^']+)'")
INLINE_PASS = re.compile(r'"PASS`t([^`]+)`')
SECTION = re.compile(r"^# --- ([A-Z])\.")
TEST_DEF = re.compile(r"^[ \t]+def test_[A-Za-z0-9_]+\(", re.M)
UNITTEST_MODULE = re.compile(r"-m unittest\b.*?(\S+\.py)")
# Column-0 `run_py` invocations are the contract commands. The helper itself
# calls `python`, and that inner line must not be counted again.
PYTHON_CMD = re.compile(r"^run_py ")

# Current-claim surfaces. Historical logs may still say 59/59 below a marker.
FRACTION_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "PROJECT_PLAN.md",
)
HISTORICAL_MARKER = "<!-- historical-log -->"
PASS_FRACTION = re.compile(r"\b\d+/\d+\b")
BADGE = re.compile(r"badge\.svg|shields\.io|img\.shields\.io", re.I)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_verify_all(text: str) -> list[dict[str, str]]:
    section = None
    found: list[dict[str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        mark = SECTION.match(line)
        if mark:
            section = mark.group(1)
            if section not in SECTION_TIER:
                raise SystemExit(f"verify_all.ps1:{lineno}: unknown section {section}")
        names = [(name, "Run-Check") for name in RUN_CHECK.findall(line)]
        names += [(name, "inline_pass") for name in INLINE_PASS.findall(line)]
        for name, kind in names:
            if section is None:
                raise SystemExit(f"verify_all.ps1:{lineno}: check {name} before a section")
            found.append(
                {
                    "name": name,
                    "section": section,
                    "tier": SECTION_TIER[section],
                    "kind": kind,
                }
            )
    names = [item["name"] for item in found]
    dupes = sorted({name for name in names if names.count(name) > 1})
    if dupes:
        raise SystemExit(f"duplicate verify_all.ps1 check names: {dupes}")
    return found


def contract_python_commands(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if PYTHON_CMD.match(line.strip())]


def unittest_methods(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        n = len(TEST_DEF.findall(text))
        if n:
            counts[path.relative_to(root).as_posix()] = n
    return counts


def contract_unittest_methods(commands: list[str], methods: dict[str, int]) -> dict[str, int]:
    covered: dict[str, int] = {}
    for command in commands:
        match = UNITTEST_MODULE.search(command)
        if not match:
            continue
        name = match.group(1)
        if name not in methods:
            raise SystemExit(f"contract runs {name} but it has no def test_ methods")
        covered[name] = methods[name]
    return covered


def compute() -> dict:
    checks = parse_verify_all(_read(PS1))
    commands = contract_python_commands(_read(CONTRACT))
    methods = unittest_methods(ROOT)
    covered = contract_unittest_methods(commands, methods)
    by_tier: dict[str, int] = {}
    for item in checks:
        by_tier[item["tier"]] = by_tier.get(item["tier"], 0) + 1
    not_in_contract = {
        name: count for name, count in methods.items() if name not in covered
    }
    return {
        "committed_contract_python_commands": len(commands),
        "committed_contract_script": "scripts/run_offline_chaos_contract.sh",
        "fresh_clone_entry": "scripts/verify_committed.sh",
        "historical_host_log": {
            "date": "2026-08-15",
            "claimed_passes": 59,
            "rerun_for_this_claim": False,
            "note": (
                "One Windows laptop with weights, CUDA, and Docker present. "
                "Not the fresh-clone contract. Optional groups SKIP when those "
                "inputs are absent. This commit does not re-run that log."
            ),
        },
        "unittest_methods_in_committed_contract": sum(covered.values()),
        "unittest_methods_in_tree": sum(methods.values()),
        "unittest_methods_not_in_committed_contract": not_in_contract,
        "unittest_methods_run_by_contract": covered,
        "verify_all_by_tier": by_tier,
        "verify_all_checks": checks,
        "verify_all_named_checks": len(checks),
        "verify_all_script": "verify_all.ps1",
    }


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def check_manifest(payload: dict, errors: list[str]) -> None:
    if not MANIFEST.is_file():
        errors.append("verification_manifest.json is missing (run with --write)")
        return
    on_disk = MANIFEST.read_text(encoding="utf-8")
    expected = canonical(payload)
    if on_disk != expected:
        errors.append(
            "verification_manifest.json does not match the tree. "
            "Run: python scripts/check_verification_drift.py --write"
        )


def _between_problem(text: str) -> list[str]:
    start = "<!-- problem-start -->"
    end = "<!-- problem-end -->"
    if start not in text or end not in text:
        raise ValueError("README.md is missing problem markers")
    body = text.split(start, 1)[1].split(end, 1)[0]
    return [line for line in body.splitlines() if line.strip()]


def before_historical(text: str, label: str) -> str:
    if HISTORICAL_MARKER not in text:
        raise ValueError(f"{label} is missing {HISTORICAL_MARKER}")
    return text.split(HISTORICAL_MARKER, 1)[0]


def check_prose(payload: dict, errors: list[str]) -> None:
    readme = _read(ROOT / "README.md")
    try:
        problem = _between_problem(readme)
    except ValueError as exc:
        errors.append(str(exc))
        problem = []
    if len(problem) > 3:
        errors.append(f"README problem statement has {len(problem)} lines; maximum is 3")
    if len(problem) < 1:
        errors.append("README problem statement is empty")
    for phrase in (
        "**Science status:**",
        "**Infrastructure status:**",
        "STATUS: WORK-IN-PROGRESS",
        f"committed contract commands: {payload['committed_contract_python_commands']}",
        f"unittest methods in tree: {payload['unittest_methods_in_tree']}",
        f"unittest methods run by that contract: {payload['unittest_methods_in_committed_contract']}",
        f"verify_all.ps1 named checks: {payload['verify_all_named_checks']}",
    ):
        if phrase not in readme:
            errors.append(f"README.md is missing exact phrase: {phrase}")
    if BADGE.search(readme):
        errors.append("README.md contains a badge image")
    if PASS_FRACTION.search(readme):
        errors.append("README.md contains a pass fraction (N/N). Cite the manifest counts instead.")

    for rel in FRACTION_FILES:
        if rel == "README.md":
            continue
        text = _read(ROOT / rel)
        if PASS_FRACTION.search(text):
            errors.append(f"{rel} contains a pass fraction (N/N)")
        if BADGE.search(text):
            errors.append(f"{rel} contains a badge image")

    for rel in ("STATUS.md", "VERIFY_RESULTS.md"):
        text = _read(ROOT / rel)
        try:
            head = before_historical(text, rel)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if PASS_FRACTION.search(head):
            errors.append(f"{rel} current claim contains a pass fraction (N/N)")
        if BADGE.search(head):
            errors.append(f"{rel} current claim contains a badge image")
        if rel == "STATUS.md" and "STATUS: WORK-IN-PROGRESS" not in head:
            errors.append("STATUS.md current claim must contain STATUS: WORK-IN-PROGRESS")
        if rel == "VERIFY_RESULTS.md" and "verification_manifest.json" not in head:
            errors.append("VERIFY_RESULTS.md current claim must point at verification_manifest.json")

    if "FAILURE_MODES.md" not in readme:
        errors.append("README.md does not link FAILURE_MODES.md")


def check_author_paths(errors: list[str]) -> None:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in FORBIDDEN_PATH:
                if pattern.search(line):
                    rel = path.relative_to(ROOT).as_posix()
                    errors.append(f"{rel}:{lineno}: author-machine path matches {pattern.pattern}")
                    break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite verification_manifest.json")
    args = parser.parse_args()
    payload = compute()
    if args.write:
        MANIFEST.write_text(canonical(payload), encoding="utf-8")
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
    errors: list[str] = []
    check_manifest(payload, errors)
    check_prose(payload, errors)
    check_author_paths(errors)
    if errors:
        print("verification drift:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(
        "drift ok: "
        f"contract_commands={payload['committed_contract_python_commands']} "
        f"unittest_tree={payload['unittest_methods_in_tree']} "
        f"unittest_contract={payload['unittest_methods_in_committed_contract']} "
        f"verify_all_named={payload['verify_all_named_checks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
