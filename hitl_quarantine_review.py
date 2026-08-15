"""Human-in-the-loop review ledger for quarantined soft-manipulation cases.

The module records review evidence and creates approved *data correction patches*.
It cannot emit robot actions, relax safety thresholds, train a model, or promote a
release. High-risk soft/contact cases require a second reviewer before the patch
can enter a SimJob/curated data manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    NEEDS_SECOND_REVIEW = "NEEDS_SECOND_REVIEW"
    APPROVED_FOR_DATA = "APPROVED_FOR_DATA"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


@dataclass(frozen=True)
class QuarantinedCase:
    case_id: str
    episode_commit_uri: str
    evidence_sha256: str
    reason: str
    risk_level: str
    object_class: str
    task_template: str


@dataclass(frozen=True)
class CorrectionPatch:
    case_id: str
    reviewer_id: str
    action: str  # e.g. correct_event, approve_physics_prior, reject_data
    corrected_events: Dict[str, int]
    physics_prior_patch: Dict[str, float]
    simulation_focus: Dict[str, str]
    rationale: str


class ReviewLedger:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                status TEXT NOT NULL, first_reviewer TEXT, first_patch_json TEXT,
                second_reviewer TEXT, second_patch_json TEXT, updated_ns INTEGER NOT NULL
            )"""
        )

    def register(self, case: QuarantinedCase) -> None:
        now = time.time_ns()
        encoded = json.dumps(asdict(case), sort_keys=True)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute("SELECT payload_json FROM cases WHERE case_id=?", (case.case_id,)).fetchone()
            if old is None:
                self.conn.execute("INSERT INTO cases VALUES(?,?,?,?,?,?,?,?)", (case.case_id, encoded, ReviewStatus.PENDING.value, None, None, None, None, now))
            elif old["payload_json"] != encoded:
                raise RuntimeError("case ID collides with different evidence")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def submit_first_review(self, patch: CorrectionPatch) -> ReviewStatus:
        row = self.conn.execute("SELECT * FROM cases WHERE case_id=?", (patch.case_id,)).fetchone()
        if row is None:
            raise KeyError("unknown case")
        if row["status"] not in (ReviewStatus.PENDING.value, ReviewStatus.CLAIMED.value):
            raise RuntimeError(f"case is not reviewable: {row['status']}")
        if not patch.rationale.strip():
            raise ValueError("review rationale required")
        case = json.loads(row["payload_json"])
        # Never auto-approve an explicit hardware/sensor safety issue as training data.
        status = ReviewStatus.ESCALATED if case["reason"] in {"hardware_fault", "sensor_contract_violation"} else ReviewStatus.NEEDS_SECOND_REVIEW
        self.conn.execute(
            "UPDATE cases SET status=?, first_reviewer=?, first_patch_json=?, updated_ns=? WHERE case_id=?",
            (status.value, patch.reviewer_id, json.dumps(asdict(patch), sort_keys=True), time.time_ns(), patch.case_id),
        )
        return status

    def submit_second_review(self, patch: CorrectionPatch, approve: bool) -> ReviewStatus:
        row = self.conn.execute("SELECT * FROM cases WHERE case_id=?", (patch.case_id,)).fetchone()
        if row is None or row["status"] != ReviewStatus.NEEDS_SECOND_REVIEW.value:
            raise RuntimeError("case is not awaiting second review")
        if patch.reviewer_id == row["first_reviewer"]:
            raise ValueError("second reviewer must be independent")
        status = ReviewStatus.APPROVED_FOR_DATA if approve else ReviewStatus.REJECTED
        self.conn.execute(
            "UPDATE cases SET status=?, second_reviewer=?, second_patch_json=?, updated_ns=? WHERE case_id=?",
            (status.value, patch.reviewer_id, json.dumps(asdict(patch), sort_keys=True), time.time_ns(), patch.case_id),
        )
        return status

    def approved_data_patch(self, case_id: str) -> Dict:
        row = self.conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if row is None or row["status"] != ReviewStatus.APPROVED_FOR_DATA.value:
            raise RuntimeError("case is not approved for data generation")
        first, second = json.loads(row["first_patch_json"]), json.loads(row["second_patch_json"])
        return {
            "patch_version": "hitl-soft-grasp-v1",
            "case_id": case_id,
            "source_evidence": json.loads(row["payload_json"]),
            "first_review": first,
            "second_review": second,
            "review_hash": hashlib.sha256((row["first_patch_json"] + row["second_patch_json"]).encode()).hexdigest(),
            "allowed_downstream": ["simjob_compiler", "curated_dataset_manifest"],
            "forbidden_downstream": ["robot_control", "safety_threshold_change", "direct_production_deploy"],
        }


if __name__ == "__main__":
    # Use a fresh temp ledger per demo run so re-runs are idempotent on any OS.
    path = os.path.join(tempfile.gettempdir(), f"hitl_soft_grasp_demo_{uuid4().hex}.sqlite")
    ledger = ReviewLedger(path)
    case = QuarantinedCase("case-001", "file://episode-commit", "evidence-hash", "soft_physics_gap", "high", "cloth", "reposition")
    ledger.register(case)
    patch_a = CorrectionPatch("case-001", "reviewer-a", "approve_physics_prior", {"slip_onset": 7}, {"young_modulus": 700.0}, {"geometry": "wrinkled"}, "Contact onset was late; use approved wrinkled-cloth stratum.")
    patch_b = CorrectionPatch("case-001", "reviewer-b", "confirm", {"slip_onset": 7}, {"young_modulus": 700.0}, {"geometry": "wrinkled"}, "Evidence and proposed bounded prior agree.")
    print(ledger.submit_first_review(patch_a).value)
    print(ledger.submit_second_review(patch_b, approve=True).value)
    print(json.dumps(ledger.approved_data_patch("case-001"), indent=2))
