"""Tamper-evident multimodal HITL audit ledger for quarantined robot data.

This research/reference module stores content hashes and signed review events;
large RGB-D, point-cloud, action and contact artifacts remain in access-controlled
object storage. It is *not* a robot-control authority, safety-policy store or
model-release mechanism. A verified ledger means modifications are detectable
under the stated threat model, not that a compromised administrator cannot delete
both the database and every external anchor.

Prototype cryptography
----------------------
- SHA-256 content addressing and a deterministic binary Merkle tree bind all
  modality manifests to an evidence_root.
- Every event is Ed25519-signed by an enrolled actor and linked to the prior
  event hash. The SQLite file is a replicated projection, not the trust root.
- Periodic anchors sign a chain head. Production must additionally publish the
  anchor to an independent append-only/WORM system or transparency service; an
  anchor kept only in the same SQLite file cannot reveal total-file deletion.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import unquote, urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature


GENESIS_HASH = "0" * 64
SCHEMA_VERSION = "multimodal-hitl-ledger-v1"
REQUIRED_MODALITIES = frozenset(
    {"rgbd_video", "pointcloud", "robot_proprio", "executed_actions", "contact_events", "calibration"}
)
ROLE_ALLOWED_EVENTS = {
    "capture_gateway": {"EVIDENCE_REGISTERED"},
    "validator": {"VALIDATION_ATTESTED"},
    "reviewer": {"REVIEW_SUBMITTED", "REVIEW_CONFIRMED", "REVIEW_REJECTED"},
    "anchor": set(),  # anchors are a separate signed table, never review events.
}


class LedgerIntegrityError(RuntimeError):
    """Raised when a ledger, signature, hash, validator report or anchor fails verification."""


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    """Stable JSON encoding used by all content hashes and signatures."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def merkle_root(hex_leaves: Sequence[str]) -> str:
    """Compute an order-independent-by-manifest deterministic binary Merkle root."""

    if not hex_leaves:
        raise ValueError("Merkle tree requires at least one leaf")
    nodes = [bytes.fromhex(value) for value in sorted(hex_leaves)]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [hashlib.sha256(nodes[index] + nodes[index + 1]).digest() for index in range(0, len(nodes), 2)]
    return nodes[0].hex()


@dataclass(frozen=True)
class EvidenceArtifact:
    modality: str
    uri: str
    content_sha256: str
    byte_length: int
    start_ns: int
    end_ns: int
    schema_version: str
    producer_id: str

    def leaf_hash(self) -> str:
        # URI, physical content, clock interval, producer and parsing contract are
        # bound together. A same-byte artifact at a different unreviewed location
        # therefore cannot be silently swapped into the evidence package.
        return sha256_bytes(canonical_bytes(asdict(self)))

    def validate(self) -> None:
        if not self.modality or not self.uri or not self.producer_id or not self.schema_version:
            raise ValueError("artifact modality, URI, producer and schema are required")
        if len(self.content_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.content_sha256):
            raise ValueError("artifact content_sha256 must be lowercase SHA-256")
        if self.byte_length < 0 or self.start_ns > self.end_ns:
            raise ValueError("artifact length or timestamp interval is invalid")


@dataclass(frozen=True)
class EvidenceManifest:
    case_id: str
    episode_commit_uri: str
    capture_session_id: str
    action_schema_version: str
    preprocess_version: str
    artifacts: tuple[EvidenceArtifact, ...]

    def validate_multimodal_contract(self, *, tolerance_ns: int = 1_000_000_000) -> None:
        if not self.case_id or not self.capture_session_id or not self.episode_commit_uri:
            raise ValueError("case, session and immutable episode commit URI are required")
        if not self.action_schema_version or not self.preprocess_version:
            raise ValueError("action and preprocessing schema versions are required")
        seen = {artifact.modality for artifact in self.artifacts}
        missing = REQUIRED_MODALITIES - seen
        duplicates = len(seen) != len(self.artifacts)
        if missing or duplicates:
            raise ValueError(f"multimodal evidence requires exactly one of each modality; missing={sorted(missing)}")
        for artifact in self.artifacts:
            artifact.validate()
        start = max(artifact.start_ns for artifact in self.artifacts if artifact.modality != "calibration")
        end = min(artifact.end_ns for artifact in self.artifacts if artifact.modality != "calibration")
        if start > end + tolerance_ns:
            raise ValueError("modalities fail the bounded temporal-overlap contract")

    @property
    def evidence_root(self) -> str:
        self.validate_multimodal_contract()
        return merkle_root([artifact.leaf_hash() for artifact in self.artifacts])

    def canonical_payload(self) -> Dict[str, Any]:
        # Sorting makes the manifest stable even if ingestion sees modality files in
        # a different order. Keep human-visible URIs outside an immutable blob store.
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "episode_commit_uri": self.episode_commit_uri,
            "capture_session_id": self.capture_session_id,
            "action_schema_version": self.action_schema_version,
            "preprocess_version": self.preprocess_version,
            "artifacts": [asdict(item) for item in sorted(self.artifacts, key=lambda item: item.modality)],
        }

    @classmethod
    def from_local_files(
        cls,
        *,
        case_id: str,
        episode_commit_uri: str,
        capture_session_id: str,
        action_schema_version: str,
        preprocess_version: str,
        local_modalities: Mapping[str, Path],
        start_ns: int,
        end_ns: int,
        producer_id: str,
        schema_version: str = "artifact-v1",
    ) -> "EvidenceManifest":
        artifacts = []
        for modality, path in local_modalities.items():
            digest, size = sha256_file(path)
            artifacts.append(
                EvidenceArtifact(
                    modality=modality,
                    uri=path.resolve().as_uri(),
                    content_sha256=digest,
                    byte_length=size,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    schema_version=schema_version,
                    producer_id=producer_id,
                )
            )
        return cls(
            case_id=case_id,
            episode_commit_uri=episode_commit_uri,
            capture_session_id=capture_session_id,
            action_schema_version=action_schema_version,
            preprocess_version=preprocess_version,
            artifacts=tuple(artifacts),
        )


class Ed25519Signer:
    """Small signer wrapper; production private keys belong in KMS/HSM, not SQLite."""

    def __init__(self, private_key: Ed25519PrivateKey | None = None):
        self._private = private_key or Ed25519PrivateKey.generate()

    def sign(self, message: bytes) -> str:
        return base64.b64encode(self._private.sign(message)).decode("ascii")

    def public_key_b64(self) -> str:
        raw = self._private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw).decode("ascii")


def _verify_signature(public_key_b64: str, message: bytes, signature_b64: str) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64.encode("ascii"), validate=True))
        key.verify(base64.b64decode(signature_b64.encode("ascii"), validate=True), message)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


class MultimodalAuditLedger:
    """Append-safe SQLite projection with verified evidence and review provenance."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS identities (
                signer_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                public_key_b64 TEXT NOT NULL,
                enrolled_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
                case_id TEXT PRIMARY KEY,
                manifest_json TEXT NOT NULL,
                evidence_root TEXT NOT NULL,
                registered_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES evidence(case_id),
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                evidence_root TEXT NOT NULL,
                signer_id TEXT NOT NULL REFERENCES identities(signer_id),
                created_ns INTEGER NOT NULL,
                previous_event_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE,
                signature_b64 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS anchors (
                anchor_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence INTEGER NOT NULL,
                event_hash TEXT NOT NULL,
                signer_id TEXT NOT NULL REFERENCES identities(signer_id),
                created_ns INTEGER NOT NULL,
                signature_b64 TEXT NOT NULL,
                UNIQUE(sequence, event_hash, signer_id)
            );
            """
        )

    def enroll_identity(self, *, signer_id: str, role: str, public_key_b64: str) -> None:
        if role not in ROLE_ALLOWED_EVENTS:
            raise ValueError(f"unknown ledger role: {role}")
        # Validate the key before persisting it.
        Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64.encode("ascii"), validate=True))
        existing = self.conn.execute("SELECT role, public_key_b64 FROM identities WHERE signer_id=?", (signer_id,)).fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO identities VALUES(?,?,?,?)", (signer_id, role, public_key_b64, time.time_ns())
            )
        elif existing["role"] != role or existing["public_key_b64"] != public_key_b64:
            raise LedgerIntegrityError("identity collision with a different role or public key")

    def register_evidence(self, manifest: EvidenceManifest, *, signer_id: str, signer: Ed25519Signer) -> str:
        manifest.validate_multimodal_contract()
        payload = manifest.canonical_payload()
        root = manifest.evidence_root
        encoded = canonical_bytes(payload).decode("utf-8")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute("SELECT manifest_json, evidence_root FROM evidence WHERE case_id=?", (manifest.case_id,)).fetchone()
            if old is None:
                self.conn.execute(
                    "INSERT INTO evidence VALUES(?,?,?,?)", (manifest.case_id, encoded, root, time.time_ns())
                )
            elif old["manifest_json"] != encoded or old["evidence_root"] != root:
                raise LedgerIntegrityError("case ID collides with a different multimodal evidence package")
            event_hash = self._append_event_in_tx(
                case_id=manifest.case_id,
                event_type="EVIDENCE_REGISTERED",
                payload={"manifest_sha256": sha256_bytes(encoded.encode("utf-8"))},
                evidence_root=root,
                signer_id=signer_id,
                signer=signer,
            )
            self.conn.execute("COMMIT")
            return event_hash
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def attest_validation(
        self,
        *,
        case_id: str,
        signer_id: str,
        signer: Ed25519Signer,
        validator_version: str,
        report_sha256: str,
        checks: Mapping[str, bool],
    ) -> str:
        # Perception/model checks are valuable tamper/poisoning signals, but no
        # classifier can prove an input is non-adversarial. We record versions,
        # report hash and pass/fail facts for later independent re-evaluation.
        if not validator_version or len(report_sha256) != 64 or not all(isinstance(result, bool) for result in checks.values()):
            raise ValueError("validator version, SHA-256 report hash and boolean checks are required")
        if not checks or not all(checks.values()):
            raise LedgerIntegrityError("failed multimodal validation keeps the case quarantined")
        return self.append_event(
            case_id=case_id,
            event_type="VALIDATION_ATTESTED",
            payload={
                "validator_version": validator_version,
                "report_sha256": report_sha256,
                "checks": dict(sorted(checks.items())),
            },
            signer_id=signer_id,
            signer=signer,
        )

    def submit_review(
        self,
        *,
        case_id: str,
        signer_id: str,
        signer: Ed25519Signer,
        patch_sha256: str,
        decision: str,
        rationale_sha256: str,
        confirms_existing_review: bool = False,
    ) -> str:
        if decision not in {"approve_for_data", "reject", "escalate"}:
            raise ValueError("review decision is not permitted")
        for name, digest in {"patch": patch_sha256, "rationale": rationale_sha256}.items():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{name} hash must be lowercase SHA-256")
        return self.append_event(
            case_id=case_id,
            event_type="REVIEW_CONFIRMED" if confirms_existing_review else "REVIEW_SUBMITTED",
            payload={
                "patch_sha256": patch_sha256,
                "decision": decision,
                "rationale_sha256": rationale_sha256,
            },
            signer_id=signer_id,
            signer=signer,
        )

    def append_event(
        self,
        *,
        case_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        signer_id: str,
        signer: Ed25519Signer,
    ) -> str:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            evidence = self.conn.execute("SELECT evidence_root FROM evidence WHERE case_id=?", (case_id,)).fetchone()
            if evidence is None:
                raise KeyError("evidence must be registered before any review event")
            event_hash = self._append_event_in_tx(
                case_id=case_id,
                event_type=event_type,
                payload=payload,
                evidence_root=evidence["evidence_root"],
                signer_id=signer_id,
                signer=signer,
            )
            self.conn.execute("COMMIT")
            return event_hash
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _append_event_in_tx(
        self,
        *,
        case_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        evidence_root: str,
        signer_id: str,
        signer: Ed25519Signer,
    ) -> str:
        identity = self.conn.execute("SELECT role, public_key_b64 FROM identities WHERE signer_id=?", (signer_id,)).fetchone()
        if identity is None:
            raise PermissionError("signer is not enrolled")
        if event_type not in ROLE_ALLOWED_EVENTS[identity["role"]]:
            raise PermissionError(f"role {identity['role']} cannot append event {event_type}")
        if identity["public_key_b64"] != signer.public_key_b64():
            raise PermissionError("signing key does not match the enrolled identity")

        latest = self.conn.execute("SELECT sequence, event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        sequence = 1 if latest is None else int(latest["sequence"]) + 1
        previous = GENESIS_HASH if latest is None else latest["event_hash"]
        created_ns = time.time_ns()
        payload_json = canonical_bytes(dict(payload)).decode("utf-8")
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "case_id": case_id,
            "event_type": event_type,
            "payload_json": payload_json,
            "evidence_root": evidence_root,
            "signer_id": signer_id,
            "created_ns": created_ns,
            "previous_event_hash": previous,
        }
        event_hash = sha256_bytes(canonical_bytes(unsigned))
        signature_b64 = signer.sign(bytes.fromhex(event_hash))
        self.conn.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
            (sequence, case_id, event_type, payload_json, evidence_root, signer_id, created_ns, previous, event_hash, signature_b64),
        )
        return event_hash

    def create_anchor(self, *, signer_id: str, signer: Ed25519Signer) -> Dict[str, Any]:
        """Sign the current chain head for replication to an independent anchor store."""

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            identity = self.conn.execute("SELECT role, public_key_b64 FROM identities WHERE signer_id=?", (signer_id,)).fetchone()
            latest = self.conn.execute("SELECT sequence, event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
            if identity is None or identity["role"] != "anchor":
                raise PermissionError("only an enrolled anchor identity may anchor a chain head")
            if identity["public_key_b64"] != signer.public_key_b64():
                raise PermissionError("anchor signing key mismatch")
            if latest is None:
                raise LedgerIntegrityError("cannot anchor an empty event chain")
            created_ns = time.time_ns()
            message = canonical_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": int(latest["sequence"]),
                    "event_hash": latest["event_hash"],
                    "created_ns": created_ns,
                }
            )
            signature = signer.sign(message)
            self.conn.execute(
                "INSERT INTO anchors(sequence,event_hash,signer_id,created_ns,signature_b64) VALUES(?,?,?,?,?)",
                (int(latest["sequence"]), latest["event_hash"], signer_id, created_ns, signature),
            )
            self.conn.execute("COMMIT")
            return {"sequence": int(latest["sequence"]), "event_hash": latest["event_hash"], "created_ns": created_ns, "signature_b64": signature}
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def eligible_patch_for_data(self, *, case_id: str, patch_sha256: str) -> bool:
        """Return eligibility only; downstream still requires independent release gates.

        Eligibility requires one successful validator attestation and two distinct
        reviewers approving the same immutable patch. It never invokes training,
        changes safety thresholds or promotes a robot model.
        """

        validation = self.conn.execute(
            "SELECT 1 FROM events WHERE case_id=? AND event_type='VALIDATION_ATTESTED' LIMIT 1", (case_id,)
        ).fetchone()
        rows = self.conn.execute(
            "SELECT signer_id,payload_json FROM events WHERE case_id=? AND event_type IN ('REVIEW_SUBMITTED','REVIEW_CONFIRMED') ORDER BY sequence",
            (case_id,),
        ).fetchall()
        approvals = {
            row["signer_id"]
            for row in rows
            if (payload := json.loads(row["payload_json"])).get("patch_sha256") == patch_sha256
            and payload.get("decision") == "approve_for_data"
        }
        return validation is not None and len(approvals) >= 2

    def verify_integrity(self) -> Dict[str, int]:
        """Verify chain linkage, signatures, evidence roots and local anchors."""

        identities = {
            row["signer_id"]: row
            for row in self.conn.execute("SELECT signer_id,role,public_key_b64 FROM identities")
        }
        evidence = {row["case_id"]: row for row in self.conn.execute("SELECT case_id,manifest_json,evidence_root FROM evidence")}
        for case_id, row in evidence.items():
            manifest = _manifest_from_payload(json.loads(row["manifest_json"]))
            if manifest.case_id != case_id or manifest.evidence_root != row["evidence_root"]:
                raise LedgerIntegrityError(f"evidence manifest/root mismatch for case {case_id}")

        previous = GENESIS_HASH
        expected_sequence = 1
        events = self.conn.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        for row in events:
            if row["sequence"] != expected_sequence or row["previous_event_hash"] != previous:
                raise LedgerIntegrityError("event sequence or hash chain is broken")
            identity = identities.get(row["signer_id"])
            if identity is None or row["event_type"] not in ROLE_ALLOWED_EVENTS[identity["role"]]:
                raise LedgerIntegrityError("event signer is unknown or role is unauthorized")
            case = evidence.get(row["case_id"])
            if case is None or case["evidence_root"] != row["evidence_root"]:
                raise LedgerIntegrityError("event is not bound to its registered evidence root")
            unsigned = {
                "schema_version": SCHEMA_VERSION,
                "sequence": row["sequence"],
                "case_id": row["case_id"],
                "event_type": row["event_type"],
                "payload_json": row["payload_json"],
                "evidence_root": row["evidence_root"],
                "signer_id": row["signer_id"],
                "created_ns": row["created_ns"],
                "previous_event_hash": row["previous_event_hash"],
            }
            expected_hash = sha256_bytes(canonical_bytes(unsigned))
            if expected_hash != row["event_hash"] or not _verify_signature(
                identity["public_key_b64"], bytes.fromhex(expected_hash), row["signature_b64"]
            ):
                raise LedgerIntegrityError("event content or signature was altered")
            previous = row["event_hash"]
            expected_sequence += 1

        anchors = self.conn.execute("SELECT * FROM anchors ORDER BY anchor_id").fetchall()
        for row in anchors:
            identity = identities.get(row["signer_id"])
            at_sequence = self.conn.execute("SELECT event_hash FROM events WHERE sequence=?", (row["sequence"],)).fetchone()
            message = canonical_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": row["sequence"],
                    "event_hash": row["event_hash"],
                    "created_ns": row["created_ns"],
                }
            )
            if (
                identity is None
                or identity["role"] != "anchor"
                or at_sequence is None
                or at_sequence["event_hash"] != row["event_hash"]
                or not _verify_signature(identity["public_key_b64"], message, row["signature_b64"])
            ):
                raise LedgerIntegrityError("anchor is invalid, missing its event, or signed by an unauthorized identity")
        return {"events": len(events), "evidence_cases": len(evidence), "anchors": len(anchors)}

    def verify_local_artifacts(self, case_id: str) -> int:
        """Re-hash `file://` artifacts. Remote object checks use immutable ETag/version APIs externally."""

        row = self.conn.execute("SELECT manifest_json FROM evidence WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            raise KeyError("unknown case")
        manifest = _manifest_from_payload(json.loads(row["manifest_json"]))
        checked = 0
        for artifact in manifest.artifacts:
            parsed = urlparse(artifact.uri)
            if parsed.scheme != "file":
                continue
            path_text = unquote(parsed.path)
            # Windows: file:///F:/... -> /F:/... ; strip the leading slash so
            # Path does not interpret it as rooted on the current drive.
            if os.name == "nt" and len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
                path_text = path_text[1:]
            path = Path(path_text)
            digest, size = sha256_file(path)
            if digest != artifact.content_sha256 or size != artifact.byte_length:
                raise LedgerIntegrityError(f"local artifact changed: {artifact.modality}")
            checked += 1
        return checked


def _manifest_from_payload(payload: Mapping[str, Any]) -> EvidenceManifest:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LedgerIntegrityError("unsupported evidence manifest version")
    return EvidenceManifest(
        case_id=payload["case_id"],
        episode_commit_uri=payload["episode_commit_uri"],
        capture_session_id=payload["capture_session_id"],
        action_schema_version=payload["action_schema_version"],
        preprocess_version=payload["preprocess_version"],
        artifacts=tuple(EvidenceArtifact(**artifact) for artifact in payload["artifacts"]),
    )


def run_smoke_test() -> None:
    """Exercise dual review, local content re-hash and deliberate DB tamper detection."""

    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        modalities = {}
        for modality in REQUIRED_MODALITIES:
            path = root / f"{modality}.bin"
            path.write_bytes(f"stable-{modality}".encode("utf-8"))
            modalities[modality] = path
        ledger = MultimodalAuditLedger(str(root / "ledger.sqlite"))
        capture, validator, reviewer_a, reviewer_b, anchor = (Ed25519Signer() for _ in range(5))
        for signer_id, role, signer in [
            ("capture-1", "capture_gateway", capture),
            ("validator-1", "validator", validator),
            ("reviewer-a", "reviewer", reviewer_a),
            ("reviewer-b", "reviewer", reviewer_b),
            ("anchor-1", "anchor", anchor),
        ]:
            ledger.enroll_identity(signer_id=signer_id, role=role, public_key_b64=signer.public_key_b64())
        manifest = EvidenceManifest.from_local_files(
            case_id="case-demo-001",
            episode_commit_uri="file://immutable/episode-commit-demo",
            capture_session_id="session-demo",
            action_schema_version="action-block-v1",
            preprocess_version="camera-proprio-v1",
            local_modalities=modalities,
            start_ns=1_000,
            end_ns=2_000,
            producer_id="capture-rig-1",
        )
        ledger.register_evidence(manifest, signer_id="capture-1", signer=capture)
        report_hash = sha256_bytes(b"validator-report-v1")
        ledger.attest_validation(
            case_id=manifest.case_id,
            signer_id="validator-1",
            signer=validator,
            validator_version="cross-modal-validator-v3",
            report_sha256=report_hash,
            checks={"time_alignment": True, "calibration_bound": True, "action_contact_consistency": True},
        )
        patch_hash = sha256_bytes(b"approved bounded cloth-physics patch")
        rationale_a = sha256_bytes(b"reviewer a rationale")
        rationale_b = sha256_bytes(b"reviewer b rationale")
        ledger.submit_review(case_id=manifest.case_id, signer_id="reviewer-a", signer=reviewer_a, patch_sha256=patch_hash, decision="approve_for_data", rationale_sha256=rationale_a)
        ledger.submit_review(case_id=manifest.case_id, signer_id="reviewer-b", signer=reviewer_b, patch_sha256=patch_hash, decision="approve_for_data", rationale_sha256=rationale_b, confirms_existing_review=True)
        ledger.create_anchor(signer_id="anchor-1", signer=anchor)
        assert ledger.verify_local_artifacts(manifest.case_id) == len(REQUIRED_MODALITIES)
        assert ledger.eligible_patch_for_data(case_id=manifest.case_id, patch_sha256=patch_hash)
        verified = ledger.verify_integrity()

        # Simulate an unauthorized SQL update. Recomputed hashes and signatures do
        # not match, so verification must fail closed.
        ledger.conn.execute("UPDATE events SET payload_json='{}' WHERE sequence=2")
        try:
            ledger.verify_integrity()
            raise AssertionError("tampering was not detected")
        except LedgerIntegrityError:
            pass
        ledger.close()
        print(json.dumps({"smoke_test": "passed", **verified}, sort_keys=True), flush=True)


if __name__ == "__main__":
    run_smoke_test()
