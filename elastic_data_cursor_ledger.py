"""Global data-cursor / checkpoint alignment for elastic heterogeneous DDP.

The ledger implements an at-least-once *work-window* contract:
- Only a checkpoint verified and marked COMMITTED advances `next_offset`.
- A node failure leaves its in-flight reservation UNCOMMITTED. After rendezvous,
  a fresh worker group re-reserves the same global work range from the previous
  committed cursor; rank assignment may change but the set of work IDs cannot.
- The local rank, world size and topology epoch are not data cursor identities.
  They are validation fields bound to each attempt and must be regenerated.

This module is offline training bookkeeping. It neither controls robots nor
performs network operations. The SQLite ledger is an experiment-local reference;
production should replicate commit records and checkpoint hashes to durable,
access-controlled storage.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Dict, Iterable, Mapping, Sequence, Tuple
from uuid import uuid4


class CursorContractError(RuntimeError):
    """Raised when a plan, checkpoint or elastic identity violates cursor semantics."""


def canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkWindow:
    """One immutable, pre-batched training item in a canonical global order."""

    work_id: str
    provenance_hash: str
    cost_units: float


@dataclass(frozen=True)
class GlobalWorkManifest:
    dataset_commit: str
    ordered_windows: Tuple[WorkWindow, ...]

    def validate(self) -> None:
        if not self.dataset_commit or not self.ordered_windows:
            raise ValueError("dataset commit and at least one work window are required")
        ids = [item.work_id for item in self.ordered_windows]
        if any(not item.work_id or not item.provenance_hash or item.cost_units <= 0 for item in self.ordered_windows):
            raise ValueError("work windows require IDs, provenance hashes and positive costs")
        if len(ids) != len(set(ids)):
            raise ValueError("work IDs must be unique in canonical global order")

    @property
    def digest(self) -> str:
        self.validate()
        return sha256_bytes(
            canonical_bytes(
                {
                    "dataset_commit": self.dataset_commit,
                    "ordered_windows": [asdict(item) for item in self.ordered_windows],
                }
            )
        )

    def slice(self, start_offset: int, end_offset: int) -> Tuple[WorkWindow, ...]:
        if not (0 <= start_offset <= end_offset <= len(self.ordered_windows)):
            raise CursorContractError("requested cursor range is outside the immutable manifest")
        return self.ordered_windows[start_offset:end_offset]


@dataclass(frozen=True)
class ElasticIdentity:
    """Identity of the *current* rendezvoused worker group, never a durable shard key."""

    run_id: str
    restart_count: int
    world_size: int
    topology_epoch: str
    topology_digest: str

    def validate(self) -> None:
        if not self.run_id or not self.topology_epoch or not self.topology_digest:
            raise ValueError("elastic identity requires run and topology facts")
        if self.restart_count < 0 or self.world_size <= 0:
            raise ValueError("invalid elastic restart count or world size")


@dataclass(frozen=True)
class CursorBoundPlan:
    """A data-facing projection of TopologyAwareUpdatePlan.

    `rank_work_ids` contains deterministic rank-local iterator order. The global
    cursor validates only the *set* of IDs against one contiguous manifest range;
    topology-aware planning may reorder that set across ranks after a restart.
    """

    plan_version: int
    topology_epoch: str
    topology_digest: str
    work_manifest_digest: str
    world_size: int
    rank_work_ids: Tuple[Tuple[str, ...], ...]

    @property
    def all_work_ids(self) -> Tuple[str, ...]:
        return tuple(item for rank_items in self.rank_work_ids for item in rank_items)

    def validate(self) -> None:
        if self.plan_version < 1 or self.world_size <= 0 or len(self.rank_work_ids) != self.world_size:
            raise ValueError("plan version/world-size/rank work entries are invalid")
        if not self.topology_epoch or not self.topology_digest or not self.work_manifest_digest:
            raise ValueError("plan is missing topology or work manifest binding")
        all_ids = self.all_work_ids
        if not all_ids or len(all_ids) != len(set(all_ids)):
            raise ValueError("plan must assign each non-empty work ID exactly once")

    @classmethod
    def from_topology_aware(cls, plan) -> "CursorBoundPlan":
        """Adapt `TopologyAwareUpdatePlan` without importing it at module load time."""

        return cls(
            plan_version=plan.plan_version,
            topology_epoch=plan.topology_epoch,
            topology_digest=plan.topology_digest,
            work_manifest_digest=plan.work_manifest_digest,
            world_size=plan.world_size,
            rank_work_ids=tuple(tuple(rank_plan.work_item_ids) for rank_plan in plan.ranks),
        )


@dataclass(frozen=True)
class CheckpointArtifact:
    uri: str
    sha256: str
    model_state_hash: str
    ema_state_hash: str
    optimizer_state_hash: str
    validation_hash: str

    def validate(self) -> None:
        values = (self.sha256, self.model_state_hash, self.ema_state_hash, self.optimizer_state_hash, self.validation_hash)
        if not self.uri or any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in values):
            raise ValueError("checkpoint URI and lowercase SHA-256 hashes are required")


@dataclass(frozen=True)
class CommittedCursor:
    commit_id: str
    committed_step: int
    next_offset: int
    checkpoint: CheckpointArtifact
    dataset_commit: str
    manifest_digest: str
    committed_world_size: int
    committed_topology_epoch: str
    committed_work_manifest_digest: str


@dataclass(frozen=True)
class UpdateReservation:
    attempt_id: str
    parent_commit_id: str
    start_offset: int
    end_offset: int
    plan: CursorBoundPlan
    reserved_global_work_ids: Tuple[str, ...]
    elastic_identity: ElasticIdentity


class ElasticCursorLedger:
    """Transactional global cursor ledger with prepare/commit/abort semantics."""

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
            CREATE TABLE IF NOT EXISTS committed_cursors (
                run_key TEXT PRIMARY KEY,
                commit_id TEXT NOT NULL,
                committed_step INTEGER NOT NULL,
                next_offset INTEGER NOT NULL,
                checkpoint_json TEXT NOT NULL,
                dataset_commit TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                committed_world_size INTEGER NOT NULL,
                committed_topology_epoch TEXT NOT NULL,
                committed_work_manifest_digest TEXT NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS update_attempts (
                attempt_id TEXT PRIMARY KEY,
                run_key TEXT NOT NULL,
                parent_commit_id TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                plan_json TEXT NOT NULL,
                elastic_json TEXT NOT NULL,
                global_work_ids_json TEXT NOT NULL,
                status TEXT NOT NULL,
                checkpoint_json TEXT,
                reason TEXT,
                created_ns INTEGER NOT NULL,
                committed_ns INTEGER
            );
            CREATE UNIQUE INDEX IF NOT EXISTS only_one_prepared_attempt
            ON update_attempts(run_key) WHERE status='PREPARED';
            """
        )

    def bootstrap(self, *, run_key: str, manifest: GlobalWorkManifest, checkpoint: CheckpointArtifact, identity: ElasticIdentity) -> CommittedCursor:
        manifest.validate()
        checkpoint.validate()
        identity.validate()
        payload = json.dumps(asdict(checkpoint), sort_keys=True)
        initial = CommittedCursor(
            commit_id="genesis-" + sha256_bytes(canonical_bytes({"run_key": run_key, "checkpoint": asdict(checkpoint), "manifest": manifest.digest}))[:16],
            committed_step=0,
            next_offset=0,
            checkpoint=checkpoint,
            dataset_commit=manifest.dataset_commit,
            manifest_digest=manifest.digest,
            committed_world_size=identity.world_size,
            committed_topology_epoch=identity.topology_epoch,
            committed_work_manifest_digest=manifest.digest,
        )
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute("SELECT * FROM committed_cursors WHERE run_key=?", (run_key,)).fetchone()
            if old is None:
                self.conn.execute(
                    "INSERT INTO committed_cursors VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_key,
                        initial.commit_id,
                        initial.committed_step,
                        initial.next_offset,
                        payload,
                        initial.dataset_commit,
                        initial.manifest_digest,
                        initial.committed_world_size,
                        initial.committed_topology_epoch,
                        initial.committed_work_manifest_digest,
                        time.time_ns(),
                    ),
                )
            else:
                existing = self._cursor_from_row(old)
                if existing.dataset_commit != manifest.dataset_commit or existing.manifest_digest != manifest.digest:
                    raise CursorContractError("run key collides with a different global work manifest")
                initial = existing
            self.conn.execute("COMMIT")
            return initial
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def latest_committed(self, run_key: str) -> CommittedCursor:
        row = self.conn.execute("SELECT * FROM committed_cursors WHERE run_key=?", (run_key,)).fetchone()
        if row is None:
            raise KeyError("unknown run key; bootstrap first")
        return self._cursor_from_row(row)

    def prepare_next_update(
        self,
        *,
        run_key: str,
        manifest: GlobalWorkManifest,
        plan: CursorBoundPlan,
        identity: ElasticIdentity,
    ) -> UpdateReservation:
        """Reserve exactly the next contiguous global range without advancing it."""

        manifest.validate()
        plan.validate()
        identity.validate()
        self._validate_plan_identity(plan, identity, manifest)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.latest_committed(run_key)
            self._validate_cursor_manifest(cursor, manifest)
            prepared = self.conn.execute(
                "SELECT attempt_id FROM update_attempts WHERE run_key=? AND status='PREPARED'", (run_key,)
            ).fetchone()
            if prepared is not None:
                raise CursorContractError("a prepared update exists; abort or commit it before reserving another")
            start = cursor.next_offset
            end = start + len(plan.all_work_ids)
            expected = manifest.slice(start, end)
            expected_ids = tuple(window.work_id for window in expected)
            if set(plan.all_work_ids) != set(expected_ids) or len(plan.all_work_ids) != len(expected_ids):
                raise CursorContractError(
                    "plan work IDs must equal the next contiguous global manifest range; "
                    "topology may reassign ranks but may not skip/duplicate data"
                )
            attempt_id = "attempt-" + uuid4().hex
            reservation = UpdateReservation(
                attempt_id=attempt_id,
                parent_commit_id=cursor.commit_id,
                start_offset=start,
                end_offset=end,
                plan=plan,
                reserved_global_work_ids=expected_ids,
                elastic_identity=identity,
            )
            self.conn.execute(
                "INSERT INTO update_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    run_key,
                    cursor.commit_id,
                    start,
                    end,
                    json.dumps(asdict(plan), sort_keys=True),
                    json.dumps(asdict(identity), sort_keys=True),
                    json.dumps(expected_ids),
                    "PREPARED",
                    None,
                    None,
                    time.time_ns(),
                    None,
                ),
            )
            self.conn.execute("COMMIT")
            return reservation
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def abort_uncommitted(self, *, attempt_id: str, reason: str) -> None:
        """Mark an attempt aborted without moving the committed global cursor."""

        if not reason.strip():
            raise ValueError("abort reason is required")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT status FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise KeyError("unknown attempt")
            if row["status"] != "PREPARED":
                raise CursorContractError(f"cannot abort attempt in state {row['status']}")
            self.conn.execute("UPDATE update_attempts SET status='ABORTED', reason=? WHERE attempt_id=?", (reason, attempt_id))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def recover_after_rendezvous(
        self,
        *,
        run_key: str,
        manifest: GlobalWorkManifest,
        new_identity: ElasticIdentity,
        reason: str,
    ) -> CommittedCursor:
        """Fence old attempts then return the only valid restart point.

        The caller must rebuild DDP/NCCL before calling `prepare_next_update`.
        This method intentionally does not use old rank-local sampler positions.
        """

        manifest.validate()
        new_identity.validate()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.latest_committed(run_key)
            self._validate_cursor_manifest(cursor, manifest)
            rows = self.conn.execute(
                "SELECT attempt_id FROM update_attempts WHERE run_key=? AND status='PREPARED'", (run_key,)
            ).fetchall()
            for row in rows:
                self.conn.execute(
                    "UPDATE update_attempts SET status='ABORTED', reason=? WHERE attempt_id=?",
                    (f"rendezvous_rebuild:{reason}", row["attempt_id"]),
                )
            # A new identity may keep the same world size but must be treated as a
            # new process group. Reject an exact stale epoch/restart identity.
            old_epoch = cursor.committed_topology_epoch
            if new_identity.topology_epoch == old_epoch and new_identity.restart_count == 0:
                raise CursorContractError("recovery identity did not prove a new rendezvous/topology epoch")
            self.conn.execute("COMMIT")
            return cursor
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def commit_update(
        self,
        *,
        attempt_id: str,
        manifest: GlobalWorkManifest,
        checkpoint: CheckpointArtifact,
        current_identity: ElasticIdentity,
    ) -> CommittedCursor:
        """Atomically advance cursor only after checkpoint/hash/identity validation."""

        manifest.validate()
        checkpoint.validate()
        current_identity.validate()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None or row["status"] != "PREPARED":
                raise CursorContractError("only a PREPARED attempt can be committed")
            plan = CursorBoundPlan(
                **{**json.loads(row["plan_json"]), "rank_work_ids": tuple(tuple(item) for item in json.loads(row["plan_json"])["rank_work_ids"])}
            )
            prepared_identity = ElasticIdentity(**json.loads(row["elastic_json"]))
            if prepared_identity != current_identity:
                raise CursorContractError("elastic identity changed during update; abort and re-plan")
            self._validate_plan_identity(plan, current_identity, manifest)
            cursor = self.latest_committed(row["run_key"])
            if cursor.commit_id != row["parent_commit_id"] or cursor.next_offset != row["start_offset"]:
                raise CursorContractError("parent commit/cursor moved; stale attempt cannot commit")
            checkpoint_path = _local_path_from_uri(checkpoint.uri)
            if checkpoint_path is not None and sha256_file(checkpoint_path) != checkpoint.sha256:
                raise CursorContractError("checkpoint bytes do not match declared SHA-256")
            new_commit = "commit-" + sha256_bytes(
                canonical_bytes(
                    {
                        "parent": cursor.commit_id,
                        "attempt": attempt_id,
                        "checkpoint": asdict(checkpoint),
                        "end_offset": row["end_offset"],
                        "identity": asdict(current_identity),
                    }
                )
            )[:16]
            payload = json.dumps(asdict(checkpoint), sort_keys=True)
            self.conn.execute(
                "UPDATE committed_cursors SET commit_id=?, committed_step=?, next_offset=?, checkpoint_json=?, "
                "committed_world_size=?, committed_topology_epoch=?, committed_work_manifest_digest=?, updated_ns=? WHERE run_key=?",
                (
                    new_commit,
                    cursor.committed_step + 1,
                    int(row["end_offset"]),
                    payload,
                    current_identity.world_size,
                    current_identity.topology_epoch,
                    plan.work_manifest_digest,
                    time.time_ns(),
                    row["run_key"],
                ),
            )
            self.conn.execute(
                "UPDATE update_attempts SET status='COMMITTED', checkpoint_json=?, committed_ns=? WHERE attempt_id=?",
                (payload, time.time_ns(), attempt_id),
            )
            self.conn.execute("COMMIT")
            return self.latest_committed(row["run_key"])
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _validate_cursor_manifest(cursor: CommittedCursor, manifest: GlobalWorkManifest) -> None:
        if cursor.dataset_commit != manifest.dataset_commit or cursor.manifest_digest != manifest.digest:
            raise CursorContractError("checkpoint cursor points to a different dataset/work manifest")

    @staticmethod
    def _validate_plan_identity(plan: CursorBoundPlan, identity: ElasticIdentity, manifest: GlobalWorkManifest) -> None:
        if plan.world_size != identity.world_size:
            raise CursorContractError("plan world size differs from current rendezvoused worker group")
        if plan.topology_epoch != identity.topology_epoch or plan.topology_digest != identity.topology_digest:
            raise CursorContractError("plan belongs to a stale topology/process-group epoch")
        if plan.work_manifest_digest != manifest.digest:
            raise CursorContractError("plan work manifest hash differs from canonical global work manifest")

    @staticmethod
    def _cursor_from_row(row: sqlite3.Row) -> CommittedCursor:
        return CommittedCursor(
            commit_id=row["commit_id"],
            committed_step=int(row["committed_step"]),
            next_offset=int(row["next_offset"]),
            checkpoint=CheckpointArtifact(**json.loads(row["checkpoint_json"])),
            dataset_commit=row["dataset_commit"],
            manifest_digest=row["manifest_digest"],
            committed_world_size=int(row["committed_world_size"]),
            committed_topology_epoch=row["committed_topology_epoch"],
            committed_work_manifest_digest=row["committed_work_manifest_digest"],
        )


def _local_path_from_uri(uri: str) -> Path | None:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        path = unquote(parsed.path)
        # Windows: file:///F:/... -> /F:/... ; strip the leading slash so Path
        # does not interpret it as a rooted path on the current drive.
        if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path)
    return None


def _checkpoint(path: Path, label: str) -> CheckpointArtifact:
    path.write_bytes(f"checkpoint:{label}".encode("utf-8"))
    digest = sha256_file(path)
    return CheckpointArtifact(
        uri=path.resolve().as_uri(),
        sha256=digest,
        model_state_hash=sha256_bytes(f"model:{label}".encode()),
        ema_state_hash=sha256_bytes(f"ema:{label}".encode()),
        optimizer_state_hash=sha256_bytes(f"optimizer:{label}".encode()),
        validation_hash=sha256_bytes(f"validation:{label}".encode()),
    )


def _plan(*, epoch: str, topology_digest: str, manifest_digest: str, assignment: Sequence[Sequence[str]]) -> CursorBoundPlan:
    return CursorBoundPlan(
        plan_version=1,
        topology_epoch=epoch,
        topology_digest=topology_digest,
        work_manifest_digest=manifest_digest,
        world_size=len(assignment),
        rank_work_ids=tuple(tuple(items) for items in assignment),
    )


def run_smoke_test() -> None:
    """Simulate 2:1 failure before commit, rendezvous rebuild and cursor replay."""

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manifest = GlobalWorkManifest(
            dataset_commit="dataset-commit-demo",
            ordered_windows=tuple(
                WorkWindow(f"work-{index}", sha256_bytes(f"source-{index}".encode()), 1.0 + index)
                for index in range(7)
            ),
        )
        epoch0 = "epoch-initial"
        identity0 = ElasticIdentity("run-demo", 0, 2, epoch0, "topology-initial")
        ledger = ElasticCursorLedger(str(root / "cursor.sqlite"))
        initial = ledger.bootstrap(run_key="run-demo", manifest=manifest, checkpoint=_checkpoint(root / "genesis.pt", "genesis"), identity=identity0)
        assert initial.next_offset == 0

        # First 2:1 update has rank 0 = work-0/work-1 and rank 1 = work-2.
        attempt0 = ledger.prepare_next_update(
            run_key="run-demo",
            manifest=manifest,
            plan=_plan(epoch=epoch0, topology_digest="topology-initial", manifest_digest=manifest.digest, assignment=(("work-0", "work-1"), ("work-2",))),
            identity=identity0,
        )
        assert (attempt0.start_offset, attempt0.end_offset, attempt0.reserved_global_work_ids) == (0, 3, ("work-0", "work-1", "work-2"))
        # Simulate node loss during the final DDP synchronization: nothing commits.
        ledger.abort_uncommitted(attempt_id=attempt0.attempt_id, reason="node_lost_during_final_allreduce")
        assert ledger.latest_committed("run-demo").next_offset == 0

        # New rendezvous may reassign the same three global windows across ranks.
        identity1 = ElasticIdentity("run-demo", 1, 2, "epoch-after-rendezvous", "topology-after-rendezvous")
        recovered = ledger.recover_after_rendezvous(
            run_key="run-demo", manifest=manifest, new_identity=identity1, reason="node-a-replaced"
        )
        assert recovered.next_offset == 0 and recovered.commit_id == initial.commit_id
        attempt1 = ledger.prepare_next_update(
            run_key="run-demo",
            manifest=manifest,
            plan=_plan(epoch=identity1.topology_epoch, topology_digest=identity1.topology_digest, manifest_digest=manifest.digest, assignment=(("work-0",), ("work-1", "work-2"))),
            identity=identity1,
        )
        committed1 = ledger.commit_update(
            attempt_id=attempt1.attempt_id,
            manifest=manifest,
            checkpoint=_checkpoint(root / "step1.pt", "step1"),
            current_identity=identity1,
        )
        assert committed1.next_offset == 3 and committed1.committed_step == 1

        # The next committed 2:1 update progresses from the *global* cursor, not
        # from any old rank-local sampler offset.
        attempt2 = ledger.prepare_next_update(
            run_key="run-demo",
            manifest=manifest,
            plan=_plan(epoch=identity1.topology_epoch, topology_digest=identity1.topology_digest, manifest_digest=manifest.digest, assignment=(("work-3", "work-4"), ("work-5",))),
            identity=identity1,
        )
        committed2 = ledger.commit_update(
            attempt_id=attempt2.attempt_id,
            manifest=manifest,
            checkpoint=_checkpoint(root / "step2.pt", "step2"),
            current_identity=identity1,
        )
        assert committed2.next_offset == 6 and committed2.committed_step == 2
        ledger.close()
        print(
            json.dumps(
                {
                    "smoke_test": "passed",
                    "replayed_uncommitted_work": list(attempt1.reserved_global_work_ids),
                    "committed_next_offset": committed2.next_offset,
                    "committed_step": committed2.committed_step,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    run_smoke_test()
