"""Offline contract tests for resumable jobs and dataset-commit training conversion."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from dataset_commit_to_acvjepa_windows import convert_episode, sha256
from resumable_simjob_ledger import LeaseLedger
from sim2real_pointcloud_video_pipeline import EpisodeWriter, SyntheticDeformableBackend, demo_job


class LeaseLedgerTests(unittest.TestCase):
    def test_expired_lease_is_reclaimed_and_completion_is_idempotent_by_job_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = LeaseLedger(str(Path(directory) / "ledger.sqlite"))
            payload = {"job_id": "j1", "simulator_version": "v1"}
            ledger.register("key-j1", payload)
            ledger.register("key-j1", payload)  # idempotent registration
            first = ledger.acquire("worker-a", lease_seconds=0.001)
            self.assertIsNotNone(first)
            time.sleep(0.01)
            second = ledger.acquire("worker-b", lease_seconds=10.0)
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.worker_id, "worker-b")
            ledger.complete(second, artifact_sha256="a", metadata_sha256="b", remote_commit_uri="file://commit")
            self.assertEqual(ledger.summary(), {"COMPLETE": 1})
            self.assertIsNone(ledger.acquire("worker-c", lease_seconds=10.0))
            ledger.conn.close()


class TrainingInputTests(unittest.TestCase):
    def test_verified_episode_produces_windows_with_auxiliary_point_cloud(self) -> None:
        request = demo_job()
        raw = SyntheticDeformableBackend(frames=8, height=20, width=24).rollout(request)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir, report = EpisodeWriter(root / "episodes", max_points=16).write(request, raw)
            self.assertTrue(report.accepted)
            windows = convert_episode(
                episode_dir,
                root / "windows",
                context_steps=2,
                horizon=3,
                proprio_dim=8,
                event_dim=4,
                dataset_commit_sha="commit-demo",
            )
            self.assertEqual(len(windows), 4)
            import torch

            payload = torch.load(windows[0], map_location="cpu", weights_only=True)
            self.assertIn("context_point_cloud_xyz", payload)
            self.assertEqual(tuple(payload["context_point_cloud_xyz"].shape), (2, 16, 3))
            self.assertEqual(tuple(payload["executed_actions"].shape[:1]), (3,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
