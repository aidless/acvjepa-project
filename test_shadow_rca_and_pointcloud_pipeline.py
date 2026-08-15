"""Offline tests for RCA and contract-test RGB-D/point-cloud generation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from shadow_degradation_rca import (
    RootCause,
    ShadowDegradationAnalyzer,
    ShadowTelemetry,
    demo_records,
)
from sim2real_pointcloud_video_pipeline import (
    EpisodeWriter,
    SyntheticDeformableBackend,
    demo_job,
)


class ShadowRCATests(unittest.TestCase):
    def test_degraded_demo_reports_blocking_runtime_and_uncertainty_causes(self) -> None:
        report = ShadowDegradationAnalyzer().analyze(demo_records())
        causes = {item.root_cause for item in report.hypotheses}
        self.assertIn(RootCause.EDGE_RUNTIME, causes)
        self.assertIn(RootCause.UNCERTAINTY_MISCALIBRATION, causes)
        self.assertIn(RootCause.SOFT_PHYSICS_GAP, causes)

    def test_insufficient_evidence_is_explicit(self) -> None:
        telemetry = demo_records()[:2]
        report = ShadowDegradationAnalyzer().analyze(telemetry)
        self.assertEqual(report.hypotheses[0].root_cause, RootCause.INSUFFICIENT_EVIDENCE)


class PointCloudPipelineTests(unittest.TestCase):
    def test_contract_backend_emits_aligned_fixed_size_pairs(self) -> None:
        request = demo_job()
        raw = SyntheticDeformableBackend(frames=4, height=24, width=32).rollout(request)
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir, quality = EpisodeWriter(Path(temp_dir), max_points=64).write(request, raw)
            self.assertTrue(quality.accepted, quality.reasons)
            payload = np.load(episode_dir / "episode.npz")
            try:
                self.assertEqual(payload["rgb_video"].shape, (4, 24, 32, 3))
                self.assertEqual(payload["point_cloud_xyz"].shape, (4, 64, 3))
                self.assertEqual(payload["point_mask"].shape, (4, 64))
                self.assertEqual(payload["executed_actions"].shape[0], 4)
            finally:
                payload.close()
            self.assertTrue((episode_dir / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
