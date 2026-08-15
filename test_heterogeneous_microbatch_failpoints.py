"""Independent unit tests for each 2:1 recovery chaos failpoint.

These tests run only the logical/offline framework. Each test uses a new
framework and unique experiment ID so alert freeze/idempotency state cannot leak
between scenarios.
"""
from __future__ import annotations

import unittest

from heterogeneous_microbatch_chaos_framework import ChaosFault, ChaosScenario, HeterogeneousMicrobatchChaosFramework


class HeterogeneousMicrobatchFailpointTests(unittest.TestCase):
    def _run(self, fault: ChaosFault):
        framework = HeterogeneousMicrobatchChaosFramework()
        result = framework.run(ChaosScenario(fault=fault, experiment_id=f"unit-{fault.value}", seed=101))
        self.assertTrue(result.passed)
        self.assertEqual(result.fault, fault.value)
        return result

    def test_node_loss_at_final_allreduce_replays_exact_global_range(self) -> None:
        """2:1 [w0,w1]/[w2] must become an uncommitted replay, never a partial commit."""
        result = self._run(ChaosFault.NODE_LOSS_FINAL_ALLREDUCE)
        self.assertEqual(
            result.assertions,
            (
                "uncommitted_cursor_unchanged",
                "same_global_range_replayed",
                "rank_layout_reassigned",
                "new_checkpoint_committed",
            ),
        )
        # The implementation asserts offset=0 before recovery and offset=3 only
        # after a verified new checkpoint under the new elastic identity.
        self.assertEqual(len(result.assertions), 4)

    def test_partition_after_prepare_aborts_without_cursor_advance_and_freezes_plan(self) -> None:
        """A logical data/control-plane partition is not a reason to retry/commit old work."""
        result = self._run(ChaosFault.NETWORK_PARTITION_AFTER_PREPARE)
        self.assertIn("reservation_aborted", result.assertions)
        self.assertIn("cursor_not_advanced", result.assertions)
        self.assertIn("new_plans_frozen", result.assertions)
        self.assertIn("no_network_operation_executed", result.assertions)
        self.assertEqual(len(result.assertions), 4)

    def test_old_attempt_cannot_commit_after_rendezvous_epoch_changes(self) -> None:
        """New identity fences the old PREPARED attempt even if its checkpoint bytes are valid."""
        result = self._run(ChaosFault.STALE_PLAN_AFTER_RENDEZVOUS)
        self.assertEqual(
            result.assertions,
            ("old_attempt_fenced", "stale_commit_rejected", "cursor_not_advanced"),
        )
        self.assertNotIn("new_checkpoint_committed", result.assertions)

    def test_plan_with_wrong_topology_digest_is_rejected_before_second_reservation(self) -> None:
        """Topology mismatch must be fail-closed at prepare, not later during collective."""
        result = self._run(ChaosFault.PLAN_TOPOLOGY_MISMATCH)
        self.assertEqual(
            result.assertions,
            ("topology_digest_mismatch_rejected", "no_second_reservation_created"),
        )
        self.assertEqual(len(result.assertions), 2)

    def test_corrupt_optimizer_cache_entry_is_evicted_and_durable_checkpoint_is_used(self) -> None:
        """The cache can accelerate a committed load but cannot supply unverified AdamW state."""
        result = self._run(ChaosFault.CACHE_CORRUPTION_DURING_RESTORE)
        self.assertEqual(
            result.assertions,
            ("cache_corruption_detected", "entry_evicted", "durable_fallback_verified", "commit_pointer_unchanged"),
        )
        self.assertEqual(len(result.assertions), 4)

    def test_framework_rejects_non_isolated_or_non_dry_run_scenario(self) -> None:
        framework = HeterogeneousMicrobatchChaosFramework()
        with self.assertRaises(PermissionError):
            framework.run(
                ChaosScenario(
                    fault=ChaosFault.NODE_LOSS_FINAL_ALLREDUCE,
                    experiment_id="must-reject",
                    seed=0,
                    environment="production",
                    dry_run=False,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
