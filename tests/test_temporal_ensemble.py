"""Temporal-reference edge cases and full copied-limiter RTC forecast audit."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi0_repro.config import PROJECT_ROOT, load_config
from pi0_repro.execution import ChunkExecutor, JointTargetLimiter, JOINTS, TemporalEnsembleReference


class DummyClient:
    def infer(self, observation):
        return np.zeros((100, 14), np.float32), 20.0, {"rtc": {"schema": "pi0-rtc-v1"}}

    def close(self):
        pass


class TemporalEnsembleTests(unittest.TestCase):
    def settings(self):
        return load_config(PROJECT_ROOT / "configs/banana_v6_temporal3_pi0.toml").simulation

    def test_alignment_zero_actions_expiry_and_grippers(self):
        old = np.zeros((100, 14), np.float32)
        new = np.repeat(np.arange(100, dtype=np.float32)[:, None], 14, axis=1)
        old[:, [6, 13]] = 0.2
        new[:, [6, 13]] = 0.8
        ref = TemporalEnsembleReference(((0, old), (10, new)), 0.01)
        weight_new = np.exp(-0.01) / (1 + np.exp(-0.01))
        np.testing.assert_allclose(ref.target(15)[JOINTS], 5 * weight_new)
        np.testing.assert_array_equal(ref.target(15)[[6, 13]], new[5, [6, 13]])
        self.assertEqual(len(ref.contributors(15)), 2)  # Zero targets are valid.
        np.testing.assert_array_equal(ref.target(100), new[90])
        with self.assertRaises(ValueError):
            ref.target(110)

    def test_bounded_cache_and_rejected_stale_reply(self):
        executor = ChunkExecutor(DummyClient(), self.settings())
        try:
            for origin in (0, 10, 20, 30):
                chunk = np.full((100, 14), origin, np.float32)
                executor._accept(origin, (chunk, 20, {}), origin + 2)
            self.assertEqual([p[0] for p in executor.reference.plans], [10, 20, 30])
            reference = executor.reference
            executor._accept(40, (np.zeros((100, 14), np.float32), 2000, {}), 140)
            self.assertIs(executor.reference, reference)
            self.assertEqual(executor.stats["stale_chunks_discarded"], 1)
        finally:
            executor.close()

    def test_rtc_forecast_matches_execution_through_plan_expiry(self):
        executor = ChunkExecutor(DummyClient(), self.settings())
        low, high = np.full(14, -2.0), np.full(14, 2.0)
        low[[6, 13]], high[[6, 13]] = 0, 0.8
        executor.limiter = JointTargetLimiter(np.zeros(14), executor.settings, low, high)
        executor.project_target = lambda target: np.clip(target, low, high).astype(np.float32)
        try:
            for origin in (0, 5, 10):
                chunk = np.random.default_rng(origin).normal(0, 0.05, (100, 14)).astype(np.float32)
                chunk[:, [6, 13]] = 1
                executor._accept(origin, (chunk, 20, {}), origin + 2)
            executor._request(17, {"state": np.zeros(14, np.float32)})
            _, future = executor.pending
            result = future.result(timeout=5)
            expected = executor.request_prefixes[17]
            commands = []
            for step in range(17, 110):
                action = executor.project_target(executor.limiter.limit(executor._target_for_step(step)[1]))
                executor.limiter.update_executed(action)
                executor.record_executed(step, action)
                commands.append(action)
            np.testing.assert_array_equal(commands, expected[:93])
            self.assertEqual(len(executor.reference.contributors(99)), 3)
            self.assertEqual(len(executor.reference.contributors(105)), 1)
            # At overlap exhaustion, even a still-current new prediction is rejected.
            executor.pending = None
            executor._accept(17, result, 110)
            self.assertFalse(executor.timings[-1]["accepted"])
            self.assertEqual(executor.timings[-1]["rtc_request"]["planned_vs_executed_prefix_max_error"], 0)
        finally:
            executor.close()

    def test_invalid_ensemble_settings_are_rejected(self):
        source = (PROJECT_ROOT / "configs/banana_v6_temporal3_pi0.toml").read_text()
        mutations = [("chunk_blend_steps = 0", "chunk_blend_steps = 20"),
                     ("temporal_ensemble_max_chunks = 3", "temporal_ensemble_max_chunks = 1"),
                     ("temporal_ensemble_coeff = 0.01", "temporal_ensemble_coeff = nan"),
                     ("temporal_ensemble = true", "temporal_ensemble = 1")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.toml"
            for before, after in mutations:
                path.write_text(source.replace(before, after))
                with self.subTest(after=after), self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
