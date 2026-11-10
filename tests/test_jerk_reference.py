import copy
import importlib.util
import unittest

import numpy as np

from pi0_repro.jerk_reference import JerkLimitedArmReference
from pi0_repro.config import PROJECT_ROOT, load_config
from pi0_repro.execution import ChunkExecutor, JointTargetLimiter


@unittest.skipUnless(importlib.util.find_spec("ruckig"), "Optional motion extra not installed")
class JerkReferenceTests(unittest.TestCase):
    def test_gripper_rate_limits_are_unchanged(self):
        old_settings = load_config(PROJECT_ROOT / "configs/banana_v6_bridge20_pi0.toml").simulation
        new_settings = load_config(PROJECT_ROOT / "configs/banana_v6_jerk300_pi0.toml").simulation
        old, new = JointTargetLimiter(np.zeros(14), old_settings), JointTargetLimiter(np.zeros(14), new_settings)
        for step in range(100):
            target = np.full(14, np.sin(step * 0.1), np.float32)
            a, b = old.limit(target), new.limit(target)
            np.testing.assert_array_equal(a[[6, 13]], b[[6, 13]])
            old.update_executed(a)
            new.update_executed(b)

    def test_full_rtc_forecast_with_filter_and_projection(self):
        class Client:
            def infer(self, observation):
                return np.zeros((100, 14), np.float32), 20, {"rtc": {"schema": "pi0-rtc-v1"}}
            def close(self):
                pass
        settings = load_config(PROJECT_ROOT / "configs/banana_v6_jerk300_pi0.toml").simulation
        executor = ChunkExecutor(Client(), settings)
        low, high = np.full(14, -2), np.full(14, 2)
        low[[6, 13]], high[[6, 13]] = 0, 0.8
        executor.limiter = JointTargetLimiter(np.zeros(14), settings, low, high)
        executor.project_target = lambda a: np.clip(a, low, high).astype(np.float32)
        try:
            for origin in (0, 5, 10):
                chunk = np.random.default_rng(origin).normal(0, .05, (100, 14)).astype(np.float32)
                executor._accept(origin, (chunk, 20, {}), origin + 2)
            executor._request(17, {"state": np.zeros(14, np.float32)})
            executor.pending[1].result(timeout=5)
            expected = executor.request_prefixes[17].copy()
            for step in range(17, 110):
                action = executor.project_target(executor.limiter.limit(executor._target_for_step(step)[1]))
                np.testing.assert_array_equal(action, expected[step - 17])
                executor.limiter.update_executed(action)
        finally:
            executor.close()

    def test_step_reversals_obey_bounds_and_settle(self):
        ref = JerkLimitedArmReference(np.zeros(12), 1.1, 40, 150)
        positions = [ref.position.copy()]
        previous_acceleration = ref.acceleration.copy()
        for step in range(500):
            target = np.full(12, 0.8 if step < 100 else -0.5)
            positions.append(ref.advance(target))
            self.assertLessEqual(abs(ref.velocity).max(), 1.1 + 1e-8)
            self.assertLessEqual(abs(ref.acceleration).max(), 40 + 1e-8)
            self.assertLessEqual(abs(ref.acceleration - previous_acceleration).max() / 0.02, 150 + 1e-6)
            previous_acceleration = ref.acceleration.copy()
            ref.update_executed(positions[-1])
        np.testing.assert_allclose(ref.position, -0.5, atol=1e-9)
        self.assertLessEqual(abs(np.diff(np.asarray(positions), n=3, axis=0)).max() / 0.02**3, 150 + 1e-5)

    def test_copied_forecast_matches_with_float32_commands(self):
        live = JerkLimitedArmReference(np.zeros(12), 1.1, 40, 150)
        for _ in range(30):
            live.update_executed(live.advance(np.ones(12)).astype(np.float32))
        forecast = copy.deepcopy(live)
        for step in range(100):
            target = np.full(12, np.sin(step * 0.03))
            a = live.advance(target).astype(np.float32)
            b = forecast.advance(target).astype(np.float32)
            np.testing.assert_array_equal(a, b)
            live.update_executed(a)
            forecast.update_executed(b)

    def test_hard_projection_reanchors_only_clipped_axes(self):
        ref = JerkLimitedArmReference(np.zeros(12), 1.1, 40, 150)
        for _ in range(10):
            command = ref.advance(np.ones(12))
            ref.update_executed(command)
        command = ref.advance(np.ones(12))
        projected = command.copy()
        projected[0] = 0
        ref.update_executed(projected)
        self.assertEqual(ref.position[0], 0)
        self.assertEqual(ref.velocity[0], 0)
        self.assertEqual(ref.acceleration[0], 0)
        self.assertGreater(ref.velocity[1], 0)
        self.assertEqual(ref.projection_resets, 1)


if __name__ == "__main__":
    unittest.main()
