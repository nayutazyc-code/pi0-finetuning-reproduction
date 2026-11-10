import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
from banana_temporal import expert_difference_loss


class TemporalLossTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.actions = rng.normal(size=(2, 5, 32)).astype(np.float32)
        self.noise = rng.normal(size=self.actions.shape).astype(np.float32)
        self.t = np.array([.2, .8], np.float32)
        self.x = self.t[:, None, None]*self.noise + (1-self.t[:, None, None])*self.actions
        self.flow = self.noise - self.actions
        self.valid = np.ones((2, 5), bool)

    def loss(self, flow, valid=None, **extra):
        return expert_difference_loss(self.x, flow, self.actions, self.t,
                                      self.valid if valid is None else valid, np.ones(12), xp=np, **extra)

    def test_perfect_flow_preserves_nonzero_expert_motion(self):
        np.testing.assert_allclose(self.loss(self.flow), 0, atol=1e-12)
        shifted = self.flow.copy()
        shifted[:, :, :6] += .7
        np.testing.assert_allclose(self.loss(shifted), 0, atol=1e-12)

    def test_grippers_and_padded_axes_excluded(self):
        flow = self.flow.copy()
        flow[:, 2, [6, 13, *range(14, 32)]] += 100
        np.testing.assert_allclose(self.loss(flow), 0, atol=1e-12)

    def test_episode_tail_not_inferred_from_equal_values(self):
        flow = self.flow.copy()
        flow[:, 3:, 7] += 5
        valid = self.valid.copy(); valid[:, 3:] = False
        np.testing.assert_allclose(self.loss(flow, valid), 0, atol=1e-12)
        self.assertTrue((self.loss(flow) > 0).all())
        np.testing.assert_array_equal(self.loss(flow, np.zeros_like(valid)), 0)

    def test_native_scale_and_noise_time_are_applied(self):
        flow = self.flow.copy(); flow[:, 2, 7] += 1
        a = self.loss(flow)
        b = expert_difference_loss(self.x, flow, self.actions, self.t, self.valid, np.full(12, 2), xp=np)
        np.testing.assert_allclose(b, a*4, rtol=2e-5)
        self.assertGreater(a[1], a[0])

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX gradient test runs in cloud environment')
    def test_jax_gradient_and_real_static_pose(self):
        import jax
        import jax.numpy as jnp
        a=jnp.zeros((1, 4, 32)); x=a; t=jnp.ones(1)*.5; valid=jnp.ones((1, 4), bool)
        v=a.at[0, 2, 7].set(1)
        loss=lambda value:jnp.sum(expert_difference_loss(x,value,a,t,valid,jnp.ones(12),xp=jnp))
        g=jax.grad(loss)(v)
        self.assertTrue(np.isfinite(g).all())
        self.assertGreater(float(loss(v)), 0)
        self.assertGreater(float(jnp.abs(g[..., 7]).sum()), 0)
        np.testing.assert_array_equal(np.asarray(g[..., [6,13]]), 0)


if __name__ == '__main__':
    unittest.main()
