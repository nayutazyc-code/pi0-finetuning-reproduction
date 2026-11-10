from pathlib import Path
import sys
import importlib.util
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'server'))
from banana_temporal import near_term_flow_loss


class NearTermLossTests(unittest.TestCase):
    def test_constant_loss_scale_preserved_with_padding(self):
        flow=np.ones((2,100))*.7;valid=np.ones_like(flow,bool);valid[1,5:]=False
        result=near_term_flow_loss(flow,valid,30,2,xp=np)
        np.testing.assert_allclose(result.mean(axis=-1),.7)
        np.testing.assert_array_equal(result[1,5:],0)

    def test_recent_actions_have_declared_relative_weight(self):
        flow=np.ones((1,100));valid=np.ones_like(flow,bool)
        result=near_term_flow_loss(flow,valid,30,2,xp=np)
        self.assertEqual(result[0,0]/result[0,99],2)
        self.assertAlmostEqual(float(result.mean()),1)

    def test_static_valid_targets_not_removed(self):
        flow=np.zeros((1,100));flow[0,2]=1;valid=np.ones_like(flow,bool)
        self.assertGreater(near_term_flow_loss(flow,valid,30,2,xp=np).sum(),0)
        np.testing.assert_array_equal(near_term_flow_loss(flow,np.zeros_like(valid),30,2,xp=np),0)

    @unittest.skipUnless(importlib.util.find_spec('jax'),'JAX gradient test runs on cloud CPU')
    def test_jax_gradient_excludes_padding_and_preserves_relative_weight(self):
        import jax
        import jax.numpy as jnp
        flow=jnp.ones((1,100));valid=jnp.arange(100)[None]<50
        gradient=jax.grad(lambda x:jnp.mean(near_term_flow_loss(x,valid,30,2,xp=jnp)))(flow)
        np.testing.assert_array_equal(np.asarray(gradient[0,50:]),0)
        self.assertAlmostEqual(float(gradient[0,0]/gradient[0,40]),2)
        self.assertAlmostEqual(float(gradient.sum()),1,places=6)


if __name__=='__main__':unittest.main()
