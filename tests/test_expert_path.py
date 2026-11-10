from pathlib import Path
import tempfile
import unittest
import numpy as np

from pi0_repro.expert_path import RoundedPath,SmoothCruise
from pi0_repro.banana_expert import ExpertConfig,load_expert_config


class ExpertPathTests(unittest.TestCase):
    def test_corner_boundaries_match_position_velocity_acceleration(self):
        points=np.array([[0,0,.02],[0,0,.2],[.03,.2,.2],[.03,.2,.03]])
        path=RoundedPath(points,.01)
        np.testing.assert_allclose(path.evaluate(0),points[0],atol=1e-15)
        np.testing.assert_allclose(path.evaluate(path.length),points[-1],atol=1e-15)
        for i,(lo,hi,c,direction) in enumerate(path.pieces):
            if direction is not None:continue
            previous=path.pieces[i-1];following=path.pieces[i+1]
            np.testing.assert_allclose(c[0],previous[2]+(previous[1]-previous[0])*previous[3],atol=1e-14)
            np.testing.assert_allclose(c[-1],following[2],atol=1e-14)
            np.testing.assert_allclose(5*(c[1]-c[0])/(hi-lo),previous[3],atol=1e-13)
            np.testing.assert_allclose(5*(c[-1]-c[-2])/(hi-lo),following[3],atol=1e-13)
            np.testing.assert_allclose(20*(c[2]-2*c[1]+c[0])/(hi-lo)**2,0,atol=1e-10)
            np.testing.assert_allclose(20*(c[-1]-2*c[-2]+c[-3])/(hi-lo)**2,0,atol=1e-10)
        s=np.linspace(0,path.length,5001);p=np.array([path.evaluate(x) for x in s])
        self.assertLessEqual(float(np.max(np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(s))),1+1e-10)
        self.assertTrue((p[:,2]>=.02-1e-12).all())

    def test_clock_monotonic_bounded_and_rest_endpoints(self):
        for distance in (.001,.53):
            clock=SmoothCruise(distance,.048)
            t=np.linspace(0,clock.duration,10001);s=np.array([clock.evaluate(x) for x in t]);v=np.diff(s)/np.diff(t)
            self.assertAlmostEqual(s[0],0);self.assertAlmostEqual(s[-1],distance)
            self.assertTrue((np.diff(s)>=-1e-14).all())
            self.assertLessEqual(float(v.max()),.048+1e-10)
            self.assertLess(abs(v[0]),.00001);self.assertLess(abs(v[-1]),.00001)
            self.assertEqual(clock.evaluate(-1),0);self.assertAlmostEqual(clock.evaluate(clock.duration+1),distance)

    def test_bad_geometry_and_clock_are_rejected(self):
        for points,radius in [([[0,0,0],[0,0,0]],.01),([[0,0,0],[1,2,float('nan')]],.01),([[0,0,0],[1,2,3]],0)]:
            with self.assertRaises(ValueError):RoundedPath(points,radius)
        for args in [(0,.1,.5),(.1,0,.5),(.1,.1,float('nan'))]:
            with self.assertRaises(ValueError):SmoothCruise(*args)

    def test_opt_in_default_and_bounds(self):
        self.assertEqual(ExpertConfig().continuous_transport_radius,0)
        for value in ('-0.01','0.04','true','nan'):
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'invalid.toml';p.write_text('[expert]\ncontinuous_transport_radius = '+value+'\n')
                with self.assertRaises(ValueError):load_expert_config(p)


if __name__=='__main__':unittest.main()
