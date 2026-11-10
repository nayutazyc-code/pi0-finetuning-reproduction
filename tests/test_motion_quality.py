import unittest
import numpy as np
from pi0_repro.motion_quality import motion_quality


class MotionQualityTests(unittest.TestCase):
    def rows(self,n,**extra):
        return [dict(near_grasp_region=True,**extra) for _ in range(n)]

    def test_stationary_failure_is_not_perfect_active_smoothness(self):
        q=np.zeros((100,14));r=motion_quality(q,q,self.rows(100))
        self.assertIsNone(r['active_frame_max_jerk_rad_s3'])
        self.assertFalse(r['coverage']['active_motion_scored'])
        self.assertFalse(r['coverage']['transport_motion_comparable'])
        self.assertGreater(r['low_motion_pause_max_seconds'],1.5)

    def test_gripper_motion_and_required_success_hold_not_stalls(self):
        q=np.zeros((100,14));a=q.copy();a[:,13]=np.arange(100)*.002
        r=motion_quality(q,a,self.rows(100));self.assertEqual(r['low_motion_pause_count'],0)
        r=motion_quality(q,q,self.rows(100,inside_box=True,released=True,supported_by_box=True,stable=True))
        self.assertEqual(r['low_motion_pause_count'],0)

    def test_constant_speed_scored_and_spike_detected(self):
        q=np.zeros((100,14));q[:,7]=np.arange(100)*.002
        r=motion_quality(q,q,self.rows(100,ever_grasped=True,inside_box=True))
        self.assertTrue(r['coverage']['transport_motion_comparable'])
        self.assertLess(r['active_frame_max_jerk_rad_s3']['max'],1e-7)
        q[50:,7]+=.1
        s=motion_quality(q,q,self.rows(100))
        self.assertGreater(s['active_frame_max_jerk_rad_s3']['max'],1000)

    def test_disjoint_short_holds_not_joined(self):
        q=np.zeros((100,14));q[::20,7]=.1
        r=motion_quality(q,q,self.rows(100))
        self.assertEqual(r['low_motion_pause_count'],0)


if __name__=='__main__':unittest.main()
