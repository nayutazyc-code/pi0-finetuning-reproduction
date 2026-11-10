"""Disagreement must compare the same time and never score expired plans as zero."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from trajectory_jump_metrics import trajectory_jumps


class JumpMetricsTests(unittest.TestCase):
    def data(self,horizon=6):
        origins=np.array([0,2]);indices=np.array([0,0,1,1,1,1])
        chunks=np.zeros((2,horizon,14))
        for i,origin in enumerate(origins):chunks[i,:,0]=.01*(origin+np.arange(horizon))+.5*i
        requested=np.zeros((6,14))
        for step,i in enumerate(indices):requested[step]=chunks[i,step-origins[i]]
        return dict(state=np.zeros((6,14)),action_prediction_index=indices,prediction_steps=origins,
            predicted_chunks=chunks,requested_action=requested,blended_action=requested,
            submitted_action=requested,action=requested)

    def test_same_time_disagreement_excludes_nominal_motion(self):
        result=trajectory_jumps(self.data())
        self.assertAlmostEqual(result['same_time_raw_disagreement_rad']['max'],.5)
        self.assertAlmostEqual(result['command_switch_increments_rad']['requested_action']['max'],.51)
        self.assertEqual(result['old_plan_overlap_count'],1)

    def test_expired_old_plan_is_unavailable(self):
        data=self.data();data['prediction_steps'][0]=-4
        result=trajectory_jumps(data)
        self.assertIsNone(result['same_time_raw_disagreement_rad'])
        self.assertEqual(result['old_plan_overlap_unavailable'],1)


if __name__=='__main__':unittest.main()
