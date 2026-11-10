from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'server'))
from banana_event_sampling import event_window_weights


class EventSamplingTests(unittest.TestCase):
    def test_true_entries_and_clipped_windows(self):
        phases=np.array(['release']*10+['retreat']*10+['close_gripper']*10)
        weights,events=event_window_weights(phases,before=3,after=4)
        np.testing.assert_array_equal(events,[0,20]);np.testing.assert_array_equal(weights[:4],2)
        self.assertEqual(weights[-1],1);self.assertEqual(weights[16],1);self.assertEqual(weights[17],2)

    def test_overlap_not_multiplicative(self):
        weights,events=event_window_weights(['close_gripper','release','retreat'],before=2,after=3,boost=2)
        np.testing.assert_array_equal(events,[0,1]);np.testing.assert_array_equal(weights,2)

    def test_no_events_keeps_positive_weights(self):
        weights,events=event_window_weights(['transport']*5)
        np.testing.assert_array_equal(weights,1);self.assertEqual(len(events),0)


if __name__=='__main__':unittest.main()
