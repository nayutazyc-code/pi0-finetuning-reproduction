import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pi0_repro.multiview import MultiViewComposer


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.composer = MultiViewComposer()
        self.front = np.full((480, 640, 3), [251, 7, 13], np.uint8)
        self.side = np.full((240, 320, 3), [17, 241, 29], np.uint8)
        self.wrist = np.full((240, 320, 3), [31, 37, 233], np.uint8)

    def compose(self, **keywords):
        return self.composer.compose(self.front, self.side, self.wrist, step=keywords.pop('step', 12),
                                     info=keywords.pop('info', {}), **keywords)

    def test_rgb_layout_dimensions_borders_and_separate_status_panel(self):
        frame = self.compose()
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertTrue(frame.flags.c_contiguous)
        np.testing.assert_array_equal(frame[400, 400], [251, 7, 13])
        np.testing.assert_array_equal(frame[120, 1120], [17, 241, 29])
        np.testing.assert_array_equal(frame[360, 1120], [31, 37, 233])
        np.testing.assert_array_equal(frame[700, 1260], self.composer._BACKGROUND)
        np.testing.assert_array_equal(frame[719, 400], self.composer._BORDER)
        np.testing.assert_array_equal(frame[479, 1120], self.composer._BORDER)

    def test_inputs_and_info_remain_unchanged_outputs_do_not_alias(self):
        originals = [array.copy() for array in (self.front, self.side, self.wrist)]
        info = {'simulation_time_seconds': 1.25, 'grasped': True, 'inside_box': False, 'is_success': False,
                'unused': {'value': [1, 2, 3]}}
        snapshot = copy.deepcopy(info)
        for image in (self.front, self.side, self.wrist):
            image.setflags(write=False)
        first = self.compose(info=info)
        second = self.compose(info=info)
        first[:] = 0
        self.assertFalse(np.shares_memory(first, second))
        self.assertTrue(np.any(second))
        for image, original in zip((self.front, self.side, self.wrist), originals):
            np.testing.assert_array_equal(image, original)
            self.assertFalse(np.shares_memory(second, image))
        self.assertEqual(info, snapshot)

    def test_wrist_uses_new_caller_frame_without_reusing_or_synthesizing_view(self):
        first = self.compose()
        self.wrist[:] = [193, 113, 53]
        second = self.compose()
        np.testing.assert_array_equal(second[360, 1120], [193, 113, 53])
        self.assertFalse(np.array_equal(first[240:479, 960:], second[240:479, 960:]))
        np.testing.assert_array_equal(first[:, :960], second[:, :960])
        np.testing.assert_array_equal(first[:239, 960:], second[:239, 960:])

    def test_invalid_rgb_channels_shapes_dtypes_and_nonfinite_float_rejected(self):
        for bad in (np.zeros((480, 640), np.uint8), np.zeros((480, 640, 4), np.uint8),
                    np.zeros((640, 480, 3), np.uint8), np.zeros((480, 640, 3), np.float32),
                    np.full((480, 640, 3), np.nan), [[0, 1, 2]]):
            with self.subTest(shape=getattr(bad, 'shape', None)):
                with self.assertRaisesRegex(ValueError, 'front must be a uint8 RGB'):
                    self.composer.compose(bad, self.side, self.wrist, step=0, info={})
        for name in ('side', 'wrist'):
            args = {'front': self.front, 'side': self.side, 'wrist': self.wrist}
            args[name] = np.zeros((480, 640, 3), np.uint8)
            with self.assertRaisesRegex(ValueError, name+' must be a uint8 RGB'):
                self.composer.compose(**args, step=0, info={})

    def test_noncontiguous_rgb_inputs_work_without_modifying_them(self):
        side = self.side[:, ::-1]
        frame = self.composer.compose(self.front, side, self.wrist, step=0, info={})
        np.testing.assert_array_equal(frame[120, 1120], side[120, 160])

    def test_status_changes_only_status_panel_and_distinguishes_unknown(self):
        unknown = self.compose()
        yes = self.compose(step=13, info={'simulation_time_seconds': 1.5, 'grasped': True,
                                        'inside_box': True, 'is_success': True})
        no = self.compose(info={'grasped': False, 'inside_box': False, 'is_success': False})
        np.testing.assert_array_equal(unknown[:, :960], yes[:, :960])
        np.testing.assert_array_equal(unknown[:480, 960:], yes[:480, 960:])
        self.assertFalse(np.array_equal(unknown[480:, 960:], yes[480:, 960:]))
        self.assertFalse(np.array_equal(unknown[480:, 960:], no[480:, 960:]))

    def test_step_time_and_flags_require_valid_scalar_metadata(self):
        for step in (-1, True, 1.5, '12'):
            with self.assertRaises(ValueError):
                self.compose(step=step)
        for seconds in (float('nan'), float('inf'), -1, True, '1.0', None):
            with self.assertRaises(ValueError):
                self.compose(info={'simulation_time_seconds': seconds})
        with self.assertRaises(ValueError):
            self.compose(info=[])
        with self.assertRaises(ValueError):
            self.compose(info={'grasped': 'yes'})
        self.assertEqual(self.compose(step=np.int64(3), info={'grasped': np.bool_(True)}).shape, (720, 1280, 3))

    def test_describe_is_json_safe_and_explicitly_excludes_policy_cam_high(self):
        metadata = self.composer.describe()
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        self.assertTrue(metadata['display_only'])
        self.assertFalse(metadata['policy_frame_included'])
        self.assertEqual(metadata['excluded_policy_camera'], 'cam_high')
        self.assertEqual(metadata['panels_xywh']['wrist'], [960, 240, 320, 240])
        metadata['panels_xywh']['wrist'][0] = 0
        self.assertEqual(self.composer.describe()['panels_xywh']['wrist'][0], 960)

    def test_font_lookup_failure_uses_installed_pillow_default(self):
        from pi0_repro import multiview
        fallback = multiview.ImageFont.load_default()
        with mock.patch.object(multiview.ImageFont, 'truetype', side_effect=OSError), \
             mock.patch.object(multiview.ImageFont, 'load_default', return_value=fallback):
            composer = MultiViewComposer()
        self.assertEqual(composer.compose(self.front, self.side, self.wrist, step=0, info={}).dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
