"""Optional scene damping checks; compile only, with no physics steps or rendering."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from gym_aloha.constants import ASSETS_DIR
import mujoco
import numpy as np

from pi0_repro.scene import SceneConfig, build_scene_xml, load_scene_config


WRISTS = {f'vx300s_{side}/{kind}' for side in ('left', 'right')
          for kind in ('forearm_roll', 'wrist_angle', 'wrist_rotate')}


class WristDampingTests(unittest.TestCase):
    def test_default_xml_matches_prechange_output_for_both_profiles(self):
        # Recorded before this option existed. Only installation-specific mesh
        # path prefixes are normalized; XML values and formatting stay exact.
        prechange = {
            'original': '8c1bd507221c04d860f5add2864cc025e5cf757c0a1e51801c49ebb5277c2b73',
            'position_pd': '68afcc225bf3f7e7691fd39aa26afc07045f383dcc9d95581d4c5bbe68551875',
        }
        for profile, expected in prechange.items():
            with self.subTest(profile=profile):
                implicit = build_scene_xml(SceneConfig(controller_profile=profile))
                explicit = build_scene_xml(SceneConfig(controller_profile=profile, wrist_damping_scale=1.0))
                self.assertEqual(implicit, explicit)
                normalized = implicit.replace(str(Path(ASSETS_DIR).resolve()), 'GYM_ALOHA_ASSETS')
                self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(), expected)

    def test_plus25_percent_changes_only_six_joint_damping_attributes(self):
        baseline = build_scene_xml(SceneConfig(controller_profile='position_pd'))
        adjusted = build_scene_xml(SceneConfig(controller_profile='position_pd', wrist_damping_scale=1.25))
        left, right = ET.fromstring(baseline), ET.fromstring(adjusted)
        changed = set()
        for old, new in zip(left.iter(), right.iter(), strict=True):
            self.assertEqual(old.tag, new.tag)
            self.assertEqual(old.text, new.text)
            self.assertEqual(old.tail, new.tail)
            if old.attrib != new.attrib:
                name = old.get('name')
                self.assertEqual(old.tag, 'joint')
                self.assertIn(name, WRISTS)
                self.assertEqual({key for key in old.attrib if old.get(key) != new.get(key)}, {'damping'})
                self.assertEqual(set(old.attrib), set(new.attrib))
                self.assertEqual(float(new.get('damping')), float(old.get('damping')) * 1.25)
                changed.add(name)
                # Restoring only these values must restore the complete XML.
                new.set('damping', old.get('damping'))
        self.assertEqual(changed, WRISTS)
        self.assertEqual(ET.tostring(right, encoding='unicode'), baseline)

    def test_compiled_model_changes_only_wrist_dofs_and_preserves_actuator_gains(self):
        baseline = mujoco.MjModel.from_xml_string(build_scene_xml(SceneConfig(controller_profile='position_pd')))
        adjusted = mujoco.MjModel.from_xml_string(build_scene_xml(
            SceneConfig(controller_profile='position_pd', wrist_damping_scale=1.25)))
        expected_dofs = set()
        for name in WRISTS:
            joint = mujoco.mj_name2id(baseline, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.assertGreaterEqual(joint, 0)
            dof = int(baseline.jnt_dofadr[joint])
            expected_dofs.add(dof)
            self.assertEqual(adjusted.dof_damping[dof], baseline.dof_damping[dof] * 1.25)
        self.assertEqual(set(np.flatnonzero(baseline.dof_damping != adjusted.dof_damping)), expected_dofs)
        for field in ('actuator_gainprm', 'actuator_biasprm', 'dof_frictionloss', 'body_gravcomp',
                      'jnt_range', 'geom_friction', 'geom_solref', 'geom_solimp'):
            np.testing.assert_array_equal(getattr(baseline, field), getattr(adjusted, field), err_msg=field)

    def test_scene_config_loads_and_serializes_without_client_config_duplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scene.toml'
            path.write_text('[scene]\ncontroller_profile = "position_pd"\nwrist_damping_scale = 1.25\n')
            settings = load_scene_config(path)
            self.assertEqual(settings.wrist_damping_scale, 1.25)
            self.assertEqual(json.loads(json.dumps(asdict(settings)))['wrist_damping_scale'], 1.25)
            path.write_text('[scene]\ncontroller_profile = "position_pd"\n')
            self.assertEqual(load_scene_config(path).wrist_damping_scale, 1.0)

    def test_invalid_scales_rejected_at_load_and_direct_build(self):
        literals = ('0', '-1', 'nan', 'inf', '-inf', 'true', '"1.25"', '[1.25]', '1e308')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scene.toml'
            for literal in literals:
                with self.subTest(literal=literal):
                    path.write_text('[scene]\ncontroller_profile = "position_pd"\nwrist_damping_scale = ' + literal + '\n')
                    with self.assertRaisesRegex(ValueError, 'wrist_damping_scale'):
                        load_scene_config(path)
        for value in (0, -1, float('nan'), float('inf'), True, '1.25', None, 1e308):
            with self.subTest(direct=value), patch('pi0_repro.scene._read_with_includes') as parse:
                with self.assertRaisesRegex(ValueError, 'wrist_damping_scale'):
                    build_scene_xml(SceneConfig(controller_profile='position_pd', wrist_damping_scale=value))
                parse.assert_not_called()

    def test_non_default_scale_rejected_for_original_controller(self):
        with self.assertRaisesRegex(ValueError, 'requires controller_profile=position_pd'):
            build_scene_xml(SceneConfig(controller_profile='original', wrist_damping_scale=1.25))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scene.toml'
            path.write_text('[scene]\nwrist_damping_scale = 1.25\n')
            with self.assertRaisesRegex(ValueError, 'requires controller_profile=position_pd'):
                load_scene_config(path)


if __name__ == '__main__':
    unittest.main()
