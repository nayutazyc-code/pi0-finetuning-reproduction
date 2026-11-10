"""A scalar candidate must change only the intended compiled position servo."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from gym_aloha.constants import ASSETS_DIR
import mujoco
import numpy as np
from pi0_repro.scene import SceneConfig,build_scene_xml,load_scene_config


class RightWristKpTests(unittest.TestCase):
    def test_default_xml_retains_original_bytes(self):
        expected={'original':'8c1bd507221c04d860f5add2864cc025e5cf757c0a1e51801c49ebb5277c2b73',
                  'position_pd':'68afcc225bf3f7e7691fd39aa26afc07045f383dcc9d95581d4c5bbe68551875'}
        for profile,digest in expected.items():
            xml=build_scene_xml(SceneConfig(controller_profile=profile))
            self.assertEqual(hashlib.sha256(xml.replace(str(Path(ASSETS_DIR).resolve()),'GYM_ALOHA_ASSETS').encode()).hexdigest(),digest)

    def test_xml_and_compiled_changes_are_one_servo_only(self):
        a=build_scene_xml(SceneConfig(controller_profile='position_pd'))
        b=build_scene_xml(SceneConfig(controller_profile='position_pd',right_wrist_angle_kp_scale=.75))
        left,right=ET.fromstring(a),ET.fromstring(b);changes=[]
        for old,new in zip(left.iter(),right.iter(),strict=True):
            if old.attrib!=new.attrib:
                changes.append(new);self.assertEqual(new.tag,'position');self.assertEqual(new.get('joint'),'vx300s_right/wrist_angle')
                self.assertEqual(float(new.get('kp')),150);new.set('kp',old.get('kp'))
        self.assertEqual(len(changes),1);self.assertEqual(ET.tostring(left),ET.tostring(right))
        x,y=mujoco.MjModel.from_xml_string(a),mujoco.MjModel.from_xml_string(b)
        joint=mujoco.mj_name2id(x,mujoco.mjtObj.mjOBJ_JOINT,'vx300s_right/wrist_angle')
        actuator=int(np.flatnonzero(x.actuator_trnid[:,0]==joint)[0])
        self.assertEqual(np.argwhere(x.actuator_gainprm!=y.actuator_gainprm).tolist(),[[actuator,0]])
        self.assertEqual(np.argwhere(x.actuator_biasprm!=y.actuator_biasprm).tolist(),[[actuator,1]])
        self.assertEqual(y.actuator_gainprm[actuator,0],150);self.assertEqual(y.actuator_biasprm[actuator,1],-150)
        for field in ['dof_damping','dof_frictionloss','jnt_range','geom_friction','geom_solref','geom_solimp','body_mass','body_gravcomp','actuator_ctrlrange']:
            np.testing.assert_array_equal(getattr(x,field),getattr(y,field),err_msg=field)

    def test_invalid_scales_and_wrong_profile_rejected(self):
        for value in [0,-1,True,None,'0.75',float('inf'),float('nan'),1e308]:
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'right_wrist_angle_kp_scale'):
                build_scene_xml(SceneConfig(controller_profile='position_pd',right_wrist_angle_kp_scale=value))
        with self.assertRaisesRegex(ValueError,'position_pd'):
            build_scene_xml(SceneConfig(right_wrist_angle_kp_scale=.75))

    def test_toml_defaults_and_candidate_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scene.toml';p.write_text('[scene]\ncontroller_profile="position_pd"\nright_wrist_angle_kp_scale=0.75\n')
            self.assertEqual(load_scene_config(p).right_wrist_angle_kp_scale,.75)
            p.write_text('[scene]\n');self.assertEqual(load_scene_config(p).right_wrist_angle_kp_scale,1.)


if __name__=='__main__':unittest.main()
