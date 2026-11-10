"""Recovery-only grasp clearance never changes ordinary or default experts."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from pi0_repro.banana_expert import ExpertConfig,ScriptedBananaExpert,load_expert_config,recovery_grasp_height,trajectory_alpha
from pi0_repro.config import PROJECT_ROOT


class RecoveryClearanceTests(unittest.TestCase):
    def test_default_and_ordinary_experts_unchanged(self):
        default=ExpertConfig();enabled=ExpertConfig(recovery_grasp_min_height=.008)
        for height in (-.005,.002,.01,.05):
            self.assertEqual(recovery_grasp_height(height,default,recovery_mode=True),height)
            self.assertEqual(recovery_grasp_height(height,enabled,recovery_mode=False),height)

    def test_floor_only_changes_low_recovery_target(self):
        cfg=ExpertConfig(recovery_grasp_min_height=.008)
        self.assertEqual(recovery_grasp_height(.002,cfg,recovery_mode=True),.008)
        self.assertEqual(recovery_grasp_height(.012,cfg,recovery_mode=True),.012)

    def test_config_rejects_invalid_height(self):
        source=(PROJECT_ROOT/'configs/banana_motion_expert.toml').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.toml';p.write_text(source+'\nrecovery_grasp_min_height = 0.008\n')
            self.assertEqual(load_expert_config(p).recovery_grasp_min_height,.008)
            for value in ('true','"0.008"','nan','inf','-0.001','0.051'):
                p.write_text(source+'\nrecovery_grasp_min_height = '+value+'\n')
                with self.assertRaises(ValueError):load_expert_config(p)

    def test_bridge_uses_prior_target_and_finishes_without_changing_default(self):
        initial=np.ones(14)*.2;goal=np.ones(14)*.4;output=[]
        expert=object.__new__(ScriptedBananaExpert)
        expert.config=ExpertConfig(recovery_target_bridge_steps=10)
        expert._bridge_initial=initial.copy();expert._bridge_index=0;expert.action=goal.copy()
        expert.phase='descend';expert.target=np.zeros(3);expert.execute=lambda action,*a:output.append(action.copy())
        for _ in range(11):expert._step(goal[7:13],goal[13])
        np.testing.assert_allclose(output[0],initial+(goal-initial)*trajectory_alpha(.1,'quintic'))
        np.testing.assert_array_equal(output[9],goal);np.testing.assert_array_equal(output[10],goal)
        np.testing.assert_array_equal(expert.action,goal)
        expert._bridge_initial=None;expert._step(goal[7:13],goal[13]);np.testing.assert_array_equal(output[-1],goal)

    def test_bridge_config_requires_bounded_integer(self):
        source=(PROJECT_ROOT/'configs/banana_motion_expert.toml').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.toml'
            for value in ('true','1.5','-1','51'):
                p.write_text(source+'\nrecovery_target_bridge_steps = '+value+'\n')
                with self.assertRaises(ValueError):load_expert_config(p)

    def test_unload_keeps_grippers_and_ends_at_original_measured_arm_pose(self):
        initial=np.full(14,.2);measured=np.full(14,.3);output=[]
        expert=object.__new__(ScriptedBananaExpert);expert.config=ExpertConfig(recovery_unload_steps=10)
        expert.env=SimpleNamespace(_state_from_physical=lambda x:x.copy(),physics=SimpleNamespace(data=SimpleNamespace(ctrl=initial,qpos=measured,site_xpos=np.zeros((1,3)))))
        expert.site=0;expert.execute=lambda action,*a:output.append(action.copy());expert._unload_recovery_targets()
        self.assertEqual(len(output),10)
        for a in output:np.testing.assert_array_equal(a[[6,13]],initial[[6,13]])
        end=measured.copy();end[[6,13]]=initial[[6,13]];np.testing.assert_array_equal(output[-1],end)
        np.testing.assert_array_equal(expert.last_q,measured[7:13])

    def test_unload_rejects_invalid_and_conflicting_modes(self):
        source=(PROJECT_ROOT/'configs/banana_motion_expert.toml').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.toml'
            for extra in ('recovery_unload_steps = true','recovery_unload_steps = 51','recovery_unload_steps = 10\nrecovery_target_bridge_steps = 10'):
                p.write_text(source+'\n'+extra+'\n')
                with self.assertRaises(ValueError):load_expert_config(p)


if __name__=='__main__':unittest.main()
