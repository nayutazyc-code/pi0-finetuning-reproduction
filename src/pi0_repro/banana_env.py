"""Banana placement environment for original-pi0 remote policy rollouts."""

import math
from dataclasses import asdict

from dm_control import mujoco as dm_mujoco
import gymnasium as gym
from gym_aloha.constants import (
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    unnormalize_puppet_gripper_position,
)
import mujoco
import numpy as np

from .scene import SceneConfig, build_scene_xml
from .random_pose import sample_banana_scene


class BananaBoxEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    control_dt = 0.02
    success_hold_seconds = 0.5
    linear_speed_threshold = 0.02  # m/s
    angular_speed_threshold = 0.5  # rad/s
    containment_tolerance = 0.003  # allow a few mm of soft-contact penetration
    lift_height = 0.025  # all banana geometry must clear the tabletop by 2.5 cm

    def __init__(self, scene: SceneConfig, *, max_episode_steps: int, terminate_on_success: bool = True, render_observations: bool = True, display_cameras: bool = False, reuse_policy_camera: bool = False, collect_step_timings: bool = False):
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps 必须大于 0。")
        self.scene = scene
        self.episode_scene = None
        self.last_reset = None
        self.display_cameras = display_cameras
        self.render_observations = render_observations
        self.reuse_policy_camera = reuse_policy_camera
        self.collect_step_timings = collect_step_timings
        self.last_step_timings = None
        self._policy_camera = None
        self.max_episode_steps = max_episode_steps
        self.terminate_on_success = bool(terminate_on_success)
        self.render_mode = "rgb_array"
        self.xml = build_scene_xml(scene, display_cameras=display_cameras)
        self.physics = dm_mujoco.Physics.from_xml_string(self.xml)
        try:
            model = self.physics.model
            joint = model.name2id("banana_joint", "joint")
            if model.nq != 23 or model.nu != 16 or int(model.jnt_qposadr[joint]) != 16:
                raise ValueError("香蕉场景的关节顺序与 ALOHA 适配器不匹配。")
            self.substeps = round(self.control_dt / float(model.opt.timestep))
            if not math.isclose(self.substeps * model.opt.timestep, self.control_dt):
                raise ValueError("物理步长必须能整除 0.02 秒控制周期。")
            self.hold_steps_required = math.ceil(self.success_hold_seconds / self.control_dt)
            self.banana_geoms = tuple(
                model.name2id(name, "geom") for name in (
                    "banana_segment_0", "banana_segment_1", "banana_segment_2", "banana_segment_3",
                    "banana_tip_left", "banana_tip_right",
                )
            )
            if any(int(model.geom_type[index]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE)
                   for index in self.banana_geoms):
                raise ValueError("当前完整入盒判定针对胶囊体香蕉；更换 mesh 后需同步更新判定。")
            self.banana_collision_geoms = set(self.banana_geoms[:4])
            self.box_geoms = {
                model.name2id(name, "geom") for name in (
                    "box_bottom", "box_wall_left", "box_wall_right", "box_wall_front", "box_wall_back",
                )
            }
            self.robot_geoms = {
                index for index in range(model.ngeom)
                if (model.id2name(index, "geom") or "").startswith(("vx300s_left/", "vx300s_right/"))
            }
            self.finger_pairs = tuple(
                {model.name2id(f"vx300s_{side}/10_{finger}_gripper_finger", "geom")
                 for finger in ("left", "right")}
                for side in ("left", "right")
            )
            self.box_body = model.name2id("open_box", "body")
            self.table_geom = model.name2id("table", "geom")
            self.inner_low, self.inner_high = self._box_interior()
            ranges = np.asarray(model.actuator_ctrlrange)
            limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
            self.ctrl_low = np.where(limited, ranges[:, 0], -np.inf)
            self.ctrl_high = np.where(limited, ranges[:, 1], np.inf)
            action_low = np.r_[self.ctrl_low[:6], 0, self.ctrl_low[8:14], 0].astype(np.float32)
            action_high = np.r_[self.ctrl_high[:6], 1, self.ctrl_high[8:14], 1].astype(np.float32)
            self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)
            self.observation_space = gym.spaces.Dict({
                "pixels": gym.spaces.Dict({"top": gym.spaces.Box(0, 255, (480, 640, 3), np.uint8)} if self.render_observations else {}),
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, (14,), np.float32),
            })
            self.steps = 0
            self.hold_steps = 0
            self.ever_grasped = False
            self.ever_lifted = False
            self._needs_reset = True
        except BaseException:
            self.close()
            raise

    def _box_interior(self):
        model = self.physics.model
        ids = {name: model.name2id("box_" + name, "geom")
               for name in ("bottom", "wall_left", "wall_right", "wall_front", "wall_back")}
        for index in ids.values():
            if int(model.geom_type[index]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("盒子的底板和四壁应为 box 几何体。")
            if not np.allclose(model.geom_quat[index], (1, 0, 0, 0)):
                raise ValueError("盒壁在盒子局部坐标中应与坐标轴对齐。")
        pos, size = model.geom_pos, model.geom_size
        left, right = ids["wall_left"], ids["wall_right"]
        front, back, bottom = ids["wall_front"], ids["wall_back"], ids["bottom"]
        low = np.array((pos[left, 0] + size[left, 0], pos[front, 1] + size[front, 1],
                        pos[bottom, 2] + size[bottom, 2]))
        high = np.array((pos[right, 0] - size[right, 0], pos[back, 1] - size[back, 1],
                         min(pos[ids[name], 2] + size[ids[name], 2]
                             for name in ("wall_left", "wall_right", "wall_front", "wall_back"))))
        if np.any(low >= high):
            raise ValueError("盒子内腔尺寸无效，请检查底板和墙的位置。")
        return low, high

    @staticmethod
    def _state_from_physical(positions):
        return np.r_[
            positions[:6], normalize_puppet_gripper_position(positions[6]),
            positions[8:14], normalize_puppet_gripper_position(positions[14]),
        ].astype(np.float32)

    def _observation(self):
        return {"pixels": {"top": self.render()} if self.render_observations else {},
                "agent_pos": self._state_from_physical(self.physics.data.qpos[:16])}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = True
        rng = self.np_random if self.scene.randomize_banana_pose else None
        self.episode_scene = sample_banana_scene(self.scene, rng)
        half_yaw = math.radians(self.episode_scene.banana_yaw_degrees) / 2
        quaternion = (math.cos(half_yaw), 0, 0, math.sin(half_yaw))
        with self.physics.reset_context():
            self.physics.data.qpos[:16] = START_ARM_POSE
            self.physics.data.ctrl[:] = START_ARM_POSE
            self.physics.data.qvel[:] = 0
            self.physics.named.data.qpos["banana_joint"] = (
                *self.episode_scene.banana_position, *quaternion
            )
        settle_steps = round(self.episode_scene.settle_seconds / float(self.physics.model.opt.timestep))
        if settle_steps:
            self.physics.step(nstep=settle_steps)
        self.physics.forward()
        self.steps = self.hold_steps = 0
        self.ever_grasped = self.ever_lifted = False
        self.last_reset = {
            'seed': int(seed) if seed is not None else None,
            'rng_seed': getattr(self, '_np_random_seed', None),
            'randomized': self.scene.randomize_banana_pose,
            'sample': {'position': list(self.episode_scene.banana_position),
                       'quaternion_wxyz': list(quaternion), 'yaw_degrees': self.episode_scene.banana_yaw_degrees},
            'settled': {'position': self.physics.named.data.xpos['banana'].tolist(),
                        'quaternion_wxyz': self.physics.named.data.xquat['banana'].tolist()},
            'scene': asdict(self.episode_scene),
        }
        observation = self._observation()
        info = self._task_info(update=False)
        self._needs_reset = False
        return observation, info

    def _banana_bounds(self):
        data, model = self.physics.data, self.physics.model
        box_position = data.xpos[self.box_body]
        box_rotation = data.xmat[self.box_body].reshape(3, 3)
        lows, highs, world_bottoms = [], [], []
        for index in self.banana_geoms:
            # A compiled capsule's local Z axis is its centerline; size[1] is half-length.
            axis = data.geom_xmat[index].reshape(3, 3)[:, 2]
            center = data.geom_xpos[index]
            half_length, radius = model.geom_size[index, 1], model.geom_size[index, 0]
            endpoints = np.stack((center - axis * half_length, center + axis * half_length))
            local = (endpoints - box_position) @ box_rotation
            lows.append(local.min(axis=0) - radius)
            highs.append(local.max(axis=0) + radius)
            world_bottoms.append(float(endpoints[:, 2].min() - radius))
        return np.min(lows, axis=0), np.max(highs, axis=0), min(world_bottoms)

    def _task_info(self, *, update: bool):
        low, high, bottom_z = self._banana_bounds()
        tolerance = self.containment_tolerance
        inside = bool(np.all(low >= self.inner_low - tolerance)
                      and np.all(high <= self.inner_high + tolerance))
        touching = set()
        for index in range(self.physics.data.ncon):
            contact = self.physics.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.banana_collision_geoms:
                touching.add(second)
            if second in self.banana_collision_geoms:
                touching.add(first)
        released = not bool(touching & self.robot_geoms)
        supported = bool(touching & self.box_geoms)
        lifted = bottom_z > self.lift_height
        grasped = lifted and any(pair <= touching for pair in self.finger_pairs)
        velocity = np.asarray(self.physics.named.data.qvel["banana_joint"])
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
        stable = linear_speed < self.linear_speed_threshold and angular_speed < self.angular_speed_threshold
        if update:
            self.ever_grasped |= bool(grasped)
            self.ever_lifted |= bool(lifted)
            candidate = self.ever_grasped and inside and released and supported and stable
            self.hold_steps = self.hold_steps + 1 if candidate else 0
        banana_position = np.asarray(self.physics.named.data.xpos["banana"])
        finger_midpoints = [np.asarray(self.physics.data.geom_xpos[list(pair)]).mean(axis=0)
                            for pair in self.finger_pairs]
        grasp_distance = min(float(np.linalg.norm(point - banana_position)) for point in finger_midpoints)
        return {
            # Ground-truth diagnostic only; never used to choose policy actions.
            "grasp_distance_m": grasp_distance,
            "near_grasp_region": grasp_distance < 0.04,
            "is_success": self.hold_steps >= self.hold_steps_required,
            "grasped": bool(grasped), "ever_grasped": self.ever_grasped,
            "lifted": bool(lifted), "ever_lifted": self.ever_lifted,
            "inside_box": inside, "released": released, "supported_by_box": supported,
            "stable": bool(stable), "touching_table": self.table_geom in touching,
            "success_hold_steps": self.hold_steps, "required_hold_steps": self.hold_steps_required,
            "linear_speed_m_s": linear_speed, "angular_speed_rad_s": angular_speed,
            "banana_position": self.physics.named.data.xpos["banana"].tolist(),
            "simulation_time_seconds": float(self.physics.data.time),
            "control_step": self.steps,
        }

    def _action_to_control(self, action):
        action = np.asarray(action, dtype=np.float32)
        left = unnormalize_puppet_gripper_position(float(np.clip(action[6], 0, 1)))
        right = unnormalize_puppet_gripper_position(float(np.clip(action[13], 0, 1)))
        return np.clip(np.r_[action[:6], left, -left, action[7:13], right, -right],
                       self.ctrl_low, self.ctrl_high)

    def project_action(self, action):
        """Pure target projection, including physical finger actuator bounds.

        Normalized gripper 1 can exceed the XML actuator limit. RTC must
        forecast the same native target that step() actually submits.
        """
        return self._state_from_physical(self._action_to_control(action))

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("请先 reset；episode 结束后也需要重新 reset。")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("香蕉任务需要有限数值的 14 维 ALOHA 动作。")
        control = self._action_to_control(action)
        executed = self._state_from_physical(control)
        self.physics.data.ctrl[:] = control
        if self.collect_step_timings:
            from .control_timing import measured_call
            self.last_step_timings = {}
            measured_call(self.last_step_timings, 'physics_step', self.physics.step, nstep=self.substeps)
            measured_call(self.last_step_timings, 'forward', self.physics.forward)
        else:
            self.physics.step(nstep=self.substeps)
            self.physics.forward()
        self.steps += 1
        info = (measured_call(self.last_step_timings, 'task_info', self._task_info, update=True)
                if self.collect_step_timings else self._task_info(update=True))
        info["executed_action"] = executed.tolist()
        info["action_clipped"] = not bool(np.allclose(action, executed, atol=1e-6, rtol=0))
        terminated = bool(info["is_success"]) and self.terminate_on_success
        truncated = self.steps >= self.max_episode_steps and not terminated
        self._needs_reset = terminated or truncated
        observation = (measured_call(self.last_step_timings, 'observation', self._observation)
                       if self.collect_step_timings else self._observation())
        return observation, float(info["is_success"]), terminated, truncated, info

    def render(self):
        if self.reuse_policy_camera:
            from dm_control.mujoco import engine
            if self._policy_camera is None:
                self._policy_camera = engine.Camera(self.physics, height=480, width=640, camera_id="top")
            # Camera.render reuses its pixel buffer. Preserve the existing
            # caller-owned frame lifetime across later renders and resets.
            return self._policy_camera.render().copy()
        return self.physics.render(height=480, width=640, camera_id="top")

    def render_camera(self, name, *, height=480, width=640):
        if not isinstance(name, str) or not name:
            raise ValueError('Camera name must be a nonempty string')
        if type(height) is not int or type(width) is not int or min(height, width) < 1:
            raise ValueError('Camera dimensions must be positive integers')
        if name not in {self.physics.model.id2name(index, 'camera')
                        for index in range(self.physics.model.ncam)}:
            raise ValueError('Unknown camera name: ' + name)
        return self.physics.render(height=height, width=width, camera_id=name)

    def close(self):
        try:
            if self._policy_camera is not None:
                camera, self._policy_camera = self._policy_camera, None
                camera._scene.free()
        finally:
            if self.physics is not None:
                self.physics.free()
                self.physics = None

    def description(self):
        return {
            "task": "pi0_repro/BananaBox-v0", "control_dt_seconds": self.control_dt,
            "success_rule": "both-finger contact while lifted at least once; then whole banana inside, no robot contact, box support, stable for hold duration",
            "success_hold_seconds": self.success_hold_seconds,
            "linear_speed_threshold_m_s": self.linear_speed_threshold,
            "angular_speed_threshold_rad_s": self.angular_speed_threshold,
            "containment_tolerance_m": self.containment_tolerance,
            "lift_height_m": self.lift_height,
            "layout_randomized": self.scene.randomize_banana_pose, "terminate_on_success": self.terminate_on_success,
        }
