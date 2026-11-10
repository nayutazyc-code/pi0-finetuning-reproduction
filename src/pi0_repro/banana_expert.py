"""Geometric expert: bounded IK and physical position-control actions only."""
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import tomllib

import mujoco
import numpy as np


@dataclass
class ExpertConfig:
    scene_config: str = "banana_demo_scene.toml"
    prompt: str = "Pick up the banana and place it inside the open box."
    max_steps: int = 2000
    approach_height: float = 0.16
    transport_height: float = 0.20
    grasp_local_x: float = 0.0
    grasp_local_y: float = -0.01
    grasp_z_offset: float = -0.014
    closed_gripper: float = 0.35
    cartesian_speed: float = 0.08
    joint_speed: float = 0.7
    close_seconds: float = 1.0
    open_seconds: float = 0.7
    settle_seconds: float = 0.8
    position_tolerance: float = 0.006
    precision_speed_scale: float = 1.0
    transit_speed_scale: float = 1.0
    direct_recovery_grasp: bool = False
    trajectory_shape: str = "cubic"
    ik_position_tolerance: float = 0.0015
    ik_rotation_tolerance: float = 0.025
    interpolate_orientation: bool = False
    respect_peak_speed: bool = False
    gripper_trajectory_shape: str = "linear"
    direct_recovery_release: bool = False
    direct_recovery_place: bool = False
    recovery_grasp_min_height: float | None = None
    recovery_target_bridge_steps: int = 0
    recovery_unload_steps: int = 0
    continuous_transport_radius: float = 0.0


def load_expert_config(path: Path) -> ExpertConfig:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"expert"}:
        raise ValueError("脚本专家配置应仅包含 [expert]。")
    config = ExpertConfig(**data["expert"])
    for name in ("direct_recovery_grasp", "interpolate_orientation", "respect_peak_speed",
                 "direct_recovery_release", "direct_recovery_place"):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be boolean")
    if config.trajectory_shape not in ("cubic", "quintic"):
        raise ValueError("trajectory_shape must be cubic or quintic")
    if config.gripper_trajectory_shape not in ("linear", "cubic", "quintic"):
        raise ValueError("gripper_trajectory_shape must be linear, cubic or quintic")
    for name in ("approach_height", "transport_height", "grasp_local_x", "grasp_local_y",
                 "grasp_z_offset", "closed_gripper", "cartesian_speed", "joint_speed", "close_seconds",
                 "open_seconds", "settle_seconds", "position_tolerance", "precision_speed_scale", "transit_speed_scale",
                 "ik_position_tolerance", "ik_rotation_tolerance"):
        if not math.isfinite(getattr(config, name)):
            raise ValueError(f"{name} 必须是有限数值。")
    if min(config.cartesian_speed, config.joint_speed, config.close_seconds,
           config.open_seconds, config.settle_seconds, config.position_tolerance,
           config.ik_position_tolerance, config.ik_rotation_tolerance) <= 0:
        raise ValueError("速度、时间和容差必须大于 0。")
    if not 0 < config.precision_speed_scale <= 1 or not 0 < config.transit_speed_scale <= 2:
        raise ValueError("Invalid precision/transit speed scale")
    if not 0 <= config.closed_gripper <= 1:
        raise ValueError("closed_gripper 必须在 0 到 1 之间。")
    if config.recovery_grasp_min_height is not None:
        value=config.recovery_grasp_min_height
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= .05:
            raise ValueError('recovery_grasp_min_height must be a finite height in [0,0.05] metres')
    if type(config.recovery_target_bridge_steps) is not int or not 0 <= config.recovery_target_bridge_steps <= 50:
        raise ValueError('recovery_target_bridge_steps must be an integer in [0,50]')
    if type(config.recovery_unload_steps) is not int or not 0 <= config.recovery_unload_steps <= 50:
        raise ValueError('recovery_unload_steps must be an integer in [0,50]')
    if config.recovery_unload_steps and config.recovery_target_bridge_steps:
        raise ValueError('Use either stationary recovery unloading or a moving target bridge')
    if (isinstance(config.continuous_transport_radius,bool)
            or not isinstance(config.continuous_transport_radius,(int,float))
            or not math.isfinite(config.continuous_transport_radius)
            or not 0 <= config.continuous_transport_radius <= .03):
        raise ValueError('continuous_transport_radius must be finite in [0,0.03] metres')
    if not isinstance(config.max_steps, int) or config.max_steps < 1:
        raise ValueError("max_steps 必须为正整数。")
    scene_path = Path(config.scene_config).expanduser()
    if not scene_path.is_absolute():
        scene_path = path.resolve().parent / scene_path
    config.scene_config = str(scene_path.resolve())
    return config


class IKError(RuntimeError):
    pass


def trajectory_alpha(t, shape):
    """Normalized interpolation, retaining the historical cubic expression."""
    if shape == "cubic":
        return t * t * (3 - 2 * t)
    if shape == "quintic":
        return t * t * t * (10 + t * (-15 + 6 * t))
    raise ValueError("Unknown trajectory shape")


def recovery_grasp_height(height, config, *, recovery_mode):
    if recovery_mode and config.recovery_grasp_min_height is not None:
        return max(height,config.recovery_grasp_min_height)
    return height


def trajectory_duration(distance, speed, minimum, shape, *, respect_peak, cartesian):
    """Time bounds for the interpolation parameter, not Cartesian-IK joint motion."""
    peak = {"cubic": 1.5, "quintic": 1.875}[shape]
    factor = peak if respect_peak else (1.5 if cartesian else 1.0)
    return max(minimum, float(distance) / speed * factor)


def interpolate_rotation(start, target, alpha):
    """Shortest-geodesic SO(3) interpolation using hemisphere-aligned quaternions.

    Interpolating orientation rather than issuing the final rotation immediately
    avoids an orientation-only target step at a recovery takeover. No Euler-angle
    interpolation or extra dependency is used. The chosen path is continuous at
    near-pi rotations; an exactly pi rotation has no unique shortest axis.
    """
    if alpha <= 0:
        return np.asarray(start).copy()
    if alpha >= 1:
        return np.asarray(target).copy()
    q0, q1 = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(q0, np.ascontiguousarray(start).ravel())
    mujoco.mju_mat2Quat(q1, np.ascontiguousarray(target).ravel())
    cosine = float(np.dot(q0, q1))
    if cosine < 0:
        q1 = -q1
        cosine = -cosine
    cosine = float(np.clip(cosine, 0, 1))
    if cosine > 0.9995:
        quaternion = (1 - alpha) * q0 + alpha * q1
    else:
        angle = math.acos(cosine)
        quaternion = (math.sin((1 - alpha) * angle) * q0 + math.sin(alpha * angle) * q1) / math.sin(angle)
    quaternion /= np.linalg.norm(quaternion)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


class ScriptedBananaExpert:
    site_name = "cali_right_site1"

    def __init__(self, env, config: ExpertConfig, execute, *, recovery_mode=False):
        self.env, self.config, self.execute = env, config, execute
        self.recovery_mode = recovery_mode
        self._bridge_index = 0
        self._bridge_initial = (env._state_from_physical(env.physics.data.ctrl).copy()
                                if recovery_mode and config.recovery_target_bridge_steps else None)
        self.planner = env.physics.copy(share_model=True)
        self.site = env.physics.model.name2id(self.site_name, "site")
        joints = [env.physics.model.name2id("vx300s_right/" + name, "joint") for name in (
            "waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")]
        self.q_indices = np.asarray([env.physics.model.jnt_qposadr[i] for i in joints])
        self.v_indices = np.asarray([env.physics.model.jnt_dofadr[i] for i in joints])
        self.low = np.asarray(env.physics.model.jnt_range[joints, 0]) + 1e-5
        self.high = np.asarray(env.physics.model.jnt_range[joints, 1]) - 1e-5
        self.last_q = np.asarray(env.physics.data.qpos[self.q_indices]).copy()
        self.action = env._state_from_physical(env.physics.data.qpos[:16])
        self.phase = "initial"
        self.target = None
        self.rotation = None

    def close(self):
        self.planner.free()

    @staticmethod
    def _rotation_error(current, target):
        current_q, target_q, inverse_q, delta = (np.empty(4) for _ in range(4))
        mujoco.mju_mat2Quat(current_q, np.ascontiguousarray(current).ravel())
        mujoco.mju_mat2Quat(target_q, np.ascontiguousarray(target).ravel())
        mujoco.mju_negQuat(inverse_q, current_q)
        mujoco.mju_mulQuat(delta, target_q, inverse_q)
        error = np.empty(3)
        mujoco.mju_quat2Vel(error, delta, 1.0)
        return error

    def _ik(self, position, rotation):
        physics = self.planner
        physics.data.qpos[:] = self.env.physics.data.qpos
        physics.data.qpos[self.q_indices] = self.last_q
        physics.data.qvel[:] = 0
        jac_p = np.zeros((3, physics.model.nv))
        jac_r = np.zeros_like(jac_p)
        weight = 0.15
        for _ in range(160):
            mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
            dp = position - physics.data.site_xpos[self.site]
            dr = self._rotation_error(physics.data.site_xmat[self.site].reshape(3, 3), rotation)
            if (np.linalg.norm(dp) < self.config.ik_position_tolerance
                    and np.linalg.norm(dr) < self.config.ik_rotation_tolerance):
                self.last_q = physics.data.qpos[self.q_indices].copy()
                return self.last_q.copy()
            mujoco.mj_jacSite(physics.model.ptr, physics.data.ptr, jac_p, jac_r, self.site)
            jac = np.vstack((jac_p[:, self.v_indices], weight * jac_r[:, self.v_indices]))
            error = np.r_[dp, weight * dr]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
            norm = np.linalg.norm(delta)
            if norm > 0.15:
                delta *= 0.15 / norm
            old = physics.data.qpos[self.q_indices].copy()
            cost = np.linalg.norm(error)
            improved = False
            for scale in (1.0, 0.5, 0.2, 0.05):
                physics.data.qpos[self.q_indices] = np.clip(old + scale * delta, self.low, self.high)
                mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
                new_p = position - physics.data.site_xpos[self.site]
                new_r = self._rotation_error(physics.data.site_xmat[self.site].reshape(3, 3), rotation)
                if np.linalg.norm(np.r_[new_p, weight * new_r]) < cost:
                    improved = True
                    break
            if not improved:
                break
        raise IKError(f"{self.phase}: IK 未收敛；位置误差 {np.linalg.norm(dp):.4f}m，旋转误差 {np.linalg.norm(dr):.3f}rad，目标 {position.tolist()}")

    def _step(self, joints, gripper):
        self.action[7:13] = joints
        self.action[13] = gripper
        submitted=self.action.copy()
        if self.config.recovery_target_bridge_steps and self._bridge_initial is not None and self._bridge_index < self.config.recovery_target_bridge_steps:
            self._bridge_index += 1
            alpha=trajectory_alpha(self._bridge_index/self.config.recovery_target_bridge_steps, 'quintic')
            submitted=self._bridge_initial+(submitted-self._bridge_initial)*alpha
        self.execute(submitted, self.phase, self.target)

    def _hold(self, seconds, gripper):
        start = float(self.action[13])
        count = max(1, math.ceil(seconds / self.env.control_dt))
        for index in range(count):
            alpha = (index + 1) / count
            if self.config.gripper_trajectory_shape != "linear":
                alpha = trajectory_alpha(alpha, self.config.gripper_trajectory_shape)
            self._step(self.last_q, start + (gripper - start) * alpha)

    def _move(self, position, *, initial=False):
        self.target = np.asarray(position, dtype=float)
        start_p = self.env.physics.data.site_xpos[self.site].copy()
        if initial:
            start_q = self.env.physics.data.qpos[self.q_indices].copy()
            goal = self._ik(self.target, self.rotation)
            seconds = trajectory_duration(np.max(np.abs(goal - start_q)), self.config.joint_speed,
                                          1.8, self.config.trajectory_shape,
                                          respect_peak=self.config.respect_peak_speed, cartesian=False)
            count = math.ceil(seconds / self.env.control_dt)
            for index in range(count):
                t = (index + 1) / count
                alpha = trajectory_alpha(t, self.config.trajectory_shape)
                self._step(start_q + alpha * (goal - start_q), float(self.action[13]))
            self.last_q = goal
        else:
            scale = (self.config.precision_speed_scale if self.phase in {"descend", "lower_into_box"}
                     else self.config.transit_speed_scale)
            seconds = trajectory_duration(np.linalg.norm(self.target - start_p), self.config.cartesian_speed * scale,
                                          0.5, self.config.trajectory_shape,
                                          respect_peak=self.config.respect_peak_speed, cartesian=True)
            count = math.ceil(seconds / self.env.control_dt)
            start_rotation = (self.env.physics.data.site_xmat[self.site].reshape(3, 3).copy()
                              if self.config.interpolate_orientation else None)
            for index in range(count):
                t = (index + 1) / count
                alpha = trajectory_alpha(t, self.config.trajectory_shape)
                point = start_p + alpha * (self.target - start_p)
                rotation = (interpolate_rotation(start_rotation, self.rotation, alpha)
                            if start_rotation is not None else self.rotation)
                joints = self._ik(point, rotation)
                self._step(joints, float(self.action[13]))
        for _ in range(100):
            actual = self.env.physics.data.site_xpos[self.site]
            if np.linalg.norm(actual - self.target) <= self.config.position_tolerance:
                return
            self._step(self.last_q, float(self.action[13]))
        raise RuntimeError(f"{self.phase}: 控制器未到达目标；TCP={self.env.physics.data.site_xpos[self.site].tolist()}")

    def _set_phase(self, name):
        self.phase = name
        logging.info("专家阶段：%s", name)

    def run(self):
        info = self.env._task_info(update=False)
        if (self.recovery_mode and self.config.recovery_unload_steps
                and info.get('touching_table') and not info.get('grasped')):
            self._unload_recovery_targets()
            info = self.env._task_info(update=False)
        if (self.recovery_mode and self.config.direct_recovery_release
                and info.get("inside_box") and info.get("supported_by_box")
                and info.get("stable") and not info.get("released")):
            # Offline labels for a replayed, supported-object state. Preserve
            # existing arm actuator targets during opening, not measured qpos.
            # Grasp/lift history remains whatever the source replay recorded.
            self.action = self.env._state_from_physical(self.env.physics.data.ctrl)
            self.last_q = self.action[7:13].copy()
            self.rotation = self.env.physics.data.site_xmat[self.site].reshape(3, 3).copy()
            self.target = self.env.physics.data.site_xpos[self.site].copy()
            retreat = self.target.copy()
            retreat[2] = self.config.transport_height
            self._release_and_retreat(retreat)
            return
        if info["grasped"]:
            # Contact displaces measured finger positions away from commanded
            # closure. Reissuing measured positions can open the grasp at takeover.
            self.action = self.env._state_from_physical(self.env.physics.data.ctrl)
            self.action[13] = min(float(self.action[13]), self.config.closed_gripper)
            # A learner-held object is already off the table. Keep its measured
            # orientation and compute resting height from geometry, not current z.
            self.rotation = self.env.physics.data.site_xmat[self.site].reshape(3, 3).copy()
            banana_z = float(self.env.physics.named.data.xpos["banana"][2])
            resting_height = banana_z - self.env._banana_bounds()[2]
            self._place(resting_height)
            return
        if not info["touching_table"]:
            raise RuntimeError("Recovery only supports table-resting or physically pinched bananas")
        config = self.config
        banana_position = self.env.physics.named.data.xpos["banana"].copy()
        banana_rotation = self.env.physics.named.data.xmat["banana"].reshape(3, 3)
        long_axis = banana_rotation[:, 0]
        yaw = math.atan2(long_axis[1], long_axis[0])
        x_axis = np.array((0.0, 0.0, -1.0))
        y_axis = np.array((math.sin(yaw), -math.cos(yaw), 0.0))
        self.rotation = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))
        grasp = banana_position + banana_rotation @ np.array((config.grasp_local_x, config.grasp_local_y, 0.0))
        grasp[2] += config.grasp_z_offset
        grasp[2] = recovery_grasp_height(grasp[2], config, recovery_mode=self.recovery_mode)
        hover = grasp.copy()
        hover[2] = config.approach_height
        direct = (self.recovery_mode and config.direct_recovery_grasp
                  and np.linalg.norm(self.env.physics.data.site_xpos[self.site] - grasp) < 0.05
                  and np.linalg.norm(self._rotation_error(
                      self.env.physics.data.site_xmat[self.site].reshape(3, 3), self.rotation)) < 0.35)
        if not direct:
            self._set_phase("open_gripper")
            self._hold(config.open_seconds, 1.0)
            self._set_phase("approach")
            self._move(hover, initial=True)
        self._set_phase("descend")
        self._move(grasp)
        self._set_phase("close_gripper")
        self._hold(config.close_seconds, config.closed_gripper)
        self._set_phase("lift")
        raised = grasp.copy()
        raised[2] = config.transport_height
        if config.continuous_transport_radius:
            self._continuous_pick_place(float(banana_position[2]),raised)
            return
        self._move(raised)
        info = self.env._task_info(update=False)
        if not info["grasped"]:
            raise RuntimeError("未确认两指夹持并抬起香蕉，保留为失败尝试。")
        self._place(float(banana_position[2]))

    def _unload_recovery_targets(self):
        # Offline preparation only: relax the old arm reference before moving
        # toward the grasp. Keep commanded grippers unchanged throughout.
        initial=self.env._state_from_physical(self.env.physics.data.ctrl).copy()
        goal=self.env._state_from_physical(self.env.physics.data.qpos[:16]).copy()
        goal[[6,13]]=initial[[6,13]]
        self._set_phase('unload_contact')
        self.target=self.env.physics.data.site_xpos[self.site].copy()
        for index in range(self.config.recovery_unload_steps):
            alpha=trajectory_alpha((index+1)/self.config.recovery_unload_steps,'quintic')
            self.execute(initial+(goal-initial)*alpha,self.phase,self.target)
        self.action=self.env._state_from_physical(self.env.physics.data.qpos[:16]).copy()
        self.action[[6,13]]=initial[[6,13]]
        self.last_q=self.action[7:13].copy()

    def _continuous_pick_place(self, resting_height, raised):
        # Offline labels only. Round the two high-clearance corners; retain
        # deliberate stops for grasp and release, and the existing retreat.
        from .expert_path import RoundedPath,SmoothCruise
        config=self.config
        start=self.env.physics.data.site_xpos[self.site].copy()
        start_rotation=self.env.physics.data.site_xmat[self.site].reshape(3,3).copy()
        offset=self.env.physics.named.data.xpos['banana']-start
        box=self.env.physics.named.data.xpos['open_box'].copy()
        over_box=box-offset;over_box[2]=config.transport_height
        place=over_box.copy();place[2]=box[2]+self.env.inner_low[2]+resting_height-offset[2]
        path=RoundedPath([start,raised,over_box,place],config.continuous_transport_radius)
        speed=config.cartesian_speed*min(config.precision_speed_scale,config.transit_speed_scale)
        clock=SmoothCruise(path.length,speed)
        count=math.ceil(clock.duration/self.env.control_dt)
        last_phase=None
        for index in range(count):
            s=clock.evaluate(min((index+1)*self.env.control_dt,clock.duration))
            phase=('lift' if s<path.bounds[1] else 'transport' if s<path.bounds[2] else 'lower_into_box')
            if phase!=last_phase:
                if last_phase=='lift' and not self.env._task_info(update=False)['grasped']:
                    raise RuntimeError('Continuous expert did not confirm a lifted pinch before transport')
                self._set_phase(phase);last_phase=phase
            self.target=path.evaluate(s)
            rotation=(interpolate_rotation(start_rotation,self.rotation,s/path.length)
                      if config.interpolate_orientation else self.rotation)
            self._step(self._ik(self.target,rotation),float(self.action[13]))
        self.target=place
        for _ in range(100):
            if np.linalg.norm(self.env.physics.data.site_xpos[self.site]-place)<=config.position_tolerance:break
            self._step(self.last_q,float(self.action[13]))
        else:raise RuntimeError('Continuous expert did not reach the placement target')
        self._release_and_retreat(over_box)

    def _place(self, resting_height):
        config = self.config
        # Preserve the measured object-to-TCP offset after grasping.
        offset = self.env.physics.named.data.xpos["banana"] - self.env.physics.data.site_xpos[self.site]
        box = self.env.physics.named.data.xpos["open_box"].copy()
        over_box = box - offset
        over_box[2] = config.transport_height
        direct = (self.recovery_mode and config.direct_recovery_place
                  and np.linalg.norm(self.env.physics.data.site_xpos[self.site][:2] - over_box[:2]) < 0.03)
        if not direct:
            self._set_phase("transport")
            self._move(over_box)
        self._set_phase("lower_into_box")
        place = over_box.copy()
        # Match the object's resting height on the box floor, then release it.
        place[2] = box[2] + self.env.inner_low[2] + resting_height - offset[2]
        self._move(place)
        self._release_and_retreat(over_box)

    def _release_and_retreat(self, over_box):
        config = self.config
        self._set_phase("release")
        self._hold(config.open_seconds, 1.0)
        self._set_phase("retreat")
        self._move(over_box)
        self._set_phase("settle")
        self._hold(config.settle_seconds, 1.0)
