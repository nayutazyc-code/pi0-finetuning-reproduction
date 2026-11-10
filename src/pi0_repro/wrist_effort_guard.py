"""Optional sustained tracking-effort guard, without task/contact inputs."""
import math


class SustainedEffortState:
    def __init__(self, limit_nm=1., high_steps=3, release_steps=3):
        if isinstance(limit_nm,bool) or not math.isfinite(limit_nm) or limit_nm<=0:
            raise ValueError('Effort limit must be finite and positive')
        if type(high_steps) is not int or high_steps<1 or type(release_steps) is not int or release_steps<1:
            raise ValueError('Effort dwell counts must be positive integers')
        self.limit_nm=limit_nm;self.high_steps=high_steps;self.release_steps=release_steps
        self.reset()

    def reset(self):self.active=False;self.high=0;self.low=0

    def update(self, demand_nm):
        if not math.isfinite(demand_nm):raise ValueError('Non-finite actuator demand')
        if not self.active:
            self.high=self.high+1 if abs(demand_nm)>self.limit_nm else 0
            if self.high>=self.high_steps:self.active=True;self.low=0
        else:
            self.low=self.low+1 if abs(demand_nm)<self.limit_nm*.5 else 0
            if self.low>=self.release_steps:self.reset()
        return self.active


class WristEffortGuard:
    """Bind one fixed-gain, affine-bias, unit-gear right wrist position servo.

    Samples the previous command's post-step demand before the next command.
    The unbounded demand remains observable while applied effort is capped.
    """
    def __init__(self, physics, *, limit_nm=1.):
        import mujoco
        import numpy as np
        self.physics=physics;model=physics.model
        joint=model.name2id('vx300s_right/wrist_angle','joint')
        ids=np.flatnonzero(np.asarray(model.actuator_trnid)[:,0]==joint)
        if len(ids)!=1:raise ValueError('Expected exactly one wrist actuator')
        self.actuator=int(ids[0]);a=self.actuator
        if (int(model.actuator_gaintype[a])!=int(mujoco.mjtGain.mjGAIN_FIXED) or
            int(model.actuator_biastype[a])!=int(mujoco.mjtBias.mjBIAS_AFFINE) or
            int(model.jnt_type[joint])!=int(mujoco.mjtJoint.mjJNT_HINGE) or
            not np.array_equal(model.actuator_gear[a],[1,0,0,0,0,0])):
            raise ValueError('Effort guard requires a unit-gear hinge position servo')
        self.original_limited=bool(model.actuator_forcelimited[a])
        self.original_range=np.asarray(model.actuator_forcerange[a]).copy()
        self.state=SustainedEffortState(limit_nm=limit_nm)

    def reset(self):
        self.state.reset();self._apply(False)

    def _apply(self, active):
        m=self.physics.model;a=self.actuator
        if active:
            m.actuator_forcelimited[a]=1;m.actuator_forcerange[a]=[-self.state.limit_nm,self.state.limit_nm]
        else:
            m.actuator_forcelimited[a]=self.original_limited;m.actuator_forcerange[a]=self.original_range

    def before_step(self):
        m=self.physics.model;d=self.physics.data;a=self.actuator
        bias=m.actuator_biasprm[a]
        demand=float(m.actuator_gainprm[a,0]*d.ctrl[a]+bias[0]+bias[1]*d.actuator_length[a]+bias[2]*d.actuator_velocity[a])
        active=self.state.update(demand);self._apply(active)
        return dict(active=active,unclipped_previous_command_demand_nm=demand,limit_nm=self.state.limit_nm,
                    high_dwell_steps=self.state.high,release_dwell_steps=self.state.low)
