"""Offline motion-first diagnostics; geometry and thresholds never choose actions."""
import numpy as np

ARM_JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def _distribution(values):
    values = np.asarray(values, float)
    if not values.size:
        return None
    return {'p50': float(np.quantile(values, .5)), 'p95': float(np.quantile(values, .95)),
            'p99': float(np.quantile(values, .99)), 'max': float(values.max())}


def _spans(mask, dt, minimum_seconds):
    mask = np.asarray(mask, bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [{'start_seconds': float(start*dt), 'end_seconds': float(end*dt),
             'duration_seconds': float((end-start)*dt)} for start,end in zip(starts,ends)
            if (end-start)*dt >= minimum_seconds - 1e-9]


def motion_quality(states, actions, task_rows, *, dt=.02, active_speed=.01,
                   pause_joint_speed=.015, pause_gripper_speed=.05, pause_min_seconds=.5):
    """Frame-max active jerk and low-motion dwell with separate task coverage.

    Derivatives use simulation time. Pause labels describe observed low movement,
    not intent: deliberate holds may be included; the required stable-success hold
    is excluded. Gripper motion is tracked separately and excludes arm-only still
    periods from the all-command-idle measure. A stationary failure is unscored
    for active-motion smoothness, never assigned a perfect score.
    """
    states, actions = np.asarray(states, float), np.asarray(actions, float)
    if states.ndim != 2 or states.shape[1] != 14 or actions.shape != states.shape or len(states) < 4:
        raise ValueError('Expected matching Nx14 states/actions with at least four frames')
    if len(task_rows) != len(states) or not np.isfinite(states).all() or not np.isfinite(actions).all() or dt <= 0:
        raise ValueError('Invalid finite synchronized recording')
    q = states[:, ARM_JOINTS]
    delta = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    velocity = delta / dt
    acceleration = np.max(np.abs(np.diff(q, n=2, axis=0)), axis=1) / dt**2
    jerk = np.max(np.abs(np.diff(q, n=3, axis=0)), axis=1) / dt**3
    active = velocity[2:] > active_speed
    joint_speed = np.r_[np.inf, velocity]
    gripper_speed = np.r_[np.inf, np.max(np.abs(np.diff(actions[:, [6,13]], axis=0)), axis=1)/dt]
    success_hold = np.array([bool(t.get('inside_box') and t.get('released') and t.get('supported_by_box') and t.get('stable')) for t in task_rows])
    all_idle = (joint_speed < pause_joint_speed) & (gripper_speed < pause_gripper_speed) & ~success_hold
    phases = np.array(['inside_before_success' if t.get('inside_box') else
                       'carrying_before_box' if t.get('ever_grasped') else
                       'near_grasp_before_lift' if t.get('near_grasp_region') else 'approach'
                       for t in task_rows])
    pauses = _spans(all_idle, dt, pause_min_seconds)
    phase_pauses = {phase: _spans(all_idle & (phases == phase), dt, pause_min_seconds)
                    for phase in ('approach','near_grasp_before_lift','carrying_before_box','inside_before_success')}
    ever_grasped = any(t.get('ever_grasped') for t in task_rows)
    ever_inside = any(t.get('inside_box') for t in task_rows)
    success = any(t.get('is_success') for t in task_rows)
    return {'schema':'motion-first-v1', 'active_frame_max_jerk_rad_s3':_distribution(jerk[active]),
            'all_frame_max_jerk_rad_s3':_distribution(jerk),
            'frame_max_acceleration_rad_s2':_distribution(acceleration),
            'physical_joint_increment_rad':_distribution(delta),
            'active_motion_seconds':float(active.sum()*dt),
            'arm_path_l1_rad':float(np.abs(np.diff(q,axis=0)).sum()),
            'low_motion_pause_count':len(pauses),
            'low_motion_pause_total_seconds':float(sum(s['duration_seconds'] for s in pauses)),
            'low_motion_pause_max_seconds':float(max((s['duration_seconds'] for s in pauses),default=0)),
            'low_motion_pause_spans':pauses,'phase_pause_spans':phase_pauses,
            'coverage':{'lifted_pinch':ever_grasped,'entered_box':ever_inside,'success':success,
                        'transport_motion_comparable':ever_grasped and ever_inside,
                        'active_motion_scored':bool(active.any())},
            'thresholds':{'active_joint_speed_rad_s':active_speed,'pause_joint_speed_rad_s':pause_joint_speed,
                          'pause_gripper_speed_normalized_s':pause_gripper_speed,'pause_min_seconds':pause_min_seconds},
            'scope':'Offline frame-max derivatives, not a wall-clock servo guarantee. Low-motion dwell is not uniquely a policy stall. Do not rank stationary/incomplete coverage as smooth successful transport.'}
