"""Offline task-progress and target-continuity diagnostics; no policy decisions."""
import numpy as np

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def distribution(values):
    values = np.asarray(values, dtype=float)
    return ({"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
             "max": float(values.max())} if values.size else None)


def meaningful_reversals(values, threshold=0.05):
    """Count direction changes after excursions of threshold normalized units."""
    values = np.asarray(values)
    if not len(values):
        return 0
    low = high = float(values[0])
    direction = reversals = 0
    for value in values[1:]:
        value = float(value)
        if direction == 0:
            if value - low >= threshold:
                direction, high = 1, value
            elif high - value >= threshold:
                direction, low = -1, value
            else:
                low, high = min(low, value), max(high, value)
        elif direction > 0:
            high = max(high, value)
            if high - value >= threshold:
                reversals += 1
                direction, low = -1, value
        else:
            low = min(low, value)
            if value - low >= threshold:
                reversals += 1
                direction, high = 1, value
    return reversals


def task_progress(task_rows, actions, dt=0.02):
    first = lambda key: next((i for i, row in enumerate(task_rows) if row.get(key)), None)
    near, grasp, inside, success = [first(key) for key in
                                   ("near_grasp_region", "ever_grasped", "inside_box", "is_success")]
    seconds = lambda index: (index + 1) * dt if index is not None else None
    end = grasp + 1 if grasp is not None else len(task_rows)
    actions = np.asarray(actions)
    return {
        "first_near_grasp_seconds": seconds(near),
        "first_confirmed_lifted_pinch_seconds": seconds(grasp),
        "near_to_lifted_pinch_seconds": (grasp - near) * dt if near is not None and grasp is not None else None,
        "near_region_seconds_before_lifted_pinch": (sum(bool(r.get("near_grasp_region")) for r in task_rows[:end]) * dt
                                                   if any("near_grasp_region" in r for r in task_rows) else None),
        "first_inside_box_seconds": seconds(inside),
        "success_seconds": seconds(success),
        "gripper_command_reversals_before_lifted_pinch": {
            side: meaningful_reversals(actions[:end, axis]) if len(actions) else 0
            for side, axis in (("left", 6), ("right", 13))},
        "gripper_command_reversals_near_grasp_before_lifted_pinch": {
            side: meaningful_reversals(actions[near:end, axis]) if near is not None and len(actions) else None
            for side, axis in (("left", 6), ("right", 13))},
        "gripper_reversal_threshold_normalized": 0.05,
        "scope": "Simulator geometry is diagnostic only; near region uses a 0.04 m finger-midpoint to banana-center radius. Old logs without proximity fields report null, not zero.",
    }


def chunk_continuity(data):
    indices = np.asarray(data["action_prediction_index"])
    switches = np.diff(indices) != 0
    result = {"switches": int(switches.sum()), "delta_units": "radians per control step"}
    for key in ("requested_action", "blended_action", "submitted_action", "action"):
        if key not in data:
            continue
        delta = np.abs(np.diff(np.asarray(data[key])[:, JOINTS], axis=0))
        result[key] = {"at_switch": distribution(delta[switches]),
                       "within_chunk": distribution(delta[~switches])}
    return result
