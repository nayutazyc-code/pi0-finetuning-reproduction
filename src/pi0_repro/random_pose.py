"""Seeded initial-pose sampling only; no policy or task-phase decisions."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene import SceneConfig


RANGE_FIELDS = ('banana_x_range', 'banana_y_range', 'banana_yaw_range_degrees')


def validate_pose_randomization(scene: SceneConfig) -> dict[str, tuple[float, float]]:
    if type(scene.randomize_banana_pose) is not bool:
        raise ValueError('randomize_banana_pose must be a boolean')
    ranges = {}
    for name in RANGE_FIELDS:
        values = getattr(scene, name)
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError(f'{name} must contain two finite increasing numbers')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values) or not values[0] < values[1]:
            raise ValueError(f'{name} must contain two finite increasing numbers')
        ranges[name] = tuple(float(value) for value in values)
    return ranges


def sample_banana_scene(scene: SceneConfig, rng=None) -> SceneConfig:
    """Return a replayable realized scene, using no RNG in the fixed branch.

    An enabled sample uses three independent uniform draws in x, y and yaw.
    Height, box pose and every other physical setting retain the requested values.
    The returned scene disables further randomization so it can be replayed as-is.
    """
    ranges = validate_pose_randomization(scene)
    if not scene.randomize_banana_pose:
        return replace(scene)
    if rng is None:
        raise ValueError('Randomized banana pose requires an explicit episode RNG')
    x, y, yaw = (float(rng.uniform(*ranges[name])) for name in RANGE_FIELDS)
    return replace(scene, randomize_banana_pose=False,
                   banana_position=(x, y, float(scene.banana_position[2])), banana_yaw_degrees=yaw)
