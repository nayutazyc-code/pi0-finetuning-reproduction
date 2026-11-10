"""Optional local, stateless Ruckig calculations with copyable numeric state."""
import numpy as np


class JerkLimitedArmReference:
    """Track joint targets with bounded kinematic derivatives, excluding grippers.

    State-to-state calculations are local; intermediate/cloud waypoints are not
    used. Targets may carry a reference velocity, with zero terminal acceleration. Physical contact
    response and subsequent hard actuator projection are outside this guarantee.
    """

    def __init__(self, position, max_velocity, max_acceleration, max_jerk, dt=0.02):
        from ruckig import Ruckig  # Fail at setup when the optional extra is missing.
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros_like(self.position)
        self.acceleration = np.zeros_like(self.position)
        self.max_velocity, self.max_acceleration, self.max_jerk = max_velocity, max_acceleration, max_jerk
        self.dt = dt
        self.projection_resets = 0

    def advance(self, target, target_velocity=None):
        from ruckig import InputParameter, Result, Ruckig, Synchronization, Trajectory
        n = len(self.position)
        current = InputParameter(n)
        current.current_position = self.position.tolist()
        current.current_velocity = self.velocity.tolist()
        current.current_acceleration = self.acceleration.tolist()
        current.target_position = np.asarray(target, dtype=np.float64).tolist()
        current.target_velocity = (np.clip(np.asarray(target_velocity, np.float64),
                                          -self.max_velocity, self.max_velocity).tolist()
                                   if target_velocity is not None else [0.0] * n)
        current.target_acceleration = [0.0] * n
        current.max_velocity = [self.max_velocity] * n
        current.max_acceleration = [self.max_acceleration] * n
        current.max_jerk = [self.max_jerk] * n
        current.synchronization = Synchronization.No
        trajectory = Trajectory(n)
        result = Ruckig(n, self.dt).calculate(current, trajectory)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig failed to calculate the arm reference: {result}")
        position, velocity, acceleration = trajectory.at_time(self.dt)
        self.position = np.asarray(position, np.float64)
        self.velocity = np.asarray(velocity, np.float64)
        self.acceleration = np.asarray(acceleration, np.float64)
        return self.position.copy()

    def update_executed(self, executed):
        executed = np.asarray(executed, np.float64)
        clipped = np.abs(executed - self.position) > 1e-6
        if clipped.any():
            # A hard actuator stop overrides smoothness. Restart those axes from
            # rest rather than forecasting from an unattained command position.
            self.velocity[clipped] = 0
            self.acceleration[clipped] = 0
            self.projection_resets += int(clipped.sum())
        self.position = executed.copy()
