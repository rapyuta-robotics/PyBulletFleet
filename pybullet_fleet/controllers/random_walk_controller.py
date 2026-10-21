"""Random walk controller."""

from __future__ import annotations

import math
import random

from pybullet_fleet._replan import refused_until_stopped
from pybullet_fleet.controller import Controller
from pybullet_fleet.geometry import Pose


class RandomWalkController(Controller):
    """Move to random nearby positions within a radius of the starting point.

    Args:
        radius: Maximum distance from origin for random targets.
        wait_range: (min, max) seconds to wait at each target.
    """

    _registry_name = "random_walk"

    def __init__(
        self,
        params=None,
        radius: float = 5.0,
        wait_range: tuple | list = (1.0, 5.0),
    ):
        super().__init__(params)
        self._radius = radius
        self._wait_range = tuple(wait_range)
        self._origin: list | None = None
        self._wait_timer = 0.0

    def compute(self, agent, dt: float) -> bool:
        if self._origin is None:
            pos = agent.get_pose().position
            self._origin = [float(pos[0]), float(pos[1])]

        if self._wait_timer > 0:
            self._wait_timer -= dt
            return False

        if not agent.is_moving:
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, self._radius)
            z = float(agent.get_pose().position[2])
            target = [
                self._origin[0] + dist * math.cos(angle),
                self._origin[1] + dist * math.sin(angle),
                z,
            ]
            # A refused goal (see Agent.set_path) is replaced on the next update.
            goal = Pose.from_xyz(*target)
            if agent.set_goal_pose(goal) is not False or not refused_until_stopped(agent, [goal]):
                self._wait_timer = random.uniform(*self._wait_range)

        return False
