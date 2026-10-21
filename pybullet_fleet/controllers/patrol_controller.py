"""Waypoint patrol controller."""

from __future__ import annotations

from pybullet_fleet._replan import refused_until_stopped
from pybullet_fleet.controller import Controller
from pybullet_fleet.geometry import Pose
from pybullet_fleet.logging_utils import get_lazy_logger

logger = get_lazy_logger(__name__)


class PatrolController(Controller):
    """Cycle through waypoints. Delegates actual movement to the base controller
    by calling ``agent.set_goal_pose()``.

    Args:
        waypoints: List of [x, y, z] positions.
        wait_time: Seconds to wait at each waypoint before advancing.
        loop: If True, restart from first waypoint after reaching last.
    """

    _registry_name = "patrol"

    def __init__(
        self,
        params=None,
        waypoints: list | None = None,
        wait_time: float = 0.0,
        loop: bool = True,
    ):
        super().__init__(params)
        self._waypoints = waypoints or []
        self._current_idx = 0
        self._wait_time = wait_time
        self._wait_timer = 0.0
        self._loop = loop
        self._started = False

    def compute(self, agent, dt: float) -> bool:
        if not self._waypoints:
            return False

        if self._wait_timer > 0:
            self._wait_timer -= dt
            return False

        # First call: set initial waypoint
        if not self._started:
            target = self._waypoints[self._current_idx]
            # Refused while the agent is still moving somewhere it cannot turn
            # from (see Agent.set_path): try again on the next update.
            goal = Pose.from_xyz(*target)
            self._started = agent.set_goal_pose(goal) is not False or not refused_until_stopped(agent, [goal])
            return False

        # Advance when agent stops moving
        if not agent.is_moving:
            next_idx = self._current_idx + 1
            if next_idx >= len(self._waypoints):
                if self._loop:
                    next_idx = 0
                else:
                    return False  # done
            target = self._waypoints[next_idx]
            # Advance only once the goal is taken; a refused one (see
            # Agent.set_path) is tried again on the next update.
            goal = Pose.from_xyz(*target)
            if agent.set_goal_pose(goal) is not False or not refused_until_stopped(agent, [goal]):
                self._current_idx = next_idx
                self._wait_timer = self._wait_time

        return False  # actual movement handled by base controller
