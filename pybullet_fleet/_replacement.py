"""What a new path does to the current motion: the one ``goal_replacement`` decision.

Both kinematic controller families -- per-agent (``KinematicController``) and
batched (``BatchKinematicController``) -- take a new path through
:meth:`GoalReplacementPolicy._replace_path`, so the rules cannot drift apart
between them. A controller supplies only where its state lives: its limits,
whether it is following a path, how it holds, drops and starts a request, and
its braking flag.

The rules (see :class:`~pybullet_fleet.types.GoalReplacement`):

- ``wait_complete`` holds every path while one is in progress.
- A path the agent can continue its motion into (any path at rest) starts at
  once, continuing it.
- Otherwise ``reject`` refuses it, ``immediate`` starts it from rest, and
  ``brake`` -- or ``wait_complete`` with nothing to wait for (a velocity
  command, or :meth:`Agent.brake <pybullet_fleet.agent.Agent.brake>`
  preempting) -- brakes along the motion and holds it until the agent stops.

The mode is read for each request: changing it affects the paths requested
afterwards. A held path keeps the decision made when it was held and starts
once the agent stops (or the current path ends), whatever the mode is then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

import math

from pybullet_fleet._motion_planning import heading_toward
from pybullet_fleet._replan import HEADING_TOL, REST_SPEED, PathRequest, replacement_rejection
from pybullet_fleet.geometry import quat_angle_between
from pybullet_fleet.logging_utils import get_lazy_logger
from pybullet_fleet.types import GoalReplacement, MovementDirection

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.controller_params import ControllerParams
    from pybullet_fleet.geometry import Pose

logger = get_lazy_logger(__name__)


def _checked_braking_request(controller: Any, agent: "Agent") -> Optional[PathRequest]:
    """``controller.braking_request(agent)`` if it continues the current motion.

    A braking request is started from the agent's current motion, so it must
    be one the controller can continue into (``_replacement_obstacle``). One
    that cannot -- a custom request stopping short of the braking distance --
    is not started: a warning naming the agent, and the built-in braking
    point instead (:func:`~pybullet_fleet._replan.braking_request`), or
    ``None`` when that cannot be used either. ``None`` also at rest.
    """
    request = controller.braking_request(agent)
    if request is None:
        return None
    obstacle = controller._replacement_obstacle(agent, request.path, request.direction)
    if obstacle is None:
        return request
    builtin = controller._builtin_braking_request(agent)
    usable = builtin is not None and controller._replacement_obstacle(agent, builtin.path, builtin.direction) is None
    logger.warning(
        f"Braking request for {agent.name or agent.object_id} cannot continue the motion ({obstacle}); "
        + ("using the built-in braking point" if usable else "no braking path continues it")
    )
    return builtin if usable else None


class GoalReplacementPolicy:
    """Mixin carrying the ``goal_replacement`` decision for a kinematic controller.

    The host controller implements the hooks below and, as both kinematic
    controllers do, ``_replacement_obstacle`` and ``continuation_velocity``.
    """

    def _replace_path(self, agent: "Agent", request: PathRequest, preempt: bool = False) -> bool:
        """Take ``request`` as ``goal_replacement`` says.

        ``preempt`` skips the ``wait_complete`` hold. Returns ``False`` when
        the path is refused (``reject`` only); ``True`` once it is set or held
        to start later.
        """
        name = str(agent.name or agent.object_id)
        params = self._replacement_params(agent)
        mode = params._eff_goal_replacement()
        if mode is GoalReplacement.WAIT_COMPLETE and not preempt and self._following_path(agent):
            logger.info(f"Path for {name} starts once the current path is finished (wait_complete)")
            self._hold_request(agent, request)
            return True

        obstacle = self._replacement_obstacle(agent, request.path, request.direction)
        if obstacle is None:
            self._drop_held(agent, "a newer path started at once")
            self._start_request(agent, request, continue_motion=True)
            return True

        if mode is GoalReplacement.REJECT:
            logger.debug(f"Path rejected for {name}: {obstacle}")
            return False
        if mode is GoalReplacement.IMMEDIATE:
            logger.info(f"Path for {name} starts from rest at once: {obstacle}")
            self._drop_held(agent, "a newer path started at once")
            self._start_request(agent, request, continue_motion=False)
            return True
        if mode is GoalReplacement.BRAKE or mode is GoalReplacement.WAIT_COMPLETE:
            # wait_complete gets here only with nothing to wait for (a
            # velocity command, or brake() preempting): it brakes as brake does.
            if not self._braking_for(agent):
                # The controller's braking_request, checked: a custom one that
                # cannot continue the motion falls back to the built-in point.
                braking = _checked_braking_request(self, agent)
                if braking is None:
                    # Nothing left to brake, or no braking request that
                    # continues the motion: start from rest.
                    self._drop_held(agent, "a newer path started at once")
                    self._start_request(agent, request, continue_motion=False)
                    return True
                self._start_request(agent, braking, continue_motion=True)
                self._mark_braking(agent)
            logger.info(f"Path for {name} starts once the agent stops (brake): {obstacle}")
            self._hold_request(agent, request)
            return True
        raise ValueError(f"Unknown goal_replacement {mode!r}")

    def path_rejection(
        self, agent: "Agent", path: "list[Pose]", direction: Optional[MovementDirection] = None
    ) -> Optional[str]:
        """Why :meth:`_replace_path` would refuse ``path`` now, or ``None`` if it would take it.

        Only ``goal_replacement="reject"`` refuses a path: one that cannot
        continue the current motion (see ``_replacement_obstacle``). The other
        modes take every path.
        """
        if not path or self._replacement_params(agent)._eff_goal_replacement() is not GoalReplacement.REJECT:
            return None
        return self._replacement_obstacle(agent, path, direction)

    # -- the continuation rules both families apply ------------------------

    def _omni_obstacle(self, agent: "Agent", path: "list[Pose]") -> Optional[str]:
        """Why ``path`` cannot continue an omnidirectional agent's motion, or ``None``.

        The motion is :meth:`continuation_velocity` (``None``: nothing to
        continue, every path can be taken). The straight-line trajectory to the
        first waypoint starts from it, signed along the line, so a goal behind
        is reached by reversing; it does not exist for a goal inside the
        braking distance, where the agent is, or off the line of motion. See
        :mod:`pybullet_fleet._replan`.
        """
        velocity = self.continuation_velocity(agent)
        if velocity is None or not path:
            return None
        return self._line_obstacle(agent, velocity, path[0])

    def _differential_obstacle(
        self, agent: "Agent", path: "list[Pose]", direction: Optional[MovementDirection]
    ) -> Optional[str]:
        """Why ``path`` cannot continue a differential-drive agent's motion, or ``None``.

        It can only continue straight along its heading: a first waypoint that
        needs a turn in place beyond ``HEADING_TOL`` cannot be taken while
        moving. On the heading line, ahead or behind (``BACKWARD``, or ``AUTO``
        for a goal behind), the omnidirectional rule applies with the
        forward-axis limits. ``direction`` ``None`` is the agent's
        ``default_direction``, the one the new path will use.
        """
        velocity = self.continuation_velocity(agent)
        if velocity is None or not path or float(np.linalg.norm(velocity)) < REST_SPEED:
            return None
        params = self._replacement_params(agent)
        requested = direction if direction is not None else params.default_direction
        pose = agent.get_pose()
        _, target_quat, _, distance = heading_toward(pose, path[0], requested, params.navigation_2d)
        if distance > 1e-6:
            turn = quat_angle_between(np.asarray(pose.orientation, dtype=np.float64), target_quat)
            if turn > HEADING_TOL:
                return f"goal needs a {math.degrees(turn):.3g} degree turn in place"
        return self._line_obstacle(agent, velocity, path[0])

    def _line_obstacle(self, agent: "Agent", velocity: np.ndarray, goal: "Pose") -> Optional[str]:
        """:func:`~pybullet_fleet._replan.replacement_rejection` for ``agent`` heading for ``goal``."""
        pose = agent.get_pose()
        now = agent.sim_core.sim_time if agent.sim_core is not None else 0.0
        return replacement_rejection(
            self._replacement_params(agent), pose.position, pose.orientation, velocity, goal.position, now
        )

    # -- hooks the controller implements ---------------------------------

    def _replacement_params(self, agent: "Agent") -> "ControllerParams":
        """The limits and ``goal_replacement`` setting that apply to ``agent``."""
        raise NotImplementedError

    def _following_path(self, agent: "Agent") -> bool:
        """Whether ``agent`` follows a path (not a velocity command) that ``wait_complete`` waits for."""
        raise NotImplementedError

    def _hold_request(self, agent: "Agent", request: PathRequest) -> None:
        """Make ``request`` the agent's held path, dropping (and logging) the one before."""
        raise NotImplementedError

    def _drop_held(self, agent: "Agent", reason: str) -> None:
        """Drop the agent's held path, logging it with ``reason``, and clear its braking flag."""
        raise NotImplementedError

    def _start_request(self, agent: "Agent", request: PathRequest, continue_motion: bool) -> None:
        """Make ``request`` the active path, from the current motion or (``False``) from rest."""
        raise NotImplementedError

    def _braking_for(self, agent: "Agent") -> bool:
        """Whether ``agent`` is already braking for a held path."""
        raise NotImplementedError

    def _mark_braking(self, agent: "Agent") -> None:
        """Record that ``agent`` is braking for a held path."""
        raise NotImplementedError

    if TYPE_CHECKING:
        # Provided by both kinematic controllers; declared for type checking
        # only, so they never shadow the controller's own at run time.
        def _replacement_obstacle(
            self, agent: "Agent", path: "list[Pose]", direction: Optional[MovementDirection] = None
        ) -> Optional[str]: ...

        def continuation_velocity(self, agent: "Agent") -> Optional[np.ndarray]: ...

        def braking_request(self, agent: "Agent") -> Optional[PathRequest]: ...

        def _builtin_braking_request(self, agent: "Agent") -> Optional[PathRequest]: ...
