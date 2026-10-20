"""Whether a moving omni agent can take a new goal within its limits.

A goal or path set while the agent is moving continues the motion: the new
straight-line trajectory to the first waypoint starts from the agent's current
velocity -- its signed component along the new direction, so a goal behind is
reached by slowing, reversing and driving back. The goal is taken when that
trajectory exists within the configured limits and rejected otherwise: the
current motion then continues unchanged and the caller is told. Rejected are a
goal closer than the braking distance at the configured deceleration, a goal
where the agent is (it cannot stop on the spot), and a goal off the line of
motion, whose sideways velocity a straight-line trajectory cannot represent.
At rest any goal the trajectory solver accepts is taken.

:func:`replacement_rejection` is the one check both omni controllers
(per-agent and batched) apply. It uses the same solver call as the trajectory
itself (:func:`~pybullet_fleet._tpi.try_build_tpi`), so the check
and the motion cannot disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Optional, Protocol, Sequence, Tuple

import numpy as np

from pybullet_fleet._tpi import try_build_tpi
from pybullet_fleet.geometry import rotate_vector

if TYPE_CHECKING:
    from pybullet_fleet.controller_params import ControllerParams
    from pybullet_fleet.geometry import Pose


class PathRejection(NamedTuple):
    """Why an agent refuses a path.

    ``temporary`` is ``True`` for a refusal that lasts only while the agent is
    moving (no trajectory from its current velocity): the same path is taken
    once that motion ends, so a caller can try again. ``False`` is permanent
    for this agent and path (a fixed-base robot, an empty path).
    """

    reason: str
    temporary: bool


class RefusesPaths(Protocol):
    """An agent that says why it would refuse a path (:meth:`Agent.path_rejection
    <pybullet_fleet.agent.Agent.path_rejection>`)."""

    def path_rejection(self, path: List["Pose"]) -> Optional[PathRejection]: ...


def refused_until_stopped(agent: RefusesPaths, path: Sequence["Pose"]) -> bool:
    """Whether ``agent`` refuses ``path`` only until its current motion ends.

    ``True`` means the same path is taken once the agent stops, so a caller
    retries it then. ``False`` when the path is taken now or refused for good
    (a fixed base, an empty path), which must not be retried forever.
    """
    rejection = agent.path_rejection(list(path))
    return rejection is not None and rejection.temporary


#: Below this speed (m/s) the agent counts as at rest.
REST_SPEED = 1e-6

#: Velocity across the line to the new goal that still counts as moving along
#: it: an absolute floor plus a fraction of the speed. It absorbs numerical
#: noise and is dropped when the trajectory starts; anything larger is a change
#: of direction and is rejected.
CROSS_SPEED_ABS = 1e-6
CROSS_SPEED_REL = 1e-3


def limits_along(
    params: "ControllerParams", orientation: Sequence[float], direction_unit: np.ndarray
) -> Tuple[float, float, float]:
    """``(vmax, accel, decel)`` along a world-frame unit direction.

    The per-axis caps are body-frame, so the direction is rotated into the
    body frame first -- the same projection both omni controllers use when
    they build a trajectory.
    """
    qx, qy, qz, qw = orientation
    direction_body = np.asarray(rotate_vector(tuple(direction_unit), (-qx, -qy, -qz, qw)))
    return (
        params.linear_vel_along_direction(direction_body),
        params.linear_accel_along_direction(direction_body),
        params.linear_decel_along_direction(direction_body),
    )


#: Relative margin added to the braking distance by :func:`braking_position`.
#: A goal exactly at the braking distance has no acceleration phase, which the
#: trajectory solver refuses as "cannot stop in time"; a hair further it is a
#: pure deceleration in effect (the extra distance is micrometres).
BRAKING_MARGIN_REL = 1e-6


def braking_position(
    params: "ControllerParams",
    position: Sequence[float],
    orientation: Sequence[float],
    velocity: Sequence[float],
) -> Optional[np.ndarray]:
    """Where an agent moving at ``velocity`` comes to rest at its configured deceleration.

    The point straight ahead along the velocity at ``|v|^2 / (2 * decel)``, the
    deceleration being the one along that direction (see :func:`limits_along`),
    plus :data:`BRAKING_MARGIN_REL`. A goal there is always taken by
    :func:`replacement_rejection`: it lies on the line of motion, and the
    trajectory from the current speed is a deceleration at exactly the
    configured rate. Returns ``None`` at rest.
    """
    pos = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float).copy()
    if params.navigation_2d:
        vel[2] = 0.0
    speed = float(np.linalg.norm(vel))
    if speed < REST_SPEED:
        return None
    unit = vel / speed
    _, _, decel = limits_along(params, orientation, unit)
    distance = speed * speed / (2.0 * decel) * (1.0 + BRAKING_MARGIN_REL)
    return pos + unit * distance


def replacement_rejection(
    params: "ControllerParams",
    position: Sequence[float],
    orientation: Sequence[float],
    velocity: Sequence[float],
    goal_position: Sequence[float],
    t0: float,
) -> Optional[str]:
    """Why an agent moving at ``velocity`` cannot head for ``goal_position`` now.

    Returns ``None`` when it can: the straight-line trajectory from its
    current state exists within its limits. Otherwise returns a short reason.
    """
    pos = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    goal = np.asarray(goal_position, dtype=float)
    if params.navigation_2d:
        goal = goal.copy()
        goal[2] = pos[2]
        vel = vel.copy()
        vel[2] = 0.0
    speed = float(np.linalg.norm(vel))
    if speed < REST_SPEED:
        return None

    disp = goal - pos
    distance = float(np.linalg.norm(disp))
    if distance <= 1e-9:
        return f"moving at {speed:.3g} m/s, cannot stop where it is"
    unit = disp / distance
    along = float(np.dot(vel, unit))
    across = float(np.linalg.norm(vel - along * unit))
    if across > CROSS_SPEED_ABS + CROSS_SPEED_REL * speed:
        return f"goal is off the line of motion ({across:.3g} m/s across it)"
    vmax, accel, decel = limits_along(params, orientation, unit)
    if try_build_tpi(p0=0.0, pe=distance, vmax=vmax, accel=accel, t0=t0, v0=along, decel=decel) is None:
        braking = along * along / (2.0 * decel)
        return f"goal is {distance:.3g} m ahead, inside the {braking:.3g} m braking distance"
    return None
