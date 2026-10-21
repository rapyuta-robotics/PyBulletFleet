"""How a moving agent takes a new goal or path.

A path set while the agent is moving continues the motion when it can: the new
straight-line trajectory to the first waypoint starts from the agent's current
velocity -- its signed component along the new direction, so a goal behind is
reached by slowing, reversing and driving back -- provided that trajectory
exists within the configured limits. It does not exist for a goal closer than
the braking distance at the configured deceleration, a goal where the agent is
(it cannot stop on the spot), a goal off the line of motion (whose sideways
velocity a straight-line trajectory cannot represent), or -- for differential
drive -- a goal that needs a turn in place. At rest every path can be taken.

When the path cannot continue the motion, the controller's
``goal_replacement`` setting (:class:`~pybullet_fleet.types.GoalReplacement`)
decides: brake to a stop and then start it (the default), refuse it, finish the
current path first, or start it from rest at once.

:func:`replacement_rejection` is the one feasibility check every controller
(per-agent and batched) applies. It uses the same solver call as the
trajectory itself (:func:`~pybullet_fleet._tpi.try_build_tpi`), so
the check and the motion cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Protocol, Sequence, Tuple


import numpy as np

from pybullet_fleet._tpi import try_build_tpi
from pybullet_fleet.geometry import rotate_vector

from pybullet_fleet.types import MovementDirection

if TYPE_CHECKING:
    from pybullet_fleet.controller_params import ControllerParams
    from pybullet_fleet.geometry import Pose


@dataclass(frozen=True)
class PathRequest:
    """A path a controller has accepted but not started yet.

    Held while the agent brakes (``GoalReplacement.BRAKE``) or finishes its
    current path (``GoalReplacement.WAIT_COMPLETE``), and started from rest
    once it has stopped. A newer request replaces it.
    """

    path: List[Any]
    final_orientation_align: bool = True
    direction: Optional[MovementDirection] = None

    def to_record(self) -> dict:
        """Plain-data form for checkpoints."""
        return {
            "path": [{"position": list(p.position), "orientation": list(p.orientation)} for p in self.path],
            "final_orientation_align": self.final_orientation_align,
            "direction": self.direction.value if self.direction is not None else None,
        }

    @classmethod
    def from_record(cls, record: dict) -> "PathRequest":
        """Inverse of :meth:`to_record`."""
        from pybullet_fleet.geometry import Pose

        direction = record["direction"]
        return cls(
            path=[Pose(position=list(p["position"]), orientation=list(p["orientation"])) for p in record["path"]],
            final_orientation_align=bool(record["final_orientation_align"]),
            direction=MovementDirection(direction) if direction is not None else None,
        )


class PathRejection(NamedTuple):
    """Why an agent refuses a path.

    ``temporary`` is ``True`` for a refusal that lasts only while the agent is
    moving (no trajectory from its current velocity): the same path is taken
    once that motion ends, so a caller can try again. ``False`` is permanent
    for this agent and path (a fixed-base robot, an empty path).
    """

    reason: str
    temporary: bool


def braking_request(
    params: "ControllerParams",
    position: Sequence[float],
    orientation: Sequence[float],
    velocity: Optional[Sequence[float]],
) -> Optional[PathRequest]:
    """A one-waypoint path that brings the motion to rest at the configured deceleration.

    The goal is :func:`braking_position`, with the current orientation and no
    final alignment. ``direction`` is ``AUTO`` so that a differential-drive
    agent backing up brakes backwards instead of turning round. ``None`` at
    rest or without a velocity.
    """
    from pybullet_fleet.geometry import Pose

    if velocity is None:
        return None
    target = braking_position(params, position, orientation, velocity)
    if target is None:
        return None
    goal = Pose(position=[float(c) for c in target], orientation=list(orientation))
    return PathRequest(path=[goal], final_orientation_align=False, direction=MovementDirection.AUTO)


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

#: Turn (rad) between a differential-drive agent's heading and the line to its
#: new goal that still counts as straight on while it moves: the sideways speed
#: such a heading leaves is ``speed * sin(turn)``, within the omni cross-track
#: tolerance (:data:`CROSS_SPEED_REL`). A moving agent continues into such a
#: goal, snapping its heading onto the line; a larger turn needs a turn in place.
HEADING_TOL = CROSS_SPEED_REL


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
