"""A request to follow a path: the waypoints and how to follow them.

:class:`PathRequest` is what :meth:`Agent.request_path
<pybullet_fleet.agent.Agent.request_path>` and the controllers'
``request_path`` take, what a controller holds while it brakes or finishes
its current path (see :class:`~pybullet_fleet.types.GoalReplacement`), and
what a :class:`~pybullet_fleet.action.MoveAction` sends. ``Agent.set_path``
and the controllers' ``set_path`` build one from their arguments.

A request keeps its identity through the copies made along the way (the
approach waypoint ``Agent.request_path`` adds is a copy with
:func:`dataclasses.replace`), so a caller can ask a controller what became of
the request it sent (:meth:`~pybullet_fleet.agent.Agent.path_request_state`).
Identity is not equality: two requests with equal fields are different
requests, and a ``replace()`` copy is the same one -- build a new
:class:`PathRequest` to send a new request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from pybullet_fleet.geometry import Pose
from pybullet_fleet.types import MovementDirection


class _RequestToken:
    """A request's identity (see :meth:`PathRequest.same_request`); weakly referenceable,
    so a record of requests can forget the ones nobody holds any more."""

    __slots__ = ("__weakref__",)


@dataclass(frozen=True)
class PathRequest:
    """Waypoints to follow and how to follow them.

    Args:
        path: The waypoints, in order (a sequence of
            :class:`~pybullet_fleet.geometry.Pose`; stored as a tuple).
        final_orientation_align: Rotate to the last waypoint's orientation
            after reaching its position.
        direction: Movement direction (differential drive only): a
            :class:`~pybullet_fleet.types.MovementDirection` or its string
            value; ``None`` uses the controller's ``default_direction``.
        auto_approach: Let :meth:`Agent.request_path
            <pybullet_fleet.agent.Agent.request_path>` put the current pose in
            front of the path when the agent is at rest and far from its first
            waypoint. Controllers ignore it.

    A copy made with :func:`dataclasses.replace` is the same request (see
    :meth:`same_request`); to send a new request, build a new one. Frozen, but
    not hashable: the :class:`~pybullet_fleet.geometry.Pose` waypoints are
    mutable. Identity does not survive :meth:`to_record` /
    :meth:`from_record`, so a request restored from a checkpoint is a new one.
    """

    path: Sequence["Pose"]
    final_orientation_align: bool = True
    direction: Optional[MovementDirection] = None
    auto_approach: bool = True
    # Identity kept by dataclasses.replace() copies; see same_request().
    _token: object = field(default_factory=_RequestToken, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        if self.direction is not None:
            object.__setattr__(self, "direction", MovementDirection(self.direction))

    def same_request(self, other: Optional["PathRequest"]) -> bool:
        """Whether ``other`` is this request or a copy of it made on its way to a controller."""
        return other is not None and other._token is self._token

    def to_record(self) -> dict:
        """Plain-data form for checkpoints (without ``auto_approach``: a held
        request has had its approach applied already)."""
        return {
            "path": [{"position": list(p.position), "orientation": list(p.orientation)} for p in self.path],
            "final_orientation_align": self.final_orientation_align,
            "direction": self.direction.value if self.direction is not None else None,
        }

    @classmethod
    def from_record(cls, record: dict) -> "PathRequest":
        """Inverse of :meth:`to_record`."""
        direction = record["direction"]
        return cls(
            path=[Pose(position=list(p["position"]), orientation=list(p["orientation"])) for p in record["path"]],
            final_orientation_align=bool(record["final_orientation_align"]),
            direction=MovementDirection(direction) if direction is not None else None,
            auto_approach=False,
        )


class PathRequestState(Enum):
    """What became of a :class:`PathRequest` sent to a controller."""

    #: Being followed.
    ACTIVE = "active"
    #: Accepted and held, to start once the agent stops or finishes its path.
    PENDING = "pending"
    #: Followed to its end; the agent came to rest there.
    FINISHED = "finished"
    #: Not (or no longer) the controller's: replaced by another request,
    #: dropped from the hold, stopped, braked, cancelled or never sent.
    GONE = "gone"
    #: The controller does not track requests (a custom controller written
    #: against ``set_path``).
    UNKNOWN = "unknown"


def request_state(
    request: PathRequest,
    active: Optional[PathRequest],
    pending: Optional[PathRequest],
    finished: Optional[PathRequest],
) -> PathRequestState:
    """``request``'s state given a controller's active, pending and last finished requests."""
    if request.same_request(active):
        return PathRequestState.ACTIVE
    if request.same_request(pending):
        return PathRequestState.PENDING
    if request.same_request(finished):
        return PathRequestState.FINISHED
    return PathRequestState.GONE


__all__: List[str] = ["PathRequest", "PathRequestState", "request_state"]
