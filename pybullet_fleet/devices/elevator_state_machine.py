"""Backend-neutral elevator request state machine."""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Mapping

from pybullet_fleet.devices.elevator_motion_adapter import ElevatorMotionAdapter


class ElevatorState(str, Enum):
    """Lifecycle states exposed by an elevator device."""

    IDLE = "idle"
    MOVING = "moving"
    ARRIVED = "arrived"


class ElevatorRequestPolicy(str, Enum):
    """How requests received while the cabin is moving are handled."""

    REJECT = "reject"
    REPLACE_NEXT = "replace_next"
    QUEUE = "queue"
    #: Turn the moving cabin toward the new floor at once; the latest request wins.
    REDIRECT = "redirect"


class ElevatorRequestResult(str, Enum):
    """Outcome of an :meth:`ElevatorStateMachine.request_floor` call."""

    ACCEPTED = "accepted"
    QUEUED = "queued"
    REPLACED = "replaced"
    REDIRECTED = "redirected"
    REJECTED = "rejected"


class ElevatorStateMachine:
    """Coordinate floor requests independently of a physics backend.

    ``ARRIVED`` is observable after a completed transition until the next
    update or a newly accepted floor request.
    """

    def __init__(
        self,
        floors: Mapping[str, float],
        initial_floor: str,
        adapter: ElevatorMotionAdapter,
        request_policy: ElevatorRequestPolicy | str = ElevatorRequestPolicy.REJECT,
    ) -> None:
        if not floors:
            raise ValueError("ElevatorStateMachine requires at least one floor")
        if initial_floor not in floors:
            raise ValueError(f"Unknown initial floor {initial_floor!r}")
        self._floors = dict(floors)
        self._adapter = adapter
        self.request_policy = ElevatorRequestPolicy(request_policy)
        self._pending_floors: deque[str] = deque()
        self.current_floor = initial_floor
        self.target_floor = initial_floor
        self.state = ElevatorState.IDLE

    @property
    def is_moving(self) -> bool:
        """Whether the cabin is currently in a physical movement transition."""
        return self.state is ElevatorState.MOVING

    @property
    def pending_floors(self) -> tuple[str, ...]:
        """Floor requests waiting for the current transition to complete."""
        return tuple(self._pending_floors)

    def request_floor(self, floor_name: str) -> ElevatorRequestResult:
        """Accept, queue, replace, redirect, or reject a requested destination.

        Only ``REDIRECT`` turns a moving cabin toward a new floor; it keeps its
        passengers and drops any pending floor, so the latest request wins.
        ``REPLACE_NEXT`` replaces the pending destination after the current
        trip, and a request for the floor the cabin is already heading to
        drops it -- the caller changed its mind back. ``QUEUE`` preserves
        requests in arrival order.
        """
        if floor_name not in self._floors:
            return ElevatorRequestResult.REJECTED

        if self.state is ElevatorState.MOVING:
            if floor_name == self.target_floor:
                if self.request_policy in (ElevatorRequestPolicy.REPLACE_NEXT, ElevatorRequestPolicy.REDIRECT) and (
                    self._pending_floors
                ):
                    self._pending_floors.clear()
                    return ElevatorRequestResult.REPLACED
                return ElevatorRequestResult.REJECTED
            if self.request_policy is ElevatorRequestPolicy.REDIRECT:
                self._pending_floors.clear()
                self.target_floor = floor_name
                self._adapter.begin_motion(self._floors[floor_name])
                return ElevatorRequestResult.REDIRECTED
            if self.request_policy is ElevatorRequestPolicy.REJECT:
                return ElevatorRequestResult.REJECTED
            if self.request_policy is ElevatorRequestPolicy.REPLACE_NEXT:
                replaced = bool(self._pending_floors)
                self._pending_floors.clear()
                self._pending_floors.append(floor_name)
                return ElevatorRequestResult.REPLACED if replaced else ElevatorRequestResult.QUEUED
            self._pending_floors.append(floor_name)
            return ElevatorRequestResult.QUEUED

        if floor_name == self.current_floor and self.state is not ElevatorState.MOVING:
            return ElevatorRequestResult.REJECTED

        self._start_motion(floor_name)
        return ElevatorRequestResult.ACCEPTED

    def _start_motion(self, floor_name: str) -> None:
        self._adapter.attach_platform_passengers()
        self.target_floor = floor_name
        self._adapter.begin_motion(self._floors[floor_name])
        self.state = ElevatorState.MOVING

    def update(self) -> bool:
        """Observe the adapter and complete an arrival when movement has ended.

        Returns ``True`` only on the update that observes arrival.
        """
        if self.state is ElevatorState.ARRIVED:
            self.state = ElevatorState.IDLE
            if self._pending_floors:
                self._start_motion(self._pending_floors.popleft())
            return False
        if self.state is not ElevatorState.MOVING or self._adapter.motion_in_progress():
            return False

        self.current_floor = self.target_floor
        self._adapter.detach_passengers()
        self.state = ElevatorState.ARRIVED
        return True
