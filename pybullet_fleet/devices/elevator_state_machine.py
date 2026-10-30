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


class ElevatorRequestResult(str, Enum):
    """Outcome of an :meth:`ElevatorStateMachine.request_floor` call."""

    ACCEPTED = "accepted"
    QUEUED = "queued"
    REPLACED = "replaced"
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

    def request_floor(self, floor_name: str, instant: bool = False) -> ElevatorRequestResult:
        """Accept, queue, replace, or reject a requested destination.

        The physical cabin is never redirected mid-motion.  ``REPLACE_NEXT``
        replaces the pending destination after the current trip; ``QUEUE``
        preserves requests in arrival order.

        ``instant`` puts the cabin at the floor at once instead of riding there,
        with its passengers aboard; the next :meth:`update` observes the
        arrival as after any trip. It is always accepted for another floor,
        also while the cabin is moving: the cabin is then put at the new floor
        with the passengers of the current trip, and pending requests are
        dropped.
        """
        if floor_name not in self._floors:
            return ElevatorRequestResult.REJECTED

        if instant:
            if self.state is ElevatorState.MOVING:
                if floor_name == self.target_floor and not self._adapter.motion_in_progress():
                    return ElevatorRequestResult.REJECTED
                self._pending_floors.clear()
                self.target_floor = floor_name
                self._adapter.begin_motion(self._floors[floor_name], instant=True)
                return ElevatorRequestResult.ACCEPTED
            if floor_name == self.current_floor:
                return ElevatorRequestResult.REJECTED
            self._start_motion(floor_name, instant=True)
            return ElevatorRequestResult.ACCEPTED

        if self.state is ElevatorState.MOVING:
            if floor_name == self.target_floor:
                return ElevatorRequestResult.REJECTED
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

    def _start_motion(self, floor_name: str, instant: bool = False) -> None:
        self._adapter.attach_platform_passengers()
        self.target_floor = floor_name
        if instant:
            self._adapter.begin_motion(self._floors[floor_name], instant=True)
        else:
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
