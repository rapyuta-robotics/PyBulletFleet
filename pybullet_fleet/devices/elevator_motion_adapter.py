"""Backend-neutral physical adapter contract for elevator state machines."""

from __future__ import annotations

from typing import Protocol


class ElevatorMotionAdapter(Protocol):
    """Physical operations required by :class:`ElevatorStateMachine`.

    Implementations command and observe a particular physics backend. The
    state machine owns request validation and lifecycle transitions; an adapter
    owns joints and passenger attachment mechanics.
    """

    def begin_motion(self, target_height: float, instant: bool = False) -> None:
        """Start or replace motion of the cabin toward ``target_height``.

        With ``instant``, put the cabin and its attached passengers at
        ``target_height`` before returning, so that :meth:`motion_in_progress`
        is already False. The state machine passes it only when a request asks
        for it.
        """

    def motion_in_progress(self) -> bool:
        """Return whether the backend still reports commanded motion."""

    def attach_platform_passengers(self) -> int:
        """Attach eligible passengers before the cabin moves; return their count."""

    def detach_passengers(self) -> int:
        """Detach passengers after arrival; return their count."""
