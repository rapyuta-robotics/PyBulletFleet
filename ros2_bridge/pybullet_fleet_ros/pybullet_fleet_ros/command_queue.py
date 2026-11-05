"""Hand commands from ROS callbacks to the simulation thread at the step boundary.

A ROS service or topic callback runs on the executor thread while the
simulation steps on its own thread. Applying a command inside the callback
means writing simulation state in the middle of a step, and, at full speed,
contending with the step for the interpreter lock: at 1000 robots a
``/fleet/navigate`` that solves every robot's trajectory inside the callback
answers in half a second.

:class:`StepCommandQueue` takes the work instead. A callback validates its
request, submits the application as a callable, and answers at once (or waits
for the application when the caller needs its result). The bridge applies the
queue at ``PRE_STEP``, on the simulation thread, before the step reads any
command.

The queue applies a submission inline, in the caller's thread, while it is not
``active`` (no simulation loop drives steps) or while the simulation is paused:
nothing would otherwise apply it. That keeps paused and stepped-by-service runs
and in-process tests behaving as before. It also applies inline when the
simulation thread is between steps (sleeping for pacing at a target RTF): the
bridge brackets each step with :meth:`StepCommandQueue.begin_step` and
:meth:`StepCommandQueue.end_step`, and a submission that finds the step lock
free takes it and applies at once. Only a submission that arrives during a
step waits for the next boundary.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger(__name__)

DEFAULT_APPLY_TIMEOUT_S = 1.0


@dataclass
class _Submission:
    apply: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class StepCommandQueue:
    """Commands to apply on the simulation thread at the next step boundary."""

    def __init__(self, sim_core: Any = None) -> None:
        self._sim = sim_core
        self._lock = threading.Lock()
        self._pending: Deque[_Submission] = deque()
        self._active = False
        # Held by the simulation thread from begin_step() to end_step(). A caller
        # that takes it between steps applies inline instead of queueing.
        self._step_lock = threading.Lock()
        self._in_step = False

    # ------------------------------------------------------------------
    # Lifecycle (bridge side)
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether a simulation loop will apply submissions at its steps."""
        return self._active

    def activate(self) -> None:
        """Start deferring: a simulation loop now applies the queue at every step."""
        self._active = True

    def deactivate(self) -> None:
        """Stop deferring and apply what is still pending, so no waiter hangs."""
        self._active = False
        self.apply_pending()

    @property
    def deferred(self) -> bool:
        """Whether a submission now would wait for the step boundary."""
        if not self._active:
            return False
        paused = getattr(self._sim, "is_paused", False)
        return not bool(paused)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # ------------------------------------------------------------------
    # Caller side (ROS callbacks)
    # ------------------------------------------------------------------

    def submit(self, apply: Callable[[], Any], *, wait: bool = False, timeout: float = DEFAULT_APPLY_TIMEOUT_S) -> Any:
        """Queue *apply* for the next step boundary, or run it now when nothing defers.

        With ``wait=True`` the call blocks until the simulation thread has
        applied the submission and returns its result; a timeout returns
        ``None`` and leaves the submission queued, so it still applies at the
        next step. An exception raised by *apply* is re-raised in the waiter;
        without a waiter it is logged and dropped.
        """
        if not self.deferred:
            return apply()
        if self._step_lock.acquire(blocking=False):
            # The simulation thread is between steps: apply now, before it starts the next one.
            try:
                return apply()
            finally:
                self._step_lock.release()
        submission = _Submission(apply)
        with self._lock:
            self._pending.append(submission)
        if not wait:
            return None
        if not submission.done.wait(timeout):
            return None
        if submission.error is not None:
            raise submission.error
        return submission.result

    # ------------------------------------------------------------------
    # Simulation side (PRE_STEP / POST_STEP)
    # ------------------------------------------------------------------

    def begin_step(self) -> int:
        """Take the step lock (waiting for an inline application in progress) and apply what is queued."""
        self._step_lock.acquire()
        self._in_step = True
        return self.apply_pending()

    def end_step(self) -> None:
        """Release the step lock: until the next begin_step(), submissions apply inline."""
        if self._in_step:
            self._in_step = False
            self._step_lock.release()

    def apply_pending(self) -> int:
        """Apply every queued submission in order; returns how many ran."""
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
        for submission in batch:
            try:
                submission.result = submission.apply()
            except Exception as exc:
                # A waiter re-raises it; a fire-and-forget submission only logs it.
                submission.error = exc
                logger.error("Queued command failed: %s", exc)
            finally:
                submission.done.set()
        return len(batch)
