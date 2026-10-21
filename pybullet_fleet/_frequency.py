"""When a frequency-limited handler is due, on simulated time.

Simulated time carries floating point error: ten steps of 0.1 s add up to
0.9999999999999999 s. A check of ``now - last >= interval`` would miss a beat
whenever that error falls short, and measuring from the time a handler
actually ran would let its phase drift when the interval is not a whole number
of timesteps.

:class:`FrequencyGate` instead fires on a fixed grid of due times, ``k *
interval`` from time zero, with a tolerance far below one timestep. Its caller
passes a time computed from the step count rather than accumulated step by
step, whose error grows with the run until it exceeds any such tolerance.

A gate fires at most once per call (one call per step). One that starts mid-run
-- a handler registered late, or after :meth:`FrequencyGate.reset` -- first
fires at the next grid point. If calls stop while time moves on, the first
call past the missed due times fires once and the schedule returns to the
grid, rather than firing once for every point it missed.
"""

import math
from typing import Optional

from pybullet_fleet.logging_utils import get_lazy_logger

logger = get_lazy_logger(__name__)

#: Fraction of a timestep by which simulated time may fall short of a due time
#: and still count as reaching it. The rounding error of a time computed from
#: the step count is many orders of magnitude smaller than this; a real
#: shortfall is a whole timestep.
_TOLERANCE = 1e-6


class FrequencyGate:
    """Decide, once per step, whether a handler running at ``frequency`` Hz is due.

    ``frequency`` ``None`` means every step. The gate keeps no reference to
    the simulation; callers pass the step's simulated time and timestep.
    An owner creates its gate once and changes the rate with
    :meth:`set_frequency`.
    """

    __slots__ = ("_frequency", "_next_index", "_last_time", "_warned")

    def __init__(self, frequency: Optional[float] = None) -> None:
        self._frequency = self._checked(frequency)
        self._warned = False
        self.reset()

    @property
    def frequency(self) -> Optional[float]:
        """Rate in Hz, or ``None`` for every step. Change it with :meth:`set_frequency`."""
        return self._frequency

    def set_frequency(self, frequency: Optional[float]) -> None:
        """Run at ``frequency`` from now on; a new rate starts a fresh schedule.

        Setting the current rate again changes nothing.
        """
        frequency = self._checked(frequency)
        if frequency == self._frequency:
            return
        self._frequency = frequency
        self.reset()

    @staticmethod
    def _checked(frequency: Optional[float]) -> Optional[float]:
        if frequency is not None and not (math.isfinite(frequency) and frequency > 0):
            raise ValueError("frequency must be None or a positive finite number, got {!r}".format(frequency))
        return frequency

    def reset(self) -> None:
        """Forget the schedule, as at the start of a run."""
        self._next_index: Optional[int] = None
        self._last_time: Optional[float] = None

    def due(self, now: float, timestep: float) -> Optional[float]:
        """Whether the handler runs at ``now``, and with which ``dt``.

        Returns:
            ``None`` when the handler is not due at ``now``. When it is due, the
            simulated time since it last ran (``dt_since_last_run``), to pass to
            the handler: always positive, and one interval (at least one
            timestep) on the first run. Test with ``is not None``.

        A handler first runs one interval into the run (the first grid point
        after time zero), or at once when its first call is already on a due
        time.
        """
        if self._frequency is None:
            first_dt = timestep
        else:
            # Grid points within ``tolerance`` of ``now`` count as ``now`` itself,
            # in either direction, so float error in the time neither skips a due
            # time nor runs the same one twice.
            interval = 1.0 / self._frequency
            tolerance = _TOLERANCE * timestep
            if self._next_index is None:
                # The first grid point at or after now, but never time zero.
                self._next_index = max(1, math.ceil((now - tolerance) / interval))
            if now + tolerance < self._next_index * interval:
                return None
            # Next: the following grid point, or when calls stopped past several,
            # the first one still ahead of now.
            self._next_index = max(self._next_index + 1, math.floor((now + tolerance) / interval) + 1)
            # The first run has no previous one: one interval, at least a step.
            first_dt = max(interval, timestep)
        last, self._last_time = self._last_time, now
        dt = first_dt if last is None else now - last
        return self._positive(dt, now, last, timestep)

    def _positive(self, dt: float, now: float, last: Optional[float], timestep: float) -> float:
        """``dt``, or one timestep with a warning when it is not positive.

        A non-positive ``dt`` means the gate was asked twice at the same time or
        the clock went back without :meth:`reset` -- a caller error. Handlers
        still get a usable ``dt``; the warning is logged once per gate.
        """
        if dt > 0:
            return dt
        if not self._warned:
            self._warned = True
            logger.warning(
                "FrequencyGate: non-positive dt %.9g at time %.9g (previous run at %r); "
                "was the gate called twice in one step, or the clock moved back without reset()? "
                "Using one timestep (%.9g) instead.",
                dt,
                now,
                last,
                timestep,
            )
        return timestep
