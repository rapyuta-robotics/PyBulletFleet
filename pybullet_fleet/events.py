"""Lightweight event pub/sub for simulation lifecycle events.

Two usage levels:

- **Global bus** (``sim.events``): sim-wide events (pre_step, agent_spawned, etc.)
- **Per-entity bus** (``agent.events``): entity-specific hooks for subclasses

Handlers execute in priority order (lower = first). Same priority = FIFO.
Exceptions in handlers are logged but do not block other handlers.

The simulation's bus also carries the other per-step hooks: callbacks from
``register_callback()`` and plugin ``on_step()`` subscribe to the step events
in tiers of their own, after the handlers registered with :meth:`EventBus.on`.
A handler on a step event can be frequency-limited; the bus then schedules it
on the simulation clock with the same fixed grid as callbacks and plugins.

Pre-defined event names are available as :class:`SimEvents` constants.
Custom event names (any ``str``) are also supported for user-defined extensions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from pybullet_fleet._frequency import FrequencyGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-defined event name constants
# ---------------------------------------------------------------------------


class SimEvents:
    """Pre-defined event name constants.

    Using these constants prevents typos and enables IDE auto-completion,
    while the :class:`EventBus` still accepts any ``str`` for user-defined
    events.

    **Simulation lifecycle** (emitted on ``sim.events``)::

        sim.events.on(SimEvents.PRE_STEP, handler)
        sim.events.on(SimEvents.POST_STEP, handler)

    **Entity lifecycle** (emitted on both ``sim.events`` and ``entity.events``)::

        sim.events.on(SimEvents.OBJECT_SPAWNED, handler)
        agent.events.on(SimEvents.ACTION_COMPLETED, handler)

    **Custom events** (any string — no constant needed)::

        sim.events.on("my_plugin_event", handler)
    """

    # --- Simulation step -------------------------------------------------
    PRE_STEP: str = "pre_step"
    """Emitted at the start of each step, before the clock advances.  kwargs: ``dt``, ``sim_time``
    (the time of the previous state)."""

    STEP: str = "step"
    """Emitted after the objects have been updated, before their new poses are
    written to PyBullet and before the collision check -- the point where
    ``register_callback()`` callbacks run by default.  kwargs: ``dt``, ``sim_time``
    (the time of the new state)."""

    POST_STEP: str = "post_step"
    """Emitted at the end of each step.  kwargs: ``dt``, ``sim_time``."""

    PAUSED: str = "paused"
    """Emitted when simulation is paused.  No kwargs."""

    RESUMED: str = "resumed"
    """Emitted when simulation is resumed.  No kwargs."""

    # --- Object / Agent lifecycle ----------------------------------------
    OBJECT_SPAWNED: str = "object_spawned"
    """Global: ``obj``.  Per-entity: no kwargs."""

    OBJECT_REMOVED: str = "object_removed"
    """Global: ``obj``.  Per-entity: no kwargs."""

    AGENT_SPAWNED: str = "agent_spawned"
    """Global only: ``agent``."""

    AGENT_REMOVED: str = "agent_removed"
    """Global only: ``agent``."""

    # --- Collision -------------------------------------------------------
    COLLISION_STARTED: str = "collision_started"
    """Global: ``obj_a``, ``obj_b``.  Per-entity: ``other``."""

    COLLISION_ENDED: str = "collision_ended"
    """Global: ``obj_a``, ``obj_b``.  Per-entity: ``other``."""

    # --- Agent update -----------------------------------------------------
    PRE_UPDATE: str = "pre_update"
    """Per-entity only: ``dt``.  Emitted before agent movement/action processing."""

    POST_UPDATE: str = "post_update"
    """Per-entity only: ``dt``, ``moved`` (bool).  Emitted after agent update completes."""

    # --- Action ----------------------------------------------------------
    ACTION_STARTED: str = "action_started"
    """Global: ``agent``, ``action``.  Per-entity: ``action``."""

    ACTION_COMPLETED: str = "action_completed"
    """Global: ``agent``, ``action``, ``status``.  Per-entity: ``action``, ``status``."""


#: Events emitted once per step with ``dt`` and ``sim_time``; the only ones a
#: subscription may be frequency-limited on.
STEP_EVENTS = frozenset({SimEvents.PRE_STEP, SimEvents.STEP, SimEvents.POST_STEP})

# Subscription tiers (internal), run in this order within one event.  Handlers from
# EventBus.on() come first, then register_callback() callbacks, then
# plugin on_step().
_TIER_HANDLER = 0
_TIER_CALLBACK = 1
_TIER_PLUGIN = 2

#: ``clock()`` returns the simulation time to schedule on and the timestep.
Clock = Callable[[], Tuple[float, float]]


class _Subscription:
    """One handler on one event: how to call it, when, and what to do when it fails."""

    __slots__ = ("handler", "call", "priority", "tier", "seq", "gate", "propagate", "takes_dt", "owner")

    def __init__(
        self,
        handler: Callable,
        call: Callable,
        priority: int,
        tier: int,
        seq: int,
        gate: Optional[FrequencyGate],
        propagate: bool,
        takes_dt: bool,
        owner: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.handler = handler  # what the subscriber registered; matched by off()
        self.call = call  # what emit() calls: with the event's kwargs, or with dt alone (takes_dt)
        self.priority = priority
        self.tier = tier
        self.seq = seq
        self.gate = gate
        self.propagate = propagate
        self.takes_dt = takes_dt
        self.owner = owner  # with takes_dt: a weak reference whose target is passed before dt

    def sort_key(self) -> Tuple[int, int, int]:
        return (self.tier, self.priority, self.seq)


class EventBus:
    """Lightweight event pub/sub bus.

    Usage::

        bus = EventBus()
        bus.on("collision_started", my_handler, priority=0)
        bus.emit("collision_started", obj_a=a, obj_b=b)
        bus.off("collision_started", my_handler)

    Args:
        clock: Returns ``(now, timestep)`` for scheduling frequency-limited
            subscriptions.  The simulation's bus (``sim.events``) has one;
            a bus without a clock (a bare or per-entity bus) rejects
            ``frequency``.

    An emit calls the subscriptions present when it starts: a handler added or
    removed by a handler takes effect from the next emit.
    """

    __slots__ = ("_handlers", "_tiers", "_clock", "_seq")

    def __init__(self, clock: Optional[Clock] = None) -> None:
        # event -> every subscription, in call order; only non-empty entries.
        self._handlers: Dict[str, Tuple[_Subscription, ...]] = {}
        # step event -> one tuple per tier (indexed by tier), each in call order.
        self._tiers: Dict[str, Tuple[Tuple[_Subscription, ...], ...]] = {}
        self._clock = clock
        self._seq = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on(self, event: str, handler: Callable, priority: int = 0, frequency: Optional[float] = None) -> None:
        """Register ``handler`` for ``event``.  Lower ``priority`` runs first; equal ones in registration order.

        Args:
            event: Event name, a :class:`SimEvents` constant or any string.
            handler: Called with the event's keyword arguments.
            priority: Order among this event's handlers (lower = first).
            frequency: Rate in Hz, for the step events (``pre_step``, ``step``,
                ``post_step``) on a bus with a clock.  The handler then runs on
                a fixed grid of ``k / frequency`` seconds of simulated time (as
                ``register_callback(frequency=...)`` does), scheduled on the
                time it observes, and its ``dt`` is the simulated time since it
                last ran.  ``None`` or ``0`` runs it on every emit, as
                ``0`` does for ``register_callback()``.

        Raises:
            ValueError: ``frequency`` is negative or not finite, is given for
                an event other than a step event, or the bus has no clock.
        """
        gate = None
        if frequency:  # None or 0: every emit, as for register_callback()
            if event not in STEP_EVENTS:
                raise ValueError(
                    "frequency is only supported for the step events {}, got {!r}".format(sorted(STEP_EVENTS), event)
                )
            gate = FrequencyGate(frequency)  # validates; _subscribe() checks for a clock
        self._subscribe(event, handler, handler, priority=priority, gate=gate)

    def off(self, event: str, handler: Callable) -> None:
        """Unregister ``handler`` from ``event`` (every registration of it made with :meth:`on`)."""
        self._remove(event, lambda sub: sub.tier == _TIER_HANDLER and sub.handler is handler)

    def emit(self, event: str, **kwargs: Any) -> None:
        """Fire event.  Handlers called in priority order, then callbacks, then plugins.

        The simulation emits the step events (``pre_step``, ``step``,
        ``post_step``) once per step.  Emitting one on ``sim.events`` yourself
        runs everything subscribed to it, ``register_callback()`` callbacks and
        plugin ``on_step()`` included, uses up the due beats of the
        frequency-limited ones, and lets a callback's exception propagate.
        """
        subscriptions = self._handlers.get(event)
        if subscriptions:
            self._dispatch(event, subscriptions, kwargs)

    def clear(self, event: Optional[str] = None) -> None:
        """Remove the handlers registered with :meth:`on`; for ``event`` only when given.

        Callbacks and plugins the simulation subscribed are kept.
        """
        for name in [event] if event else list(self._handlers):
            self._remove(name, lambda sub: sub.tier == _TIER_HANDLER)

    def has_handlers(self, event: str) -> bool:
        """Whether emitting ``event`` would call anything (handlers, callbacks or plugins)."""
        return bool(self._handlers.get(event))

    # ------------------------------------------------------------------
    # Simulation-internal API
    # ------------------------------------------------------------------

    def _subscribe(
        self,
        event: str,
        handler: Callable,
        call: Callable,
        *,
        priority: int = 0,
        tier: int = _TIER_HANDLER,
        gate: Optional[FrequencyGate] = None,
        propagate: bool = False,
        takes_dt: bool = False,
        owner: Optional[Callable[[], Any]] = None,
    ) -> _Subscription:
        """Add a subscription; ``gate`` needs a clock.

        ``propagate`` re-raises a handler's exception instead of logging it.
        ``takes_dt`` calls ``call(dt)`` positionally instead of with the
        event's keyword arguments (for step events, which carry ``dt``);
        with ``owner``, a weak reference, ``call(owner(), dt)``.  The bus
        holds the owner weakly so that a simulation is not kept alive by
        the callbacks on its own bus.
        """
        if gate is not None and self._clock is None:
            raise ValueError("frequency needs a bus with a simulation clock, such as sim.events")
        self._seq += 1
        subscription = _Subscription(handler, call, priority, tier, self._seq, gate, propagate, takes_dt, owner)
        # Insert in call order: after every subscription that sorts before it.
        # It has the highest seq, so in the common case (same tier and
        # priority as the last one, or later) that is the end.
        subscriptions = self._handlers.get(event, ())
        key = subscription.sort_key()
        index = len(subscriptions)
        while index and subscriptions[index - 1].sort_key() > key:
            index -= 1
        self._store(event, subscriptions[:index] + (subscription,) + subscriptions[index:])
        return subscription

    def _remove(self, event: str, match: Callable[[_Subscription], bool]) -> None:
        subscriptions = self._handlers.get(event)
        if subscriptions:
            kept = tuple(sub for sub in subscriptions if not match(sub))
            if len(kept) != len(subscriptions):
                self._store(event, kept)

    def _unsubscribe(self, event: str, subscription: _Subscription) -> None:
        self._remove(event, lambda sub: sub is subscription)

    def _store(self, event: str, ordered: Tuple[_Subscription, ...]) -> None:
        """Replace ``event``'s subscriptions, already in call order, with new tuples, so a running emit keeps its own.

        Only the step events, which the simulation emits tier by tier, keep
        the split by tier.
        """
        if not ordered:
            self._handlers.pop(event, None)
            self._tiers.pop(event, None)
            return
        self._handlers[event] = ordered
        if event in STEP_EVENTS:
            self._tiers[event] = tuple(
                tuple(sub for sub in ordered if sub.tier == tier) for tier in (_TIER_HANDLER, _TIER_CALLBACK, _TIER_PLUGIN)
            )

    def _subscriptions(self, tier: Optional[int] = None) -> List[Tuple[str, _Subscription]]:
        """``(event, subscription)`` pairs, of ``tier`` when given, in registration order."""
        if tier is None or tier == _TIER_HANDLER:
            pairs = [
                (event, sub) for event, subs in self._handlers.items() for sub in subs if tier is None or sub.tier == tier
            ]
        else:
            # Callbacks and plugins only subscribe to the step events.
            pairs = [(event, sub) for event, tiers in self._tiers.items() for sub in tiers[tier]]
        pairs.sort(key=lambda pair: pair[1].seq)
        return pairs

    def _tier_snapshot(self, event: str) -> Optional[Tuple[Tuple[_Subscription, ...], ...]]:
        """``event``'s subscriptions split by tier (indexed by tier), or ``None`` when it has none.

        The tuples are never mutated, so holding them for a whole emit keeps
        the snapshot guarantee across tiers: a subscription added or removed
        while the tiers run takes effect from the next emit.
        """
        return self._tiers.get(event)

    def _reset_schedules(self) -> None:
        """Restart every frequency-limited subscription's schedule, as at the start of a run."""
        for subscriptions in self._handlers.values():
            for sub in subscriptions:
                if sub.gate is not None:
                    sub.gate.reset()

    def _dispatch(self, event: str, subscriptions: Tuple[_Subscription, ...], kwargs: Dict[str, Any]) -> None:
        clock_reading: Optional[Tuple[float, float]] = None
        for sub in subscriptions:
            dt: Optional[float] = None
            if sub.gate is not None:
                if clock_reading is None:
                    clock_reading = self._clock()  # type: ignore[misc]  # a gate implies a clock
                dt = sub.gate.due(*clock_reading)
                if dt is None:
                    continue
            try:
                if sub.takes_dt:
                    if dt is None:
                        dt = kwargs["dt"]
                    if sub.owner is None:
                        sub.call(dt)
                    else:
                        sub.call(sub.owner(), dt)
                elif dt is None:
                    sub.call(**kwargs)
                else:
                    sub.call(**dict(kwargs, dt=dt))
            except Exception:
                if sub.propagate:
                    raise
                logger.exception("EventBus handler error for event '%s'", event)
