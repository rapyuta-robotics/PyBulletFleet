# Unify per-step hooks on the event bus

**Date:** 2026-11-05
**Status:** Implemented (issue #84)

## Context

A simulation step had three hook mechanisms with overlapping purposes:

| | `register_callback()` | `events.on()` | `SimPlugin.on_step()` |
|---|---|---|---|
| When | `STEP` point or end of step | `pre_step`, `post_step`, discrete events | after `STEP` callbacks |
| Frequency limit | yes | no | yes |
| Arguments | `(sim_core, dt)` | event keyword arguments | `dt` |
| Exceptions | propagate | logged | logged |
| Order | registration | `priority` | load order |

There was no event at the `STEP` point (after the object update, before the
pose flush and the collision check), and an event subscription could not be
frequency-limited, so neither could stand in for the other two.

## Decision

`sim.events` is the one mechanism. Callbacks and plugin `on_step` are
subscriptions on it; their public APIs stay and behave as before.

1. **`SimEvents.STEP` (`"step"`)** is emitted at the point where `STEP`
   callbacks run, with `dt` and `sim_time` (the time of the new state). Its
   name is the value of `CallbackPhase.STEP`, as `"post_step"` already was of
   `CallbackPhase.POST_STEP`: a callback phase *is* the event name.
2. **`events.on(event, handler, priority=0, frequency=None)`.** With a
   frequency the bus gives the subscription a `FrequencyGate` (the fixed grid
   of #80) and passes `dt` as the simulated time since the handler last ran.
   `frequency=0` means every emit, as it does for `register_callback()`, so
   moving a callback to `on()` keeps its schedule.
3. **Tiers.** A subscription has a tier: handlers from `on()` (0), callbacks
   from `register_callback()` (1), plugin `on_step` (2). An emit calls them
   ordered by `(tier, priority, registration order)`. This keeps the old order
   exactly: `post_step` handlers of any priority before `POST_STEP` callbacks,
   callbacks before plugins, each in registration / load order. New `step`
   handlers run before `STEP` callbacks, as `post_step` handlers already do
   before `POST_STEP` callbacks.
4. **Snapshot emits.** Subscriptions are stored as immutable tuples, rebuilt by
   `on()`/`off()`. An emit runs the subscriptions present when it starts, as
   `register_callback()` always did; before, a handler added by a handler of
   the same event ran in the same emit.
5. **Every phase for every tier.** `CallbackPhase` gains `PRE_STEP`, so a
   callback can run at any of the three step events, and a plugin's `on_step`
   takes a phase the same way (`SimPlugin.step_phase`: the class attribute
   `default_step_phase`, `register_plugin(phase=...)` or the YAML `phase`
   field; `step` by default; fixed once registered). Named `step_phase`, not
   `phase`, so a plugin's own state can keep that common name. The simulation emits all
   three events tier by tier, so at each one handlers, callbacks and plugins
   run in that order with their own frequency. Added in review: plugins could
   otherwise only reach `pre_step` or `post_step` through `events.on()` in
   `on_init()`, losing what `on_step` gives them (load order, the error log
   naming the plugin, the YAML `frequency`, the schedule reset).

## Open questions from the issue

**How does a frequency-limited subscription get the simulation time?** The bus
takes an optional `clock` returning `(now, timestep)`. `sim.events` is created
with one that returns the step-count-based schedule time (`_schedule_time()`),
the same time callbacks and plugins were scheduled on, so a 3 Hz handler and a
3 Hz callback fire on the same steps. A bus without a clock — a bare `EventBus`
or an entity's `obj.events` — raises `ValueError` for `frequency`. Frequency is
accepted only for the step events (`pre_step`, `step`, `post_step`), which are
emitted once per step and carry `dt`; on a discrete event (a collision) a
frequency limit would silently drop occurrences. Each event is scheduled on the
time it observes, so a `pre_step` handler is due when the previous state
reaches `k / frequency`.

**One exception policy?** Not one, but one *place*: the policy is a property of
the subscription, decided by the API that created it.

- `events.on()`: logged, the step goes on (unchanged; one buggy observer must
  not stop the others).
- `register_callback()`: propagates out of `step_once()` and skips the rest of
  the step (unchanged; it is documented, tested, and scenario scripts rely on
  a failing callback stopping the run).
- plugin `on_step`: logged with the plugin's name (unchanged).

Changing a policy would break the documented behaviour of a public API; the
issue's goal is one mechanism, not one behaviour. Propagation stays internal
(`EventBus._subscribe(..., propagate=True)`); a public opt-in is left for a
follow-up if a need appears.

**Argument shape and migration.** Events keep keyword arguments (`dt`,
`sim_time`); callbacks keep `(sim_core, dt)` and plugins `on_step(dt)` through
thin adapters, so no existing callback, the recorder, behaviour trees, the
agent-manager callbacks or the examples need to change. `events.on()` is the
keyword form of a callback; new code may use either.

## Preserved behaviour

Pinned by `tests/test_step_hooks.py` (passing before and after the change):
order across the three mechanisms and within each; the frequency grid and `dt`;
`frequency=0` meaning every step; the exception policies above; `unregister_callback`
removing the earliest registration across phases; registering during a step;
`events.clear()` leaving callbacks and plugins; schedules restarting with
`initialize_simulation()` / `reset()`; profiling keys (`callbacks` excludes
plugins, `events_post_step` apart from `callbacks_post_step`). Handlers of the
new `step` event are timed as `events_step`.

## Internal changes

- `sim._callbacks` is a read-only view built from the bus (same record shape:
  `func`, `phase`, `gate`), for the recorder, replay checks and tests.
- The kinematic replay profile checks only tier-0 handlers against its
  allowlist, and rejects a frequency-limited handler as it rejects a
  frequency-limited callback (its schedule is not checkpointed).
- `step_once()` emits `step` and `post_step` one tier at a time to keep the
  profiling breakdown; `emit()` from outside runs every tier.

## Performance

Subscriptions are kept as immutable tuples, pre-sorted and pre-split by tier,
so an emit does no sorting or copying. A step event with no subscriber costs a
single dict lookup (`has_handlers`), where the old step looped over the
callback list twice and over the plugins once. A callback's `_Subscription`
holds the callback itself as `call` and a weak reference to the simulation as
`owner`; the dispatch calls `call(owner(), dt)` with both arguments positional,
so no keyword-argument dict is built for a callback (other handlers get the
event's keyword arguments).

Step time (`step_once()` wall time, `p.DIRECT`, kinematic omni agents with far
goals, collision check at 10 Hz, timestep 0.1 s; one process per run pinned to
one CPU, runs of main and this branch interleaved; "hooks" = 10 `STEP`
callbacks, 5 `POST_STEP` callbacks at 2 Hz, 2 plugins (one at 5 Hz), a
`pre_step` and a `post_step` handler):

| Scene | Hooks | main (6ee451e) | this branch |
| --- | --- | --- | --- |
| 300 agents, 7 x 600 steps | none | 4.45 ms median (4.08 min) | 4.14 ms (4.08) |
| 300 agents, 7 x 600 steps | hooks | 4.43 ms (4.13) | 4.28 ms (4.12) |
| 1 agent, 9 x 5000 steps | none | 7.8 us (7.3) | 7.2 us (6.7) |
| 1 agent, 9 x 5000 steps | hooks | 13.9 us (13.6) | 15.0 us (14.1) |

With a fleet the difference is within run-to-run noise. Without hooks the step
is about 0.6 us faster; with 17 hooks (about 15 calls per step) it is about
1 us slower, under 0.1 us per call.

## Left out

- `SimObject.register_callback()` / `execute_callbacks()` (per-object,
  `last_executed`-based) and agent `AgentPlugin` updates are separate
  mechanisms, untouched.
- The ROS bridge's own throttled post-step handler list could move to
  `events.on(POST_STEP, ..., frequency=...)`.
- No API is deprecated here.
