# Plugins and Events

This page describes framework extension points for PyBulletFleet developers.
If you only configure or operate a simulation, use callbacks and the public
Python/ROS APIs instead of depending on this lifecycle.

## Choose the extension point

| Need | Use | Why |
| --- | --- | --- |
| Small scenario-specific update function | `register_callback()` or `events.on(SimEvents.STEP, ...)` | Minimal setup; stays local to the simulation |
| Reusable lifecycle-managed integration | `SimPlugin` | Receives simulation core and init/reset/shutdown hooks |
| React only to a discrete state transition | `EventBus` subscription | Avoids polling every step |
| Expose a remote transport | Public Fleet API or ROS bridge | Keeps transport code outside the simulation loop |

`SimPlugin` is the framework equivalent of a world plugin. It is appropriate
for reusable integrations such as workcell and RMF bridge plugins, not for a
single action sequence in an example.

## Plugin lifecycle

Every hook is optional. A plugin receives `sim_core` at construction and follows
this lifecycle:

```text
construct -> on_init -> on_step (zero or more times) -> on_reset -> on_shutdown
```

`on_init()` runs after the world and robots have been created. Use it to inspect
agents, allocate resources, or subscribe to events. `on_step(dt)` runs at every
simulation step by default, at the `step` event; a `phase` (`pre_step`, `step`
or `post_step`, set by the class's `default_step_phase`, by
`register_plugin(phase=...)` or by the YAML `phase` field) moves it to another
step event, as for callbacks. A configured `frequency` invokes it less often, on
the fixed schedule described in [the two-phase step](two-phase-step.md), and
passes the simulated time since its previous run. `on_reset()` must discard state that does
not survive a reset, and `on_shutdown()` must release external resources before
PyBullet disconnects.

Plugins can be configured by a registry name or dotted Python class path:

```yaml
plugins:
  - class: my_package.plugins.TrafficMonitor
    frequency: 2.0
    config:
      log_interval: 5.0
```

Configuration keys are matched to the subclass constructor. Keep plugin
configuration explicit and validate any values that affect external I/O.

## EventBus contract

`sim.events` is the global bus for simulation lifecycle, object/agent lifecycle,
collision, pause/resume, and fleet-command events. Each object also lazily
creates `object.events` for entity-local events. Use `SimEvents` constants for
the built-in names; custom string names are permitted for package-local events.

Handlers run in ascending priority order; equal priorities preserve registration
order. An exception is logged and does not prevent later handlers from running.
Handlers should therefore keep work bounded and make externally visible side
effects idempotent where practical. An emit calls the subscriptions present when
it starts -- handlers, and for the step events also callbacks and plugins; one
added or removed by a handler, callback or plugin takes effect from the next
emit.

## Per-step hooks

`sim.events` is the one mechanism behind every per-step hook. Three events are
emitted once per step, each with `dt` and `sim_time`:

| Event | When | `sim_time` |
| --- | --- | --- |
| `SimEvents.PRE_STEP` | start of the step, before the clock advances | previous state |
| `SimEvents.STEP` | after the object update, before the pose flush and the collision check | new state |
| `SimEvents.POST_STEP` | end of the step, after the collision check | new state |

`register_callback(f, frequency, phase)` subscribes `f` to the `phase` event
(`CallbackPhase.PRE_STEP`, `CallbackPhase.STEP` and `CallbackPhase.POST_STEP`
are the `pre_step`, `step` and `post_step` events), and a plugin's `on_step`
subscribes to its `step_phase` event (`step` unless set). They form tiers that
keep a fixed order within each event:

1. handlers from `events.on()`, by `priority`, then registration order;
2. `register_callback()` callbacks, in registration order;
3. plugin `on_step`, in load order.

So a `post_step` handler of any priority runs before every post-step callback,
and every callback of an event runs before any plugin of it. Each of the three
can run at any of the three events, with a `frequency`; they differ only in
call shape and error handling:

| Registered with | Called as | An exception |
| --- | --- | --- |
| `events.on(event, h, frequency=...)` | `h(dt=..., sim_time=...)` | is logged; the step goes on |
| `register_callback(f, frequency, phase)` | `f(sim_core, dt)` | propagates out of `step_once()` |
| `SimPlugin.on_step` with `frequency` | `plugin.on_step(dt)` | is logged; the step goes on |

All three can be frequency-limited and share one schedule: due times on a
fixed grid of `k / frequency` seconds of simulated time, at most once per step,
first one interval into the run, and `dt` is the simulated time since the hook
last ran. A 3 Hz handler, callback and plugin run on the same steps.

```python
from pybullet_fleet.events import SimEvents

def publish(dt, sim_time):
    ...  # runs when the state reaches 0.1 s, 0.2 s, ...; dt is about 0.1

sim.events.on(SimEvents.POST_STEP, publish, frequency=10.0)
```

`frequency` is accepted for the three step events only, and only on
`sim.events`, which carries the simulation clock; a per-entity `obj.events`
bus or a bare `EventBus()` raises `ValueError`. Each event is scheduled on the
`sim_time` it carries, so a `pre_step` handler is due when the previous state
reaches a grid point. `initialize_simulation()` and `reset()` restart every
schedule. `events.off()` and `events.clear()` remove only handlers registered
with `events.on()`; use `unregister_callback()` for callbacks. As for
`register_callback()`, `frequency=None` or `frequency=0` means every step.

```python
from pybullet_fleet.events import SimEvents

def record_collision(obj_a, obj_b):
    ...

sim.events.on(SimEvents.COLLISION_STARTED, record_collision, priority=10)
```

Use events for transitions such as `ACTION_COMPLETED` or `COLLISION_STARTED`.
Do not subscribe to `PRE_STEP` just to poll state that can be queried on demand.

## Step-order and pose rule

Plugin `on_step`, callbacks and handlers at the `pre_step` and `step` events run
in Phase 1 of `step_once()`; at `pre_step` nothing has moved yet and `sim_time`
is still the previous state's. Kinematic `set_pose()` calls are buffered there. Framework
getters, such as `agent.get_pose()`, see the new cached pose; direct PyBullet
queries still see the previous pose until the pose flush completes.

Avoid raw `p.getBasePositionAndOrientation()` and `p.getAABB()` in `on_step`
when a same-step value is required. Use framework getters, a post-step event, or
defer the operation to the next step. The complete ordering and physics boundary
are documented in [Two-Phase Step](two-phase-step).

## Related public examples

`basics/event_bus_demo.py` is the runnable EventBus example. The ROS/RMF bridge
uses plugins for its in-process path; its deployment-level choices are documented
separately in [ROS 2 and Open-RMF](../ros2/index).
