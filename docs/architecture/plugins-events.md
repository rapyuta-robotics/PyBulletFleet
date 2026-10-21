# Plugins and Events

This page describes framework extension points for PyBulletFleet developers.
If you only configure or operate a simulation, use callbacks and the public
Python/ROS APIs instead of depending on this lifecycle.

## Choose the extension point

| Need | Use | Why |
| --- | --- | --- |
| Small scenario-specific update function | `register_callback()` | Minimal setup; stays local to the simulation |
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
simulation step by default; a configured `frequency` invokes it less often, on
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
effects idempotent where practical.

```python
from pybullet_fleet.events import SimEvents

def record_collision(obj_a, obj_b):
    ...

sim.events.on(SimEvents.COLLISION_STARTED, record_collision, priority=10)
```

Use events for transitions such as `ACTION_COMPLETED` or `COLLISION_STARTED`.
Do not subscribe to `PRE_STEP` just to poll state that can be queried on demand.

## Step-order and pose rule

Plugin `on_step`, callbacks, and pre-step handlers run in Phase 1 of
`step_once()`. Kinematic `set_pose()` calls are buffered there. Framework
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
