# Tutorial 8: Event Bus

The Event Bus lets an application observe discrete simulation and entity
transitions without polling every object on every simulation step. It is aimed
at integrations, logging, UI updates, and coordination code rather than robot
motion commands.

**Source file:** [`examples/basics/event_bus_demo.py`](https://github.com/yuokamoto/PyBulletFleet/blob/main/pybullet_fleet/examples/basics/event_bus_demo.py)

## Run the demo

Run the event example from a source checkout, or copy it from the installed
package first:

```bash
python examples/basics/event_bus_demo.py
```

The demo subscribes to global simulation events, per-agent events, and a custom
event. Global events are accessed through `sim.events`; use an entity's `events`
bus when the handler concerns only that entity.

```python
from pybullet_fleet.events import SimEvents

def on_complete(agent, action, status):
    print(agent.name, status)

sim.events.on(SimEvents.ACTION_COMPLETED, on_complete)
```

## Built-in events

Use `SimEvents` constants for built-in event names. They prevent spelling
mistakes while still allowing custom string names for application events.

| Group | Event | Global handler arguments | Per-entity handler arguments |
|---|---|---|---|
| Step | `PRE_STEP`, `STEP`, `POST_STEP` | `dt`, `sim_time` | — |
| Object lifecycle | `OBJECT_SPAWNED`, `OBJECT_REMOVED` | `obj` | no arguments |
| Agent lifecycle | `AGENT_SPAWNED`, `AGENT_REMOVED` | `agent` | — |
| Collision | `COLLISION_STARTED`, `COLLISION_ENDED` | `obj_a`, `obj_b` | `other` |
| Agent update | — | — | `PRE_UPDATE`: `dt`; `POST_UPDATE`: `dt`, `moved` |
| Action | `ACTION_STARTED` | `agent`, `action` | `action` |
| Action | `ACTION_COMPLETED` | `agent`, `action`, `status` | `action`, `status` |
| Simulation state | `PAUSED`, `RESUMED` | no arguments | — |

Accept `**_` in a handler when it only needs some of the documented arguments;
this also makes a handler resilient when the framework adds context later.

```python
def on_collision(obj_a, obj_b, **_):
    print(f"{obj_a.name} collided with {obj_b.name}")

sim.events.on(SimEvents.COLLISION_STARTED, on_collision)
```

## Choosing the event scope

| Scope | Access | Use for |
|---|---|---|
| Simulation-wide | `sim.events` | Global logging, fleet coordination, objects entering or leaving the simulation |
| Per entity | `agent.events` / `sim_object.events` | Behavior tied to one robot or object |
| Application-defined | `emit()` on the relevant bus | Events produced by application or integration code |

## Registration, order, and cleanup

`on()` registers a handler and `off()` removes that exact handler object. Lower
priority values run first; handlers with the same priority retain their
registration order.

```python
def record_completion(agent, action, status, **_):
    ...

# Record before a default-priority UI handler.
sim.events.on(SimEvents.ACTION_COMPLETED, record_completion, priority=-10)

# Remove the subscription when this integration is shut down.
sim.events.off(SimEvents.ACTION_COMPLETED, record_completion)
```

Use `clear(event)` to remove all handlers for one event, or `clear()` to reset a
bus entirely. `has_handlers(event)` is useful when an integration wants to
avoid producing optional event data unless it is observed.

## Custom events

Custom event names are ordinary strings. They are local to the selected bus;
emitting on `sim.events` does not also emit on every entity bus.

```python
def on_battery_low(agent_id, level, **_):
    print(f"{agent_id}: battery at {level:.0%}")

sim.events.on("battery_low", on_battery_low)
sim.events.emit("battery_low", agent_id=agent.object_id, level=0.15)
```

## Handler rules

Event handlers run synchronously in the simulation loop. An exception is logged
and later handlers still run, but a slow handler delays the next simulation
step. Use events for transitions such as `ACTION_COMPLETED` or
`COLLISION_STARTED`; keep `PRE_STEP`, `STEP`, `POST_STEP`, and per-entity update
handlers bounded. Move blocking I/O or expensive analysis to another queue or
thread.

A handler of a step event can run at a fixed rate instead of every step:
`sim.events.on(SimEvents.POST_STEP, publish, frequency=10.0)` calls `publish`
when the simulated time reaches 0.1 s, 0.2 s, ..., with `dt` the time since it
last ran -- the same schedule as `register_callback(..., frequency=10.0)`.

Developers writing reusable integrations should read
[Plugins and Events](../architecture/plugins-events) for lifecycle and
buffered-pose rules. In particular, framework pose getters reflect same-step
buffered changes, while direct PyBullet queries may not until the step flushes.

## See also

- [Tutorial 1 — Spawning Objects](spawning-objects): entities that produce lifecycle and collision events
- [Tutorial 2 — Action System](action-system): `ACTION_STARTED` and `ACTION_COMPLETED` handlers
- [Tutorial 9 — Plugins](plugins): reusable lifecycle-managed extensions
- [Plugins and Events](../architecture/plugins-events): extension-point choice, lifecycle, and buffered-pose rules
- [Two-Phase Step](../architecture/two-phase-step): exact handler timing relative to pose flush and physics
