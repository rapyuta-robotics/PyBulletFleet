# Controller Configuration

Controllers turn goals, paths, and velocity commands into agent motion. This
guide covers per-agent controller selection, controller chains, and manager
level batch execution.

## Choose a controller

Use an explicit `controller.type` for new code. `motion_mode` remains a
compatibility fallback for selecting the built-in omni or differential base
controller, but it is not the recommended configuration interface.

```python
from pybullet_fleet.agent import Agent

# Built-in controller by registry name.
agent = Agent.from_urdf("robots/mobile_robot.urdf", controller="differential")

# Recommended config form: controller type plus shared limits and implementation options.
agent = Agent.from_urdf(
    "robots/mobile_robot.urdf",
    controller={
        "type": "differential",
        "max_linear_vel": 1.5,
        "max_angular_vel": 2.0,
        "wheel_separation": 0.3,
    },
)
```

### Accepted `controller` forms

| Form | Example | Use |
|------|---------|-----|
| `None` | `controller=None` | Use the legacy `motion_mode` fallback. The default selects `DifferentialController`. |
| String | `controller="omni"` | Select one registered controller. |
| Mapping | `controller={"type": "differential", ...}` | Recommended for YAML and ordinary Python configuration. |
| List / tuple | `controller=[{"type": "omni"}, {"type": "patrol", ...}]` | Build a controller chain. |
| `Controller` instance | `controller=MyController(...)` | Python-only injection of a pre-built controller. |
| `ControllerParams` instance | `controller=ControllerParams(...)` | Python-only shared-parameter override while retaining the `motion_mode` base. |

YAML can use `null`, a string, a mapping, or a list of mappings. Controller
instances and `ControllerParams` instances are Python-only.

```yaml
entities:
  - urdf_path: robots/mobile_robot.urdf
    controller:
      type: differential
      max_linear_vel: 1.5
      max_angular_vel: 2.0
      wheel_separation: 0.3
```

## Controller chains

An agent has one base movement controller and may have high-level controllers
above it. During an update, high-level controllers run first and set goals for
the base controller to execute.

```yaml
entities:
  - urdf_path: robots/mobile_robot.urdf
    controller:
      - type: differential
        default_direction: auto
      - type: patrol
        waypoints: [[1.0, 0.0, 0.05], [1.0, 1.0, 0.05]]
        loop: true
```

- A `KinematicController` entry (`omni`, `differential`, or a custom
  kinematic class) becomes the base controller.
- A non-kinematic entry such as `patrol` or `random_walk` is appended as a
  high-level controller.
- If no kinematic entry is present, the base is created from `motion_mode`.
  Thus a lone `patrol` controller still has a base controller to move the
  agent.

Each mapping accepts `type` (a registered name) or `class` (a dotted Python
class path). Keys matching `ControllerParams` configure shared limits; the
remaining keys are passed to the selected controller's constructor. Batch
controller types are not valid as an entity `controller.type`; configure them
on an `AgentManager` instead.

At runtime, use `set_controller()` to replace the base controller, and
`add_controller()` / `remove_controller()` to manage high-level controllers:

```python
from pybullet_fleet.controller import create_controller

agent.set_controller(create_controller("differential", {"max_linear_vel": 1.5}))
agent.add_controller(create_controller("patrol", {"waypoints": [[1.0, 0.0, 0.05]]}))
```

## `motion_mode` compatibility

```{warning}
**Deprecated for new configurations:** use `controller.type` instead of
`motion_mode`. `motion_mode` remains supported for existing configurations and
fallback behaviour, but new YAML, examples, and integrations should not add it.
```

`motion_mode` is retained for existing configurations and for the coarse
mobile-base hint used by batch controllers. When no kinematic controller is
specified, it selects the built-in base:

| `motion_mode` | Fallback base controller |
|---------------|--------------------------|
| `omnidirectional` | `OmniController` |
| `differential` | `DifferentialController` |

An explicit kinematic `controller.type` selects the base controller instead.
Prefer omitting `motion_mode` in new entity configurations. The legacy
`agent.set_motion_mode()` method replaces the base with the corresponding
built-in controller; use `set_controller()` when changing controllers at
runtime.

## `ControllerParams`

`ControllerParams` holds the limits and behaviour flags shared by kinematic
controllers. `Agent.controller_params` is the authoritative instance. The
legacy `Agent.max_linear_vel`, `max_angular_vel`, `max_linear_accel`,
`max_linear_decel`, and `max_angular_accel` properties are read-only views of
it.

```python
agent.controller_params.max_linear_vel = [2.0, 1.0, 0.0]

# A vehicle that brakes harder than it accelerates.
agent.controller_params.max_linear_accel = [1.5, 1.5, 0.0]
agent.controller_params.max_linear_decel = [3.0, 3.0, 0.0]
```

| Field | Meaning |
|-------|---------|
| `max_linear_vel` | Scalar magnitude cap, or `[vx, vy, vz]` per-axis cap in the body frame. |
| `max_angular_vel` | Scalar magnitude cap, or `[wx, wy, wz]` per-axis cap. Differential drive uses `wz`. |
| `max_linear_accel` | Scalar acceleration cap, or `[ax, ay, az]` per-axis cap in the body frame, used to construct translation trajectories. Differential drive uses `ax`. |
| `max_linear_decel` | Braking cap, same form and frame as `max_linear_accel`. Unset mirrors `max_linear_accel`, which is the symmetric trapezoid every configuration had before this field existed. |
| `max_angular_accel` | Scalar acceleration cap, or `[αx, αy, αz]` per-axis cap, used to construct rotation trajectories. Differential drive uses `αz`. |
| `cmd_vel_timeout` | Velocity-command watchdog in seconds; `0.0` disables it. |
| `navigation_2d` | `True` preserves current Z for pose trajectories. `None` and `False` follow goal Z. |
| `default_direction` | Default `set_path()` direction for differential drive. |
| `goal_replacement` | What a moving agent does with a goal it cannot reach from its current motion: `"brake"` (default), `"immediate"`, `"reject"` or `"wait_complete"`. See [A new goal while moving](#a-new-goal-while-moving). |

All numeric values default to `None`, which resolves to the framework default
when used. This lets a manager provide only the fields that an agent did not
set explicitly.

### A new goal while moving

An agent (omnidirectional or differential, per-agent or batched) that gets a
new goal or path while it is moving continues its motion when it can: the
straight-line trajectory to the first waypoint starts from its current
velocity, signed along the line. A goal ahead keeps the speed; a goal behind
is reached by slowing, reversing and driving back. Slowing down before a
reversal is the trajectory's first phase, which uses `max_linear_accel`. A
differential agent continues only into a goal that needs no turn: one straight
ahead, or straight behind with `direction="auto"` or `"backward"`.

The agent cannot continue into a goal inside the braking distance at
`max_linear_decel`, where the agent is, off the line of motion (a
straight-line trajectory cannot carry the sideways velocity), or, for
differential drive, one that needs a turn in place. `goal_replacement`
decides what happens then:

| `goal_replacement` | A goal the agent cannot continue into | A goal it can continue into |
|--------------------|---------------------------------------|-----------------------------|
| `"brake"` (default) | Brakes along the current motion at `max_linear_decel`, then starts the goal from rest. | Taken at once. |
| `"immediate"` | Starts the goal from rest at once: the velocity drops to zero in one step, ignoring `max_linear_decel`. | Taken at once. |
| `"reject"` | Refused: `set_path()` / `set_goal_pose()` return `False` and the current motion and path continue unchanged. | Taken at once. |
| `"wait_complete"` | Held until the current path is finished, then started from rest. | Also held: the current goal has priority. |

In short: a goal the agent can continue into -- further along its line of
motion, beyond the braking distance, such as an extended target -- is always
taken at once and keeps the speed, whatever the mode (`reject` included); the
mode decides only what happens to the other goals. The exception is
`wait_complete`, which holds every goal while a path is in progress.

To stop at a point further along the line, send that point as the goal: the
agent continues into it and comes to rest there (if it is beyond the braking
distance; a closer one is a goal it cannot continue into). To stop at points on
the way and then go on, send them together as the waypoints of one path: every
waypoint is reached at rest.

At rest every goal is taken at once. A held goal (`brake`, `wait_complete`) is
accepted: `set_path()` returns `True`. Only the latest one is kept; a newer
request replaces it, and `stop()`, `brake()` or a velocity command drops it.
Each drop is logged at INFO with the agent, the goal and the reason. Under a
velocity command there is no path to wait for, so `wait_complete` brakes like
`brake`.

Set the mode for one agent, for every agent of a manager, or globally:

```python
from pybullet_fleet.types import GoalReplacement

# One agent (the string value "reject" works too).
agent.controller_params.goal_replacement = GoalReplacement.REJECT

# Every agent of a manager that does not set its own.
manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni", "goal_replacement": "wait_complete"})
```

The global default is `_defaults.CONTROLLER["goal_replacement"]`, which the
`PBF_CONTROLLER_GOAL_REPLACEMENT` environment variable overrides (see
[Defaults and environment overrides](../configuration/reference.md#defaults-and-environment-overrides)).

The mode is read for each request, so a change applies to the paths requested
after it, also while the agent moves. A path already held keeps the decision
made when it was held: it starts once the agent stops (or the current path
ends), whatever the mode is by then.

Under `reject`, check the result:

```python
if not agent.set_goal_pose(goal):  # True when taken, False when refused
    print(agent.path_rejection([goal]))  # the reason, e.g. "goal is off the line of motion"
```

`MoveAction`, `PatrolController`, `RandomWalkController` and the behaviour
tree's `NavigateTo` retry a refused goal on their next update;
`FleetCommandDispatcher.navigate()` reports it in the ack's `rejected` map. In
the other modes nothing is refused, so these never retry.

### Stopping: `brake()` and `stop()`

`agent.brake()` brings a moving agent (per-agent or batched) to rest at
`max_linear_decel`: it keeps its line of motion and heading, drops the rest of
its path and any held goal, and stops at the braking point `|v|^2 / (2 * decel)`
ahead. A differential agent brakes straight along its heading, backwards when
it is backing up. `brake()` also ends a velocity command the same way, and it
does not wait for the current path under `wait_complete`. It returns `False`
when there is nothing to brake (at rest, fixed base). `agent.stop()` is the
immediate stop: the velocity is zeroed at once.

To change direction mid-motion, send the new goal: under the default `brake`
mode the agent brakes and then heads for it. `brake()` leaves the action queue
alone; a `MoveAction` still executing ends `CANCELLED` once the agent is at
rest, short of its path's end, and the queue moves on to the next action, so
call `clear_actions()` first to drop the queued actions too.

#### Custom braking

The controller does the braking: `agent.brake()` calls the controller's
`brake(agent)`, and the `brake` replacement mode runs the same braking before
the held goal. Both use one method, `braking_request(agent)`, which returns the
path request that brings the agent to rest (`None` at rest). Override it in a
`KinematicController` subclass (or a batch controller) to brake differently,
for example to stop at a set point further ahead:

```python
from pybullet_fleet.controller import OmniController


class CarefulOmni(OmniController):
    def braking_request(self, agent):
        request = super().braking_request(agent)  # stop |v|^2 / (2 * decel) ahead
        ...  # adjust or replace request.path
        return request


agent.set_controller(CarefulOmni(params))
```

The request starts from the current motion, so it must continue it: on its
line, no closer than the braking distance. One that does not is not started;
the controller logs a warning naming the agent and brakes at the default
braking point instead. A
controller that does not override `brake()` from the base `Controller` cannot
brake, and `agent.brake()` returns `False`.

## Batch controllers

Batch controllers use one vectorised `batch_advance()` call for all agents in
an `AgentManager`. Use them when profiling shows per-agent controller dispatch
is a meaningful cost. They do not replace collision detection, pose commit, or
spatial-grid work.

| Batch type | Per-agent base |
|------------|----------------|
| `batch_omni` | `omni` |
| `batch_differential` | `differential` |

They honour `max_linear_decel` like the per-agent controllers. The batched
path evaluates a closed form rather than the `TwoPointInterpolation` itself,
and that form takes the braking scalar as its own argument, so the two agree
to floating-point noise on the same profile.

Rotation is the exception: angular limits carry no separate deceleration, so
a turn is always symmetric.

### Python API

```python
from pybullet_fleet.agent_manager import AgentManager

mgr = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
agents = mgr.spawn_agents_grid(100, grid_params, spawn_params)

# Agent-level commands remain the normal public API.
agents[0].set_path([goal_pose])
```

Agents added with `mgr.add_object(agent)` are registered with the active batch
controller automatically. Existing manager agents can be converted later, or
returned to per-agent execution:

```python
mgr.enable_batch("batch_omni")
mgr.disable_batch()
```

`MoveAction`, `PickAction`, and other action-queue operations still run on
each agent. Batch execution replaces only the base controller's movement
calculation.

### YAML: named managers

Declare a named manager and route entity groups to it. `fleet_controller.type`
enables batch execution; omit `type` to retain per-agent execution while
sharing controller defaults.

```yaml
managers:
  - name: delivery_fleet
    fleet_controller:
      type: batch_differential
      max_linear_vel: 1.5
      max_linear_accel: 2.0
      navigation_2d: true

entities:
  - urdf_path: robots/mobile_robot.urdf
    manager: delivery_fleet
    controller:
      type: differential
    grid:
      count: 50
      spacing: [2.0, 2.0]
```

Manager defaults are applied field by field: an entity's explicitly set
`ControllerParams` fields win, while fields still set to `None` inherit from
`fleet_controller`. For per-agent execution with shared defaults, omit the
batch type:

```python
mgr = AgentManager(
    sim_core=sim,
    fleet_controller={"max_linear_vel": 1.0, "navigation_2d": True},
)
```

Multiple named managers may use different batch types. Retrieve a manager
after config loading with `sim.get_manager("delivery_fleet")`.

### Custom batch controllers

Subclass `BatchKinematicController`, give it a registry name, and import the
module before the manager is constructed:

```python
from pybullet_fleet.controllers.batch_base import BatchKinematicController

class MyBatchController(BatchKinematicController):
    _registry_name = "my_batch"

    def batch_advance(self, dt):
        ...

mgr = AgentManager(sim_core=sim, fleet_controller={"type": "my_batch"})
```

For a custom per-agent controller, use the same `type` registry mechanism or
a dotted `class` path in an entity's `controller` mapping.

## See also

- [Multi-Robot Fleet Tutorial](../examples/multi-robot-fleet) — Config-driven named managers and batch fleets
- [Two-Phase Step](../architecture/two-phase-step) — Where batch movement runs in the step lifecycle
- [Benchmark Results](../benchmarking/results) — Reproducible performance measurements
