# Two-Phase Step

PyBulletFleet's `step_once()` exposes a **two-phase pose-commit contract**.
This page explains that contract, the following synchronization work, and what
they mean for callback / plugin authors.

> If you only call high-level APIs (`Agent.add_action`, `controller.set_velocity`,
> `agent.get_pose`), the two-phase split is fully transparent — you do not need
> to change any code. This page is for callback / plugin / ROS-bridge authors
> who interact with poses or PyBullet directly inside a step.

## Design rationale

Separating base-pose computation from the corresponding **PyBullet C-API
writes** lets the simulator:

- batch pose writes for N agents in one pass (`set_poses()`),
- vectorize per-agent compute with NumPy across N agents,
- guarantee that collision detection always sees a consistent world snapshot
  (all writes applied, all AABBs refreshed) within a single step.

## Two-phase pose commit in the full step timeline

```
step_once()
├── PHASE 1 — UPDATE         (base-pose writes are buffered)
│     ├── pre-step events
│     ├── for manager with a batch controller:  batch_advance(dt)
│     ├── for obj in sim_objects:  obj.update(dt)
│     │       └── controller.compute → agent.set_pose(...)   ← buffered
│     ├── user callbacks               (registered via register_callback)
│     └── plugin on_step hooks
│
├── PHASE 2 — POSE FLUSH     (tight C-API loop)
│     └── for obj_id in _pending_pose_ids:
│             p.resetBasePositionAndOrientation(...)
│     └── recompute and flush link-attached children after their parent moves
│
├── stepSimulation()         (only when physics is enabled)
│     └── refresh AABBs + spatial grid for physics-driven objects
│
├── POST-COMMIT SYNCHRONIZATION
│     └── for each flushed kinematic object:
│             refresh AABB (p.getAABB)
│             update spatial-hash grid
│
├── check_collisions         (frequency-gated; sees up-to-date AABBs/grid)
├── post-step events
├── post-step callbacks      (register_callback(..., phase="post_step"))
└── monitor update
```

The set of objects with pending base-pose writes (`_pending_pose_ids`) is built
up during Phase 1, snapshotted for pose commit and post-commit synchronization,
and then cleared. Link-attached
children may add a second, internal flush after their parent's new base pose is
visible to PyBullet.

## `set_pose()` behaviour: buffered vs immediate

Calls to `SimObject.set_pose()` (and `agent.set_pose_raw()`) behave
differently depending on **when** they are made:

| When | Behaviour |
|---|---|
| **Outside `step_once()`** (test setup, REPL, between steps) | Immediate — writes to PyBullet right away. |
| **Inside `step_once()`** (controllers, callbacks, plugin `on_step`) | **Buffered** — only the cache is updated; the actual PyBullet base pose is written at Phase 2 later in the same step. |

The buffering rule applies to every movable `SimObject` that calls `set_pose()`
inside the step, including a physics-driven object that is explicitly
teleported. In normal use, kinematic objects are moved through this path;
physics-driven bodies should be allowed to move through `stepSimulation()`
instead of receiving repeated pose writes.

For a buffered **kinematic** write, the framework cache is the source of truth
mid-step, so:

- ✅ `agent.get_pose()` / `sim_object.get_pose()` — returns the new cached pose.
- ⚠️ `p.getBasePositionAndOrientation(object_id)` — returns the **previous
  step's** pose during Phase 1, because PyBullet hasn't been written yet.
- ⚠️ PyBullet AABB queries and the framework collision/spatial-grid caches are
  not a same-step pose API. The framework refreshes kinematic AABBs and grid
  entries during post-commit synchronization, after the base-pose flush.

### Rule of thumb

> Inside a callback or plugin `on_step`, **always use the framework getter**
> (`agent.get_pose()`, `sim_object.get_pose()`), never the raw PyBullet API.

If you genuinely need PyBullet to reflect the latest pose mid-step (e.g. for a
raycast against just-moved agents), subscribe to the `POST_STEP` event or defer
the operation to the next step — by then Phase 2 has flushed:

```python
from pybullet_fleet.events import SimEvents

sim.events.on(SimEvents.POST_STEP, on_post_step)
```

`register_callback()` runs in Phase 1 by default. Pass
`phase=CallbackPhase.POST_STEP` (or `"post_step"`) for
a callback that runs right after the `POST_STEP` event, with its own
`frequency` -- for example to publish or record each step's outcome at a fixed
rate. Its schedule runs on `sim_time`, the time of the state it sees, so a 10 Hz
post-step callback runs when the poses reach 0.1 s, 0.2 s, ...

`set_pose()` from a `POST_STEP` handler or a post-step callback writes to
PyBullet at once, but the step's collision check has already run. A collision
such a write causes is found by the next collision check.

### Which time a pose belongs to

One rule: **`sim_core.sim_time` (and `step_count`) is the time (and step) of
the state you see.** `step_once()` advances the clock and the step counter
right after the `pre_step` event, before anything computes the new state:

| Where | `sim_time` / `step_count` | State it describes |
|---|---|---|
| `pre_step` subscriber (payload `sim_time` too) | previous step, `t` / `k` | the previous step's state |
| object update, Phase-1 callbacks, plugin `on_step`, behavior trees | new step, `t + dt` / `k + 1` | being computed for `t + dt` |
| collision check, `post_step` subscriber (payload `sim_time` too), post-step callbacks | `t + dt` / `k + 1` | the step's result |
| outside a step | completed step, `t + dt` / `k + 1` | the completed step's result |

`pre_step` is the only place that still sees the previous step. Everything
read anywhere else -- a pose, an action's start or end time, a collision
observation -- is stamped correctly with `sim_time`.

A step that raises (an exception from a Phase-1 callback, say) propagates the
exception and leaves the simulation in an unspecified state; nothing is rolled
back.

`sim_time` is a read-only view of the simulation's single clock. Outside a step it
equals the completed boundary returned by `get_completed_step_boundary()`. The
only supported way to move the clock is `restore_completed_step_boundary()`,
when resuming a fresh simulation from a checkpoint.

Position controllers evaluate their trajectories at `sim_time`, so the pose a
step ends with is the trajectory at the end of the step, the same rule
kinematic joints follow. A trajectory started now begins at `sim_time`: a path
set in a `pre_step` subscriber therefore moves in that same step; one set later
in the step, or outside it, moves from the next step. With `Agent.set_path()`'s
default `auto_approach=True` (which `set_goal_pose()` uses), a goal more than
0.5 m away gets the current pose as a zero-length first waypoint, and that
waypoint takes the first step.
A goal replaced mid-motion continues from the speed of the pose in effect at
`sim_time`.

### Frequency-limited handlers

Callbacks and plugins with a `frequency`, `collision_check_frequency` and the
DataMonitor refresh run on a fixed grid of simulated time, `k / frequency`
seconds from the start of the run: on the first step at or after each due
time, at most once per step. A handler registered mid-run first runs at the
next due time; missed due times never cause a burst of runs. The schedule
reads the time as the step count times the timestep (counted again from any
change of timestep or restore) rather than the accumulated simulation time,
whose float error grows with the run, and tolerates the rounding left. So a
10 Hz handler on a 0.1 s timestep runs every step, however long the run, and
an interval that is not a whole number of timesteps does not drift.

### What Phase 1 does not promise

Phase 1 batches framework **base-pose** writes. It is not a general prohibition
on PyBullet calls: joint control may issue PyBullet calls, and application
callbacks/plugins can call PyBullet directly. Framework extensions should avoid
direct base-pose and AABB reads/writes in Phase 1 when they require a
same-step-consistent scene; use the framework pose APIs and the post-step event
instead.

## Profiling

The per-step profiling dict (returned by `MultiRobotSimulationCore.get_profiling_stats()`)
exposes the phase split:

| Key | What it measures |
|---|---|
| `phase1_update` | End-to-end Phase 1 wall time (pre-step events → batch-controller advance → object update loop → callbacks → plugin hooks). Framework base-pose writes are buffered, but joint-control or user PyBullet calls may still occur. |
| `phase2_pose_flush` | Time spent writing buffered poses to PyBullet (`resetBasePositionAndOrientation` loop). |
| `step_simulation` | `p.stepSimulation()` plus AABB/spatial-grid refresh for physics-driven objects. |
| `phase3_aabb_grid_flush` | Time spent refreshing AABBs and spatial-hash grid entries for the kinematic objects flushed in Phase 2. This is the implementation/profiling key; conceptually it is post-commit synchronization, not a third public phase. |
| `agent_update` | Time spent in the per-object `update(dt)` loop inside Phase 1 (subset of `phase1_update`). |

## See also

- {doc}`overview` — overall architecture
- {doc}`../how-to/custom-profiling` — adding your own profiling metrics

## Owned navigation replay sessions

`ReplaySession.step()` applies its ordered input batch at the step-start time,
then calls the ordinary `MultiRobotSimulationCore.step_once()` and records
results once the step has completed. Replay control and required
recording calls stay in the session, outside EventBus exception isolation.
The core has no replay-specific step hooks or mutation guards; ordinary
simulations keep their existing step and event behavior.

The replay transition is `S_k -> inputs(k, order) -> update -> S_(k+1)`.
Input/event records identify transition `k`; observations identify completed
state `k+1`. Replay uses `k * dt` as its canonical step-start timestamp; it does
not change legacy timestamps for ordinary runs or ROS publication. Custom
state-mutating event handlers are unsupported in the v1 replay session.
