# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Snapshot/replay entries in this section describe development and validation
profiles. A declared kinematic profile supports fixed-arm and mobile-arm validation programs, but
they do not provide general save/load for arbitrary simulations and
are not recommended for production use yet.

### Added

- `ControllerParams.stop_on_collision`: an agent that sets it stops dead, as
  `stop()` does, on the step a collision starts that involves it or an object
  attached to it (a carried load stops its carrier), instead of driving or
  braking on past the point of contact. A contact with its own attached
  object does not count, and an agent struck while it stands is left alone.
  Off by default; per agent, through a fleet's
  controller config, or for every agent with `PBF_CONTROLLER_STOP_ON_COLLISION`.

- `Agent.brake()`: come to rest at `max_linear_decel` along the current line of
  motion, keeping the heading and dropping the rest of the path, for
  omnidirectional and differential-drive agents (per-agent and batched) under
  a path or a velocity command. A differential agent brakes straight along its
  heading, backwards when it is backing up. `stop()` remains the immediate
  stop. Controllers gain `continuation_velocity(agent)`, the velocity a new
  path continues from.

- Braking is a controller method: `Agent.brake()` delegates to the
  controller's `brake(agent)`, and both it and `goal_replacement="brake"` take
  the braking path from `braking_request(agent)`, which a custom controller
  overrides to brake differently. A request that cannot continue the current
  motion is not started: a warning naming the agent, and the default braking
  point instead. The base `Controller.brake()` returns
  `False` (cannot brake).

- `ControllerParams.goal_replacement` (`GoalReplacement`): what a moving agent
  does with a goal it cannot reach from its current motion within its limits.
  `brake` (the default) brakes along the current motion at `max_linear_decel`
  and starts the goal from rest; `immediate` starts it from rest at once;
  `reject` refuses it (`set_path` returns `False`); `wait_complete` holds every
  goal set while a path is in progress, reachable or not, and starts it once
  that path is finished. A goal the agent can reach is taken at once,
  continuing the motion, in every mode but `wait_complete`. Only the latest
  held goal is kept; a newer request, `stop()`, `brake()` or a velocity
  command drops it, with an INFO log line naming the agent, the goal and the
  reason. The global default is `_defaults.CONTROLLER["goal_replacement"]`
  (environment variable `PBF_CONTROLLER_GOAL_REPLACEMENT`); a manager's
  `fleet_controller` and an agent's own `controller` settings override it.
  Per-agent and batched, omnidirectional and differential controllers behave
  alike. Differential agents now continue the motion into a goal that needs no
  turn (issue #78). Checkpoints record the held goal and the braking state;
  the kinematic profile's controller state is now version 3, and the omni
  checkpoint proof's record version 3.

### Changed

- `ignore_attached_collision` now covers every object on one attachment tree,
  not only a direct parent and child: two loads on one carrier and a load on a
  load no longer count as colliding, the same rule `stop_on_collision` uses.

- Setting a goal on a moving agent is cheaper: the check whether the agent
  can continue into it and the trajectory it then follows share one solve
  instead of solving the same trajectory twice, and the differential heading
  computation avoids a generic cross product. Per goal, 5-45% less time
  across per-agent and batched, omnidirectional and differential controllers
  (`benchmark/experiments/goal_replacement_cost.py`).

- The ROS 2 bridge Docker images build from the Open-RMF demos image pinned
  by digest (`RMF_DEMOS_IMAGE` build argument) instead of the nightly
  `jazzy-rmf-latest` tag, whose rebuilds changed the demo worlds' model URIs
  and broke the Fuel model cache stage. The Fuel model download now always
  creates the cache directory and fails with an explanation when the image has
  no demo worlds or the worlds name no Fuel models (the latter can be allowed
  with `ALLOW_NO_FUEL_MODELS=1`).

- Spawning an agent no longer copies the controller's construction config a
  second time for its spawn descriptor, and the parameter names of spawn
  factories and controller constructors are introspected once per callable
  instead of once per object: spawning 1000 agents takes 0.30 s instead of
  0.36 s, as in v0.8.0. Each agent still keeps its own descriptor: a
  controller that overrides `_construction_config()` should return a new dict
  each call; its result is still deep-copied unless the class sets
  `_construction_config_is_fresh = True`. That declaration is the class's
  responsibility: the first spawn of each class only checks that two calls
  return two dicts, and logs a warning and keeps copying if not. The
  introspection cache holds its callables weakly and skips callables that
  cannot be cached (an unhashable callable object, say).

- The DataMonitor no longer costs more than the step in a large scene. Its
  entity list is rebuilt only when an object is added, removed or renamed
  (`SimObject.name` is now a property that tells the simulation), the list's
  JSON is serialized once per change, and the file is written from its parts
  without joining them into a new string. With the monitor window open,
  `monitor_update_interval=None` now refreshes every 0.25 simulated seconds
  instead of every step (the window reads the file every 0.5 s); without the
  window it is still 1 s, and `0` is still every step. Each refresh builds
  `last_monitor_frame` and writes the file.
  While paused, a selection, follow or pause change and a single step
  refresh at once; the final frame of a run is written, `close()` writes the
  last state, and the new `flush_monitor()` does so on demand. `reset()`
  empties the entity list. The JSON a reader sees is unchanged.
  Median step time with 5000 static boxes and 20 moving agents (headless,
  `target_rtf=0`, timestep 0.1, a 312 KB monitor file): 25.9 ms before with
  the window flag on (a refresh every step), 0.84 ms now with the window
  flag on (default, every 0.25 s), and 0.63 ms with the monitor off. A
  refresh every step (`monitor_update_interval=0`) still writes the whole
  file each step: 0.95 ms per step, about 0.3 ms per write on that scene.

- `ControllerParams` takes either enum field -- `default_direction` and
  `goal_replacement` -- as the enum or its string value, on construction and
  on any later assignment, and stores the enum; an unknown value raises
  `ValueError` there.
  `default_direction` given as a string to the constructor used to stay a
  string (only `from_dict()` converted it). `ControllerParams.as_config()`
  returns the fields as plain data, enums as their string values.

- A goal or path set on a moving agent that it cannot continue into within
  its limits now brakes first by default (`goal_replacement="brake"`): the
  agent stops along its current motion at `max_linear_decel`, then starts the
  goal from rest. The last release started such a goal from rest at once,
  dropping the current speed in one step; set `goal_replacement="immediate"`
  (per agent, per manager's `fleet_controller`, or globally through
  `PBF_CONTROLLER_GOAL_REPLACEMENT`) for that behaviour.

- `MoveAction` ends `COMPLETED` only when the agent comes to rest at the
  path's last waypoint. Stopped elsewhere -- the path was braked
  (`Agent.brake()`) or replaced by a newer one, held goals included -- it ends
  `CANCELLED`, and a path the agent refuses for good (a fixed base, an empty
  path) ends `FAILED`; both used to end `COMPLETED`. The check no longer reads
  `Agent.path`, which a batch-owned agent always reports empty.

- A moving differential-drive agent takes a goal whose line is within
  `_replan.HEADING_TOL` (1e-3 rad) of its heading as straight on: it snaps its
  heading onto the line and continues at its speed, as an omnidirectional
  agent does within its cross-track tolerance. A larger turn still needs a
  turn in place (braked for under `brake`). From rest the agent still turns
  for any heading error above 1e-6 rad.

- An agent's velocity is its state, set from the trajectory it follows. The
  batched controllers (`BatchOmniController`, `BatchDifferentialController`)
  now report each agent's profile velocity at the step's time, as the
  per-agent controllers do, instead of the pose change averaged over the step,
  which lagged while accelerating, led while braking and stayed non-zero on the
  arrival step. `Agent.velocity`, `angular_velocity` and the odometry built
  from them therefore change for batched agents: the twist is the
  instantaneous velocity at the end of the step, so integrating it while the
  agent accelerates or brakes no longer reproduces the pose change exactly
  (the pose itself is unchanged). A goal replaced mid-motion
  continues from that velocity state in every controller (the batched ones
  re-derived it from their trajectory arrays), so a restored velocity is what a
  new goal continues from. The yaw rate reported during a turn in place now
  carries the turn's sense (negative clockwise) in the per-agent controllers
  too; it was the unsigned angular speed.

- `step_once()` advances `sim_time`, the elapsed time and `step_count` right
  after the `pre_step` event, before the object update, instead of after the
  `post_step` event. `sim_time` is now the time of the state you see everywhere
  except a `pre_step` subscriber:
  - Position controllers (`OmniController`, `DifferentialController` and the
    batched omni and differential controllers) evaluate their trajectories at
    the end of the step, as kinematic joints already did; before, a step's pose
    was the trajectory one step earlier. A path set in a `pre_step` subscriber
    moves in that same step (with `auto_approach=False`; the default
    zero-length approach waypoint still takes a step), and a motion completes
    one step sooner.
  - `post_step` subscribers (and the `post_step` payload's `sim_time`) see the
    step they complete: `step_count` and `sim_time` have already advanced. Code
    that added one to `step_count` in a `post_step` handler must drop it.
  - Outside a step `sim_time` is the completed time (it used to lag one step
    behind `step_count`); actions' start and end times, behavior-tree ticks and
    collision observations carry the time of the step that produced them.
  - A step that raises before it completes returns the clock and counter to
    the last completed boundary.
  - `sim_time` is read-only: it reads the simulation's one clock, so it can no
    longer drift from the elapsed time by assignment. Move the clock with
    `restore_completed_step_boundary()` (resuming from a checkpoint); assigning
    `sim_time` raises `AttributeError`.
  - The snapshot-replay evidence documents keep the values recorded before this
    change, with a note.

- A goal or path set on a moving omnidirectional agent (per-agent
  `OmniController` and `BatchOmniController`) now continues the motion: the
  new trajectory starts from the agent's speed toward the new goal instead of
  from rest, and `Agent.set_path` no longer puts a zero-length approach
  waypoint (the current pose) in front of it while the agent moves. The speed
  is signed along the new line, so a goal behind is reached by slowing,
  reversing and driving back. A goal is taken when that trajectory exists
  within the configured limits; otherwise `goal_replacement` decides (see
  above), and under `reject` it is refused -- inside the braking
  distance, where the agent is, or off the line of motion: the current motion
  and path continue unchanged, no limit is ever exceeded, and
  `Agent.set_path` / `set_goal_pose` return `False` (`True` when taken; the
  `AgentManager.set_goal_pose*` wrappers pass it on). `Agent.path_rejection()`
  gives the reason without acting, as a `PathRejection` whose `temporary`
  flag tells a refusal that lasts only while the agent moves from a
  permanent one (fixed base, empty path). At rest every goal is taken, as
  before. Callers inside the library retry a temporarily refused goal on
  their next update
  (`MoveAction`, `PatrolController`, `RandomWalkController`, the behaviour
  tree's `NavigateTo`, whose `NavigationAdapter.start_navigation()` may now
  return `False`), and `FleetCommandDispatcher.navigate()` reports it in the
  ack's `rejected` map -- also when the refusal comes only after the command
  event was emitted (a handler of it changed the robot): the returned ack and
  the recorded input then list only the goals actually applied. A replaced
  goal is judged on the speed of the pose in effect when it is set
  (`sim_time`), per-agent and batched alike, whichever phase sets it.
  This needs
  `two-point-interpolation>=1.3.2`: earlier versions could return a trajectory
  that never reached its target for a body starting away from it, which a
  reversal is. Navigation checkpoints record the trajectory's initial
  speed (`forward.v0`); the kinematic profile's controller state is now
  version 2. A refusal is logged at DEBUG. From a speed above `vmax` (lowered
  during the motion) the new trajectory first slows down to it.

- `RobotGoalCommand2D.z` defaults to `None`: the robot keeps its present
  height (`to_pose(current_z)`, which `FleetCommandDispatcher.navigate()`
  passes). It was `0.0`, which made a planar goal a 3D move for a robot whose
  pose is not at `z = 0` -- one a moving omni agent refuses as off its line of
  motion. An explicit `z` is kept; `ReplayInput.navigate()` and recorded
  `navigate` inputs accept `z: null` the same way.

- URDF SimObject and mesh Agent factories retain their effective spawn
  descriptors for recording handlers, including URDF fixed-base and scale
  settings. The development kinematic profile rejects batch-owned Agents,
  changed Agent pickability and malformed floor settings before restore.

- The development kinematic recording profile accepts a physics-off world
  containing supported primitives without requiring an initial Agent.

- Keep the supported kinematic recording guide as the how-to entry point;
  move earlier re-execution and checkpoint validation guides to the snapshot
  design records. Demo video capture remains a separate how-to.

- Custom controller subclasses without their own `_registry_name` no longer
  replace an inherited built-in registry entry such as `omni`.

- Kinematic joint state moves into a `KinematicJoint` dataclass, one per
  joint, replacing the five dictionaries keyed by joint index that held
  position, last target, motion profile, speed and trajectory between them.
  No behaviour change: the same fields, read and written in the same places.

- `Agent.from_urdf()` and `SimObject.from_urdf()` load through one shared
  helper, `load_urdf_body()`. `Agent.from_urdf()` gains `global_scaling` and
  now raises `FileNotFoundError` for a URDF it cannot read, where it
  previously let PyBullet's own `error` escape -- the two factories had
  drifted on both points.

- `build_tpi()` accepts a `decel` argument instead of pinning `dec_max` to
  `accel`. `TwoPointInterpolation` has always supported an independent
  deceleration; the wrapper was discarding it.

- The batched controllers honour `max_linear_decel` too.
  `trapezoid_distance()` takes the braking scalar as its own argument instead
  of reusing the acceleration one, and `extract_phase_params()` returns it
  alongside -- the phase durations in `tpi.dt` were already the asymmetric
  ones. Batched and per-agent trajectories agree to floating-point noise on
  the same profile. Rotation stays symmetric, angular limits having no
  separate deceleration.

- `Agent.set_path()` marks the agent moving only once the controller has
  accepted the path, and the batched controllers set the flag inside the lock
  that guards their array writes, so a controller that refuses a path does
  not leave an idle agent reporting `is_moving`.

- The validation-only omni checkpoint proof now uses the shared kinematic
  navigation capture/restore methods, including asymmetric deceleration.
  Its artifact version advances to 2. The redundant straight-only controller
  methods have been removed, and per-joint checkpoint conversion now lives
  with `KinematicJoint` while `Agent` applies the rebuilt joints to PyBullet.
- An optional state-recorder error during a command, object spawn/removal, or
  completed-step capture now stops recording with an explicitly incomplete
  artifact while the simulation operation continues. GUI playback selects the
  first available checkpoint at the configured cadence. Checkpoint restore
  also verifies the recorded result and input streams before loading state.
  Accepted inputs issued outside the final step are flushed when recording
  closes, including runs with no completed steps. Closing a recorder detaches
  it from the simulation, so subsequent steps continue without recording.

- Group corridor policy evaluation programs under `examples/evaluation/` and
  checkpoint/state-discovery programs under `examples/validation/`. Their
  Python module paths and documented run commands now use those folders.

- A negative or non-finite `frequency` for `register_callback()`,
  `register_plugin()` or a plugin config now raises `ValueError`; a negative
  one used to mean every step. `None` (and `0` for a callback) still means
  every step.

- `SimulationParams.monitor_update_interval` is read once, when the
  simulation is created; changing `params` afterwards no longer changes the
  DataMonitor refresh rate.

- Time profiling reports `CallbackPhase.POST_STEP` callbacks as
  `callbacks_post_step`, apart from `events_post_step`.

### Added

- Add `register_callback(..., phase=CallbackPhase.POST_STEP)` (`CallbackPhase`
  in `pybullet_fleet.types`; the string `"post_step"` works too): a callback that runs at the
  end of the step, right after the `post_step` event, once poses are written
  and the collision check is done, with the same `frequency` option as Phase-1
  callbacks, scheduled on `sim_time`, the time of the state it sees.

- Add `MultiRobotSimulationCore.set_collision_ignore_filter(fn)`: touching
  pairs for which `fn(obj_a, obj_b)` returns `True` are excluded from
  collision results -- active pairs, `collision_count`, collision colouring,
  `COLLISION_STARTED` and the collision observation -- and skip the narrow
  phase. Filtering in a `COLLISION_STARTED` handler instead left every other
  figure counting the contact. A pair that becomes ignored while active
  resolves on the next check, and one that stops being ignored is checked
  again on the next one even if neither body moved, so the filter may depend
  on run-time state.

- Add `SimulationParams.ignore_attached_collision` (default `true`): contact
  between an object and the object it is attached to is no longer reported
  as a collision. Set it to `false` for the previous behaviour.

- Add `SimulationParams.monitor_log_interval`: the monitor's figures --
  real-time factor, collision totals, step count -- are logged as an INFO
  `[MONITOR]` line every that many wall-clock seconds, whether or not
  `monitor` or `gui` is on. They were otherwise computed only for the
  DataMonitor and logged at DEBUG, so a run kept no record of its real-time
  factor or collisions. Defaults to 1 s; `0` turns it off.

- Add `SimulationParams.monitor_update_interval`: simulated seconds between
  DataMonitor refreshes, which were fixed at every step with a GUI and every
  second without one. The default (`None`) is every step while the monitor
  window is open (`enable_monitor_gui`, or `gui` when that is unset), so a
  monitor window on a headless run stays live too, and every second
  otherwise; `0` is every step. It is independent of `monitor_log_interval`.

- The monitor's real-time factor is measured against the newest sample older
  than its 10 s window as well, so updates sparser than the window no longer
  report `rtf=0.00`.

- Add `SimObject.from_urdf()` and a `urdf_path` field on
  `SimObjectSpawnParams`, so a URDF-defined body with no behaviour can be
  loaded as a plain `SimObject` instead of an `Agent`. `from_sdf()` already
  covered SDF; URDF had no non-agent loader, and `urdf_path` existed only on
  `AgentSpawnParams`. Static scenery created this way stays out of the
  per-step update loop (`_needs_update` is `False`): measured on a scene of
  2376 fixed-base, jointless, controller-less bodies, 15.98 ms per step as
  Agents against 0.78 ms as SimObjects.

- Add `ControllerParams.max_linear_decel` (scalar or per-axis, same semantics
  as `max_linear_accel`) so vehicles that brake harder — or softer — than they
  accelerate can be modelled. Leaving it unset mirrors `max_linear_accel`, so
  existing configurations keep their symmetric trapezoid unchanged.
  `Agent.max_linear_decel`, `ControllerParams.linear_decel_along_direction()`
  and `ControllerParams.scalar_max_linear_decel()` round out the accessors.
- Add `MultiRobotSimulationCore.close()` and context-manager support.
  `run_simulation()` has always released the client on its way out, but a
  program driving `step_once()` itself, or a test that only builds a scene,
  never reached that code and nothing else disconnected. Because PyBullet's
  module-level calls resolve against a default client, a leaked client made
  the *next* simulation in the process report bodies and joints "not found"
  rather than failing where the leak happened. `close()` is safe to call more
  than once and on a simulation that was never run.
- Add `Agent.set_joint_motion_profile()`, and `max_speed` / `max_accel` /
  `max_decel` on `ElevatorParams`. A kinematic joint ran at the URDF's
  `<limit velocity="...">` and reached it instantly, so its speed was a
  property of the model file -- two installations differing only in how fast
  a cabin travels needed two URDFs, and
  `changeDynamics(maxJointVelocity=...)` does not help because
  `getJointInfo()` keeps reporting the original. Giving an acceleration makes
  the travel a trapezoid solved by `TwoPointInterpolation`, the same solver
  the linear controllers use: 0.6 m at 0.4 m/s takes 1.5 s flat and 1.9 s
  with 1.0 m/s^2 ramps. Both are opt-in; with neither, and with a speed but
  no acceleration, the joint keeps the constant-speed behaviour it always
  had. A target a ramped joint cannot stop at -- one inside its braking
  distance, one it is moving away from, or one set to where it already is --
  brakes at the configured rate and re-plans rather than snapping to the
  target. A profile can be changed while the joint moves: it keeps the speed
  it is carrying and re-plans from it under the new limits.
  `Agent.is_joint_moving()` reports whether a ramped joint is still in
  motion, which `Elevator` arrival now waits for instead of taking the
  action's tolerance as arrival.
- Add a PBF-owned, versioned kinematic recording profile for supported URDF
  Agents, all kinematic joints, constructed primitives and attachments. A trusted
  registry and `load_supported_simulation()` reconstruct a complete artifact
  in a fresh process; the first ordinary `run_simulation()` then continues
  from its completed-step checkpoint. A fixed-arm validation program records
  and restores the existing demo's callback behavior through a declared
  application data restorer; the ordinary demo remains unchanged. Unsupported
  live state or missing restore code fails closed; recording errors mark
  artifacts incomplete while simulation continues. The same profile renders
  recorded kinematic entity frames in the GUI without executing simulation
  steps; examples may supply an initial camera configuration. The manipulation
  validation scenario now uses the one-call loader and ordinary
  `run_simulation()` continuation with a registered external-stage restorer.
  The supported profile now records joint motion limits, signed speed and
  in-progress ramp phase for fresh-process continuation of ramped joints.
  Its development artifact schema is `pbf.kinematic.v4`; older artifacts
  lack parts of the current execution or object-construction state and must be
  recorded again.
  Active differential pose navigation and asymmetric omni/differential forward
  deceleration are now restorable through shared TPI checkpoint primitives.
  Physics-off standard `SimObject` box, sphere and cylinder shapes can be
  reconstructed independently for display and collision, including local
  shape offsets. Meshes, URDF objects and custom subclasses remain unsupported.

- Add name-based simulation entity lookup on `MultiRobotSimulationCore`:
  ordered multi-result searches for objects and agents, plus unique-result
  helpers that raise `LookupError` when a name is missing or ambiguous.

- Add opt-in, per-completed-step execution recording for the supported
  physics-off omni manipulation scenario, including its generated approach
  waypoint and final orientation turn, with fresh-process checkpoint
  restore/resume and result playback through the same example CLI. Capture
  every joint position/configured target on the supported robot, kinematic
  attachment relation, box lifecycle and
  scenario progress in a versioned profile; unsupported active state marks
  recording incomplete rather than being silently omitted.
  The common recorder accepts declared profiles and named, versioned data
  callbacks. The example's Omni/box profile lives under `examples/validation/`.

- Add a profile-limited, fresh-process checkpoint/restore proof for one
  physics-off omni robot in straight navigation, with explicit
  `run_simulation(resume=True)` continuation and a three-process example.
  All three proof modes support optional GUI viewing and auto-named result
  paths when `--output` is omitted.
- Add an optional-GUI mobile-manipulator state-discovery example that exercises
  runtime object spawn/delete, base and joint motion, link attachment and
  detachment, and completed-step observations without adding checkpoint APIs.
- Add a one-sided corridor traffic experiment that compares ordinary kinematic
  passage through an entrance-side merge with an external collision-triggered
  stop/restart rule, optional
  single-policy GUI observation, auto-named result directories, and reports
  of all-robot passage time and blocking evidence.
  A stopped conflict-group winner resumes immediately so the external response
  does not stop every moving member of that group.
- Expose the latest collision-check observation, including active pair IDs,
  detection branch, signed closest-point distance and sample time, without a
  second all-pairs query.
- Add a synthetic 20-robot corridor evaluation example with independent entry
  policies, optional single-policy GUI observation, task/collision evidence,
  and simulated-time fleet metrics.
- Add a macOS launcher for viewing Linux GUI examples in a browser via Colima
  and noVNC.
- Opt-in, versioned kinematic navigation replay: record effective Fleet API navigate/stop inputs, reconstruct a fresh supported initial world, and compare acknowledgements, events and full observations. Supports omni/batch_omni, stable artifact entity IDs, integrity validation and explicit changed-configuration comparisons; does not provide checkpoint/resume or ROS/DDS replay.
- `run_simulation()` now warns when real time between consecutive loop
  iterations exceeds 2 seconds, regardless of `target_rtf`. The existing
  wall-vs-monotonic clock-jump diagnostic only logged in the `target_rtf > 0`
  branch, so `target_rtf <= 0` (run-as-fast-as-possible) had no loop-timing
  instrumentation at all — a host overload or a stalled GUI event left
  nothing to go on after the fact.

### Changed

- Align the one-sided traffic example's `run_policy()` and report `policy` name
  with the bidirectional evaluation example. In its version 2 report, name route
  phases `entrance`, `exit` and `destination` instead of using numeric indices.
- Extend the one-sided traffic example's external collision response through
  endpoint arrival, and report all-robot endpoint time separately from corridor
  passage time.
- Shorten both corridor examples to a 1.5 m passage; start the bidirectional
  evaluation GUI closer to the action, auto-name its output directory when
  no path is supplied, and allow both examples to show the live DataMonitor
  with `--gui --monitor`.
- Move the traffic-experiment GUI camera closer and allow arrow-key panning in
  PBF's interactive camera controller.
- Keep replay input application and result capture in `ReplaySession`, without replay-specific mutation guards in ordinary simulation and Fleet APIs. Direct core calls are outside the recorded-input contract.

### Fixed

- `BatchDifferentialController` uses the agent's
  `controller_params.default_direction` for a path set without a direction, as
  the per-agent `DifferentialController` does; it always drove `FORWARD`. A
  goal replacement on a batched differential agent is judged with the new
  path's direction, not the previous path's: a goal set without a direction
  while backing up under `auto` was taken and turned around at once instead of
  being refused (`reject`) or braked to (`brake`).

- A null `shadow_map_world_size` or `shadow_map_resolution` in the lighting
  config now takes the default for the GUI lighting sliders, as a null
  `light_position` already did; it used to fail an internal assertion.
  Internal consistency checks raise `RuntimeError` or `ValueError` with a
  message instead of using `assert`, so they hold under `python -O`.

- Frequency-limited callbacks and plugins, `collision_check_frequency` and the
  DataMonitor refresh no longer skip beats or drift. They compared the
  accumulated (float) simulation time with the interval since their last run,
  so a 10 Hz callback on a 0.1 s timestep ran at gaps of 0.1 s and 0.2 s, a
  1 Hz plugin missed every other second, and an interval that is not a whole
  number of timesteps drifted. They now run on a fixed grid, `k / frequency`
  from the start of the run, at most once per step, with a tolerance for float
  error, and they read the time as the step count times the timestep rather
  than the accumulated simulation time, whose error grew past that tolerance
  within minutes at a 1 ms timestep. A handler registered mid-run first runs
  at the next due time. A DataMonitor with a GUI now refreshes on every step,
  as documented.

- `Agent.set_controller()` now adopts the controller's own
  `ControllerParams` as the agent's `controller_params`, matching what
  `Agent.__init__` does when the controller is passed at construction.
  Previously the two diverged silently: the controller moved the agent by its
  own limits while `controller_params` -- which `Agent.max_linear_vel` and its
  siblings delegate to, and which a batch controller builds its trajectory
  from -- still held the defaults. A fleet that set its limits this way ran
  correctly under the per-agent controllers and at the default 2.0 m/s and
  5.0 m/s under a batch controller.
- An `AgentManager` holding a plain `SimObject` no longer raises
  `AttributeError`. `spawn_from_config()` dispatches on each entry's `type`,
  so a config may mix `agent` and `sim_object` and a mixed manager is a
  supported arrangement, but every method that commands movement, reads
  motion state or queues actions reached for attributes only an `Agent` has
  -- `repr()` among them, by way of `get_moving_count()`. They now read a new
  `AgentManager.agents` property, which filters. `set_goal_pose()` indexes
  that property rather than `objects`, so a non-agent no longer shifts the
  index of every agent after it.
- `AgentSpawnParams.pickable` now reaches the agent. Neither
  `Agent.from_mesh` nor `Agent.from_urdf` took the argument, so
  `forward_spawn_params()` dropped it and `Agent.__init__` overwrote the flag
  with `False` unconditionally -- the field was inert. `attach_object()`
  refuses a non-pickable body, so an agent meant to be carried failed to
  attach, far from where it was configured. The default is unchanged:
  `AgentSpawnParams` declares `pickable = False` explicitly, narrowing
  `SimObjectSpawnParams`' `True`, and `from_dict()` applies that default
  rather than inheriting the parent's resolved `True`. The inherited field
  order is unchanged.
- PyBulletFleet no longer configures the root logger. `logging.basicConfig()`
  on import changed the format of the host application's own output, and
  constructing a simulation called `logging.getLogger().setLevel(...)` from
  `log_level` -- which defaults to `warn`, so merely building a simulation
  turned off every INFO line in the process. Both now apply to the
  `pybullet_fleet` logger, which carries its own handler.

- A headless simulation no longer opens a monitor window. `enable_monitor_gui`
  now defaults to `None`, meaning "follow `gui`"; an explicit `True` or
  `False` is still honoured, so a monitor window without the PyBullet viewer
  remains available. Building several simulations in one process -- a test
  suite, a parameter sweep -- used to put a Tk window on screen per
  simulation.

- Include positive-gap pairs within `collision_margin` in closest-point
  collision checks, including pairs whose AABBs do not overlap.
- Isolate the macOS Tkinter monitor in its own process to avoid native GUI
  conflicts with PyBullet, while preserving monitor controls while paused.
  Keep the existing threaded startup and shutdown behavior on Windows and Linux.

## v0.8.0 (2026-09-01)

### Added

- Add OpenUSD static-world import, including PointInstancer expansion,
  PreviewSurface/Isaac material support, collision controls, and a warehouse
  example.
- Add portable BehaviorTree.CPP XML runners for generic, Agent, and Worker
  trees, with typed action/navigation contracts and a worker demo.
- Add backend-neutral elevator state-machine and motion-adapter contracts,
  including reject, queue, and replace-next request policies.
- Add configurable GUI lighting and native runtime lighting controls.
- Add interactive DataMonitor controls for simulation execution, selection,
  camera following, live timing, and paused pose editing.
- Add manager-scoped Fleet ROS interfaces, transport probes, and scale checks.

### Changed

- Reduce FleetState conversion overhead by populating generated ROS messages
  in place, and add profiler fields for collection, conversion, and publish.

### Fixed

- Serialize batch-controller path updates with simulation advancement to avoid
  concurrent ROS command dispatch corrupting vectorized trajectory state.

### Documentation

- Document the USD/USO portability roadmap, supported behavior-tree profile,
  planned device coordination, and refreshed core and ROS scale results.

## v0.7.4 (2026-08-03)

### Added

- Add `pybullet-fleet config --list`, `--path`, and `--copy` for managing
  bundled YAML configuration templates after a pip installation.
- Add model overrides to the grid demos: `--robot` for the mobile-only grid
  and `--mobile-robot` / `--arm-robot` for the mixed fleet.
- Add recorded fleet-interface and RMF office-demo videos to the ROS 2
  quickstarts.

## v0.7.3 (2026-08-02)

### Documentation

- Document the optional `python3-tk` dependency for the PyBulletFleet
  DataMonitor GUI in the ROS 2 and APT installation paths, and link the RMF
  Docker steps to the official Ubuntu Docker Engine installation guide.

### Fixed

- Avoid repeated `InvalidHandle` and shutdown errors during ROS 2 bridge Ctrl-C
  handling by stopping simulation publish callbacks before rclpy destroys
  publishers and making context shutdown idempotent.

### Performance

- Refresh core and ROS bridge benchmark references for this release; no
  release-blocking regression was observed.

## v0.7.2 (2026-08-02)

### Features

- Add an installed ROS 2 fleet demo launch for 100-robot, batched Fleet API
  navigation, shared by native Jazzy and Docker environments, with an optional
  lightweight RViz view and non-overlapping multi-robot navigation client.
- Use a Franka Panda in the ROS 2 arm trajectory demo without requiring an
  additional model package.

### Fixed

- Default the Open-RMF Office demo to real-time pacing so dispatched tasks
  progress with the simulated clock.
- Resolve RMF map `model://` furniture includes from the Gazebo Fuel cache,
  so prefetched Office and other RMF demo assets are rendered in PyBullet.

### Documentation

- Add native Jazzy bridge, Fleet API, and RMF quickstarts, with Docker moved to
  the alternative environment path, add an RMF scenario catalog and full RMF
  Web Docker workflow including a no-build RMF Web companion stack for native
  demos, and document prefetching Gazebo Fuel assets for the supported RMF demos.

## v0.7.1 (2026-08-01)

### Fixed

- Constrain NumPy and SciPy to compatible releases so a fresh installation
  continues to work with the PyBullet binary extension.

## v0.7.0 (2026-08-01)

### Features

- Add a transport-neutral fleet API with state snapshots, fleet navigation and
  joint, stop, attach, and generic action commands, `FleetStateProvider`, and
  `FleetCommandDispatcher`.
- Add stable generated names for grid-spawned robots and objects, allowing
  fleet commands and integrations to address them deterministically.
- Add `MultiRobotSimulationCore.last_profiling`, a read-only snapshot of the
  most recently measured simulation step for plugins and callbacks.
- Add fleet-level ROS 2 state and command interfaces, including 2D/3D
  navigation, joint control, stop, attach, and generic action requests.
- Add RMF fleet client modes for per-robot ROS endpoints, fleet-level ROS
  endpoints, and direct in-process Python dispatch. The direct mode dispatches
  simulator changes on the simulation step thread.
- Add configurable, Gazebo-faithful RMF workcell results and expanded support
  for RMF doors and lifts.

### Changed

- Make per-robot bridge interface groups independently configurable. Disabling
  all groups avoids creating per-robot handlers; disabling one group suppresses
  only its associated publishers, subscribers, services, or actions.
- Configure vectorized batch robot controllers through a named manager's
  `fleet_controller` mapping; entity-level `batch_controller` and
  `fleet_controller` settings are no longer accepted.
- Keep objects attached to a robot link synchronized with the current pose in
  the same simulation step.

### Breaking Changes

- ROS 2 bridge internals: `RobotHandler` is now a facade over dedicated
  interface groups. Private callback overrides such as `_cmd_vel_cb`,
  `_navigate_execute`, `_execute_action_execute`, and service callbacks are not
  supported extension points. Custom integrations should use the public ROS
  topics/actions/services or compose/replace the relevant interface group.

### Documentation

- Document the ROS 2 Jazzy APT release workflow, including a reproducible
  GitHub-hosted APT supplement repository and the required RMF demo overlay.
- Add native ROS 2, fleet API, bridge scale, and RMF dispatch guides and
  runnable Docker validation commands.

### Testing

- Add headless RMF integration coverage and end-to-end patrol and delivery
  dispatch checks, including door, lift, and workcell flows.

## v0.6.0 (2026-06-26)

### Bug Fixes

- **Fixed intermittent multi-second freezes during `run_simulation()`.** The
  real-time loop paced off the wall clock, so a wall-clock jump (NTP correction,
  host suspend/resume, WSL2 drift) could desync it into a long `time.sleep()` —
  the sim "froze" then "suddenly resumed". It now paces on the monotonic clock,
  converts the catch-up sleep correctly for `target_rtf != 1` (it previously
  over-slept ~`rtf`×), and clamps any residual sleep so it can never freeze.

### Features

- Added `SimulationParams.max_sleep_frames` (default 4.0) to bound the real-time
  sleep, mirroring the existing `max_steps_per_frame` catch-up cap.

## v0.5.0 (2026-06-25)

Try the examples straight from a `pip install` — no repo clone needed.

### Features

- **Examples ship in the wheel + a `pybullet-fleet` CLI.** After
  `pip install pybullet-fleet`, list / locate / copy / run the bundled demos:
  ```
  pybullet-fleet examples --list
  pybullet-fleet examples --path
  pybullet-fleet examples --run path_following_demo.py [--robot racecar]
  pybullet-fleet examples --copy ./my-examples
  ```
  `--run` forwards extra flags to the demo; `--copy` extracts them for editing.
- Expose `pybullet_fleet.__version__`.

### Internal

- Bridge: fixed the previously-failing `pybullet_fleet_ros` tests and now run
  `colcon test` in CI (the bridge unit tests were never gated before).
- Added a clean-install CI check that the published wheel actually ships the
  bundled examples, and per-demo compile tests so a broken example is caught.

## v0.4.1 (2026-06-24)

Packaging patch: the v0.4.0 wheel shipped **without** its bundled data, so
pip-installed users could not load the built-in robots, configs, or meshes.
This release makes that data ship in the wheel and resolve correctly.

### Bug Fixes

- Bundle the built-in `robots/`, `config/`, and `mesh/` data in the wheel —
  pip-installed users can now load the bundled URDFs, configs, and meshes that
  were missing from the v0.4.0 wheel.
- Resolve bundled assets from a relative path (e.g. `"config/config.yaml"`,
  `"robots/arm_robot.urdf"`) whether you run from the repo or a pip install.

### Internal

- Add a clean-install packaging smoke test (`make test-clean-install`, a
  standalone script, and a Docker image) plus a release-time wheel-verify gate,
  so missing bundled data is caught before publishing.
- Add CI coverage: path-filtered packaging (clean-venv install) and ROS 2
  bridge (Docker integration) workflows.
- Pin `numpy<2` in the bridge image to match the ROS 2 Jazzy runtime.

## v0.4.0 (2026-06-22)

First release with the **ROS 2 bridge and Open-RMF integration**, an event
system, SDF world loading, vectorized batch controllers, and a plugin/device
architecture.

### Highlights

- **ROS 2 bridge** (`ros2_bridge/`, ROS 2 Jazzy) — drive PyBulletFleet from ROS 2:
  per-robot odom / TF / cmd_vel / goal_pose / joint topics; NavigateToPose,
  FollowPath and FollowJointTrajectory actions; and `simulation_interfaces`
  services (spawn / delete / get entities, step simulation, simulator features).
  Dockerized with an integration smoke test.
- **Open-RMF integration** — fleet adapter plus handlers for doors, lifts,
  delivery, cleaning, and battery, with runnable demos: office, hotel, clinic,
  airport_terminal, battle_royale, and campus.
- **EventBus** — global and per-entity lifecycle events (object/agent spawn &
  remove, pre/post step & update, action start/complete, collision enter/exit)
  for plugins and external integrations.
- **SDF loader** — load multi-model SDF worlds, with `<submesh>` extraction from
  multi-part meshes, `model_yaw_offset`, and `force_color` flat-shading for
  meshes PyBullet cannot texture.
- **Batch controllers** — vectorized omnidirectional and differential controllers
  (two-phase step) for efficiently simulating large fleets.
- **Plugin & device architecture** — per-agent plugins (e.g. battery) and
  simulation plugins (e.g. workcell delivery); `Door` and `Elevator` agent
  subclasses with joint control and automatic passenger attachment.

### Features

- Unified `controller=` API: registry name, dict, list, `ControllerParams`, or a
  prebuilt controller; `type:` (registry) / `class:` (dotted import path)
  selection; patrol and random-walk high-level controllers; per-axis kinematic
  limits and `navigation_2d`.
- `resolve_model()` resolves Tier 1/2/3 model names (bundled, ROS description
  packages, `robot_descriptions`) to URDF paths.
- `CameraController` — interactive right-drag pan, zoom, and top-down view.
- `pybullet_fleet_msgs` package (ExecuteAction action).

### Bug Fixes

_(Fixes to behaviour from v0.3.0; issues introduced and resolved within this
release's new features are omitted.)_

- Collision detection was silently disabled when the simulation was loaded from
  YAML — the `collision_detection_method` string is now coerced to its enum.
- Per-axis velocity/acceleration limits with a zero-capped axis (e.g.
  `max_linear_vel: [0.3, 3.0, 0.0]`) no longer produce a NaN limit.

### Documentation

- ROS 2 bridge / Docker guide; corrected headless, env-var, and config-file docs;
  2026-06-20 implementation audit; a single shared agent-instructions source for
  GitHub Copilot and Claude Code.

### Performance

- Kinematics mode (headless, AMD Ryzen AI 7 PRO 350): ~64× real-time at 100
  agents, ~10× at 500, ~4.4× at 1000. See `docs/benchmarking/results.md`.

## v0.3.0 (2026-04-08)

### Breaking Changes

- `DifferentialPhase` removed — motion phase logic is now owned by `DifferentialController` / `OmnidirectionalController` in `controller.py`

### Highlights

- **SimulationRecorder** — Headless and GUI video capture (GIF/MP4) via `start_recording()` API or `RECORD` environment variable
- **Centralized defaults** — All parameter defaults in `_defaults.py` with `PBF_*` env-var overrides and `.env` file support
- **Visual documentation** — Embedded demo videos across all doc pages, YAML-driven batch capture scripts
- **Robot model resolution** — `resolve_model("panda")` finds URDFs/SDFs by name across local, pybullet_data, and robot_descriptions sources (`resolve_urdf` kept as deprecated alias)
- **Controller refactor** — `Controller` extracted from `Agent` into `controller.py`, reducing `agent.py` by ~500 lines
- **Entity registry** — `register_entity_class()` for custom spawn types via YAML config

### Features

- `SimulationRecorder` — camera modes (auto, gui, orbit, manual), time bases (sim, real), GIF/MP4 output
- `start_recording()` / `stop_recording()` on `MultiRobotSimulationCore`; `RECORD` env-var enables auto-recording in `run_simulation()`
- `_defaults.py` — single source of truth for all defaults; `PBF_{SECTION}_{KEY}` env-var overrides; `.env` auto-loading via python-dotenv
- `robot_models.py` — `resolve_model()`, `auto_detect_profile()`, `register_model()`, `discover_models()`, `add_search_path()` (`resolve_urdf()` kept as deprecated alias)
- `controller.py` — `OmnidirectionalController`, `DifferentialController` extracted from Agent
- `entity_registry.py` — `register_entity_class()` for extensible YAML-driven spawning
- `compute_scene_bounds()` — scene AABB for auto-framing cameras
- `unregister_callback()` — remove callbacks by function identity
- `SimulationParams.enable_floor` — option to skip ground plane loading
- `[models]` extra in `pyproject.toml` — `pip install pybullet-fleet[models]` pulls `robot_descriptions`
- New demo scripts: `capture_demo.py` (headless), `capture_screen_demo.py` (GUI), `capture_model_catalog.py`
- Examples reorganized into `arm/`, `basics/`, `mobile/`, `scale/`, `models/` subdirectories

### Documentation

- Embedded demo videos on all example pages; Quickstart page with single demo video
- How-To: Capturing Demo Videos (Python API, `RECORD` env var, batch scripts)
- Configuration Reference — centralized defaults, override priority, `.env` usage
- Tutorial 6: Robot Models — resolution, auto-detect, search paths, auto-discovery
- Roadmap page with performance profiling plan

### Testing

- 960 tests (up from 740), coverage 82%
- New modules: `test_recorder.py`, `test_defaults.py`, `test_robot_models.py`, `test_controller.py`, `test_entity_registry.py`

### Performance

- 1000-agent RTF improved **+38%** (2.4× → 3.3×, 40.9 ms → 30.0 ms/step)
- 500-agent RTF improved **+12%** (6.8× → 7.6×, 14.7 ms → 13.2 ms/step)

## v0.2.0 (2026-03-20)

### Highlights

- **IK end-effector control** — `Agent.move_end_effector()` → `PoseAction` → `IKParams` for Cartesian arm control
- **Prismatic (linear) joints** — Rail arm support with mixed revolute+prismatic chains
- **Mobile manipulator** — Arm-on-mobile-base with IK auto-locking wheels and kinematic EE tracking
- **AI-native DX** — Repository-level AI instructions and `Makefile` entry points

### Features

- `PoseAction` — move end-effector to Cartesian target; uses `Agent.move_end_effector()` with IK solved via `IKParams`
- `IKParams` dataclass for IK solver configuration (attached to `AgentSpawnParams`)
- `PickAction` / `DropAction` EE extensions via `ee_target_position` parameter
- `DropAction.drop_relative_pose` — offset from current object pose instead of absolute world position
- `JointAction` — per-joint tolerance (scalar, list, or dict) and prismatic joint support
- `Agent.joint_tolerance` — agent-level default with fallback chain (Action → Agent → 0.01)
- Prismatic joint kinematic fallback (0.5 m/s) alongside revolute (2.0 rad/s)
- Kinematic arm joint position cache for velocity interpolation
- `.github/copilot-instructions.md` — AI-readable project context (architecture, guard rails, patterns)
- Root `Makefile` with 11 targets (`make verify`, `make test`, `make bench-smoke`, etc.)

### New URDFs & Demos

- `robots/rail_arm_robot.urdf` — 1 prismatic + 4 revolute joints (5-DOF)
- New demo scripts: `pick_drop_arm_ee_demo`, `pick_drop_arm_ee_action_demo`, `rail_arm_demo`

### Bug Fixes

- Fix link-level object attachment (`computeForwardKinematics` for correct link poses)
- Fix packaging configuration for pip install compatibility
- Standardize collision method names to lowercase
- Resolve CI and type-checking issues

### Documentation

- Tutorials: EE/IK control, prismatic joints, mobile manipulator
- Design specs for all new features
- Updated architecture overview with IK and action system extensions

### Testing

- 740 tests (up from ~560), coverage 80%
- Mobile manipulator E2E tests with shared assertion helpers
- Parametrized arm tests across physics/kinematic/physics_off modes

### Performance

- No regressions: 1000 agents at 3.2× real-time, 31 ms/step

## v0.1.0 (2026-03-14)

Initial public release of PyBulletFleet — a kinematics-first simulation framework for large-scale multi-robot fleets.

### Highlights

- **Kinematics-first simulation** — Teleport-based stepping enables N× real-time execution for fleet-scale evaluation
- **Scalable to 1000+ agents** — Spatial-hash collision detection and shared-shape caching keep step times low
- **Action system** — Built-in MoveTo, Pick, Drop, and Wait actions with queue-based execution
- **Flexible physics** — Full PyBullet physics available per-scenario when needed (grasping, conveyor dynamics)

### Features

- Multi-robot simulation core with configurable physics/kinematics modes
- Agent and AgentManager with grid spawning, YAML-driven configuration
- Spatial-hash collision detection with configurable cell sizes and modes
- Differential drive motion with smooth Slerp rotation
- Action queue system (MoveTo, Pick, Drop, Wait) with status tracking
- SimObject base class for robots, structures, and custom objects
- Callback-driven simulation loop with pause/resume support
- Time and memory profiling built in
- URDF and mesh object support with shared-shape caching

### CI & Release Infrastructure

- GitHub Actions CI: lint, test matrix (Python 3.10/3.11/3.12), docs build, security audit, license check
- Automated PyPI release via tag push (Trusted Publisher / OIDC)
- Pre-release and publish scripts with semver validation
- Releasing Copilot skill for guided release orchestration

### Documentation

- Complete ReadTheDocs documentation: tutorials, API reference, architecture guide, how-to guides
- Zero Sphinx warnings — builds cleanly with `-W` flag
- Benchmark results and profiling guides

### Performance

- Quaternion Slerp replaces scipy dependency (7× rotation speedup)
- 1000 agents at 3.2× real-time, 31 ms/step (Intel i7-1185G7)
