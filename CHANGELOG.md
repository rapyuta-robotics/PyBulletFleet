# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking

- `Pose` fields may be NumPy array views. What `get_pose()` returns for a
  kinematic object in a simulation is the object's live row of the pose
  columns: its `position` and `orientation` follow the object, and they are
  arrays, not lists. `pose.position == [x, y, z]` raises (compare
  `list(pose.position)` or the `Pose` objects, which compare by value),
  `pos + [0, 0, 1]` adds element by element, and `json.dumps` needs
  `list(...)`. Hold the `Pose`, not its arrays, across steps: an array taken
  from it stops following the object once the object's row moves (after
  another object's removal, or the columns growing) and it no longer follows
  it after `remove_object()` or `reset()`. Copy a value to keep it. A
  physics object's `get_pose()` still returns a snapshot.
- `mirror: lazy` is the default: a kinematic body's pose reaches PyBullet
  only when something reads it there. Code that calls PyBullet with a body
  id kept from earlier reads the pose of the last sync; set `mirror: always`
  or give the plugin or callback `needs_mirror = True`. With `physics` on,
  every pose is written at every step whatever `mirror` says.

### Changed

- A rider that leaves its carrier counts as moved, and so does every
  object that leaves with it: after `a.detach_object(b)` the riders of `b`
  are checked against `a` on the next collision check too. A removed
  object leaves its attachment tree: its riders are detached, in place,
  each at the top of its own tree (two former riders are a pair for the
  broad phase again), and its carrier no longer lists it; they used to
  stay attached to a body that was gone. A change of
  `ignore_attached_collision` at run time checks every pair again on the
  next collision check, as a change of `ignore_static_collision` does.
- With `physics` on, every body's AABB is read from PyBullet after the
  step, as before. The local box a kinematic body's box is computed from
  is read by putting the body at identity orientation and back, and the
  orientation it comes back with is the one PyBullet reads back, about an
  ulp from the one it was given; a contact with a dynamic body would
  amplify that. The computed boxes serve kinematic simulations.
- Agent plugins can be updated as a batch. `AgentPlugin.create_batch(sim_core)`
  may return a `PluginBatch`; the simulation then adds every plugin of that
  class to it as the agents join and calls the batch's `update(dt)` once
  per step in place of each plugin's `on_update()`. `BatteryPlugin` does:
  its state (rates, state of charge, charging flag) lives in the batch's
  columns, its properties read and write its row, and the batch drains or
  charges every battery in one array operation, reading whether each agent
  moves from the simulation's column. A `BatteryPlugin` subclass stays
  plugin by plugin whatever it overrides, as does a plain battery on an
  agent whose class answers `is_moving` itself (`PluginBatch.add()` may
  return `False` to leave a plugin to its agent), and a plugin on an agent
  outside a simulation. A batched plugin is updated by the simulation's
  step, not by `Agent.update()`. An agent whose plugins are all in batches
  takes the batch-driven fast path in `update()`.
- A `MoveAction` on a batch-driven agent no longer polls its request every
  step: once the batch controller follows (or holds) the request, the agent
  sleeps, and the controller wakes it when the request's fate changes (the
  row ends, a path from outside replaces or drops it, the agent is stopped
  or leaves the controller). An agent whose joints have all settled, and
  one whose action queue has an interruption to end, are treated the same
  way as before. 1000 batch robots driven through `MoveAction` (the
  behaviour-tree and ROS navigate path) step in about the same time as
  through `set_path` (1.3 ms against 2.7 ms before).
- With `ignore_attached_collision` (the default) the broad phase no longer
  offers a pair of two bodies of one attachment tree (a carrier and its
  load, two loads on one carrier): every row carries the id of its tree's
  root and pairs with the same root are dropped as arrays, instead of
  reaching the narrow phase and being set aside there one by one each
  step. A rider that leaves its carrier counts as moved, so a pair the two
  make is checked on the next collision check even if neither moves, as
  before. The riders of the carriers a buffered `set_poses()` moved are
  found through a `parent_row` column, level by level, with no Python loop
  over the carriers, and a batch-driven agent whose only attachments ride
  on its base takes the fast path (they ride along in the batch write).
- The step's per-object update loop visits only the objects that have a
  per-step `update()` (agents and devices), kept in a list as objects are
  added and removed, instead of testing every body in the scene; a scene
  of tens of thousands of static tiles no longer pays for them each step.
  `_needs_update` is a property: a subclass may set it at any time, also
  after `super().__init__()`, and the loop follows the change at once; a
  removed object leaves the loop whatever its flag says.
  A batch-driven agent with nothing of its own to run this step (no action,
  no joint on its way, nothing attached, no plugin, no entity event
  handlers) is skipped: its motion is the batch controller's. An agent's
  reported velocity and yaw rate are columns of the simulation (its
  `velocity` / `angular_velocity` read its row), which the batch controller
  writes as arrays instead of agent by agent, and the Phase 3 AABB flush
  and `set_poses()` resolve their rows as arrays too.
- The PyBullet mirror is lazy (`mirror: lazy`, the default). A kinematic
  body's buffered pose is no longer written to PyBullet at every step: it
  stays in the simulation's pose columns until something reads it in
  PyBullet, and the simulation syncs by itself before anything of its own
  does (the narrow phase of a collision pair, an AABB query, a recording,
  the accessors `SimObject.body_id` and `sim.client`). Physics bodies,
  articulated bodies and link-attachment parents are written at every
  step, and with `physics` on every body is (PyBullet integrates against the
  kinematic bodies and computes the contacts of `CONTACT_POINTS` and
  `HYBRID` in `stepSimulation()`). `sim.sync_mirror()` writes every pending
  pose on demand. Code that calls PyBullet with a body id kept from earlier
  should run with `mirror: always` (in `SimulationParams` or YAML), or a
  plugin or callback can set `needs_mirror = True` to ask for the same: the
  poses earlier steps left pending are written when it is registered, and
  `unregister_callback()` releases the request.
- Poses are columns of the simulation. Every registered object's position
  and orientation live in one array each (in the collision store), an
  object's pose cache (what `get_pose()` returns for a kinematic object) is
  a view into its row, and `sim.get_poses()` is a gather from the columns.
  A buffered `sim.set_poses()` (inside a step) judges which rows moved and
  writes them in one array operation instead of one `set_pose` per object,
  and carries the objects attached to the moved ones' bases the same way,
  level by level (their attachment offsets are columns too; the composed
  pose is computed in double precision, where `p.multiplyTransforms` works
  in single, so the two agree to about 1e-7). A physics body, an object
  with link attachments, a carrier with a rider that is not a kinematic
  base attachment, and a STATIC object take the per-object path there (the
  last does not move, with one warning for the batch). The AABBs of the moved kinematic bodies are then computed from
  the columns directly. A physics body's row is read back after every
  `stepSimulation()`. `Pose` compares by value whether its fields are lists
  or arrays; compare or serialize a field with `list(pose.position)`, as
  the package does (`pose.position == [x, y, z]` on an array view raises).
- The collision broad phase works on arrays. Every object's AABB, local
  box, collision mode and physics flag live in one column store
  (`pybullet_fleet/collision_store.py`) and the spatial hash is two sorted
  arrays of cell keys over it, one for the static bodies and one for the
  rest: the boxes of all the kinematic single-body
  objects that moved in a step are computed from their poses in one array
  operation (no `getAABB()` call per object; physics and articulated bodies
  are still read from PyBullet), and the candidate pairs of all the moved
  bodies come out of one `searchsorted` and one array AABB filter. The
  registration rule, the per-axis clearance, the NORMAL_2D z rule, the
  zero clearance of a HYBRID pair with a physics body, `ignore_static` and
  DISABLED are as before, and the collision parity traces record the same
  events and distances. `sim.cached_aabb(object_id)` and
  `sim.grid_cells_of(object_id)` replace reading the former
  `_cached_aabbs_dict` and `_cached_object_to_cell` dicts, which are gone
  along with `_cached_spatial_grid`.

### Added

- Collision parity traces (`tests/collision_parity/`): eleven small scripted
  scenes (lanes crossing within the margin, a per-axis clearance and margin
  changes, collision mode changes, ignored static bodies, attachment, the
  ignore filter, a collision check every other step, `CONTACT_POINTS` and
  `HYBRID` with physics bodies, agents on paths) are recorded step by step
  (`COLLISION_STARTED` / `COLLISION_ENDED` events, active pairs and their
  `signed_distance`) and compared with the traces recorded on the reference
  revision, kept under `tests/collision_parity/golden/`.
  `tests/test_collision_parity.py` fails on a difference that is not written
  down in its `DOCUMENTED_DIFFERENCES`, and on a written-down difference
  that no longer occurs. `python -m tests.collision_parity.record` records
  the goldens again (`--check` compares without writing).
- Steady-state fleet benchmark: `benchmark/steady_bench.py` and
  `run_benchmark.py --type steady` measure a fixed window of steps in which
  every robot keeps moving, with the per-phase step time, the collision event
  counts and the RTF from the median step. Five scenes: `sparse` (2 m grid),
  `dense` (two groups driving head-on through each other, every tenth robot
  stopped), `carrying` (a box attached to every robot), `warehouse` (robots in
  the aisles of tens of thousands of static boxes) and `idle`.
  `make bench-steady` runs the reference set. `batch_perf.py`'s scene helpers
  are public (`make_sim`, `make_fleet`, ...) so the workers share them.

## v0.9.0 (2026-11-08)

### Added

- Add `PathRequest`: a path and how to follow it (final orientation, direction,
  approach) as one object. `Controller.request_path()` takes it, and
  `PathRequestState` tells whether it is pending, active, finished or gone.
- Add `ControllerParams.goal_replacement` (`GoalReplacement`) for a goal set on
  a moving agent that it cannot reach from its current motion: `brake` first
  (default), `reject` it, `wait_complete` (finish the current path first) or
  `immediate`. Add `Agent.brake()`, and `ControllerParams.max_linear_decel`,
  `max_angular_decel` and `stop_on_collision`.
- Add step hooks: the `STEP` event, `events.on(..., frequency=...)`,
  `CallbackPhase` (`PRE_STEP`, `STEP`, `POST_STEP`) for `register_callback()`,
  and `SimPlugin.step_phase` for plugins.
- Add collision controls: per-axis `collision_margin`,
  `set_collision_ignore_filter()`, `ignore_attached_collision` (default on),
  `query_aabb()`, `notify_geometry_changed()` and access to the latest
  collision-check result.
- Add joint control by name or index for all joint methods,
  `Agent.set_joint_position()` / `set_joints_positions()` to place joints at
  once, and `set_joint_motion_profile()` for kinematic joint speed and
  acceleration limits.
- Add elevator options: `request_floor(..., instant=True)`, the `REDIRECT`
  request policy, `boarding_height`, `report_platform_contacts`, and speed and
  acceleration limits.
- Add `SimObject.from_urdf()`, name-based entity lookup,
  `MultiRobotSimulationCore.close()` and context-manager support, and the
  monitor settings `monitor_update_interval` and `monitor_log_interval`.
- Add kinematic state recording and fresh-process restore for a declared
  profile: supported URDF agents, kinematic joints (including ramped motion),
  primitive objects and attachments. `load_supported_simulation()` rebuilds a
  recorded world in a new process and `run_simulation()` continues from the
  last completed step; recordings can also be played back in the GUI without
  stepping. Unsupported state fails closed, and a recording error marks the
  artifact incomplete without stopping the simulation. The artifact schema is
  `pbf.kinematic.v4`.
- Add kinematic navigation replay: record the Fleet API navigate and stop
  inputs, rebuild the initial world, replay them and compare acknowledgements,
  events and observations (`ReplaySession`; omni and `batch_omni`).
- Recording, restore and replay are development and validation profiles. They
  are not a general save and load for any simulation, and are not recommended
  for production use yet.
- Add examples: link collision for fixed and mobile arms, corridor traffic
  evaluation (20 robots, entry policies, collision-triggered stop and restart),
  mobile-manipulator state discovery, checkpoint and restore validation, and a
  macOS launcher that shows Linux GUI examples in a browser.
- `run_simulation()` warns when more than 2 s of real time passes between loop
  iterations, at any `target_rtf`.

### Changed

- A goal or path set on a moving agent now continues the current motion when it
  can and brakes first when it cannot (`goal_replacement="brake"`), instead of
  restarting from rest.
- `MoveAction` ends `COMPLETED` only when the agent comes to rest at the last
  waypoint, and `CANCELLED` as soon as another request replaces its path.
- `step_once()` advances `sim_time` before the object update, so `sim_time` is
  the time of the new state. Frequency-limited callbacks, plugins, collision
  checks and monitor updates run on a fixed grid of simulated time.
- Callbacks, plugins and event handlers run through one mechanism on
  `sim.events`, with the same order, arguments and error handling as before.
- The collision broadphase covers every link of an articulated body, and
  `ignore_attached_collision` covers every object on one attachment tree.
- PyBulletFleet no longer configures the root logger on import, and a headless
  simulation no longer opens a monitor window (`enable_monitor_gui` follows
  `gui` by default).
- `RobotGoalCommand2D.z` defaults to `None`: the robot keeps its height.
- The ROS 2 bridge Docker images build from a pinned Open-RMF demos image.

### Deprecated

- Overriding `Controller.set_path()` in a subclass: override `request_path()`.
  Removal in 0.10.0 (#142).
- The `joint_index=` keyword and the single-joint `*_by_name` aliases: pass the
  joint name or index to the main methods. Removal in 0.10.0 (#146).
- `CollisionMode.NORMAL_2D` (#104).
- `SimObject.create_shared_shapes()` without `physics_client_id`: it falls back
  to PyBullet's default client, which is wrong when more than one simulation
  runs in a process. Pass `physics_client_id=sim_core.client`.

### Fixed

- Batched controllers: braking (`max_linear_decel`), per-axis angular limits,
  `default_direction`, and goals replaced above the speed limit.
- Joint and link names are looked up on the right body and physics client;
  `Agent.from_mesh()` uses the simulation's client.
- Closest-point collision checks include pairs within `collision_margin` whose
  AABBs do not overlap; `NORMAL_2D` pairs resolve once they no longer overlap
  in z.
- `AgentSpawnParams.pickable` reaches the agent; an `AgentManager` can hold a
  plain `SimObject`; `set_controller()` adopts the controller's parameters.
- `import pybullet_fleet` works without the Tk library, and the monitor no
  longer fails on a host with no display. On macOS the monitor runs in its own
  process.
- `REPLACE_NEXT` elevators drop the pending floor when the new request is the
  floor the cabin is already heading to.

### Performance

- Against v0.8.0 on the same machine and Python, the core benchmark step time
  is 24-42% shorter from 100 to 2000 agents (1000 agents: 9.7 ms to 5.6 ms,
  real-time factor 10.3x to 17.8x), and spawning is 20-30% faster. Memory per
  agent is about 20% higher.
- Giving every mobile agent a path takes about twice as long (1000 agents:
  about 25 ms to about 40 ms), mostly from tracking each request and the
  garbage collection it causes; the step time of the control path is not
  slower (#150).
- The DataMonitor no longer costs more than the step in a large scene.
- The ROS 2 bridge reaches 13.45x real time with 1000 robots on the fleet
  interface (10.98x before).

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
