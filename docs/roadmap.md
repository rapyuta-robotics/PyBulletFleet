# Roadmap

Planned additions and improvements for PyBulletFleet.
Items are grouped by category.  A roadmap item states the intended outcome and
its boundary; implementation detail belongs in the linked architecture or
design documentation.


## Assets

New robot and infrastructure models:

- **Physics Mobile Robot** — Wheeled robot driven by PyBullet physics (motor torques, friction, contact forces)
- **Physics Mobile Manipulator** — Physics-mode mobile manipulator with motor-driven base and arm
- **Conveyor / Mobile Rack** — Add warehouse infrastructure entities for
  material-handling scenarios. Define a portable conveyor state machine and
  backend motion/sensor adapter, including start/stop, jam, item-arrival, and
  recovery states. Develop its device-request BT leaves (`StartConveyor`,
  `StopConveyor`, and `WaitForItemAtSensor`) with the same portable action
  registry as Pick/Drop and lift requests; retain backend physics and sensor
  mechanics in adapters. Door and Elevator devices, and the WorkcellPlugin
  used for RMF dispenser/ingestor flows, are available today.
- **Portable device execution contracts** — Keep infrastructure state machines
  and request/event schemas independent of the physics backend; add adapter
  conformance tests and backend implementations for doors, conveyors, and
  elevators as additional simulators are supported.  The current PyBullet
  elevator is the first extraction; shared packaging and USO node ownership
  remain future work. Validate the portable elevator request contract and its
  policy/configuration conformance across backends, including `REJECT`, FIFO
  `QUEUE`, and `REPLACE_NEXT`; treat an in-motion physical redirect as a
  separate, safety-reviewed capability.
- **RMF-compatible elevator doors** — Extend the simplified cabin-motion
  elevator into a coordinated lift state machine with cabin-door and
  destination-floor shaft-door pairs. Model closed/opening/open/closing and
  cabin motion separately; close and verify the relevant doors before moving,
  then open the paired doors on arrival. Add backend door adapters, URDF/SDF
  door-pair configuration, passenger safety/occupancy checks, and RMF
  `LiftState` / `LiftRequest` door-state, mode, and session semantics. Keep
  standalone building `Door` devices separate from lift-owned doors.
- **Portable mobile-manipulation actions** — Define backend-neutral `Pick` and
  `Drop` Agent contracts, XML ports, capability reporting, and conformance
  tests before adding them to `AgentBehaviorTree`. Forklift and picker trees
  will extend that shared profile with their equipment-specific actions.
- **Portable behavior-tree core** — Move the dependency-free XML runner,
  control-node semantics, blackboard, and documented portable leaf contracts
  into a shared library after a second simulator validates the adapter
  boundary. Keep PyBulletFleet's Agent/worker action adapters in this package;
  preserve `pybullet_fleet.behavior_tree` as a compatibility import path.
- **Portable composite action trees (USO)** — Express reusable warehouse-task
  choreography as portable trees: for example `Pick` as navigate-to-pregrasp,
  reach, grasp, and verify-holding; `Drop` as navigate, release, and verify;
  and `Charge` as find, dock, start, and wait. Define backend-neutral leaf
  Action contracts and lifecycle results; PyBullet, MuJoCo, and other
  simulators then implement only physical leaf adapters such as navigation,
  IK, grasp/release, contact verification, and charging observation. Retain
  existing PyBulletFleet `PickAction`/`DropAction` and related public APIs as
  compatibility facades while progressively implementing their orchestration
  through the portable trees. Keep simple physical movement as a leaf Action;
  represent recovery, replanning, congestion, and device use as tree policy.

## Simulation Capabilities

- **Articulated-body collision candidates** — Correct the collision broadphase
  so contacts involving child links outside an articulated URDF's base-link
  AABB are not missed by PBF collision observations, events, or counts. Cover
  joint movement, both candidate-extraction paths, and collision margins
  without regressing the jointless fast path. Track the correctness fix in
  [issue #68](https://github.com/yuokamoto/PyBulletFleet/issues/68).

- **Snapshot, Event log & Replay** —
  **Development status:** The replay, recording, checkpoint and playback work
  below consists of restricted validation profiles. It is not ready for
  general use with ordinary simulations or production artifacts. The intended
  save/restart/playback user workflow remains under development.

  Restricted kinematic navigation re-execution from a recorded initial state
  is implemented: a session-owned
  fresh instance, effective Fleet API navigate/stop inputs at a defined
  step/order, full observations, supported outcomes, and comparison. The
  versioned artifact is PBF-owned;
  it is evidence for evolving USO concepts, not a canonical/USO-compatible
  schema or a shared runtime dependency. See the [initial-state navigation re-execution guide](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/navigation-reexecution-guide.md)
  and the [design record](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/spec.md).

  Before scoping new work in this area, read the product goals in
  `docs/design/snapshot-replay/product-goals.md` and state which user workflow
  the change advances and which state remains unsupported.

  The implemented artifact separates initial state, input journal, observations
  and completeness/provenance. Full observations are complete only for the
  declared observation profile; they are not execution checkpoints. A changed
  code environment or explicit PBF configuration override produces a variant
  from the same initial state, not a continuation from an intermediate state.

  The final user goals are:

  1. Play recorded simulation data like `rosbag play`, without recomputing it.
  2. Restore a snapshot from an arbitrary recorded point and resume normal
     simulation.
  3. After restore, change an algorithm and reproduce a failure or compare
     variants in an A/B test. A controlled comparison also needs the relevant
     later external inputs and conditions.

  The intended user entry points are opt-in recording during ordinary
  simulation, a time-selectable restart followed by normal execution, RTF-
  controlled playback of recorded results, and a time-aligned trace of events
  and effective parameters. Names such as `save=on`, `pbf-restart --time` and
  `pbf-playback --rtf` are provisional. Time selection should begin from the
  nearest usable checkpoint **not later than** the requested time; exact
  advance-to-time semantics remain to be designed. These are overall goals;
  only declared supported profiles have the capabilities described below.

  The current initial-state input re-execution is a limited first step toward
  these goals. It can compare changed code/configuration from the beginning of
  a supported run, but cannot display a recorded timeline or resume from an
  intermediate point. Schema, journal, execution boundary, identity, USO
  mapping and tracing serve these goals or other future uses; none is a goal
  by itself.

  A first execution-checkpoint proof now supports one physics-off per-agent
  omni robot on a single straight path: capture at completed `S_k`, terminate
  the source process, restore in a fresh process and continue with
  `run_simulation(resume=True)`. Its suffix is compared against an uninterrupted
  run. See `docs/design/snapshot-replay/checkpoint-evidence.md` for the exact
  profile and limits; this does not make arbitrary simulations restorable.
  The checkpoint state capability checklist in
  `docs/design/snapshot-replay/product-goals.md` distinguishes the proven
  single-omni subset from later save/load work. The manipulation scenario's
  `docs/design/snapshot-replay/manipulation-state-scenario-evidence.md`
  records the observation and ownership findings behind part of that list.
  The example-side physics-off omni manipulation profile now offers opt-in
  per-step state recording, fresh-process restore/resume and result playback
  through the validation example. This is one robot with all kinematic joint
  positions/targets recorded (one driven by the scenario), and one box, with
  the ordinary omni pose path, generated approach waypoint, final turn,
  direct supported commands and an explicit application-stage hook;
  it is not arbitrary-example save/load or Action/BT/plugin restoration. See
  [manipulation validation guide](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/manipulation-validation-guide.md) and the
  [V1 feature plan](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/feature-v1-spec-plan.md).

  Distinguish three records where the user goals require them:

  | Record | Primary question | Current status |
  | --- | --- | --- |
  | State observation / execution checkpoint | What was the world state? Can execution resume? | The omni manipulation profile records a full checkpoint each completed step and resumes from it in a fresh process. Arbitrary controllers and state owners remain unsupported. |
  | Input and event journal | Which effective input was applied, when, and what ack/outcome followed? | The manipulation profile records supported Fleet API commands and box spawn/remove operations; arbitrary direct mutators and broader typed events remain future work. |
  | Operation trace | Why did a task take this path across RMF, ROS, agents and devices? | Future sampled action/operation spans; not a replay input source. |

  Snapshots alone cannot yield the causal operation trace: sampled poses omit
  requests, decisions, retries, rejection reasons and cross-process timing.
  The input/event journal can anchor a trace timeline, but end-to-end spans
  require context propagated at each integration boundary. Correlate sinks with
  stable run/step/command/operation IDs and explicit simulation versus wall time;
  v1 persists run, step/phase/order, source and command IDs, but not end-to-end
  operation/trace IDs or wall-time spans. Keep
  durable replay writes on a failure-visible path; optional EventBus/OTel sinks
  may share emissions but must not be the sole source of required replay data.
  See [Observability](#observability) for the future trace exporter.

  For the next scope review, a **provisional v2 candidate** is tooling around
  the supported artifact: inspect/re-execute/compare CLI, opt-in recording in
  compatible examples, and limited result playback. Playback reads
  saved observations; it does not require variable `dt` or controller-state
  serialization, but needs explicit sampling, timing and display semantics. A
  **provisional v3 candidate** is broader execution profiles based on concrete
  failures, starting with differential navigation and attach/detach plus their
  required inputs/state. Capturing controller, attachment and other relevant
  internal execution state would support a later checkpoint contract; expanding
  recorded observations and example capture would support useful playback.
  Neither expansion alone establishes a complete restore boundary. Generic
  recording of arbitrary examples and checkpoint/restore remain separate scope
  decisions. These version labels are discussion aids, not approved milestones;
  playback, input re-execution and checkpoint/restore can advance independently.

  Remaining candidates, requiring concrete use cases and separate scope:

  - Extend the opt-in execution boundary incrementally to other simulation
    commands and state-changing subsystems, then evaluate whether it should
    become the default architecture. Define the reproducibility guarantee per
    profile: explicit input order alone is insufficient without controlled
    randomness, time, external inputs, scheduling, assets and relevant runtime
    state. Preserve ordinary API compatibility during any migration.
  - Result playback/seek for supported observations, with GUI viewing as the
    primary user operation. Resolve a registered renderer/profile from the
    artifact manifest for a future `pbf-playback <artifact>` command, rather
    than requiring users to instantiate an example-specific profile manually.
    Share profile resolution with restart tooling; keep headless frame access
    available for tests and external analysis. Define a separate explicit GUI
    renderer contract instead of discovering optional `play_gui` with `getattr`.
    Replace the validation renderer's hard-coded robot/box updates with a loop
    over declared frame entities and registered display handlers; include
    spawn/remove behavior and stable-key-to-rendered-object mapping before
    claiming dynamic-world playback. Keep camera controls usable during
    playback. Treat a recorded camera pose, if useful, as optional initial
    display metadata with a user override, not as execution-checkpoint state
    or a per-frame camera reset. Delta encoding is a separate storage
    optimization, not a playback prerequisite.
  - Additional profiles, starting from concrete failure cases: differential
    navigation, attach/detach and the controller/attachment state needed to
    explain their outcomes; then joint/action/device/BT and dynamic worlds as
    justified. Specify initial re-execution state separately from any
    intermediate checkpoint state. The checkpoint follow-up profile inventory
    in `docs/design/snapshot-replay/checkpoint-candidate.md` records what the
    single-omni restore proof still omits; differential
    navigation is a plausible next narrow checkpoint profile, pending its own
    Scope Review and fresh-process continuation test.
  - **Action-based manipulation checkpoint discovery:** after the explicit
    Fleet API manipulation-state scenario has established the base, joint,
    attachment and entity-lifecycle boundaries, have an external driver queue
    an equivalent sequence through existing `MoveAction`, `JointAction`,
    `PickAction` and `DropAction`
    (or the smallest applicable subset). Compare the user-facing simplicity
    and observable outcomes with the direct-command scenario, then identify
    the additional execution state needed to continue mid-action: queued and
    active action identity/order, phase/sub-action progress, targets, timers,
    completion/failure state and any external driver state. Keep both examples
    while they reveal different boundaries; decide whether to consolidate them
    only after this comparison. Do not assume a CLI alone makes arbitrary
    action-based demos restorable, or expand the current state-discovery slice
    into an action framework. This is a later state-requirements investigation,
    not part of the initial low-level navigation checkpoint proof or a promise
    that either manipulation example can already be restored.
  - Reconstruct supported `Agent` and `SimObject` instances from their effective
    spawn information, instead of extending the replay-only `simple_cube` /
    `static_box` vocabulary as a general PBF entity model. Record enough spawn
    parameters and asset references to create the same initial objects. For
    user-provided URDFs, assume the referenced files remain available and
    unchanged; maintaining those files is the user's responsibility. Do not
    require bundling or hashing arbitrary user assets by default. State changed
    after spawn still needs its own capture and input boundary.
  - A CLI to inspect, re-execute and compare supported artifacts, plus an
    opt-in record mode for examples that explicitly use a supported input
    boundary. Generic `record: true` for arbitrary examples requires broader
    command/state coverage and is not a config-only change. A recorder attached
    to an existing simulation must capture its actual effective configuration,
    not fill omitted fields with ReplaySession creation defaults.
  - ROS ingress adapters and rosbag correlation; DDS/executor replay is not
    covered by core input re-execution.
  - Checkpoint/restore with an explicitly sufficient execution-state contract:
    capture and load the required controller, action, attachment/device and
    other mutable state for the approved profile, then continue normal
    simulation. Record and reapply post-checkpoint inputs separately when an
    identical continuation is required. Evaluate whether a PyBullet-native
    state format can preserve the relevant engine state; if usable, define how
    it combines with PBF-owned execution state and user-managed asset files.
  - Shared lazy state capture for ROS/co-simulation/replay after validating
    capture consistency and cost.
  - USO mapping/conformance, then cross-backend validation before extracting
    common libraries.
  - Independent observability sinks. Sampled traces cannot replace a complete
    input journal. The general EventBus remains an in-process name/kwargs API;
    only supported replay records have a persisted schema today.
  - If broader audit/debugging cases require it, define a versioned generic
    event-record schema and correlation fields before adding an independent
    durable event sink. Do not assume every EventBus emission is a complete
    state-change record or a safe required-I/O boundary.

- **Behavior tree integration** — The current portable profile supports
  `Sequence`, `Fallback`, `Repeat`, and registered `Action` nodes. It is not a
  full BehaviorTree.CPP runtime. Remaining work, in dependency order:

  1. Define portable `Condition` handler contracts and capability/query
     adapters, with deterministic tests.
  2. Add `SubTree` resolution, explicit input/output port remapping, scoped
     blackboards, recursion/cycle validation, and XML source locations in
     diagnostics.
  3. Add only the control/decorator nodes justified by scenarios (for example
     `Parallel`, reactive control nodes, cancellation/halting, timeout, and
     retry); document unsupported nodes rather than implying full compatibility.
  4. Specify typed port conversion and validation, tree/node lifecycle events,
     and snapshot-safe execution-state serialization.
  5. Define portable Agent `Pick`/`Drop` and device-request leaf contracts,
     including lift calls/status, door requests/status, and conveyor
     start/stop/sensor operations. Implement forklift, picker, conveyor, and
     device-tree profiles through adapters and verify their coordinated
     state-machine/BT flows.
  6. Move the validated XML core and portable contracts to the shared library
     only after a second simulator exercises the same profile.
  7. Define portable geometric navigation-zone schemas and sampling rules;
     retain explicitly named waypoint sets for deterministic MVP wandering.

### Crowd Simulation (lightweight)

Lightweight crowd simulation for NPC pedestrians in RMF demo environments. Provides more realistic pedestrian behavior than the current `RandomWalkController` (random point within radius) while staying far simpler than Gazebo's full Menge/ORCA `crowd_simulator` plugin.

**Motivation:** RMF demos (airport_terminal) use Gazebo's `crowd_simulator` with Menge BFSM + ORCA + nav mesh for 10 pedestrians. PyBulletFleet currently substitutes `RandomWalkController` which has no obstacle avoidance and no agent-agent interaction. A middle-ground is useful for visual fidelity in demo environments without the complexity of a full crowd simulation framework.

**Scope:**

| Feature | RandomWalkController (current) | Proposed | Gazebo crowd_simulator |
|---------|-------------------------------|----------|----------------------|
| Obstacle avoidance | None | Simple raycast or AABB check | ORCA + nav mesh |
| Agent-agent avoidance | None | Separation force (boids-style) | ORCA mutual |
| Goal selection | Uniform random in radius | Weighted goal sets (configurable) | BFSM state machine |
| Nav mesh | Not required | Not required | Required |
| Config complexity | 2 params | ~5 params (YAML) | 3 XML files |
| Computation | ~0 | O(n·k) neighbor lookup | O(n²) ORCA |

**Approach:** A new `CrowdController` (or enhancement of `RandomWalkController`) that adds:
1. **Goal sets** — Named groups of positions with transition weights (like Menge BFSM but in YAML)
2. **Simple separation** — Repulsive force from nearby agents (boids separation rule, not full ORCA)
3. **Static obstacle avoidance** — Optional raycast or AABB query to avoid walls (PyBullet `rayTest`)

**Priority:** Low — current `RandomWalkController` is functional for demo purposes. This is primarily for RMF demo visual fidelity (再現度).

#### Crowd-as-dynamic-obstacle throughput benchmark

The higher-value use of a crowd is not visual fidelity but **turning crowd density into a benchmark variable**: measure how much a fleet's *task* throughput degrades when robots must contend with pedestrians. Pure visual pedestrians (a human mesh wandering) have little functional value here, because the fleet is RMF-path-driven and does not avoid arbitrary dynamic obstacles — the NPCs are currently just eye-candy. Making them *dynamic obstacles the fleet reacts to* is what aligns with PyBulletFleet's strength (throughput/scale benchmarking at hundreds of robots, a regime too heavy for Gazebo).

**Three pieces:**
1. **Crowd as dynamic obstacles** — the lightweight `CrowdController` above, with pedestrian positions known to the sim.
2. **Robot reaction** —
   - *Minimal:* "pedestrian within X m ahead → slow/stop" via PyBullet `rayTest` / collision checks. No RMF changes; measures stop-and-wait delay.
   - *Full:* publish detected pedestrians as `rmf_obstacle_msgs/Obstacles` so RMF reroutes / closes lanes (heavier; integrates with `rmf_visualization_obstacles` and the obstacle pipeline).
3. **Task-throughput harness** — dispatch N patrol/delivery tasks and measure completion time / completions-per-hour. This is a **new metric** distinct from the existing RTF/step-time benchmarks (which measure sim performance, not fleet task throughput).

**Experiment:** sweep crowd density × avoidance{off, stop, reroute} → task throughput, at 100–1000 robots. Reuses existing collision detection, `rayTest`, EventBus, and the benchmark harness scaffolding.

**Priority:** Low / research — schedule after the ROSCon minimum (working demos + packaging). Start with the minimal reaction (stop-on-detect, no RMF changes).

### Devices

Door, Elevator, and workcell simulation are implemented in the core/bridge
integration.  ROS/RMF-specific gaps and device-model enhancements are tracked
in `ros2_bridge/README.md`; this roadmap retains only cross-project asset work.
See [device-external-agent/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/2026-04-21-device-external-agent/spec.md)
for the implemented device/external-actor model.

## RMF Demo Feature Coverage

Gaps between the standard Gazebo `rmf_demos` and the PyBulletFleet bridge. Core
flows (navigation, doors, **lifts**, delivery pick/drop, battery/charging,
**cleaning coverage**, **georeferenced maps**) are implemented. Ported demos:
office, clinic, hotel, airport_terminal, battle_royale, **campus**. Outstanding:
See [open-rmf/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/open-rmf/spec.md)
for the bridge design and the original integration boundary.

**Demos not ported — require an RMF mode the bridge doesn't implement:**
- `office_mock_traffic_light` and `triple_H` — both use the **`EasyTrafficLight`** adapter (the robot owns navigation; RMF only gates it with go/stop). This inverts the bridge's full-control model and is a poor fit for the kinematic, RMF-planned sim — see the [TrafficLight discussion]. Relevant only once a co-simulation/robot-proxy layer exists. Run them in Gazebo for now.

**RMF tasks/actions — partial** (`fleet_adapter.execute_action`):
- **dock** — treated as a plain navigate (no dedicated dock-path approach; endpoint is reached, only the precise approach trajectory is skipped — negligible in a kinematic sim).
- Other custom performable actions — logged and finished, not executed.

**Substituted** (works, but differs from Gazebo):
- Crowd — Menge/ORCA `crowd_simulator` → lightweight `RandomWalkController` (no avoidance). See [Crowd Simulation](#crowd-simulation-lightweight).
- Caddy — Gazebo keyboard teleop (`diff_drive` plugin) → autonomous `PatrolController`.

**Not modeled by design** (kinematic focus):
- Robot sensors (lidar/camera), physics dynamics (wheel torque/contact).
- `rmf_obstacle` detection pipeline (also unused in stock Gazebo demos; see the "Crowd-as-dynamic-obstacle throughput benchmark" section above).

Detailed ROS/RMF release and validation plans live in `ros2_bridge/README.md`;
keep this section focused on user-visible feature coverage.

## Interfaces

External communication layers:

- **Fleet API actor scopes** — Separate simulation actor ownership from public
  fleet visibility. Today `FleetStateProvider` publishes all `sim.agents` and
  `FleetCommandDispatcher` resolves every named Agent, while `AgentManager`
  groups execution only. Add a short-term filter for fleet-state publication
  so mock people, external robots, and devices do not inflate `/fleet/states`.
  Then add named API scopes that can be attached to one or more managers and
  independently select state publication and command ingress (with distinct
  endpoint namespaces when multiple fleets are exported). Keep the state and
  command scopes explicit rather than inferring ownership from an Agent type.
  Acceptance criteria: a state-only filter can be deployed first; a named
  scope can select one or more managers; and changing command ingress never
  implicitly changes state publication (or vice versa).
  See [fleet-control-api/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/fleet-control-api/spec.md).

- **ROS 2** — Topic / service / action bridge for ROS 2 ecosystem integration
  (see `ros2_bridge/README.md` and
  [ros2-bridge/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/ros2-bridge/spec.md))
- **gRPC** — Language-agnostic RPC interface for orchestrators, WMS, and fleet managers
- **Distributed co-simulation (Robot Proxy layer)** — Per-robot proxy processes that translate between simulated robots and real Robot Apps (Nav2, task assigners, BTs). Enables running 100+ unmodified single-robot software stacks against a centralized batched simulator. Sim Central stays single-process and batched; only a thin fixed-schema boundary (`StateSnapshot` + `CommandBuffer` + Events) crosses the IPC. Its earlier proposal to share one `StateSnapshot` dataclass with replay is a future design candidate, not a contract of the v1 navigation replay artifact. See [co-simulation/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/co-simulation/spec.md) for layer separation, transport options (shared memory / gRPC / DDS), lockstep vs async sync modes, and the implementation phases.

## Refactoring

- **`MultiRobotSimulationCore` responsibility decomposition** — The core class has grown to ~3200 lines / 66 methods. Extract self-contained subsystems into dedicated classes composed into the core:

  | Subsystem | Target class | Methods to extract | Est. lines |
  |---|---|---|---|
  | Collision detection (spatial hash, AABB, broad/narrow phase) | `CollisionSystem` | `check_collisions`, `filter_aabb_pairs`, `_update_object_spatial_grid`, … (~20 methods) | ~1000 |
  | Visualizer (camera, transparency, keyboard) | `VisualizerController` | `configure_visualizer`, `setup_camera`, `_handle_keyboard_events`, … (~6 methods) | ~250 |
  | Profiling & memory tracking | `SimProfiler` | `record_profiling`, `_print_profiling_summary`, `_print_memory_profiling_summary`, … (~5 methods) | ~200 |

  Each subsystem is held by composition (`self._collision = CollisionSystem(self)`).
  The core class delegates; public API does not change.
  Natural stepping stone toward the SimBackend ABC (Long-Term Phase 1).

- **Move `controller.py` into `controllers/` package** — Base controller classes (`Controller`, `KinematicController`, `OmniController`, `DifferentialController`, `create_controller`, `register_controller`) live in `pybullet_fleet/controller.py` while higher-level controllers (`PatrolController`, `RandomWalkController`) are already in `pybullet_fleet/controllers/`. Consolidate by moving the base module into the package as `controllers/base.py` (or splitting omni/differential into separate files) and re-exporting from `controllers/__init__.py`. Preserve the `from pybullet_fleet.controller import ...` path via a compatibility shim or `__init__.py` re-export.
  See [controller-refactor/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/controller-refactor/spec.md).

- **Remove scipy runtime dependency** — Quaternion helpers now cover vector
  rotation, multiplication, rotation-vector conversion, and angle comparison;
  the batch omni hot path already uses them.  Audit and migrate the remaining
  `Rotation` callers in `geometry.py`, `_motion_planning.py`, and the batch
  differential controller, then remove the dependency only after equivalent
  numerical tests cover the replacements.  This is an install-size and
  dependency-simplification task, not a promised performance improvement.

## GUI / Interactive Tools

Improve `p.GUI` interactivity for development and debugging.
Long-term plan is to replace `p.GUI` with Rerun (see [Long-Term Phase 1](#long-term-backend-abstraction--beyond-pybullet)) where many of these capabilities come built-in. These items are the **interim layer** until that migration lands, and naturally collect under the planned `VisualizerController` (see Refactoring).
The existing camera interaction boundary is described in
[camera-control/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/2026-04-18-camera-control/spec.md).

Current state:
- `sim.pause()` / `resume()` Python API ✅
- `SPACE` keybinding for pause toggle ✅
- `pause` / `resume` events on EventBus ✅
- `CameraController` with right-drag pan, zoom, top-down view ✅

Planned:

- **Pause / Step / Speed control panel** — In-window UI for pause/resume, single-step, and real-time multiplier. Short-term: `p.addUserDebugParameter` sliders/buttons. Mid-term: tkinter side panel composed with `DataMonitor`.
- **SimObject inspector panel** — Live list of all `_agents` / `_sim_objects` with `object_id`, `name`, current `Pose`, action queue length, and status. Click-to-select with viewport highlight.
- **Camera focus on selection** — Selected entity becomes camera target; auto-follow toggle; distance / yaw / pitch sliders. Replaces the current static `setup_camera`.
- **Object manipulation** — Drag selected `SimObject` in XY plane, edit pose via numeric input, delete via UI (calls `remove_object()` properly so collision/EventBus stay consistent). Respects `pickable` flag.
- **Action queue inspector** — Per-Agent visualization of pending and active actions (type, status, target). Right-click to cancel an action.
- **Keyboard shortcut overlay** — Help panel listing all bound keys (`SPACE`, future shortcuts), surfaced via a `?` key.

Implementation notes:
- All new GUI affordances must be **opt-out via `SimulationParams(gui=False)`** so headless / CI paths stay zero-cost.
- UI state changes must publish through the EventBus (`gui.select`, `gui.manipulate`, `gui.action_cancel`) so plugins / recordings see them.
- Avoid per-step UI redraws — throttle to ≤30 Hz independent of sim FPS.

## Performance

Near-term optimizations within the current PyBullet-backed architecture.
Goal: reduce per-step cost at 100–1000 agents without a full backend swap (see Long-Term section for that).

Measured throughput and profiling methodology are maintained in
[Benchmark Results](benchmarking/results) and the profiling guide.  They are
configuration- and hardware-dependent, so this roadmap intentionally does not
duplicate point-in-time timings or projected speedups.

### Two-Phase Pose Commit: Decouple Computation from PyBullet C API ✅ (batch controller path)

Base-pose updates are buffered during simulation work and committed together,
which lets manager-level batch controllers vectorize compatible agents.  The
public API remains pose/controller based; joint control and plugin code may
still make necessary PyBullet calls.  See [Two-Phase Simulation Step](architecture/two-phase-step)
for the operational model and extension boundaries, and
[two-phase-step/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/two-phase-step/spec.md)
for the design record.

### Vectorized Agent Update (NumPy Batch) ✅

`fleet_controller.type` selects the batch omni or differential implementation
for a named manager.  Agents that are not compatible with that batch path keep
their normal per-agent controller behavior.  See the fleet tutorial and
controller configuration reference for user-facing configuration.

### C++ Extensions for Hot-Path Functions

Consider native extensions only after a reproducible profile identifies a
dominant Python hot path.  Likely candidates are interpolation/quaternion work
and broad-phase spatial hashing; PyBullet narrow-phase calls are already
native and are not an initial target.

### Deferred AABB Update

Audit whether collision-disabled or low-frequency configurations perform more
AABB/grid work than required.  Any change must preserve collision-event timing
and the current two-phase synchronization guarantees.

## SDF & DAE Support Improvements

PyBulletFleet currently has two self-implemented workarounds for Gazebo ecosystem interop:

See [sdf-loader/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/sdf-loader/spec.md)
for the original SDF-loading scope and constraints.

1. **`resolve_sdf_to_urdf`** — Hand-rolled SDF→URDF XML converter (PyBullet cannot load SDF directly)
2. **DAE defensive fallbacks** — Colour extraction, texture symlinks, collision try/except (PyBullet has poor DAE support)

Both are functional but hacky. This section tracks improvements to make them more robust or replace them entirely.

### SDF → URDF Conversion: `gz sdf -p` Replacement

`resolve_sdf_to_urdf()` in `sdf_loader.py` is a minimal self-implemented SDF→URDF converter using `xml.etree`. It covers only `<link>`, `<joint>`, `<mesh>`, and primitive geometries — enough for simple robots (DeliveryRobot) but fragile for complex SDF models.

**Why self-implemented?** No usable Python SDF→URDF tool exists:
- `sdformat_urdf` — C++ only, no Python bindings
- `pysdf` (PyPI) — fails to build on Python 3.12+
- `libsdformat14` — C++ template API, not callable via ctypes
- `gz sdf -p model.sdf` — converts SDF→URDF via CLI subprocess (see evaluation below)

**`gz sdf -p` evaluation (Gazebo Harmonic / sdformat 14, as of 2026-04):**

| | Self-implemented parser | `gz sdf -p` |
|---|---|---|
| Dependencies | None (`xml.etree` stdlib) | `sdformat14` + `gz-tools` (~200 MB) |
| Coverage | Minimal (link/joint/mesh only) | Broader (frames, nested models partial) |
| Edge cases | Fix ourselves immediately | Wait for upstream fix |
| DAE colour injection | `_extract_dae_diffuse_color` embeds into URDF | Not supported (no colour in output URDF) |
| Nested `<model>` | Not supported | Partial support |
| Known issues | Pose hierarchy may drift for deep nesting | Nested model bugs, some pose frame resolution errors |
| Stability | Tested for our models | "Best effort" — known regressions between versions |

**Verdict:** `gz sdf -p` is **usable but not fully stable**. For simple robots it works; for complex SDF it has regressions. The dependency cost (~200 MB) is also significant. **Current self-implemented parser is the pragmatic choice.**

**Future action:**
- Monitor `gz sdf -p` stability across Gazebo releases
- If it stabilises + we already have `gz-tools` in Docker for other reasons → switch to subprocess call and delete `resolve_sdf_to_urdf()`
- If we support complex SDF robots (nested models, sensors) → worth the dependency

### DAE → OBJ Automatic Conversion

PyBullet has poor DAE (COLLADA) support: textures are not loaded, diffuse colours are ignored, and some meshes fail `createCollisionShape`. Gazebo Fuel models ship exclusively as DAE.

**Proposal:** Use `assimp export` (from `assimp-utils`, ~2 MB) to convert DAE meshes to OBJ + MTL at download time. OBJ is PyBullet's best-supported mesh format — colours, textures via MTL, and collision all work reliably.

Changes needed:
1. **`docker/Dockerfile.rmf_demos`** — `apt install assimp-utils`
2. **`docker/download_fuel_models.py`** — Post-download `assimp export *.dae *.obj` conversion step
3. **`pybullet_fleet/sdf_loader.py`** — Prefer `.obj` over `.dae` when both exist (`mesh_path.replace(".dae", ".obj")`)

**Simplification benefit:** Once all DAE meshes have OBJ counterparts, three defensive workarounds in `sdf_loader.py` become unnecessary and can be removed:

| Current workaround | Why it exists | Removed after conversion |
|---|---|---|
| `is_fuel` flag on `_IncludeInfo` | Fuel DAE has embedded colours; local OBJ does not → different `rgba_color` handling | OBJ+MTL always carries material info → uniform `rgba_color=None` for all models |
| `_ensure_texture_symlinks()` | DAE references textures by bare filename; PyBullet resolves relative to `meshes/` not `materials/textures/` | OBJ+MTL uses relative paths that resolve correctly |
| `try/except` around `createCollisionShape` in `sim_object.py` | Some DAE meshes crash PyBullet's collision shape loader | OBJ collision shapes load reliably |

The `is_fuel` flag would be replaced by a mesh-format check (e.g. "does an `.obj` file exist alongside the `.dae`?"), making the loader source-agnostic — any DAE model from any provider would be handled the same way.

### Multi-material mesh textures (per-material visuals)

`createVisualShape(GEOM_MESH)` applies a **single texture per mesh**, so a
multi-material mesh (one OBJ/DAE packing many materials) renders with just one
texture tiled over everything. The campus env is a single 50-material OBJ and
showed one (a person) texture across the whole scene. Current workaround: the
`world.force_color` config flat-shades such a mesh grey (strips its `mtllib`
so no texture loads — see `_obj_without_materials`). Proper fix: split the mesh
by material into per-material sub-visuals (reusing the trimesh `<submesh>`
extraction path), one texture each. Moderate effort; helps any multi-material
environment.

### Priority

Low — SDF parsing, DAE handling, and multi-material textures are all functional
enough for demos (with the flat-shade workaround for multi-material meshes).
These improvements primarily affect:
- **Visual fidelity** (DAE→OBJ: textures and materials render correctly)
- **Maintainability** (gz sdf -p: delete ~150 lines of hand-rolled XML conversion)
- **Robustness** (both: fewer edge-case failures with complex models)

## Environments

Simulation environment assets (warehouse floors, factory layouts, etc.):

- **`pybullet-fleet-environments` package** — Manage environment assets in a separate repository, installable via `pip install pybullet-fleet-environments` for on-demand retrieval. Keep PyBulletFleet core lightweight by not bundling meshes.
  - AWS RoboMaker Small Warehouse (MIT-0): DAE→OBJ converted meshes + URDF wrappers
  - Open-RMF rmf_demos maps (office, hotel, clinic, airport, campus): OBJ mesh export
  - pybullet_data bundled environments (kiva_shelf, samurai, stadium) wrappers
  - Original license clearly noted per environment
- **`resolve_environment()` API** — Name resolution similar to `resolve_model()` for loading environments. Shows install hints when not installed.
- **USD entity overrides and Agent promotion** — Keep imported USD geometry
  static by default; use a versioned PyBulletFleet sidecar to promote explicit
  semantic candidates (for example an Isaac forklift root) into movable
  entities with controller and collision-proxy settings. See
  `docs/design/usd-behavior-tree/spec.md`.

## CI / DevOps

- **GitHub Actions refactoring** — Streamlined CI pipeline
- **Automated performance tracking** — Run time / memory benchmarks in CI, auto-update results in documentation, and alert on significant performance regressions
  See [benchmark-refactor/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/benchmark-refactor/spec.md).

## Observability

Operational visibility for long benchmark runs and production deployments.
Optional traces and metrics may consume EventBus emissions, while the v1 replay
session records its supported inputs/results directly so I/O failures remain
visible. The sinks should share correlation IDs where available, not
depend on one universal emission point.
See [observability/spec.md](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/observability/spec.md)
for the design.

- **Structured logging (JSON)** — Replace prefix-string `NamedLazyLogger` with `extra={...}` dict fields (`agent_id`, `object_id`, `sim_time`, `step`). Enables Loki/Grafana label queries.
- **`sim_time` injection filter** — `logging.Filter` registered by `MultiRobotSimulationCore` auto-attaches `sim_time` and `step` to every record. Removes per-call boilerplate on hot paths.
- **Prometheus / OTLP metrics** — Promote `DataMonitor` internals (`pbf_step_duration_seconds`, `pbf_active_agents`, `pbf_actions_total{type,status}`, `pbf_collision_pairs`) to scrapable metrics. Opt-in to keep default cost zero.
- **OpenTelemetry trace exporter** — EventBus subscriber that converts action
  lifecycle events into spans.  Sampling is allowed; trace and operation IDs
  are copied into structured event records so Tempo "Trace to Logs" can locate
  the corresponding JSONL record in Loki.
- **End-to-end robot-operation tracing** — Make a single fleet operation
  debuggable across RMF task ingress, fleet/ROS command handling, the target
  agent action, and workcell/plugin completion.  Propagate a trace context and
  stable operation ID at each boundary; attach `fleet_id`, `robot_id`,
  `task_id`, and request/command IDs to spans and structured logs.  The same
  attribute vocabulary must work for a simulator-only run, ROS 2 deployment,
  and a future Robot Proxy, so traces can be compared with real-robot
  operations.  This is action/operation-level tracing, not a span per
  simulation step.
- **Hot-path safety** — All exporters use batch / async processors. Per-step spans are forbidden; Action-level granularity only. `LazyLogger.isEnabledFor()` semantics preserved.

## Long-Term: Backend Abstraction & Beyond PyBullet

Current architecture is tightly coupled to PyBullet (`body_id`, per-entity FFI
calls).  The following exploratory items investigate a backend boundary while
preserving the Python user API.  They require workload-specific benchmarks and
are not performance commitments.

### Short / Mid-Term Performance TODOs

- **Release-like profiler mode** — Add a fixed first-window profile mode (for example 1000 agents, 50% moving, first 100 measured steps after warmup) so profiler output matches release benchmark conditions. Long profiling runs can understate moving-agent cost after robots reach their goals.
- **Active moving set** — Keep an explicit set of moving agents so stationary agents skip more of the hot path.
- **AABB/grid flush reduction** — Audit whether low-frequency or disabled collision modes still refresh more AABB/spatial-grid state than needed.
- **Ground-robot 2D path** — Make the 2D collision/grid path easier to use for AGV-style fleets where Z-axis neighbor checks are unnecessary.
- **Manager-scoped ROS interface deployment evaluation** — Optional `/fleet/<manager>/...` endpoints, manager-specific state rate/QoS, command isolation, and the global `/fleet/...` compatibility API are implemented. Same-host transport checks confirm that manager streams benefit selective subscribers but do not increase total throughput when every stream is subscribed. Before defining a manager-size limit or adding same-manager state partitions, measure total CPU, full-snapshot completion latency, slow-subscriber behavior, and networked deployments. A manager name is not implicitly an RMF fleet name. See the [manager-scoped interface design](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/fleet-control-api/partitioned-interfaces.md) and `ros2_bridge/PERFORMANCE.md`.

### Phase 1: SimBackend ABC + Numpy Pure Kinematic Backend

- **SimBackend ABC** — Abstract interface (`set_positions_batch`, `detect_collisions`, `load_model`, `step_physics`) that `Agent` and `SimObject` program against instead of raw `pybullet` calls
- **NumpyBackend (candidate default)** — Contiguous numpy arrays for
  positions/orientations and a spatial-index collision implementation.  It
  would have no physics-engine dependency in kinematic-only mode; validate
  correctness and throughput before choosing it as the default.
- **PyBulletBackend (compat)** — Wraps existing PyBullet calls behind SimBackend ABC for backward compatibility and physics-mode users
- **URDF parsing without PyBullet** — Use `yourdfpy` or similar to parse URDF into internal model data, removing the last hard dependency on PyBullet for kinematic-only mode
- **Visualization decoupling** — Replace `p.GUI` with Rerun, Open3D, or RViz for rendering. Backend-agnostic scene display

### Phase 2: Native Backend (Rust/C++ via PyO3/pybind11)

Only justified if Phase 1 numpy performance is insufficient (e.g., 5000+ agents at 240 Hz).

- **Rust kinematic core** — Position update, yaw integration, and AABB
  collision in Rust, exposed through PyO3.  Evaluate only against a measured
  NumpyBackend baseline.
- **Batch collision in Rust** — Sweep-and-prune or spatial hash for O(n log n) broad-phase, replacing Python KDTree
- **Zero-copy interop** — Share numpy arrays directly with Rust via buffer protocol (no serialization)

### Phase 3: GPU Backend (optional)

For 10,000+ agent scenarios or RL training workloads.

- **MuJoCo MJX backend** — JAX-accelerated batch physics on GPU. URDF via MJCF conversion
- **JAX pure kinematic** — `jax.numpy` drop-in for NumpyBackend, JIT-compiled, GPU-parallel

### Phase 4: ECS Architecture (v2.0 candidate)

Considered when entity diversity explodes beyond what Agent/SimObject OOP can express cleanly (e.g., drones + ground robots + conveyors + elevators + dynamic obstacles all in one scene with distinct component sets).

- **Component-based entity model** — Replace Agent/SimObject inheritance with composable components (Transform, Collision, Kinematics, JointState, ActionQueue, etc.)
- **System pipeline** — Each system (MovementSystem, CollisionSystem, ActionSystem) operates on component arrays. Maps naturally from current Controller ABC → System
- **Rust ECS runtime** — Leverage `bevy_ecs` or `hecs` for cache-friendly SoA memory layout. Python API via PyO3 for user-facing logic
- **Migration path** — Controller ABC → System, EventBus → ECS Events, Registry → ECS resource/archetype queries. Plugin Architecture phases are designed as stepping stones toward this

> **Note:** At this point the project may evolve beyond "PyBulletFleet" in name, as PyBullet would be just one optional backend among several, and the core would no longer depend on it. A rename (e.g., *FleetSim*, *KinematicFleet*) may be appropriate.
