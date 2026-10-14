# Snapshot / Replay — product goals and capability boundary

**Status:** Direction for future scope reviews, not an approved implementation
plan or a claim that these capabilities already work.

The existing replay and checkpoint implementations are development validation
profiles. General recording, restart and playback of ordinary simulations are
not ready for general or production use.

Read this document before scoping further snapshot, restart, playback or trace
work. Each change should state which user operation it advances, what it makes
observable, and which supported profiles remain excluded. The [v1 scope](spec.md)
and [single-omni evidence](checkpoint-evidence.md) describe narrower delivered
capabilities.

The approved end-to-end feature theme is specified in the
[Snapshot / Restore / Replay / Playback V1 plan](feature-v1-spec-plan.md).
PR #59 delivered a validation profile, not general V1 support.

## Intended user workflow

1. **Record:** A user opts in when running an ordinary simulation (conceptually
   `save=on`). PBF records enough time-indexed simulation results for observation
   and, for declared supported profiles, enough execution state and effective
   inputs to restart. Recording must identify its coverage and completeness;
   merely saving poses does not create a resumable checkpoint.

   The application chooses a goal or attachment operation, but once PBF
   accepts it, the effective command and its simulation step/phase/order and
   parameters must be available for recording. This includes movement and,
   for future supported profiles, attach/detach, spawn/delete and permitted
   runtime configuration changes. PBF must expose trustworthy execution facts;
   the current PBF recorder owns the durable journal. The supported Fleet API
   commands and box lifecycle path are recorded, but direct mutators and other
   inputs are not covered. A current active goal in a checkpoint does not
   recover the full earlier input history.
2. **Restart:** A user selects a recorded time (conceptually
   `pbf-restart <artifact> --time <t>`). The tool selects the latest usable
   checkpoint **at or before** `t`, never one after `t`, so the requested
   interval is not skipped. Whether it then advances from that checkpoint to
   exactly `t` using recorded inputs, or starts at the selected checkpoint,
   remains a later design decision. After restore, normal PBF execution runs.
   An external application can then try another policy; it must restore or
   reconstruct any state it owns. PBF settings may be changed at runtime only
   where the ordinary API permits that change. A changed policy/configuration
   is a variant, not an identical reproduction.

   The Python API should make the normal supported case approximately a
   single load operation followed by ordinary `run_simulation()` (conceptually
   `sim = load(dir); sim.run_simulation()`). The exact method name and whether
   it constructs a new core or initializes an existing fresh core remain to be
   designed. The caller should not have to load a manifest, construct a
   profile, select a checkpoint and pass raw state dictionaries separately.
   Keep lower-level operations available for inspection and custom workflows.
   Internal schema/profile/handler boundaries are implementation details of
   this simple user operation, not separate user steps or automatic reasons to
   split the feature into further slices.
3. **Playback:** A user observes recorded results without rerunning the original
   policy or physics (conceptually `pbf-playback <artifact> --rtf <rate>`).
   Playback pacing is a display control, distinct from simulation RTF and from
   restarting an executable simulation. Sampling and seek behavior must be
   defined so short events are not silently claimed as observed.
4. **Debug:** A user inspects a time-aligned trace of relevant inputs, events,
   their start/end times, outcomes and effective parameters to understand what
   happened. The trace must state what is measured versus inferred and what
   crosses the PBF/external-application boundary. It complements snapshots;
   snapshots alone cannot explain causal decisions.

The desired result is to reproduce a failure, watch it, restart before it to
try a changed policy or permitted runtime parameter, and compare outcomes.
CLI names, configuration names, artifact layout, checkpoint cadence and exact
`--time` semantics are provisional. Do not treat all recorded samples as
execution checkpoints or assume that arbitrary PBF/controller/plugin/ROS state
is serializable. The checklist below tracks known checkpoint profile limits; it is not exhaustive.

## Current position

- PR #50 implements restricted initial-state + recorded-input re-execution and
  comparison. That API remains separate from the supported manipulation
  recording artifact.
- The separate single-omni proof restores one fixed physics-off straight
  navigation profile at a completed step in a fresh process and continues
  with `run_simulation(resume=True)`. It does not provide opt-in recording of
  arbitrary simulations, selectable-time restart or the proposed CLI. Its
  reference mode uses `run_simulation()` with `POST_STEP` observations, while
  source mode calls `step_once()` to stop at one predetermined completed-step
  boundary. Ordinary `run_simulation()` with opt-in checkpoint capture remains
  a distinct TODO in that earlier proof.
- The physics-off omni manipulation validation scenario now uses the PBF-owned
  kinematic profile for recording, restore and GUI result playback. Its
  registered data restorer reinstalls the external scenario stage and callback
  during the public one-call load; plain `run_simulation()` continues. It drives one
  named joint but captures every joint's position, configured target and ramp
  execution on
  the supported robot, plus one box; this does not imply arbitrary-example
  save/load or general recorded-input re-execution.
- The declared PBF-owned kinematic profile now covers a fixed-arm validation
  program and an omni mobile-arm composition. One public load call restores a
  complete artifact and plain `run_simulation()` continues. The fixed-arm
  callback stage is an explicitly registered external data restorer. The
  ordinary teaching demo remains unchanged; its opt-in recording and general
  CLI save/restore/playback commands are still user-visible gaps. A small CLI
  recording entry point for supported ordinary examples is a useful next
  validation tool before expanding owner and controller coverage, but it must
  report unsupported state instead of claiming universal capture.
- Broader controller, entity lifecycle, Action, device, physics, external app
  and ROS/RMF state, plus a general trace and playback UI, remain separate
  scope decisions. Implementation should expand from concrete supported
  profiles and evidence rather than claiming whole-simulation capture.

## One-call restore and reusable supported profile

The [approved spec and plan](user-facing-restore-spec-plan.md) promotes the
validation proof to a supported PBF operation without claiming universal state
capture. Its acceptance conditions are:

1. A supported ordinary PBF example can enable recording during its normal run.
   **Open:** the fixed-arm acceptance was moved to validation after Human
   review to keep the ordinary demo simple.
2. In a fresh process, one public API accepts an artifact and constructs and
   restores a simulation ready to run. Profile/handler resolution and artifact
   validation are PBF's responsibility; application-owned continuation state
   remains explicit where required.
3. The restored simulation continues through ordinary `run_simulation()` with
   no scenario-specific resume sequence.
4. A second, differently composed supported simulation matches an equivalent
   uninterrupted reference at the checkpoint, during continuation, and at the
   final outcome under stated tolerances.
5. Unsupported or missing execution state prevents a claim of successful
   restore. During recording, such failures must be reported without stopping
   an otherwise healthy simulation.
6. Measure artifact size, per-step recording overhead and load time, including
   the cost of storing both frames and checkpoints. Use those results to decide
   whether the format needs a separate change; format redesign is not a
   completion condition for this user operation.

Choose the ordinary examples and supported state owners in the implementation
plan. Reuse their normal execution paths rather than adding another validation
program solely to satisfy this list.

## Checkpoint state capability checklist

"Observed" means a value can be read or supplied in one run; it does **not**
mean that it can be persisted, loaded or resumed. **Done means proven only for
the named profile**, not for every robot or for the manipulation scenario that
motivated this inventory. This checklist tracks execution-checkpoint state,
not playback or trace coverage. It is a capability inventory, not an approved
implementation sequence or a guarantee that every future plugin or controller
state has already been identified.

| State or operation | Checkpoint status | Remaining boundary |
| --- | --- | --- |
| Recording profile and owner-state extension | **Partial.** PBF now has a versioned kinematic profile, owner handlers and trusted registry, proven with fixed-arm and mobile-arm compositions. | Other controller/entity types and custom `SimObject` state need declared handlers and separate continuation evidence. A versioned envelope does not make arbitrary state restorable. |
| Single-call supported restore entry point | **Done for the declared kinematic profile.** `load_supported_simulation()` validates the complete artifact, selects the earlier checkpoint, resolves trusted handlers and returns a ready core. Plain `run_simulation()` continues it. | Required application restorers must be registered by installed code; arbitrary worlds are not supported. |
| Separate state schemas, handlers and profiles | **Partial.** PBF-owned base/joint/primitive/attachment handlers are separated from the example stage. One profile iterates declared live owners and composes them in the acceptance simulations. | The current profile still owns explicit dict validation and restore order. Mesh/URDF objects and custom owner schemas need separate evidence; avoid claiming generic serialization. |
| Checkpoint validation responsibility and cadence | **TODO.** The recorder runs common-envelope and profile-specific validation after every completed step; the validation profile also rechecks invariant manifest fields and the robot asset hash. Loading and restoring validate again. | Separate construction/manifest compatibility checks at recording start and artifact load from changing-state checks per step and full restore checks. Compare declarative validation (for example, strict Pydantic models or JSON Schema) for both the common manifest/checkpoint envelope and concrete owner state before selecting one approach; check exact JSON types, versioning, added dependencies and per-step cost. Keep cross-field restore invariants and file hashes explicit where a schema alone is insufficient. Retain profile-specific support limits without placing scenario expected outcomes in reusable schemas. Measure cost before changing validation cadence, and ensure a failed capture is reported as an incomplete recording rather than silently trusted. Reuse state invariants in tests where useful; test assertions for trajectories, timing and final outcomes remain separate. |
| Duplicate frame/checkpoint storage and long recordings | **TODO.** V1 writes a result frame and a full checkpoint at every completed step, duplicating much of the agent/object state; the one-robot measurement does not establish long-run or fleet-scale cost. | Prefer investigating full per-step checkpoints as the single stored state source, with playback frames projected from them without advancing controllers or physics. Preserve step/seek, lifecycle visibility, recording integrity and the ability to distinguish display data from restorable execution state. Compare artifact size, per-step write cost/RTF, playback seek/read cost and memory against the current format over longer runs and representative fleets before changing the format or checkpoint cadence. Playback may pay more read cost, but recording should remain practical. |
| User-defined state extension | **Partial.** Versioned profile factories and required `DataRecord` restorers can be explicitly registered as trusted installed code; the fixed-arm stage uses this contract. | User-defined controller/entity execution-state composition inside the built-in profile remains unsupported. A `DataRecord` does not itself make arbitrary behavior resumable. |
| Caller-defined data records | **Partial.** `DataRecord` callbacks receive a completed-step context and contribute named, versioned JSON data to results/checkpoints. The manipulation scenario records its stage through a required `scenario` record. | Callers must re-register callback code when continuing; its execution is not serialized. PBF does not infer how arbitrary record values affect future behavior. Callback failures end recording and mark the artifact incomplete; partial-record recovery remains open. |
| Completed step/time and base pose, velocity and moving flag | **Done: single-omni proof and supported manipulation profile.** A completed-step clock and `Agent` motion fields are restored in a fresh process. | Other world/controller profiles must verify their own state and step semantics. |
| Active omni pose goal/trajectory | **Done for the supported manipulation profile.** The ordinary `set_goal_pose()` path, generated approach waypoint, current waypoint and asymmetric forward deceleration are captured and resumed. | Omni velocity-mode commands are not part of this profile. `Agent.restore_motion_state()` alone does not restore controller execution. |
| Omni final-orientation rotation phase | **Done for the supported pose-navigation profile.** The final turn's slerp/TPI state resumes from an in-progress checkpoint and reaches completion. | Other controller modes and custom controller implementations need separate profiles. |
| Omni multi-waypoint path and auto-approach | **Done for the profile's generated approach + destination path.** The saved path and current waypoint survive fresh-process restore. | General multi-goal input commands are not exposed by this scenario's Fleet API surface. |
| Omni velocity-mode execution | **TODO: not captured or restored.** | Capture body-frame linear/angular commands and their watchdog/mode state when a supported scenario needs it. |
| AgentManager construction, membership and callbacks | **TODO: not captured or restored.** The current kinematic profile rejects every registered manager, including one without a batch controller. | Reconstruct supported manager identity/configuration and ordered membership using stable entity names rather than runtime IDs. Inventory callback registration, external callback-owned state and frequency/scheduler phase separately; require installed callback code and an explicit state contract where continuation depends on it. A manager must not become implicitly restorable merely because its agents are. |
| Batch controller execution state | **TODO: not captured or restored.** The current kinematic profile rejects registered batch managers even if their commands entered through Fleet API. | Build on a supported AgentManager reconstruction, then inventory omni/differential batch-owned member order, goals/paths, phase, velocity and timing state; define versioned capture/restore for a supported composition; compare a fresh-process mid-motion multi-robot continuation with an uninterrupted run. Keep unsupported batch modes fail-closed. Input journaling alone does not complete this row. |
| Active differential-drive goal/trajectory | **Done for the supported kinematic profile.** Resolved forward/backward direction, turn/forward phase, path progress and TPI state resume in a fresh process; tests cover mid-turn and mid-forward checkpoints with asymmetric linear deceleration. | Differential velocity mode, custom controllers and batch execution remain unsupported. |
| World construction, configuration and assets | **Done for the supported kinematic profile.** PBF reconstructs declared URDF Agents and physics-off primitive `SimObject` instances from construction/state and verifies Agent asset hashes. Effective `AgentSpawnParams` from the supported URDF factory and `SimObjectSpawnParams` from the primitive factory supply their construction data; changing execution state is stored separately. | The earlier proof still uses hard-coded `_make_sim()`; arbitrary worlds, mesh/URDF objects and other settings are not covered. |
| Collision active pairs and check cadence | **TODO: excluded from the proof.** The kinematic profile rejects enabled collision checks; geometry alone does not preserve sampled collision history. | For a collision-relevant profile, inventory detection method, margin, ignore/static settings, check frequency and last-check time, active pairs and their observations, moved-object tracking and collision count. Decide between (a) restoring event/count continuity and (b) an explicitly documented reset-on-resume mode that performs an initial full check and treats existing contacts as new observations. In mode (b), `COLLISION_STARTED`/`COLLISION_ENDED` and counts can differ from an uninterrupted run; never claim equivalent event history. Test persistent overlap, separation after restore, skipped checks and callbacks before enabling either mode. |
| Joint position and active target/interpolation | **Done for all joints in the supported kinematic profile.** Every URDF joint position, target, motion limit, signed speed and active TPI phase is captured and restored. A fresh-process mobile-arm continuation with an asymmetric ramp matches the uninterrupted trajectory. | `get_joint_state()` still reports `0.0` velocity for kinematic joints; physics motors and other joint execution models are not covered. |
| Attachment parent link and relative transform | **Done for the supported kinematic box scenario.** Parent name, link name and offset are restored at CP3; the same attachment handler accepts supported primitives. | Physics constraints, arbitrary attachment graphs and non-box attachment acceptance are not covered. |
| Live entity roster and construction data | **Done for standard physics-off primitives.** Checkpoints reconstruct independent box/sphere/cylinder visual and collision shapes, pose and presence. The box scenario also proves detach/delete continuation. A primitive-only world can record and restore without any Agent. `validate_checkpoint()` accepts changing primitive rosters after validating each present object's construction/state, but requires the Agent roster to match the initial manifest exactly. | Agent spawn **and removal** after recording starts are not yet supported: the manifest lacks later Agent construction, and exact Agent-roster validation rejects both changes. A checkpoint's primitive roster is not cross-checked against the ordered spawn/delete input journal; define whether that consistency check is needed for lifecycle replay. Mesh/URDF objects, custom subclasses and their execution state need declared construction contracts. |
| Common entity construction capture and subclass extension | **Partial.** Supported URDF Agents now retain effective `AgentSpawnParams`; standard primitive `SimObject` instances retain effective `SimObjectSpawnParams`. The kinematic profile reads those parameters for construction and rejects unsupported URDF options such as non-default scaling. Runtime spawn encoding is still an optional profile method, `spawn_record`, discovered with `getattr`; profiles without it reject a recorded spawn. | Normalize effective `SpawnParams` across remaining public Agent/SimObject factory paths, URDF/SDF objects, and custom subclasses only when their reconstruction fields and runtime state are explicitly supported. Define a common post-factory construction-state contract and a natural versioned subclass extension point; avoid serializing arbitrary instance fields. Define an explicit contract or handler registration for optional spawn support instead of leaving it implicit in `getattr`. |
| Durable entity identity and ordered spawn/delete | **Done for this artifact profile.** Unique stable names map to new runtime IDs; ordered spawn/delete inputs are recorded. | General cross-run identity and exact re-execution of arbitrary lifecycle histories remain open. |
| External-driver stage and later inputs | **Partial.** The example saves/restores its stage and observation flags through a required `scenario` data record, so its existing callbacks continue after CP1/CP3. | Arbitrary application state and replay of later recorded inputs under an absent application are not supported. |
| Effective PBF input history and timing | **Partial.** Fleet API `navigate`, `joint_command`, `stop`, `attach` and the supported primitive spawn/remove path are recorded with phase, step and order. | Add a single accepted-operation recording boundary for supported direct calls such as `Agent.set_path()`, joint-target variants and `set_joint_motion_profile()`. Preserve effective arguments, step/phase/order and available source/command ID; record Fleet API delegation once, and preserve the identity of aggregate joint commands rather than emitting misleading duplicates for every internal setter call. Test both direct and Fleet API paths and ensure rejected attempts are not recorded as accepted. `execute_action`, other APIs and general input re-execution remain separate coverage decisions; this work does not make batch execution checkpointable. |
| Recording failure isolation / best-effort capture | **Partial.** #59 isolates runtime recorder failures from normal spawn/step operations, closes the recorder and marks the artifact incomplete; restore rejects incomplete artifacts. A plugin registered after recording starts fails the next capture, while simulation continues. | Existing simulation or Agent plugins are rejected by `KinematicProfile` at recording setup; the caller must handle that exception to continue the run without recording. A preflight/status API for unsupported profiles has not been designed. Recovery of valid records after one provider fails, per-owner/per-step gap reporting and a policy for partially usable artifacts remain open; do not advertise failed capture as a restorable checkpoint. |
| Randomness, external clocks and integration state | **TODO: excluded from the proof.** | Identify seeds/generator state, timing and ROS/RMF or other external inputs when a concrete continuation depends on them. |
| Action queue and in-progress Action | **TODO: not exercised in the manipulation scenario.** Other examples can queue Actions. | Investigate queue/order, phase, sub-action progress, targets, timers and outcomes before mid-Action restore. |
| Devices/plugins and physics-engine state | **TODO: excluded from the proof.** The built-in kinematic profile rejects simulation and Agent plugins instead of capturing their state. | Inventory plugin construction/configuration, mutable execution state, callback registration and scheduling phase only for a concrete supported profile; Agent/controller restore does not imply arbitrary Python or engine-state serialization. Evaluate elevator and conveyor execution-state handlers when those devices have concrete behavior and continuation tests. |

Progress should be marked against a concrete supported profile and a passing
fresh-process restore test, rather than marking a row complete merely because
its current value can be observed. Recorded-result playback is a separate
capability and is not tracked by this checklist.

For failure isolation, distinguish an invalid profile or output setting found
before recording starts from a new unsupported state encountered during a run.
The latter includes a runtime spawn outside the declared profile: the object
operation should retain its normal PBF outcome, while recording reports the
failure and does not advertise an incomplete checkpoint as usable. #59 keeps
simulation running but ends that recording. Retaining unrelated valid records
and defining partial-artifact use remain future decisions.
