# Snapshot / Restore / Replay / Playback V1 — feature specification and plan

**Status:** Restricted validation profile merged in PR #59; broader V1 remains
under development.
The current APIs and artifacts are not recommended for general or production
use. The profile and extension boundary are still being reviewed.
Material changes to the supported profile or state ownership return for Human
Review. This applies to the whole feature theme; controller and entity-state
tests are not separate feature themes.

## Current architecture correction

Human review identified that the first implementation embedded the validation
scenario's one mobile manipulator, one box and `application_state` callback in
PBF's recording module. Before this feature is ready for review, make the PBF
recorder accept a versioned recording profile and named, versioned data-record
callbacks with a defined completed-step input and JSON-compatible output.
The recorder owns capture timing, validation of the record envelope and file
I/O; the profile owns construction, supported state capture, validation and
restore. A caller registers its own data provider, and a fresh process reads
its saved value and re-registers its callback code. The scenario stage is one
such provider, not a special `application` field in PBF.

The scenario-specific profile belongs with the manipulation validation
example. Profile state uses explicit `sim`, `agents` and `objects` sections
keyed by durable names and type/version declarations; controller state is
nested under each agent. Only the built-in omni controller, the demonstrated
mobile manipulator and the constructed box need concrete handlers now.
Unsupported live state must
remain visible as unsupported. Verify the generic recorder with a second,
different profile as well as the existing CP1/CP3 fresh-process demonstration.
This is an extension contract, not a claim that arbitrary kinematic state can
already be restored.

The correction was merged in PR #59. `RecordingProfile`
declares construction, capture, validation, restore, profile ID/version and
coverage. The common recorder stores named `DataRecord` values returned at a
completed boundary. The artifact has versioned `sim`, `agents` and `objects`
sections, with a versioned controller state inside its agent; the
Omni/mobile-manipulator/box handlers and
GUI renderer live with the validation example. A second empty-world profile
test records custom data and restores its clock without changing the writer.
Fresh-process CP1, active-turn and CP3 continuation tests remain the acceptance
checks. This proves extensibility of the writer, not support for every
kinematic entity or controller. Runtime data callback failures now mark the
artifact incomplete and stop recording while normal simulation continues;
partial-record recovery remains on the product-goal checklist.

Read the [product goals and capability checklist](product-goals.md) before
changing this scope. The goal here is a usable path through those goals, not a
claim that every checklist row will be complete in V1.

## Product goal and user operation

**Today:** PR #50 can re-execute a restricted initial state and recorded
navigate/stop inputs. PR #55 can observe a manipulation/lifecycle sequence.
PR #56 can resume one fixed straight-moving omni robot from a predetermined
checkpoint in a fresh process. A user still cannot opt into recording an
ordinary supported run, choose its completed-step checkpoint, restart the
manipulation sequence from a saved artifact, or inspect its recorded results
without running the simulation again.

**V1 outcome:** A user runs one declared supported PBF scenario normally with
recording enabled, obtains time-indexed results and selectable usable
execution checkpoints, terminates the source process, restores a selected
checkpoint in a fresh process, resumes normal simulation execution, and opens
the recorded results for step/seek playback. The artifact reports exactly
which inputs, state owners and steps it covers. A changed external policy or
runtime setting is a new continuation, not proof of identical replay.

The current supported-profile Python shape is (application callbacks and
their data restorer must be installed by the caller):

```python
sim = build_supported_simulation()  # ordinary PBF setup
sim.configure_state_recording(
    profile=KinematicProfile(allowed_callbacks=(scenario_callback,)),
    records=(DataRecord("scenario", 1, capture_scenario_state, required_for_restore=True),),
    output="run-dir",
    checkpoint_every_steps=1,
)
register_scenario_callbacks(sim, scenario_state=ScenarioState())
sim.run_simulation()  # PBF records completed steps; no user step loop

# Separate process, after the source has exited:
register_data_restorer("scenario", 1, restore_scenario_state)
sim = load_supported_simulation("run-dir", at_or_before=2.1)
sim.run_simulation()

with ResultPlayback("run-dir") as playback:
    playback.seek_step(21)
    print(playback.state, playback.events)
    playback.step()
```

`configure_state_recording()` is an opt-in setting; the output location can
be auto-named when omitted. PBF reconstructs its supported world, while the
application restores its own stage and callback code. V1 does **not** promise
that an artifact can construct arbitrary user Python applications. The
scenario CLI exposes record, restart and playback with optional GUI. The
existing video `SimulationRecorder` and
`start_recording()` are for GIF/MP4 capture; state recording needs a distinct
name.

## Capability and responsibility boundary

| Capability | V1 meaning |
| --- | --- |
| Observation | Read current PBF facts in memory; this alone cannot restart a run. |
| Recording | Write a time-indexed history of supported inputs, results, events and checkpoints. It is an operation over time, not a smaller snapshot. |
| Snapshot / checkpoint | One profile-qualified execution state sufficient to restore at its completed-step boundary. A result frame can have fewer fields; a recording can contain many checkpoints. |
| Restore | Validate the artifact, reconstruct the supported world in a fresh process, then apply its execution state. |
| Resume | Continue from the restored boundary through the ordinary `run_simulation(resume=True)` path. |
| Re-execution | Apply recorded effective inputs to an executable initial/checkpoint state. #50 already provides this for its restricted fixed-world profile; a general V1 manipulation re-execution claim requires its own coverage proof. |
| Playback | Read recorded results in time order, step or seek, without advancing PBF controllers or physics. Playback pacing is display pacing, separate from simulation RTF. |

PBF core, Agent, controller and SimObject may expose their owned state and
supported restore operations. **PBF's recording facility** should own capture
scheduling, artifact I/O, validation, checkpoint selection, restoration tools
and playback reading. These belong in the PBF package, but the simulation
core need not directly manage files or launch processes. The caller owns when
to start/stop its application, the policy it runs and any application-only
state. It should be able to restore using PBF's supported APIs/tools without
hand-writing the state reconstruction sequence. PBF must expose accepted
operations and effective parameters where V1 needs them, but it does not own
the external application's decisions.

The default proposal for this small V1 profile is a **full checkpoint after
every completed step** (`checkpoint_every_steps=1`), plus each step's result
frame and ordered events. The time at which a failure will matter is usually
unknown during recording. If every-step checkpoints prove too expensive,
measure the cost and return for a frequency/retention decision rather than
silently weakening selectable-time restart. A lower frequency is a storage
and performance option, but selecting the latest earlier checkpoint then
advancing to an arbitrary requested step requires complete intervening input
capture and re-execution; V1 does not assume that mechanism is already proven.

## Proposed supported V1 profile

Use the existing [manipulation state scenario](manipulation-validation-guide.md)
as the **acceptance sequence**, adapting rather than duplicating its setup:
spawn a box; navigate while the base is moving (CP1); move one named joint;
attach the box to a named robot link; move the attached joint while it is in
motion (CP3); detach; then remove the box. CP1 and CP3 are mandatory restore
tests, not the only checkpoint times. CP2 and CP4 remain useful observation
and regression points; under every-step recording they also have checkpoints.
The external scenario stage must resume after CP1 and CP3, or those checkpoints are not
usable for this operation. A reference run and both restored continuations
must exercise the same supported sequence.

The profile is one mobile manipulator, one supported built-in **omni**
controller, one actively driven named kinematic joint (with every joint
position and configured target captured), one constructed box, direct
supported PBF/Fleet API operations, physics off, fixed timestep and
packaged/available unchanged assets. Its navigation uses the ordinary
`set_goal_pose()` path, including the generated approach waypoint and final
orientation turn. Tests restore during forward travel and during that turn.
This extends #56's controller state path across normal goal handling, joint,
attachment and entity lifecycle. The default observation-only differential
scenario remains available; restoring differential execution is a later
controller profile.

Existing `name` values are not unique and runtime PBF `object_id`/PyBullet
body IDs are run-local. V1 therefore uses a recording-local stable entity key,
validated for uniqueness within the artifact, mapped to runtime objects in
each process. This is an artifact identity map, not a new global PBF identity
system. User-provided assets remain the user's responsibility; the artifact
records locators and a compatibility fingerprint and fails visibly if the
needed asset differs or is missing.

## Non-goals and explicit later profiles

V1 does not support arbitrary PBF examples; arbitrary controller/Action/BT
state; differential or batch navigation; arbitrary plugins, callbacks,
devices or custom SimObjects; physics-engine state; ROS/RMF/DDS state;
multi-robot execution; universal schema; USO runtime unification; delta
checkpoints; GUI editing; or warehouse evaluation. The selected scenario's
registered callback code is recreated by the application, and its mutable
stage is saved through a named, versioned data callback. An undeclared
stateful callback or plugin makes a checkpoint unsupported; V1 must fail
closed instead of silently omitting it.

The extension contract must remain possible: a future owner can declare a
state key, schema version, capture/restore operation, and restore-order
dependencies for a controller, Action, BT, plugin, callback or custom
SimObject. Restoring code itself is not implied. The V1 implementation should
define the completeness and unsupported-state rule and one concrete external
scenario-state participant; it should not introduce an unused generic
`SimObject.save/load` interface or generic plugin serializer. A later profile
can test and refine that participant contract before claiming arbitrary
extension support. Physics-off lets PyBullet engine-state restoration wait.

## End-to-end acceptance demonstration

1. Reuse the PR #55 sequence under the declared V1 profile in an ordinary
   `run_simulation()` call. Opt into state recording and persist ordered
   input/lifecycle records, every completed-step result frame and a full
   checkpoint at every completed boundary. CP1 and CP3 are selected later for
   restore tests. Show a finished manifest with coverage, assets, version,
   step/time convention and no incomplete checkpoint presented as usable.
2. Exit the source process. In a genuinely fresh process, select CP1, rebuild
   the supported world from artifact construction data plus the supported
   application entrypoint, restore the completed-step clock, live entity
   roster, robot/controller/joint state and external scenario stage, then
   continue with `run_simulation(resume=True)` through attach, detach and
   delete. Repeat from CP3 while the box is attached and the joint is moving.
3. Compare each restored boundary *before* continuation with the source and
   compare every subsequent completed-step robot pose, joint position, box
   presence/pose/attachment, step/time, terminal navigation state and ordered
   lifecycle outcomes with an uninterrupted reference run. Use declared
   numeric tolerances on one supported environment; do not claim bitwise or
   cross-platform identity. Assert box absence before spawn, attachment at
   CP3, detachment before removal and no stale live attachment after removal.
4. Open the source artifact for playback in another process. Seek CP1 and CP3,
   step through attach/detach/delete, inspect frames and ordered events, and
   prove that no controller/physics `step_once()` or application policy
   execution is used. Optional GUI display must use the same recorded data;
   `--rtf` controls display rate, not simulation execution. Playback must
   identify missing frames or intervals rather than invent intermediate state.
5. Reject incompatible profile/version, changed `dt` or asset fingerprint,
   missing required state, duplicate stable keys, unsupported active
   callbacks/plugins/Actions, and corrupt/incomplete artifacts before
   claiming restore. A failed durable write must fail the recording operation
   visibly. Document the exact coverage limits in the user-facing guide.

One V1 example command/entrypoint must let a normal user run the sequence,
record it, restart from CP1 or CP3 and play it back. Tests may use narrower
state fixtures, but cannot replace this complete demonstration.

## State and input inventory for implementation

| State/fact | Owner and current evidence | V1 capture/restore or record requirement |
| --- | --- | --- |
| Fixed timestep, physics mode, controller/robot configuration, asset identity | Core/world construction; #56 hard-codes `_make_sim()` | Persist effective supported construction and validate compatibility. Reconstruct without requiring the old process or its runtime IDs. |
| Completed step and elapsed simulation time | Core; `POST_STEP` occurs before counters advance | Capture only after a completed boundary. On restore, the next step evaluates from that boundary; distinguish callback phase/order for inputs issued inside a step. |
| Base pose/motion, active destination and omni trajectory phase | Agent/controller; #56 proves one forward phase | Capture the effective path, waypoint index, forward TPI or final-turn rotation TPI and alignment state; compare forward and turn continuations against the same ordinary goal run. |
| Joint position, target and interpolation progress | Agent; #55 shows public kinematic velocity is `0.0` while moving | Capture private execution facts through a supported Agent operation; restore target/progress and prove subsequent positions. Do not relabel reported `0.0` as measured velocity. |
| Live box construction, pose, membership and stable key | SimObject/core; #55 driver owns spawn parameters | Persist effective construction and checkpoint roster. Map artifact key to fresh runtime object on restore. |
| Parent, named link and relative attachment transform | SimObject/Agent; #55 public attachment view is incomplete | Provide a narrow supported capture/restore operation and rebuild relation after both entities exist; do not infer it from current world pose. |
| Accepted navigate/joint/attach/detach and spawn/delete operations | Fleet API plus core lifecycle paths | Record effective parameters, result/ack where available, stable targets, step, phase and order. A checkpoint's active goal does not replace the input history; no claim of complete arbitrary-input capture. |
| Scenario stage, next intended operation and any scenario timer | External application | Save via the example's required `scenario` data record, read it after PBF restore and before resuming callbacks. This is not a PBF core variable. |
| Per-step poses, joint/attachment/lifecycle results and events | PBF observations plus scenario-owned outcomes | Store immutable result frames separately from execution checkpoints and ordered events separately from sampled roster changes. |

## Architecture and implementation sequence

1. **Confirm the vertical path.** Run the PR #55 sequence with the omni
   profile and inspect its actual controller phases. Inventory active mutable
   owners at CP1/turn/CP3 and label each as PBF-owned or application-owned.
2. **Establish one supported capture contract.** Add only the core/Agent/
   controller/SimObject state operations needed by the inventory, with
   profile-qualified validation. Expose a failure-visible completed-step
   capture point **after** `step_once()` advances its counters, usable while
   ordinary `run_simulation()` runs. `POST_STEP` and `EventBus.emit()` alone
   cannot provide this contract: POST_STEP observes old counters and EventBus
   logs handler exceptions instead of propagating durable write failures.
   Keep recording scheduling and file I/O in a PBF recording component outside
   the simulation core. Prefer a narrow synchronous completed-boundary
   observer or an equivalent explicit API;
   finalize its public shape during Architecture Review.
3. **Use one versioned artifact with distinct parts.** Proposed logical parts
   are a manifest/coverage declaration, supported construction, ordered
   effective-input and lifecycle journal, per-completed-step result frames,
   and full per-step checkpoints with external scenario-state sections. Reuse
   #50's streaming writer, checksums and validation ideas where they fit;
   its `simple_cube`/`static_box` initial schema cannot be relabeled as a
   mobile-manipulator checkpoint. Write checkpoints atomically and publish
   them as usable only after successful validation. A full checkpoint is
   proposed for one robot/box; delta encoding is not required. Measure
   per-step write cost and artifact size before confirming this default.
4. **Restore in a fresh process.** Validate profile, artifact integrity,
   assets and dependencies first. Construct the core and static world;
   construct live entities with new runtime IDs; initialize once; restore
   joint/base/controller execution state, then attachment relation and core
   clock; restore scenario stage and re-register callback code; compare the
   boundary; finally call `run_simulation(resume=True)`. Test the exact order
   against CP1 and CP3. Do not replay an attach or navigate command merely
   to synthesize the checkpoint state.
5. **Playback the result stream.** Provide a read-only step/seek cursor over
   recorded frames and ordered events. A small CLI/example may render those
   frames in GUI mode, but it must never execute the original controller or
   application. Playback can share artifact validation with restore while
   keeping a distinct capability claim.
6. **Verify and simplify.** Run the fresh-process comparison and playback
   demonstrations, focused state/negative tests, repository verification for
   the final source diff, docs build, and self-review against the product
   operation. Measure artifact size and enabled-vs-disabled step/RTF cost on
   the supported scenario; report measured cost without inventing a scale
   target. Replace redundant proof code only after its user/test coverage is
   retained. Do not add another evaluation scenario.

The candidate PR boundary is (1) ordinary recording plus CP1/CP3
checkpoint/restore/resume, which yields a useful restart operation, and (2)
recorded-result playback using the same artifact, which yields an independent
viewing operation. They remain one V1 theme and one acceptance demonstration;
each PR must say what remains. Combine them if implementation is reviewable.
Do not split by omni, joint, attachment or lifecycle simply because tests
cover them separately. A material change to the approved state ownership or
unsupported-state policy returns to Human Architecture Review.

## Relationship to existing work

- **#50:** Keep its restricted initial-state/input re-execution working while
  V1 is built. Reuse writer/validation mechanisms selectively; do not force
  the new checkpoint profile into its fixed-world schema. Its public API is a
  deprecation candidate **only after** a replacement reproduces its concrete
  user operation, a migration path is documented, and Human Architecture
  Review approves the compatibility change. General manipulation
  re-execution is not claimed merely because the V1 journal exists.
- **#55:** Reuse the scenario sequence, URDF, lifecycle and CP inventory.
  Adapt the example or share small setup helpers only where this reduces
  duplication without creating a general manipulation framework.
- **#56:** Reuse the completed-step clock restore, `run_simulation(resume=True)`
  and supported omni trajectory state. Replace its hard-coded world
  reconstruction for V1 rather than adding another isolated proof.
- **SimulationRecorder:** Keep the existing video recorder distinct from
  execution-state recording. The two may run together but have separate
  completeness claims.

## Human decisions recorded before implementation

1. **Scope:** Approve the single-manipulator, direct-command, physics-off
   sequence with CP1 and CP3 as the V1 acceptance profile. Confirm that full
   manipulation re-execution is later, while #50's restricted re-execution
   remains available during V1.
2. **Controller:** Omni was approved and the manipulation route now runs
   through ordinary goal navigation, including the generated approach waypoint
   and final orientation turn. Differential restore remains a later profile.
3. **Architecture:** Approve PBF-owned state recording, artifact handling and
   restore/playback tools, implemented outside the simulation core, plus a
   narrow failure-visible completed-step capture point and profile-specific
   PBF state operations. The application owns its process and policy. Decide
   whether the synchronous observer is a public hook or internal to the
   supported recorder attachment API.
4. **Extension boundary:** Approve explicit completeness declarations and
   fail-closed handling of undeclared stateful plugins/callbacks/custom
   objects. V1 defines named, versioned data-record callbacks and uses one
   required `scenario` record without claiming arbitrary callback serialization.
5. **User surface:** Approve a Python API plus one scenario-specific CLI with
   optional GUI for record/restart/playback. Prefer a simple recording config
   (`record=True` plus optional output/frequency) over mandatory `with` usage.
   Exact names, directory naming, checkpoint selection syntax and display
   pacing can be settled during implementation without changing the
   capability boundary.
6. **Checkpoint cadence:** Approve full checkpoints every completed step for
   the small V1 profile, subject to measured storage and step-time cost.
   Changing that default or relying on re-execution between sparse
   checkpoints would require explicit review of the user-visible time
   selection behavior.

`--time` selecting the latest checkpoint at or before the requested time is
the product direction. Whether V1 continues from that checkpoint directly or
re-executes recorded inputs to exactly the requested time remains open; the
CP1/CP3 acceptance demonstration tests two representative choices from the
per-step checkpoint sequence. The former is the proposed V1 behavior, and any
stronger claim needs a separate test of the intervening input coverage.

## Implementation evidence for Human Review

The existing manipulation scenario now runs normally with `--record`, then
supports `--restore DIR --time T` and `--playback DIR` in a fresh process.
The recorded omni route reached CP1 at step 3 / 0.3 s, had its active final
turn at steps 14–17 (the test restores at step 14), reached CP3 at step 25 /
2.5 s, and removed the box at step 33. It uses the standard Fleet `navigate()` call with no
checkpoint-only navigation flags. Separate-process tests compared the
selected checkpoint before continuation and every subsequent sample with an
uninterrupted reference. Playback seeks and steps through 50 recorded frames
without calling `step_once()`. The default observation-only differential
invocation remains available.

On this machine, a five-run median for the 50-step headless route was about
10.2 ms without recording and 44.7 ms with per-step full checkpoints, a 4.4×
ratio; the artifact was about 193 KiB. This is a one-robot profile measurement,
not a fleet-scale performance claim. The final core verification passed 1843
tests with 12 skipped and 79.81% coverage; the documentation build passed.
GUI playback is offered but was not exercised in the headless test environment.

The completeness boundary remains explicit: direct unobserved mutators, omni
velocity-mode commands, differential navigation, Actions, plugins, arbitrary
callbacks or objects, physics and ROS/RMF state are outside the artifact. Supported input history
does not yet constitute general manipulation input re-execution. The earlier
PR #50 re-execution API remains separate.
