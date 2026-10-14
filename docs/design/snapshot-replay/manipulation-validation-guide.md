# Inspect the manipulation state scenario

```{warning}
Development validation only. The recording, checkpoint/restore and playback
commands below exercise one declared scenario and profile. They are not ready
for general use with ordinary PBF examples or custom simulations; do not rely
on their artifacts for production runs.
```

This example began as a checkpoint-state inventory. Its default invocation
still runs that observation-only scenario. With `--record`, it now exercises
the supported Snapshot / Restore / Playback V1 profile.
It uses existing PBF APIs to create a box during a run, move a mobile
manipulator, attach the box to its end-effector link, move a joint while
carrying it, detach the box, and delete it. Attachment uses an explicit
link-relative offset; it is not a contact-validated grasp. Collision checks
are disabled in this state-discovery scenario.
The existing `mobile_manipulator_demo.py` uses an Action chain for a more
natural application-level sequence. This example deliberately issues direct
commands and keeps its stage in the external driver, so the two ownership
boundaries can be compared. An external driver could also queue Actions; an
action-based state-discovery counterpart is a follow-up, not part of this
example.

Run the deterministic headless scenario:

```bash
python -m pybullet_fleet.examples.validation.manipulation_state_scenario
```

To watch it in PyBullet GUI at normal simulated speed:

```bash
python -m pybullet_fleet.examples.validation.manipulation_state_scenario --gui
```

`--rtf 2` requests 2× viewing speed. The GUI closes when the finite scenario
run ends. The program prints the completed step and simulation time for CP1
(base moving), CP2 (joint moving), CP3 (attached box and joint moving), CP4
(detached box), and the before-spawn and after-delete states. Without
`--record`, these are in-memory observations.

## Record, restart and play back the supported profile

Use the ordinary simulation loop with opt-in recording. Omitting a directory
creates a unique path under `recordings/` and prints it:

```bash
python -m pybullet_fleet.examples.validation.manipulation_state_scenario --record
```

You can also choose a directory and watch the run:

```bash
python -m pybullet_fleet.examples.validation.manipulation_state_scenario --record my-run --gui
```

After that process has exited, restart from the latest completed checkpoint
at or before the requested simulation time. CP1 is step 3 / 0.3 s and CP3 is
step 25 / 2.5 s in this omni profile:

```bash
python -m pybullet_fleet.examples.validation.manipulation_state_scenario --restore my-run --time 2.5 --gui
```

Playback reads saved result frames without stepping the controller or physics:

```bash
python -m pybullet_fleet.examples.validation.manipulation_state_scenario --playback my-run
python -m pybullet_fleet.examples.validation.manipulation_state_scenario --playback my-run --gui --rtf 0.5
```

## What PBF and this scenario each save

`_ScenarioRun` is this example's external driver state. Its event callbacks
update that state while the simulation runs; registering an event callback
does **not** itself save the callback or its data. With `--record`,
`_configure_scenario()` separately installs a `DataRecord("scenario", 1, ...)`
capture callback. After each completed step, PBF calls the reusable
`KinematicProfile` and that data callback to write a checkpoint and a result
frame.

| Owner | Saved data | Playback | Resume |
| --- | --- | --- | --- |
| PBF `KinematicProfile` | `sim`, `agents` and `objects`: completed clock, supported robot pose/navigation and joint execution, live primitive construction/state and attachment | `ResultPlayback` reads saved frames; GUI playback renders robot and object state without running controllers | `load_supported_simulation()` validates a checkpoint and reconstructs the supported simulation |
| This example's `DataRecord` | `records.scenario.value`: `stage`, `observations` and `lifecycle` from `_ScenarioRun` | The value is available in each saved frame for application-specific inspection; this CLI's text output and GUI do not display those fields | The registered `_restore_scenario_state()` recreates `_ScenarioRun` and re-registers the scenario's event callbacks before loading returns |

`_ScenarioRun.identity`, `trajectory`, `callback_error` and `recording_path`
are **not** saved in `records.scenario`. On resume, the box identity is rebuilt
from the restored object's current runtime ID, the trajectory starts anew,
and callback errors and the recording path start empty. `_result()` packages
the current run for this example's caller; it is not the checkpoint writer.

The artifact contains `manifest.json`, `frames.jsonl`, `inputs.jsonl` and
`checkpoints/`. The manifest declares the profile and custom record. Result
frames support playback; a complete checkpoint supplies the execution state
needed for resume. The recorder writes one checkpoint and one frame per
completed step here, plus ordered supported inputs and lifecycle operations.
If capture or an unsupported runtime spawn fails, the recorder marks the
artifact incomplete while the simulation continues. Incomplete artifacts
cannot be loaded for restore or playback.

For playback, `main()` constructs `ResultPlayback` with `KinematicProfile`.
The CLI prints step/time and box presence, or calls `play_gui()` to render
saved entities. An application can also read the custom value from a frame's
`frame["records"]["scenario"]["value"]`. Neither playback path calls
`_restore_scenario_state()` or advances the scenario's event callbacks.

For resume, `resume_scenario()` first registers the trusted local restorer for
`("scenario", 1)`, then calls `load_supported_simulation()`. PBF restores its
owned state and passes the saved custom value to `_restore_scenario_state()`.
That function installs the example's driver callbacks and stores their
`_ScenarioRun` in `_RESTORED_RUNS` keyed by the new core. `resume_scenario()`
retrieves it, calls ordinary `run_simulation()`, then uses `_result()` to
report the continued run. The artifact stores an ID and data, not executable
Python callback code; this example module must be installed in the restore
process. Runtime PBF object IDs are remapped, and the robot asset must remain
available and unchanged. `--rtf` during playback controls display pacing.

The supported profile uses the ordinary Omni pose navigation path, including
its generated approach waypoint and final orientation turn. This example uses
one built-in mobile manipulator and one box. Checkpoints include all kinematic
joint positions and configured targets, with fixed timestep and physics off.
This scenario actively drives one named arm joint. It does not save arbitrary
callbacks, plugins, Actions, custom entities, physics state or ROS/RMF state.
This scenario uses an omni controller; the current reusable kinematic profile
also supports declared differential pose navigation in its own tests. A
recorded result frame is useful for viewing; only a complete
checkpoint can resume execution. Input history here covers the supported
Fleet API commands and box spawn/remove path, not arbitrary direct mutators.

The box has a scenario-owned stable label `box-001`. PBF assigns a runtime
`object_id` and PyBullet assigns a `body_id`; neither numeric ID is promised
to match another process. `PRE_STEP` runs before the next motion step. In
`POST_STEP`, the physical state already reflects the completed step, while
the core's public step/time counters advance **after** callbacks return. The
example labels observations with the completed step and `step × 0.1 s`.

Kinematic joint state currently reports velocity `0.0` even while the joint
position changes. `get_attached_objects()` and `is_attached()` show the
relationship exists, but public APIs do not reconstruct the full parent-link
and relative-transform state from an arbitrary child. The driver knows the
link and offset it requested; the V1 profile now captures those fields for
this supported kinematic attachment. See
`docs/design/snapshot-replay/manipulation-state-scenario-evidence.md` in the
repository for the observed values and limits.
