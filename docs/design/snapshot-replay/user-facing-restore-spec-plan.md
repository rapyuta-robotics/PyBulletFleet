# User-facing supported recording and one-call restore — spec and plan

**Status:** Architecture approved; the later Human review moved fixed-arm
recording into `examples/validation/` to preserve the ordinary teaching demo.
Ordinary-example opt-in and general CLI restore remain product-goal work.

Read the [product goal and capability checklist](product-goals.md) first. This
slice advances the normal-user operation after PR #59's validation proof; it
does not replace the broader V1 goal of replay, playback and trace.

## Problem and outcome

PR #59 can record and restore a declared manipulation validation profile, but
ordinary examples cannot yet use a reusable PBF-owned profile. Fresh-process
restart still makes the application load a manifest, create a profile, select a
checkpoint, restore the core and call `run_simulation(resume=True)`. The
validation profile embeds one robot name, joint, box and scenario-specific
checks. Those steps and assumptions obscure the supported state boundary.

The desired supported operation is conceptually:

```python
# In an ordinary example, before its normal run:
sim.configure_state_recording(profile=..., output=...)
sim.run_simulation()

# In a fresh process with the required extension code installed:
sim = load_supported_simulation(artifact_dir, at_or_before=chosen_time)
sim.run_simulation()
```

The loader returns a fully initialized and restored core. It selects the
latest usable completed-step checkpoint **at or before** the requested time;
it does not silently fast-forward to that exact time. The exact API name and
default time selection remain design decisions. A returned core means the
declared PBF execution state is restorable and ready to continue, not that
arbitrary application code or policy decisions were serialized.

## Supported boundary and acceptance

Use physics-off, fixed-step kinematic execution. Reuse the implemented omni
navigation, joint, attachment and supported box-lifecycle state operations.
Compose a PBF-owned supported profile from explicit owner handlers for the
state actually exercised, rather than moving the validation scenario's class
unchanged into the library. The profile must enumerate live owners and reject
unknown controller/entity execution state or undeclared stateful callbacks,
Actions, BTs, plugins and devices. A versioned envelope alone is insufficient.
Do not serialize arbitrary instance attributes.

The fixed-arm acceptance candidate is based on
[`pick_drop_arm_demo.py`](../../../pybullet_fleet/examples/arm/pick_drop_arm_demo.py):
it uses a fixed arm, one box, joint targets, attachment and an external
callback state machine, without Actions. This differs materially from the
mobile-manipulator validation profile. The adapted validation program makes
its module globals import-safe so a fresh process can re-register the callback
with its saved stage. Give the robot and box
durable artifact keys; their runtime PBF/PyBullet IDs may differ. Check its
effective controller and collision configuration before declaring support.
The existing validation example remains the baseline for omni navigation.
Do not choose `mobile_manipulator_demo.py` as the second case for this slice:
its Action sequence would add unproven Action queue/in-progress state. During
implementation review, the Human chose a separate
`examples/validation/pick_drop_arm_recording.py` for the adapted callback,
leaving the ordinary demo unchanged. This supersedes acceptance condition 1
below; ordinary-example opt-in is an explicit remaining gap.

Acceptance conditions:

1. A fixed-arm validation program enables recording by an option and uses
   `run_simulation()`. The ordinary teaching demo remains unchanged; its
   configuration-based opt-in and general CLI restore are deferred by the
   later Human review. GUI use remains available in validation.
2. A fresh process takes the artifact through **one public PBF load call**;
   callers do not manually parse the manifest, instantiate owner profiles,
   choose a checkpoint file or apply raw state dictionaries. Lower-level APIs
   remain available for advanced use.
3. The returned simulation continues by calling ordinary
   `run_simulation()`; no `resume=True`, `step_once()` loop or scenario-specific
   restore sequence is required. Application-owned callbacks may require
   installed, explicitly declared extension code, but no hidden in-core policy.
4. The existing mobile-manipulator validation composition and the differently
   composed fixed-arm validation program each have a fresh-process restored continuation
   compared against an uninterrupted reference. Compare checkpoint state,
   every subsequent completed-step pose/joint/attachment fact, completion
   events, final state and step/time under stated tolerances.
5. Incomplete/corrupt artifacts, missing assets/handlers and unsupported live
   execution state fail before the loader claims success. An unsupported state
   found during recording marks that artifact incomplete without terminating
   an otherwise healthy simulation; recovery of partial artifacts is separate.
6. Report artifact bytes, recording overhead per step and load time for both
   supported compositions. Attribute frame and checkpoint bytes separately.
   Decide whether format optimization is needed; changing the format is **not**
   a completion condition here.

## Pre-implementation baseline (superseded by this PR)

- `StateRecorder` already owns completed-step capture, `manifest.json`,
  `frames.jsonl`, `inputs.jsonl` and checkpoint files. `RecordingProfile` is
  a Protocol with construction/capture/validate/restore. Its concrete
  `KinematicManipulationProfile` lives in `examples/validation` and assumes
  exactly one named omni mobile manipulator plus at most one named box.
- `load_supported_checkpoint()` already validates checksums and selects the
  latest earlier checkpoint. `restore_supported_simulation()` accepts a
  profile instance and state dict. The new public entry point can compose
  these operations; it should not duplicate artifact parsing or validation.
- `restore_completed_step_boundary()` marks the core resume-ready, but
  `run_simulation()` currently initializes by default and only preserves that
  state with `resume=True`. The new user operation needs a safe lifecycle rule
  for a restored core while preserving ordinary fresh-run behavior.
- The fixed-arm example uses `Agent.from_urdf()`, `SimObject.from_mesh()` and
  module-global `picked`, `step_state` and `_targets_set`. The state machine is
  external application state. PBF can save its declared data and invoke an
  installed restorer, but must not interpret these stages as PBF semantics.
  The example's direct `set_all_joints_targets()`, `attach_object()` and
  `set_pose()` paths also need an explicit coverage statement; checkpoint
  continuation is not proof that every such input has a replayable journal.

## Implementation plan

1. **Inventory both compositions.** Run the existing validation profile and
   the fixed-arm example headlessly with fixed assets and finite duration.
   Identify every live core, Agent, controller, object and external callback
   owner at two useful checkpoint boundaries: mid-joint motion and attached
   motion. Determine the arm's actual controller and collision settings.
   If its active state requires a new controller/physics/collision contract,
   return for review before broadening support.
2. **Extract reusable owner state.** Move only proven construction, joint,
   omni-navigation, box and attachment capture/validate/restore operations
   from the validation profile into versioned PBF handlers. A supported
   profile declares the applicable combination and restore dependencies;
   owner-specific logic remains with its owner or a narrow handler, and
   example stages remain outside PBF. Iterate declared live entities rather
   than coding one robot and one box into the reusable path. Keep custom
   handler registration explicit and reject absent support.
3. **Add the public load operation.** Read and verify the manifest/streams;
   resolve a trusted installed profile/handler by artifact ID and version;
   select and validate the requested earlier checkpoint; construct and
   restore the supported world, relationships and completed-step clock; then
   return a ready core. On any failure, release the partly constructed
   PyBullet client and report the unsupported owner or bad artifact. Do not
   import arbitrary Python code named by an artifact. Retain the current
   lower-level loader for advanced callers.
4. **Make normal run resume restored state.** The chosen minimal surface is
   `load_supported_simulation(...)` followed by plain `run_simulation()`.
   Only a core restored at a validated completed-step boundary is resume-ready;
   a fresh core keeps its existing initialization behavior. This uses the
   existing core lifecycle flag instead of a new wrapper or a second run API.
   The first plain run consumes that flag and avoids a second initialization.
   Explicit `resume=True` remains available for lower-level/manual restore.
   Verify simulation time, pacing/monitor time, cutoff and disconnect behavior.
5. **Adapt the fixed-arm callback in validation.** Add an opt-in record option
   and callback state provider/restorer using the generic declared extension
   contract. Keep callback decisions outside PBF. Ensure a
   fresh-process entry can load the artifact and call plain
   `run_simulation()` without reconstructing entities or stages by hand.
6. **Verify and measure.** Add focused negative tests for unknown owners,
   missing callback code, incomplete/corrupt streams, changed assets and
   partial-restore cleanup. Compare both compositions against reference runs
   across checkpoint and subsequent steps. Measure disabled/enabled step cost,
   frame/checkpoint/input bytes and load time for fixed run lengths, noting
   hardware and configuration. Run relevant source checks, docs build and
   self-review before a feature PR.

## Approved architecture and continuation decision

1. **Profile resolution and application code.** Use a trusted registry
   of installed profile/handler factories keyed by versioned IDs. The artifact
   contains IDs and data, never executable code. An example or user package
   registers its callback-state restorer before the one load call; the loader
   invokes it through a documented contract. Missing code fails closed.
2. **Core lifecycle after load.** Two choices were compared: a separate
   `run_restored_simulation()` surface would duplicate run parameters and
   lifecycle logic, while a restored core can already carry a validated
   resume-ready flag. The latter keeps the user operation to one load call
   and the ordinary run call, and does not affect fresh cores. It is selected
   for this unreleased replay API. The distinction must remain documented:
   plain `run_simulation()` resumes only on a core returned from supported
   restore (or otherwise explicitly restored at a completed-step boundary).
3. **First reusable owner coverage.** Use PBF-owned state handlers for
   the proven kinematic base/joints/box/attachment operations and a profile
   declaring their supported composition. This is not a universal
   `SimObject.save/load()` or a shared PBF/USO runtime. Confirm the fixed-arm
   example as the second acceptance case after its state inventory; if it
   exposes a material unsupported owner, return with an alternative ordinary
   example rather than silently adding Action/physics support.

The user approved these boundaries and allowed the continuation API choice
without another gate. If implementation exposes a material new controller,
physics or collision responsibility, return for Human Architecture Review.
