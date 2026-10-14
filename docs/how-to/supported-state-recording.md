# Supported kinematic recording and restore (development)

```{warning}
Development validation only. The commands below exercise a declared,
physics-off kinematic profile through a validation program. They do not enable
general recording or restore for ordinary PBF examples or custom simulations.
Do not rely on these artifacts for production runs.
```

The kinematic artifact profile is now `pbf.kinematic.v4` (version 4). Earlier
development artifacts use older execution or object-construction schemas and
are not accepted by this loader; record a new artifact before testing restore.
The table describes **execution checkpoints** for this declared profile. The
last column uses row names from the [checkpoint capability
checklist](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/product-goals.md#checkpoint-state-capability-checklist).
“No” means the state is outside this profile and cannot be claimed restored.
Detectable undeclared live PBF state is rejected; state owned outside PBF may
not be detectable. “Conditional” requires the application code to be installed
and explicitly registered for capture and load.

| State or operation | Recorded | Restored in a fresh process | Checklist row / boundary |
| --- | --- | --- | --- |
| Completed step/time and Agent base pose, velocity, moving flag | Yes | Yes | Completed step/time and base pose, velocity and moving flag |
| Active omni or differential **pose** navigation | Yes | Yes | Active omni pose goal/trajectory; Omni final-orientation rotation phase; Active differential-drive goal/trajectory. Includes path/phase and forward acceleration/deceleration; differential direction is retained. |
| Every kinematic URDF joint | Yes | Yes | Joint position and active target/interpolation. Includes position, target, limits, signed speed, TPI phase and trajectory-free `braking` state. |
| Standard physics-off `SimObject` box, sphere or cylinder | Yes | Yes | Live entity roster and construction data. Visual and collision shapes are separate, including dimensions, color and local frame offset. |
| Kinematic attachment to a supported Agent link | Yes | Yes | Attachment parent link and relative transform. Parent/link identity and relative pose are retained. |
| Fleet API `navigate`, `joint_command`, `stop`, `attach` and primitive spawn/remove hooks | Partial | Their resulting checkpoint state is restored; earlier commands are not re-executed | Effective PBF input history and timing; Durable entity identity and ordered spawn/delete. `execute_action` and direct mutators are not journaled. |
| Application stage or other caller-defined `DataRecord` | Conditional | Conditional | Caller-defined data records; External-driver stage and later inputs. PBF cannot infer the application's continuation logic. |
| Registered `AgentManager` (with or without batch) | No | No | AgentManager construction, membership and callbacks. Manager registration is rejected even when its agents are otherwise supported. |
| Omni/differential velocity mode and batch controller execution | No | No | Omni velocity-mode execution; Batch controller execution state. A Fleet API command does not make its batch execution state checkpointable. |
| Collision active pairs, event continuity and check cadence | No | No | Collision active pairs and check cadence. |
| Action/BT queues, plugins, devices and physics-engine state | No | No | Action queue and in-progress Action; Devices/plugins and physics-engine state. |
| Mesh/URDF `SimObject`, custom subclasses or nonempty `user_data` | No | No | Common entity construction capture and subclass extension. URDF **Agents** above are a separate supported case. |
| Randomness, external clocks and ROS/RMF state | No | No | Randomness, external clocks and integration state. |

A checkpoint restores the supported state at one completed step; it does not
replay the original policy's later decisions. A recorded result frame can be
displayed, but a frame alone is not a restorable checkpoint. For the joint's
trajectory-free `braking` behavior, see {doc}`../examples/arm-pick-drop`.

Here “unobserved mutator” means a direct state-changing call whose **input is
not written to the effective-input journal**. Examples include calling
`agent.set_path(...)`, `agent.set_joint_target_by_name(...)`,
`agent.set_joint_motion_profile(...)`, `obj.set_pose(...)`, or
`agent.attach_object(...)` directly instead of using an instrumented Fleet
API command where one exists. A supported result of such a call may still
appear in the next completed-step checkpoint. The checkpoint does not recover
the call's original time, arguments or ordering, so it cannot replay the
decision itself. Direct PyBullet mutations are outside the declared profile.
The Fleet API can dispatch commands to other controller arrangements, but
this recording profile accepts only its declared per-Agent omni/differential
pose execution. It rejects a simulation with a registered batch manager at
recording setup; it cannot claim a resumable batch checkpoint merely because
the corresponding command entered through Fleet API.

The fixed-arm validation program adapts the callback from the ordinary
`pick_drop_arm_demo.py`. The ordinary teaching demo is unchanged; enabling
recording and restoring it through a general CLI is future work. During
recording, this 0.1 s timestep validation program runs the callback every step
because the normal 10 Hz scheduler phase is not checkpointed. `--record`
without a directory chooses an automatically named artifact:

```bash
python -m pybullet_fleet.examples.validation.pick_drop_arm_recording --robot arm_robot --headless --duration 8 --record
```

The printed directory contains `manifest.json`, result `frames.jsonl`,
`inputs.jsonl` and `checkpoints/`. The manifest records the declared profile,
coverage and completeness. Only a complete artifact is eligible for restore.
Checkpoint selection uses the latest completed step **at or before** the
requested simulated time. Restore in a fresh process with the same example
code installed:

```bash
python -m pybullet_fleet.examples.validation.pick_drop_arm_recording --restore recordings/state-... --time 2.0 --duration 8
```

Omit `--headless` to view either run in the GUI. Here `--duration` is the
absolute simulated-time cutoff, so the restored run continues until 8 s; it
does not run for eight additional seconds. The validation program registers its own stage
restorer on import. PBF restores the supported robot and primitive state but does not
interpret the example's pick/drop stages.

The equivalent Python operation is:

```python
from pybullet_fleet.replay import load_supported_simulation

# Import or install application code that registers required data restorers.
sim = load_supported_simulation("recordings/state-...", at_or_before=2.0, gui=True)
sim.run_simulation(duration=8.0)
```

The first plain `run_simulation()` on a successfully restored core continues
from the saved completed-step boundary without reinitializing it. Fresh cores
retain their normal initialization behavior. The artifact stores versioned
profile and data IDs; the loader resolves only trusted installed handlers and
does not import Python code named by an artifact. If required application code
is absent, or a live state owner is unsupported, restore fails closed. A
recording error marks the artifact incomplete while an otherwise healthy
simulation continues. Partial-artifact recovery is not supported.

Result playback and input re-execution are distinct operations. The built-in
kinematic profile can display any recorded supported Agent and primitive by its
artifact name, without stepping the simulation:

```python
from pybullet_fleet.replay import ResultPlayback
from pybullet_fleet.replay.kinematic_profile import KinematicProfile

with ResultPlayback("recordings/state-...", profile=KinematicProfile()) as playback:
    playback.play_gui(rtf=1.0)
```

`camera_config` may be passed to `play_gui()` for an initial view; the
camera can then be moved interactively. A general CLI GUI playback command
and effective input history for every direct mutator are still future work.
The remaining work is tracked in `docs/design/snapshot-replay/product-goals.md`.

Earlier validation procedures remain in the design records: the
[manipulation scenario](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/manipulation-validation-guide.md),
[single-omni checkpoint proof](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/omni-checkpoint-proof-guide.md),
and [initial-state input re-execution](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/navigation-reexecution-guide.md).
They document distinct experiments rather than additional steps required by
this guide.
