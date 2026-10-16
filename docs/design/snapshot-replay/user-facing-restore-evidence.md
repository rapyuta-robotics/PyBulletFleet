# Supported kinematic restore — implementation evidence

**Status:** Implementation evidence for Human Review; this does not certify
arbitrary simulation checkpointing or production readiness.

## User operation and continuation semantics

The fixed-arm `examples/validation/pick_drop_arm_recording.py` adapts the
ordinary demo's callback and enables recording with `--record`. The ordinary
teaching demo itself remains unchanged. A new process imports the installed
validation extension, calls
`load_supported_simulation(artifact, at_or_before=...)` once, then calls plain
`run_simulation()`. The loader selects a completed-step checkpoint no later
than the requested time. Only a validated restored core is resume-ready; a
newly constructed core still initializes normally. This avoids a second
user-facing run method or wrapper, while preventing double initialization.
The application stage remains example-owned and is restored through a
versioned, explicitly registered `DataRecord` handler.
The original eight-stage callback logic is retained in the validation program.
The ordinary demo's 10 Hz registration is unchanged; at the validation program's
0.1 s timestep, recording uses every-step callback registration because the
frequency scheduler phase itself is not in the checkpoint profile. A focused
restore comparison exposed a one-step attachment mismatch when the restored
run used the 10 Hz scheduler without its phase.

PBF-owned kinematic handlers cover supported URDF Agent base and all joints,
active omni and differential pose navigation, constructed non-physics box/sphere/cylinder primitives and named
attachments. The profile enumerates live owners and rejects unknown Agents,
objects, callbacks, event handlers, plugins, BTs and managers. Active
velocity-mode navigation, collision state, physics, Actions and undeclared
application state remain outside this profile. Artifact IDs resolve trusted
installed code; the artifact cannot request an arbitrary Python import.
The documented headless CLI record/restore path was also run for the fixed-arm
example at a 1.0 s cutoff with a restore point at 0.5 s.

## Fresh-process acceptance

Focused acceptance tests run source and restored continuation in separate
Python processes. For the fixed-arm validation program, checkpoints during
unattached joint motion and attached joint motion were selected. Every
subsequent completed-step joint position, box pose and attachment presence
matched uninterrupted result frames within `1e-6`; the restored clock ended
at the same 8.0 s cutoff. For a different composition, one omni mobile arm
with a moving base, active joint and attached box was restored at step 8;
every subsequent base pose, joint position, box pose and attachment presence
matched its 2.0 s reference run within `1e-6`. These are deterministic
physics-off same-machine comparisons, not cross-platform bitwise guarantees.
After rebasing onto #66 and #72, the mobile-arm continuation also checkpoints
an in-progress shoulder ramp with asymmetric deceleration. Its restored joint
positions and signed speeds match every subsequent reference frame within
`1e-6`. The PBF-owned joint record now includes motion limits, speed and the
primitive TPI inputs with a start-time offset; the development profile schema
was `pbf.kinematic.v3` for the navigation acceptance. The current profile is
`pbf.kinematic.v4` after adding primitive construction state. Omni navigation
also records asymmetric forward deceleration; the mobile-arm fresh-process
comparison exercises it. A separate
differential-drive comparison restores both mid-turn and mid-forward checkpoints,
including a forward trajectory with asymmetric deceleration, and compares the
remaining pose and completion sequence with the uninterrupted run.
Fresh-process primitive coverage additionally restores independent display and
collision shapes for spheres, cylinders, mixed box/sphere geometry and a
visual-only sphere, including local frame offsets and a sphere attached to an
Agent base link. Unsupported shape records
are rejected before simulation construction.

Negative tests confirm that a missing required application restorer rejects
the artifact before constructing the world, a restorer error disconnects the
partially restored PyBullet client, and an undeclared callback introduced
during recording marks the artifact incomplete while the simulation reaches
its normal cutoff. Existing manipulation recording tests remain applicable.

## Recording and load measurement

Exploratory local measurements on macOS x86_64 used physics off, headless
execution, a 0.1 s timestep, `target_rtf=0`, three independent runs for each
baseline and recording condition, and three artifact loads. Source run wall
time includes process setup and model loading; these small runs are noisy and
are **not** a steady-state per-step overhead estimate. These timings and sizes
were measured before the v2 joint-ramp schema; they remain historical context,
not current overhead estimates.

| Composition | Steps | Baseline run median | Recording run median | Gross median delta / step | Frames | Checkpoints | Manifest | Inputs | Load median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed arm + box/stage | 80 | 0.907 s | 1.065 s | 1.99 ms | 104,807 B | 108,327 B | 7,395 B | 0 B | 0.023 s |
| Mobile arm + box | 20 | 0.980 s | 1.613 s | 31.65 ms | 37,906 B | 38,786 B | 2,863 B | 0 B | 0.022 s |

After the v2 schema change, one complete headless test artifact at every-step
cadence had these sizes (bytes). The mobile composition includes an active
asymmetric joint ramp. Runtime and load-time overhead were not remeasured;
the v3 navigation and v4 primitive schemas have not been measured separately.

| Composition | Steps | Frames | Checkpoints | Manifest | Inputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed arm + box/stage | 80 | 140,647 | 144,167 | 7,431 | 0 |
| Mobile arm + box/ramp | 20 | 56,397 | 57,277 | 2,901 | 0 |

The mobile result's three baseline runs ranged from 0.976 to 1.647 s, so its
gross delta cannot isolate recorder cost. In both compositions, frames and
checkpoints each store roughly the same amount of data, confirming the known
duplication for these runs. Keep the current format for this user-operation
slice. A separate, longer steady-state and fleet-size study should decide
whether to derive playback frames from checkpoints and change the format.

Asset-validation microbenchmark on this macOS development machine: 20 references
to the bundled `mobile_robot.urdf` over 100 steps took 0.136 s when hashing at
every step (1.36 ms/step), versus 0.025 s for file metadata checks (0.25
ms/step). This isolates asset validation, not total recording overhead. The
profile now hashes at recording setup, caches the verified file metadata during
per-step validation, and hashes again before restore. The cache also rehashes
if file size or modification time changes during recording.

## Remaining V1 user-visible gaps

- General playback GUI/CLI and trace views are not supplied by this operation.
- An ordinary teaching example still cannot opt into recording through its
  configuration, nor can a general CLI restore that example. This is the next
  user-visible integration gap; the current fixed-arm program is validation.
- Effective input history is incomplete for direct mutators; a checkpoint can
  continue with installed application code, but the artifact alone cannot
  re-execute every later application decision.
- The supported profile excludes differential velocity/batch execution, physics, collision,
  Action/BT, device and undeclared custom owner state. A custom entity needs
  explicit versioned capture/restore support and a fresh-process test.
- The example's stage restorer requires its code to be installed/imported in
  the new process. PBF does not serialize Python callbacks or arbitrary app
  state. The explicit callback allowlist is a developer declaration; PBF
  cannot infer whether every variable captured by an allowed callback was
  included in its required data record. This remains a completeness risk for
  custom applications, even though the tested example declares its stage.
- Frame/checkpoint duplication and validation cost need representative long
  recordings before a storage design change is justified.

These are product-goal gaps to prioritize after review, not automatic new
proof slices. The full [capability checklist](product-goals.md) tracks the
supported and excluded state by owner.
