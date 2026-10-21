# Single-omni checkpoint proof — implementation evidence

**Usage status:** Evidence for one development proof, not a general or
production checkpoint/restore capability.

**Status:** Implemented locally for Human Review. Only the fixed one-robot
straight-navigation profile is claimed.

**Note (2026-10-19):** these values were recorded with an earlier step timing; a rerun reaches each row one step earlier. The current timing is described in "Which time a pose belongs to" in [two-phase-step.md](../../architecture/two-phase-step.md).

## Four capability levels

| Level | Evidence in this profile | Boundary |
| --- | --- | --- |
| Observable | Pose, velocity, moving flag, completed step and the controller's active goal/trajectory can be read at `S_3`. | Observation alone is not a checkpoint. |
| Serializable | A strict JSON record contains the fixed construction conditions, completed clock, Agent motion and TPI-defining values. | No arbitrary `SimObject`, Action or plugin serialization. |
| Restorable | A fresh process validates the record, constructs the fixed robot, initializes once, applies motion/controller state and restores `S_3`. | Does not replay the prefix or issue another navigation goal. |
| Resumable | `run_simulation(resume=True)` advances the loaded state through arrival using the normal step engine. | Supported only for the fixed physics-off omni profile. |

The proof uses a packaged simple-cube robot, one effective waypoint at
`(0.4, 0, 0.1)`, acceleration `1.0 m/s²`, maximum linear speed `0.8 m/s`,
`dt=0.1 s` and a completed-step checkpoint at `S_3`. The source interpreter
exits before the restore interpreter starts. The record holds the original
trajectory origin and `t0`, rather than creating a new trajectory from the
checkpoint pose. It also holds the active goal, reported velocity and moving
flag; the runtime PyBullet/PBF object IDs are not persisted.

| Boundary | Reference X / velocity X / moving | Restored X / velocity X / moving |
| --- | --- | --- |
| `S_3`, 0.3 s | 0.020 / 0.200 / true | 0.020 / 0.200 / true |
| `S_4`, 0.4 s | 0.045 / 0.300 / true | 0.045 / 0.300 / true |
| `S_8`, 0.8 s | 0.24044 / 0.56491 / true | 0.24044 / 0.56491 / true |
| `S_12`, 1.2 s | 0.38640 / 0.16491 / true | 0.38640 / 0.16491 / true |
| `S_14`, 1.4 s | 0.400 / 0.000 / false | 0.400 / 0.000 / false |

The fresh-process test compares **every** suffix step, not only these display
rows. The largest observed position difference in the inspected headless run
was about `1.4e-17 m`; tests use an explicit `1e-8` absolute pose/velocity
tolerance rather than asserting bitwise identity. A negative control that
reissues the goal at the checkpoint pose differs from the reference on the
next step. Invalid profile/version, changed construction conditions, clock
inconsistency, missing/nonfinite trajectory fields and inconsistent pose or
velocity are rejected before restoring core state.

`step_once()` evaluates controller motion at the step's starting time, then
updates elapsed time and step count after `POST_STEP`. Capture therefore runs
after `step_once()` returns. At completed `S_3`, elapsed time is `0.3 s` and
the last pose was evaluated at `0.2 s`; the next evaluation is at `0.3 s`.
The restored run starts at the same next evaluation time. A `POST_STEP`
subscriber collecting the resumed trace labels it with the completing step
because the core counter has not advanced yet. The proof does not change this
existing callback-time contract.

## Scope and findings

The necessary mutable state for this profile is smaller than a generic object
snapshot: core completed-step clock, Agent pose/reported motion, the active
single goal/path, and the original forward TPI parameters. Direction and
distance are reconstructed from the original trajectory origin and goal;
the third-party TPI object is not serialized. `run_simulation()` required an
explicit resume entry because its ordinary entry resets the clock. The core
does not own file storage or cross-process workflow.

This does **not** establish restore for the manipulation scenario in PR #55.
Joint interpolation, attachments, entity lifecycle, external-driver progress,
Actions, devices, physics, recorded-result playback and later inputs remain
separate profiles or work. It also does not guarantee cross-platform exact
identity or prove GUI/RTF timing accuracy.
