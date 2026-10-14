# Run the single-omni checkpoint proof

```{warning}
Development proof only. This checkpoint format and restore procedure validate
one fixed scenario; they are not a supported general save/load facility and
should not be used for production runs.
```

This example tests a **supported, limited execution checkpoint**: one
physics-off `OmniController` robot on a single straight path with a fixed
0.1 s timestep. It saves after completed step `S_3`, ends that process,
loads the checkpoint in a new process, and resumes the normal
`run_simulation(resume=True)` loop without reissuing the goal. It is not
recorded-result playback or a checkpoint for arbitrary robots, Actions,
attachments, plugins, physics or callbacks.

This is a historical proof for a fixed scenario. The current development
profile and its one-call loader are described in the
[supported kinematic recording guide](../../how-to/supported-state-recording.md).

From the repository root, run the three commands in order:

```bash
python -m pybullet_fleet.examples.validation.omni_checkpoint_proof reference --output /tmp/omni-reference.json
python -m pybullet_fleet.examples.validation.omni_checkpoint_proof source --checkpoint /tmp/omni-checkpoint.json --output /tmp/omni-source.json
python -m pybullet_fleet.examples.validation.omni_checkpoint_proof restore --checkpoint /tmp/omni-checkpoint.json --output /tmp/omni-restored.json
```

`--output` is optional in every mode. If omitted, the program creates a
uniquely named temporary directory and prints the result path. `source` also
creates and prints a checkpoint path when `--checkpoint` is omitted; pass that
path to a later `restore` command. Explicit output paths retain the existing
write behavior.

Each command starts a separate process. The source writes one complete JSON
checkpoint; the other output files contain observed per-step states. The
automated test compares the restored suffix with the uninterrupted reference
at every completed step, including movement completion at `S_14`. The
reference mode observes `POST_STEP` while using ordinary `run_simulation()`.
The source mode deliberately calls `step_once()` three times so it can capture
only after the completed boundary; `POST_STEP` runs before the core advances
its completed-step counters, and `run_simulation()` closes the PyBullet
connection on exit. Opt-in checkpoint capture during ordinary simulation is
still a future capability. The checkpoint's `elapsed_time` is `0.3` s; the
first resumed step evaluates the
original trajectory at `0.3` s. This distinction matters because a fresh
navigation command at the checkpoint pose would restart acceleration.
The controller checkpoint includes the **active accepted goal** needed to
continue. This proof does not record the time or order of the earlier goal
command, nor arbitrary later movement or attach/detach inputs. Those require
a supported effective-input journal for re-execution; the fixed step 3 and
`moving=True` checks are assertions of this experiment, not general capture
rules.

Add `--gui` to **reference**, **source**, or **restore** to watch that mode in
PyBullet. `--rtf` sets the GUI viewing rate (default 1); headless runs use
maximum speed and make no wall-time accuracy claim. GUI commands hold their
final frame until the window is closed or Ctrl+C is pressed. In `source`, only
three steps (0.3 simulated seconds) run before the held checkpoint frame, so
use `reference` to watch the entire original movement. GUI viewing is optional
and is not part of deterministic verification.

The public core clock restore is restricted to a fresh initialized instance.
`Agent.restore_motion_state()` restores Agent-owned pose, velocity and moving
fields independently of controller type. It does not restore an arbitrary
controller's goal or execution state; this example separately restores the
supported straight-navigation state through `OmniController`.
`run_simulation()` without `resume=True` keeps its existing fresh-run behavior
and resets the clock. For a resumed run, `duration` remains an **absolute
simulation-time cutoff**, not a duration measured from the checkpoint. The
monitor's Sim Time continues from the checkpoint; its Real Time starts at the
resumed run in the new process and does not include checkpoint loading.
The simulator owns normal stepping; the example owns file I/O, process creation,
restore sequencing and comparison.

See `docs/design/snapshot-replay/checkpoint-evidence.md` in the repository for
the observed results and supported-state inventory.
