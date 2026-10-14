# Record and re-execute fleet navigation

```{warning}
Development preview. This API is a restricted validation profile and is not
recommended for production use or general simulation recording. The broader
snapshot/restore/playback workflow is still under development.
```

`pybullet_fleet.replay` records effective Fleet API inputs and re-executes them
from the **recorded initial state** in a fresh PyBulletFleet instance. It applies
the saved inputs at their original steps and compares computed results. Explicit
`pbf_overrides` or a changed code environment allow a variant run from that same
initial state; they do not resume an intermediate state.

This is the earlier, separate initial-state re-execution profile. For the
current development checkpoint/restore profile, see the
[supported kinematic recording guide](../../how-to/supported-state-recording.md).

This is an opt-in PBF-owned v1 profile. It is not complete simulation-state
restoration, result playback, or an execution checkpoint. It has no ROS or USO
runtime dependency and makes no USO compatibility or canonical-schema claim.
The longer-term goals are to play recorded simulation data like `rosbag play`,
restore a sufficiently complete snapshot from an arbitrary recorded point and
resume simulation, then change an algorithm after restore to reproduce a
failure or compare A/B results. V1 delivers none of those end-to-end workflows:
it compares runs restarted from the initial state. Reproducing the same
continuation after restore would also require the later external inputs and
relevant conditions.

## Quick start

```python
from pybullet_fleet.replay import ReplaySession, ReplayInput, reexecute, compare

initial = {
    "world": {"entities": [
        {"entity_id": "robot-001", "name": "amr", "position": [0, 0, 0.1]}
    ]},
    "pbf": {"controller": "batch_omni", "timestep": 0.1},
}
with ReplaySession.create(initial, output="run-original") as session:
    session.step([ReplayInput.navigate("amr", (2, 0), command_id="go-1")])
    for _ in range(10):
        session.step()
    session.step([ReplayInput.stop(["amr"], command_id="stop-1")])

copy = reexecute("run-original", "run-copy")
assert compare("run-original", copy).status == "matched"
```

Output directories must not already exist. The context manager closes the owned
PyBullet connection. Without `output`, the session still enforces its execution
profile but does not construct or serialize full observations.

A representative two-robot navigation/stop example, including a partial rejection
and an explicitly changed velocity limit, is available as:

```bash
python -m pybullet_fleet.examples.replay.navigation_reexecution /tmp/replay-demo
```

It is a synthetic representative scenario, not a claimed historical failure.

## V1 limitations at a glance

The following restrictions apply to **recording and re-execution**, not to
PyBulletFleet as a whole. V1 does not record an arbitrary running simulation.

| Area | V1 boundary |
| --- | --- |
| Starting point | `ReplaySession.create()` creates a fresh, session-owned simulation from a supplied S_0 definition. It cannot attach to an existing simulation or start from an intermediate snapshot. |
| Robots and models | Only bundled `simple_cube.urdf` robots; no user URDF, joints, arbitrary `Agent` spawn parameters or other robot types. |
| Other objects | Only fixed box obstacles through the replay-only `static_box` shorthand. This creates a normal PBF `SimObject` with a box shape and `CollisionMode.STATIC`; `static_box` is not a general PBF entity type. No arbitrary shapes, floors, devices or runtime spawning. |
| Controllers and motion | `omni` or `batch_omni` planar navigation; no differential controller, physics-driven motion, other actions or behavior trees. Initial controllers are idle, with zero velocity. |
| Inputs | Only effective Fleet API `navigate` **and `stop`** requests passed through `session.step(inputs)` are recorded and dispatched. Direct agent/controller/PyBullet mutations and existing ROS/RMF input paths are not captured. |
| Timing and execution | Fixed timestep, fixed entities, headless and physics off; no mid-run configuration changes, pause/resume, GUI input, plugins or callbacks. Steps must advance through `ReplaySession.step()`. |
| Saved state | Sampled poses, reported velocities and `is_moving`, plus supported command results/events. No complete controller/action/device/physics state, so observations are not restorable checkpoints. |
| User workflows | Re-execution and comparison from S_0 only. No result player, arbitrary-time restore/resume, post-restore A/B test, generic example recorder or replay CLI. |

The session checks some persistent runtime configuration/entity changes at its
step boundary and rejects a run when it detects them. Direct commands or state
changes made outside `session.step(inputs)` are neither captured nor guaranteed
to be detected. The roadmap tracks broader capture, playback and
checkpoint/restore as separate future work.

## Supported profile

`pbf.kinematic_navigation` version 1 supports `omni` and `batch_omni`, physics off,
fixed timestep, fixed entities, headless execution, bundled `simple_cube` robots,
and static box geometry. Initial controllers are idle, with zero velocity and
no pending actions. There is one session-owned fleet manager, with entity and
batch registration order taken from the initial definition.

An entity has `entity_id`, `name`, `position: [x,y,z]`, `yaw`, and `kind`:

- `robot` (default): `model: simple_cube` (default).
- `static_box`: positive `half_extents: [x,y,z]`; no external model.

Names and entity IDs must be unique within the session. Missing entity IDs are
generated once and saved. Names remain Fleet API addresses; durable references
and comparison use entity IDs. `object_id` and PyBullet `body_id` are not saved as
persistent identities. This does not change ordinary SimObject naming semantics.

The `pbf` section of the initial definition passed to `ReplaySession.create()`
accepts the following fields. The listed defaults apply **only when that
definition omits a field**. V1 does not attach to an already running simulation
or read its configuration. It resolves the definition before constructing its
owned simulation and saves every resolved value in `initial_state.json`.

| Field | Value if omitted / meaning |
| --- | --- |
| `controller` | `omni`; alternatively `batch_omni` |
| `timestep` | `0.1` seconds, positive and fixed |
| `limits.max_linear_vel` | `2.0` m/s |
| `limits.max_linear_accel` | `5.0` m/s² |
| `limits.max_angular_vel` | `2.0` rad/s |
| `limits.max_angular_accel` | `5.0` rad/s² |
| `collision_frequency` | `null`: every step; zero disables; positive means sim Hz |
| `collision_margin` | `0.02` m, nonnegative |
| `position_tolerance` | `0.001` m for the arrival outcome |
| `angle_tolerance` | `0.001` rad for the arrival outcome |

This table is the allowlist of **execution-affecting initial `pbf` settings**
that v1 can reconstruct. It is not a limit on descriptive metadata:
`ReplaySession.create(..., provenance={...})` stores finite JSON data in the
artifact manifest, and inputs retain their source/command IDs. Provenance is
recorded for correlation but is not interpreted as executable configuration.
Other simulation parameters cannot be passed through the v1 initial definition;
silently saving an unsupported parameter would not make its behavior replayable.
For a future record mode that attaches to an existing simulation, the recorder
would need to capture and validate its *effective runtime configuration* rather
than substitute these creation defaults.

V1 fixes navigation to XY (preserving initial Z), closest-points collision,
NORMAL_2D robots, STATIC boxes, automatic initial spatial-grid sizing, no implicit
floor, no physics, no GUI/monitor, and no plugin/callback/BT execution. Other
simulation defaults remain tied to the recorded PBF source fingerprint; the
profile version and fingerprint must match for strict reproduction. Unknown
configuration fields are rejected instead of silently discarded.
Headless execution is a restriction of this tested profile, not a claim that
displaying a replay inherently changes simulation results. A GUI or interactive
example may add unrecorded user input and scheduling effects, so it needs a
separately defined capture/display boundary before being supported.

## Effective inputs and step semantics

`session.step(inputs)` accepts an ordered iterable of `ReplayInput` values.
Use `navigate()` / `stop()` helpers, or a batch payload:

With `output` set, v1 writes these complete inputs to `journal.jsonl`.
`reexecute(recording, new_output)` reads the saved input intents, creates a
fresh supported simulation and applies them at their original steps/orders;
`compare(recording, new_output)` checks the computed results. This is already
input re-execution. A timeline player that displays saved observations without
running the simulation is a separate, unimplemented result-playback feature.

```python
request = ReplayInput(
    "navigate",
    {"goals": [{"name": "amr", "position": [1, 2], "yaw": 0.0, "z": 0.1}]},
    source="ros",
    command_id="request-123",
    allowed_names=("amr",),
)
```

`allowed_names=None` means unrestricted target selection. Goal payloads also
preserve optional per-goal `command_id`. The envelope command ID identifies the
ack; repeated IDs are not deduplicated. Empty batches and normal Fleet API
rejections (unknown/duplicate/out-of-scope targets) remain valid recorded inputs.

For transition `S_k -> S_(k+1)`:

1. Input phase `(step=k, order=0..n)` runs before PRE_STEP/controller update.
2. Each complete input intent is recorded, dispatched and followed by its actual ack.
3. The core advances normally, including pose commit and collision checks.
4. After POST_STEP and counter advancement, observations describe `state_step=k+1`.

Input/event `sim_time` is `k * dt`; observation `sim_time` is `state_step * dt`.
Ordering is by integer step/phase/order, never floating timestamps alone. Paused
steps and arbitrary reinitialization are unsupported in this owned session.
Recording correctness does not depend on EventBus subscriber ordering.
The v1 timestep is fixed in `initial_state.json`: a mid-run `dt` change is
rejected and there is no per-step `dt` journal or variable-timestep replay.

Producer-specific translation (including ROS frame conversion) happens **before**
this boundary. A ROS-origin effective input can therefore replay without ROS.
`source`, `command_id`, `run_id`, and optional JSON `provenance` support later
correlation. Existing asynchronous ROS endpoints are not automatically connected
to the new boundary. ROS/DDS timing, executor scheduling, RMF decisions and
rosbag integration are outside this change. No bag is required.

## State, observations and outcomes

World state uses metres, seconds, Z-up, and `xyzw` quaternions. Initial pose is
expressed as position/yaw; the profile supplies zero initial velocities.
Observations include position, orientation, controller-reported linear/angular
velocity and `is_moving`, keyed by entity ID. Linear velocity is world-frame;
angular Z is the controller's reported yaw rate/magnitude. In particular, the
per-agent rotation path may report angular speed rather than signed yaw rate.
These fields are not a physics-solver state or a cross-controller velocity contract.

`observation_interval=N` captures S_0, every N completed steps, and the final
state (without duplicating it). Each observation contains every profile entity's
observation fields; static definitions are referenced from the initial state.
“Full” does not mean the observation alone can recreate the runtime or assets.
There are no deltas or general dirty tracking. Future checkpoint state may
share these observable fields, but the current observations are a
display/comparison view, not a saved execution state.

Collision start/end events use sorted stable-ID pairs. Events are recorded at
collision-check opportunities, not interpolated contact times. For each robot,
the latest accepted navigation/stop request is tracked: a newer request replaces
the previous outcome candidate. `arrived` means not moving and within the saved
position/yaw tolerances; `stopped` means not moving after a stop request. Outcome
events include the originating input step/order and command ID. These are profile
outcomes, not Action lifecycle or RMF task-completion events. Events are not
sampled with observations.

## Artifact and failures

| File | Meaning |
| --- | --- |
| `manifest.json` | Schema/profile version, run ID, environment/source fingerprint, asset hashes, provenance, observation interval |
| `initial_state.json` | Normalized common world definition and separate `pbf` execution configuration |
| `journal.jsonl` | Ordered command intents/results and supported events |
| `observations.jsonl` | Full profile observations |
| `completion.json` | Final step, counts, SHA-256 integrity information; only written on successful close |

JSON/JSONL are streamed synchronously with normal bounded file buffering. There
is no asynchronous pipeline or per-input fsync. A completion marker follows
successful close/flush and integrity calculation; this is not a power-loss
transaction guarantee. Readers validate both integrity and record semantics.
Do not edit an artifact while it is being validated or replayed. Hashes detect
accidental corruption, not malicious rewriting by an actor who can also rewrite
the completion metadata.

Sessions fail fast when required session-managed data cannot be recorded.
Unsupported `ReplayInput` commands, detected persistent runtime configuration
changes, writer failure and interrupted session execution cannot produce a
successful footer. A command intent without its result is incomplete; it is not
assumed to have executed or to have been rolled back.

`ReplaySession.step()` applies supported inputs before the ordinary core step,
then reads outcomes and observations afterward. The core, Agent, SimObject and
Fleet API do not switch behavior when a replay session exists. The owned core is
private: direct calls to it can run normally, but they bypass input recording.
The session's step-time checks catch some lasting configuration/entity/callback
changes, not transient or arbitrary private/raw PyBullet mutations. A completed
artifact therefore attests to the session-managed input stream and recorded
results, **not** to every possible change in the underlying simulation. Do not
use private session/core objects as an integration API. The session is a
single-owner stepping API, not a concurrent producer queue.

`ReplayError.status` / comparison status distinguish:

- `matched`: compared records agree within the declared tolerance.
- `different`: valid records differ; the first recorded difference is returned.
- `unsupported`: unknown schema/profile, unsupported execution or reproduction environment/asset mismatch.
- `incomplete`: unfinished recording or missing required execution evidence.
- `invalid`: malformed data or failed integrity/semantic validation.

## Reproduction versus changed conditions

Strict `reexecute()` checks the environment (Python/platform/dependency versions
and PBF source-content hash) and bundled asset hash. The source hash works for
wheels and dirty source trees without needing git. It does not prove that every
hardware/OS/library detail is identical. Only the same-environment supported
profile is promised; no bitwise, cross-platform or cross-backend guarantee exists.

Explicit `pbf_overrides={...}` and/or `allow_environment_change=True` produce a
variant with recorded provenance. Assets cannot silently change. `compare()`
reports changed conditions separately from behavioral differences. A `matched`
variant means matching observed behavior, not identical conditions.

Comparison checks journal payload/order/ack/events exactly and observation numeric
state within `tolerance` (default `1e-6`, SI units; orientation uses rotation angle).
Discrete state remains exact. Different observation schedules or initial identity
mappings are unsupported comparisons. With sparse observations, the report gives
the first **observed** difference, not an invented exact divergence step.

See `docs/design/snapshot-replay/evidence.md` in the repository for verification,
performance measurements, limitations and findings to feed back into USO.

## Example and command-line tooling

The bundled replay example above uses the supported session API explicitly.
There is no `pybullet-fleet replay <file>` command or generic `record: true`
switch for existing examples in v1. A command-line wrapper around
`reexecute()`/`compare()` for a supported artifact would be relatively small.
Recording an arbitrary existing example requires its initial world and every
result-affecting command to cross a supported input boundary; a config switch
alone cannot capture direct agent, controller, callback or GUI mutations.
