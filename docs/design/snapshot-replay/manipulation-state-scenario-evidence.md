# Manipulation/entity-lifecycle scenario — implementation evidence

**Note (2026-10-19):** these values were recorded with an earlier step timing; a rerun reaches each row one step earlier. The current timing is described in "Which time a pose belongs to" in [two-phase-step.md](../../architecture/two-phase-step.md).

**Usage status:** State-discovery evidence; general snapshot/restore is still
under development and not ready for production use.

**Status:** Historical evidence for the scenario merged in PR #55. That slice
did not implement checkpoint/restore or recording/playback. The later V1 proof
and its current limits are described in [feature-v1-spec-plan.md](feature-v1-spec-plan.md).

## Headless run

The scenario ran with one kinematic mobile manipulator, one dynamically
created box, a 0.1 s timestep and physics disabled. The fixed driver uses
existing Fleet API navigation/joint/attach commands and `SimObject.from_params`
and `sim.remove_object()` for box lifecycle. Its observations remain in memory.

| Point | Completed step/time | Base X | Joint position | Box state and position |
| --- | --- | ---: | ---: | --- |
| Before spawn | 0 / 0.0 s | 0.00 | 0.00 | Absent |
| CP1, base moving | 3 / 0.3 s | 0.01 | 0.00 | Independent at (1.00, 0.00, 0.70) |
| CP2, joint moving | 16 / 1.6 s | 0.60 | 0.18 | Independent at (1.00, 0.00, 0.70) |
| CP3, attached joint moving | 21 / 2.1 s | 0.60 | 0.62 | Attached; approximately (0.85, -0.273, 1.183) |
| CP4, after detach | 28 / 2.8 s | 0.60 | -0.60 | Independent; approximately (0.85, 0.265, 1.188) |
| After delete | 29 / 2.9 s | 0.60 | -0.60 | Absent |

At CP1 the base reports `is_moving=True`. At CP2 and CP3 the joint position
changes across steps, but `get_joint_state_by_name()` reports velocity `0.0`:
this is the current kinematic-joint API behavior, not measured zero motion.
The box position changes with the link during the attached arm movement.
Attachment places the box at the requested link-relative offset; this is an
explicit kinematic operation, not a contact-validated grasp. Collision checks
are disabled for this state-discovery run, so no manipulation safety claim is
made.
At CP4 the box is detached and retains its world pose. After deletion it is
absent from `sim.sim_objects`, and no robot attachment or live driver box
reference remains. No localized `remove_object()` correctness issue appeared
for the supported **detach-then-delete** path; deleting an attached object was
not exercised.

The driver maps stable label `box-001` to runtime PBF `object_id=1` in this
run. Robot `object_id=0` and box `object_id=1` are observations of this run,
not cross-run identity promises. The `OBJECT_SPAWNED` callback ran in
`PRE_STEP` with core counter 0; the box exists before step 1. The
`OBJECT_REMOVED` callback ran in `POST_STEP` after step 29's state update,
while the core counter still read 28. The driver labels this as completed
step 29 / 2.9 s. In general, `POST_STEP` callbacks see the resulting state
before the core increments `step_count` and `sim_time`; operations issued
inside that callback need an explicit phase and completed-step label. The
example's CP4 is observed immediately after detach within `POST_STEP`, so
the detach is a boundary mutation, not motion performed by step 28.

## State inventory and ownership

| State | Public PBF API | Private PBF state needed for future continuation | External driver state |
| --- | --- | --- | --- |
| Base | `Agent.get_pose()`, `FleetStateProvider.get_states_3d()` provide pose, reported velocity and movement flag | Controller goal, trajectory, start time/progress and completion flags are not exposed as a restore contract | Stage trigger and later command timing |
| Arm | `get_joint_state_by_name()` provides position and reported velocity; Fleet API issues target | Kinematic joint position cache and `_last_joint_targets` govern interpolation; joint command completion is not a checkpoint contract | Chosen joint name, target values and stage trigger |
| Box | Live `sim.sim_objects` list, `SimObject.get_pose()` and runtime `object_id` | Constructed shape/asset settings and any engine state must be reconstructible for a future checkpoint | Stable label `box-001` and runtime-ID mapping; initial spawn step and parameters |
| Attachment | `get_attached_objects()` and `is_attached()` show that a relation exists | Child `_attached_to`, `_attached_link_index`, `_attach_offset` and optional constraint state hold parent/link/transform details; no complete public read/restore contract | Requested `end_effector` link, explicit 0.07 m offset, detach trigger |
| Lifecycle | `OBJECT_SPAWNED`/`OBJECT_REMOVED` events and the live roster expose observed changes | Core registration/collision caches are maintained internally | Spawn/delete decisions and phase; a future durable journal would need ordered effective inputs |
| Time | `step_count`, `sim_time`, PRE/POST events | Counter update is after POST_STEP; future restore must define one completed-step boundary | Driver labels callback-time operations and CP observations |

The example does not serialize this inventory. Its stable label is local to the
driver. A full roster at a checkpoint can describe what exists *then*; a
sequence of sampled rosters reveals only net additions/deletions between
samples. Exact re-execution needs the ordered spawn/delete operations and
effective construction data. A spawn and deletion between two samples would
otherwise be invisible. Entity identity must remain stable independently of
PyBullet `body_id` and PBF's run-local `object_id`.

For current cross-profile checkpoint status, see the
[checkpoint state capability checklist](product-goals.md#checkpoint-state-capability-checklist).
The table above records only what this manipulation scenario observed and who
owns it.

## Implications for checkpoint scope

The [single-robot checkpoint proof](checkpoint-evidence.md) now resumes one
omni robot during straight navigation, with no dynamic objects or callbacks.
It does not restore this manipulation run. This scenario shows
what a later manipulation profile would additionally require: articulated
joint targets/interpolation state, a live object roster with construction
information, link-level attachment relation and relative transform, safe
detach/delete semantics, and any later external lifecycle inputs. The
external driver stage must be saved separately if it is expected to resume
autonomously; PBF cannot infer it from world state. A complete `PickAction` or
other action-queue restore would require still more action phase/state and is
outside this example.

The direct-command driver makes those ownership boundaries visible but needs
more explicit stage code than the existing `mobile_manipulator_demo.py` action
chain. The external driver can also queue Actions; "external driver" and
"Action-based execution" are separate choices. An action-based counterpart
is a deliberate follow-up: compare equivalent
behavior and inspect queued/current action identity, phase and sub-action
progress, targets, timers and terminal status at the same observation points.
Keep both styles during that comparison. The direct example documents what an
external driver owns; the action example would show what PBF's action system
must expose for a supported mid-action continuation. A future CLI does not
remove the need to define and capture that state.

The completed single-robot, physics-off omni navigation proof is not a claim
to restore this entire manipulation scenario. The present scenario should be
used as a later conformance case when joint, attachment and entity-lifecycle
state boundaries are deliberately added. Simulator/controller code may expose
supported state operations; external orchestration should own checkpoint
timing, files and continuation.

The work is separated into three scopes: this example only inventories state
under direct commands; the proposed initial checkpoint proof covers supported
low-level navigation state; a later Action-based manipulation investigation
must first establish its additional queue/progress state before any mid-Action
restore is claimed. None of these scopes supplies recorded-result playback.

## USO comparison and capability gaps

USO's current draft snapshot has asset identity, pose, logical
`connected_to`, illustrative joint fields and delta `new_assets` /
`removed_assets`. These concepts describe much of the **observed** state.
They do not yet specify PBF's parent-link/relative-transform attachment,
interpolating joint targets, runtime-ID mapping or the ordered lifecycle
inputs required for exact continuation. The joint field shape remains open in
USO. These are implementation findings for later cross-backend validation,
not grounds to adopt a common schema now.

No new PBF core API was required to execute the scenario. The relevant
future observability gaps are a complete public attachment-state view,
meaningful kinematic joint-motion reporting, and a defined read/restore
contract for controller/joint execution state. The run-local ID boundary and
external driver stage are separate from those PBF capabilities. Nothing here
establishes arbitrary save/load for existing demos.
