# Kinematic Arm Documentation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add documentation for robot arm joint control, kinematic mode, and the arm-specific demos/actions.

**Architecture:** Three documentation changes: (1) a new Tutorial 4 covering arm pick/drop demos, (2) additions to existing Tutorial 1 explaining kinematic mode (`mass=0`) and joint control, (3) architecture doc updates reflecting kinematic joint interpolation. All changes are additive — no existing content is removed or restructured.

**Tech Stack:** Sphinx/MyST Markdown docs in `docs/`

---

### Task 1: Add Tutorial 4 — Robot Arm Pick & Drop  (SERIAL)

**Files:**
- Create: `docs/examples/arm-pick-drop.md`
- Modify: `docs/examples/index.md`

**Step 1: Create `docs/examples/arm-pick-drop.md`**

Write a new tutorial page covering both arm demos. Structure:

```markdown
# Tutorial 4: Robot Arm — Joint Control & Pick/Drop

**Source files:**
- [`examples/arm/pick_drop_arm_demo.py`](https://github.com/yuokamoto/PyBulletFleet/blob/main/examples/arm/pick_drop_arm_demo.py) — low-level callback approach
- [`examples/arm/pick_drop_arm_action_demo.py`](https://github.com/yuokamoto/PyBulletFleet/blob/main/examples/arm/pick_drop_arm_action_demo.py) — action-queue approach

This tutorial demonstrates **fixed-base robot arm** simulation: controlling joints,
picking up objects at the end-effector, and dropping them at target locations.

**What you'll learn:**

- Spawning a URDF arm with `use_fixed_base=True`
- Controlling joints with `set_all_joints_targets()` and `JointAction`
- Kinematic vs physics joint control (how `mass` / `physics` affect behaviour)
- Link-level object attachment (`attach_link` parameter)
- Building a pick/drop cycle with low-level callbacks **and** with the action queue

If you haven't read [Tutorial 2 — Action System](action-system), do so first —
this tutorial builds on the `PickAction` / `DropAction` concepts introduced there.

---

## 1. Fixed-Base Arm Setup

```python
arm_agent = Agent.from_urdf(
    urdf_path="robots/arm_robot.urdf",
    pose=Pose.from_xyz(0, 0, 0),
    use_fixed_base=True,   # pin base to world
    sim_core=sim_core,
)
```

`use_fixed_base=True` prevents the arm from falling or sliding.
The base link is pinned to the world frame at the given pose.

---

## 2. Kinematic vs Physics Joint Control

How joints are driven depends on two factors: the agent's `mass` and the simulation's `physics` setting.

| `mass` | `physics` | Joint control method | Behaviour |
|--------|-----------|---------------------|-----------|
| `> 0`  | `True`    | PyBullet motor control (`setJointMotorControl2`) | Physics-accurate; torque/velocity limits from motor |
| `0.0`  | any       | Kinematic interpolation (`resetJointState`) | Smooth movement respecting URDF `<limit velocity="...">` |
| `> 0`  | `False`   | Kinematic interpolation (auto-override) | Warning logged; kinematic used because `stepSimulation()` not called |

**Kinematic mode** (`mass=0.0`) is recommended for arm demonstrations and fleet-scale
scenarios where physics fidelity is not needed. Joints move smoothly at URDF-defined
velocity limits without requiring `stepSimulation()`.

> **Note:** When `physics=False`, even `mass > 0` agents use kinematic joint control
> because motor commands have no effect without `stepSimulation()`. A warning is logged
> when this override occurs.

---

## 3. Joint Control API

```python
# Set all joints at once
arm_agent.set_all_joints_targets([1.5, 1.5, 1.5, 0.0])

# Set a single joint by index or by name
arm_agent.set_joint_target(0, 1.5)
arm_agent.set_joint_target("base_to_shoulder", 1.5)

# Check arrival
if arm_agent.are_all_joints_at_targets([1.5, 1.5, 1.5, 0.0], tolerance=0.05):
    print("Arm is in position")
```

These methods work identically regardless of kinematic/physics mode — the underlying
implementation switches transparently.

---

## 4. Link-Level Object Attachment

For arm robots, objects are attached to a specific **link** (e.g., the end-effector)
rather than the base:

```python
PICK_LINK_INDEX = p.getNumJoints(arm_agent.body_id) - 1  # last link

arm_agent.attach_object(
    box_sim,
    parent_link_index=PICK_LINK_INDEX,
    relative_pose=Pose.from_xyz(0, 0, 0.14),  # offset from link frame
)
```

The attached object follows the link as joints move.
Use `arm_agent.detach_object(box_sim)` to release.

---

## 5. Approach A — Low-Level Callback (`pick_drop_arm_demo.py`)

The callback approach uses a state machine driven by `are_all_joints_at_targets()`:

```python
def pick_drop_callback(sim_core, dt):
    if step_state == 0:
        arm_agent.set_all_joints_targets(joint_init)
        if arm_agent.are_all_joints_at_targets(joint_init, tolerance=0.05):
            advance_state()
    elif step_state == 1:
        arm_agent.set_all_joints_targets(pick_joints)
        if arm_agent.are_all_joints_at_targets(pick_joints, tolerance=0.05):
            arm_agent.attach_object(box_sim, parent_link_index=PICK_LINK_INDEX, ...)
            advance_state()
    # ... more states for drop, reverse cycle
```

This gives full control but requires manual state management.

---

## 6. Approach B — Action Queue (`pick_drop_arm_action_demo.py`)

The action-queue approach expresses the same cycle declaratively:

```python
from pybullet_fleet.action import JointAction, PickAction, DropAction, WaitAction

actions = [
    JointAction(target_joint_positions=joint_init, tolerance=0.05),
    JointAction(target_joint_positions=pick_joints, tolerance=0.05),
    PickAction(
        target_object_id=box_sim.body_id,
        use_approach=False,       # arm is already positioned by JointAction
        attach_link=PICK_LINK_INDEX,
        attach_relative_pose=Pose.from_xyz(0, 0, 0.14),
    ),
    JointAction(target_joint_positions=place_joints, tolerance=0.05),
    DropAction(drop_pose=box_place_pose, use_approach=False),
    JointAction(target_joint_positions=joint_init, tolerance=0.05),
    WaitAction(duration=0.5, action_type="idle"),
]
arm_agent.add_action_sequence(actions)
```

**Key differences from mobile-robot actions:**

| Parameter | Mobile robot | Arm robot |
|-----------|-------------|-----------|
| `use_approach` | `True` (navigate to approach waypoint) | `False` (arm is pre-positioned by `JointAction`) |
| `attach_link` | `-1` (base link) | End-effector link index |
| Movement before pick/drop | `MoveAction` (path following) | `JointAction` (joint targets) |

---

## 7. `JointAction` Reference

`JointAction` moves all joints to target positions and completes when every joint
is within `tolerance` of its target:

```python
JointAction(
    target_joint_positions=[1.5, 1.5, 1.5, 0.0],
    tolerance=0.05,       # radians; default 0.01
    max_force=500.0,      # only used in physics mode
)
```

---

## 8. Running the Demos

```bash
# Low-level callback approach
python examples/arm/pick_drop_arm_demo.py

# Action-queue approach
python examples/arm/pick_drop_arm_action_demo.py
```

Both demos support switching between kinematic and physics mode by changing the
`SimulationParams` line at the top of the file.

---

## See Also

- [Tutorial 1 — Spawning Objects](spawning-objects): `from_urdf`, `set_all_joints_targets` basics
- [Tutorial 2 — Action System](action-system): `PickAction`, `DropAction` for mobile robots
- [Architecture Overview](../architecture/overview): kinematic joint interpolation internals
```

**Step 2: Update `docs/examples/index.md`**

Add Tutorial 4 to the table, toctree, and API quick-reference:

In the table:

```markdown
| Simulate a robot arm picking and dropping objects | [Tutorial 4 — Arm Pick & Drop](arm-pick-drop) |
```

In the toctree:

```
arm-pick-drop
```

In the API Quick-Reference table:

```markdown
| `JointAction` | Tutorial 4 |
| `Agent.from_urdf` / `use_fixed_base` | Tutorials 1, 4 |
| `set_all_joints_targets` / `set_joint_target` | Tutorials 1, 4 |
| `attach_object` with `parent_link_index` | Tutorial 4 |
```

**Step 3: Commit**

```bash
git add docs/examples/arm-pick-drop.md docs/examples/index.md
git commit -m "docs: add Tutorial 4 — Robot Arm Pick & Drop"
```

---

### Task 2: Expand Tutorial 1 — Kinematic Mode & `mass` Explanation  (SERIAL, after T1)

**Files:**
- Modify: `docs/examples/spawning-objects.md`

**Step 1: Add kinematic mode explanation after the Agent parameters table (after line ~156)**

After the existing `mass` row in the Agent parameters table (`| mass | 0.0 for kinematic...`),
add a new subsection:

```markdown
### Understanding `mass` and Kinematic Mode

The `mass` parameter controls whether an object uses **kinematic** or **physics-based** movement:

| `mass` | Mode | Behaviour |
|--------|------|-----------|
| `0.0` | Kinematic | Teleport-based movement. No gravity, no collisions response. Agent is moved directly by `set_pose()` / `set_goal_pose()` / `resetJointState()`. Fastest option for fleet-scale. |
| `> 0` | Dynamic | Physics-driven. Subject to gravity, friction, collision forces. Requires `physics=True` in `SimulationParams` for full effect. |

**Kinematic mode is the default** for PyBulletFleet's fleet-scale use case.
Set `mass=0.0` (or omit it — `0.0` is the default) for agents that don't need physics.

> **For URDF robots with `use_fixed_base=True`:** PyBullet internally sets the base link
> mass to 0 when using a fixed base. If you want physics-based joint control (motor torques),
> pass an explicit `mass > 0` to the constructor. Otherwise the agent will be detected as
> kinematic and use interpolation-based joint control.

```

**Step 2: Expand Section 9 (Controlling Arm Joints) to mention kinematic mode**

Replace the note about `physics=True` at the end of Section 9:

Before:
```markdown
- `set_all_joints_targets(positions, max_force=500.0)` — sets position targets for joints
  in joint-index order. Joint control requires `physics=True` in `SimulationParams`.
```

After:
```markdown
- `set_all_joints_targets(positions, max_force=500.0)` — sets position targets for joints
  in joint-index order. Works in both physics and kinematic modes:
  - **Physics mode** (`mass > 0`, `physics=True`): uses PyBullet motor control
  - **Kinematic mode** (`mass=0.0` or `physics=False`): uses smooth interpolation
    respecting URDF velocity limits

For a dedicated arm tutorial with pick/drop, see [Tutorial 4 — Arm Pick & Drop](arm-pick-drop).
```

**Step 3: Commit**

```bash
git add docs/examples/spawning-objects.md
git commit -m "docs: add kinematic mode explanation to Tutorial 1"
```

---

### Task 3: Update Architecture Overview — Kinematic Joint Interpolation  (PARALLEL with T2)

**Files:**
- Modify: `docs/architecture/overview.md`

**Step 1: Add kinematic joint control to Agent section (after line ~154, after "Control Algorithm")**

After the existing "Control Algorithm" bullet points, add:

```markdown
**Joint Control Modes:**
- **Physics mode** (`mass > 0`, `physics=True`): `setJointMotorControl2` — PyBullet motor control with torque limits
- **Kinematic mode** (`mass=0.0` or `physics=False`): `resetJointState` with per-step interpolation — joints move at URDF `<limit velocity="...">` rates, falling back to `_KINEMATIC_JOINT_FALLBACK_VELOCITY` (2.0 rad/s) if unspecified. Mode selected once at init via `_compute_use_kinematic_joints()` and cached in `_use_kinematic_joints`.

**Key Joint Methods:**
- `set_joint_target(index, position)`: Set single joint target (transparent mode switching)
- `set_all_joints_targets(positions)`: Set all joint targets at once
- `are_all_joints_at_targets(targets, tolerance)`: Check if all joints reached targets
- `_update_kinematic_joints(dt)`: Internal per-step interpolation (called from `update()`)
```

**Step 2: Add `JointAction` to action.py section (after existing Wait description, around line ~223)**

```markdown
##### JointAction
Move all joints to target positions.

**Key Parameters:**
- `target_joint_positions`: List of target angles (radians) for all controllable joints
- `tolerance`: Completion threshold per joint (default: 0.01 rad)
- `max_force`: Motor force for physics mode (default: 500.0 N·m)

**Completion:** All joints within `tolerance` of their targets.
```

**Step 3: Commit**

```bash
git add docs/architecture/overview.md
git commit -m "docs: add kinematic joint control to architecture overview"
```

---

### Task 4: Update Benchmark Documentation — Arm Results & Performance  (PARALLEL with T2/T3)

**Files:**
- Modify: `docs/benchmarking/results.md`
- Modify: `benchmark/README.md`
- Modify: `benchmark/profiling/README.md`

**Why:** The benchmark suite has been significantly refactored (Worker + Orchestrator pattern, unified `--agents` CLI, YAML config support, `suppress_stdout()`) and new performance data exists for arm joint control (physics vs kinematic, kinematic joint cache optimization).

**Step 1: Add Arm Joint Control Performance section to `docs/benchmarking/results.md`**

Append after the "Collision Config-Based Comparison" section (before "See Also"):

```markdown
---

## Arm Joint Control Performance

**Script:** `benchmark/profiling/arm_joint_update.py --test scaling`
**Conditions:** arm_robot.urdf (4 revolute joints), fixed-base, JointAction cycling, 100 steps per count

### Physics vs Kinematic Scaling

| Arms | Joints | Physics (ms/step) | Kinematic (ms/step) | Ratio |
|------|--------|--------------------|---------------------|-------|
| 1    | 4      | 0.029              | 0.010               | 2.8×  |
| 5    | 20     | 0.082              | 0.056               | 1.5×  |
| 10   | 40     | 0.152              | 0.090               | 1.7×  |
| 25   | 100    | 0.415              | 0.253               | 1.6×  |
| 50   | 200    | 0.886              | 0.552               | 1.6×  |

**→ Kinematic mode is consistently 1.5–1.7× faster** than physics mode for joint control.
The gap comes from skipping `stepSimulation()` — kinematic mode uses `resetJointState()`
with per-step interpolation at URDF velocity limits.

### Component Breakdown (10 arms)

**Script:** `benchmark/profiling/arm_joint_update.py --test builtin --arms 10`

| Component | Physics | Kinematic |
|-----------|---------|-----------|
| agent_update | 27.0% (0.050 ms) | 96.7% (0.087 ms) |
| step_simulation | 69.6% (0.129 ms) | 0.2% (0.000 ms) |
| callbacks | 1.0% | 0.4% |
| **total** | **0.186 ms** | **0.090 ms** |

In physics mode, `stepSimulation()` dominates (70%). In kinematic mode, the physics engine
is bypassed entirely — agent_update (joint interpolation + `resetJointState`) is the sole cost.

### Kinematic Joint Cache Optimization

**Problem:** cProfile showed `p.getJointState()` consuming ~36% of kinematic update time
(called per-joint per-step to read current positions before interpolating).

**Solution:** `_kinematic_joint_positions` cache — joint positions stored in a Python dict,
initialized via batch `p.getJointStates()` at agent creation, updated after each `resetJointState()`.
`get_joint_state()` returns cached values for kinematic robots (zero PyBullet calls).

| Metric | Before cache | After cache | Improvement |
|--------|-------------|-------------|-------------|
| 50 arms step time | 0.826 ms | 0.552 ms | **1.5× faster** |
| `p.getJointState` calls/step | 200 (50 arms × 4 joints) | 0 | **eliminated** |

**→ Design decision:** Kinematic joint cache is always active for `mass=0.0` URDF robots.
The cache is invisible to callers — `get_joint_state()` API is unchanged.
```

**Step 2: Update `benchmark/README.md` CLI Reference (arm section)**

Replace the outdated arm CLI examples to reflect the current unified `--agents`/`--scenario` pattern:

In the Quick Start section, replace the arm examples:

```markdown
# Arm: single benchmark (10 arms, kinematic mode)
python benchmark/run_benchmark.py --type arm --agents 10 --duration 5 --scenario kinematic

# Arm: physics mode
python benchmark/run_benchmark.py --type arm --agents 10 --duration 5 --scenario physics

# Arm: compare physics vs kinematic
python benchmark/run_benchmark.py --type arm --compare physics kinematic

# Arm: sweep arm counts
python benchmark/run_benchmark.py --type arm --sweep 1 10 50 100
```

In the CLI Reference section at the bottom, replace the arm worker direct call example:

```markdown
# Workers (direct call — usually called by orchestrator)
python benchmark/mobile_benchmark.py --agents 1000 --duration 10 --scenario no_collision
python benchmark/arm_benchmark.py --agents 10 --duration 5 --scenario physics
```

**Step 3: Update `benchmark/profiling/README.md` arm_joint_update section**

Add or update the arm_joint_update.py entry to document the `suppress_stdout()` usage and
the kinematic cache optimization context. After the existing tool summary table, ensure:

```markdown
## Arm Joint Update Profiler (`arm_joint_update.py`)

Detailed profiling of arm robot joint update performance. Compares physics
(motor control via `setJointMotorControl2`) vs kinematic (interpolation via `resetJointState()`)
modes across varying arm counts.

PyBullet C-level inertia warnings are suppressed via `suppress_stdout()` context manager
to produce clean output.

### Analysis Methods

| Method | Command | Purpose |
|--------|---------|---------|
| Built-in Profiling | `--test builtin` (default) | Component breakdown (agent_update, step_simulation, etc.) |
| cProfile | `--test cprofile` | Function-level bottleneck analysis |
| Scaling | `--test scaling` | Physics vs kinematic step time across 1–50 arms |

### CLI Usage

`` `bash
# Component breakdown (10 arms, both modes)
python benchmark/profiling/arm_joint_update.py --arms 10 --steps 100

# Scaling analysis (1, 5, 10, 25, 50 arms)
python benchmark/profiling/arm_joint_update.py --test scaling

# cProfile (kinematic mode only)
python benchmark/profiling/arm_joint_update.py --arms 10 --test cprofile

# All analyses
python benchmark/profiling/arm_joint_update.py --arms 10 --test all
`` `

### Key Findings

- Kinematic mode is **1.5–1.7× faster** than physics mode (no `stepSimulation()` overhead)
- Joint position cache (`_kinematic_joint_positions`) eliminates `p.getJointState()` calls,
  providing an additional **1.5× speedup** for kinematic mode
- Scaling is approximately linear: 200 joints (50 arms) = 0.55 ms kinematic
```

**Step 4: Commit**

```bash
git add docs/benchmarking/results.md benchmark/README.md benchmark/profiling/README.md
git commit -m "docs: add arm joint control performance results and updated benchmark CLI"
```

---

### Task 5: Verify docs build  (SERIAL, after T1-T4)

**Step 1: Check for broken cross-references**

```bash
cd docs && python -m sphinx -b linkcheck . _build/linkcheck 2>&1 | tail -20
```

If sphinx is not available, verify manually:

```bash
grep -r "arm-pick-drop" docs/ --include="*.md"
```

Expected: references in `index.md`, `spawning-objects.md`, and `arm-pick-drop.md` itself.

**Step 2: Commit any fixes if needed**

---

## Summary

| Task | Scope | Dependencies |
|------|-------|-------------|
| T1 | New Tutorial 4 page + index update | None |
| T2 | Expand Tutorial 1 (mass/kinematic explanation) | After T1 (cross-reference) |
| T3 | Architecture overview update | PARALLEL with T2 |
| T4 | Benchmark/performance docs update | PARALLEL with T2/T3 |
| T5 | Build verification | After T1-T4 |
