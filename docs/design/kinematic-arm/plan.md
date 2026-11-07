# Kinematic Robot Arm Implementation Plan

> **Note:** This is the original design record. The single-joint APIs have since
> changed: `set_joint_target()`, `get_joint_state()` and `is_joint_at_target()`
> take `joint: Union[int, str]` (a joint index or a URDF joint name) instead of
> `joint_index: int`, and kinematic joint state lives in `Agent._joints`. See
> `docs/architecture/overview.md` for the current API.

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Enable arm joint control for kinematic robots (`mass=0`) using `resetJointState()` with smooth per-step interpolation, so `set_joint_target()` and `JointAction` work transparently in both physics and kinematic modes.

**Architecture:** `set_joint_target()` branches on `self.is_kinematic`: physics robots use existing `setJointMotorControl2()`; kinematic robots store targets in a dict. A new private method `_update_kinematic_joints(dt)` is called from `update()` each step, interpolating joints toward targets at URDF velocity limits using `resetJointState()`. All `getLinkState()` calls add `computeForwardKinematics=1` for correct link poses.

**Tech Stack:** PyBullet (`resetJointState`, `getLinkState`), pytest parametrize

---

## Task Dependency Map

```
T1 (getLinkState fix) ───────────────────────────────┐
T2 (core impl: dict + set_joint_target + update) ────┤──► T4 (parametrize tests) ──► T5 (action integration tests) ──┐
T3 (PARALLEL: roadmap update)                         │                                                               ├──► T7 (verify all)
                                                      └──► T6 (benchmark & perf docs) ────────────────────────────────┘
```

- **T1, T2**: PARALLEL (independent changes)
- **T3**: PARALLEL (standalone — no code depends on it)
- **T4**: SERIAL after T2 (tests need the implementation)
- **T5**: SERIAL after T4 (builds on test infrastructure)
- **T6**: PARALLEL with T4/T5 (documentation-only, no code dependencies)
- **T7**: SERIAL final verification (after all tasks)

---

### Task 1: Fix `getLinkState()` calls — add `computeForwardKinematics=1` (PARALLEL)

**Files:**
- Modify: `pybullet_fleet/agent.py:1799`
- Modify: `pybullet_fleet/action.py:569`
- Modify: `pybullet_fleet/sim_object.py:965`
- Modify: `pybullet_fleet/collision_visualizer.py:176`

**Why:** `resetJointState()` (used in T2) does not trigger forward kinematics automatically. Without `computeForwardKinematics=1`, `getLinkState()` returns stale link positions. This flag is harmless for physics mode (negligible overhead) so we apply it universally.

**Step 1: Apply all 4 fixes**

In `pybullet_fleet/agent.py`, `update_attached_objects_kinematics()` (~line 1799):
```python
# BEFORE:
link_state = p.getLinkState(self.body_id, obj._attached_link_index, physicsClientId=self._pid)
# AFTER:
link_state = p.getLinkState(self.body_id, obj._attached_link_index, computeForwardKinematics=1, physicsClientId=self._pid)
```

In `pybullet_fleet/action.py`, PickAction.execute() (~line 569):
```python
# BEFORE:
link_state = p.getLinkState(agent.body_id, self._attach_link_index)
# AFTER:
link_state = p.getLinkState(agent.body_id, self._attach_link_index, computeForwardKinematics=1)
```

In `pybullet_fleet/sim_object.py`, attach_object() (~line 965):
```python
# BEFORE:
link_state = p.getLinkState(self.body_id, parent_link_index, physicsClientId=self._pid)
# AFTER:
link_state = p.getLinkState(self.body_id, parent_link_index, computeForwardKinematics=1, physicsClientId=self._pid)
```

In `pybullet_fleet/collision_visualizer.py` (~line 176):
```python
# BEFORE:
link_state = p.getLinkState(body_id, link_index)
# AFTER:
link_state = p.getLinkState(body_id, link_index, computeForwardKinematics=1)
```

**Step 2: Run existing tests to verify no regressions**

```bash
pytest tests/ -x -q
```
Expected: All 555 tests pass. The flag adds no behavioral change for physics mode.

**Step 3: Commit**

```bash
git add pybullet_fleet/agent.py pybullet_fleet/action.py pybullet_fleet/sim_object.py pybullet_fleet/collision_visualizer.py
git commit -m "fix: add computeForwardKinematics=1 to all getLinkState calls

Required for correct link poses after resetJointState() in kinematic mode.
Harmless for physics mode (negligible overhead)."
```

---

### Task 2: Core implementation — kinematic joint interpolation (PARALLEL with T1)

**Files:**
- Modify: `pybullet_fleet/agent.py` (3 locations: `__init__`, `set_joint_target`, `update`)
- Add import: `import math` at top of `agent.py`

**Step 1: Add `_kinematic_joint_targets` dict to `__init__`**

In `pybullet_fleet/agent.py`, after line ~270 (after `self._action_queue` / `self._current_action`), add:

```python
        # Kinematic joint control (mass=0 robots): targets for smooth interpolation
        # Dict: {joint_index: (target_position, max_force)}
        self._kinematic_joint_targets: Dict[int, Tuple[float, float]] = {}
```

Place it after the action queue block (after `self._current_action: Optional[Action] = None`) and before `self.pickable = False`.

**Step 2: Modify `set_joint_target()` to branch on `is_kinematic`**

In `pybullet_fleet/agent.py`, replace the implementation of `set_joint_target()` (~line 1605):

```python
    def set_joint_target(self, joint_index: int, target_position: float, max_force: float = 500.0):
        """
        Set target position for a joint (for URDF robots only).

        Args:
            joint_index: Joint index (0-based)
            target_position: Target position (radians for revolute, meters for prismatic)
            max_force: Maximum force to apply

        Example::

            robot.set_joint_target(0, 1.57)  # Move first joint to 90 degrees
        """
        if not self.is_urdf_robot():
            self._log.warning("set_joint_target() only works for URDF robots")
            return

        if joint_index >= len(self.joint_info):
            self._log.warning(f"joint_index {joint_index} out of range (max: {len(self.joint_info)-1})")
            return

        if self.is_kinematic:
            # Kinematic mode (mass=0): store target for smooth interpolation
            # Actual movement happens in _update_kinematic_joints() called from update()
            self._kinematic_joint_targets[joint_index] = (target_position, max_force)
        else:
            # Physics mode (mass>0): existing motor control
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=max_force,
                physicsClientId=self._pid,
            )
```

**Step 3: Add `_update_kinematic_joints()` method**

Add this new private method in `pybullet_fleet/agent.py`, right before the `update()` method (~before line 1448). Place it in the "URDF-specific methods (joint control)" section, after `set_joints_targets()` and before `is_joint_at_target()` (roughly around line 1700 area), OR just before `update()`. The cleanest place is right before `update()`:

```python
    def _update_kinematic_joints(self, dt: float) -> None:
        """Interpolate joints toward targets for kinematic robots (mass=0).

        Called from update() each step. Mirrors the role of stepSimulation()
        for physics-mode motor control.

        Each joint moves at most ``velocity_limit * dt`` per step, where
        velocity_limit comes from the URDF ``<limit velocity="...">`` attribute.
        Falls back to 2.0 rad/s if the URDF limit is 0 or missing.
        """
        if not self._kinematic_joint_targets:
            return

        reached: list = []
        for joint_index, (target, _max_force) in self._kinematic_joint_targets.items():
            current_pos, _ = self.get_joint_state(joint_index)
            # URDF velocity limit: joint_info[joint_index][11] is maxVelocity
            max_vel = self.joint_info[joint_index][11]
            if max_vel <= 0:
                max_vel = 2.0  # fallback default
            max_step = max_vel * dt
            diff = target - current_pos
            if abs(diff) <= max_step:
                new_pos = target
                reached.append(joint_index)
            else:
                new_pos = current_pos + math.copysign(max_step, diff)
            p.resetJointState(
                self.body_id, joint_index, new_pos, physicsClientId=self._pid
            )

        for idx in reached:
            del self._kinematic_joint_targets[idx]
```

**Step 4: Modify `update()` to call `_update_kinematic_joints(dt)`**

In `pybullet_fleet/agent.py`, `update()` method (~line 1448). After `self._update_actions(dt)` and before the `if not self.use_fixed_base:` block, add:

```python
        # Process action queue first (actions may set goals/paths)
        self._update_actions(dt)

        # Kinematic joint interpolation (mirrors stepSimulation role for mass=0 URDF robots)
        if self.is_urdf_robot() and self.is_kinematic:
            self._update_kinematic_joints(dt)

        if not self.use_fixed_base:
```

**Step 5: Add `import math`**

At the top of `pybullet_fleet/agent.py`, add `import math` with the other stdlib imports:

```python
import logging
import math
from dataclasses import dataclass
```

**Step 6: Run existing tests to verify no regressions**

```bash
pytest tests/ -x -q
```
Expected: All 555 tests pass. The new code only activates for `is_kinematic=True` agents, which existing tests don't create (they default to `mass=1.0`).

**Step 7: Commit**

```bash
git add pybullet_fleet/agent.py
git commit -m "feat: kinematic joint interpolation for mass=0 robots

- set_joint_target() branches on is_kinematic:
  physics (mass>0) uses setJointMotorControl2 (unchanged)
  kinematic (mass=0) stores targets for smooth interpolation
- New _update_kinematic_joints(dt) method: interpolates per step
  using URDF velocity limits and resetJointState()
- update() calls _update_kinematic_joints() for kinematic URDF robots"
```

---

### Task 3: Update `Kinematic Robot Arm` roadmap entry (PARALLEL)

**Files:**
- Modify: `docs/roadmap.md`

**Step 1: Update the roadmap entry**

The current entry says:
> **Kinematic Robot Arm** — Articulated arm controlled via inverse kinematics (kinematics mode)

Update to reflect that kinematic arm support is now implemented (but IK is separate):
> ~~**Kinematic Robot Arm** — Articulated arm controlled via inverse kinematics (kinematics mode)~~

Remove the line from roadmap since kinematic arm is being implemented in this PR. The IK item was already added separately under Features.

**Step 2: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs: remove kinematic arm from roadmap (now implemented)"
```

---

### Task 4: Parametrize existing joint control tests (SERIAL — after T2)

**Files:**
- Modify: `tests/test_agent_core.py` — `TestAgentJointControl` class (~line 1291)

**Context:** The existing `TestAgentJointControl` class has tests that call `p.stepSimulation()` directly in loops. To parametrize for kinematic mode, we need a helper that abstracts the stepping mechanism.

**Step 1: Add `_step` helper and parametrize `_create_arm`**

Add a module-level fixture or class-level helper. The cleanest approach is to keep it as a class method since `TestAgentJointControl` already has `_create_arm`:

```python
@pytest.mark.parametrize("mass", [1.0, 0.0], ids=["physics", "kinematic"])
class TestAgentJointControl:
    """Test URDF joint state and control methods.

    arm_robot.urdf has 4 revolute joints:
        0: base_to_shoulder   (axis Z, ±π)
        1: shoulder_to_elbow  (axis X, ±2.0)
        2: elbow_to_wrist     (axis X, ±2.5)
        3: wrist_to_end       (axis Z, ±1.57)

    Parametrized with mass=1.0 (physics motor control) and mass=0.0 (kinematic
    interpolation) to verify both modes produce the same end results.
    """

    ARM_JOINT_NAMES = [
        "base_to_shoulder",
        "shoulder_to_elbow",
        "elbow_to_wrist",
        "wrist_to_end",
    ]

    def _create_arm(self, pybullet_env, mass, *, use_fixed_base=True):
        agent = Agent.from_urdf(
            urdf_path=ARM_URDF,
            pose=Pose.from_xyz(0, 0, 0.5),
            mass=mass,
            use_fixed_base=use_fixed_base,
        )
        return agent

    def _step(self, agent, mass, n=500, dt=1.0 / 240.0):
        """Advance simulation n steps using the right mechanism for the mode.

        Physics (mass>0): p.stepSimulation() drives motor control.
        Kinematic (mass=0): agent.update(dt) drives interpolation.
        """
        for _ in range(n):
            if mass == 0.0:
                agent.update(dt)
            else:
                p.stepSimulation()
```

**Step 2: Update every test that calls `_create_arm` and `p.stepSimulation()`**

Each test method needs `mass` in its signature (provided by `parametrize`). Replace:
- `self._create_arm(pybullet_env)` → `self._create_arm(pybullet_env, mass)`
- Direct `for _ in range(N): p.stepSimulation()` loops → `self._step(agent, mass, n=N)`

Tests that don't use stepping (read-only tests like `test_num_joints_matches_urdf`, `test_joint_names_match_urdf`, etc.) just need the `mass` parameter added to `_create_arm`.

Full list of methods to update:

| Test method | Has stepping loop? | Changes |
|---|---|---|
| `test_num_joints_matches_urdf` | No | `_create_arm(pybullet_env, mass)` |
| `test_joint_names_match_urdf` | No | `_create_arm(pybullet_env, mass)` |
| `test_joint_types_all_revolute` | No | `_create_arm(pybullet_env, mass)` |
| `test_is_urdf_robot` | No | `_create_arm(pybullet_env, mass)` |
| `test_mesh_agent_no_joints` | No | No change (doesn't use arm) |
| `test_initial_joint_positions_zero` | No | `_create_arm(pybullet_env, mass)` |
| `test_set_joint_target_and_simulate` | Yes (500 steps) | `_create_arm` + `_step` |
| `test_get_joint_state_by_name` | No | `_create_arm(pybullet_env, mass)` |
| `test_set_joint_target_by_name` | Yes (500 steps) | `_create_arm` + `_step` |
| `test_get_all_joints_state` | No | `_create_arm(pybullet_env, mass)` |
| `test_get_all_joints_state_by_name` | No | `_create_arm(pybullet_env, mass)` |
| `test_set_all_joints_targets` | Yes (1000 steps) | `_create_arm` + `_step` |
| `test_set_joints_targets_by_name` | Yes (1000 steps) | `_create_arm` + `_step` |
| `test_set_joints_targets_list_and_dict` | Yes (1000×2 steps) | `_create_arm` + `_step` |
| `test_mesh_agent_joint_state_returns_zero` | No | No change (doesn't use arm) |
| `test_get_joints_state_by_name_subset` | No | `_create_arm(pybullet_env, mass)` |

**Example transformation for `test_set_joint_target_and_simulate`:**

```python
    def test_set_joint_target_and_simulate(self, pybullet_env, mass):
        """set_joint_target should move joint toward target after simulation steps."""
        agent = self._create_arm(pybullet_env, mass)
        target = 0.5  # radians
        agent.set_joint_target(0, target)

        self._step(agent, mass, n=500)

        pos, _ = agent.get_joint_state(0)
        pb_pos = p.getJointState(agent.body_id, 0)[0]
        assert abs(pos - pb_pos) < 1e-6, f"get_joint_state pos={pos} != pybullet pos={pb_pos}"
        assert abs(pos - target) < 0.05, f"Joint 0 should be near {target}, got {pos}"
        assert agent.is_joint_at_target(0, target, tolerance=0.05)
```

**Example for tests that don't step but need mass:**

```python
    def test_num_joints_matches_urdf(self, pybullet_env, mass):
        """get_num_joints() should match the URDF (4 joints for arm_robot)."""
        agent = self._create_arm(pybullet_env, mass)
        assert agent.get_num_joints() == 4
```

**Step 3: Run parametrized tests**

```bash
pytest tests/test_agent_core.py::TestAgentJointControl -v
```
Expected: Each test runs twice (`[physics]` and `[kinematic]`), all pass.

**Step 4: Commit**

```bash
git add tests/test_agent_core.py
git commit -m "test: parametrize joint control tests for physics and kinematic modes

@pytest.mark.parametrize('mass', [1.0, 0.0]) runs TestAgentJointControl
in both physics (motor control) and kinematic (interpolation) modes.
_step() helper abstracts p.stepSimulation() vs agent.update(dt)."
```

---

### Task 5: Parametrize JointAction integration tests (SERIAL — after T4)

**Files:**
- Modify: `tests/test_action_integration.py` — `TestJointActionIntegration` class and helpers (~line 973)

**Context:** The current `PhysicsSimCore` and `run_arm_until_idle()` only support physics mode. We need a kinematic equivalent.

**Step 1: Add `KinematicSimCore` class**

Add alongside `PhysicsSimCore` in `tests/test_action_integration.py` (~line 977):

```python
class KinematicSimCore:
    """Minimal sim_core for kinematic (mass=0) agents.

    Does NOT call p.stepSimulation(). Joints move via agent.update() →
    _update_kinematic_joints() using resetJointState().
    """

    def __init__(self, dt: float = 1.0 / 240.0):
        self.sim_time = 0.0
        self._dt = dt
        self.sim_objects: list = []
        self._kinematic_objects: set = set()
        self._client = 0
        self._params = type("Params", (), {"physics": False})()

    @property
    def client(self):
        return self._client

    def add_object(self, obj):
        self.sim_objects.append(obj)

    def remove_object(self, obj):
        if obj in self.sim_objects:
            self.sim_objects.remove(obj)

    def _mark_object_moved(self, object_id):
        pass

    def tick(self, n: int = 1):
        """Advance sim_time without physics stepping."""
        for _ in range(n):
            self.sim_time += self._dt
```

**Step 2: Add parametrized fixture and update `create_arm_agent`**

```python
def create_arm_agent(sim_core, mass=1.0):
    """Create a fixed-base arm robot wired to *sim_core*."""
    return Agent.from_urdf(
        urdf_path=ARM_URDF,
        pose=Pose.from_xyz(0, 0, 0),
        mass=mass,
        use_fixed_base=True,
        sim_core=sim_core,
    )


@pytest.fixture(params=[
    pytest.param("physics", id="physics"),
    pytest.param("kinematic", id="kinematic"),
])
def arm_sim(request, pybullet_env):
    """Parametrized fixture providing (sim_core, mass) for both modes."""
    if request.param == "physics":
        return PhysicsSimCore(), 1.0
    else:
        return KinematicSimCore(), 0.0
```

**Step 3: Update `run_arm_until_idle` to use `agent.update()`**

The existing helper already calls `agent.update(dt)` after `sim_core.tick()`. For kinematic mode, `sim_core.tick()` just advances time (no `stepSimulation()`), and `agent.update(dt)` runs `_update_kinematic_joints()`. This should work as-is, BUT we need to make sure the agent gets the dt. The code is:

```python
def run_arm_until_idle(agent, sim_core, *, max_steps: int = 5000) -> int:
    """Run agent.update() with stepping until actions complete."""
    dt = sim_core._dt
    for step in range(max_steps):
        sim_core.tick()
        agent.update(dt)
        if agent.is_action_queue_empty():
            return step + 1
    raise AssertionError(f"Agent did not finish actions within {max_steps} steps")
```

This already works for both modes — no change needed.

**Step 4: Parametrize `TestJointActionIntegration`**

Replace `physics_sim` fixture with `arm_sim` in test methods:

```python
class TestJointActionIntegration:
    """Test JointAction execution with a real arm robot in both modes."""

    def test_reaches_target_positions(self, arm_sim):
        """JointAction moves all joints to target and completes."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        targets = [0.5, 0.3, -0.3, 0.2]

        action = JointAction(target_joint_positions=targets, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=0.05)

    def test_dict_targets_by_name(self, arm_sim):
        """JointAction accepts dict keyed by joint name."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        targets = {"base_to_shoulder": 0.4, "elbow_to_wrist": -0.5}

        action = JointAction(target_joint_positions=targets, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=0.05)

    def test_sequential_joint_actions(self, arm_sim):
        """Two JointActions execute in sequence, both reach targets."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        targets_1 = [0.5, 0.3, -0.3, 0.2]
        targets_2 = [-0.3, 0.5, 0.2, -0.1]

        action1 = JointAction(target_joint_positions=targets_1, tolerance=0.05)
        action2 = JointAction(target_joint_positions=targets_2, tolerance=0.05)
        agent.add_action_sequence([action1, action2])

        run_arm_until_idle(agent, sim_core)

        assert action1.status is ActionStatus.COMPLETED
        assert action2.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets_2, tolerance=0.05)

    def test_duration_recorded(self, arm_sim):
        """JointAction records non-zero duration after completion."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        action = JointAction(target_joint_positions=[0.3, 0.0, 0.0, 0.0], tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        duration = action.get_duration()
        assert duration is not None and duration > 0, f"Duration should be positive, got {duration}"
```

**Step 5: Add kinematic-specific test — multi-step convergence**

Add a new test that verifies kinematic mode takes multiple steps (not instant):

```python
    def test_kinematic_takes_multiple_steps(self, arm_sim):
        """In kinematic mode, joints should NOT reach target in 1 step."""
        sim_core, mass = arm_sim
        if mass > 0:
            pytest.skip("multi-step convergence test is kinematic-specific")
        agent = create_arm_agent(sim_core, mass)
        targets = [1.0, 1.0, 1.0, 0.5]

        action = JointAction(target_joint_positions=targets, tolerance=0.01)
        agent.add_action(action)

        # Single step should NOT complete
        sim_core.tick()
        agent.update(sim_core._dt)

        assert action.status is ActionStatus.IN_PROGRESS, \
            "Kinematic joint interpolation should take multiple steps"
```

**Step 6: Run tests**

```bash
pytest tests/test_action_integration.py::TestJointActionIntegration -v
```
Expected: Each test runs twice (`[physics]` and `[kinematic]`), all pass. The multi-step test skips in physics mode.

**Step 7: Commit**

```bash
git add tests/test_action_integration.py
git commit -m "test: parametrize JointAction integration tests for both modes

Add KinematicSimCore for mass=0 testing. arm_sim fixture parametrizes
physics and kinematic modes. run_arm_until_idle works for both modes.
New test: kinematic_takes_multiple_steps verifies smooth interpolation."
```

---

### Task 6: Benchmark & Performance Documentation (PARALLEL with T4/T5)

**Files:**
- Modify: `docs/benchmarking/results.md`
- Modify: `benchmark/README.md`
- Modify: `benchmark/profiling/README.md`

**Why:** Arm joint control performance data (physics vs kinematic scaling, component breakdown, kinematic joint cache optimization) needs to be documented. The benchmark CLI was also refactored (`--arms` → `--agents`, `--mode` → `--scenario`, shared `load_config()`).

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

Kinematic mode is consistently 1.5–1.7× faster than physics mode.

### Component Breakdown (10 arms)

| Component | Physics | Kinematic |
|-----------|---------|-----------|
| agent_update | 27.0% (0.050 ms) | 96.7% (0.087 ms) |
| step_simulation | 69.6% (0.129 ms) | 0.2% (0.000 ms) |
| **total** | **0.186 ms** | **0.090 ms** |

### Kinematic Joint Cache Optimization

| Metric | Before cache | After cache | Improvement |
|--------|-------------|-------------|-------------|
| 50 arms step time | 0.826 ms | 0.552 ms | 1.5× faster |
| p.getJointState calls/step | 200 | 0 | eliminated |
```

**Step 2: Update `benchmark/README.md` CLI examples**

Replace outdated `--arms`/`--mode` examples with current `--agents`/`--scenario` pattern:

```markdown
# Arm: single benchmark (kinematic mode)
python benchmark/run_benchmark.py --type arm --agents 10 --duration 5 --scenario kinematic

# Arm: compare physics vs kinematic
python benchmark/run_benchmark.py --type arm --compare physics kinematic

# Workers (direct)
python benchmark/arm_benchmark.py --agents 10 --duration 5 --scenario physics
```

**Step 3: Update `benchmark/profiling/README.md` arm_joint_update section**

Document `suppress_stdout()` usage, scaling findings, and kinematic cache context.

**Step 4: Commit**

```bash
git add docs/benchmarking/results.md benchmark/README.md benchmark/profiling/README.md
git commit -m "docs: add arm joint control performance results and benchmark CLI updates"
```

---

### Task 7: Final verification (SERIAL — last)

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: 584 tests pass, 3 skipped, 2 xfailed. No failures, no errors.

**Step 2: Verify test count**

- `TestAgentJointControl`: 17 tests × 2 modes (physics/kinematic) = 34
- `TestJointActionIntegration`: parametrized for both modes
- Total: 584 passed

**Step 3: Verify demos work (manual — GUI required)**

Test both arm demos with `mass=0.0` to confirm visual correctness:

```bash
python examples/arm/pick_drop_arm_demo.py
python examples/arm/pick_drop_arm_action_demo.py
```
Expected: Arms move smoothly in both demos with `mass=0.0`.

**Step 4: Build docs**

```bash
cd docs && sphinx-build -W -b html . _build/html 2>&1 | tail -3
```
Expected: Build succeeds with no warnings.
