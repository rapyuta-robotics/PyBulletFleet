# Kinematic Robot Arm - Agent Specification

> **Note:** This is the original design record. The single-joint APIs have since
> changed: `set_joint_target()`, `get_joint_state()` and `is_joint_at_target()`
> take `joint: Union[int, str]` (a joint index or a URDF joint name) instead of
> `joint_index: int`, and kinematic joint state lives in `Agent._joints`. See
> `docs/architecture/overview.md` for the current API.

## Requirements

- `Agent.set_joint_target()` must work for kinematic robots (`mass=0` / `is_kinematic=True`) using `resetJointState()` with smooth per-step interpolation
- Interpolation speed governed by URDF `<limit velocity="...">` (joint_info[11]); fallback to 2.0 rad/s if 0 or missing
- All `getLinkState()` calls across the codebase must add `computeForwardKinematics=1` for correct link poses after `resetJointState()`
- No new public classes/methods — only internal implementation changes
- Existing physics-mode (mass>0) behavior must be completely unchanged

## Constraints

- `resetJointState()` resets joint velocity to 0 — this is fine for kinematic mode
- Detection uses `self.is_kinematic` (= `self.mass == 0.0`), set in `SimObject.__init__()`. Consistent with mobile robot kinematic pattern. Additionally, `_compute_use_kinematic_joints()` falls back to kinematic mode when `sim_core._params.physics` is `False` (motor control has no effect without `stepSimulation()`)
- `update()` is already called for all agents every step (including fixed-base) by `core_simulation.py`

## Approach

Mirror the physics-mode pattern:

| Step | mass>0 (physics) | mass=0 (kinematic, new) |
|------|-----------|-------------------|
| Action sets target | `setJointMotorControl2()` | Store in `_kinematic_joint_targets` dict (`Dict[int, float]`) |
| Engine moves joints | `stepSimulation()` | `_update_kinematic_joints(dt)` in `update()` |
| Action checks completion | `are_joints_at_targets()` | Same (unchanged) |

## Design

### Architecture

No new components. Changes are internal to `Agent` class with a secondary fix to `getLinkState()` calls.

### Key Components

| Component | Change | Location |
|-----------|--------|----------|
| `Agent._kinematic_joint_targets` | New dict: `{int: Tuple[float, float]}` → `{joint_index: (target_pos, max_force)}` | `pybullet_fleet/agent.py` |
| `Agent.set_joint_target()` | Branch: `is_kinematic` → store in dict, else → existing motor control | `pybullet_fleet/agent.py:1605` |
| `Agent._update_kinematic_joints()` | New private method: interpolate each joint | `pybullet_fleet/agent.py` (new) |
| `Agent.update()` | Call `_update_kinematic_joints(dt)` after `_update_actions(dt)`, before `update_attached_objects_kinematics()` | `pybullet_fleet/agent.py:1448` |
| `getLinkState()` calls (4 sites) | Add `computeForwardKinematics=1` | See list below |

### `_update_kinematic_joints(dt)` pseudocode

```python
def _update_kinematic_joints(self, dt: float) -> None:
    """Interpolate joints toward targets in kinematic mode (physics=False)."""
    if not self._kinematic_joint_targets:
        return

    reached = []
    for joint_index, (target, max_force) in self._kinematic_joint_targets.items():
        current_pos, _ = self.get_joint_state(joint_index)
        # URDF velocity limit (joint_info[11] is maxVelocity)
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
        p.resetJointState(self.body_id, joint_index, new_pos, physicsClientId=self._pid)

    for idx in reached:
        del self._kinematic_joint_targets[idx]
```

### `set_joint_target()` modification

```python
def set_joint_target(self, joint, target_position, max_force=500.0):
    # joint: index or URDF joint name
    # ... existing validation; resolves joint to joint_index ...
    if self.is_kinematic:
        # Kinematic mode (mass=0): store target for smooth interpolation
        self._kinematic_joint_targets[joint_index] = (target_position, max_force)
    else:
        # Physics mode (mass>0): existing motor control
        p.setJointMotorControl2(...)
```

### `update()` modification

```python
def update(self, dt):
    self._update_actions(dt)

    # NEW: Kinematic joint interpolation (mirrors stepSimulation role for mass=0 robots)
    if self.is_urdf_robot() and self.is_kinematic:
        self._update_kinematic_joints(dt)

    if not self.use_fixed_base:
        # ... existing navigation code ...

    if self.is_urdf_robot():
        self.update_attached_objects_kinematics()

    return True
```

### `getLinkState()` fix sites

| File | Line | Current | Fix |
|------|------|---------|-----|
| `agent.py` | ~1799 | `getLinkState(self.body_id, idx, physicsClientId=self._pid)` | Add `computeForwardKinematics=1` |
| `action.py` | ~569 | `getLinkState(agent.body_id, self._attach_link_index)` | Add `computeForwardKinematics=1` |
| `sim_object.py` | ~965 | `getLinkState(self.body_id, parent_link_index, physicsClientId=self._pid)` | Add `computeForwardKinematics=1` |
| `collision_visualizer.py` | ~176 | `getLinkState(body_id, link_index)` | Add `computeForwardKinematics=1` |

### Data Flow

```
User calls set_joint_target() or JointAction
    │
    ├── mass>0 (is_kinematic=False): setJointMotorControl2() ──► stepSimulation() moves joints
    │
    └── mass=0 (is_kinematic=True): store in _kinematic_joint_targets dict
                           │
                           ▼
                   update() each step
                           │
                           ▼
                   _update_kinematic_joints(dt)
                     - read current joint pos
                     - clamp movement to velocity_limit * dt
                     - resetJointState(new_pos)
                     - remove from dict if reached
                           │
                           ▼
                   update_attached_objects_kinematics()
                     - getLinkState(computeForwardKinematics=1)
                     - update attached object poses
```

### Code Patterns

Follow existing patterns in `agent.py`:
- Use `self._pid` for `physicsClientId` consistently
- Use `self._log` for lazy logging
- Validate with `is_urdf_robot()` guard
- Use `self.joint_info[idx][11]` for URDF velocity limits

## File References

Files the plan agent MUST read before planning:
- `pybullet_fleet/agent.py:1448-1500` - `update()` method
- `pybullet_fleet/agent.py:1605-1670` - `set_joint_target()` and related methods
- `pybullet_fleet/agent.py:1789-1810` - `update_attached_objects_kinematics()`
- `pybullet_fleet/agent.py:1700-1790` - `are_joints_at_targets()` family (unchanged)
- `pybullet_fleet/action.py:281-330` - `JointAction` (unchanged)
- `pybullet_fleet/action.py:560-580` - PickAction `getLinkState()` usage
- `pybullet_fleet/sim_object.py:960-970` - `attach_object()` `getLinkState()` usage
- `pybullet_fleet/core_simulation.py:2220-2240` - `step_once()` physics branching
- `tests/test_agent_core.py:1291-1500` - `TestAgentJointControl` class (parametrize target)
- `tests/test_action_integration.py` - existing action integration tests
- `examples/arm/pick_drop_arm_demo.py` — callback-based arm demo (uses `set_all_joints_targets` directly)
- `examples/arm/pick_drop_arm_action_demo.py` — action-based arm demo (uses JointAction/PickAction/DropAction)
- `robots/arm_robot.urdf` - URDF with velocity limits

## Success Criteria

- [ ] `mass=0` + `JointAction`: completes over multiple steps (not 1 step)
- [ ] `mass=0` + `set_joint_target()`: joint moves smoothly toward target across steps
- [ ] `mass=0` + arm demos: both `pick_drop_arm_demo.py` and `pick_drop_arm_action_demo.py` work with only `mass=0.0` added to `Agent.from_urdf()` — no other demo code changes
- [ ] `mass>0` tests: zero regressions (all 555 tests pass)
- [ ] Existing `TestAgentJointControl` tests parametrized with `@pytest.mark.parametrize("mass", [1.0, 0.0])` to run in both modes
- [ ] Step helper `_step()` abstracts `p.stepSimulation()` (physics) vs `agent.update(dt)` (kinematic)
- [ ] Kinematic-specific tests: multi-step convergence (not instant), attached object following with `computeForwardKinematics`
- [ ] `getLinkState(computeForwardKinematics=1)` applied to all 4 call sites
