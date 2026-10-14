# Tutorial 4: Robot Arm — Joint Control & Pick/Drop

<video src="../pick_drop_arm_action.mp4" width="100%" autoplay loop muted playsinline></video>

**Source files:**
- [`examples/arm/pick_drop_arm_demo.py`](https://github.com/yuokamoto/PyBulletFleet/blob/main/pybullet_fleet/examples/arm/pick_drop_arm_demo.py) — low-level callback approach
- [`examples/arm/pick_drop_arm_action_demo.py`](https://github.com/yuokamoto/PyBulletFleet/blob/main/pybullet_fleet/examples/arm/pick_drop_arm_action_demo.py) — action-queue approach

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
> because motor commands have no effect without `stepSimulation()`. An info-level
> message is logged once (per process) when this fallback activates.

Kinematic joints have two motion patterns:

| Configuration | Movement toward a target |
|---------------|--------------------------|
| Default, or `max_velocity` only | Constant speed. The joint reaches the URDF velocity limit (or the configured limit) immediately and moves up to that limit each step. |
| `max_accel` configured with `set_joint_motion_profile()` | A two-point interpolation (TPI) trajectory accelerates, optionally cruises, then decelerates to the target. `max_decel` can set a different braking rate; otherwise it follows `max_accel`. |

For example, `arm_agent.set_joint_motion_profile("base_to_shoulder",
max_velocity=1.0, max_accel=2.0)` enables the TPI pattern for that joint.
Setting only `max_velocity` keeps the constant-speed pattern. These are
kinematic joint settings, separate from the base-navigation controller.

In the TPI pattern, changing the target while the joint moves away from it,
or placing the target inside its stopping distance, can leave no usable
trajectory yet. The joint then reduces its speed without a trajectory and
tries to plan again on the next step. Its internal `braking` flag marks this
temporary state, including the step when speed reaches zero but the target
has not been reached. This is different from the normal deceleration phase
of a valid TPI trajectory.

### Performance

Kinematic mode is **significantly faster** than physics mode for joint control because
`stepSimulation()` is skipped entirely. Joint positions are cached internally
(`_kinematic_joint_positions`), eliminating per-step `p.getJointState()` calls.

See [Benchmark Results — Arm Joint Control](../benchmarking/results) for scaling data
from a representative test environment.

---

## 3. Joint Control API

```python
# Set all joints at once
arm_agent.set_all_joints_targets([1.5, 1.5, 1.5, 0.0])

# Set a single joint by index
arm_agent.set_joint_target(joint_index=0, target_position=1.5)

# Set by joint name
arm_agent.set_joint_target_by_name("base_to_shoulder", 1.5)

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
    tolerance=0.05,       # radians for revolute, metres for prismatic; default 0.01
    max_force=500.0,      # only used in physics mode
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_joint_positions` | `list` or `dict` | required | Target positions — list by index or dict by joint name. Units: radians (revolute), metres (prismatic). |
| `tolerance` | `float`, `dict`, or `None` | `None` | Completion threshold per joint. See [Tolerance](#tolerance) below. |
| `max_force` | `float` | `500.0` | Motor force (N·m) — only used in physics mode |

### Tolerance

`tolerance` controls how close each joint must be to its target before the action
completes. It accepts several forms:

| Form | Example | Behaviour |
|------|---------|----------|
| **Scalar** (`float`) | `0.05` | Same threshold for every joint |
| **List** (`list`) | `[0.005, 0.05, 0.01, 0.01, 0.01]` | Per-joint by absolute joint index; out-of-range indices fall back to the class default (0.01) |
| **Dict** (`{name: float}`) | `{"rail_joint": 0.005, "base_to_shoulder": 0.05}` | Per-joint by name; joints not listed use the class default (0.01) |
| **`None`** (default) | — | Resolved from `agent.joint_tolerance` on the first tick (see below) |

#### Agent-level fallback

When `tolerance` is `None`, the action resolves it from the agent's `joint_tolerance`
property on the first `execute()` call and **writes the resolved value back** to
`action.tolerance` — making it inspectable after the action starts:

```python
# Set agent-level tolerance at construction
agent = Agent.from_urdf(
    urdf_path="robots/rail_arm_robot.urdf",
    ...,
    joint_tolerance={"rail_joint": 0.005, "base_to_shoulder": 0.05},
)

# Action with no explicit tolerance → inherits from agent
action = JointAction(target_joint_positions=[0.3, 0.2, 0.0, 0.0, 0.0])
assert action.tolerance is None  # before execute

agent.add_action(action)
# ... after first tick ...
assert action.tolerance == {"rail_joint": 0.005, "base_to_shoulder": 0.05}
```

The full fallback chain is:

1. **Action-level** — `JointAction(tolerance=0.05)` takes priority
2. **Agent-level** — `agent.joint_tolerance` (set at construction or via property)
3. **Class default** — `0.01` (applied when both are `None`)

This is especially useful for **mixed prismatic + revolute** arms where prismatic joints
measure in metres and revolute joints in radians. A dict tolerance lets you tighten
prismatic accuracy (e.g., ±5 mm) without over-constraining revolute joints.

### Prismatic (linear) joints

`JointAction` works transparently with prismatic joints — target values are in
**metres** instead of radians. See [`examples/arm/rail_arm_demo.py`](https://github.com/yuokamoto/PyBulletFleet/blob/main/pybullet_fleet/examples/arm/rail_arm_demo.py)
for a complete example using a rail arm with 1 prismatic + 4 revolute joints.

---

## 8. Running the Demos

```bash
# Low-level callback approach (1 arm)
python examples/arm/pick_drop_arm_demo.py

# Action-queue approach (1 arm)
python examples/arm/pick_drop_arm_action_demo.py

# 100 arms — AgentManager grid spawn + bulk action management
python examples/scale/pick_drop_arm_100robots_demo.py

# Rail arm (prismatic + revolute) with EE control and per-joint tolerance
python examples/arm/rail_arm_demo.py

# Mobile manipulator (base + arm) — kinematic IK pick/drop
python examples/arm/mobile_manipulator_demo.py
```

Both single-arm demos use kinematic mode by default (`physics=False`). To switch to physics mode,
change the `SimulationParams` line at the top of the file.

### Switching Robot Models

All arm demos accept a `--robot` argument to swap the arm model.
Pass an arm model name or a direct URDF path:

```bash
python examples/arm/pick_drop_arm_demo.py --robot kuka_iiwa
python examples/arm/pick_drop_arm_action_demo.py --robot arm_robot
```

| `--robot` default | Alternatives |
|-------------------|-------------|
| `panda` | `kuka_iiwa`, `arm_robot` |

See [Tutorial 6 — Robot Models](robot-models) for the full model resolution system.

The 100-arm demo combines `AgentManager` / `GridSpawnParams` (see
[Tutorial 3](multi-robot-fleet)) with `JointAction` for fleet-scale arm control.

---

## See Also

- [Tutorial 1 — Spawning Objects](spawning-objects): `from_urdf`, `set_all_joints_targets` basics
- [Tutorial 2 — Action System](action-system): `PickAction`, `DropAction` for mobile robots
- [Tutorial 5 — EE Control & IK](arm-ee-control): control the arm by EE position instead of joint angles
- [Mobile Manipulator Demo](https://github.com/yuokamoto/PyBulletFleet/blob/main/pybullet_fleet/examples/arm/mobile_manipulator_demo.py): kinematic mobile base + arm with IK pick/drop
- [Rail Arm Demo](https://github.com/yuokamoto/PyBulletFleet/blob/main/pybullet_fleet/examples/arm/rail_arm_demo.py): prismatic + revolute joint control with per-joint tolerance
- [Architecture Overview](../architecture/overview): kinematic joint interpolation internals
- [Benchmark Results](../benchmarking/results): arm joint control performance data
