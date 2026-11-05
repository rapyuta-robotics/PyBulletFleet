---
name: adding-sim-entities
description: "Use when adding new robot types, simulation objects, or infrastructure entities (conveyors, elevators, linear joints, kinematic arms) to PyBulletFleet - covers URDF creation, spawn parameters, config, actions, and demo scripts"
---

# Adding Simulation Entities

Checklist-driven process for adding new robots, objects, or infrastructure to PyBulletFleet.

## Step 0: Decide — SimObject vs Agent

```
Does the entity have joints or URDF?
  YES → Agent (extends SimObject)
    └── Supports: URDF loading, joint control, navigation, action queue

  NO → SimObject (single rigid body)
    └── Supports: Primitive/mesh shapes, pose, attachments
```

**Complex entities** (conveyor, elevator) may combine both:
- Structure body → SimObject (static)
- Moving parts → Agent with joints or custom logic

| Planned Entity | Base Class | MotionMode | Actions Needed |
|---------------|-----------|------------|----------------|
| Linear joint | Agent | N/A (joint) | JointAction (existing) |
| Kinematic arm | Agent | N/A (joint) | PickAction/DropAction (existing) |
| Physics mobile robot | Agent | DIFFERENTIAL | MoveAction (existing) |
| Conveyor | SimObject + logic | N/A | New ConveyAction |
| Elevator | SimObject + logic | N/A | New ElevatorAction |

## Step 1: Create URDF

Place in `robots/` directory. Reference existing URDFs:

```bash
ls robots/
# arm_robot.urdf           — articulated arm with joints
# mobile_manipulator.urdf  — arm on mobile base
# mobile_robot.urdf        — wheeled platform
# simple_cube.urdf         — minimal box
```

**URDF Checklist:**
- [ ] `<visual>` geometry defined for all links
- [ ] `<collision>` geometry defined (required for collision detection)
- [ ] Joint limits set (`<limit lower="" upper="" effort="" velocity="">`)
- [ ] Mass/inertia defined if `physics: true` will be used
- [ ] Reasonable scale (meters — PyBullet uses SI units)

See [references/urdf-checklist.md](references/urdf-checklist.md) for detailed URDF requirements.

## Step 2: Define SpawnParams

```python
from pybullet_fleet import AgentSpawnParams, Pose
from pybullet_fleet.types import CollisionMode, MotionMode

params = AgentSpawnParams(
    urdf_path="robots/your_robot.urdf",
    initial_pose=Pose.from_xyz(0, 0, 0.1),
    motion_mode=MotionMode.OMNIDIRECTIONAL,  # or DIFFERENTIAL
    max_linear_vel=2.0,
    max_linear_accel=5.0,
    max_angular_vel=3.0,
    max_angular_accel=10.0,
    collision_mode=CollisionMode.NORMAL_3D,
    use_fixed_base=False,
)
```

For SimObject (no URDF):
```python
from pybullet_fleet.sim_object import SimObjectSpawnParams, ShapeParams

params = SimObjectSpawnParams(
    visual_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.1]),
    collision_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.1]),
    initial_pose=Pose.from_xyz(3, 3, 0.1),
    mass=0.0,  # 0 = kinematic
    collision_mode=CollisionMode.STATIC,
)
```

## Step 3: Add Config (if needed)

If the entity has tunable parameters, add to `config/config.yaml` or create a new YAML:

```yaml
# New robot type parameters
your_robot:
  urdf_path: "robots/your_robot.urdf"
  max_linear_vel: 2.0
  motion_mode: "omnidirectional"
  collision_mode: "normal_3d"
```

Map new params in `SimulationParams.from_dict()` if needed.

## Step 4: Spawn and Test

```bash
# Create test FIRST (test-driven-development)
# tests/test_your_entity.py
```

```python
def test_spawn_your_robot(pybullet_env):
    params = SimulationParams(gui=False, monitor=False)
    sim = MultiRobotSimulationCore(params)
    agent = Agent.from_params(spawn_params, sim)
    assert agent.object_id is not None
    assert agent.get_pose() is not None
```

**Test navigation, actions, collision mode** — see existing tests for patterns:
```bash
ls tests/test_agent_*.py tests/test_action*.py
```

## Step 5: Add Actions (if needed)

New actions extend `Action` ABC:

```python
from pybullet_fleet.action import Action
from pybullet_fleet.types import ActionStatus

class ConveyAction(Action):
    def __init__(self, target_position, speed):
        super().__init__()
        self.target_position = target_position
        self.speed = speed

    def execute(self, agent, dt) -> bool:
        """Return True when complete."""
        # Move object toward target
        if reached_target:
            self._status = ActionStatus.COMPLETED
            return True
        return False

    def reset(self):
        super().reset()
        # Reset action-specific state
```

**Action lifecycle:** NOT_STARTED → IN_PROGRESS → COMPLETED/FAILED/CANCELLED

## Step 6: Create Demo

Place in `examples/`. Follow existing demo pattern:

```bash
# Reference demos:
# examples/basics/robot_demo.py            — basic agent demo
# examples/basics/action_system_demo.py    — action queue demo
# examples/arm/pick_drop_arm_demo.py       — arm pick/drop
# examples/mobile/path_following_demo.py   — path following
# examples/scale/100robots_grid_demo.py    — large-scale spawn
```

Standard demo structure:
1. Config → SimulationParams
2. Create sim → MultiRobotSimulationCore
3. Spawn entities → Agent/SimObject.from_params or AgentManager
4. Set up callback → sim.register_callback
5. Run → sim.run_simulation()

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| URDF missing `<collision>` | Add collision geometry — required for collision detection |
| Wrong MotionMode for URDF | DIFFERENTIAL needs proper wheel layout; OMNIDIRECTIONAL for simple robots |
| `_shared_shapes` key collision | Use unique URDF path or shape params |
| `use_fixed_base=True` blocking movement | Fixed base agents can't navigate — use for stationary arms only |
| Forgetting to add to `__init__.py` | If entity type is a new public class, export it |

## Cross-References

- **REQUIRED:** working-with-pybullet-fleet — codebase patterns and architecture
- **pybullet-collision-tuning** — choosing collision mode for new entity
- **test-driven-development** — write tests before implementation
