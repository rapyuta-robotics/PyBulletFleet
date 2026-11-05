---
name: working-with-pybullet-fleet
description: "Use when working on PyBulletFleet codebase - adding features, debugging, refactoring, or reviewing code involving simulation, agents, collision, actions, or configuration. Provides domain knowledge, architecture reference, code patterns, and common pitfalls."
---

# Working with PyBulletFleet

Foundation domain reference for all PyBulletFleet development work.

## Architecture

```
User Application
    │
    ▼
MultiRobotSimulationCore  (core_simulation.py)
    │
    ├── Agent (agent.py)           — URDF loading, navigation, action queue, joint control
    ├── SimObject (sim_object.py)  — Single rigid body, pose, attachments, shared shapes
    ├── AgentManager               — Batch spawn, goal management
    ├── Action System (action.py)  — MoveAction, PickAction, DropAction, WaitAction, JointAction
    ├── Tools (tools.py)           — Offset pose calculation utilities
    ├── CollisionVisualizer        — Collision pair rendering
    └── DataMonitor                — Real-time FPS/step-time display
```

## Component Quick Reference

| Component | File | Key Class | Responsibility |
|-----------|------|-----------|----------------|
| Sim Core | `core_simulation.py` | `MultiRobotSimulationCore` | PyBullet init, sim loop, collision, camera |
| Sim Params | `core_simulation.py` | `SimulationParams` | YAML→dataclass config |
| Agent | `agent.py` | `Agent` (extends SimObject) | URDF, nav, actions, joints |
| SimObject | `sim_object.py` | `SimObject` | Single rigid body, pose, attach |
| Agent Mgr | `agent_manager.py` | `AgentManager` | Batch spawn, grid layout, goals |
| Actions | `action.py` | `Action` ABC | Move/Pick/Drop/Wait/Joint |
| Types | `types.py` | Enums | CollisionMode, MotionMode, ActionStatus, etc. |
| Geometry | `geometry.py` | `Pose`, `Path` | Position/orientation, waypoints |
| Tools | `tools.py` | Functions | `calculate_offset_pose()` |
| Config | `config_utils.py` | Functions | YAML loading with merge |

## Before Working — Required Reading

Before modifying PyBulletFleet code, read these files for current state:

```bash
# Architecture overview
cat docs/architecture/overview.md

# All enums (CollisionMode, MotionMode, ActionStatus, etc.)
cat pybullet_fleet/types.py

# Default config with all parameters
cat config/config.yaml

# Test infrastructure (autouse fixtures)
cat tests/conftest.py

# Public API surface
cat pybullet_fleet/__init__.py
```

## Key Enums (types.py)

| Enum | Values | Used By |
|------|--------|---------|
| `MotionMode` | OMNIDIRECTIONAL, DIFFERENTIAL | Agent navigation |
| `MovementDirection` | FORWARD, BACKWARD | Differential drive |
| `ActionStatus` | NOT_STARTED, IN_PROGRESS, COMPLETED, FAILED, CANCELLED | Action system |
| `CollisionMode` | NORMAL_3D, STATIC, DISABLED (NORMAL_2D is deprecated) | Per-object collision |
| `CollisionDetectionMethod` | CLOSEST_POINTS, CONTACT_POINTS, HYBRID | Narrow-phase |
| `SpatialHashCellSizeMode` | CONSTANT, AUTO_ADAPTIVE, AUTO_INITIAL | Broad-phase cell sizing |

## Key Patterns

### Factory Methods — Never call `__init__` directly

```python
# SimulationParams
params = SimulationParams.from_config("config/config.yaml")
params = SimulationParams.from_dict(config_dict)

# Sim Core
sim = MultiRobotSimulationCore.from_yaml("config.yaml")
sim = MultiRobotSimulationCore.from_dict(config_dict)

# Agent
agent = Agent.from_params(spawn_params, sim_core)
agent = Agent.from_urdf("robots/mobile_robot.urdf", sim_core, ...)
agent = Agent.from_mesh(..., sim_core)

# SimObject
obj = SimObject.from_params(spawn_params, sim_core)
obj = SimObject.from_mesh(..., sim_core)
```

### Dataclass Spawn Params

```python
from pybullet_fleet import AgentSpawnParams, Pose
from pybullet_fleet.types import CollisionMode, MotionMode

params = AgentSpawnParams(
    urdf_path="robots/mobile_robot.urdf",
    initial_pose=Pose.from_xyz(0, 0, 0.1),
    motion_mode=MotionMode.OMNIDIRECTIONAL,
    max_linear_vel=2.0,
    collision_mode=CollisionMode.NORMAL_3D,
)
```

### Lazy Logging — Always use `get_lazy_logger()`

```python
from pybullet_fleet.logging_utils import get_lazy_logger
logger = get_lazy_logger(__name__)
# f-string NOT evaluated if log level disabled
logger.debug("Agent %s at %s", agent.name, agent.get_pose())
```

### Shared Shape Cache

`SimObject._shared_shapes` caches PyBullet mesh shape IDs to avoid redundant OpenGL creation.
- Class-level dict: `Dict[str, Tuple[int, int]]` → `(visual_id, collision_id)`
- Only caches mesh shapes (primitives are cheap)
- **Must clear between tests** — IDs become stale after `p.disconnect()`
- `tests/conftest.py` has autouse fixture for this

### Action Lifecycle

```python
# Actions execute via: agent.add_action(action) → agent.update(dt) processes queue
action = MoveAction(path=Path.from_points([[5, 5, 0]]))
agent.add_action(action)
# status: NOT_STARTED → IN_PROGRESS → COMPLETED/FAILED/CANCELLED
```

### Simulation Loop

```python
sim = MultiRobotSimulationCore.from_yaml("config.yaml")
agent = Agent.from_params(params, sim)
sim.register_callback(my_callback, frequency=1.0)
sim.run_simulation()  # or: sim.step_once() for manual control
```

## Testing Patterns

### Fixtures (tests/conftest.py)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `_clear_shared_shapes` | autouse | Clears shape cache (stale IDs after disconnect) |
| `_disable_monitor_gui` | autouse | Prevents matplotlib windows in headless tests |

### Standard PyBullet Test Setup

```python
import pybullet as p

@pytest.fixture
def pybullet_env():
    client = p.connect(p.DIRECT)  # ALWAYS p.DIRECT for tests
    p.setAdditionalSearchPath(pybullet_data.getDataSearchPath())
    p.loadURDF("plane.urdf", physicsClientId=client)
    yield client
    p.disconnect(client)
```

### Test Style

- Use `p.DIRECT` mode — never open GUI in tests
- Create `SimulationParams(gui=False, monitor=False)` for headless
- Use `Agent.from_params()` / `SimObject.from_params()` for consistency
- Assert with keyword-based helpers when available

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Stale shape cache | Crash on `createMultiBody` after disconnect | Clear `SimObject._shared_shapes` (autouse fixture) |
| `physics: false` + `stepSimulation()` | Kinematic objects jump/break | Never call `stepSimulation()` when physics disabled |
| `physics: false` + `contact_points` | Collision always empty | Use `closest_points` (auto-selected by default) |
| `collision_margin` too small | Objects pass through each other | Increase to 5-10% of object size |
| `collision_margin` too large | False-positive collisions | Decrease; check `<collision>` geometry |
| GUI in tests | Test hangs with window open | Always use `p.DIRECT`, not `p.GUI` |
| `TYPE_CHECKING` import guard missing | Circular import error | Use `if TYPE_CHECKING:` for type hints |
| Editing `_agents` list directly | Step loop breaks | Use `add_object()` / `remove_object()` |

## Existing URDFs

| File | Description |
|------|-------------|
| `robots/simple_cube.urdf` | Minimal box — benchmarks, basic tests |
| `robots/mobile_robot.urdf` | Wheeled platform — navigation demos |
| `robots/arm_robot.urdf` | Articulated arm — pick/drop demos |
| `robots/mobile_manipulator.urdf` | Arm on mobile base — combined demos |

## Cross-References

- **REQUIRED:** `DESIGN.md` for full architecture
- **pybullet-performance-workflow** — performance optimization cycle
- **adding-sim-entities** — adding new robots/objects
- **pybullet-collision-tuning** — collision debugging/tuning
- **integrating-with-uso** — USO snapshot/replay integration
- **test-driven-development** — test-first workflow
- **systematic-debugging** — root-cause investigation
- **Details:** See [references/code-patterns.md](references/code-patterns.md) for extended code examples
