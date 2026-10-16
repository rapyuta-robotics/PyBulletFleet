# Collision Detection Narrow-Phase Details

Narrow-phase PyBullet APIs, collision mode implementation details, and
configuration reference.

For the overall system flow and object lifecycle, see
[Collision Detection Overview](system-flow-object-lifecycle).

---

(narrow-phase-details-pybullet-apis)=
## Narrow-Phase Details: PyBullet APIs

After broad-phase filtering, we use PyBullet's APIs for exact collision detection.

### API Comparison

| Feature | `getContactPoints()` | `getClosestPoints()` |
|---------|----------------------|----------------------|
| **Freshness requirement** | Explicit collision-detection pass required | Query is immediate |
| **Works with kinematics** | ⚠️ Cache/solver-dependent | ✅ Yes |
| **Safety margin support** | ❌ No | ✅ Yes (distance param) |
| **Result** | Contact manifold; solver results after a physics step | Geometric closest-point result |
| **Best for** | Physics mode | Kinematics mode |

### `getClosestPoints(distance)` - Recommended for Kinematics

**What it does**: Geometric closest-point query between two collision bodies.

**Key Parameters**:
```python
closest_points = p.getClosestPoints(
    bodyA=obj1.body_id,
    bodyB=obj2.body_id,
    distance=0.02,  # Safety margin (2cm)
    physicsClientId=self.client
)
```

**Behavior**:
- `distance > 0`: Returns points when the closest separation is within the margin
- `distance = 0`: Returns points for touching **or penetrating** geometry
- `len(closest_points) > 0`: Collision detected

**Advantages**:

- ✅ Works with `resetBasePositionAndOrientation()` (kinematic)
- ✅ Stable for kinematic-kinematic pairs
- ✅ Safety margin support (detect "near miss")
- ✅ No `stepSimulation()` required

**Use cases**:
- Kinematics-focused planning
- Collision avoidance with safety buffer
- High-speed simulation that does not need rigid-body integration or contact solving

### `getContactPoints()` - For Physics Mode

**What it does**: Query PyBullet's most recently computed contact manifold.

**Key Parameters**:
```python
contact_points = p.getContactPoints(
    bodyA=obj1.body_id,
    bodyB=obj2.body_id,
    physicsClientId=self.client
)
```

**Behavior**:
- Returns the contact manifold from the latest collision-detection pass
  (`stepSimulation()` or `performCollisionDetection()`)
- Includes signed contact distance, normal, and contact position
- Includes solver-derived normal force after `stepSimulation()`; a
  `performCollisionDetection()`-only pass does not run the contact solver
- Empty list if no contact

**Advantages**:

- ✅ Represents contacts used by the physics simulation
- ✅ Contact normals and signed distances are available
- ✅ Normal force is available after the physics solver runs

**Limitations**:

- ❌ **Needs a collision-detection pass**: `stepSimulation()` is the normal
  physics-mode path; `performCollisionDetection()` refreshes contacts without
  advancing dynamics
- ❌ Limited for kinematics: results depend on the contact cache and solver;
  use `getClosestPoints()` for kinematic--kinematic checks
- ❌ No caller-provided safety margin: unlike `getClosestPoints()`, this API
  does not take a query distance

**Why not ideal for kinematics**:
- Kinematics mode uses `resetBasePositionAndOrientation()` to set positions directly
- `getContactPoints()` reads a contact cache; a pose reset alone does not refresh it
- `performCollisionDetection()` can refresh the cache, but does not calculate
  contact forces or advance physics
- `getClosestPoints()` is therefore the direct, margin-aware query for
  kinematic objects

**Use cases**:
- Physics mode
- Contact-force analysis after a physics step
- Debugging actual contacts

### stepSimulation() - The Physics Update

`stepSimulation()` performs **comprehensive physics update**:

1. **Time Integration**: Update positions, velocities (F=ma)
2. **Broad-Phase**: Internal collision detection
3. **Narrow-Phase**: Exact collision testing
4. **Contact Solving**: Resolve penetrations, apply friction
5. **Cache Update**: Update contact manifold

**Cost**: Computationally expensive (scene-dependent)

**When Required**:
- ✅ Physics ON mode (dynamic objects, mass > 0)
- ❌ Kinematics mode (pure planning)

### Two-Mode Design Philosophy

**Mode 1: Kinematics (Default)** ⚡
```yaml
physics: false
collision_detection_method: CLOSEST_POINTS
collision_margin: 0.02
```
- No rigid-body integration or contact solving
- `getClosestPoints()` → Stable
- Use case: Planning, coordination, high-speed simulation

**Mode 2: Physics** 🔬
```yaml
physics: true
collision_detection_method: CONTACT_POINTS  # or HYBRID
collision_margin: 0.02
```
- `stepSimulation()` every frame → Accurate
- `getContactPoints()` → Realistic contacts
- Use case: Physics simulation, debugging, contact analysis

**Hybrid Mode** (Advanced):
```python
if either_object_is_physics:
    use getContactPoints()  # Accurate for dynamics
else:
    use getClosestPoints()  # Stable for kinematics
```

In the current implementation, an object is physics-enabled when it has
`mass > 0` and is not kinematic.  A pair uses `getContactPoints()` when either
member is physics-enabled; only pairs where both members are kinematic use
`getClosestPoints()` and `collision_margin`.

---

(collision-mode-details)=
## Collision Mode Details

This section provides comprehensive details for each collision mode.

### NORMAL_3D - Full 3D Collision

**Characteristics**:
- **Neighbor Check**: 27 cells (3×3×3 cube)
- **AABB Updates**: Every movement
- **Grid Registration**: Updated on movement
- **PyBullet Collision**: Enabled

**Use Cases**:
- Flying drones (need Z-axis collision)
- 3D manipulators
- Multi-level warehouses
- Any object that moves in 3D space

**Configuration**:
```python
drone = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.NORMAL_3D,
    mass=0.0  # Kinematic
)
```

**Performance**:
- Checks 27 neighbor cells vs 9 for 2D
- ~3x more cells, but filtered by AABB
- Typical overhead: <10% vs NORMAL_2D

### NORMAL_2D - 2D Collision (XY Plane)

**Characteristics**:
- **Neighbor Check**: 9 cells (3×3×1, XY plane only)
- **AABB Updates**: Every movement
- **Grid Registration**: Updated on movement
- **Z-Axis**: Ignored for neighbor search (but AABB Z must still overlap!)

**Use Cases**:
- Ground robots (AGVs, AMRs)
- Warehouse robots (single floor)
- 2D path planning
- Most typical use cases

**Configuration**:
```python
agv = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.NORMAL_2D,
    mass=0.0
)
```

### STATIC - Fixed Objects

**Characteristics**:
- **Neighbor Check**: 27 cells (when checked by moving objects)
- **AABB Updates**: NEVER (calculated once at spawn)
- **Grid Registration**: Once at spawn, never updated
- **Movement**: Assumed to never move

**Use Cases**:
- Walls, pillars, structures
- Shelves, racks
- Fixed obstacles
- Any object that never moves

**Configuration**:
```python
wall = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.STATIC,
    is_static=True
)
```

**Performance Benefits**:
```
Many static walls (e.g., 1000):
  NORMAL_3D: All walls updated every frame → High overhead
  STATIC: 0 AABB updates/frame → No overhead ✓

  → Moving robots can still collide with walls!
  → Walls never trigger updates themselves
```

**Important**: If a STATIC object needs to move, change mode first:
```python
wall.set_collision_mode(CollisionMode.NORMAL_3D)  # Enable updates
wall.set_pose([new_x, new_y, new_z])               # Move
wall.set_collision_mode(CollisionMode.STATIC)      # Re-optimize
```

### DISABLED - No Collision

**Characteristics**:
- **Neighbor Check**: NEVER (not in spatial grid)
- **AABB Updates**: NEVER
- **Grid Registration**: NEVER
- **PyBullet Collision**: Disabled via `setCollisionFilterPair()`

**Use Cases**:
- Visualization markers
- Debug geometry
- Temporary exclusion
- Non-physical objects

**Configuration**:
```python
marker = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.DISABLED,
)
```

**Behavior**:
```python
# Object is completely invisible to collision system
marker.set_pose([x, y, z])  # Visual update only
pairs = sim_core.check_collisions()  # Marker ignored ✓
```

**Performance**:
- Zero overhead (not processed at all)
- Use liberally for visualization objects

### Runtime Mode Changes

**Supported Transitions**:
```python
# Any mode → Any mode
obj.set_collision_mode(new_mode)

# Example: Temporarily disable
obj.set_collision_mode(CollisionMode.DISABLED)
# ... do something ...
obj.set_collision_mode(CollisionMode.NORMAL_3D)  # Re-enable
```

**What Happens Internally**:
1. Remove from old mode's data structures
2. Update AABB if needed
3. Register to new mode's data structures
4. Update PyBullet collision filter if DISABLED ↔ other

**Use Cases**:
- Dynamic environment changes
- Temporary object exclusion
- Performance optimization
- A/B testing different modes

---

## Implementation & Configuration

### SimulationParams

```python
from pybullet_fleet.core_simulation import SimulationParams
from pybullet_fleet.types import CollisionDetectionMethod

# Kinematics Mode (Recommended Default)
params = SimulationParams(
    physics=False,
    collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
    collision_margin=0.02,  # 2cm safety buffer
    spatial_hash_cell_size=2.0,  # 2m cells
    multi_cell_threshold=1.5,  # Large object threshold
)

# Physics Mode
params = SimulationParams(
    physics=True,
    collision_detection_method=CollisionDetectionMethod.CONTACT_POINTS,
    collision_margin=0.02,
    spatial_hash_cell_size=2.0,
)

# Hybrid Mode (Advanced)
params = SimulationParams(
    physics=True,
    collision_detection_method=CollisionDetectionMethod.HYBRID,
    collision_margin=0.05,  # Larger margin for kinematics safety
    spatial_hash_cell_size=2.0,
)
```

### Per-Object Configuration

```python
# Moving 3D robot
robot_3d = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.NORMAL_3D,
    mass=0.0,  # Kinematic
)

# Moving 2D AGV
agv = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.NORMAL_2D,
    mass=0.0,
)

# Fixed structure
wall = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.STATIC,
    is_static=True,
)

# Visualization marker
marker = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.DISABLED,
)
```

### Config File (YAML)

```yaml
# config.yaml
physics: false
collision_detection_method: "closest_points"
collision_margin: 0.02
spatial_hash_cell_size: 2.0
multi_cell_threshold: 1.5
ignore_static_collision: false
```

### Auto-Selection Logic

**Collision Detection Method**:
- `physics=False` → `CLOSEST_POINTS` (auto)
- `physics=True` → `CONTACT_POINTS` (auto)
- Explicit setting overrides auto-selection

**Collision Mode**:
- Default: `NORMAL_3D` if not specified
- Recommended: Explicitly set for clarity

### Ignoring Expected Contact

Some touching pairs are meant to touch and should not count as collisions.

- `ignore_attached_collision` (default `true`) drops contact between an
  object and the object it is attached to with `attach_object()`. Only the
  direct parent and child are dropped; anything else the carrier touches is
  still reported.
- `set_collision_ignore_filter(fn)` drops every touching pair for which
  `fn(obj_a, obj_b)` returns `True`, for rules only the application knows:

```python
def is_expected_contact(obj_a, obj_b):
    # e.g. a vehicle and the platform it is riding on
    return obj_b in platform_riders.get(obj_a, ()) or obj_a in platform_riders.get(obj_b, ())

sim.set_collision_ignore_filter(is_expected_contact)
```

An ignored pair is excluded everywhere: it is not an active pair, is not
counted in `collision_count`, raises no `COLLISION_STARTED` and is absent from
`get_collision_observation()`. Filtering in a `COLLISION_STARTED` handler
instead would leave those figures counting the contact. A pair that becomes
ignored while active resolves on the next check with `COLLISION_ENDED`.
Ignored pairs skip the narrow phase, so the filter runs once per candidate or
active pair on each check and should be cheap.

### Collision Margin Recommendations

| Use Case | Margin | Rationale |
|----------|--------|-----------|
| Tight spaces | 0.01-0.02m | Minimal clearance |
| Normal operation | 0.02-0.05m | Comfortable safety |
| Conservative | 0.05-0.10m | Extra safety for high-speed |
| Contact detection | 0.0m | Actual touching only |

---

## See Also

- [Collision Detection Overview](collision-overview) — Design goals, mode summary, architecture
- [Collision Detection Broad-Phase Details: Spatial Hash Grid](collision-spatial-hash) — Algorithm details
- [Collision Configuration Guide](../how-to/collision-config) — Practical configuration
