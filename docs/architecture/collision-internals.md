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

A per-axis `collision_margin` `(x, y, z)` acts on AABBs only: the broad phase
enlarges each AABB axis by that axis's clearance, and the narrow phase first
holds a pair it checks again without the broad phase (an active pair) to the
same AABB test. `getClosestPoints()` is then queried with the largest
component and judges the straight-line gap, with no per-axis test of the
points; `(m, m, 0)` is therefore the same as `NORMAL_2D` with `m` for every
pair. See [Per-axis clearance](per-axis-clearance).

---

(collision-mode-details)=
## Collision Mode Details

This section provides comprehensive details for each collision mode.

### NORMAL_3D - Full 3D Collision

**Characteristics**:
- **Candidates**: the bodies sharing one of its cells
- **AABB Updates**: Every movement
- **Grid Registration**: Updated on movement
- **PyBullet Collision**: Enabled

**Use Cases**:
- Any moving object: ground robots (AGVs, AMRs), drones, manipulators

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
- Registered with its z extent enlarged by half the margin; a NORMAL_2D body's z extent is not enlarged
- Candidates are the bodies sharing a cell, for both modes
- Same cost as NORMAL_2D: the two differ only in the z clearance

### NORMAL_2D - No Clearance in Z (Deprecated)

NORMAL_2D is deprecated: setting it emits a `FutureWarning`, and it will
be removed in a future release. Use NORMAL_3D; the only difference is that a
body within the clearance above or below is reported. A
`collision_margin` of `(m, m, 0)` removes the z clearance instead, for every
pair, and reports exactly what NORMAL_2D did (see
[Per-axis clearance](per-axis-clearance)).

**Characteristics**:
- **Candidates**: the bodies sharing one of its cells
- **AABB Updates**: Every movement
- **Grid Registration**: Updated on movement; z extent not enlarged by the margin
- **Z-Axis**: AABB Z must actually overlap (no margin in z); Z is not ignored
- **Differs from NORMAL_3D** only with a non-zero clearance (CLOSEST_POINTS / HYBRID): a body within the clearance above or below is not reported


### STATIC - Fixed Objects

**Characteristics**:
- **Candidates**: found from a moving body's side, through a cell both are registered in
- **AABB Updates**: NEVER (calculated once at spawn)
- **Grid Registration**: At spawn, and again whenever the grid is rebuilt (cell size or margin change)
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

# Moving ground AGV
agv = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.NORMAL_3D,
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

- `ignore_attached_collision` (default `true`) drops contact between objects
  on one attachment tree built with `attach_object()`: a parent and child, two
  loads on one carrier, and a load on a load. Contact with anything outside
  the tree is still reported. Attaching or detaching takes effect on the next
  collision check: an active pair that becomes excluded ends
  (`COLLISION_ENDED`), and an excluded pair is checked again once it is not.
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
ignored while active resolves on the next check with `COLLISION_ENDED`, and
an ignored pair that stops being ignored -- an in-place `detach_object()`, a
filter whose answer changes -- is checked again on the next one, even if
neither body moved. Ignored pairs skip the narrow phase, so the filter runs
once per candidate, active or previously ignored pair on each check and
should be cheap.

### Collision Margin Recommendations

| Use Case | Margin | Rationale |
|----------|--------|-----------|
| Tight spaces | 0.01-0.02m | Minimal clearance |
| Normal operation | 0.02-0.05m | Comfortable safety |
| Conservative | 0.05-0.10m | Extra safety for high-speed |
| Contact detection | 0.0m | Actual touching only |

### Parity Traces

`tests/collision_parity/` keeps what the pipeline reports over ten small
scripted scenes: for every step, the `COLLISION_STARTED` / `COLLISION_ENDED`
events, the active pairs of the collision observation and each pair's
`signed_distance`. The traces recorded on the reference revision live under
`tests/collision_parity/golden/`, and `tests/test_collision_parity.py`
compares a fresh recording with them. A change to the broad phase, the narrow
phase or the pair bookkeeping that is meant to report the same thing shows it
by passing unchanged; one that knowingly reports something else writes the
difference, pair by pair, in the test's `DOCUMENTED_DIFFERENCES`, and the
test also fails when a written-down difference stops occurring. Record the
goldens again with `python -m tests.collision_parity.record` and say in the
commit which revision they come from.

Two properties of the current pipeline that the traces pin down, and that a
rewrite should keep or document: a body set to `DISABLED` drops its active
pairs without a `COLLISION_ENDED` event, and a pair enabled again later
raises `COLLISION_STARTED` afresh; and a `detach_object()` in place leaves
the former carrier and load touching, so they are reported as a pair from the
next check on.

---

## See Also

- [Collision Detection Overview](collision-overview) — Design goals, mode summary, architecture
- [Collision Detection Broad-Phase Details: Spatial Hash Grid](collision-spatial-hash) — Algorithm details
- [Collision Configuration Guide](../how-to/collision-config) — Practical configuration
