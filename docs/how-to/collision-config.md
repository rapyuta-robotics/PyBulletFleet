# Collision Configuration Guide

All collision-related settings in one place — detection method, per-object
modes, safety margin, spatial hash cell size, and multi-cell threshold.

---

## Quick Decision Guide

```
┌─ What kind of simulation? ─────────────────────────────────┐
│                                                             │
│  Kinematics (path planning, fleet coordination)             │
│  → physics: false                                           │
│  → collision_detection_method: "closest_points" (auto)      │
│  → collision_margin: 0.02                                   │
│                                                             │
│  Physics (contact verification, dynamics)                   │
│  → physics: true                                            │
│  → collision_detection_method: "contact_points" (auto)      │
│                                                             │
│  Mixed (some physics + some kinematic objects)              │
│  → physics: true                                            │
│  → collision_detection_method: "hybrid"                     │
│  → collision_margin: 0.05                                   │
└─────────────────────────────────────────────────────────────┘
```

(collision-features-demo)=
## Read collision observations

The Python simulation core exposes `get_collision_observation()`. It
returns `None` before the first collision check or after collision state is
invalidated by object removal/disable; otherwise it reports the
latest check step/time, configured margin, and active qualifying pairs. Each
pair has PBF object IDs, the actual `detection_method` used and its own sample
step/time. `signed_distance` is available for closest-point checks: positive
values are within the configured margin, while zero or negative values mean
geometric touch/overlap. Contact-point checks report `None` for distance.
When collision checks are skipped, the previous observation remains available
with its original timestamp. This is not a history or an all-pairs distance
query.
The check-level `configured_method` records the simulation setting; in hybrid
mode, each pair's `detection_method` identifies the branch actually used.

## Try the visual collision demo

Run `pybullet_fleet/examples/basics/collision_features_demo.py` to observe
`NORMAL_3D`, `NORMAL_2D` (deprecated), and `DISABLED` objects, collision
highlighting, and large-object multi-cell registration in one scene:

```bash
python3 pybullet_fleet/examples/basics/collision_features_demo.py --duration 30
```

Press `w` in the PyBullet window to toggle collision-shape wireframes. The
demo uses kinematics and `closest_points`, so it is intended to illustrate
collision modes and broad-phase configuration rather than contact forces.

---

## Collision Detection Method

Controls **how** collisions are confirmed in the narrow phase.

| Method | Freshness requirement | Safety margin | Best for |
|--------|-----------------------|---------------|----------|
| `closest_points` | Immediate query | ✅ Yes | Kinematics (default) |
| `contact_points` | Explicit collision-detection pass | ❌ No query-distance parameter | Physics |
| `hybrid` | Pass for pairs involving physics | Kinematic--kinematic pairs only | Mixed scenes |

### Configuration

**YAML**:
```yaml
physics: false
collision_detection_method: "closest_points"  # or "contact_points", "hybrid"
```

**Python**:
```python
from pybullet_fleet.core_simulation import SimulationParams
from pybullet_fleet.types import CollisionDetectionMethod

params = SimulationParams(
    physics=False,
    collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
)
```

### Auto-Selection

If `collision_detection_method` is not specified, it is selected automatically:

```python
if physics == False:
    collision_detection_method = "closest_points"  # Kinematics-safe
else:
    collision_detection_method = "contact_points"  # Physics-accurate
```

For architecture details, see [Narrow-Phase Details](narrow-phase-details-pybullet-apis).

---

## Per-Object Collision Mode

Controls collision behavior **per object**. Set at spawn time via `AgentSpawnParams` or `SimObject` constructor.

```python
from pybullet_fleet.types import CollisionMode
```

| Mode | Updates | Grid | Clearance in z | Use Case |
|------|---------|------|----------------|----------|
| `NORMAL_3D` | Every move | Yes | Yes | Default for moving bodies |
| `NORMAL_2D` | Every move | Yes | No (z must overlap) | Deprecated; use `NORMAL_3D` |
| `STATIC` | Once | Yes | Yes | Walls, shelves |
| `DISABLED` | Never | No | — | Markers, visualization |

### Setting via AgentSpawnParams

```python
from pybullet_fleet.types import CollisionMode
from pybullet_fleet.agent import Agent, AgentSpawnParams

# Moving robot (NORMAL_3D is the default and may be omitted)
params = AgentSpawnParams(
    urdf_path="robot.urdf",
    collision_mode=CollisionMode.NORMAL_3D,
)
agent = Agent.from_params(params, sim_core)
```

`NORMAL_2D` is deprecated. It differs from `NORMAL_3D` only in having no
collision clearance on the z axis: a pair with a `NORMAL_2D` body must really
overlap in z. That matters only with `CLOSEST_POINTS` or `HYBRID` and a
non-zero `collision_margin`; with `CONTACT_POINTS` the two behave the same.
Setting it (in spawn parameters, a constructor, `set_collision_mode()` or a
config's `collision_mode: normal_2d`) still works but emits a
`FutureWarning` that points at the calling line. Python shows it by
default, once per line however many objects the line spawns. Switch to
`NORMAL_3D`; the only change is that near misses within the clearance above or
below are reported too. To keep no clearance on the z axis, also set
`collision_margin: [m, m, 0]` (see [Per-axis clearance](per-axis-clearance)).
It reports exactly what `NORMAL_2D` with margin `m` did, for every pair
rather than only pairs with a former `NORMAL_2D` body.

### Setting via SimObject

```python
from pybullet_fleet.sim_object import SimObject

# Fixed structure
wall = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.STATIC,
)

# Visualization marker (no collision)
marker = SimObject(
    body_id=body_id,
    sim_core=sim_core,
    collision_mode=CollisionMode.DISABLED,
)
```

### Runtime Mode Changes

```python
# Temporarily disable collision
obj.set_collision_mode(CollisionMode.DISABLED)
# ... do something ...
obj.set_collision_mode(CollisionMode.NORMAL_3D)  # Re-enable
```

For detailed mode behavior, see [Collision Mode Details](collision-mode-details).

---

(collision-margin)=
## Collision Margin

Safety clearance for `getClosestPoints()`. Only affects `CLOSEST_POINTS` and `HYBRID` modes.

```yaml
collision_margin: 0.02  # meters (2cm default)
```

Set `collision_margin: 0.0` to exclude positive-clearance near misses; touching
and penetrating geometry is still reported. Choose a non-zero margin from the
clearance requirements of the robot and its task.

Use `sim.set_collision_margin(value)` to change the clearance during a run.
It refreshes the spatial grid and rechecks stationary pairs at the next
collision check. The value must be finite and non-negative.

(per-axis-clearance)=
### Per-axis clearance

`collision_margin` also takes one clearance per world axis, `(x, y, z)` in
Python or a three-item list in YAML:

```yaml
collision_margin: [0.05, 0.05, 0.0]  # side by side within 5 cm; nothing above or below
```

```python
SimulationParams(collision_margin=(0.05, 0.05, 0.0))
sim.set_collision_margin((0.05, 0.05, 0.0))
```

This reports ground vehicles that come within 5 cm of each other side by side,
but not one passing close under a shelf or a carried load. Each component must
be finite and non-negative.

The per-axis clearance acts on axis-aligned bounding boxes (AABBs), not on
the shapes themselves. A pair that touches or penetrates is always reported.
A pair that does not is reported when both hold:

- **AABBs**: their AABBs, each axis enlarged by that axis's clearance,
  overlap. With `(0.05, 0.05, 0.0)`, two bodies whose AABBs do not overlap in
  z are never reported, however close they are.
- **Gap**: the straight-line gap between the bodies, from
  `getClosestPoints()`, is within the largest component (5 cm here), as for a
  scalar margin of that size.

So `(m, m, m)` behaves exactly as the scalar `m`, and `(m, m, 0)` behaves
exactly as `CollisionMode.NORMAL_2D` with `m` did, for every pair. The
smaller clearance restricts the pair at AABB level only. Two bodies whose
AABBs overlap on that axis are judged by the straight-line gap, so a near
miss whose gap runs mostly along that axis can still be reported, for example
two spheres at different heights whose AABBs overlap in z, or a rounded or
tilted shape beside a box. Two boxes side by side, both tilted by a fraction
of a degree, are reported like upright ones.

The axes are the world axes, as AABBs are, not a body's own axes. A scalar
behaves exactly as before.
`CONTACT_POINTS`, and the pairs with a physics body under `HYBRID`, ignore the
clearance as they ignore a scalar one. The clearance is global; there is no
per-object clearance. The replay profile records a scalar margin only.

---

(spatial-hash-cell-size)=
## Spatial Hash Cell Size

Controls how 3D space is partitioned for the broad-phase collision grid. Three modes are available.

### Mode 1: `auto_initial` (Default — Recommended)

Calculates `cell_size` once at first collision check from existing object sizes.

```yaml
spatial_hash_cell_size_mode: "auto_initial"  # or omit (default)
```

```python
params = SimulationParams(
    spatial_hash_cell_size_mode="auto_initial",
)
```

- Calculates `max(median AABB extent × 2, 0.5 m)` from the objects present
  when the grid is initialized
- Does not recalculate on later add/remove operations unless you call the
  recalculation method yourself
- Spawn representative objects before first simulation step
- Recalculate manually after a material change in object-size distribution

### Mode 2: `constant` (Fixed)

Use a fixed cell size provided by the user.

```yaml
spatial_hash_cell_size_mode: "constant"
spatial_hash_cell_size: 2.5  # meters (required)
```

```python
params = SimulationParams(
    spatial_hash_cell_size_mode="constant",
    spatial_hash_cell_size=2.5,
)
```

- Predictable: add/remove operations do not recalculate the cell size
- Requires tuning against a representative workload

Choose a value using the AABB extents of the collision-enabled objects and
measure it with the intended scene. The automatic modes use twice the median
of each AABB's largest extent, with a 0.5 m minimum, which is a useful initial
value for a constant configuration.

### Mode 3: `auto_adaptive` (Dynamic)

Automatically recalculates `cell_size` when objects are added or removed.

```yaml
spatial_hash_cell_size_mode: "auto_adaptive"
```

```python
params = SimulationParams(
    spatial_hash_cell_size_mode="auto_adaptive",
)
```

- Recalculates the same median-based heuristic whenever an object is added or
  removed
- Rebuilds the spatial grid after each recalculation; batch spawning defers
  this until the batch completes
- Appropriate when the object set changes frequently enough to justify that
  rebuild cost

### Manual Recalculation

For `auto_initial` mode, you can manually trigger recalculation:

```python
# Recalculate from current objects
new_cell_size = sim.set_collision_spatial_hash_cell_size_mode()
print(f"Cell size: {new_cell_size:.3f} m")

# Switch mode at runtime
from pybullet_fleet.types import SpatialHashCellSizeMode

sim.set_collision_spatial_hash_cell_size_mode(
    mode=SpatialHashCellSizeMode.CONSTANT,
    cell_size=3.0,
)
```

### Recalculation Behavior

| Mode | Initial calculation | Add / remove | Use when |
|------|---------------------|--------------|----------|
| `auto_initial` | First grid initialization | Keeps the existing cell size | The initial scene is representative |
| `constant` | User-provided value | Keeps the configured cell size | You have measured a suitable value |
| `auto_adaptive` | First grid initialization | Recalculates and rebuilds the grid | Object sizes change during the run |

All modes use the same incremental AABB/grid maintenance and moved-object
candidate filtering during collision checks. Their trade-off is when the cell
size and grid are rebuilt, not a fixed big-O cost for every collision check.

For the cell size calculation algorithm, see [Cell Size & Grid Mapping](cell-size-and-grid-mapping).

---

## Multi-Cell Threshold

`multi_cell_threshold` is accepted for compatibility and no longer used. Every
body is registered in every spatial-hash cell its AABB, enlarged by half the
collision margin (not in z for a deprecated NORMAL_2D body), touches, and a moved body is
checked against the bodies sharing one of its cells. Every pair within the
margin is found whatever the bodies' sizes.

For implementation details, see [Multi-Cell Registration](multi-cell-registration).

---

## Static-Collision Filter

`ignore_static_collision` controls whether collision checks include pairs with
a `STATIC` object. Its default is `true`.

```yaml
# Set false to detect moving robots against walls, shelves, and other STATIC objects.
ignore_static_collision: false
```

When it is `true`, the collision system skips **any pair containing a static
object**—not only static--static pairs. Keep the default for scenes where
static structures are only visual context; set it to `false` when structures
are obstacles that must produce collision reports.

`check_collisions(ignore_static=False)` provides the same override for one
explicit collision check.

---

## Complete YAML Reference

```yaml
simulation:
  # Simulation mode
  physics: false  # true for physics, false for kinematics

  # Detection method (auto-selected if omitted)
  collision_detection_method: "closest_points"  # or "contact_points", "hybrid"

  # Safety margin (CLOSEST_POINTS / kinematic--kinematic HYBRID pairs only)
  collision_margin: 0.02  # meters

  # Spatial hash cell size
  spatial_hash_cell_size_mode: "auto_initial"  # "constant" or "auto_adaptive"
  # spatial_hash_cell_size: 2.0  # required only for "constant" mode

  # Whether pairs containing STATIC objects are checked (default: true)
  ignore_static_collision: true

entities:
  # collision_mode is per entity, not a simulation setting.
  - urdf_path: robots/mobile_robot.urdf
    collision_mode: normal_3d  # the default for moving bodies

  - urdf_path: robots/simple_cube.urdf
    collision_mode: static  # fixed; cannot be moved until its mode changes

  - urdf_path: robots/simple_cube.urdf
    collision_mode: disabled  # excluded from framework and PyBullet collision
```

---

## See Also

- [Collision Detection Overview](../architecture/collision-overview) — Design goals and architecture
- [Collision Detection Broad-Phase Details: Spatial Hash Grid](../architecture/collision-spatial-hash) — Algorithm details
- [Collision Detection Narrow-Phase Details](../architecture/collision-internals) — System flow and narrow-phase APIs
- [Configuration Files Guide](../configuration/reference) — Pre-built configuration files
