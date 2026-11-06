# Collision Detection Broad-Phase Details: Spatial Hash Grid

Implementation details of the broad-phase spatial hash grid — cell calculation,
AABB caching, registration and candidate search, and
performance optimizations.

---

(cell-size-and-grid-mapping)=
## Cell Size & Grid Mapping

The spatial hash grid divides 3D space into uniform cells. The `cell_size` determines how large each cell is, and directly affects collision detection performance.

### Cell Size Calculation Modes

The system supports three modes. See [Collision Configuration Guide](spatial-hash-cell-size) for practical configuration.

1. **AUTO_INITIAL** (default): Calculate once at simulation start
   - Calculated on first `run_simulation()` call
   - Based on actual object sizes in your scene at startup
   - Uses formula: `cell_size = median_aabb_extent × scale_factor`
   - Best balance of performance and adaptability

2. **CONSTANT**: Use a fixed, manually-specified cell size
   - Must explicitly set both `spatial_hash_cell_size_mode` and `spatial_hash_cell_size`
   - No automatic calculation or overriding
   - Best when you know the optimal size in advance
   - Fastest (no calculation overhead)

3. **AUTO_ADAPTIVE**: Recalculate on object add/remove
   - Adapts to changing object distributions during simulation
   - Recalculates whenever objects are added or removed
   - Higher overhead, useful when object sizes vary significantly

**Trade-off**:
- **Small cells**: More grid lookups, fewer objects per cell
- **Large cells**: Fewer grid lookups, more objects per cell

### Calculation Algorithm

For auto modes (`auto_initial`, `auto_adaptive`):

```python
# 1. Get all object AABBs
aabbs = [p.getAABB(obj.body_id) for obj in objects]

# 2. Calculate maximum extent for each object
extents = [max(aabb[1][i] - aabb[0][i] for i in range(3)) for aabb in aabbs]

# 3. Use median extent (robust to outliers)
median_extent = sorted(extents)[len(extents) // 2]

# 4. cell_size = 2 * median_extent (covers typical neighbors)
cell_size = max(median_extent * 2.0, 0.5)  # minimum 0.5m
```

**Why median?**
- Robust to outliers (large structures don't skew grid)
- Represents typical object size in mixed environments
- Better than mean for heterogeneous scenes

### Position → Cell Mapping

```python
def get_cell(position: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert world position to grid cell indices"""
    x, y, z = position
    return (int(x / cell_size), int(y / cell_size), int(z / cell_size))
```

**Example**:
```
Object at position [3.2, 5.7, 1.1], cell_size=2.0
→ Cell = (int(3.2/2), int(5.7/2), int(1.1/2)) = (1, 2, 0)
```

### Usage

```python
# Default: cell_size calculated from median AABB on first run_simulation()
sim_core = MultiRobotSimulationCore(params)
sim_core.add_object(...)  # Add objects first
sim_core.run_simulation(...)  # cell_size auto-calculated here (first call)

# Explicit: Calculate before simulation
sim_core = MultiRobotSimulationCore(params)
sim_core.add_object(...)  # Add objects first
sim_core.calculate_optimal_cell_size()  # Explicitly calculate
sim_core.run_simulation(...)  # Uses pre-calculated cell_size
```

---

## AABB Columns

**Purpose**: hold every body's AABB where the broad phase can read them all at once

The simulation core keeps one row per object in a column store
(`pybullet_fleet/collision_store.py`): the AABB, the body's box in its own
frame, its collision mode and whether it is a physics body. Rows are
internal; `sim.cached_aabb(object_id)` returns the box the collision system
holds for an object (``None`` while collision is DISABLED).

**How a box is computed**:

- A kinematic single-body object (a box, a cylinder, a mesh, a URDF with one
  link): from its pose. Its local box is read from PyBullet once, at identity
  orientation; the world box at any pose is that box transformed by the
  element-wise absolute rotation, which is exactly what PyBullet's
  `getAABB()` returns (`btTransformAabb`); a sphere's extents do not rotate.
  The boxes of every moved object are computed in one array operation in the
  step's Phase 3, with no PyBullet call.
- A physics body: read from PyBullet after every `stepSimulation()`.
- An articulated body: the union of its links' boxes, one `getAABB()` per
  link with a collision shape (`p.getAABB(body)` alone is the base link's).

An articulated body's box covers every link, so it changes when a joint
moves. `notify_geometry_changed(obj)` marks the body moved and refreshes its
box: at once outside a step, with the flushed poses (Phase 3) inside one.
`Agent.set_joint_position()`, `set_joints_positions()`, the kinematic joint
interpolation and a joint checkpoint restore call it; physics bodies are
re-read after every `stepSimulation()`. Call it after moving a body's joints
directly with `p.resetJointState()`.
`examples/arm/link_collision_demo.py` shows all three cases: a fixed arm whose
end effector sweeps through a box well outside its base, a mobile manipulator
whose stretched-out arm hits a pillar its base clears, and a joint reset by hand
followed by `notify_geometry_changed`.

---

(neighbor-search-patterns)=
(multi-cell-registration)=
## Registration and Candidate Search

Every collision body (NORMAL_3D, NORMAL_2D and STATIC; not DISABLED) is
registered in **every cell its AABB, enlarged by half the collision margin on
every side, touches**; with a per-axis margin `(x, y, z)`, each axis is
enlarged by half that axis's margin. A NORMAL_2D body's z extent is not
enlarged (NORMAL_2D is deprecated). A moved
body's candidates are the bodies registered in **its own cells**: there is no
neighbour search.

```
cell size 1.0, margin 0.02 (enlargement 0.01 per side)

  x: 0.0         1.0         2.0
     ┌───────────┬───────────┐
     │   A       │           │   A: 0.40-0.60        -> cell 0 only
     │        B══╪══         │   B: 0.90-1.10        -> cells 0 and 1
     │           │  C        │   C: 1.15-1.35        -> cell 1 only
     └───────────┴───────────┘
  A-B share cell 0, B-C share cell 1: both pairs are candidates.
  A-C share no cell: never looked at.
```

**Why no pair is missed**: two bodies within the margin `m` have enlarged AABBs
(each grown by `m/2` per side) that overlap. The cell holding a point of that
overlap is one both are registered in. A pair with a NORMAL_2D body must
overlap in z, so the cell holding a point of that overlap is one both reach
in z without enlargement. Without a margin (`m = 0`) every body is registered
in exactly the cells its AABB touches, which is the same argument.

**How the hash is kept**: two sorted arrays of packed cell keys, with the
row of each entry alongside: one for the STATIC bodies, patched only when a
static body changes, and one for the others, rebuilt (one sort) whenever
one of them moved, since in a driving fleet most of them did. A search for
the candidates of the moved bodies is one `searchsorted` per array over all
of their cells at once (the static array is skipped under `ignore_static`);
the pairs then go through the margin-based AABB filter (with its NORMAL_2D
z rule, the zero margin for a pair with a physics body and DISABLED) as
arrays, and come out as object id pairs. Nothing in it is per object Python.

**What it costs**: a body small next to a cell is usually in one cell, in two
or four near a cell boundary. A body larger than a cell, a wall say, is in
every cell it covers; a *moving* one pays for its size on every move, as its
entries are rebuilt each time it moves. A large static body pays this once.
The pair search reads only a body's own cells, not the 9 or 27 around it.

`multi_cell_threshold` is accepted for compatibility and no longer used: every
body is registered wherever it reaches, whatever its size, so it is no longer a
way to cap that cost for a large moving body. A larger `cell_size` is.

```
Cell size: 2.0m
Wall: 5.0m × 5.0m, centred on a cell corner -> registered in the 3×3 cells it covers:
┌─────┬─────┬─────┐
│  W  │  W  │  W  │
├─────┼─────┼─────┤
│  W  │  W  │  W  │
├─────┼─────┼─────┤
│  W  │  W  │  W  │
└─────┴─────┴─────┘
A robot in any of them finds the wall in its own cell.
```

A DISABLED body is in no cell, even after it moves.

---

## Movement-Based Optimization

**Key Insight**: Most objects don't move every frame

**Strategy**: Only check pairs involving moved objects

```python
def check_collisions(self):
    # Track objects that moved since last check
    moved_objects = self._moved_this_step.copy()
    self._moved_this_step.clear()

    # The candidates of all the moved bodies, from their own cells, as arrays
    ids_a, ids_b = store.candidate_pairs(store.rows(moved_objects), clearance, ...)
```

---

(incremental-updates)=
## Incremental Updates

**Concept**: Only update what changed

- The boxes of the objects that moved are recomputed (kinematic single
  bodies in one array operation, the rest from PyBullet); the others keep
  theirs.
- The static bodies' entries are kept as they are; the moving bodies'
  entries are rebuilt from their new boxes, one sort over the moving bodies
  only.
- Only the moved objects are searched for pairs; a static object is never
  in `_moved_this_step`.

```
33,000 static tiles and 200 moving robots:
  the tiles' entries stay; each step sorts the 200 robots' entries afresh
  and searches from them only.
```

---

## See Also

- [Collision Detection Overview](collision-overview) — Design goals, mode summary, architecture
- [Collision Detection Narrow-Phase Details](collision-internals) — System flow, narrow-phase APIs, mode details
- [Collision Configuration Guide](../how-to/collision-config) — Practical cell size configuration
