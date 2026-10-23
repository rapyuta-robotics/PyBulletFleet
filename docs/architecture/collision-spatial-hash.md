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

## AABB Caching

**Purpose**: Avoid redundant `getAABB()` calls

**Data Structure**:
```python
_cached_aabbs_dict: Dict[int, Tuple[Tuple[float, float, float],
                                     Tuple[float, float, float]]]
# Key: object_id (body_id)
# Value: ((min_x, min_y, min_z), (max_x, max_y, max_z))
```

**Update Strategy**:
```python
def _update_object_aabb(self, object_id: int):
    """Called when object moves"""
    obj = self._sim_objects_dict[object_id]

    # Query PyBullet (expensive!)
    self._cached_aabbs_dict[object_id] = p.getAABB(
        obj.body_id,
        physicsClientId=self.client
    )

    # Update grid registration
    self._update_object_spatial_grid(object_id)
```

**Invalidation**: Cache invalidated on movement, mode change, or removal

---

(neighbor-search-patterns)=
(multi-cell-registration)=
## Registration and Candidate Search

Every collision body (NORMAL_3D, NORMAL_2D and STATIC; not DISABLED) is
registered in **every cell its AABB, enlarged by half the collision margin on
every side, touches**. A NORMAL_2D body's z extent is not enlarged. A moved
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

**What it costs**: a body small next to a cell is usually in one cell, in two
or four near a cell boundary. A body larger than a cell, a wall say, is in
every cell it covers. The pair search then reads only the body's own cells,
not the 9 or 27 around it, and the margin-based AABB filter (with its NORMAL_2D
z rule, the zero margin for a pair with a physics body, `ignore_static` and
DISABLED) decides on each candidate.

`multi_cell_threshold` is accepted for compatibility and no longer used: every
body is registered wherever it reaches, whatever its size.

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

    # Only iterate through moved objects
    for obj_id in moved_objects:
        # Find neighbors in grid
        # Check AABB overlap
        # Add to candidate pairs
```

---

(incremental-updates)=
## Incremental Updates

**Concept**: Only update what changed

**Benefits**:
```
1000 objects, 10 move per frame:
  Full update: 1000 getAABB() calls
  Incremental: 10 getAABB() calls
  → 100x faster! ✓
```

**Implementation**:
- Track moved objects in `_moved_this_step`
- Only these objects trigger collision checks
- Static objects never added to `_moved_this_step`

---

## See Also

- [Collision Detection Overview](collision-overview) — Design goals, mode summary, architecture
- [Collision Detection Narrow-Phase Details](collision-internals) — System flow, narrow-phase APIs, mode details
- [Collision Configuration Guide](../how-to/collision-config) — Practical cell size configuration
