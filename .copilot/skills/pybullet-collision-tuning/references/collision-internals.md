# Collision Internals

Implementation details of the spatial hash collision detection system. Read when debugging collision code or making implementation changes.

## Table of Contents

1. [Spatial Hash Grid Implementation](#spatial-hash-grid-implementation)
2. [AABB Management](#aabb-management)
3. [Incremental Update Strategy](#incremental-update-strategy)
4. [Narrow Phase Method Comparison](#narrow-phase-method-comparison)
5. [Known Historical Issues](#known-historical-issues)
6. [Key Variables in core_simulation.py](#key-variables-in-core_simulationpy)

## Spatial Hash Grid Implementation

Objects are indexed into a 3D grid:

```
cell_index = (floor(x / cell_size), floor(y / cell_size), floor(z / cell_size))
```

**Data structures (in MultiRobotSimulationCore):**
- `_cached_spatial_grid: Dict[Tuple[int,int,int], Set[int]]` — cell → object IDs
- `_cached_aabbs_dict: Dict[int, Tuple]` — object ID → AABB bounds

**Registration and search:**
- Every body is registered in each cell its AABB, enlarged by half the collision margin, touches (`NORMAL_2D` is not enlarged in z).
- Candidates for a body are the bodies registered in its own cells; no neighbour cells are searched. Two bodies within the margin always share a cell.
- `STATIC`: registered once, never re-checked for movement
- `DISABLED`: not in grid at all
- `multi_cell_threshold` is accepted for compatibility and not used.

## AABB Management

Each object's Axis-Aligned Bounding Box (AABB) is queried from PyBullet:

```python
aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=client)
```

**Caching strategy:**
- Kinematic objects: AABB cached, updated only when object moves (via `_moved_this_step`)
- Physics objects: AABB updated every step (conservative — physics can move them)
- Static objects: AABB cached once at spawn, never updated

## Incremental Update Strategy

The key optimization: only check pairs involving moved objects.

```
step_once():
  1. Mark physics objects as moved (conservative)
  2. Agent.update(dt) → if moved, mark as moved
  3. collision check:
     a. For each moved object → update AABB, re-hash into grid
     b. For each moved object → check neighbors for overlap
     c. Narrow-phase → confirm overlap with PyBullet API
  4. Clear _moved_this_step
```

`_moved_this_step: Set[int]` accumulates across steps when `collision_check_frequency` is < step frequency.

## Narrow Phase Method Comparison

### CLOSEST_POINTS — `p.getClosestPoints(bodyA, bodyB, distance)`

- Returns closest point pairs between two bodies within `distance` threshold
- Works without `stepSimulation()` — pure geometric query
- `distance` parameter = `collision_margin` from config
- Stable with `resetBasePositionAndOrientation()` (kinematic motion)
- **Default for physics=false**

### CONTACT_POINTS — `p.getContactPoints(bodyA, bodyB)`

- Returns contact manifold from physics engine
- Requires `stepSimulation()` to be called (updates contact manifold)
- Returns empty for kinematic-kinematic pairs if not stepped
- Faster per-call than CLOSEST_POINTS in physics mode
- **Default for physics=true**

### HYBRID

- Uses CONTACT_POINTS for physics objects (in `_physics_objects` set)
- Uses CLOSEST_POINTS for kinematic objects
- Branching overhead makes it slower for homogeneous scenes
- Useful only when mixing physics and kinematic objects

## Known Historical Issues

Issues discovered during development (from git history):

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| Static objects re-checked every frame | `_moved_this_step` flag not filtered for STATIC | Skip STATIC objects in movement tracking |
| Cell size = 0 crash | Object with zero-size AABB (degenerate) | Clamp minimum cell size to 0.1 |
| Empty collision with physics=false | Using CONTACT_POINTS without stepSimulation() | Auto-select CLOSEST_POINTS when physics=false |
| Collision miss after teleport | `set_pose()` not marking object as moved | `set_pose()` now calls `_mark_object_moved()` |
| Excessive memory with many cells | Very small cell_size with large world | Added AUTO_INITIAL mode with reasonable defaults |

## Key Variables in core_simulation.py

To understand or debug collision, search for these:

```bash
grep -n "_cached_spatial_grid\|_cached_aabbs_dict\|_moved_this_step\|_kinematic_objects\|_physics_objects\|_active_collision_pairs\|check_collisions\|filter_aabb_pairs" pybullet_fleet/core_simulation.py
```

| Variable | Type | Purpose |
|----------|------|---------|
| `_cached_spatial_grid` | `Dict[Tuple, Set[int]]` | Spatial hash grid |
| `_cached_aabbs_dict` | `Dict[int, Tuple]` | Cached AABBs |
| `_moved_this_step` | `Set[int]` | Objects moved since last collision check |
| `_kinematic_objects` | `Set[int]` | Kinematically controlled objects |
| `_physics_objects` | `Set[int]` | Physics-enabled objects |
| `_active_collision_pairs` | `Set[Tuple]` | Currently colliding pairs |
| `_collision_modes` | `Dict[int, CollisionMode]` | Per-object mode |
