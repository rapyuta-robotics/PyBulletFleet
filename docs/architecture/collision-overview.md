# Collision Detection Overview

A high-level overview of PyBulletFleet's collision detection system — design goals,
available modes, and the broad-phase/narrow-phase architecture.

---

## TL;DR

PyBulletFleet's collision detection system provides:

- ✅ **Scalability**: O(N) via spatial hashing (not O(N²))
- ✅ **Flexibility**: 4 collision modes per object
- ✅ **Performance**: Incremental updates, mode-based optimization
- ✅ **Determinism**: Same input → same output
- ✅ **Simplicity**: Automatic configuration, minimal setup

**Recommended Workflow**:
1. Start with default (physics=False, CLOSEST_POINTS)
2. Set collision modes (NORMAL_3D/2D, STATIC, DISABLED)
3. Run and profile
4. Optimize cell_size if needed
5. Use Physics mode when needed

---

## Overview & Design Philosophy

### Why This Design?

PyBulletFleet's collision detection system is designed for **scalability, determinism, and flexibility**:

**The Challenge**:
- Naive all-pairs collision detection: **O(N²)** complexity
- 1000 robots → 1,000,000 pairs to check every frame → **infeasible**

**Our Solution**:
- **Spatial Hash Grid** (Broad-Phase): O(N) → filters to ~1% of pairs
- **Incremental Updates**: Only check moved objects → 10-100x faster
- **Per-Object Modes**: Fine-grained control (3D, 2D, static, disabled)
- **Two-Mode Design**: Kinematics (fast) or Physics (accurate)

### Core Principles

1. **Scalability First**
   - O(N) broad-phase filtering via spatial hashing
   - Incremental updates (only moved objects)
   - Scales to **100-1000+ objects** in real-time

2. **Kinematics First, Physics When Needed**
   - **Default**: Pure kinematics (no `stepSimulation()`)
   - **Optional**: Physics mode

3. **Deterministic & Reproducible**
   - Same input → same output (critical for planning)
   - Stable at high speeds (10-100x real-time)
   - No race conditions or timing dependencies

4. **Per-Object Flexibility**
   - Each object can have different collision modes
   - Runtime mode changes supported
   - No global "one size fits all"

---

## Architecture Overview

### Two-Phase Collision Detection

PyBulletFleet uses a classic **Broad-Phase → Narrow-Phase** pipeline:

> This is the collision system's two-phase pipeline, not the simulator's
> two-phase pose-commit contract. Pose commit and post-commit synchronization
> happen before this pipeline runs in `step_once()`.

```
┌─────────────────────────────────────────────────────────────┐
│                    Collision Detection Pipeline             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: BROAD-PHASE (Spatial Hash Grid)                  │
│  ────────────────────────────────────                       │
│  Input:  N objects (e.g., 1000 robots)                     │
│  Goal:   Filter to candidate pairs                          │
│  Method: Spatial hash grid + AABB overlap                   │
│  Complexity: O(N)                                           │
│                                                             │
│  ↓                                                          │
│                                                             │
│  Phase 2: NARROW-PHASE (PyBullet APIs)                     │
│  ───────────────────────────────────                        │
│  Input:  Candidate pairs from Phase 1                       │
│  Goal:   Exact collision detection                          │
│  Method: getClosestPoints() or getContactPoints()           │
│  Output: Confirmed collision pairs                          │
│  Complexity: O(k) where k << N²                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

→ [Collision Detection Broad-Phase Details: Spatial Hash Grid](collision-spatial-hash)
→ [Collision Detection Narrow-Phase Details](collision-internals)

### Why Two Phases?

**Without Broad-Phase** (Naive All-Pairs):
```
1000 objects → 1000 × 999 / 2 = 499,500 pairs
Every frame: 499,500 × getClosestPoints()  ❌
```

**With Broad-Phase** (Spatial Hash):
```
1000 objects → Spatial grid → ~100 candidate pairs
Every frame: 100 × getClosestPoints()  ✅
```

**Performance Gain**: **500x faster** for large scenes!

### Spatial Hash Grid: How It Works

**Concept**: Divide 3D space into uniform grid cells (→ [full details](collision-spatial-hash))

```
World Space (top view):
┌─────┬─────┬─────┬─────┬─────┬─────┐
│  ·  │  ·  │  R1 │  ·  │  ·  │  ·  │  Cell size: 2.0m × 2.0m
├─────┼─────┼─────┼─────┼─────┼─────┤
│  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  R1: Robot at (4, 10)
├─────┼─────┼─────┼─────┼─────┼─────┤  R2: Robot at (2, 4)
│  ·  │  R2 │  W  │  ·  │  ·  │  ·  │  W: Wall at (4, 4)
├─────┼─────┼─────┼─────┼─────┼─────┤  R3: Robot at (10, 2)
│  ·  │  ·  │  ·  │  ·  │  R3 │  ·  │  Only check neighbors!
├─────┼─────┼─────┼─────┼─────┼─────┤
│  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │
└─────┴─────┴─────┴─────┴─────┴─────┘

R2 only checks the bodies registered in its own cell(s)
   → Finds W (wall), which is registered in every cell it covers
   → No need to check R1, R3 (no shared cell)
```

**Key Optimizations** (→ [Spatial Hash Grid](collision-spatial-hash)):
1. **[Incremental Updates](incremental-updates)**: Only update AABBs for moved objects
2. **[Registration and Candidate Search](multi-cell-registration)**: every body is in each cell its margin-enlarged AABB touches, and a moved body's candidates are the bodies sharing one of its cells

---

(system-flow-object-lifecycle)=
## System Flow & Object Lifecycle

This section explains **when and how** AABBs and spatial grids are updated throughout an object's lifetime.

**Note**: In standard usage with `run_simulation()`, `step_once()` handles
movement, physics, synchronization, and collision detection. You normally do
not call `check_collisions()` yourself.

### Object Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Object Creation (add_object / spawn_agent)               │
├──────────────────────────────────────────────────────────────┤
│  → Calculate initial AABB via getAABB()                      │
│  → Determine collision mode (NORMAL_3D/2D/STATIC/DISABLED)  │
│  → Register to spatial grid (if not DISABLED)                │
│  → Mark as "moved" for first collision check                 │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. Movement, pose commit, and synchronization                │
├──────────────────────────────────────────────────────────────┤
│  Kinematic set_pose() inside step_once():                    │
│  → Phase 1: update cached pose and mark the object moved     │
│  → Phase 2: resetBasePositionAndOrientation()                │
│  → Post-commit: refresh AABB and spatial-grid registration   │
│                                                              │
│  Mode-specific behavior:                                     │
│  • NORMAL_3D/2D: Refresh after every committed movement ✓   │
│  • STATIC: Never updated (optimization) ✓                   │
│  • DISABLED: Not in grid (skip) ✓                           │
│                                                              │
│  Physics mode:                                               │
│  • stepSimulation() updates all physics objects every step   │
│  • All physics objects automatically marked as moved         │
│  • Their AABBs and grid entries refresh after stepSimulation │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. Collision Detection (check_collisions)                    │
├──────────────────────────────────────────────────────────────┤
│  Step 1: Broad-Phase (Spatial Hash)                         │
│    → Iterate through _moved_this_step objects               │
│    → For each moved object:                                  │
│      - Get its grid cells                                    │
│      - Collect the bodies registered in those cells          │
│      - Collect potential pairs                               │
│    → AABB overlap test → Filter to candidates               │
│                                                              │
│  Step 2: Narrow-Phase (PyBullet API)                        │
│    → For each candidate pair:                                │
│      - getClosestPoints() or getContactPoints()              │
│      - Exact distance/contact check                          │
│    → Return confirmed collision pairs                        │
│                                                              │
│  Step 3: Cleanup                                             │
│    → Clear _moved_this_step                                 │
│    → Update _active_collision_pairs                         │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. Object Removal (remove_object)                            │
├──────────────────────────────────────────────────────────────┤
│  → Remove from _sim_objects_dict                            │
│  → Remove from spatial grid                                  │
│  → Remove from _moved_this_step                             │
│  → Clear AABB cache                                          │
└──────────────────────────────────────────────────────────────┘
```

### Standard step ordering

Use `run_simulation()` for normal applications, or call `step_once()` when an
application owns the loop:

```python
sim_core.step_once()
# 1. agent/controller/callback/plugin updates queue kinematic poses
# 2. framework commits queued poses to PyBullet
# 3. physics step (when enabled) and AABB/grid synchronization
# 4. broad-phase + narrow-phase collision detection (frequency-gated)
```

Outside `step_once()`, `set_pose(Pose(...))` updates a kinematic object's
PyBullet pose and collision caches immediately. Inside a controller, callback,
or plugin update, the same call is buffered until the commit/synchronization
sequence above. See [Two-Phase Step](two-phase-step) for the extension-author
rules.

---

## FAQ & Troubleshooting

### Q: How do I debug collision detection?

**A**: Several options:

1. **Enable collision visualization**:
```python
params = SimulationParams(gui=True, enable_collision_shapes=True)
# Or press PyBullet's built-in 'w' key to toggle wireframes at runtime.
```

2. **Check AABB caches**:
```python
aabb = sim_core._cached_aabbs_dict.get(obj.body_id)
print(f"AABB: {aabb}")
```

3. **Inspect spatial grid**:
```python
cells = sim_core._cached_object_to_cell.get(obj.body_id)
print(f"Object in cells: {cells}")
```

4. **Run comprehensive tests**:
```bash
python -m pytest tests/test_collision_comprehensive.py -v
```

### Q: Performance is slow. How to optimize?

**A**: Checklist:

1. ✅ Use STATIC for fixed objects (walls, structures)
2. ✅ Use DISABLED for visualization objects
3. ✅ Set appropriate cell_size (~2× object size)
4. ✅ Use NORMAL_2D instead of NORMAL_3D for ground robots
5. ✅ Enable physics=False for kinematics (huge speedup!)

**Benchmark**:
```bash
cd benchmark
python performance_benchmark.py
```

---

## Further Reading

- [Collision Detection Broad-Phase Details: Spatial Hash Grid](collision-spatial-hash) — Algorithm details, AABB caching, multi-cell registration
- [Collision Detection Narrow-Phase Details](collision-internals) — PyBullet APIs, collision mode implementation details, configuration
- [Collision Configuration Guide](../how-to/collision-config) — Practical configuration: detection method, modes, margins, cell size
- [Visual Collision Demo](collision-features-demo) — Observe collision modes, highlighting, and multi-cell registration

**Source code**:
- `pybullet_fleet/core_simulation.py` — Main implementation
- `pybullet_fleet/sim_object.py` — SimObject with collision modes
- `pybullet_fleet/types.py` — Enums and types
- `tests/test_collision_comprehensive.py` — Comprehensive tests
