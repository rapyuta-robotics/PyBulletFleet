---
name: pybullet-collision-tuning
description: "Use when debugging collision detection issues, tuning collision parameters, adding collision behavior for new entities, or investigating false positives/negatives in PyBulletFleet simulations"
---

# PyBulletFleet Collision Tuning

Collision detection debugging, parameter tuning, and diagnostic guide.

## Before Touching Collision Code

**Always read these first:**

```bash
# Official design document (architecture, rationale, modes)
cat docs/COLLISION_DETECTION_DESIGN_v3.md

# All collision enums
grep -A5 "class Collision" pybullet_fleet/types.py

# Current config defaults
grep -A20 "Collision" config/config.yaml

# Spatial hash cell size documentation
cat docs/spatial_hash_cell_size_modes.md
```

## Architecture Quick Reference

```
Broad Phase: Spatial Hash Grid — O(N) filtering
  │
  ├── Objects hashed into cells by AABB
  ├── Each body in every cell its AABB (+ margin/2) touches
  ├── Only check pairs sharing a cell (no neighbour search)
  └── Incremental: only re-hash moved objects (_moved_this_step)
  │
  ▼
Narrow Phase: PyBullet API — per-pair verification
  │
  ├── CLOSEST_POINTS: getClosestPoints() — distance-based, works without physics
  ├── CONTACT_POINTS: getContactPoints() — contact manifold, needs stepSimulation()
  └── HYBRID: contact for physics objects, closest for kinematic
```

## CollisionMode Decision Table

| Entity Type | Recommended Mode | Why |
|------------|-----------------|-----|
| Ground mobile robot | `NORMAL_2D` | Z-axis margin not applied; pairs need real z overlap |
| Flying drone / 3D robot | `NORMAL_3D` | Full 3D detection needed |
| Wall / shelf / structure | `STATIC` | Never moves → skip AABB/grid updates |
| Visual marker / helper | `DISABLED` | Exclude from collision entirely |
| Carried object (during pick) | `DISABLED` | Parent agent handles collision |
| Conveyor belt surface | `STATIC` | Structure doesn't move |

```python
from pybullet_fleet.types import CollisionMode

# Set at spawn
params = AgentSpawnParams(collision_mode=CollisionMode.NORMAL_2D, ...)

# Change at runtime
obj.set_collision_mode(CollisionMode.DISABLED)
```

## CollisionDetectionMethod Selection

| Method | When to Use | Requires |
|--------|------------|----------|
| `CLOSEST_POINTS` | physics=false (default), safety margins | Nothing special |
| `CONTACT_POINTS` | physics=true, actual contact logging | `stepSimulation()` called |
| `HYBRID` | Mixed physics/kinematic objects | Both conditions |

**Auto-selection** (when method not specified in config):
- `physics: false` → CLOSEST_POINTS
- `physics: true` → CONTACT_POINTS

## Tuning Parameters

| Parameter | Config Key | Default | Guide |
|-----------|-----------|---------|-------|
| Collision margin | `collision_margin` | 0.02m | Object size × 5-10%. Too small → miss, too large → false positive |
| Cell size mode | `spatial_hash_cell_size_mode` | auto_initial | `constant` fastest, `auto_adaptive` most flexible |
| Fixed cell size | `spatial_hash_cell_size` | (auto) | 2× median object extent. Too small → many cells, too large → many objects per cell |
| Check frequency | `collision_check_frequency` | null (every step) | null=every step, 0=disabled, N=Hz. 10Hz usually sufficient |
| Ignore static | `ignore_static_collision` | true | Skip static-static pairs |
| Multi-cell threshold | `multi_cell_threshold` | 1.5 | Accepted for compatibility; not used |

## Diagnostic Flowcharts

### Collision Not Detected

```
Collision not detected?
├── Is CollisionMode DISABLED?
│   └── YES → Change to NORMAL_2D/3D
├── Is <collision> geometry defined in URDF?
│   └── NO → Add <collision> element
├── Is collision_check_frequency = 0?
│   └── YES → Set to null (every step) or positive Hz
├── Is collision_margin too small?
│   └── Try increasing to 0.05 or 0.1
├── Using CONTACT_POINTS with physics=false?
│   └── Switch to CLOSEST_POINTS
├── Is spatial_hash_cell_size too small?
│   └── Increase — objects may not share cells
└── Are objects actually overlapping?
    └── Check with enable_collision_shapes: true
```

### False Positive Collisions

```
Detecting collision when objects aren't touching?
├── Is collision_margin too large?
│   └── Decrease to 0.01 - 0.02
├── Is <collision> geometry larger than <visual>?
│   └── Shrink collision geometry to match visual
├── Using NORMAL_2D but objects at different Z heights?
│   └── Switch to NORMAL_3D
└── AABB overlap without actual geometric contact?
    └── Expected with spatial hash — narrow phase should filter
```

### Collision Performance Issues

```
Collision too slow?
├── Profile first: enable_time_profiling: true
│   └── What % of step time is collision_check?
├── Can objects be STATIC or DISABLED?
│   └── Mark non-moving objects as STATIC
├── Can NORMAL_3D be NORMAL_2D?
│   └── Ground robots don't need 3D checking
├── Is collision_check_frequency too high?
│   └── Try 10 Hz instead of every step
├── Is cell size appropriate?
│   └── constant mode + benchmarked cell size
└── Use pybullet-performance-workflow for systematic optimization
```

## Visualizing Collisions

```yaml
# See collision shapes in GUI
enable_collision_shapes: true

# Color change on collision
enable_collision_color_change: true
```

Keyboard shortcuts in GUI:
- `c` — toggle collision shape display
- `v` — toggle visual shape display
- `t` — toggle structure transparency

## Cross-References

- **REQUIRED:** working-with-pybullet-fleet — codebase domain knowledge
- **pybullet-performance-workflow** — when collision is the performance bottleneck
- **adding-sim-entities** — choosing collision mode for new entities
- **systematic-debugging** — root-cause investigation for collision bugs
- **Details:** See [references/collision-internals.md](references/collision-internals.md) for spatial hash implementation details
