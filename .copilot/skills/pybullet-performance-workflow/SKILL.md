---
name: pybullet-performance-workflow
description: "Use when optimizing PyBulletFleet performance - profiling slow simulations, tuning parameters, running benchmarks, investigating memory or CPU bottlenecks, or comparing simulation configurations"
---

# PyBulletFleet Performance Workflow

Structured benchmark→profile→optimize→verify cycle for PyBulletFleet.

## The Iron Law

```
NEVER optimize without measurements.
Benchmark BEFORE. Profile to find the bottleneck. THEN optimize. THEN verify.
```

If you haven't measured, you cannot optimize. If you optimized multiple things at once, you can't attribute improvement. One change at a time.

## The Four Phases

```
Phase 1: MEASURE         Phase 2: PROFILE
  baseline numbers  ──→    find the bottleneck
        │                        │
        │                        ▼
Phase 4: VERIFY           Phase 3: OPTIMIZE
  re-run baseline   ◀──    implement fix (one change)
  compare numbers
```

### Phase 1: MEASURE (Baseline)

Run benchmarks to establish current performance:

```bash
# Single benchmark
python benchmark/run_benchmark.py --agents 1000 --duration 10 --repetitions 3

# Sweep agent counts
python benchmark/run_benchmark.py --sweep 100 500 1000 2000

# Compare scenarios
python benchmark/run_benchmark.py --compare no_collision collision_10hz collision_3d_full

# With specific config
python benchmark/run_benchmark.py --config benchmark/configs/general.yaml
```

Save results — you'll compare against these after optimization.

**Key metrics:** avg_step_time (ms), FPS, memory RSS, spawn_time, CPU%

### Phase 2: PROFILE (Find Bottleneck)

**CPU profiling:**
```bash
# Collision-specific profiling
python benchmark/collision_check_profile.py

# General cProfile
python -m cProfile -s cumulative your_script.py | head -40
```

**Time profiling (built-in):**
```yaml
# In config:
enable_time_profiling: true
profiling_interval: 100
log_level: debug
```

This prints per-step breakdown: agent_update, callbacks, step_simulation, collision_check, monitor_update.

**Memory profiling:**
```yaml
enable_memory_profiling: true
```

**What to look for:**
- Which phase dominates step time? (collision? agent update? callbacks?)
- Is memory growing over time? (leak)
- Does performance degrade with agent count? (scaling issue)

### Phase 3: OPTIMIZE (One Change)

Apply **one** change at a time. Use **test-driven-development** for the change.

Common optimization levers:

| Lever | Config/Code | Impact |
|-------|-------------|--------|
| Disable GUI | `gui: false` | 2-3x faster |
| Disable monitor | `monitor: false` | Removes matplotlib overhead |
| Reduce collision frequency | `collision_check_frequency: 10` | Major if collision dominates |
| Use NORMAL_2D | Per-object CollisionMode | No z margin; fewer z cells per body |
| Mark static objects | `CollisionMode.STATIC` | Skip AABB updates |
| Disable unnecessary collision | `CollisionMode.DISABLED` | Remove from spatial hash |
| Increase timestep | `timestep: 0.1` | Fewer steps per second |
| Tune cell size | `spatial_hash_cell_size_mode: constant` | Optimal broad-phase |
| Simple shapes | Use cubes instead of mesh | Faster shape creation |

### Phase 4: VERIFY (Compare)

Re-run **exactly** the same benchmark as Phase 1. Compare numbers.

- Improved? → Commit the change.
- No improvement? → Revert. Profile again.
- Regression? → Revert immediately.

## Red Flags — STOP

| Red Flag | Action |
|----------|--------|
| Optimizing without profiling | STOP → Phase 2 first |
| Multiple changes at once | STOP → One change, measure, repeat |
| GUI ON during benchmark | STOP → `gui: false`, `monitor: false` |
| Comparing different agent counts | STOP → Use identical benchmark params |
| "It feels faster" | STOP → Numbers or nothing |

## Benchmark CLI Quick Reference

```bash
# Core options
--agents N          # Agent count (default: 1000)
--duration N        # Seconds (default: 10.0)
--repetitions N     # Runs for stats (default: 3)
--gui               # Enable GUI (avoid for benchmarks)
--config PATH       # Config file

# Analysis modes
--sweep N1 N2 ...   # Test at multiple agent counts
--compare S1 S2 ... # Compare named scenarios
--scenario NAME     # Use predefined scenario
```

**Predefined scenarios** (in `benchmark/configs/general.yaml`):

| Scenario | What it tests |
|----------|---------------|
| `no_collision` | Baseline — collision disabled |
| `collision_3d_full` | Every-step 3D collision |
| `collision_10hz` | 10 Hz collision checking |
| `stress_5k_agents` | 5000 agents scaling test |

## Cross-References

- **REQUIRED:** working-with-pybullet-fleet — codebase domain knowledge
- **pybullet-collision-tuning** — when bottleneck is collision detection
- **systematic-debugging** — when investigating unexpected performance issues
- **verification-before-completion** — verifying optimization results
- **Details:** See [references/benchmark-reference.md](references/benchmark-reference.md) for performance targets and known bottlenecks
