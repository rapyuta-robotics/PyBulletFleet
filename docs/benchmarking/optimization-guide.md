# Performance Optimization Guide

Optimise from measurements of the actual workload. Robot models, moving-agent
ratio, collision settings, rendering, and command ingress all affect results.
Use [Benchmark Results](results) for reproducible measurements and their test
environment; this page explains how to locate and address a bottleneck.

## Optimisation workflow

1. Reproduce the workload headlessly where possible, with a fixed warm-up and
   measured window.
2. Collect built-in fields with `step_once(return_profiling=True)`, or enable
   profiling and inspect `sim.last_profiling` from a callback or plugin.
3. Identify the dominant field before changing settings.
4. Change one axis at a time and record the scenario and environment.

```python
for _ in range(100):                 # warm-up
    sim.step_once()

samples = [sim.step_once(return_profiling=True) for _ in range(300)]
mean_total_ms = sum(s["total"] for s in samples) / len(samples)
print(f"mean step: {mean_total_ms:.2f} ms")
```

The two-phase fields distinguish `phase1_update`, `phase2_pose_flush`, and
`phase3_aabb_grid_flush`. The legacy `agent_update` field is only the
per-object update loop, and is a subset of Phase 1.

## Pacing and timestep

| `target_rtf` | Behaviour | Typical use |
|---|---|---|
| `0` | No pacing sleep | Offline throughput measurement and batch processing |
| `1.0` | Target real-time pacing | Interactive control and visualisation |
| Other positive value | Target a faster or slower simulation-time / wall-time ratio | Scenario-specific integration testing |

`target_rtf` changes pacing, not computation cost. Use `0` to measure maximum
throughput and record the achieved RTF rather than assuming a value transfers
to another host.

Choose `timestep` for the required controller and physics fidelity. Larger
values reduce steps per simulation second but can reduce trajectory and physics
accuracy. Validate the value with the actual robot and contact behaviour.

`max_steps_per_frame`, `max_sleep_frames`, and `gui_min_fps` are pacing safety
and responsiveness controls, not normal throughput tuning knobs. See
[Real-time Synchronization](../architecture/realtime-sync) before changing them.

## Controllers and command ingress

Motion execution and command ingress are separate optimisation axes:

| Axis | Choice | What it changes |
|---|---|---|
| Motion execution | Per-agent controller or manager batch controller | How movement is computed in the simulation loop |
| Command ingress | Per-agent calls or `FleetCommandDispatcher` | How commands enter the simulation |

For a large, homogeneous mobile fleet, declare a named manager and select a
batch controller. It runs one vectorised `batch_advance()` in Phase 1; action
queues and other per-agent work still run normally.

```yaml
simulation:
  gui: false
  target_rtf: 0

managers:
  - name: delivery_fleet
    fleet_controller:
      type: batch_omni

entities:
  - urdf_path: robots/mobile_robot.urdf
    manager: delivery_fleet
    grid:
      count: 500
      spacing: [2.0, 2.0]
```

Use `FleetCommandDispatcher` when an application submits many commands as one
fleet-level request. It does not replace the controller: a manager's
`fleet_controller` selects execution, while the dispatcher changes command
ingress. Measure those axes independently when setup latency matters.

See [Controller Configuration](../how-to/controller-config),
[Fleet API](../architecture/fleet-api), and
[Batch Execution](../architecture/batch-execution).

## Collision work

`collision_check_frequency` controls collision scheduling in simulation time:

| Value | Behaviour |
|---|---|
| `null` | Check every simulation step |
| Positive Hz value | Check no more often than that simulation-time rate |
| `0` | Disable collision checks |

Select the rate from the required safety and response time, not an assumed
step-count speedup. For a timestep `dt`, frequency `f` has an intended interval
of `1 / f` simulation seconds.

Set `collision_mode` per entity:

- `normal_2d` for ground-constrained entities;
- `normal_3d` for full 3D motion;
- `static` for fixed structures;
- `disabled` only when an object must not participate in collision detection.

`ignore_static_collision: true` skips every pair involving `static` objects.
Use it only when structure collision is deliberately out of scope.

If collision is dominant, inspect `collision_breakdown` from
`step_once(return_profiling=True)`, then evaluate spatial-hash settings:

- `auto_initial`: choose cell size once after initial population;
- `auto_adaptive`: recalculate when objects are added or removed;
- `constant`: use a measured fixed `spatial_hash_cell_size`.

Every body occupies each cell it reaches, so the cell size alone sets the
trade-off: choose it against the representative scene.

See [Collision Configuration](../how-to/collision-config) and
[Collision Internals](../architecture/collision-internals).

## Rendering, monitors, and profiling

For headless throughput measurements, disable work not required by the
scenario:

```yaml
simulation:
  gui: false
  monitor: false
  enable_monitor_gui: false
  enable_time_profiling: false
  enable_memory_profiling: false
  enable_collision_shapes: false
  enable_shadows: false
```

Enable time profiling while locating a bottleneck, then disable it for the
final throughput measurement. Memory profiling uses `tracemalloc` and is for
allocation/leak investigation, not RTF comparisons. GUI rendering, DataMonitor,
recording, collision-shape wireframes, and shadows change the workload and
should be recorded as benchmark conditions.

## Diagnose common bottlenecks

| Dominant measurement | First checks |
|---|---|
| `phase1_update` / `agent_update` | Moving-agent ratio, controller chain, custom callbacks/plugins, batch eligibility |
| `phase2_pose_flush` | Number of kinematic pose writes; redundant `set_pose()` calls |
| `phase3_aabb_grid_flush` | Moved kinematic-object count, collision mode, spatial-hash settings |
| `collision_check` | Collision rate, static filtering, 2D versus 3D mode, cell-size strategy |
| `step_simulation` | Physics timestep, dynamic-body count, contact complexity |
| `callbacks` or custom field | Frequency and algorithmic cost of application code |
| `monitor_update` | Disable monitor/GUI for headless measurements |

For function-level attribution after finding a field, use the scripts in
`benchmark/profiling/` or `cProfile`. Profilers alter execution
characteristics, so use them to locate work rather than publish absolute
throughput values.

## Reproducible measurements

Run the release benchmark suite for comparable core results:

```bash
make bench-release
```

For controller and command-interface comparisons, use the mobile control-path
benchmark with fixed agent count, step count, and repetitions. See
[Benchmark Suite](benchmark-suite) and [Profiling Guide](profiling-guide) for
commands and methodology.

## See Also

- [Benchmark Results](results)
- [Benchmark Suite](benchmark-suite)
- [Profiling Guide](profiling-guide)
- [Time Profiling](../how-to/time-profiling)
- [Memory Profiling](../how-to/memory-profiling)
