# Time Profiling User Guide

## Overview

Time profiling measures **per-step execution time** of each simulation component inside `step_once()`.
It helps identify performance bottlenecks by breaking down where time is spent every simulation step.

| Component | What it measures |
|-----------|-----------------|
| `agent_update` | All `Agent.update(dt)` calls (kinematics, path following, actions) |
| `callbacks` | `register_callback()` callbacks of the default `STEP` phase |
| `step_simulation` | `p.stepSimulation()` (physics mode) or AABB/spatial grid updates |
| `collision_check` | Spatial-hash broad phase + PyBullet narrow phase |
| `monitor_update` | `DataMonitor` overlay refresh |
| `total` | End-to-end step time |

**Overhead**: < 0.1% CPU — safe to leave enabled during development.

### ROS 2 Bridge state publication

When a bridge configuration enables the Fleet API state publisher, the same
profile also includes these fields for every endpoint that actually publishes
in that step:

| Component | What it measures |
|-----------|-----------------|
| `fleet_state_collect` | Build transport-neutral `RobotState3D` values from the selected agents. |
| `fleet_state_message` | Convert those values into the ROS `FleetState` message. |
| `fleet_state_publish` | The local `rclpy` `Publisher.publish()` call. This is not DDS delivery time. |

The three values are summed across the global and all active manager-scoped
endpoints. `events_post_step` remains the complete bridge callback cost,
including `/clock`, other ROS interfaces, and plugins. Use the optional
transport probe separately when measuring same-host DDS delivery time.

---

## Quick Start

### 1. Enable Time Profiling in Configuration

```yaml
# config.yaml
enable_time_profiling: true   # Enable step-timing reports
profiling_interval: 100       # Print average every 100 steps
log_level: "info"             # Required: reports use logger.info()
```

### 2. Run Your Simulation

```python
from pybullet_fleet.core_simulation import SimulationParams, MultiRobotSimulationCore

params = SimulationParams.from_config("config/config.yaml")
# Or directly:
params = SimulationParams(
    enable_time_profiling=True,
    profiling_interval=100,
)
sim = MultiRobotSimulationCore(params)
```

### 3. Read the Output

Every `profiling_interval` steps, a summary line is logged:

```
[PROFILING] Last 100 steps average: agent_update=0.45ms (35.2%), callbacks=0.02ms (1.6%),
step_simulation=0.00ms (0.0%), collision_check=0.75ms (58.6%), monitor_update=0.06ms (4.7%),
total=1.28ms (100.0%)
```

Each component shows **average time (ms)** and **percentage of total**.

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_time_profiling` | bool | `false` | Enable per-step timing measurement |
| `profiling_interval` | int | `100` | Steps between summary reports |

Both parameters can be set via YAML config or `SimulationParams` constructor.

`profiling_interval` is shared with memory profiling — it controls the reporting
interval for both systems.

---

## Interpreting Results

### Healthy Profile (kinematics mode, 100 robots)

```
[PROFILING] Last 100 steps average: agent_update=0.45ms (35.2%), collision_check=0.75ms (58.6%), total=1.28ms
```

- `collision_check` dominates — normal for large scenes
- `step_simulation` ≈ 0 — expected in physics-off mode
- Total < 2ms per step — good performance

### Bottleneck: Collision Detection

```
[PROFILING] Last 100 steps average: agent_update=0.50ms (10.0%), collision_check=4.50ms (90.0%), total=5.00ms
```

- `collision_check` > 80% → optimize spatial hash cell size or reduce collision-enabled objects
- See [Collision Configuration](collision-config) for tuning

### Bottleneck: Agent Update

```
[PROFILING] Last 100 steps average: agent_update=3.20ms (80.0%), collision_check=0.75ms (18.8%), total=4.00ms
```

- `agent_update` > 70% → check action complexity, path length, or number of attached objects

### Bottleneck: Physics Step

```
[PROFILING] Last 100 steps average: step_simulation=8.00ms (72.7%), collision_check=2.50ms (22.7%), total=11.00ms
```

- `step_simulation` dominates → expected in physics-on mode
- Consider switching to kinematics mode if physics fidelity is not required

---

## Programmatic Access

Use `return_profiling=True` on `step_once()` to get timing data as a dictionary
instead of logging:

```python
for _ in range(1000):
    timings = sim.step_once(return_profiling=True)
    if timings:
        print(f"Agent update: {timings['phase1_update']:.2f}ms")
        print(f"Collision:    {timings['collision_check']:.2f}ms")
        print(f"Total:        {timings['total']:.2f}ms")
```

### Return value fields

All times are in milliseconds.

| Key | What it measures |
|-----|-----------------|
| `phase1_update` | Full Phase 1: pre-step events + batch-controller advance + object update + callbacks + plugin hooks (framework base-pose writes are buffered) |
| `phase2_pose_flush` | Phase 2: flush buffered poses to PyBullet |
| `phase3_aabb_grid_flush` | Post-commit synchronization: kinematic AABB refresh + spatial-grid update (the metric name is retained for compatibility) |
| `agent_update` | Per-object update loop only (subset of `phase1_update`, kept for backward compat) |
| `callbacks` | `register_callback()` callbacks of the default `STEP` phase |
| `step_simulation` | `p.stepSimulation()` plus physics-object AABB/grid refresh — non-zero only when `physics=True` |
| `collision_check` | Broad phase (spatial hash) + narrow phase total |
| `collision_breakdown` | Nested dict — present only when collision ran (see below) |
| `events_pre_step` | EventBus `pre_step` dispatch |
| `events_step` | EventBus `step` handlers (before the `STEP` callbacks; plugin `on_step` is counted only in `phase1_update`) |
| `events_post_step` | EventBus `post_step` handlers |
| `callbacks_post_step` | `register_callback(..., phase="post_step")` callbacks |
| `events_collision` | EventBus collision-pair dispatch |
| `monitor_update` | DataMonitor overlay refresh |
| `total` | End-to-end step time |

`collision_breakdown` sub-fields:

| Key | What it measures |
|-----|-----------------|
| `get_aabbs` | Fetch AABB bounds from PyBullet |
| `spatial_hashing` | Rebuild spatial hash buckets |
| `aabb_filtering` | Broad-phase candidate pair filtering |
| `contact_points` | Narrow-phase contact query (`physics=True` only) |
| `total` | Sum of collision sub-phases |

### Example: per-step CSV logging

```python
import csv, sys

writer = csv.DictWriter(sys.stdout,
    fieldnames=["step", "phase1_update", "collision_check", "total"])
writer.writeheader()

for i in range(500):
    t = sim.step_once(return_profiling=True)
    if t:
        writer.writerow({"step": i,
                         "phase1_update": round(t["phase1_update"], 3),
                         "collision_check": round(t["collision_check"], 3),
                         "total": round(t["total"], 3)})
```

This is useful for custom analysis, plotting, or CI performance regression tests.

### Latest completed result during `run_simulation()`

When using `run_simulation()`, enable time profiling and read
`sim.last_profiling` from a callback or `SimPlugin`. It is a read-only mapping
for the **previous completed step**, so all phases (including physics and
collision detection) have finished. It has the same fields and millisecond
units as `step_once(return_profiling=True)`.

```python
def adaptive_monitor(sim, dt):
    previous = sim.last_profiling
    if previous is not None and previous["collision_check"] > 5.0:
        print(f"collision work: {previous['collision_check']:.2f} ms")


sim.register_callback(adaptive_monitor, frequency=10.0)
sim.run_simulation()
```

The property is `None` until one profiled step has completed, and remains
`None` while profiling is disabled. Use `step_once(return_profiling=True)`
when every individual result must be retained or exported.

---

## Best Practices

1. **Always enable during development** — overhead is negligible (< 0.1%)
2. **Disable for production** — avoids log noise
3. **Use `profiling_interval=100`** for stable averages; lower values (10-50) for debugging spikes
4. **Combine with memory profiling** for comprehensive analysis:
   ```yaml
   enable_time_profiling: true
   enable_memory_profiling: true
   profiling_interval: 100
   ```
5. **Use `return_profiling=True`** for automated benchmarks instead of parsing logs

---

## Troubleshooting

### No profiling output appears

**Cause:** `enable_time_profiling` is not enabled or log level is too high.

**Solution:**
```python
params = SimulationParams(
    enable_time_profiling=True,
    profiling_interval=100,
    log_level="info",  # Must be info or lower
)
```

Or in YAML:
```yaml
enable_time_profiling: true
profiling_interval: 100
log_level: "info"
```

### Output appears too frequently / infrequently

**Cause:** `profiling_interval` does not match your needs.

**Solution:** Adjust `profiling_interval`:
- `10-50`: Fine-grained debugging
- `100`: Normal development (default)
- `500-1000`: Long-running simulations

---

## Benchmark scripts

For repeatable, process-isolated measurement use the scripts under `benchmark/`:

| Script | Purpose |
|--------|---------|
| `benchmark/batch_perf.py` | Controller and command-interface comparison with phase breakdown. Runs a fixed number of steps for stable statistics. |
| `benchmark/run_benchmark.py` | Multi-run sweep with subprocess isolation and median/stdev. Supports scenario comparison and controller/command-interface axes. |
| `benchmark/mobile_benchmark.py` | Worker process called by `run_benchmark.py`. Outputs JSON. |

The default mobile whole-system benchmark uses the expected fastest path from
`benchmark/configs/general.yaml`: `agents.batch_controller: batch_omni` and
`agents.command_interface: fleet`. Override those fields in a scenario or custom
config to compare against the legacy per-agent path. `--collision-freq` can
override the config value for mobile, arm, and mobile control-path runs.

**Controller vs command-interface comparison**:

| Axis | Values | Meaning |
|------|--------|---------|
| Controller implementation | `per_agent`, `batch` | How robot motion is computed inside the simulation. |
| Command interface | `per_agent`, `fleet` | How navigation commands are submitted to the simulation. |

```bash
# Quick comparison in one process (phase breakdown shown)
python3 benchmark/batch_perf.py --agents 1000 --mode omni \
    --controller per_agent batch --command-interface per_agent fleet

# Process-isolated comparison across agent counts
python3 benchmark/run_benchmark.py --type mobile_control_path \
    --controller per_agent batch --command-interface per_agent fleet \
    --sweep 100 500 1000 2000 --steps 300 --repetitions 3
```

The default for the new path is `--controller batch --command-interface fleet`.
Use `--steps 300` or more for stable means; very short runs can be dominated by
startup, GC, or one-time cache effects.

## References

- [Memory Profiling User Guide](memory-profiling) — Memory usage tracking
- [Collision Configuration](collision-config) — Collision settings and cell size tuning
