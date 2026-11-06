# Profiling Guide

Standalone scripts for identifying performance bottlenecks in PyBulletFleet simulation components. These scripts are run from the command line against a live simulation.

All scripts live in `benchmark/profiling/`. For overall benchmark results and quick start, see `benchmark/README.md`.

> **Looking to add profiling to your own code?** See the [Time Profiling User Guide](../how-to/time-profiling) (API-based) or [Custom Class Profiling](../how-to/custom-profiling) (subclass profiling).

---

## Tool Summary

| Tool | Purpose | What It Measures |
|------|---------|-----------------|
| `simulation_profiler.py` | Step-level component breakdown | Agent Update, Collision Check, PyBullet Step, etc. |
| `collision_check.py` | Detailed collision detection analysis | AABBs, spatial hashing, candidate filtering, and narrow-phase work |
| `agent_update.py` | Detailed `Agent.update()` analysis | cProfile, manual, PyBullet API, stationary/moving, and controller comparison |
| `arm_joint_update.py` | Arm joint update profiling | Physics vs kinematic mode, scaling analysis |
| `agent_manager_set_goal.py` | Goal setting profiling | `set_goal_pose()` overhead and trajectory calculation |
| `wrapper_overhead.py` | Wrapper-layer overhead | Spawn time, update time, and memory: direct PyBullet vs SimObject vs Agent vs Manager |

## Measurement Methods by Script

Each profiling script uses one or more measurement techniques. The table below shows which `--test` options are available and what technique they use.

| Script | `--test` Option | Technique | Description |
|--------|----------------|-----------|-------------|
| `simulation_profiler.py` | `builtin` (default) | `time.perf_counter` | Per-component timing via `step_once(return_profiling=True)` |
| | `cprofile` | `cProfile` | Function-level call graph analysis |
| | `motion_modes` | `time.perf_counter` | `omni` vs `differential` controller comparison |
| `collision_check.py` | `builtin` | `time.perf_counter` | Collision-pipeline timing via `check_collisions(return_profiling=True)` |
| | `cprofile` | `cProfile` | Function-level analysis of collision path |
| `collision_check.py` | `all` (default) | Both | Runs built-in timing and cProfile |
| `agent_update.py` | `cprofile` | `cProfile` | Function-level call graph of `agent.update()` |
| | `manual` | `time.perf_counter` | Manual timing of update sub-steps |
| | `pybullet` | `time.perf_counter` | PyBullet API call timing (resetBasePositionAndOrientation, etc.) |
| | `stationary` | `time.perf_counter` | Stationary vs moving agent cost comparison |
| | `motion_modes` | `time.perf_counter` | Per-controller update cost (`omni` vs `differential`) |
| `agent_update.py` | `all` (default) | Mixed | Runs every listed analysis |
| `agent_manager_set_goal.py` | _(no option)_ | Both | `time.perf_counter` for wall time + `cProfile` for call graph (always runs both) |

## Measurement Method Comparison

| Attribute | cProfile | CPU Time | Wall Time |
|-----------|----------|----------|-----------|
| **Goal** | Bottleneck identification | CPU usage measurement | Real-time measurement |
| **Granularity** | Function level | Process-wide | Process-wide |
| **Overhead** | Yes (high if many calls) | Almost none | Almost none |
| **Detail** | High (Python layer) | Low | Low |
| **Stability** | Medium | High (same environment) | Low (environment-sensitive) |
| **Use case** | Find optimization targets | Compare CPU work when a script reports it | Compare measured sections within one run |

- **cProfile** — Function-level call counts and cumulative times. It perturbs
  timings and cannot see inside PyBullet C++ internals; use it to locate Python
  call paths, not to report throughput.
- **CPU Time** (`psutil` / `time.process_time()`) — Actual CPU consumption.
  `wrapper_overhead.py` reports it alongside wall time; it is useful as one
  before/after signal under the same workload.
- **Wall Time** (`time.perf_counter()`) — Elapsed time for the section a script
  measures. It is not, by itself, the simulation-loop RTF reported by
  `run_benchmark.py`.
- **`step_once(return_profiling=True)`** — Built-in profiling in
  `MultiRobotSimulationCore` that returns a per-component timing dict. The same
  completed-step snapshot is available through `sim.last_profiling`, including
  custom fields recorded with `record_profiling()`.

## When to Use Each Tool

1. **Identify overall bottleneck** → `simulation_profiler.py`
2. **Collision detection is slow** → `collision_check.py`
3. **Agent Update is slow** → `agent_update.py`
4. **Goal setting is slow** → `agent_manager_set_goal.py`
5. **Wrapper-layer overhead?** → `wrapper_overhead.py`

---

## Simulation Profiler (`simulation_profiler.py`)

Component-level time measurement and bottleneck identification within
`step_once()`. Unlike `run_benchmark.py`, which reports an aggregate
simulation-loop benchmark, this tool reports per-component statistics for a
fixed number of direct steps.

### Measured Components

| Component | Description |
|-----------|-------------|
| Agent Update | State updates for all agents (trajectory following and controller work) |
| Collision Check | Collision detection for the configured scene and collision mode |
| PyBullet Step | Physics-engine work when physics is enabled |
| Monitor Update | Data-monitor work when enabled |

### Analysis Methods

| Method | Command | Purpose |
|--------|---------|---------|
| Built-in Profiling | `--test=builtin` (default) | Component time distribution from `step_once()` |
| cProfile | `--test=cprofile` | All-function bottleneck search |
| Controller comparison | `--test=motion_modes` | `differential` vs `omni` controller comparison |

### CLI Usage

```bash
# Built-in profiling (default)
python benchmark/profiling/simulation_profiler.py --agents=1000 --steps=100

# Detailed analysis with cProfile
python benchmark/profiling/simulation_profiler.py --agents=1000 --test=cprofile

# Controller comparison
python benchmark/profiling/simulation_profiler.py --agents=1000 --test=motion_modes

# Run all analyses
python benchmark/profiling/simulation_profiler.py --agents=1000 --test=all
```

### Output Example

```text
Step Breakdown (<controller>): <agents> agents (<steps> steps)

Agent Update:
  Mean:   <mean_ms> ms (<share>%)
  Median: <median_ms> ms

Collision Check:
  Mean:   <mean_ms> ms (<share>%)

Total Step Time:
  Mean:   <mean_ms> ms
```

The report prints mean, median, standard deviation, and range for each
measured component. Compare runs only when agent count, controller type,
physics, collision configuration, and step count are the same. Values depend
on how many agents are moving and on the host environment.

### Follow-Up Analysis

- Agent Update is slow → use `agent_update.py` for detailed analysis
- Collision Check is slow → use `collision_check.py` for detailed analysis

---

## Collision Check Profiler (`collision_check.py`)

Breaks the collision detection pipeline into 4 steps for bottleneck identification.

### 4-Step Breakdown

| Step | Description |
|------|-------------|
| Get AABBs | Fetch bounding boxes from PyBullet |
| Spatial Hashing | Update/query the spatial grid |
| AABB Filtering | Select candidate pairs |
| Narrow phase | Run the configured collision query, when applicable |

### CLI Usage

```bash
# Built-in profiling
python benchmark/profiling/collision_check.py --agents=1000 --iterations=100

# Detailed analysis with cProfile (function level)
python benchmark/profiling/collision_check.py --agents=1000 --test=cprofile

# Run both
python benchmark/profiling/collision_check.py --agents=1000 --test=all
```

### Output Example

```text
Collision Check Breakdown: <agents> agents (<iterations> iterations)

Get AABBs:      mean=<mean_ms> ms  median=<median_ms> ms
Spatial Hashing: mean=<mean_ms> ms  median=<median_ms> ms
AABB Filtering:  mean=<mean_ms> ms  median=<median_ms> ms
Narrow phase:    mean=<mean_ms> ms  median=<median_ms> ms
Total:           mean=<mean_ms> ms
```

Use `--test=cprofile` to drill into what is slow *inside* the measured path
(for example, AABB overlap checks or grid dictionary lookups). Interpret the
reported phases for the selected collision configuration rather than assuming a
fixed percentage split.

### Optimization Hints

- If candidate filtering dominates, inspect grid cell size and object density.
- If narrow-phase work dominates, profile the configured collision method and
  reduce check frequency only when the application's detection requirements
  allow it.

---

## Agent Update Profiler (`agent_update.py`)

`Agent.update()` runs every frame for every agent. This tool provides 5 analysis methods.

### Five Analysis Methods

| Method | Command | Purpose | Overhead |
|--------|---------|---------|----------|
| cProfile | `--test=cprofile` | All-function bottleneck search | Perturbs timing |
| Manual Timing | `--test=manual` | Precise measurement of specific methods | Low, workload-dependent |
| PyBullet API | `--test=pybullet` | C++ API cost measurement | Low, workload-dependent |
| Stationary vs Moving | `--test=stationary` | Impact of movement/update processing | None |
| Controller comparison | `--test=motion_modes` | `differential` vs `omni` | None |

### CLI Usage

```bash
# Run all analyses
python benchmark/profiling/agent_update.py --agents=1000 --updates=100

# Individual tests
python benchmark/profiling/agent_update.py --agents=1000 --test=cprofile
python benchmark/profiling/agent_update.py --agents=1000 --test=manual
python benchmark/profiling/agent_update.py --agents=100  --test=pybullet
python benchmark/profiling/agent_update.py --agents=1000 --test=stationary
python benchmark/profiling/agent_update.py --agents=1000 --test=motion_modes
```

### Output Example

```text
# --test=stationary
Stationary vs Moving: <agents> agents
<state>: total=<total_ms> ms  per_agent=<per_agent_us> us

# --test=motion_modes
Controller comparison: differential vs omni
<controller>: total=<total_ms> ms  per_agent=<per_agent_us> us
```

Each method produces a focused report:

- **Manual Timing** — per-component mean/median/max for controller and action-update work
- **PyBullet API** — call counts, total time, and per-call average for each PyBullet function
- **Stationary vs Moving** — total time and per-agent cost for idle vs. moving agents, plus overhead ratio
- **Controller comparison** — side-by-side `differential` vs `omni` controller cost comparison

### When to Use Which Method

1. **Bottleneck identification** → `--test=cprofile` — find slow functions
2. **Optimization verification** → `--test=manual` — accurate before/after comparison
3. **PyBullet API optimization** → `--test=pybullet` — identify expensive API calls
4. **Stationary agent impact** → `--test=stationary` — quantify movement overhead
5. **Controller comparison** → `--test=motion_modes` — compare `differential` and `omni`

---

## Goal Setting Profiler (`agent_manager_set_goal.py`)

Uses cProfile to analyze `AgentManager.set_goal_pose()` and the trajectory calculation call chain.

### CLI Usage

```bash
python benchmark/profiling/agent_manager_set_goal.py --agents=1000
```

### Reading Output

The cProfile report identifies the call chain used to prepare goals and paths.
Focus on cumulative time under the current controller and goal distribution;
there is no fixed function or percentage that is expected to dominate every
scene.

---

## Typical Bottlenecks and Fixes

### 1. Collision check is a material part of the measured step

**Fixes:**
- Reduce `collision_check_frequency` only after deciding the permitted
  detection latency.
- Use `ignore_static_collision=True` only when static-structure collisions are
  intentionally out of scope.

**Verification:**

```bash
python benchmark/profiling/collision_check.py --agents=1000
```

### 2. Agent update dominates the measured step

**Fixes:**
- Skip updates for stationary agents
- Reduce PyBullet API calls (caching)

**Verification:**

```bash
python benchmark/profiling/agent_update.py --agents=1000 --test=stationary
python benchmark/profiling/agent_update.py --agents=100  --test=pybullet
```

### 3. Goal setup is slow for the target fleet size

**Fixes:**
- Cache trajectory calculations
- Use simplified trajectory generation

**Verification:**

```bash
python benchmark/profiling/agent_manager_set_goal.py --agents=1000
```

---

## Wrapper Overhead (`wrapper_overhead.py`)

Measures the overhead introduced by PyBulletFleet wrapper classes relative to bare PyBullet API calls. Useful for detecting regressions after refactoring SimObject, Agent, or Manager layers.

### What It Measures

| Test | Description |
|------|-------------|
| Direct PyBullet (baseline) | Bare `createMultiBody` / `resetBasePositionAndOrientation` |
| SimObject Wrapper | `SimObject.spawn()` + `get_pose()` + `set_pose()` |
| SimObjectManager (Bulk) | `SimObjectManager.spawn_grid()` + bulk pose operations |
| Agent Wrapper | `Agent.from_params()` + `get_pose()` + `set_pose()` |
| AgentManager (Bulk) | `AgentManager.spawn_agents_grid()` + `get_all_poses()` |

Each test runs in a **separate child process** (via `subprocess`) to eliminate cross-test contamination and measure clean-state memory.

### CLI Usage

```bash
# Default: 10000 objects, 5 repetitions
python benchmark/profiling/wrapper_overhead.py

# Custom object count and repetitions
python benchmark/profiling/wrapper_overhead.py --n=1000 --reps=3
```

### Metrics

- **Spawn time** — wall-clock and CPU (user+sys) time to create N objects
- **Update time** — wall-clock and CPU time for get_pose + set_pose per step
- **Memory overhead** — RSS delta between before/after spawn (MB, per-object KB)
- **CPU utilization** — `cpu_time / wall_time` ratio (stability indicator)
- **Reference extrapolation** — scaled to the script's `N_MAX_OBJECTS` setting.
  Its thresholds are local reference values, not production acceptance criteria.

---

## Troubleshooting

### Profiling logs are not displayed

**Cause:** `enable_time_profiling=False` or the logger suppresses `INFO` output.

**Fix:**

```python
params = SimulationParams(
    enable_time_profiling=True,
    log_level="info"
)
```

Or in config:

```yaml
simulation:
  enable_time_profiling: true
  log_level: info
```

### Segfault with cProfile

**Cause:** Compatibility issues between cProfile and PyBullet's C++ extension.

**Fix:**
- Use Manual Timing or Built-in Profiling instead
- Try `agent_update.py --test=manual`
- Use `agent_update.py --test=stationary` (does not use cProfile)

### High variance in measurement results

**Cause:** Background process interference, thermal throttling.

**Fix:**
- Run multiple iterations and compute statistics
- Increase iteration count with `--iterations`
- Check CPU utilization
