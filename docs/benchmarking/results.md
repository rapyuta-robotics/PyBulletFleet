# Benchmark Results Reference

Results are kept up-to-date with each release. For how to reproduce them,
see [Benchmark Suite](benchmark-suite) and [Profiling Guide](profiling-guide).

---

## Test Environment

| Item | Value |
|------|-------|
| OS | Ubuntu (Linux 6.6 microsoft-standard-WSL2) |
| CPU | AMD Ryzen AI 7 PRO 350 w/ Radeon 860M (8 cores / 16 logical) |
| RAM | 29 GB |
| Python | 3.12.3 |
| Conditions | Headless (`gui=false`), `physics=false`, `timestep=0.1`, 3 repetitions |

---

## Simulation Throughput

**Command:** `make bench-release` (runs the core throughput sweep and the mobile control-path comparison)
**Base config:** `benchmark/configs/general.yaml` — headless `simple_cube`
robots, `collision_check_frequency=null` (every step), `batch_omni` fleet
controller, and the fleet command interface
**Last measured:** 2026-09-01

| Agents | Step Time (ms) | RTF   | Spawn Time | Memory Delta |
|--------|----------------|-------|------------|--------------|
| 100    | 0.74 ± 0.02    | 135.4× | 62 ms     | +2.6 MB      |
| 250    | 1.82 ± 0.03    | 54.8×  | 160 ms    | +6.2 MB      |
| 500    | 4.26 ± 0.16    | 23.5×  | 346 ms    | +12.4 MB     |
| 1000   | 9.73 ± 0.03    | 10.3×  | 644 ms    | +24.9 MB     |
| 2000   | 24.70 ± 1.30   | 4.0×   | 1468 ms   | +47.9 MB     |

**Source:** `benchmark/results/benchmark_sweep_10.0s.json`

The 2026-09-01 sweep was collected on a WSL2 host using the lightweight
`simple_cube` model as the large-scale baseline. The worker currently gives
goals to half of the spawned agents; this workload is defined in
`mobile_benchmark.py`. Although `general.yaml` contains
`moving_agents_ratio` and `max_moving_agents`, the mobile worker does not yet
read those two settings.

The default
`collision_check_frequency=null` setting means collision checks run every
simulation step. More detailed robot models have higher spawn and update costs
and should be benchmarked separately. Timed simulation runs keep
`tracemalloc` disabled because Python allocation tracing materially distorts
step-time and RTF measurements; memory deltas are RSS-based.

**Metric scope:** RTF is the simulation-loop value
`duration / simulation_wall_s`. `simulation_wall_s` measures only the
`run_simulation()` call. It excludes Python process startup, PyBullet setup,
spawning, command setup, warmup, cleanup, and JSON aggregation. Therefore it
must not be read as the elapsed time of a complete benchmark worker.

## ROS 2 Bridge Scale Check

Measured 2026-08-02 with Docker, headless `simple_cube` robots, physics off,
`timestep=0.1`, `publish_rate=5 Hz`, and `target_rtf=0`; the `fleet` 1000-row
was refreshed on 2026-09-01. These are diagnostic single-run values and are
not comparable to the core simulation table above.

| Mode | Robots | Command result | Max RTF |
|------|--------|----------------|---------|
| `fleet` | 1000 | ack 0.550 s; all moved in 0.083 s | 10.98× |
| `per_robot` | 1000 | publish 0.217 s; 0/1000 moved in 60 s | 0.64× |
| `per_robot` | 100 | publish 0.056 s; all moved in 15.033 s | not measured |
| `hybrid` | 1000 | fleet ack 7.998 s; per-robot publish 0.220 s | 0.30× |
| `hybrid` | 1000 | fleet ack 1.862 s; all moved in 0.412 s (1 Hz, target RTF 1.0) | target-rate check |

The 1000-robot per-robot result shows why topic publication latency must not be
treated as end-to-end command latency. See `ros2_bridge/PERFORMANCE.md` for
the bridge-specific methodology and recommendations.

With the same generated 1000-robot static scenario, a 2026-08-16 matched
comparison measured 274.14x RTF for core-only, 239.08x with the Fleet bridge
but no FleetState subscriber, and 10.94x with one `/fleet/states` subscriber.
The dominant ROS cost is requested FleetState collection/conversion/publication,
not bridge residency. Full methodology and per-step costs are in
`ros2_bridge/PERFORMANCE.md`.

### Manager-Scoped State Check

Measured on the same Docker host with ten equally sized manager scopes,
`publish_rate=5 Hz`, `target_rtf=0`, and transport timing enabled. This is a
state-only diagnostic. The 1000-robot global, selective, and complete rows
were refreshed on 2026-08-16 after the FleetState message-conversion
optimization; the remaining rows are from 2026-08-13.

| Robots | Subscription case | Observed RTF | State p50 / p99 |
|-------:|-------------------|-------------:|----------------:|
| 100 | global / one stream | 83.24x | 1.90 / 23.60 ms |
| 100 | one 10-robot manager | 305.76x | 0.37 / 1.25 ms |
| 100 | all ten manager streams | 47.78x | 0.45--0.49 / 1.82--2.37 ms |
| 500 | global / one stream | 18.21x | 9.57 / 33.23 ms |
| 500 | one 50-robot manager | 114.83x | 1.02 / 2.65 ms |
| 500 | all ten manager streams | 16.22x | 1.07--1.15 / 2.56--3.17 ms |
| 1000 | global / one stream | 10.60x | 19.11 / 45.75 ms |
| 1000 | one 100-robot manager | 70.58x | 1.91 / 20.68 ms |
| 1000 | all ten manager streams | 10.17x | 1.89--2.04 / 17.18--32.96 ms |

Manager scopes allow the bridge to skip state conversion for unsubscribed
managers. They are therefore appropriate for separate operating fleets and
selective consumers. Replacing a global snapshot with subscriptions to every
manager stream adds message overhead and is slower in this single-process
bridge. Full measurements and the partitioning decision are in
`ros2_bridge/PERFORMANCE.md`.

## Mobile Control Path Comparison

**Script:** `benchmark/run_benchmark.py --type mobile_control_path --controller per_agent batch --command-interface per_agent fleet --sweep 100 500 1000 --steps 600 --repetitions 3`
**Source:** `benchmark/results/mobile_control_path_sweep_600steps.json`

This is a fixed-step micro-benchmark of the controller and command-ingress
paths, not an end-to-end simulation benchmark. It uses differential
controllers (`--mode diff`), `collision_freq=60`, five warmup steps, and 600
measured `step_once()` calls. Setup time begins after spawning; step time
includes the configured simulation-step work, including collision work.

The command runs the Cartesian product of the two controller implementations
and the two command interfaces. The table below intentionally shows the two
matched, representative paths: a fully per-agent path and a batch-controller
path reached through `FleetCommandDispatcher`. The JSON source also contains
the cross combinations (`per-agent` + fleet and batch + per-agent), which are
useful when isolating one axis at a time.

| Agents | Controller | Command Interface | Setup Time | Step Time | P95 Step |
|--------|------------|-------------------|------------|-----------|----------|
| 100    | per-agent  | per-agent         | 0.0022 s   | 0.114 ms  | 0.122 ms |
| 100    | batch      | fleet             | 0.0017 s   | 0.109 ms  | 0.118 ms |
| 500    | per-agent  | per-agent         | 0.0100 s   | 0.680 ms  | 0.397 ms |
| 500    | batch      | fleet             | 0.0085 s   | 0.519 ms  | 0.288 ms |
| 1000   | per-agent  | per-agent         | 0.0224 s   | 1.355 ms  | 1.084 ms |
| 1000   | batch      | fleet             | 0.0162 s   | 1.156 ms  | 0.723 ms |

**Memory note:** Memory delta is process RSS and should be treated as
environment-dependent, especially on WSL2.

**Scalability:** ~O(n^1.2) across the measured 100-2000 robot sweep on this
WSL2 host.

---

## Steady-State Scenarios

**Script:** `make bench-steady` (`run_benchmark.py --type steady`, worker
`benchmark/steady_bench.py`). Every robot is sent 200 m away, so the whole
fleet is still moving during the measured window: 20 warmup steps, then 150
measured steps, dt = 0.1 s, differential drive, batch controller, collision
check every step, headless. Values are the median over 3 repetitions of each
run's median step; the RTF is derived from that step. Measured in the
environment above at commit 2e2a86a (after #115), 2026-11-06, with nothing
else running on the machine.

| Scene | Agents | Static bodies | Step p50 | RTF | phase1 (agent update) | pose flush | AABB / grid | collision | Collision events in window |
|-------|--------|---------------|----------|-----|-----------------------|------------|-------------|-----------|----------------------------|
| sparse (2 m grid, all driving +x) | 1000 | 0 | 11.1 ms | 9.0x | 4.8 ms | 1.5 ms | 2.9 ms | 1.4 ms | 0 |
| dense (two groups head-on, 0.05 m margin, every tenth robot stopped) | 1000 | 0 | 8.7 ms | 11.6x | 3.8 ms | 1.1 ms | 2.1 ms | 1.4 ms | 610 started, 610 ended |
| carrying (one box attached to every robot) | 1000 | 0 | 27.0 ms | 3.7x | 10.3 ms | 3.5 ms | 6.8 ms | 5.7 ms | 0 |
| warehouse (aisles between floor tiles and bins) | 200 | 32,924 | 5.9 ms | 16.9x | 3.6 ms | 0.8 ms | 0.7 ms | 0.5 ms | 0 |
| idle (sparse layout, no goals) | 1000 | 0 | 0.42 ms | 241x | 0.4 ms | 0 | 0 | 0.01 ms | 0 |

The same scenes measured at 6ee451e on 2026-11-05, while other work ran on
the machine, gave 14.2 ms (sparse), 13.0 ms (dense), 50.1 ms (carrying),
7.4 ms (warehouse) and 0.45 ms (idle). Runs of 6ee451e and 2e2a86a alternated
in one quiet session agree within noise (sparse 10.2-11.5 ms, carrying
30.7-31.7 ms for both), so the commits between them do not change the step
time; the difference between the two days is the load on the machine. Compare
revisions by alternating them in one session, not by numbers from different
days.

What the scenes show:

- After #94 the collision check is no longer the largest phase at 1000 moving
  robots (1.4 ms); the per-object agent update (4.8 ms), the AABB / grid flush
  (2.9 ms) and the pose flush (1.5 ms) are.
- An attached object per robot more than doubles the step (11 to 27 ms): the
  child pose is propagated per object, and every child joins the pose flush,
  the AABB / grid flush and the collision check.
- With 33k static bodies and 200 movers, the agent update is still 3.6 ms:
  the per-object loop visits every static body each step. The spawn of that
  scene takes 17.3 s.
- Head-on traffic (610 contacts in 15 simulated seconds) adds little to the
  collision phase here; the narrow phase is cheap next to the broad phase.

Raw results: `benchmark/results/steady_sparse_dense_carrying_idle_1000agents.json`
and `benchmark/results/steady_warehouse_200agents.json`.

---

## Step Time Component Breakdown

**Script:** supplemental profiling with 1000 agents, 50% moving,
`collision_check_frequency=null` (collision checks every simulation step), and
the first 100 measured steps after warmup.

| Component | Mean Time | Share | Notes |
|-----------|-----------|-------|-------|
| Agent / controller update | 4.43 ms | ~46% | Dominant cost; much higher for moving than stationary |
| Pose flush | 0.98 ms | ~10% | Buffered kinematic pose writes to PyBullet |
| AABB / spatial-grid flush | 2.07 ms | ~22% | Refreshes moved-object collision state |
| Collision check | 2.07 ms | ~22% | AABB filtering dominates within collision |
| Monitor update | 0.04 ms | <1% | Near-zero if monitor GUI is disabled |
| Step simulation | 0 ms | 0% | Physics off; up to ~40% with physics on |

> **Profiler note:** Long profiling runs can understate moving-agent cost because
> robots may reach their goals and become stationary. Use a fixed first-window
> profile when comparing with the benchmark table.

---

## Agent Update Cost

**Script:** `benchmark/profiling/agent_update.py --agents 500`

### Stationary vs Moving (500 agents)

| State | Total time | Per agent |
|-------|-----------|-----------|
| Stationary | 0.29 ms | 0.57 μs |
| Moving | 61.27 ms | 122.5 μs |
| Ratio | **214×** | |

**→ Design decision:** Benchmarks use 50% moving agents as a representative workload.
Moving agent cost dominates: kinematics, trajectory following, and PyBullet API calls
only execute for agents with an active goal.

### Motion Mode Comparison (500 agents, all moving)

| Mode | Total time | Per agent | Relative |
|------|-----------|-----------|---------|
| OMNIDIRECTIONAL | 9.26 ms | 18.5 μs | 1× (baseline) |
| DIFFERENTIAL | 26.66 ms | 53.3 μs | 2.9× slower |

**→ Optimisation note (2026-03-12):** DIFFERENTIAL was previously ~5× slower (90.6 μs/agent).
Replacing scipy Slerp with a precomputed quaternion slerp and removing per-tick numpy allocations
reduced the ROTATE phase from 37 μs to 5 μs. See [Differential Drive Optimization](#differential-drive-optimization) below.

**→ Design decision:** Benchmarks default to OMNIDIRECTIONAL. DIFFERENTIAL requires
heading alignment computation which adds ~2.9× update cost.

---

## Wrapper Layer Overhead

**Script:** `benchmark/profiling/wrapper_overhead.py --n 500 --reps 3`
**Conditions:** 500 objects, process-isolated per layer

| Layer | Spawn time | Update time (get+set pose) | Memory (RSS delta) |
|-------|-----------|---------------------------|-------------------|
| Direct PyBullet (baseline) | 63 ms | 2.21 ms | 5.2 MB |
| SimObject | 106 ms (+68%) | 6.80 ms (+3.1×) | 9.2 MB |
| SimObjectManager | 207 ms (+229%) | 10.75 ms (+4.9×) | 9.3 MB |
| Agent | 237 ms (+276%) | 10.60 ms (+4.8×) | 11.5 MB |
| AgentManager | 255 ms (+305%) | 10.20 ms (+4.6×) | 11.6 MB |

**Extrapolated to 10,000 objects (production feasibility, thresholds: spawn < 10 s, update < 150 ms/step, memory < +200 MB):**

| Layer | Spawn | Update/step | Memory | Pass? |
|-------|-------|-------------|--------|-------|
| SimObject | 2.1 s | 136 ms | +81 MB | ✅ all pass |
| Agent | 4.7 s | 212 ms | +127 MB | spawn/mem ✅, update ❌ |

**→ Design decision:** SimObject layer is production-viable at 10,000 scale.
Agent layer exceeds the 150 ms/step update threshold at 10,000 (extrapolated 212 ms),
but comfortably passes at 5,000 or below. For ≥10,000 agents, consider using SimObject
directly or batching update calls.

---

## Collision Mode Comparison (DISABLED / NORMAL_2D / NORMAL_3D)

**Script:** `benchmark/experiments/collision_mode_comparison.py --agents 500 --iterations 200`

Historical results, measured before the same-cell broadphase. The neighbor
counts describe the broadphase of that time; both modes now search the same cells.

| Mode | Step time (mean) | Collision time | vs Disabled |
|------|-----------------|---------------|-------------|
| DISABLED | 1.37 ms | — | baseline |
| NORMAL_2D (9 neighbors, historical) | 2.43 ms | 0.21 ms | +77% |
| NORMAL_3D (27 neighbors, historical) | 2.26 ms | 0.39 ms | +64% |

Collision check breakdown (both modes): AABB Filtering accounts for ~97–99% of collision check time.
Spatial Hashing and Contact Points together are <3%.

**→ Design decision:** NORMAL_2D yields only modest speedup (7%) over NORMAL_3D at this scale
because AABB filtering dominates in both. These numbers predate the same-cell broadphase, after
which both modes search the same cells. NORMAL_2D is now deprecated; use NORMAL_3D.

---

## Collision Algorithm Comparison (Spatial Hashing vs Alternatives)

**Script:** `benchmark/experiments/collision_method_comparison.py --agents 100,500 --iterations 100`
**Conditions:** spacing=0.08m (dense), kinematic objects (mass=0)

| Method | 100 agents | 500 agents | Collisions detected | Valid? |
|--------|-----------|-----------|---------------------|--------|
| Spatial Hashing (current) | 8.0 ms | 92.1 ms | ✅ correct | ✅ |
| Brute Force AABB | 8.7 ms | 231.9 ms (2.5×) | ✅ correct | ✅ |
| `getClosestPoints` all-pairs | 23.0 ms | 370.6 ms (4.0×) | ✅ correct | ✅ |
| `getContactPoints()` no-args | 0.06 ms | 0.24 ms | ❌ 0 — invalid | ❌ |
| `getContactPoints(A,B)` pairwise | 12.1 ms | 190.3 ms | ❌ 0 — invalid | ❌ |

**Key finding:** `getContactPoints` variants return 0 collisions for kinematic objects (mass=0).
PyBullet's contact point solver only activates between physics-enabled bodies.
Spatial hashing with `getClosestPoints` is the only valid fast method for kinematic AGV-type robots.

**→ Design decision:** Spatial hashing (O(N) average) is used because:
1. It is the only algorithmically valid method for kinematic robots.
2. It is 2.5–4× faster than the valid alternatives at 500 agents.

---

## Differential Drive Optimization

**Script:** `benchmark/profiling/differential_breakdown.py`
**Conditions:** 1000 agents, DIFFERENTIAL mode (ROTATE phase)

The differential drive ROTATE phase was the single most expensive per-agent per-tick operation.
Profiling identified two root causes:

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Quaternion slerp (scipy Slerp → `_quat_slerp`) | 26.8 μs | 2.3 μs | **11.8×** |
| Scalar clip (`np.clip` → `max/min`) | 6.8 μs | 0.15 μs | **44×** |
| Dead `np.array()` allocations (P0) | ~0.6 μs | 0 μs | eliminated |
| Redundant `np.dot` in slerp (P1) | ~0.3 μs | 0 μs | eliminated |
| Velocity array reallocation (P2) | ~0.3 μs | 0 μs | in-place |
| FORWARD direction recompute (P3) | ~0.5 μs | 0 μs | cached |
| **ROTATE phase total** | **36.1 μs** | **4.9 μs** | **7.3×** |

**End-to-end impact (1000 agents, differential):**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Per-agent update | 57 μs | 12.4 μs | 4.6× faster |
| DIFFERENTIAL / OMNIDIRECTIONAL ratio | 5.0× | 2.9× | Closer to parity |
| 500-agent ROTATE frame time | 18.0 ms | 2.5 ms | −15.6 ms/frame |

**Why was scipy Slerp slow?**

`scipy.spatial.transform.Slerp` is a generic array-oriented API designed for batch interpolation
(e.g., `slerp(np.linspace(0, 1, 10000))`). When called with a single scalar `t` per agent per tick,
the fixed overhead — input normalisation, dtype promotion, Rotation object wrapping — dominates
the actual trigonometry. The replacement `_quat_slerp` sidesteps this by precomputing
`dot/theta/sin(theta)` once per waypoint transition and using `math.sin`/`math.cos` (C-level scalar
functions) instead of numpy array dispatch.

Absolute timings vary by environment; the ratios above are the meaningful comparison metric.

---

## Collision Config-Based Comparison (Physics ON vs OFF)

**Script:** `benchmark/experiments/collision_methods_config_based.py`
**Conditions:** 100 objects, 500 steps

| Config | Physics | Method | Step time | Collision time |
|--------|---------|--------|-----------|---------------|
| `physics_off_closest.yaml` | OFF | `CLOSEST_POINTS` | 1.51 ms | 1.24 ms |
| `physics_on_contact.yaml` | ON | `CONTACT_POINTS` | 2.64 ms | 1.41 ms |
| `hybrid.yaml` | ON | `HYBRID` | 2.00 ms | 1.27 ms |

Physics ON adds ~75% step overhead due to the physics engine stepping each simulation step.

**→ Design decision:** Default is `physics=false` (kinematics mode) for maximum throughput.
Use `physics=true` only when rigid-body dynamics are required.

**Source:** `benchmark/results/collision_methods_config_based.txt`

---

## Arm Joint Control Performance

**Script:** `benchmark/profiling/arm_joint_update.py --test scaling`
**Conditions:** arm_robot.urdf (4 revolute joints), fixed-base, JointAction cycling, 100 steps per count

### Physics vs Kinematic Scaling

| Arms | Joints | Physics (ms/step) | Kinematic (ms/step) | Ratio |
|------|--------|--------------------|---------------------|-------|
| 1    | 4      | 0.029              | 0.010               | 2.8×  |
| 5    | 20     | 0.082              | 0.056               | 1.5×  |
| 10   | 40     | 0.152              | 0.090               | 1.7×  |
| 25   | 100    | 0.415              | 0.253               | 1.6×  |
| 50   | 200    | 0.886              | 0.552               | 1.6×  |

Kinematic mode is consistently faster than physics mode for joint control.
The gap comes from skipping `stepSimulation()` — kinematic mode uses `resetJointState()`
with per-step interpolation at URDF velocity limits.
Exact ratios are environment-dependent; the trend (kinematic faster) is consistent.

### Component Breakdown (10 arms)

**Script:** `benchmark/profiling/arm_joint_update.py --test builtin --arms 10`

| Component | Physics | Kinematic |
|-----------|---------|----------|
| agent_update | 27.0% (0.050 ms) | 96.7% (0.087 ms) |
| step_simulation | 69.6% (0.129 ms) | 0.2% (0.000 ms) |
| callbacks | 1.0% | 0.4% |
| **total** | **0.186 ms** | **0.090 ms** |

In physics mode, `stepSimulation()` dominates (70%). In kinematic mode, the physics engine
is bypassed entirely — agent_update (joint interpolation + `resetJointState`) is the sole cost.

### Kinematic Joint Cache Optimization

**Problem:** cProfile showed `p.getJointState()` consuming ~36% of kinematic update time
(called per-joint per-step to read current positions before interpolating).

**Solution:** `_kinematic_joint_positions` cache — joint positions stored in a Python dict,
initialized via batch `p.getJointStates()` at agent creation, updated after each `resetJointState()`.
`get_joint_state()` returns cached values for kinematic robots (zero PyBullet calls).

| Metric | Before cache | After cache | Improvement |
|--------|-------------|-------------|-------------|
| 50 arms step time | 0.826 ms | 0.552 ms | **1.5× faster** |
| `p.getJointState` calls/step | 200 (50 arms × 4 joints) | 0 | **eliminated** |

**→ Design decision:** Kinematic joint cache is always active for `mass=0.0` URDF robots.
The cache is invisible to callers — `get_joint_state()` API is unchanged.

*Data collected 2026-03-15. Absolute timings and ratios are environment-dependent
(CPU, OS, PyBullet version). The qualitative trends — kinematic faster than physics,
cache eliminating per-step PyBullet calls — are expected to hold across environments.
Re-run `benchmark/profiling/arm_joint_update.py` to obtain numbers for your setup.*

---

## See Also

- [Benchmark Suite](benchmark-suite) — How to run benchmarks and reproduce these results
- [Experiment Scripts](experiments) — Collision algorithm comparison scripts
- [Profiling Guide](profiling-guide) — Per-component profiling scripts
- [Optimization Guide](optimization-guide) — Parameter recommendations based on these results
