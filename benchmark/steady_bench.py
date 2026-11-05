"""Steady-state fleet benchmark: every robot keeps moving for the whole measurement.

``batch_perf.py`` and ``mobile_benchmark.py`` give each robot a short goal, so
part of the fleet stands still before the run ends and the step time is a mix
of moving and idle cost. This worker sends every robot 200 m away, warms up,
then measures a fixed window of steps in which the fleet is still moving, and
reports the per-phase step time of that window.

Scenarios (``--scenario``):

- ``sparse``: N robots on a 2 m grid, all driving +x. AABBs never overlap, so
  the collision cost is the broad phase alone.
- ``dense``: two groups of N/2 robots on 1 m rows drive head-on through each
  other with a 0.05 m collision margin; every tenth robot stands still as an
  obstacle. Exercises the narrow phase and the collision events.
- ``carrying``: ``sparse`` with one box attached to every robot's base, as a
  fleet carrying bins.
- ``warehouse``: N robots in the aisles of a field of static boxes: floor tiles
  on a 1 m grid over x in [-2, 230] and y in [-2, --field-width], and bins on
  three of four tiles of every odd row. Robots and statics never overlap in z
  or y, so every static pair is broad-phase work only.
- ``idle``: the ``sparse`` layout with no goals.

Run (default: 1000 robots, diff, sparse, 20 warmup + 150 measured steps)::

    python3 benchmark/steady_bench.py
    python3 benchmark/steady_bench.py --scenario warehouse --agents 200 --field-width 100
    python3 benchmark/steady_bench.py --scenario dense --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pybullet as p
from batch_perf import make_fleet, make_manager, make_sim, make_spawn_params
from pybullet_fleet import Agent, MultiRobotSimulationCore, Pose
from pybullet_fleet.agent_manager import SimObjectManager
from pybullet_fleet.commands import RobotGoalCommand2D
from pybullet_fleet.events import SimEvents
from pybullet_fleet.fleet_api import FleetCommandDispatcher
from pybullet_fleet.sim_object import CollisionMode, ShapeParams, SimObjectSpawnParams

SCENARIOS = ("sparse", "dense", "carrying", "warehouse", "idle")

TRAVEL_M = 200.0
DENSE_MARGIN_M = 0.05
DENSE_ROW_M = 1.0
DENSE_GROUP_GAP_M = 100.0
DENSE_STOPPED_EVERY = 10
LOAD_SHAPE = ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.15])
LOAD_LIFT_M = 0.45
TILE_SHAPE = ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.05])
BIN_SHAPE = ShapeParams(shape_type="box", half_extents=[0.22, 0.33, 0.17])
FIELD_X_RANGE = (-2, 231)


def _goal(agent: Agent, dx: float) -> RobotGoalCommand2D:
    pose = agent.get_pose()
    return RobotGoalCommand2D(name=agent.name, position=(pose.x + dx, pose.y), yaw=0.0 if dx > 0 else math.pi, z=pose.z)


def _static_box(shape: ShapeParams, x: float, y: float, z: float) -> SimObjectSpawnParams:
    return SimObjectSpawnParams(
        collision_shape=shape,
        visual_shape=shape,
        mass=0.0,
        collision_mode=CollisionMode.STATIC,
        initial_pose=Pose.from_xyz(x, y, z),
    )


def build_sparse(n: int, mode: str, collision_freq: int) -> tuple[MultiRobotSimulationCore, list[RobotGoalCommand2D], int]:
    sim, _, agents = make_fleet(n, collision_freq, mode, "batch")
    return sim, [_goal(a, TRAVEL_M) for a in agents], 0


def build_idle(n: int, mode: str, collision_freq: int) -> tuple[MultiRobotSimulationCore, list[RobotGoalCommand2D], int]:
    sim, _, _ = make_fleet(n, collision_freq, mode, "batch")
    return sim, [], 0


def build_carrying(n: int, mode: str, collision_freq: int) -> tuple[MultiRobotSimulationCore, list[RobotGoalCommand2D], int]:
    sim, _, agents = make_fleet(n, collision_freq, mode, "batch")
    loads = []
    for i, agent in enumerate(agents):
        pose = agent.get_pose()
        loads.append(
            replace(
                _static_box(LOAD_SHAPE, pose.x, pose.y, pose.z + LOAD_LIFT_M),
                collision_mode=CollisionMode.NORMAL_3D,
                name=f"load_{i}",
            )
        )
    boxes = SimObjectManager(sim_core=sim).spawn_objects_batch(loads)
    for agent, box in zip(agents, boxes):
        agent.attach_object(box, keep_world_pose=True)
    return sim, [_goal(a, TRAVEL_M) for a in agents], 0


def build_dense(n: int, mode: str, collision_freq: int) -> tuple[MultiRobotSimulationCore, list[RobotGoalCommand2D], int]:
    sim = make_sim(collision_freq, collision_margin=DENSE_MARGIN_M)
    mgr = make_manager(sim, mode, "batch")
    spawn = make_spawn_params(mode)
    half = max(1, n // 2)
    rows = int(math.ceil(math.sqrt(half)))
    params = []
    for i in range(n):
        k = i // 2
        col, row = divmod(k, rows)
        x = 2.0 * col + (DENSE_GROUP_GAP_M if i % 2 else 0.0)
        yaw = math.pi if i % 2 else 0.0
        params.append(replace(spawn, initial_pose=Pose.from_yaw(x, DENSE_ROW_M * row, 0.1, yaw), name=f"robot_{i}"))
    agents = mgr.spawn_objects_batch(params)
    goals = [_goal(a, -TRAVEL_M if i % 2 else TRAVEL_M) for i, a in enumerate(agents) if i % DENSE_STOPPED_EVERY]
    return sim, goals, 0


def build_warehouse(
    n: int, mode: str, collision_freq: int, field_width: int
) -> tuple[MultiRobotSimulationCore, list[RobotGoalCommand2D], int]:
    sim, _, agents = make_fleet(n, collision_freq, mode, "batch")
    fleet_rows = int(math.ceil(math.sqrt(n)))
    y_max = max(field_width, 2 * fleet_rows)
    statics = []
    for i in range(*FIELD_X_RANGE):
        for j in range(-2, y_max + 1):
            statics.append(_static_box(TILE_SHAPE, float(i), float(j), -0.3))
            if j % 2 == 1 and i % 4 < 3:
                statics.append(_static_box(BIN_SHAPE, float(i), float(j), 0.17))
    with sim.batch_spawn():
        spawned = SimObjectManager(sim_core=sim).spawn_objects_batch(statics)
    return sim, [_goal(a, TRAVEL_M) for a in agents], len(spawned)


def build_scenario(
    scenario: str, n: int, mode: str, collision_freq: int, *, field_width: int = 100
) -> tuple[MultiRobotSimulationCore, list[RobotGoalCommand2D], int]:
    """The simulation for *scenario*, its navigation goals (not yet sent) and its static count."""
    if scenario == "sparse":
        return build_sparse(n, mode, collision_freq)
    if scenario == "idle":
        return build_idle(n, mode, collision_freq)
    if scenario == "carrying":
        return build_carrying(n, mode, collision_freq)
    if scenario == "dense":
        return build_dense(n, mode, collision_freq)
    if scenario == "warehouse":
        return build_warehouse(n, mode, collision_freq, field_width)
    raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")


def _p95(samples: list[float]) -> float:
    return sorted(samples)[int(0.95 * (len(samples) - 1))]


def measure(sim: MultiRobotSimulationCore, warmup: int, steps: int) -> dict:
    """Step *warmup* times, then *steps* times with profiling, and summarize the measured window."""
    events = {"collision_started": 0, "collision_ended": 0}

    def _started(**_: object) -> None:
        events["collision_started"] += 1

    def _ended(**_: object) -> None:
        events["collision_ended"] += 1

    for _ in range(warmup):
        sim.step_once()
    sim.events.on(SimEvents.COLLISION_STARTED, _started)
    sim.events.on(SimEvents.COLLISION_ENDED, _ended)

    step_ms: list[float] = []
    phases: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for _ in range(steps):
        s = time.perf_counter()
        prof = sim.step_once(return_profiling=True) or {}
        step_ms.append((time.perf_counter() - s) * 1000.0)
        for key, value in prof.items():
            if key == "collision_breakdown":
                for sub, sub_val in value.items():
                    phases.setdefault(f"col.{sub}", []).append(sub_val)
            else:
                phases.setdefault(key, []).append(value)
    wall_s = time.perf_counter() - t0

    p50 = statistics.median(step_ms)
    return {
        "warmup": warmup,
        "steps": steps,
        "wall_s": wall_s,
        "mean_ms": statistics.mean(step_ms),
        "p50_ms": p50,
        "p95_ms": _p95(step_ms),
        "max_ms": max(step_ms),
        "rtf": (sim.params.timestep * 1000.0) / p50 if p50 > 0 else 0.0,
        "phases": {k: (statistics.mean(v), statistics.median(v), _p95(v)) for k, v in phases.items() if v},
        "collision_started": events["collision_started"],
        "collision_ended": events["collision_ended"],
        "active_pairs_end": len(sim.get_active_collision_pairs()),
    }


def run(
    scenario: str,
    n: int,
    *,
    mode: str = "diff",
    collision_freq: int = 60,
    warmup: int = 20,
    steps: int = 150,
    field_width: int = 100,
) -> dict:
    """Build *scenario*, send its goals, measure, disconnect, and return the result dict."""
    t0 = time.perf_counter()
    sim, goals, num_static = build_scenario(scenario, n, mode, collision_freq, field_width=field_width)
    spawn_s = time.perf_counter() - t0
    accepted = 0
    if goals:
        accepted = len(FleetCommandDispatcher(sim).navigate(goals, source="steady_bench", command_id="steady").accepted_names)
    result = measure(sim, warmup, steps)
    p.disconnect(sim.client)
    result.update(
        {
            "scenario": scenario,
            "num_agents": n,
            "num_static": num_static,
            "mode": mode,
            "collision_freq": collision_freq,
            "spawn_s": spawn_s,
            "accepted": accepted,
        }
    )
    return result


_PHASE_ORDER = ["phase1_update", "agent_update", "phase2_pose_flush", "phase3_aabb_grid_flush", "collision_check", "total"]


def print_result(r: dict) -> None:
    print(
        f"\n  {r['scenario']}: {r['num_agents']} agents, {r['num_static']} statics, mode={r['mode']}, "
        f"collision_freq={r['collision_freq']}, spawn={r['spawn_s']:.1f}s, goals accepted={r['accepted']}"
    )
    print(
        f"    step mean={r['mean_ms']:.3f}ms  p50={r['p50_ms']:.3f}  p95={r['p95_ms']:.3f}  max={r['max_ms']:.3f}  "
        f"RTF(p50)={r['rtf']:.1f}x  collisions started={r['collision_started']} ended={r['collision_ended']} "
        f"active at end={r['active_pairs_end']}"
    )
    print("    phase                     mean       p50       p95")
    seen = set(_PHASE_ORDER)
    for key in _PHASE_ORDER + sorted(k for k in r["phases"] if k not in seen):
        if key in r["phases"]:
            m, p50, p95 = r["phases"][key]
            print(f"      {key:<22}  {m:7.3f}ms {p50:7.3f}ms {p95:7.3f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=1000, help="Number of robots (default: 1000)")
    ap.add_argument("--scenario", choices=SCENARIOS, default="sparse", help="Scene (default: sparse)")
    ap.add_argument("--mode", choices=["omni", "diff"], default="diff", help="Motion mode (default: diff)")
    ap.add_argument("--collision-freq", type=int, default=60, help="collision_check_frequency (default 60; 0 = off)")
    ap.add_argument("--warmup", type=int, default=20, help="Steps before the measured window (default: 20)")
    ap.add_argument("--steps", type=int, default=150, help="Measured steps (default: 150)")
    ap.add_argument("--field-width", type=int, default=100, help="warehouse: field extent in y, metres (default: 100)")
    ap.add_argument("--json", action="store_true", help="Print the result as one JSON line (for run_benchmark.py)")
    args = ap.parse_args()

    result = run(
        args.scenario,
        args.agents,
        mode=args.mode,
        collision_freq=args.collision_freq,
        warmup=args.warmup,
        steps=args.steps,
        field_width=args.field_width,
    )
    if args.json:
        print(json.dumps(result))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
