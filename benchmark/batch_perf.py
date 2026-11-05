"""Controller and command-interface micro-benchmark.

Spawns N agents on a grid, sends one navigation goal per agent, and measures
wall time + per-step phase costs over a fixed step budget.

This script compares two independent axes:

- ``--controller per_agent|batch`` controls how robot motion is computed inside
  the simulation loop.
- ``--command-interface per_agent|fleet`` controls how navigation commands are
  submitted: one call per robot or one fleet-level dispatcher call.

Omni agents follow a straight-line path (no rotation needed); differential
agents follow a zigzag path that forces a ROTATE at every waypoint — the
workload most representative of the batch controller's extra slerp computation.

Run (default: diff, 500 agents, collision on)::

    python3 benchmark/batch_perf.py

Omni mode::

    python3 benchmark/batch_perf.py --mode omni

No collision (isolates controller speedup)::

    python3 benchmark/batch_perf.py --collision-freq 0

All controller/command-interface combinations::

    python3 benchmark/batch_perf.py --controller per_agent batch --command-interface per_agent fleet

GUI visualisation (batch only)::

    python3 benchmark/batch_perf.py --gui
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pybullet as p
from pybullet_fleet import (
    Agent,
    AgentSpawnParams,
    MultiRobotSimulationCore,
    Pose,
    SimulationParams,
)
from pybullet_fleet.agent_manager import AgentManager, GridSpawnParams
from pybullet_fleet.commands import RobotGoalCommand2D
from pybullet_fleet.fleet_api import FleetCommandDispatcher, FleetStateProvider
from pybullet_fleet.types import CollisionMode, SpatialHashCellSizeMode


def make_sim(collision_freq: int, gui: bool = False, *, collision_margin: float | None = None) -> MultiRobotSimulationCore:
    """The headless, kinematic, dt=0.1 simulation every fleet benchmark runs in."""
    extra = {} if collision_margin is None else {"collision_margin": collision_margin}
    return MultiRobotSimulationCore(
        SimulationParams(
            gui=gui,
            monitor=False,
            physics=False,
            timestep=0.1,
            collision_check_frequency=collision_freq,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=2.0,
            log_level="warning",
            enable_time_profiling=False,
            **extra,
        )
    )


def make_grid_params(n: int, spacing: float = 2.0) -> GridSpawnParams:
    side = int(np.ceil(np.sqrt(n)))
    return GridSpawnParams(
        x_min=0,
        x_max=side - 1,
        y_min=0,
        y_max=side - 1,
        spacing=[spacing, spacing, 0.0],
        offset=[0.0, 0.0, 0.1],
    )


def make_spawn_params(mode: str) -> AgentSpawnParams:
    """A simple_cube agent with the benchmark's velocity limits, omni or differential."""
    ctrl: dict = {"max_linear_vel": 1.5, "max_linear_accel": 2.0}
    if mode == "diff":
        ctrl["max_angular_vel"] = 2.0
        ctrl["max_angular_accel"] = 4.0
    ctrl["type"] = "omni" if mode == "omni" else "differential"
    return AgentSpawnParams(
        urdf_path="robots/simple_cube.urdf",
        collision_mode=CollisionMode.NORMAL_3D,
        controller=ctrl,
    )


def _build_waypoints(start: Pose, mode: str) -> list[Pose]:
    if mode == "omni":
        # Straight line: no rotation needed, tests pure translation throughput.
        return [Pose.from_xyz(start.x + i * 0.3, start.y, start.z) for i in range(1, 6)]
    # Diff: zigzag forces a ROTATE→FORWARD cycle at every waypoint.
    leg = 0.4
    return [Pose.from_xyz(start.x + leg * i, start.y + leg * (1 if i % 2 else -1), start.z) for i in range(1, 4)]


def make_manager(sim: MultiRobotSimulationCore, mode: str, controller: str) -> AgentManager:
    """An AgentManager with the batch controller for *mode*, or none for per-agent control."""
    batch_controller = None
    if controller == "batch":
        batch_controller = "batch_omni" if mode == "omni" else "batch_differential"
    return AgentManager(sim_core=sim, fleet_controller={"type": batch_controller} if batch_controller else None)


def make_fleet(
    n: int,
    collision_freq: int,
    mode: str,
    controller: str,
    *,
    gui: bool = False,
    collision_margin: float | None = None,
) -> tuple[MultiRobotSimulationCore, AgentManager, list[Agent]]:
    """N agents on a 2 m grid in a fresh simulation: the shared starting point of the fleet benchmarks."""
    sim = make_sim(collision_freq, gui=gui, collision_margin=collision_margin)
    mgr = make_manager(sim, mode, controller)
    agents = mgr.spawn_agents_grid(n, make_grid_params(n), make_spawn_params(mode), name_prefix="robot")
    return sim, mgr, agents


def _apply_per_agent_commands(agents: list[Agent], mode: str) -> tuple[float, int, int]:
    setup_start = time.perf_counter()
    for agent in agents:
        goal = _build_waypoints(agent.get_pose(), mode)[-1]
        agent.set_goal_pose(goal)
    return time.perf_counter() - setup_start, len(agents), 0


def _apply_fleet_commands(
    sim: MultiRobotSimulationCore,
    agents: list[Agent],
    mode: str,
) -> tuple[float, int, int, float, int]:
    dispatcher = FleetCommandDispatcher(sim)
    commands = []
    for agent in agents:
        goal = _build_waypoints(agent.get_pose(), mode)[-1]
        commands.append(
            RobotGoalCommand2D(
                name=agent.name,
                position=(goal.x, goal.y),
                yaw=goal.yaw,
                z=goal.z,
            )
        )

    setup_start = time.perf_counter()
    ack = dispatcher.navigate(commands, source="batch-perf", command_id="batch-perf-nav")
    setup_s = time.perf_counter() - setup_start

    state_provider = FleetStateProvider(sim)
    state_start = time.perf_counter()
    state_count = len(state_provider.get_states())
    state_snapshot_s = time.perf_counter() - state_start
    return setup_s, len(ack.accepted_names), len(ack.rejected), state_snapshot_s, state_count


def bench_case(
    n: int,
    steps: int,
    collision_freq: int,
    mode: str,
    controller_impl: str,
    command_interface: str,
    *,
    gui: bool = False,
) -> dict:
    sim, mgr, agents = make_fleet(n, collision_freq, mode, controller_impl, gui=gui)

    state_snapshot_s = 0.0
    state_count = 0
    if command_interface == "per_agent":
        setup_s, accepted, rejected = _apply_per_agent_commands(agents, mode)
    else:
        setup_s, accepted, rejected, state_snapshot_s, state_count = _apply_fleet_commands(sim, agents, mode)

    if gui:
        sim.setup_camera(
            camera_config={
                "camera_mode": "auto",
                "camera_view_type": "perspective",
                "camera_auto_scale": 0.7,
            },
            entity_positions=[agent.get_pose().position for agent in agents],
        )
        sim.run_simulation(duration=None)
        return {}

    controller_label = "Batch" if controller_impl == "batch" else "PerAgent"
    command_label = "FleetCommandDispatcher" if command_interface == "fleet" else "per-agent API"
    result = _run(
        sim,
        steps,
        label=f"{controller_label} controller via {command_label}, n={n}",
        controller_impl=controller_impl,
        command_interface=command_interface,
        setup_s=setup_s,
        accepted=accepted,
        rejected=rejected,
    )
    if command_interface == "fleet":
        result["state_snapshot_s"] = state_snapshot_s
        result["state_count"] = state_count
    return result


def _run(
    sim: MultiRobotSimulationCore,
    steps: int,
    *,
    label: str,
    controller_impl: str,
    command_interface: str,
    setup_s: float,
    accepted: int,
    rejected: int,
) -> dict:
    for _ in range(5):
        sim.step_once()

    step_times: list[float] = []
    collected: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for _ in range(steps):
        s = time.perf_counter()
        result = sim.step_once(return_profiling=True) or {}
        step_times.append((time.perf_counter() - s) * 1000.0)
        for key, value in result.items():
            if key == "collision_breakdown":
                if isinstance(value, dict):
                    for sub, sub_val in value.items():
                        collected.setdefault(f"col.{sub}", []).append(sub_val)
                continue
            collected.setdefault(key, []).append(value)
    t1 = time.perf_counter()

    profile_stats: dict[str, tuple[float, float, float]] = {
        key: (
            statistics.mean(samples),
            statistics.median(samples),
            sorted(samples)[int(0.95 * (len(samples) - 1))],
        )
        for key, samples in collected.items()
        if samples
    }
    p.disconnect(sim.client)
    return {
        "label": label,
        "controller_impl": controller_impl,
        "command_interface": command_interface,
        "setup_s": setup_s,
        "accepted": accepted,
        "rejected": rejected,
        "wall_s": t1 - t0,
        "steps": steps,
        "mean_ms": statistics.mean(step_times),
        "p50_ms": statistics.median(step_times),
        "p95_ms": sorted(step_times)[int(0.95 * len(step_times))],
        "max_ms": max(step_times),
        "phases": profile_stats,
    }


_PHASE_KEYS = [
    "phase1_update",
    "agent_update",
    "callbacks",
    "phase2_pose_flush",
    "step_simulation",
    "phase3_aabb_grid_flush",
    "collision_check",
    "total",
]


def _print(r: dict) -> None:
    print(f"\n  {r['label']}")
    print(
        f"    setup={r.get('setup_s', 0.0):.4f}s  accepted={r.get('accepted', 0)}  "
        f"rejected={r.get('rejected', 0)}  wall={r['wall_s']:.2f}s  "
        f"step mean={r['mean_ms']:.3f}ms  p50={r['p50_ms']:.3f}  "
        f"p95={r['p95_ms']:.3f}  max={r['max_ms']:.3f}"
    )
    if "state_snapshot_s" in r:
        print(f"    state_snapshot={r['state_count']} robots in {r['state_snapshot_s']:.6f}s")
    print("    phase                     mean       p50       p95")
    seen = set(_PHASE_KEYS)
    for key in _PHASE_KEYS:
        if key not in r["phases"]:
            continue
        m, p50, p95 = r["phases"][key]
        print(f"      {key:<22}  {m:7.3f}ms {p50:7.3f}ms {p95:7.3f}ms")
    for key in sorted(r["phases"]):
        if key in seen:
            continue
        m, p50, p95 = r["phases"][key]
        print(f"      {key:<22}  {m:7.3f}ms {p50:7.3f}ms {p95:7.3f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--agents",
        "--n",
        dest="agents",
        type=int,
        default=500,
        help="Number of agents (default: 500; --n is kept as a compatibility alias)",
    )
    ap.add_argument("--steps", type=int, default=600, help="Measurement steps (default: 600)")
    ap.add_argument(
        "--mode",
        choices=["omni", "diff"],
        default="diff",
        help="omni (BatchOmniController) or diff (BatchDifferentialController, default)",
    )
    ap.add_argument(
        "--collision-freq",
        type=int,
        default=60,
        help="collision_check_frequency (default 60; 0 = disabled)",
    )
    ap.add_argument(
        "--controller",
        nargs="+",
        choices=["per_agent", "batch"],
        default=["batch"],
        help="Controller implementation(s) to benchmark (default: batch)",
    )
    ap.add_argument(
        "--command-interface",
        nargs="+",
        choices=["per_agent", "fleet"],
        default=["fleet"],
        help="Command interface granularity to benchmark (default: fleet)",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON results for run_benchmark.py")
    ap.add_argument("--gui", action="store_true", help="Open the GUI (batch run only)")
    args = ap.parse_args()

    ctrl_name = "OmniController" if args.mode == "omni" else "DifferentialController"
    print(
        f"\n=== Controller/API interface {ctrl_name} perf  "
        f"(n={args.agents}, steps={args.steps}, collision_freq={args.collision_freq}) ===\n"
    )
    if args.gui:
        controller = args.controller[0]
        command_interface = args.command_interface[0]
        controller_name = (
            ("BatchOmniController" if args.mode == "omni" else "BatchDifferentialController")
            if controller == "batch"
            else ctrl_name
        )
        print(
            f"{controller_name} via {command_interface} command interface — GUI  (n={args.agents})\n"
            "Close the window to exit.\n"
        )
        bench_case(args.agents, args.steps, args.collision_freq, args.mode, controller, command_interface, gui=True)
    else:
        results = [
            bench_case(args.agents, args.steps, args.collision_freq, args.mode, controller, command_interface)
            for controller in args.controller
            for command_interface in args.command_interface
        ]
        if args.json:
            print(json.dumps(results[-1] if len(results) == 1 else results))
            return
        for result in results:
            _print(result)
        print()


if __name__ == "__main__":
    main()
