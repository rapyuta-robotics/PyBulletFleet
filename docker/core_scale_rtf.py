#!/usr/bin/env python3
"""Measure a generated fleet-scale YAML without constructing a ROS bridge."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pybullet_fleet import MultiRobotSimulationCore
from pybullet_fleet.commands import RobotGoalCommand2D
from pybullet_fleet.config_utils import load_yaml_config
from pybullet_fleet.fleet_api import FleetCommandDispatcher

from fleet_scale_config import moving_workload_direction


def _run_for_wall_time(sim: MultiRobotSimulationCore, seconds: float) -> tuple[float, int]:
    start_wall = time.monotonic()
    start_sim = sim.sim_time
    deadline = start_wall + seconds
    steps = 0
    while time.monotonic() < deadline:
        sim.step_once()
        steps += 1
    return sim.sim_time - start_sim, steps


def _start_moving_workload(sim: MultiRobotSimulationCore, distance: float) -> None:
    """Send every agent the goal the scale checker's moving workload sends through /fleet/navigate."""
    agents = sim.agents
    goals = []
    for index, agent in enumerate(agents):
        position = agent.get_pose().position
        x, y = float(position[0]), float(position[1])
        goals.append(
            RobotGoalCommand2D(
                name=agent.name,
                position=(x + moving_workload_direction(index, len(agents)) * distance, y),
            )
        )
    ack = FleetCommandDispatcher(sim, retain_command_events=False).navigate(goals, source="core_scale_rtf")
    if ack.rejected:
        raise RuntimeError(f"moving workload rejected for {len(ack.rejected)} agents: {dict(list(ack.rejected.items())[:5])}")


def _moving_fraction(sim: MultiRobotSimulationCore) -> float:
    agents = sim.agents
    return sum(agent.is_moving for agent in agents) / len(agents) if agents else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Generated bridge-scale YAML")
    parser.add_argument("--warmup", type=float, default=1.0, help="Wall-clock warm-up seconds")
    parser.add_argument("--duration", type=float, default=6.0, help="Wall-clock measurement seconds")
    parser.add_argument(
        "--moving-distance",
        type=float,
        default=None,
        help=(
            "Measure with every robot moving: send each one a goal this many metres away "
            "(rows alternate direction, as fleet_scale_check.py --rtf-workload moving does)"
        ),
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.duration <= 0:
        parser.error("--warmup must be non-negative and --duration must be positive")
    if args.moving_distance is not None and args.moving_distance <= 0:
        parser.error("--moving-distance must be positive")

    sim = MultiRobotSimulationCore.from_dict(load_yaml_config(args.config))
    sim.initialize_simulation()
    workload = "idle"
    if args.moving_distance is not None:
        _start_moving_workload(sim, args.moving_distance)
        workload = "moving"
    if args.warmup:
        _run_for_wall_time(sim, args.warmup)
    start_wall = time.monotonic()
    sim_elapsed, steps = _run_for_wall_time(sim, args.duration)
    wall_elapsed = time.monotonic() - start_wall
    print(
        "CORE_ONLY_RTF "
        f"sim={sim_elapsed:.3f}s wall={wall_elapsed:.3f}s steps={steps} rtf={sim_elapsed / wall_elapsed:.2f}x "
        f"workload={workload} moving_at_end={_moving_fraction(sim):.1%}"
    )
    if workload == "moving" and _moving_fraction(sim) < 0.95:
        print("FAIL: fewer than 95% of the robots were still moving at the end; raise --moving-distance")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
