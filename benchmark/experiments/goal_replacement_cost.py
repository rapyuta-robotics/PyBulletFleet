#!/usr/bin/env python3
"""
goal_replacement_cost.py
Per-goal cost of setting a new goal on agents that are already moving

Overview:
    The periodic re-send pattern: every agent drives toward a far goal, then
    receives a new goal every few steps. Measures the wall-clock time per goal
    of ``Agent.set_goal_pose`` (and of ``FleetCommandDispatcher.navigate`` for
    the same goals) for per-agent and batched, omnidirectional and
    differential controllers, with two kinds of goal:

    - ``ahead``: further along the line of motion, which the agent continues
      into (the common re-send pattern);
    - ``offline``: off the line of motion, handled by ``goal_replacement``
      (``brake`` by default: brake, hold the goal, start it once stopped).

    Only the goal-setting calls are timed, not the simulation steps.

Usage:
    python benchmark/experiments/goal_replacement_cost.py --agents 300 --rounds 16
    python benchmark/experiments/goal_replacement_cost.py --only omni/ahead
    python benchmark/experiments/goal_replacement_cost.py --profile omni:ahead:set

Output: a JSON object mapping ``<controller>/<pattern>/<call>`` to the median
microseconds per goal over the rounds (the first two rounds are dropped).
"""

import argparse
import cProfile
import dataclasses
import json
import logging
import pstats
import time

import numpy as np
import pybullet as p

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.types import CollisionMode

DT = 0.05
CASES = {
    "omni": (MotionMode.OMNIDIRECTIONAL, False),
    "batch_omni": (MotionMode.OMNIDIRECTIONAL, True),
    "diff": (MotionMode.DIFFERENTIAL, False),
    "batch_diff": (MotionMode.DIFFERENTIAL, True),
}


def _build(n_agents, motion, batch):
    names = {f.name for f in dataclasses.fields(SimulationParams)}
    params = dict(
        gui=False,
        monitor=False,
        physics=False,
        timestep=DT,
        target_rtf=0,
        collision_check_frequency=0,
        monitor_log_interval=0,
        log_level="error",
    )
    sim = MultiRobotSimulationCore(SimulationParams(**{k: v for k, v in params.items() if k in names}))
    sim.initialize_simulation()
    manager = None
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        kind = "batch_omni" if motion is MotionMode.OMNIDIRECTIONAL else "batch_differential"
        manager = AgentManager(sim_core=sim, fleet_controller={"type": kind})
    agents = []
    for i in range(n_agents):
        agent = Agent.from_params(
            AgentSpawnParams(
                name=f"r{i}",
                urdf_path="robots/simple_cube.urdf",
                initial_pose=Pose.from_xyz(0.0, 3.0 * i, 0.1),
                motion_mode=motion,
                collision_mode=CollisionMode.NORMAL_3D,
                controller={"max_linear_vel": 2.0, "max_linear_accel": 1.0, "max_linear_decel": 1.0},
            ),
            sim_core=sim,
        )
        if manager is not None:
            manager.add_object(agent)
        agents.append(agent)
    return sim, agents


def run(n_agents, rounds, motion, batch, pattern, call, profiler=None):
    """Median microseconds per goal for one case."""
    sim, agents = _build(n_agents, motion, batch)
    dispatcher = None
    if call == "navigate":
        from pybullet_fleet.commands import RobotGoalCommand2D
        from pybullet_fleet.fleet_api import FleetCommandDispatcher

        dispatcher = FleetCommandDispatcher(sim)
    for agent in agents:
        agent.set_goal_pose(Pose.from_xyz(1000.0, agent.get_pose().position[1], 0.1))
    for _ in range(60):  # up to speed
        sim.step_once()
    times = []
    try:
        for r in range(rounds):
            goals = []
            for agent in agents:
                x, y, _ = agent.get_pose().position
                goals.append((x + 6.0, y) if pattern == "ahead" else (x + 3.0, y + (1.0 if r % 2 else -1.0)))
            if profiler is not None:
                profiler.enable()
            start = time.perf_counter()
            if dispatcher is None:
                for agent, (gx, gy) in zip(agents, goals):
                    agent.set_goal_pose(Pose.from_xyz(gx, gy, 0.1))
            else:
                dispatcher.navigate([RobotGoalCommand2D(f"r{i}", position=goal) for i, goal in enumerate(goals)])
            times.append(time.perf_counter() - start)
            if profiler is not None:
                profiler.disable()
            for _ in range(10 if pattern == "ahead" else 70):  # off-line goals: let braking settle
                sim.step_once()
    finally:
        p.disconnect(sim.client)
    return float(np.median(times[2:]) / n_agents * 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--only", default="", help="one case, '<controller>/<pattern>' (e.g. omni/ahead)")
    parser.add_argument("--profile", default="", help="cProfile one case, '<controller>:<pattern>:<set|navigate>'")
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    if args.profile:
        name, pattern, call = args.profile.split(":")
        profiler = cProfile.Profile()
        run(args.agents, args.rounds, *CASES[name], pattern, call, profiler=profiler)
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(30)
        return

    results = {}
    for name, (motion, batch) in CASES.items():
        for pattern in ("ahead", "offline"):
            if args.only and args.only != f"{name}/{pattern}":
                continue
            results[f"{name}/{pattern}/set_goal_pose"] = round(run(args.agents, args.rounds, motion, batch, pattern, "set"), 1)
        if not args.only:
            results[f"{name}/ahead/navigate"] = round(run(args.agents, args.rounds, motion, batch, "ahead", "navigate"), 1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
