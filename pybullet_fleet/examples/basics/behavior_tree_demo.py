#!/usr/bin/env python3
"""Run portable BehaviorTree.CPP worker trees without a USD dependency."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pybullet as p

from pybullet_fleet import Agent, MultiRobotSimulationCore, SimulationParams, WorkerBehaviorTree
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams
from pybullet_fleet.types import CollisionMode


def _worker_tree() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "worker_wander.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, help="Number of BT-driven workers (default: 2)")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds (default: 20)")
    parser.add_argument("--headless", action="store_true", help="Run without opening the PyBullet GUI")
    args = parser.parse_args()

    sim = MultiRobotSimulationCore(
        SimulationParams(gui=not args.headless, monitor=not args.headless, enable_floor=True, duration=args.duration)
    )
    waypoint_sets = {"aisle": [(-3.0, -2.0, 0.4), (-3.0, 2.0, 0.4), (3.0, -2.0, 0.4), (3.0, 2.0, 0.4)]}
    starts = [(-1.5, -1.5, 0.4), (1.5, 1.5, 0.4)]
    for index in range(max(0, args.workers)):
        worker = Agent.from_mesh(
            visual_shape=ShapeParams(shape_type="cylinder", radius=0.22, height=0.8, rgba_color=[0.2, 0.8, 1.0, 1.0]),
            collision_shape=ShapeParams(shape_type="cylinder", radius=0.22, height=0.8),
            pose=Pose.from_xyz(*starts[index % len(starts)]),
            controller={"type": "omni", "max_linear_vel": 1.2},
            collision_mode=CollisionMode.NORMAL_3D,
            name=f"worker_{index}",
            sim_core=sim,
        )
        sim.register_behavior_tree(
            WorkerBehaviorTree.from_file(
                _worker_tree(),
                agent=worker,
                waypoint_sets=waypoint_sets,
                blackboard={"worker_waypoint_set": "aisle"},
                seed=index,
            )
        )

    if not args.headless:
        sim.setup_camera()
        print("Blue workers follow the portable BehaviorTree.CPP profile. Close the window or Ctrl-C to exit.")
    try:
        sim.run_simulation()
    finally:
        try:
            p.disconnect(sim.client)
        except p.error:
            pass


if __name__ == "__main__":
    main()
