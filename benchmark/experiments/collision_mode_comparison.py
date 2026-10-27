#!/usr/bin/env python3
"""
collision_mode_comparison.py
Collision detection mode comparison tool

Overview:
    Compares the performance of 3 collision detection modes:
    1. No Collision (CollisionMode.DISABLED): Completely disable collision detection
    2. 2D Collision (CollisionMode.NORMAL_2D): XY collision; z must actually overlap
    3. 3D Collision (CollisionMode.NORMAL_3D): Full 3D collision

Usage:
    python collision_mode_comparison.py --agents=1000 --iterations=100

Historical output example (illustrative only; do not use these values as a
reference or as a mode-selection recommendation):
    ===================================================================
    Collision Mode Comparison for 1000 Agents
    ===================================================================

    Mode: No Collision (Disabled)
    ------------------------------
    Total simulation step: 2.345ms
    Collision check: 0.000ms (disabled)
    Other processing: 2.345ms

    Mode: 2D Collision
    ------------------
    Total simulation step: 5.678ms
    Collision check breakdown:
      Get AABBs:       0.523ms ( 10.2%)
      Spatial Hashing: 0.312ms (  6.1%)
      AABB Filtering:  1.234ms ( 24.1%)  ← 67% reduction vs 3D
      Contact Points:  0.432ms (  8.5%)
      Total collision: 2.501ms
    Other processing: 3.177ms

    Speedup vs 3D: 1.45x

    Mode: 3D Collision
    ------------------
    Total simulation step: 8.234ms
    Collision check breakdown:
      Get AABBs:       0.523ms ( 10.2%)
      Spatial Hashing: 0.312ms (  6.1%)
      AABB Filtering:  3.845ms ( 75.2%)  ← Bottleneck
      Contact Points:  0.432ms (  8.5%)
      Total collision: 5.112ms
    Other processing: 3.122ms

    ===================================================================
    Summary
    ===================================================================
    No Collision:  2.345ms (baseline, 0% collision overhead)
    2D Collision:  5.678ms (+142% vs no collision, 67% reduction vs 3D)
    3D Collision:  8.234ms (+251% vs no collision, baseline for collision)

    Recommendation:
      - For 2D navigation: Use 2D mode for 1.45x speedup
      - For 3D navigation: Use 3D mode (required for accuracy)
      - For visualization only: Disable collision checks
"""
import os
import sys
import time
import math
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pybullet as p
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.agent_manager import AgentManager, GridSpawnParams
from pybullet_fleet.agent import AgentSpawnParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.types import CollisionMode


def benchmark_collision_mode(
    mode_name: str, collision_freq: float, collision_mode: CollisionMode, num_agents: int, num_iterations: int
):
    """Benchmark a specific collision mode"""

    # Clean disconnect
    if p.isConnected():
        p.disconnect()

    # Setup
    config_path = os.path.join(os.path.dirname(__file__), "profiling_config.yaml")
    params = SimulationParams.from_config(config_path)
    params.monitor = False
    params.collision_check_frequency = collision_freq

    wants_collision = collision_freq is None or collision_freq > 0

    sim_core = MultiRobotSimulationCore(params)
    agent_manager = AgentManager(sim_core=sim_core)

    # Spawn agents
    grid_size = int(math.ceil(math.sqrt(num_agents)))
    grid_params = GridSpawnParams(
        x_min=0,
        x_max=grid_size - 1,
        y_min=0,
        y_max=grid_size - 1,
        z_min=0,
        z_max=0,
        spacing=[1.0, 1.0, 0.0],
        offset=[0.0, 0.0, 0.1],
    )

    robot_urdf = "simple_cube"
    agent_spawn_params = AgentSpawnParams(
        urdf_path=robot_urdf,
        controller={
            "type": "omni",
            "max_linear_vel": 2.0,
            "max_angular_vel": 3.0,
        },
        mass=0.0,
        collision_mode=collision_mode,
    )

    agents = agent_manager.spawn_agents_grid(
        num_agents=num_agents,
        grid_params=grid_params,
        spawn_params=agent_spawn_params,
    )

    # Set goals so agents actually move (required for meaningful collision benchmarks)
    # Without goals, agents are stationary → _moved_this_step is empty →
    # collision check short-circuits and 2D/3D difference is not measured.
    for agent in agents:
        pos = agent.get_pose().position
        agent.set_goal_pose(Pose.from_xyz(pos[0] + 5.0, pos[1] + 5.0, pos[2]))

    # Collect timing data
    step_times = []
    collision_timings = {
        "get_aabbs": [],
        "spatial_hashing": [],
        "aabb_filtering": [],
        "contact_points": [],
        "total": [],
    }

    # Warm-up
    for _ in range(10):
        sim_core.step_once()

    # Benchmark
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        profiling_data = sim_core.step_once(return_profiling=True)
        step_time = (time.perf_counter() - t0) * 1000
        step_times.append(step_time)

        # Extract per-phase collision breakdown from step_once profiling data
        if profiling_data and "collision_breakdown" in profiling_data:
            breakdown = profiling_data["collision_breakdown"]
            for key in collision_timings:
                if key in breakdown:
                    collision_timings[key].append(breakdown[key])

    p.disconnect()

    # Calculate statistics
    results = {
        "mode_name": mode_name,
        "step_time_mean": statistics.mean(step_times),
        "step_time_stdev": statistics.stdev(step_times) if len(step_times) > 1 else 0,
        "collision_breakdown": {},
    }

    if wants_collision:
        for key, times in collision_timings.items():
            if times:
                results["collision_breakdown"][key] = {
                    "mean": statistics.mean(times),
                    "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                }

    return results


def print_results(results_list, num_agents):
    """Format and output results"""
    print("\n" + "=" * 70)
    print(f"Collision Mode Comparison for {num_agents} Agents")
    print("=" * 70)

    for results in results_list:
        print(f"\nMode: {results['mode_name']}")
        print("-" * 70)
        print(f"Total simulation step: {results['step_time_mean']:.3f}ms (±{results['step_time_stdev']:.3f}ms)")

        if results["collision_breakdown"]:
            print("\nCollision check breakdown:")
            total_collision = results["collision_breakdown"].get("total", {}).get("mean", 0)

            for component in ["get_aabbs", "spatial_hashing", "aabb_filtering", "contact_points", "total"]:
                if component in results["collision_breakdown"]:
                    mean = results["collision_breakdown"][component]["mean"]
                    percentage = (mean / total_collision * 100) if total_collision > 0 else 0
                    print(f"  {component.replace('_', ' ').title():.<20} {mean:>7.3f}ms ({percentage:>5.1f}%)")

            other_time = results["step_time_mean"] - total_collision
            print(f"\nOther processing: {other_time:.3f}ms")
        else:
            print("Collision check: disabled")

    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    no_collision_time = results_list[0]["step_time_mean"]
    collision_2d_time = results_list[1]["step_time_mean"]
    collision_3d_time = results_list[2]["step_time_mean"]

    print(f"No Collision:  {no_collision_time:.3f}ms (baseline, 0% collision overhead)")
    print(f"2D Collision:  {collision_2d_time:.3f}ms (+{(collision_2d_time/no_collision_time - 1)*100:.1f}% vs no collision)")
    print(f"3D Collision:  {collision_3d_time:.3f}ms (+{(collision_3d_time/no_collision_time - 1)*100:.1f}% vs no collision)")

    if collision_3d_time > collision_2d_time:
        speedup = collision_3d_time / collision_2d_time
        reduction = (1 - collision_2d_time / collision_3d_time) * 100
        print(f"\nObserved 2D/3D ratio for this workload: {speedup:.2f}x ({reduction:.1f}% reduction)")

    print("\nInterpretation:")
    print("  - Select NORMAL_2D only when ignoring Z-axis neighbours is correct.")
    print("  - Select NORMAL_3D when the scene requires Z-axis neighbours.")
    print("  - Select DISABLED only when collision detection is intentionally out of scope.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare collision detection modes")
    parser.add_argument("--agents", type=int, default=1000, help="Number of agents")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    print(f"Benchmarking collision modes with {args.agents} agents...")
    print(f"Running {args.iterations} iterations per mode\n")

    # Run benchmarks for each mode
    results = []

    print("[1/3] Benchmarking No Collision mode...")
    results.append(
        benchmark_collision_mode(
            "No Collision (Disabled)",
            collision_freq=0,
            collision_mode=CollisionMode.DISABLED,
            num_agents=args.agents,
            num_iterations=args.iterations,
        )
    )

    print("[2/3] Benchmarking 2D Collision mode...")
    results.append(
        benchmark_collision_mode(
            "2D Collision",
            collision_freq=None,  # Check every step
            collision_mode=CollisionMode.NORMAL_2D,
            num_agents=args.agents,
            num_iterations=args.iterations,
        )
    )

    print("[3/3] Benchmarking 3D Collision mode...")
    results.append(
        benchmark_collision_mode(
            "3D Collision",
            collision_freq=None,  # Check every step
            collision_mode=CollisionMode.NORMAL_3D,
            num_agents=args.agents,
            num_iterations=args.iterations,
        )
    )

    print_results(results, args.agents)
