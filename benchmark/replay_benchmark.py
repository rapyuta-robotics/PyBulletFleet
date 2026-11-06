"""Matched replay measurements; workers run in separate processes.

Example::

    python benchmark/replay_benchmark.py --baseline-root /tmp/pbf-before --output /tmp/replay-results.json

The baseline tree must be the parent revision. No benchmark thresholds are
implied. Source, writer-disabled session and recording costs are separate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import warnings


def initial_definition(count, controller):
    return {
        "world": {
            "entities": [
                {"entity_id": f"entity-{i}", "name": f"robot-{i}", "position": [float(i % 40) * 2, float(i // 40) * 2, 0.1]}
                for i in range(count)
            ]
        },
        "pbf": {"controller": controller, "timestep": 0.1},
    }


def worker(args):
    import numpy as np
    import psutil
    import pybullet as p
    from pybullet_fleet.agent import Agent, AgentSpawnParams
    from pybullet_fleet.agent_manager import AgentManager
    from pybullet_fleet.commands import RobotGoalCommand2D
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore
    from pybullet_fleet.fleet_api import FleetCommandDispatcher
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.types import CollisionMode
    import pybullet_fleet

    definition = initial_definition(args.agents, args.controller)
    start = time.perf_counter()
    session = None
    with tempfile.TemporaryDirectory(prefix="pbf-replay-benchmark-") as directory:
        artifact = Path(directory) / "run"
        if args.mode == "core":
            sim = MultiRobotSimulationCore.from_dict(
                {
                    "simulation": {
                        "gui": False,
                        "physics": False,
                        "monitor": False,
                        "enable_monitor_gui": False,
                        "enable_floor": False,
                        "target_rtf": 0,
                        "timestep": 0.1,
                        "collision_check_frequency": None,
                        "collision_margin": 0.02,
                        "ignore_static_collision": False,
                        "collision_detection_method": "closest_points",
                        "spatial_hash_cell_size_mode": "auto_initial",
                        "log_level": "warning",
                    }
                }
            )
            manager = AgentManager(
                sim, name="replay_fleet", fleet_controller={"type": "batch_omni"} if args.controller == "batch_omni" else None
            )
            # Matches the replay session, whose v1 profile fixes NORMAL_2D robots (deprecated).
            with sim.batch_spawn(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", "CollisionMode.NORMAL_2D is deprecated", FutureWarning)
                for entity in definition["world"]["entities"]:
                    obj = Agent.from_params(
                        AgentSpawnParams(
                            name=entity["name"],
                            urdf_path=str(Path(pybullet_fleet.__file__).parent / "robots/simple_cube.urdf"),
                            initial_pose=Pose.from_xyz(*entity["position"]),
                            mass=0.0,
                            pickable=False,
                            collision_mode=CollisionMode.NORMAL_2D,
                            controller={
                                "type": "omni",
                                "navigation_2d": True,
                                "max_linear_vel": 2.0,
                                "max_linear_accel": 5.0,
                                "max_angular_vel": 2.0,
                                "max_angular_accel": 5.0,
                                "cmd_vel_timeout": 0.0,
                                "default_direction": "forward",
                            },
                        ),
                        sim,
                    )
                    manager.add_object(obj)
            sim.initialize_simulation()
            dispatcher = FleetCommandDispatcher(sim)
        else:
            from pybullet_fleet.replay import ReplayInput, ReplaySession

            session = ReplaySession.create(
                definition, output=artifact if args.mode == "record" else None, observation_interval=args.interval
            )
            sim = session._sim
        setup_s = time.perf_counter() - start
        for _ in range(10):
            session.step() if session else sim.step_once()
        process = psutil.Process()
        rss_start = process.memory_info().rss
        rss_peak = rss_start
        step_times = []
        commands = 0
        measured_start = time.perf_counter()
        for step in range(args.steps):
            tick = time.perf_counter()
            goals = []
            if step % 20 == 0:
                goals = [
                    {"name": e["name"], "position": [e["position"][0] + 100.0, e["position"][1]], "yaw": 0.0, "z": 0.1}
                    for e in definition["world"]["entities"]
                ]
                commands += len(goals)
            if session:
                inputs = [ReplayInput("navigate", {"goals": goals}, command_id=f"command-{step}")] if goals else []
                session.step(inputs)
            else:
                if goals:
                    dispatcher.navigate([RobotGoalCommand2D(**g) for g in goals], command_id=f"command-{step}")
                sim.step_once()
            step_times.append(time.perf_counter() - tick)
            if step % 10 == 0:
                rss_peak = max(rss_peak, process.memory_info().rss)
        loop_s = time.perf_counter() - measured_start
        flush_start = time.perf_counter()
        if session:
            session.close()
        else:
            p.disconnect(sim.client)
        flush_s = time.perf_counter() - flush_start
        output_bytes = sum(f.stat().st_size for f in artifact.glob("*")) if artifact.exists() else 0
        result = {
            "agents": args.agents,
            "controller": args.controller,
            "mode": args.mode,
            "interval": args.interval,
            "steps": args.steps,
            "warmup_steps": 10,
            "command_targets": commands,
            "setup_s": setup_s,
            "loop_s": loop_s,
            "rtf": args.steps * 0.1 / loop_s,
            "finalize_s": flush_s,
            "total_measured_s": loop_s + flush_s,
            "step_mean_ms": float(np.mean(step_times) * 1000),
            "step_p95_ms": float(np.percentile(step_times, 95) * 1000),
            "step_p99_ms": float(np.percentile(step_times, 99) * 1000),
            "rss_start_bytes": rss_start,
            "rss_sampled_peak_bytes": rss_peak,
            "artifact_bytes": output_bytes,
            "bytes_per_sim_second": output_bytes / ((args.steps + 10) * 0.1),
            "effective_writer_bytes_per_wall_second": output_bytes / (loop_s + flush_s) if output_bytes else 0,
            "package_root": str(Path(pybullet_fleet.__file__).parent),
        }
        Path(args.worker_output).write_text(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root")
    parser.add_argument("--output", default="/tmp/pbf-replay-results.json")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker-output")
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--controller", default="batch_omni")
    parser.add_argument("--mode", default="core")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    if args.worker_output:
        worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    cases = [
        ("disabled", root, "core", 10),
        ("session", root, "session", 10),
        ("record_1hz", root, "record", 10),
        ("record_every_step", root, "record", 1),
    ]
    if args.baseline_root:
        cases.insert(0, ("baseline", Path(args.baseline_root).resolve(), "core", 10))
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "storage": "local tempfile directory",
        },
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="pbf-replay-workers-") as temporary:
        for count in (100, 1000):
            for controller in ("omni", "batch_omni"):
                for repetition in range(args.repetitions):
                    for label, source, mode, interval in cases:
                        destination = str(Path(temporary) / "worker.json")
                        env = dict(os.environ, PYTHONPATH=str(source))
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker-output",
                            destination,
                            "--agents",
                            str(count),
                            "--controller",
                            controller,
                            "--mode",
                            mode,
                            "--interval",
                            str(interval),
                            "--steps",
                            str(args.steps),
                        ]
                        result = subprocess.run(command, cwd=temporary, env=env, capture_output=True, text=True)
                        if result.returncode:
                            raise RuntimeError(result.stdout + result.stderr)
                        row = json.loads(Path(destination).read_text())
                        row.update(case=label, repetition=repetition)
                        report["runs"].append(row)
                        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                        print(f"{count} {controller} {label} #{repetition}: {row['step_mean_ms']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
