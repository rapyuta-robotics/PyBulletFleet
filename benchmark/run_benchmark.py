#!/usr/bin/env python3
"""
run_benchmark.py
Meta-script for running performance benchmarks with process isolation and statistical analysis.

Features:
- Runs benchmark worker in isolated child processes
- Supports both mobile (simple cube) and arm (robot arm) benchmarks
- Supports mobile per-agent/batch/fleet control-path comparison
- Collects statistics (median, mean, stdev) from multiple repetitions
- Supports sweep across multiple agent/arm counts
- Supports multiple scenarios comparison for both types
- Clean separation of concerns (no self-recursion)

Usage:
    # Mobile benchmarks (default)
    python benchmark/run_benchmark.py --agents 1000 --duration 10 --repetitions 3
    python benchmark/run_benchmark.py --sweep 100 500 1000 2000
    python benchmark/run_benchmark.py --compare no_collision collision_10hz

    # Arm benchmarks
    python benchmark/run_benchmark.py --type arm --agents 10 --duration 5
    python benchmark/run_benchmark.py --type arm --sweep 1 10 50 100
    python benchmark/run_benchmark.py --type arm --compare physics kinematic

    # Mobile control-path benchmarks
    python benchmark/run_benchmark.py --type mobile --controller batch \
        --command-interface per_agent fleet --agents 500 --steps 600

Scenarios:
    Scenario names passed to --scenario / --compare are defined in
    benchmark/configs/general.yaml under the ``scenarios:`` key.
    Each scenario can override simulation, agents, and benchmark settings.
    Built-in scenarios include: no_collision, collision_3d_full, collision_10hz,
    per_agent, batch_omni.  Use --config to point at a different YAML file.
"""

import os
import sys
import json
import subprocess
import argparse
import statistics
from typing import List, Optional

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Worker dispatch
# ---------------------------------------------------------------------------

# Map benchmark type → worker script filename
_WORKER_SCRIPTS = {
    "mobile": "mobile_benchmark.py",
    "arm": "arm_benchmark.py",
    "mobile_control_path": "batch_perf.py",
    "steady": "steady_bench.py",
    # Phase 5 will add an arm_control_path worker after fleet joint-command
    # dispatch is implemented. Keep control-path workers explicit so they can
    # use command-oriented CLI options instead of the duration/scenario worker API.
}

_CONTROL_PATH_BENCHMARK_TYPES = {"mobile_control_path"}
# Workers measured over a fixed number of steps (--steps) rather than a duration.
_STEP_AXIS_BENCHMARK_TYPES = _CONTROL_PATH_BENCHMARK_TYPES | {"steady"}

_STEADY_SCENARIOS = ("sparse", "dense", "carrying", "warehouse", "idle")
_STEADY_DEFAULT_STEPS = 150
_STEADY_DEFAULT_WARMUP = 20


def _parse_worker_json(stdout: str) -> dict:
    """Extract JSON dict from worker stdout.

    PyBullet's C library may print warnings (b3Warning, build time) to stdout
    before the JSON line.  We find the last ``{...}`` block on a single line.
    """
    # Try the whole thing first (fast path)
    stripped = stdout.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Fallback: find the last line that looks like JSON
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Last resort: PyBullet C warnings may be concatenated on the same line
    # as the JSON output (e.g. "...end_effector{"num_arms":...}").
    # Scan backwards for each '{' and try to parse from there.
    search_from = len(stripped)
    while True:
        idx = stripped.rfind("{", 0, search_from)
        if idx < 0:
            break
        candidate = stripped[idx:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            search_from = idx
            continue

    raise json.JSONDecodeError("No valid JSON found in worker output", stdout, 0)


def run_worker(
    num_agents: int,
    duration: float,
    benchmark_type: str = "mobile",
    *,
    gui: bool = False,
    config_path: Optional[str] = None,
    scenario: Optional[str] = None,
    steps: Optional[int] = None,
    mode: str = "diff",
    collision_freq: Optional[int] = None,
    controller: Optional[str] = None,
    command_interface: Optional[str] = None,
    warmup: Optional[int] = None,
) -> dict:
    """Run benchmark worker in isolated child process.

    Args:
        num_agents: Number of agents/arms.
        duration: Simulation duration.
        benchmark_type: ``"mobile"``, ``"arm"``, ``"steady"``, or internal ``"mobile_control_path"``.
        gui: Enable GUI (mobile only).
        config_path: Path to config file.
        scenario: Scenario name (e.g., ``"physics"``, ``"kinematic"``; a steady scene name for ``"steady"``).
        warmup: Steps before the measured window (steady only).

    Returns:
        Benchmark results dictionary.
    """
    script = _WORKER_SCRIPTS[benchmark_type]
    script_path = os.path.join(os.path.dirname(__file__), script)

    if benchmark_type == "steady":
        cmd = [
            sys.executable,
            script_path,
            "--agents",
            str(num_agents),
            "--scenario",
            scenario or "sparse",
            "--steps",
            str(_STEADY_DEFAULT_STEPS if steps is None else steps),
            "--warmup",
            str(_STEADY_DEFAULT_WARMUP if warmup is None else warmup),
            "--mode",
            mode,
            "--collision-freq",
            str(60 if collision_freq is None else collision_freq),
            "--json",
        ]
    elif benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        effective_steps = 600 if steps is None else steps
        effective_collision_freq = 60 if collision_freq is None else collision_freq
        cmd = [
            sys.executable,
            script_path,
            "--agents",
            str(num_agents),
            "--steps",
            str(effective_steps),
            "--mode",
            mode,
            "--collision-freq",
            str(effective_collision_freq),
            "--controller",
            controller or "batch",
            "--command-interface",
            command_interface or "fleet",
            "--json",
        ]
    else:
        cmd = [
            sys.executable,
            script_path,
            "--agents",
            str(num_agents),
            "--duration",
            str(duration),
        ]
        if gui and benchmark_type == "mobile":
            cmd.append("--gui")

        if config_path:
            cmd.extend(["--config", config_path])
        if scenario:
            cmd.extend(["--scenario", scenario])
        if benchmark_type in {"mobile", "arm"} and collision_freq is not None:
            cmd.extend(["--collision-freq", str(collision_freq)])

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed ({benchmark_type}, n={num_agents}, scenario={scenario})\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    try:
        return _parse_worker_json(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse worker JSON: {e}\n" f"Raw stdout:\n{proc.stdout}\n")


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------


def run_multiple(
    num_agents: int,
    duration: float,
    num_reps: int,
    benchmark_type: str = "mobile",
    *,
    gui: bool = False,
    config_path: Optional[str] = None,
    scenario: Optional[str] = None,
    steps: Optional[int] = None,
    mode: str = "diff",
    collision_freq: Optional[int] = None,
    controller: Optional[str] = None,
    command_interface: Optional[str] = None,
    warmup: Optional[int] = None,
) -> dict:
    """Run benchmark multiple times and compute statistics."""
    results = []

    if benchmark_type == "steady":
        effective_steps = _STEADY_DEFAULT_STEPS if steps is None else steps
        effective_collision_freq = 60 if collision_freq is None else collision_freq
        effective_controller = controller
        effective_command_interface = command_interface
    elif benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        effective_steps = 600 if steps is None else steps
        effective_collision_freq = 60 if collision_freq is None else collision_freq
        effective_controller = controller or "batch"
        effective_command_interface = command_interface or "fleet"
    else:
        effective_steps = steps
        effective_collision_freq = collision_freq
        effective_controller = controller
        effective_command_interface = command_interface

    entity_label = "arms" if benchmark_type == "arm" else "agents"
    duration_label = f"{effective_steps} steps" if benchmark_type in _STEP_AXIS_BENCHMARK_TYPES else f"{duration}s"
    label_parts = [f"{num_agents} {entity_label}", duration_label]
    if benchmark_type == "steady":
        label_parts.append(f"scenario={scenario or 'sparse'}")
    elif benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        label_parts.append(f"controller={effective_controller}")
        label_parts.append(f"command_interface={effective_command_interface}")
    elif scenario:
        label_parts.append(f"scenario={scenario}")

    print(f"\n{'='*70}")
    print(f"Benchmark: {', '.join(label_parts)}")
    print(f"{'='*70}")

    for i in range(num_reps):
        print(f"  Run {i+1}/{num_reps}...", end="", flush=True)
        r = run_worker(
            num_agents,
            duration,
            benchmark_type,
            gui=gui,
            config_path=config_path,
            scenario=scenario,
            steps=steps,
            mode=mode,
            collision_freq=collision_freq,
            controller=controller,
            command_interface=command_interface,
            warmup=warmup,
        )
        results.append(r)
        print(" done")

    def stats(xs):
        if len(xs) == 0:
            return {"median": 0.0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "median": statistics.median(xs),
            "mean": statistics.mean(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    if benchmark_type == "steady":
        # Per-phase: the median over repetitions of each repetition's own median.
        phase_keys = sorted({k for r in results for k in r.get("phases", {})})
        return {
            "num_agents": num_agents,
            "benchmark_type": benchmark_type,
            "scenario": scenario or "sparse",
            "steps": effective_steps,
            "warmup": results[0].get("warmup", warmup) if results else warmup,
            "mode": mode,
            "collision_freq": effective_collision_freq,
            "num_reps": num_reps,
            "num_static": results[0].get("num_static", 0) if results else 0,
            "spawn_s": stats([r["spawn_s"] for r in results]),
            "accepted": stats([r["accepted"] for r in results]),
            "wall_s": stats([r["wall_s"] for r in results]),
            "avg_step_time_ms": stats([r["mean_ms"] for r in results]),
            "p50_step_time_ms": stats([r["p50_ms"] for r in results]),
            "p95_step_time_ms": stats([r["p95_ms"] for r in results]),
            "max_step_time_ms": stats([r["max_ms"] for r in results]),
            "rtf": stats([r["rtf"] for r in results]),
            "collision_started": stats([r["collision_started"] for r in results]),
            "collision_ended": stats([r["collision_ended"] for r in results]),
            "active_pairs_end": stats([r["active_pairs_end"] for r in results]),
            "phases_p50_ms": {k: stats([r["phases"][k][1] for r in results if k in r.get("phases", {})]) for k in phase_keys},
        }
    if benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        return {
            "num_agents": num_agents,
            "benchmark_type": benchmark_type,
            "steps": effective_steps,
            "mode": mode,
            "collision_freq": effective_collision_freq,
            "controller_impl": effective_controller,
            "command_interface": effective_command_interface,
            "num_reps": num_reps,
            "setup_s": stats([r["setup_s"] for r in results]),
            "accepted": stats([r["accepted"] for r in results]),
            "rejected": stats([r["rejected"] for r in results]),
            "wall_s": stats([r["wall_s"] for r in results]),
            "avg_step_time_ms": stats([r["mean_ms"] for r in results]),
            "p50_step_time_ms": stats([r["p50_ms"] for r in results]),
            "p95_step_time_ms": stats([r["p95_ms"] for r in results]),
            "max_step_time_ms": stats([r["max_ms"] for r in results]),
            "state_snapshot_s": stats([r.get("state_snapshot_s", 0.0) for r in results]),
        }
    else:
        # Aggregate results — use keys that both mobile and arm workers provide.
        aggregated = {
            "num_agents": num_agents,
            "benchmark_type": benchmark_type,
            "duration_s": duration,
            "scenario": scenario,
            "num_reps": num_reps,
            "gui": gui,
            "spawn_time_s": stats([r["spawn_time_s"] for r in results]),
            "spawn_cpu_s": stats([r["spawn_cpu_s"] for r in results]),
            "spawn_cpu_percent": stats([r["spawn_cpu_percent"] for r in results]),
            "simulation_wall_s": stats([r["simulation_wall_s"] for r in results]),
            "simulation_cpu_s": stats([r["simulation_cpu_s"] for r in results]),
            "simulation_cpu_percent": stats([r["simulation_cpu_percent"] for r in results]),
            "real_time_factor": stats([r["real_time_factor"] for r in results]),
            "avg_step_time_ms": stats([r["avg_step_time_ms"] for r in results]),
            "system_info": results[0]["system_info"] if results else {},
            "mem_spawn_rss_mb": stats([r["mem_spawn_mb"]["rss_mb"] for r in results]),
            "mem_total_rss_mb": stats([r["mem_total_mb"]["rss_mb"] for r in results]),
        }
        if benchmark_type == "mobile":
            aggregated["batch_controller"] = results[0].get("batch_controller") if results else None
            aggregated["command_interface"] = results[0].get("command_interface") if results else None
            aggregated["command_setup_s"] = stats([r.get("command_setup_s", 0.0) for r in results])
            aggregated["accepted_commands"] = stats([r.get("accepted_commands", 0) for r in results])
            aggregated["rejected_commands"] = stats([r.get("rejected_commands", 0) for r in results])
        if results and "collision_check_frequency" in results[0]:
            aggregated["collision_check_frequency"] = results[0].get("collision_check_frequency")

    # Mobile-specific: expected_steps
    if results and benchmark_type not in _STEP_AXIS_BENCHMARK_TYPES and "expected_steps" in results[0]:
        aggregated["expected_steps"] = results[0]["expected_steps"]

    # Arm-specific: total_joints, mode (from worker result)
    if benchmark_type == "arm":
        aggregated["mode"] = results[0].get("mode", "N/A") if results else "N/A"
        aggregated["timestep"] = results[0].get("timestep", 0.01) if results else 0.01
        aggregated["total_joints"] = stats([r.get("total_joints", 0) for r in results])

    return aggregated


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


_STEADY_PHASE_ORDER = [
    "phase1_update",
    "agent_update",
    "phase2_pose_flush",
    "phase3_aabb_grid_flush",
    "collision_check",
    "total",
]


def _print_steady_results(results: dict) -> None:
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Scenario: {results['scenario']}")
    print(f"  Agents: {results['num_agents']}, static bodies: {results['num_static']}")
    print(f"  Steps: {results['steps']} measured after {results['warmup']} warmup, mode={results['mode']}")
    print(f"  Collision frequency: {results['collision_freq']}")
    print(f"  Repetitions: {results['num_reps']}")
    print("\nPerformance (median over repetitions):")
    print(f"  Step p50: {results['p50_step_time_ms']['median']:.3f}ms (±{results['p50_step_time_ms']['stdev']:.3f})")
    print(f"  Step mean: {results['avg_step_time_ms']['median']:.3f}ms, p95: {results['p95_step_time_ms']['median']:.3f}ms")
    print(f"  RTF (from p50): {results['rtf']['median']:.1f}x")
    print(f"  Spawn: {results['spawn_s']['median']:.2f}s")
    print(
        f"  Collisions: started={results['collision_started']['median']:.0f}, "
        f"ended={results['collision_ended']['median']:.0f}, active at end={results['active_pairs_end']['median']:.0f}"
    )
    phases = results.get("phases_p50_ms", {})
    if phases:
        print("\nPhase p50 (ms):")
        seen = set(_STEADY_PHASE_ORDER)
        for key in _STEADY_PHASE_ORDER + sorted(k for k in phases if k not in seen):
            if key in phases:
                print(f"  {key:<24} {phases[key]['median']:8.3f} (±{phases[key]['stdev']:.3f})")
    print("=" * 70)


def _print_steady_summary(all_results: List[dict]) -> None:
    print("\n" + "=" * 90)
    print("STEADY-STATE SUMMARY (median over repetitions)")
    print("=" * 90)
    header = f"{'Agents':<8} {'Statics':<8} {'Scenario':<11} {'Step p50 (ms)':<15} {'RTF':<8} "
    print(f"\n{header}{'Collision (ms)':<15} {'Phase1 (ms)':<12}")
    print("-" * 90)
    for r in all_results:
        phases = r.get("phases_p50_ms", {})
        col = phases.get("collision_check", {}).get("median", 0.0)
        ph1 = phases.get("phase1_update", {}).get("median", 0.0)
        print(
            f"{r['num_agents']:<8} {r['num_static']:<8} {r['scenario']:<11} "
            f"{r['p50_step_time_ms']['median']:>7.3f}±{r['p50_step_time_ms']['stdev']:<6.3f} "
            f"{r['rtf']['median']:>6.1f}x  {col:>8.3f}        {ph1:>8.3f}"
        )
    print("=" * 90)


def print_results(results: dict):
    """Print formatted benchmark results."""
    benchmark_type = results.get("benchmark_type", "mobile")
    if benchmark_type == "steady":
        _print_steady_results(results)
        return
    entity = "Arms" if benchmark_type == "arm" else "Agents"

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  {entity}: {results['num_agents']}")
    if benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        print(f"  Steps: {results['steps']}")
        print(f"  Controller: {results.get('controller_impl')}")
        print(f"  Command Interface: {results.get('command_interface')}")
        print(f"  Mode: {results.get('mode', 'diff')}")
        print(f"  Collision frequency: {results.get('collision_freq')}")
    else:
        print(f"  Duration: {results['duration_s']}s")
        print(f"  Scenario: {results.get('scenario') or results.get('mode') or 'default'}")
        if benchmark_type == "mobile":
            print(f"  Batch Controller: {results.get('batch_controller')}")
            print(f"  Command Interface: {results.get('command_interface')}")
        if "collision_check_frequency" in results:
            print(f"  Collision frequency: {results.get('collision_check_frequency')}")
    print(f"  Repetitions: {results['num_reps']}")

    print("\nPerformance:")
    if benchmark_type not in _CONTROL_PATH_BENCHMARK_TYPES:
        print(
            f"  Real-Time Factor: {results['real_time_factor']['median']:.2f}x "
            f"(±{results['real_time_factor']['stdev']:.2f})"
        )
    print(f"  Step Time: {results['avg_step_time_ms']['median']:.2f}ms " f"(±{results['avg_step_time_ms']['stdev']:.2f})")
    if benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        print(f"  Setup Time: {results['setup_s']['median']:.6f}s " f"(±{results['setup_s']['stdev']:.6f})")
        print(f"  State Snapshot: {results['state_snapshot_s']['median']:.6f}s")
        print(f"  Accepted: {results['accepted']['median']:.0f}, Rejected: {results['rejected']['median']:.0f}")
    else:
        print(f"  Spawn Time: {results['spawn_time_s']['median']:.3f}s " f"(±{results['spawn_time_s']['stdev']:.3f})")
        print(f"  Memory: {results['mem_total_rss_mb']['median']:.2f}MB " f"(±{results['mem_total_rss_mb']['stdev']:.2f})")
        if benchmark_type == "mobile":
            print(f"  Command Setup: {results['command_setup_s']['median']:.6f}s")
            print(
                f"  Commands: accepted={results['accepted_commands']['median']:.0f}, "
                f"rejected={results['rejected_commands']['median']:.0f}"
            )

    if benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        print("=" * 70)
        return

    # Assessment
    rtf = results["real_time_factor"]["median"]

    if rtf >= 10.0:
        rtf_status = "✅ EXCELLENT"
    elif rtf >= 5.0:
        rtf_status = "✅ GOOD"
    elif rtf >= 1.0:
        rtf_status = "⚠️  ACCEPTABLE"
    else:
        rtf_status = "❌ POOR"

    print("\nAssessment:")
    print(f"  RTF: {rtf_status} ({rtf:.2f}x)")
    print("=" * 70)


def print_sweep_summary(all_results: List[dict]):
    """Print summary table for sweep results."""
    benchmark_type = all_results[0].get("benchmark_type", "mobile") if all_results else "mobile"
    if benchmark_type == "steady":
        _print_steady_summary(all_results)
        return
    entity = "Arms" if benchmark_type == "arm" else "Agents"

    if benchmark_type in _CONTROL_PATH_BENCHMARK_TYPES:
        print("\n" + "=" * 90)
        print("MOBILE CONTROL-PATH SUMMARY")
        print("=" * 90)
        print(f"\n{entity:<10} {'Controller':<12} {'Command If':<12} " f"{'Setup (s)':<15} {'Step (ms)':<15} {'P95 (ms)':<15}")
        print("-" * 90)
        for r in all_results:
            print(
                f"{r['num_agents']:<10} {r.get('controller_impl', ''):<12} {r.get('command_interface', ''):<12} "
                f"{r['setup_s']['median']:>8.6f}±{r['setup_s']['stdev']:<6.6f}  "
                f"{r['avg_step_time_ms']['median']:>6.3f}±{r['avg_step_time_ms']['stdev']:<5.3f}  "
                f"{r['p95_step_time_ms']['median']:>6.3f}±{r['p95_step_time_ms']['stdev']:<5.3f}"
            )
        print("=" * 90)
        return

    print("\n" + "=" * 90)
    print("SWEEP SUMMARY")
    print("=" * 90)

    print(f"\n{entity:<10} {'Scenario':<20} {'RTF (x)':<15} {'Step (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 90)

    for r in all_results:
        label = r.get("scenario") or r.get("mode") or "default"

        rtf = r["real_time_factor"]["median"]
        rtf_std = r["real_time_factor"]["stdev"]
        step = r["avg_step_time_ms"]["median"]
        step_std = r["avg_step_time_ms"]["stdev"]
        mem = r["mem_total_rss_mb"]["median"]
        mem_std = r["mem_total_rss_mb"]["stdev"]

        rtf_status = "✅" if rtf >= 5.0 else "⚠️" if rtf >= 1.0 else "❌"

        print(
            f"{r['num_agents']:<10} {label:<20} {rtf_status} {rtf:>5.2f}±{rtf_std:<4.2f}  "
            f"{step:>6.2f}±{step_std:<5.2f}  "
            f"{mem:>7.2f}±{mem_std:<5.2f}"
        )

    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyBullet Fleet Performance Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mobile (default)
  %(prog)s --sweep 100 500 1000 2000
  %(prog)s --compare no_collision collision_10hz

  # Mobile per-agent/batch/fleet control-path comparison
  %(prog)s --type mobile_control_path --controller per_agent batch --command-interface per_agent fleet --agents 500 --steps 600
  %(prog)s --type mobile_control_path --controller batch --command-interface per_agent fleet --sweep 100 500 1000 --steps 300

  # Arm
  %(prog)s --type arm --agents 10 --duration 5
  %(prog)s --type arm --sweep 1 10 50 100
  %(prog)s --type arm --compare physics kinematic
  %(prog)s --type arm --agents 50 --scenario kinematic

Scenario names (--scenario / --compare) are defined in benchmark/configs/general.yaml.
Use --config to point at a custom YAML file.
""",
    )

    parser.add_argument(
        "--type",
        choices=["mobile", "arm", "mobile_control_path", "steady"],
        default="mobile",
        help="Benchmark type: mobile agents, arm robots, or mobile control-path comparison (default: mobile)",
    )

    # Basic options (shared)
    parser.add_argument("--agents", type=int, default=None, help="Number of agents/arms")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Simulation duration in seconds for mobile/arm benchmarks " "(omitted: use 10.0s; not supported for control-path)"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions (default: 3)")
    parser.add_argument("--gui", action="store_true", help="Enable GUI (mobile only)")
    parser.add_argument("--config", type=str, default=None, help="Path to benchmark config file")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name from config")
    parser.add_argument("--steps", type=int, default=None, help="Measurement steps for control-path and steady runs")
    parser.add_argument("--warmup", type=int, default=None, help="Steps before the measured window (steady only; default 20)")
    parser.add_argument("--mode", choices=["omni", "diff"], default="diff", help="Motion mode for mobile control-path runs")
    parser.add_argument(
        "--collision-freq",
        type=int,
        default=None,
        help=(
            "Override collision_check_frequency for mobile benchmarks "
            "(0 = disabled; control-path default is 60 when omitted)"
        ),
    )
    parser.add_argument(
        "--controller",
        nargs="+",
        choices=["per_agent", "batch"],
        default=None,
        help="Mobile controller implementation(s) to compare instead of scenario benchmarking",
    )
    parser.add_argument(
        "--command-interface",
        nargs="+",
        choices=["per_agent", "fleet"],
        default=None,
        help="Mobile command interface(s) to compare instead of scenario benchmarking",
    )

    # Sweep / compare
    parser.add_argument("--sweep", type=int, nargs="+", help="Sweep agent/arm counts")
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Compare scenarios (e.g., --compare physics kinematic)",
    )

    return parser.parse_args()


def _effective_benchmark_type(args) -> str:
    if args.type == "mobile" and (args.controller or args.command_interface):
        return "mobile_control_path"
    return args.type


def _validate_time_axis(args, effective_type: str) -> None:
    if effective_type in _STEP_AXIS_BENCHMARK_TYPES and args.duration is not None:
        raise SystemExit(f"--duration is not supported for --type {effective_type}; use --steps")
    if effective_type not in _STEP_AXIS_BENCHMARK_TYPES and args.steps is not None:
        raise SystemExit(
            f"--steps is only supported for control-path and steady benchmarks; use --duration for --type {args.type}"
        )
    if effective_type != "steady" and args.warmup is not None:
        raise SystemExit("--warmup is only supported for --type steady")


def get_output_dir(config_path: Optional[str]) -> str:
    """Get output directory from config file, falling back to benchmark/results/."""
    if config_path:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            output_dir = config.get("benchmark", {}).get("output_dir", ".")
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
        except (FileNotFoundError, yaml.YAMLError):
            pass
    # Default: benchmark/results/
    default_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


# ---------------------------------------------------------------------------
# Arm benchmark logic
# ---------------------------------------------------------------------------


def _run_arm_benchmark(args):
    """Run arm-specific benchmark flows."""
    if args.steps is not None:
        raise SystemExit("--steps is only supported for control-path benchmarks; use --duration for --type arm")
    duration = 10.0 if args.duration is None else args.duration
    default_agents = 10
    num_agents = args.agents or default_agents
    output_dir = get_output_dir(args.config)
    all_results = []

    if args.sweep:
        # Sweep mode: iterate arm counts × scenarios
        scenarios = [args.scenario] if args.scenario else ["physics", "kinematic"]
        for n in args.sweep:
            for s in scenarios:
                result = run_multiple(
                    num_agents=n,
                    duration=duration,
                    num_reps=args.repetitions,
                    benchmark_type="arm",
                    config_path=args.config,
                    scenario=s,
                    collision_freq=args.collision_freq,
                )
                all_results.append(result)

        print_sweep_summary(all_results)

        output_file = os.path.join(output_dir, f"arm_sweep_{duration}s.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSweep results saved to: {output_file}")

    elif args.compare:
        # Compare scenarios
        for scenario in args.compare:
            result = run_multiple(
                num_agents=num_agents,
                duration=duration,
                num_reps=args.repetitions,
                benchmark_type="arm",
                config_path=args.config,
                scenario=scenario,
                collision_freq=args.collision_freq,
            )
            all_results.append(result)

        print_sweep_summary(all_results)

        output_file = os.path.join(output_dir, f"arm_compare_{num_agents}arms.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison results saved to: {output_file}")

    else:
        # Single scenario
        result = run_multiple(
            num_agents=num_agents,
            duration=duration,
            num_reps=args.repetitions,
            benchmark_type="arm",
            config_path=args.config,
            scenario=args.scenario,
            collision_freq=args.collision_freq,
        )
        print_results(result)

        scenario_suffix = f"_{args.scenario}" if args.scenario else ""
        output_file = os.path.join(
            output_dir,
            f"arm_results_{num_agents}arms_{duration}s{scenario_suffix}.json",
        )
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# Mobile benchmark logic
# ---------------------------------------------------------------------------


def _run_mobile_benchmark(args):
    """Run mobile-specific benchmark flows (original behaviour)."""
    if args.controller or args.command_interface:
        print(
            "Note: --controller/--command-interface selected the mobile_control_path benchmark. "
            "Use --type mobile_control_path to make this explicit.",
            file=sys.stderr,
        )
        _run_mobile_control_path_benchmark(args)
        return
    if args.steps is not None:
        raise SystemExit("--steps is only supported for control-path benchmarks; use --duration for --type mobile")
    duration = 10.0 if args.duration is None else args.duration

    num_agents = args.agents or 1000
    output_dir = get_output_dir(args.config)
    all_results = []

    if args.sweep:
        for n in args.sweep:
            result = run_multiple(
                num_agents=n,
                duration=duration,
                num_reps=args.repetitions,
                benchmark_type="mobile",
                gui=args.gui,
                config_path=args.config,
                scenario=args.scenario,
                collision_freq=args.collision_freq,
            )
            all_results.append(result)

        print_sweep_summary(all_results)

        output_file = os.path.join(output_dir, f"benchmark_sweep_{duration}s.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSweep results saved to: {output_file}")

    elif args.compare:
        for scenario in args.compare:
            result = run_multiple(
                num_agents=num_agents,
                duration=duration,
                num_reps=args.repetitions,
                benchmark_type="mobile",
                gui=args.gui,
                config_path=args.config,
                scenario=scenario,
                collision_freq=args.collision_freq,
            )
            all_results.append(result)

        print_sweep_summary(all_results)

        output_file = os.path.join(output_dir, f"benchmark_compare_{num_agents}agents.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison results saved to: {output_file}")

    else:
        result = run_multiple(
            num_agents=num_agents,
            duration=duration,
            num_reps=args.repetitions,
            benchmark_type="mobile",
            gui=args.gui,
            config_path=args.config,
            scenario=args.scenario,
            collision_freq=args.collision_freq,
        )
        print_results(result)

        scenario_suffix = f"_{args.scenario}" if args.scenario else ""
        output_file = os.path.join(
            output_dir,
            f"benchmark_results_{num_agents}agents_{duration}s{scenario_suffix}.json",
        )
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# Mobile control-path benchmark logic
# ---------------------------------------------------------------------------


def _run_mobile_control_path_benchmark(args):
    """Run mobile controller/command-ingress benchmark flows."""
    if args.duration is not None:
        raise SystemExit("--duration is not supported for --type mobile_control_path; use --steps")
    steps = 600 if args.steps is None else args.steps
    num_agents = args.agents or 500
    output_dir = get_output_dir(args.config)
    all_results = []

    controllers = args.controller or ["batch"]
    command_interfaces = args.command_interface or ["fleet"]

    def run_one(n: int, controller: str, command_interface: str) -> dict:
        return run_multiple(
            num_agents=n,
            duration=0.0,
            num_reps=args.repetitions,
            benchmark_type="mobile_control_path",
            steps=steps,
            mode=args.mode,
            collision_freq=60 if args.collision_freq is None else args.collision_freq,
            controller=controller,
            command_interface=command_interface,
        )

    if args.sweep:
        for n in args.sweep:
            for controller in controllers:
                for command_interface in command_interfaces:
                    all_results.append(run_one(n, controller, command_interface))

        print_sweep_summary(all_results)

        output_file = os.path.join(output_dir, f"mobile_control_path_sweep_{steps}steps.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSweep results saved to: {output_file}")

    else:
        for controller in controllers:
            for command_interface in command_interfaces:
                all_results.append(run_one(num_agents, controller, command_interface))

        if len(all_results) > 1:
            print_sweep_summary(all_results)
        else:
            print_results(all_results[0])

        suffix = "_".join(controllers) + "__" + "_".join(command_interfaces)
        output_file = os.path.join(output_dir, f"mobile_control_path_{suffix}_{num_agents}agents.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison results saved to: {output_file}")


# ---------------------------------------------------------------------------
# Steady-state benchmark logic
# ---------------------------------------------------------------------------


def _run_steady_benchmark(args):
    """Run the steady-state scenes: a fixed window of steps with the whole fleet moving.

    ``--scenario`` picks one scene, ``--compare`` several; ``--sweep`` repeats them
    for each agent count. Every result is the median over ``--repetitions`` runs.
    """
    scenarios = args.compare or [args.scenario or "sparse"]
    unknown = [s for s in scenarios if s not in _STEADY_SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown steady scenario(s) {unknown}; choose from {list(_STEADY_SCENARIOS)}")
    counts = args.sweep or [args.agents or 1000]
    output_dir = get_output_dir(args.config)
    all_results = []
    for n in counts:
        for scenario in scenarios:
            all_results.append(
                run_multiple(
                    num_agents=n,
                    duration=0.0,
                    num_reps=args.repetitions,
                    benchmark_type="steady",
                    scenario=scenario,
                    steps=args.steps,
                    mode=args.mode,
                    collision_freq=args.collision_freq,
                    warmup=args.warmup,
                )
            )

    if len(all_results) > 1:
        print_sweep_summary(all_results)
    else:
        print_results(all_results[0])

    if args.sweep or args.compare:
        output_file = os.path.join(output_dir, f"steady_{'_'.join(scenarios)}_{'_'.join(str(c) for c in counts)}agents.json")
    else:
        output_file = os.path.join(output_dir, f"steady_{scenarios[0]}_{counts[0]}agents.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    effective_type = _effective_benchmark_type(args)
    _validate_time_axis(args, effective_type)

    type_label = {
        "arm": "Arm Robot",
        "mobile": "Mobile Agent",
        "mobile_control_path": "Mobile Control-Path",
        "steady": "Steady-State Fleet",
    }[effective_type]
    print("=" * 70)
    print(f"PyBullet Fleet - {type_label} Benchmark Runner")
    print("=" * 70)

    if args.type == "arm":
        _run_arm_benchmark(args)
    elif args.type == "mobile_control_path":
        _run_mobile_control_path_benchmark(args)
    elif args.type == "steady":
        _run_steady_benchmark(args)
    else:
        _run_mobile_benchmark(args)


if __name__ == "__main__":
    main()
