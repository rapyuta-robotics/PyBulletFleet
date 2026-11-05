"""Tests for benchmark/run_benchmark.py aggregation logic."""

import sys
import os
from unittest.mock import patch

# Ensure benchmark/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))

from run_benchmark import run_multiple, run_worker


def _make_mobile_result(**overrides):
    """Return a minimal mobile-worker result dict."""
    base = {
        "spawn_time_s": 0.1,
        "spawn_cpu_s": 0.05,
        "spawn_cpu_percent": 50.0,
        "simulation_wall_s": 1.0,
        "simulation_cpu_s": 0.8,
        "simulation_cpu_percent": 80.0,
        "real_time_factor": 1.0,
        "avg_step_time_ms": 1.0,
        "system_info": {"cpu": "test"},
        "mem_spawn_mb": {"rss_mb": 10.0},
        "mem_total_mb": {"rss_mb": 20.0},
        "expected_steps": 100,
    }
    base.update(overrides)
    return base


def _make_arm_result(**overrides):
    """Return a minimal arm-worker result dict."""
    base = _make_mobile_result()
    base.pop("expected_steps", None)
    base.update({"mode": "kinematic", "timestep": 0.01, "total_joints": 6})
    base.update(overrides)
    return base


def _make_batch_result(**overrides):
    """Return a minimal batch-worker result dict."""
    base = {
        "setup_s": 0.01,
        "accepted": 10,
        "rejected": 0,
        "wall_s": 0.2,
        "mean_ms": 1.0,
        "p50_ms": 0.9,
        "p95_ms": 1.5,
        "max_ms": 2.0,
        "state_snapshot_s": 0.001,
    }
    base.update(overrides)
    return base


class _Completed:
    returncode = 0
    stdout = "{}"
    stderr = ""


class TestRunWorker:
    @patch("run_benchmark.subprocess.run")
    def test_arm_forwards_collision_freq(self, mock_run):
        mock_run.return_value = _Completed()
        run_worker(3, 1.0, "arm", collision_freq=0)
        argv = mock_run.call_args.args[0]
        assert "--collision-freq" in argv
        assert argv[argv.index("--collision-freq") + 1] == "0"


class TestRunMultipleEmptyResults:
    """run_multiple must not crash when num_reps=0 (empty results)."""

    @patch("run_benchmark.run_worker")
    def test_mobile_zero_reps_does_not_raise(self, mock_worker):
        """IndexError guard: results[0] must not be accessed when results is empty."""
        mock_worker.return_value = None  # never called anyway
        result = run_multiple(
            num_agents=10,
            duration=1.0,
            num_reps=0,
            benchmark_type="mobile",
        )
        assert result["num_agents"] == 10
        assert result["num_reps"] == 0
        # Mobile-specific key should be absent
        assert "expected_steps" not in result
        assert result["command_setup_s"]["median"] == 0.0
        assert result["accepted_commands"]["median"] == 0.0
        assert result["rejected_commands"]["median"] == 0.0

    @patch("run_benchmark.run_worker")
    def test_arm_zero_reps_does_not_raise(self, mock_worker):
        """IndexError guard: arm-specific results[0] must be guarded."""
        mock_worker.return_value = None
        result = run_multiple(
            num_agents=5,
            duration=1.0,
            num_reps=0,
            benchmark_type="arm",
        )
        assert result["num_agents"] == 5
        assert result["num_reps"] == 0

    @patch("run_benchmark.run_worker")
    def test_batch_zero_reps_does_not_raise(self, mock_worker):
        mock_worker.return_value = None
        result = run_multiple(
            num_agents=5,
            duration=1.0,
            num_reps=0,
            benchmark_type="mobile_control_path",
            steps=10,
            controller="batch",
            command_interface="fleet",
        )
        assert result["num_agents"] == 5
        assert result["num_reps"] == 0
        assert result["controller_impl"] == "batch"
        assert result["command_interface"] == "fleet"

    @patch("run_benchmark.run_worker")
    def test_batch_zero_reps_reports_effective_defaults(self, mock_worker):
        mock_worker.return_value = None
        result = run_multiple(
            num_agents=5,
            duration=1.0,
            num_reps=0,
            benchmark_type="mobile_control_path",
        )
        assert result["steps"] == 600
        assert result["collision_freq"] == 60
        assert result["controller_impl"] == "batch"
        assert result["command_interface"] == "fleet"

    @patch("run_benchmark.run_worker")
    def test_batch_zero_reps_prints_effective_defaults(self, mock_worker, capsys):
        mock_worker.return_value = None
        run_multiple(
            num_agents=5,
            duration=1.0,
            num_reps=0,
            benchmark_type="mobile_control_path",
        )
        output = capsys.readouterr().out
        assert "600 steps" in output
        assert "controller=batch" in output
        assert "command_interface=fleet" in output
        assert "controller=None" not in output
        assert "command_interface=None" not in output


class TestRunMultipleWithResults:
    """run_multiple aggregates results correctly when num_reps >= 1."""

    @patch("run_benchmark.run_worker")
    def test_mobile_single_rep(self, mock_worker):
        mock_worker.return_value = _make_mobile_result()
        result = run_multiple(
            num_agents=10,
            duration=1.0,
            num_reps=1,
            benchmark_type="mobile",
        )
        assert result["num_agents"] == 10
        assert result["expected_steps"] == 100

    @patch("run_benchmark.run_worker")
    def test_arm_single_rep(self, mock_worker):
        mock_worker.return_value = _make_arm_result()
        result = run_multiple(
            num_agents=5,
            duration=1.0,
            num_reps=1,
            benchmark_type="arm",
        )
        assert result["mode"] == "kinematic"
        assert "expected_steps" not in result

    @patch("run_benchmark.run_worker")
    def test_batch_single_rep(self, mock_worker):
        mock_worker.return_value = _make_batch_result()
        result = run_multiple(
            num_agents=10,
            duration=1.0,
            num_reps=1,
            benchmark_type="mobile_control_path",
            steps=20,
            mode="diff",
            collision_freq=0,
            controller="batch",
            command_interface="fleet",
        )
        assert result["benchmark_type"] == "mobile_control_path"
        assert result["steps"] == 20
        assert result["controller_impl"] == "batch"
        assert result["command_interface"] == "fleet"
        assert result["mode"] == "diff"
        assert result["collision_freq"] == 0
        assert result["setup_s"]["median"] == 0.01
        assert result["accepted"]["median"] == 10


def _make_steady_result(**overrides):
    """Return a minimal steady-worker result dict."""
    base = {
        "warmup": 20,
        "steps": 150,
        "wall_s": 0.3,
        "mean_ms": 2.0,
        "p50_ms": 1.8,
        "p95_ms": 2.5,
        "max_ms": 3.0,
        "rtf": 55.0,
        "phases": {"collision_check": (0.8, 0.7, 1.0), "total": (2.0, 1.8, 2.5)},
        "collision_started": 3,
        "collision_ended": 2,
        "active_pairs_end": 1,
        "scenario": "sparse",
        "num_agents": 10,
        "num_static": 0,
        "mode": "diff",
        "collision_freq": 60,
        "spawn_s": 0.1,
        "accepted": 10,
    }
    base.update(overrides)
    return base


class TestSteadyWorker:
    @patch("run_benchmark.subprocess.run")
    def test_forwards_scene_steps_and_warmup(self, mock_run):
        mock_run.return_value = _Completed()
        run_worker(10, 0.0, "steady", scenario="warehouse", steps=50, warmup=5, collision_freq=0)
        argv = mock_run.call_args.args[0]
        assert argv[argv.index("--scenario") + 1] == "warehouse"
        assert argv[argv.index("--steps") + 1] == "50"
        assert argv[argv.index("--warmup") + 1] == "5"
        assert argv[argv.index("--collision-freq") + 1] == "0"
        assert "--json" in argv

    @patch("run_benchmark.subprocess.run")
    def test_defaults_to_sparse_150_steps_20_warmup(self, mock_run):
        mock_run.return_value = _Completed()
        run_worker(10, 0.0, "steady")
        argv = mock_run.call_args.args[0]
        assert argv[argv.index("--scenario") + 1] == "sparse"
        assert argv[argv.index("--steps") + 1] == "150"
        assert argv[argv.index("--warmup") + 1] == "20"


class TestSteadyAggregation:
    @patch("run_benchmark.run_worker")
    def test_medians_over_repetitions_and_per_phase(self, mock_worker):
        mock_worker.side_effect = [
            _make_steady_result(p50_ms=1.0, phases={"collision_check": (0.5, 0.4, 0.6), "total": (1.0, 1.0, 1.2)}),
            _make_steady_result(p50_ms=3.0, phases={"collision_check": (0.9, 0.8, 1.0), "total": (3.0, 3.0, 3.5)}),
            _make_steady_result(p50_ms=2.0, phases={"collision_check": (0.7, 0.6, 0.8), "total": (2.0, 2.0, 2.4)}),
        ]
        agg = run_multiple(10, 0.0, 3, "steady", scenario="dense", steps=30, warmup=5)
        assert agg["benchmark_type"] == "steady"
        assert agg["scenario"] == "dense"
        assert agg["steps"] == 30
        assert agg["num_reps"] == 3
        assert agg["p50_step_time_ms"]["median"] == 2.0
        assert agg["phases_p50_ms"]["collision_check"]["median"] == 0.6
        assert agg["phases_p50_ms"]["total"]["median"] == 2.0
        assert agg["collision_started"]["median"] == 3

    @patch("run_benchmark.run_worker")
    def test_empty_results_do_not_raise(self, mock_worker):
        mock_worker.side_effect = []
        agg = run_multiple(10, 0.0, 0, "steady")
        assert agg["num_reps"] == 0
        assert agg["p50_step_time_ms"]["median"] == 0.0
        assert agg["phases_p50_ms"] == {}
