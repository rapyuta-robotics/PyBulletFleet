"""Smoke tests for benchmark/steady_bench.py: every scene builds, runs and reports."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))

import steady_bench  # noqa: E402

_RESULT_KEYS = {"mean_ms", "p50_ms", "p95_ms", "max_ms", "rtf", "phases", "collision_started", "collision_ended"}


@pytest.mark.parametrize("scenario", steady_bench.SCENARIOS)
def test_scene_runs_and_reports(scenario):
    result = steady_bench.run(scenario, 4, warmup=1, steps=3, field_width=2)
    assert _RESULT_KEYS <= set(result)
    assert result["scenario"] == scenario
    assert result["num_agents"] == 4
    assert result["steps"] == 3
    assert "total" in result["phases"]
    assert result["p50_ms"] > 0


def test_idle_sends_no_goals_and_sparse_sends_one_per_robot():
    idle = steady_bench.run("idle", 4, warmup=1, steps=2)
    sparse = steady_bench.run("sparse", 4, warmup=1, steps=2)
    assert idle["accepted"] == 0
    assert sparse["accepted"] == 4


def test_dense_keeps_every_tenth_robot_stopped_and_collides():
    result = steady_bench.run("dense", 20, warmup=1, steps=2)
    assert result["accepted"] == 18
    assert result["num_static"] == 0


def test_warehouse_counts_its_static_boxes():
    result = steady_bench.run("warehouse", 4, warmup=1, steps=2, field_width=2)
    # Tiles on every (i, j) of x in [-2, 230] and y in [-2, 4] (4 agents sit on a 2 x 2 grid, 2 m apart),
    # plus a bin on three of four columns of each odd row.
    columns = range(-2, 231)
    rows = range(-2, 5)
    tiles = len(columns) * len(rows)
    bins = sum(1 for i in columns for j in rows if j % 2 == 1 and i % 4 < 3)
    assert result["num_static"] == tiles + bins


def test_build_scenario_rejects_unknown_scene():
    with pytest.raises(ValueError):
        steady_bench.build_scenario("nope", 1, "diff", 60)
