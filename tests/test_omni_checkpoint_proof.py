"""A fresh-process continuation must match the uninterrupted omni suffix."""

import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pybullet as p
import pytest

from pybullet_fleet.events import SimEvents
from pybullet_fleet.examples.validation import omni_checkpoint_proof as proof


def _command(mode, output, checkpoint=None):
    command = [sys.executable, "-m", "pybullet_fleet.examples.validation.omni_checkpoint_proof", mode, "--output", str(output)]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(output.read_text())


def test_fresh_process_restore_matches_uninterrupted_suffix(tmp_path):
    reference = _command("reference", tmp_path / "reference.json")
    checkpoint_path = tmp_path / "checkpoint.json"
    source = _command("source", tmp_path / "source.json", checkpoint_path)
    # The source process has exited before a separate interpreter restores it.
    restored = _command("restore", tmp_path / "restored.json", checkpoint_path)

    assert source == json.loads(checkpoint_path.read_text())
    assert source["state"]["step"] == proof.CHECKPOINT_STEP
    assert source["state"]["moving"] is True
    assert len(restored) == len(reference[proof.CHECKPOINT_STEP :])
    for expected, actual in zip(reference[proof.CHECKPOINT_STEP :], restored):
        assert actual["step"] == expected["step"]
        assert actual["elapsed_time"] == pytest.approx(expected["elapsed_time"], abs=1e-10)
        assert actual["position"] == pytest.approx(expected["position"], abs=1e-8)
        assert actual["orientation"] == pytest.approx(expected["orientation"], abs=1e-8)
        assert actual["velocity"] == pytest.approx(expected["velocity"], abs=1e-8)
        assert actual["moving"] is expected["moving"]
    assert restored[0]["position"] == pytest.approx(source["state"]["position"])
    assert restored[-1]["step"] == reference[-1]["step"]
    assert restored[-1]["moving"] is False
    assert restored[-1]["position"] == pytest.approx(proof.GOAL.position)
    assert sum(not sample["moving"] for sample in restored) == 1


def test_cli_omitted_paths_are_named_and_reported(tmp_path):
    command = [sys.executable, "-m", "pybullet_fleet.examples.validation.omni_checkpoint_proof"]
    environment = {**os.environ, "TMPDIR": str(tmp_path)}
    reference = subprocess.run(command + ["reference"], check=True, capture_output=True, text=True, env=environment)
    reference_path = next(
        Path(line.removeprefix("Saved result to "))
        for line in reference.stdout.splitlines()
        if line.startswith("Saved result to ")
    )
    assert reference_path.parent.parent == tmp_path
    assert json.loads(reference_path.read_text())[-1]["moving"] is False

    source = subprocess.run(command + ["source"], check=True, capture_output=True, text=True, env=environment)
    paths = {
        label: Path(line.removeprefix(prefix))
        for line in source.stdout.splitlines()
        for label, prefix in (("result", "Saved result to "), ("checkpoint", "Saved checkpoint to "))
        if line.startswith(prefix)
    }
    assert paths["result"].parent.parent == tmp_path
    assert paths["checkpoint"].parent == paths["result"].parent
    assert json.loads(paths["result"].read_text()) == json.loads(paths["checkpoint"].read_text())


def test_gui_rtf_is_checked_before_opening_window():
    command = [
        sys.executable,
        "-m",
        "pybullet_fleet.examples.validation.omni_checkpoint_proof",
        "reference",
        "--gui",
        "--rtf",
        "0",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "--rtf must be finite and positive" in completed.stderr


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(version=3), "profile or version"),
        (lambda c: c["conditions"].update(timestep=0.2), "conditions differ"),
        (lambda c: c["state"].update(elapsed_time=0.4), "clock is inconsistent"),
        (lambda c: c["state"]["controller"]["forward"].update(vmax=float("nan")), "finite number"),
        (lambda c: c["state"]["controller"]["forward"].pop("origin"), "missing or unsupported"),
        (lambda c: c["state"].update(position=[0.2, 0, 0.1]), "does not match"),
    ],
)
def test_incomplete_or_incompatible_checkpoint_is_rejected(tmp_path, mutate, message):
    checkpoint = proof.run_source(tmp_path / "valid.json")
    invalid = copy.deepcopy(checkpoint)
    mutate(invalid)
    with pytest.raises(ValueError, match=message):
        proof.validate_checkpoint(invalid)


def test_reissue_goal_from_checkpoint_pose_changes_trajectory(tmp_path):
    checkpoint = proof.run_source(tmp_path / "checkpoint.json")
    reference = proof.run_reference()
    sim, robot = proof._make_sim()
    try:
        state = checkpoint["state"]
        robot.set_pose(proof.Pose(position=state["position"], orientation=state["orientation"]))
        sim.restore_completed_step_boundary(state["step"], state["elapsed_time"])
        proof._issue_goal(robot)
        sim.step_once()
        assert math.dist(robot.get_pose().position, reference[proof.CHECKPOINT_STEP + 1]["position"]) > 0.01
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_capture_rejected_inside_post_step_and_resume_requires_restore():
    sim, robot = proof._make_sim()
    try:
        with pytest.raises(RuntimeError, match="restored completed-step"):
            sim.run_simulation(resume=True)
        proof._issue_goal(robot)
        errors = []

        def after_step(**_):
            with pytest.raises(RuntimeError, match="during step_once"):
                sim.get_completed_step_boundary()
            errors.append(sim.step_count)

        sim.events.on(SimEvents.POST_STEP, after_step)
        sim.step_once()
        assert errors == [1]  # POST_STEP sees the counter of the step it completes
        assert sim.get_completed_step_boundary() == (1, proof.DT)
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_resumed_monitor_wall_time_starts_at_run_not_checkpoint_load(tmp_path):
    checkpoint = proof.run_source(tmp_path / "checkpoint.json")
    sim, robot = proof._make_sim()
    try:
        proof.restore_checkpoint(sim, robot, checkpoint)
        sim._start_time = 0.0  # Distinguish the monitor origin from initialization.
        before = time.monotonic()
        sim.run_simulation(duration=checkpoint["state"]["elapsed_time"], resume=True)
        after = time.monotonic()
        assert before <= sim._start_time <= after
        assert sim.get_completed_step_boundary() == (proof.CHECKPOINT_STEP, checkpoint["state"]["elapsed_time"])
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)
