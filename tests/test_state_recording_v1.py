"""End-to-end guarantees for the supported manipulation recording profile."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from pybullet_fleet.replay.state_recording import (
    DataRecord,
    ResultPlayback,
    load_recording_manifest,
    load_supported_checkpoint,
)
from pybullet_fleet.replay.kinematic_profile import KinematicProfile


@pytest.mark.parametrize(
    ("name", "version", "capture", "required"),
    [
        ("", 1, lambda context: None, False),
        ("custom", 0, lambda context: None, False),
        ("custom", 1, None, False),
        ("custom", 1, lambda context: None, 1),
    ],
)
def test_data_record_rejects_invalid_declaration_at_creation(name, version, capture, required) -> None:
    with pytest.raises(ValueError, match="Data record requires"):
        DataRecord(name, version, capture, required_for_restore=required)


def _profile(directory: Path) -> KinematicProfile:
    return KinematicProfile.from_manifest(load_recording_manifest(directory))


def _fresh_process(script: str, *args: str) -> dict:
    result = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout.split("RESULT:", 1)[1].strip().splitlines()[0])


@pytest.fixture
def recorded_run(tmp_path: Path) -> tuple[Path, dict]:
    directory = tmp_path / "recording"
    script = (
        "import json,sys; "
        "from pybullet_fleet.examples.validation.manipulation_state_scenario import run_scenario; "
        "r=run_scenario(record_output=sys.argv[1]); "
        "print('RESULT:'+json.dumps(r))"
    )
    source = _fresh_process(script, str(directory))
    return directory, source


def test_fresh_process_resume_matches_uninterrupted_run(recorded_run: tuple[Path, dict]) -> None:
    directory, source = recorded_run
    reference = _fresh_process(
        "import json; from pybullet_fleet.examples.validation.manipulation_state_scenario import run_scenario; "
        "print('RESULT:'+json.dumps(run_scenario(omni_profile=True)))"
    )
    assert source["final_stage"] == reference["final_stage"] == "done"
    assert source["trajectory"] == reference["trajectory"]

    script = (
        "import json,sys; "
        "from pybullet_fleet.examples.validation.manipulation_state_scenario import resume_scenario; "
        "r=resume_scenario(sys.argv[1],at_or_before=float(sys.argv[2])); "
        "print('RESULT:'+json.dumps(r))"
    )
    checkpoints = [json.loads(path.read_text()) for path in sorted((directory / "checkpoints").glob("*.json"))]
    active_turn_step = next(
        checkpoint["sim"]["step"]
        for checkpoint in checkpoints
        if checkpoint["agents"]["mobile-arm"]["state"]["controller"]["state"] is not None
        and checkpoint["agents"]["mobile-arm"]["state"]["controller"]["state"]["phase"] == "rotate"
        and checkpoint["agents"]["mobile-arm"]["state"]["controller"]["state"]["rotation"] is not None
    )
    for label, step in (
        ("CP1", source["observations"]["CP1"]["step"]),
        ("active-turn", active_turn_step),
        ("CP3", source["observations"]["CP3"]["step"]),
    ):
        manifest, checkpoint = load_supported_checkpoint(directory, at_or_before=step * 0.1, profile=_profile(directory))
        assert checkpoint["sim"]["step"] == step
        assert checkpoint["sim"]["elapsed_time"] == pytest.approx(step * manifest["construction"]["timestep"])
        if label == "CP3":
            assert checkpoint["objects"]["box-001"]["state"]["attachment"]["parent_name"] == "mobile-arm"
            joints = {joint["name"]: joint for joint in checkpoint["agents"]["mobile-arm"]["state"]["joints"]}
            assert joints["shoulder_to_elbow"]["target"] == -0.6
            assert "elbow_to_wrist" in joints
        boundary_script = (
            "import json,sys,pybullet as p; "
            "from pybullet_fleet.replay.state_recording import "
            "load_recording_manifest,load_supported_checkpoint,restore_supported_simulation; "
            "from pybullet_fleet.replay.kinematic_profile import KinematicProfile; "
            "profile=KinematicProfile.from_manifest(load_recording_manifest(sys.argv[1])); "
            "m,c=load_supported_checkpoint(sys.argv[1],at_or_before=float(sys.argv[2]),profile=profile); "
            "s=restore_supported_simulation(m,c,profile=profile); "
            "r=s.get_unique_agent_by_name('mobile-arm'); "
            "b=next((obj for obj in s.sim_objects if obj.name=='box-001'),None); "
            "print('RESULT:'+json.dumps({'step':s.get_completed_step_boundary()[0],"
            "'time':s.get_completed_step_boundary()[1],"
            "'pose':list(r.get_pose().position),"
            "'orientation':list(r.get_pose().orientation),"
            "'joints':{name:state[0] for name,state in r.get_all_joints_state_by_name().items()},"
            "'box':list(b.get_pose().position) if b else None,"
            "'attachment':b.get_kinematic_attachment_state() if b else None})); "
            "p.disconnect(s.client)"
        )
        boundary = _fresh_process(boundary_script, str(directory), str(step * 0.1))
        assert boundary["step"] == step
        assert boundary["time"] == pytest.approx(checkpoint["sim"]["elapsed_time"])
        assert boundary["pose"] == pytest.approx(checkpoint["agents"]["mobile-arm"]["state"]["pose"]["position"], abs=1e-6)
        assert boundary["orientation"] == pytest.approx(
            checkpoint["agents"]["mobile-arm"]["state"]["pose"]["orientation"], abs=1e-6
        )
        assert boundary["joints"] == pytest.approx(
            {joint["name"]: joint["position"] for joint in checkpoint["agents"]["mobile-arm"]["state"]["joints"]}, abs=1e-6
        )
        assert boundary["box"] == pytest.approx(checkpoint["objects"]["box-001"]["state"]["pose"]["position"], abs=1e-5)
        assert boundary["attachment"] == checkpoint["objects"]["box-001"]["state"]["attachment"]
        resumed = _fresh_process(script, str(directory), str(step * 0.1))
        assert resumed["final_stage"] == "done"
        assert resumed["trajectory"] == reference["trajectory"][step:]
        assert resumed["observations"]["after_delete"]["box_present"] is False


def test_playback_and_ordered_lifecycle_without_simulation_steps(
    recorded_run: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, source = recorded_run
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore

    def no_steps(*_args, **_kwargs):
        raise AssertionError("Playback must not execute a simulation step")

    monkeypatch.setattr(MultiRobotSimulationCore, "step_once", no_steps)
    playback = ResultPlayback(directory, profile=_profile(directory))
    assert len(playback.frames) == len(source["trajectory"])
    assert playback.seek_step(source["observations"]["CP3"]["step"])["objects"]["box-001"]["state"]["attachment"] is not None
    assert playback.step()["step"] == source["observations"]["CP3"]["step"] + 1
    assert "box-001" not in playback.seek_step(source["observations"]["after_delete"]["step"])["objects"]
    operations = [record["operation"] for record in playback.inputs]
    assert operations == ["spawn_object", "navigate", "joint_command", "attach", "joint_command", "attach", "remove_object"]
    assert playback.inputs[0]["step"] == 1
    assert playback.inputs[0]["phase"] == "PRE_STEP"
    assert playback.inputs[1]["phase"] == "POST_STEP"
    assert playback.inputs[-1]["step"] == source["observations"]["after_delete"]["step"]


def test_checkpoint_selection_and_rejection(recorded_run: tuple[Path, dict]) -> None:
    directory, _ = recorded_run
    _, state = load_supported_checkpoint(directory, at_or_before=1.95, profile=_profile(directory))
    assert state["sim"]["step"] == 19
    _, before_boundary = load_supported_checkpoint(directory, at_or_before=0.29999999999, profile=_profile(directory))
    assert before_boundary["sim"]["step"] == 2
    with pytest.raises(ValueError, match="No checkpoint"):
        load_supported_checkpoint(directory, at_or_before=0, profile=_profile(directory))

    manifest_path = directory / "manifest.json"
    original = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**original, "complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        load_supported_checkpoint(directory, at_or_before=1.9, profile=KinematicProfile.from_manifest(original))
    manifest_path.write_text(json.dumps(original))

    changed_asset = json.loads(manifest_path.read_text())
    changed_asset["construction"]["agents"]["mobile-arm"]["asset_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(changed_asset))
    with pytest.raises(ValueError, match="asset changed"):
        load_supported_checkpoint(directory, at_or_before=1.9, profile=KinematicProfile.from_manifest(original))
    manifest_path.write_text(json.dumps(original))

    checkpoint_path = directory / "checkpoints" / "000000025.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["objects"]["box-001"]["state"]["attachment"].pop("relative_position")
    checkpoint_path.write_text(json.dumps(checkpoint))
    original["checkpoint_sha256"]["25"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="attachment"):
        load_supported_checkpoint(directory, at_or_before=2.5, profile=_profile(directory))


@pytest.mark.parametrize("stream", ["frames.jsonl", "inputs.jsonl"])
def test_restore_rejects_corrupt_recording_stream(recorded_run: tuple[Path, dict], stream: str) -> None:
    directory, _ = recorded_run
    path = directory / stream
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="stream integrity"):
        load_supported_checkpoint(directory, at_or_before=2.5, profile=_profile(directory))


def test_required_custom_data_and_state_type_are_checked_before_restore(recorded_run: tuple[Path, dict]) -> None:
    directory, _ = recorded_run
    manifest_path = directory / "manifest.json"
    checkpoint_path = directory / "checkpoints" / "000000025.json"
    manifest = json.loads(manifest_path.read_text())
    original = json.loads(checkpoint_path.read_text())

    for change, message in (
        (lambda state: state["records"].pop("scenario"), "data records"),
        (lambda state: state["agents"]["mobile-arm"].update({"version": 3}), "Agent state type"),
    ):
        changed = json.loads(json.dumps(original))
        change(changed)
        checkpoint_path.write_text(json.dumps(changed))
        manifest["checkpoint_sha256"]["25"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=message):
            load_supported_checkpoint(directory, at_or_before=2.5, profile=_profile(directory))


@pytest.mark.parametrize("change", ["speed", "trajectory"])
def test_invalid_joint_execution_is_rejected_before_restore(recorded_run: tuple[Path, dict], change: str) -> None:
    directory, _ = recorded_run
    manifest_path = directory / "manifest.json"
    checkpoint_path = directory / "checkpoints" / "000000025.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    joint = checkpoint["agents"]["mobile-arm"]["state"]["joints"][0]
    if change == "speed":
        joint["speed"] = "invalid"
    else:
        joint["trajectory"] = {"p0": 0.0}
    checkpoint_path.write_text(json.dumps(checkpoint))
    manifest["checkpoint_sha256"]["25"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="joint"):
        load_supported_checkpoint(directory, at_or_before=2.5, profile=_profile(directory))


def test_undeclared_event_handler_is_not_silently_checkpointed() -> None:
    import pybullet as p

    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.events import SimEvents

    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, monitor=False, enable_monitor_gui=False, collision_check_frequency=0)
    )
    try:
        sim.events.on(SimEvents.PRE_STEP, lambda **_: None)
        profile = KinematicProfile()
        with pytest.raises(ValueError, match="Undeclared event handler"):
            profile.capture(sim)
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_all_kinematic_joint_positions_and_targets_round_trip() -> None:
    import pybullet as p

    from pybullet_fleet.agent import Agent, AgentSpawnParams
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose

    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, monitor=False, enable_monitor_gui=False, physics=False, enable_floor=False)
    )
    try:
        robot = Agent.from_params(
            AgentSpawnParams(
                name="mobile-arm",
                urdf_path=str(Path(__file__).resolve().parents[1] / "pybullet_fleet/robots/mobile_manipulator.urdf"),
                initial_pose=Pose.from_xyz(0, 0, 0.3),
                mass=0.0,
                controller={"type": "omni"},
            ),
            sim,
        )
        sim.initialize_simulation()
        indices = {info[1].decode("utf-8"): index for index, info in enumerate(robot.joint_info)}
        robot.set_joint_target(indices["shoulder_to_elbow"], 0.5)
        robot.set_joint_target(indices["elbow_to_wrist"], -0.4)
        sim.step_once()
        captured = robot.capture_kinematic_joint_execution()
        assert len(captured) == robot.get_num_joints()
        by_name = {joint["name"]: joint for joint in captured}
        assert by_name["shoulder_to_elbow"]["target"] == 0.5
        assert by_name["elbow_to_wrist"]["target"] == -0.4
        assert by_name["elbow_to_wrist"]["position"] != 0.0

        cleared = [{**joint, "position": 0.0, "target": None} for joint in captured]
        robot.restore_kinematic_joint_execution(cleared)
        robot.restore_kinematic_joint_execution(captured)
        assert robot.capture_kinematic_joint_execution() == captured
        with pytest.raises(ValueError, match="names or order"):
            robot.restore_kinematic_joint_execution(list(reversed(captured)))
        assert robot.capture_kinematic_joint_execution() == captured
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_recording_contract_accepts_an_independent_profile_and_custom_data(tmp_path: Path) -> None:
    """The writer must not know the manipulation example's robot or box names."""
    import pybullet as p
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.replay.state_recording import DataRecord, restore_supported_simulation

    class EmptyWorldProfile:
        profile_id = "test.empty_world.v1"
        version = 1
        coverage = {"inputs": "none", "results": "clock only", "checkpoints": "clock only", "unsupported": "entities"}

        def construction(self, sim):
            return {"timestep": sim.params.timestep}

        def capture(self, sim):
            step, elapsed_time = sim.get_completed_step_boundary()
            return {
                "sim": {"type": "kinematic", "version": 1, "step": step, "elapsed_time": elapsed_time},
                "agents": {},
                "objects": {},
            }

        def validate_checkpoint(self, manifest, state):
            if state["agents"] or state["objects"]:
                raise ValueError("Unexpected entity in empty-world profile")

        def restore(self, manifest, state, *, gui, target_rtf):
            restored = MultiRobotSimulationCore(
                SimulationParams(gui=False, physics=False, enable_floor=False, timestep=manifest["construction"]["timestep"])
            )
            restored.initialize_simulation()
            restored.restore_completed_step_boundary(state["sim"]["step"], state["sim"]["elapsed_time"])
            return restored

    profile = EmptyWorldProfile()
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, enable_floor=False, timestep=0.1))
    try:
        sim.configure_state_recording(
            output=str(tmp_path / "other-profile"),
            profile=profile,
            records=(DataRecord("custom", 2, lambda context: {"observed_step": context.step}, required_for_restore=True),),
        )
        sim.run_simulation(duration=0.2)
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)

    directory = tmp_path / "other-profile"
    manifest, checkpoint = load_supported_checkpoint(directory, at_or_before=0.2, profile=profile)
    assert manifest["profile"] == profile.profile_id
    assert checkpoint["records"]["custom"] == {"version": 2, "value": {"observed_step": 2}}
    assert checkpoint["agents"] == checkpoint["objects"] == {}
    restored = restore_supported_simulation(manifest, checkpoint, profile=profile)
    try:
        assert restored.get_completed_step_boundary() == (2, pytest.approx(0.2))
    finally:
        if p.isConnected(restored.client):
            p.disconnect(restored.client)


def test_failed_capture_does_not_stop_simulation_or_publish_complete_artifact(tmp_path: Path) -> None:
    import pybullet as p
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    class FailingProfile:
        profile_id = "test.failing.v1"
        version = 1
        coverage = {}

        def construction(self, sim):
            return {"timestep": sim.params.timestep}

        def capture(self, sim):
            raise ValueError("unsupported state")

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, enable_floor=False, timestep=0.1))
    directory = tmp_path / "failed"
    try:
        sim.configure_state_recording(output=str(directory), profile=FailingProfile())
        sim.run_simulation(duration=0.2)
        assert sim.get_completed_step_boundary() == (2, pytest.approx(0.2))
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["complete"] is False
        assert "unsupported state" in manifest["recording_error"]
        with pytest.raises(ValueError, match="Incomplete"):
            load_recording_manifest(directory)
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_gui_playback_uses_first_available_checkpoint(recorded_run, monkeypatch: pytest.MonkeyPatch) -> None:
    import pybullet_fleet.replay.kinematic_profile as module

    directory, _ = recorded_run
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_sha256"].pop("1")
    manifest_path.write_text(json.dumps(manifest))
    playback = ResultPlayback(directory, profile=_profile(directory))

    def inspect_first_checkpoint(_manifest, state, **_kwargs):
        assert state["sim"]["step"] == 2
        raise RuntimeError("selected second checkpoint")

    monkeypatch.setattr(module, "restore_supported_simulation", inspect_first_checkpoint)
    with pytest.raises(RuntimeError, match="selected second checkpoint"):
        playback.play_gui(rtf=1.0)


def test_gui_playback_follows_box_attachment_and_removal_without_steps(
    recorded_run: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    import pybullet_fleet.replay.kinematic_profile as module
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore

    directory, _ = recorded_run
    restore = module.restore_supported_simulation
    simulations = []

    def headless_restore(manifest, state, **kwargs):
        sim = restore(manifest, state, **{**kwargs, "gui": False})
        simulations.append(sim)
        return sim

    def no_steps(*_args, **_kwargs):
        raise AssertionError("Playback must not execute a simulation step")

    monkeypatch.setattr(module, "restore_supported_simulation", headless_restore)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(MultiRobotSimulationCore, "step_once", no_steps)
    playback = ResultPlayback(directory, profile=_profile(directory))
    playback.play_gui(rtf=100.0)
    assert simulations and simulations[0].find_objects_by_name("box-001") == []


def test_spawn_record_failure_preserves_spawn_and_marks_artifact_incomplete(tmp_path: Path) -> None:
    import pybullet as p
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams

    class RejectingProfile:
        profile_id = "test.reject_spawn.v1"
        version = 1
        coverage = {}

        def construction(self, sim):
            return {"timestep": sim.params.timestep}

        def spawn_record(self, obj, params):
            raise ValueError("unsupported spawn")

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, enable_floor=False))
    directory = tmp_path / "spawn-failure"
    try:
        sim.configure_state_recording(output=str(directory), profile=RejectingProfile())
        sim.initialize_simulation()
        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            initial_pose=Pose.from_xyz(0, 0, 0.5),
            name="recording-rejected-box",
        )
        obj = SimObject.from_params(params, sim)
        assert obj in sim.sim_objects
        assert p.getBodyInfo(obj.body_id, physicsClientId=sim.client)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["complete"] is False
        assert "unsupported spawn" in manifest["recording_error"]
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


@pytest.mark.parametrize("completed_steps", [0, 1])
def test_close_flushes_inputs_issued_outside_last_step(tmp_path: Path, completed_steps: int) -> None:
    import pybullet as p
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    class ClockProfile:
        profile_id = "test.clock.v1"
        version = 1
        coverage = {}

        def construction(self, sim):
            return {"timestep": sim.params.timestep}

        def capture(self, sim):
            step, elapsed_time = sim.get_completed_step_boundary()
            return {
                "sim": {"type": "kinematic", "version": 1, "step": step, "elapsed_time": elapsed_time},
                "agents": {},
                "objects": {},
            }

        def validate_checkpoint(self, manifest, state):
            pass

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, enable_floor=False, timestep=0.1))
    directory = tmp_path / "outside-step"
    try:
        recorder = sim.configure_state_recording(output=str(directory), profile=ClockProfile())
        sim.initialize_simulation()
        for _ in range(completed_steps):
            sim.step_once()
        sim._record_state_input("test_command", {"accepted": True})
        recorder.close()
        assert sim._state_recorder is None
        playback = ResultPlayback(directory)
        assert len(playback.frames) == completed_steps
        assert playback.inputs == [
            {
                "operation": "test_command",
                "details": {"accepted": True},
                "step": completed_steps,
                "phase": "outside_step",
                "order": 0,
            }
        ]
        sim.step_once()
        assert json.loads((directory / "manifest.json").read_text())["last_completed_step"] == completed_steps
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)
