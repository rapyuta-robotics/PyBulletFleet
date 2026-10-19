"""Fresh-process acceptance for the PBF-owned kinematic profile."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pybullet_fleet.replay import load_recording_manifest


def _run_python(script: str, *args: str) -> dict | None:
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    marked = [line.removeprefix("RESULT:") for line in result.stdout.splitlines() if line.startswith("RESULT:")]
    return json.loads(marked[-1]) if marked else None


def test_default_panda_recording_scenario_reaches_place(tmp_path: Path) -> None:
    artifact = tmp_path / "panda-recording"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pybullet_fleet.examples.validation.pick_drop_arm_recording",
            "--headless",
            "--duration",
            "5",
            "--rtf",
            "0",
            "--record",
            str(artifact),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    assert "Dropped box (detached)" in result.stdout
    assert "Expected 12 targets, got 11" not in result.stderr
    assert load_recording_manifest(artifact)["complete"] is True


def test_supported_agent_construction_uses_effective_spawn_params() -> None:
    import pybullet as p

    from pybullet_fleet.agent import Agent, AgentSpawnParams
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        initial_pose = Pose.from_xyz(1, 2, 0.05)
        agent = Agent.from_urdf(
            "pybullet_fleet/robots/mobile_robot.urdf",
            pose=initial_pose,
            mass=0,
            name="direct-agent",
            sim_core=sim,
        )
        params = agent._checkpoint_spawn_params
        assert type(params) is AgentSpawnParams
        initial_pose.position[0] = 9
        assert params.initial_pose.position[0] == pytest.approx(1)
        assert params.controller == agent.controller._construction_config()
        assert isinstance(params.controller, dict)
        assert params.controller["type"] == "differential"
        construction = KinematicProfile().construction(sim)["agents"]["direct-agent"]
        assert construction["asset"] == str(Path(params.urdf_path).resolve())
        assert construction["initial_pose"]["position"] == list(params.initial_pose.position)
        assert construction["mass"] == params.mass
        assert construction["controller"]["type"] == params.controller["type"]
        assert construction["controller"]["parameters"] == {
            key: value for key, value in params.controller.items() if key != "type"
        }
    finally:
        p.disconnect(sim.client)


def test_unrepresented_agent_spawn_option_is_not_checkpointable() -> None:
    import pybullet as p

    from pybullet_fleet.agent import Agent, AgentSpawnParams
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        agent = Agent.from_urdf(
            "pybullet_fleet/robots/mobile_robot.urdf",
            name="scaled-agent",
            global_scaling=1.2,
            sim_core=sim,
        )
        params = agent._checkpoint_spawn_params
        assert type(params) is AgentSpawnParams
        assert params.global_scaling == 1.2
        with pytest.raises(ValueError, match="Unsupported Agent spawn options"):
            KinematicProfile().construction(sim)
    finally:
        p.disconnect(sim.client)


@pytest.fixture(scope="module")
def fixed_arm_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("supported-arm") / "recording"
    _run_python(
        "from pybullet_fleet.examples.validation.pick_drop_arm_recording import *; "
        "s=build_example('arm_robot',gui=False,recording=True); s.params.target_rtf=0; "
        "configure_demo(s,'arm_robot',recording=True); "
        "s.configure_state_recording(profile=KinematicProfile(allowed_callbacks=(pick_drop_callback,)),"
        "output=__import__('sys').argv[1],records=(DataRecord(RECORD_NAME,1,capture_stage,True),)); "
        "s.run_simulation(duration=8.0)",
        str(path),
    )
    assert load_recording_manifest(path)["complete"] is True
    return path


def _reference_frames(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]


def test_mismatched_joint_trajectory_target_is_rejected(fixed_arm_artifact: Path) -> None:
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile

    manifest = load_recording_manifest(fixed_arm_artifact)
    checkpoint = next((fixed_arm_artifact / "checkpoints").glob("*.json"))
    state = json.loads(checkpoint.read_text())
    joint = state["agents"]["pick-drop-arm"]["state"]["joints"][0]
    joint["max_accel"] = 1.0
    joint["max_decel"] = 1.0
    joint["target"] = 1.0
    joint["trajectory"] = {
        "p0": 0.0,
        "pe": 1.0,
        "v0": 0.0,
        "ve": 0.0,
        "vmax": 1.0,
        "accel": 1.0,
        "decel": 1.0,
        "t0_offset": 0.0,
        "target": 1.1,
    }
    with pytest.raises(ValueError, match="Joint trajectory target differs"):
        KinematicProfile().validate_checkpoint(manifest, state)


def test_kinematic_gui_playback_supports_fixed_arm_entities_without_stepping(
    fixed_arm_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared renderer must use artifact names, not mobile-arm names."""
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore
    from pybullet_fleet.replay import ResultPlayback
    from pybullet_fleet.replay import kinematic_profile
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile

    restore = kinematic_profile.restore_supported_simulation
    rendered: list[str] = []

    def headless_restore(manifest, state, **kwargs):
        sim = restore(manifest, state, **{**kwargs, "gui": False})
        original = sim.get_unique_agent_by_name("pick-drop-arm").set_pose

        def set_pose(pose, preserve_velocity=True):
            rendered.append("pick-drop-arm")
            return original(pose, preserve_velocity=preserve_velocity)

        sim.get_unique_agent_by_name("pick-drop-arm").set_pose = set_pose
        return sim

    def no_steps(*_args, **_kwargs):
        raise AssertionError("Playback must not execute a simulation step")

    monkeypatch.setattr(kinematic_profile, "restore_supported_simulation", headless_restore)
    monkeypatch.setattr(kinematic_profile.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(MultiRobotSimulationCore, "step_once", no_steps)
    playback = ResultPlayback(fixed_arm_artifact, profile=KinematicProfile())
    playback.play_gui(rtf=100.0)
    assert rendered == ["pick-drop-arm"] * len(playback.frames)


@pytest.mark.parametrize("phase", ["moving_joint", "attached_joint"])
def test_fixed_arm_validation_restores_and_continues_in_fresh_process(fixed_arm_artifact: Path, phase: str) -> None:
    frames = _reference_frames(fixed_arm_artifact)
    checkpoints = [json.loads(path.read_text()) for path in sorted((fixed_arm_artifact / "checkpoints").glob("*.json"))]
    candidates = []
    for state in checkpoints:
        arm = state["agents"]["pick-drop-arm"]["state"]
        box = state["objects"]["pick-drop-box"]["state"]
        moving_joint = any(j["target"] is not None and abs(j["position"] - j["target"]) > 0.1 for j in arm["joints"])
        if moving_joint and (box["attachment"] is not None) == (phase == "attached_joint"):
            candidates.append(state)
    assert candidates, f"No {phase} checkpoint in the normal example"
    checkpoint = candidates[len(candidates) // 2]
    step = checkpoint["sim"]["step"]
    result = _run_python(
        """import json, sys
from pybullet_fleet.examples.validation.pick_drop_arm_recording import load_supported_simulation
from pybullet_fleet.events import SimEvents
s = load_supported_simulation(sys.argv[1], at_or_before=float(sys.argv[2]))
out = []
def sample(**kwargs):
    arm = s.get_unique_agent_by_name('pick-drop-arm')
    box = s.get_unique_object_by_name('pick-drop-box')
    out.append({'step': s.step_count,
                'joints': arm.capture_kinematic_joint_execution(),
                'box': list(box.get_pose().position),
                'attached': box.get_kinematic_attachment_state()})
s.events.on(SimEvents.POST_STEP, sample)
s.run_simulation(duration=8.0)
print('RESULT:' + json.dumps({'trajectory': out, 'time': s.get_completed_step_boundary()[1]}))
""",
        str(fixed_arm_artifact),
        str(step * 0.1),
    )
    assert result is not None
    actual = result["trajectory"]
    assert actual and actual[0]["step"] == step + 1
    assert len(actual) == len(frames) - step
    for observed in actual:
        expected = frames[observed["step"] - 1]
        arm_state = expected["agents"]["pick-drop-arm"]["state"]
        box_state = expected["objects"]["pick-drop-box"]["state"]
        assert [j["position"] for j in observed["joints"]] == pytest.approx(
            [j["position"] for j in arm_state["joints"]], abs=1e-6
        )
        assert observed["box"] == pytest.approx(box_state["pose"]["position"], abs=1e-6)
        assert (observed["attached"] is not None) == (box_state["attachment"] is not None)
    assert result["time"] == pytest.approx(8.0)


def test_missing_application_restorer_fails_before_construction(fixed_arm_artifact: Path) -> None:
    output = _run_python(
        "import json,sys; from pybullet_fleet.replay import load_supported_simulation; "
        "\ntry: load_supported_simulation(sys.argv[1],at_or_before=1.0)"
        "\nexcept ValueError as e: print('RESULT:'+json.dumps({'error':str(e)}))",
        str(fixed_arm_artifact),
    )
    assert output is not None and "No trusted data restorer" in output["error"]


def test_mobile_manipulator_composition_uses_same_public_loader(tmp_path: Path) -> None:
    """Base navigation plus joints/attachment shares the fixed-arm handlers."""
    directory = tmp_path / "mobile-recording"
    _run_python(
        """import sys
from pathlib import Path
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import SimObject, SimObjectSpawnParams, ShapeParams
from pybullet_fleet.types import CollisionMode
from pybullet_fleet.replay.kinematic_profile import KinematicProfile
s = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, enable_floor=False,
    timestep=0.1, collision_check_frequency=0, target_rtf=0))
arm = Agent.from_params(AgentSpawnParams(name='mobile-arm',
    urdf_path=str(Path('pybullet_fleet/robots/mobile_manipulator.urdf').resolve()),
    initial_pose=Pose.from_xyz(0, 0, 0.3), mass=0, use_fixed_base=False,
    motion_mode='omnidirectional',
    controller={'type': 'omni', 'navigation_2d': True,
                'max_linear_vel': 1, 'max_linear_accel': 2, 'max_linear_decel': 0.5}), s)
box = SimObject.from_params(SimObjectSpawnParams(name='box', initial_pose=Pose.from_xyz(0.5, 0, 0.5),
    mass=0, collision_mode=CollisionMode.DISABLED,
    visual_shape=ShapeParams(shape_type='box', half_extents=[0.05]*3),
    collision_shape=ShapeParams(shape_type='box', half_extents=[0.05]*3)), s)
arm.set_goal_pose(Pose.from_xyz(2, 0, 0.3))
arm.set_joint_motion_profile('shoulder_to_elbow', max_velocity=0.4, max_accel=1.0, max_decel=0.3)
arm.set_joint_target_by_name('shoulder_to_elbow', 0.6)
assert arm.attach_object(box, parent_link_index='end_effector', relative_pose=Pose.from_xyz(0, 0, 0.1))
s.configure_state_recording(profile=KinematicProfile(), output=sys.argv[1])
s.run_simulation(duration=2.0)
""",
        str(directory),
    )
    frames = _reference_frames(directory)
    checkpoint = json.loads((directory / "checkpoints" / "000000008.json").read_text())
    assert load_recording_manifest(directory)["construction"]["agents"]["mobile-arm"]["motion_mode"] == "omnidirectional"
    assert checkpoint["agents"]["mobile-arm"]["state"]["moving"] is True
    assert checkpoint["agents"]["mobile-arm"]["state"]["controller"]["state"]["forward"]["decel"] == pytest.approx(0.5)
    assert checkpoint["objects"]["box"]["state"]["attachment"] is not None
    checkpoint_joint = next(
        joint for joint in checkpoint["agents"]["mobile-arm"]["state"]["joints"] if joint["name"] == "shoulder_to_elbow"
    )
    assert checkpoint_joint["speed"] > 0
    assert checkpoint_joint["trajectory"] is not None
    assert checkpoint_joint["max_decel"] == pytest.approx(0.3)
    result = _run_python(
        """import json, sys
from pybullet_fleet.replay import load_supported_simulation
from pybullet_fleet.events import SimEvents
s = load_supported_simulation(sys.argv[1], at_or_before=0.8)
out = []
def sample(**kwargs):
    arm = s.get_unique_agent_by_name('mobile-arm')
    box = s.get_unique_object_by_name('box')
    out.append({'step': s.step_count, 'pose': list(arm.get_pose().position),
                'joints': arm.capture_kinematic_joint_execution(),
                'box': list(box.get_pose().position), 'attached': box.get_kinematic_attachment_state()})
s.events.on(SimEvents.POST_STEP, sample)
s.run_simulation(duration=2.0)
print('RESULT:' + json.dumps({'trajectory': out, 'time': s.get_completed_step_boundary()[1],
                              'motion_mode': s.get_unique_agent_by_name('mobile-arm').motion_mode.value}))
""",
        str(directory),
    )
    assert result is not None
    assert result["motion_mode"] == "omnidirectional"
    assert len(result["trajectory"]) == len(frames) - 8
    for observed in result["trajectory"]:
        frame = frames[observed["step"] - 1]
        arm = frame["agents"]["mobile-arm"]["state"]
        box = frame["objects"]["box"]["state"]
        assert observed["pose"] == pytest.approx(arm["pose"]["position"], abs=1e-6)
        assert [j["position"] for j in observed["joints"]] == pytest.approx([j["position"] for j in arm["joints"]], abs=1e-6)
        assert [j["speed"] for j in observed["joints"]] == pytest.approx([j["speed"] for j in arm["joints"]], abs=1e-6)
        assert observed["box"] == pytest.approx(box["pose"]["position"], abs=1e-6)
        assert observed["attached"] is not None
    assert result["time"] == pytest.approx(2.0)


@pytest.mark.parametrize("checkpoint_time, phase", [(0.2, "rotate"), (3.0, "forward")])
def test_asymmetric_differential_navigation_resumes_in_fresh_process(
    tmp_path: Path, checkpoint_time: float, phase: str
) -> None:
    directory = tmp_path / "differential"
    _run_python(
        """import sys
from pathlib import Path
from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.replay.kinematic_profile import KinematicProfile
s = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, timestep=0.1,
    collision_check_frequency=0, target_rtf=0))
a = Agent.from_params(AgentSpawnParams(name='diff',
    urdf_path=str(Path('pybullet_fleet/robots/mobile_robot.urdf').resolve()),
    initial_pose=Pose.from_yaw(0, 0, 0.05, 1.57), mass=0,
    controller={'type': 'differential', 'max_linear_vel': 1.0,
                'max_linear_accel': 1.0, 'max_linear_decel': 0.3,
                'max_angular_vel': 1.0, 'max_angular_accel': 1.0}), s)
a.set_goal_pose(Pose.from_xyz(2, 0, 0.05))
s.configure_state_recording(profile=KinematicProfile(), output=sys.argv[1])
s.run_simulation(duration=7.0)
""",
        str(directory),
    )
    frames = _reference_frames(directory)
    checkpoint = max(
        (
            json.loads(path.read_text())
            for path in (directory / "checkpoints").glob("*.json")
            if int(path.stem) <= round(checkpoint_time / 0.1)
        ),
        key=lambda state: state["sim"]["step"],
    )
    navigation = checkpoint["agents"]["diff"]["state"]["controller"]["state"]
    assert navigation is not None
    if phase == "forward":
        from pybullet_fleet.replay.kinematic_profile import KinematicProfile

        changed = json.loads(json.dumps(checkpoint))
        changed["agents"]["diff"]["state"]["controller"]["state"]["goal"]["position"][0] += 1
        with pytest.raises(ValueError, match="Navigation goal differs from active waypoint"):
            KinematicProfile().validate_checkpoint(load_recording_manifest(directory), changed)
    assert navigation["phase"] == phase
    assert navigation["movement_direction"] == "forward"
    assert navigation["requested_direction"] == "forward"
    if navigation["phase"] == "forward":
        assert navigation["forward"]["decel"] == pytest.approx(0.3)
    result = _run_python(
        """import json, sys
from pybullet_fleet.replay import load_supported_simulation
from pybullet_fleet.events import SimEvents
s = load_supported_simulation(sys.argv[1], at_or_before=float(sys.argv[2]))
out = []
def sample(**kwargs):
    a = s.get_unique_agent_by_name('diff')
    out.append({'step': s.step_count, 'position': list(a.get_pose().position),
                'orientation': list(a.get_pose().orientation), 'moving': a.is_moving})
s.events.on(SimEvents.POST_STEP, sample)
s.run_simulation(duration=7.0)
print('RESULT:' + json.dumps(out))
""",
        str(directory),
        str(checkpoint_time),
    )
    assert result is not None
    assert len(result) == len(frames) - checkpoint["sim"]["step"]
    for sample in result:
        expected = frames[sample["step"] - 1]["agents"]["diff"]["state"]
        assert sample["position"] == pytest.approx(expected["pose"]["position"], abs=1e-6)
        assert sample["orientation"] == pytest.approx(expected["pose"]["orientation"], abs=1e-6)
        assert sample["moving"] is expected["moving"]


def test_runtime_unsupported_callback_aborts_recording_but_not_simulation(tmp_path: Path) -> None:
    from pybullet_fleet.examples.validation.pick_drop_arm_recording import build_example
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile

    sim = build_example("arm_robot", gui=False, recording=True)
    sim.params.target_rtf = 0
    recorder = sim.configure_state_recording(profile=KinematicProfile(), output=str(tmp_path / "incomplete"))
    sim.register_callback(lambda *_: None)
    sim.run_simulation(duration=0.3)
    assert sim.step_count == 3
    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["complete"] is False
    assert "Undeclared simulation callback" in manifest["recording_error"]


def test_missing_pick_link_is_rejected_before_callback_setup() -> None:
    from pybullet_fleet.examples.validation.pick_drop_arm_recording import restore_pick_drop_stage

    stage = {"model": "arm_robot", "step_state": 1, "targets_set": False, "picked": False, "pick_link_name": None}
    with pytest.raises(ValueError, match="Saved pick link name is missing"):
        restore_pick_drop_stage(None, stage)


def test_runtime_plugin_aborts_recording_but_not_simulation(tmp_path: Path) -> None:
    from pybullet_fleet.examples.validation.pick_drop_arm_recording import build_example
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile
    from pybullet_fleet.sim_plugin import SimPlugin

    sim = build_example("arm_robot", gui=False, recording=True)
    sim.params.target_rtf = 0
    recorder = sim.configure_state_recording(profile=KinematicProfile(), output=str(tmp_path / "plugin-incomplete"))
    sim.register_plugin(SimPlugin)
    sim.run_simulation(duration=0.3)
    assert sim.step_count == 3
    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["complete"] is False
    assert "Plugins, BTs or managers" in manifest["recording_error"]


def test_existing_plugin_rejects_recording_setup_without_changing_simulation(tmp_path: Path) -> None:
    from pybullet_fleet.examples.validation.pick_drop_arm_recording import build_example
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile
    from pybullet_fleet.sim_plugin import SimPlugin

    sim = build_example("arm_robot", gui=False, recording=True)
    sim.params.target_rtf = 0
    sim.register_plugin(SimPlugin)
    output = tmp_path / "no-artifact"
    with pytest.raises(ValueError, match="Plugins, BTs or managers"):
        sim.configure_state_recording(profile=KinematicProfile(), output=str(output))
    assert not output.exists()
    sim.run_simulation(duration=0.2)
    assert sim.step_count == 2


def test_failed_application_restore_disconnects_partial_world(
    fixed_arm_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pybullet as p

    from pybullet_fleet.replay import load_supported_simulation
    from pybullet_fleet.replay import state_recording

    seen: list[int] = []

    def fail_after_construction(sim, value):
        seen.append(sim.client)
        raise ValueError("Application restorer rejected its stage")

    monkeypatch.setitem(state_recording._DATA_RESTORERS, ("pbf.example.pick_drop_arm.stage", 1), fail_after_construction)
    with pytest.raises(ValueError, match="Application restorer rejected"):
        load_supported_simulation(fixed_arm_artifact, at_or_before=1.0)
    assert seen and not p.isConnected(seen[0])
