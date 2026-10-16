"""Fresh-process coverage for built-in physics-off primitive objects."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pybullet as p
import pytest

from pybullet_fleet.replay.kinematic_profile import KinematicProfile
from pybullet_fleet.replay.state_recording import load_recording_manifest


def _run(script: str, *args: str) -> dict | None:
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    marked = [line.removeprefix("RESULT:") for line in result.stdout.splitlines() if line.startswith("RESULT:")]
    return json.loads(marked[-1]) if marked else None


@pytest.fixture
def primitive_artifact(tmp_path: Path) -> Path:
    directory = tmp_path / "primitives"
    _run(
        """import sys
from pathlib import Path
from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.replay.kinematic_profile import KinematicProfile
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams
s = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, timestep=0.1,
    collision_check_frequency=0, target_rtf=0))
agent = Agent.from_params(AgentSpawnParams(name='robot',
    urdf_path=str(Path('pybullet_fleet/robots/mobile_robot.urdf').resolve()),
    initial_pose=Pose.from_xyz(-2, 0, 0.05), mass=0), s)
objects = {
    'sphere': (ShapeParams(shape_type='sphere', radius=0.2, rgba_color=[1, 0, 0, 1],
                           frame_pose=Pose.from_xyz(0.01, 0.02, 0.03)),
               ShapeParams(shape_type='sphere', radius=0.18,
                           frame_pose=Pose.from_xyz(-0.01, 0, 0))),
    'cylinder': (ShapeParams(shape_type='cylinder', radius=0.15, height=0.4,
                             frame_pose=Pose.from_yaw(0, 0, 0.03, 0.4)),
                 ShapeParams(shape_type='cylinder', radius=0.13, height=0.35)),
    'mixed': (ShapeParams(shape_type='box', half_extents=[0.2, 0.1, 0.3]),
              ShapeParams(shape_type='sphere', radius=0.12)),
    'visual-only': (ShapeParams(shape_type='sphere', radius=0.1), None),
}
for index, (name, (visual, collision)) in enumerate(objects.items()):
    obj = SimObject.from_params(SimObjectSpawnParams(name=name, mass=0,
        initial_pose=Pose.from_xyz(index, 1, 0.5),
        visual_shape=visual, collision_shape=collision), s)
    if name == 'sphere':
        assert agent.attach_object(obj, parent_link_index='base_link',
            relative_pose=Pose.from_xyz(0.2, 0, 0.1))
s.configure_state_recording(profile=KinematicProfile(), output=sys.argv[1])
s.run_simulation(duration=0.2)
""",
        str(directory),
    )
    assert load_recording_manifest(directory)["complete"] is True
    return directory


def test_primitives_restore_in_fresh_process(primitive_artifact: Path) -> None:
    checkpoint = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    source_objects = checkpoint["objects"]
    assert {name: entry["type"] for name, entry in source_objects.items()} == {
        name: "kinematic_primitive" for name in ("sphere", "cylinder", "mixed", "visual-only")
    }
    assert source_objects["cylinder"]["state"]["construction"]["visual_shape"]["frame_pose"] is not None
    assert source_objects["sphere"]["state"]["attachment"]["parent_name"] == "robot"
    assert source_objects["mixed"]["state"]["construction"]["collision_shape"]["type"] == "sphere"
    assert source_objects["visual-only"]["state"]["construction"]["collision_shape"] is None
    restored = _run(
        """import json, sys, pybullet as p
from pybullet_fleet.replay import load_supported_simulation
from pybullet_fleet.replay.kinematic_profile import capture_primitive_state
s = load_supported_simulation(sys.argv[1], at_or_before=0.1)
objects = {obj.name: obj for obj in s.sim_objects if obj not in s.agents}
print('RESULT:' + json.dumps({
    'states': {name: capture_primitive_state(obj) for name, obj in objects.items()},
    'geometry': {name: {
        'visual': [shape[2] for shape in p.getVisualShapeData(obj.body_id, physicsClientId=s.client)],
        'collision': [shape[2] for shape in p.getCollisionShapeData(obj.body_id, -1, physicsClientId=s.client)],
        'visual_offsets': [list(shape[5]) for shape in p.getVisualShapeData(obj.body_id, physicsClientId=s.client)],
        'collision_offsets': [list(shape[5]) for shape in p.getCollisionShapeData(obj.body_id, -1, physicsClientId=s.client)]
    } for name, obj in objects.items()}
}))
""",
        str(primitive_artifact),
    )
    assert restored is not None
    assert restored["states"] == {name: entry["state"] for name, entry in source_objects.items()}
    geometry = restored["geometry"]
    assert geometry["sphere"]["visual"] == [p.GEOM_SPHERE]
    assert geometry["sphere"]["collision"] == [p.GEOM_SPHERE]
    assert geometry["cylinder"]["visual"] == [p.GEOM_CYLINDER]
    assert geometry["cylinder"]["collision"] == [p.GEOM_CYLINDER]
    assert geometry["mixed"]["visual"] == [p.GEOM_BOX]
    assert geometry["mixed"]["collision"] == [p.GEOM_SPHERE]
    assert geometry["sphere"]["visual_offsets"][0] == pytest.approx([0.01, 0.02, 0.03])
    assert geometry["sphere"]["collision_offsets"][0] == pytest.approx([-0.01, 0, 0])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state["objects"]["sphere"]["state"]["construction"]["visual_shape"].update(type="mesh"),
        lambda state: state["objects"]["cylinder"]["state"]["construction"]["collision_shape"].update(height=-1),
        lambda state: state["objects"]["mixed"]["state"]["construction"]["visual_shape"].update(radius=1),
    ],
)
def test_unsupported_or_invalid_shape_is_rejected_before_restore(primitive_artifact: Path, mutation) -> None:
    manifest = load_recording_manifest(primitive_artifact)
    state = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    state = copy.deepcopy(state)
    mutation(state)
    with pytest.raises(ValueError):
        KinematicProfile().validate_checkpoint(manifest, state)


def test_invalid_agent_angular_velocity_is_rejected(primitive_artifact: Path) -> None:
    manifest = load_recording_manifest(primitive_artifact)
    checkpoint = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    for invalid in (True, "0", None, float("nan"), float("inf")):
        state = copy.deepcopy(checkpoint)
        state["agents"]["robot"]["state"]["angular_velocity"] = invalid
        with pytest.raises(ValueError, match="Invalid Agent movement state"):
            KinematicProfile().validate_checkpoint(manifest, state)


@pytest.mark.parametrize("field,bad_value", [("type", "other"), ("version", 2)])
def test_unsupported_sim_state_is_rejected(primitive_artifact: Path, field: str, bad_value: object) -> None:
    manifest = load_recording_manifest(primitive_artifact)
    state = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    state["sim"][field] = bad_value
    with pytest.raises(ValueError, match="Unsupported kinematic checkpoint state"):
        KinematicProfile().validate_checkpoint(manifest, state)


@pytest.mark.parametrize("location", ["agent", "primitive", "frame", "attachment", "initial"])
def test_nonunit_quaternion_is_rejected(primitive_artifact: Path, location: str) -> None:
    manifest = copy.deepcopy(load_recording_manifest(primitive_artifact))
    state = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    if location == "agent":
        orientation = state["agents"]["robot"]["state"]["pose"]["orientation"]
    elif location == "primitive":
        orientation = state["objects"]["sphere"]["state"]["pose"]["orientation"]
    elif location == "frame":
        orientation = state["objects"]["sphere"]["state"]["construction"]["visual_shape"]["frame_pose"]["orientation"]
    elif location == "attachment":
        orientation = state["objects"]["sphere"]["state"]["attachment"]["relative_orientation"]
    else:
        orientation = manifest["construction"]["agents"]["robot"]["initial_pose"]["orientation"]
    orientation[:] = [0, 0, 0, 2]
    with pytest.raises(ValueError, match="quaternion"):
        KinematicProfile().validate_checkpoint(manifest, state)


def test_existing_agent_callbacks_reject_recording_setup(tmp_path: Path) -> None:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        agent = Agent.from_urdf("pybullet_fleet/robots/mobile_robot.urdf", name="robot", sim_core=sim)
        agent.callbacks.append({"func": lambda: None})
        with pytest.raises(ValueError, match="Unsupported Agent execution state"):
            sim.configure_state_recording(profile=KinematicProfile(), output=str(tmp_path / "recording"))
        assert not (tmp_path / "recording").exists()
    finally:
        p.disconnect(sim.client)


def test_unknown_agent_motion_mode_is_rejected(primitive_artifact: Path) -> None:
    manifest = copy.deepcopy(load_recording_manifest(primitive_artifact))
    state = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    manifest["construction"]["agents"]["robot"]["motion_mode"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported Agent motion mode"):
        KinematicProfile().validate_checkpoint(manifest, state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass", -1),
        ("fixed_base", "false"),
        ("collision_mode", "unknown"),
        ("asset_sha256", "bad"),
    ],
)
def test_invalid_agent_construction_is_rejected(primitive_artifact: Path, field: str, value: object) -> None:
    manifest = copy.deepcopy(load_recording_manifest(primitive_artifact))
    state = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    manifest["construction"]["agents"]["robot"][field] = value
    with pytest.raises(ValueError):
        KinematicProfile().validate_checkpoint(manifest, state)


def test_nonboolean_enable_floor_is_rejected(primitive_artifact: Path) -> None:
    manifest = copy.deepcopy(load_recording_manifest(primitive_artifact))
    state = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    manifest["construction"]["enable_floor"] = "false"
    with pytest.raises(ValueError, match="Unsupported kinematic construction"):
        KinematicProfile().validate_checkpoint(manifest, state)


def test_unsupported_urdf_object_retains_effective_spawn_descriptor() -> None:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.sim_object import SimObject, SimObjectSpawnParams
    from pybullet_fleet.types import CollisionMode

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        obj = SimObject.from_urdf(
            "pybullet_fleet/robots/simple_cube.urdf",
            name="fixture",
            pose=Pose.from_xyz(1, 2, 3),
            mass=0,
            pickable=True,
            collision_mode=CollisionMode.STATIC,
            use_fixed_base=True,
            global_scaling=0.5,
            sim_core=sim,
        )
        saved = obj._checkpoint_spawn_params
        assert saved.name == "fixture"
        assert list(saved.initial_pose.position) == [1, 2, 3]
        assert saved.mass == 0
        assert saved.pickable is True
        assert saved.collision_mode is CollisionMode.STATIC
        assert getattr(saved, "use_fixed_base") is True
        assert getattr(saved, "global_scaling") == 0.5
        assert Path(saved.urdf_path).is_file()
        via_params = SimObject.from_params(
            SimObjectSpawnParams(name="via-params", urdf_path="pybullet_fleet/robots/simple_cube.urdf", mass=0), sim
        )
        assert via_params._checkpoint_spawn_params.name == "via-params"
        assert getattr(via_params._checkpoint_spawn_params, "global_scaling") == 1.0
    finally:
        p.disconnect(sim.client)


def test_mesh_agent_retains_effective_spawn_descriptor() -> None:
    from pybullet_fleet.agent import Agent, AgentSpawnParams
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.sim_object import ShapeParams

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        agent = Agent.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
            pose=Pose.from_xyz(1, 2, 3),
            mass=0,
            name="mesh-agent",
            sim_core=sim,
        )
        saved = agent._checkpoint_spawn_params
        assert isinstance(saved, AgentSpawnParams)
        assert saved.visual_shape.shape_type == "box"
        assert saved.collision_shape.shape_type == "box"
        assert list(saved.initial_pose.position) == [1, 2, 3]
        assert isinstance(saved.controller, dict)
        assert saved.controller["type"] == agent.controller._registry_name
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize(
    "field,value",
    [("max_linear_vel", "fast"), ("max_linear_accel", [1, 2]), ("cmd_vel_timeout", -1), ("navigation_2d", "yes")],
)
def test_invalid_controller_construction_is_rejected(primitive_artifact: Path, field: str, value: object) -> None:
    manifest = copy.deepcopy(load_recording_manifest(primitive_artifact))
    state = json.loads((primitive_artifact / "checkpoints" / "000000001.json").read_text())
    manifest["construction"]["agents"]["robot"]["controller"]["parameters"][field] = value
    with pytest.raises(ValueError, match="controller"):
        KinematicProfile().validate_checkpoint(manifest, state)


def test_live_pickability_change_is_rejected(primitive_artifact: Path) -> None:
    from pybullet_fleet.replay import load_supported_simulation

    sim = load_supported_simulation(primitive_artifact, at_or_before=0.1)
    try:
        obj = sim.get_unique_object_by_name("sphere")
        obj.pickable = not obj.pickable
        with pytest.raises(ValueError, match="supported primitive construction"):
            KinematicProfile().capture(sim)
    finally:
        p.disconnect(sim.client)


def test_frequency_callback_is_rejected_even_if_allowlisted(tmp_path: Path) -> None:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:

        def callback(_sim, _dt):
            pass

        sim.register_callback(callback, frequency=10)
        with pytest.raises(ValueError, match="Undeclared simulation callback"):
            sim.configure_state_recording(
                profile=KinematicProfile(allowed_callbacks=(callback,)), output=str(tmp_path / "recording")
            )
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("change", ["metadata", "controller", "pickable", "batch"])
def test_live_agent_construction_change_is_rejected(change: str) -> None:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        agent = Agent.from_urdf("pybullet_fleet/robots/mobile_robot.urdf", name="robot", sim_core=sim)
        profile = KinematicProfile()
        profile.construction(sim)
        if change == "metadata":
            agent.user_data["extra"] = 1
        elif change == "controller":
            agent.controller.params.max_linear_vel = 0.5
        elif change == "pickable":
            agent.pickable = not agent.pickable
        else:
            agent._batch_controller = object()
        with pytest.raises(ValueError):
            profile.capture(sim)
    finally:
        p.disconnect(sim.client)


def test_motion_mode_changed_after_spawn_is_not_silently_captured() -> None:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        agent = Agent.from_urdf("pybullet_fleet/robots/mobile_robot.urdf", name="robot", sim_core=sim)
        assert agent.set_motion_mode("omnidirectional")
        with pytest.raises(ValueError, match="motion mode changed"):
            KinematicProfile().capture(sim)
    finally:
        p.disconnect(sim.client)


def test_agent_collision_mode_change_is_not_silently_captured() -> None:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.types import CollisionMode

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        agent = Agent.from_urdf("pybullet_fleet/robots/mobile_robot.urdf", name="robot", sim_core=sim)
        profile = KinematicProfile()
        profile.construction(sim)
        agent.set_collision_mode(CollisionMode.DISABLED)
        with pytest.raises(ValueError, match="Agent mode changed"):
            profile.capture(sim)
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("factory", ["direct", "params"])
def test_short_lived_runtime_agent_spawn_invalidates_recording(tmp_path: Path, factory: str) -> None:
    from pybullet_fleet.agent import Agent, AgentSpawnParams
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    path = tmp_path / factory
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, collision_check_frequency=0, target_rtf=0, timestep=0.1)
    )
    try:
        sim.configure_state_recording(profile=KinematicProfile(), output=str(path))
        if factory == "direct":
            agent = Agent.from_urdf("pybullet_fleet/robots/mobile_robot.urdf", name="robot", sim_core=sim)
        else:
            agent = Agent.from_params(AgentSpawnParams(name="robot", urdf_path="pybullet_fleet/robots/mobile_robot.urdf"), sim)
        sim.remove_object(agent)
        sim.run_simulation(duration=0.1)
        assert json.loads((path / "manifest.json").read_text())["complete"] is False
        assert sim.get_completed_step_boundary()[0] == 1
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_asset_hash_is_cached_during_recording_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.replay import kinematic_profile

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        Agent.from_urdf("pybullet_fleet/robots/mobile_robot.urdf", name="robot", sim_core=sim)
        original = kinematic_profile._asset_fingerprint
        calls = []

        def counted(path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(kinematic_profile, "_asset_fingerprint", counted)
        profile = KinematicProfile()
        construction = profile.construction(sim)
        state = profile.capture(sim)
        for _ in range(3):
            profile.validate_checkpoint({"construction": construction}, state)
        assert len(calls) == 1
        KinematicProfile().validate_checkpoint({"construction": construction}, state)
        assert len(calls) == 2
    finally:
        p.disconnect(sim.client)


def test_direct_primitive_spawn_is_recorded_once(tmp_path: Path) -> None:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams

    path = tmp_path / "direct-spawn"
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, collision_check_frequency=0, target_rtf=0, timestep=0.1)
    )
    try:
        sim.configure_state_recording(profile=KinematicProfile(), output=str(path))
        shape = ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1])
        SimObject.from_mesh(visual_shape=shape, pose=Pose.from_xyz(0, 0, 0.5), name="direct", sim_core=sim)
        SimObject.from_params(SimObjectSpawnParams(name="params", visual_shape=shape), sim)
        sim.run_simulation(duration=0.1)
        inputs = [json.loads(line) for line in (path / "inputs.jsonl").read_text().splitlines()]
        assert [item["details"]["key"] for item in inputs] == ["direct", "params"]
        assert load_recording_manifest(path)["complete"] is True
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_factory_spawn_params_are_independent_of_mutable_inputs() -> None:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.sim_object import ShapeParams, SimObject

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        shape = ShapeParams(shape_type="box", half_extents=[0.1, 0.2, 0.3])
        pose = Pose.from_xyz(1, 2, 3)
        obj = SimObject.from_mesh(visual_shape=shape, pose=pose, name="box", sim_core=sim)
        shape.half_extents[0] = 9
        pose.position[0] = 9
        state = KinematicProfile().capture(sim)["objects"]["box"]["state"]["construction"]
        assert state["visual_shape"]["half_extents"] == [0.1, 0.2, 0.3]
        assert list(obj._checkpoint_spawn_params.initial_pose.position) == [1, 2, 3]
        assert obj.get_pose().position[0] == pytest.approx(1)
    finally:
        p.disconnect(sim.client)


def test_spawn_hook_is_optional_for_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.sim_object import ShapeParams, SimObject

    monkeypatch.delattr(MultiRobotSimulationCore, "_record_state_spawn")
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    try:
        SimObject.from_mesh(visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]), sim_core=sim)
        SimObject.from_urdf("pybullet_fleet/robots/simple_cube.urdf", sim_core=sim)
        assert len(sim.sim_objects) == 2
    finally:
        p.disconnect(sim.client)


def test_noncopyable_metadata_does_not_prevent_normal_spawns() -> None:
    import threading

    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.sim_object import ShapeParams, SimObject

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, collision_check_frequency=0))
    lock = threading.Lock()
    try:
        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            name="box",
            user_data={"lock": lock},
            sim_core=sim,
        )
        agent = Agent.from_urdf(
            "pybullet_fleet/robots/mobile_robot.urdf", name="robot", user_data={"lock": lock}, sim_core=sim
        )
        assert obj._checkpoint_spawn_params.user_data["lock"] is lock
        assert agent._checkpoint_spawn_params.user_data["lock"] is lock
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("change", ["collision", "physics"])
def test_runtime_profile_change_makes_recording_incomplete(tmp_path: Path, change: str) -> None:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.sim_object import ShapeParams, SimObject

    path = tmp_path / change
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, collision_check_frequency=0, target_rtf=0, timestep=0.1)
    )
    try:
        SimObject.from_mesh(visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]), name="box", sim_core=sim)
        sim.configure_state_recording(profile=KinematicProfile(), output=str(path))
        if change == "collision":
            sim.set_collision_check_frequency(10)
        else:
            sim.params.physics = True
        sim.run_simulation(duration=0.1)
        assert json.loads((path / "manifest.json").read_text())["complete"] is False
        assert sim.get_completed_step_boundary()[0] == 1
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def test_playback_recreates_removed_name(primitive_artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pybullet_fleet.replay import ResultPlayback
    from pybullet_fleet.replay import kinematic_profile

    original_restore = kinematic_profile.restore_supported_simulation
    simulations = []

    def headless_restore(manifest, state, **kwargs):
        sim = original_restore(manifest, state, **{**kwargs, "gui": False})
        simulations.append(sim)
        return sim

    monkeypatch.setattr(kinematic_profile, "restore_supported_simulation", headless_restore)
    monkeypatch.setattr(kinematic_profile.time, "sleep", lambda _seconds: None)
    playback = ResultPlayback(primitive_artifact, profile=KinematicProfile())
    first = copy.deepcopy(playback.frames[0])
    first["objects"]["mixed"]["state"]["construction"]["visual_shape"]["half_extents"][0] = 0.4
    without = copy.deepcopy(first)
    del without["objects"]["mixed"]
    del without["objects"]["sphere"]
    respawned = copy.deepcopy(first)
    respawned["objects"]["mixed"]["state"]["construction"]["visual_shape"]["half_extents"][0] = 0.6
    playback.frames = [first, without, respawned]
    playback.play_gui(rtf=100)
    assert len(simulations) == 1
    objects = simulations[0].find_objects_by_name("mixed")
    assert len(objects) == 1
    assert objects[0]._checkpoint_spawn_params.visual_shape.half_extents[0] == 0.6
    attached = simulations[0].get_unique_agent_by_name("robot").get_attached_objects()
    assert [obj.name for obj in attached] == ["sphere"]


def test_playback_skips_frames_before_first_checkpoint(primitive_artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pybullet_fleet.replay import ResultPlayback
    from pybullet_fleet.replay import kinematic_profile

    restore = kinematic_profile.restore_supported_simulation
    monkeypatch.setattr(
        kinematic_profile,
        "restore_supported_simulation",
        lambda manifest, state, **kwargs: restore(manifest, state, **{**kwargs, "gui": False}),
    )
    monkeypatch.setattr(kinematic_profile.time, "sleep", lambda _seconds: None)
    playback = ResultPlayback(primitive_artifact, profile=KinematicProfile())
    playback.manifest["checkpoint_sha256"].pop("1")
    playback.frames[0]["agents"]["not-in-checkpoint"] = {}
    playback.play_gui(rtf=100)


def test_primitive_only_world_records_and_restores_without_an_agent(tmp_path: Path) -> None:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.geometry import Pose
    from pybullet_fleet.replay import load_supported_simulation
    from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams

    path = tmp_path / "primitive-only"
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, collision_check_frequency=0, target_rtf=0, timestep=0.1)
    )
    try:
        SimObject.from_params(
            SimObjectSpawnParams(
                name="box",
                mass=0,
                initial_pose=Pose.from_xyz(1, 2, 0.5),
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            ),
            sim,
        )
        sim.configure_state_recording(profile=KinematicProfile(), output=str(path))
        sim.run_simulation(duration=0.1)
        assert load_recording_manifest(path)["construction"]["agents"] == {}
        restored = load_supported_simulation(path, at_or_before=0.1)
        try:
            assert restored.agents == []
            assert restored.get_unique_object_by_name("box").get_pose().position == pytest.approx([1, 2, 0.5])
            assert restored.get_completed_step_boundary() == (1, pytest.approx(0.1))
        finally:
            p.disconnect(restored.client)
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)
