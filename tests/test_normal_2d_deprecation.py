"""CollisionMode.NORMAL_2D is deprecated (#104).

Every public way to give an object NORMAL_2D warns once, attributed to the
caller's line, and the mode keeps working. NORMAL_3D and the other modes do
not warn.
"""

import warnings

import pybullet as p
import pytest

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.devices.door import DoorParams
from pybullet_fleet.devices.elevator import ElevatorParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams
from pybullet_fleet.types import CollisionMode

BOX = ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1])


def _normal_2d_warnings(record):
    return [w for w in record if issubclass(w.category, FutureWarning) and "NORMAL_2D" in str(w.message)]


def _assert_one_warning_from_this_file(record):
    found = _normal_2d_warnings(record)
    assert len(found) == 1, [str(w.message) for w in found]
    assert found[0].filename == __file__
    return found[0]


def test_message_says_what_it_does_and_what_to_use():
    with pytest.warns(FutureWarning) as record:
        SimObjectSpawnParams(collision_mode=CollisionMode.NORMAL_2D)
    message = str(_assert_one_warning_from_this_file(record).message)
    assert "z" in message and "clearance" in message
    assert "NORMAL_3D" in message
    assert "collision_margin=(m, m, 0)" in message


def test_spawn_params_constructor_warns():
    with pytest.warns(FutureWarning) as record:
        params = SimObjectSpawnParams(name="a", collision_mode=CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)
    assert params.collision_mode is CollisionMode.NORMAL_2D


def test_agent_spawn_params_constructor_warns():
    with pytest.warns(FutureWarning) as record:
        AgentSpawnParams(name="a", visual_shape=BOX, collision_mode=CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)


@pytest.mark.parametrize("cls", [SimObjectSpawnParams, AgentSpawnParams, DoorParams, ElevatorParams])
def test_parsing_the_string_from_a_config_warns(cls):
    config = {"name": "a", "urdf_path": "x.urdf", "open_positions": {"j": 1.0}, "floors": {"1F": 0.0}}
    with pytest.warns(FutureWarning) as record:
        params = cls.from_dict({**config, "collision_mode": "normal_2d"})
    _assert_one_warning_from_this_file(record)
    assert params.collision_mode is CollisionMode.NORMAL_2D


def test_object_constructors_warn_once(pybullet_env):
    with pytest.warns(FutureWarning) as record:
        obj = SimObject.from_mesh(collision_shape=BOX, pose=Pose.from_xyz(0, 0, 1), collision_mode=CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)
    assert obj.collision_mode is CollisionMode.NORMAL_2D


def test_agent_constructor_warns_once(pybullet_env):
    with pytest.warns(FutureWarning) as record:
        agent = Agent.from_mesh(visual_shape=BOX, collision_shape=BOX, collision_mode=CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)
    assert agent.collision_mode is CollisionMode.NORMAL_2D


@pytest.mark.parametrize("spawn", [SimObject.from_params, Agent.from_params])
def test_spawning_from_params_does_not_warn_a_second_time(pybullet_env, spawn):
    with pytest.warns(FutureWarning):
        params = AgentSpawnParams(name="a", visual_shape=BOX, collision_shape=BOX, collision_mode=CollisionMode.NORMAL_2D)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        obj = spawn(params)
    assert _normal_2d_warnings(record) == []
    assert obj.collision_mode is CollisionMode.NORMAL_2D


@pytest.mark.parametrize("spawn", [SimObject.from_params, Agent.from_params])
def test_spawning_from_params_changed_to_normal_2d_after_construction_warns(pybullet_env, spawn):
    params = AgentSpawnParams(name="a", visual_shape=BOX, collision_shape=BOX)
    params.collision_mode = CollisionMode.NORMAL_2D
    with pytest.warns(FutureWarning) as record:
        obj = spawn(params)
    _assert_one_warning_from_this_file(record)
    assert obj.collision_mode is CollisionMode.NORMAL_2D


def _spawn_two_on_a_grid(manager_cls, params, how):
    from pybullet_fleet.agent_manager import GridSpawnParams

    grid = GridSpawnParams(x_min=0, x_max=1, y_min=0, y_max=0, spacing=[1.0, 1.0, 0.0], offset=[0.0, 0.0, 0.5])
    manager = manager_cls()
    if how == "counts":
        return manager.spawn_objects_grid(2, grid, params)
    return manager.spawn_grid_mixed(2, grid, [(params, 1.0)])


@pytest.mark.parametrize("how", ["counts", "mixed"])
@pytest.mark.parametrize("manager", ["SimObjectManager", "AgentManager"])
def test_grid_spawning_from_params_changed_to_normal_2d_after_construction_warns_once(pybullet_env, manager, how):
    import pybullet_fleet.agent_manager as agent_manager

    params = AgentSpawnParams(name="a", visual_shape=BOX, collision_shape=BOX)
    params.collision_mode = CollisionMode.NORMAL_2D
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        objs = _spawn_two_on_a_grid(getattr(agent_manager, manager), params, how)
    assert len(objs) == 2
    assert all(obj.collision_mode is CollisionMode.NORMAL_2D for obj in objs)
    _assert_one_warning_from_this_file(record)


@pytest.mark.parametrize("how", ["counts", "mixed"])
def test_grid_spawning_from_warned_params_does_not_warn_again(pybullet_env, how):
    from pybullet_fleet.agent_manager import AgentManager

    with pytest.warns(FutureWarning):
        params = AgentSpawnParams(name="a", visual_shape=BOX, collision_shape=BOX, collision_mode=CollisionMode.NORMAL_2D)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _spawn_two_on_a_grid(AgentManager, params, how)
    assert _normal_2d_warnings(record) == []


class _SpawnsAHelperFirst(SimObject):
    @classmethod
    def from_mesh(cls, collision_shape=None, pose=None, collision_mode=CollisionMode.NORMAL_3D, sim_core=None):
        SimObject.from_mesh(collision_shape=BOX, pose=Pose.from_xyz(5, 0, 1), collision_mode=CollisionMode.NORMAL_2D)
        return super().from_mesh(collision_shape=collision_shape, pose=pose, collision_mode=collision_mode, sim_core=sim_core)


class _SpawnsAHelperInInit(SimObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SimObject.from_mesh(collision_shape=BOX, pose=Pose.from_xyz(5, 0, 1), collision_mode=CollisionMode.NORMAL_2D)


@pytest.mark.parametrize("cls", [_SpawnsAHelperFirst, _SpawnsAHelperInInit])
def test_spawning_from_params_still_warns_about_nested_spawns(pybullet_env, cls):
    with pytest.warns(FutureWarning):
        params = SimObjectSpawnParams(collision_shape=BOX, collision_mode=CollisionMode.NORMAL_2D)
    with pytest.warns(FutureWarning) as record:
        obj = cls.from_params(params)
    _assert_one_warning_from_this_file(record)
    assert obj.collision_mode is CollisionMode.NORMAL_2D


def test_set_collision_mode_warns(pybullet_env):
    obj = SimObject.from_mesh(collision_shape=BOX, pose=Pose.from_xyz(0, 0, 1))
    with pytest.warns(FutureWarning) as record:
        obj.set_collision_mode(CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)
    assert obj.collision_mode is CollisionMode.NORMAL_2D


@pytest.mark.parametrize("mode", [CollisionMode.NORMAL_3D, CollisionMode.STATIC, CollisionMode.DISABLED])
def test_other_modes_do_not_warn(pybullet_env, mode):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        SimObjectSpawnParams.from_dict({"name": "a", "collision_mode": mode.value})
        AgentSpawnParams(name="a", visual_shape=BOX, collision_mode=mode)
        obj = SimObject.from_mesh(collision_shape=BOX, collision_mode=mode)
        Agent.from_mesh(visual_shape=BOX, collision_shape=BOX, collision_mode=mode)
        obj.set_collision_mode(CollisionMode.NORMAL_3D)


def test_a_spawn_loop_warns_once_under_the_default_filter(pybullet_env):
    """Python's default filter shows a warning once per call site, so a loop
    spawning many objects from one line does not flood the output."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for _ in range(20):
            SimObject.from_mesh(collision_shape=BOX, collision_mode=CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)


def test_a_config_warns_once_per_spawn_call():
    """Config-driven spawning attributes the warning to the spawn call, and the
    default filter shows it once however many entities the config lists."""
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams

    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, monitor=False, log_level="warning"))
    entities = [
        {
            # sim_object: Agent.from_mesh builds its body on the default physics client.
            "type": "sim_object",
            "name": f"r{i}",
            "collision_shape": {"shape_type": "box", "half_extents": [0.1, 0.1, 0.1]},
            "collision_mode": "normal_2d",
            "pose": [float(i), 0.0, 0.1],
        }
        for i in range(10)
    ]
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("default")
            spawned = sim.spawn_robots_from_config(entities)
    finally:
        p.disconnect(sim.client)
    assert len(spawned) == 10
    assert all(obj.collision_mode is CollisionMode.NORMAL_2D for obj in spawned)
    _assert_one_warning_from_this_file(record)


def test_dataclasses_replace_in_user_code_is_attributed_to_the_caller():
    from dataclasses import replace

    with pytest.warns(FutureWarning) as record:
        replace(SimObjectSpawnParams(name="a"), collision_mode=CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)


def test_python_shows_it_by_default_for_a_call_in_an_imported_module(tmp_path):
    """A DeprecationWarning would be hidden here: by default Python shows one only
    for code in __main__. The spawn call sits in a module the script imports."""
    import os
    import subprocess
    import sys

    (tmp_path / "spawner.py").write_text(
        "from pybullet_fleet.sim_object import SimObjectSpawnParams\n"
        "from pybullet_fleet.types import CollisionMode\n"
        "def spawn():\n"
        "    SimObjectSpawnParams(name='a', collision_mode=CollisionMode.NORMAL_2D)\n"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), repo_root]))
    env.pop("PYTHONWARNINGS", None)
    result = subprocess.run(
        [sys.executable, "-c", "import spawner; spawner.spawn()"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "spawner.py:4: FutureWarning: CollisionMode.NORMAL_2D is deprecated" in result.stderr


def test_code_run_from_a_string_is_attributed_to_that_code():
    """python -c and exec() run user code from "<string>": that is the caller, not
    the package file that emits the warning."""
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=repo_root)
    env.pop("PYTHONWARNINGS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pybullet_fleet.sim_object import SimObjectSpawnParams\n"
            "from pybullet_fleet.types import CollisionMode\n"
            "SimObjectSpawnParams(name='a', collision_mode=CollisionMode.NORMAL_2D)\n",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "<string>:3: FutureWarning: CollisionMode.NORMAL_2D is deprecated" in result.stderr


_TWO_BOX_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
{models}
</sdf>
"""
_BOX_MODEL = """  <model name="box{i}">
    <pose>{i} 0 0.5 0 0 0</pose>
    <link name="link">
      <collision name="c"><geometry><box><size>0.2 0.2 0.2</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>0.2 0.2 0.2</size></box></geometry></visual>
    </link>
  </model>"""


@pytest.mark.parametrize("cls", [SimObject, Agent])
def test_from_urdf_warns_once(pybullet_env, cls):
    with pytest.warns(FutureWarning) as record:
        obj = cls.from_urdf("cube_small.urdf", collision_mode=CollisionMode.NORMAL_2D)
    _assert_one_warning_from_this_file(record)
    assert obj.collision_mode is CollisionMode.NORMAL_2D


def test_from_sdf_warns_once_per_body(pybullet_env, tmp_path):
    """Under pytest.warns every warning is recorded: one per body the file holds."""
    sdf = tmp_path / "two_boxes.sdf"
    sdf.write_text(_TWO_BOX_SDF.format(models="\n".join(_BOX_MODEL.format(i=i) for i in range(2))))
    with pytest.warns(FutureWarning) as record:
        objects = SimObject.from_sdf(str(sdf), collision_mode=CollisionMode.NORMAL_2D)
    assert len(objects) == 2
    found = _normal_2d_warnings(record)
    assert len(found) == 2 and all(w.filename == __file__ for w in found)
    assert all(obj.collision_mode is CollisionMode.NORMAL_2D for obj in objects)


@pytest.mark.parametrize("bodies", [1, 5])
def test_from_sdf_warns_once_under_the_default_filter(pybullet_env, tmp_path, bodies):
    sdf = tmp_path / "boxes.sdf"
    sdf.write_text(_TWO_BOX_SDF.format(models="\n".join(_BOX_MODEL.format(i=i) for i in range(bodies))))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        objects = SimObject.from_sdf(str(sdf), collision_mode=CollisionMode.NORMAL_2D)
    assert len(objects) == bodies
    _assert_one_warning_from_this_file(record)


def test_restoring_a_recorded_normal_2d_world_does_not_warn(tmp_path):
    """The recording keeps the mode the user already chose (and was warned about)."""
    from pathlib import Path

    from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
    from pybullet_fleet.replay import load_supported_simulation
    from pybullet_fleet.replay.kinematic_profile import KinematicProfile

    path = tmp_path / "normal-2d"
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, collision_check_frequency=0, target_rtf=0, timestep=0.1)
    )
    try:
        with pytest.warns(FutureWarning):
            Agent.from_params(
                AgentSpawnParams(
                    name="robot",
                    urdf_path=str(Path("pybullet_fleet/robots/mobile_robot.urdf").resolve()),
                    initial_pose=Pose.from_xyz(-2, 0, 0.05),
                    mass=0,
                    collision_mode=CollisionMode.NORMAL_2D,
                ),
                sim,
            )
            SimObject.from_params(
                SimObjectSpawnParams(
                    name="box",
                    mass=0,
                    initial_pose=Pose.from_xyz(1, 2, 0.5),
                    collision_shape=BOX,
                    collision_mode=CollisionMode.NORMAL_2D,
                ),
                sim,
            )
        sim.configure_state_recording(profile=KinematicProfile(), output=str(path))
        sim.run_simulation(duration=0.1)
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        restored = load_supported_simulation(path, at_or_before=0.1)
    try:
        assert _normal_2d_warnings(record) == []
        assert restored.get_unique_agent_by_name("robot").collision_mode is CollisionMode.NORMAL_2D
        assert restored.get_unique_object_by_name("box").collision_mode is CollisionMode.NORMAL_2D
    finally:
        p.disconnect(restored.client)
