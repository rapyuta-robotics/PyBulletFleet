"""The lazy PyBullet mirror: a kinematic body's buffered pose reaches PyBullet
only when something reads it there, and everything of the simulation's own
that reads PyBullet syncs first."""

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.sim_plugin import SimPlugin
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode


def make_sim(**overrides):
    params = dict(
        gui=False,
        physics=False,
        monitor=False,
        collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
        collision_margin=0.02,
        log_level="warning",
    )
    params.update(overrides)
    return MultiRobotSimulationCore(SimulationParams(**params))


@pytest.fixture
def sim():
    sim = make_sim()
    yield sim
    p.disconnect(sim._client)


def box(sim, x, z=0.1, mode=CollisionMode.NORMAL_3D, mass=0.0):
    shape = ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1])
    return SimObject.from_mesh(
        visual_shape=shape, collision_shape=shape, pose=Pose.from_xyz(x, 0.0, z), mass=mass, sim_core=sim, collision_mode=mode
    )


def raw_x(sim, body_id):
    return p.getBasePositionAndOrientation(body_id, physicsClientId=sim._client)[0][0]


def move_in_step(sim, obj, x):
    sim.register_callback(lambda _sim, _dt: obj.set_pose(Pose.from_xyz(x, 0.0, 0.1)))
    sim.step_once()


def test_a_buffered_pose_stays_out_of_pybullet_until_read(sim):
    obj = box(sim, 0.0)
    body = obj._body_id
    move_in_step(sim, obj, 5.0)
    assert obj.get_pose().x == 5.0
    assert raw_x(sim, body) == 0.0  # a kept id sees the old pose
    assert obj.body_id == body  # the accessor syncs
    assert raw_x(sim, body) == 5.0


@pytest.mark.parametrize("how", ["sync_mirror", "sync_mirror_ids", "client"])
def test_the_sync_entry_points_write_every_pending_pose(sim, how):
    objects = [box(sim, float(k)) for k in range(3)]
    bodies = [o._body_id for o in objects]
    sim.register_callback(lambda _sim, _dt: [o.set_pose(Pose.from_xyz(10.0 + k, 0.0, 0.1)) for k, o in enumerate(objects)])
    sim.step_once()
    assert [raw_x(sim, b) for b in bodies] == [0.0, 1.0, 2.0]
    if how == "sync_mirror":
        sim.sync_mirror()
    elif how == "sync_mirror_ids":
        sim.sync_mirror([o.object_id for o in objects])
    else:
        assert sim.client == sim._client
    assert [raw_x(sim, b) for b in bodies] == [10.0, 11.0, 12.0]
    assert not sim._store.mirror_dirty[: len(sim._store)].any()


def test_mirror_always_writes_every_step():
    sim = make_sim(mirror="always")
    try:
        obj = box(sim, 0.0)
        body = obj._body_id
        move_in_step(sim, obj, 5.0)
        assert raw_x(sim, body) == 5.0
    finally:
        p.disconnect(sim._client)


def test_a_bad_mirror_value_is_rejected():
    with pytest.raises(ValueError, match="mirror"):
        SimulationParams(gui=False, monitor=False, mirror="sometimes")


def test_an_immediate_write_is_current_and_clears_a_pending_one(sim):
    obj = box(sim, 0.0)
    body = obj._body_id
    move_in_step(sim, obj, 5.0)
    obj.set_pose(Pose.from_xyz(7.0, 0.0, 0.1))  # outside a step: immediate
    assert raw_x(sim, body) == 7.0
    assert not sim._store.mirror_dirty[sim._store.row_of[obj.object_id]]


def test_the_narrow_phase_sees_the_poses_it_judges(sim):
    a = box(sim, 0.0)
    b = box(sim, 5.0)
    sim.step_once()
    assert sim.get_active_collision_pairs() == []
    move_in_step(sim, b, 0.15)  # into contact, written only to the columns
    assert tuple(sorted((a.object_id, b.object_id))) in sim.get_active_collision_pairs()
    move_in_step(sim, b, 5.0)
    assert sim.get_active_collision_pairs() == []


def test_a_query_sees_a_disabled_body_where_it_is(sim):
    obj = box(sim, 0.0, mode=CollisionMode.DISABLED)
    move_in_step(sim, obj, 5.0)
    assert obj in sim.query_aabb((4.5, -1.0, -1.0), (5.5, 1.0, 1.0))
    assert obj not in sim.query_aabb((-0.5, -1.0, -1.0), (0.5, 1.0, 1.0))


def test_physics_and_articulated_bodies_are_always_mirrored():
    sim = make_sim(physics=True)
    try:
        ball = box(sim, 0.0, z=2.0, mass=1.0)
        arm = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/arm_robot.urdf", initial_pose=Pose.from_xyz(3.0, 0.0, 0.0), use_fixed_base=True
            ),
            sim_core=sim,
        )
        ball_body, arm_body = ball._body_id, arm._body_id
        rows = sim._store
        assert rows.needs_mirror[rows.row_of[ball.object_id]]
        assert rows.needs_mirror[rows.row_of[arm.object_id]]
        sim.register_callback(
            lambda _sim, _dt: (ball.set_pose(Pose.from_xyz(1.0, 0.0, 2.0)), arm.set_pose(Pose.from_xyz(4.0, 0.0, 0.0)))
        )
        sim.step_once()
        assert raw_x(sim, ball_body) == pytest.approx(1.0, abs=0.05)
        assert raw_x(sim, arm_body) == 4.0
    finally:
        p.disconnect(sim._client)


def test_a_plugin_asking_for_the_mirror_gets_every_pose_written(sim):
    class RawReader(SimPlugin):
        needs_mirror = True

        def on_step(self, dt):
            pass

    obj = box(sim, 0.0)
    body = obj._body_id
    sim.register_plugin(RawReader, config={})
    move_in_step(sim, obj, 5.0)
    assert raw_x(sim, body) == 5.0


def test_a_callback_asking_for_the_mirror_gets_every_pose_written(sim):
    obj = box(sim, 0.0)
    body = obj._body_id

    def reader(_sim, _dt):
        pass

    reader.needs_mirror = True
    sim.register_callback(reader)
    move_in_step(sim, obj, 5.0)
    assert raw_x(sim, body) == 5.0


def test_set_poses_rows_are_lazy_too(sim):
    objects = [box(sim, float(k)) for k in range(3)]
    ids = [o.object_id for o in objects]
    positions = np.array([[20.0 + k, 0.0, 0.1] for k in range(3)])
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    sim.register_callback(lambda _sim, _dt: sim.set_poses(ids, positions, orientations))
    sim.step_once()
    assert raw_x(sim, objects[1]._body_id) == 1.0
    assert sim.cached_aabb(ids[1])[0][0] == pytest.approx(20.9)
    assert raw_x(sim, objects[1].body_id) == 21.0
