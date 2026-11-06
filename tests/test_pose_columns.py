"""The simulation's pose columns: an object's pose cache is a view into its
row, ``set_poses`` writes arrays straight into the rows inside a step, and
the rows follow the objects through growth, removal and physics steps."""

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode


@pytest.fixture
def sim():
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            collision_margin=0.02,
            log_level="warning",
        )
    )
    yield sim
    p.disconnect(sim.client)


def box(sim, x, y=0.0, z=0.1, mode=CollisionMode.NORMAL_3D, mass=0.0, name=None):
    shape = ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1])
    return SimObject.from_mesh(
        visual_shape=shape,
        collision_shape=shape,
        pose=Pose.from_xyz(x, y, z),
        mass=mass,
        sim_core=sim,
        collision_mode=mode,
        name=name,
    )


def row_pose(sim, obj):
    row = sim._store.row_of[obj.object_id]
    return sim._store.pos[row].tolist(), sim._store.orn[row].tolist()


def test_pose_equality_holds_across_lists_and_arrays():
    a = Pose(position=[1.0, 2.0, 3.0])
    b = Pose(position=np.array([1.0, 2.0, 3.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    assert a == b
    assert b == a
    assert a != Pose(position=[1.0, 2.0, 3.5])
    assert a != "pose"


def test_the_cache_is_the_row_and_set_pose_writes_it(sim):
    obj = box(sim, 1.0)
    assert row_pose(sim, obj)[0] == [1.0, 0.0, 0.1]
    obj.set_pose(Pose.from_xyz(2.0, 3.0, 0.1))
    assert row_pose(sim, obj)[0] == [2.0, 3.0, 0.1]
    assert obj.get_pose() == Pose.from_xyz(2.0, 3.0, 0.1)
    assert list(obj.get_pose().position) == [2.0, 3.0, 0.1]


def test_views_survive_a_reallocation_of_the_columns(sim):
    first = box(sim, 0.0)
    capacity = sim._store.capacity
    others = [box(sim, float(k + 1)) for k in range(capacity + 3)]
    assert sim._store.capacity > capacity
    first.set_pose(Pose.from_xyz(-5.0, 0.0, 0.1))
    assert row_pose(sim, first)[0] == [-5.0, 0.0, 0.1]
    for k, obj in enumerate(others):
        assert obj.get_pose().x == pytest.approx(k + 1)
        assert row_pose(sim, obj)[0][0] == pytest.approx(k + 1)


def test_views_survive_the_removal_of_another_object(sim):
    a, b, c = box(sim, 1.0), box(sim, 2.0), box(sim, 3.0)
    sim.remove_object(a)  # c's row moves into a's
    c.set_pose(Pose.from_xyz(30.0, 0.0, 0.1))
    assert row_pose(sim, c)[0] == [30.0, 0.0, 0.1]
    assert b.get_pose().x == 2.0
    assert sim.get_poses([b.object_id, c.object_id])[0][:, 0].tolist() == [2.0, 30.0]
    # The removed object keeps its last pose, in storage of its own.
    assert list(a.get_pose().position) == [1.0, 0.0, 0.1]
    a.get_pose().position[0] = 99.0
    assert row_pose(sim, c)[0][0] == 30.0


def test_set_poses_inside_a_step_writes_the_moved_rows_as_arrays(sim):
    objects = [box(sim, float(k)) for k in range(4)]
    ids = [o.object_id for o in objects]
    positions, orientations = sim.get_poses(ids)
    positions[1, 1] = 5.0  # moved
    orientations[2] = -orientations[2]  # the same rotation, other sign: not a move
    seen = {}

    def cb(_sim, _dt):
        sim.set_poses(ids, positions, orientations)
        seen["pending"] = set(sim._pending_pose_ids)
        seen["pose"] = objects[1].get_pose() == Pose(position=[1.0, 5.0, 0.1])

    sim.register_callback(cb)
    sim.step_once()
    assert seen["pending"] == {ids[1]}
    assert seen["pose"]
    assert objects[1].get_pose() == Pose(position=[1.0, 5.0, 0.1])
    assert p.getBasePositionAndOrientation(objects[1].body_id, physicsClientId=sim.client)[0] == pytest.approx((1.0, 5.0, 0.1))
    assert sim.cached_aabb(ids[1])[0][1] == pytest.approx(4.9)


def test_set_poses_inside_a_step_leaves_static_objects_and_propagates_attachments(sim):
    wall = box(sim, 0.0, mode=CollisionMode.STATIC)
    carrier = box(sim, 2.0)
    load = box(sim, 2.0, z=0.3)
    carrier.attach_object(load, keep_world_pose=True)
    ids = [wall.object_id, carrier.object_id]
    positions = np.array([[9.0, 0.0, 0.1], [4.0, 0.0, 0.1]])
    orientations = np.array([[0.0, 0.0, 0.0, 1.0]] * 2)

    sim.register_callback(lambda _sim, _dt: sim.set_poses(ids, positions, orientations))
    sim.step_once()

    assert wall.get_pose().x == 0.0
    assert carrier.get_pose().x == 4.0
    assert load.get_pose().x == pytest.approx(4.0)
    assert load.get_pose().z == pytest.approx(0.3)


def test_set_poses_keeps_or_drops_the_velocity_as_asked():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=True, monitor=False, log_level="warning"))
    try:
        ball = box(sim, 0.0, z=5.0, mass=1.0)
        p.resetBaseVelocity(ball.body_id, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=sim.client)
        ids = [ball.object_id]
        target = np.array([[1.0, 0.0, 5.0]])
        orn = np.array([[0.0, 0.0, 0.0, 1.0]])
        sim.register_callback(lambda _sim, _dt: sim.set_poses(ids, target, orn, preserve_velocity=False))
        sim.step_once()
        linear, _ = p.getBaseVelocity(ball.body_id, physicsClientId=sim.client)
        assert linear[0] == pytest.approx(0.0, abs=1e-9)
    finally:
        p.disconnect(sim.client)


def test_a_physics_body_pose_is_in_its_row_after_the_step():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=True, monitor=False, log_level="warning"))
    try:
        ball = box(sim, 0.0, z=5.0, mass=1.0)
        for _ in range(10):
            sim.step_once()
        expected = p.getBasePositionAndOrientation(ball.body_id, physicsClientId=sim.client)[0]
        assert row_pose(sim, ball)[0] == pytest.approx(expected)
        assert expected[2] < 5.0
        assert ball.get_pose().z == pytest.approx(expected[2])
    finally:
        p.disconnect(sim.client)
