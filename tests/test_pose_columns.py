"""The simulation's pose columns: an object's pose cache is a view into its
row, ``set_poses`` writes arrays straight into the rows inside a step, and
the rows follow the objects through growth, removal and physics steps."""

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams
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


# --- base attachments carried as arrays -------------------------------------


def test_quaternion_products_compose_as_multiply_transforms():
    import random

    from pybullet_fleet.collision_store import quaternion_products

    rng = random.Random(2)
    for _ in range(20):
        a = np.array([rng.gauss(0, 1) for _ in range(4)])
        b = np.array([rng.gauss(0, 1) for _ in range(4)])
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        _, expected = p.multiplyTransforms([0, 0, 0], a.tolist(), [0, 0, 0], b.tolist())
        # PyBullet composes in single precision and reads the result back from a matrix.
        assert quaternion_products(a[None, :], b[None, :])[0] == pytest.approx(expected, abs=1e-6)


def _step_with_poses(sim, ids, positions, orientations):
    sim.register_callback(lambda _sim, _dt: sim.set_poses(ids, positions, orientations))
    sim.step_once()


def test_riders_follow_their_carrier_through_the_array_path(sim):
    carrier = box(sim, 0.0)
    load = box(sim, 0.1, y=0.05, z=0.3)
    on_top = box(sim, 0.1, y=0.05, z=0.5)
    carrier.attach_object(load, keep_world_pose=True)
    load.attach_object(on_top, keep_world_pose=True)
    positions = np.array([[3.0, -1.0, 0.1]])
    orientations = np.array([p.getQuaternionFromEuler([0.1, -0.2, 1.3])])
    seen = {}

    def cb(_sim, _dt):
        sim.set_poses([carrier.object_id], positions, orientations)
        seen["pending"] = set(sim._pending_pose_ids)

    sim.register_callback(cb)
    sim.step_once()

    assert seen["pending"] == {carrier.object_id, load.object_id, on_top.object_id}
    pos, orn = p.multiplyTransforms(
        positions[0], orientations[0], load._attach_offset.position, load._attach_offset.orientation
    )
    assert list(load.get_pose().position) == pytest.approx(pos, abs=1e-6)  # PyBullet composes in single precision
    assert list(load.get_pose().orientation) == pytest.approx(orn, abs=1e-6)
    pos2, orn2 = p.multiplyTransforms(pos, orn, on_top._attach_offset.position, on_top._attach_offset.orientation)
    assert list(on_top.get_pose().position) == pytest.approx(pos2, abs=1e-6)
    assert list(on_top.get_pose().orientation) == pytest.approx(orn2, abs=1e-6)
    # PyBullet and the collision boxes followed too.
    assert p.getBasePositionAndOrientation(on_top.body_id, physicsClientId=sim.client)[0] == pytest.approx(pos2, abs=1e-6)
    assert sim.cached_aabb(on_top.object_id)[0][0] == pytest.approx(pos2[0] - 0.1, abs=0.2)


def test_a_detached_rider_stays_behind(sim):
    carrier = box(sim, 0.0)
    load = box(sim, 0.0, z=0.3)
    carrier.attach_object(load, keep_world_pose=True)
    _step_with_poses(sim, [carrier.object_id], np.array([[1.0, 0.0, 0.1]]), np.array([[0.0, 0.0, 0.0, 1.0]]))
    assert load.get_pose().x == pytest.approx(1.0)
    carrier.detach_object(load)
    assert sim._base_children == {}
    _step_with_poses(sim, [carrier.object_id], np.array([[2.0, 0.0, 0.1]]), np.array([[0.0, 0.0, 0.0, 1.0]]))
    assert load.get_pose().x == pytest.approx(1.0)
    assert carrier.get_pose().x == pytest.approx(2.0)


def test_removing_a_carrier_or_rider_clears_the_attachment_bookkeeping(sim):
    carrier = box(sim, 0.0)
    load = box(sim, 0.0, z=0.3)
    other = box(sim, 5.0)
    rider = box(sim, 5.0, z=0.3)
    carrier.attach_object(load, keep_world_pose=True)
    other.attach_object(rider, keep_world_pose=True)
    sim.remove_object(load)
    assert carrier.object_id not in sim._base_children
    sim.remove_object(other)
    assert sim._base_children == {}


def test_a_carrier_with_a_physics_rider_takes_the_per_object_path(sim):
    carrier = box(sim, 0.0)
    heavy = box(sim, 0.0, z=0.3, mass=1.0)
    carrier.attach_object(heavy, keep_world_pose=True)
    assert not sim._carries_as_arrays(carrier)
    _step_with_poses(sim, [carrier.object_id], np.array([[2.0, 0.0, 0.1]]), np.array([[0.0, 0.0, 0.0, 1.0]]))
    assert heavy.get_pose().x == pytest.approx(2.0)


# --- one attachment tree is one body for the broad phase ---------------------


def test_a_carrier_and_its_riders_are_never_a_candidate_pair(sim):
    carrier = box(sim, 0.0)
    load = box(sim, 0.0, z=0.3)
    on_top = box(sim, 0.0, z=0.5)
    carrier.attach_object(load, keep_world_pose=True)
    load.attach_object(on_top, keep_world_pose=True)
    sim._moved_this_step.update({carrier.object_id, load.object_id, on_top.object_id})
    assert sim.filter_aabb_pairs()[0] == []
    store = sim._store
    assert {int(store.attach_root[store.row_of[o.object_id]]) for o in (carrier, load, on_top)} == {carrier.object_id}
    assert store.parent_row[store.row_of[load.object_id]] == store.row_of[carrier.object_id]
    assert store.parent_row[store.row_of[on_top.object_id]] == store.row_of[load.object_id]


def test_with_attached_collisions_counted_the_pair_is_offered():
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            ignore_attached_collision=False,
            log_level="warning",
        )
    )
    try:
        carrier = box(sim, 0.0)
        load = box(sim, 0.0, z=0.3)
        carrier.attach_object(load, keep_world_pose=True)
        sim._moved_this_step.update({carrier.object_id, load.object_id})
        assert sim.filter_aabb_pairs()[0] == [tuple(sorted((carrier.object_id, load.object_id)))]
    finally:
        p.disconnect(sim._client)


def test_a_detached_rider_in_contact_is_reported_without_moving(sim):
    carrier = box(sim, 0.0)
    load = box(sim, 0.0, z=0.3)
    carrier.attach_object(load, keep_world_pose=True)
    sim.step_once()
    assert sim.get_active_collision_pairs() == []
    carrier.detach_object(load)  # still touching, nothing moves
    sim.step_once()
    assert sim.get_active_collision_pairs() == [tuple(sorted((carrier.object_id, load.object_id)))]
    store = sim._store
    assert int(store.attach_root[store.row_of[load.object_id]]) == load.object_id
    assert store.parent_row[store.row_of[load.object_id]] == -1


def test_parent_rows_follow_a_removal(sim):
    first = box(sim, 5.0)
    carrier = box(sim, 0.0)
    load = box(sim, 0.0, z=0.3)
    carrier.attach_object(load, keep_world_pose=True)
    sim.remove_object(first)  # the last row (load's) moves into first's
    store = sim._store
    assert store.parent_row[store.row_of[load.object_id]] == store.row_of[carrier.object_id]
    sim.remove_object(carrier)
    assert store.parent_row[store.row_of[load.object_id]] == -1
    assert sim._base_children == {} and sim._base_parent_of == {}


def test_a_carrier_with_only_base_riders_takes_the_batch_fast_path(sim):
    from pybullet_fleet.agent_manager import AgentManager

    manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
    a = Agent.from_params(
        AgentSpawnParams(urdf_path="robots/simple_cube.urdf", initial_pose=Pose.from_xyz(0.0, 0.0, 0.0)), sim_core=sim
    )
    manager.add_object(a)
    load = box(sim, 0.0, z=0.4)
    a.attach_object(load, keep_world_pose=True)
    assert a._link_attached_count == 0
    assert a.update(0.1) is False  # nothing of its own to run
    a.set_goal_pose(Pose.from_xyz(3.0, 0.0, 0.0))
    for _ in range(10):
        sim.step_once()
    assert a.get_pose().x > 0.1
    assert load.get_pose().x == pytest.approx(a.get_pose().x, abs=1e-6)


def test_reset_gives_a_kept_object_its_own_pose_back(sim):
    kept = box(sim, 1.0)
    sim.reset()
    fresh = box(sim, 7.0)  # reuses the row the kept object had
    assert kept.get_pose().x == 1.0
    assert fresh.get_pose().x == 7.0


def test_a_physics_pose_is_a_snapshot_of_its_step():
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=True, monitor=False, timestep=1.0 / 60.0, log_level="warning")
    )
    try:
        falling = box(sim, 0.0, z=2.0, mass=1.0)
        start = falling.get_pose()
        for _ in range(30):
            sim.step_once()
        assert start.z == 2.0  # the pose kept from before the fall did not fall with the box
        assert falling.get_pose().z < 1.5
    finally:
        p.disconnect(sim.client)
