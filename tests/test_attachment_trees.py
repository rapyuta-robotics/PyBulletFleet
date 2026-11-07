"""Attachment trees in the broad phase: what counts as a pair after a detach, a
removed carrier, or ``ignore_attached_collision`` changing; the local box read
puts a kinematic body back exactly; a DISABLED object holds no box."""

import pybullet as p
import pytest

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode

HALF = 0.1
SHAPE = ShapeParams(shape_type="box", half_extents=[HALF, HALF, HALF])


def make_sim(**overrides):
    params = dict(
        gui=False,
        physics=False,
        monitor=False,
        collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
        collision_margin=0.0,
        log_level="warning",
    )
    params.update(overrides)
    return MultiRobotSimulationCore(SimulationParams(**params))


@pytest.fixture
def sim():
    sim = make_sim()
    yield sim
    p.disconnect(sim._client)


def box(sim, x, mode=CollisionMode.NORMAL_3D, pose=None):
    return SimObject.from_mesh(
        visual_shape=SHAPE,
        collision_shape=SHAPE,
        pose=pose or Pose.from_xyz(x, 0.0, HALF),
        mass=0.0,
        sim_core=sim,
        collision_mode=mode,
    )


def pairs(sim):
    sim.step_once()
    return {tuple(sorted(pair)) for pair in sim.get_active_collision_pairs()}


def pair(a, b):
    return tuple(sorted((a.object_id, b.object_id)))


def test_a_detached_rider_and_its_own_rider_are_pairs_of_the_old_carrier_again(sim):
    a = box(sim, 0.0)
    b = box(sim, 0.15)  # overlaps a
    c = box(sim, 0.0)  # overlaps a, rides on b
    a.attach_object(b, relative_pose=Pose.from_xyz(0.15, 0.0, 0.0))
    b.attach_object(c, relative_pose=Pose.from_xyz(-0.15, 0.0, 0.0))
    assert pairs(sim) == set()  # one tree
    assert a.detach_object(b)  # nothing moves
    assert pairs(sim) == {pair(a, b), pair(a, c)}  # b's tree (b, c) against a; b and c stay one tree


def test_the_riders_of_a_removed_carrier_are_a_pair(sim):
    a = box(sim, 0.0)
    b = box(sim, 0.15)
    c = box(sim, 0.25)  # overlaps b
    a.attach_object(b, relative_pose=Pose.from_xyz(0.15, 0.0, 0.0))
    a.attach_object(c, relative_pose=Pose.from_xyz(0.25, 0.0, 0.0))
    assert pairs(sim) == set()
    sim.remove_object(a)
    assert pairs(sim) == {pair(b, c)}


def test_ignore_attached_collision_changed_at_run_time_applies_to_the_next_check(sim):
    a = box(sim, 0.0)
    b = box(sim, 0.15)
    a.attach_object(b, relative_pose=Pose.from_xyz(0.15, 0.0, 0.0))
    assert pairs(sim) == set()
    sim._params.ignore_attached_collision = False
    assert pairs(sim) == {pair(a, b)}  # neither moved
    sim._params.ignore_attached_collision = True
    assert pairs(sim) == set()


def test_with_physics_a_rotated_kinematic_body_keeps_the_orientation_it_was_given():
    sim = make_sim(physics=True)
    try:
        pose = Pose.from_euler(1.0, 2.0, HALF, yaw=0.3)
        obj = box(sim, 0.0, pose=pose)  # its box is read from PyBullet; no trip to identity and back
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[HALF] * 3, physicsClientId=sim._client)
        raw = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            basePosition=pose.position,
            baseOrientation=pose.orientation,
            physicsClientId=sim._client,
        )
        _, read_back = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=sim._client)
        _, untouched = p.getBasePositionAndOrientation(raw, physicsClientId=sim._client)
        assert tuple(read_back) == tuple(untouched)  # not an ulp apart
        assert sim.cached_aabb(obj.object_id) is not None
    finally:
        p.disconnect(sim._client)


def test_a_disabled_object_holds_no_box_after_a_full_rebuild(sim):
    enabled = box(sim, 0.0)
    disabled = box(sim, 5.0, mode=CollisionMode.DISABLED)
    sim._aabb_cache_valid = False
    sim.check_collisions()
    assert sim.cached_aabb(enabled.object_id) is not None
    assert sim.cached_aabb(disabled.object_id) is None
