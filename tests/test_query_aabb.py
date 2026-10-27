"""``query_aabb`` finds what a full scan finds, through the spatial hash.

``find_nearest_pickable`` and an elevator's passenger scan used to test every
object in the simulation on each call. In a scene with tens of thousands of
static bodies that made one call cost tens of milliseconds. They now ask
``query_aabb`` for the objects near the point, so these tests pin that the
answer is the one the full scan would give.
"""

import random
from typing import List, Optional

import pybullet as p
import pytest

from pybullet_fleet import MultiRobotSimulationCore, SimObject, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode


def make_sim(method: CollisionDetectionMethod = CollisionDetectionMethod.CLOSEST_POINTS, **params) -> MultiRobotSimulationCore:
    return MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=method,
            log_level="warning",
            **params,
        )
    )


# CONTACT_POINTS has no broadphase margin: the hash then holds each body in
# the cells its bare AABB reaches, the case most likely to miss an overlap.
@pytest.fixture(params=[CollisionDetectionMethod.CLOSEST_POINTS, CollisionDetectionMethod.CONTACT_POINTS])
def sim(request):
    sim = make_sim(request.param)
    yield sim
    p.disconnect(sim.client)


def box(
    sim: MultiRobotSimulationCore,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
    half: float = 0.2,
    name: Optional[str] = None,
    mode: CollisionMode = CollisionMode.NORMAL_3D,
    pickable: bool = False,
) -> SimObject:
    shape = ShapeParams(shape_type="box", half_extents=[half, half, half])
    obj = SimObject.from_mesh(
        visual_shape=shape,
        collision_shape=shape,
        pose=Pose.from_xyz(x, y, z),
        sim_core=sim,
        collision_mode=mode,
        name=name,
    )
    obj.pickable = pickable
    return obj


def brute_force(sim: MultiRobotSimulationCore, lo, hi) -> List[SimObject]:
    found = []
    for obj in sim.sim_objects:
        (x0, y0, z0), (x1, y1, z1) = p.getAABB(obj.body_id, physicsClientId=sim.client)
        if x0 <= hi[0] and x1 >= lo[0] and y0 <= hi[1] and y1 >= lo[1] and z0 <= hi[2] and z1 >= lo[2]:
            found.append(obj)
    return found


def names(objects) -> List[str]:
    return sorted(o.name for o in objects)


def scatter(sim: MultiRobotSimulationCore, count: int = 300, seed: int = 3) -> None:
    rng = random.Random(seed)
    modes = [CollisionMode.STATIC, CollisionMode.NORMAL_3D, CollisionMode.NORMAL_2D, CollisionMode.DISABLED]
    for i in range(count):
        box(
            sim,
            rng.uniform(-10, 10),
            rng.uniform(-10, 10),
            rng.uniform(0, 3),
            half=rng.choice([0.05, 0.2, 1.5]),
            name="o{}".format(i),
            mode=modes[i % len(modes)],
        )


BOXES = [
    ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    ((-3.0, 2.0, 0.0), (1.0, 6.0, 3.0)),
    ((4.9, -7.2, 1.1), (5.1, -7.0, 1.3)),
    ((-12.0, -12.0, -1.0), (12.0, 12.0, 4.0)),
]


class TestQueryAabb:
    @pytest.mark.parametrize("lo, hi", BOXES)
    def test_matches_a_full_scan_before_the_hash_exists(self, sim, lo, hi):
        scatter(sim)
        assert names(sim.query_aabb(lo, hi)) == names(brute_force(sim, lo, hi))

    @pytest.mark.parametrize("lo, hi", BOXES)
    def test_matches_a_full_scan_through_the_hash(self, sim, lo, hi):
        scatter(sim)
        sim.step_once()  # builds the spatial hash
        assert sim._cached_cell_size is not None
        assert names(sim.query_aabb(lo, hi)) == names(brute_force(sim, lo, hi))

    def test_finds_an_object_moved_since_the_last_step(self, sim):
        scatter(sim, count=40)
        mover = box(sim, 50.0, name="mover")
        sim.step_once()
        mover.set_pose(Pose.from_xyz(0.0, 0.0, 0.0))
        assert "mover" in names(sim.query_aabb((-0.3, -0.3, -0.3), (0.3, 0.3, 0.3)))

    def test_finds_an_object_whose_centre_cell_is_outside_the_box(self, sim):
        """A body overlapping the box with its centre in a cell the box does not
        reach. A hash that registered such a body only in its centre cell would
        miss it."""
        from pybullet_fleet.types import SpatialHashCellSizeMode

        sim.set_collision_spatial_hash_cell_size_mode(mode=SpatialHashCellSizeMode.CONSTANT, cell_size=1.0)
        box(sim, 2.5, 0.5, 0.5, half=0.6, name="wide")  # AABB x 1.9..3.1, centre cell 2
        sim.step_once()
        lo, hi = (1.92, 0.4, 0.4), (1.98, 0.6, 0.6)  # cell 1 only
        assert names(sim.query_aabb(lo, hi)) == names(brute_force(sim, lo, hi)) == ["wide"]

    def test_a_box_wider_than_the_scene_scans_everything(self, sim):
        scatter(sim, count=40)
        sim.step_once()
        lo, hi = (-1000.0, -1000.0, -10.0), (1000.0, 1000.0, 10.0)
        assert names(sim.query_aabb(lo, hi)) == names(sim.sim_objects)

    def test_finds_an_object_with_collision_disabled(self, sim):
        scatter(sim, count=40)
        box(sim, 30.0, name="ghost", mode=CollisionMode.DISABLED)
        sim.step_once()
        assert names(sim.query_aabb((29.9, -0.1, -0.1), (30.1, 0.1, 0.1))) == ["ghost"]


class TestNearestPickable:
    @pytest.mark.parametrize("radius", [0.3, 1.0, 4.0])
    def test_same_answer_as_a_full_scan(self, sim, radius):
        from pybullet_fleet import tools

        rng = random.Random(5)
        for i in range(120):
            box(sim, rng.uniform(-5, 5), rng.uniform(-5, 5), 0.0, half=0.1, name="b{}".format(i), pickable=i % 3 != 0)
        finder = box(sim, 0.0, name="finder")
        sim.step_once()
        expected = tools.find_nearest(
            sim.sim_objects,
            position=finder.get_pose().position,
            search_radius=radius,
            predicate=lambda o: o.pickable and not o.is_attached(),
            exclude=finder,
        )
        assert finder.find_nearest_pickable(search_radius=radius) is expected


class TestElevatorPassengers:
    def test_same_passengers_as_a_full_scan(self, sim):
        from pybullet_fleet.devices.elevator import Elevator, ElevatorParams

        scatter(sim, count=200)
        car = Elevator.from_params(
            ElevatorParams(
                urdf_path="robots/elevator.urdf",
                initial_pose=Pose.from_xyz(20.0, 0.0, 0.0),
                use_fixed_base=True,
                floors={"L1": 0.0, "L2": 8.0},
                initial_floor="L1",
                joint_name="lift",
                platform_link="platform",
            ),
            sim,
        )
        box(sim, 20.0, 0.0, 0.2, name="rider")
        box(sim, 23.0, 0.0, 0.2, name="next door")
        sim.step_once()

        platform_min, platform_max = car._get_platform_aabb()
        search_max = (platform_max[0], platform_max[1], platform_max[2] + 1.0)
        expected = [o for o in brute_force(sim, platform_min, search_max) if o is not car and not o.is_attached()]
        assert names(car._find_objects_on_platform()) == names(expected)
        assert "rider" in names(expected)


class TestHashFreshness:
    @pytest.mark.parametrize("frequency", [0, 0.1])
    def test_an_agent_driven_across_cells_is_found_where_it_is(self, frequency):
        """The hash follows a controller-driven agent every step, whether or not
        a collision check runs on that step."""
        from pybullet_fleet import Agent, AgentSpawnParams

        sim = make_sim(collision_check_frequency=None)
        try:
            scatter(sim, count=60)
            sim.step_once()  # builds the hash
            assert sim._cached_cell_size is not None
            sim.set_collision_check_frequency(frequency)
            agent = Agent.from_params(
                AgentSpawnParams(urdf_path="cube_small.urdf", name="walker", initial_pose=Pose.from_xyz(-20.0, -20.0, 0.0)),
                sim_core=sim,
            )
            agent.set_goal_pose(Pose.from_xyz(-14.0, -20.0, 0.0))
            for _ in range(400):
                sim.step_once()
            x, y, z = agent.get_pose().position
            assert x > -19.0  # it moved several cells
            here = sim.query_aabb((x - 0.1, y - 0.1, z - 0.1), (x + 0.1, y + 0.1, z + 0.1))
            assert "walker" in names(here)
            start = sim.query_aabb((-20.1, -20.1, -0.1), (-19.9, -19.9, 0.1))
            assert "walker" not in names(start)
        finally:
            p.disconnect(sim.client)


class TestOriginOutsideItsAabb:
    def test_an_object_whose_shape_sits_off_its_origin_is_found(self, sim):
        """Distance is measured from the base origin; a body whose shape is
        offset from it is still found."""
        shape = ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1], frame_pose=Pose.from_xyz(0.6, 0.0, 0.0))
        shelf = SimObject.from_mesh(
            visual_shape=shape, collision_shape=shape, pose=Pose.from_xyz(5.0, 5.0, 0.0), sim_core=sim, name="shelf"
        )
        shelf.pickable = True
        finder = box(sim, 5.2, 5.0, 0.0, half=0.05, name="finder")
        sim.step_once()
        # Origin 0.2 m away, AABB 0.3 m away beyond the origin: outside a 0.25 m box.
        assert finder.find_nearest_pickable(search_radius=0.25) is shelf
