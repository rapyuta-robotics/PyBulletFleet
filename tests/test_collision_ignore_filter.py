"""Excluding expected contact from collision results.

``ignore_attached_collision`` drops contact between an attached object and
its carrier; ``set_collision_ignore_filter()`` drops whatever pairs the
application says are meant to touch. Either way the pair must be gone from
every place a collision shows up -- active pairs, the count, the events and
the observation -- not just from one of them.
"""

from typing import List, Optional, Tuple

import pybullet as p
import pytest

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode


def make_sim(**overrides) -> MultiRobotSimulationCore:
    params = dict(
        gui=False,
        physics=False,
        monitor=False,
        collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
        collision_margin=0.02,
        log_level="warning",
    )
    params.update(overrides)
    sim = MultiRobotSimulationCore(SimulationParams(**params))
    sim.set_collision_spatial_hash_cell_size_mode()
    return sim


@pytest.fixture
def sim():
    sim = make_sim()
    yield sim
    p.disconnect(sim.client)


def box(
    sim: MultiRobotSimulationCore,
    x: float,
    z: float = 0.0,
    name: Optional[str] = None,
    mode: CollisionMode = CollisionMode.NORMAL_3D,
) -> SimObject:
    shape = ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2])
    return SimObject.from_mesh(
        visual_shape=shape,
        collision_shape=shape,
        pose=Pose.from_xyz(x, 0.0, z),
        sim_core=sim,
        collision_mode=mode,
        name=name,
    )


def pair_of(a: SimObject, b: SimObject) -> Tuple[int, int]:
    return tuple(sorted((a.object_id, b.object_id)))  # type: ignore[return-value]


def record_events(sim: MultiRobotSimulationCore) -> List[Tuple[str, str, str]]:
    events: List[Tuple[str, str, str]] = []

    def started(obj_a=None, obj_b=None, **_):
        events.append(("started", obj_a.name, obj_b.name))

    def ended(obj_a=None, obj_b=None, **_):
        events.append(("ended", obj_a.name, obj_b.name))

    sim.events.on(SimEvents.COLLISION_STARTED, started)
    sim.events.on(SimEvents.COLLISION_ENDED, ended)
    return events


def observed_pairs(sim: MultiRobotSimulationCore):
    observation = sim.get_collision_observation()
    return [] if observation is None else list(observation.pairs)


class TestIgnoreAttachedCollision:
    def test_a_carried_object_does_not_collide_with_its_carrier(self, sim):
        carrier = box(sim, 0.0, name="carrier")
        load = box(sim, 0.0, z=0.3, name="load")
        assert carrier.attach_object(load, keep_world_pose=True)
        events = record_events(sim)

        pairs, _ = sim.check_collisions()

        assert pairs == []
        assert sim.collision_count == 0
        assert events == []
        assert observed_pairs(sim) == []

    def test_the_carried_object_still_collides_with_anything_else(self, sim):
        carrier = box(sim, 0.0, name="carrier")
        load = box(sim, 0.0, z=0.3, name="load")
        other = box(sim, 0.3, z=0.3, name="other")
        assert carrier.attach_object(load, keep_world_pose=True)

        pairs, _ = sim.check_collisions()

        assert pair_of(load, other) in pairs
        assert pair_of(carrier, load) not in pairs

    def test_attaching_resolves_a_contact_that_was_already_active(self, sim):
        carrier = box(sim, 0.0, name="carrier")
        load = box(sim, 0.0, z=0.3, name="load")
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == [pair_of(carrier, load)]
        events = record_events(sim)

        assert carrier.attach_object(load, keep_world_pose=True)
        pairs, _ = sim.check_collisions()

        assert pairs == []
        assert events == [("ended", "carrier", "load")] or events == [("ended", "load", "carrier")]

    def test_detaching_in_place_reports_the_contact(self, sim):
        """detach_object() without a drop pose moves nothing, so the pair is
        found again only because ignored pairs are re-evaluated."""
        carrier = box(sim, 0.0, name="carrier")
        load = box(sim, 0.0, z=0.3, name="load")
        assert carrier.attach_object(load, keep_world_pose=True)
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == []
        events = record_events(sim)

        assert carrier.detach_object(load)
        pairs, _ = sim.check_collisions()

        assert pairs == [pair_of(carrier, load)]
        assert [kind for kind, _, _ in events] == ["started"]

    def test_can_be_turned_off(self):
        sim = make_sim(ignore_attached_collision=False)
        try:
            carrier = box(sim, 0.0, name="carrier")
            load = box(sim, 0.0, z=0.3, name="load")
            assert carrier.attach_object(load, keep_world_pose=True)

            pairs, _ = sim.check_collisions()

            assert pairs == [pair_of(carrier, load)]
        finally:
            p.disconnect(sim.client)


def loads_on_one_carrier(sim: MultiRobotSimulationCore) -> Tuple[SimObject, SimObject, SimObject]:
    """A carrier with two loads side by side on it, overlapping each other."""
    carrier = box(sim, 0.0, name="carrier")
    first = box(sim, -0.15, z=0.3, name="first")
    second = box(sim, 0.15, z=0.3, name="second")
    assert carrier.attach_object(first, keep_world_pose=True)
    assert carrier.attach_object(second, keep_world_pose=True)
    return carrier, first, second


def load_on_a_load(sim: MultiRobotSimulationCore) -> Tuple[SimObject, SimObject, SimObject]:
    """A carrier carrying a load that carries a second load, which overlaps the carrier."""
    carrier = box(sim, 0.0, name="carrier")
    first = box(sim, 0.0, z=0.3, name="first")
    second = box(sim, 0.1, z=0.05, name="second")
    assert carrier.attach_object(first, keep_world_pose=True)
    assert first.attach_object(second, keep_world_pose=True)
    return carrier, first, second


class TestOneAttachmentTree:
    """``ignore_attached_collision`` covers every object on an attachment tree:
    a carrier, its loads and a load's loads, not only a direct parent/child."""

    @pytest.mark.parametrize("build", [loads_on_one_carrier, load_on_a_load], ids=["siblings", "chain"])
    def test_never_collides_with_itself(self, sim, build):
        build(sim)
        events = record_events(sim)

        pairs, _ = sim.check_collisions()

        assert pairs == []
        assert sim.collision_count == 0
        assert events == []
        assert observed_pairs(sim) == []

    @pytest.mark.parametrize("build", [loads_on_one_carrier, load_on_a_load], ids=["siblings", "chain"])
    def test_turned_off_the_whole_tree_is_reported(self, build):
        sim = make_sim(ignore_attached_collision=False)
        try:
            carrier, first, second = build(sim)

            pairs, _ = sim.check_collisions()

            assert pair_of(first, second) in pairs
            assert pair_of(carrier, second) in pairs
        finally:
            p.disconnect(sim.client)

    def test_attaching_two_touching_objects_to_one_carrier_resolves_their_contact(self, sim):
        """Nothing moves: the pair is re-evaluated because it is active."""
        carrier = box(sim, -1.0, name="carrier")
        first = box(sim, -0.15, z=0.3, name="first")
        second = box(sim, 0.15, z=0.3, name="second")
        sim.check_collisions()
        assert pair_of(first, second) in sim.get_active_collision_pairs()
        events = record_events(sim)

        assert carrier.attach_object(first, keep_world_pose=True)
        assert carrier.attach_object(second, keep_world_pose=True)
        pairs, _ = sim.check_collisions()

        assert pair_of(first, second) not in pairs
        assert [kind for kind, _, _ in events] == ["ended"]

    def test_detaching_one_load_reports_its_contact_with_the_other(self, sim):
        """detach_object() moves nothing: the ignored pair is found again only
        because ignored pairs are re-evaluated."""
        carrier, first, second = loads_on_one_carrier(sim)
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == []
        events = record_events(sim)

        assert carrier.detach_object(second)
        pairs, _ = sim.check_collisions()

        assert pair_of(first, second) in pairs
        assert [kind for kind, _, _ in events].count("started") >= 1


class TestCollisionIgnoreFilter:
    def test_an_ignored_pair_is_absent_everywhere(self, sim):
        a = box(sim, 0.0, name="a")
        b = box(sim, 0.3, name="b")
        sim.set_collision_ignore_filter(lambda x, y: {x.name, y.name} == {"a", "b"})
        events = record_events(sim)

        pairs, _ = sim.check_collisions()

        assert pairs == []
        assert sim.collision_count == 0
        assert events == []
        assert observed_pairs(sim) == []
        assert a.object_id not in sim._active_pairs_by_object
        assert b.object_id not in sim._active_pairs_by_object

    def test_only_the_pairs_it_names_are_dropped(self, sim):
        a = box(sim, 0.0, name="a")
        b = box(sim, 0.3, name="b")
        c = box(sim, -0.3, name="c")
        sim.set_collision_ignore_filter(lambda x, y: {x.name, y.name} == {"a", "b"})

        pairs, _ = sim.check_collisions()

        assert pairs == [pair_of(a, c)]
        assert pair_of(a, b) not in pairs
        assert b.object_id not in sim._active_pairs_by_object

    def test_setting_it_resolves_an_active_pair_without_movement(self, sim):
        box(sim, 0.0, name="a")
        box(sim, 0.3, name="b")
        sim.check_collisions()
        assert sim.collision_count == 1
        events = record_events(sim)

        sim.set_collision_ignore_filter(lambda x, y: True)
        pairs, _ = sim.check_collisions()

        assert pairs == []
        assert [kind for kind, _, _ in events] == ["ended"]

    def test_a_filter_whose_answer_changes_resolves_the_pair(self, sim):
        """The filter may read state that changes without anything moving --
        a body boarding a platform, say. The active pair is still resolved on
        the next check, not left until one of the two happens to move."""
        a = box(sim, 0.0, name="a")
        b = box(sim, 0.3, name="b")
        expected = set()
        sim.set_collision_ignore_filter(lambda x, y: frozenset((x.name, y.name)) in expected)
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == [pair_of(a, b)]

        expected.add(frozenset(("a", "b")))
        pairs, _ = sim.check_collisions()

        assert pairs == []

    def test_a_filter_that_stops_ignoring_reports_the_pair(self, sim):
        """The reverse of the case above, without calling the setter again."""
        a = box(sim, 0.0, name="a")
        b = box(sim, 0.3, name="b")
        expected = {frozenset(("a", "b"))}
        sim.set_collision_ignore_filter(lambda x, y: frozenset((x.name, y.name)) in expected)
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == []

        expected.clear()
        pairs, _ = sim.check_collisions()

        assert pairs == [pair_of(a, b)]

    def test_a_static_pair_that_becomes_ignored_resolves(self):
        """Active STATIC-STATIC pairs are otherwise never rechecked; the ignore
        rule has to reach them anyway."""
        sim = make_sim(ignore_static_collision=False)
        try:
            a = box(sim, 0.0, name="a", mode=CollisionMode.STATIC)
            b = box(sim, 0.3, name="b", mode=CollisionMode.STATIC)
            expected = set()
            sim.set_collision_ignore_filter(lambda x, y: frozenset((x.name, y.name)) in expected)
            sim.check_collisions()
            assert sim.get_active_collision_pairs() == [pair_of(a, b)]
            events = record_events(sim)

            expected.add(frozenset(("a", "b")))
            pairs, _ = sim.check_collisions()

            assert pairs == []
            assert observed_pairs(sim) == []
            assert [kind for kind, _, _ in events] == ["ended"]
        finally:
            p.disconnect(sim.client)

    def test_an_ignored_pair_that_moves_apart_is_forgotten(self, sim):
        a = box(sim, 0.0, name="a")
        b = box(sim, 0.3, name="b")
        sim.set_collision_ignore_filter(lambda x, y: True)
        sim.check_collisions()
        assert sim._ignored_collision_pairs == {pair_of(a, b)}

        b.set_pose(Pose.from_xyz(10.0, 0.0, 0.0))
        sim.check_collisions()

        assert sim._ignored_collision_pairs == set()

    def test_removing_it_reports_a_stationary_pair_again(self, sim):
        a = box(sim, 0.0, name="a")
        b = box(sim, 0.3, name="b")
        sim.set_collision_ignore_filter(lambda x, y: True)
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == []
        events = record_events(sim)

        sim.set_collision_ignore_filter(None)
        pairs, _ = sim.check_collisions()

        assert pairs == [pair_of(a, b)]
        assert [kind for kind, _, _ in events] == ["started"]
        assert sim.collision_count == 1

    def test_an_ignored_pair_skips_the_narrow_phase(self, sim, monkeypatch):
        box(sim, 0.0, name="a")
        box(sim, 0.3, name="b")
        sim.set_collision_ignore_filter(lambda x, y: True)
        calls = []
        real = p.getClosestPoints

        def counting(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(p, "getClosestPoints", counting)
        sim.check_collisions()

        assert calls == []
