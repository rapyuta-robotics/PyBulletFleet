"""Per-axis collision clearance (#105).

``collision_margin`` takes a scalar, as before, or an ``(x, y, z)`` triple in
world axes. The triple acts in the broad phase only: each AABB axis is
enlarged by that axis's clearance. The narrow phase (``CLOSEST_POINTS`` and the
kinematic pairs of ``HYBRID``) then reports a pair whose straight-line gap is
within the largest component, as for a scalar.
"""

import itertools
import math
import random

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode, SpatialHashCellSizeMode

XY_ONLY = (0.05, 0.05, 0.0)
_PITCHED = (0.0, math.radians(0.5), 0.0)


def _sim(margin, method=CollisionDetectionMethod.CLOSEST_POINTS, cell=1.0):
    return MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=method,
            collision_margin=margin,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=cell,
            ignore_static_collision=False,
            log_level="warning",
        )
    )


def _shape(kind, size):
    if kind == "sphere":
        return ShapeParams(shape_type="sphere", radius=size)
    return ShapeParams(shape_type="box", half_extents=list(size))


def _body(sim, xyz, size=(0.1, 0.1, 0.1), kind="box", mode=CollisionMode.NORMAL_3D, rpy=(0.0, 0.0, 0.0)):
    return SimObject.from_mesh(
        visual_shape=_shape(kind, size),
        collision_shape=_shape(kind, size),
        pose=Pose.from_euler(*xyz, *rpy),
        mass=0.0,
        sim_core=sim,
        collision_mode=mode,
    )


def _pair(a, b):
    return (min(a.object_id, b.object_id), max(a.object_id, b.object_id))


@pytest.fixture
def make_sim():
    sims = []

    def make(*args, **kwargs):
        sims.append(_sim(*args, **kwargs))
        return sims[-1]

    yield make
    for sim in sims:
        p.disconnect(sim.client)


# --- Configuration ---------------------------------------------------------


@pytest.mark.parametrize("value", [(0.05, 0.05, 0.0), [0.05, 0.05, 0]])
def test_params_accept_a_triple(value):
    params = SimulationParams(gui=False, monitor=False, collision_margin=value)
    assert params.collision_margin == (0.05, 0.05, 0.0)
    assert isinstance(params.collision_margin, tuple)


def test_params_keep_a_scalar_as_it_is():
    params = SimulationParams(gui=False, monitor=False, collision_margin=0.03)
    assert params.collision_margin == 0.03
    assert isinstance(params.collision_margin, float)


@pytest.mark.parametrize("value", [(0.05, 0.05), (0.1, 0.1, 0.1, 0.1), (0.05, -0.01, 0.0), (0.05, math.inf, 0.0)])
def test_params_reject_a_bad_triple(value):
    with pytest.raises(ValueError, match="collision_margin"):
        SimulationParams(gui=False, monitor=False, collision_margin=value)


@pytest.mark.parametrize("value", [-0.01, np.float64(-0.01), math.inf, math.nan])
def test_params_reject_a_bad_scalar(value):
    """As set_collision_margin() does: the broad phase would clamp it to 0 while
    getClosestPoints got the negative distance."""
    with pytest.raises(ValueError, match="collision_margin"):
        SimulationParams(gui=False, monitor=False, collision_margin=value)


def test_from_dict_reads_a_list(tmp_path):
    assert SimulationParams.from_dict({"collision_margin": [0.05, 0.05, 0.0]}).collision_margin == XY_ONLY
    config = tmp_path / "sim.yaml"
    config.write_text("simulation:\n  collision_margin: [0.05, 0.05, 0.0]\n")
    assert SimulationParams.from_config(str(config)).collision_margin == XY_ONLY


# --- Narrow and broad phase -------------------------------------------------


@pytest.mark.parametrize("method", [CollisionDetectionMethod.CLOSEST_POINTS, CollisionDetectionMethod.HYBRID])
def test_side_by_side_within_the_xy_clearance_is_reported(make_sim, method):
    sim = make_sim(XY_ONLY, method=method)
    a = _body(sim, (0.0, 0.0, 0.1))
    b = _body(sim, (0.23, 0.0, 0.1))  # 0.03 apart in x
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, b)]


@pytest.mark.parametrize("method", [CollisionDetectionMethod.CLOSEST_POINTS, CollisionDetectionMethod.HYBRID])
def test_under_a_shelf_within_the_scalar_clearance_is_not_reported_with_no_z_clearance(make_sim, method):
    scalar = make_sim(0.05, method=method)
    a0 = _body(scalar, (0.0, 0.0, 0.1))
    shelf0 = _body(scalar, (0.05, 0.0, 0.33), size=(0.3, 0.3, 0.1))  # 0.03 above
    scalar.check_collisions()
    assert scalar.get_active_collision_pairs() == [_pair(a0, shelf0)]

    sim = make_sim(XY_ONLY, method=method)
    a = _body(sim, (0.0, 0.0, 0.1))
    shelf = _body(sim, (0.05, 0.0, 0.33), size=(0.3, 0.3, 0.1))
    sim._moved_this_step = {a.object_id, shelf.object_id}
    assert sim.filter_aabb_pairs()[0] == []
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == []


def test_touching_is_reported_whatever_the_clearance(make_sim):
    sim = make_sim((0.0, 0.0, 0.0))
    a = _body(sim, (0.0, 0.0, 0.1))
    b = _body(sim, (0.15, 0.05, 0.15))  # overlapping in every axis
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, b)]


def test_each_axis_has_its_own_clearance(make_sim):
    """Boxes 0.03 apart in x only: within (0.05, 0.02, 0) but not (0.02, 0.05, 0.05)."""
    wide_x = make_sim((0.05, 0.02, 0.0))
    a, b = _body(wide_x, (0.0, 0.0, 0.1)), _body(wide_x, (0.23, 0.0, 0.1))
    wide_x.check_collisions()
    assert wide_x.get_active_collision_pairs() == [_pair(a, b)]

    narrow_x = make_sim((0.02, 0.05, 0.05))
    _body(narrow_x, (0.0, 0.0, 0.1)), _body(narrow_x, (0.23, 0.0, 0.1))
    narrow_x.check_collisions()
    assert narrow_x.get_active_collision_pairs() == []


def test_the_narrow_phase_judges_the_straight_line_gap(make_sim):
    """Spheres whose AABBs overlap in z, with closest points 1.5 mm apart in z:
    the broad phase passes them and, with no z clearance, the pair is still
    reported, because the narrow phase judges the straight-line gap."""
    sim = make_sim(XY_ONLY)
    a = _body(sim, (0.0, 0.0, 0.2), size=0.1, kind="sphere")
    b = _body(sim, (0.2, 0.0, 0.25), size=0.1, kind="sphere")
    sim._moved_this_step = {a.object_id, b.object_id}
    assert sim.filter_aabb_pairs()[0] == [_pair(a, b)]
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, b)]


def test_the_narrow_phase_queries_the_largest_component(make_sim):
    """Boxes 0.04 apart in x and 0.008 in y pass the broad phase of
    (0.05, 0.01, 0); their 0.041 gap is within 0.05, the largest component."""
    sim = make_sim((0.05, 0.01, 0.0))
    a = _body(sim, (0.0, 0.0, 0.1))
    b = _body(sim, (0.24, 0.208, 0.1))
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, b)]
    observation = sim.get_collision_observation()
    assert observation is not None
    assert observation.pairs[0].signed_distance == pytest.approx(math.hypot(0.04, 0.008), abs=1e-3)


def test_a_near_miss_between_tilted_boxes_is_reported(make_sim):
    """Boxes 3 cm apart, both pitched by 0.5 degree (as on a ramp): the closest
    points are not level, but the AABBs overlap in z, so (0.05, 0.05, 0)
    reports the pair."""
    sim = make_sim(XY_ONLY)
    a = _body(sim, (0.0, 0.0, 0.1), rpy=_PITCHED)
    b = _body(sim, (0.23, 0.0, 0.1), rpy=_PITCHED)
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, b)]


def test_contact_points_ignore_the_triple(make_sim):
    sim = make_sim((0.5, 0.5, 0.5), method=CollisionDetectionMethod.CONTACT_POINTS)
    a, b = _body(sim, (0.0, 0.0, 0.1)), _body(sim, (0.23, 0.0, 0.1))
    sim._moved_this_step = {a.object_id, b.object_id}
    assert sim.filter_aabb_pairs()[0] == []


def test_observation_reports_the_configured_triple(make_sim):
    sim = make_sim(XY_ONLY)
    _body(sim, (0.0, 0.0, 0.1)), _body(sim, (0.23, 0.0, 0.1))
    sim.check_collisions()
    observation = sim.get_collision_observation()
    assert observation is not None and observation.margin == XY_ONLY
    assert observation.pairs[0].signed_distance == pytest.approx(0.03, abs=1e-4)


@pytest.mark.parametrize("seed", range(3))
def test_an_equal_triple_behaves_as_the_scalar(seed):
    """(m, m, m) finds the same pairs, candidates and signed distances as m."""
    results = []
    for margin in (0.05, (0.05, 0.05, 0.05)):
        rng = random.Random(seed)
        sim = _sim(margin, cell=0.5)
        try:
            objects = []
            for _ in range(40):
                xyz = [rng.uniform(0.0, 1.5) for _ in range(3)]
                kind = rng.choice(["box", "sphere"])
                size = rng.uniform(0.03, 0.1) if kind == "sphere" else [rng.uniform(0.03, 0.1) for _ in range(3)]
                objects.append(_body(sim, xyz, size=size, kind=kind))
            sim._moved_this_step = {o.object_id for o in objects}
            candidates = sorted(sim.filter_aabb_pairs()[0])
            sim._moved_this_step = {o.object_id for o in objects}
            sim.check_collisions()
            observation = sim.get_collision_observation()
            assert observation is not None
            results.append((candidates, [(o.object_ids, o.signed_distance) for o in observation.pairs]))
        finally:
            p.disconnect(sim.client)
    assert results[0] == results[1]
    assert results[0][1]  # the scene has near misses


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("margin", [(0.1, 0.02, 0.0), (0.0, 0.0, 0.25), (0.02, 0.1, 0.05)])
def test_broadphase_returns_every_pair_within_the_per_axis_clearance(seed, margin):
    """Against brute force: boxes on and off cell boundaries; a pair whose
    AABBs, each axis enlarged by that axis's clearance, overlap is a
    candidate, and one whose do not is not."""
    rng = random.Random(seed)
    cell = rng.choice([0.2, 0.5, 1.0])
    sim = _sim(margin, cell=cell)
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        objects = []
        for _ in range(30):
            half = [0.5 * cell * rng.choice([0.04, 0.1, 0.9, 1.4, 2.8]) for _ in range(3)]
            offsets = [0.0, 0.001, -0.001, rng.uniform(-cell, cell)]
            xyz = [round(rng.uniform(0.0, 3.0) / cell) * cell + rng.choice(offsets) for _ in range(3)]
            objects.append(_body(sim, xyz, size=half))
        sim._moved_this_step = {o.object_id for o in objects}
        found = set(sim.filter_aabb_pairs()[0])
        aabbs = sim._cached_aabbs_dict
        expected = set()
        for a, b in itertools.combinations([o.object_id for o in objects], 2):
            box_a, box_b = aabbs[a], aabbs[b]
            if all(box_a[1][d] + margin[d] >= box_b[0][d] and box_a[0][d] - margin[d] <= box_b[1][d] for d in range(3)):
                expected.add((min(a, b), max(a, b)))
        assert expected and found == expected
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_each_axis_registers_with_its_own_clearance(axis):
    """Two boxes 0.04 apart across a cell boundary (cell 1.0) on one axis, with
    0.1 of clearance on that axis and 0.02 on the others: they share a cell
    only if each is registered with that axis's clearance, not another's."""
    margin = tuple(0.1 if d == axis else 0.02 for d in range(3))
    sim = _sim(margin, cell=1.0)
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        xyz_a, xyz_b = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        xyz_a[axis], xyz_b[axis] = 0.74, 1.26
        a = _body(sim, xyz_a, size=(0.24, 0.24, 0.24))
        b = _body(sim, xyz_b, size=(0.24, 0.24, 0.24))
        sim._moved_this_step = {a.object_id}
        assert sim.filter_aabb_pairs()[0] == [_pair(a, b)]
    finally:
        p.disconnect(sim.client)


# --- Changing the clearance -------------------------------------------------


@pytest.mark.parametrize("change", ["setter", "params"])
def test_a_clearance_change_rechecks_a_stationary_pair(make_sim, change):
    """Two STATIC bodies 0.03 apart in z: reported with 0.05, resolved when the
    z clearance drops to 0, reported again when it is restored."""
    sim = make_sim(0.05)
    a = _body(sim, (0.0, 0.0, 0.1), mode=CollisionMode.STATIC)
    shelf = _body(sim, (0.05, 0.0, 0.33), size=(0.3, 0.3, 0.1), mode=CollisionMode.STATIC)
    sim._moved_this_step = {a.object_id, shelf.object_id}
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, shelf)]

    def set_margin(value):
        if change == "setter":
            sim.set_collision_margin(value)
        else:
            sim.params.collision_margin = value

    set_margin(XY_ONLY)
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == []

    set_margin((0.05, 0.05, 0.05))
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, shelf)]


@pytest.mark.parametrize("value", [-0.01, math.nan, (0.05, -0.01, 0.0), "0.05"])
def test_a_bad_margin_set_on_params_raises_before_the_check_changes_anything(make_sim, value):
    """A bad margin set on params directly is rejected when the check starts:
    the pair stays active and the grid and last observation are untouched."""
    sim = make_sim(0.05)
    a = _body(sim, (0.0, 0.0, 0.1), mode=CollisionMode.STATIC)
    shelf = _body(sim, (0.05, 0.0, 0.33), size=(0.3, 0.3, 0.1), mode=CollisionMode.STATIC)
    sim._moved_this_step = {a.object_id, shelf.object_id}
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, shelf)]
    grid_margin = sim._grid_margin
    observation = sim.get_collision_observation()

    sim.params.collision_margin = value
    with pytest.raises(ValueError, match="collision_margin"):
        sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, shelf)]
    assert sim._grid_margin == grid_margin
    assert sim.get_collision_observation() == observation

    sim.params.collision_margin = XY_ONLY
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == []


def test_set_collision_margin_stores_a_tuple(make_sim):
    sim = make_sim(0.02)
    sim.set_collision_margin([0.05, 0.05, 0])
    assert sim.params.collision_margin == XY_ONLY


@pytest.mark.parametrize("value", [(0.05, 0.05), (0.05, -0.01, 0.0), (0.05, math.nan, 0.0)])
def test_set_collision_margin_rejects_a_bad_triple(make_sim, value):
    sim = make_sim(0.02)
    with pytest.raises(ValueError):
        sim.set_collision_margin(value)
    assert sim.params.collision_margin == 0.02


@pytest.mark.parametrize("value", [np.float32(0.02), np.float64(0.02), np.int64(0)])
def test_a_numpy_scalar_is_a_scalar(make_sim, value):
    params = SimulationParams(gui=False, monitor=False, collision_margin=value)
    assert params.collision_margin == pytest.approx(float(value))
    assert type(params.collision_margin) is float
    sim = make_sim(0.05)
    sim.set_collision_margin(value)
    assert sim.params.collision_margin == pytest.approx(float(value))
    assert type(sim.params.collision_margin) is float


@pytest.mark.parametrize("value", ["005", "0.05", True, (True, True, True)])
def test_a_string_or_bool_is_rejected(make_sim, value):
    with pytest.raises(ValueError, match="collision_margin"):
        SimulationParams(gui=False, monitor=False, collision_margin=value)
    sim = make_sim(0.02)
    with pytest.raises(ValueError, match="collision_margin"):
        sim.set_collision_margin(value)
    assert sim.params.collision_margin == 0.02


def test_set_collision_margin_stores_an_int_as_a_float(make_sim):
    sim = make_sim(0.02)
    sim.set_collision_margin(1)
    assert type(sim.params.collision_margin) is float


def test_observation_holds_a_tuple_after_a_list_is_set_on_params(make_sim):
    sim = make_sim(0.05)
    _body(sim, (0.0, 0.0, 0.1)), _body(sim, (0.23, 0.0, 0.1))
    margin = [0.05, 0.05, 0.0]
    sim.params.collision_margin = margin
    sim.check_collisions()
    margin[0] = 9.0
    observation = sim.get_collision_observation()
    assert observation is not None and observation.margin == XY_ONLY
    assert isinstance(observation.margin, tuple)
    hash(observation.margin)


# --- Against NORMAL_2D ------------------------------------------------------


def _active_pairs(margin, mode, bodies):
    """Active pairs, by index in ``bodies``, with every body of ``mode``.

    Each body is ``(xyz, size, kind, rpy)``.
    """
    sim = _sim(margin, cell=0.5)
    try:
        objects = [_body(sim, xyz, size=size, kind=kind, mode=mode, rpy=rpy) for xyz, size, kind, rpy in bodies]
        index = {o.object_id: k for k, o in enumerate(objects)}
        sim.check_collisions()
        return sorted(tuple(sorted((index[a], index[b]))) for a, b in sim.get_active_collision_pairs())
    finally:
        p.disconnect(sim.client)


_BOX = (0.1, 0.1, 0.1)
_LEVEL = (0.0, 0.0, 0.0)
_SCENES = {
    "side by side": [((0.0, 0.0, 0.1), _BOX, "box", _LEVEL), ((0.23, 0.0, 0.1), _BOX, "box", _LEVEL)],
    "under a shelf": [((0.0, 0.0, 0.1), _BOX, "box", _LEVEL), ((0.05, 0.0, 0.33), (0.3, 0.3, 0.1), "box", _LEVEL)],
    "diagonal": [((0.0, 0.0, 0.1), _BOX, "box", _LEVEL), ((0.22, 0.22, 0.1), _BOX, "box", _LEVEL)],
    "diagonal above": [((0.0, 0.0, 0.1), _BOX, "box", _LEVEL), ((0.22, 0.0, 0.32), _BOX, "box", _LEVEL)],
    "spheres at different heights": [
        ((0.0, 0.0, 0.2), 0.1, "sphere", _LEVEL),
        ((0.2, 0.0, 0.25), 0.1, "sphere", _LEVEL),
    ],
    "tilted boxes": [
        ((0.0, 0.0, 0.1), _BOX, "box", _PITCHED),
        ((0.23, 0.0, 0.1), _BOX, "box", _PITCHED),
    ],
}


def _random_scene(seed):
    """40 boxes and spheres, some boxes tilted."""
    rng = random.Random(seed)
    bodies = []
    for _ in range(40):
        xyz = tuple(rng.uniform(0.0, 1.2) for _ in range(3))
        if rng.random() < 0.3:
            bodies.append((xyz, rng.uniform(0.03, 0.12), "sphere", _LEVEL))
        else:
            rpy = tuple(rng.choice([0.0, rng.uniform(-0.3, 0.3)]) for _ in range(3))
            bodies.append((xyz, tuple(rng.uniform(0.03, 0.12) for _ in range(3)), "box", rpy))
    return bodies


@pytest.mark.filterwarnings("ignore:CollisionMode.NORMAL_2D is deprecated:FutureWarning")
@pytest.mark.parametrize(
    "bodies",
    list(_SCENES.values()) + [_random_scene(seed) for seed in range(3)],
    ids=list(_SCENES) + ["random 0", "random 1", "random 2"],
)
def test_a_zero_z_clearance_matches_normal_2d(bodies):
    """Every body NORMAL_2D with margin m reports exactly what every body
    NORMAL_3D with (m, m, 0) does."""
    assert _active_pairs(0.05, CollisionMode.NORMAL_2D, bodies) == _active_pairs(
        (0.05, 0.05, 0.0), CollisionMode.NORMAL_3D, bodies
    )


@pytest.mark.filterwarnings("ignore:CollisionMode.NORMAL_2D is deprecated:FutureWarning")
@pytest.mark.parametrize(
    "margin, mode",
    [(0.05, CollisionMode.NORMAL_2D), ((0.05, 0.05, 0.0), CollisionMode.NORMAL_3D)],
    ids=["normal_2d", "per_axis"],
)
def test_an_active_pair_that_moves_out_of_the_axis_clearance_resolves(make_sim, margin, mode):
    """Side by side, then one body lifted 3 cm clear in z (gap about 0.042):
    the pair resolves, as it would not be reported in a fresh scene."""
    sim = make_sim(margin)
    a = _body(sim, (0.0, 0.0, 0.1), mode=mode)
    b = _body(sim, (0.23, 0.0, 0.1), mode=mode)
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == [_pair(a, b)]
    b.set_pose(Pose.from_xyz(0.23, 0.0, 0.33))
    sim.check_collisions()
    assert sim.get_active_collision_pairs() == []


@pytest.mark.parametrize("seed", range(3))
def test_a_zero_z_clearance_reports_fewer_pairs_than_the_scalar(seed):
    """NORMAL_3D with m also reports the near misses above or below."""
    bodies = _random_scene(seed)
    per_axis = _active_pairs((0.05, 0.05, 0.0), CollisionMode.NORMAL_3D, bodies)
    normal_3d = _active_pairs(0.05, CollisionMode.NORMAL_3D, bodies)
    assert set(per_axis) < set(normal_3d)
