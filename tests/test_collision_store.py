"""The collision store: AABBs from poses as PyBullet computes them, and an
array spatial hash whose candidate pairs are exactly the pairs within the
clearance, however the rows move, change mode or leave."""

import itertools
import random

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet.collision_store import (
    MODE_DISABLED,
    MODE_NORMAL_2D,
    MODE_NORMAL_3D,
    MODE_STATIC,
    CollisionStore,
    rotation_matrices,
)
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionMode

MODES = (MODE_NORMAL_3D, MODE_NORMAL_2D, MODE_STATIC, MODE_DISABLED)


def _random_store(rng, n, *, cell=1.0, half_margin=(0.0, 0.0, 0.0), spread=4.0):
    store = CollisionStore(capacity=4)
    for k in range(n):
        row = store.add(
            100 + k,
            mode=rng.choice(MODES),
            physics=rng.random() < 0.2,
            pickable=False,
            pybullet_aabb=False,
        )
        lo = np.array([rng.uniform(-spread, spread) for _ in range(3)])
        size = np.array([rng.choice([0.05, 0.3, 0.9, 1.4, 2.6]) for _ in range(3)])
        store.set_aabb(row, (tuple(lo), tuple(lo + size)))
    store.set_hash(cell, half_margin)
    return store


def _brute_pairs(store, clearance, moved_rows=None, *, ignore_static=False, hybrid=False):
    n = len(store)
    pairs = set()
    for a, b in itertools.combinations(range(n), 2):
        if moved_rows is not None and a not in moved_rows and b not in moved_rows:
            continue
        if not (store.registered(np.array([a]))[0] and store.registered(np.array([b]))[0]):
            continue
        if ignore_static and (store.mode[a] == MODE_STATIC or store.mode[b] == MODE_STATIC):
            continue
        m = list(clearance)
        if store.mode[a] == MODE_NORMAL_2D or store.mode[b] == MODE_NORMAL_2D:
            m[2] = 0.0
        if hybrid and (store.physics[a] or store.physics[b]):
            m = [0.0, 0.0, 0.0]
        if all(
            store.aabb_max[a, d] + m[d] >= store.aabb_min[b, d] and store.aabb_min[a, d] - m[d] <= store.aabb_max[b, d]
            for d in range(3)
        ):
            pairs.add((int(store.object_id[a]), int(store.object_id[b])))
    return pairs


def _found(store, rows, clearance, **kw):
    ids_a, ids_b = store.candidate_pairs(np.asarray(rows, dtype=np.int64), clearance, **kw)
    return {(min(a, b), max(a, b)) for a, b in zip(ids_a.tolist(), ids_b.tolist())}


# --- AABBs from poses -----------------------------------------------------


@pytest.fixture
def sim():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, monitor=False, log_level="warning"))
    yield sim
    p.disconnect(sim.client)


def _shape(kind):
    if kind == "sphere":
        return ShapeParams(shape_type="sphere", radius=0.17)
    if kind == "cylinder":
        return ShapeParams(shape_type="cylinder", radius=0.2, height=0.5)
    if kind == "mesh":
        return ShapeParams(shape_type="mesh", mesh_path="pybullet_fleet/mesh/11pallet.obj")
    return ShapeParams(shape_type="box", half_extents=[0.1, 0.25, 0.4])


@pytest.mark.parametrize("kind", ["box", "cylinder", "sphere", "mesh"])
def test_aabb_from_pose_is_pybullets(sim, kind):
    shape = _shape(kind)
    obj = SimObject.from_mesh(
        visual_shape=shape,
        collision_shape=shape,
        pose=Pose.from_xyz(0, 0, 0),
        sim_core=sim,
        collision_mode=CollisionMode.NORMAL_3D,
    )
    p.performCollisionDetection(physicsClientId=sim.client)
    local = p.getAABB(obj.body_id, physicsClientId=sim.client)
    store = CollisionStore()
    row = store.add(obj.object_id, mode=MODE_NORMAL_3D, physics=False, pickable=False, pybullet_aabb=False)
    store.set_local_box(row, local[0], local[1], invariant=kind == "sphere")
    rng = random.Random(kind)
    for _ in range(12):
        pose = Pose.from_euler(*(rng.uniform(-3, 3) for _ in range(3)), *(rng.uniform(-3, 3) for _ in range(3)))
        obj.set_pose(pose)
        p.performCollisionDetection(physicsClientId=sim.client)
        expected = p.getAABB(obj.body_id, physicsClientId=sim.client)
        store.set_aabbs_from_poses(np.array([row]), np.array([pose.position]), np.array([pose.orientation]))
        got = store.aabb(row)
        assert got[0] == pytest.approx(expected[0], abs=1e-9)
        assert got[1] == pytest.approx(expected[1], abs=1e-9)


def test_the_one_row_path_equals_the_array_path():
    rng = random.Random(5)
    store = CollisionStore()
    for k, invariant in enumerate((False, True)):
        row = store.add(k, mode=MODE_NORMAL_3D, physics=False, pickable=False, pybullet_aabb=False)
        store.set_local_box(row, (-0.1, -0.3, 0.05), (0.2, 0.1, 0.45), invariant=invariant)
    for _ in range(20):
        q = np.array([rng.gauss(0, 1) for _ in range(4)])
        q /= np.linalg.norm(q)
        pos = [rng.uniform(-5, 5) for _ in range(3)]
        for row in (0, 1):
            store.set_aabbs_from_poses(np.array([row]), np.array([pos]), q[None, :])
            expected = store.aabb(row)
            store.set_aabb_from_pose(row, pos, q.tolist())
            got = store.aabb(row)
            assert got[0] == pytest.approx(expected[0], abs=1e-12)
            assert got[1] == pytest.approx(expected[1], abs=1e-12)


def test_rotation_matrices_match_pybullet():
    rng = random.Random(1)
    for _ in range(20):
        q = np.array([rng.gauss(0, 1) for _ in range(4)])
        q /= np.linalg.norm(q)
        expected = np.array(p.getMatrixFromQuaternion(q.tolist())).reshape(3, 3)
        assert rotation_matrices(q[None, :])[0] == pytest.approx(expected, abs=1e-12)


# --- candidate pairs ------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("clearance", [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.3, 0.02, 0.0), (0.0, 0.0, 0.7)])
def test_every_pair_within_the_clearance_is_a_candidate_and_no_other(seed, clearance):
    rng = random.Random(seed)
    store = _random_store(rng, 60, cell=rng.choice([0.5, 1.0, 2.0]), half_margin=[c / 2 for c in clearance])
    rows = np.arange(len(store))
    for ignore_static, hybrid in itertools.product((False, True), (False, True)):
        assert _found(store, rows, clearance, ignore_static=ignore_static, hybrid=hybrid) == _brute_pairs(
            store, clearance, ignore_static=ignore_static, hybrid=hybrid
        )


@pytest.mark.parametrize("seed", range(6))
def test_only_pairs_of_moved_rows_are_searched(seed):
    rng = random.Random(seed)
    clearance = (0.1, 0.1, 0.0)
    store = _random_store(rng, 50, half_margin=(0.05, 0.05, 0.0))
    moved = sorted(rng.sample(range(50), 12))
    for ignore_static in (False, True):
        assert _found(store, moved, clearance, ignore_static=ignore_static, hybrid=False) == _brute_pairs(
            store, clearance, set(moved), ignore_static=ignore_static
        )


def test_no_rows_or_no_moved_rows_give_no_pairs():
    store = CollisionStore()
    store.set_hash(1.0, (0, 0, 0))
    a, b = store.candidate_pairs(np.empty(0, np.int64), (0, 0, 0), ignore_static=False, hybrid=False)
    assert len(a) == 0 and len(b) == 0
    row = store.add(7, mode=MODE_NORMAL_3D, physics=False, pickable=False, pybullet_aabb=False)
    store.set_aabb(row, ((0, 0, 0), (1, 1, 1)))
    a, b = store.candidate_pairs(np.array([row]), (0, 0, 0), ignore_static=False, hybrid=False)
    assert len(a) == 0


# --- the hash follows the rows ---------------------------------------------


def _fresh_contents(store):
    """The hash contents rebuilt from scratch, independent of the incremental sync."""
    rebuilt = CollisionStore()
    for row in range(len(store)):
        r = rebuilt.add(
            int(store.object_id[row]),
            mode=int(store.mode[row]),
            physics=bool(store.physics[row]),
            pickable=False,
            pybullet_aabb=False,
        )
        if store.has_aabb[row]:
            rebuilt.set_aabb(r, store.aabb(row))
    rebuilt.set_hash(store.cell_size, store.half_margin)
    return rebuilt.hash_contents()


@pytest.mark.parametrize("seed", range(5))
def test_incremental_sync_equals_a_rebuild(seed):
    rng = random.Random(seed)
    store = _random_store(rng, 40, cell=0.5, half_margin=(0.01, 0.01, 0.0))
    store.sync_hash()
    assert store.hash_contents() == _fresh_contents(store)
    for _ in range(60):
        action = rng.random()
        if action < 0.6:  # a few rows move
            for row in rng.sample(range(len(store)), rng.randint(1, 4)):
                lo = np.array([rng.uniform(-4, 4) for _ in range(3)])
                store.set_aabb(row, (tuple(lo), tuple(lo + rng.choice([0.05, 0.4, 1.3]))))
        elif action < 0.85:  # a mode change, registration included
            row = rng.randrange(len(store))
            store.mode[row] = rng.choice(MODES)
            store.hash_dirty[row] = True
        elif action < 0.95 and len(store) > 5:  # a removal
            store.remove(int(store.object_id[rng.randrange(len(store))]))
        else:  # an addition
            row = store.add(1000 + _, mode=rng.choice(MODES), physics=False, pickable=False, pybullet_aabb=False)
            store.set_aabb(row, ((0.1, 0.2, 0.3), (0.5, 0.6, 0.7)))
        store.sync_hash()
        assert store.hash_contents() == _fresh_contents(store)
        assert all(store.object_id[row] == object_id for object_id, row in store.row_of.items())


def test_removing_a_row_moves_the_last_row_into_its_place():
    store = CollisionStore(capacity=2)
    rows = [store.add(i, mode=MODE_NORMAL_3D, physics=False, pickable=False, pybullet_aabb=False) for i in range(5)]
    for i, row in enumerate(rows):
        store.set_aabb(row, ((i, i, i), (i + 0.5, i + 0.5, i + 0.5)))
    store.set_hash(1.0, (0, 0, 0))
    store.sync_hash()
    store.remove(1)
    assert len(store) == 4
    assert store.row_of == {0: 0, 4: 1, 2: 2, 3: 3}
    assert store.aabb(1)[0] == (4.0, 4.0, 4.0)
    assert store.hash_contents() == {(0, 0, 0): {0}, (2, 2, 2): {2}, (3, 3, 3): {3}, (4, 4, 4): {4}}


def test_cells_cover_the_enlarged_box_and_negative_coordinates():
    store = CollisionStore()
    store.set_hash(2.0, (0.1, 0.1, 0.1))
    row = store.add(1, mode=MODE_NORMAL_3D, physics=False, pickable=False, pybullet_aabb=False)
    store.set_aabb(row, ((-0.05, 0.2, 0.2), (1.95, 0.5, 0.5)))
    assert store.cells_of(row) == [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
    assert store.cells_of(row, margin=(0.0, 0.0, 0.0)) == [(-1, 0, 0), (0, 0, 0)]
    flat = store.add(2, mode=MODE_NORMAL_2D, physics=False, pickable=False, pybullet_aabb=False)
    store.set_aabb(flat, ((0.5, 0.5, 1.95), (0.6, 0.6, 1.99)))
    assert store.cells_of(flat) == [(0, 0, 0)]  # not enlarged in z
    store.mode[flat] = MODE_NORMAL_3D
    assert store.cells_of(flat) == [(0, 0, 0), (0, 0, 1)]


def test_query_overlapping_reports_rows_with_an_aabb_in_the_box():
    store = CollisionStore()
    a = store.add(1, mode=MODE_NORMAL_3D, physics=False, pickable=False, pybullet_aabb=False)
    b = store.add(2, mode=MODE_DISABLED, physics=False, pickable=False, pybullet_aabb=False)
    store.add(3, mode=MODE_NORMAL_3D, physics=False, pickable=False, pybullet_aabb=False)  # no AABB yet
    store.set_aabb(a, ((0, 0, 0), (1, 1, 1)))
    store.set_aabb(b, ((5, 5, 5), (6, 6, 6)))
    assert store.query_overlapping((0.5, 0.5, 0.5), (5.5, 5.5, 5.5)).tolist() == [a, b]
    assert store.query_overlapping((2, 2, 2), (3, 3, 3)).tolist() == []
    assert store.median_extent() == 1.0


def test_capacity_grows_and_keeps_the_columns():
    store = CollisionStore(capacity=1)
    for i in range(10):
        row = store.add(i, mode=MODE_STATIC, physics=bool(i % 2), pickable=False, pybullet_aabb=False)
        store.set_aabb(row, ((i, 0, 0), (i + 1, 1, 1)))
    assert store.capacity >= 10
    assert [bool(x) for x in store.physics[:10]] == [bool(i % 2) for i in range(10)]
    assert store.aabb(9)[0] == (9.0, 0.0, 0.0)
