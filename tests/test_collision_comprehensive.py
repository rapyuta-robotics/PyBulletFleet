"""
Comprehensive Collision Detection Tests

Tests all collision detection modes, methods, and edge cases to ensure
robust and correct collision detection behavior.

Test Categories:
1. Basic Collision Detection (mode combinations)
2. CollisionDetectionMethod verification
3. 2D vs 3D neighbor search
4. STATIC object optimization
5. DISABLED mode
6. Multi-cell registration
7. Movement detection
8. Runtime mode changes
9. Edge cases
10. Integration tests

Author: Auto-generated from COLLISION_TEST_DESIGN.md
Date: 2026-01-21
"""

import logging
import itertools
from typing import List, Tuple, Optional

import pybullet as p
import pytest

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.sim_object import SimObject, ShapeParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.types import CollisionMode, CollisionDetectionMethod, SpatialHashCellSizeMode

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sim_core_kinematics():
    """Simulation with physics=False, CLOSEST_POINTS (recommended default)"""
    params = SimulationParams(
        gui=False,
        physics=False,
        monitor=False,
        collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
        collision_margin=0.02,  # 2cm safety margin
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        ignore_static_collision=False,  # Enable for testing
        log_level="warning",
    )
    sim_core = MultiRobotSimulationCore(params)
    sim_core.set_collision_spatial_hash_cell_size_mode()
    yield sim_core
    p.disconnect(sim_core.client)


@pytest.fixture
def sim_core_physics():
    """Simulation with physics=True, CONTACT_POINTS"""
    params = SimulationParams(
        gui=False,
        physics=True,
        monitor=False,
        collision_detection_method=CollisionDetectionMethod.CONTACT_POINTS,
        collision_margin=0.02,
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        ignore_static_collision=False,
        log_level="warning",
    )
    sim_core = MultiRobotSimulationCore(params)
    sim_core.set_collision_spatial_hash_cell_size_mode()
    yield sim_core
    p.disconnect(sim_core.client)


@pytest.fixture
def sim_core_hybrid():
    """Simulation with physics=True, HYBRID method"""
    params = SimulationParams(
        gui=False,
        physics=True,
        monitor=False,
        collision_detection_method=CollisionDetectionMethod.HYBRID,
        collision_margin=0.05,
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        ignore_static_collision=False,
        log_level="warning",
    )
    sim_core = MultiRobotSimulationCore(params)
    sim_core.set_collision_spatial_hash_cell_size_mode()
    yield sim_core
    p.disconnect(sim_core.client)


# ============================================================================
# Helper Functions
# ============================================================================


def create_test_box(
    sim_core: MultiRobotSimulationCore,
    position: List[float],
    size: float = 0.5,
    collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
    mass: float = 0.0,
    color: Optional[List[float]] = None,
) -> SimObject:
    """
    Create a simple box object for testing.

    Args:
        sim_core: Simulation core instance
        position: [x, y, z] position
        size: Half-extent of the box (box is 2*size in each dimension)
        collision_mode: Collision detection mode
        mass: Object mass (0 = kinematic)
        color: RGBA color (default: random)

    Returns:
        SimObject instance
    """
    if color is None:
        color = [0.6, 0.6, 0.6, 1.0]

    # Create box using SimObject.from_mesh() with ShapeParams
    sim_obj = SimObject.from_mesh(
        visual_shape=ShapeParams(
            shape_type="box",
            half_extents=[size, size, size],
            rgba_color=color,
        ),
        collision_shape=ShapeParams(
            shape_type="box",
            half_extents=[size, size, size],
        ),
        pose=Pose.from_xyz(position[0], position[1], position[2]),
        mass=mass,
        sim_core=sim_core,
        collision_mode=collision_mode,
    )

    return sim_obj


def get_collision_pairs(sim_core: MultiRobotSimulationCore) -> List[Tuple[int, int]]:
    """
    Get current collision pairs.

    Args:
        sim_core: Simulation core instance

    Returns:
        List of (body_id1, body_id2) tuples
    """
    # Trigger collision detection
    # check_collisions() returns (pairs, timings) tuple
    collision_pairs, _ = sim_core.check_collisions()

    return collision_pairs


def assert_collision_detected(
    sim_core: MultiRobotSimulationCore, obj1: SimObject, obj2: SimObject, should_collide: bool = True, message: str = ""
):
    """
    Assert that collision between two objects is detected (or not).

    Args:
        sim_core: Simulation core instance
        obj1: First object
        obj2: Second object
        should_collide: True if collision should be detected
        message: Additional error message
    """
    pairs = get_collision_pairs(sim_core)
    expected_pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))

    collision_detected = expected_pair in pairs

    if should_collide:
        assert collision_detected, (
            f"Collision NOT detected between obj{obj1.object_id} (body {obj1.body_id}) "
            f"and obj{obj2.object_id} (body {obj2.body_id}). "
            f"Expected pair: {expected_pair}, Found pairs: {pairs}. {message}"
        )
    else:
        assert not collision_detected, (
            f"Collision INCORRECTLY detected between obj{obj1.object_id} (body {obj1.body_id}) "
            f"and obj{obj2.object_id} (body {obj2.body_id}). "
            f"Unexpected pair: {expected_pair}, Found pairs: {pairs}. {message}"
        )


def get_object_cells(sim_core: MultiRobotSimulationCore, sim_obj: SimObject) -> List[Tuple[int, int, int]]:
    """
    Get the spatial grid cells an object is registered to.

    Args:
        sim_core: Simulation core instance
        sim_obj: Object to check

    Returns:
        List of (x, y, z) cell tuples
    """
    return sim_core._cached_object_to_cell.get(sim_obj.object_id, [])


# ============================================================================
# Category 1: Basic Collision Detection
# ============================================================================

# Test data generation for comprehensive collision mode testing
ALL_MODES = [
    CollisionMode.NORMAL_3D,
    CollisionMode.NORMAL_2D,
    CollisionMode.STATIC,
    CollisionMode.DISABLED,
]

DISTANCES = [
    (0.5, "overlapping"),  # Objects overlap (0.5 distance < 1.0 total size)
    (5.0, "separated"),  # Objects separated (5.0 distance > 1.0 total size)
]

# Special cases where behavior differs from the default rule
# Key: (mode1, mode2, distance) -> expected collision result
SPECIAL_CASES = {
    # Example: If there are any special collision rules, add them here
    # (CollisionMode.STATIC, CollisionMode.STATIC, 0.5): False,
}


def expected_collision(mode1: CollisionMode, mode2: CollisionMode, distance: float) -> bool:
    """
    Determine expected collision behavior based on collision modes and distance.

    Rules:
    1. Special cases take highest priority (if defined in SPECIAL_CASES)
    2. DISABLED mode: Never collides (regardless of distance or other mode)
    3. All other modes: Collide when overlapping (distance=0.5), not when separated (distance=5.0)

    Args:
        mode1: First object's collision mode
        mode2: Second object's collision mode
        distance: Distance between objects

    Returns:
        True if collision should be detected, False otherwise
    """
    # Check special cases first
    key = (mode1, mode2, distance)
    if key in SPECIAL_CASES:
        return SPECIAL_CASES[key]

    # DISABLED mode: Never collides
    if mode1 == CollisionMode.DISABLED or mode2 == CollisionMode.DISABLED:
        return False

    # Default rule: Collision depends on distance
    # Objects with size=0.5 have total extent of 1.0m
    # distance=0.5 → overlapping → should collide
    # distance=5.0 → separated → should NOT collide
    return distance == 0.5


def case_id(mode1: CollisionMode, mode2: CollisionMode, distance: float, dist_label: str) -> str:
    """Generate a readable test case ID."""
    return f"{mode1.name}-{mode2.name} {dist_label} (d={distance})"


def generate_collision_test_cases():
    """
    Generate all combinations of collision modes and distances.

    Yields:
        pytest.param with (mode1, mode2, distance, should_collide) and descriptive ID
    """
    for mode1, mode2 in itertools.product(ALL_MODES, repeat=2):
        for distance, dist_label in DISTANCES:
            should_collide = expected_collision(mode1, mode2, distance)
            yield pytest.param(
                mode1,
                mode2,
                distance,
                should_collide,
                id=case_id(mode1, mode2, distance, dist_label),
            )


class TestBasicCollisionDetection:
    """Test basic collision detection for all mode combinations"""

    @pytest.mark.parametrize("mode1,mode2,distance,should_collide", list(generate_collision_test_cases()))
    def test_collision_mode_combinations(self, sim_core_kinematics, mode1, mode2, distance, should_collide):
        """
        Test all collision mode combinations with overlapping and separated objects.

        Auto-generated test covering all combinations of:
        - Collision modes: NORMAL_3D, NORMAL_2D, STATIC, DISABLED
        - Distances: overlapping (0.5m), separated (5.0m)

        Total: 4 modes × 4 modes × 2 distances = 32 test cases

        Expected behavior:
        - DISABLED mode: Never collides (regardless of distance)
        - Other modes: Collide when overlapping, not when separated
        """
        obj1 = create_test_box(sim_core_kinematics, [0, 0, 0], size=0.5, collision_mode=mode1)
        obj2 = create_test_box(sim_core_kinematics, [distance, 0, 0], size=0.5, collision_mode=mode2)

        assert_collision_detected(
            sim_core_kinematics,
            obj1,
            obj2,
            should_collide=should_collide,
            message=f"mode1={mode1.name}, mode2={mode2.name}, distance={distance}",
        )


# ============================================================================
# Category 2: CollisionDetectionMethod Verification
# ============================================================================


class TestCollisionDetectionMethod:
    """Test different collision detection methods"""

    def test_closest_points_detects_near_miss(self, sim_core_kinematics):
        """TC101: CLOSEST_POINTS detects near-miss within margin"""
        margin = sim_core_kinematics.params.collision_margin  # 0.02m

        obj1 = create_test_box(sim_core_kinematics, [0, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        # Position just within margin (0.5 + 0.5 + 0.01 = 1.01, margin = 0.02)
        obj2 = create_test_box(sim_core_kinematics, [1.01, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)

        # Should detect because distance (0.01) < margin (0.02)
        assert_collision_detected(
            sim_core_kinematics,
            obj1,
            obj2,
            should_collide=True,
            message=f"CLOSEST_POINTS should detect near-miss within margin ({margin}m)",
        )

    def test_contact_points_requires_step_simulation(self, sim_core_physics):
        """TC102: CONTACT_POINTS detects overlapping objects after step_once()

        CONTACT_POINTS requires stepSimulation() to generate contact manifold.
        step_once() internally runs stepSimulation + check_collisions, so
        a single call is sufficient.
        """
        obj1 = create_test_box(sim_core_physics, [0, 0, 0.75], size=0.5, collision_mode=CollisionMode.NORMAL_3D, mass=1.0)
        obj2 = create_test_box(sim_core_physics, [0, 0, 0.25], size=0.5, collision_mode=CollisionMode.NORMAL_3D, mass=0.0)

        sim_core_physics.step_once()

        pairs = sim_core_physics.get_active_collision_pairs()
        expected = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert expected in pairs, f"CONTACT_POINTS should detect overlapping objects. pairs={pairs}"


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("mode", [CollisionMode.NORMAL_2D, CollisionMode.NORMAL_3D])
@pytest.mark.parametrize("method", [CollisionDetectionMethod.CLOSEST_POINTS, CollisionDetectionMethod.HYBRID])
def test_margin_pair_across_nonadjacent_cells(axis, mode, method):
    """Enlarged AABBs must share a cell even when object centers are far apart."""
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=method,
            collision_margin=0.25,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=0.05,
            ignore_static_collision=False,
        )
    )
    try:
        first = create_test_box(sim, [0, 0, 0], size=0.05, collision_mode=mode)
        position = [0.0, 0.0, 0.0]
        position[axis] = 0.30  # Surface gap 0.20 m, inside the 0.25 m margin.
        second = create_test_box(sim, position, size=0.05, collision_mode=mode)
        sim._moved_this_step = {first.object_id, second.object_id}
        pair = (first.object_id, second.object_id)
        candidates, _ = sim.filter_aabb_pairs()
        expected = axis != 2 or mode == CollisionMode.NORMAL_3D
        assert candidates.count(pair) == int(expected)
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == ([pair] if expected else [])
    finally:
        p.disconnect(sim.client)


def test_normal_2d_pair_requires_actual_z_overlap(sim_core_kinematics):
    first = create_test_box(sim_core_kinematics, [0, 0, 0], size=0.05, collision_mode=CollisionMode.NORMAL_2D)
    second = create_test_box(sim_core_kinematics, [0, 0, 0.11], size=0.05, collision_mode=CollisionMode.NORMAL_3D)
    sim_core_kinematics._moved_this_step = {first.object_id, second.object_id}
    candidates, _ = sim_core_kinematics.filter_aabb_pairs()
    assert candidates == []


def test_collision_observation_tracks_signed_distance_and_lifecycle(sim_core_kinematics):
    sim = sim_core_kinematics
    assert sim.get_collision_observation() is None
    first = create_test_box(sim, [0, 0, 0], size=0.05)
    second = create_test_box(sim, [0.11, 0, 0], size=0.05)
    sim.check_collisions()
    observation = sim.get_collision_observation()
    assert observation is not None
    assert observation.step == 0
    assert observation.margin == 0.02
    assert len(observation.pairs) == 1
    pair = observation.pairs[0]
    assert pair.object_ids == (first.object_id, second.object_id)
    assert pair.detection_method == CollisionDetectionMethod.CLOSEST_POINTS
    assert pair.signed_distance == pytest.approx(0.01, abs=1e-4)
    assert pair.sample_step == 0

    second.set_pose(Pose.from_xyz(0.09, 0, 0))
    sim.check_collisions()
    overlap = sim.get_collision_observation()
    assert overlap is not None
    assert overlap.pairs[0].signed_distance <= 0

    sim.set_collision_margin(0)
    second.set_pose(Pose.from_xyz(0.11, 0, 0))
    sim.check_collisions()
    zero_margin = sim.get_collision_observation()
    assert zero_margin is not None and zero_margin.margin == 0
    assert zero_margin.pairs == ()

    second.set_pose(Pose.from_xyz(0.30, 0, 0))
    sim.check_collisions()
    cleared = sim.get_collision_observation()
    assert cleared is not None and cleared.pairs == ()
    assert sim.get_active_collision_pairs() == []
    assert sim._active_pairs_by_object == {}


def test_missing_detection_method_rejected_before_collision_state_changes(sim_core_kinematics):
    sim = sim_core_kinematics
    sim._moved_this_step.add(42)
    sim.params.collision_detection_method = None
    with pytest.raises(ValueError, match="collision_detection_method must be set"):
        sim.check_collisions()
    assert sim._moved_this_step == {42}
    assert sim.get_collision_observation() is None


def test_collision_observation_keeps_check_and_pair_sample_times(sim_core_kinematics):
    sim = sim_core_kinematics
    create_test_box(sim, [0, 0, 0], size=0.05)
    create_test_box(sim, [0.09, 0, 0], size=0.05)
    seen_in_post_step = []
    sim.events.on(SimEvents.POST_STEP, lambda **_: seen_in_post_step.append(sim.get_collision_observation()))
    sim.step_once()
    initial = sim.get_collision_observation()
    assert initial is not None and initial.step == sim.step_count == 1
    assert initial.sim_time == pytest.approx(sim.params.timestep)
    assert seen_in_post_step[0] == initial

    sim.set_collision_check_frequency(0)
    sim.step_once()
    assert sim.get_collision_observation() == initial
    sim.check_collisions()
    manual = sim.get_collision_observation()
    assert manual is not None and manual.step == 2
    assert manual.sim_time == pytest.approx(2 * sim.params.timestep)
    assert manual.pairs[0].sample_step == 1

    sim.set_collision_check_frequency(None)
    sim.step_once()
    later = sim.get_collision_observation()
    assert later is not None and later.step == 3
    assert later.pairs[0].sample_step == 1  # Stationary pair was not rechecked.


def test_manual_collision_check_in_post_step_uses_completed_step_time(sim_core_kinematics):
    sim = sim_core_kinematics
    create_test_box(sim, [0, 0, 0], size=0.05)
    create_test_box(sim, [0.09, 0, 0], size=0.05)
    observations = []

    def check_in_post_step(**_):
        sim.check_collisions()
        observations.append(sim.get_collision_observation())

    sim.events.on(SimEvents.POST_STEP, check_in_post_step)
    sim.step_once()

    assert observations[0] is not None
    assert observations[0].step == sim.step_count == 1
    assert observations[0].sim_time == pytest.approx(sim.params.timestep)


@pytest.mark.parametrize("method", [CollisionDetectionMethod.CLOSEST_POINTS, CollisionDetectionMethod.HYBRID])
def test_collision_observation_hybrid_kinematic_branch(method):
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, monitor=False, collision_detection_method=method, collision_margin=0.02)
    )
    try:
        create_test_box(sim, [0, 0, 0], size=0.05)
        create_test_box(sim, [0.11, 0, 0], size=0.05)
        sim.check_collisions()
        observation = sim.get_collision_observation()
        assert observation is not None and len(observation.pairs) == 1
        assert observation.pairs[0].detection_method == CollisionDetectionMethod.CLOSEST_POINTS
        assert observation.pairs[0].signed_distance is not None
    finally:
        p.disconnect(sim.client)


def test_collision_observation_contact_mode_has_no_claimed_distance(sim_core_physics):
    sim = sim_core_physics
    create_test_box(sim, [0, 0, 0], size=0.05, mass=1.0)
    create_test_box(sim, [0.09, 0, 0], size=0.05, mass=1.0)
    for _ in range(3):
        sim.step_once()
    observation = sim.get_collision_observation()
    assert observation is not None
    assert observation.configured_method == CollisionDetectionMethod.CONTACT_POINTS
    assert observation.pairs
    assert all(
        pair.detection_method == CollisionDetectionMethod.CONTACT_POINTS and pair.signed_distance is None
        for pair in observation.pairs
    )


def test_collision_observation_invalidated_when_object_is_disabled(sim_core_kinematics):
    sim = sim_core_kinematics
    create_test_box(sim, [0, 0, 0], size=0.05)
    second = create_test_box(sim, [0.09, 0, 0], size=0.05)
    sim.check_collisions()
    observation = sim.get_collision_observation()
    assert observation is not None and observation.pairs
    second.set_collision_mode(CollisionMode.DISABLED)
    assert sim.get_collision_observation() is None


def test_reset_clears_active_pairs_and_collision_observation(sim_core_kinematics):
    sim = sim_core_kinematics
    create_test_box(sim, [0, 0, 0], size=0.05)
    create_test_box(sim, [0.09, 0, 0], size=0.05)
    sim.check_collisions()
    observation = sim.get_collision_observation()
    assert observation is not None and observation.pairs

    sim.reset()

    assert sim.get_active_collision_pairs() == []
    assert sim.get_collision_observation() is None


def test_contact_points_margin_update_does_not_rebuild(sim_core_physics, monkeypatch):
    def unexpected_rebuild():
        pytest.fail("CONTACT_POINTS does not use collision_margin")

    monkeypatch.setattr(sim_core_physics, "_rebuild_spatial_grid", unexpected_rebuild)
    sim_core_physics.set_collision_margin(0.25)
    assert sim_core_physics.params.collision_margin == 0.25
    assert not sim_core_physics._recheck_stationary_pairs


def test_contact_points_candidates_ignore_positive_margin(sim_core_physics):
    first = create_test_box(sim_core_physics, [0, 0, 0], size=0.05)
    second = create_test_box(sim_core_physics, [0.30, 0, 0], size=0.05)
    sim_core_physics._moved_this_step = {first.object_id, second.object_id}
    candidates, _ = sim_core_physics.filter_aabb_pairs()
    assert candidates == []


def test_set_collision_margin_rechecks_stationary_pair():
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            collision_margin=0.0,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=0.05,
            ignore_static_collision=False,
        )
    )
    try:
        first = create_test_box(sim, [0, 0, 0], size=0.05, collision_mode=CollisionMode.STATIC)
        second = create_test_box(sim, [0.30, 0, 0], size=0.05, collision_mode=CollisionMode.STATIC)
        sim._moved_this_step = {first.object_id, second.object_id}
        sim.check_collisions()
        assert not sim.get_active_collision_pairs()

        sim.set_collision_margin(0.25)
        sim.check_collisions()
        assert sim.get_active_collision_pairs() == [(first.object_id, second.object_id)]

        sim.set_collision_margin(0.0)
        sim.check_collisions()
        assert not sim.get_active_collision_pairs()
    finally:
        p.disconnect(sim.client)


def test_registration_covers_the_aabb_enlarged_by_half_the_margin():
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_margin=0.02,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=1.0,
        )
    )
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        inside = create_test_box(sim, [0.5, 0.5, 0.5], size=0.05)
        edge = create_test_box(sim, [1.0, 0.5, 0.5], size=0.05)
        near = create_test_box(sim, [0.945, 0.5, 0.5], size=0.05)  # 0.005 from x=1, within margin/2
        large = create_test_box(sim, [1.0, 0.5, 0.5], size=1.5)
        flat = create_test_box(sim, [0.5, 0.5, 0.945], size=0.05, collision_mode=CollisionMode.NORMAL_2D)
        cells = sim._cached_object_to_cell
        assert cells[inside.object_id] == [(0, 0, 0)]
        # Every cell the AABB, enlarged by margin/2, touches.
        assert cells[edge.object_id] == [(0, 0, 0), (1, 0, 0)]
        assert cells[near.object_id] == [(0, 0, 0), (1, 0, 0)]
        assert len(cells[large.object_id]) > 1
        # A NORMAL_2D body's z extent is not enlarged.
        assert cells[flat.object_id] == [(0, 0, 0)]
    finally:
        p.disconnect(sim.client)


# ============================================================================
# Category 6: Multi-cell Registration
# ============================================================================


class TestMultiCellRegistration:
    """Test spatial hash multi-cell registration for large objects."""

    @pytest.fixture
    def sim_core_multicell(self):
        """Exercise the legacy size threshold without margin expansion."""
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            collision_margin=0.0,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=2.0,
            multi_cell_threshold=1.5,
            ignore_static_collision=False,
            log_level="warning",
        )
        sim_core = MultiRobotSimulationCore(params)
        sim_core.set_collision_spatial_hash_cell_size_mode()
        yield sim_core
        p.disconnect(sim_core.client)

    def test_small_object_single_cell(self, sim_core_multicell):
        """A small object inside one cell is registered in that cell only."""
        obj = create_test_box(
            sim_core_multicell,
            [1, 1, 1],
            size=0.25,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        cells = sim_core_multicell._cached_object_to_cell.get(obj.object_id, [])
        assert len(cells) == 1, f"Small object should use 1 cell, got {len(cells)}"

    def test_small_object_on_a_cell_boundary_is_in_both_cells(self, sim_core_multicell):
        """Registration follows the AABB, not its centre: straddling x=0 means two cells."""
        obj = create_test_box(sim_core_multicell, [0, 1, 1], size=0.25, collision_mode=CollisionMode.NORMAL_3D)
        assert sorted(sim_core_multicell._cached_object_to_cell[obj.object_id]) == [(-1, 0, 0), (0, 0, 0)]

    def test_get_overlapping_cells(self, sim_core_multicell):
        """_get_overlapping_cells returns cells covering object AABB."""
        large = create_test_box(
            sim_core_multicell,
            [0, 0, 0],
            size=2.5,
            collision_mode=CollisionMode.STATIC,
        )
        cells = sim_core_multicell._get_overlapping_cells(large.object_id)
        # 5m object with 2m cells → should span at least 4 cells
        assert len(cells) >= 4, f"Expected ≥4 overlapping cells, got {len(cells)}"

    def test_large_wall_collision_with_small_robot(self, sim_core_multicell):
        """Small robot collides with large wall via multi-cell lookup."""
        wall = create_test_box(
            sim_core_multicell,
            [0, 0, 0],
            size=2.5,
            collision_mode=CollisionMode.STATIC,
        )
        robot = create_test_box(
            sim_core_multicell,
            [2.3, 2.3, 0],
            size=0.25,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        # Mark robot as moved so check_collisions considers it
        sim_core_multicell._moved_this_step.add(robot.object_id)
        assert_collision_detected(
            sim_core_multicell,
            wall,
            robot,
            should_collide=True,
            message="Small robot near large wall edge should collide",
        )

    def test_threshold_is_accepted_and_ignored(self):
        """multi_cell_threshold no longer changes registration (kept for compatibility)."""
        registered = []
        for threshold in (1.2, 10.0):
            params = SimulationParams(
                gui=False,
                physics=False,
                monitor=False,
                collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
                spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
                spatial_hash_cell_size=2.0,
                multi_cell_threshold=threshold,
                ignore_static_collision=False,
                log_level="warning",
            )
            sim_core = MultiRobotSimulationCore(params)
            sim_core.set_collision_spatial_hash_cell_size_mode()
            try:
                obj = create_test_box(sim_core, [0, 0, 0], size=1.25, collision_mode=CollisionMode.NORMAL_3D)
                registered.append(sorted(sim_core._cached_object_to_cell[obj.object_id]))
            finally:
                p.disconnect(sim_core.client)
        assert registered[0] == registered[1] and len(registered[0]) > 1

    def test_coord_to_cell_positive(self, sim_core_multicell):
        """_coord_to_cell converts positive coordinates to correct cell index."""
        # cell_size=2.0  →  coord 3.5 → floor(3.5/2.0) = floor(1.75) = 1
        assert sim_core_multicell._coord_to_cell(3.5) == 1
        assert sim_core_multicell._coord_to_cell(0.0) == 0
        assert sim_core_multicell._coord_to_cell(1.99) == 0
        assert sim_core_multicell._coord_to_cell(2.0) == 1

    def test_coord_to_cell_negative(self, sim_core_multicell):
        """_coord_to_cell converts negative coordinates (floor toward -inf)."""
        # cell_size=2.0  →  coord -0.1 → floor(-0.05) = -1
        assert sim_core_multicell._coord_to_cell(-0.1) == -1
        assert sim_core_multicell._coord_to_cell(-2.0) == -1
        assert sim_core_multicell._coord_to_cell(-2.1) == -2

    def test_coord_to_cell_used_in_get_overlapping_cells(self, sim_core_multicell):
        """_get_overlapping_cells results are consistent with _coord_to_cell."""
        large = create_test_box(
            sim_core_multicell,
            [0, 0, 0],
            size=2.5,
            collision_mode=CollisionMode.STATIC,
        )
        cells = sim_core_multicell._get_overlapping_cells(large.object_id)
        aabb = sim_core_multicell._cached_aabbs_dict[large.object_id]
        # Each cell coordinate should match _coord_to_cell range
        for cell in cells:
            for d in range(3):
                min_c = sim_core_multicell._coord_to_cell(aabb[0][d])
                max_c = sim_core_multicell._coord_to_cell(aabb[1][d])
                assert min_c <= cell[d] <= max_c

    def test_same_cell_update_skips_grid_operations(self, sim_core_multicell):
        """_update_object_spatial_grid skips discard/add when cell unchanged."""
        obj = create_test_box(
            sim_core_multicell,
            [1, 1, 1],
            size=0.25,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        oid = obj.object_id
        cells_before = sim_core_multicell._cached_object_to_cell[oid]
        assert len(cells_before) == 1
        cell_key = cells_before[0]

        # Capture the set object identity for this cell
        grid_set_before = sim_core_multicell._cached_spatial_grid[cell_key]
        set_id_before = id(grid_set_before)

        # Move slightly within the same cell (cell_size=2.0, move 0.1m)
        from pybullet_fleet.geometry import Pose

        obj.set_pose(Pose(position=[1.1, 1.0, 1.0]))

        # Call _update_object_spatial_grid (simulating what set_pose triggers)
        sim_core_multicell._update_object_spatial_grid(oid)

        # Cell should be unchanged
        cells_after = sim_core_multicell._cached_object_to_cell[oid]
        assert cells_after == cells_before, "Cell should not change for small movement"

        # The grid set should be the same object (not recreated via delete+setdefault)
        grid_set_after = sim_core_multicell._cached_spatial_grid[cell_key]
        assert id(grid_set_after) == set_id_before, "Grid set was recreated — same-cell optimization not applied"

    def test_discard_from_cells_removes_object(self, sim_core_multicell):
        """_discard_from_cells removes object from spatial grid and cleans empty cells."""
        obj = create_test_box(
            sim_core_multicell,
            [1, 1, 1],
            size=0.25,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        oid = obj.object_id
        cells = sim_core_multicell._cached_object_to_cell[oid]
        assert len(cells) == 1
        cell_key = cells[0]
        assert oid in sim_core_multicell._cached_spatial_grid[cell_key]

        # Call _discard_from_cells directly
        sim_core_multicell._discard_from_cells(oid, cells)

        # Object should be removed from the cell's set
        # If the cell had only this object, the cell key should be deleted
        if cell_key in sim_core_multicell._cached_spatial_grid:
            assert oid not in sim_core_multicell._cached_spatial_grid[cell_key]

    def test_discard_from_cells_preserves_other_objects(self, sim_core_multicell):
        """_discard_from_cells only removes the target object, not others in the same cell."""
        obj1 = create_test_box(
            sim_core_multicell,
            [0, 0, 0],
            size=0.25,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        obj2 = create_test_box(
            sim_core_multicell,
            [0.1, 0, 0],
            size=0.25,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        cells1 = sim_core_multicell._cached_object_to_cell[obj1.object_id]
        cells2 = sim_core_multicell._cached_object_to_cell[obj2.object_id]
        # Both should be in the same cell (cell_size=2.0, both near origin)
        assert cells1 == cells2, "Both objects should be in the same cell"
        cell_key = cells1[0]

        # Remove obj1 only
        sim_core_multicell._discard_from_cells(obj1.object_id, cells1)

        # obj2 should still be in the cell
        assert obj2.object_id in sim_core_multicell._cached_spatial_grid[cell_key]
        # Cell should NOT be deleted (obj2 still there)
        assert cell_key in sim_core_multicell._cached_spatial_grid

    def test_discard_from_cells_deletes_empty_cell(self, sim_core_multicell):
        """_discard_from_cells deletes cell key when the set becomes empty."""
        obj = create_test_box(
            sim_core_multicell,
            [100, 100, 0],  # Far away to be the only object in this cell
            size=0.25,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        oid = obj.object_id
        cells = sim_core_multicell._cached_object_to_cell[oid]
        cell_key = cells[0]
        # Ensure only this object is in the cell
        assert sim_core_multicell._cached_spatial_grid[cell_key] == {oid}

        sim_core_multicell._discard_from_cells(oid, cells)

        # Cell key should be completely removed
        assert cell_key not in sim_core_multicell._cached_spatial_grid


# ============================================================================
# Category 7: Movement Detection
# ============================================================================


class TestMovementDetection:
    """Test that moved objects are correctly re-evaluated for collisions."""

    def test_move_into_collision(self, sim_core_kinematics):
        """Object moved into collision range is detected."""
        obj1 = create_test_box(sim_core_kinematics, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        obj2 = create_test_box(sim_core_kinematics, [5, 0, 0], collision_mode=CollisionMode.NORMAL_3D)

        # Initially separated → no collision
        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=False)

        # Move obj2 next to obj1 (set_pose handles mark_moved, AABB, grid update)
        obj2.set_pose(Pose.from_xyz(0.5, 0, 0))

        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=True)

    def test_move_out_of_collision(self, sim_core_kinematics):
        """Object moved away from collision range is no longer detected."""
        obj1 = create_test_box(sim_core_kinematics, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        obj2 = create_test_box(sim_core_kinematics, [0.5, 0, 0], collision_mode=CollisionMode.NORMAL_3D)

        # Initially overlapping → collision
        sim_core_kinematics._moved_this_step.add(obj2.object_id)
        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=True)

        # Move obj2 far away (set_pose handles mark_moved, AABB, grid update)
        obj2.set_pose(Pose.from_xyz(10, 0, 0))

        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=False)


class TestCollisionPairsAccuracy:
    """Test that active_collision_pairs exactly matches expected pairs.

    Unlike assert_collision_detected which checks a single pair,
    these tests verify the complete set of collision pairs — ensuring
    no extra or missing pairs exist.
    """

    @staticmethod
    def _sorted_pairs(sim_core: MultiRobotSimulationCore):
        """Return active collision pairs as a sorted-tuple set for exact comparison."""
        return {(min(a, b), max(a, b)) for a, b in sim_core.get_active_collision_pairs()}

    @staticmethod
    def _pair(obj_a: SimObject, obj_b: SimObject) -> tuple:
        return (min(obj_a.object_id, obj_b.object_id), max(obj_a.object_id, obj_b.object_id))

    def test_three_object_chain_pairs(self, sim_core_kinematics):
        """A↔B↔C chain: only adjacent pairs collide, A↔C does not.

        Layout (size=0.5 → full extent 1.0m each):
          A at x=0, B at x=0.8, C at x=1.6
          A↔B overlap 0.2m (collide), B↔C overlap 0.2m (collide),
          A↔C gap=0.6m > margin 0.02m (no collide)
        """
        sc = sim_core_kinematics
        a = create_test_box(sc, [0, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        b = create_test_box(sc, [0.8, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        c = create_test_box(sc, [1.6, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)

        # Mark all as moved for initial detection
        for obj in [a, b, c]:
            sc._moved_this_step.add(obj.object_id)

        sc.check_collisions()
        pairs = self._sorted_pairs(sc)

        expected = {self._pair(a, b), self._pair(b, c)}
        assert pairs == expected, f"Expected chain pairs {expected}, got {pairs}"

    def test_all_separated_empty_pairs(self, sim_core_kinematics):
        """Three well-separated objects → pairs must be empty set."""
        sc = sim_core_kinematics
        a = create_test_box(sc, [0, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        b = create_test_box(sc, [10, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        c = create_test_box(sc, [20, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)

        for obj in [a, b, c]:
            sc._moved_this_step.add(obj.object_id)

        sc.check_collisions()
        pairs = self._sorted_pairs(sc)

        assert pairs == set(), f"All separated → expected empty, got {pairs}"

    def test_no_movement_pairs_unchanged(self, sim_core_kinematics):
        """Without movement, active_collision_pairs must not change.

        Verifies: check_collisions() with empty _moved_this_step
        preserves existing pairs and adds no new ones.
        """
        sc = sim_core_kinematics
        a = create_test_box(sc, [0, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        b = create_test_box(sc, [0.5, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        c = create_test_box(sc, [10, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)

        # Establish initial pairs: {(a, b)}
        for obj in [a, b, c]:
            sc._moved_this_step.add(obj.object_id)
        sc.check_collisions()
        pairs_before = self._sorted_pairs(sc)
        assert pairs_before == {self._pair(a, b)}, f"Pre-condition failed: {pairs_before}"

        # No movement → call check_collisions again
        sc.check_collisions()
        pairs_after = self._sorted_pairs(sc)

        assert pairs_after == pairs_before, f"Pairs changed without movement: before={pairs_before}, after={pairs_after}"

    def test_move_adds_and_removes_pairs(self, sim_core_kinematics):
        """Moving one object updates pairs: old pair removed, new pair added.

        Step 1: A↔B collide, C far → {(A,B)}
        Step 2: Move B away from A, next to C → {(B,C)}
        """
        sc = sim_core_kinematics
        a = create_test_box(sc, [0, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        b = create_test_box(sc, [0.5, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        c = create_test_box(sc, [10, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)

        for obj in [a, b, c]:
            sc._moved_this_step.add(obj.object_id)
        sc.check_collisions()

        assert self._sorted_pairs(sc) == {self._pair(a, b)}, "Step 1 failed"

        # Move B next to C (set_pose handles mark_moved, AABB, grid)
        b.set_pose(Pose.from_xyz(10.5, 0, 0))
        sc.check_collisions()

        pairs = self._sorted_pairs(sc)
        assert self._pair(a, b) not in pairs, "A↔B should be resolved after B moved away"
        assert self._pair(b, c) in pairs, "B↔C should collide after B moved next to C"
        assert pairs == {self._pair(b, c)}, f"Expected only {{(B,C)}}, got {pairs}"

    def test_progressive_pair_accumulation(self, sim_core_kinematics):
        """Pairs evolve as objects are moved together step by step.

        Step 0: A, B, C all separated       → {}
        Step 1: Move B next to A             → {(A,B)}
        Step 2: Move C next to B (and A)     → {(A,B), (A,C), (B,C)}
        Step 3: Move B far away              → {(A,C)}
        """
        sc = sim_core_kinematics
        a = create_test_box(sc, [0, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        b = create_test_box(sc, [10, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)
        c = create_test_box(sc, [20, 0, 0], size=0.5, collision_mode=CollisionMode.NORMAL_3D)

        for obj in [a, b, c]:
            sc._moved_this_step.add(obj.object_id)
        sc.check_collisions()

        # Step 0: all separated
        assert self._sorted_pairs(sc) == set(), "Step 0: all separated"

        # Step 1: Move B next to A
        b.set_pose(Pose.from_xyz(0.5, 0, 0))
        sc.check_collisions()
        assert self._sorted_pairs(sc) == {self._pair(a, b)}, "Step 1: A↔B only"

        # Step 2: Move C overlapping both A and B (at origin area)
        c.set_pose(Pose.from_xyz(0.25, 0, 0))
        sc.check_collisions()
        expected_all = {self._pair(a, b), self._pair(a, c), self._pair(b, c)}
        assert self._sorted_pairs(sc) == expected_all, "Step 2: all three collide"

        # Step 3: Move B far away — only A↔C remains
        b.set_pose(Pose.from_xyz(30, 0, 0))
        sc.check_collisions()
        assert self._sorted_pairs(sc) == {self._pair(a, c)}, "Step 3: only A↔C"


# ============================================================================
# Category 8: Runtime Mode Changes
# ============================================================================


class TestRuntimeModeChanges:
    """Test collision mode changes at runtime."""

    def test_remove_from_collision_system_clears_pairs(self, sim_core_kinematics):
        """_remove_object_from_collision_system also removes active collision pairs."""
        obj = create_test_box(sim_core_kinematics, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id

        # Seed both the active set and its reverse index.
        for pair in ((oid, 99), (50, oid), (50, 99)):
            sim_core_kinematics._add_active_collision_pair(pair)

        sim_core_kinematics._remove_object_from_collision_system(oid)

        # Pairs containing oid should be gone
        assert (oid, 99) not in sim_core_kinematics._active_collision_pairs
        assert (50, oid) not in sim_core_kinematics._active_collision_pairs
        # Unrelated pair survives
        assert (50, 99) in sim_core_kinematics._active_collision_pairs
        assert oid not in sim_core_kinematics._active_pairs_by_object
        assert sim_core_kinematics._active_pairs_by_object[50] == {(50, 99)}

    def test_remove_from_collision_system_no_op_pairs(self, sim_core_kinematics):
        """_remove_object_from_collision_system is no-op for pairs when id absent."""
        obj = create_test_box(sim_core_kinematics, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id
        sim_core_kinematics._add_active_collision_pair((50, 99))

        sim_core_kinematics._remove_object_from_collision_system(oid)

        assert sim_core_kinematics._active_collision_pairs == {(50, 99)}

    def test_same_mode_update_is_no_op(self, sim_core_kinematics):
        """_update_object_collision_mode with old_mode == new_mode does nothing."""
        obj = create_test_box(sim_core_kinematics, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id

        # Clear moved set to detect spurious additions
        sim_core_kinematics._moved_this_step.clear()

        # Call with same mode
        sim_core_kinematics._update_object_collision_mode(oid, CollisionMode.NORMAL_3D, CollisionMode.NORMAL_3D)

        # Should NOT add to _moved_this_step (early return, no side effects)
        assert oid not in sim_core_kinematics._moved_this_step, "Same-mode update should not add object to _moved_this_step"

    def test_disable_collision_at_runtime(self, sim_core_kinematics):
        """Disabling collision mode removes object from detection."""
        obj1 = create_test_box(sim_core_kinematics, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        obj2 = create_test_box(sim_core_kinematics, [0.5, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        sim_core_kinematics._moved_this_step.add(obj2.object_id)

        # Initially colliding
        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=True)

        # Disable obj2 collision
        old_mode = obj2.collision_mode
        obj2.set_collision_mode(CollisionMode.DISABLED)
        sim_core_kinematics._update_object_collision_mode(
            obj2.object_id,
            old_mode,
            CollisionMode.DISABLED,
        )

        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=False)

    def test_enable_collision_at_runtime(self, sim_core_kinematics):
        """Re-enabling collision mode re-adds object to detection."""
        obj1 = create_test_box(sim_core_kinematics, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        obj2 = create_test_box(sim_core_kinematics, [0.5, 0, 0], collision_mode=CollisionMode.DISABLED)

        sim_core_kinematics._moved_this_step.add(obj1.object_id)

        # Initially disabled → no collision
        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=False)

        # Enable obj2
        old_mode = obj2.collision_mode
        obj2.set_collision_mode(CollisionMode.NORMAL_3D)
        sim_core_kinematics._update_object_collision_mode(
            obj2.object_id,
            old_mode,
            CollisionMode.NORMAL_3D,
        )
        sim_core_kinematics._moved_this_step.add(obj2.object_id)

        assert_collision_detected(sim_core_kinematics, obj1, obj2, should_collide=True)


# ============================================================================
# Run Tests
# ============================================================================


def _broadphase_scene(
    seed, margin, widths, modes=(CollisionMode.NORMAL_3D,), one_moved=False, ignore_static=False, change=None
):
    """Random boxes on and off spatial-hash cell boundaries; returns (missed, checked).

    ``widths`` are AABB widths in cells; each box takes a random mode from
    ``modes``. A pair is missed when its AABBs, enlarged by the collision
    margin, overlap but the broadphase does not return it as a candidate. A
    pair with a NORMAL_2D body needs actual overlap in z, as the broadphase
    requires. With ``one_moved``, each body is searched for alone, as when it
    is the only one that moved: its pairs must be found from its side.

    STATIC bodies never move, so they are never searched from; a pair with
    one must be found from the other body's side, unless ``ignore_static``.
    Two STATIC bodies are never a pair.

    ``change`` spawns everything with an effective margin of 0 and only then
    sets ``margin``, through ``"setter"`` (``set_collision_margin``),
    ``"params_margin"`` (assigning ``params.collision_margin``) or
    ``"params_method"`` (switching ``params.collision_detection_method`` from
    contact points, whose broadphase margin is 0, to closest points): every
    body, STATIC ones included, must then be registered with the new margin.
    """
    import random

    rng = random.Random(seed)
    cell = rng.choice([0.2, 0.5, 1.0])
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_margin=margin if change in (None, "params_method") else 0.0,
            collision_detection_method=(
                CollisionDetectionMethod.CONTACT_POINTS if change == "params_method" else CollisionDetectionMethod.CLOSEST_POINTS
            ),
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=cell,
            ignore_static_collision=ignore_static,
        )
    )
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        objects = []
        for _ in range(30):
            half = 0.5 * cell * rng.choice(widths)
            # On a cell boundary, a hair off it, or anywhere.
            offsets = [0.0, 0.001, -0.001, rng.uniform(-cell, cell)]
            xyz = [round(rng.uniform(0.0, 3.0) / cell) * cell + rng.choice(offsets) for _ in range(3)]
            objects.append(create_test_box(sim, xyz, size=half, collision_mode=rng.choice(modes)))
        static = {obj.object_id for obj in objects if obj.collision_mode == CollisionMode.STATIC}
        movers = [obj.object_id for obj in objects if obj.object_id not in static]
        if change == "setter":
            sim.set_collision_margin(margin)
        elif change == "params_margin":
            sim.params.collision_margin = margin
        elif change == "params_method":
            sim.params.collision_detection_method = CollisionDetectionMethod.CLOSEST_POINTS
        if one_moved:
            found = {}
            for object_id in movers:
                sim._moved_this_step = {object_id}
                found[object_id] = set(sim.filter_aabb_pairs()[0])
        else:
            sim._moved_this_step = set(movers)
            everything = set(sim.filter_aabb_pairs()[0])
            found = {object_id: everything for object_id in movers}
        aabbs = sim._cached_aabbs_dict
        two_d = {obj.object_id for obj in objects if obj.collision_mode == CollisionMode.NORMAL_2D}
        missed = checked = 0
        for a, b in itertools.combinations([obj.object_id for obj in objects], 2):
            if a in static and b in static:
                continue
            if ignore_static and (a in static or b in static):
                continue
            box_a, box_b = aabbs[a], aabbs[b]
            z_margin = 0.0 if a in two_d or b in two_d else margin
            margins = (margin, margin, z_margin)
            overlap = all(
                box_a[1][d] + margins[d] >= box_b[0][d] and box_a[0][d] - margins[d] <= box_b[1][d] for d in range(3)
            )
            checked += 1
            pair = (min(a, b), max(a, b))
            if overlap and any(pair not in found[s] for s in (a, b) if s not in static):
                missed += 1
        return missed, checked
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("margin", [0.02, 0.1, 0.25])
def test_broadphase_returns_every_pair_within_the_margin(seed, margin):
    """Against brute force: tiny to multi-cell bodies, on and off cell
    boundaries; every body is in each cell its AABB enlarged by margin/2
    touches, and two within the margin share one."""
    missed, checked = _broadphase_scene(seed, margin, widths=[0.04, 0.1, 0.2, 0.9, 1.0, 1.4, 2.8, 5.0])
    assert checked > 0
    assert missed == 0


@pytest.mark.parametrize("seed", range(6))
def test_broadphase_without_margin_returns_every_touching_pair(seed):
    """margin == 0 (contact points): every width, including bodies between one
    and one and a half cells wide that issue #89 missed."""
    missed, checked = _broadphase_scene(seed, 0.0, widths=[0.04, 0.1, 0.2, 0.9, 1.0, 1.2, 1.4, 1.6, 2.8, 5.0])
    assert checked > 0
    assert missed == 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("margin", [0.0, 0.02, 0.1])
def test_broadphase_finds_pairs_with_normal_2d_bodies(seed, margin):
    """A NORMAL_2D body is registered by its real z extent, so a pair with one
    that overlaps in z shares a cell. Against brute force, with 2D, 3D and
    mixed pairs placed on and off z cell boundaries."""
    widths = [0.04, 0.1, 0.2, 0.9, 1.0, 1.4, 2.8, 5.0]
    missed, checked = _broadphase_scene(seed, margin, widths=widths, modes=(CollisionMode.NORMAL_2D, CollisionMode.NORMAL_3D))
    assert checked > 0
    assert missed == 0


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("margin", [0.0, 0.02, 0.1])
def test_broadphase_finds_normal_2d_pairs_from_either_side(seed, margin):
    """As above, but each body searched alone: a NORMAL_2D body that moved
    must itself find a NORMAL_3D body in the z layer next to its own, which
    it only looks into while such a body is registered there."""
    widths = [0.04, 0.1, 0.2, 0.9, 1.0, 2.8, 5.0]
    modes = (CollisionMode.NORMAL_2D, CollisionMode.NORMAL_3D)
    missed, checked = _broadphase_scene(seed, margin, widths=widths, modes=modes, one_moved=True)
    assert checked > 0
    assert missed == 0


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("margin", [0.0, 0.02, 0.1])
@pytest.mark.parametrize("ignore_static", [False, True], ids=["static_counts", "static_ignored"])
@pytest.mark.parametrize("one_moved", [False, True], ids=["all_moved", "one_moved"])
def test_broadphase_finds_pairs_with_static_bodies(seed, margin, ignore_static, one_moved):
    """As above, with STATIC bodies among the 2D and 3D ones. A NORMAL_2D body
    looks into the next z layer for a STATIC one only while static collisions
    count; when they are ignored no pair with a STATIC body is required."""
    widths = [0.04, 0.1, 0.2, 0.9, 1.0, 2.8, 5.0]
    modes = (CollisionMode.NORMAL_2D, CollisionMode.NORMAL_3D, CollisionMode.STATIC)
    missed, checked = _broadphase_scene(
        seed, margin, widths=widths, modes=modes, one_moved=one_moved, ignore_static=ignore_static
    )
    assert checked > 0
    assert missed == 0


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("margin", [0.02, 0.1])
@pytest.mark.parametrize("change", ["setter", "params_margin", "params_method"])
@pytest.mark.parametrize("one_moved", [False, True], ids=["all_moved", "one_moved"])
def test_broadphase_follows_a_margin_change(seed, margin, change, one_moved):
    """Bodies spawned with no broadphase margin, the margin raised afterwards:
    every registration, STATIC bodies' included, must follow it however it was
    changed -- the setter, the params field, or the detection method."""
    widths = [0.04, 0.1, 0.2, 0.9, 1.0, 2.8]
    modes = (CollisionMode.NORMAL_2D, CollisionMode.NORMAL_3D, CollisionMode.STATIC)
    missed, checked = _broadphase_scene(seed, margin, widths=widths, modes=modes, one_moved=one_moved, change=change)
    assert checked > 0
    assert missed == 0


@pytest.mark.parametrize("change", ["setter", "params_margin", "params_method"])
@pytest.mark.parametrize("kind", [CollisionMode.NORMAL_3D, CollisionMode.STATIC])
def test_a_margin_change_reregisters_bodies_registered_with_the_old_one(change, kind):
    """Two boxes 0.04 apart across a cell boundary (cell 1.0, x = 1.0): with
    no margin they share no cell. Raised to 0.1 afterwards, the pair is within
    the margin and must be a candidate -- which needs both bodies, the other
    one STATIC or not, re-registered with the new margin however it changed."""
    margin = 0.1
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_margin=0.0 if change != "params_method" else margin,
            collision_detection_method=(
                CollisionDetectionMethod.CONTACT_POINTS if change == "params_method" else CollisionDetectionMethod.CLOSEST_POINTS
            ),
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=1.0,
            ignore_static_collision=False,
        )
    )
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        mover = create_test_box(sim, [0.74, 0.5, 0.5], size=0.24, collision_mode=CollisionMode.NORMAL_3D)
        other = create_test_box(sim, [1.26, 0.5, 0.5], size=0.24, collision_mode=kind)
        pair = (min(mover.object_id, other.object_id), max(mover.object_id, other.object_id))
        sim._moved_this_step = {mover.object_id}
        assert pair not in set(sim.filter_aabb_pairs()[0])  # not within a margin of 0
        if change == "setter":
            sim.set_collision_margin(margin)
        elif change == "params_margin":
            sim.params.collision_margin = margin
        else:
            sim.params.collision_detection_method = CollisionDetectionMethod.CLOSEST_POINTS
        sim._moved_this_step = {mover.object_id}
        assert pair in set(sim.filter_aabb_pairs()[0])
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("kind", [CollisionMode.NORMAL_3D, CollisionMode.STATIC])
@pytest.mark.parametrize("ignore_static", [False, True], ids=["static_counts", "static_ignored"])
@pytest.mark.parametrize("margin", [0.0, 0.02])
def test_a_moved_normal_2d_body_and_one_centred_in_the_next_layer(kind, ignore_static, margin):
    """The 2D body lies in z layer 1 only (z 0.50-0.70, cell 0.5); the other
    body is centred in layer 0 (z 0.30-0.52) and reaches into layer 1, where
    the two overlap. Only the 2D one moved, and it looks in its own cells only:
    the pair is found there, except for a STATIC body while static collisions
    are ignored."""
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_margin=margin,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=0.5,
            ignore_static_collision=ignore_static,
        )
    )
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        flat = create_test_box(sim, [0.2, 0.2, 0.60], size=0.10, collision_mode=CollisionMode.NORMAL_2D)
        other = create_test_box(sim, [0.2, 0.2, 0.41], size=0.11, collision_mode=kind)
        assert {c[2] for c in sim._cached_object_to_cell[flat.object_id]} == {1}
        assert {c[2] for c in sim._cached_object_to_cell[other.object_id]} == {0, 1}
        looked_up = set()

        class Grid(dict):
            def get(self, key, default=None):
                looked_up.add(key)
                return super().get(key, default)

        sim._cached_spatial_grid = Grid(sim._cached_spatial_grid)
        sim._moved_this_step = {flat.object_id}
        pair = tuple(sorted((flat.object_id, other.object_id)))
        expected = not (ignore_static and kind == CollisionMode.STATIC)
        assert (pair in set(sim.filter_aabb_pairs()[0])) is expected
        assert looked_up == set(sim._cached_object_to_cell[flat.object_id])
    finally:
        p.disconnect(sim.client)


def test_the_grid_follows_every_body():
    """Each body is in exactly the cells its AABB, enlarged by margin/2 (not in
    z for NORMAL_2D), touches, through moves, mode changes between NORMAL_2D,
    NORMAL_3D, STATIC and DISABLED (also ones that keep the same cells) and
    removals; a DISABLED body is in no cell."""
    import random

    rng = random.Random(3)
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_margin=0.02,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=0.5,
            ignore_static_collision=False,
        )
    )

    def consistent():
        half = sim._collision_broadphase_margin() / 2
        grid = {}
        for object_id, cells in sim._cached_object_to_cell.items():
            if sorted(cells) != sorted(sim._get_overlapping_cells(object_id, half)):
                return False
            for cell in cells:
                grid.setdefault(cell, set()).add(object_id)
        return grid == sim._cached_spatial_grid and not sim._disabled_collision_objects & set(sim._cached_object_to_cell)

    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        modes = (CollisionMode.NORMAL_2D, CollisionMode.NORMAL_3D, CollisionMode.STATIC)
        objects = [
            create_test_box(
                sim,
                [rng.uniform(0, 2), rng.uniform(0, 2), rng.uniform(0, 2)],
                size=rng.choice([0.05, 0.3, 0.8]),
                collision_mode=rng.choice(modes),
            )
            for _ in range(20)
        ]
        assert consistent()
        for _ in range(90):
            obj = rng.choice(objects)
            action = rng.random()
            if action < 0.5:
                pos = [rng.uniform(0, 2), rng.uniform(0, 2), rng.uniform(0, 2)]
                p.resetBasePositionAndOrientation(obj.body_id, pos, [0, 0, 0, 1], physicsClientId=sim.client)
                sim._update_object_aabb(obj.object_id)
            elif action < 0.93:
                others = [m for m in modes + (CollisionMode.DISABLED,) if m != obj.collision_mode]
                obj.set_collision_mode(rng.choice(others))
            else:
                sim.remove_object(obj)
                objects.remove(obj)
            assert consistent()
    finally:
        p.disconnect(sim.client)


def test_issue_89_bodies_one_to_one_and_a_half_cells_wide_touching_without_margin():
    """Issue #89: without a margin, two touching bodies 1-1.5 cells wide were
    missed when each sat in one cell (centre cells two apart). Every body is
    now in every cell it touches."""
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            collision_margin=0.0,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=1.0,
            ignore_static_collision=False,
        )
    )
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        # 1.4 cells wide each, overlapping by 0.01 around x=1.4: centre cells 0 and 2.
        left = create_test_box(sim, [0.7, 0.5, 0.5], size=0.7)
        right = create_test_box(sim, [2.09, 0.5, 0.5], size=0.7)
        sim._moved_this_step = {left.object_id}
        pair = tuple(sorted((left.object_id, right.object_id)))
        assert pair in set(sim.filter_aabb_pairs()[0])
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("modes", [(CollisionMode.NORMAL_2D,) * 2, (CollisionMode.NORMAL_2D, CollisionMode.NORMAL_3D)])
def test_normal_2d_pair_across_a_z_cell_boundary(modes):
    """Overlapping in z, with centres in different z cells (cell 0.5: z 0.40-0.60
    and 0.30-0.45)."""
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_margin=0.02,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=0.5,
            ignore_static_collision=False,
        )
    )
    try:
        sim.set_collision_spatial_hash_cell_size_mode()
        upper = create_test_box(sim, [0.2, 0.2, 0.50], size=0.10, collision_mode=modes[0])
        lower = create_test_box(sim, [0.2, 0.2, 0.375], size=0.075, collision_mode=modes[1])
        for moved in (upper, lower):
            sim._moved_this_step = {moved.object_id}
            pair = tuple(sorted((upper.object_id, lower.object_id)))
            assert pair in set(sim.filter_aabb_pairs()[0]), f"missed when only {moved.collision_mode} moved"
    finally:
        p.disconnect(sim.client)


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
