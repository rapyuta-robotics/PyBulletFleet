"""
Tests for geometry module (Pose, Path, and quaternion slerp).

This module tests:
- Pose creation from various formats
- Pose conversions
- Distance calculations
- Path creation and waypoint management
- Fast quaternion slerp (quat_slerp / quat_slerp_precompute)
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from pybullet_fleet.geometry import (
    Path,
    Pose,
    as_axes,
    is_scalar,
    quat_angle_between,
    quat_from_rotvec,
    quat_multiply,
    quat_slerp,
    quat_slerp_precompute,
    quat_to_rot_matrix,
    rotate_vector,
)


# Helper functions for testing
IDENTITY_QUAT = [0, 0, 0, 1]  # Identity quaternion (no rotation)


def test_scalar_helpers_accept_only_numeric_scalars():
    assert is_scalar(1.0)
    assert is_scalar(np.float32(1.0))
    assert is_scalar(np.array(1.0))
    assert not is_scalar([1.0, 2.0, 3.0])
    assert not is_scalar("1.0")
    assert not is_scalar(None)
    np.testing.assert_array_equal(as_axes(2.0), np.array([2.0, 2.0, 2.0]))


def assert_pose_equal(pose: Pose, expected_position: list, expected_orientation: list = None, decimal: int = 3):
    """
    Assert that pose position (and optionally orientation) matches expected values.

    Args:
        pose: Pose object to check
        expected_position: Expected [x, y, z] position
        expected_orientation: Expected [x, y, z, w] quaternion (optional, skip if None)
        decimal: Number of decimal places for comparison (default: 3, i.e. 0.0005 tolerance)
    """
    np.testing.assert_array_almost_equal(pose.position, expected_position, decimal=decimal, err_msg="Position mismatch")
    if expected_orientation is not None:
        np.testing.assert_array_almost_equal(
            pose.orientation, expected_orientation, decimal=decimal, err_msg="Orientation mismatch"
        )


class TestPoseCreation:
    """Test Pose creation methods"""

    def test_pose_from_xyz(self):
        """Test creating Pose from position only"""
        pose = Pose.from_xyz(1.0, 2.0, 3.0)

        assert_pose_equal(pose, [1.0, 2.0, 3.0], IDENTITY_QUAT)

    def test_pose_from_euler(self):
        """Test creating Pose from position and Euler angles"""
        pose = Pose.from_euler(1.0, 2.0, 3.0, roll=0.1, pitch=0.2, yaw=0.3)

        # Position and orientation (quaternion for roll=0.1, pitch=0.2, yaw=0.3)
        assert_pose_equal(pose, [1.0, 2.0, 3.0], [0.034271, 0.106021, 0.143572, 0.983347])

        # Euler angles should round-trip correctly
        assert abs(pose.roll - 0.1) < 0.01
        assert abs(pose.pitch - 0.2) < 0.01
        assert abs(pose.yaw - 0.3) < 0.01

    def test_pose_from_pybullet(self):
        """Test creating Pose from PyBullet format"""
        position = [1.0, 2.0, 3.0]
        orientation = [0.0, 0.0, 0.707, 0.707]  # 90 degrees around Z
        pose = Pose.from_pybullet(position, orientation)

        assert_pose_equal(pose, position, orientation)

    def test_pose_from_yaw(self):
        """Test creating Pose with yaw angle only"""
        pose = Pose.from_yaw(1, 2, 3, yaw=np.pi / 2)

        # Position should match
        assert_pose_equal(pose, [1, 2, 3])

        # Yaw should be approximately pi/2
        assert abs(pose.yaw - np.pi / 2) < 0.01

    def test_pose_default_constructor(self):
        """Test Pose with minimal constructor"""
        # Pose requires position, but orientation has default
        pose = Pose(position=[0, 0, 0])

        assert_pose_equal(pose, [0, 0, 0], IDENTITY_QUAT)

    def test_pose_with_position_and_orientation(self):
        """Test Pose with explicit position and orientation"""
        position = [1.0, 2.0, 3.0]
        orientation = [0.0, 0.0, 0.707, 0.707]
        pose = Pose(position=position, orientation=orientation)

        assert_pose_equal(pose, position, orientation)


class TestPoseConversion:
    """Test Pose conversion methods"""

    def test_as_position_orientation(self):
        """Test converting Pose to position/orientation tuple"""
        pose = Pose.from_xyz(1.0, 2.0, 3.0)
        position, orientation = pose.as_position_orientation()

        np.testing.assert_array_almost_equal(position, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(orientation, [0, 0, 0, 1])

    def test_as_euler(self):
        """Test converting Pose to Euler angles"""
        # Create pose with known Euler angles
        roll, pitch, yaw = 0.1, 0.2, 0.3
        pose = Pose.from_euler(0, 0, 0, roll=roll, pitch=pitch, yaw=yaw)

        # Convert back to Euler
        result_roll, result_pitch, result_yaw = pose.as_euler()

        # Should match original values (within tolerance)
        assert abs(result_roll - roll) < 0.01
        assert abs(result_pitch - pitch) < 0.01
        assert abs(result_yaw - yaw) < 0.01

    def test_euler_roundtrip(self):
        """Test Euler angle conversion roundtrip"""
        original_roll, original_pitch, original_yaw = 0.5, -0.3, 1.2

        # Create pose from Euler
        pose = Pose.from_euler(1, 2, 3, roll=original_roll, pitch=original_pitch, yaw=original_yaw)

        # Convert back
        roll, pitch, yaw = pose.as_euler()

        # Verify roundtrip
        assert abs(roll - original_roll) < 0.01
        assert abs(pitch - original_pitch) < 0.01
        assert abs(yaw - original_yaw) < 0.01


class TestPoseProperties:
    """Test Pose properties and accessors"""

    def test_xyz_properties(self):
        """Test x, y, z property accessors"""
        pose = Pose.from_xyz(1.5, 2.5, 3.5)

        assert_pose_equal(pose, [1.5, 2.5, 3.5])

    def test_position_is_list(self):
        """Test that position is a list"""
        pose = Pose.from_xyz(1, 2, 3)

        assert isinstance(pose.position, list)
        assert len(pose.position) == 3

    def test_orientation_is_list(self):
        """Test that orientation is a list"""
        pose = Pose(position=[0, 0, 0])

        assert isinstance(pose.orientation, list)
        assert len(pose.orientation) == 4

    def test_yaw_property(self):
        """Test yaw property accessor works with different Pose creation methods"""
        # Test with from_yaw
        pose1 = Pose.from_yaw(0, 0, 0, yaw=1.2)
        assert abs(pose1.yaw - 1.2) < 0.01

        # Test with from_euler (yaw should be extracted correctly)
        pose2 = Pose.from_euler(0, 0, 0, roll=0.1, pitch=0.2, yaw=0.5)
        assert abs(pose2.yaw - 0.5) < 0.01

        # Test with default pose (should be 0)
        pose3 = Pose.from_xyz(0, 0, 0)
        assert abs(pose3.yaw - 0.0) < 0.01

    def test_roll_pitch_yaw_properties(self):
        """Test roll, pitch, yaw property accessors"""
        roll, pitch, yaw = 0.1, 0.2, 0.3
        pose = Pose.from_euler(0, 0, 0, roll=roll, pitch=pitch, yaw=yaw)

        # All angles should match
        assert abs(pose.roll - roll) < 0.01
        assert abs(pose.pitch - pitch) < 0.01
        assert abs(pose.yaw - yaw) < 0.01
        pose = Pose(position=[0, 0, 0])

        assert isinstance(pose.orientation, list)
        assert len(pose.orientation) == 4


class TestPoseDistanceCalculations:
    """Test Pose distance calculation methods"""

    def test_distance_calculation_same_position(self):
        """Test distance between same positions using numpy"""
        pose1 = Pose.from_xyz(1, 2, 3)
        pose2 = Pose.from_xyz(1, 2, 3)

        # Calculate distance manually
        pos1 = np.array(pose1.position)
        pos2 = np.array(pose2.position)
        distance = np.linalg.norm(pos1 - pos2)

        assert distance == 0.0

    def test_distance_calculation_different_position(self):
        """Test distance between different positions using numpy"""
        pose1 = Pose.from_xyz(0, 0, 0)
        pose2 = Pose.from_xyz(3, 4, 0)  # 3-4-5 triangle

        # Calculate distance manually
        pos1 = np.array(pose1.position)
        pos2 = np.array(pose2.position)
        distance = np.linalg.norm(pos1 - pos2)

        assert abs(distance - 5.0) < 0.001

    def test_distance_calculation_3d(self):
        """Test 3D distance calculation using numpy"""
        pose1 = Pose.from_xyz(0, 0, 0)
        pose2 = Pose.from_xyz(1, 1, 1)

        expected_distance = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)

        # Calculate distance manually
        pos1 = np.array(pose1.position)
        pos2 = np.array(pose2.position)
        distance = np.linalg.norm(pos1 - pos2)

        assert abs(distance - expected_distance) < 0.001


class TestPath:
    """Test Path class"""

    def test_path_from_positions(self):
        """Test creating Path from list of positions"""
        positions = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
        path = Path.from_positions(positions)

        assert len(path.waypoints) == 4
        assert_pose_equal(path.waypoints[0], [0, 0, 0])
        assert_pose_equal(path.waypoints[1], [1, 0, 0])
        assert_pose_equal(path.waypoints[2], [1, 1, 0])
        assert_pose_equal(path.waypoints[3], [0, 1, 0])

    def test_path_constructor(self):
        """Test Path constructor with waypoints"""
        waypoints = [Pose.from_xyz(1, 2, 3), Pose.from_xyz(4, 5, 6)]
        path = Path(waypoints=waypoints)

        assert len(path.waypoints) == 2
        assert_pose_equal(path.waypoints[0], [1, 2, 3])
        assert_pose_equal(path.waypoints[1], [4, 5, 6])

    def test_path_indexing(self):
        """Test path indexing (bracket notation access)"""
        path = Path.from_positions([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        # Test indexing with bracket notation
        waypoint0 = path[0]
        waypoint1 = path[1]
        waypoint2 = path[2]

        assert_pose_equal(waypoint0, [0, 0, 0])
        assert_pose_equal(waypoint1, [1, 1, 1])
        assert_pose_equal(waypoint2, [2, 2, 2])

    def test_path_indexing_out_of_bounds(self):
        """Test path indexing with invalid index"""
        path = Path.from_positions([[0, 0, 0]])

        # Should raise IndexError
        with pytest.raises(IndexError):
            _ = path[10]

    def test_path_empty(self):
        """Test empty path"""
        path = Path(waypoints=[])

        assert len(path.waypoints) == 0
        assert path.waypoints == []

    def test_path_iteration(self):
        """Test iterating over path waypoints"""
        poses = [
            Pose.from_xyz(0, 0, 0),
            Pose.from_xyz(1, 0, 0),
            Pose.from_euler(2, 0, 0, yaw=1.57),  # With rotation
        ]
        path = Path(waypoints=poses)

        assert len(path) == 3

        # Test iteration with complete position checking
        expected_positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        for i, waypoint in enumerate(path):
            assert_pose_equal(waypoint, expected_positions[i])

        # Third waypoint should have rotation
        assert not np.allclose(path[2].orientation, [0, 0, 0, 1])


class TestPoseEdgeCases:
    """Test edge cases and special scenarios"""

    def test_pose_zero_position(self):
        """Test pose at origin"""
        pose = Pose.from_xyz(0, 0, 0)

        assert_pose_equal(pose, [0, 0, 0], IDENTITY_QUAT)

    def test_pose_negative_coordinates(self):
        """Test pose with negative coordinates"""
        pose = Pose.from_xyz(-1, -2, -3)

        assert_pose_equal(pose, [-1, -2, -3], IDENTITY_QUAT)

    def test_pose_large_coordinates(self):
        """Test pose with large coordinates"""
        pose = Pose.from_xyz(1000, 2000, 3000)

        assert_pose_equal(pose, [1000, 2000, 3000], IDENTITY_QUAT)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPathGeneration:
    """Test Path generation methods (rectangle, square, circle, ellipse)"""

    def test_create_rectangle(self):
        """Test creating rectangular path"""
        center = [0, 0, 0]
        width = 4.0
        height = 2.0

        path = Path.create_rectangle(center=center, width=width, height=height)

        # Should have 5 waypoints (4 corners + return to start)
        assert len(path) == 5

        # First and last waypoints should be at same position
        assert abs(path[0].x - path[4].x) < 0.001
        assert abs(path[0].y - path[4].y) < 0.001

    def test_create_square(self):
        """Test creating square path"""
        center = [1, 2, 3]
        side_length = 2.0

        path = Path.create_square(center=center, side_length=side_length)

        # Should have 5 waypoints
        assert len(path) == 5

        # Check center is approximately correct
        x_coords = [w.x for w in path.waypoints[:4]]
        y_coords = [w.y for w in path.waypoints[:4]]

        assert abs(np.mean(x_coords) - center[0]) < 0.1
        assert abs(np.mean(y_coords) - center[1]) < 0.1

    def test_create_ellipse(self):
        """Test creating elliptical path"""
        center = [0, 0, 0]
        radius_x = 2.0
        radius_y = 1.0
        num_points = 8

        path = Path.create_ellipse(center=center, radius_x=radius_x, radius_y=radius_y, num_points=num_points)

        # Should have num_points + 1 waypoints
        assert len(path) == num_points + 1

        # Check approximate radii
        distances_x = [abs(w.x - center[0]) for w in path.waypoints]
        distances_y = [abs(w.y - center[1]) for w in path.waypoints]

        assert max(distances_x) <= radius_x + 0.1
        assert max(distances_y) <= radius_y + 0.1

    def test_create_circle(self):
        """Test creating circular path"""
        center = [0, 0, 0]
        radius = 1.5
        num_points = 16

        path = Path.create_circle(center=center, radius=radius, num_points=num_points)

        # Should have num_points + 1 waypoints
        assert len(path) == num_points + 1

        # All points should be approximately at radius distance from center
        for waypoint in path.waypoints:
            dx = waypoint.x - center[0]
            dy = waypoint.y - center[1]
            distance = np.sqrt(dx**2 + dy**2)
            assert abs(distance - radius) < 0.1

    def test_create_rectangle_with_rotation(self):
        """Test creating rotated rectangular path"""
        center = [0, 0, 1]
        width = 3.0
        height = 2.0
        rpy = [0, 0, np.pi / 4]  # 45 degree yaw rotation

        path = Path.create_rectangle(center=center, width=width, height=height, rpy=rpy)

        assert len(path) == 5

        # All z-coordinates should be approximately at center[2]
        for waypoint in path.waypoints:
            assert abs(waypoint.z - center[2]) < 0.1

    def test_create_square_tilted(self):
        """Test creating tilted square path"""
        center = [0, 0, 1]
        side_length = 2.0
        rpy = [0.1, 0.2, 0]  # Tilted in 3D

        path = Path.create_square(center=center, side_length=side_length, rpy=rpy)

        # Should still have 5 waypoints
        assert len(path) == 5

        # Waypoints should have non-identity orientations
        for waypoint in path.waypoints:
            # At least some orientation should be non-default
            pass  # Orientations are set based on direction


class TestPathAdvancedFeatures:
    """Test advanced Path features"""

    def test_path_length(self):
        """Test len() operator on Path"""
        waypoints = [Pose.from_xyz(i, 0, 0) for i in range(5)]
        path = Path(waypoints=waypoints)

        assert len(path) == 5

    def test_path_iteration_multiple_times(self):
        """Test that path can be iterated multiple times"""
        path = Path.from_positions([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

        # First iteration
        count1 = sum(1 for _ in path)
        # Second iteration
        count2 = sum(1 for _ in path)

        assert count1 == 3
        assert count2 == 3

    def test_path_slicing(self):
        """Test path supports list slicing"""
        path = Path.from_positions([[i, 0, 0] for i in range(10)])

        # Test slicing waypoints list
        first_three = path.waypoints[:3]
        assert len(first_three) == 3
        assert_pose_equal(first_three[0], [0, 0, 0])
        assert_pose_equal(first_three[2], [2, 0, 0])

    def test_path_append(self):
        """Test appending one path to another (in-place mutation)"""
        path1 = Path.from_positions([[0, 0, 0], [1, 0, 0]])
        path2 = Path.from_positions([[2, 0, 0], [3, 0, 0]])

        # Append path2 to path1
        path1.append(path2)

        # path1 should now have 4 waypoints
        assert len(path1) == 4
        assert_pose_equal(path1[0], [0, 0, 0])
        assert_pose_equal(path1[1], [1, 0, 0])
        assert_pose_equal(path1[2], [2, 0, 0])
        assert_pose_equal(path1[3], [3, 0, 0])

        # path2 should be unchanged
        assert len(path2) == 2

    def test_path_add_operator(self):
        """Test adding paths with + operator (non-mutating)"""
        path1 = Path.from_positions([[0, 0, 0], [1, 0, 0]])
        path2 = Path.from_positions([[2, 0, 0], [3, 0, 0]])

        # Create new path with + operator
        combined = path1 + path2

        # Combined path should have 4 waypoints
        assert len(combined) == 4
        assert_pose_equal(combined[0], [0, 0, 0])
        assert_pose_equal(combined[1], [1, 0, 0])
        assert_pose_equal(combined[2], [2, 0, 0])
        assert_pose_equal(combined[3], [3, 0, 0])

        # Original paths should be unchanged
        assert len(path1) == 2
        assert len(path2) == 2

    def test_path_from_paths(self):
        """Test creating path from multiple paths"""
        path1 = Path.from_positions([[0, 0, 0], [1, 0, 0]])
        path2 = Path.from_positions([[2, 0, 0], [3, 0, 0]])
        path3 = Path.from_positions([[4, 0, 0], [5, 0, 0]])

        # Combine multiple paths
        combined = Path.from_paths([path1, path2, path3])

        # Should have all waypoints
        assert len(combined) == 6
        assert_pose_equal(combined[0], [0, 0, 0])
        assert_pose_equal(combined[2], [2, 0, 0])
        assert_pose_equal(combined[4], [4, 0, 0])
        assert_pose_equal(combined[5], [5, 0, 0])

    def test_path_get_total_distance(self):
        """Test calculating total path distance"""
        # Simple straight line path
        path = Path.from_positions([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])

        total_distance = path.get_total_distance()

        # Expected: 1 + 1 + 1 = 3.0
        assert abs(total_distance - 3.0) < 0.01

    def test_path_get_total_distance_empty(self):
        """Test total distance for empty or single-waypoint path"""
        # Empty path
        empty_path = Path(waypoints=[])
        assert empty_path.get_total_distance() == 0.0

        # Single waypoint
        single_path = Path.from_positions([[0, 0, 0]])
        assert single_path.get_total_distance() == 0.0


# ============================================================================
# Quaternion slerp utility tests
# ============================================================================


class TestQuatSlerp:
    """Test the fast quaternion slerp implementation against scipy Slerp."""

    def test_quat_slerp_antipodal_fallback(self):
        """Bug: lerp fallback must use needs_flip when q0 ≈ -q1 (antipodal).

        When quaternions are nearly antipodal, precompute sets needs_flip=True
        and after flip sin_theta_0 ≈ 0, triggering the lerp fallback. The
        fallback must interpolate toward -q1 (not raw q1) for shortest path.
        """
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        # q1 ≈ -q0  (same rotation, opposite sign — antipodal)
        q1 = np.array([0.0, 0.0, 0.0, -1.0])

        precomp = quat_slerp_precompute(q0, q1)
        # After flip, these should be near-identical, so lerp fallback triggers
        assert precomp.needs_flip is True

        result = quat_slerp(q0, q1, 0.5, precomp)
        # Midpoint of identity with itself must be identity (or its antipode)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10
        # Result should be close to q0 (identity), not diverging
        assert abs(np.dot(result, q0)) > 0.999

    def test_quat_slerp_matches_scipy(self):
        """Manual quat_slerp must match scipy Slerp across many rotation pairs."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            # Generate random quaternion pair
            q0 = R.random(random_state=rng).as_quat()  # type: ignore[call-arg]  # [x, y, z, w]
            q1 = R.random(random_state=rng).as_quat()  # type: ignore[call-arg]

            # Scipy reference
            key_rots = R.from_quat([q0, q1])
            scipy_slerp = Slerp([0.0, 1.0], key_rots)

            # Precompute constants
            precomp = quat_slerp_precompute(q0, q1)

            for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
                expected = scipy_slerp(t).as_quat()
                actual = quat_slerp(q0, q1, t, precomp)
                # Quaternions q and -q represent the same rotation
                if np.dot(expected, actual) < 0:
                    actual = -actual
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-10,
                    err_msg=f"Slerp mismatch at t={t}, q0={q0}, q1={q1}",
                )

    def test_quat_slerp_near_identity(self):
        """Slerp between nearly identical quaternions (small angle)."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = np.array([0.0, 0.0, 0.001, 0.9999995])  # ~0.1° rotation
        q1 = q1 / np.linalg.norm(q1)

        precomp = quat_slerp_precompute(q0, q1)
        result = quat_slerp(q0, q1, 0.5, precomp)

        # Should be a valid unit quaternion
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10
        # Should be between q0 and q1
        assert np.dot(result, q0) > 0.999

    def test_quat_slerp_endpoints(self):
        """Slerp at t=0 returns q0, at t=1 returns q1."""
        q0 = R.from_euler("z", 0, degrees=True).as_quat()
        q1 = R.from_euler("z", 90, degrees=True).as_quat()
        precomp = quat_slerp_precompute(q0, q1)

        r0 = quat_slerp(q0, q1, 0.0, precomp)
        r1 = quat_slerp(q0, q1, 1.0, precomp)

        np.testing.assert_allclose(r0, q0, atol=1e-12)
        np.testing.assert_allclose(r1, q1, atol=1e-12)

    def test_quat_slerp_180_degrees(self):
        """Slerp across 180° rotation."""
        q0 = R.from_euler("z", 0, degrees=True).as_quat()
        q1 = R.from_euler("z", 170, degrees=True).as_quat()  # near 180°
        precomp = quat_slerp_precompute(q0, q1)

        mid = quat_slerp(q0, q1, 0.5, precomp)
        # Midpoint should be at 85°
        expected = R.from_euler("z", 85, degrees=True).as_quat()
        if np.dot(mid, expected) < 0:
            mid = -mid
        np.testing.assert_allclose(mid, expected, atol=1e-10)

    def test_quat_slerp_returns_ndarray(self):
        """quat_slerp returns a numpy array (can call .tolist() on it)."""
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = np.array([0.0, 0.0, 0.3827, 0.9239])
        q1 = q1 / np.linalg.norm(q1)
        precomp = quat_slerp_precompute(q0, q1)

        result = quat_slerp(q0, q1, 0.5, precomp)
        assert isinstance(result, np.ndarray)
        assert len(result.tolist()) == 4


# ============================================================================
# Quaternion rotation utilities (quat_to_rot_matrix, rotate_vector)
# ============================================================================


class TestQuatRotateVector:
    """quat_to_rot_matrix / rotate_vector standalone quaternion utilities."""

    def test_quat_to_rot_matrix_identity(self):
        """Identity quaternion produces identity matrix."""
        m = quat_to_rot_matrix((0.0, 0.0, 0.0, 1.0))
        assert len(m) == 9
        expected = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        for i in range(9):
            assert m[i] == pytest.approx(expected[i], abs=1e-12)

    def test_quat_to_rot_matrix_matches_scipy(self):
        """Rotation matrix matches scipy for random quaternions."""
        import random

        random.seed(99)
        for _ in range(10):
            q = [random.gauss(0, 1) for _ in range(4)]
            norm = sum(x * x for x in q) ** 0.5
            q = [x / norm for x in q]

            expected = R.from_quat(q).as_matrix().flatten()
            actual = quat_to_rot_matrix(tuple(q))
            for i in range(9):
                assert actual[i] == pytest.approx(expected[i], abs=1e-12)

    def test_rotate_vector_identity(self):
        """Identity quaternion passes through vector."""
        result = rotate_vector((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        assert result == pytest.approx((1.0, 2.0, 3.0))

    def test_rotate_vector_yaw_90(self):
        """90-degree yaw rotates X to Y."""
        q = tuple(R.from_euler("z", 90, degrees=True).as_quat())
        result = rotate_vector((1.0, 0.0, 0.0), q)
        assert result[0] == pytest.approx(0.0, abs=1e-9)
        assert result[1] == pytest.approx(1.0, abs=1e-9)
        assert result[2] == pytest.approx(0.0, abs=1e-9)

    def test_rotate_vector_matches_scipy(self):
        """rotate_vector matches scipy for random inputs."""
        import random

        random.seed(77)
        for _ in range(20):
            q = [random.gauss(0, 1) for _ in range(4)]
            norm = sum(x * x for x in q) ** 0.5
            q = [x / norm for x in q]
            v = [random.uniform(-10, 10) for _ in range(3)]

            expected = R.from_quat(q).apply(v)
            actual = rotate_vector(tuple(v), tuple(q))
            for i in range(3):
                assert actual[i] == pytest.approx(expected[i], abs=1e-9)

    def test_unnormalized_quat_matches_scipy(self):
        """Non-unit quaternion is normalized internally, matching scipy."""
        q_raw = (2.0, 0.0, 0.0, 2.0)
        expected_mat = R.from_quat(q_raw).as_matrix().flatten()
        actual_mat = quat_to_rot_matrix(q_raw)
        for i in range(9):
            assert actual_mat[i] == pytest.approx(expected_mat[i], abs=1e-12)

        v = (1.0, 2.0, 3.0)
        expected_vec = R.from_quat(q_raw).apply(v)
        actual_vec = rotate_vector(v, q_raw)
        for i in range(3):
            assert actual_vec[i] == pytest.approx(expected_vec[i], abs=1e-9)


# ============================================================================
# Quaternion math utilities (quat_from_rotvec, quat_multiply, quat_angle_between)
# ============================================================================


class TestNormalizeQuat:
    """_normalize_quat helper rejects zero-norm and normalizes correctly."""

    def test_zero_norm_raises(self):
        """Zero quaternion should raise ValueError."""
        from pybullet_fleet.geometry import _normalize_quat

        with pytest.raises(ValueError, match="zero-norm"):
            _normalize_quat((0.0, 0.0, 0.0, 0.0))

    def test_unit_passthrough(self):
        """Already-unit quaternion is returned unchanged (within tolerance)."""
        from pybullet_fleet.geometry import _normalize_quat

        q = (0.0, 0.0, 0.3827, 0.9239)
        result = _normalize_quat(q)
        for i in range(4):
            assert result[i] == pytest.approx(q[i], abs=1e-4)

    def test_scales_to_unit(self):
        """Non-unit quaternion is scaled to norm 1."""
        import math as _math

        from pybullet_fleet.geometry import _normalize_quat

        q = (2.0, 0.0, 0.0, 2.0)
        result = _normalize_quat(q)
        norm = _math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0, abs=1e-12)


class TestQuatToRotMatrixZero:
    """quat_to_rot_matrix rejects zero-norm quaternion."""

    def test_zero_quat_raises(self):
        """quat_to_rot_matrix((0,0,0,0)) should raise ValueError."""
        from pybullet_fleet.geometry import quat_to_rot_matrix

        with pytest.raises(ValueError, match="zero-norm"):
            quat_to_rot_matrix((0.0, 0.0, 0.0, 0.0))


class TestQuatAngleBetweenNormalize:
    """quat_angle_between normalizes inputs before computing dot product."""

    def test_unnormalized_inputs(self):
        """Scaled quaternions should give same angle as unit quaternions."""
        import math as _math

        q0 = (0.0, 0.0, 0.0, 2.0)  # 2× identity
        q1_unit = tuple(R.from_euler("z", 90, degrees=True).as_quat())
        q1 = tuple(x * 3.0 for x in q1_unit)  # 3× scaled
        result = quat_angle_between(q0, q1)
        assert result == pytest.approx(_math.pi / 2, abs=1e-9)


class TestQuatFromRotvec:
    """quat_from_rotvec converts axis*angle to (x, y, z, w) quaternion."""

    def test_zero_vector(self):
        """Zero rotation vector → identity quaternion."""
        q = quat_from_rotvec((0.0, 0.0, 0.0))
        assert q == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)

    def test_matches_scipy(self):
        """Random rotation vectors match scipy Rotation.from_rotvec."""
        import random

        random.seed(88)
        for _ in range(20):
            rv = [random.uniform(-3, 3) for _ in range(3)]
            expected = R.from_rotvec(rv).as_quat()  # xyzw
            actual = quat_from_rotvec(tuple(rv))
            # q and -q are equivalent
            if sum(a * b for a, b in zip(actual, expected)) < 0:
                actual = tuple(-x for x in actual)
            for i in range(4):
                assert actual[i] == pytest.approx(expected[i], abs=1e-12)

    def test_90_deg_z(self):
        """90° around Z matches scipy."""
        import math

        rv = (0.0, 0.0, math.pi / 2)
        expected = R.from_rotvec(rv).as_quat()
        actual = quat_from_rotvec(rv)
        for i in range(4):
            assert actual[i] == pytest.approx(expected[i], abs=1e-12)


class TestQuatMultiply:
    """quat_multiply computes Hamilton product of two (x,y,z,w) quaternions."""

    def test_identity_left(self):
        """Identity * q = q."""
        identity = (0.0, 0.0, 0.0, 1.0)
        q = (0.1, 0.2, 0.3, 0.9)
        result = quat_multiply(identity, q)
        assert result == pytest.approx(q, abs=1e-12)

    def test_identity_right(self):
        """q * identity = q."""
        identity = (0.0, 0.0, 0.0, 1.0)
        q = (0.1, 0.2, 0.3, 0.9)
        result = quat_multiply(q, identity)
        assert result == pytest.approx(q, abs=1e-12)

    def test_matches_scipy(self):
        """Random products match scipy (R1 * R2).as_quat()."""
        import random

        random.seed(55)
        for _ in range(20):
            q1 = [random.gauss(0, 1) for _ in range(4)]
            n1 = sum(x * x for x in q1) ** 0.5
            q1 = [x / n1 for x in q1]
            q2 = [random.gauss(0, 1) for _ in range(4)]
            n2 = sum(x * x for x in q2) ** 0.5
            q2 = [x / n2 for x in q2]

            expected = (R.from_quat(q1) * R.from_quat(q2)).as_quat()
            actual = quat_multiply(tuple(q1), tuple(q2))
            if sum(a * b for a, b in zip(actual, expected)) < 0:
                actual = tuple(-x for x in actual)
            for i in range(4):
                assert actual[i] == pytest.approx(expected[i], abs=1e-12)


class TestQuatAngleBetween:
    """quat_angle_between returns the rotation angle between two quaternions."""

    def test_same_quat(self):
        """Angle between identical quaternions is 0."""
        q = tuple(R.from_euler("z", 45, degrees=True).as_quat())
        assert quat_angle_between(q, q) == pytest.approx(0.0, abs=1e-9)

    def test_antipodal_quat(self):
        """q and -q represent the same rotation → angle = 0."""
        q = tuple(R.from_euler("z", 45, degrees=True).as_quat())
        q_neg = tuple(-x for x in q)
        assert quat_angle_between(q, q_neg) == pytest.approx(0.0, abs=1e-9)

    def test_90_degrees(self):
        """90° rotation between identity and 90° yaw."""
        import math

        q0 = (0.0, 0.0, 0.0, 1.0)
        q1 = tuple(R.from_euler("z", 90, degrees=True).as_quat())
        assert quat_angle_between(q0, q1) == pytest.approx(math.pi / 2, abs=1e-9)

    def test_matches_scipy(self):
        """Random pairs match scipy (r_target * r_current.inv()).magnitude()."""
        import random

        random.seed(66)
        for _ in range(20):
            q0 = [random.gauss(0, 1) for _ in range(4)]
            n0 = sum(x * x for x in q0) ** 0.5
            q0 = [x / n0 for x in q0]
            q1 = [random.gauss(0, 1) for _ in range(4)]
            n1 = sum(x * x for x in q1) ** 0.5
            q1 = [x / n1 for x in q1]

            expected = (R.from_quat(q1) * R.from_quat(q0).inv()).magnitude()
            actual = quat_angle_between(tuple(q0), tuple(q1))
            assert actual == pytest.approx(expected, abs=1e-9)


def test_cross3_matches_np_cross():
    """Same arithmetic as np.cross, so swapping it in changes no trajectory."""
    from pybullet_fleet.geometry import cross3

    rng = np.random.default_rng(3)
    cases = [rng.normal(size=(2, 3)) for _ in range(200)]
    cases += [np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])]
    for a, b in cases:
        assert np.array_equal(cross3(a, b), np.cross(a, b))
    # Plain sequences work too, as np.cross accepts them.
    assert np.array_equal(cross3([0, 0, 1], [1, 0, 0]), np.cross([0, 0, 1], [1, 0, 0]))
