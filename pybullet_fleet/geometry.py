"""Geometric primitives: Pose, Path, and quaternion slerp utilities."""

import math
from dataclasses import dataclass, field
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

#: Quaternion in ``(x, y, z, w)`` convention — same as PyBullet.
Quat = Tuple[float, float, float, float]
#: 3-D vector ``(x, y, z)``.
Vec3 = Tuple[float, float, float]

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

#: Scalar limit or per-axis sequence (used for kinematic limits and similar).
ScalarOrAxes = Union[float, Sequence[float], np.ndarray]


def is_scalar(v: ScalarOrAxes) -> bool:
    """Return ``True`` if *v* should be treated as a scalar (broadcastable) value.

    Accepts Python ``int``/``float``, numpy numeric scalars, or a 0-d numpy
    array. Anything else (list, tuple, 1-d ndarray, …) is treated as per-axis.
    """
    if isinstance(v, (int, float, np.number)):
        return True
    return isinstance(v, np.ndarray) and getattr(v, "ndim", None) == 0


def as_axes(v: ScalarOrAxes, dim: int = 3) -> np.ndarray:
    """Coerce *v* to a ``(dim,)`` numpy array, broadcasting scalars.

    Raises:
        ValueError: If *v* is array-like but its shape is not ``(dim,)``.
    """
    if is_scalar(v):
        return np.full(dim, float(v), dtype=float)  # type: ignore[arg-type]
    arr = np.asarray(v, dtype=float)
    if arr.shape != (dim,):
        raise ValueError(f"Expected scalar or length-{dim} sequence; got shape {arr.shape}.")
    return arr


def projected_axis_limit(limit: ScalarOrAxes, direction_unit: np.ndarray) -> float:
    """Direction-projected scalar limit from per-axis caps.

    Given a per-axis cap ``arr`` and a unit direction ``d``, returns the
    largest scalar ``v`` such that ``|d[i]| * v <= arr[i]`` for every axis::

        v_max = 1 / max_i ( |d[i]| / arr[i] )

    Used to convert per-axis kinematic limits (velocity or acceleration) into
    the equivalent scalar bound along a given travel direction.

    Args:
        limit: Scalar (returned as-is) or per-axis cap ``(dim,)``.
        direction_unit: Unit direction vector ``(dim,)``. A zero vector
            (no constraining axis) returns ``math.inf``.

    Returns:
        Effective scalar limit along ``direction_unit``.
    """
    if is_scalar(limit):
        return float(limit)  # type: ignore[arg-type]
    arr = as_axes(limit, dim=direction_unit.shape[0])
    abs_dir = np.abs(direction_unit)
    # ratio_i = |d_i| / arr_i. Axes the direction does not touch (|d_i| == 0)
    # never constrain — even when their cap is 0 (a 0/0 that must read as 0, not
    # NaN; e.g. max_linear_vel [0.3, 3.0, 0.0] moving in the XY plane). An axis
    # with |d_i| > 0 but a 0 cap forbids motion → inf ratio → 0 effective limit.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(abs_dir == 0.0, 0.0, abs_dir / arr)
    max_ratio = float(np.max(ratios))
    if max_ratio == 0.0:
        return math.inf
    return 0.0 if math.isinf(max_ratio) else 1.0 / max_ratio


def clamp_vec_to_limit(vec: np.ndarray, limit: ScalarOrAxes) -> np.ndarray:
    """Clamp a vector against a scalar or per-axis cap.

    Semantics depend on the shape of *limit*:

    * **Scalar** ⇒ Euclidean magnitude clamp: if ``|vec| > limit``, the vector
      is scaled down to length ``limit`` while preserving direction. Acts
      as projection onto a sphere of radius ``limit``.
    * **Per-axis** ⇒ component-wise clamp: each ``vec[i]`` is independently
      clipped to ``[-limit[i], +limit[i]]``. Acts as projection onto an
      axis-aligned box.

    Non-finite scalar limits (``math.inf``) short-circuit and return *vec*
    unchanged.

    Args:
        vec: Vector ``(dim,)`` to clamp (e.g. body-frame linear or angular
            velocity).
        limit: Scalar magnitude cap or per-axis cap ``(dim,)``.

    Returns:
        Clamped vector (may be *vec* itself when already within bounds).
    """
    if is_scalar(limit):
        vmax = float(limit)  # type: ignore[arg-type]
        if not math.isfinite(vmax):
            return vec
        mag = float(np.linalg.norm(vec))
        if mag > vmax:
            return vec * (vmax / mag)
        return vec
    arr = as_axes(limit, dim=vec.shape[0])
    return np.clip(vec, -arr, arr)


@dataclass
class Pose:
    """
    Position and orientation representation for any object in the simulation.
    Compatible with ROS2 geometry_msgs/Pose and PyBullet's (position, orientation) tuples.

    Supports both quaternion and Euler angle representations.

    Attributes:
        position: [x, y, z] in world coordinates
        orientation: [x, y, z, w] quaternion (defaults to no rotation)
    """

    position: List[float]  # [x, y, z]
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w] quaternion

    # Convenient property accessors
    @property
    def x(self) -> float:
        """X coordinate."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Y coordinate."""
        return self.position[1]

    @property
    def z(self) -> float:
        """Z coordinate."""
        return self.position[2]

    @property
    def roll(self) -> float:
        """Roll angle in radians (rotation around x-axis)."""
        return p.getEulerFromQuaternion(self.orientation)[0]

    @property
    def pitch(self) -> float:
        """Pitch angle in radians (rotation around y-axis)."""
        return p.getEulerFromQuaternion(self.orientation)[1]

    @property
    def yaw(self) -> float:
        """Yaw angle in radians (rotation around z-axis)."""
        return p.getEulerFromQuaternion(self.orientation)[2]

    def __str__(self) -> str:
        """Concise string: ``Pose(x=1.00, y=2.00, z=0.00, yaw=90.0°)``."""
        yaw_deg = np.degrees(self.yaw)
        return f"Pose(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, yaw={yaw_deg:.1f}°)"

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float):
        """Create Pose from x, y, z coordinates."""
        return cls(position=[x, y, z])

    @classmethod
    def from_euler(cls, x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        """
        Create Pose from position and Euler angles.

        Args:
            x, y, z: Position in world coordinates
            roll, pitch, yaw: Euler angles in radians
        """
        quat = p.getQuaternionFromEuler([roll, pitch, yaw])
        return cls(position=[x, y, z], orientation=list(quat))

    @classmethod
    def from_pybullet(cls, position: Tuple[float, float, float], orientation: Tuple[float, float, float, float]):
        """Create Pose from PyBullet position and orientation tuples."""
        return cls(position=list(position), orientation=list(orientation))

    def as_position(self) -> List[float]:
        """Return position as list [x, y, z]."""
        return self.position

    def as_orientation(self) -> List[float]:
        """Return orientation as quaternion [x, y, z, w]."""
        return self.orientation

    def as_position_orientation(self) -> Tuple[List[float], List[float]]:
        """Return (position, orientation) tuple for PyBullet."""
        return self.position, self.orientation

    def as_tuple(self) -> Tuple[List[float], List[float]]:
        """
        Return (position, orientation) tuple.
        Alias for as_position_orientation() for convenience.
        """
        return self.position, self.orientation

    def as_euler(self) -> Tuple[float, float, float]:
        """Return orientation as Euler angles (roll, pitch, yaw)."""
        return p.getEulerFromQuaternion(self.orientation)

    @classmethod
    def from_yaw(cls, x: float, y: float, z: float, yaw: float):
        """
        Create Pose from position and yaw angle (rotation around z-axis).

        Args:
            x, y, z: Position in world coordinates
            yaw: Yaw angle in radians

        Returns:
            Pose instance
        """
        # Use PyBullet to convert yaw to quaternion (roll=0, pitch=0, yaw=yaw)
        quaternion = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        return cls(position=[x, y, z], orientation=list(quaternion))


@dataclass
class Path:
    """
    Represents a path as a sequence of waypoints (Poses).

    Attributes:
        waypoints: List of Pose objects representing the path

    Example::

        # Create from Pose list (recommended)
        path = Path([
            Pose.from_xyz(0, 0, 0),
            Pose.from_xyz(1, 0, 0),
            Pose.from_xyz(1, 1, 0)
        ])

        # Create from positions
        path = Path.from_positions([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        # Create predefined shapes
        circle = Path.create_circle(center=[0, 0], radius=1.5)
        square = Path.create_square(center=[0, 0], side_length=2.0)
    """

    waypoints: List[Pose]

    def __str__(self) -> str:
        """Multi-line string showing each waypoint.

        Example::

            Path(3 waypoints):
              [0] Pose(x=0.00, y=0.00, z=0.00, yaw=0.0°)
              [1] Pose(x=1.00, y=0.00, z=0.00, yaw=0.0°)
              [2] Pose(x=1.00, y=1.00, z=0.00, yaw=90.0°)
        """
        lines = [f"Path({len(self.waypoints)} waypoints):"]
        for i, wp in enumerate(self.waypoints):
            lines.append(f"  [{i}] {wp}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of waypoints."""
        return len(self.waypoints)

    def __getitem__(self, index: int) -> Pose:
        """Get waypoint at index."""
        return self.waypoints[index]

    def __iter__(self):
        """Iterate over waypoints."""
        return iter(self.waypoints)

    @classmethod
    def from_positions(cls, positions: List[List[float]], orientation: Optional[List[float]] = None) -> "Path":
        """
        Create Path from list of positions with optional shared orientation.

        Args:
            positions: List of [x, y, z] positions
            orientation: Optional quaternion [x, y, z, w] applied to all waypoints
                        (default: [0, 0, 0, 1] - no rotation)

        Returns:
            Path instance

        Example::

            path = Path.from_positions([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0]
            ])
        """
        if orientation is None:
            orientation = [0.0, 0.0, 0.0, 1.0]

        waypoints = [Pose(position=list(pos), orientation=list(orientation)) for pos in positions]
        return cls(waypoints=waypoints)

    @staticmethod
    def _calculate_orientation_for_plane(direction: np.ndarray, plane_normal: np.ndarray) -> List[float]:
        """
        Calculate quaternion orientation where X+ points in direction and Z+ points in plane_normal.

        Args:
            direction: 3D direction vector (will be normalized) - robot's X+ axis
            plane_normal: 3D plane normal vector (will be normalized) - robot's Z+ axis

        Returns:
            Quaternion [x, y, z, w] that aligns X+ with direction and Z+ with plane_normal
        """
        # Normalize inputs
        x_axis = direction / np.linalg.norm(direction)
        z_axis = plane_normal / np.linalg.norm(plane_normal)

        # Build Y-axis (right direction) using right-hand rule: Y = Z × X
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)

        if y_norm < 1e-6:
            # Edge case: direction parallel to plane normal
            # Choose arbitrary perpendicular Y axis
            if abs(x_axis[0]) < 0.9:
                y_axis = np.cross(z_axis, np.array([1, 0, 0]))
            else:
                y_axis = np.cross(z_axis, np.array([0, 1, 0]))
            y_axis = y_axis / np.linalg.norm(y_axis)
        else:
            y_axis = y_axis / y_norm

        # Recompute Z to ensure orthogonality: Z = X × Y
        z_axis = np.cross(x_axis, y_axis)

        # Build rotation matrix [X, Y, Z] as columns
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to quaternion using scipy
        rot = R.from_matrix(rot_matrix)
        quat = rot.as_quat()  # [x, y, z, w]
        return quat.tolist()

    @classmethod
    def create_rectangle(
        cls,
        center: List[float],
        width: float,
        height: float,
        rpy: List[float] = [0.0, 0.0, 0.0],
    ) -> "Path":
        """Create a rectangular path in 3D.

        Args:
            center: [x, y, z] center position in world coordinates
            width: Width of rectangle (X direction in local frame)
            height: Height of rectangle (Y direction in local frame)
            rpy: [roll, pitch, yaw] in radians. Defaults to [0, 0, 0].
                 When non-zero, the Z+ axis of the robot will be
                 perpendicular to the rectangle plane.

        Returns:
            Path instance with waypoints at rectangle corners

        Example::

            rect = Path.create_rectangle(center=[0, 0, 1], width=3.0, height=2.0)
            tilted_rect = Path.create_rectangle(center=[0, 0, 1], width=3.0, height=2.0,
                                                rpy=[0.1, 0.2, 0])
        """
        if len(center) != 3:
            raise ValueError("Path.create_rectangle: center must be [x, y, z]")

        if len(rpy) != 3:
            raise ValueError("Path.create_rectangle: rpy must be [roll, pitch, yaw]")

        roll, pitch, yaw = rpy

        half_width = width / 2.0
        half_height = height / 2.0

        # Local coordinates of rectangle corners in the XY plane (z=0) around origin
        local_points = np.array(
            [
                [-half_width, -half_height, 0.0],  # Bottom-left
                [half_width, -half_height, 0.0],  # Bottom-right
                [half_width, half_height, 0.0],  # Top-right
                [-half_width, half_height, 0.0],  # Top-left
                [-half_width, -half_height, 0.0],  # Back to start
            ]
        )

        # Build rotation using scipy (more reliable than manual calculation)
        rot = R.from_euler("XYZ", [roll, pitch, yaw])

        # Calculate plane normal from rotation (Z+ axis)
        plane_normal = rot.as_matrix()[:, 2]

        # Center position in world coordinates
        center_world = np.array(center)

        # Transform points to world coordinates
        positions = []
        for pt in local_points:
            world_pt = center_world + rot.as_matrix().dot(pt)
            positions.append(world_pt.tolist())

        # Calculate orientation for each waypoint based on travel direction
        waypoints = []
        for i in range(len(positions)):
            pos = positions[i]

            # Calculate direction to next waypoint
            next_i = (i + 1) % len(positions)
            direction = np.array(positions[next_i]) - np.array(pos)
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-6:
                # Calculate orientation with X+ pointing in travel direction, Z+ perpendicular to plane
                quat = cls._calculate_orientation_for_plane(direction, plane_normal)
            else:
                # For duplicate points (e.g., closing point of path), use previous waypoint's orientation
                if i > 0:
                    quat = waypoints[-1].orientation
                else:
                    quat = [0.0, 0.0, 0.0, 1.0]

            waypoints.append(Pose(position=pos, orientation=quat))

        return cls(waypoints=waypoints)

    @classmethod
    def create_square(
        cls,
        center: List[float],
        side_length: float,
        rpy: List[float] = [0.0, 0.0, 0.0],
    ) -> "Path":
        """Create a square path in 3D.

        Args:
            center: [x, y, z] center position in world coordinates
            side_length: Length of square sides
            rpy: [roll, pitch, yaw] in radians. Defaults to [0, 0, 0].
                 When non-zero, the Z+ axis of the robot will be
                 perpendicular to the square plane.

        Returns:
            Path instance with waypoints at square corners

        Example::

            square = Path.create_square(center=[0, 0, 0], side_length=2.0)
            tilted_square = Path.create_square(center=[0, 0, 1], side_length=2.0,
                                              rpy=[0.1, 0.2, 0])

        Note::

            This is a wrapper for create_rectangle with width == height.
        """
        return cls.create_rectangle(center=center, width=side_length, height=side_length, rpy=rpy)

    @classmethod
    def create_ellipse(
        cls,
        center: List[float],
        radius_x: float,
        radius_y: float,
        num_points: int = 16,
        rpy: List[float] = [0.0, 0.0, 0.0],
        start_angle: float = 0.0,
        end_angle: float = 2.0 * np.pi,
    ) -> "Path":
        """Create an elliptical path in 3D.

        Args:
            center: [x, y, z] center position in world coordinates
            radius_x: Radius along X axis in local frame
            radius_y: Radius along Y axis in local frame
            num_points: Number of waypoints along the arc
            rpy: [roll, pitch, yaw] in radians. Defaults to [0, 0, 0].
            start_angle: Start angle [rad] in the local XY-plane
            end_angle: End angle [rad] in the local XY-plane

        Returns:
            Path instance with waypoints along ellipse

        Example::

            ellipse = Path.create_ellipse(center=[0, 0, 0], radius_x=2.0, radius_y=1.0)
            tilted_ellipse = Path.create_ellipse(center=[0, 0, 1], radius_x=2.0, radius_y=1.0,
                                                 rpy=[0, 0.3, 0])
        """
        if len(center) != 3:
            raise ValueError("Path.create_ellipse: center must be [x, y, z]")

        if len(rpy) != 3:
            raise ValueError("Path.create_ellipse: rpy must be [roll, pitch, yaw]")

        roll, pitch, yaw = rpy

        # Local ellipse/arc points in XY-plane with z=0
        local_points = []
        # Ensure we always include the end point by sampling num_points
        # segments between start_angle and end_angle.
        for i in range(num_points + 1):
            t = i / float(num_points)
            angle = start_angle + (end_angle - start_angle) * t
            x = radius_x * np.cos(angle)
            y = radius_y * np.sin(angle)
            local_points.append([x, y, 0.0])

        local_points = np.array(local_points)

        # Build rotation using scipy
        rot = R.from_euler("XYZ", [roll, pitch, yaw])

        # Calculate plane normal from rotation (Z+ axis)
        plane_normal = rot.as_matrix()[:, 2]

        # Center position in world coordinates
        center_world = np.array(center)

        positions = []
        for pt in local_points:
            world_pt = center_world + rot.as_matrix().dot(pt)
            positions.append(world_pt.tolist())

        # Calculate orientation for each waypoint based on travel direction
        waypoints = []
        for i in range(len(positions)):
            pos = positions[i]

            # Calculate direction to next waypoint
            next_i = (i + 1) % len(positions)
            direction = np.array(positions[next_i]) - np.array(pos)
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-6:
                # Calculate orientation with X+ pointing in travel direction, Z+ perpendicular to plane
                quat = cls._calculate_orientation_for_plane(direction, plane_normal)
            else:
                # Fallback for duplicate points (e.g., last point == first point)
                if i > 0:
                    # Use previous waypoint's orientation
                    quat = waypoints[i - 1].orientation
                else:
                    quat = [0.0, 0.0, 0.0, 1.0]

            waypoints.append(Pose(position=pos, orientation=quat))

        return cls(waypoints=waypoints)

    @classmethod
    def create_circle(
        cls,
        center: List[float],
        radius: float,
        num_points: int = 16,
        rpy: List[float] = [0.0, 0.0, 0.0],
        start_angle: float = 0.0,
        end_angle: float = 2.0 * np.pi,
    ) -> "Path":
        """Create a circular path in 3D.

        Args:
            center: [x, y, z] center position in world coordinates
            radius: Circle radius
            num_points: Number of waypoints along the arc
            rpy: [roll, pitch, yaw] in radians. Defaults to [0, 0, 0].
            start_angle: Start angle [rad] in the local XY-plane
            end_angle: End angle [rad] in the local XY-plane

        Returns:
            Path instance with waypoints along circle

        Example::

            circle = Path.create_circle(center=[0, 0, 0], radius=1.5)
            arc = Path.create_circle(center=[0, 0, 0], radius=1.5,
                                    start_angle=0, end_angle=np.pi)

        Note::

            This is a wrapper for create_ellipse with radius_x == radius_y.
        """
        return cls.create_ellipse(
            center=center,
            radius_x=radius,
            radius_y=radius,
            num_points=num_points,
            rpy=rpy,
            start_angle=start_angle,
            end_angle=end_angle,
        )

    # --------------------------------------------------------------
    # Path composition utilities
    # --------------------------------------------------------------

    def append(self, other: "Path") -> None:
        """Append waypoints from another path in-place.

        Note::

            This mutates the current Path. For a non-mutating version,
            use `combined = path1 + path2`.
        """

        self.waypoints.extend(other.waypoints)

    def __add__(self, other: "Path") -> "Path":
        """Concatenate two paths and return a new Path.

        The resulting path simply has waypoints = self.waypoints + other.waypoints.
        """

        return Path(waypoints=list(self.waypoints) + list(other.waypoints))

    @classmethod
    def from_paths(cls, paths: Iterable["Path"]) -> "Path":
        """Create a single Path by concatenating multiple paths.

        Example::

            circle_arc = Path.create_circle(center=[0, 0, 0.5], radius=2.0,
                                           num_points=16,
                                           start_angle=0.0,
                                           end_angle=np.pi/2)
            line = Path.from_positions([[1, 1, 0.5], [3, 1, 0.5]])
            full = Path.from_paths([circle_arc, line])
        """

        waypoints: List[Pose] = []
        for path in paths:
            waypoints.extend(path.waypoints)
        return cls(waypoints=waypoints)

    def get_total_distance(self) -> float:
        """
        Calculate total path length.

        Returns:
            Total distance in meters
        """
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            p1 = np.array(self.waypoints[i].position)
            p2 = np.array(self.waypoints[i + 1].position)
            total += np.linalg.norm(p2 - p1)

        return total

    def visualize(
        self,
        show_lines: bool = True,
        line_color: List[float] = None,
        line_width: float = 2.0,
        show_waypoints: bool = False,
        show_axes: bool = False,
        axis_length: float = 0.3,
        show_points: bool = False,
        point_size: float = 0.05,
        lifetime: float = 0.0,
    ) -> List[int]:
        """
        Visualize path with lines and optional waypoint markers.

        Args:
            show_lines: If True, draw lines connecting waypoints
            line_color: RGB color for path lines [r, g, b] (0-1 range), None for default
            line_width: Width of path lines
            show_waypoints: If True, show waypoint coordinate axes and points
            show_axes: If True, draw X, Y, Z axes at each waypoint
            axis_length: Length of coordinate axes in meters
            show_points: If True, draw spheres at waypoint positions
            point_size: Radius of waypoint spheres in meters
            lifetime: Debug line lifetime (0 = permanent until reset)

        Returns:
            List of debug item IDs (for later removal if needed)

        Example::

            # Just show path lines
            path = Path.create_circle(center=[0, 0, 0.5], radius=2.0)
            path.visualize(line_color=[1, 0, 0])

            # Show path with waypoint orientations
            path.visualize(show_waypoints=True, show_axes=True)
        """
        debug_ids = []

        # Draw path lines
        if show_lines and len(self.waypoints) >= 2:
            # Default color if not specified
            if line_color is None:
                line_color = [0.5, 0.5, 0.5]

            for i in range(len(self.waypoints) - 1):
                p1 = self.waypoints[i].position
                p2 = self.waypoints[i + 1].position
                debug_id = p.addUserDebugLine(p1, p2, lineColorRGB=line_color, lineWidth=line_width, lifeTime=lifetime)
                debug_ids.append(debug_id)

        # Draw waypoint markers if requested
        if show_waypoints:
            waypoint_ids = self.visualize_waypoints(
                show_axes=show_axes,
                axis_length=axis_length,
                show_points=show_points,
                point_size=point_size,
                lifetime=lifetime,
            )
            debug_ids.extend(waypoint_ids)

        return debug_ids

    def visualize_waypoints(
        self,
        show_axes: bool = True,
        axis_length: float = 0.3,
        show_points: bool = True,
        point_size: float = 0.05,
        lifetime: float = 0.0,
    ) -> List[int]:
        """
        Visualize waypoints with coordinate axes for debugging.

        Args:
            show_axes: If True, draw X, Y, Z axes at each waypoint
            axis_length: Length of coordinate axes in meters
            show_points: If True, draw spheres at waypoint positions
            point_size: Radius of waypoint spheres in meters
            lifetime: Debug line lifetime (0 = permanent until reset)

        Returns:
            List of debug item IDs (for later removal if needed)

        Example::

            path = Path.create_square(center=[0, 0, 1], side_length=2.0, rpy=[0.1, 0.2, 0])
            debug_ids = path.visualize_waypoints(axis_length=0.5)
        """
        debug_ids = []

        for i, waypoint in enumerate(self.waypoints):
            pos = waypoint.position

            # Draw point at waypoint
            if show_points:
                # Use small sphere for waypoint marker
                for offset in [
                    [-point_size, 0, 0],
                    [point_size, 0, 0],
                    [0, -point_size, 0],
                    [0, point_size, 0],
                    [0, 0, -point_size],
                    [0, 0, point_size],
                ]:
                    p1 = [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]]
                    debug_id = p.addUserDebugLine(pos, p1, lineColorRGB=[1, 1, 1], lineWidth=3.0, lifeTime=lifetime)
                    debug_ids.append(debug_id)

            # Draw coordinate axes
            if show_axes:
                # Get rotation matrix from quaternion
                from scipy.spatial.transform import Rotation as R

                rot = R.from_quat(waypoint.orientation)
                rot_matrix = rot.as_matrix()

                # Extract axis vectors
                x_axis = rot_matrix[:, 0]  # X+ (forward/red)
                y_axis = rot_matrix[:, 1]  # Y+ (right/green)
                z_axis = rot_matrix[:, 2]  # Z+ (up/blue)

                # Draw X axis (red)
                x_end = [pos[0] + x_axis[0] * axis_length, pos[1] + x_axis[1] * axis_length, pos[2] + x_axis[2] * axis_length]
                debug_id = p.addUserDebugLine(
                    pos,
                    x_end,
                    lineColorRGB=[1, 0, 0],  # Red
                    lineWidth=5.0,  # Thicker line for better visibility
                    lifeTime=lifetime,
                )
                debug_ids.append(debug_id)

                # Draw Y axis (green)
                y_end = [pos[0] + y_axis[0] * axis_length, pos[1] + y_axis[1] * axis_length, pos[2] + y_axis[2] * axis_length]
                debug_id = p.addUserDebugLine(
                    pos,
                    y_end,
                    lineColorRGB=[0, 1, 0],  # Green
                    lineWidth=5.0,  # Thicker line for better visibility
                    lifeTime=lifetime,
                )
                debug_ids.append(debug_id)

                # Draw Z axis (blue)
                z_end = [pos[0] + z_axis[0] * axis_length, pos[1] + z_axis[1] * axis_length, pos[2] + z_axis[2] * axis_length]
                debug_id = p.addUserDebugLine(
                    pos,
                    z_end,
                    lineColorRGB=[0, 0, 1],  # Blue
                    lineWidth=5.0,  # Thicker line for better visibility
                    lifeTime=lifetime,
                )
                debug_ids.append(debug_id)

                # Add waypoint number label
                debug_id = p.addUserDebugText(
                    f"WP{i}",
                    [pos[0], pos[1], pos[2] + axis_length * 1.2],
                    textColorRGB=[1, 1, 1],
                    textSize=1.0,
                    lifeTime=lifetime,
                )
                debug_ids.append(debug_id)

        return debug_ids


# ============================================================================
# Lightweight quaternion rotation (no SciPy Rotation object on hot paths)
# ============================================================================


def _normalize_quat(quat: Quat) -> Quat:
    """Normalize a quaternion to unit length.

    Args:
        quat: Quaternion ``(x, y, z, w)``.

    Returns:
        Unit quaternion ``(x, y, z, w)``.

    Raises:
        ValueError: If the quaternion has zero norm.
    """
    qx, qy, qz, qw = quat
    norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
    if norm_sq == 0.0:
        raise ValueError("zero-norm quaternion is not allowed")
    inv_norm = 1.0 / math.sqrt(norm_sq)
    return (qx * inv_norm, qy * inv_norm, qz * inv_norm, qw * inv_norm)


def quat_to_rot_matrix(
    quat: Quat,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Convert a quaternion ``(x, y, z, w)`` to a 3×3 rotation matrix.

    Returns the matrix as a flat 9-element tuple in **row-major** order::

        (r00, r01, r02,  r10, r11, r12,  r20, r21, r22)

    This is allocation-free and suitable for hot-path use.

    Args:
        quat: Quaternion ``(x, y, z, w)`` — same convention as PyBullet.
              Automatically normalized if not already unit-length.

    Returns:
        9-element tuple representing the 3×3 rotation matrix (row-major).

    Raises:
        ValueError: If the quaternion has zero norm.
    """
    qx, qy, qz, qw = _normalize_quat(quat)
    return (
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy - qz * qw),
        2.0 * (qx * qz + qy * qw),
        2.0 * (qx * qy + qz * qw),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz - qx * qw),
        2.0 * (qx * qz - qy * qw),
        2.0 * (qy * qz + qx * qw),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )


def rotate_vector(vec: Vec3, quat: Quat) -> Vec3:
    """Rotate a 3-D vector by a quaternion.

    Equivalent to ``scipy.spatial.transform.Rotation.from_quat(quat).apply(vec)``
    but avoids SciPy object allocation, making it suitable for per-tick hot paths.

    Args:
        vec: 3-D vector ``(x, y, z)``.
        quat: Quaternion ``(x, y, z, w)`` — same convention as PyBullet.

    Returns:
        Rotated vector ``(x, y, z)``.
    """
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = quat_to_rot_matrix(quat)
    vx, vy, vz = vec
    return (
        r00 * vx + r01 * vy + r02 * vz,
        r10 * vx + r11 * vy + r12 * vz,
        r20 * vx + r21 * vy + r22 * vz,
    )


def quat_from_rotvec(rotvec: Vec3) -> Quat:
    """Convert a rotation vector (axis × angle) to a quaternion ``(x, y, z, w)``.

    Equivalent to ``scipy.spatial.transform.Rotation.from_rotvec(rotvec).as_quat()``.

    Args:
        rotvec: 3-D rotation vector whose direction is the rotation axis
                and magnitude is the rotation angle in radians.

    Returns:
        Quaternion ``(x, y, z, w)`` — same convention as PyBullet.
    """
    rx, ry, rz = rotvec
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    half = 0.5 * angle
    s = math.sin(half) / angle
    return (rx * s, ry * s, rz * s, math.cos(half))


def quat_multiply(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product of two quaternions in ``(x, y, z, w)`` convention.

    Equivalent to ``(Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()``.
    For active rotations this corresponds to applying *q2* first, then *q1*.

    Args:
        q1: Left quaternion ``(x, y, z, w)``.
        q2: Right quaternion ``(x, y, z, w)``.

    Returns:
        Product quaternion ``(x, y, z, w)``.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_angle_between(q0: Quat, q1: Quat) -> float:
    """Compute the rotation angle (radians) between two quaternions.

    Equivalent to ``(Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()).magnitude()``.
    Handles the double-cover: ``q`` and ``-q`` represent the same rotation.
    Inputs are normalized internally, so non-unit quaternions are accepted.

    Args:
        q0: First quaternion ``(x, y, z, w)``.
        q1: Second quaternion ``(x, y, z, w)``.

    Returns:
        Rotation angle in ``[0, π]`` radians.

    Raises:
        ValueError: If either quaternion has zero norm.
    """
    x0, y0, z0, w0 = _normalize_quat(q0)
    x1, y1, z1, w1 = _normalize_quat(q1)
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    # Clamp |dot| to [0, 1] for acos safety; abs handles double-cover
    dot_abs = min(abs(dot), 1.0)
    return 2.0 * math.acos(dot_abs)


def rotation_axis_z(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """World-frame ``z`` component of the unit axis of the shortest rotation ``q_from -> q_to``.

    Accepts ``(4,)`` or ``(N, 4)`` arrays of ``(x, y, z, w)`` quaternions and
    returns a scalar array or ``(N,)``. A slerp between the two turns about
    this axis, so its yaw rate is this times the angular speed: ``+1`` for a
    pure counter-clockwise turn about ``z``, ``-1`` clockwise, ``0`` with no
    rotation.
    """
    a = np.asarray(q_to, dtype=np.float64)
    b = np.asarray(q_from, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    # q_rel = q_to * conj(q_from): the world-frame rotation between them.
    w = aw * bw + ax * bx + ay * by + az * bz
    z = -aw * bz - ax * by + ay * bx + az * bw
    z = np.where(w < 0.0, -z, z)  # the shorter way round
    sin_half = np.sqrt(np.maximum(1.0 - np.minimum(w * w, 1.0), 0.0))
    return np.where(sin_half > 1e-12, z / np.where(sin_half > 1e-12, sin_half, 1.0), 0.0)


# ============================================================================
# Fast two-keyframe scalar quaternion slerp
#
# Why not scipy.spatial.transform.Slerp?
# ---------------------------------------
# The main reason SciPy's Slerp appears slow for single-scalar calls is not
# the interpolation formula itself, but the fixed overhead of a general-purpose
# API:
#   - Input normalization, array promotion, and dtype coercion for both
#     scalar and array inputs
#   - Backend selection (including Array API support in recent SciPy)
#   - Shape and range validation
#   - Wrapping the result as a Rotation object
#
# Slerp is designed for array-oriented use cases where many query points are
# interpolated in a single call.  For a hot path that interpolates one scalar
# per tick, this generality becomes overhead.
#
# This implementation is specialized for the two-keyframe case where q0/q1
# are fixed and only t varies each call.  By precomputing dot/theta/sin(theta)
# and using math.sin/cos for scalar arithmetic, the above generalization costs
# are avoided entirely.
#
# Benchmark measurements (environment-dependent, shown as a rough reference):
#   scipy Slerp + as_quat + tolist : ~29 µs/call
#   this implementation + tolist   : ~2.4 µs/call  (~12x)
# Absolute numbers vary with SciPy/NumPy version, CPU, and Python version;
# treat them as indicative, not definitive.
# ============================================================================


class SlerpPrecomp(NamedTuple):
    """Precomputed constants for fast quaternion slerp between two keyframes."""

    dot: float
    theta_0: float
    sin_theta_0: float
    needs_flip: bool


def quat_slerp_precompute(q0: np.ndarray, q1: np.ndarray) -> SlerpPrecomp:
    """Precompute constants for fast scalar quaternion slerp.

    Args:
        q0: Normalized start quaternion [x, y, z, w], shape (4,).
        q1: Normalized end quaternion [x, y, z, w], shape (4,).

    Returns:
        SlerpPrecomp with (dot, theta_0, sin_theta_0, needs_flip).
    """
    dot = float(np.dot(q0, q1))
    # Shortest-path: flip if hemispheres differ
    needs_flip = dot < 0.0
    if needs_flip:
        dot = -dot
    # Clamp to valid acos domain (both sides, for rounding safety)
    dot = max(0.0, min(dot, 1.0))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    return SlerpPrecomp(dot, theta_0, sin_theta_0, needs_flip)


def quat_slerp(
    q0: np.ndarray,
    q1: np.ndarray,
    t: float,
    precomp: SlerpPrecomp,
) -> np.ndarray:
    """Fast scalar quaternion slerp for normalized quaternions.

    Specialized for the two-keyframe case where q0/q1 are fixed and only t
    varies per call.  Uses math.sin/cos for scalar operations instead of numpy.

    Note::

        q0 and q1 must be unit quaternions.  The normal slerp path preserves
        unit length by construction; the near-identical fallback normalizes
        explicitly.

    Args:
        q0: Normalized start quaternion [x, y, z, w], shape (4,).
        q1: Normalized end quaternion [x, y, z, w], shape (4,).
        t: Interpolation parameter (typically [0, 1]; extrapolation permitted).
        precomp: Precomputed constants from quat_slerp_precompute().

    Returns:
        Interpolated quaternion as numpy array [x, y, z, w].
    """
    q1_effective = -q1 if precomp.needs_flip else q1

    # Near-identical quaternions: fall back to normalized lerp.
    # Threshold 1e-8 balances numerical stability of sin division against
    # precision loss from the lerp approximation.
    if precomp.sin_theta_0 < 1e-8:
        result = q0 + t * (q1_effective - q0)
        return result / np.linalg.norm(result)

    theta = precomp.theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - precomp.dot * sin_theta / precomp.sin_theta_0
    s1 = sin_theta / precomp.sin_theta_0
    return s0 * q0 + s1 * q1_effective


def quat_slerp_batch(
    q0: np.ndarray,
    q1: np.ndarray,
    t: np.ndarray,
    dot: np.ndarray,
    theta0: np.ndarray,
    sin_theta0: np.ndarray,
) -> np.ndarray:
    """Vectorised SLERP for N quaternion pairs.

    Evaluates slerp simultaneously for all N agents using NumPy — the
    hot path for :class:`BatchDifferentialController` and
    :class:`BatchOmniController`.

    The caller is responsible for shortest-path correction: ``q1`` must
    already have had its sign flipped where ``dot(q0, q1) < 0``, and
    ``dot`` must be the corrected (non-negative) dot product.

    Args:
        q0: ``(N, 4)`` start quaternions.
        q1: ``(N, 4)`` end quaternions (sign-corrected for shortest path).
        t: ``(N,)`` interpolation parameters in ``[0, 1]``.
        dot: ``(N,)`` dot products ``dot(q0, q1) >= 0``.
        theta0: ``(N,)`` half-angles ``acos(dot)``.
        sin_theta0: ``(N,)`` ``sin(theta0)``.

    Returns:
        ``(N, 4)`` interpolated unit quaternions.
    """
    sin_theta0_safe = np.where(sin_theta0 > 1e-8, sin_theta0, 1.0)
    theta = theta0 * t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0_safe
    s1 = sin_theta / sin_theta0_safe
    result = s0[:, None] * q0 + s1[:, None] * q1

    # Near-identical quaternions: fall back to normalized lerp.
    use_lerp = sin_theta0 <= 1e-8
    if use_lerp.any():
        lerp = q0 + t[:, None] * (q1 - q0)
        norms = np.linalg.norm(lerp, axis=1, keepdims=True)
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        result[use_lerp] = (lerp / norms_safe)[use_lerp]

    return result
