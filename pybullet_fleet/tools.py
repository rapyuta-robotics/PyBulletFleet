from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any, Iterable, TYPE_CHECKING
import logging
import math

import numpy as np
import pybullet as p

from pybullet_fleet.geometry import rotate_vector  # used by body_to_world_velocity_3d

if TYPE_CHECKING:
    from pybullet_fleet.geometry import Pose

# Create logger for this module
logger = logging.getLogger(__name__)


def body_to_world_velocity_2d(
    vx_body: float,
    vy_body: float,
    yaw: float,
) -> Tuple[float, float]:
    """Rotate 2-D body-frame velocity to world frame using yaw only.

    Args:
        vx_body: Forward velocity in body frame.
        vy_body: Lateral velocity in body frame.
        yaw: Current heading in world frame (radians).

    Returns:
        ``(vx, vy)`` in world frame.

    Example::

        >>> body_to_world_velocity_2d(1.0, 0.0, yaw=math.pi / 2)
        (0.0, 1.0)   # facing +Y, forward maps to +Y
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx = vx_body * cos_yaw - vy_body * sin_yaw
    vy = vx_body * sin_yaw + vy_body * cos_yaw
    return vx, vy


def body_to_world_velocity_3d(
    vx_body: float,
    vy_body: float,
    vz_body: float,
    orientation: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    """Rotate 3-D body-frame velocity to world frame using full quaternion.

    Uses the same quaternion convention as PyBullet: ``(x, y, z, w)``.

    Args:
        vx_body: Forward velocity in body frame (body X).
        vy_body: Lateral velocity in body frame (body Y).
        vz_body: Vertical velocity in body frame (body Z).
        orientation: Quaternion ``(x, y, z, w)`` representing robot orientation.

    Returns:
        ``(vx, vy, vz)`` in world frame.

    Example::

        >>> from scipy.spatial.transform import Rotation
        >>> q = Rotation.from_euler("z", 90, degrees=True).as_quat()  # xyzw
        >>> body_to_world_velocity_3d(1.0, 0.0, 0.0, tuple(q))
        (0.0, 1.0, 0.0)
    """
    return rotate_vector((vx_body, vy_body, vz_body), orientation)


def normalize_vector_param(value: Union[float, List[float]], param_name: str, dim: int = 3) -> np.ndarray:
    """
    Normalize a parameter to a numpy array of specified dimension.

    This is a utility function for handling parameters that can be either:
    - A single float (applied to all dimensions)
    - A list of floats (one per dimension)

    Commonly used for velocity/acceleration limits that can be uniform or per-axis.

    Args:
        value: Float (applied to all dimensions) or list of floats (per-dimension)
        param_name: Parameter name for error messages
        dim: Number of dimensions (default: 3)

    Returns:
        Numpy array of shape (dim,)

    Raises:
        ValueError: If list length doesn't match dimension

    Example::

        >>> normalize_vector_param(2.0, "max_vel", 3)
        array([2., 2., 2.])
        >>> normalize_vector_param([1.0, 2.0, 3.0], "max_vel", 3)
        array([1., 2., 3.])
    """
    if isinstance(value, (int, float)):
        return np.array([value] * dim, dtype=float)
    else:
        arr = np.array(value, dtype=float)
        if len(arr) != dim:
            raise ValueError(f"{param_name} must be a float or a list of {dim} floats")
        return arr


def resolve_link_index(body_id: int, link: Union[int, str], physics_client_id: int = 0) -> int:
    """
    Resolve link name to index.

    Args:
        body_id: PyBullet body ID
        link: Link index (int) or name (str). -1 or "base_link" for base link.

    Returns:
        Link index (int). Returns -1 if link is "base_link" or if link name not found.

    Example::

        >>> index = resolve_link_index(robot.body_id, "sensor_mast")
        >>> index = resolve_link_index(robot.body_id, 2)  # Also accepts int
        >>> index = resolve_link_index(robot.body_id, "base_link")  # Returns -1
    """
    if isinstance(link, int):
        return link

    # Handle special case for base link
    if link.lower() == "base_link":
        return -1

    # Search for link by name
    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(body_id, i, physicsClientId=physics_client_id)
        link_name = joint_info[12].decode("utf-8")  # Link name is at index 12
        if link_name == link:
            return i

    # Link not found
    logger.warning(f"Link '{link}' not found on body {body_id}, using base link (-1)")
    return -1


def resolve_joint_index(body_id: int, joint: Union[int, str], physics_client_id: int = 0) -> int:
    """
    Resolve joint name to index.

    Args:
        body_id: PyBullet body ID
        joint: Joint index (int) or name (str)
        physics_client_id: The client the body lives on (PyBullet's default otherwise)

    Returns:
        Joint index (int). Returns -1 if joint name not found.

    Example::

        >>> idx = resolve_joint_index(robot.body_id, "elbow_joint")
        >>> idx = resolve_joint_index(robot.body_id, 2)
    """
    if isinstance(joint, int):
        return joint
    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(body_id, i, physicsClientId=physics_client_id)
        joint_name = joint_info[1].decode("utf-8")  # Joint name is at index 1
        if joint_name == joint:
            return i
    logger.warning(f"Joint '{joint}' not found on body {body_id}, returning -1")
    return -1


def calculate_offset_pose(
    target_position: List[float], current_position: List[float], offset: float, keep_height: bool = True
) -> "Pose":
    """
    Calculate pose at specified offset distance from target.

    This unified function can be used for various offset calculations:
    - Approach poses (approach_offset): Position before reaching target
    - Pick poses (pick_offset): Position where pick operation occurs
    - Drop poses (drop_offset): Position where drop operation occurs

    Args:
        target_position: Target position [x, y, z]
        current_position: Current agent position [x, y, z]
        offset: Distance from target (positive = away from target, 0 = at target)
        keep_height: Whether to keep current height (default: True)

    Returns:
        Pose object at the offset position, oriented to face the target

    Example::

        >>> # Approach pose (1.5m away from target)
        >>> approach = calculate_offset_pose([5, 0, 0.1], [0, 0, 0.3], offset=1.5)
        >>>
        >>> # Pick pose (0.3m away from target)
        >>> pick = calculate_offset_pose([5, 0, 0.1], [0, 0, 0.3], offset=0.3)
        >>>
        >>> # At target position (0m offset)
        >>> drop = calculate_offset_pose([5, 0, 0.1], [0, 0, 0.3], offset=0.0)
    """
    from pybullet_fleet.geometry import Pose

    target_pos = np.array(target_position)
    current_pos = np.array(current_position)

    # Calculate direction from current to target
    direction = target_pos - current_pos
    direction[2] = 0  # Project to XY plane

    direction_norm = np.linalg.norm(direction)
    if direction_norm > 1e-6:
        direction = direction / direction_norm
        yaw = np.arctan2(direction[1], direction[0])
    else:
        # Current and target positions are the same (in XY plane)
        # Direction is undefined, cannot apply offset properly
        logger.warning(
            f"calculate_offset_pose: current and target positions are same in XY plane "
            f"(distance={direction_norm:.6f}). Direction is undefined. "
            f"Defaulting to yaw=0 and ignoring offset={offset}."
        )
        direction = np.array([0.0, 0.0, 0.0])
        yaw = 0.0

    # Calculate offset position (away from target by offset distance)
    offset_pos = target_pos - direction * offset

    # Use current height or target height
    z = current_pos[2] if keep_height else target_pos[2]
    offset_pos[2] = z

    # Orientation: face the target with horizontal orientation
    return Pose.from_euler(x=offset_pos[0], y=offset_pos[1], z=offset_pos[2], roll=0.0, pitch=0.0, yaw=yaw)


def world_to_grid(pos: List[float], spacing: List[float], offset: Optional[List[float]] = None) -> List[int]:
    """
    Convert world coordinates to grid indices (always 3D), with offset.

    Args:
        pos: [x, y, z] or [x, y] world coordinates
        spacing: [spacing_x, spacing_y, spacing_z] or [spacing_x, spacing_y] grid spacing
        offset: [offset_x, offset_y, offset_z] or [offset_x, offset_y] offset (optional)

    Returns:
        [ix, iy, iz] grid indices
    """
    x = pos[0]
    y = pos[1]
    z = pos[2] if len(pos) > 2 else 0.0
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2] if len(spacing) > 2 else 1.0
    offset_x = offset[0] if offset and len(offset) > 0 else 0.0
    offset_y = offset[1] if offset and len(offset) > 1 else 0.0
    offset_z = offset[2] if offset and len(offset) > 2 else 0.0
    grid_ix = int(round((x - offset_x) / spacing_x))
    grid_iy = int(round((y - offset_y) / spacing_y))
    grid_iz = int(round((z - offset_z) / spacing_z))
    return [grid_ix, grid_iy, grid_iz]


def grid_to_world(grid: List[int], spacing: List[float], offset: Optional[List[float]] = None) -> List[float]:
    """
    Convert grid indices to world coordinates (always 3D), with offset.

    Args:
        grid: [ix, iy, iz] or [ix, iy] grid indices
        spacing: [spacing_x, spacing_y, spacing_z] or [spacing_x, spacing_y] grid spacing
        offset: [offset_x, offset_y, offset_z] or [offset_x, offset_y] offset (optional)

    Returns:
        [x, y, z] world coordinates
    """
    ix = grid[0]
    iy = grid[1]
    iz = grid[2] if len(grid) > 2 else 0
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2] if len(spacing) > 2 else 1.0
    offset_x = offset[0] if offset and len(offset) > 0 else 0.0
    offset_y = offset[1] if offset and len(offset) > 1 else 0.0
    offset_z = offset[2] if offset and len(offset) > 2 else 0.0
    x = ix * spacing_x + offset_x
    y = iy * spacing_y + offset_y
    z = iz * spacing_z + offset_z
    return [x, y, z]


def world_to_grid_2d(pos: List[float], spacing: List[float]) -> List[int]:
    """
    2D wrapper for world_to_grid.

    Args:
        pos: [x, y] world coordinates
        spacing: [spacing_x, spacing_y] grid spacing

    Returns:
        [ix, iy] grid indices
    """
    return world_to_grid(pos[:2], spacing[:2])


def grid_to_world_2d(grid: List[int], spacing: List[float]) -> List[float]:
    """
    2D wrapper for grid_to_world.

    Args:
        grid: [ix, iy] grid indices
        spacing: [spacing_x, spacing_y] grid spacing

    Returns:
        [x, y] world coordinates
    """
    return grid_to_world(grid[:2], spacing[:2])


def grid_execution(
    grid_num: List[int] = None,
    spacing: List[float] = None,
    offset: Optional[List[float]] = None,
    func: Optional[Callable] = None,
    args: Optional[Dict] = None,
) -> None:
    r"""
    Calls the callback function func(grid_index, world_pos, \*\*args) at each grid point.

    Args:
        grid_num: [x_num, y_num, z_num] number of grids in each axis (default: [1, 1, 1])
        spacing: [spacing_x, spacing_y, spacing_z] grid spacing (default: [1.0, 1.0, 1.0])
        offset: [offset_x, offset_y, offset_z] offset (default: [0.0, 0.0, 0.0])
        func: Callback function(grid_index: List[int], world_pos: List[float], \*\*args)
              - grid_index: [ix, iy, iz] grid indices
              - world_pos: [x, y, z] world coordinates
        args: Additional arguments to pass to the callback function
    """
    if func is None:
        raise ValueError("Please specify func (callback function)")
    if grid_num is None:
        grid_num = [1, 1, 1]
    if spacing is None:
        spacing = [1.0, 1.0, 1.0]
    if offset is None:
        offset = [0.0, 0.0, 0.0]

    x_num, y_num, z_num = grid_num[0], grid_num[1], grid_num[2]
    for ix in range(x_num):
        for iy in range(y_num):
            for iz in range(z_num):
                grid_index = [ix, iy, iz]
                world_pos = grid_to_world(grid_index, spacing, offset)
                if args:
                    func(grid_index, world_pos, **args)
                else:
                    func(grid_index, world_pos)


def find_nearest(
    objects: Iterable[Any],
    position: Sequence[float],
    search_radius: float,
    predicate: Optional[Callable[[Any], bool]] = None,
    exclude: Optional[Any] = None,
    use_2d: bool = False,
) -> Optional[Any]:
    """Find the nearest object from *position* within *search_radius*.

    A generic spatial nearest-neighbour query over objects that expose
    ``get_pose().position``.  Filtering is done via an optional
    *predicate* callable and an *exclude* instance.

    Args:
        objects: Iterable of objects with ``get_pose().position`` attribute.
        position: ``[x, y, z]`` reference position in world frame.
        search_radius: Maximum search distance (metres).
        predicate: Optional filter — called with each object, return
                   ``True`` to include.  When ``None`` all objects are
                   candidates.
        exclude: Single object instance to skip (typically ``self``).
        use_2d: When ``True``, compare only XY distance (ignore Z).
                Useful for finding items on tables/shelves near a
                floor-level waypoint.

    Returns:
        The nearest matching object, or ``None`` if nothing found.

    Example::

        # Find nearest pickable, unattached object from a world position
        obj = find_nearest(
            sim_core.sim_objects,
            position=[5, 0, 0],
            search_radius=2.0,
            predicate=lambda o: o.pickable and not o.is_attached(),
            exclude=agent,
        )
    """
    ref = np.array(position, dtype=float)
    min_dist = search_radius
    nearest: Optional[Any] = None

    for obj in objects:
        if obj is exclude:
            continue
        if predicate is not None and not predicate(obj):
            continue
        obj_pos = np.array(obj.get_pose().position, dtype=float)
        if use_2d:
            dist = float(np.linalg.norm(obj_pos[:2] - ref[:2]))
        else:
            dist = float(np.linalg.norm(obj_pos - ref))
        if dist < min_dist:
            min_dist = dist
            nearest = obj

    return nearest
