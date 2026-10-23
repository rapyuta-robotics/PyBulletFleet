"""Shared kinematic-planning helpers used by both per-agent and batched controllers.

These helpers de-duplicate the geometry that previously lived in:

* ``OmniController._init_pose_trajectory`` / ``DifferentialController._init_pose_trajectory``
  (per-agent, in :mod:`pybullet_fleet.controller`)
* ``BatchOmniController._begin_waypoint`` / ``BatchDifferentialController._begin_waypoint``
  (vectorised, in :mod:`pybullet_fleet.controllers.batch_omni` / ``batch_differential``)

Two helpers are provided:

* :func:`plan_translation_segment` — computes the (start, displacement,
  total_distance, direction_unit) tuple for a straight-line goto, with
  ``navigation_2d`` z-flattening.
* :func:`align_x_axis_quat` — solves for a goal-orientation quaternion whose
  body-X axis points along a requested world-frame direction; matches the
  prior inline construction (alignment > 0.95 shortcut + cross-product
  fallback for parallel z-axis).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from pybullet_fleet.geometry import cross3
from pybullet_fleet.types import MovementDirection

if TYPE_CHECKING:
    from pybullet_fleet.geometry import Pose


@dataclass(frozen=True)
class TranslationSegment:
    """Result of :func:`plan_translation_segment`."""

    start_pos: np.ndarray  # (3,) float64 — start position (input, not z-flattened)
    displacement: np.ndarray  # (3,) float64 — goal_pos - start_pos (post nav_2d flattening)
    total_distance: float
    direction_unit: np.ndarray  # (3,) float64 — displacement / total_distance, or zeros


def plan_translation_segment(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    navigation_2d: bool,
) -> TranslationSegment:
    """Compute the straight-line segment data for a single goto.

    When ``navigation_2d`` is True, the goal z is clamped to the start z
    before computing displacement (so motion is purely planar). Inputs are
    not mutated.
    """
    start = np.asarray(start_pos, dtype=np.float64).copy()
    goal = np.asarray(goal_pos, dtype=np.float64).copy()
    if navigation_2d:
        goal[2] = start[2]
    displacement = goal - start
    total = float(np.linalg.norm(displacement))
    if total > 1e-9:
        direction_unit = displacement / total
    else:
        direction_unit = np.zeros(3, dtype=np.float64)
    return TranslationSegment(
        start_pos=start,
        displacement=displacement,
        total_distance=total,
        direction_unit=direction_unit,
    )


def align_x_axis_quat(
    start_quat: np.ndarray,
    goal_quat: np.ndarray,
    x_axis_target: np.ndarray,
) -> np.ndarray:
    """Solve for a unit quaternion whose body-X axis points along ``x_axis_target``.

    The algorithm matches the prior inline implementation in
    ``DifferentialController._init_pose_trajectory``:

    1. If the goal orientation's X axis already aligns with the target
       (dot > 0.95), return ``goal_quat`` unchanged — this preserves the
       user-specified roll/yaw at the destination.
    2. Otherwise build an orthonormal basis where:
       - X' = ``x_axis_target``
       - Y' = ``z_goal × X'`` (normalised; fallback to world Z or Y if parallel)
       - Z' = X' × Y'
       and return the quaternion of the resulting rotation matrix.

    ``start_quat`` is currently accepted for API symmetry (caller-side
    decisions sometimes depend on it) but not consumed; if the
    construction degenerates the function still returns a valid quaternion.

    Returns:
        ``(4,)`` float64 quaternion ``[x, y, z, w]``.
    """
    _ = start_quat  # reserved for future use / API symmetry
    x_axis_target = np.asarray(x_axis_target, dtype=np.float64)
    goal_q = np.asarray(goal_quat, dtype=np.float64)

    goal_rot_matrix = R.from_quat(goal_q).as_matrix()
    x_axis_goal = goal_rot_matrix[:, 0]
    z_axis_goal = goal_rot_matrix[:, 2]

    alignment = float(np.dot(x_axis_goal, x_axis_target))
    if alignment > 0.95:
        return goal_q

    y_axis = cross3(z_axis_goal, x_axis_target)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-6:
        # ``z_axis_goal`` is parallel to the desired X — fall back to a world
        # up vector (Z if target isn't near-vertical, else Y).
        fallback_up = np.array([0.0, 0.0, 1.0]) if abs(x_axis_target[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y_axis = cross3(fallback_up, x_axis_target)
        y_axis = y_axis / np.linalg.norm(y_axis)
    else:
        y_axis = y_axis / y_norm
    z_axis_final = cross3(x_axis_target, y_axis)
    rotation_matrix = np.column_stack([x_axis_target, y_axis, z_axis_final])
    return np.asarray(R.from_matrix(rotation_matrix).as_quat(), dtype=np.float64)


__all__ = ["TranslationSegment", "plan_translation_segment", "align_x_axis_quat"]


def heading_toward(
    pose: "Pose", goal: "Pose", direction: "Optional[MovementDirection | str]", navigation_2d: Optional[bool]
) -> Tuple[MovementDirection, np.ndarray, np.ndarray, float]:
    """How a differential-drive body at ``pose`` heads for ``goal``.

    Returns ``(direction, target_quat, displacement, distance)``. ``direction``
    ``None`` is ``FORWARD``; ``AUTO`` picks ``BACKWARD`` for a goal more than 90
    degrees behind the heading, judged in the plane (a goal straight above or
    below is ``FORWARD``). ``target_quat`` points the body x-axis along the
    travel, or against it when backing up (:func:`align_x_axis_quat`).
    ``navigation_2d`` keeps the goal at the body's height.

    The one rule for the per-agent and the batched differential controllers.
    """
    resolved = MovementDirection.FORWARD if direction is None else MovementDirection(direction)
    start = np.asarray(pose.position, dtype=np.float64)
    goal_pos = np.array(goal.position, dtype=np.float64)
    if navigation_2d:
        goal_pos[2] = start[2]
    displacement = goal_pos - start
    distance = float(np.linalg.norm(displacement))

    if resolved == MovementDirection.AUTO:
        resolved = MovementDirection.FORWARD
        if float(np.linalg.norm(displacement[:2])) > 1e-6:
            goal_heading = math.atan2(displacement[1], displacement[0])
            delta = (goal_heading - pose.yaw + math.pi) % (2.0 * math.pi) - math.pi
            if abs(delta) > math.pi / 2.0:
                resolved = MovementDirection.BACKWARD

    if distance > 1e-6:
        unit = displacement / distance
        x_axis_target = -unit if resolved == MovementDirection.BACKWARD else unit
        target_quat = align_x_axis_quat(
            np.asarray(pose.orientation, dtype=np.float64), np.asarray(goal.orientation, dtype=np.float64), x_axis_target
        )
    else:
        target_quat = np.asarray(goal.orientation, dtype=np.float64)
    return resolved, target_quat, displacement, distance
