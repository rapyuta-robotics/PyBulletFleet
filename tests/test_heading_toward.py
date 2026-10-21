"""How a differential-drive body heads for a goal: one rule, per-agent and batched."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pybullet_fleet._motion_planning import heading_toward
from pybullet_fleet.geometry import Pose
from pybullet_fleet.types import MovementDirection


def _x_axis(quat):
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(quat).as_matrix()[:, 0]


def test_forward_points_the_body_at_the_goal():
    direction, quat, displacement, distance = heading_toward(Pose.from_yaw(0, 0, 0, 0.0), Pose.from_xyz(0, 2, 0), None, False)
    assert direction is MovementDirection.FORWARD
    assert distance == pytest.approx(2.0)
    assert _x_axis(quat) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("yaw, expected", [(0.0, MovementDirection.BACKWARD), (math.pi, MovementDirection.FORWARD)])
def test_auto_backs_up_to_a_goal_behind(yaw, expected):
    direction, quat, _, _ = heading_toward(Pose.from_yaw(0, 0, 0, yaw), Pose.from_xyz(-3, 0, 0), "auto", False)
    assert direction is expected
    assert _x_axis(quat) == pytest.approx(
        [1.0, 0.0, 0.0] if expected is MovementDirection.BACKWARD else [-1.0, 0.0, 0.0], abs=1e-9
    )


def test_auto_judges_in_the_plane():
    """A goal straight above is FORWARD, whatever the heading."""
    direction, _, _, distance = heading_toward(Pose.from_yaw(0, 0, 0, math.pi), Pose.from_xyz(0, 0, 1), "auto", False)
    assert direction is MovementDirection.FORWARD
    assert distance == pytest.approx(1.0)


def test_navigation_2d_keeps_the_height():
    _, _, displacement, distance = heading_toward(Pose.from_xyz(0, 0, 0.1), Pose.from_xyz(3, 0, 0.0), None, True)
    assert displacement == pytest.approx([3.0, 0.0, 0.0])
    assert distance == pytest.approx(3.0)


def test_a_goal_at_the_body_keeps_the_goal_orientation():
    goal = Pose.from_yaw(0, 0, 0, 0.7)
    _, quat, _, distance = heading_toward(Pose.from_yaw(0, 0, 0, 0.0), goal, None, False)
    assert distance == 0.0
    assert np.asarray(quat) == pytest.approx(goal.orientation)
