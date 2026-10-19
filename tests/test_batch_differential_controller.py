"""Tests for BatchDifferentialController (Task 9).

Covers:
- ROTATE -> FORWARD phase transitions
- Multi-waypoint paths
- Equivalence vs the per-agent DifferentialController for the supported scope
  (direction=FORWARD, no final-orientation align)
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import (
    Agent,
    AgentSpawnParams,
    MotionMode,
    MultiRobotSimulationCore,
    Pose,
    SimulationParams,
)
from pybullet_fleet.controllers import BatchDifferentialController
from pybullet_fleet.types import CollisionMode, MovementDirection, SpatialHashCellSizeMode


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def sim():
    params = SimulationParams(
        gui=False,
        monitor=False,
        physics=False,
        timestep=1.0 / 60.0,
        collision_check_frequency=0,
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        log_level="warning",
    )
    sc = MultiRobotSimulationCore(params)
    yield sc
    p.disconnect(sc.client)


def _spawn_diff(sim, x=0.0, y=0.0, z=0.1, yaw=0.0, max_vel=1.0, max_ang_vel=2.0):
    return Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_yaw(x, y, z, yaw),
            motion_mode=MotionMode.DIFFERENTIAL,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={
                "max_linear_vel": max_vel,
                "max_linear_accel": 2.0,
                "max_angular_vel": max_ang_vel,
                "max_angular_accel": 4.0,
            },
        ),
        sim_core=sim,
    )


def _make_diff_manager(sim):
    from pybullet_fleet.agent_manager import AgentManager

    return AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})


# ----------------------------------------------------------------------
# Basic behaviour
# ----------------------------------------------------------------------


class TestBatchDiffBasics:
    def test_set_path_requires_registered_agent(self, sim):
        mgr = _make_diff_manager(sim)
        bc = mgr.batch_controller
        sentinel = _spawn_diff(sim, x=-1.0)
        mgr.add_object(sentinel)
        a = _spawn_diff(sim)
        with pytest.raises(KeyError):
            bc.set_path(a, [Pose.from_xyz(1.0, 0.0, 0.1)])

    def test_set_path_requires_attached_sim(self):
        bc = BatchDifferentialController()
        with pytest.raises(RuntimeError):
            bc.set_path(None, [Pose.from_xyz(1.0, 0.0, 0.1)])  # type: ignore[arg-type]

    def test_inactive_agent_does_not_move(self, sim):
        mgr = _make_diff_manager(sim)
        bc = mgr.batch_controller
        a = _spawn_diff(sim)
        mgr.add_object(a)
        start = a.get_pose().position[:]
        for _ in range(30):
            sim.step_once()
        assert np.allclose(a.get_pose().position, start)

    def test_reaches_goal_with_rotation(self, sim):
        mgr = _make_diff_manager(sim)
        bc = mgr.batch_controller
        a = _spawn_diff(sim, x=0.0, y=0.0, yaw=0.0, max_vel=1.5, max_ang_vel=2.0)
        mgr.add_object(a)
        bc.set_path(a, [Pose.from_xyz(0.0, 2.0, 0.1)])
        for _ in range(600):
            sim.step_once()
        pos = a.get_pose().position
        assert pos[0] == pytest.approx(0.0, abs=1e-2)
        assert pos[1] == pytest.approx(2.0, abs=1e-2)

    def test_multi_waypoint_path(self, sim):
        mgr = _make_diff_manager(sim)
        bc = mgr.batch_controller
        a = _spawn_diff(sim, max_vel=1.5, max_ang_vel=2.0)
        mgr.add_object(a)
        bc.set_path(
            a,
            [
                Pose.from_xyz(1.0, 0.0, 0.1),
                Pose.from_xyz(1.0, 1.0, 0.1),
            ],
        )
        for _ in range(900):
            sim.step_once()
        pos = a.get_pose().position
        assert pos[0] == pytest.approx(1.0, abs=2e-2)
        assert pos[1] == pytest.approx(1.0, abs=2e-2)


# ----------------------------------------------------------------------
# Equivalence with per-agent DifferentialController
# ----------------------------------------------------------------------


class TestBatchDiffEquivalence:
    """BatchDifferentialController should track DifferentialController for the
    supported scope (FORWARD direction, no final-orientation align) within
    a small per-step tolerance.

    The tolerance is wider than BatchOmni's 1e-3 because the differential
    controller has two TPI phases per waypoint and the per-agent path uses
    the scalar quat_slerp while the batch path uses a vectorised slerp; tiny
    floating-point divergence accumulates over the ROTATE phase but should
    stay well below visually observable error.
    """

    @pytest.mark.parametrize(
        "goal_xyz",
        [(2.0, 0.0, 0.1), (0.0, 1.5, 0.1), (1.0, 1.0, 0.1), (-1.0, 0.5, 0.1)],
    )
    def test_per_step_pose_matches_diff(self, sim, goal_xyz):
        # Reference agent uses per-agent DifferentialController.
        ref = _spawn_diff(sim, x=0.0, y=0.0, yaw=0.0, max_vel=1.2, max_ang_vel=1.5)
        # The same path on both sides: bc.set_path() adds no approach waypoint.
        ref.set_path(
            [Pose.from_xyz(*goal_xyz)],
            final_orientation_align=False,
            direction=MovementDirection.FORWARD,
            auto_approach=False,
        )

        # Target agent uses the batch controller (disable final-orient to match ref).
        mgr = _make_diff_manager(sim)
        bc = mgr.batch_controller
        tgt = _spawn_diff(sim, x=0.0, y=0.0, yaw=0.0, max_vel=1.2, max_ang_vel=1.5)
        mgr.add_object(tgt)
        bc.set_path(tgt, [Pose.from_xyz(*goal_xyz)], final_orientation_align=False)

        max_pos_err = 0.0
        max_ang_err = 0.0
        for _ in range(400):
            sim.step_once()
            ref_pose = ref.get_pose()
            tgt_pose = tgt.get_pose()
            pos_err = float(np.linalg.norm(np.array(ref_pose.position) - np.array(tgt_pose.position)))
            ang_err = _quat_angle(np.array(ref_pose.orientation), np.array(tgt_pose.orientation))
            max_pos_err = max(max_pos_err, pos_err)
            max_ang_err = max(max_ang_err, ang_err)

        # Tight enough to catch any structural mismatch but allows for
        # vectorised vs scalar slerp float drift.
        assert max_pos_err < 5e-3, f"position divergence {max_pos_err:.6f} m"
        assert max_ang_err < 5e-3, f"orientation divergence {max_ang_err:.6f} rad"


# ----------------------------------------------------------------------
# Integration: buffered pose write
# ----------------------------------------------------------------------


class TestBatchDiffIntegration:
    def test_uses_buffered_pose_write(self, sim):
        mgr = _make_diff_manager(sim)
        bc = mgr.batch_controller
        a = _spawn_diff(sim, x=0.0, y=0.0, yaw=0.0, max_vel=1.5, max_ang_vel=2.0)
        mgr.add_object(a)
        bc.set_path(a, [Pose.from_xyz(0.0, 1.0, 0.1)])

        sim.step_once()
        pb_pos, pb_orn = p.getBasePositionAndOrientation(a.body_id, physicsClientId=sim.client)
        cache_pose = a.get_pose()
        assert pb_pos[0] == pytest.approx(cache_pose.position[0], abs=1e-9)
        assert pb_pos[1] == pytest.approx(cache_pose.position[1], abs=1e-9)
        for i in range(4):
            assert pb_orn[i] == pytest.approx(cache_pose.orientation[i], abs=1e-9)

    def test_lifecycle_basic(self, sim):
        """Register/unregister keeps internal arrays consistent."""
        bc = BatchDifferentialController()
        a = _spawn_diff(sim, x=0.0)
        b = _spawn_diff(sim, x=1.0)
        c = _spawn_diff(sim, x=2.0)
        bc.register_agent(a)
        bc.register_agent(b)
        bc.register_agent(c)
        assert bc._phase.shape == (3,)
        assert bc._rot_start_quat.shape == (3, 4)
        bc.unregister_agent(b)
        assert bc._phase.shape == (2,)
        assert bc._rot_start_quat.shape == (2, 4)
        assert bc._agents == [a, c]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest-path angle between two unit quaternions (rad)."""
    dot = abs(float(np.dot(q0, q1)))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)
