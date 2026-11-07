"""
Integration tests for Action execution with real Agent + PyBullet.

Unlike test_action.py (unit tests with MockAgent), these tests run actions
through a real Agent.update() loop in a headless PyBullet environment to
verify end-to-end behaviour:

- MoveAction: agent follows path and reaches destination
- PickAction: agent approaches target, attaches object
- DropAction: agent approaches drop pose, detaches object at correct position
- Sequence: Move → Pick → Move → Drop full workflow
"""

import os

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet.action import (
    JointAction,
    MoveAction,
    PoseAction,
    PickAction,
    DropAction,
    WaitAction,
)
from pybullet_fleet.agent import Agent
from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet.geometry import Pose, Path
from pybullet_fleet.sim_object import SimObject, ShapeParams
from pybullet_fleet.types import ActionStatus, MotionMode, MovementDirection
from tests.conftest import MockSimCore


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MESH_PATH = os.path.join(os.path.dirname(__file__), "..", "pybullet_fleet", "mesh", "cube.obj")
ARM_URDF = "robots/arm_robot.urdf"
RAIL_ARM_URDF = "robots/rail_arm_robot.urdf"


# ---------------------------------------------------------------------------
# Assertion tolerances
# ---------------------------------------------------------------------------

TIGHT_TOL = 0.01  # Final position accuracy (±1 cm)
PROXIMITY_TOL = 0.01  # "Near target" checks
COARSE_TOL = 0.01  # Box placement / waypoint proximity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_until_idle(agent, sim_core, *, max_steps: int = 20_000) -> int:
    """Run agent.update() until the action queue is empty.

    Returns the number of steps executed.
    Raises AssertionError if *max_steps* is exceeded.
    """
    dt = sim_core._dt
    for step in range(max_steps):
        sim_core.tick()
        agent.update(dt)
        if agent.is_action_queue_empty() and not agent.is_moving:
            return step + 1
    raise AssertionError(f"Agent did not finish actions within {max_steps} steps")


def create_agent(sim_core, *, pose=None, motion_mode=MotionMode.OMNIDIRECTIONAL):
    """Create an omnidirectional mesh agent wired to *sim_core*."""
    if pose is None:
        pose = Pose.from_xyz(0, 0, 0)
    return Agent.from_mesh(
        visual_shape=ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        pose=pose,
        mass=0.0,
        controller=ControllerParams(
            max_linear_vel=5.0,
            max_linear_accel=20.0,
            max_angular_vel=6.0,
            max_angular_accel=20.0,
        ),
        motion_mode=motion_mode,
        sim_core=sim_core,
    )


def create_pickable_box(sim_core, *, pos=(2, 0, 0)):
    """Create a small pickable SimObject registered with *sim_core*."""
    return SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
        pose=Pose.from_xyz(*pos),
        mass=0.0,
        pickable=True,
        sim_core=sim_core,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sim(pybullet_env):
    """MockSimCore wired to the PyBullet session."""
    return MockSimCore()


# ---------------------------------------------------------------------------
# MoveAction
# ---------------------------------------------------------------------------


class TestMoveActionIntegration:
    """MoveAction executed through Agent.update() loop."""

    def test_single_waypoint(self, sim):
        """Agent reaches a single waypoint and action completes."""
        agent = create_agent(sim)
        goal = [2, 0, 0]
        action = MoveAction(path=Path.from_positions([goal]))
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        pos = agent.get_pose().position
        assert np.allclose(pos[:2], goal[:2], atol=PROXIMITY_TOL)

    def test_multi_waypoint_path(self, sim):
        """Agent follows an L-shaped path through all intermediate waypoints."""
        agent = create_agent(sim)
        waypoints = [[1, 0, 0], [1, 1, 0], [0, 1, 0]]
        action = MoveAction(path=Path.from_positions(waypoints))
        agent.add_action(action)

        # Record positions during update loop to verify intermediate waypoints
        trajectory: list = []
        dt = sim._dt
        for _ in range(20_000):
            sim.tick()
            agent.update(dt)
            trajectory.append(list(agent.get_pose().position[:2]))
            if agent.is_action_queue_empty() and not agent.is_moving:
                break

        assert action.status is ActionStatus.COMPLETED

        # Final position
        pos = agent.get_pose().position
        assert np.allclose(pos[:2], [0, 1], atol=PROXIMITY_TOL)

        # Verify agent passed near each intermediate waypoint
        for wp in waypoints[:-1]:
            dists = [np.linalg.norm(np.array(p) - np.array(wp[:2])) for p in trajectory]
            assert min(dists) < COARSE_TOL, f"Agent never passed near intermediate waypoint {wp}"

    def test_backward_direction(self, sim):
        """Agent moves backward: reaches goal while facing away from travel direction."""
        agent = create_agent(sim, motion_mode=MotionMode.DIFFERENTIAL)
        goal = [2, 0, 0]
        action = MoveAction(
            path=Path.from_positions([goal]),
            direction=MovementDirection.BACKWARD,
            final_orientation_align=False,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        pos = agent.get_pose().position
        assert np.allclose(pos[:2], goal[:2], atol=PROXIMITY_TOL)

        # Backward movement: agent's X-axis should point OPPOSITE to travel direction.
        # Travel was +X, so agent should face -X (yaw ≈ π or -π).
        orn = agent.get_pose().orientation
        _, _, yaw = p.getEulerFromQuaternion(orn)
        assert (
            abs(abs(yaw) - np.pi) < 0.3
        ), f"Backward agent should face away from travel direction (yaw ≈ ±π), got {np.degrees(yaw):.1f}°"

    def test_duration_is_recorded(self, sim):
        """Start and end time are recorded after completion."""
        agent = create_agent(sim)
        action = MoveAction(path=Path.from_positions([[1, 0, 0]]))
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.start_time is not None
        assert action.end_time is not None
        assert action.get_duration() >= 0  # type: ignore[operator]

    def test_sequential_moves(self, sim):
        """Two MoveActions execute in sequence; both complete."""
        agent = create_agent(sim)
        a1 = MoveAction(path=Path.from_positions([[1, 0, 0]]))
        a2 = MoveAction(path=Path.from_positions([[1, 1, 0]]))
        agent.add_action_sequence([a1, a2])

        run_until_idle(agent, sim)

        assert a1.status is ActionStatus.COMPLETED
        assert a2.status is ActionStatus.COMPLETED
        pos = agent.get_pose().position
        assert np.allclose(pos[:2], [1, 1], atol=PROXIMITY_TOL)


# ---------------------------------------------------------------------------
# PickAction
# ---------------------------------------------------------------------------


class TestPickActionIntegration:
    """PickAction executed through Agent.update() loop."""

    def test_pick_with_approach(self, sim):
        """Full 5-phase pick: approach → move_to_pick → pick → retreat.

        Verify via trajectory that agent approached the target, then retreated.
        """
        approach_offset = 0.5
        target_pos = (2, 0, 0)

        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=target_pos)
        target_xy = np.array(target_pos[:2])

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=True,
            approach_offset=approach_offset,
            pick_offset=0.0,
        )
        agent.add_action(action)

        # Record distance-to-target over time
        dists: list = []
        dt = sim._dt
        for _ in range(20_000):
            sim.tick()
            agent.update(dt)
            pos = agent.get_pose().position
            dists.append(float(np.linalg.norm(np.array(pos[:2]) - target_xy)))
            if agent.is_action_queue_empty() and not agent.is_moving:
                break

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()

        # Agent should have reached near the target (pick_offset=0 → at target)
        min_dist = min(dists)
        assert min_dist < PROXIMITY_TOL, f"Agent should have reached near target (pick_offset=0), closest was {min_dist:.2f}m"

        # After reaching target, distance should have increased (retreat).
        # Find the index of closest approach, then verify distance grew afterward.
        min_idx = dists.index(min_dist)
        final_dist = dists[-1]
        assert final_dist > min_dist + COARSE_TOL, (
            f"Agent should have retreated after pick " f"(closest={min_dist:.2f}m at step {min_idx}, final={final_dist:.2f}m)"
        )

        # Final position should be at approach_offset from target
        assert (
            abs(final_dist - approach_offset) < TIGHT_TOL
        ), f"Agent should end ~{approach_offset}m (approach_offset) from target, but is {final_dist:.2f}m"

    def test_pick_with_approach_and_pick_offset(self, sim):
        """Pick with approach and pick_offset: agent stops at pick_offset, not at target.

        Verify agent approached to pick_offset distance (not to target itself),
        then retreated to approach_offset distance.
        """
        approach_offset = 0.5
        pick_offset = 0.3
        target_pos = (2, 0, 0)

        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=target_pos)
        target_xy = np.array(target_pos[:2])

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=True,
            approach_offset=approach_offset,
            pick_offset=pick_offset,
        )
        agent.add_action(action)

        dists: list = []
        dt = sim._dt
        for _ in range(20_000):
            sim.tick()
            agent.update(dt)
            pos = agent.get_pose().position
            dists.append(float(np.linalg.norm(np.array(pos[:2]) - target_xy)))
            if agent.is_action_queue_empty() and not agent.is_moving:
                break

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()

        # Closest distance should be ~pick_offset, NOT at target (0.0m)
        min_dist = min(dists)
        assert (
            abs(min_dist - pick_offset) < PROXIMITY_TOL
        ), f"Agent should approach to ~{pick_offset}m (pick_offset) from target, closest was {min_dist:.2f}m"

        # Final position should be at approach_offset
        final_dist = dists[-1]
        assert (
            abs(final_dist - approach_offset) < TIGHT_TOL
        ), f"Agent should end ~{approach_offset}m (approach_offset) from target, but is {final_dist:.2f}m"

    def test_pick_without_approach(self, sim):
        """No approach/retreat: agent moves to pick_pose and stays there.

        With pick_offset=0.0, agent ends at the target position.
        """
        target_pos = (2, 0, 0)

        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim, pos=target_pos)

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=False,
            pick_offset=0.0,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert box in agent.attached_objects

        # No retreat → agent stays at pick_pose = target position (pick_offset=0).
        agent_pos = agent.get_pose().position
        target_xy = np.array(target_pos[:2])
        dist_to_target = np.linalg.norm(np.array(agent_pos[:2]) - target_xy)

        assert dist_to_target < PROXIMITY_TOL, (
            f"Agent should be at target (pick_offset=0, no retreat), "
            f"but is {dist_to_target:.2f}m away (pos={agent_pos[:2]})"
        )

    def test_pick_without_approach_with_pick_offset(self, sim):
        """No approach/retreat with pick_offset: agent stops at pick_offset distance.

        With pick_offset=0.3, agent ends 0.3m from the target.
        """
        pick_offset = 0.3
        target_pos = (2, 0, 0)

        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=target_pos)

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=False,
            pick_offset=pick_offset,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()

        # No retreat → agent stays at pick_offset from target.
        agent_pos = agent.get_pose().position
        target_xy = np.array(target_pos[:2])
        dist_to_target = np.linalg.norm(np.array(agent_pos[:2]) - target_xy)
        assert abs(dist_to_target - pick_offset) < TIGHT_TOL, (
            f"Agent should be ~{pick_offset}m (pick_offset) from target, "
            f"but is {dist_to_target:.2f}m away (pos={agent_pos[:2]})"
        )

    def test_pick_with_relative_pose(self, sim):
        """Picked object is attached at the specified relative pose offset."""
        rel_offset = 0.3
        approach_offset = 0.5

        agent = create_agent(sim)
        box = create_pickable_box(sim)

        rel_pose = Pose.from_euler(rel_offset, 0, 0)
        action = PickAction(
            target_object_id=box.body_id,
            use_approach=True,
            approach_offset=approach_offset,
            attach_relative_pose=rel_pose,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()

        # Verify relative pose: box should be ~0.3m from agent in the agent's local X direction.
        agent_pos = np.array(agent.get_pose().position)
        box_pos = np.array(box.get_pose().position)
        offset_world = box_pos - agent_pos
        offset_dist = np.linalg.norm(offset_world)
        assert abs(offset_dist - rel_offset) < TIGHT_TOL, (
            f"Box should be ~{rel_offset}m from agent (relative_pose offset), "
            f"but is {offset_dist:.3f}m (offset={offset_world})"
        )

    def test_pick_nonexistent_object_fails(self, sim):
        """PickAction fails if the target body_id is not found."""
        agent = create_agent(sim)

        action = PickAction(target_object_id=9999)
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.FAILED
        assert action.error_message is not None

    def test_pick_non_pickable_object_fails(self, sim):
        """PickAction fails if the target object is not pickable."""
        agent = create_agent(sim)
        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
            pose=Pose.from_xyz(2, 0, 0),
            mass=0.0,
            pickable=False,
            sim_core=sim,
        )

        action = PickAction(target_object_id=obj.body_id)
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.FAILED

    def test_pick_by_position(self, sim):
        """PickAction can find nearest pickable object by position."""
        target_pos = (2, 0, 0)

        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=target_pos)

        action = PickAction(
            target_position=list(target_pos),
            search_radius=1.0,
            use_approach=True,
            approach_offset=0.5,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()

    def test_pick_by_position_offset_within_radius(self, sim):
        """PickAction finds nearest pickable object when position is not exact.

        Box is at (2, 0.4, 0), search from (2, 0, 0) with radius=0.5.
        The box is 0.4m away — within search_radius.
        """
        search_pos = (2, 0, 0)
        search_radius = 0.5

        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=(2, 0.4, 0))

        action = PickAction(
            target_position=list(search_pos),
            search_radius=search_radius,
            use_approach=True,
            approach_offset=0.5,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()

    def test_pick_by_position_outside_radius_fails(self, sim):
        """PickAction fails when no pickable object is within search_radius."""
        search_pos = (2, 0, 0)
        search_radius = 0.5

        agent = create_agent(sim)
        # Box at (2, 1, 0) is 1.0m from search position — outside radius
        create_pickable_box(sim, pos=(2, 1, 0))

        action = PickAction(
            target_position=list(search_pos),
            search_radius=search_radius,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.FAILED

    def test_pick_by_position_skips_attached(self, sim):
        """PickAction with target_position must skip already-attached objects."""
        search_pos = (2, 0, 0)
        agent = create_agent(sim)

        # Create two pickable boxes near the search position
        box_near = create_pickable_box(sim, pos=(2.0, 0.1, 0))
        box_far = create_pickable_box(sim, pos=(2.0, 0.3, 0))

        # Attach the nearer box to a different agent (simulate already picked)
        other = create_agent(sim, pose=Pose.from_xyz(5, 0, 0))
        other.attach_object(box_near)

        action = PickAction(
            target_position=list(search_pos),
            search_radius=1.0,
            use_approach=True,
            approach_offset=0.5,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        # Should pick box_far (skip attached box_near)
        assert action.status is ActionStatus.COMPLETED
        assert box_far.is_attached()
        assert action._target_object is box_far

    def test_pick_with_explicit_approach_pose(self, sim):
        """PickAction with explicit approach_pose not on the agent-target line.

        Agent at (0,0), target at (2,0), approach_pose at (2,1) — off to the side.
        Agent should go to (2,1) first, then forward to target, then retreat to (2,1).
        """
        agent = create_agent(sim)
        box = create_pickable_box(sim)
        approach = Pose.from_xyz(2, 1, 0)

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=True,
            approach_pose=approach,
            pick_offset=0.0,
        )
        agent.add_action(action)

        # Record trajectory to verify lateral approach
        trajectory: list = []
        dt = sim._dt
        for _ in range(20_000):
            sim.tick()
            agent.update(dt)
            trajectory.append(list(agent.get_pose().position[:2]))
            if agent.is_action_queue_empty() and not agent.is_moving:
                break

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()

        # Agent should have passed near approach_pose (2,1), not just gone straight
        dists_to_approach = [np.linalg.norm(np.array(t) - np.array([2, 1])) for t in trajectory]
        assert min(dists_to_approach) < COARSE_TOL, (
            f"Agent should have passed near explicit approach_pose (2,1), " f"closest was {min(dists_to_approach):.2f}m"
        )

        # Final position should be back near approach_pose after retreat
        final_pos = np.array(trajectory[-1])
        dist_to_approach = np.linalg.norm(final_pos - np.array([2, 1]))
        assert dist_to_approach < 0.3, f"Agent should retreat to approach_pose (2,1), but ended at {final_pos}"

    def test_pick_zero_distance_skip(self, sim):
        """Pick with pick_offset == distance(robot, item) skips MOVING_TO_PICK.

        When the robot is already at the pick pose (distance < threshold),
        MOVING_TO_PICK is skipped and the item is attached immediately.
        This mimics the workcell handler dispensing in-place.
        """
        # Place agent right at the item
        box = create_pickable_box(sim, pos=(1, 0, 0))
        agent = create_agent(sim, pose=Pose.from_xyz(1, 0, 0))

        # pick_offset = distance = 0 → skip MOVING_TO_PICK
        action = PickAction(
            target_object_id=box.body_id,
            use_approach=False,
            pick_offset=0.0,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert box.is_attached()
        # Agent should not have moved significantly
        assert abs(agent.get_pose().position[0] - 1.0) < TIGHT_TOL


# ---------------------------------------------------------------------------
# DropAction
# ---------------------------------------------------------------------------


class TestDropActionIntegration:
    """DropAction executed through Agent.update() loop."""

    def _pick_box(self, agent, box, sim):
        """Helper: pick a box and wait for completion."""
        action = PickAction(
            target_object_id=box.body_id,
            use_approach=False,
        )
        agent.add_action(action)
        run_until_idle(agent, sim)
        assert box.is_attached()

    def test_drop_with_approach(self, sim):
        """Full 5-phase drop: approach → move_to_drop → drop → retreat.

        Verify via trajectory that agent approached the drop pose, then retreated.
        """
        approach_offset = 0.5
        drop_target = [4, 0, 0]

        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        target_xy = np.array(drop_target[:2])
        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=True,
            approach_offset=approach_offset,
            drop_offset=0.0,
            place_gently=True,
        )
        agent.add_action(action)

        # Record distance-to-drop-pose over time
        dists: list = []
        dt = sim._dt
        for _ in range(20_000):
            sim.tick()
            agent.update(dt)
            pos = agent.get_pose().position
            dists.append(float(np.linalg.norm(np.array(pos[:2]) - target_xy)))
            if agent.is_action_queue_empty() and not agent.is_moving:
                break

        assert action.status is ActionStatus.COMPLETED
        assert not box.is_attached()

        # Box should be at drop position
        box_pos = box.get_pose().position
        assert np.allclose(box_pos[:2], drop_target[:2], atol=COARSE_TOL)

        # Agent should have reached near the drop pose (drop_offset=0 → at target)
        min_dist = min(dists)
        assert (
            min_dist < PROXIMITY_TOL
        ), f"Agent should have reached near drop pose (drop_offset=0), closest was {min_dist:.2f}m"

        # Final position should be at approach_offset from drop pose
        final_dist = dists[-1]
        assert (
            abs(final_dist - approach_offset) < TIGHT_TOL
        ), f"Agent should end ~{approach_offset}m (approach_offset) from drop pose, but is {final_dist:.2f}m"

    def test_drop_with_approach_and_drop_offset(self, sim):
        """Drop with approach and drop_offset: agent stops at drop_offset, not at target.

        Verify agent approached to drop_offset distance (not to target itself),
        then retreated to approach_offset distance.
        """
        approach_offset = 0.5
        drop_offset = 0.3
        drop_target = [4, 0, 0]

        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        target_xy = np.array(drop_target[:2])
        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=True,
            approach_offset=approach_offset,
            drop_offset=drop_offset,
            place_gently=True,
        )
        agent.add_action(action)

        dists: list = []
        dt = sim._dt
        for _ in range(20_000):
            sim.tick()
            agent.update(dt)
            pos = agent.get_pose().position
            dists.append(float(np.linalg.norm(np.array(pos[:2]) - target_xy)))
            if agent.is_action_queue_empty() and not agent.is_moving:
                break

        assert action.status is ActionStatus.COMPLETED
        assert not box.is_attached()

        # Closest distance should be ~drop_offset, NOT at target (0.0m)
        min_dist = min(dists)
        assert (
            abs(min_dist - drop_offset) < PROXIMITY_TOL
        ), f"Agent should approach to ~{drop_offset}m (drop_offset) from drop pose, closest was {min_dist:.2f}m"

        # Final position should be at approach_offset
        final_dist = dists[-1]
        assert (
            abs(final_dist - approach_offset) < TIGHT_TOL
        ), f"Agent should end ~{approach_offset}m (approach_offset) from drop pose, but is {final_dist:.2f}m"

    def test_drop_without_approach(self, sim):
        """No approach/retreat: agent moves to drop_pose and stays there.

        With drop_offset=0.0, agent ends at the drop position.
        """
        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        drop_target = [3, 0, 0]
        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=False,
            drop_offset=0.0,
            place_gently=True,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert not box.is_attached()

        # No retreat → agent stays at drop_pose (drop_offset=0).
        agent_pos = agent.get_pose().position
        target_pos = np.array(drop_target[:2])
        dist_to_target = np.linalg.norm(np.array(agent_pos[:2]) - target_pos)
        assert dist_to_target < PROXIMITY_TOL, (
            f"Agent should be at drop pose (drop_offset=0, no retreat), "
            f"but is {dist_to_target:.2f}m away (pos={agent_pos[:2]})"
        )

        # Box should be at drop position
        box_pos = box.get_pose().position
        assert np.allclose(box_pos[:2], drop_target[:2], atol=COARSE_TOL)

    def test_drop_without_approach_with_drop_offset(self, sim):
        """No approach/retreat with drop_offset: agent stops at drop_offset distance.

        With drop_offset=0.3, agent ends 0.3m from the drop pose.
        """
        drop_offset = 0.3
        drop_target = [4, 0, 0]

        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=False,
            drop_offset=drop_offset,
            place_gently=True,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert not box.is_attached()

        # No retreat → agent stays at drop_pose offset by drop_offset.
        agent_pos = agent.get_pose().position
        target_xy = np.array(drop_target[:2])
        dist_to_target = np.linalg.norm(np.array(agent_pos[:2]) - target_xy)
        assert abs(dist_to_target - drop_offset) < TIGHT_TOL, (
            f"Agent should be ~{drop_offset}m (drop_offset) from drop pose, "
            f"but is {dist_to_target:.2f}m away (pos={agent_pos[:2]})"
        )

    def test_drop_with_orientation(self, sim):
        """Dropped object receives the specified orientation."""
        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        yaw_quat = list(p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        action = DropAction(
            drop_pose=Pose(position=[3, 0, 0], orientation=yaw_quat),
            use_approach=False,
            place_gently=True,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        box_orn = box.get_pose().orientation
        assert np.allclose(box_orn, yaw_quat, atol=TIGHT_TOL)

    def test_drop_zero_distance_teleports(self, sim):
        """Drop with drop_offset == distance skips MOVING_TO_DROP and teleports.

        When the robot is far from the drop_pose but drop_offset equals
        that distance, _drop_pose == robot position → distance < threshold
        → skip MoveAction → go straight to DROPPING → item teleports to
        drop_pose.  This mimics the workcell handler ingestor drop.
        """
        drop_target = [5, 0, 0]

        # Agent at (2, 0), box picked, drop at (5, 0) with offset=3
        agent = create_agent(sim, pose=Pose.from_xyz(2, 0, 0))
        box = create_pickable_box(sim, pos=(2, 0, 0))
        self._pick_box(agent, box, sim)

        # drop_offset = distance from agent(2,0) to drop_pose(5,0) = 3.0
        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=False,
            drop_offset=3.0,
            place_gently=True,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert not box.is_attached()

        # Box should be at the drop_pose (teleported), NOT at the agent
        box_pos = box.get_pose().position
        assert abs(box_pos[0] - 5.0) < TIGHT_TOL, f"Box should teleport to x=5, got x={box_pos[0]:.3f}"
        assert abs(box_pos[1] - 0.0) < TIGHT_TOL

        # Agent should NOT have moved significantly
        agent_pos = agent.get_pose().position
        assert abs(agent_pos[0] - 2.0) < TIGHT_TOL, f"Agent should stay at x=2, got x={agent_pos[0]:.3f}"

    def test_drop_no_attached_object_fails(self, sim):
        """DropAction fails if agent has nothing attached."""
        agent = create_agent(sim)
        action = DropAction(drop_pose=Pose(position=[1, 0, 0]), use_approach=False)
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.FAILED
        assert action.error_message is not None

    def test_drop_from_height(self, sim):
        """When place_gently=False, object is placed at drop_height above drop_pose."""
        drop_height = 0.5

        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        action = DropAction(
            drop_pose=Pose(position=[2, 0, 0]),
            use_approach=False,
            place_gently=False,
            drop_height=drop_height,
        )
        agent.add_action(action)

        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        # Object should be at z = 0.0 + 0.5 = 0.5
        box_z = box.get_pose().position[2]
        assert abs(box_z - drop_height) < PROXIMITY_TOL

    def test_drop_with_explicit_approach_pose(self, sim):
        """DropAction with explicit approach_pose not on the agent-target line.

        Agent at (1.9,0), drop pose at (4,0), approach_pose at (4,1) — off to the side.
        Agent should go to (4,1) first, then forward to drop pose, then retreat to (4,1).
        """
        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        approach = Pose.from_xyz(4, 1, 0)
        action = DropAction(
            drop_pose=Pose(position=[4, 0, 0]),
            use_approach=True,
            approach_pose=approach,
            drop_offset=0.0,
            place_gently=True,
        )
        agent.add_action(action)

        # Record trajectory to verify lateral approach
        trajectory: list = []
        dt = sim._dt
        for _ in range(20_000):
            sim.tick()
            agent.update(dt)
            trajectory.append(list(agent.get_pose().position[:2]))
            if agent.is_action_queue_empty() and not agent.is_moving:
                break

        assert action.status is ActionStatus.COMPLETED
        assert not box.is_attached()

        # Box should be at drop position
        box_pos = box.get_pose().position
        assert np.allclose(box_pos[:2], [4, 0], atol=COARSE_TOL)

        # Agent should have passed near approach_pose (4,1), not just gone straight
        dists_to_approach = [np.linalg.norm(np.array(t) - np.array([4, 1])) for t in trajectory]
        assert min(dists_to_approach) < COARSE_TOL, (
            f"Agent should have passed near explicit approach_pose (4,1), " f"closest was {min(dists_to_approach):.2f}m"
        )

        # Final position should be back near approach_pose after retreat
        final_pos = np.array(trajectory[-1])
        dist_to_approach = np.linalg.norm(final_pos - np.array([4, 1]))
        assert dist_to_approach < 0.3, f"Agent should retreat to approach_pose (4,1), but ended at {final_pos}"

    def test_drop_relative_pose_in_place(self, sim):
        """drop_relative_pose=Pose(0,0,0) drops object at its current EE position.

        When drop_relative_pose is set, the object should stay where it was
        at detach time (at the attachment point) instead of being teleported
        to drop_pose.  The robot still moves to drop_pose first.
        """
        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        drop_target = [3, 0, 0]
        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=False,
            drop_offset=0.0,
            place_gently=True,
            drop_relative_pose=Pose.from_xyz(0, 0, 0),  # drop in place
        )
        agent.add_action(action)
        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED
        assert not box.is_attached()

        # Agent should have moved to drop_pose
        agent_pos = np.array(agent.get_pose().position[:2])
        dist_to_drop = np.linalg.norm(agent_pos - np.array(drop_target[:2]))
        assert dist_to_drop < 0.3, (
            f"Agent should move to drop_pose {drop_target[:2]}, " f"but ended at {list(agent_pos)} (dist={dist_to_drop:.3f})"
        )

        # drop_relative_pose=(0,0,0) → box stays at its pre-detach world position.
        # Default attach_link=-1 and attach_relative_pose=(0,0,0), so box ≈ agent position.
        box_pos = np.array(box.get_pose().position)
        dist_box_agent = np.linalg.norm(box_pos[:2] - agent_pos)
        assert dist_box_agent < COARSE_TOL, (
            f"drop_relative_pose=(0,0,0) with zero attachment offset means "
            f"box should be at agent position, but dist={dist_box_agent:.3f} "
            f"(box={list(box_pos[:2])}, agent={list(agent_pos)})"
        )

    def test_drop_relative_pose_with_offset(self, sim):
        """drop_relative_pose with Z offset places object below detach position."""
        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        self._pick_box(agent, box, sim)

        drop_target = [3, 0, 0]
        z_offset = -0.15
        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=False,
            drop_offset=0.0,
            place_gently=True,
            drop_relative_pose=Pose.from_xyz(0, 0, z_offset),
        )
        agent.add_action(action)
        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED

        # Agent should have moved to drop_pose
        agent_pos = np.array(agent.get_pose().position)
        dist_to_drop = np.linalg.norm(agent_pos[:2] - np.array(drop_target[:2]))
        assert dist_to_drop < 0.3, f"Agent should move to drop_pose, but ended at {list(agent_pos)}"

        # Box should be at agent's position + z_offset (relative placement)
        box_pos = np.array(box.get_pose().position)
        assert abs(box_pos[2] - (agent_pos[2] + z_offset)) < COARSE_TOL, (
            f"Box Z should be agent_z({agent_pos[2]:.3f}) + offset({z_offset}) = "
            f"{agent_pos[2] + z_offset:.3f}, got {box_pos[2]:.3f}"
        )

    def test_drop_relative_pose_mobile_moves_to_drop_pose(self, sim):
        """Mobile robot with drop_relative_pose still drives to drop_pose.

        drop_relative_pose only controls how the object is placed (relative
        offset from its current EE position). The base should still move to
        drop_pose in the MOVING_TO_DROP phase.
        """
        agent = create_agent(sim, pose=Pose.from_xyz(0, 0, 0))
        box = create_pickable_box(sim, pos=(0.5, 0, 0))
        self._pick_box(agent, box, sim)

        drop_target = [3, 0, 0]
        action = DropAction(
            drop_pose=Pose(position=drop_target),
            use_approach=False,
            drop_offset=0.0,
            place_gently=True,
            drop_relative_pose=Pose.from_xyz(0, 0, 0),
        )
        agent.add_action(action)
        run_until_idle(agent, sim)

        assert action.status is ActionStatus.COMPLETED

        # Agent should have moved near drop_pose
        agent_pos = np.array(agent.get_pose().position[:2])
        dist = np.linalg.norm(agent_pos - np.array(drop_target[:2]))
        assert dist < COARSE_TOL, (
            f"Agent should move to drop_pose {drop_target[:2]} even with "
            f"drop_relative_pose, but ended at {list(agent_pos)} (dist={dist:.3f})"
        )

        # drop_relative_pose=(0,0,0) + attachment offset ≈ 0 → box ≈ agent position
        assert not box.is_attached()
        box_pos = np.array(box.get_pose().position[:2])
        dist_box_agent = np.linalg.norm(box_pos - agent_pos)
        assert dist_box_agent < COARSE_TOL, (
            f"drop_relative_pose=(0,0,0) means box should be at agent position, "
            f"but dist={dist_box_agent:.3f} (box={list(box_pos)}, agent={list(agent_pos)})"
        )


# ---------------------------------------------------------------------------
# Full sequence: Move → Pick → Move → Drop
# ---------------------------------------------------------------------------


class TestActionSequenceIntegration:
    """End-to-end action sequence mimicking a warehouse pick-and-place task."""

    def test_move_pick_move_drop(self, sim):
        """Full workflow: move to object, pick, move to drop zone, drop."""
        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=(3, 0, 0))

        move_to_pick = MoveAction(
            path=Path.from_positions([[2.5, 0, 0]]),
            final_orientation_align=False,
        )
        pick = PickAction(
            target_object_id=box.body_id,
            use_approach=True,
            approach_offset=0.5,
        )
        move_to_drop = MoveAction(
            path=Path.from_positions([[3, 3, 0]]),
            final_orientation_align=False,
        )
        drop = DropAction(
            drop_pose=Pose(position=[4, 3, 0]),
            use_approach=False,
            place_gently=True,
        )

        agent.add_action_sequence([move_to_pick, pick, move_to_drop, drop])

        run_until_idle(agent, sim)

        # All four actions should be completed
        assert move_to_pick.status is ActionStatus.COMPLETED
        assert pick.status is ActionStatus.COMPLETED
        assert move_to_drop.status is ActionStatus.COMPLETED
        assert drop.status is ActionStatus.COMPLETED

        # Box should be detached and near drop position
        assert not box.is_attached()
        box_pos = box.get_pose().position
        assert np.allclose(box_pos[:2], [4, 3], atol=COARSE_TOL)

    def test_move_pick_move_drop_with_wait(self, sim):
        """Full workflow with WaitAction interleaved."""
        agent = create_agent(sim)
        box = create_pickable_box(sim)

        actions = [
            MoveAction(path=Path.from_positions([[1.5, 0, 0]]), final_orientation_align=False),
            PickAction(target_object_id=box.body_id, use_approach=True, approach_offset=0.3),
            WaitAction(duration=0.1, action_type="loading"),
            MoveAction(path=Path.from_positions([[2, 2, 0]]), final_orientation_align=False),
            DropAction(drop_pose=Pose(position=[3, 2, 0]), use_approach=False, place_gently=True),
        ]
        agent.add_action_sequence(actions)

        run_until_idle(agent, sim)

        for a in actions:
            assert a.status is ActionStatus.COMPLETED, f"{a.__class__.__name__} not completed: {a.status}"

        assert not box.is_attached()

    def test_queue_size_decreases(self, sim):
        """Action queue size decreases as actions complete."""
        agent = create_agent(sim)

        actions = [
            MoveAction(path=Path.from_positions([[1, 0, 0]]), final_orientation_align=False),
            WaitAction(duration=0.01),
            MoveAction(path=Path.from_positions([[1, 1, 0]]), final_orientation_align=False),
        ]
        agent.add_action_sequence(actions)

        initial_size = agent.get_action_queue_size()
        assert initial_size == 3

        run_until_idle(agent, sim)

        assert agent.get_action_queue_size() == 0
        assert agent.is_action_queue_empty()


# ---------------------------------------------------------------------------
# JointAction
# ---------------------------------------------------------------------------


def create_arm_agent(sim_core, mass=None):
    """Create a fixed-base arm robot wired to *sim_core*.

    Args:
        mass: None (default) uses URDF mass values, 0.0 for kinematic.
    """
    return Agent.from_urdf(
        urdf_path=ARM_URDF,
        pose=Pose.from_xyz(0, 0, 0),
        mass=mass,
        use_fixed_base=True,
        sim_core=sim_core,
    )


def run_arm_until_idle(agent, sim_core, *, max_steps: int = 2_000) -> int:
    """Run agent.update() with stepping until actions complete."""
    dt = sim_core._dt
    for step in range(max_steps):
        sim_core.tick()
        agent.update(dt)
        if agent.is_action_queue_empty():
            return step + 1
    raise AssertionError(f"Agent did not finish actions within {max_steps} steps")


class TestJointActionIntegration:
    """Test JointAction execution with a real arm robot in both modes."""

    def test_reaches_target_positions(self, arm_sim):
        """JointAction moves all joints to target and completes."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        targets = [0.5, 0.3, -0.3, 0.2]

        action = JointAction(target_joint_positions=targets, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=0.05)

    def test_dict_targets_by_name(self, arm_sim):
        """JointAction accepts dict keyed by joint name."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        targets = {"base_to_shoulder": 0.4, "elbow_to_wrist": -0.5}

        action = JointAction(target_joint_positions=targets, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=0.05)

    def test_sequential_joint_actions(self, arm_sim):
        """Two JointActions execute in sequence, both reach targets."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        targets_1 = [0.5, 0.3, -0.3, 0.2]
        targets_2 = [-0.3, 0.5, 0.2, -0.1]

        action1 = JointAction(target_joint_positions=targets_1, tolerance=0.05)
        action2 = JointAction(target_joint_positions=targets_2, tolerance=0.05)
        agent.add_action_sequence([action1, action2])

        run_arm_until_idle(agent, sim_core)

        assert action1.status is ActionStatus.COMPLETED
        assert action2.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets_2, tolerance=0.05)

    def test_duration_recorded(self, arm_sim):
        """JointAction records non-zero duration after completion."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        action = JointAction(target_joint_positions=[0.3, 0.0, 0.0, 0.0], tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        duration = action.get_duration()
        assert duration is not None and duration > 0, f"Duration should be positive, got {duration}"

    def test_kinematic_takes_multiple_steps(self, arm_sim):
        """Kinematic interpolation respects velocity limits, not teleporting.

        With large target angles (1.0 rad), a single dt step cannot cover the
        full distance at the URDF velocity limit.  If the joint reaches the
        target in 1 step, it means resetJointState is teleporting instead of
        interpolating — which would be a bug.
        """
        sim_core, mass = arm_sim
        if mass != 0.0:
            pytest.skip("multi-step convergence test is kinematic-specific")
        agent = create_arm_agent(sim_core, mass)
        targets = [1.0, 1.0, 1.0, 0.5]

        action = JointAction(target_joint_positions=targets, tolerance=0.01)
        agent.add_action(action)

        # Single step should NOT complete
        sim_core.tick()
        agent.update(sim_core._dt)

        assert action.status is ActionStatus.IN_PROGRESS, "Kinematic joint interpolation should take multiple steps"


# ---------------------------------------------------------------------------
# PoseAction (arm integration)
# ---------------------------------------------------------------------------


class TestPoseActionIntegration:
    """Test PoseAction execution with a real arm robot."""

    def test_ee_at_target(self, arm_sim):
        """PoseAction completes and EE is near target position."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        target = [0.0, 0.0, 0.75]
        action = PoseAction(target_position=target, tolerance=0.03)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.COMPLETED
        ee_idx = agent._get_end_effector_link_index()
        link_state = p.getLinkState(agent.body_id, ee_idx, computeForwardKinematics=1, physicsClientId=agent._pid)
        actual_pos = np.array(link_state[0])
        distance = np.linalg.norm(actual_pos - np.array(target))
        assert distance < 0.05

    def test_with_orientation(self, arm_sim):
        """PoseAction with orientation completes, EE position and orientation are near targets."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        target_pos = [0.0, 0.0, 0.75]
        target_orn = [0.0, 0.0, 0.0, 1.0]
        action = PoseAction(
            target_position=target_pos,
            target_orientation=target_orn,
            tolerance=0.03,
        )
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.COMPLETED
        ee_idx = agent._get_end_effector_link_index()
        link_state = p.getLinkState(agent.body_id, ee_idx, computeForwardKinematics=1, physicsClientId=agent._pid)
        # Position check
        actual_pos = np.array(link_state[0])
        distance = np.linalg.norm(actual_pos - np.array(target_pos))
        assert distance < 0.05, f"EE position mismatch: distance={distance:.3f}, actual={actual_pos}"
        # Orientation check: quaternion dot product, |dot| close to 1 means match
        actual_orn = np.array(link_state[1])
        dot = abs(np.dot(actual_orn, np.array(target_orn)))
        assert dot > 0.95, f"EE orientation mismatch: dot={dot:.3f}, actual={actual_orn}"

    def test_explicit_ee_link(self, arm_sim):
        """PoseAction with explicit end_effector_link."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        action = PoseAction(
            target_position=[0.0, 0.0, 0.75],
            end_effector_link="end_effector",
        )
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.COMPLETED

    def test_unreachable_target_fails(self, arm_sim):
        """PoseAction with unreachable target should fail gracefully."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        # Target far beyond reachable workspace of the 4-joint arm
        action = PoseAction(target_position=[10.0, 10.0, 10.0])
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.FAILED

    def test_checks_ee_position(self, arm_sim):
        """PoseAction checks EE Cartesian distance, not just joint convergence."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)
        target = [0.0, 0.0, 0.75]
        action = PoseAction(target_position=target, tolerance=0.03)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.COMPLETED
        # Verify are_ee_at_target is True
        assert agent.are_ee_at_target(target, tolerance=0.05) is True


# ---------------------------------------------------------------------------
# PickAction / DropAction with ee_target_position (arm integration)
# ---------------------------------------------------------------------------


class TestPickDropEEPoseIntegration:
    """Test PickAction/DropAction with ee_target_position on arm robot."""

    def test_pick_ee_pose_completes(self, arm_sim):
        """PickAction with ee_target_position resolves IK and completes."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)

        # EE target at home position [0,0,0.75] to ensure trivial IK and
        # reliable convergence under physics (arm URDF has no inertia data).
        # Box is placed far from the arm to avoid collision interference.
        box = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            pose=Pose.from_xyz(2.0, 0.0, 0.1),
            mass=0.0,
            sim_core=sim_core,
        )
        ee_link = agent.get_num_joints() - 1

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=False,
            ee_target_position=[0.0, 0.0, 0.75],
            attach_link=ee_link,
            attach_relative_pose=Pose.from_xyz(0, 0, 0.06),
            joint_tolerance=0.05,
            joint_max_force=5000.0,
        )
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.COMPLETED
        assert box.is_attached()

    def test_drop_ee_pose_completes(self, arm_sim):
        """DropAction with ee_target_position resolves IK and completes."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)

        box = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            pose=Pose.from_xyz(0.0, 0.0, 0.8),
            mass=0.0,
            sim_core=sim_core,
        )
        ee_link = agent.get_num_joints() - 1
        agent.attach_object(box, parent_link_index=ee_link, relative_pose=Pose.from_xyz(0, 0, 0.06))

        # Near-home EE target minimises joint displacement under gravity
        action = DropAction(
            drop_pose=Pose.from_xyz(0.0, 0.0, 0.5),
            use_approach=False,
            ee_target_position=[0.0, 0.0, 0.75],
            joint_tolerance=0.05,
            joint_max_force=5000.0,
        )
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.COMPLETED
        assert not box.is_attached()

    def test_pick_unreachable_with_continue_true(self, arm_sim):
        """PickAction with unreachable ee_target and continue_on_ik_failure=True completes with warning."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)

        box = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            pose=Pose.from_xyz(2.0, 0.0, 0.1),
            mass=0.0,
            sim_core=sim_core,
        )
        ee_link = agent.get_num_joints() - 1

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=False,
            ee_target_position=[10.0, 10.0, 10.0],  # unreachable
            attach_link=ee_link,
            attach_relative_pose=Pose.from_xyz(0, 0, 0.06),
            continue_on_ik_failure=True,
        )
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        # Should still complete (continue despite IK failure)
        assert action.status == ActionStatus.COMPLETED
        assert box.is_attached()

    def test_pick_unreachable_with_continue_false(self, arm_sim):
        """PickAction with unreachable ee_target and continue_on_ik_failure=False fails."""
        sim_core, mass = arm_sim
        agent = create_arm_agent(sim_core, mass)

        box = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.03, 0.03, 0.03]),
            pose=Pose.from_xyz(2.0, 0.0, 0.1),
            mass=0.0,
            sim_core=sim_core,
        )
        ee_link = agent.get_num_joints() - 1

        action = PickAction(
            target_object_id=box.body_id,
            use_approach=False,
            ee_target_position=[10.0, 10.0, 10.0],  # unreachable
            attach_link=ee_link,
            attach_relative_pose=Pose.from_xyz(0, 0, 0.06),
            continue_on_ik_failure=False,
        )
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.FAILED
        assert not box.is_attached()


# ---------------------------------------------------------------------------
# Rail Arm (prismatic + revolute) — JointAction
# ---------------------------------------------------------------------------


def create_rail_arm_agent(sim_core, mass=None):
    """Create a fixed-base rail arm robot (1 prismatic + 4 revolute) wired to *sim_core*."""
    return Agent.from_urdf(
        urdf_path=RAIL_ARM_URDF,
        pose=Pose.from_xyz(0, 0, 0),
        mass=mass,
        use_fixed_base=True,
        sim_core=sim_core,
    )


class TestPrismaticJointActionIntegration:
    """Test JointAction with prismatic + revolute joints."""

    def test_prismatic_joint_reaches_target(self, arm_sim):
        """JointAction moves the prismatic rail joint to target and completes."""
        sim_core, mass = arm_sim
        agent = create_rail_arm_agent(sim_core, mass)
        # Only set the prismatic joint (index 0) to 0.5 m; leave revolute at 0
        targets = [0.5, 0.0, 0.0, 0.0, 0.0]

        action = JointAction(target_joint_positions=targets, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=0.05)

    def test_mixed_prismatic_revolute_targets(self, arm_sim):
        """JointAction moves all 5 joints (1 prismatic + 4 revolute) to targets."""
        sim_core, mass = arm_sim
        agent = create_rail_arm_agent(sim_core, mass)
        # prismatic=0.7m, then 4 revolute angles
        targets = [0.7, 0.5, 0.3, -0.3, 0.2]

        action = JointAction(target_joint_positions=targets, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=0.05)

    def test_prismatic_kinematic_multi_step(self, arm_sim):
        """Prismatic kinematic interpolation takes multiple steps (no teleport)."""
        sim_core, mass = arm_sim
        if mass != 0.0:
            pytest.skip("multi-step test is kinematic-specific")
        agent = create_rail_arm_agent(sim_core, mass)
        # 1.0 m target at 0.5 m/s velocity = 2 seconds = 480 steps at 240 Hz
        targets = [1.0, 0.0, 0.0, 0.0, 0.0]

        action = JointAction(target_joint_positions=targets, tolerance=0.01)
        agent.add_action(action)

        # Single step should NOT complete
        sim_core.tick()
        agent.update(sim_core._dt)

        assert action.status is ActionStatus.IN_PROGRESS, "Prismatic kinematic interpolation should take multiple steps"

    def test_prismatic_by_name(self, arm_sim):
        """JointAction accepts dict keyed by joint name for prismatic joint."""
        sim_core, mass = arm_sim
        agent = create_rail_arm_agent(sim_core, mass)
        targets = {"rail_joint": 0.4, "elbow_to_wrist": -0.5}

        action = JointAction(target_joint_positions=targets, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=0.05)

    def test_per_joint_tolerance(self, arm_sim):
        """Tolerance can be specified per-joint for mixed prismatic/revolute.

        Prismatic joints measure in metres, revolute in radians.  A scalar
        tolerance of e.g. 0.05 means 5 cm for prismatic but only ~3° for
        revolute.  Per-joint tolerance lets users tighten prismatic accuracy
        without over-constraining revolute joints.
        """
        sim_core, mass = arm_sim
        agent = create_rail_arm_agent(sim_core, mass)
        targets = {"rail_joint": 0.5, "base_to_shoulder": 0.3, "elbow_to_wrist": -0.3}
        # Tight for prismatic (±5 mm), looser for revolute (±0.05 rad ≈ 3°)
        tolerance = {"rail_joint": 0.005, "base_to_shoulder": 0.05, "elbow_to_wrist": 0.05}

        action = JointAction(target_joint_positions=targets, tolerance=tolerance)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status is ActionStatus.COMPLETED
        assert agent.are_joints_at_targets(targets, tolerance=tolerance)


# ---------------------------------------------------------------------------
# Agent-level joint_tolerance (not prismatic-specific)
# ---------------------------------------------------------------------------


class TestAgentJointToleranceIntegration:
    """Test Agent.joint_tolerance fallback chain with real PyBullet agent.

    These tests exercise the tolerance resolution logic (Action > Agent > default)
    and are not specific to prismatic joints.  The rail arm URDF is used because
    its mixed joint types make dict tolerance meaningful.
    """

    def test_agent_level_tolerance_fallback(self, arm_sim):
        """JointAction with tolerance=None falls back to agent.joint_tolerance.

        Fallback chain: Action.tolerance > Agent.joint_tolerance > class default (0.01).
        Tolerance is resolved on the first tick (no need to run to completion).
        """
        sim_core, mass = arm_sim

        # --- Case 1: Agent has custom joint_tolerance ---
        agent_tol = {"rail_joint": 0.005, "base_to_shoulder": 0.05, "elbow_to_wrist": 0.05}
        agent = Agent.from_urdf(
            urdf_path=RAIL_ARM_URDF,
            pose=Pose.from_xyz(0, 0, 0),
            mass=mass,
            use_fixed_base=True,
            sim_core=sim_core,
            joint_tolerance=agent_tol,
        )
        assert agent.joint_tolerance == agent_tol

        action_no_tol = JointAction(target_joint_positions=[0.3, 0.2, 0.0, 0.0, 0.0])
        assert action_no_tol.tolerance is None
        agent.add_action(action_no_tol)
        sim_core.tick()
        agent.update(sim_core._dt)
        # Tolerance resolved from agent on first tick
        assert action_no_tol.tolerance == agent_tol

        # --- Case 2: Action-level tolerance overrides agent-level ---
        agent.clear_actions()
        action_override = JointAction(target_joint_positions=[0.0] * 5, tolerance=0.1)
        agent.add_action(action_override)
        sim_core.tick()
        agent.update(sim_core._dt)
        assert action_override.tolerance == 0.1  # NOT overwritten by agent dict

        # --- Case 3: Agent without custom tolerance → class default 0.01 ---
        agent_default = create_rail_arm_agent(sim_core, mass)
        assert agent_default.joint_tolerance == 0.01

    def test_agent_tolerance_setter(self, arm_sim):
        """Agent.joint_tolerance can be updated after construction."""
        sim_core, mass = arm_sim
        agent = create_rail_arm_agent(sim_core, mass)

        # Default
        assert agent.joint_tolerance == 0.01

        # Set via property
        agent.joint_tolerance = 0.05
        assert agent.joint_tolerance == 0.05

        # Set dict
        agent.joint_tolerance = {"rail_joint": 0.002}
        assert agent.joint_tolerance == {"rail_joint": 0.002}

        # Reset to None → falls back to class default
        agent.joint_tolerance = None
        assert agent.joint_tolerance == 0.01


# ---------------------------------------------------------------------------
# Rail Arm — PoseAction (IK with prismatic joint)
# ---------------------------------------------------------------------------


class TestPrismaticPoseActionIntegration:
    """Test PoseAction / IK with a prismatic joint in the chain."""

    def test_ee_at_target_with_rail(self, arm_sim):
        """PoseAction completes and EE reaches target with rail arm."""
        sim_core, mass = arm_sim
        agent = create_rail_arm_agent(sim_core, mass)
        # Target above base — IK should use a combination of rail + arm joints
        target = [0.1, 0.0, 1.2]
        action = PoseAction(target_position=target, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core)

        assert action.status == ActionStatus.COMPLETED
        ee_idx = agent._get_end_effector_link_index()
        link_state = p.getLinkState(agent.body_id, ee_idx, computeForwardKinematics=1, physicsClientId=agent._pid)
        actual_pos = np.array(link_state[0])
        distance = np.linalg.norm(actual_pos - np.array(target))
        assert distance < 0.1, f"EE too far from target: {distance:.3f}m, actual={actual_pos}"

    def test_ee_at_elevated_rail_height(self, arm_sim):
        """PoseAction reaches elevated target requiring rail displacement."""
        sim_core, mass = arm_sim
        agent = create_rail_arm_agent(sim_core, mass)

        # Target above arm-only reach — IK must use rail (prismatic) + arm joints
        target = [0.1, 0.0, 1.4]
        action = PoseAction(target_position=target, tolerance=0.05)
        agent.add_action(action)

        run_arm_until_idle(agent, sim_core, max_steps=5_000)

        assert action.status == ActionStatus.COMPLETED
        ee_idx = agent._get_end_effector_link_index()
        link_state = p.getLinkState(agent.body_id, ee_idx, computeForwardKinematics=1, physicsClientId=agent._pid)
        actual_pos = np.array(link_state[0])
        distance = np.linalg.norm(actual_pos - np.array(target))
        assert distance < 0.1, f"EE too far from target: {distance:.3f}m, " f"actual={actual_pos}, target={target}"


class TestNestedMoveEndsTheAction:
    """A Pick/Drop whose nested move does not complete -- its path replaced, braked
    or stopped from outside the queue -- ends that way, and does not pick or drop."""

    def _start(self, agent, action, sim, steps=5):
        agent.add_action(action)
        dt = sim._dt
        for _ in range(steps):
            sim.tick()
            agent.update(dt)
        assert action.status is ActionStatus.IN_PROGRESS

    def _run(self, agent, sim, steps=200):
        dt = sim._dt
        for _ in range(steps):
            sim.tick()
            agent.update(dt)

    def test_a_stopped_approach_cancels_the_pick(self, sim):
        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=(4, 0, 0))
        action = PickAction(target_object_id=box.body_id, use_approach=True, approach_offset=0.5)
        self._start(agent, action, sim)
        agent.stop()
        self._run(agent, sim)
        assert action.status is ActionStatus.CANCELLED
        assert "approach pose ended CANCELLED" in (action.error_message or "")
        assert not box.is_attached()

    def test_a_replaced_move_to_the_pick_pose_cancels_the_pick(self, sim):
        agent = create_agent(sim)
        box = create_pickable_box(sim, pos=(4, 0, 0))
        action = PickAction(target_object_id=box.body_id, use_approach=False)
        self._start(agent, action, sim)
        agent.set_path([Pose.from_xyz(0.0, 2.0, 0.0)])  # from outside the queue
        self._run(agent, sim, steps=2000)
        assert action.status is ActionStatus.CANCELLED
        assert "pick pose ended CANCELLED" in (action.error_message or "")
        assert not box.is_attached()

    def test_a_stopped_approach_cancels_the_drop(self, sim):
        agent = create_agent(sim, pose=Pose.from_xyz(1.9, 0, 0))
        box = create_pickable_box(sim)
        pick = PickAction(target_object_id=box.body_id, use_approach=False)
        agent.add_action(pick)
        run_until_idle(agent, sim)
        assert box.is_attached()
        action = DropAction(drop_pose=Pose(position=[5, 0, 0]), use_approach=True, approach_offset=0.5)
        self._start(agent, action, sim)
        agent.stop()
        self._run(agent, sim)
        assert action.status is ActionStatus.CANCELLED
        assert box.is_attached(), "nothing is dropped where the agent was stopped"
