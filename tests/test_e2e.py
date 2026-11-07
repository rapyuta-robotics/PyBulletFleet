"""
End-to-end tests for PyBulletFleet simulation workflows.

Each test runs a complete scenario through MultiRobotSimulationCore.step_once(),
verifying final state after action sequences complete.

Based on examples/:
- action_system_demo.py (Pick → Drop → Move → Wait)
- pick_drop_mobile_100robots_demo.py (bulk Pick → Move → Drop)
- path_following_demo.py (waypoint path following)
- collision_features_demo.py (multi-mode collision detection)
"""

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet.action import (
    DropAction,
    JointAction,
    MoveAction,
    PickAction,
    PoseAction,
    WaitAction,
)
from pybullet_fleet.agent import Agent, AgentSpawnParams, IKParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Path, Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams
from pybullet_fleet.types import (
    CollisionDetectionMethod,
    CollisionMode,
    MotionMode,
    SpatialHashCellSizeMode,
)

TOL = 0.15  # E2E position tolerance (generous — accounts for kinematic stepping)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sim():
    """Headless kinematics sim (fast timestep for quick convergence)."""
    params = SimulationParams(
        gui=False,
        physics=False,
        monitor=False,
        timestep=0.1,
        target_rtf=0,  # max speed (no sleep)
        collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        ignore_static_collision=True,
        log_level="warning",
    )
    sc = MultiRobotSimulationCore(params)
    sc.set_collision_spatial_hash_cell_size_mode()
    yield sc
    p.disconnect(sc.client)


def make_mobile_agent(sim, position=(0, 0, 0), motion_mode=MotionMode.OMNIDIRECTIONAL):
    """Spawn a mobile robot agent."""
    return Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(*position),
            collision_mode=CollisionMode.NORMAL_3D,
            motion_mode=motion_mode,
        ),
        sim_core=sim,
    )


def make_pickable_box(sim, position=(2, 0, 0)):
    """Spawn a pickable box object."""
    return SimObject.from_params(
        SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            initial_pose=Pose.from_xyz(*position),
            mass=0.0,
            pickable=True,
        ),
        sim_core=sim,
    )


def run_until_done(sim, agents, *, max_steps=5000):
    """Step sim until all agent action queues are empty."""
    for step in range(max_steps):
        sim.step_once()
        if all(a.is_action_queue_empty() and not a.is_moving for a in agents):
            return step + 1
    raise AssertionError(f"Agents did not finish within {max_steps} steps")


# ---------------------------------------------------------------------------
# Shared assertion helpers (usable by all test classes)
# ---------------------------------------------------------------------------


def get_ee_pos(agent, ee_link):
    """Get end-effector world position via forward kinematics."""
    state = p.getLinkState(
        agent.body_id,
        ee_link,
        computeForwardKinematics=1,
        physicsClientId=agent._pid,
    )
    return np.array(state[0])


def assert_joints_near(agent, targets, tol, label=""):
    """Assert all joints are near target values (list-based, by index)."""
    joint_states = agent.get_all_joints_state()
    for i, target in enumerate(targets):
        pos = joint_states[i][0]
        assert abs(pos - target) < tol, f"{label} Joint {i}: expected ~{target}, got {pos:.3f}"


def assert_joints_near_dict(agent, targets_dict, tol, label=""):
    """Assert named joints are near their target values (dict-based, by name)."""
    for name, target in targets_dict.items():
        pos = agent.get_joint_state(name)[0]
        assert abs(pos - target) < tol, f"{label} Joint '{name}': expected ~{target:.3f}, got {pos:.3f}"


def assert_base_near(agent, expected_xy, tol, label=""):
    """Assert agent base XY is near expected position."""
    base_pos = np.array(agent.get_pose().position[:2])
    dist = np.linalg.norm(base_pos - np.array(expected_xy))
    assert dist < tol, f"{label} Base should be near {expected_xy}, got {base_pos} (dist={dist:.3f})"


def assert_ee_pos_near(agent, ee_link, expected_pos, tol, label=""):
    """Assert end-effector position is near expected 3D position."""
    ee_pos = get_ee_pos(agent, ee_link)
    dist = np.linalg.norm(ee_pos - np.array(expected_pos))
    assert dist < tol, f"{label} EE should be near {expected_pos}, got {ee_pos} (dist={dist:.3f})"


# ============================================================================
# E2E: Pick → Move → Drop
# ============================================================================


class TestPickMoveDropE2E:
    """Full pick-move-drop workflow (based on action_system_demo.py)."""

    def test_pick_move_drop_single_agent(self, sim):
        agent = make_mobile_agent(sim, position=(0, 0, 0))
        box = make_pickable_box(sim, position=(2, 0, 0))

        # Verify initial box position
        box_pos = np.array(box.get_pose().position)
        assert np.linalg.norm(box_pos[:2] - np.array([2, 0])) < TOL, f"Box should start at (2,0), got {box_pos}"

        drop_pose = Pose.from_xyz(5, 3, 0)

        # Note: PickAction.target_object_id expects body_id (PyBullet ID), not object_id
        agent.add_action_sequence(
            [
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=False,
                    pick_offset=0.3,
                    attach_relative_pose=Pose.from_xyz(0, 0, 0.3),
                ),
                MoveAction(
                    path=Path.from_positions([drop_pose.position]),
                    final_orientation_align=False,
                ),
                DropAction(
                    drop_pose=drop_pose,
                    use_approach=False,
                    drop_offset=0.3,
                ),
            ]
        )

        run_until_done(sim, [agent])

        # Box should be near drop position
        box_pos = np.array(box.get_pose().position)
        drop_pos = np.array(drop_pose.position)
        assert (
            np.linalg.norm(box_pos[:2] - drop_pos[:2]) < TOL
        ), f"Box should be near drop pose. box={box_pos}, drop={drop_pos}"

    def test_pick_move_drop_with_approach(self, sim):
        agent = make_mobile_agent(sim, position=(0, 0, 0))
        box = make_pickable_box(sim, position=(3, 0, 0))
        drop_pose = Pose.from_xyz(6, 4, 0)

        agent.add_action_sequence(
            [
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=True,
                    approach_offset=1.0,
                    pick_offset=0.3,
                    attach_relative_pose=Pose.from_xyz(0, 0, 0.3),
                ),
                MoveAction(
                    path=Path.from_positions([drop_pose.position]),
                ),
                DropAction(
                    drop_pose=drop_pose,
                    use_approach=True,
                    approach_offset=1.0,
                    drop_offset=0.3,
                ),
            ]
        )

        run_until_done(sim, [agent])

        box_pos = np.array(box.get_pose().position)
        drop_pos = np.array(drop_pose.position)
        assert np.linalg.norm(box_pos[:2] - drop_pos[:2]) < TOL

    def test_move_then_wait(self, sim):
        agent = make_mobile_agent(sim, position=(0, 0, 0))
        goal = [4, 2, 0]

        agent.add_action_sequence(
            [
                MoveAction(path=Path.from_positions([goal])),
                WaitAction(duration=0.5, action_type="idle"),
            ]
        )

        steps = run_until_done(sim, [agent])

        assert_base_near(agent, goal[:2], TOL, "After move+wait:")
        # Move + Wait(0.5s) with timestep=0.1s → at least 5 steps for wait alone
        min_wait_steps = int(0.5 / sim.params.timestep)
        assert steps >= min_wait_steps, f"Expected ≥{min_wait_steps} steps (wait portion), got {steps}"


# ============================================================================
# E2E: Path Following
# ============================================================================


class TestPathFollowingE2E:
    """Path following to completion (based on path_following_demo.py)."""

    def test_square_path(self, sim):
        agent = make_mobile_agent(sim, position=(0, 0, 0))

        waypoints = [[2, 0, 0], [2, 2, 0], [0, 2, 0], [0, 0, 0]]
        agent.add_action_sequence(
            [
                MoveAction(path=Path.from_positions(waypoints)),
            ]
        )

        run_until_done(sim, [agent])

        assert_base_near(agent, [0, 0], TOL, "After square path:")

    def test_multi_agent_paths(self, sim):
        agents = [make_mobile_agent(sim, position=(i * 3, 0, 0)) for i in range(3)]

        for i, agent in enumerate(agents):
            goal = [i * 3, 5, 0]
            agent.add_action_sequence(
                [
                    MoveAction(path=Path.from_positions([goal])),
                ]
            )

        run_until_done(sim, agents)

        for i, agent in enumerate(agents):
            assert_base_near(agent, [i * 3, 5], TOL, f"Agent {i}:")


# ============================================================================
# E2E: Collision Detection During Movement
# ============================================================================


class TestCollisionDuringMovementE2E:
    """Collision detection while agents are moving (based on collision_features_demo.py)."""

    @pytest.fixture
    def sim_with_collision(self):
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            timestep=0.1,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            collision_margin=0.05,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=2.0,
            ignore_static_collision=False,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        sc.set_collision_spatial_hash_cell_size_mode()
        yield sc
        p.disconnect(sc.client)

    def test_collision_detected_during_step(self, sim_with_collision):
        """Two agents converging should produce collision pairs at some step."""
        sc = sim_with_collision
        a1 = make_mobile_agent(sc, position=(0, 0, 0))
        a2 = make_mobile_agent(sc, position=(3, 0, 0))

        a1.set_goal_pose(Pose.from_xyz(3, 0, 0))
        a2.set_goal_pose(Pose.from_xyz(0, 0, 0))

        collision_ever = False
        for _ in range(500):
            sc.step_once()
            if len(sc.get_active_collision_pairs()) > 0:
                collision_ever = True
                break

        assert collision_ever, "Converging agents should collide at some point"

    def test_disabled_mode_no_collision(self, sim_with_collision):
        """Agent with DISABLED collision mode is never in collision pairs."""
        sc = sim_with_collision
        a1 = make_mobile_agent(sc, position=(0, 0, 0))
        a2 = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0.3, 0, 0),
                collision_mode=CollisionMode.DISABLED,
            ),
            sim_core=sc,
        )
        sc._moved_this_step.add(a1.object_id)

        sc.step_once()
        pairs = sc.get_active_collision_pairs()
        ids = {obj_id for pair in pairs for obj_id in pair}
        assert a2.object_id not in ids, "DISABLED agent should never appear in collision pairs"


# ============================================================================
# E2E: Multi-Agent Bulk Operations
# ============================================================================


class TestMultiAgentBulkE2E:
    """Multi-agent scenarios (based on pick_drop_mobile_100robots_demo.py)."""

    def test_multiple_agents_pick_and_drop(self, sim):
        """3 agents each pick a box, move, and drop it."""
        num_agents = 3
        agents = []
        boxes = []

        for i in range(num_agents):
            agent = make_mobile_agent(sim, position=(i * 4, 0, 0))
            box = make_pickable_box(sim, position=(i * 4, 1, 0))
            agents.append(agent)
            boxes.append(box)

        drop_y = 5.0
        for i, (agent, box) in enumerate(zip(agents, boxes)):
            drop_pose = Pose.from_xyz(i * 4, drop_y, 0)
            agent.add_action_sequence(
                [
                    PickAction(
                        target_object_id=box.body_id,
                        use_approach=False,
                        pick_offset=0.3,
                        attach_relative_pose=Pose.from_xyz(0, 0, 0.3),
                    ),
                    MoveAction(path=Path.from_positions([drop_pose.position])),
                    DropAction(drop_pose=drop_pose, use_approach=False, drop_offset=0.3),
                ]
            )

        run_until_done(sim, agents, max_steps=10000)

        for i, box in enumerate(boxes):
            box_pos = np.array(box.get_pose().position)
            expected_y = drop_y
            assert abs(box_pos[1] - expected_y) < TOL, f"Box {i} should be near y={expected_y}, got y={box_pos[1]}"


# ============================================================================
# E2E: Arm Robot Pick/Drop (based on pick_drop_arm_action_demo.py)
# ============================================================================


class TestArmPickDropE2E:
    """Robot arm JointAction → Pick → JointAction → Drop workflow."""

    @pytest.fixture(
        params=[
            pytest.param((True, None), id="physics"),
            pytest.param((False, 0.0), id="kinematic"),
        ]
    )
    def arm_env(self, request):
        """Parametrized fixture returning (sim_core, mass).

        - physics:   stepSimulation enabled, URDF mass (motor control)
        - kinematic: no stepSimulation, mass=0.0 (kinematic interpolation)
        """
        physics, mass = request.param
        params = SimulationParams(
            gui=False,
            physics=physics,
            monitor=False,
            timestep=1.0 / 240.0,
            target_rtf=0,
            collision_detection_method=CollisionDetectionMethod.CONTACT_POINTS,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=2.0,
            ignore_static_collision=True,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        sc.set_collision_spatial_hash_cell_size_mode()
        yield sc, mass
        p.disconnect(sc.client)

    # Helper: run until action queue is empty
    JOINT_TOL = 0.1  # generous tolerance for E2E joint checks
    EE_TOL = 0.15  # generous tolerance for E2E EE position checks
    BOX_TOL = 0.3  # generous tolerance for box final position

    @staticmethod
    def _run_arm_until_done(sc, arm, max_steps=2_000):
        for _ in range(max_steps):
            sc.step_once()
            if arm.is_action_queue_empty():
                return
        raise AssertionError(f"Arm did not finish actions within {max_steps} steps")

    def test_arm_joint_pick_joint_drop(self, arm_env):
        """Arm moves to pick pose, picks box, moves to place pose, drops box.

        Asserts intermediate state after each phase:
        - After pick: box attached, joints at pick_joints
        - After drop: box detached, joints at place_joints, box near drop pose
        """
        sc, arm_mass = arm_env

        arm = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/arm_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0),
                mass=arm_mass,
                collision_mode=CollisionMode.DISABLED,
                use_fixed_base=True,
            ),
            sim_core=sc,
        )

        box = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                initial_pose=Pose.from_xyz(0.3, 0, 0.1),
                mass=0.0,
                pickable=True,
            ),
            sim_core=sc,
        )

        ee_link = arm.get_num_joints() - 1
        pick_joints = [1.5, 1.5, 1.5, 0.0]
        place_joints = [-1.5, 1.5, 1.5, 0.0]
        offset_pose = Pose.from_xyz(0, 0, 0.14)

        # --- Phase 1: Move to pick + Pick ---
        arm.add_action_sequence(
            [
                JointAction(target_joint_positions=pick_joints, tolerance=0.05),
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=False,
                    attach_link=ee_link,
                    attach_relative_pose=offset_pose,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after pick
        assert box.is_attached(), "Box should be attached after pick"
        assert_joints_near(arm, pick_joints, self.JOINT_TOL, "After pick:")

        # --- Phase 2: Move to place + Drop ---
        arm.add_action_sequence(
            [
                JointAction(target_joint_positions=place_joints, tolerance=0.05),
                DropAction(
                    drop_pose=Pose.from_xyz(-0.3, 0, 0.1),
                    use_approach=False,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after drop
        assert not box.is_attached(), "Box should be detached after drop"
        assert_joints_near(arm, place_joints, self.JOINT_TOL, "After drop:")
        box_pos = np.array(box.get_pose().position)
        assert (
            np.linalg.norm(box_pos - np.array([-0.3, 0, 0.1])) < self.BOX_TOL
        ), f"Box should be near drop pose (-0.3, 0, 0.1), got {box_pos}"

    def test_pick_drop_with_inline_joint_targets(self, arm_env):
        """PickAction/DropAction with joint_targets parameter (joints move inside action).

        Asserts intermediate state after each phase:
        - After pick: box attached, joints at pick_joints
        - After drop: box detached, joints at place_joints, box near drop pose
        """
        sc, arm_mass = arm_env

        arm = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/arm_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0),
                mass=arm_mass,
                collision_mode=CollisionMode.DISABLED,
                use_fixed_base=True,
            ),
            sim_core=sc,
        )

        box = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                initial_pose=Pose.from_xyz(0.3, 0, 0.1),
                mass=0.0,
                pickable=True,
            ),
            sim_core=sc,
        )

        ee_link = arm.get_num_joints() - 1
        pick_joints = [1.5, 1.5, 1.5, 0.0]
        place_joints = [-1.5, 1.5, 1.5, 0.0]
        offset_pose = Pose.from_xyz(0, 0, 0.14)

        # --- Phase 1: Pick with inline joint_targets ---
        arm.add_action_sequence(
            [
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=False,
                    attach_link=ee_link,
                    attach_relative_pose=offset_pose,
                    joint_targets=pick_joints,
                    joint_tolerance=0.05,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after pick
        assert box.is_attached(), "Box should be attached after pick"
        assert_joints_near(arm, pick_joints, self.JOINT_TOL, "After pick:")

        # --- Phase 2: Drop with inline joint_targets ---
        arm.add_action_sequence(
            [
                DropAction(
                    drop_pose=Pose.from_xyz(-0.3, 0, 0.1),
                    use_approach=False,
                    joint_targets=place_joints,
                    joint_tolerance=0.05,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after drop
        assert not box.is_attached(), "Box should be detached after drop"
        assert_joints_near(arm, place_joints, self.JOINT_TOL, "After drop:")
        box_pos = np.array(box.get_pose().position)
        assert (
            np.linalg.norm(box_pos - np.array([-0.3, 0, 0.1])) < self.BOX_TOL
        ), f"Box should be near drop pose (-0.3, 0, 0.1), got {box_pos}"

    def test_arm_pose_pick_pose_drop(self, arm_env):
        """Arm moves via PoseAction, picks box, moves via PoseAction, drops box.

        Mirrors test_arm_joint_pick_joint_drop but uses PoseAction (IK-based EE
        control) instead of JointAction.

        Asserts intermediate state after each phase:
        - After pick: box attached, EE near pick_pos
        - After drop: box detached, EE near drop_pos, box near drop pose
        """
        sc, arm_mass = arm_env

        arm = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/arm_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0),
                mass=arm_mass,
                collision_mode=CollisionMode.DISABLED,
                use_fixed_base=True,
            ),
            sim_core=sc,
        )

        box = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                initial_pose=Pose.from_xyz(0.0, -0.3, 0.1),
                mass=0.0,
                pickable=True,
            ),
            sim_core=sc,
        )

        ee_link = arm.get_num_joints() - 1
        pick_pos = [0.0, -0.3, 0.3]
        drop_pos = [0.0, 0.3, 0.3]
        offset_pose = Pose.from_xyz(0, 0, 0.14)

        # --- Phase 1: PoseAction to pick position + Pick ---
        arm.add_action_sequence(
            [
                PoseAction(target_position=pick_pos, tolerance=0.02),
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=False,
                    attach_link=ee_link,
                    attach_relative_pose=offset_pose,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after pick
        assert box.is_attached(), "Box should be attached after pick"
        assert_ee_pos_near(arm, ee_link, pick_pos, self.EE_TOL, "After pick:")

        # --- Phase 2: PoseAction to drop position + Drop ---
        arm.add_action_sequence(
            [
                PoseAction(target_position=drop_pos, tolerance=0.02),
                DropAction(
                    drop_pose=Pose.from_xyz(0.0, 0.3, 0.1),
                    use_approach=False,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after drop
        assert not box.is_attached(), "Box should be detached after drop"
        assert_ee_pos_near(arm, ee_link, drop_pos, self.EE_TOL, "After drop:")
        box_pos = np.array(box.get_pose().position)
        assert (
            np.linalg.norm(box_pos - np.array([0.0, 0.3, 0.1])) < self.BOX_TOL
        ), f"Box should be near drop pose (0, 0.3, 0.1), got {box_pos}"

    def test_pick_drop_with_ee_target_position(self, arm_env):
        """PickAction/DropAction with ee_target_position (IK-based EE control inside action).

        Asserts intermediate state after each phase:
        - After pick: box attached, EE near pick_pos
        - After drop: box detached, EE near drop_pos, box near drop pose
        """
        sc, arm_mass = arm_env

        arm = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/arm_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0),
                mass=arm_mass,
                collision_mode=CollisionMode.DISABLED,
                use_fixed_base=True,
            ),
            sim_core=sc,
        )

        box = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05]),
                initial_pose=Pose.from_xyz(0.0, -0.3, 0.1),
                mass=0.0,
                pickable=True,
            ),
            sim_core=sc,
        )

        ee_link = arm.get_num_joints() - 1
        pick_pos = [0.0, -0.3, 0.3]
        drop_pos = [0.0, 0.3, 0.3]
        offset_pose = Pose.from_xyz(0, 0, 0.14)

        # --- Phase 1: Pick with ee_target_position ---
        arm.add_action_sequence(
            [
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=False,
                    ee_target_position=pick_pos,
                    attach_link=ee_link,
                    attach_relative_pose=offset_pose,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after pick
        assert box.is_attached(), "Box should be attached after pick"
        assert_ee_pos_near(arm, ee_link, pick_pos, self.EE_TOL, "After pick:")

        # --- Phase 2: Drop with ee_target_position ---
        arm.add_action_sequence(
            [
                DropAction(
                    drop_pose=Pose.from_xyz(0.0, 0.3, 0.1),
                    use_approach=False,
                    ee_target_position=drop_pos,
                ),
            ]
        )
        self._run_arm_until_done(sc, arm)

        # Assert after drop
        assert not box.is_attached(), "Box should be detached after drop"
        assert_ee_pos_near(arm, ee_link, drop_pos, self.EE_TOL, "After drop:")
        box_pos = np.array(box.get_pose().position)
        assert (
            np.linalg.norm(box_pos - np.array([0.0, 0.3, 0.1])) < self.BOX_TOL
        ), f"Box should be near drop pose (0, 0.3, 0.1), got {box_pos}"


# ============================================================================
# E2E: Mobile Manipulator Pick/Drop (based on mobile_manipulator_demo.py)
# ============================================================================

MOBILE_MANIPULATOR_URDF = "robots/mobile_manipulator.urdf"

_ARM_JOINT_NAMES = (
    "mount_to_shoulder",
    "shoulder_to_elbow",
    "elbow_to_wrist",
    "wrist_to_end",
)


class TestMobileManipulatorE2E:
    """Mobile manipulator: Navigate → Pick → Navigate → Drop workflows.

    Covers two scenarios mirroring mobile_manipulator_demo.py:
    - test_joint_target_pick_drop: Joint-target arm control (Part 1)
    - test_ik_pick_drop: IK-based ee_target_position control (Part 2)

    Both verify:
    - Base navigation moves agent near target
    - Pick attaches box to EE link
    - Attached box follows agent during navigation
    - Drop releases box near the drop area
    """

    BASE_TOL = 0.2  # generous tolerance for base position
    EE_TOL = 0.15  # EE Cartesian tolerance
    BOX_TOL = 0.3  # box final position tolerance

    @pytest.fixture
    def sim(self):
        """Headless kinematic sim for mobile manipulator tests."""
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            timestep=0.1,
            target_rtf=0,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=2.0,
            ignore_static_collision=True,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        sc.set_collision_spatial_hash_cell_size_mode()
        yield sc
        p.disconnect(sc.client)

    @staticmethod
    def _run_until_done(sc, agent, max_steps=3_000):
        for _ in range(max_steps):
            sc.step_once()
            if agent.is_action_queue_empty():
                return
        raise AssertionError(f"Agent did not finish actions within {max_steps} steps")

    def _make_robot(self, sc, ik_joint_names=None):
        """Spawn mobile manipulator with optional ik_joint_names."""
        ik_params = IKParams(ik_joint_names=ik_joint_names) if ik_joint_names else None
        return Agent.from_params(
            AgentSpawnParams(
                urdf_path=MOBILE_MANIPULATOR_URDF,
                initial_pose=Pose.from_xyz(0, 0, 0.3),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
                controller={
                    "max_linear_vel": 2.0,
                    "max_angular_vel": 1.5,
                },
                mass=0.0,
                use_fixed_base=False,
                collision_mode=CollisionMode.DISABLED,
                ik_params=ik_params,
            ),
            sim_core=sc,
        )

    def _make_box(self, sc, position):
        """Spawn pickable box."""
        return SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(
                    shape_type="box",
                    half_extents=[0.05, 0.05, 0.05],
                ),
                collision_shape=ShapeParams(
                    shape_type="box",
                    half_extents=[0.05, 0.05, 0.05],
                ),
                initial_pose=Pose.from_xyz(*position),
                mass=0.0,
                pickable=True,
            ),
            sim_core=sc,
        )

    @staticmethod
    def _find_ee_link(agent):
        """Find end_effector link index."""
        for i in range(p.getNumJoints(agent.body_id)):
            info = p.getJointInfo(agent.body_id, i)
            if info[12].decode("utf-8") == "end_effector":
                return i
        raise ValueError("end_effector link not found")

    def test_joint_target_pick_drop(self, sim):
        """Navigate → JointAction pick → Navigate → JointAction drop.

        Scenario (simplified Part 1 of mobile_manipulator_demo.py):
        1. Navigate near box
        2. Arm to pick pose via JointAction → Pick box at EE
        3. Arm to carry pose → Navigate to drop area
        4. Arm to drop pose → Drop box (drop_relative_pose)
        Verify: box attached after pick, box detached after drop, box near drop area.
        """
        robot = self._make_robot(sim)
        box = self._make_box(sim, [1.5, 0, 0.8])
        ee_link = self._find_ee_link(robot)

        import math

        arm_pick = {
            "mount_to_shoulder": math.pi / 2,
            "shoulder_to_elbow": 1.2,
            "elbow_to_wrist": 1.0,
            "wrist_to_end": 0.0,
        }
        arm_carry = {
            "mount_to_shoulder": math.pi / 2,
            "shoulder_to_elbow": -0.3,
            "elbow_to_wrist": 0.8,
            "wrist_to_end": 0.0,
        }

        # Phase 1: Navigate near box + pick
        robot.add_action_sequence(
            [
                MoveAction(
                    path=Path.from_positions([[0.9, 0.0, 0.3]]),
                    final_orientation_align=False,
                ),
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=False,
                    attach_link=ee_link,
                    attach_relative_pose=Pose.from_xyz(0, 0, 0.07),
                    joint_targets=arm_pick,
                    joint_tolerance=0.05,
                ),
            ]
        )
        self._run_until_done(sim, robot)

        # Assert after pick: box attached, base near box, joints at arm_pick
        assert box.is_attached(), "Box should be attached after pick"
        assert_base_near(robot, [1.5, 0.0], self.BASE_TOL, "After pick:")
        assert_joints_near_dict(robot, arm_pick, 0.1, "After pick:")

        # Phase 2: Carry + navigate + drop
        robot.add_action_sequence(
            [
                JointAction(target_joint_positions=arm_carry, tolerance=0.05),
                MoveAction(
                    path=Path.from_positions([[0.0, 0.9, 0.3]]),
                    final_orientation_align=False,
                ),
                DropAction(
                    drop_pose=Pose.from_xyz(0, 0.9, 0.8),
                    place_gently=True,
                    use_approach=False,
                    joint_targets={
                        "mount_to_shoulder": 0.0,
                        "shoulder_to_elbow": 1.2,
                        "elbow_to_wrist": 1.0,
                        "wrist_to_end": 0.0,
                    },
                    joint_tolerance=0.05,
                    drop_relative_pose=Pose.from_xyz(0, 0, 0),
                ),
            ]
        )
        self._run_until_done(sim, robot)

        arm_drop = {
            "mount_to_shoulder": 0.0,
            "shoulder_to_elbow": 1.2,
            "elbow_to_wrist": 1.0,
            "wrist_to_end": 0.0,
        }

        # Assert after drop: box detached, joints at arm_drop, base near drop area, box near EE
        assert not box.is_attached(), "Box should be detached after drop"
        assert_joints_near_dict(robot, arm_drop, 0.1, "After drop:")
        assert_base_near(robot, [0.0, 0.9], self.BASE_TOL, "After drop:")
        box_pos = np.array(box.get_pose().position)
        ee_pos = get_ee_pos(robot, ee_link)
        dist_box_ee = np.linalg.norm(box_pos - ee_pos)
        assert dist_box_ee < self.BOX_TOL, (
            f"Box should be near EE after drop_relative_pose=(0,0,0), " f"box={box_pos}, ee={ee_pos}, dist={dist_box_ee:.3f}"
        )

    def test_ik_pick_drop(self, sim):
        """Navigate → IK pick (ee_target_position) → Navigate → IK drop.

        Scenario (simplified Part 2 of mobile_manipulator_demo.py):
        1. Navigate near box
        2. Pick box via ee_target_position (IK auto-locks wheels)
        3. Navigate to drop area
        4. Drop box via ee_target_position + drop_relative_pose
        Verify: IK works on mobile manipulator, box follows during nav, box released.
        """
        robot = self._make_robot(sim, ik_joint_names=_ARM_JOINT_NAMES)
        box = self._make_box(sim, [0.5, -1.5, 0.6])
        ee_link = self._find_ee_link(robot)

        import math

        arm_carry = {
            "mount_to_shoulder": math.pi / 2,
            "shoulder_to_elbow": -0.3,
            "elbow_to_wrist": 0.8,
            "wrist_to_end": 0.0,
        }

        # Phase 1: Navigate near box + IK pick
        robot.add_action_sequence(
            [
                MoveAction(
                    path=Path.from_positions([[0.0, -1.0, 0.3]]),
                    final_orientation_align=False,
                ),
                PickAction(
                    target_object_id=box.body_id,
                    use_approach=False,
                    attach_link=ee_link,
                    attach_relative_pose=Pose.from_xyz(0, 0, 0.07),
                    ee_target_position=[0.5, -1.5, 0.6],
                    ee_end_effector_link=ee_link,
                    ee_tolerance=0.05,
                ),
            ]
        )
        self._run_until_done(sim, robot)

        # Assert after IK pick: box attached, base near box, EE near pick target
        assert box.is_attached(), "Box should be attached after IK pick"
        assert_base_near(robot, [0.5, -1.5], self.BASE_TOL, "After IK pick:")
        assert_ee_pos_near(robot, ee_link, [0.5, -1.5, 0.6], self.EE_TOL, "After IK pick:")
        # Wheels should still be near 0 (IK locked them)
        for wheel_name in ("base_to_left_wheel", "base_to_right_wheel"):
            wheel_pos = robot.get_joint_state(wheel_name)[0]
            assert abs(wheel_pos) < 0.1, f"Wheel '{wheel_name}' should be near 0 (IK locked), got {wheel_pos:.3f}"

        # Phase 2: Carry + navigate + IK drop
        robot.add_action_sequence(
            [
                JointAction(target_joint_positions=arm_carry, tolerance=0.05),
                MoveAction(
                    path=Path.from_positions([[-1.0, 0.0, 0.3]]),
                    final_orientation_align=False,
                ),
                DropAction(
                    drop_pose=Pose.from_xyz(-0.35, 0.0, 0.6),
                    place_gently=True,
                    use_approach=False,
                    ee_target_position=[-0.35, 0.0, 0.6],
                    ee_end_effector_link=ee_link,
                    ee_tolerance=0.05,
                    drop_relative_pose=Pose.from_xyz(0, 0, 0),
                ),
            ]
        )
        self._run_until_done(sim, robot)

        # Assert after IK drop: box detached, EE near target, base near drop area, box near EE
        assert not box.is_attached(), "Box should be detached after IK drop"
        assert_ee_pos_near(robot, ee_link, [-0.35, 0.0, 0.6], self.EE_TOL, "After IK drop:")
        assert_base_near(robot, [-0.35, 0.0], self.BASE_TOL, "After IK drop:")
        # drop_relative_pose=(0,0,0) → box should be near EE
        ee_pos = get_ee_pos(robot, ee_link)
        box_pos = np.array(box.get_pose().position)
        dist_box_ee = np.linalg.norm(box_pos - ee_pos)
        assert dist_box_ee < self.BOX_TOL, (
            f"Box should be near EE after drop, box={box_pos}, ee={ee_pos}, " f"dist={dist_box_ee:.3f}"
        )
        # Wheels should still be near 0 (IK locked them)
        for wheel_name in ("base_to_left_wheel", "base_to_right_wheel"):
            wheel_pos = robot.get_joint_state(wheel_name)[0]
            assert abs(wheel_pos) < 0.1, f"Wheel '{wheel_name}' should be near 0 (IK locked), got {wheel_pos:.3f}"
