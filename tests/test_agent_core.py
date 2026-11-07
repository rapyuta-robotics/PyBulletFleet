"""
Tests for Agent class core functionality.

This module tests:
- Agent creation from mesh and URDF
- Agent properties and parameters
- Goal setting and path following
- Motion modes (omnidirectional vs differential)
- Velocity and acceleration constraints
- Agent state management
- Motion update and goal reaching
- URDF joint control (set_joint_target / get_joint_state)
- URDF link attach/detach
- Velocity/acceleration capping during motion
"""

import os
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data
import pytest
from scipy.spatial.transform import Rotation as R

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.events import EventBus
from pybullet_fleet.geometry import Pose, Path
from pybullet_fleet.sim_object import SimObject, ShapeParams
from pybullet_fleet.types import CollisionMode, MotionMode, PosePhase
from tests.conftest import MockSimCore

MESH_PATH = os.path.join(os.path.dirname(__file__), "..", "pybullet_fleet", "mesh", "cube.obj")
ARM_URDF = "robots/arm_robot.urdf"
MOBILE_URDF = "robots/mobile_robot.urdf"
MOBILE_MANIPULATOR_URDF = "robots/mobile_manipulator.urdf"


@pytest.fixture
def pybullet_env():
    """Setup and teardown PyBullet environment"""
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF("plane.urdf")

    yield physics_client, plane_id

    p.disconnect()


_SKIP = object()  # Sentinel to indicate "skip this check"


def assert_agent_properties(
    agent,
    *,
    # SimObject properties (defaults match create_mesh_agent defaults)
    mass=0.0,
    pickable=False,  # Agent default: not pickable
    name=None,
    position=(0, 0, 0),
    orientation=(0, 0, 0, 1),
    is_kinematic=True,  # mass=0 → kinematic
    collision_mode=CollisionMode.NORMAL_3D,
    # Agent-specific properties (defaults match Agent.__init__ defaults)
    motion_mode=MotionMode.DIFFERENTIAL,
    max_linear_vel=(2.0, 2.0, 2.0),
    max_linear_accel=(5.0, 5.0, 5.0),
    max_angular_vel=(3.0, 3.0, 3.0),
    max_angular_accel=(10.0, 10.0, 10.0),
    is_moving=False,
    is_urdf=False,
    use_fixed_base=False,
    user_data=_SKIP,  # {} from SimObject but _SKIP avoids dict-in-default-arg issues
    # Tolerances
    pos_tolerance=0.01,
    orient_tolerance=0.01,
    vel_tolerance=1e-6,
):
    """
    Assert multiple Agent properties in one call.

    Defaults match the most common test pattern: a mesh agent created by
    create_mesh_agent() with all default arguments.  This means calling
    ``assert_agent_properties(agent)`` with no extra kwargs verifies the
    full default state, just like test_sim_object's assert_object_properties.

    Pass _SKIP to skip a specific check.

    Args:
        agent: Agent instance to verify
        mass: Expected mass (default 0.0 — create_mesh_agent default)
        pickable: Expected pickable flag (default False — Agent overrides SimObject)
        name: Expected name (default None)
        position: Expected (x, y, z) position tuple (default (0,0,0))
        orientation: Expected (x, y, z, w) quaternion (default (0,0,0,1))
        is_kinematic: Expected is_kinematic flag (default True — mass=0)
        collision_mode: Expected CollisionMode (default NORMAL_3D)
        motion_mode: Expected MotionMode (default DIFFERENTIAL)
        max_linear_vel: Expected [vx, vy, vz] (default [2,2,2])
        max_linear_accel: Expected [ax, ay, az] (default [5,5,5])
        max_angular_vel: Expected [wx, wy, wz] (default [3,3,3])
        max_angular_accel: Expected [ex, ey, ez] (default [10,10,10])
        is_moving: Expected is_moving flag (default False)
        is_urdf: Expected is_urdf_robot() result (default False)
        use_fixed_base: Expected use_fixed_base flag (default False)
        user_data: Expected user_data dict (_SKIP by default)
        pos_tolerance: Position comparison tolerance (default 0.01)
        orient_tolerance: Orientation comparison tolerance (default 0.01)
        vel_tolerance: Velocity/acceleration comparison tolerance (default 1e-6)
    """
    assert agent is not None
    assert agent.body_id >= 0

    # SimObject properties
    if mass is not _SKIP:
        assert agent.mass == mass, f"Expected mass={mass}, got {agent.mass}"
    if pickable is not _SKIP:
        assert agent.pickable is pickable, f"Expected pickable={pickable}, got {agent.pickable}"
    if name is not _SKIP:
        assert agent.name == name, f"Expected name={name!r}, got {agent.name!r}"
    if position is not _SKIP or orientation is not _SKIP:
        pose = agent.get_pose()
        if position is not _SKIP:
            for axis, (actual, expected) in enumerate(zip(pose.position, position)):
                assert abs(actual - expected) < pos_tolerance, f"Position axis {axis}: expected {expected}, got {actual}"
        if orientation is not _SKIP:
            for axis, (actual, expected) in enumerate(zip(pose.orientation, orientation)):
                assert abs(actual - expected) < orient_tolerance, f"Orientation axis {axis}: expected {expected}, got {actual}"
    if is_kinematic is not _SKIP:
        assert agent.is_kinematic is is_kinematic, f"Expected is_kinematic={is_kinematic}, got {agent.is_kinematic}"
    if collision_mode is not _SKIP:
        assert agent.collision_mode == collision_mode, f"Expected collision_mode={collision_mode}, got {agent.collision_mode}"

    # Agent-specific properties
    if motion_mode is not _SKIP:
        assert agent.motion_mode == motion_mode, f"Expected motion_mode={motion_mode}, got {agent.motion_mode}"
    if max_linear_vel is not _SKIP:
        expected = np.array(max_linear_vel) if not isinstance(max_linear_vel, np.ndarray) else max_linear_vel
        assert np.allclose(
            agent.max_linear_vel, expected, atol=vel_tolerance
        ), f"Expected max_linear_vel={expected.tolist()}, got {agent.max_linear_vel.tolist()}"
    if max_linear_accel is not _SKIP:
        expected = np.array(max_linear_accel) if not isinstance(max_linear_accel, np.ndarray) else max_linear_accel
        assert np.allclose(
            agent.max_linear_accel, expected, atol=vel_tolerance
        ), f"Expected max_linear_accel={expected.tolist()}, got {agent.max_linear_accel.tolist()}"
    if max_angular_vel is not _SKIP:
        expected = np.array(max_angular_vel) if not isinstance(max_angular_vel, np.ndarray) else max_angular_vel
        assert np.allclose(
            agent.max_angular_vel, expected, atol=vel_tolerance
        ), f"Expected max_angular_vel={expected.tolist()}, got {agent.max_angular_vel.tolist()}"
    if max_angular_accel is not _SKIP:
        expected = np.array(max_angular_accel) if not isinstance(max_angular_accel, np.ndarray) else max_angular_accel
        assert np.allclose(
            agent.max_angular_accel, expected, atol=vel_tolerance
        ), f"Expected max_angular_accel={expected.tolist()}, got {agent.max_angular_accel.tolist()}"
    if is_moving is not _SKIP:
        assert agent.is_moving is is_moving, f"Expected is_moving={is_moving}, got {agent.is_moving}"
    if is_urdf is not _SKIP:
        assert agent.is_urdf_robot() is is_urdf, f"Expected is_urdf_robot()={is_urdf}, got {agent.is_urdf_robot()}"
    if use_fixed_base is not _SKIP:
        assert agent.use_fixed_base is use_fixed_base, f"Expected use_fixed_base={use_fixed_base}, got {agent.use_fixed_base}"
    if user_data is not _SKIP:
        assert agent.user_data == user_data, f"Expected user_data={user_data}, got {agent.user_data}"


def create_mesh_agent(
    pose=None,
    mass=0.0,
    max_linear_vel=2.0,
    max_linear_accel=5.0,
    max_angular_vel=3.0,
    max_angular_accel=10.0,
    motion_mode=MotionMode.DIFFERENTIAL,
    use_fixed_base=False,
    collision_mode=CollisionMode.NORMAL_3D,
    name=None,
    user_data=None,
):
    """
    Create a mesh-based Agent with sensible defaults for testing.

    Reduces boilerplate for the common case of cube.obj visual + box collision.
    """
    from pybullet_fleet.controller_params import ControllerParams

    return Agent.from_mesh(
        visual_shape=ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        pose=pose,
        mass=mass,
        controller=ControllerParams(
            max_linear_vel=max_linear_vel,
            max_linear_accel=max_linear_accel,
            max_angular_vel=max_angular_vel,
            max_angular_accel=max_angular_accel,
        ),
        motion_mode=motion_mode,
        use_fixed_base=use_fixed_base,
        collision_mode=collision_mode,
        name=name,
        user_data=user_data,
    )


class TestAgentCreationFromMesh:
    """Test Agent creation from mesh files"""

    def test_from_mesh_basic(self, pybullet_env):
        """Test basic agent creation with all default properties"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), mass=1.0)

        # mass=1.0 and user_data={} differ from assert_agent_properties defaults
        assert_agent_properties(
            agent,
            mass=1.0,
            position=(0, 0, 0.5),
            is_kinematic=False,
            user_data={},
        )

    def test_from_mesh_kinematic(self, pybullet_env):
        """Test kinematic agent (mass=0)"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5))

        assert_agent_properties(
            agent,
            mass=0.0,
            is_kinematic=True,
            position=(0, 0, 0.5),
        )

    def test_from_mesh_with_custom_params(self, pybullet_env):
        """Test agent creation with custom velocity/acceleration/mode"""
        agent = create_mesh_agent(
            pose=Pose.from_xyz(1, 2, 0.5),
            mass=2.5,
            max_linear_vel=0.8,
            max_linear_accel=1.5,
            motion_mode=MotionMode.DIFFERENTIAL,
        )

        assert_agent_properties(
            agent,
            mass=2.5,
            position=(1, 2, 0.5),
            is_kinematic=False,
            motion_mode=MotionMode.DIFFERENTIAL,
            max_linear_vel=[0.8, 0.8, 0.8],
            max_linear_accel=[1.5, 1.5, 1.5],
        )

    def test_from_mesh_with_name(self, pybullet_env):
        """Test agent with name"""
        agent = create_mesh_agent(
            pose=Pose.from_xyz(0, 0, 0.5),
            name="TestRobot",
        )

        assert_agent_properties(agent, name="TestRobot", position=(0, 0, 0.5))

    def test_from_mesh_with_user_data(self, pybullet_env):
        """Test agent with user_data"""
        agent = create_mesh_agent(
            pose=Pose.from_xyz(0, 0, 0.5),
            user_data={"role": "carrier", "priority": 1},
        )

        assert_agent_properties(
            agent,
            position=(0, 0, 0.5),
            user_data={"role": "carrier", "priority": 1},
        )

    def test_from_mesh_default_pose(self, pybullet_env):
        """Test that pose=None defaults to origin"""
        agent = create_mesh_agent()

        assert_agent_properties(agent, position=(0, 0, 0))

    def test_from_mesh_different_colors(self, pybullet_env):
        """Test creating agents with different colors have different body_ids"""
        agent1 = Agent.from_mesh(
            visual_shape=ShapeParams(
                shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2], rgba_color=[1, 0, 0, 1]
            ),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(0, 0, 0.5),
        )
        agent2 = Agent.from_mesh(
            visual_shape=ShapeParams(
                shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2], rgba_color=[0, 1, 0, 1]
            ),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(1, 0, 0.5),
        )

        assert agent1.body_id != agent2.body_id

    def test_from_mesh_collision_mode_passed_through_init(self, pybullet_env):
        """Agent.from_mesh should pass collision_mode through __init__ to SimObject.__init__.

        Previously, Agent.__init__ did NOT forward collision_mode to super().__init__(),
        so add_object always received NORMAL_3D, then from_mesh called set_collision_mode
        to fix it – a wasteful 2-step transition.  After the fix, add_object should
        receive the correct collision_mode directly and _update_object_collision_mode
        should NOT be called during from_mesh.
        """

        class TrackingSimCore:
            """Mock that records collision_mode at add_object time and
            tracks _update_object_collision_mode calls."""

            def __init__(self):
                self._next_object_id = 0
                self.sim_time = 0.0
                self._kinematic_objects = set()
                self._client = 0
                self.collision_mode_at_add: Optional[CollisionMode] = None
                self.update_collision_mode_calls = []
                # Agent.__init__ emits AGENT_SPAWNED at the end of construction.
                self.events = EventBus()

            @property
            def client(self):
                return self._client

            def add_object(self, obj):
                self.collision_mode_at_add = obj.collision_mode

            def _update_object_collision_mode(self, object_id, old_mode, new_mode):
                self.update_collision_mode_calls.append((object_id, old_mode, new_mode))

            def _mark_object_moved(self, object_id):
                pass

        tracker = TrackingSimCore()
        agent = Agent.from_mesh(
            visual_shape=ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_mode=CollisionMode.STATIC,
            sim_core=tracker,
        )

        # add_object should have been called with collision_mode already set to STATIC
        assert (
            tracker.collision_mode_at_add == CollisionMode.STATIC
        ), f"add_object received collision_mode={tracker.collision_mode_at_add}, expected STATIC"
        # set_collision_mode should NOT have been called (no transition needed)
        assert (
            len(tracker.update_collision_mode_calls) == 0
        ), f"_update_object_collision_mode was called {len(tracker.update_collision_mode_calls)} times, expected 0"
        # Final collision_mode should be STATIC
        assert agent.collision_mode == CollisionMode.STATIC

    def test_from_urdf_collision_mode_passed_through_init(self, pybullet_env):
        """Same as mesh test but for URDF path."""

        class TrackingSimCore:
            def __init__(self):
                self._next_object_id = 0
                self.sim_time = 0.0
                self._kinematic_objects = set()
                self._client = 0
                self.collision_mode_at_add: Optional[CollisionMode] = None
                self.update_collision_mode_calls = []
                # Agent.__init__ emits AGENT_SPAWNED at the end of construction.
                self.events = EventBus()

            @property
            def client(self):
                return self._client

            def add_object(self, obj):
                self.collision_mode_at_add = obj.collision_mode

            def _update_object_collision_mode(self, object_id, old_mode, new_mode):
                self.update_collision_mode_calls.append((object_id, old_mode, new_mode))

            def _mark_object_moved(self, object_id):
                pass

        tracker = TrackingSimCore()
        agent = Agent.from_urdf(
            urdf_path="robots/mobile_robot.urdf",
            collision_mode=CollisionMode.STATIC,
            sim_core=tracker,
        )

        assert (
            tracker.collision_mode_at_add == CollisionMode.STATIC
        ), f"add_object received collision_mode={tracker.collision_mode_at_add}, expected STATIC"
        assert (
            len(tracker.update_collision_mode_calls) == 0
        ), f"_update_object_collision_mode was called {len(tracker.update_collision_mode_calls)} times, expected 0"
        assert agent.collision_mode == CollisionMode.STATIC


class TestAgentCreationFromURDF:
    """Test Agent creation from URDF files"""

    def test_from_urdf_basic(self, pybullet_env):
        """Test creating agent from URDF with default mass=None (uses URDF values)"""
        agent = Agent.from_urdf(urdf_path="robots/mobile_robot.urdf", pose=Pose.from_xyz(0, 0, 0.5))

        assert_agent_properties(
            agent,
            mass=_SKIP,  # Mass comes from URDF
            is_kinematic=_SKIP,  # Depends on URDF mass
            position=(0, 0, 0.5),
            is_urdf=True,
        )
        assert agent.mass >= 0.5  # URDF should have some mass

    def test_from_urdf_mass_none_uses_urdf_values(self, pybullet_env):
        """mass=None (default) should use URDF mass values, not override them."""
        agent = Agent.from_urdf(
            urdf_path="robots/mobile_robot.urdf",
            pose=Pose.from_xyz(0, 0, 0.5),
            mass=None,  # explicit None — same as default
        )
        # URDF has non-zero mass → agent should not be kinematic
        assert agent.mass > 0, "mass=None should preserve URDF mass values"
        assert not agent.is_kinematic, "mass=None agent should not be kinematic"

    def test_from_urdf_with_custom_params(self, pybullet_env):
        """Test URDF agent with custom velocity/mode parameters (mass=None default)"""
        from pybullet_fleet.controller_params import ControllerParams

        agent = Agent.from_urdf(
            urdf_path="robots/mobile_robot.urdf",
            pose=Pose.from_xyz(1, 1, 0.5),
            controller=ControllerParams(
                max_linear_vel=1.5,
                max_linear_accel=3.0,
            ),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )

        assert_agent_properties(
            agent,
            mass=_SKIP,  # URDF mass, not overridden
            is_kinematic=_SKIP,  # Depends on URDF mass
            position=(1, 1, 0.5),
            is_urdf=True,
            max_linear_vel=[1.5, 1.5, 1.5],
            max_linear_accel=[3.0, 3.0, 3.0],
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )

    def test_from_urdf_kinematic(self, pybullet_env):
        """Test URDF agent with mass=0 (kinematic control)"""
        agent = Agent.from_urdf(
            urdf_path="robots/mobile_robot.urdf",
            pose=Pose.from_xyz(0, 0, 0.5),
            mass=0.0,
        )

        assert_agent_properties(
            agent,
            mass=0.0,
            is_kinematic=True,
            is_urdf=True,
            position=(0, 0, 0.5),
        )

    def test_from_urdf_resolves_model_name(self, pybullet_env):
        """from_urdf should accept a model name (e.g. 'arm_robot') and resolve it."""
        agent = Agent.from_urdf(urdf_path="arm_robot", pose=Pose.from_xyz(0, 0, 0))
        assert agent.body_id >= 0
        assert agent.get_num_joints() > 0

    def test_from_urdf_resolves_pybullet_data_name(self, pybullet_env):
        """from_urdf should resolve tier-1 pybullet_data names like 'panda'."""
        agent = Agent.from_urdf(urdf_path="panda", pose=Pose.from_xyz(0, 0, 0), use_fixed_base=True)
        assert agent.body_id >= 0
        assert agent.get_num_joints() >= 7  # Panda has 7 DOF + fingers


class TestAgentSpawnParams:
    """Test AgentSpawnParams dataclass"""

    def test_spawn_params_defaults(self):
        """Test spawn params with default values"""
        params = AgentSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
        )

        assert params.mass == 0.0
        # Narrows SimObjectSpawnParams' True: an agent is usually the one
        # doing the picking. This used to read True while Agent.__init__
        # overwrote the flag with False regardless, so the field was inert;
        # now the params default states what the agent actually gets.
        assert params.pickable is False
        assert params.name is None
        # No controller dict provided → stays None; framework defaults are
        # applied later at Agent construction time.
        assert params.controller is None
        assert params.motion_mode == MotionMode.DIFFERENTIAL
        assert params.use_fixed_base is False
        assert params.urdf_path is None
        assert params.collision_mode == CollisionMode.NORMAL_3D
        assert params.user_data == {}

    def test_spawn_params_custom(self):
        """Test spawn params with custom values"""
        params = AgentSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
            mass=1.5,
            controller={"max_linear_vel": 2.0},
            name="Robot_A",
            motion_mode=MotionMode.DIFFERENTIAL,
            user_data={"team": "blue"},
        )

        assert params.mass == 1.5
        assert params.controller == {"max_linear_vel": 2.0}
        assert params.name == "Robot_A"
        assert params.motion_mode == MotionMode.DIFFERENTIAL
        assert params.user_data == {"team": "blue"}

    def test_from_params_mesh(self, pybullet_env):
        """Test creating agent from mesh spawn parameters"""
        params = AgentSpawnParams(
            visual_shape=ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            initial_pose=Pose.from_xyz(0, 0, 0.5),
            mass=2.0,
            controller={"max_linear_vel": 1.0},
            motion_mode=MotionMode.DIFFERENTIAL,
            name="ParamsAgent",
            user_data={"source": "params"},
        )

        agent = Agent.from_params(params)

        assert_agent_properties(
            agent,
            mass=2.0,
            position=(0, 0, 0.5),
            is_kinematic=False,
            motion_mode=MotionMode.DIFFERENTIAL,
            max_linear_vel=[1.0, 1.0, 1.0],
            is_urdf=False,
            name="ParamsAgent",
            user_data={"source": "params"},
        )

    def test_from_params_urdf(self, pybullet_env):
        """Test creating agent from URDF spawn parameters"""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(1, 1, 0.5),
            controller={"max_linear_vel": 1.5},
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )

        agent = Agent.from_params(params)

        assert_agent_properties(
            agent,
            mass=_SKIP,  # URDF mass
            is_kinematic=_SKIP,  # Depends on URDF mass
            position=(1, 1, 0.5),
            is_urdf=True,
            max_linear_vel=[1.5, 1.5, 1.5],
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )


class TestAgentSpawnParamsFromDict:
    """AgentSpawnParams.from_dict() creates params from YAML-style dicts."""

    def test_minimal_urdf(self):
        """Minimal dict with name + urdf_path produces correct params."""
        result = AgentSpawnParams.from_dict({"name": "r0", "urdf_path": "robots/mobile_robot.urdf"})
        assert result.name == "r0"
        assert result.urdf_path == "robots/mobile_robot.urdf"
        # No kinematic limits provided in dict → controller stays None and
        # framework defaults are applied at Agent construction time.
        assert result.controller is None

    def test_pose_and_yaw(self):
        """pose list + yaw are converted to Pose."""
        result = AgentSpawnParams.from_dict(
            {"name": "r0", "urdf_path": "robots/mobile_robot.urdf", "pose": [1.0, 2.0, 0.05], "yaw": 1.57}
        )
        assert result.initial_pose.x == pytest.approx(1.0)
        assert result.initial_pose.y == pytest.approx(2.0)
        assert result.initial_pose.z == pytest.approx(0.05)
        assert result.initial_pose.yaw == pytest.approx(1.57, abs=0.01)

    def test_controller_string_shortcut(self):
        """controller string is preserved as-is; wired later by Agent._init_controller_chain."""
        result = AgentSpawnParams.from_dict(
            {"name": "r0", "urdf_path": "robots/mobile_robot.urdf", "controller": "differential"}
        )
        assert result.controller == "differential"

    def test_controller_dict(self):
        """controller dict is passed through."""
        result = AgentSpawnParams.from_dict(
            {"name": "r0", "urdf_path": "robots/mobile_robot.urdf", "controller": {"type": "omni_velocity"}}
        )
        assert result.controller == {"type": "omni_velocity"}

    def test_missing_name_raises(self):
        """Missing name raises ValueError (inherited from SimObjectSpawnParams)."""
        with pytest.raises(ValueError, match="missing required 'name'"):
            AgentSpawnParams.from_dict({"urdf_path": "robots/mobile_robot.urdf"})


class TestAgentFromDict:
    """Agent.from_dict creates instance directly from config dict."""

    def test_subclass_returns_subclass(self, pybullet_env):
        """Subclass.from_dict returns an instance of the subclass."""

        class ForkliftAgent(Agent):
            pass

        config = {
            "name": "fl0",
            "urdf_path": "robots/mobile_robot.urdf",
        }
        agent = ForkliftAgent.from_dict(config)
        assert isinstance(agent, ForkliftAgent)


class TestAgentGoalSetting:
    """Test agent goal and path setting"""

    def test_set_goal_pose(self, pybullet_env):
        """Test setting goal pose"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5))

        # Initially no goal and not moving
        assert_agent_properties(agent, is_moving=False, position=(0, 0, 0.5))
        assert agent.goal_pose is None

        # Set goal
        agent.set_goal_pose(Pose.from_xyz(5, 0, 0.5))

        # Agent should be moving with a goal set
        assert agent.is_moving or agent.goal_pose is not None

    def test_set_path(self, pybullet_env):
        """Test setting path with waypoints"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5))

        path = Path.from_positions([[1, 0, 0.5], [2, 0, 0.5], [2, 1, 0.5]])

        # Set path
        agent.set_path(path)

        # Agent should be moving after set_path
        assert agent.is_moving


class TestAgentMotionModes:
    """Test different motion modes"""

    def test_restore_agent_motion_fields_does_not_require_omni_controller(self, pybullet_env):
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), motion_mode=MotionMode.DIFFERENTIAL)
        restored_pose = Pose.from_xyz(1, 2, 0.5)

        agent.restore_motion_state(restored_pose, [0.2, 0.0, 0.0], 0.1, moving=True)

        assert agent.get_pose().position == pytest.approx(restored_pose.position)
        assert agent.get_velocity() == pytest.approx([0.2, 0.0, 0.0])
        assert agent.angular_velocity == pytest.approx(0.1)
        assert agent.is_moving is True

    def test_omnidirectional_mode(self, pybullet_env):
        """Test omnidirectional motion mode"""
        agent = create_mesh_agent(
            pose=Pose.from_xyz(0, 0, 0.5),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )

        assert_agent_properties(agent, motion_mode=MotionMode.OMNIDIRECTIONAL, position=(0, 0, 0.5))

    def test_differential_mode(self, pybullet_env):
        """Test differential drive motion mode"""
        agent = create_mesh_agent(
            pose=Pose.from_xyz(0, 0, 0.5),
            motion_mode=MotionMode.DIFFERENTIAL,
        )

        assert_agent_properties(agent, motion_mode=MotionMode.DIFFERENTIAL, position=(0, 0, 0.5))

    def test_motion_mode_string(self, pybullet_env):
        """Test motion mode accepts string values"""
        agent = create_mesh_agent(
            pose=Pose.from_xyz(0, 0, 0.5),
            motion_mode="differential",
        )

        assert_agent_properties(agent, motion_mode=MotionMode.DIFFERENTIAL, position=(0, 0, 0.5))

    def test_set_motion_mode(self, pybullet_env):
        """Test changing motion mode"""
        agent = create_mesh_agent(
            pose=Pose.from_xyz(0, 0, 0.5),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )

        assert_agent_properties(agent, motion_mode=MotionMode.OMNIDIRECTIONAL, position=(0, 0, 0.5))

        # Change to differential
        success = agent.set_motion_mode("differential")
        assert success is True
        assert_agent_properties(agent, motion_mode=MotionMode.DIFFERENTIAL, position=(0, 0, 0.5))


class TestAgentVelocityConstraints:
    """Test velocity and acceleration constraints"""

    def test_scalar_velocity_constraint(self, pybullet_env):
        """Test scalar velocity constraint applied to all axes"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), max_linear_vel=1.5)

        assert_agent_properties(agent, max_linear_vel=[1.5, 1.5, 1.5], position=(0, 0, 0.5))

    def test_vector_velocity_constraint(self, pybullet_env):
        """Test per-axis velocity constraints"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), max_linear_vel=[1.0, 2.0, 0.5])

        assert_agent_properties(agent, max_linear_vel=[1.0, 2.0, 0.5], position=(0, 0, 0.5))

    def test_scalar_acceleration_constraint(self, pybullet_env):
        """Test scalar acceleration constraint"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), max_linear_accel=3.0)

        assert_agent_properties(agent, max_linear_accel=[3.0, 3.0, 3.0], position=(0, 0, 0.5))

    def test_vector_acceleration_constraint(self, pybullet_env):
        """Test per-axis acceleration constraints"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), max_linear_accel=[2.0, 4.0, 1.0])

        assert_agent_properties(agent, max_linear_accel=[2.0, 4.0, 1.0], position=(0, 0, 0.5))

    def test_scalar_angular_velocity(self, pybullet_env):
        """Test scalar angular velocity constraint"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), max_angular_vel=5.0)

        assert_agent_properties(agent, max_angular_vel=[5.0, 5.0, 5.0], position=(0, 0, 0.5))

    def test_scalar_angular_acceleration(self, pybullet_env):
        """Test scalar angular acceleration constraint"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), max_angular_accel=8.0)

        assert_agent_properties(agent, max_angular_accel=[8.0, 8.0, 8.0], position=(0, 0, 0.5))


class TestAgentFixedBase:
    """Test fixed base agents"""

    def test_fixed_base_agent(self, pybullet_env):
        """Test creating fixed base agent"""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5), use_fixed_base=True)

        assert_agent_properties(agent, use_fixed_base=True, position=(0, 0, 0.5))

        # Position should not change even after simulation steps
        initial_pos, _ = p.getBasePositionAndOrientation(agent.body_id)

        for _ in range(100):
            p.stepSimulation()

        final_pos, _ = p.getBasePositionAndOrientation(agent.body_id)

        # Position should remain the same (fixed base)
        assert np.allclose(initial_pos, final_pos, atol=0.001)


# ============================================================================
# Motion update / goal reaching tests
# ============================================================================


class TestAgentMotionUpdate:
    """Test agent.update() moves toward goals and eventually reaches them."""

    def _create_agent_with_simcore(
        self,
        pose,
        *,
        max_linear_vel=2.0,
        max_linear_accel=5.0,
        max_angular_vel=3.0,
        max_angular_accel=10.0,
        motion_mode=MotionMode.OMNIDIRECTIONAL,
    ):
        """Helper: create mesh agent wired to a MockSimCore."""
        from pybullet_fleet.controller_params import ControllerParams

        sim_core = MockSimCore()
        agent = Agent.from_mesh(
            visual_shape=ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=pose,
            mass=0.0,
            controller=ControllerParams(
                max_linear_vel=max_linear_vel,
                max_linear_accel=max_linear_accel,
                max_angular_vel=max_angular_vel,
                max_angular_accel=max_angular_accel,
            ),
            motion_mode=motion_mode,
            sim_core=sim_core,
        )
        return agent, sim_core

    # -- Initial state ---------------------------------------------------

    def test_initial_velocity_zero(self, pybullet_env):
        """Agent should start with zero linear velocity."""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5))
        assert np.allclose(agent.velocity, [0.0, 0.0, 0.0])

    def test_initial_angular_velocity_zero(self, pybullet_env):
        """Agent should start with zero angular velocity."""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5))
        assert agent.angular_velocity == 0.0

    # -- Omnidirectional -------------------------------------------------

    def _run_omni_motion_loop(
        self,
        agent,
        sim_core,
        goal_pos,
        expected_quat,
        *,
        final_quat=None,
        max_steps=5000,
        warmup_steps=5,
    ):
        """Run an omnidirectional update loop checking per-step invariants.

        **Translation phase** (while ``_is_final_orientation_aligning`` is False):
          - distance to goal never increases
          - orientation stays at *expected_quat*
          - velocity is non-zero after warmup while far from goal

        **Final-rotation phase** (optional, when *final_quat* is given and the
        agent enters ``_is_final_orientation_aligning``):
          - position stays fixed at goal
          - angle error to *final_quat* decreases monotonically (after warmup)

        Returns:
            dict with keys:
                reached (bool): whether the agent stopped within max_steps
        """
        prev_dist = np.linalg.norm(np.array(agent.get_pose().position) - goal_pos)
        dt = sim_core._dt
        reached = False

        # Final-rotation phase tracking
        in_final_rotation = False
        final_rot_pos = None
        prev_angle_error = float("inf")
        final_rot_steps = 0

        for step in range(max_steps):
            sim_core.tick()
            agent.update(dt)
            pose = agent.get_pose()
            cur_pos = np.array(pose.position)
            cur_dist = np.linalg.norm(cur_pos - goal_pos)

            # Detect transition into final-rotation phase
            if final_quat is not None and agent._controller._is_final_orientation_aligning and not in_final_rotation:
                in_final_rotation = True
                final_rot_pos = cur_pos.copy()
                prev_angle_error = self._quat_angle_error(pose.orientation, final_quat)
                final_rot_steps = 0

            if in_final_rotation:
                final_rot_steps += 1
                # Position must stay fixed
                assert np.allclose(cur_pos, final_rot_pos, atol=1e-4), (
                    f"Step {step} FINAL_ROTATE: position moved " f"from {list(final_rot_pos)} to {list(cur_pos)}"
                )
                # Angle error must decrease monotonically (after warmup)
                cur_angle_error = self._quat_angle_error(pose.orientation, final_quat)
                if final_rot_steps >= warmup_steps:
                    assert cur_angle_error <= prev_angle_error + 1e-4, (
                        f"Step {step} FINAL_ROTATE: angle error increased "
                        f"{prev_angle_error:.6f} -> {cur_angle_error:.6f} rad"
                    )
                prev_angle_error = cur_angle_error
            else:
                # Translation phase checks
                assert cur_dist <= prev_dist + 1e-6, f"Step {step}: distance increased {prev_dist:.6f} -> {cur_dist:.6f}"
                assert np.allclose(pose.orientation, expected_quat, atol=1e-4), (
                    f"Step {step}: orientation changed " f"from {expected_quat} to {list(pose.orientation)}"
                )
                if warmup_steps <= step and agent.is_moving and cur_dist > 0.1:
                    assert np.linalg.norm(agent.velocity) > 0.01, (
                        f"Step {step}: velocity should be positive during motion, " f"got {agent.velocity}"
                    )

            prev_dist = cur_dist
            if not agent.is_moving:
                reached = True
                break

        return {"reached": reached}

    def test_omni_update_toward_goal(self, pybullet_env):
        """Omnidirectional agent moves monotonically toward goal and reaches it.

        Verifies every step:
        - distance to goal never increases
        - orientation stays at identity (omni does not rotate)
        - velocity is non-zero after warmup
        After reaching:
        - is_moving is False, velocity ≈ 0, position ≈ goal
        """
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(0, 0, 0))
        goal_pos = np.array([1, 0, 0])
        agent.set_goal_pose(Pose.from_xyz(*goal_pos))

        result = self._run_omni_motion_loop(agent, sim_core, goal_pos, [0, 0, 0, 1])

        assert result["reached"], "Agent should reach the goal within max_steps"
        assert not agent.is_moving
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos, goal_pos, atol=0.05), f"Final position {pos} not close to goal"
        assert np.allclose(agent.velocity, [0, 0, 0], atol=1e-4), f"Velocity should be ~0 after goal, got {agent.velocity}"

    def test_omni_second_goal_no_teleport(self, pybullet_env):
        """After completing a first goal, a second goal must produce gradual motion.

        Regression test: _is_final_orientation_aligning was not reset after the
        first goal completed with final-orientation alignment.  On the second
        goal, update() dispatched to _update_differential (rotation) instead of
        _update_omnidirectional (TPI), causing an immediate teleport to the goal.

        Verifies:
        - First goal is reached normally
        - After setting a second goal, distance does NOT jump to zero in one step
        - The agent eventually reaches the second goal with monotonic progress
        """
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(0, 0, 0))
        dt = sim_core._dt

        # ----- Goal 1: (0,0,0) → (1,0,0) with default final_orientation_align=True -----
        goal1 = np.array([1.0, 0.0, 0.0])
        agent.set_goal_pose(Pose.from_xyz(*goal1))
        for _ in range(5000):
            sim_core.tick()
            agent.update(dt)
            if not agent.is_moving:
                break
        assert not agent.is_moving, "Goal 1 should be reached"
        # After goal 1, _is_final_orientation_aligning must be cleared
        assert (
            not agent._controller._is_final_orientation_aligning
        ), "_is_final_orientation_aligning should be reset after goal completion"

        # ----- Goal 2: (1,0,0) → (0,1,0) -----
        pos_before = np.array(agent.get_pose().position)
        goal2 = np.array([0.0, 1.0, 0.0])
        agent.set_goal_pose(Pose.from_xyz(*goal2))

        # After a few steps, robot should have moved GRADUALLY, not teleported
        for _ in range(10):
            sim_core.tick()
            agent.update(dt)
        pos_after_10 = np.array(agent.get_pose().position)
        dist_moved = np.linalg.norm(pos_after_10 - pos_before)
        dist_to_goal = np.linalg.norm(pos_after_10[:2] - goal2[:2])

        # In 10 steps at dt=1/240, max distance ≈ 0.5*5.0*(10/240)^2 ≈ 0.004m (accel phase)
        # It should NOT have teleported the full ~1.4m
        assert dist_to_goal > 0.5, (
            f"Robot teleported to goal! pos={list(pos_after_10)}, dist_to_goal={dist_to_goal:.4f}. "
            f"Expected gradual motion, not instant arrival."
        )

        # Run until goal 2 reached
        for _ in range(5000):
            sim_core.tick()
            agent.update(dt)
            if not agent.is_moving:
                break
        assert not agent.is_moving, "Goal 2 should be reached"
        pos_final = np.array(agent.get_pose().position)
        assert np.allclose(pos_final[:2], goal2[:2], atol=0.05), f"Final position {pos_final} not close to goal2 {goal2}"

    def test_omni_straight_line_motion(self, pybullet_env):
        """Omnidirectional agent must move in a straight line toward the goal.

        When moving to a diagonal goal (e.g. (3, 4, 0)), the robot should
        follow a straight-line path, NOT move along each axis independently.
        Independent per-axis TPI causes the shorter axis to arrive first,
        producing an L-shaped or curved trajectory instead of a straight line.

        Verifies:
        - Every intermediate position lies on (or very near) the line from
          start to goal (cross-track error < threshold)
        - The agent actually reaches the goal
        """
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(0, 0, 0))
        dt = sim_core._dt

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([3.0, 4.0, 0.0])
        direction = goal - start
        total_dist = np.linalg.norm(direction)
        unit_dir = direction / total_dist

        agent.set_path([Pose.from_xyz(*goal)], auto_approach=False, final_orientation_align=False)

        max_cross_track = 0.0
        reached = False
        warmup_steps = 10  # allow a few steps for TPI to kick in

        for step in range(5000):
            sim_core.tick()
            agent.update(dt)

            if not agent.is_moving:
                reached = True
                break

            if step < warmup_steps:
                continue

            pos = np.array(agent.get_pose().position)
            # Cross-track error: perpendicular distance from pos to line (start → goal)
            to_pos = pos - start
            projection = np.dot(to_pos, unit_dir) * unit_dir
            cross_track = np.linalg.norm(to_pos - projection)

            if cross_track > max_cross_track:
                max_cross_track = cross_track

            # Allow 5cm tolerance for numerical drift
            assert cross_track < 0.05, (
                f"Step {step}: cross-track error {cross_track:.4f}m exceeds 0.05m. "
                f"pos={list(pos)}, expected on line from {list(start)} to {list(goal)}. "
                f"Robot is NOT moving in a straight line."
            )

        assert reached, "Agent should reach the diagonal goal"
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos[:2], goal[:2], atol=0.05), f"Final pos {pos} not close to goal {goal}"

    def test_omni_update_toward_goal_3d(self, pybullet_env):
        """Omnidirectional agent with non-identity initial orientation reaches 3D goal.

        Same invariants as test_omni_update_toward_goal, but:
        - start position and goal differ in all three axes
        - initial orientation is non-zero (rotated around Z)
        - orientation must stay at that initial value throughout (omni does not rotate)
        - final_orientation_align disabled so the agent does not rotate after reaching
        """
        initial_pose = Pose.from_euler(2, 1, 0, yaw=0.5)  # rotated ~28.6° around Z
        agent, sim_core = self._create_agent_with_simcore(initial_pose)
        goal_pos = np.array([1, 2, 0.5])
        agent.set_path([Pose.from_xyz(*goal_pos)], final_orientation_align=False)

        initial_quat = list(agent.get_pose().orientation)
        result = self._run_omni_motion_loop(agent, sim_core, goal_pos, initial_quat)

        assert result["reached"], "Agent should reach the 3D goal within max_steps"
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos, goal_pos, atol=0.05), f"Final position {pos} not close to goal"
        assert np.allclose(agent.velocity, [0, 0, 0], atol=1e-4), f"Velocity should be ~0 after goal, got {agent.velocity}"

    def test_omni_path_multi_waypoint(self, pybullet_env):
        """Agent follows an L-shaped multi-waypoint path and reaches the final position.

        Verifies every step:
        - waypoint index never decreases (monotonic progress through path)
        - within each waypoint segment, distance to current goal never increases
        - orientation stays at identity (omni does not rotate)
        - velocity is non-zero while moving and far from current waypoint
        After reaching:
        - is_moving is False, velocity ≈ 0, position ≈ final waypoint
        """
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(0, 0, 0))
        waypoints = [Pose.from_xyz(1, 0, 0), Pose.from_xyz(1, 1, 0), Pose.from_xyz(0, 1, 0)]
        agent.set_path(waypoints, auto_approach=False, final_orientation_align=False)

        identity_quat = [0, 0, 0, 1]
        dt = sim_core._dt
        warmup_steps = 5
        prev_wp_idx = agent.current_waypoint_index
        goal_pos = np.array(agent.goal_pose.position)
        prev_dist = np.linalg.norm(np.array(agent.get_pose().position) - goal_pos)
        reached = False

        for step in range(10000):
            sim_core.tick()
            agent.update(dt)
            if not agent.is_moving:
                reached = True
                break

            pose = agent.get_pose()
            cur_wp_idx = agent.current_waypoint_index

            # Waypoint index must never decrease
            assert cur_wp_idx >= prev_wp_idx, f"Step {step}: waypoint index went backwards {prev_wp_idx} -> {cur_wp_idx}"

            # On waypoint switch, reset distance tracking to new goal
            if cur_wp_idx != prev_wp_idx:
                goal_pos = np.array(agent.goal_pose.position)
                prev_dist = np.linalg.norm(np.array(pose.position) - goal_pos)
                prev_wp_idx = cur_wp_idx
                continue

            # Within segment: distance to current goal must not increase
            cur_dist = np.linalg.norm(np.array(pose.position) - goal_pos)
            assert (
                cur_dist <= prev_dist + 1e-6
            ), f"Step {step} (wp={cur_wp_idx}): distance increased {prev_dist:.6f} -> {cur_dist:.6f}"

            # Orientation must stay at identity
            assert np.allclose(
                pose.orientation, identity_quat, atol=1e-4
            ), f"Step {step}: orientation changed to {pose.orientation}"

            # Velocity should be non-zero while moving and far from waypoint
            if step >= warmup_steps and cur_dist > 0.1:
                assert np.linalg.norm(agent.velocity) > 0.01, f"Step {step}: velocity should be positive, got {agent.velocity}"

            prev_dist = cur_dist

        assert reached, "Agent should complete the multi-waypoint path"
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos, [0, 1, 0], atol=0.05), f"Final pos {pos} not close to last waypoint"
        assert np.allclose(agent.velocity, [0, 0, 0], atol=1e-4), f"Velocity should be ~0 after path, got {agent.velocity}"

    def test_omni_stop_clears_goal(self, pybullet_env):
        """Calling stop() should clear the goal and stop motion."""
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(0, 0, 0))
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0))
        assert agent.is_moving

        agent.stop()
        assert not agent.is_moving
        assert agent.goal_pose is None
        assert np.allclose(agent.velocity, [0, 0, 0])

    def test_omni_stop_halts_motion_after_updates(self, pybullet_env):
        """Calling stop() should prevent further pose changes on later updates."""
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(0, 0, 0))
        dt = 0.1
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0))

        for _ in range(30):
            sim_core.tick()
            agent.update(dt)

        moving_pos = np.array(agent.get_pose().position)
        goal_pos = np.array([10.0, 0.0, 0.0])
        assert moving_pos[0] > 0.01
        assert np.linalg.norm(goal_pos - moving_pos) > 1.0
        assert np.linalg.norm(agent.velocity) > 0.0

        agent.stop()
        stopped_pos = np.array(agent.get_pose().position)
        for _ in range(20):
            sim_core.tick()
            agent.update(dt)

        final_pos = np.array(agent.get_pose().position)
        assert np.allclose(final_pos, stopped_pos)
        assert not agent.is_moving
        assert agent.goal_pose is None
        assert np.allclose(agent.velocity, [0, 0, 0])

    def test_omni_final_orientation_align(self, pybullet_env):
        """Omnidirectional agent should align to goal orientation after reaching position.

        Goal is at (1,0,0) with yaw=1.0 rad.  The agent should:
        1. Move to goal position (omni, orientation stays at identity)
        2. Rotate in place to match goal yaw (via _is_final_orientation_aligning flag)
        """
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(0, 0, 0))
        goal = Pose.from_euler(1, 0, 0, yaw=1.0)
        goal_pos = np.array(goal.position)
        goal_quat = np.array(goal.orientation)
        initial_quat = list(agent.get_pose().orientation)
        agent.set_goal_pose(goal)

        assert agent._controller._align_final_orientation is True

        result = self._run_omni_motion_loop(agent, sim_core, goal_pos, initial_quat, final_quat=goal_quat)

        assert result["reached"], "Omni agent should reach goal"
        final_orient = np.array(agent.get_pose().orientation)
        angle_err = self._quat_angle_error(final_orient, goal_quat)
        assert angle_err < 0.05, (
            f"Final orientation error {angle_err:.4f} rad; " f"final={list(final_orient)}, goal={list(goal_quat)}"
        )

    # -- Differential ----------------------------------------------------

    def _run_pose_phase_loop(
        self,
        agent,
        sim_core,
        goal_pos,
        *,
        max_steps=10000,
        warmup_steps=5,
    ):
        """Run a differential-drive update loop tracking ROTATE/FORWARD phase invariants.

        Handles multiple ROTATE→FORWARD cycles (e.g. initial rotation, forward
        motion, then final orientation alignment). On each cycle:

        ROTATE:
          - position stays fixed
          - angle error to target quaternion decreases monotonically (after warmup)
          - angular_velocity is non-zero after warmup

        FORWARD:
          - distance to goal decreases monotonically
          - orientation stays fixed
          - linear velocity is non-zero while far from goal

        Returns:
            dict with keys:
                reached (bool): whether the agent stopped within max_steps
                rotate_cycles (int): number of ROTATE phases entered
                final_rotate_at_goal (bool): whether a ROTATE phase started near goal_pos
        """
        dt = sim_core._dt

        prev_phase = None
        target_quat = None
        prev_angle_error = float("inf")
        rotate_start_pos = None
        rotate_steps_in_cycle = 0

        forward_orientation = None
        prev_forward_dist = float("inf")
        forward_steps_in_cycle = 0

        rotate_cycles = 0
        final_rotate_at_goal = False
        reached = False

        for step in range(max_steps):
            sim_core.tick()
            agent.update(dt)
            if not agent.is_moving:
                reached = True
                break

            phase = agent._controller._pose_phase
            pose = agent.get_pose()
            cur_pos = np.array(pose.position)

            # Detect phase transition → reset per-cycle state
            if phase != prev_phase:
                if phase == PosePhase.ROTATE:
                    rotate_cycles += 1
                    ctrl_tgt = agent._controller._rotation_target_quat
                    target_quat = ctrl_tgt.copy()
                    prev_angle_error = self._quat_angle_error(pose.orientation, target_quat)
                    rotate_start_pos = cur_pos.copy()
                    rotate_steps_in_cycle = 0
                    if np.allclose(cur_pos, goal_pos, atol=0.1):
                        final_rotate_at_goal = True
                elif phase == PosePhase.FORWARD:
                    forward_orientation = list(pose.orientation)
                    prev_forward_dist = np.linalg.norm(cur_pos - goal_pos)
                    forward_steps_in_cycle = 0
                prev_phase = phase
                continue  # skip checks on the transition step

            if phase == PosePhase.ROTATE:
                rotate_steps_in_cycle += 1

                # Position must stay fixed during rotation
                assert np.allclose(
                    cur_pos, rotate_start_pos, atol=1e-4
                ), f"Step {step} ROTATE: position moved from {rotate_start_pos} to {cur_pos}"

                # Angle error must decrease monotonically (after warmup)
                cur_angle_error = self._quat_angle_error(pose.orientation, target_quat)
                if rotate_steps_in_cycle >= warmup_steps:
                    assert cur_angle_error <= prev_angle_error + 1e-4, (
                        f"Step {step} ROTATE: angle error increased " f"{prev_angle_error:.6f} -> {cur_angle_error:.6f} rad"
                    )
                prev_angle_error = cur_angle_error

                # Angular velocity should be non-zero after warmup
                if rotate_steps_in_cycle >= warmup_steps:
                    assert abs(agent.angular_velocity) > 0.01, (
                        f"Step {step} ROTATE: angular_velocity should be non-zero, " f"got {agent.angular_velocity}"
                    )

            elif phase == PosePhase.FORWARD:
                forward_steps_in_cycle += 1
                cur_dist = np.linalg.norm(cur_pos - goal_pos)

                # Distance to goal must decrease monotonically
                assert cur_dist <= prev_forward_dist + 1e-6, (
                    f"Step {step} FORWARD: distance increased " f"{prev_forward_dist:.6f} -> {cur_dist:.6f}"
                )

                # Orientation must stay fixed during FORWARD phase
                assert np.allclose(pose.orientation, forward_orientation, atol=1e-4), (
                    f"Step {step} FORWARD: orientation changed from " f"{forward_orientation} to {list(pose.orientation)}"
                )

                # Velocity should be non-zero while far from goal
                if forward_steps_in_cycle >= warmup_steps and cur_dist > 0.1:
                    assert np.linalg.norm(agent.velocity) > 0.01, (
                        f"Step {step} FORWARD: velocity should be positive, " f"got {agent.velocity}"
                    )

                prev_forward_dist = cur_dist

        return {
            "reached": reached,
            "rotate_cycles": rotate_cycles,
            "final_rotate_at_goal": final_rotate_at_goal,
        }

    def test_diff_update_toward_goal_forward(self, pybullet_env):
        """Differential agent reaching a goal straight ahead (no rotation needed).

        Uses _pose_phase to verify phase-specific invariants:
        - ROTATE phase should be skipped or very short (goal on X-axis)
        - FORWARD phase: distance decreases monotonically, orientation stays fixed,
          velocity is non-zero while far from goal
        After reaching:
        - is_moving is False, velocity ≈ 0, position ≈ goal
        """
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        goal_pos = np.array([2, 0, 0])
        agent.set_goal_pose(Pose.from_xyz(*goal_pos))

        result = self._run_pose_phase_loop(agent, sim_core, goal_pos)

        # Straight-ahead goal: ROTATE phase should be very short or skipped
        assert result["rotate_cycles"] <= 1, (
            f"ROTATE phase should be skipped for straight-ahead goal, " f"got {result['rotate_cycles']} cycles"
        )
        assert result["reached"], "Differential agent should reach the forward goal"
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos, goal_pos, atol=0.05), f"Final position {pos} not close to goal"
        assert np.allclose(agent.velocity, [0, 0, 0], atol=1e-4), f"Velocity should be ~0 after goal, got {agent.velocity}"

    def test_diff_update_toward_goal_lateral(self, pybullet_env):
        """Differential agent reaching a goal at 90° (rotation then forward).

        Uses _pose_phase to verify strict per-phase invariants.
        The agent may go through multiple ROTATE→FORWARD cycles (e.g. initial
        rotation, forward motion, then final orientation alignment).

        ROTATE phase (each cycle):
        - angle error to target quaternion decreases monotonically (after warmup)
        - position stays fixed (no translation during rotation)
        - angular_velocity is non-zero after warmup

        FORWARD phase (each cycle):
        - distance to goal decreases monotonically
        - orientation stays fixed (locked after rotation completes)
        - linear velocity is non-zero while far from goal

        After reaching:
        - is_moving is False, velocity ≈ 0, position ≈ goal
        """
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        goal_pos = np.array([0, 2, 0])
        agent.set_goal_pose(Pose.from_xyz(*goal_pos))

        result = self._run_pose_phase_loop(agent, sim_core, goal_pos)

        assert result["rotate_cycles"] >= 1, "ROTATE phase should be observed for 90° lateral goal"
        assert result["reached"], "Differential agent should reach the lateral goal"
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos, goal_pos, atol=0.05), f"Final position {pos} not close to goal"
        assert np.allclose(agent.velocity, [0, 0, 0], atol=1e-4), f"Velocity should be ~0 after goal, got {agent.velocity}"

    # -- Final orientation alignment -------------------------------------

    def test_diff_final_orientation_align(self, pybullet_env):
        """Differential agent aligns to goal orientation after reaching position.

        Goal is straight ahead (X+) but with yaw=1.0 rad.  The agent should:
        1. FORWARD phase — move to goal position (no initial rotation needed)
        2. After reaching position, _start_final_orientation_alignment fires
        3. ROTATE phase — rotate in place to match goal yaw
           - position stays fixed at goal
           - angle error to goal quaternion decreases monotonically
        4. FORWARD phase (zero-distance) — completes immediately
        Final orientation must match the goal quaternion.
        """
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        goal = Pose.from_euler(2, 0, 0, yaw=1.0)
        goal_pos = np.array(goal.position)
        goal_quat = np.array(goal.orientation)
        agent.set_goal_pose(goal)

        # final_orientation_align should be enabled (set_goal_pose → set_path default)
        assert agent._controller._align_final_orientation is True

        result = self._run_pose_phase_loop(agent, sim_core, goal_pos)

        assert result["final_rotate_at_goal"], "Final orientation alignment ROTATE phase should occur at goal position"
        assert result["reached"], "Differential agent should complete motion including final alignment"
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos, goal_pos, atol=0.05), f"Final position {pos} not close to goal"
        # Final orientation must match the goal quaternion
        final_orient = np.array(agent.get_pose().orientation)
        angle_err = self._quat_angle_error(final_orient, goal_quat)
        assert angle_err < 0.05, (
            f"Final orientation error {angle_err:.4f} rad to goal; " f"final={list(final_orient)}, goal={list(goal_quat)}"
        )

    # -- Rotation-only goal (no translation) -----------------------------

    def test_diff_rotation_only_goal(self, pybullet_env):
        """Differential agent rotates in place when goal is at current position with different yaw.

        The agent should:
        - Stay at the same position throughout
        - Rotate to match the goal orientation
        - angle error decreases monotonically (after warmup)
        """
        start_pos = np.array([1, 2, 0])
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(*start_pos),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        goal_yaw = 1.5
        goal = Pose.from_euler(*start_pos, yaw=goal_yaw)
        goal_quat = np.array(goal.orientation)
        agent.set_goal_pose(goal)

        dt = sim_core._dt
        warmup_steps = 5
        prev_angle_error = self._quat_angle_error(agent.get_pose().orientation, goal_quat)
        reached = False

        for step in range(5000):
            sim_core.tick()
            agent.update(dt)
            pose = agent.get_pose()
            cur_pos = np.array(pose.position)

            # Position must stay fixed (no translation)
            assert np.allclose(cur_pos, start_pos, atol=1e-3), f"Step {step}: position moved from {start_pos} to {cur_pos}"

            if not agent.is_moving:
                reached = True
                break

            # After warmup, angle error should decrease monotonically
            cur_angle_error = self._quat_angle_error(pose.orientation, goal_quat)
            if step >= warmup_steps:
                assert cur_angle_error <= prev_angle_error + 1e-4, (
                    f"Step {step}: angle error increased " f"{prev_angle_error:.6f} -> {cur_angle_error:.6f} rad"
                )
            prev_angle_error = cur_angle_error

        assert reached, "Differential agent should complete rotation-only goal"
        final_orient = np.array(agent.get_pose().orientation)
        angle_err = self._quat_angle_error(final_orient, goal_quat)
        assert angle_err < 0.05, (
            f"Final orientation error {angle_err:.4f} rad; " f"final={list(final_orient)}, goal={list(goal_quat)}"
        )

    def test_omni_rotation_only_goal(self, pybullet_env):
        """Omnidirectional agent rotates in place when goal is at current position with different yaw.

        The agent should:
        - Stay at the same position throughout (position fixed in final-rotation phase)
        - Rotate to match the goal orientation via final orientation alignment
        - Angle error to goal decreases monotonically (after warmup)
        """
        start_pos = np.array([1, 2, 0])
        agent, sim_core = self._create_agent_with_simcore(Pose.from_xyz(*start_pos))
        goal_yaw = 1.5
        goal = Pose.from_euler(*start_pos, yaw=goal_yaw)
        goal_quat = np.array(goal.orientation)
        initial_quat = list(agent.get_pose().orientation)
        agent.set_goal_pose(goal)

        result = self._run_omni_motion_loop(agent, sim_core, start_pos, initial_quat, final_quat=goal_quat)

        assert result["reached"], "Omnidirectional agent should complete rotation-only goal"
        final_orient = np.array(agent.get_pose().orientation)
        angle_err = self._quat_angle_error(final_orient, goal_quat)
        assert angle_err < 0.05, (
            f"Final orientation error {angle_err:.4f} rad; " f"final={list(final_orient)}, goal={list(goal_quat)}"
        )

    @staticmethod
    def _quat_angle_error(q_current, q_target) -> float:
        """Compute the angular distance (radians) between two quaternions [x,y,z,w]."""
        r_current = R.from_quat(q_current)
        r_target = R.from_quat(q_target)
        r_diff = r_current.inv() * r_target
        return float(r_diff.magnitude())


# ============================================================================
# Joint control tests (URDF)
# ============================================================================


@pytest.mark.parametrize("mass", [None, 0.0], ids=["physics", "kinematic"])
class TestAgentJointControl:
    """Test URDF joint state and control methods.

    arm_robot.urdf has 4 revolute joints:
        0: base_to_shoulder   (axis Z, ±π)
        1: shoulder_to_elbow  (axis X, ±2.0)
        2: elbow_to_wrist     (axis X, ±2.5)
        3: wrist_to_end       (axis Z, ±1.57)

    Parametrized with mass=None (physics, URDF mass values) and mass=0.0
    (kinematic interpolation) to verify both modes produce the same end results.
    """

    ARM_JOINT_NAMES = [
        "base_to_shoulder",
        "shoulder_to_elbow",
        "elbow_to_wrist",
        "wrist_to_end",
    ]

    def _create_arm(self, pybullet_env, mass, *, use_fixed_base=True):
        agent = Agent.from_urdf(
            urdf_path=ARM_URDF,
            pose=Pose.from_xyz(0, 0, 0.5),
            mass=mass,
            use_fixed_base=use_fixed_base,
        )
        return agent

    def _step(self, agent, mass, n=500, dt=1.0 / 240.0):
        """Advance simulation n steps using the right mechanism for the mode.

        Physics (mass=None): p.stepSimulation() drives motor control.
        Kinematic (mass=0.0): only agent.update(dt) drives interpolation.
        agent.update(dt) is always called (handles base motion, actions, etc.).
        """
        for _ in range(n):
            if mass != 0.0:
                p.stepSimulation()
            agent.update(dt)

    # -- Joint info / consistency ----------------------------------------

    def test_num_joints_matches_urdf(self, pybullet_env, mass):
        """get_num_joints() should match the URDF (4 joints for arm_robot)."""
        agent = self._create_arm(pybullet_env, mass)
        assert agent.get_num_joints() == 4

    def test_joint_names_match_urdf(self, pybullet_env, mass):
        """joint_info names should match the URDF joint names."""
        agent = self._create_arm(pybullet_env, mass)
        names = [info[1].decode("utf-8") for info in agent.joint_info]
        assert names == self.ARM_JOINT_NAMES

    def test_joint_types_all_revolute(self, pybullet_env, mass):
        """All arm_robot joints should be revolute (type 0 in pybullet)."""
        agent = self._create_arm(pybullet_env, mass)
        for info in agent.joint_info:
            assert info[2] == p.JOINT_REVOLUTE, f"Joint {info[1]} should be revolute"

    def test_is_urdf_robot(self, pybullet_env, mass):
        """URDF agent should return True from is_urdf_robot()."""
        agent = self._create_arm(pybullet_env, mass)
        assert agent.is_urdf_robot() is True

    def test_mesh_agent_no_joints(self, pybullet_env, mass):
        """Mesh agent should have 0 joints."""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5))
        assert agent.get_num_joints() == 0
        assert agent.is_urdf_robot() is False

    # -- get_joint_state / set_joint_target (by index) -------------------

    def test_initial_joint_positions_zero(self, pybullet_env, mass):
        """All joints should start at position 0, consistent with pybullet."""
        agent = self._create_arm(pybullet_env, mass)
        for i in range(agent.get_num_joints()):
            pos, vel = agent.get_joint_state(i)
            pb_state = p.getJointState(agent.body_id, i)
            pb_pos, pb_vel = pb_state[0], pb_state[1]
            assert abs(pos - pb_pos) < 1e-6, f"Joint {i}: get_joint_state pos={pos} != pybullet pos={pb_pos}"
            assert abs(vel - pb_vel) < 1e-6, f"Joint {i}: get_joint_state vel={vel} != pybullet vel={pb_vel}"
            assert abs(pos) < 0.01, f"Joint {i} should start at 0, got {pos}"

    def test_set_joint_target_and_simulate(self, pybullet_env, mass):
        """set_joint_target should move joint toward target after simulation steps."""
        agent = self._create_arm(pybullet_env, mass)
        target = 0.5  # radians
        agent.set_joint_target(0, target)

        self._step(agent, mass, n=500)

        pos, _ = agent.get_joint_state(0)
        pb_pos = p.getJointState(agent.body_id, 0)[0]
        assert abs(pos - pb_pos) < 1e-6, f"get_joint_state pos={pos} != pybullet pos={pb_pos}"
        assert abs(pos - target) < 0.05, f"Joint 0 should be near {target}, got {pos}"
        assert agent.is_joint_at_target(0, target, tolerance=0.05)

    # -- get/set by name -------------------------------------------------

    def test_get_joint_state_by_name(self, pybullet_env, mass):
        """get_joint_state with a joint name should return the same result as by index and pybullet."""
        agent = self._create_arm(pybullet_env, mass)
        pos_idx, vel_idx = agent.get_joint_state(0)
        pos_name, vel_name = agent.get_joint_state("base_to_shoulder")
        pb_state = p.getJointState(agent.body_id, 0)
        pb_pos, pb_vel = pb_state[0], pb_state[1]

        # by-index and by-name must agree
        assert pos_idx == pos_name
        assert vel_idx == vel_name
        # both must match pybullet
        assert abs(pos_idx - pb_pos) < 1e-6, f"pos by index={pos_idx} != pybullet={pb_pos}"
        assert abs(vel_idx - pb_vel) < 1e-6, f"vel by index={vel_idx} != pybullet={pb_vel}"

    def test_set_joint_target_by_name(self, pybullet_env, mass):
        """set_joint_target with a joint name should move the joint as with its index."""
        agent = self._create_arm(pybullet_env, mass)

        # Initially at 0
        assert agent.is_joint_at_target("shoulder_to_elbow", 0.0, tolerance=0.01)

        target = 0.4
        agent.set_joint_target("shoulder_to_elbow", target)

        self._step(agent, mass, n=500)

        assert agent.is_joint_at_target("shoulder_to_elbow", target, tolerance=0.05)

    # -- Batch operations ------------------------------------------------

    def test_get_all_joints_state(self, pybullet_env, mass):
        """get_all_joints_state returns list of (pos, vel) matching pybullet."""
        agent = self._create_arm(pybullet_env, mass)
        states = agent.get_all_joints_state()
        assert len(states) == 4
        for i, (pos, vel) in enumerate(states):
            pb_state = p.getJointState(agent.body_id, i)
            assert abs(pos - pb_state[0]) < 1e-6, f"Joint {i}: pos={pos} != pybullet={pb_state[0]}"
            assert abs(vel - pb_state[1]) < 1e-6, f"Joint {i}: vel={vel} != pybullet={pb_state[1]}"

    def test_get_all_joints_state_by_name(self, pybullet_env, mass):
        """get_all_joints_state_by_name returns dict keyed by joint name matching pybullet."""
        agent = self._create_arm(pybullet_env, mass)
        states = agent.get_all_joints_state_by_name()
        assert set(states.keys()) == set(self.ARM_JOINT_NAMES)

        # Values must match pybullet for each joint
        for i, name in enumerate(self.ARM_JOINT_NAMES):
            pos, vel = states[name]
            pb_state = p.getJointState(agent.body_id, i)
            assert abs(pos - pb_state[0]) < 1e-6, f"{name}: pos={pos} != pybullet={pb_state[0]}"
            assert abs(vel - pb_state[1]) < 1e-6, f"{name}: vel={vel} != pybullet={pb_state[1]}"

    def test_set_all_joints_targets(self, pybullet_env, mass):
        """set_all_joints_targets moves all joints to specified positions."""
        agent = self._create_arm(pybullet_env, mass)
        targets = [0.3, 0.2, -0.2, 0.1]
        agent.set_all_joints_targets(targets)

        self._step(agent, mass, n=1000)

        assert agent.are_all_joints_at_targets(targets, tolerance=0.05)

    def test_set_joints_targets_by_name(self, pybullet_env, mass):
        """set_joints_targets_by_name (partial update) works correctly."""
        agent = self._create_arm(pybullet_env, mass)
        partial = {"base_to_shoulder": 0.5, "elbow_to_wrist": -0.3}
        agent.set_joints_targets_by_name(partial)

        self._step(agent, mass, n=1000)

        assert agent.are_joints_at_targets_by_name(partial, tolerance=0.05)

    def test_set_joints_targets_list_and_dict(self, pybullet_env, mass):
        """set_joints_targets accepts both list and dict."""
        agent = self._create_arm(pybullet_env, mass)

        # list form
        targets_list = [0.1, 0.2, 0.3, 0.1]
        agent.set_joints_targets(targets_list)
        self._step(agent, mass, n=1000)
        assert agent.are_joints_at_targets(targets_list, tolerance=0.05)

        # dict form
        targets_dict = {"base_to_shoulder": -0.2, "wrist_to_end": 0.4}
        agent.set_joints_targets(targets_dict)
        self._step(agent, mass, n=1000)
        assert agent.are_joints_at_targets(targets_dict, tolerance=0.05)

    # -- Edge cases / warnings -------------------------------------------

    def test_kinematic_joint_cache_avoids_pybullet_call(self, pybullet_env, mass):
        """Kinematic robots should cache joint positions internally.

        After stepping, get_joint_state must return the cached position
        WITHOUT calling p.getJointState.  We patch p.getJointState after
        stepping to prove it is never invoked during the read path.
        """
        if mass != 0.0:
            pytest.skip("cache only applies to kinematic (mass=0)")
        agent = self._create_arm(pybullet_env, mass)

        # Set target and step enough for interpolation to move
        agent.set_joint_target(0, 0.5)
        self._step(agent, mass, n=200)

        # Record ground truth BEFORE patching
        pb_pos = p.getJointState(agent.body_id, 0)[0]
        cached_pos = agent._joints[0].position

        # Patch p.getJointState — if get_joint_state calls it, the test fails
        from unittest.mock import patch

        with patch(
            "pybullet.getJointState", side_effect=AssertionError("p.getJointState should NOT be called for kinematic robots")
        ):
            pos, vel = agent.get_joint_state(0)

        # Cached value should match what get_joint_state returns
        assert abs(cached_pos - pos) < 1e-9, f"Cache ({cached_pos}) != get_joint_state ({pos})"
        # And match pybullet ground truth
        assert abs(cached_pos - pb_pos) < 1e-6, f"Cache ({cached_pos}) != pybullet ({pb_pos})"
        # Velocity should be 0 for kinematic
        assert vel == 0.0

    def test_mesh_agent_joint_state_returns_zero(self, pybullet_env, mass):
        """Calling get_joint_state on mesh agent returns (0, 0) gracefully."""
        agent = create_mesh_agent(pose=Pose.from_xyz(0, 0, 0.5))
        pos, vel = agent.get_joint_state(0)
        assert pos == 0.0
        assert vel == 0.0

    def test_get_joints_state_by_name_subset(self, pybullet_env, mass):
        """get_joints_state_by_name with a subset of joint names."""
        agent = self._create_arm(pybullet_env, mass)
        subset = ["base_to_shoulder", "wrist_to_end"]
        states = agent.get_joints_state_by_name(subset)
        assert set(states.keys()) == set(subset)


# ============================================================================
# _resolve_joint_tolerance helper
# ============================================================================


class TestResolveJointTolerance:
    """Unit tests for Agent._resolve_joint_tolerance().

    arm_robot.urdf joints (all revolute):
        0: base_to_shoulder
        1: shoulder_to_elbow
        2: elbow_to_wrist
        3: wrist_to_end
    """

    NUM_JOINTS = 4

    def _create(self, pybullet_env):
        client, _ = pybullet_env
        sim_core = MockSimCore()
        sim_core._client = client
        return Agent.from_urdf(
            urdf_path=ARM_URDF,
            pose=Pose.from_xyz(0, 0, 0),
            mass=0.0,
            use_fixed_base=True,
            sim_core=sim_core,
        )

    def test_none_falls_back_to_agent_default(self, pybullet_env):
        """tolerance=None → self.joint_tolerance → class default 0.01 for all joints."""
        agent = self._create(pybullet_env)
        for i in range(self.NUM_JOINTS):
            assert agent._resolve_joint_tolerance(None, i) == 0.01

    def test_none_falls_back_to_custom_agent_tolerance(self, pybullet_env):
        """tolerance=None with agent.joint_tolerance set → uses agent value for all joints."""
        agent = self._create(pybullet_env)
        agent.joint_tolerance = 0.05
        for i in range(self.NUM_JOINTS):
            assert agent._resolve_joint_tolerance(None, i) == 0.05

    def test_scalar_returns_as_is(self, pybullet_env):
        """Scalar float tolerance is returned directly for all joints."""
        agent = self._create(pybullet_env)
        for i in range(self.NUM_JOINTS):
            assert agent._resolve_joint_tolerance(0.1, i) == 0.1

    def test_dict_resolves_by_joint_name(self, pybullet_env):
        """Dict tolerance: present keys → custom value, missing keys → default."""
        agent = self._create(pybullet_env)
        tol = {"base_to_shoulder": 0.001, "elbow_to_wrist": 0.05}
        # Present keys
        assert agent._resolve_joint_tolerance(tol, 0) == 0.001  # base_to_shoulder
        assert agent._resolve_joint_tolerance(tol, 2) == 0.05  # elbow_to_wrist
        # Missing keys → _DEFAULT_JOINT_TOLERANCE
        assert agent._resolve_joint_tolerance(tol, 1) == agent._DEFAULT_JOINT_TOLERANCE  # shoulder_to_elbow
        assert agent._resolve_joint_tolerance(tol, 3) == agent._DEFAULT_JOINT_TOLERANCE  # wrist_to_end

    def test_list_resolves_by_index(self, pybullet_env):
        """List tolerance indexes by joint_index."""
        agent = self._create(pybullet_env)
        tol = [0.01, 0.02, 0.03, 0.04]
        for i in range(self.NUM_JOINTS):
            assert agent._resolve_joint_tolerance(tol, i) == tol[i]

    def test_list_short_falls_back_to_default(self, pybullet_env):
        """List shorter than joint count → out-of-range indices get _DEFAULT_JOINT_TOLERANCE."""
        agent = self._create(pybullet_env)
        tol = [0.01, 0.02]  # only 2 elements, 4 joints
        assert agent._resolve_joint_tolerance(tol, 0) == 0.01
        assert agent._resolve_joint_tolerance(tol, 1) == 0.02
        # Out of range → default, not clamp-to-last
        assert agent._resolve_joint_tolerance(tol, 2) == agent._DEFAULT_JOINT_TOLERANCE
        assert agent._resolve_joint_tolerance(tol, 3) == agent._DEFAULT_JOINT_TOLERANCE

    def test_none_with_dict_agent_tolerance(self, pybullet_env):
        """tolerance=None + agent.joint_tolerance is a dict → resolves per joint."""
        agent = self._create(pybullet_env)
        agent.joint_tolerance = {"base_to_shoulder": 0.001, "elbow_to_wrist": 0.05}
        assert agent._resolve_joint_tolerance(None, 0) == 0.001
        assert agent._resolve_joint_tolerance(None, 2) == 0.05
        # Missing keys → default
        assert agent._resolve_joint_tolerance(None, 1) == agent._DEFAULT_JOINT_TOLERANCE
        assert agent._resolve_joint_tolerance(None, 3) == agent._DEFAULT_JOINT_TOLERANCE


# ============================================================================
# URDF link attach / detach tests
# ============================================================================


class TestAgentURDFLinkAttach:
    """Test attaching/detaching SimObjects to URDF robot links.

    arm_robot.urdf links:
        -1: base_link
         0: shoulder_link  (child of base_to_shoulder joint)
         1: elbow_link     (child of shoulder_to_elbow joint)
         2: wrist_link     (child of elbow_to_wrist joint)
         3: end_effector   (child of wrist_to_end joint)
    """

    def _create_arm_agent(self, pybullet_env):
        return Agent.from_urdf(
            urdf_path=ARM_URDF,
            pose=Pose.from_xyz(0, 0, 0.5),
            use_fixed_base=True,  # mass=None (default) uses URDF mass values
        )

    def _create_pickable_box(self, pos=(0, 0, 0)):
        """Create a small pickable SimObject for attachment."""
        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.02, 0.02, 0.02]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.02, 0.02, 0.02]),
            pose=Pose.from_xyz(*pos),
            mass=0.0,
            pickable=True,
        )
        return obj

    def test_attach_to_end_effector(self, pybullet_env):
        """Attach a box to the end effector link (index 3)."""
        agent = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()

        success = agent.attach_object(box, parent_link_index=3)
        assert success is True
        assert box in agent.attached_objects
        assert box.is_attached()
        assert box._attached_link_index == 3

    def test_attach_to_base_link(self, pybullet_env):
        """Attach to base link (index -1)."""
        agent = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()

        success = agent.attach_object(box, parent_link_index=-1)
        assert success is True
        assert box._attached_link_index == -1

    def test_attach_by_link_name(self, pybullet_env):
        """Attach using link name instead of index; resolved index must match."""
        agent = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()

        success = agent.attach_object(box, parent_link_index="end_effector")
        assert success is True
        assert box in agent.attached_objects
        assert box.is_attached()
        assert box._attached_link_index == 3  # end_effector is link 3

    def test_attach_by_base_link_name(self, pybullet_env):
        """Attach using 'base_link' string resolves to index -1."""
        agent = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()

        success = agent.attach_object(box, parent_link_index="base_link")
        assert success is True
        assert box._attached_link_index == -1

    def test_detach_from_link(self, pybullet_env):
        """Detach a previously attached object."""
        agent = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()

        agent.attach_object(box, parent_link_index=3)
        success = agent.detach_object(box)

        assert success is True
        assert box not in agent.attached_objects
        assert not box.is_attached()
        assert box._attached_link_index == -1

    def test_attach_follows_link_after_joint_move(self, pybullet_env):
        """Kinematic attached object should follow link when joint moves."""
        agent = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()
        # Non-trivial relative offset: 5cm X, 3cm Y, 10cm Z + 45° yaw
        rel_pos = [0.05, 0.03, 0.1]
        rel_orn = list(R.from_euler("z", 45, degrees=True).as_quat())  # [x,y,z,w]

        # Attach to end effector (link 3)
        agent.attach_object(
            box,
            parent_link_index=3,
            relative_pose=Pose(position=rel_pos, orientation=rel_orn),
        )

        # Update kinematics and record initial position
        agent.update_attached_objects_kinematics()
        initial_pos = np.array(box.get_pose().position)

        # Move shoulder pitch joint (joint 1: shoulder_to_elbow, axis X)
        # This tilts the arm and significantly moves the end effector
        agent.set_joint_target(1, 1.0)
        for _ in range(1000):
            p.stepSimulation()

        # Verify joint actually moved
        pos1, _ = agent.get_joint_state(1)
        assert abs(pos1 - 1.0) < 0.1, f"Joint 1 should have moved to ~1.0, got {pos1}"

        # Update attached objects kinematics
        agent.update_attached_objects_kinematics()
        new_pos = np.array(box.get_pose().position)
        new_orn = np.array(box.get_pose().orientation)

        # Box should have moved because the end effector moved
        assert not np.allclose(
            initial_pos, new_pos, atol=0.01
        ), f"Attached box should move when joint moves: initial={initial_pos}, new={new_pos}"

        # Compute expected position from PyBullet link state + relative offset
        link_state = p.getLinkState(agent.body_id, 3)
        expected_pos, expected_orn = p.multiplyTransforms(link_state[0], link_state[1], rel_pos, rel_orn)
        assert np.allclose(new_pos, expected_pos, atol=1e-4), (
            f"Box position {list(new_pos)} should match " f"link3 + offset = {list(expected_pos)}"
        )
        assert np.allclose(new_orn, expected_orn, atol=1e-4), (
            f"Box orientation {list(new_orn)} should match " f"link3 + offset = {list(expected_orn)}"
        )

    def test_attach_multiple_to_different_links(self, pybullet_env):
        """Attach different objects to different links."""
        agent = self._create_arm_agent(pybullet_env)
        box1 = self._create_pickable_box(pos=(1, 0, 0))
        box2 = self._create_pickable_box(pos=(2, 0, 0))

        s1 = agent.attach_object(box1, parent_link_index=1)
        s2 = agent.attach_object(box2, parent_link_index=3)

        assert s1 and s2
        assert len(agent.attached_objects) == 2
        assert box1._attached_link_index == 1
        assert box2._attached_link_index == 3

    def test_detach_one_keeps_other(self, pybullet_env):
        """Detaching one object should not affect another."""
        agent = self._create_arm_agent(pybullet_env)
        box1 = self._create_pickable_box(pos=(1, 0, 0))
        box2 = self._create_pickable_box(pos=(2, 0, 0))

        agent.attach_object(box1, parent_link_index=1)
        agent.attach_object(box2, parent_link_index=3)

        agent.detach_object(box1)

        assert box1 not in agent.attached_objects
        assert box2 in agent.attached_objects
        assert box2.is_attached()

    def test_cannot_attach_non_pickable(self, pybullet_env):
        """Non-pickable objects should not be attachable."""
        agent = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()
        box.pickable = False

        success = agent.attach_object(box, parent_link_index=3)
        assert success is False

    def test_cannot_double_attach(self, pybullet_env):
        """An already-attached object cannot be attached to a second parent."""
        agent1 = self._create_arm_agent(pybullet_env)
        agent2 = self._create_arm_agent(pybullet_env)
        box = self._create_pickable_box()

        s1 = agent1.attach_object(box, parent_link_index=3)
        s2 = agent2.attach_object(box, parent_link_index=3)

        assert s1 is True
        assert s2 is False  # Already attached to agent1

    # -- URDF-to-URDF and chained attachment -----------------------------

    def test_attach_urdf_to_urdf(self, pybullet_env):
        """Attach a URDF agent to another URDF agent's link.

        A small arm (child) is attached to the end effector of the parent arm.
        After joint movement on the parent, the child's base should follow
        the parent's end effector according to PyBullet link state + offset.
        """
        parent = self._create_arm_agent(pybullet_env)
        child = Agent.from_urdf(
            urdf_path=ARM_URDF,
            pose=Pose.from_xyz(1, 0, 0),
            mass=0.0,
            use_fixed_base=False,
        )
        child.pickable = True  # Agent defaults pickable=False

        rel_pos = [0, 0, 0.05]
        rel_orn = [0, 0, 0, 1]
        success = parent.attach_object(
            child,
            parent_link_index="end_effector",
            relative_pose=Pose(position=rel_pos, orientation=rel_orn),
        )
        assert success is True
        assert child in parent.attached_objects
        assert child._attached_link_index == 3

        # Move parent joint and update kinematics
        parent.set_joint_target(1, 0.8)
        for _ in range(1000):
            p.stepSimulation()
        parent.update_attached_objects_kinematics()

        # Verify child base follows parent end effector + offset
        link_state = p.getLinkState(parent.body_id, 3)
        expected_pos, expected_orn = p.multiplyTransforms(link_state[0], link_state[1], rel_pos, rel_orn)
        child_pose = child.get_pose()
        assert np.allclose(child_pose.position, expected_pos, atol=1e-4), (
            f"Child URDF position {list(child_pose.position)} should match " f"parent link3 + offset = {list(expected_pos)}"
        )
        assert np.allclose(child_pose.orientation, expected_orn, atol=1e-4), (
            f"Child URDF orientation {list(child_pose.orientation)} should match "
            f"parent link3 + offset = {list(expected_orn)}"
        )

    def test_chained_attach_follows_joint_move(self, pybullet_env):
        """Object attached to an attached object should follow joint movement.

        Chain: agent (link 3) → box_a → box_b
        When the agent's joint moves, box_a follows via update_attached_objects_kinematics,
        which internally calls set_pose_raw on box_a, which recursively updates box_b.
        """
        agent = self._create_arm_agent(pybullet_env)
        box_a = self._create_pickable_box()
        box_b = self._create_pickable_box(pos=(0.5, 0, 0))

        rel_a_pos = [0, 0, 0.08]
        rel_a_orn = [0, 0, 0, 1]
        rel_b_pos = [0.04, 0, 0]
        rel_b_orn = [0, 0, 0, 1]

        # Chain: agent → box_a → box_b
        agent.attach_object(
            box_a,
            parent_link_index=3,
            relative_pose=Pose(position=rel_a_pos, orientation=rel_a_orn),
        )
        box_a.attach_object(
            box_b,
            relative_pose=Pose(position=rel_b_pos, orientation=rel_b_orn),
        )

        # Record initial positions
        agent.update_attached_objects_kinematics()
        initial_b_pos = np.array(box_b.get_pose().position)

        # Move joint
        agent.set_joint_target(1, 1.0)
        for _ in range(1000):
            p.stepSimulation()

        pos1, _ = agent.get_joint_state(1)
        assert abs(pos1 - 1.0) < 0.1, f"Joint 1 should have moved to ~1.0, got {pos1}"

        agent.update_attached_objects_kinematics()
        new_b_pos = np.array(box_b.get_pose().position)

        # box_b should have moved
        assert not np.allclose(
            initial_b_pos, new_b_pos, atol=0.01
        ), f"Chained box_b should move: initial={list(initial_b_pos)}, new={list(new_b_pos)}"

        # Verify the full chain: link3 → box_a → box_b
        link_state = p.getLinkState(agent.body_id, 3)
        expected_a_pos, expected_a_orn = p.multiplyTransforms(link_state[0], link_state[1], rel_a_pos, rel_a_orn)
        expected_b_pos, expected_b_orn = p.multiplyTransforms(expected_a_pos, expected_a_orn, rel_b_pos, rel_b_orn)
        assert np.allclose(new_b_pos, expected_b_pos, atol=1e-4), (
            f"box_b position {list(new_b_pos)} should match " f"link3 + offset_a + offset_b = {list(expected_b_pos)}"
        )
        new_b_orn = np.array(box_b.get_pose().orientation)
        assert np.allclose(new_b_orn, expected_b_orn, atol=1e-4), (
            f"box_b orientation {list(new_b_orn)} should match " f"expected = {list(expected_b_orn)}"
        )

    def test_update_propagates_attachment_when_base_stationary(self, pybullet_env):
        """update() must update attached objects even when base is not moving.

        Regression test: Agent.update() had an early return when the base was
        stationary (_is_moving=False), which skipped
        update_attached_objects_kinematics(). This caused attached objects to
        stay in place when only the arm joints moved (e.g. JointAction after
        PickAction).

        Uses mobile_manipulator.urdf (the real scenario: mobile base + arm)
        with joints:
          0: base_to_left_wheel  (continuous)
          1: base_to_right_wheel (continuous)
          2: base_to_mount       (FIXED)
          3: mount_to_shoulder   (revolute)
          4: shoulder_to_elbow   (revolute)
          5: elbow_to_wrist      (revolute)
          6: wrist_to_end        (revolute)  → child: end_effector (link 6)
        """
        client_id = pybullet_env[0]
        sim_core = MockSimCore(physics=False)
        sim_core._client = client_id
        ee_link_index = 6  # end_effector link
        arm_joint_index = 4  # shoulder_to_elbow (revolute, moves EE significantly)
        arm_joint_target = 1.0
        agent = Agent.from_urdf(
            urdf_path=MOBILE_MANIPULATOR_URDF,
            pose=Pose.from_xyz(0, 0, 0.5),
            mass=0.0,
            use_fixed_base=False,
            sim_core=sim_core,
        )
        box = self._create_pickable_box()
        rel_pos = [0, 0, 0.06]
        rel_orn = [0, 0, 0, 1]
        agent.attach_object(
            box,
            parent_link_index=ee_link_index,
            relative_pose=Pose(position=rel_pos, orientation=rel_orn),
        )

        # Record initial box position (at home pose)
        agent.update_attached_objects_kinematics()
        initial_pos = np.array(box.get_pose().position)

        # Set arm joint target — base remains stationary
        agent.set_joint_target(arm_joint_index, arm_joint_target)

        # Run update loop (NOT calling update_attached_objects_kinematics directly)
        dt = sim_core._dt
        for _ in range(500):
            sim_core.tick()
            agent.update(dt)

        # Joint should have moved via kinematic interpolation
        pos_j, _ = agent.get_joint_state(arm_joint_index)
        assert abs(pos_j - arm_joint_target) < 0.05, f"Joint {arm_joint_index} should be near {arm_joint_target}, got {pos_j}"

        # Box MUST have followed the end effector
        new_pos = np.array(box.get_pose().position)
        assert not np.allclose(initial_pos, new_pos, atol=0.01), (
            f"Attached box must move when arm joints move via update(): " f"initial={list(initial_pos)}, after={list(new_pos)}"
        )

        # Verify exact position matches expected link state + offset
        link_state = p.getLinkState(agent.body_id, ee_link_index, computeForwardKinematics=1, physicsClientId=client_id)
        expected_pos, expected_orn = p.multiplyTransforms(link_state[0], link_state[1], rel_pos, rel_orn)
        assert np.allclose(new_pos, expected_pos, atol=1e-4), (
            f"Box position {list(new_pos)} should match " f"link{ee_link_index} + offset = {list(expected_pos)}"
        )

    def test_update_propagates_attachment_after_base_move_then_joint_move(self, pybullet_env):
        """update() must propagate attachments during both base movement and joint movement.

        Complements test_update_propagates_attachment_when_base_stationary:
        that test covers joint-only movement; this test covers the full
        mobile manipulator workflow:
          Phase 1: Base moves to a new position (arm at home pose)
                   → attached object must follow EE throughout
          Phase 2: Base stops, arm joint moves
                   → attached object must follow EE at the new base position

        Uses mobile_manipulator.urdf with joints:
          0: base_to_left_wheel  (continuous)
          1: base_to_right_wheel (continuous)
          2: base_to_mount       (FIXED)
          3: mount_to_shoulder   (revolute)
          4: shoulder_to_elbow   (revolute)
          5: elbow_to_wrist      (revolute)
          6: wrist_to_end        (revolute)  → child: end_effector (link 6)
        """
        client_id = pybullet_env[0]
        sim_core = MockSimCore(physics=False)
        sim_core._client = client_id
        ee_link_index = 6
        arm_joint_index = 4  # shoulder_to_elbow
        arm_joint_target = 1.0
        dt = sim_core._dt

        agent = Agent.from_urdf(
            urdf_path=MOBILE_MANIPULATOR_URDF,
            pose=Pose.from_xyz(0, 0, 0),
            mass=0.0,
            use_fixed_base=False,
            sim_core=sim_core,
        )
        box = self._create_pickable_box()
        rel_pos = [0, 0, 0.06]
        rel_orn = [0, 0, 0, 1]
        agent.attach_object(
            box,
            parent_link_index=ee_link_index,
            relative_pose=Pose(position=rel_pos, orientation=rel_orn),
        )

        # Record initial box position
        agent.update_attached_objects_kinematics()
        initial_pos = np.array(box.get_pose().position)

        # --- Phase 1: Move base to (3, 0, 0) ---
        goal_pos = [3, 0, 0]
        agent.set_goal_pose(Pose.from_xyz(*goal_pos))
        for _ in range(5000):
            sim_core.tick()
            agent.update(dt)
            if not agent.is_moving:
                break

        assert not agent.is_moving, "Base should have reached goal"
        base_pos = np.array(agent.get_pose().position)
        assert np.allclose(base_pos[:2], goal_pos[:2], atol=0.1), f"Base should be near {goal_pos}, got {list(base_pos)}"

        # Box must have moved from initial position (it followed the base)
        after_base_move_pos = np.array(box.get_pose().position)
        assert not np.allclose(initial_pos, after_base_move_pos, atol=0.01), (
            f"Box must move during base movement: " f"initial={list(initial_pos)}, after={list(after_base_move_pos)}"
        )

        # Verify box is at correct FK position after base move
        link_state = p.getLinkState(agent.body_id, ee_link_index, computeForwardKinematics=1, physicsClientId=client_id)
        expected_pos, _ = p.multiplyTransforms(link_state[0], link_state[1], rel_pos, rel_orn)
        assert np.allclose(after_base_move_pos, expected_pos, atol=1e-4), (
            f"After base move: box {list(after_base_move_pos)} != " f"expected {list(expected_pos)}"
        )

        # --- Phase 2: Move arm joint (base stationary) ---
        agent.set_joint_target(arm_joint_index, arm_joint_target)
        for _ in range(500):
            sim_core.tick()
            agent.update(dt)

        pos_j, _ = agent.get_joint_state(arm_joint_index)
        assert abs(pos_j - arm_joint_target) < 0.05, f"Joint {arm_joint_index} should be near {arm_joint_target}, got {pos_j}"

        # Box must have moved again (arm moved at new base position)
        final_pos = np.array(box.get_pose().position)
        assert not np.allclose(after_base_move_pos, final_pos, atol=0.01), (
            f"Box must move during joint movement: " f"after_base={list(after_base_move_pos)}, after_joint={list(final_pos)}"
        )

        # Verify final box position matches FK at new base + new joint config
        link_state = p.getLinkState(agent.body_id, ee_link_index, computeForwardKinematics=1, physicsClientId=client_id)
        expected_pos, expected_orn = p.multiplyTransforms(link_state[0], link_state[1], rel_pos, rel_orn)
        assert np.allclose(final_pos, expected_pos, atol=1e-4), (
            f"Final box position {list(final_pos)} should match " f"link{ee_link_index} + offset = {list(expected_pos)}"
        )


# ============================================================================
# Velocity / acceleration capping tests
# ============================================================================


class TestAgentVelocityCapping:
    """Test that velocity and acceleration never exceed configured max during motion.

    The TPI trajectory planner should respect max_linear_vel and max_linear_accel.
    We verify this by sampling velocity during the entire motion and checking all
    values stay within bounds.
    """

    def _create_agent_with_simcore(
        self,
        pose,
        *,
        max_linear_vel=1.0,
        max_linear_accel=2.0,
        max_angular_vel=1.5,
        max_angular_accel=3.0,
        motion_mode=MotionMode.OMNIDIRECTIONAL,
    ):
        from pybullet_fleet.controller_params import ControllerParams

        sim_core = MockSimCore()
        agent = Agent.from_mesh(
            visual_shape=ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=pose,
            mass=0.0,
            controller=ControllerParams(
                max_linear_vel=max_linear_vel,
                max_linear_accel=max_linear_accel,
                max_angular_vel=max_angular_vel,
                max_angular_accel=max_angular_accel,
            ),
            motion_mode=motion_mode,
            sim_core=sim_core,
        )
        return agent, sim_core

    def test_omni_velocity_capped_scalar(self, pybullet_env):
        """Velocity magnitude per axis should never exceed max_linear_vel."""
        max_vel = 1.0
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            max_linear_vel=max_vel,
        )
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0))

        dt = sim_core._dt
        max_observed_vel = 0.0
        for _ in range(3000):
            sim_core.tick()
            agent.update(dt)
            vel = agent.velocity
            for axis in range(3):
                max_observed_vel = max(max_observed_vel, abs(vel[axis]))
            if not agent.is_moving:
                break

        # Allow small tolerance for floating point
        assert (
            max_observed_vel <= max_vel + 0.01
        ), f"Velocity axis max {max_observed_vel:.4f} exceeded max_linear_vel={max_vel}"

    def test_omni_velocity_capped_per_axis(self, pybullet_env):
        """Per-axis velocity limits should each be respected."""
        limits = [0.5, 1.0, 0.3]
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            max_linear_vel=limits,
        )
        agent.set_goal_pose(Pose.from_xyz(5, 5, 2))

        dt = sim_core._dt
        max_per_axis = [0.0, 0.0, 0.0]
        for _ in range(5000):
            sim_core.tick()
            agent.update(dt)
            vel = agent.velocity
            for axis in range(3):
                max_per_axis[axis] = float(max(max_per_axis[axis], abs(vel[axis])))
            if not agent.is_moving:
                break

        for axis in range(3):
            assert (
                max_per_axis[axis] <= limits[axis] + 0.01
            ), f"Axis {axis}: max vel {max_per_axis[axis]:.4f} exceeded limit {limits[axis]}"

    def test_omni_acceleration_capped(self, pybullet_env):
        """Acceleration magnitude per axis should never exceed max_linear_accel.

        We compute approximate acceleration as Δv / dt between consecutive updates.
        """
        max_accel = 2.0
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            max_linear_accel=max_accel,
        )
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0))

        dt = sim_core._dt
        prev_vel = agent.velocity.copy()
        max_observed_accel = 0.0

        for _ in range(3000):
            sim_core.tick()
            agent.update(dt)
            cur_vel = agent.velocity
            accel = np.abs((cur_vel - prev_vel) / dt) if dt > 0 else np.zeros(3)
            for axis in range(3):
                max_observed_accel = max(max_observed_accel, accel[axis])
            prev_vel = cur_vel.copy()
            if not agent.is_moving:
                break

        # Larger tolerance because discrete approximation of acceleration can overshoot slightly
        assert (
            max_observed_accel <= max_accel + 0.5
        ), f"Acceleration max {max_observed_accel:.4f} exceeded max_linear_accel={max_accel}"

    def test_diff_angular_velocity_capped(self, pybullet_env):
        """Differential drive angular velocity should not exceed max_angular_vel."""
        max_ang_vel = 1.5
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            max_angular_vel=max_ang_vel,
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        # Goal at 90° requires rotation
        agent.set_goal_pose(Pose.from_xyz(0, 3, 0))

        dt = sim_core._dt
        max_observed_ang_vel = 0.0
        for _ in range(5000):
            sim_core.tick()
            agent.update(dt)
            max_observed_ang_vel = max(max_observed_ang_vel, abs(agent.angular_velocity))
            if not agent.is_moving:
                break

        assert (
            max_observed_ang_vel <= max_ang_vel + 0.01
        ), f"Angular velocity max {max_observed_ang_vel:.4f} exceeded max_angular_vel={max_ang_vel}"

    def test_low_velocity_limit_still_reaches_goal(self, pybullet_env):
        """Even with very low velocity limits, agent should still reach goal."""
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            max_linear_vel=0.2,
            max_linear_accel=0.5,
        )
        agent.set_goal_pose(Pose.from_xyz(0.5, 0, 0))

        dt = sim_core._dt
        for _ in range(10000):
            sim_core.tick()
            agent.update(dt)
            if not agent.is_moving:
                break

        assert not agent.is_moving, "Agent should eventually reach goal even with low velocity"
        pos = np.array(agent.get_pose().position)
        assert np.allclose(pos, [0.5, 0, 0], atol=0.05)

    def test_high_velocity_limit_respects_cap(self, pybullet_env):
        """With high max_linear_vel, velocity should still not exceed it."""
        max_vel = 10.0
        agent, sim_core = self._create_agent_with_simcore(
            Pose.from_xyz(0, 0, 0),
            max_linear_vel=max_vel,
            max_linear_accel=50.0,
        )
        agent.set_goal_pose(Pose.from_xyz(100, 0, 0))

        dt = sim_core._dt
        max_observed = 0.0
        for _ in range(3000):
            sim_core.tick()
            agent.update(dt)
            vel = agent.velocity
            max_observed = max(max_observed, np.linalg.norm(vel))
            if not agent.is_moving:
                break

        # Per-axis cap: each axis capped at 10, so norm ≤ 10 for single-axis motion
        assert max_observed <= max_vel + 0.1, f"Velocity norm {max_observed:.4f} exceeded max_linear_vel={max_vel}"


class TestMovableJointIndicesCache:
    """Agent caches movable joint indices to avoid per-solve recomputation."""

    def test_cached_movable_indices_matches_computed(self, pybullet_env):
        """_cached_movable_indices matches the list comprehension in _solve_ik."""
        agent = Agent.from_urdf(
            urdf_path="robots/arm_robot.urdf",
            use_fixed_base=True,
        )
        expected = [i for i, info in enumerate(agent.joint_info) if info[2] != p.JOINT_FIXED]
        assert hasattr(agent, "_cached_movable_indices"), "Agent should have _cached_movable_indices attribute"
        assert agent._cached_movable_indices == expected

    def test_mesh_agent_has_empty_cached_movable_indices(self, pybullet_env):
        """Non-URDF (mesh) agents should have empty cached movable indices."""
        agent = create_mesh_agent()
        assert agent._cached_movable_indices == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
