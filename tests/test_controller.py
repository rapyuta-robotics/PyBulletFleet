"""Tests for Controller hierarchy, KinematicController, and Agent.set_controller().

TDD tests for the Controller system (Plugin Architecture Phase 2).
OmniController is a kinematic omnidirectional 6-DoF velocity controller:
body-frame (vx, vy, vz, wx, wy, wz) → quaternion-based world-frame integration.
"""

import math

import numpy as np
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, Pose
from pybullet_fleet.controller import (
    Controller,
    KinematicController,
    OmniController,
    DifferentialController,
    create_controller,
    list_controllers,
    register_controller,
)
from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet.types import MotionMode, MovementDirection
from tests.conftest import MockSimCore


@pytest.fixture
def sim_core(pybullet_env):
    """MockSimCore wired to the headless pybullet_env fixture."""
    sc = MockSimCore(physics=False)
    sc._client = pybullet_env
    return sc


# Module-level custom controllers used by the ``controller={"class": ...}``
# dotted-path tests.  They have no ``_registry_name`` so they are reachable
# *only* via dotted import path, exercising the resolve_class branch.
class CustomKinematicController(OmniController):
    """A custom KinematicController loadable via dotted path (replaces base)."""


class CustomHighLevelController(Controller):
    """A custom non-kinematic Controller loadable via dotted path (chained)."""

    def compute(self, agent, dt: float) -> bool:
        return False


@pytest.fixture
def mobile_agent(sim_core):
    """Spawn a simple omnidirectional mobile agent."""
    params = AgentSpawnParams(
        urdf_path="robots/mobile_robot.urdf",
        initial_pose=Pose.from_xyz(0, 0, 0.05),
        motion_mode=MotionMode.OMNIDIRECTIONAL,
        controller={"max_linear_vel": 2.0},
    )
    return Agent.from_params(params, sim_core)


@pytest.fixture
def omni_agent(mobile_agent):
    """mobile_agent with OmniController pre-attached."""
    ctrl = OmniController()
    mobile_agent.set_controller(ctrl)
    return mobile_agent


@pytest.fixture
def diff_agent(mobile_agent):
    """mobile_agent with DifferentialController pre-attached."""
    ctrl = DifferentialController()
    mobile_agent.set_controller(ctrl)
    return mobile_agent


# -----------------------------------------------------------------------
# OmniController (unified: basic + body-frame + 6-DoF)
# -----------------------------------------------------------------------


class TestOmniController:
    """OmniController: kinematic omnidirectional 6-DoF velocity controller.

    Body-frame (vx, vy, vz, wx, wy, wz) → quaternion-based world-frame integration.
    All assertions verify exact target positions, not just "moved from initial."
    """

    # -- Linear motion (exact target verification) --

    def test_vx_reaches_target(self, omni_agent):
        """vx=1.0 for dt=0.1 → x = initial + 0.1 (exact kinematic integration)."""
        omni_agent._controller.set_velocity(vx=1.0)

        initial_x = omni_agent.get_pose().x
        dt = 0.1
        omni_agent.update(dt)

        assert omni_agent.get_pose().x == pytest.approx(initial_x + 1.0 * dt, abs=0.01)

    def test_vy_reaches_target(self, omni_agent):
        """vy=1.0 at yaw=0 for dt=0.1 → y = initial + 0.1."""
        omni_agent._controller.set_velocity(vy=1.0)

        initial_y = omni_agent.get_pose().y
        dt = 0.1
        omni_agent.update(dt)

        assert omni_agent.get_pose().y == pytest.approx(initial_y + 1.0 * dt, abs=0.01)

    def test_vz_reaches_target(self, omni_agent):
        """vz=0.5 for dt=0.1 → z = initial + 0.05."""
        omni_agent._controller.set_velocity(vz=0.5)

        initial_z = omni_agent.get_pose().z
        dt = 0.1
        omni_agent.update(dt)

        assert omni_agent.get_pose().z == pytest.approx(initial_z + 0.5 * dt, abs=0.01)

    # -- Angular motion (exact target verification) --

    def test_wz_reaches_target_yaw(self, omni_agent):
        """wz=1.0 for dt=0.1 → yaw ≈ initial + 0.1 rad, roll/pitch unchanged."""
        omni_agent._controller.set_velocity(wz=1.0)

        initial_yaw = omni_agent.get_pose().yaw
        dt = 0.1
        omni_agent.update(dt)

        pose = omni_agent.get_pose()
        assert pose.yaw == pytest.approx(initial_yaw + 1.0 * dt, abs=0.01)
        # Roll and pitch must remain ~0 (pure yaw rotation)
        assert abs(pose.roll) < 0.01
        assert abs(pose.pitch) < 0.01

    def test_wx_reaches_target_roll(self, omni_agent):
        """wx=1.0 for dt=0.1 → roll ≈ 0.1 rad."""
        omni_agent._controller.set_velocity(wx=1.0)

        dt = 0.1
        omni_agent.update(dt)

        assert omni_agent.get_pose().roll == pytest.approx(1.0 * dt, abs=0.01)

    def test_wy_reaches_target_pitch(self, omni_agent):
        """wy=1.0 for dt=0.1 → pitch ≈ 0.1 rad."""
        omni_agent._controller.set_velocity(wy=1.0)

        dt = 0.1
        omni_agent.update(dt)

        assert omni_agent.get_pose().pitch == pytest.approx(1.0 * dt, abs=0.01)

    # -- Body-frame rotation (quaternion body→world) --

    def test_vx_at_yaw_90_reaches_target_y(self, omni_agent):
        """vx=1 at yaw=π/2 → y += 0.1 (body forward = world +Y)."""
        omni_agent.set_pose(Pose.from_yaw(0, 0, 0.05, math.pi / 2))
        omni_agent._controller.set_velocity(vx=1.0)

        dt = 0.1
        omni_agent.update(dt)

        pose = omni_agent.get_pose()
        assert pose.y == pytest.approx(0.0 + 1.0 * dt, abs=0.02)
        assert abs(pose.x) < 0.02  # Negligible X movement

    def test_vy_at_yaw_90_reaches_target_neg_x(self, omni_agent):
        """vy=1 at yaw=π/2 → x ≈ -0.1 (body left = world -X)."""
        omni_agent.set_pose(Pose.from_yaw(0, 0, 0.05, math.pi / 2))
        omni_agent._controller.set_velocity(vy=1.0)

        dt = 0.1
        omni_agent.update(dt)

        pose = omni_agent.get_pose()
        assert pose.x == pytest.approx(-1.0 * dt, abs=0.02)

    def test_vx_at_pitch_90_reaches_target_neg_z(self, omni_agent):
        """vx=1 with pitch=π/2 → z -= 0.1 (body forward = world -Z, full quaternion rotation)."""
        omni_agent.set_pose(Pose.from_euler(0, 0, 0.5, roll=0, pitch=math.pi / 2, yaw=0))
        omni_agent._controller.set_velocity(vx=1.0)

        initial_z = omni_agent.get_pose().z
        dt = 0.1
        omni_agent.update(dt)

        pose = omni_agent.get_pose()
        assert pose.z == pytest.approx(initial_z - 1.0 * dt, abs=0.02)
        assert abs(pose.x) < 0.02  # Negligible X movement

    def test_vz_at_pitch_90_reaches_target_x(self, omni_agent):
        """vz=1 with pitch=π/2 → x += 0.1 (body up = world +X)."""
        omni_agent.set_pose(Pose.from_euler(0, 0, 0.5, roll=0, pitch=math.pi / 2, yaw=0))
        omni_agent._controller.set_velocity(vz=1.0)

        initial_x = omni_agent.get_pose().x
        dt = 0.1
        omni_agent.update(dt)

        pose = omni_agent.get_pose()
        assert pose.x == pytest.approx(initial_x + 1.0 * dt, abs=0.02)

    # -- Combined angular (quaternion integration) --

    def test_simultaneous_wx_wy_uses_quaternion(self, omni_agent):
        """Simultaneous wx + wy uses quaternion multiplication (no gimbal lock)."""
        omni_agent._controller.set_velocity(wx=1.0, wy=1.0)

        dt = 0.1
        omni_agent.update(dt)

        pose = omni_agent.get_pose()
        # Both roll and pitch should have changed (quaternion integration)
        assert abs(pose.roll) == pytest.approx(0.1, abs=0.02)
        assert abs(pose.pitch) == pytest.approx(0.1, abs=0.02)

    # -- Orientation preservation --

    def test_linear_only_preserves_orientation(self, omni_agent):
        """Linear velocity (wx=wy=wz=0) preserves current orientation exactly."""
        # Set a non-trivial orientation
        omni_agent.set_pose(Pose.from_euler(0, 0, 0.05, roll=0.3, pitch=0.2, yaw=0.5))
        initial_quat = list(omni_agent.get_pose().orientation)

        omni_agent._controller.set_velocity(vx=1.0, vy=0.5, vz=0.2)  # No angular
        omni_agent.update(0.1)

        new_quat = omni_agent.get_pose().orientation
        np.testing.assert_allclose(new_quat, initial_quat, atol=1e-6)

    # -- State management --

    def test_zero_velocity_returns_false(self, omni_agent):
        """compute() returns False when velocity is zero."""
        result = omni_agent._controller.compute(omni_agent, 0.1)
        assert result is False

    def test_nonzero_velocity_returns_true(self, omni_agent):
        """compute() returns True when velocity is nonzero."""
        omni_agent._controller.set_velocity(vx=1.0)
        result = omni_agent._controller.compute(omni_agent, 0.1)
        assert result is True

    def test_stop_clears_all_6dof(self, omni_agent):
        """agent.stop() zeros all 6 DoF velocities via on_stop()."""
        omni_agent._controller.set_velocity(vx=1.0, vy=0.5, vz=0.2, wx=0.1, wy=0.2, wz=0.3)

        omni_agent.stop()
        # All zeros → compute returns False
        result = omni_agent._controller.compute(omni_agent, 0.1)
        assert result is False

    def test_reported_velocity_matches_command(self, omni_agent):
        """After compute(), agent.velocity reflects the commanded world-frame velocity."""
        omni_agent._controller.set_velocity(vx=1.5, vy=-0.5, vz=0.0)

        omni_agent.update(0.1)

        assert omni_agent.velocity[0] == pytest.approx(1.5, abs=0.01)
        assert omni_agent.velocity[1] == pytest.approx(-0.5, abs=0.01)

    # -- Velocity limits (magnitude clamping) --

    def test_linear_speed_clamped(self, mobile_agent):
        """Linear velocity magnitude clamped to max_linear_vel, direction preserved."""
        ctrl = OmniController(ControllerParams(max_linear_vel=1.0))
        mobile_agent.set_controller(ctrl)

        ctrl.set_velocity(vx=3.0, vy=4.0)  # magnitude=5.0, clamped to 1.0
        # Direction preserved: (3/5, 4/5) * 1.0
        np.testing.assert_allclose(ctrl._linear_velocity, [0.6, 0.8, 0.0], atol=1e-9)

    def test_angular_speed_clamped(self, mobile_agent):
        """Angular velocity magnitude clamped to max_angular_vel, direction preserved."""
        ctrl = OmniController(ControllerParams(max_angular_vel=1.0))
        mobile_agent.set_controller(ctrl)

        ctrl.set_velocity(wx=0.0, wy=0.0, wz=2.0)  # magnitude=2.0, clamped to 1.0
        np.testing.assert_allclose(ctrl._angular_velocity, [0.0, 0.0, 1.0], atol=1e-9)

    def test_below_limit_not_clamped(self, mobile_agent):
        """Velocity below limit passes through unchanged."""
        ctrl = OmniController(ControllerParams(max_linear_vel=10.0, max_angular_vel=5.0))
        mobile_agent.set_controller(ctrl)

        ctrl.set_velocity(vx=1.0, vy=2.0, wz=1.5)
        np.testing.assert_allclose(ctrl._linear_velocity, [1.0, 2.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(ctrl._angular_velocity, [0.0, 0.0, 1.5], atol=1e-9)

    def test_default_no_limit(self, omni_agent):
        """Explicit ControllerParams(inf) disables clamping."""
        omni_agent.set_controller(OmniController(ControllerParams(max_linear_vel=math.inf, max_angular_vel=math.inf)))
        omni_agent._controller.set_velocity(vx=100.0, vy=200.0, wz=50.0)
        np.testing.assert_allclose(omni_agent._controller._linear_velocity, [100.0, 200.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(omni_agent._controller._angular_velocity, [0.0, 0.0, 50.0], atol=1e-9)

    def test_from_config_with_limits(self):
        """from_config parses max_linear_vel / max_angular_vel."""
        ctrl = OmniController.from_config({"max_linear_vel": 2.0, "max_angular_vel": 1.0})
        assert ctrl.params.max_linear_vel == pytest.approx(2.0)  # type: ignore[reportAttributeAccessIssue]
        assert ctrl.params.max_angular_vel == pytest.approx(1.0)  # type: ignore[reportAttributeAccessIssue]


# -----------------------------------------------------------------------
# Agent.set_controller()
# -----------------------------------------------------------------------


class TestSetController:
    """Agent.set_controller() switches control strategy."""

    def test_set_controller_stores_reference(self, mobile_agent):
        """set_controller(ctrl) stores the controller on the agent."""
        ctrl = OmniController()
        mobile_agent.set_controller(ctrl)
        assert mobile_agent._controller is ctrl

    def test_set_controller_none_disables_movement(self, mobile_agent):
        """set_controller(None) disables all movement."""
        ctrl = OmniController()
        mobile_agent.set_controller(ctrl)
        mobile_agent.set_controller(None)
        assert mobile_agent._controller is None

    def test_goal_based_works_via_controller(self, mobile_agent, sim_core):
        """Goal-based movement works through the auto-assigned controller."""
        mobile_agent.set_goal_pose(Pose.from_xyz(5, 0, 0.05))
        initial_x = mobile_agent.get_pose().x

        # TPI movement depends on sim_core.sim_time — must tick sim_core
        dt = sim_core._dt
        for _ in range(100):
            sim_core.tick()
            mobile_agent.update(dt)

        # Should move toward goal via controller's TPI
        assert mobile_agent.get_pose().x > initial_x

    def test_controller_takes_priority_over_goal(self, mobile_agent):
        """When controller is set, it takes precedence over goal-based movement."""
        # Set a goal pointing in +X
        mobile_agent.set_goal_pose(Pose.from_xyz(10, 0, 0.05))
        # But controller commands -X velocity
        ctrl = OmniController()
        mobile_agent.set_controller(ctrl)
        ctrl.set_velocity(vx=-1.0, vy=0.0)

        initial_x = mobile_agent.get_pose().x
        mobile_agent.update(0.1)

        # Controller wins: robot moved in -X
        assert mobile_agent.get_pose().x < initial_x


# -----------------------------------------------------------------------
# Controller-instance form of ``controller=`` argument
# -----------------------------------------------------------------------


class TestControllerInstanceArg:
    """``controller=<Controller instance>`` installs it verbatim."""

    def test_preset_instance_installed_as_is(self, sim_core):
        """A pre-built controller is installed without copy or reconstruction."""
        preset_params = ControllerParams(max_linear_vel=4.2, max_angular_vel=1.7)
        preset = DifferentialController(params=preset_params, wheel_separation=0.42)

        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.OMNIDIRECTIONAL,  # intentionally mismatched
            controller=preset,
        )
        agent = Agent.from_params(params, sim_core)

        # Identity: exact same instance, not a copy.
        assert agent._controller is preset
        # controller_params is taken from the preset's params (also identity).
        assert agent.controller_params is preset_params
        # Subclass-specific kwargs survive.
        assert agent._controller._wheel_separation == pytest.approx(0.42)

    def test_preset_instance_overrides_motion_mode(self, sim_core):
        """When a Controller instance is passed, motion_mode does not pick a default."""
        preset = DifferentialController(params=ControllerParams())
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            controller=preset,
        )
        agent = Agent.from_params(params, sim_core)
        # The omnidirectional motion_mode did NOT install an OmniController.
        assert isinstance(agent._controller, DifferentialController)


# -----------------------------------------------------------------------
# Controller ABC contract
# -----------------------------------------------------------------------


class TestControllerABC:
    """Controller ABC enforces the compute() contract."""

    def test_cannot_instantiate_abstract(self):
        """Controller ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Controller()  # type: ignore[reportAbstractUsage]

    def test_velocity_controller_cannot_instantiate(self):
        """KinematicController is abstract (has abstract template hooks)."""
        with pytest.raises(TypeError):
            KinematicController()  # type: ignore[reportAbstractUsage]


# -----------------------------------------------------------------------
# Controller Registry
# -----------------------------------------------------------------------


class TestControllerRegistry:
    """Controller registry allows name-based creation."""

    def test_builtins_registered(self):
        """Built-in controllers are auto-registered."""
        registry = list_controllers()
        assert "omni" in registry
        assert "differential" in registry

    def test_create_velocity(self):
        """Subclass imports must not replace the built-in omni controller."""
        ctrl = create_controller("omni")
        assert type(ctrl) is OmniController

    def test_create_differential(self):
        """create_controller('differential_velocity') returns DifferentialController."""
        ctrl = create_controller("differential")
        assert isinstance(ctrl, DifferentialController)

    def test_create_with_config(self):
        """create_controller passes config to from_config()."""
        ctrl = create_controller("differential", {"max_linear_vel": 3.0, "max_angular_vel": 2.0})
        assert isinstance(ctrl, DifferentialController)
        assert ctrl.params.max_linear_vel == pytest.approx(3.0)
        assert ctrl.params.max_angular_vel == pytest.approx(2.0)

    def test_unknown_name_raises(self):
        """create_controller with unknown name raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            create_controller("nonexistent")

    def test_register_custom(self):
        """register_controller adds a custom controller."""

        class MyController(Controller):
            def compute(self, agent, dt):
                return False

        register_controller("my_custom", MyController)
        assert "my_custom" in list_controllers()
        ctrl = create_controller("my_custom")
        assert isinstance(ctrl, MyController)

        # Cleanup
        from pybullet_fleet.controller import _registry

        _registry.unregister("my_custom")


# -----------------------------------------------------------------------
# DifferentialController
# -----------------------------------------------------------------------


class TestDifferentialController:
    """DifferentialController uses unicycle kinematics."""

    def test_forward_only(self, diff_agent):
        """vx=1 at yaw=0 moves agent in +X only."""
        diff_agent._controller.set_velocity(vx=1.0)

        initial_x = diff_agent.get_pose().x
        initial_y = diff_agent.get_pose().y
        dt = 0.1
        diff_agent.update(dt)

        pose = diff_agent.get_pose()
        assert pose.x == pytest.approx(initial_x + 1.0 * dt, abs=0.01)
        assert pose.y == pytest.approx(initial_y, abs=0.01)

    def test_lateral_ignored(self, diff_agent):
        """vy is ignored — differential drive cannot strafe."""
        diff_agent._controller.set_velocity(vx=0.0, vy=5.0)  # vy should be ignored

        initial_x = diff_agent.get_pose().x
        initial_y = diff_agent.get_pose().y
        dt = 0.1
        diff_agent.update(dt)

        pose = diff_agent.get_pose()
        # Should not have moved at all (only vx matters, and it's 0)
        assert pose.x == pytest.approx(initial_x, abs=0.01)
        assert pose.y == pytest.approx(initial_y, abs=0.01)

    def test_rotation_only(self, diff_agent):
        """wz rotates in place to target yaw, no translation."""
        diff_agent._controller.set_velocity(vx=0.0, wz=1.0)

        initial_x = diff_agent.get_pose().x
        initial_y = diff_agent.get_pose().y
        initial_yaw = diff_agent.get_pose().yaw
        dt = 0.1
        diff_agent.update(dt)

        pose = diff_agent.get_pose()
        # Rotated to target yaw but didn't translate
        assert pose.x == pytest.approx(initial_x, abs=0.01)
        assert pose.y == pytest.approx(initial_y, abs=0.01)
        assert pose.yaw == pytest.approx(initial_yaw + 1.0 * dt, abs=0.01)

    def test_combined_arc(self, diff_agent):
        """vx + wz produces arc motion (non-zero lateral displacement proves arc, not straight line).

        Unicycle: R = v/wz = 1.0/0.5 = 2.0m.
        After 10×0.1s Euler integration: x ≈ 0.96, y ≈ 0.22, yaw ≈ 0.5 rad.
        """
        diff_agent._controller.set_velocity(vx=1.0, wz=0.5)

        initial_x = diff_agent.get_pose().x
        initial_y = diff_agent.get_pose().y
        dt = 0.1
        for _ in range(10):
            diff_agent.update(dt)

        pose = diff_agent.get_pose()
        # Y displacement > 0 proves arc, not straight line
        assert pose.y - initial_y == pytest.approx(0.22, abs=0.05)
        # Forward displacement
        assert pose.x - initial_x == pytest.approx(0.96, abs=0.05)
        # Yaw = wz * total_time = 0.5 * 1.0
        assert pose.yaw == pytest.approx(0.5, abs=0.02)

    def test_heading_affects_direction(self, diff_agent):
        """vx=1 at yaw=π/2 → moves in +Y to target y ≈ 0.1, x ≈ 0."""
        # Set initial heading to π/2
        diff_agent.set_pose(Pose.from_yaw(0, 0, 0.05, math.pi / 2))

        diff_agent._controller.set_velocity(vx=1.0)
        dt = 0.1
        diff_agent.update(dt)

        pose = diff_agent.get_pose()
        # At yaw=π/2, forward vx=1 → world +Y
        assert pose.y == pytest.approx(1.0 * dt, abs=0.02)
        assert abs(pose.x) < 0.02

    def test_speed_limit_symmetric(self, mobile_agent):
        """Speed exceeding max is clamped symmetrically (both forward and backward).

        Robots go backward, so limits are [-max, max], not [0, max].
        """
        ctrl = DifferentialController(ControllerParams(max_linear_vel=1.0, max_angular_vel=0.5))
        mobile_agent.set_controller(ctrl)

        # Positive overshoot
        ctrl.set_velocity(vx=10.0, wz=5.0)
        assert ctrl._linear_velocity[0] == pytest.approx(1.0)
        assert ctrl._angular_velocity[2] == pytest.approx(0.5)

        # Negative overshoot (backward / reverse rotation)
        ctrl.set_velocity(vx=-10.0, wz=-5.0)
        assert ctrl._linear_velocity[0] == pytest.approx(-1.0)
        assert ctrl._angular_velocity[2] == pytest.approx(-0.5)

    def test_backward_motion(self, diff_agent):
        """Negative vx moves robot backward (opposite to heading)."""
        diff_agent._controller.set_velocity(vx=-1.0)

        initial_x = diff_agent.get_pose().x
        dt = 0.1
        diff_agent.update(dt)

        pose = diff_agent.get_pose()
        # At yaw=0, backward → -X
        assert pose.x == pytest.approx(initial_x - 1.0 * dt, abs=0.01)

    def test_zero_returns_false(self, diff_agent):
        """compute() returns False for zero velocity."""
        result = diff_agent._controller.compute(diff_agent, 0.1)
        assert result is False

    def test_nonzero_returns_true(self, diff_agent):
        """compute() returns True for nonzero velocity."""
        diff_agent._controller.set_velocity(vx=1.0)
        result = diff_agent._controller.compute(diff_agent, 0.1)
        assert result is True

    def test_stop_clears_velocity(self, diff_agent):
        """on_stop() zeros velocity state."""
        diff_agent._controller.set_velocity(vx=1.0, wz=0.5)
        diff_agent._controller.on_stop(diff_agent)
        np.testing.assert_allclose(diff_agent._controller._linear_velocity, 0.0)
        np.testing.assert_allclose(diff_agent._controller._angular_velocity, 0.0)

    def test_from_config(self):
        """from_config parses params correctly."""
        ctrl = DifferentialController.from_config({"max_linear_vel": 5.0, "max_angular_vel": 3.0, "wheel_separation": 0.4})
        assert ctrl.params.max_linear_vel == pytest.approx(5.0)  # type: ignore[reportAttributeAccessIssue]
        assert ctrl.params.max_angular_vel == pytest.approx(3.0)  # type: ignore[reportAttributeAccessIssue]
        assert ctrl._wheel_separation == pytest.approx(0.4)  # type: ignore[reportAttributeAccessIssue]

    def test_from_config_defaults(self):
        """from_config with empty dict leaves params as None; _eff_* helpers resolve to framework defaults."""
        from pybullet_fleet._defaults import CONTROLLER as _CTRL_D

        ctrl = DifferentialController.from_config({})
        assert ctrl.params.max_linear_vel is None
        assert ctrl.params.scalar_max_linear_vel() == _CTRL_D["max_linear_vel"]
        assert ctrl.params.scalar_max_angular_vel() == _CTRL_D["max_angular_vel"]


# -----------------------------------------------------------------------
# MovementDirection.AUTO — auto-detect forward/backward
# -----------------------------------------------------------------------


class TestAutoDirection:
    """AUTO direction: choose FORWARD/BACKWARD based on yaw delta to goal."""

    @staticmethod
    def _run_path(agent, sim_core, steps=2000):
        dt = sim_core._dt
        for _ in range(steps):
            sim_core.sim_time += dt
            agent.update(dt)

    def test_default_direction_is_none(self):
        """set_path direction default is None (uses controller's default_direction)."""
        import inspect

        ctrl = DifferentialController()
        sig = inspect.signature(ctrl.set_path)
        default = sig.parameters["direction"].default
        assert default is None, f"Default direction should be None, got {default}"

    def test_controller_default_direction_is_forward(self):
        """DifferentialController.default_direction defaults to FORWARD."""
        from pybullet_fleet.types import MovementDirection

        ctrl = DifferentialController()
        assert ctrl.default_direction == MovementDirection.FORWARD

    def test_controller_default_direction_auto(self):
        """DifferentialController accepts default_direction=AUTO via constructor."""
        from pybullet_fleet.types import MovementDirection

        ctrl = DifferentialController(ControllerParams(default_direction=MovementDirection.AUTO))
        assert ctrl.default_direction == MovementDirection.AUTO

    def test_set_path_none_uses_controller_default(self, sim_core):
        """set_path(direction=None) uses controller.default_direction."""
        from pybullet_fleet.types import MovementDirection

        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController(ControllerParams(default_direction=MovementDirection.AUTO))
        agent.set_controller(ctrl)

        # Goal behind — AUTO should resolve to BACKWARD
        goal = Pose.from_xyz(-5.0, 0.0, 0.05)
        ctrl.set_path(agent, [goal])  # direction=None → uses AUTO

        assert ctrl._movement_direction == MovementDirection.BACKWARD

    def test_set_path_explicit_overrides_default(self, sim_core):
        """Explicit direction=FORWARD overrides controller default_direction=AUTO."""
        from pybullet_fleet.types import MovementDirection

        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController(ControllerParams(default_direction=MovementDirection.AUTO))
        agent.set_controller(ctrl)

        # Goal behind — but explicit FORWARD should override AUTO
        goal = Pose.from_xyz(-5.0, 0.0, 0.05)
        ctrl.set_path(agent, [goal], direction=MovementDirection.FORWARD)

        assert ctrl._movement_direction == MovementDirection.FORWARD

    def test_agent_set_path_inherits_controller_default(self, sim_core):
        """Agent.set_path() without direction uses controller's default_direction."""
        from pybullet_fleet.types import MovementDirection

        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController(ControllerParams(default_direction=MovementDirection.AUTO))
        agent.set_controller(ctrl)

        # Goal behind — AUTO should resolve to BACKWARD via agent.set_path
        # Use auto_approach=False to avoid approach waypoint insertion
        goal = Pose.from_xyz(-5.0, 0.0, 0.05)
        agent.set_path([goal], auto_approach=False)

        assert ctrl._movement_direction == MovementDirection.BACKWARD

    def test_movement_direction_from_config(self, sim_core):
        """movement_direction in YAML config sets controller default_direction."""
        from pybullet_fleet.types import MovementDirection

        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),
            motion_mode=MotionMode.DIFFERENTIAL,
            controller={"type": "differential", "default_direction": "auto"},
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = agent._controllers[0]

        assert ctrl.default_direction == MovementDirection.AUTO

    def test_controller_config_from_dict(self, sim_core):
        """controller in from_dict sets controller default_direction."""
        from pybullet_fleet.types import MovementDirection

        params = AgentSpawnParams.from_dict(
            {
                "name": "test_robot",
                "urdf_path": "robots/mobile_robot.urdf",
                "motion_mode": "differential",
                "controller": {"type": "differential", "default_direction": "auto"},
            }
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = agent._controllers[0]

        assert ctrl.default_direction == MovementDirection.AUTO

    def test_auto_enum_value(self):
        """MovementDirection.AUTO is a valid enum value."""
        from pybullet_fleet.types import MovementDirection

        d = MovementDirection("auto")
        assert d is MovementDirection.AUTO

    def test_auto_forward_when_goal_ahead(self, sim_core):
        """AUTO resolves to FORWARD when goal is ahead of robot (|delta_yaw| < pi/2)."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),  # Facing +X
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController()
        agent.set_controller(ctrl)

        # Goal ahead (+X direction) — should choose FORWARD
        goal = Pose.from_xyz(5.0, 0.0, 0.05)
        ctrl.set_path(agent, [goal], direction=MovementDirection.AUTO)

        # After init, movement direction should be FORWARD
        assert ctrl._movement_direction == MovementDirection.FORWARD

    def test_auto_backward_when_goal_behind(self, sim_core):
        """AUTO resolves to BACKWARD when goal is behind robot (|delta_yaw| > pi/2)."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),  # Facing +X
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController()
        agent.set_controller(ctrl)

        # Goal behind (-X direction) — should choose BACKWARD
        goal = Pose.from_xyz(-5.0, 0.0, 0.05)
        ctrl.set_path(agent, [goal], direction=MovementDirection.AUTO)

        assert ctrl._movement_direction == MovementDirection.BACKWARD

    def test_auto_forward_diagonal(self, sim_core):
        """AUTO picks FORWARD for diagonal goal within ±90° of heading."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),  # Facing +X
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController()
        agent.set_controller(ctrl)

        # Goal at 45° — ahead, should be FORWARD
        goal = Pose.from_xyz(5.0, 5.0, 0.05)
        ctrl.set_path(agent, [goal], direction=MovementDirection.AUTO)

        assert ctrl._movement_direction == MovementDirection.FORWARD

    def test_auto_backward_diagonal(self, sim_core):
        """AUTO picks BACKWARD for diagonal goal beyond ±90° of heading."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),  # Facing +X
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController()
        agent.set_controller(ctrl)

        # Goal at 135° — behind, should be BACKWARD
        goal = Pose.from_xyz(-5.0, 5.0, 0.05)
        ctrl.set_path(agent, [goal], direction=MovementDirection.AUTO)

        assert ctrl._movement_direction == MovementDirection.BACKWARD

    def test_auto_reaches_goal_ahead(self, sim_core):
        """AUTO + goal ahead: robot reaches goal (FORWARD path)."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),
            motion_mode=MotionMode.DIFFERENTIAL,
            controller={"max_linear_vel": 2.0},
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController()
        agent.set_controller(ctrl)

        goal = Pose.from_xyz(3.0, 0.0, 0.05)
        ctrl.set_path(agent, [goal], direction=MovementDirection.AUTO)

        self._run_path(agent, sim_core, steps=3000)

        final = agent.get_pose()
        assert final.x == pytest.approx(3.0, abs=0.1)

    def test_auto_reaches_goal_behind(self, sim_core):
        """AUTO + goal behind: robot reaches goal via BACKWARD motion."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),
            motion_mode=MotionMode.DIFFERENTIAL,
            controller={"max_linear_vel": 2.0},
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController()
        agent.set_controller(ctrl)

        goal = Pose.from_xyz(-3.0, 0.0, 0.05)
        ctrl.set_path(agent, [goal], direction=MovementDirection.AUTO)

        self._run_path(agent, sim_core, steps=3000)

        final = agent.get_pose()
        assert final.x == pytest.approx(-3.0, abs=0.1)

    def test_auto_string_accepted(self):
        """'auto' string is accepted and converted to MovementDirection.AUTO."""
        from pybullet_fleet.types import MovementDirection

        d = MovementDirection("auto")
        assert d is MovementDirection.AUTO

    def test_auto_reevaluates_per_waypoint(self, sim_core):
        """AUTO re-evaluates direction for each waypoint, not just the first."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_yaw(0, 0, 0.05, 0.0),  # Facing +X
            motion_mode=MotionMode.DIFFERENTIAL,
            controller={"max_linear_vel": 2.0},
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController()
        agent.set_controller(ctrl)

        # Path: forward (+X), then back to origin — second waypoint should go BACKWARD
        goal1 = Pose.from_xyz(3.0, 0.0, 0.05)
        goal2 = Pose.from_xyz(0.0, 0.0, 0.05)
        ctrl.set_path(agent, [goal1, goal2], direction=MovementDirection.AUTO)

        # _original_direction should be preserved as AUTO
        assert ctrl._original_direction == MovementDirection.AUTO
        # First waypoint resolves to FORWARD
        assert ctrl._movement_direction == MovementDirection.FORWARD

        # Run until first waypoint is reached and second starts
        self._run_path(agent, sim_core, steps=5000)

        final = agent.get_pose()
        assert final.x == pytest.approx(0.0, abs=0.3)


# -----------------------------------------------------------------------
# AgentSpawnParams.controller
# -----------------------------------------------------------------------


class TestSpawnParamsControllerConfig:
    """AgentSpawnParams.controller auto-creates controller."""

    def test_no_config_default_controller(self, sim_core):
        """Default: no controller → DifferentialController auto-assigned."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
        )
        agent = Agent.from_params(params, sim_core)
        assert isinstance(agent._controller, DifferentialController)

    def test_velocity_config(self, sim_core):
        """controller with type=omni creates OmniController."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            controller={"type": "omni"},
        )
        agent = Agent.from_params(params, sim_core)
        assert isinstance(agent._controller, OmniController)

    def test_differential_config(self, sim_core):
        """controller with type=differential creates DifferentialController."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            controller={"type": "differential", "max_linear_vel": 1.5},
        )
        agent = Agent.from_params(params, sim_core)
        assert isinstance(agent._controller, DifferentialController)
        assert agent._controller.params.max_linear_vel == pytest.approx(1.5)


# -----------------------------------------------------------------------
# navigation_2d flag
# -----------------------------------------------------------------------


class TestNavigation2DFlag:
    """Tests for the navigation_2d flag on KinematicController.

    Default (True) preserves current Z during pose-mode navigation.
    When False, goal Z is used as-is for 3D movement.
    """

    @staticmethod
    def _run_path(agent, sim_core, steps=2000):
        """Tick sim_time + agent.update in lockstep."""
        dt = sim_core._dt
        for _ in range(steps):
            sim_core.sim_time += dt
            agent.update(dt)

    def test_default_is_2d(self):
        """ControllerParams.navigation_2d defaults to None (no explicit override)."""
        ctrl = OmniController()
        assert ctrl.params.navigation_2d is None

    def test_omni_2d_preserves_z(self, sim_core):
        """Omni with navigation_2d=True: Z stays at initial value (current behavior)."""
        spawn_z = 0.05
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, spawn_z),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = OmniController(ControllerParams(navigation_2d=True))
        agent.set_controller(ctrl)

        # Goal at z=5.0 — should be ignored in 2D mode
        goal = Pose.from_xyz(3.0, 4.0, 5.0)
        ctrl.set_path(agent, [goal], final_orientation_align=False)

        self._run_path(agent, sim_core)
        final_pose = agent.get_pose()

        assert final_pose.x == pytest.approx(3.0, abs=0.05)
        assert final_pose.y == pytest.approx(4.0, abs=0.05)
        assert final_pose.z == pytest.approx(spawn_z, abs=0.02), "Z should be preserved in 2D mode"

    def test_omni_3d_reaches_goal_z(self, sim_core):
        """Omni with navigation_2d=False: Z reaches goal value."""
        spawn_z = 0.05
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, spawn_z),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = OmniController(ControllerParams(navigation_2d=False))
        agent.set_controller(ctrl)

        goal_z = 5.0
        goal = Pose.from_xyz(3.0, 4.0, goal_z)
        ctrl.set_path(agent, [goal], final_orientation_align=False)

        self._run_path(agent, sim_core)
        final_pose = agent.get_pose()

        assert final_pose.x == pytest.approx(3.0, abs=0.05)
        assert final_pose.y == pytest.approx(4.0, abs=0.05)
        assert final_pose.z == pytest.approx(goal_z, abs=0.05), "Z should reach goal in 3D mode"

    def test_diff_3d_reaches_goal_z(self, sim_core):
        """Differential with navigation_2d=False: Z reaches goal value."""
        spawn_z = 0.05
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, spawn_z),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController(ControllerParams(navigation_2d=False))
        agent.set_controller(ctrl)

        goal_z = 3.0
        goal = Pose.from_xyz(5.0, 0.0, goal_z)
        ctrl.set_path(agent, [goal], final_orientation_align=False)

        self._run_path(agent, sim_core)
        final_pose = agent.get_pose()

        assert final_pose.x == pytest.approx(5.0, abs=0.1)
        assert final_pose.z == pytest.approx(goal_z, abs=0.1), "Z should reach goal in 3D mode"

    def test_diff_2d_preserves_z(self, sim_core):
        """Differential with navigation_2d=True: Z stays at initial value."""
        spawn_z = 0.05
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, spawn_z),
            motion_mode=MotionMode.DIFFERENTIAL,
        )
        agent = Agent.from_params(params, sim_core)
        ctrl = DifferentialController(ControllerParams(navigation_2d=True))
        agent.set_controller(ctrl)

        goal = Pose.from_xyz(5.0, 0.0, 10.0)
        ctrl.set_path(agent, [goal], final_orientation_align=False)

        self._run_path(agent, sim_core)
        final_pose = agent.get_pose()

        assert final_pose.x == pytest.approx(5.0, abs=0.1)
        assert final_pose.z == pytest.approx(spawn_z, abs=0.02), "Z should be preserved in 2D mode"

    def test_from_config_navigation_2d(self):
        """from_config passes navigation_2d to controller."""
        ctrl = OmniController.from_config({"navigation_2d": True})
        assert ctrl.params.navigation_2d is True  # type: ignore[reportAttributeAccessIssue]

        ctrl2 = DifferentialController.from_config({"navigation_2d": True})
        assert ctrl2.params.navigation_2d is True  # type: ignore[reportAttributeAccessIssue]

    def test_spawn_params_navigation_2d(self, sim_core):
        """AgentSpawnParams.controller navigation_2d propagates to default controller."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            controller={"navigation_2d": True},
        )
        agent = Agent.from_params(params, sim_core)
        assert agent._controller.params.navigation_2d is True


# -----------------------------------------------------------------------
# PatrolController
# -----------------------------------------------------------------------


class _FakeAgent:
    """Minimal mock agent for high-level controller tests.

    Only implements the interface that PatrolController / RandomWalkController
    call: ``is_moving``, ``set_goal_pose()``, ``get_pose()``.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.05):
        self._pose = Pose.from_xyz(x, y, z)
        self.is_moving = False
        self.goals: list = []  # recorded set_goal_pose calls

    def set_goal_pose(self, pose):
        self.goals.append(pose)

    def get_pose(self):
        return self._pose


class TestPatrolController:
    def test_empty_waypoints_returns_false(self):
        """No waypoints → compute returns False immediately."""
        from pybullet_fleet.controllers.patrol_controller import PatrolController

        ctrl = PatrolController(waypoints=[])
        agent = _FakeAgent()
        assert ctrl.compute(agent, 0.1) is False
        assert len(agent.goals) == 0

    def test_first_call_sets_first_waypoint(self):
        """First compute() sets the initial waypoint goal."""
        from pybullet_fleet.controllers.patrol_controller import PatrolController

        wps = [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
        ctrl = PatrolController(waypoints=wps)
        agent = _FakeAgent()

        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == 1
        assert agent.goals[0].x == pytest.approx(1.0)
        assert agent.goals[0].y == pytest.approx(2.0)

    def test_advances_to_next_waypoint_when_stopped(self):
        """When agent stops, controller advances to next waypoint."""
        from pybullet_fleet.controllers.patrol_controller import PatrolController

        wps = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ctrl = PatrolController(waypoints=wps)
        agent = _FakeAgent()

        # First call: sets wp[0]
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == 1

        # Agent still moving — no advance
        agent.is_moving = True
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == 1

        # Agent stopped — advance to wp[1]
        agent.is_moving = False
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == 2
        assert agent.goals[1].x == pytest.approx(2.0)

        # Agent stopped — advance to wp[2]
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == 3
        assert agent.goals[2].x == pytest.approx(3.0)

    def test_loop_wraps_around(self):
        """With loop=True, wraps back to first waypoint."""
        from pybullet_fleet.controllers.patrol_controller import PatrolController

        wps = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ctrl = PatrolController(waypoints=wps, loop=True)
        agent = _FakeAgent()

        # wp[0]
        ctrl.compute(agent, 0.1)
        # wp[1]
        ctrl.compute(agent, 0.1)
        # wrap → wp[0]
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == 3
        assert agent.goals[2].x == pytest.approx(1.0)

    def test_no_loop_stops_at_end(self):
        """With loop=False, returns False after last waypoint."""
        from pybullet_fleet.controllers.patrol_controller import PatrolController

        wps = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ctrl = PatrolController(waypoints=wps, loop=False)
        agent = _FakeAgent()

        ctrl.compute(agent, 0.1)  # wp[0]
        ctrl.compute(agent, 0.1)  # wp[1]
        result = ctrl.compute(agent, 0.1)  # past end
        assert result is False
        assert len(agent.goals) == 2  # no new goal set

    def test_wait_time_delays_advance(self):
        """wait_time prevents immediate advance to next waypoint."""
        from pybullet_fleet.controllers.patrol_controller import PatrolController

        wps = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ctrl = PatrolController(waypoints=wps, wait_time=1.0)
        agent = _FakeAgent()

        ctrl.compute(agent, 0.1)  # wp[0], agent stopped → advance sets wait
        ctrl.compute(agent, 0.1)  # wp[1] set, wait_timer = 1.0

        # Now waiting — should not advance even though agent is stopped
        goals_before = len(agent.goals)
        ctrl.compute(agent, 0.3)  # timer: 1.0 - 0.3 = 0.7 (still waiting)
        assert len(agent.goals) == goals_before

        # Drain the timer
        ctrl.compute(agent, 0.8)  # timer: 0.7 - 0.8 = -0.1 (expired)
        # Next call should advance
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) > goals_before


# -----------------------------------------------------------------------
# RandomWalkController
# -----------------------------------------------------------------------


class TestRandomWalkController:
    def test_captures_origin_on_first_call(self):
        """Origin is captured from agent's initial position."""
        from pybullet_fleet.controllers.random_walk_controller import RandomWalkController

        ctrl = RandomWalkController(radius=5.0)
        agent = _FakeAgent(x=10.0, y=20.0)

        assert ctrl._origin is None
        ctrl.compute(agent, 0.1)
        assert ctrl._origin == [10.0, 20.0]

    def test_target_within_radius(self):
        """Generated target is within the configured radius."""
        from pybullet_fleet.controllers.random_walk_controller import RandomWalkController

        ctrl = RandomWalkController(radius=3.0, wait_range=(0.0, 0.0))
        agent = _FakeAgent(x=5.0, y=5.0)

        # Run multiple iterations to check statistically
        for _ in range(20):
            agent.is_moving = False
            ctrl.compute(agent, 0.1)

        for goal in agent.goals:
            dx = goal.x - 5.0
            dy = goal.y - 5.0
            dist = math.sqrt(dx * dx + dy * dy)
            assert dist <= 3.0 + 0.01, f"Target at dist {dist:.2f} exceeds radius 3.0"

    def test_preserves_z(self):
        """Generated target preserves agent's Z coordinate."""
        from pybullet_fleet.controllers.random_walk_controller import RandomWalkController

        ctrl = RandomWalkController(radius=5.0, wait_range=(0.0, 0.0))
        agent = _FakeAgent(x=0.0, y=0.0, z=0.42)

        agent.is_moving = False
        ctrl.compute(agent, 0.1)

        assert len(agent.goals) >= 1
        assert agent.goals[0].z == pytest.approx(0.42)

    def test_wait_range_delays_next_target(self):
        """After setting a target, wait timer prevents immediate re-target."""
        from pybullet_fleet.controllers.random_walk_controller import RandomWalkController

        ctrl = RandomWalkController(radius=5.0, wait_range=(10.0, 10.0))
        agent = _FakeAgent()

        # First call: captures origin + sets target (agent not moving)
        ctrl.compute(agent, 0.1)
        n_goals = len(agent.goals)

        # Agent stops again but timer is ~10s — should not set new target
        agent.is_moving = False
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == n_goals, "Should not set new target while waiting"

    def test_does_not_move_while_agent_moving(self):
        """No new target while agent is still moving."""
        from pybullet_fleet.controllers.random_walk_controller import RandomWalkController

        ctrl = RandomWalkController(radius=5.0, wait_range=(0.0, 0.0))
        agent = _FakeAgent()

        ctrl.compute(agent, 0.1)  # first target
        n_goals = len(agent.goals)

        agent.is_moving = True
        ctrl.compute(agent, 0.1)
        assert len(agent.goals) == n_goals, "Should not set target while agent moving"


class TestControllerChainWiring:
    """Verify that from_params correctly wires high-level controllers
    (patrol, random_walk) as chain entries while keeping the base
    navigation controller at index 0."""

    def test_patrol_config_creates_chain(self, sim_core):
        """PatrolController from controller= should be appended,
        not replace the base controller."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.DIFFERENTIAL,
            controller={"type": "patrol", "waypoints": [[1, 0, 0], [2, 0, 0]], "loop": True},
        )
        agent = Agent.from_params(params, sim_core)

        from pybullet_fleet.controllers.patrol_controller import PatrolController

        assert len(agent._controllers) == 2, "Should have base + patrol"
        assert isinstance(agent._controllers[0], KinematicController), "Index 0 should be base controller"
        assert isinstance(agent._controllers[1], PatrolController), "Index 1 should be PatrolController"

    def test_random_walk_config_creates_chain(self, sim_core):
        """RandomWalkController from controller= should be appended."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            controller={"type": "random_walk", "radius": 5.0},
        )
        agent = Agent.from_params(params, sim_core)

        from pybullet_fleet.controllers.random_walk_controller import RandomWalkController

        assert len(agent._controllers) == 2
        assert isinstance(agent._controllers[0], OmniController)
        assert isinstance(agent._controllers[1], RandomWalkController)

    def test_kinematic_controller_replaces_base(self, sim_core):
        """DifferentialController from controller= should replace base."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            controller={"type": "differential", "default_direction": "auto"},
        )
        agent = Agent.from_params(params, sim_core)

        assert len(agent._controllers) == 1
        assert isinstance(agent._controllers[0], DifferentialController)

    def test_list_config_base_plus_high_level(self, sim_core):
        """A list of controller= entries should wire base + high-level."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.DIFFERENTIAL,
            controller=[
                {"type": "differential", "default_direction": "auto"},
                {"type": "patrol", "waypoints": [[1, 0, 0]]},
            ],
        )
        agent = Agent.from_params(params, sim_core)

        from pybullet_fleet.controllers.patrol_controller import PatrolController

        assert len(agent._controllers) == 2
        assert isinstance(agent._controllers[0], DifferentialController)
        assert isinstance(agent._controllers[1], PatrolController)

    def test_dotted_path_kinematic_replaces_base(self, sim_core):
        """controller={"class": "...KinematicController..."} replaces the base."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.DIFFERENTIAL,
            controller={
                "class": "tests.test_controller.CustomKinematicController",
                "max_linear_vel": 1.5,
            },
        )
        agent = Agent.from_params(params, sim_core)

        assert len(agent._controllers) == 1
        assert isinstance(agent._controllers[0], CustomKinematicController)
        # Inline config reaches controller_params (the kinematic-limit source).
        assert float(np.max(agent.max_linear_vel)) == pytest.approx(1.5)

    def test_dotted_path_high_level_creates_chain(self, sim_core):
        """A custom non-kinematic controller via class: is appended on the base."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            controller={"class": "tests.test_controller.CustomHighLevelController"},
        )
        agent = Agent.from_params(params, sim_core)

        assert len(agent._controllers) == 2
        assert isinstance(agent._controllers[0], OmniController)
        assert isinstance(agent._controllers[1], CustomHighLevelController)

    def test_patrol_chain_actually_moves_agent(self, sim_core):
        """End-to-end: patrol sets goal, base controller drives toward it."""
        params = AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.05),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            controller={"type": "patrol", "waypoints": [[1.0, 0, 0.05]], "loop": False, "max_linear_vel": 2.0},
        )
        agent = Agent.from_params(params, sim_core)

        # Use sim_core.tick() so sim_time advances (TPI is time-based)
        sim_core.tick(100)

        pos = agent.get_pose().position
        assert pos[0] > 0.1, f"Agent should have moved toward waypoint, x={pos[0]}"
