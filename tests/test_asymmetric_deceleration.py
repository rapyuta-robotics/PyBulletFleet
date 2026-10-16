"""Asymmetric accel/decel support in the TPI layer and the per-agent controllers.

``build_tpi`` used to pin ``dec_max`` to ``accel``, so a vehicle that brakes
harder than it accelerates could not be modelled even though
``TwoPointInterpolation`` has supported ``dec_max`` all along. These tests pin
the new behaviour and the guard that keeps the batched controllers — which
integrate the decel phase with the accel scalar — from silently producing
wrong positions.
"""

import pytest

from pybullet_fleet._tpi import build_tpi, extract_phase_params
from pybullet_fleet.controller_params import ControllerParams


class TestBuildTpiDecel:
    def test_decel_defaults_to_accel(self):
        """Omitting decel reproduces the symmetric profile of earlier releases."""
        tpi = build_tpi(p0=0.0, pe=5.0, vmax=2.0, accel=1.0, t0=0.0)
        assert tpi.amax_decel == pytest.approx(tpi.amax_accel)

    def test_decel_is_forwarded(self):
        tpi = build_tpi(p0=0.0, pe=5.0, vmax=2.0, accel=1.0, t0=0.0, decel=0.25)
        assert tpi.amax_accel == pytest.approx(1.0)
        assert tpi.amax_decel == pytest.approx(0.25)

    def test_softer_braking_takes_longer(self):
        """A weaker decel must stretch the trajectory, not leave it unchanged."""
        symmetric = build_tpi(p0=0.0, pe=5.0, vmax=2.0, accel=1.0, t0=0.0)
        soft_brake = build_tpi(p0=0.0, pe=5.0, vmax=2.0, accel=1.0, t0=0.0, decel=0.25)
        assert soft_brake.get_duration() > symmetric.get_duration()

    def test_endpoint_still_reached_exactly(self):
        tpi = build_tpi(p0=0.0, pe=5.0, vmax=2.0, accel=1.0, t0=0.0, decel=0.25)
        pos, vel, _ = tpi.get_point(tpi.get_end_time())
        assert pos == pytest.approx(5.0, abs=1e-6)
        assert vel == pytest.approx(0.0, abs=1e-6)

    def test_zero_distance_fallback_keeps_decel(self):
        """The degenerate p0 -> p0 fallback must not drop the decel argument."""
        tpi = build_tpi(p0=1.0, pe=1.0, vmax=2.0, accel=1.0, t0=0.0, decel=0.25)
        assert tpi.amax_decel == pytest.approx(0.25)


class TestExtractPhaseParams:
    def test_it_carries_both_scalars_out(self):
        # Far enough that vmax is reached and all three phases are populated.
        tpi = build_tpi(p0=0.0, pe=20.0, vmax=2.5, accel=1.5, t0=0.0, decel=0.5)
        t_accel, t_const, t_total, accel, decel = extract_phase_params(tpi)
        assert accel == pytest.approx(1.5)
        assert decel == pytest.approx(0.5)
        assert t_accel > 0.0 and t_const > 0.0
        assert t_total > t_accel + t_const

    def test_the_phase_durations_are_the_asymmetric_ones(self):
        """TwoPointInterpolation computed them from both scalars already, so
        a softer brake lengthens the final phase rather than needing any
        special handling here."""
        soft = extract_phase_params(build_tpi(p0=0.0, pe=20.0, vmax=2.5, accel=1.5, t0=0.0, decel=0.5))
        hard = extract_phase_params(build_tpi(p0=0.0, pe=20.0, vmax=2.5, accel=1.5, t0=0.0, decel=6.0))
        assert soft[2] > hard[2]
        # vmax is reached either way, so the ramp up is the same and only the
        # braking phase accounts for the difference.
        assert soft[0] == pytest.approx(hard[0])
        assert soft[2] - soft[0] - soft[1] > hard[2] - hard[0] - hard[1]

    def test_a_symmetric_profile_still_reports_equal_scalars(self):
        _, _, _, accel, decel = extract_phase_params(build_tpi(p0=0.0, pe=5.0, vmax=2.0, accel=1.0, t0=0.0))
        assert accel == pytest.approx(decel)


class TestCheckpointCaptureGuard:
    def test_capture_preserves_an_asymmetric_trajectory(self):
        """Straight navigation records the braking limit separately."""
        from pybullet_fleet import Pose
        from pybullet_fleet.controller import OmniController
        from pybullet_fleet.types import ControllerMode, PosePhase

        controller = OmniController(ControllerParams(max_linear_accel=1.0, max_linear_decel=0.25))
        # Put the controller in the one state capture supports, then give it a
        # trajectory whose braking differs from its acceleration.
        controller._mode = ControllerMode.POSE
        controller._pose_phase = PosePhase.FORWARD
        controller._path = [Pose.from_xyz(1.0, 0.0, 0.0)]
        controller._current_waypoint_index = 0
        controller._goal_pose = Pose.from_xyz(1.0, 0.0, 0.0)
        controller._forward_start_pos = __import__("numpy").zeros(3)
        controller._tpi_forward = build_tpi(p0=0.0, pe=1.0, vmax=2.0, accel=1.0, t0=0.0, decel=0.25)

        state = controller.capture_navigation_state()
        assert state["forward"]["decel"] == pytest.approx(0.25)
        restored = OmniController(controller.params)
        restored.restore_navigation_state(state, [0.0, 0.0, 0.0, 1.0])
        assert restored._tpi_forward.amax_decel == pytest.approx(0.25)


class TestControllerParamsDecel:
    def test_unset_decel_mirrors_accel(self):
        p = ControllerParams(max_linear_accel=3.0)
        assert p.max_linear_decel is None
        assert p._eff_linear_decel() == 3.0
        assert p.scalar_max_linear_decel() == pytest.approx(3.0)

    def test_unset_decel_mirrors_framework_default_accel(self):
        p = ControllerParams()
        assert p._eff_linear_decel() == p._eff_linear_accel()

    def test_explicit_decel_is_independent(self):
        p = ControllerParams(max_linear_accel=3.0, max_linear_decel=0.5)
        assert p.scalar_max_linear_accel() == pytest.approx(3.0)
        assert p.scalar_max_linear_decel() == pytest.approx(0.5)

    def test_per_axis_decel_projects_like_accel(self):
        """Per-axis decel uses the same direction projection as per-axis accel."""
        import numpy as np

        p = ControllerParams(max_linear_accel=[1.0, 4.0, 0.0], max_linear_decel=[2.0, 8.0, 0.0])
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        assert p.linear_accel_along_direction(x_axis) == pytest.approx(1.0)
        assert p.linear_decel_along_direction(x_axis) == pytest.approx(2.0)
        assert p.linear_accel_along_direction(y_axis) == pytest.approx(4.0)
        assert p.linear_decel_along_direction(y_axis) == pytest.approx(8.0)

    def test_from_dict_accepts_decel(self):
        p = ControllerParams.from_dict({"max_linear_accel": 2.0, "max_linear_decel": 0.5})
        assert p.max_linear_decel == 0.5


class TestPerAgentControllersUseTheDecelLimit:
    """Raised in review: the controller wiring was not exercised at all.

    The tests above cover ``build_tpi()`` and inject ``_tpi_forward`` by hand,
    so a regression in which limit ``_init_linear_tpi()`` selects for ``dmax``
    would have passed the suite. These drive a real agent through
    ``set_path()`` and measure the trajectory it produces.
    """

    DISTANCE = 8.0
    LIMITS = dict(max_linear_vel=[2.5, 2.5, 0.0], max_linear_accel=[1.5, 1.5, 0.0], navigation_2d=True)
    DT = 0.02

    @pytest.fixture
    def sim_core(self):
        import pybullet as p

        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=False, timestep=self.DT, monitor=False, enable_monitor_gui=False)
        )
        sim.initialize_simulation()
        yield sim
        try:
            p.disconnect(sim.client)
        except p.error:
            pass

    def _agent(self, sim_core, name, controller_cls, **extra):
        from pybullet_fleet import Agent, AgentSpawnParams

        return Agent.from_params(
            AgentSpawnParams(
                urdf_path="cube_small.urdf",
                name=name,
                controller=controller_cls(ControllerParams(**{**self.LIMITS, **extra})),
            ),
            sim_core=sim_core,
        )

    def _travel_time(self, sim_core, agent, limit=4000):
        from pybullet_fleet.geometry import Pose

        agent.set_path([Pose.from_xyz(self.DISTANCE, 0.0, 0.0)], auto_approach=False)
        for step in range(limit):
            sim_core.step_once()
            if not agent.is_moving:
                return (step + 1) * self.DT
        raise AssertionError("the agent never arrived")

    def test_a_softer_decel_makes_an_omni_trajectory_longer(self, sim_core):
        from pybullet_fleet import OmniController

        symmetric = self._travel_time(sim_core, self._agent(sim_core, "sym", OmniController))
        softer = self._travel_time(sim_core, self._agent(sim_core, "soft", OmniController, max_linear_decel=[0.5, 0.5, 0.0]))
        assert softer > symmetric + 0.5

    def test_a_harder_decel_makes_it_shorter(self, sim_core):
        from pybullet_fleet import OmniController

        symmetric = self._travel_time(sim_core, self._agent(sim_core, "sym", OmniController))
        harder = self._travel_time(sim_core, self._agent(sim_core, "hard", OmniController, max_linear_decel=[6.0, 6.0, 0.0]))
        assert harder < symmetric - 0.2

    def test_the_omni_duration_matches_the_closed_form(self, sim_core):
        """v/a + v/d + (s - v^2/2a - v^2/2d)/v, with v capped by the ramps."""
        from pybullet_fleet import OmniController

        v, a, d = 2.5, 1.5, 0.5
        expected = v / a + v / d + (self.DISTANCE - v * v / (2 * a) - v * v / (2 * d)) / v
        measured = self._travel_time(sim_core, self._agent(sim_core, "closed", OmniController, max_linear_decel=[d, d, 0.0]))
        assert measured == pytest.approx(expected, abs=4 * self.DT)

    def test_a_differential_trajectory_uses_it_too(self, sim_core):
        from pybullet_fleet import DifferentialController

        symmetric = self._travel_time(sim_core, self._agent(sim_core, "dsym", DifferentialController))
        softer = self._travel_time(
            sim_core, self._agent(sim_core, "dsoft", DifferentialController, max_linear_decel=[0.5, 0.5, 0.0])
        )
        assert softer > symmetric + 0.5


class TestBatchedControllersHonourTheDecelLimit:
    """The batched controllers evaluate a closed form rather than the TPI, and
    that form now takes the braking scalar as its own argument.

    Until it did, ``trapezoid_distance()`` integrated the braking phase with
    the acceleration scalar, which would have put every agent in the wrong
    place with nothing raised -- so the profile was refused instead.
    """

    DT = 0.02
    DISTANCE = 8.0

    @pytest.fixture
    def sim_core(self):
        import pybullet as p

        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=False, timestep=self.DT, monitor=False, enable_monitor_gui=False)
        )
        sim.initialize_simulation()
        yield sim
        try:
            p.disconnect(sim.client)
        except p.error:
            pass

    def _trace(self, sim_core, mode, accel, decel, name, goal=None):
        """Drive one agent to a goal and record its x over time."""
        from pybullet_fleet import Agent, AgentManager, AgentSpawnParams, OmniController
        from pybullet_fleet.geometry import Pose

        manager = AgentManager(sim_core=sim_core)
        if mode is not None:
            manager.enable_batch(mode)
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="cube_small.urdf",
                name=name,
                controller=OmniController(
                    ControllerParams(
                        max_linear_vel=[2.5, 2.5, 0.0],
                        max_linear_accel=[accel, accel, 0.0],
                        max_linear_decel=[decel, decel, 0.0],
                        navigation_2d=True,
                    )
                ),
            ),
            sim_core=sim_core,
        )
        manager.add_object(agent)
        agent.set_path([goal or Pose.from_xyz(self.DISTANCE, 0.0, 0.0)], auto_approach=False)
        trace = []
        for _ in range(3000):
            sim_core.step_once()
            trace.append(agent.get_pose().position[0])
            if not agent.is_moving:
                break
        return trace

    def _agent(self, sim_core, manager, name, accel, decel):
        from pybullet_fleet import Agent, AgentSpawnParams, OmniController

        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="cube_small.urdf",
                name=name,
                controller=OmniController(
                    ControllerParams(
                        max_linear_vel=[2.5, 2.5, 0.0],
                        max_linear_accel=[accel, accel, 0.0],
                        max_linear_decel=[decel, decel, 0.0],
                        navigation_2d=True,
                    )
                ),
            ),
            sim_core=sim_core,
        )
        manager.add_object(agent)
        return agent

    @pytest.mark.parametrize("decel", [0.5, 6.0])
    def test_batch_matches_the_per_agent_trajectory(self, sim_core, decel):
        """The whole point: the closed form and the TPI must agree.

        Both agents live in the same core and are stepped together, so they
        share a clock and can be compared step for step.
        """
        from pybullet_fleet import AgentManager
        from pybullet_fleet.geometry import Pose

        plain = AgentManager(sim_core=sim_core)
        batched = AgentManager(sim_core=sim_core)
        batched.enable_batch("batch_omni")

        single = self._agent(sim_core, plain, "single", 1.5, decel)
        batch = self._agent(sim_core, batched, "batched", 1.5, decel)
        for agent in (single, batch):
            agent.set_path([Pose.from_xyz(self.DISTANCE, 0.0, 0.0)], auto_approach=False)

        worst = 0.0
        for _ in range(3000):
            sim_core.step_once()
            worst = max(worst, abs(single.get_pose().position[0] - batch.get_pose().position[0]))
            if not single.is_moving and not batch.is_moving:
                break

        assert worst < 1e-9, f"batched trajectory diverges by {worst} m"
        assert single.get_pose().position[0] == pytest.approx(self.DISTANCE, abs=1e-6)
        assert batch.get_pose().position[0] == pytest.approx(self.DISTANCE, abs=1e-6)

    def test_a_softer_brake_takes_longer_under_batch_too(self, sim_core):
        symmetric = self._trace(sim_core, "batch_omni", 1.5, 1.5, "sym")
        softer = self._trace(sim_core, "batch_omni", 1.5, 0.5, "soft")
        assert len(softer) * self.DT > len(symmetric) * self.DT + 0.5

    @pytest.mark.parametrize("mode", ["batch_omni", "batch_differential"])
    def test_an_asymmetric_path_is_accepted(self, sim_core, mode):
        """It used to raise from set_path()."""
        from pybullet_fleet.geometry import Pose

        trace = self._trace(sim_core, mode, 1.5, 0.5, "ok", goal=Pose.from_xyz(5.0, 0.0, 0.0))
        assert trace[-1] == pytest.approx(5.0, abs=1e-3)

    def test_a_per_axis_profile_asymmetric_only_on_y_still_works(self, sim_core):
        """accel [1, 4, 0] with decel [1, 8, 0]: x agrees, y does not. The
        earlier preflight compared the forward scalar and let this through to
        a later failure; now it simply runs."""
        from pybullet_fleet import Agent, AgentManager, AgentSpawnParams, OmniController
        from pybullet_fleet.geometry import Pose

        manager = AgentManager(sim_core=sim_core)
        manager.enable_batch("batch_omni")
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="cube_small.urdf",
                name="peraxis",
                controller=OmniController(
                    ControllerParams(
                        max_linear_vel=[2.5, 2.5, 0.0],
                        max_linear_accel=[1.0, 4.0, 0.0],
                        max_linear_decel=[1.0, 8.0, 0.0],
                        navigation_2d=True,
                    )
                ),
            ),
            sim_core=sim_core,
        )
        manager.add_object(agent)

        agent.set_path([Pose.from_xyz(0.0, 5.0, 0.0)], auto_approach=False)
        assert agent.is_moving is True
        for _ in range(3000):
            sim_core.step_once()
            if not agent.is_moving:
                break
        assert agent.get_pose().position[1] == pytest.approx(5.0, abs=1e-3)


class TestCheckpointCaptureCoversBothPaths:
    """Both navigation capture paths preserve independent braking limits."""

    LIMITS = dict(max_linear_vel=[2.5, 2.5, 0.0], navigation_2d=True)

    @pytest.fixture
    def sim_core(self):
        import pybullet as p

        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=False, timestep=0.02, monitor=False, enable_monitor_gui=False)
        )
        sim.initialize_simulation()
        yield sim
        try:
            p.disconnect(sim.client)
        except p.error:
            pass

    def _agent(self, sim_core, name, **extra):
        from pybullet_fleet import Agent, AgentSpawnParams, OmniController

        return Agent.from_params(
            AgentSpawnParams(
                urdf_path="cube_small.urdf",
                name=name,
                controller=OmniController(ControllerParams(**{**self.LIMITS, **extra})),
            ),
            sim_core=sim_core,
        )

    def test_the_generic_capture_path_preserves_deceleration(self, sim_core):
        from pybullet_fleet.geometry import Pose

        agent = self._agent(
            sim_core,
            "capture",
            max_linear_accel=[1.5, 1.5, 0.0],
            max_linear_decel=[0.5, 0.5, 0.0],
        )
        agent.set_path([Pose.from_xyz(6.0, 0.0, 0.0)], auto_approach=False)
        for _ in range(20):
            sim_core.step_once()

        state = agent._controllers[0].capture_navigation_state()
        assert state["forward"]["decel"] == pytest.approx(0.5)

    def test_a_symmetric_trajectory_still_captures(self, sim_core):
        from pybullet_fleet.geometry import Pose

        agent = self._agent(sim_core, "sym_capture", max_linear_accel=[1.5, 1.5, 0.0])
        agent.set_path([Pose.from_xyz(6.0, 0.0, 0.0)], auto_approach=False)
        for _ in range(20):
            sim_core.step_once()

        state = agent._controllers[0].capture_navigation_state()
        assert state["forward"] is not None
        assert state["forward"]["accel"] == pytest.approx(1.5)


class TestIsMovingIsSetUnderTheBatchLock:
    """Raised in review: setting the flag after the lock released races a
    concurrent step_once() that completes a zero-distance path and clears it,
    leaving an idle row reporting is_moving."""

    @pytest.fixture
    def sim_core(self):
        import pybullet as p

        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=False, timestep=0.02, monitor=False, enable_monitor_gui=False)
        )
        sim.initialize_simulation()
        yield sim
        try:
            p.disconnect(sim.client)
        except p.error:
            pass

    def _managed(self, sim_core):
        from pybullet_fleet import Agent, AgentManager, AgentSpawnParams, OmniController

        manager = AgentManager(sim_core=sim_core)
        manager.enable_batch("batch_omni")
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="cube_small.urdf",
                name="bot",
                controller=OmniController(
                    ControllerParams(max_linear_vel=[2.5, 2.5, 0.0], max_linear_accel=[1.5, 1.5, 0.0], navigation_2d=True)
                ),
            ),
            sim_core=sim_core,
        )
        manager.add_object(agent)
        return manager, agent

    def test_a_step_landing_at_lock_release_is_not_overridden(self, sim_core):
        """The race, made deterministic.

        ``synchronized_batch_advance()`` takes the same lock, so a step can
        only interleave once ``synchronized_set_path()`` has released it. This
        wraps that method to run exactly one step at that moment, with a
        zero-distance path the step completes immediately.

        With the flag set inside the lock, the step's clear is the last word
        and the agent is correctly idle. With it set by the caller afterwards,
        the assignment lands after the step and revives an idle row.
        """
        from pybullet_fleet.geometry import Pose

        manager, agent = self._managed(sim_core)
        controller = manager.batch_controller
        original = controller.synchronized_set_path

        def step_at_lock_release(a, path, **kwargs):
            original(a, path, **kwargs)
            sim_core.step_once()

        controller.synchronized_set_path = step_at_lock_release

        here = agent.get_pose().position
        agent.set_path([Pose.from_xyz(here[0], here[1], here[2])], auto_approach=False)

        assert agent.is_moving is False

    def test_a_step_completing_the_path_leaves_it_idle(self, sim_core):
        """A zero-distance path is finished by the very next step; the flag
        must not be revived afterwards."""
        from pybullet_fleet.geometry import Pose

        _, agent = self._managed(sim_core)
        here = agent.get_pose().position

        agent.set_path([Pose.from_xyz(here[0], here[1], here[2])], auto_approach=False)
        for _ in range(5):
            sim_core.step_once()

        assert agent.is_moving is False


class TestRowCompactionCarriesTheBrakingScalar:
    """Raised in review: the new per-row braking array was missing from
    ``_swap_rows()``.

    ``unregister_agent()`` compacts by moving the last row into the removed
    one. Every other piece of the survivor's trajectory moved with it, so an
    omitted array left that agent braking at the *removed* agent's rate --
    silently, and only after an ordinary agent removal.
    """

    DT = 0.02

    @pytest.fixture
    def sim_core(self):
        import pybullet as p

        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=False, timestep=self.DT, monitor=False, enable_monitor_gui=False)
        )
        sim.initialize_simulation()
        yield sim
        try:
            p.disconnect(sim.client)
        except p.error:
            pass

    def _agent(self, sim_core, manager, name, decel):
        from pybullet_fleet import Agent, AgentSpawnParams, OmniController

        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="cube_small.urdf",
                name=name,
                controller=OmniController(
                    ControllerParams(
                        max_linear_vel=[2.5, 2.5, 0.0],
                        max_linear_accel=[1.5, 1.5, 0.0],
                        max_linear_decel=[decel, decel, 0.0],
                        navigation_2d=True,
                    )
                ),
            ),
            sim_core=sim_core,
        )
        manager.add_object(agent)
        return agent

    @pytest.mark.parametrize("mode,array", [("batch_omni", "_decel_buf"), ("batch_differential", "_fwd_decel")])
    def test_removing_a_non_last_agent_keeps_each_survivor_scalar(self, sim_core, mode, array):
        from pybullet_fleet import AgentManager

        manager = AgentManager(sim_core=sim_core)
        controller = manager.enable_batch(mode)

        doomed = self._agent(sim_core, manager, "doomed", 0.5)
        survivor = self._agent(sim_core, manager, "survivor", 6.0)

        rows = getattr(controller, array)
        rows[controller._agent_index[id(doomed)]] = 0.5
        rows[controller._agent_index[id(survivor)]] = 6.0

        # Remove the first of the two, so the survivor is compacted into row 0.
        manager.remove_object(doomed)

        idx = controller._agent_index[id(survivor)]
        assert getattr(controller, array)[idx] == pytest.approx(6.0)

    def _survivor_trace(self, remove_the_other):
        """Drive two agents, optionally removing the first mid-trajectory.

        Its own core per run: two runs sharing one would start from different
        clocks and leave the earlier run's agents in the scene.
        """
        import pybullet as p

        from pybullet_fleet import AgentManager, MultiRobotSimulationCore, SimulationParams
        from pybullet_fleet.geometry import Pose

        sim_core = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=False, timestep=self.DT, monitor=False, enable_monitor_gui=False)
        )
        sim_core.initialize_simulation()
        manager = AgentManager(sim_core=sim_core)
        manager.enable_batch("batch_omni")
        # Both on a path, so each row really holds its own braking scalar; an
        # agent that never moved leaves its row at zero.
        doomed = self._agent(sim_core, manager, "doomed", 0.5)
        survivor = self._agent(sim_core, manager, "survivor", 6.0)
        doomed.set_path([Pose.from_xyz(0.0, 8.0, 0.0)], auto_approach=False)
        survivor.set_path([Pose.from_xyz(8.0, 0.0, 0.0)], auto_approach=False)
        sim_core.step_once()

        if remove_the_other:
            manager.remove_object(doomed)

        trace = []
        for _ in range(3000):
            sim_core.step_once()
            trace.append(survivor.get_pose().position[0])
            if not survivor.is_moving:
                break
        try:
            p.disconnect(sim_core.client)
        except p.error:
            pass
        return trace

    def test_a_survivor_under_way_follows_the_same_path_either_way(self):
        """The behaviour behind the array check.

        A fresh ``set_path()`` rewrites the row, so a stale scalar only bites
        an agent already on a trajectory when another is removed -- the
        ordinary case, since agents come and go while the fleet keeps driving.

        The arrival *time* is no help here: it comes from ``_t_total``, which
        swaps correctly. What a wrong braking scalar changes is where the
        agent is *during* the braking phase, so the two runs are compared
        position by position.
        """
        undisturbed = self._survivor_trace(remove_the_other=False)
        after_removal = self._survivor_trace(remove_the_other=True)

        overlap = min(len(undisturbed), len(after_removal))
        worst = max(abs(a - b) for a, b in zip(undisturbed[:overlap], after_removal[:overlap]))
        assert worst < 1e-9, f"removing another agent moved this one by {worst} m"
