"""A goal replaced mid-motion continues the motion from the agent's current speed.

The omni controllers -- per-agent and batched -- start a replaced goal's
trajectory from the agent's speed toward it, signed along the new line, so a
goal behind is reached by slowing, reversing and driving back; no zero-length
approach waypoint is put in front of a moving agent's path. A goal whose
trajectory does not exist within the configured limits -- inside the braking
distance, where the agent is, or off the line of motion -- is refused under
``goal_replacement="reject"``, which this module runs with: set_goal_pose()
returns False and the current motion continues unchanged. At rest every goal
is taken. No case exceeds the configured limits. The other modes are covered
in test_goal_replacement_modes.py.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.types import CollisionMode

pytestmark = pytest.mark.usefixtures("reject_mode")

DT = 0.05
VMAX = 2.0
ACCEL = 4.0
DECEL = 4.0


@pytest.fixture
def sim():
    sc = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            monitor=False,
            physics=False,
            timestep=DT,
            target_rtf=0,
            collision_check_frequency=0,
            monitor_log_interval=0,
            log_level="warning",
        )
    )
    sc.initialize_simulation()
    yield sc
    p.disconnect(sc.client)


def _agent(sim, batch=False):
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={"max_linear_vel": VMAX, "max_linear_accel": ACCEL, "max_linear_decel": DECEL},
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"}).add_object(agent)
    return agent


def _goal(x, y=0.0):
    return Pose.from_xyz(x, y, 0.1)


def _run(sim, agent, steps):
    """Per-step (x, y, speed) samples."""
    out = []
    for _ in range(steps):
        sim.step_once()
        x, y, _ = agent.get_pose().position
        out.append((x, y, float(np.linalg.norm(agent.velocity[:2]))))
    return out


def _run_vel(sim, agent, steps):
    """Per-step (x, y, vx, vy) samples."""
    out = []
    for _ in range(steps):
        sim.step_once()
        x, y, _ = agent.get_pose().position
        vx, vy = agent.velocity[:2]
        out.append((x, y, float(vx), float(vy)))
    return out


def _assert_within_ramps(start_velocity, samples):
    """The velocity never changes faster than the configured ramps allow.

    Compared as vectors, so a turn counts too: dropping the motion across a
    new direction at once would show up here.
    """
    vels = [np.asarray(start_velocity, dtype=float)] + [np.array([vx, vy]) for _, _, vx, vy in samples]
    jumps = [float(np.linalg.norm(b - a)) for a, b in zip(vels, vels[1:])]
    assert max(jumps) <= max(ACCEL, DECEL) * DT + 1e-6, max(jumps)


def _cruise(sim, agent, far=100.0):
    agent.set_goal_pose(_goal(far))
    _run(sim, agent, int(2 * VMAX / ACCEL / DT) + 5)
    assert np.linalg.norm(agent.velocity[:2]) == pytest.approx(VMAX, rel=1e-3)


@pytest.fixture(params=[False, True], ids=["per_agent", "batch"])
def batch(request):
    return request.param


class TestGoalReplacedMidMotion:
    def test_keeps_its_speed_toward_a_goal_further_ahead(self, sim, batch):
        agent = _agent(sim, batch)
        _cruise(sim, agent)
        start_x = agent.get_pose().position[0]
        agent.set_goal_pose(_goal(start_x + 20.0))
        samples = _run(sim, agent, 5)
        assert all(speed == pytest.approx(VMAX, rel=1e-3) for _, _, speed in samples)
        # Distance covered at full speed, not a restart from rest.
        assert samples[-1][0] - start_x == pytest.approx(5 * DT * VMAX, rel=1e-3)

    def test_a_planar_goal_keeps_the_agents_height(self, sim, batch):
        """A planar command without ``z`` keeps the agent's height, so a moving
        agent continues into it; a goal at another height is a 3D move off the
        line of motion and is refused."""
        from pybullet_fleet import RobotGoalCommand2D

        agent = _agent(sim, batch)
        _cruise(sim, agent)
        x = agent.get_pose().position[0]
        assert agent.set_goal_pose(Pose.from_xyz(x + 10.0, 0.0, 0.0)) is False  # 0.1 m down: off the line
        planar = RobotGoalCommand2D("robot", position=(x + 10.0, 0.0)).to_pose(agent.get_pose().position[2])
        assert agent.set_goal_pose(planar) is True
        samples = _run(sim, agent, 5)
        assert all(speed == pytest.approx(VMAX, rel=1e-3) for _, _, speed in samples)

    def test_speed_never_jumps_more_than_the_ramps_allow(self, sim, batch):
        agent = _agent(sim, batch)
        _cruise(sim, agent)
        x = agent.get_pose().position[0]
        speeds = [VMAX]
        for offset in (6.0, 3.0, 9.0):  # replaced again and again, always ahead
            agent.set_goal_pose(_goal(x + offset))
            samples = _run(sim, agent, 8)
            speeds += [speed for _, _, speed in samples]
            x = samples[-1][0]
        steps = np.abs(np.diff(speeds))
        assert steps.max() <= max(ACCEL, DECEL) * DT + 1e-6

    def test_stops_at_a_goal_within_reach_braking_at_the_configured_rate(self, sim, batch):
        agent = _agent(sim, batch)
        _cruise(sim, agent)
        x0 = agent.get_pose().position[0]
        braking = VMAX * VMAX / (2 * DECEL)
        goal_x = x0 + braking + 0.5
        agent.set_goal_pose(_goal(goal_x))
        samples = _run(sim, agent, int(3 * VMAX / DECEL / DT) + 20)
        xs = [x for x, _, _ in samples]
        speeds = [VMAX] + [speed for _, _, speed in samples]
        assert max(xs) <= goal_x + 1e-6, "overshot the goal"
        assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])), "reversed"
        assert np.max(np.diff(speeds)) <= ACCEL * DT + 1e-6
        assert -np.min(np.diff(speeds)) <= DECEL * DT + 1e-6
        assert xs[-1] == pytest.approx(goal_x, abs=1e-6)
        assert all(abs(y) < 1e-9 for _, y, _ in samples)

    @pytest.mark.parametrize(
        "offset",
        [(0.2, 0.0), (0.0, 5.0), (0.0, 0.0)],
        ids=["inside_braking_distance", "right_angle", "where_it_is"],
    )
    def test_an_infeasible_goal_is_refused_and_the_motion_runs_on(self, sim, batch, offset):
        agent = _agent(sim, batch)
        agent.set_goal_pose(_goal(6.0))
        _run(sim, agent, int(VMAX / ACCEL / DT) + 2)  # cruising toward x = 6
        x0, y0, _ = agent.get_pose().position
        before = [list(w.position) for w in agent.path]
        new_goal = _goal(x0 + offset[0], y0 + offset[1])

        assert agent.path_rejection([new_goal]) is not None
        assert agent.set_goal_pose(new_goal) is False
        assert [list(w.position) for w in agent.path] == before
        samples = _run_vel(sim, agent, 80)
        _assert_within_ramps((VMAX, 0.0), samples)
        assert samples[-1][0] == pytest.approx(6.0, abs=1e-6), "the previous goal is still followed"
        assert all(abs(y - y0) < 1e-9 for _, y, _, _ in samples)

    def test_a_goal_behind_is_reached_by_reversing(self, sim, batch):
        """The velocity runs continuously through zero: slow down, reverse,
        drive back -- no instant stop."""
        agent = _agent(sim, batch)
        _cruise(sim, agent)
        x0 = agent.get_pose().position[0]
        assert agent.set_goal_pose(_goal(x0 - 5.0)) is True
        samples = _run_vel(sim, agent, 120)
        _assert_within_ramps((VMAX, 0.0), samples)
        vxs = [vx for _, _, vx, _ in samples]
        assert min(vxs) < -0.5 * VMAX and samples[0][2] > 0.0, "slowed from forward, then drove back"
        assert max(x for x, _, _, _ in samples) == pytest.approx(x0 + VMAX * VMAX / (2 * ACCEL), abs=1e-3)
        assert samples[-1][0] == pytest.approx(x0 - 5.0, abs=1e-6)

    def test_at_rest_any_goal_is_taken(self, sim, batch):
        agent = _agent(sim, batch)
        assert agent.path_rejection([_goal(-3.0, 2.0)]) is None
        assert agent.set_goal_pose(_goal(-3.0, 2.0)) is True
        samples = _run_vel(sim, agent, 80)
        _assert_within_ramps((0.0, 0.0), samples)
        assert samples[-1][:2] == pytest.approx((-3.0, 2.0), abs=1e-6)

    def test_a_goal_ahead_stays_continuous(self, sim, batch):
        """No dip: the speed toward a goal further ahead is kept exactly."""
        agent = _agent(sim, batch)
        _cruise(sim, agent)
        x0 = agent.get_pose().position[0]
        agent.set_goal_pose(_goal(x0 + 3.0))
        samples = _run_vel(sim, agent, 60)
        _assert_within_ramps((VMAX, 0.0), samples)
        assert samples[0][2] == pytest.approx(VMAX, rel=1e-3)
        xs = [x for x, _, _, _ in samples]
        assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])), "reversed"
        assert xs[-1] == pytest.approx(x0 + 3.0, abs=1e-6)

    def test_a_goal_set_on_arrival_starts_from_rest(self, sim, batch):
        """Arrived is at rest, whatever the arrival step's average speed says.

        The next goal is set from the step the agent arrives in, before
        anything has zeroed the reported velocity.
        """
        agent = _agent(sim, batch)
        agent.set_goal_pose(_goal(1.0))
        handed = []

        def next_goal(sim_core, dt):
            if not agent.is_moving and not handed:
                handed.append(np.linalg.norm(agent.velocity[:2]))
                agent.set_goal_pose(_goal(5.0))

        sim.register_callback(next_goal, phase="post_step")
        for _ in range(200):
            sim.step_once()
            if handed:
                break
        assert handed, "never arrived"
        speed = _run(sim, agent, 1)[0][2]
        # From rest the first step is below one step of acceleration; carrying
        # the arrival step's average speed would put it at or above.
        assert speed < ACCEL * DT - 1e-6

    def test_a_goal_off_a_velocity_command_is_refused(self, sim):
        """The switch from a velocity command to a goal follows the same rule."""
        agent = _agent(sim)
        agent._controllers[0].set_velocity(vx=VMAX)
        for _ in range(5):
            sim.step_once()
        x0, y0, _ = agent.get_pose().position
        assert agent.set_goal_pose(_goal(x0, y0 + 3.0)) is False
        samples = _run_vel(sim, agent, 3)
        assert all(vx == pytest.approx(VMAX, rel=1e-3) for _, _, vx, _ in samples)

    def test_a_goal_behind_a_velocity_command_reverses(self, sim):
        agent = _agent(sim)
        agent._controllers[0].set_velocity(vx=VMAX)
        for _ in range(5):
            sim.step_once()
        x0 = agent.get_pose().position[0]
        assert agent.set_goal_pose(_goal(x0 - 3.0)) is True
        samples = _run_vel(sim, agent, 120)
        _assert_within_ramps((VMAX, 0.0), samples)
        assert samples[-1][0] == pytest.approx(x0 - 3.0, abs=1e-6)

    def test_a_goal_after_a_velocity_command_continues_its_speed(self, sim):
        """Velocity mode does not set is_moving; the switch to a goal must not
        insert the zero-length approach waypoint and restart from rest."""
        agent = _agent(sim)
        agent._controllers[0].set_velocity(vx=VMAX)
        for _ in range(5):
            sim.step_once()
        assert np.linalg.norm(agent.velocity[:2]) == pytest.approx(VMAX, rel=1e-3)
        x0 = agent.get_pose().position[0]
        agent.set_goal_pose(_goal(x0 + 20.0))
        samples = _run(sim, agent, 3)
        assert all(speed == pytest.approx(VMAX, rel=1e-3) for _, _, speed in samples)

    def test_a_path_set_on_arrival_keeps_its_approach_waypoint(self, sim, batch):
        """Arrived is not moving, whatever the arrival step's speed says."""
        agent = _agent(sim, batch)
        agent.set_goal_pose(_goal(1.0))
        paths = []

        def next_path(sim_core, dt):
            if not agent.is_moving and not paths:
                agent.set_goal_pose(_goal(5.0))
                if batch:
                    bc = agent._batch_controller
                    paths.append(bc._paths[bc._agent_index[id(agent)]])
                else:
                    paths.append(agent._controllers[0]._path)

        sim.register_callback(next_path, phase="post_step")
        for _ in range(200):
            sim.step_once()
            if paths:
                break
        assert paths and len(paths[0]) == 2, "approach waypoint + goal"


class TestFromRestUnchanged:
    def test_a_goal_from_rest_ramps_up_from_zero(self, sim, batch):
        agent = _agent(sim, batch)
        agent.set_goal_pose(_goal(10.0))
        speeds = [speed for _, _, speed in _run(sim, agent, 3)]
        assert speeds[0] <= ACCEL * DT + 1e-9
        assert np.max(np.diff([0.0] + speeds)) <= ACCEL * DT + 1e-6


class TestPerAgentMatchesBatch:
    @pytest.mark.parametrize("phase_steps", [3, 14], ids=["accelerating", "braking"])
    def test_per_agent_and_batch_agree_when_replaced_mid_ramp(self, phase_steps):
        """The batch row continues from its profile speed, as the per-agent
        controller does, not from the last step's average."""
        tracks = []
        for batch in (False, True):
            sc = MultiRobotSimulationCore(
                SimulationParams(gui=False, monitor=False, physics=False, timestep=DT, target_rtf=0, log_level="warning")
            )
            sc.initialize_simulation()
            try:
                agent = _agent(sc, batch)
                agent.set_goal_pose(_goal(1.2))  # reaches ~VMAX, then brakes
                for _ in range(phase_steps):
                    sc.step_once()
                x0 = agent.get_pose().position[0]
                agent.set_goal_pose(_goal(x0 + 4.0))
                tracks.append([x - x0 for x, _, _ in _run(sc, agent, 40)])
            finally:
                p.disconnect(sc.client)
        assert tracks[0] == pytest.approx(tracks[1], abs=1e-6)

    def test_per_agent_and_batch_follow_the_same_trajectory(self):
        tracks = []
        for batch in (False, True):
            sc = MultiRobotSimulationCore(
                SimulationParams(gui=False, monitor=False, physics=False, timestep=DT, target_rtf=0, log_level="warning")
            )
            sc.initialize_simulation()
            try:
                agent = _agent(sc, batch)
                _cruise(sc, agent)
                x0 = agent.get_pose().position[0]
                agent.set_goal_pose(_goal(x0 + 3.0))
                tracks.append([x - x0 for x, _, _ in _run(sc, agent, 40)])
            finally:
                p.disconnect(sc.client)
        assert tracks[0] == pytest.approx(tracks[1], abs=1e-6)


def test_a_checkpoint_keeps_the_initial_speed(sim):
    """A trajectory continued from speed restores with the same speed."""
    agent = _agent(sim)
    _cruise(sim, agent)
    x0 = agent.get_pose().position[0]
    agent.set_goal_pose(_goal(x0 + 6.0))
    sim.step_once()
    controller = agent._controllers[0]
    state = controller.capture_navigation_state()
    assert state["forward"]["v0"] == pytest.approx(VMAX, rel=1e-3)

    from pybullet_fleet.controller import OmniController

    restored = OmniController(controller.params)
    restored.restore_navigation_state(state, list(agent.get_pose().orientation))
    for t in np.linspace(sim.sim_time, sim.sim_time + 3.0, 7):
        assert restored._tpi_forward.get_point(t)[0] == pytest.approx(controller._tpi_forward.get_point(t)[0])


def test_a_refused_move_action_starts_once_the_motion_ends(sim):
    """MoveAction retries a refused path instead of completing or failing."""
    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.geometry import Path
    from pybullet_fleet.types import ActionStatus

    agent = _agent(sim)
    agent.set_goal_pose(_goal(3.0))
    _run(sim, agent, int(VMAX / ACCEL / DT) + 2)
    action = MoveAction(path=Path.from_positions([[0.5, 2.0, 0.1]]))  # off the line of motion
    agent.add_action(action)
    sim.step_once()
    assert action.status == ActionStatus.NOT_STARTED  # refused while heading along x to x = 3
    for _ in range(400):
        sim.step_once()
        if action.status == ActionStatus.COMPLETED:
            break
    assert action.status == ActionStatus.COMPLETED
    assert agent.get_pose().position[:2] == pytest.approx((0.5, 2.0), abs=1e-6)


def test_from_speed_never_raises_the_deceleration():
    from pybullet_fleet._tpi import build_tpi_from_speed, try_build_tpi

    assert try_build_tpi(p0=0.0, pe=0.2, vmax=VMAX, accel=ACCEL, t0=0.0, v0=VMAX, decel=DECEL) is None
    with pytest.raises(ValueError):
        build_tpi_from_speed(p0=0.0, pe=0.2, vmax=VMAX, accel=ACCEL, t0=0.0, v0=VMAX, decel=DECEL)


def test_an_initial_speed_over_vmax_first_slows_down_to_it():
    """Above vmax (lowered mid-motion) the trajectory keeps the speed it has
    and slows down to vmax at the ramp limit -- no jump. Cruising at the cap,
    up to float noise, starts there."""
    from pybullet_fleet._tpi import try_build_tpi

    with pytest.warns(UserWarning, match="exceeds vmax"):
        tpi = try_build_tpi(p0=0.0, pe=10.0, vmax=VMAX, accel=ACCEL, t0=0.0, v0=1.5 * VMAX, decel=DECEL)
    assert tpi is not None
    speeds = [tpi.get_point(t)[1] for t in np.arange(0.0, tpi.get_end_time(), DT)]
    assert speeds[0] == pytest.approx(1.5 * VMAX)
    assert max(np.abs(np.diff(speeds))) <= max(ACCEL, DECEL) * DT + 1e-9
    assert max(speeds[int(0.5 * VMAX / ACCEL / DT) + 1 :]) <= VMAX + 1e-9
    assert tpi.get_point(tpi.get_end_time())[0] == pytest.approx(10.0)
    cruising = try_build_tpi(p0=0.0, pe=10.0, vmax=VMAX, accel=ACCEL, t0=0.0, v0=VMAX * (1 + 1e-9), decel=DECEL)
    assert cruising is not None and cruising.get_point(0.0)[1] == pytest.approx(VMAX)


def test_trapezoid_speed_matches_the_trajectory():
    """The batched speed lookup is the trajectory's own velocity, from rest,
    from speed and reversing, and the derivative of trapezoid_distance."""
    from pybullet_fleet._tpi import build_tpi_from_speed, extract_phase_params, trapezoid_distance, trapezoid_speed

    for v0 in (0.0, 0.5, 0.999 * VMAX, -0.75 * VMAX, -VMAX):
        for distance in (0.3, 1.0, 3.0, 10.0):
            for accel, decel in ((ACCEL, DECEL), (1.0, 3.0), (3.0, 1.0)):
                try:
                    tpi = build_tpi_from_speed(p0=0.0, pe=distance, vmax=VMAX, accel=accel, t0=0.0, v0=v0, decel=decel)
                except ValueError:
                    continue  # inside the braking distance
                t_accel, t_const, t_total, accel_eff, decel_eff = extract_phase_params(tpi)
                tau = np.linspace(-0.1, 1.05 * t_total, 400)
                ta, tc, tt, a, d, v0s, dist_total = (
                    np.full_like(tau, value)
                    for value in (t_accel, t_const, t_total, accel_eff, decel_eff, float(tpi.v0), distance)
                )
                speed = trapezoid_speed(tau, ta, tc, tt, a, decel=d, v0=v0s)
                expected = [tpi.get_point(max(t, 0.0))[1] for t in tau]
                assert speed == pytest.approx(expected, abs=1e-9)
                h = 1e-6
                dist = [trapezoid_distance(tau + s, ta, tc, tt, a, dist_total, decel=d, v0=v0s) for s in (-h, h)]
                inner = (tau > h) & (tau < t_total - h)
                assert speed[inner] == pytest.approx(((dist[1] - dist[0]) / (2 * h))[inner], abs=1e-4)


def test_a_reversal_trajectory_ends_at_its_target():
    """A body starting away from the target: the solver's smaller first-phase
    root does not end there; the trajectory must."""
    from pybullet_fleet._tpi import try_build_tpi

    tpi = try_build_tpi(p0=0.0, pe=0.3, vmax=2.0, accel=4.0, t0=0.0, v0=-2.0, decel=3.0)
    end = tpi.get_end_time()
    assert tpi.get_point(end)[0] == pytest.approx(0.3)
    assert tpi.get_point(end - 1e-9)[1] == pytest.approx(0.0, abs=1e-6)


def test_a_checkpoint_of_a_reversal_restores_the_same_trajectory(sim):
    """A trajectory started moving away from its goal (negative v0) restores
    to the same motion, short reversals included."""
    agent = _agent(sim)
    _cruise(sim, agent)
    x0 = agent.get_pose().position[0]
    agent.set_goal_pose(_goal(x0 - 0.3))
    sim.step_once()
    controller = agent._controllers[0]
    state = controller.capture_navigation_state()
    assert state["forward"]["v0"] == pytest.approx(-VMAX, rel=1e-3)

    from pybullet_fleet.controller import OmniController

    restored = OmniController(controller.params)
    restored.restore_navigation_state(state, list(agent.get_pose().orientation))
    for t in np.linspace(sim.sim_time, sim.sim_time + 2.0, 9):
        assert restored._tpi_forward.get_point(t)[0] == pytest.approx(controller._tpi_forward.get_point(t)[0])


def test_a_reversal_checkpoint_passes_profile_validation(sim):
    """The recorded initial speed is signed; validation accepts a negative one."""
    from pybullet_fleet.replay.kinematic_profile import _validate_navigation

    agent = _agent(sim)
    _cruise(sim, agent)
    agent.set_goal_pose(_goal(agent.get_pose().position[0] - 2.0))
    sim.step_once()
    state = agent._controllers[0].capture_navigation_state()
    assert state["forward"]["v0"] < 0
    _validate_navigation(state, differential=False)


class TestPermanentRefusalsAreNotRetried:
    """A refusal that ends with the motion is retried; a permanent one is not."""

    def _fixed(self, sim):
        return Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/simple_cube.urdf",
                initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
                use_fixed_base=True,
            ),
            sim_core=sim,
        )

    def test_the_rejection_says_which_kind(self, sim):
        fixed = self._fixed(sim)
        assert fixed.path_rejection([_goal(1.0)]).temporary is False
        moving = _agent(sim)
        _cruise(sim, moving)
        x, y, _ = moving.get_pose().position
        rejection = moving.path_rejection([_goal(x, y + 1.0)])
        assert rejection.temporary is True and "off the line" in rejection.reason

    def test_a_move_action_on_a_fixed_base_fails(self, sim):
        """Refused for good: the move fails at once rather than being retried."""
        from pybullet_fleet.action import MoveAction
        from pybullet_fleet.geometry import Path
        from pybullet_fleet.types import ActionStatus

        agent = self._fixed(sim)
        action = MoveAction(path=Path.from_positions([[1.0, 0.0, 0.1]]))
        agent.add_action(action)
        for _ in range(3):
            sim.step_once()
        assert action.status == ActionStatus.FAILED

    def test_navigate_to_on_a_fixed_base_counts_as_started(self, sim):
        from pybullet_fleet.agent_behavior_tree import PyBulletFleetNavigationAdapter

        assert PyBulletFleetNavigationAdapter(self._fixed(sim)).start_navigation((1.0, 0.0, 0.1)) is True

    def test_navigate_to_while_moving_off_line_is_retried(self, sim):
        from pybullet_fleet.agent_behavior_tree import PyBulletFleetNavigationAdapter

        agent = _agent(sim)
        _cruise(sim, agent)
        x, y, _ = agent.get_pose().position
        assert PyBulletFleetNavigationAdapter(agent).start_navigation((x, y + 1.0, 0.1)) is False
