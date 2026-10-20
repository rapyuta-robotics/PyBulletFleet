"""Agent.brake(): come to rest at the configured deceleration.

stop() zeroes the velocity in one step. brake() keeps the line of motion and
the heading and stops at the braking point, decelerating at exactly the
configured rate -- the deceleration, not the acceleration, which the limits
here keep different on purpose.
"""

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.action import MoveAction
from pybullet_fleet.geometry import Path
from pybullet_fleet.types import ActionStatus, CollisionMode

DT = 0.05
VMAX = 2.0
ACCEL = 4.0
DECEL = 2.0  # differs from ACCEL, so braking at the wrong rate shows


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


@pytest.fixture(params=[False, True], ids=["per_agent", "batch"])
def batch(request):
    return request.param


def _agent(sim, batch=False, motion_mode=MotionMode.OMNIDIRECTIONAL, fixed=False):
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=motion_mode,
            collision_mode=CollisionMode.NORMAL_3D,
            use_fixed_base=fixed,
            controller={"max_linear_vel": VMAX, "max_linear_accel": ACCEL, "max_linear_decel": DECEL},
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"}).add_object(agent)
    return agent


def _xy(agent):
    return np.array(agent.get_pose().position[:2], dtype=float)


def _brake_and_track(sim, agent, steps=60):
    """Brake now; return (start position, start velocity, per-step positions)."""
    start = _xy(agent)
    # The velocity the controller continues from -- for a batch row its
    # profile speed, not the reported last-step average.
    controller = agent._batch_controller or agent._controllers[0]
    velocity = np.array(controller.continuation_velocity(agent)[:2], dtype=float)
    assert agent.brake() is True
    track = []
    for _ in range(steps):
        sim.step_once()
        track.append(_xy(agent))
    return start, velocity, np.array(track)


def _assert_braked(start, velocity, track, decel=DECEL):
    speed = float(np.linalg.norm(velocity))
    unit = velocity / speed
    expected = start + unit * speed * speed / (2.0 * decel)
    # Ends at rest at the braking point.
    assert track[-1] == pytest.approx(expected, abs=1e-6)
    assert np.linalg.norm(track[-1] - track[-2]) < 1e-9
    # Along the line of motion only, forward only.
    rel = track - start
    along = rel @ unit
    across = rel - np.outer(along, unit)
    assert np.abs(across).max() < 1e-9
    assert np.all(np.diff(along) >= -1e-12)
    # Per-step speed never rises and never drops faster than the deceleration.
    speeds = np.r_[speed, np.diff(np.r_[0.0, along]) / DT]
    assert np.all(np.diff(speeds) <= 1e-6)
    assert np.max(-np.diff(speeds)) <= decel * DT + 1e-6


def _cruise(sim, agent, goal=(100.0, 0.0)):
    """Accelerate toward a far ``goal`` and stop stepping while cruising at VMAX."""
    agent.set_path([Pose.from_xyz(goal[0], goal[1], 0.1)], auto_approach=False)
    for _ in range(int(2 * VMAX / ACCEL / DT) + 5):
        sim.step_once()
    # Still on the way, at full speed: the profile speed, not a batch row's
    # reported last-step average.
    controller = agent._batch_controller or agent._controllers[0]
    assert agent.is_moving
    assert np.linalg.norm(controller.continuation_velocity(agent)) == pytest.approx(VMAX, rel=1e-6)


class TestBrake:
    def test_stops_at_the_braking_point_at_the_deceleration(self, sim, batch):
        agent = _agent(sim, batch)
        _cruise(sim, agent)
        assert np.linalg.norm(agent.velocity[:2]) == pytest.approx(VMAX, rel=1e-3)
        start, velocity, track = _brake_and_track(sim, agent)
        _assert_braked(start, velocity, track)
        assert not agent.is_moving

    def test_per_agent_and_batch_brake_alike(self, sim):
        tracks = []
        for batch in (False, True):
            agent = _agent(sim, batch)
            _cruise(sim, agent)
            tracks.append(_brake_and_track(sim, agent)[2])
            sim.remove_object(agent)
        assert tracks[0] == pytest.approx(tracks[1], abs=1e-9)

    def test_while_still_accelerating(self, sim, batch):
        agent = _agent(sim, batch)
        agent.set_path([Pose.from_xyz(100.0, 0.0, 0.1)], auto_approach=False)
        for _ in range(4):
            sim.step_once()
        assert 0.0 < np.linalg.norm(agent.velocity[:2]) < VMAX
        _assert_braked(*_brake_and_track(sim, agent))

    def test_moving_diagonally(self, sim, batch):
        agent = _agent(sim, batch)
        _cruise(sim, agent, goal=(60.0, 80.0))
        start, velocity, track = _brake_and_track(sim, agent, steps=80)
        # Per-axis caps are projected onto the direction of travel; with
        # scalar limits the deceleration along it is DECEL.
        _assert_braked(start, velocity, track)

    def test_ends_a_velocity_command(self, sim):
        agent = _agent(sim)
        agent._controller.set_velocity(vx=1.5)
        for _ in range(5):
            sim.step_once()
        _assert_braked(*_brake_and_track(sim, agent))

    def test_at_rest_does_nothing(self, sim, batch):
        agent = _agent(sim, batch)
        sim.step_once()
        before = _xy(agent)
        assert agent.brake() is False
        sim.step_once()
        assert _xy(agent) == pytest.approx(before)
        assert not agent.is_moving

    def test_fixed_base_and_differential_do_not_brake(self, sim):
        assert _agent(sim, fixed=True).brake() is False
        diff = _agent(sim, motion_mode=MotionMode.DIFFERENTIAL)
        diff.set_path([Pose.from_xyz(10.0, 0.0, 0.1)], auto_approach=False)
        for _ in range(10):
            sim.step_once()
        assert diff.is_moving
        assert diff.brake() is False  # differential restarts paths from rest (issue #78)

    def test_drops_the_rest_of_the_path(self, sim, batch):
        agent = _agent(sim, batch)
        agent.set_path(
            [Pose.from_xyz(20.0, 0.0, 0.1), Pose.from_xyz(20.0, 20.0, 0.1)], auto_approach=False, final_orientation_align=False
        )
        for _ in range(15):
            sim.step_once()
        start, velocity, track = _brake_and_track(sim, agent, steps=80)
        _assert_braked(start, velocity, track)
        assert not agent.is_moving


class TestAfterBraking:
    def test_a_goal_in_another_direction_starts_once_stopped(self, sim, batch):
        """Refused while braking, taken when at rest -- MoveAction retries."""
        agent = _agent(sim, batch)
        _cruise(sim, agent)
        controller = agent._batch_controller or agent._controllers[0]
        velocity = np.array(controller.continuation_velocity(agent)[:2], dtype=float)
        speed = float(np.linalg.norm(velocity))
        braking_point = _xy(agent) + velocity / speed * speed * speed / (2.0 * DECEL)
        assert agent.brake() is True
        action = MoveAction(path=Path.from_positions([[0.0, 5.0, 0.1]]), auto_approach=False)
        agent.add_action(action)
        sim.step_once()
        assert action.status == ActionStatus.NOT_STARTED  # off the line: refused for now
        track = [_xy(agent)]
        for _ in range(300):
            sim.step_once()
            track.append(_xy(agent))
            if action.status == ActionStatus.COMPLETED:
                break
        assert action.status == ActionStatus.COMPLETED
        assert track[-1] == pytest.approx([0.0, 5.0], abs=1e-6)
        # Taken only once the agent stood at the end of the braking, not on the
        # way: the last pose before it leaves the line of motion is the braking
        # point. (A batch row takes it in the arrival step itself, after the
        # batch advance; a per-agent controller in the next one.)
        first_off_line = next(i for i, xy in enumerate(track) if abs(xy[1]) > 1e-9)
        assert track[first_off_line - 1] == pytest.approx(braking_point, abs=1e-5)

    def test_clear_actions_first_cancels_the_move(self, sim):
        agent = _agent(sim)
        action = MoveAction(path=Path.from_positions([[100.0, 0.0, 0.1]]), auto_approach=False)
        agent.add_action(action)
        for _ in range(int(2 * VMAX / ACCEL / DT) + 5):
            sim.step_once()
        agent.clear_actions()
        _assert_braked(*_brake_and_track(sim, agent))
        assert action.status == ActionStatus.CANCELLED

    def test_a_move_still_running_completes_at_rest(self, sim):
        """Documented: brake() leaves the action queue alone."""
        agent = _agent(sim)
        action = MoveAction(path=Path.from_positions([[100.0, 0.0, 0.1]]), auto_approach=False)
        agent.add_action(action)
        for _ in range(int(2 * VMAX / ACCEL / DT) + 5):
            sim.step_once()
        assert agent.brake() is True
        for _ in range(60):
            sim.step_once()
        assert action.status == ActionStatus.COMPLETED
        assert _xy(agent)[0] < 100.0


def test_a_checkpoint_mid_brake_restores_the_same_trajectory(sim):
    agent = _agent(sim)
    _cruise(sim, agent)
    assert agent.brake() is True
    for _ in range(3):
        sim.step_once()
    controller = agent._controllers[0]
    state = controller.capture_navigation_state()

    from pybullet_fleet.controller import OmniController

    restored = OmniController(controller.params)
    restored.restore_navigation_state(state, list(agent.get_pose().orientation))
    for t in np.linspace(sim.sim_time, sim.sim_time + 2.0, 9):
        assert restored._tpi_forward.get_point(t)[0] == pytest.approx(controller._tpi_forward.get_point(t)[0])
