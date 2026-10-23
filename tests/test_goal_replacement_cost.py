"""A goal set on a moving agent solves its trajectory once.

The replacement check (``_replan.replacement_rejection``) and the trajectory
build call the solver with the very same inputs; the build reuses the check's
solution (``_tpi.try_build_tpi(keep_for_build=True)`` hands it to that one
build). Counting solver calls keeps that from regressing without a timing test.
"""

from __future__ import annotations

import pybullet as p
import pytest
from two_point_interpolation import constant_acc

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet import _tpi
from pybullet_fleet._tpi import try_build_tpi
from pybullet_fleet.types import CollisionMode


@pytest.fixture(autouse=True)
def no_kept_solve():
    """Start and end every test with no kept solve.

    ``_tpi._kept_solve`` is one module-level slot: a check whose solve no build
    takes leaves it filled for whatever runs next -- under ``reject`` or
    ``immediate``, a goal the agent cannot continue into keeps the check's
    (infeasible) result. Clearing it keeps these counts independent of test
    order.
    """
    _tpi._kept_solve = None
    yield
    _tpi._kept_solve = None


@pytest.fixture
def solves(monkeypatch):
    """Count TwoPointInterpolation.calc_trajectory calls."""
    calls = [0]
    original = constant_acc.TwoPointInterpolation.calc_trajectory

    def counted(self):
        calls[0] += 1
        return original(self)

    monkeypatch.setattr(constant_acc.TwoPointInterpolation, "calc_trajectory", counted)
    return calls


def _moving_agent(motion, batch):
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            monitor=False,
            physics=False,
            timestep=0.05,
            target_rtf=0,
            collision_check_frequency=0,
            monitor_log_interval=0,
            log_level="warning",
        )
    )
    sim.initialize_simulation()
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=motion,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={"max_linear_vel": 2.0, "max_linear_accel": 1.0, "max_linear_decel": 1.0},
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        kind = "batch_omni" if motion is MotionMode.OMNIDIRECTIONAL else "batch_differential"
        AgentManager(sim_core=sim, fleet_controller={"type": kind}).add_object(agent)
    agent.set_goal_pose(Pose.from_xyz(100.0, 0.0, 0.1))
    for _ in range(40):
        sim.step_once()
    assert agent.is_moving
    return sim, agent


@pytest.mark.parametrize("batch", [False, True], ids=["per_agent", "batch"])
@pytest.mark.parametrize("motion", [MotionMode.OMNIDIRECTIONAL, MotionMode.DIFFERENTIAL], ids=["omni", "differential"])
@pytest.mark.parametrize("goal", [(20.0, 0.0), (0.0, 5.0)], ids=["ahead", "off_line"])
def test_a_goal_on_a_moving_agent_solves_once(solves, motion, batch, goal):
    """Ahead: the check and the trajectory share one solve. Off the line
    (``brake``): the braking path is checked and built with one solve; the
    goal itself is held, not solved."""
    sim, agent = _moving_agent(motion, batch)
    try:
        solves[0] = 0
        assert agent.set_goal_pose(Pose.from_xyz(goal[0], goal[1], 0.1)) is True
        assert solves[0] == 1
    finally:
        p.disconnect(sim.client)


def test_a_kept_solve_goes_to_one_build_with_the_same_inputs(solves):
    """The check keeps its solve; the next build with the same inputs takes it,
    once. A different input, or a second build, solves on its own."""
    args = dict(p0=0.0, pe=5.0, vmax=2.0, accel=1.0, t0=1.0, v0=1.5, decel=1.0)
    checked = try_build_tpi(**args, keep_for_build=True)
    assert try_build_tpi(**{**args, "v0": 1.5000001}) is not checked
    assert solves[0] == 2
    built = try_build_tpi(**args)
    assert built is checked and solves[0] == 2
    again = try_build_tpi(**args)
    assert again is not built and solves[0] == 3
    # An infeasible check is handed over as such too.
    infeasible = dict(p0=0.0, pe=0.1, vmax=2.0, accel=1.0, t0=1.0, v0=2.0, decel=1.0)
    assert try_build_tpi(**infeasible, keep_for_build=True) is None
    assert try_build_tpi(**infeasible) is None
    assert solves[0] == 4


def test_no_two_trajectories_share_a_solve(solves):
    """Callers that keep their trajectory (agents, joints, checkpoint restore)
    never get the same ``TwoPointInterpolation`` for the same inputs, so no
    trajectory can be changed through another."""
    args = dict(p0=0.0, pe=6.0, vmax=2.0, accel=1.0, t0=2.0, v0=1.0, decel=1.0)
    first, second = try_build_tpi(**args), try_build_tpi(**args)
    assert first is not second
    try_build_tpi(**args, keep_for_build=True)
    taken, after = try_build_tpi(**args), try_build_tpi(**args)
    assert len({id(first), id(second), id(taken), id(after)}) == 4


def test_two_agents_given_the_same_goal_get_their_own_trajectories(solves):
    """Same limits, same speed, same goal at the same time: identical solver
    inputs, yet each agent's trajectory is its own."""
    sims = [_moving_agent(MotionMode.OMNIDIRECTIONAL, batch=False) for _ in range(2)]
    try:
        for sim, agent in sims:
            assert agent.set_goal_pose(Pose.from_xyz(20.0, 0.0, 0.1)) is True
        (_, a), (_, b) = sims
        ta, tb = a._controllers[0]._tpi_forward, b._controllers[0]._tpi_forward
        assert (ta.p0, ta.pe, ta.v0, ta.t0) == (tb.p0, tb.pe, tb.v0, tb.t0)
        assert ta is not tb
    finally:
        for sim, _ in sims:
            p.disconnect(sim.client)
