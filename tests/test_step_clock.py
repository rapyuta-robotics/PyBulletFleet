"""The step clock: ``sim_time`` is the time of the state you see.

``step_once()`` advances the clock and step counter right after ``pre_step``, so
from the object update on -- the update itself, callbacks, plugin hooks, the
collision check, ``post_step`` -- and outside a step, ``sim_time`` and
``step_count`` are those of the state being computed or observed. A
``pre_step`` subscriber is the one place that sees the previous step. Position
controllers evaluate their trajectories at ``sim_time``, so the pose a step ends
with is the trajectory at the end of the step.

Uses 0.05 s steps and 4 m/s^2 acceleration and deceleration.
"""

from __future__ import annotations

import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.types import CollisionMode

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


def _agent(sim, batch=False, differential=False):
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=MotionMode.DIFFERENTIAL if differential else MotionMode.OMNIDIRECTIONAL,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={"max_linear_vel": VMAX, "max_linear_accel": ACCEL, "max_linear_decel": DECEL},
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        kind = "batch_differential" if differential else "batch_omni"
        AgentManager(sim_core=sim, fleet_controller={"type": kind}).add_object(agent)
    return agent


def _goal(x):
    return Pose.from_xyz(x, 0.0, 0.1)


FAR = 10.0


def _start(sim, agent):
    agent.set_path([_goal(FAR)], final_orientation_align=False, auto_approach=False)


@pytest.mark.parametrize("batch", [False, True], ids=["per-agent", "batch"])
def test_a_pose_is_stamped_with_its_trajectory_time(sim, batch):
    """From rest at 4 m/s^2 the trajectory is x = 2 t^2: every pose a step ends
    with must be that at the time it is stamped with -- e.g. x = 0.08 at
    0.20 s, not at 0.25 s."""
    agent = _agent(sim, batch=batch)
    _start(sim, agent)
    seen = []
    sim.register_callback(
        lambda core, dt: seen.append((core.sim_time, agent.get_pose().position[0])),
        frequency=None,
        phase="post_step",
    )
    for _ in range(8):
        sim.step_once()
        # Outside the step too: the completed time is the pose's time.
        seen.append((sim.sim_time, agent.get_pose().position[0]))
    assert len(seen) == 16
    for stamp, x in seen:
        assert x == pytest.approx(0.5 * ACCEL * stamp**2, abs=1e-9)
    assert (pytest.approx(0.2), pytest.approx(0.08)) in [(s, x) for s, x in seen]


def test_the_clock_advances_right_after_pre_step(sim):
    """pre_step sees the previous step; everything after it sees the new one."""
    seen = []
    sim.events.on(SimEvents.PRE_STEP, lambda sim_time, **_: seen.append(("pre", sim_time, sim.sim_time, sim.step_count)))
    sim.register_callback(lambda sc, dt: seen.append(("callback", sc.sim_time, sc.sim_time, sc.step_count)))
    sim.events.on(SimEvents.POST_STEP, lambda sim_time, **_: seen.append(("post", sim_time, sim.sim_time, sim.step_count)))
    sim.step_once()
    sim.step_once()
    assert seen == [
        ("pre", 0.0, 0.0, 0),
        ("callback", pytest.approx(DT), pytest.approx(DT), 1),
        ("post", pytest.approx(DT), pytest.approx(DT), 1),
        ("pre", pytest.approx(DT), pytest.approx(DT), 1),
        ("callback", pytest.approx(2 * DT), pytest.approx(2 * DT), 2),
        ("post", pytest.approx(2 * DT), pytest.approx(2 * DT), 2),
    ]
    # Outside a step: the completed boundary.
    assert sim.sim_time == pytest.approx(2 * DT)
    assert sim.get_completed_step_boundary() == (2, pytest.approx(2 * DT))


def test_an_action_is_stamped_with_the_step_that_finishes_it(sim):
    """A MoveAction's start and end times are the times of the steps that
    start and finish it -- the time of the state those steps produce."""
    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.geometry import Path

    agent = _agent(sim)
    action = MoveAction(path=Path.from_positions([[0.2, 0.0, 0.1]]), auto_approach=False, final_orientation_align=False)
    agent.add_action(action)
    started = finished = None
    for _ in range(40):
        sim.step_once()
        if started is None and action.start_time is not None:
            started = sim.sim_time
        if action.end_time is not None:
            finished = sim.sim_time
            break
    assert action.start_time == pytest.approx(started) == pytest.approx(DT)
    assert finished is not None and action.end_time == pytest.approx(finished)


def test_collision_observation_carries_the_observed_step(sim):
    observation = None

    def grab(**_):
        nonlocal observation
        observation = sim.get_collision_observation()

    sim.set_collision_check_frequency(None)
    sim.events.on(SimEvents.POST_STEP, grab)
    sim.step_once()
    sim.step_once()
    assert observation is not None
    assert (observation.step, observation.sim_time) == (2, pytest.approx(2 * DT))


@pytest.mark.parametrize("batch", [False, True], ids=["per-agent", "batch"])
def test_a_goal_set_in_pre_step_moves_in_that_step(sim, batch):
    agent = _agent(sim, batch=batch)
    sim.events.on(SimEvents.PRE_STEP, lambda **_: _start(sim, agent) if sim.step_count == 0 else None)
    sim.step_once()
    assert agent.get_pose().position[0] == pytest.approx(0.5 * ACCEL * DT**2, abs=1e-9)


@pytest.mark.parametrize("batch", [False, True], ids=["per-agent", "batch"])
def test_a_differential_pose_is_stamped_with_its_trajectory_time(sim, batch):
    """A differential agent already facing its goal drives straight there, on
    the same x = 2 t^2 from rest, stamped with sim_time; a pre_step path
    moves in that step."""
    agent = _agent(sim, batch=batch, differential=True)
    sim.events.on(SimEvents.PRE_STEP, lambda **_: _start(sim, agent) if sim.step_count == 0 else None)
    seen = []
    for _ in range(6):
        sim.step_once()
        seen.append((sim.sim_time, agent.get_pose().position[0]))
    for stamp, x in seen:
        assert x == pytest.approx(0.5 * ACCEL * stamp**2, abs=1e-9)


def test_sim_time_is_read_only(sim):
    """One clock: sim_time reads it and cannot drift from it by assignment."""
    with pytest.raises(AttributeError):
        sim.sim_time = 1.0


def test_sim_time_is_the_clock_everywhere(sim):
    """Outside a step it is the completed boundary's time; inside, from the
    object update on, the time of the state the step computes."""
    agent = _agent(sim)
    _start(sim, agent)
    seen = []
    sim.events.on(SimEvents.PRE_STEP, lambda **_: seen.append(("pre", sim.sim_time, sim.step_count)))
    sim.register_callback(lambda s, dt: seen.append(("callback", s.sim_time, s.step_count)))
    sim.events.on(SimEvents.POST_STEP, lambda **_: seen.append(("post", sim.sim_time, sim.step_count)))
    for k in range(1, 4):
        sim.step_once()
        step, elapsed = sim.get_completed_step_boundary()
        assert (step, sim.sim_time) == (k, elapsed)
        assert elapsed == pytest.approx(k * DT)
    for k in range(1, 4):
        pre, callback, post = seen[3 * (k - 1) : 3 * k]
        assert pre[1:] == (pytest.approx((k - 1) * DT), k - 1)
        assert callback[1:] == (pytest.approx(k * DT), k)
        assert post[1:] == (pytest.approx(k * DT), k)
