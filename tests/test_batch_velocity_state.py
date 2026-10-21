"""An agent's velocity is its state, set from the trajectory it follows.

The trajectory (TPI) is the reference the kinematics follow exactly; the
agent's pose and velocity are the result. Per-agent and batched controllers
both report the trajectory's velocity at the step's time -- not, for a batch,
the average over the step, which lags while accelerating, leads while braking
and is still non-zero on the arrival step -- and a goal replaced mid-motion
continues from that state.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.types import CollisionMode

DT = 0.05
VMAX = 2.0
ACCEL = 4.0
DECEL = 2.0
CONTROLLER = {
    "max_linear_vel": VMAX,
    "max_linear_accel": ACCEL,
    "max_linear_decel": DECEL,
    "max_angular_vel": 1.5,
    "max_angular_accel": 3.0,
}


def _sim():
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
    return sc


def _agent(sim, mode, batch, **controller):
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=mode,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={**CONTROLLER, **controller},
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        kind = "batch_omni" if mode is MotionMode.OMNIDIRECTIONAL else "batch_differential"
        AgentManager(sim_core=sim, fleet_controller={"type": kind}).add_object(agent)
    return agent


def _states(mode, batch, script, steps):
    """Per-step (vx, vy, yaw rate) of an agent driven by ``script(agent, step)``."""
    sim = _sim()
    try:
        agent = _agent(sim, mode, batch)
        out = []
        for k in range(steps):
            script(agent, k)
            sim.step_once()
            vx, vy, _ = agent.velocity
            out.append((float(vx), float(vy), float(agent.angular_velocity)))
        return out
    finally:
        p.disconnect(sim.client)


def _goal(x, y=0.0, yaw=0.0):
    return Pose.from_yaw(x, y, 0.1, yaw)


def _omni_script(agent, k):
    if k == 0:
        agent.set_goal_pose(_goal(3.0, yaw=1.2))  # accelerate, cruise, brake, then turn
    if k == 12:
        agent.set_goal_pose(_goal(-1.0, yaw=-0.8))  # replaced behind: reverse, then turn clockwise
    if k == 90:
        agent.set_goal_pose(_goal(-1.0, yaw=0.5))  # at rest: turn counter-clockwise in place


def _differential_script(agent, k):
    if k == 0:
        agent.set_goal_pose(_goal(0.0, 2.5, yaw=-1.0))  # turn left, drive, turn right
    if k == 14:
        agent.set_goal_pose(_goal(0.0, 4.0, yaw=-1.0))  # replaced further along the heading


@pytest.mark.parametrize(
    "mode, script",
    [(MotionMode.OMNIDIRECTIONAL, _omni_script), (MotionMode.DIFFERENTIAL, _differential_script)],
    ids=["omni", "differential"],
)
def test_batch_reports_the_same_velocity_as_the_per_agent_controller(mode, script):
    per_agent = _states(mode, False, script, 160)
    batch = _states(mode, True, script, 160)
    assert np.asarray(batch) == pytest.approx(np.asarray(per_agent), abs=1e-6)
    # The motion covered every phase: driving, turning both ways, and rest.
    speeds = np.hypot(*np.asarray(per_agent)[:, :2].T)
    yaw_rates = np.asarray(per_agent)[:, 2]
    assert speeds.max() > 1.0 and yaw_rates.max() > 0.1 and yaw_rates.min() < -0.1
    assert per_agent[-1] == (0.0, 0.0, 0.0)


def test_batch_velocity_is_the_profile_velocity_not_the_step_average():
    """While accelerating from rest the step average would be half a step's
    acceleration below the profile speed; on the arrival step it would still
    be non-zero."""
    states = _states(
        MotionMode.OMNIDIRECTIONAL, True, lambda a, k: k == 0 and a.set_path([_goal(3.0)], auto_approach=False), 80
    )
    speeds = [vx for vx, _, _ in states]
    assert speeds[:3] == pytest.approx([ACCEL * DT, 2 * ACCEL * DT, 3 * ACCEL * DT])
    arrival = next(k for k, s in enumerate(speeds) if s == 0.0)
    assert speeds[arrival - 1] > 0.0  # the step before arrival was still braking


def test_a_restored_velocity_state_is_continued_into_a_replaced_goal():
    """After restoring an agent mid-cruise, a new goal continues from the
    restored velocity: the agent keeps cruising instead of restarting.

    Per-agent only: the kinematic checkpoint profile does not cover
    batch-owned agents (it rejects them before capture)."""
    from pybullet_fleet.controller import OmniController

    sim = _sim()
    try:
        agent = _agent(sim, MotionMode.OMNIDIRECTIONAL, False)
        agent.set_goal_pose(_goal(100.0))
        for _ in range(int(VMAX / ACCEL / DT) + 5):
            sim.step_once()
        pose, velocity = agent.get_pose(), agent.velocity
        state = agent._controllers[0].capture_navigation_state()
        assert velocity[0] == pytest.approx(VMAX)
    finally:
        p.disconnect(sim.client)

    sim = _sim()
    try:
        agent = _agent(sim, MotionMode.OMNIDIRECTIONAL, False)
        controller = OmniController(agent._controllers[0].params)
        agent.set_controller(controller)
        controller.restore_navigation_state(state, list(pose.orientation))
        agent.restore_motion_state(pose, velocity, 0.0, moving=True)
        x0 = agent.get_pose().position[0]
        assert agent.set_goal_pose(_goal(x0 + 20.0)) is True
        speeds = []
        for _ in range(5):
            sim.step_once()
            speeds.append(float(agent.velocity[0]))
        assert speeds == pytest.approx([VMAX] * 5)
    finally:
        p.disconnect(sim.client)


@pytest.mark.parametrize("batch", [False, True], ids=["per_agent", "batch"])
def test_a_path_started_from_rest_resets_the_velocity_state(batch):
    """``immediate`` starts an off-line goal from rest: the velocity state is
    rest at once, so a brake() in the same step has nothing to brake instead of
    braking the replaced motion."""
    sim = _sim()
    try:
        agent = _agent(sim, MotionMode.OMNIDIRECTIONAL, batch, goal_replacement="immediate")
        agent.set_goal_pose(_goal(100.0))
        for _ in range(int(VMAX / ACCEL / DT) + 5):
            sim.step_once()
        assert np.linalg.norm(agent.velocity[:2]) == pytest.approx(VMAX)
        x, y, _ = agent.get_pose().position
        assert agent.set_goal_pose(_goal(x, y + 5.0)) is True  # off the line: from rest
        assert np.linalg.norm(agent.velocity) == 0.0
        assert agent.brake() is False  # at rest: nothing to brake
        for _ in range(200):
            sim.step_once()
        assert agent.get_pose().position[:2] == pytest.approx([x, y + 5.0], abs=1e-6)
    finally:
        p.disconnect(sim.client)
