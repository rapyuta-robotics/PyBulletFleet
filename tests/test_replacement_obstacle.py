"""Per-agent and batched controllers judge a replacement with the same rule.

Both families read the motion from the agent's velocity state and apply one
shared rule (GoalReplacementPolicy), so they give the same answer -- the same
reason, or none -- for any goal.
"""

from __future__ import annotations

import math

import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.types import CollisionMode

DT = 0.05
VMAX = 2.0
ACCEL = 4.0
DECEL = 2.0
BRAKING = VMAX * VMAX / (2.0 * DECEL)


def _cruising(motion, batch):
    sim = MultiRobotSimulationCore(
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
    sim.initialize_simulation()
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=motion,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={"max_linear_vel": VMAX, "max_linear_accel": ACCEL, "max_linear_decel": DECEL},
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        kind = "batch_omni" if motion is MotionMode.OMNIDIRECTIONAL else "batch_differential"
        AgentManager(sim_core=sim, fleet_controller={"type": kind}).add_object(agent)
    agent.set_path([Pose.from_xyz(50.0, 0.0, 0.1)], auto_approach=False, final_orientation_align=False)
    for _ in range(int(VMAX / ACCEL / DT) + 5):
        sim.step_once()
    controller = agent._batch_controller or agent._controllers[0]
    return sim, agent, controller


def _goal(x, y):
    # Oriented across the line, so the heading a differential agent needs is
    # the line to the goal (a goal whose own x axis is within ~18 degrees of
    # it would lend its orientation instead; see align_x_axis_quat).
    return Pose.from_yaw(x, y, 0.1, math.pi / 2)


def _goals(x):
    """(label, goal, direction) relative to an agent cruising along +x at ``x``."""
    far = 3.0 * BRAKING
    return [
        ("ahead", _goal(x + far, 0.0), None),
        ("inside braking", _goal(x + 0.5 * BRAKING, 0.0), None),
        ("behind, auto", _goal(x - far, 0.0), "auto"),
        ("behind, forward", _goal(x - far, 0.0), None),
        ("slightly off heading", _goal(x + far, far * math.tan(5e-4)), None),
        ("needs a turn", _goal(x + far, far * math.tan(0.05)), None),
        ("where it is", _goal(x, 0.0), None),
    ]


@pytest.mark.parametrize("motion", [MotionMode.OMNIDIRECTIONAL, MotionMode.DIFFERENTIAL], ids=["omni", "differential"])
def test_per_agent_and_batched_give_the_same_reasons(motion):
    reasons = {}
    for batch in (False, True):
        sim, agent, controller = _cruising(motion, batch)
        try:
            x = agent.get_pose().position[0]
            reasons[batch] = [
                (label, controller._replacement_obstacle(agent, [goal], direction)) for label, goal, direction in _goals(x)
            ]
        finally:
            p.disconnect(sim.client)
    assert reasons[True] == reasons[False]
    by_label = dict(reasons[False])
    assert by_label["ahead"] is None
    assert by_label["inside braking"] is not None
    assert by_label["where it is"] is not None
    if motion is MotionMode.DIFFERENTIAL:
        assert by_label["slightly off heading"] is None  # within HEADING_TOL: continues
        assert "turn in place" in by_label["needs a turn"]
        assert "turn in place" in by_label["behind, forward"]  # forward to a goal behind needs a half turn
        assert by_label["behind, auto"] is None  # backs up along the line instead
