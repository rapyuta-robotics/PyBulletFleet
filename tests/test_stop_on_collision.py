"""``ControllerParams.stop_on_collision``: an agent stops dead where it ran into something.

A vehicle that hits something does not brake to a convenient point further on;
it stays where the contact happened. With the flag set, the step a collision
starts that involves the agent -- or an object attached to it, such as a load
it carries -- the agent is stopped as :meth:`Agent.stop` does: velocity zeroed,
path dropped. Off by default.
"""

from __future__ import annotations

import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.action import MoveAction
from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet.geometry import Path
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import ActionStatus, CollisionMode

DT = 0.05
HALF = 0.05  # half extent of robots/simple_cube.urdf [m]


@pytest.fixture
def sim():
    sc = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            monitor=False,
            physics=False,
            timestep=DT,
            target_rtf=0,
            collision_margin=0.0,
            ignore_static_collision=False,
            monitor_log_interval=0,
            log_level="warning",
        )
    )
    sc.initialize_simulation()
    yield sc
    p.disconnect(sc.client)


def _agent(sim, x, stop_on_collision=None, batch=False):
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(x, 0.0, HALF),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={
                "max_linear_vel": 1.0,
                "max_linear_accel": 2.0,
                "max_angular_vel": 1.0,
                "stop_on_collision": stop_on_collision,
            },
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"}).add_object(agent)
    return agent


def _box(sim, x, half=0.05):
    return SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="box", half_extents=[half, half, half]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[half, half, half]),
        pose=Pose.from_xyz(x, 0.0, HALF),
        mass=0.0,
        pickable=True,
        sim_core=sim,
    )


def _run_until_contact(sim, mover, other, max_steps=400):
    """Step until ``mover`` and ``other`` are in contact; return the mover's x at that step."""
    for _ in range(max_steps):
        sim.step_once()
        if any(other.object_id in pair and mover.object_id in pair for pair in sim._active_collision_pairs):
            return mover.get_pose().position[0]
    raise AssertionError("no contact")


def _x(agent):
    return float(agent.get_pose().position[0])


def test_off_by_default():
    assert ControllerParams()._eff_stop_on_collision() is False
    assert ControllerParams(stop_on_collision=True)._eff_stop_on_collision() is True


@pytest.mark.parametrize("batch", [False, True], ids=["per_agent", "batch"])
def test_stops_where_it_hit(sim, batch):
    mover = _agent(sim, 0.0, stop_on_collision=True, batch=batch)
    wall = _agent(sim, 1.5)
    mover.set_goal_pose(Pose.from_xyz(3.0, 0.0, HALF))
    hit_x = _run_until_contact(sim, mover, wall)
    assert not mover.is_moving
    for _ in range(40):
        sim.step_once()
    assert _x(mover) == pytest.approx(hit_x, abs=1e-9)
    assert _x(mover) < _x(wall)


def test_without_the_flag_the_agent_drives_on(sim):
    mover = _agent(sim, 0.0)
    wall = _agent(sim, 1.5)
    mover.set_goal_pose(Pose.from_xyz(3.0, 0.0, HALF))
    hit_x = _run_until_contact(sim, mover, wall)
    for _ in range(40):
        sim.step_once()
    assert _x(mover) > hit_x + 0.1


def test_only_the_flagged_side_stops(sim):
    """The flag belongs to each agent: head-on, the flagged one stops and the unflagged one drives on."""
    plain = _agent(sim, 0.0)
    flagged = _agent(sim, 3.0, stop_on_collision=True)
    plain.set_goal_pose(Pose.from_xyz(3.0, 0.0, HALF))
    flagged.set_goal_pose(Pose.from_xyz(0.0, 0.0, HALF))
    _run_until_contact(sim, plain, flagged)
    assert not flagged.is_moving
    assert plain.is_moving


def test_a_standing_agent_that_is_struck_is_left_alone(sim, monkeypatch):
    """stop() is for motion: a flagged agent struck while it stands is not stopped,
    so an action it runs in place (a pick, a joint move) goes on."""
    mover = _agent(sim, 0.0)
    standing = _agent(sim, 1.5, stop_on_collision=True)
    stops = []
    monkeypatch.setattr(standing, "stop", lambda: stops.append(sim.sim_time))
    mover.set_goal_pose(Pose.from_xyz(3.0, 0.0, HALF))
    _run_until_contact(sim, mover, standing)
    assert stops == []


def test_a_carried_load_stops_its_carrier(sim):
    """The load is wider than the agent: it touches the obstacle first and the carrier stops there."""
    mover = _agent(sim, 0.0, stop_on_collision=True)
    load = _box(sim, 0.0, half=0.3)
    mover.attach_object(load, relative_pose=Pose.from_xyz(0.0, 0.0, 0.0))
    obstacle = _box(sim, 1.5)
    mover.set_goal_pose(Pose.from_xyz(3.0, 0.0, HALF))
    for _ in range(400):
        sim.step_once()
        if any(load.object_id in pair and obstacle.object_id in pair for pair in sim._active_collision_pairs):
            break
    else:
        raise AssertionError("load never reached the obstacle")
    assert not mover.is_moving
    hit_x = _x(mover)
    for _ in range(40):
        sim.step_once()
    assert _x(mover) == pytest.approx(hit_x, abs=1e-9)
    # The agent body itself stayed clear of the obstacle: it was the load that hit.
    assert hit_x + HALF < 1.5 - 0.05


def test_its_own_load_is_not_a_collision(sim):
    mover = _agent(sim, 0.0, stop_on_collision=True)
    load = _box(sim, 0.0)
    mover.attach_object(load, relative_pose=Pose.from_xyz(0.0, 0.0, 0.0))
    mover.set_goal_pose(Pose.from_xyz(1.0, 0.0, HALF))
    for _ in range(200):
        sim.step_once()
    assert _x(mover) == pytest.approx(1.0, abs=1e-3)


def test_a_move_action_cut_short_is_cancelled(sim):
    mover = _agent(sim, 0.0, stop_on_collision=True)
    _agent(sim, 1.5)
    action = MoveAction(path=Path.from_positions([[3.0, 0.0, HALF]]))
    mover.add_action(action)
    for _ in range(400):
        sim.step_once()
        if action.is_complete():
            break
    assert action.status is ActionStatus.CANCELLED


# --- Its own attachment tree never stops an agent ---------------------------
# Whether such contact counts at all is ignore_attached_collision's business
# (test_collision_ignore_filter.py); either way it is never a reason to stop.


def _carrier_with_two_loads(sim):
    """An agent carrying two overlapping loads side by side on it."""
    mover = _agent(sim, 0.0, stop_on_collision=True)
    first, second = _box(sim, 0.0), _box(sim, 0.03)
    mover.attach_object(first, relative_pose=Pose.from_xyz(0.0, 0.0, 0.0))
    mover.attach_object(second, relative_pose=Pose.from_xyz(0.03, 0.0, 0.0))
    return mover, first, second


def _carrier_with_a_chain(sim):
    """An agent carrying a load that carries another load overlapping the agent."""
    mover = _agent(sim, 0.0, stop_on_collision=True)
    first, second = _box(sim, 0.0), _box(sim, 0.03)
    mover.attach_object(first, relative_pose=Pose.from_xyz(0.0, 0.0, 0.0))
    first.attach_object(second, relative_pose=Pose.from_xyz(0.03, 0.0, 0.0))
    return mover, first, second


@pytest.fixture(params=[True, False], ids=["ignored", "counted"])
def sim_any_attached_rule(request):
    """Both settings of ignore_attached_collision: the stop rule must not depend on it."""
    sc = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            monitor=False,
            physics=False,
            timestep=DT,
            target_rtf=0,
            collision_margin=0.0,
            ignore_static_collision=False,
            ignore_attached_collision=request.param,
            monitor_log_interval=0,
            log_level="warning",
        )
    )
    sc.initialize_simulation()
    yield sc
    p.disconnect(sc.client)


@pytest.mark.parametrize("build", [_carrier_with_two_loads, _carrier_with_a_chain], ids=["siblings", "chain"])
def test_its_own_attachment_tree_never_stops_it(sim_any_attached_rule, build):
    sim = sim_any_attached_rule
    mover, _, _ = build(sim)
    mover.set_goal_pose(Pose.from_xyz(1.0, 0.0, HALF))
    for _ in range(200):
        sim.step_once()
    assert _x(mover) == pytest.approx(1.0, abs=1e-3)


def _pair(a, b):
    return tuple(sorted((a.object_id, b.object_id)))


def test_a_carried_load_hitting_another_agent_is_counted_and_stops_the_carrier(sim):
    mover = _agent(sim, 0.0, stop_on_collision=True)
    load = _box(sim, 0.0, half=0.3)
    mover.attach_object(load, relative_pose=Pose.from_xyz(0.0, 0.0, 0.0))
    other = _agent(sim, 1.5)
    mover.set_goal_pose(Pose.from_xyz(3.0, 0.0, HALF))
    for _ in range(400):
        sim.step_once()
        if _pair(load, other) in sim._active_collision_pairs:
            break
    else:
        raise AssertionError("load never reached the other agent")
    assert _pair(load, other) in sim._active_collision_pairs
    assert not mover.is_moving
