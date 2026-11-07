"""The step visits only the objects with a per-step update, a batch-driven
agent with nothing of its own to run is skipped, and an agent's velocity is
a row the batch controller writes as an array."""

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams
from pybullet_fleet.action import WaitAction
from pybullet_fleet.agent_manager import AgentManager
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionMode


@pytest.fixture
def sim():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, physics=False, monitor=False, log_level="warning"))
    yield sim
    p.disconnect(sim._client)


def box(sim, x, mode=CollisionMode.STATIC):
    shape = ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1])
    return SimObject.from_mesh(
        visual_shape=shape, collision_shape=shape, pose=Pose.from_xyz(x, 0.0, 0.1), mass=0.0, sim_core=sim, collision_mode=mode
    )


def robot(sim, x, name=None):
    return Agent.from_params(
        AgentSpawnParams(urdf_path="robots/simple_cube.urdf", initial_pose=Pose.from_xyz(x, 0.0, 0.0), name=name), sim_core=sim
    )


def test_the_step_visits_agents_and_not_static_boxes(sim):
    boxes = [box(sim, float(k)) for k in range(5)]
    a = robot(sim, 10.0)
    assert sim._updatable == [a]
    sim.remove_object(a)
    assert sim._updatable == []
    sim.remove_object(boxes[0])
    assert sim._updatable == []


def test_a_batch_driven_agent_with_nothing_to_do_is_not_updated(sim, monkeypatch):
    manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
    a = robot(sim, 0.0)
    manager.add_object(a)
    assert a._batch_controller is not None
    calls = []
    original = Agent._update_actions

    def counting(self, dt):
        calls.append(self)
        return original(self, dt)

    monkeypatch.setattr(Agent, "_update_actions", counting)
    sim.step_once()
    assert calls == []
    a.add_action(WaitAction(duration=1.0))  # something of its own to run
    sim.step_once()
    assert calls == [a]


def test_an_entity_event_handler_keeps_the_agent_updated(sim):
    manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
    a = robot(sim, 0.0)
    manager.add_object(a)
    seen = []
    a.events.on(SimEvents.POST_UPDATE, lambda **kw: seen.append(kw["dt"]))
    sim.step_once()
    assert seen == [pytest.approx(sim.params.timestep)]


def test_velocity_is_a_row_the_batch_controller_writes(sim):
    manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
    a = robot(sim, 0.0)
    manager.add_object(a)
    row = sim._store.row_of[a.object_id]
    a._current_velocity[:] = [1.0, 2.0, 0.0]
    a._current_angular_velocity = 0.5
    assert sim._store.lin_vel[row].tolist() == [1.0, 2.0, 0.0]
    assert sim._store.ang_vel[row] == 0.5
    sim._store.lin_vel[row] = [3.0, 0.0, 0.0]
    sim._store.ang_vel[row] = 0.25
    assert a.velocity.tolist() == [3.0, 0.0, 0.0]
    assert a.angular_velocity == 0.25
    a.set_goal_pose(Pose.from_xyz(5.0, 0.0, 0.0))
    for _ in range(5):
        sim.step_once()
    assert a.velocity[0] > 0.0  # the batch controller's velocity, through the row
    assert np.allclose(a.velocity, sim._store.lin_vel[row])


def test_a_removed_agent_keeps_its_own_velocity(sim):
    a = robot(sim, 0.0)
    a._current_velocity[:] = [1.0, 0.0, 0.0]
    sim.remove_object(a)
    b = robot(sim, 1.0)
    b._current_velocity[:] = [9.0, 9.0, 9.0]
    assert a.velocity.tolist() == [1.0, 0.0, 0.0]


def test_set_poses_rejects_an_unknown_id_before_writing(sim):
    a = robot(sim, 0.0)
    with pytest.raises(KeyError):
        sim.set_poses(
            [a.object_id, 12345], np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
        )
    assert a.get_pose().x == 0.0


# --- a move under a batch controller sleeps until its request is settled ----


def _batch_robot(sim, x=0.0):
    from pybullet_fleet.agent_manager import AgentManager

    manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
    a = robot(sim, x)
    manager.add_object(a)
    return a


def test_a_batch_move_action_sleeps_while_followed_and_completes_when_the_row_ends(sim, monkeypatch):
    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.geometry import Path
    from pybullet_fleet.types import ActionStatus

    a = _batch_robot(sim)
    action = MoveAction(path=Path.from_positions([[1.0, 0.0, 0.0]]))
    a.add_action(action)
    executes = []
    original = MoveAction.execute

    def counting(self, agent, dt):
        executes.append(sim.step_count)
        return original(self, agent, dt)

    monkeypatch.setattr(MoveAction, "execute", counting)
    sim.step_once()  # starts the action, sends the request, sleeps
    assert action.status == ActionStatus.IN_PROGRESS and a._action_asleep
    for _ in range(10):
        sim.step_once()
    assert executes == [1]  # not polled while the row is followed
    for _ in range(200):
        sim.step_once()
        if action.status == ActionStatus.COMPLETED:
            break
    assert action.status == ActionStatus.COMPLETED
    assert a.get_pose().x == pytest.approx(1.0, abs=1e-3)
    assert not a._action_asleep
    assert len(executes) == 2  # woken once, by the row's end


def test_a_sleeping_move_action_is_woken_by_a_path_from_outside(sim):
    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.geometry import Path
    from pybullet_fleet.types import ActionStatus

    a = _batch_robot(sim)
    action = MoveAction(path=Path.from_positions([[5.0, 0.0, 0.0]]))
    a.add_action(action)
    for _ in range(3):
        sim.step_once()
    assert a._action_asleep
    a.set_goal_pose(Pose.from_xyz(-2.0, 0.0, 0.0))  # replaces the action's path
    assert not a._action_asleep
    sim.step_once()
    assert action.status == ActionStatus.CANCELLED


def test_a_stopped_agent_ends_its_sleeping_action(sim):
    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.geometry import Path
    from pybullet_fleet.types import ActionStatus

    a = _batch_robot(sim)
    action = MoveAction(path=Path.from_positions([[5.0, 0.0, 0.0]]))
    a.add_action(action)
    for _ in range(3):
        sim.step_once()
    a.stop()
    sim.step_once()
    assert action.status == ActionStatus.CANCELLED


def test_settled_joints_let_an_arm_take_the_fast_path_until_a_new_target(sim):
    from pybullet_fleet.agent_manager import AgentManager

    manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
    arm = Agent.from_params(
        AgentSpawnParams(urdf_path="robots/arm_robot.urdf", initial_pose=Pose.from_xyz(0.0, 0.0, 0.0), use_fixed_base=True),
        sim_core=sim,
    )
    manager.add_object(arm)
    arm.set_joint_target_by_name("shoulder_to_elbow", 0.3)
    assert not arm._joints_settled
    for _ in range(400):  # the joint's speed limit and ramp decide how long it takes
        sim.step_once()
        if arm._joints_settled:
            break
    # Settled: either the update saw no joint move, or no joint has a target left to visit.
    assert arm._joints_settled or not arm._active_joints
    assert arm.are_joints_at_targets()
    assert arm.update(0.1) is False  # the fast path: nothing to do
    arm.set_joint_target_by_name("shoulder_to_elbow", 0.0)
    assert not arm._joints_settled
    assert arm.update(0.1) is True  # moving again
