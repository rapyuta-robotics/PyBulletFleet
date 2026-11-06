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
