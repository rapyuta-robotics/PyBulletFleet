"""set_controller() keeps controller_params in step with the controller.

``controller_params`` is the single source of truth for an agent's kinematic
limits: :attr:`Agent.max_linear_vel` and its siblings delegate to it, and a
batch controller builds its trajectory from it and nothing else. A controller
installed after construction used to leave it behind, so the agent moved by
one set of limits while everything that asked reported another.
"""

import pytest

from pybullet_fleet import Agent, AgentSpawnParams
from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet import OmniController

LIMITS = dict(max_linear_vel=[2.5, 1.2, 0.0], max_linear_accel=[1.5, 0.8, 0.0], navigation_2d=True)


def _agent(sim_core, **kwargs):
    return Agent.from_params(AgentSpawnParams(urdf_path="cube_small.urdf", **kwargs), sim_core=sim_core)


class TestSetControllerAdoptsParams:
    def test_params_follow_a_controller_installed_after_the_spawn(self, sim_core):
        agent = _agent(sim_core, name="a")
        agent.set_controller(OmniController(ControllerParams(**LIMITS)))
        assert list(agent.controller_params.max_linear_vel) == pytest.approx([2.5, 1.2, 0.0])
        assert list(agent.controller_params.max_linear_accel) == pytest.approx([1.5, 0.8, 0.0])
        assert agent.controller_params.navigation_2d is True

    def test_it_agrees_with_passing_the_controller_at_construction(self, sim_core):
        late = _agent(sim_core, name="late")
        late.set_controller(OmniController(ControllerParams(**LIMITS)))
        early = _agent(sim_core, name="early", controller=OmniController(ControllerParams(**LIMITS)))
        for field in ("max_linear_vel", "max_linear_accel"):
            assert list(getattr(late.controller_params, field)) == pytest.approx(list(getattr(early.controller_params, field)))

    def test_the_delegating_properties_report_the_new_limits(self, sim_core):
        """max_linear_vel and friends read controller_params, so they were the
        visible half of the divergence."""
        agent = _agent(sim_core, name="a")
        agent.set_controller(OmniController(ControllerParams(**LIMITS)))
        assert list(agent.max_linear_vel)[:2] == pytest.approx([2.5, 1.2])
        assert list(agent.max_linear_accel)[:2] == pytest.approx([1.5, 0.8])

    def test_a_controller_without_params_leaves_them_alone(self, sim_core):
        agent = _agent(sim_core, name="a", controller=OmniController(ControllerParams(**LIMITS)))

        class Bare:
            params = None

            def compute(self, agent, dt):
                return False

        agent.set_controller(Bare())
        assert list(agent.controller_params.max_linear_vel) == pytest.approx([2.5, 1.2, 0.0])

    def test_disabling_movement_keeps_the_limits(self, sim_core):
        """None means "do not move", not "forget how fast you may move"."""
        agent = _agent(sim_core, name="a", controller=OmniController(ControllerParams(**LIMITS)))
        agent.set_controller(None)
        assert agent._controllers == []
        assert list(agent.controller_params.max_linear_vel) == pytest.approx([2.5, 1.2, 0.0])
