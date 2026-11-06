"""`pickable` survives the trip from AgentSpawnParams to the agent.

``SimObjectSpawnParams.pickable`` is inherited by ``AgentSpawnParams``, but
neither ``Agent.from_mesh`` nor ``Agent.from_urdf`` took the argument, so
``forward_spawn_params()`` dropped it and every agent came out not pickable.
``attach_object()`` refuses a non-pickable body, which made the failure show
up far from its cause: an attach that silently returns False.
"""

import pytest

from pybullet_fleet import Agent, AgentSpawnParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams

URDF = "cube_small.urdf"


class TestAgentPickable:
    def test_default_is_still_not_pickable(self, sim_core):
        agent = Agent.from_params(AgentSpawnParams(urdf_path=URDF, name="bot"), sim_core=sim_core)
        assert agent.pickable is False

    def test_spawn_params_pickable_reaches_a_urdf_agent(self, sim_core):
        agent = Agent.from_params(AgentSpawnParams(urdf_path=URDF, name="cargo", pickable=True), sim_core=sim_core)
        assert agent.pickable is True

    def test_spawn_params_pickable_reaches_a_mesh_agent(self, sim_core):
        agent = Agent.from_params(
            AgentSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
                name="cargo",
                pickable=True,
            ),
            sim_core=sim_core,
        )
        assert agent.pickable is True

    def test_from_dict_carries_it(self, sim_core):
        agent = Agent.from_dict({"name": "cargo", "urdf_path": URDF, "pickable": True}, sim_core=sim_core)
        assert agent.pickable is True

    def test_a_pickable_agent_can_actually_be_attached(self, sim_core):
        """The behaviour the flag exists for: attach_object() refuses a
        non-pickable body, so dropping the flag made an attach fail far from
        where it was configured."""
        carrier = Agent.from_params(AgentSpawnParams(urdf_path=URDF, name="carrier"), sim_core=sim_core)
        cargo = Agent.from_params(
            AgentSpawnParams(urdf_path=URDF, name="cargo", pickable=True, initial_pose=Pose.from_xyz(0.0, 0.0, 0.3)),
            sim_core=sim_core,
        )
        assert carrier.attach_object(cargo, keep_world_pose=True) is True
        assert cargo in carrier.attached_objects


def test_the_agent_default_narrows_the_sim_object_one():
    """SimObjectSpawnParams defaults pickable to True; an agent is usually the
    one doing the picking, so AgentSpawnParams narrows it to False. Field
    order is unchanged, so the redeclaration cannot shift what a positional
    argument means for AgentSpawnParams or ElevatorParams."""
    import dataclasses

    from pybullet_fleet.devices.elevator import ElevatorParams
    from pybullet_fleet.sim_object import SimObjectSpawnParams

    assert SimObjectSpawnParams().pickable is True
    assert AgentSpawnParams(urdf_path=URDF).pickable is False

    expected = [f.name for f in dataclasses.fields(SimObjectSpawnParams)]
    for cls in (AgentSpawnParams, ElevatorParams):
        assert [f.name for f in dataclasses.fields(cls)][: len(expected)] == expected


class TestFromDictDefault:
    """Raised in review: the dataclass default alone does not cover from_dict.

    AgentSpawnParams.from_dict() builds its base fields from
    SimObjectSpawnParams.from_dict(), which has already resolved an absent
    `pickable` to its own True. Splatting that in reinstated True for every
    config-driven agent -- the common path -- while the direct constructor
    said False.
    """

    def test_an_omitted_key_gets_the_agent_default(self):
        assert AgentSpawnParams.from_dict({"name": "bot", "urdf_path": URDF}).pickable is False

    def test_it_agrees_with_the_direct_constructor(self):
        assert (
            AgentSpawnParams.from_dict({"name": "bot", "urdf_path": URDF}).pickable
            is AgentSpawnParams(urdf_path=URDF).pickable
        )

    @pytest.mark.parametrize("value", [True, False])
    def test_an_explicit_key_is_honoured(self, value):
        params = AgentSpawnParams.from_dict({"name": "bot", "urdf_path": URDF, "pickable": value})
        assert params.pickable is value

    def test_subclasses_that_delegate_here_get_it_too(self):
        from pybullet_fleet.devices.elevator import ElevatorParams

        params = ElevatorParams.from_dict(
            {"name": "lift", "urdf_path": "robots/elevator.urdf", "floors": {"0": 0.0}, "initial_floor": "0"}
        )
        assert params.pickable is False

    def test_the_sim_object_default_is_untouched(self):
        from pybullet_fleet.sim_object import SimObjectSpawnParams

        assert SimObjectSpawnParams.from_dict({"name": "box"}).pickable is True


class TestSpawnEventSeesTheConfiguredValue:
    """Raised in review: SimObject.__init__ registers the body and emits
    OBJECT_SPAWNED, so overwriting pickable after super() returned let every
    listener observe the default instead of what the agent was configured
    with."""

    def test_the_spawn_event_carries_the_configured_value(self, sim_core):
        from pybullet_fleet.events import SimEvents

        seen = []
        sim_core.events.on(SimEvents.OBJECT_SPAWNED, lambda obj=None, **_: seen.append(obj.pickable))

        Agent.from_params(AgentSpawnParams(urdf_path=URDF, name="cargo", pickable=True), sim_core=sim_core)

        assert seen == [True]

    def test_and_the_default_at_spawn_time_too(self, sim_core):
        from pybullet_fleet.events import SimEvents

        seen = []
        sim_core.events.on(SimEvents.OBJECT_SPAWNED, lambda obj=None, **_: seen.append(obj.pickable))

        Agent.from_params(AgentSpawnParams(urdf_path=URDF, name="bot"), sim_core=sim_core)

        assert seen == [False]
