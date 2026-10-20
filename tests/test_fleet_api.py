from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pytest

from pybullet_fleet._replan import PathRejection
from pybullet_fleet.events import EventBus
from pybullet_fleet.fleet_api import (
    FLEET_COMMAND_EVENT,
    FleetCommandDispatcher,
    FleetStateProvider,
    RobotGoalCommand2D,
    RobotGoalCommand3D,
    RobotAttachCommand,
    RobotActionCommand,
    RobotJointPositionsCommand,
    RobotNamedJointPositionsCommand,
)
from pybullet_fleet.geometry import Pose


@dataclass
class FakeObject:
    name: str


@dataclass
class FakeAgent:
    name: str
    object_id: int
    pose: Pose = field(default_factory=lambda: Pose.from_xyz(0.0, 0.0, 0.0))
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: float | tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_moving: bool = False
    battery_soc: float | None = None
    is_charging: bool = False

    def __post_init__(self) -> None:
        self.goal_calls = []
        self.stop_calls = 0
        self.joint_calls = []
        self.attach_calls = []
        self.detach_calls = []
        self.action_calls = []
        self.attached_objects = []
        self.pickable_object: FakeObject | None = None
        self.search_radii = []
        self.sim_core = None

    def get_pose(self) -> Pose:
        return self.pose

    def set_goal_pose(self, pose: Pose) -> None:
        self.goal_calls.append(pose)

    def path_rejection(self, path):
        return None  # at rest: every path is taken

    def stop(self) -> None:
        self.stop_calls += 1

    def set_all_joints_targets(self, positions, max_force=500.0):
        self.joint_calls.append(("all", list(positions), max_force))

    def set_joints_targets_by_name(self, positions_by_name, max_force=500.0):
        self.joint_calls.append(("named", dict(positions_by_name), max_force))

    def find_nearest_pickable(self, search_radius=0.5):
        self.search_radii.append(search_radius)
        return self.pickable_object

    def attach_object(self, obj, parent_link_index="base_link", relative_pose=None) -> bool:
        self.attach_calls.append((obj, parent_link_index, relative_pose))
        self.attached_objects.append(obj)
        return True

    def detach_object(self, obj) -> bool:
        self.detach_calls.append(obj)
        self.attached_objects.remove(obj)
        return True

    def get_attached_objects(self):
        return list(self.attached_objects)

    def add_action(self, action):
        self.action_calls.append(action)


class FakeSim:
    def __init__(self, agents, sim_time=12.5):
        self.agents = agents
        self.sim_time = sim_time
        self.events = EventBus()
        self.sim_objects = []
        for agent in self.agents:
            agent.sim_core = self

    def _record_state_input(self, operation, details):
        """Match the core recording hook; this fake does not record."""
        return None


def test_fleet_state_provider_returns_3d_and_2d_snapshots():
    sim = FakeSim(
        [
            FakeAgent(
                "robot0",
                7,
                pose=Pose.from_yaw(1.0, 2.0, 0.3, 1.57),
                velocity=(0.4, 0.5, 0.0),
                angular_velocity=0.2,
                is_moving=True,
                battery_soc=0.75,
            )
        ]
    )

    provider = FleetStateProvider(sim)

    states_3d = provider.get_states()
    assert len(states_3d) == 1
    assert states_3d[0].name == "robot0"
    assert states_3d[0].object_id == 7
    assert states_3d[0].position == (1.0, 2.0, 0.3)
    assert states_3d[0].linear_velocity == (0.4, 0.5, 0.0)
    assert states_3d[0].angular_velocity == (0.0, 0.0, 0.2)
    assert states_3d[0].battery_soc == 0.75

    states_2d = provider.get_states_2d(names=["robot0"])
    assert len(states_2d) == 1
    assert states_2d[0].position == (1.0, 2.0)
    assert states_2d[0].yaw == pytest.approx(1.57)
    assert states_2d[0].linear_velocity == (0.4, 0.5)
    assert states_2d[0].angular_velocity == 0.2


def test_fleet_state_provider_accepts_numpy_scalar_velocities():
    sim = FakeSim(
        [
            FakeAgent(
                "robot0",
                7,
                velocity=np.float64(0.4),
                angular_velocity=np.float64(0.2),
            )
        ]
    )

    state_3d = FleetStateProvider(sim).get_states()[0]
    state_2d = FleetStateProvider(sim).get_states_2d()[0]

    assert state_3d.linear_velocity == (0.4, 0.0, 0.0)
    assert state_3d.angular_velocity == (0.0, 0.0, 0.2)
    assert state_2d.angular_velocity == 0.2


def test_dispatcher_navigate_accepts_many_and_emits_before_mutation():
    robot0 = FakeAgent("robot0", 1)
    robot1 = FakeAgent("robot1", 2)
    sim = FakeSim([robot0, robot1])
    emitted = []
    sim.events.on(FLEET_COMMAND_EVENT, lambda command_event: emitted.append((command_event, len(robot0.goal_calls))))

    dispatcher = FleetCommandDispatcher(sim)
    ack = dispatcher.navigate(
        [
            RobotGoalCommand2D("robot0", position=(1.0, 2.0), yaw=0.5),
            RobotGoalCommand3D("robot1", position=(3.0, 4.0, 0.0)),
        ],
        source="test",
        command_id="cmd-1",
    )

    assert ack.ok
    assert ack.accepted_names == ("robot0", "robot1")
    assert ack.rejected == {}
    assert robot0.goal_calls[0].x == 1.0
    assert robot0.goal_calls[0].yaw == pytest.approx(0.5)
    assert robot1.goal_calls[0].position == [3.0, 4.0, 0.0]
    assert dispatcher.command_events[0].command_type == "navigate"
    assert emitted[0][0] == dispatcher.command_events[0]
    assert emitted[0][1] == 0


def test_dispatcher_rejects_unknown_duplicate_and_ambiguous_names():
    robot0 = FakeAgent("robot0", 1)
    duplicate_a = FakeAgent("dup", 2)
    duplicate_b = FakeAgent("dup", 3)
    sim = FakeSim([robot0, duplicate_a, duplicate_b])

    dispatcher = FleetCommandDispatcher(sim)
    ack = dispatcher.navigate(
        [
            RobotGoalCommand2D("robot0", position=(1.0, 0.0)),
            RobotGoalCommand2D("robot0", position=(2.0, 0.0)),
            RobotGoalCommand2D("missing", position=(3.0, 0.0)),
            RobotGoalCommand2D("dup", position=(4.0, 0.0)),
        ],
        command_id="cmd-2",
    )

    assert ack.accepted_names == ()
    assert ack.rejected == {
        "robot0": "duplicate target",
        "missing": "unknown robot",
        "dup": "ambiguous robot name",
    }
    assert robot0.goal_calls == []
    assert duplicate_a.goal_calls == []
    assert duplicate_b.goal_calls == []


def test_state_provider_handles_minimal_unnamed_agent_stub():
    class MinimalAgent:
        def get_pose(self):
            return Pose.from_xyz(0.0, 0.0, 0.0)

    states = FleetStateProvider(FakeSim([MinimalAgent()])).get_states()

    assert states[0].name == "agent_unknown"
    assert states[0].object_id == -1


def test_dispatcher_skips_unnamed_agents():
    class MinimalAgent:
        def __init__(self):
            self.goal_calls = []

        def get_pose(self):
            return Pose.from_xyz(0.0, 0.0, 0.0)

        def set_goal_pose(self, pose):
            self.goal_calls.append(pose)

    agent = MinimalAgent()
    dispatcher = FleetCommandDispatcher(FakeSim([agent]))

    ack = dispatcher.navigate([RobotGoalCommand2D("agent_unknown", position=(1.0, 0.0))])

    assert ack.accepted_names == ()
    assert ack.rejected == {"agent_unknown": "unknown robot"}
    assert agent.goal_calls == []


def test_dispatcher_joint_command_and_stop():
    robot0 = FakeAgent("robot0", 1)
    robot1 = FakeAgent("robot1", 2)
    dispatcher = FleetCommandDispatcher(FakeSim([robot0, robot1]))

    joint_ack = dispatcher.joint_command(
        [
            RobotJointPositionsCommand("robot0", positions=(0.1, 0.2)),
            RobotNamedJointPositionsCommand("robot1", positions={"joint": 1.5}),
        ],
        command_id="cmd-3",
    )
    stop_ack = dispatcher.stop(["robot0", "missing"], command_id="cmd-4")

    assert joint_ack.ok
    assert robot0.joint_calls == [("all", [0.1, 0.2], 500.0)]
    assert robot1.joint_calls == [("named", {"joint": 1.5}, 500.0)]
    assert stop_ack.accepted_names == ("robot0",)
    assert stop_ack.rejected == {"missing": "unknown robot"}
    assert robot0.stop_calls == 1


def test_dispatcher_allowlist_rejects_out_of_scope_targets():
    robot0 = FakeAgent("robot0", 1)
    robot1 = FakeAgent("robot1", 2)
    dispatcher = FleetCommandDispatcher(FakeSim([robot0, robot1]), allowed_names=["robot0"])

    ack = dispatcher.stop(["robot0", "robot1"], command_id="manager-stop")

    assert ack.accepted_names == ("robot0",)
    assert ack.rejected == {"robot1": "not managed by this interface"}
    assert robot0.stop_calls == 1
    assert robot1.stop_calls == 0


def test_dispatcher_attach_by_name_and_detach_attached_object():
    robot0 = FakeAgent("robot0", 1)
    box = FakeObject("box")
    sim = FakeSim([robot0])
    sim.sim_objects = [box]
    emitted = []
    sim.events.on(
        FLEET_COMMAND_EVENT,
        lambda command_event: emitted.append((command_event, len(robot0.attach_calls))),
    )
    dispatcher = FleetCommandDispatcher(sim)

    attach_ack = dispatcher.attach(
        [
            RobotAttachCommand(
                "robot0",
                attach=True,
                object_name="box",
                parent_link="tool",
                offset=Pose.from_xyz(0.0, 0.0, 0.2),
            )
        ],
        command_id="cmd-attach",
    )
    detach_ack = dispatcher.attach(
        [RobotAttachCommand("robot0", attach=False, object_name="box")],
        command_id="cmd-detach",
    )

    assert attach_ack.ok
    assert detach_ack.ok
    assert robot0.attach_calls[0][0] is box
    assert robot0.attach_calls[0][1] == "tool"
    assert robot0.attach_calls[0][2].z == pytest.approx(0.2)
    assert robot0.detach_calls == [box]
    assert emitted[0][0].command_type == "attach"
    assert emitted[0][1] == 0


def test_dispatcher_attach_defaults_non_positive_search_radius():
    robot0 = FakeAgent("robot0", 1)
    box = FakeObject("box")
    robot0.pickable_object = box
    dispatcher = FleetCommandDispatcher(FakeSim([robot0]))

    ack = dispatcher.attach(
        [RobotAttachCommand("robot0", attach=True, search_radius=-1.0)],
        command_id="cmd-attach-nearest",
    )

    assert ack.ok
    assert robot0.search_radii == [0.5]
    assert robot0.attach_calls[0][0] is box


def test_dispatcher_attach_rejects_missing_object_before_mutation():
    robot0 = FakeAgent("robot0", 1)
    dispatcher = FleetCommandDispatcher(FakeSim([robot0]))

    ack = dispatcher.attach(
        [RobotAttachCommand("robot0", attach=True, object_name="missing")],
        command_id="cmd-missing",
    )

    assert ack.accepted_names == ()
    assert ack.rejected == {"robot0": "object 'missing' not found"}
    assert robot0.attach_calls == []


def test_dispatcher_attach_rejects_failed_mutation():
    robot0 = FakeAgent("robot0", 1)
    robot0.attach_object = lambda *args, **kwargs: False
    box = FakeObject("box")
    sim = FakeSim([robot0])
    sim.sim_objects = [box]
    dispatcher = FleetCommandDispatcher(sim)

    ack = dispatcher.attach(
        [RobotAttachCommand("robot0", attach=True, object_name="box")],
        command_id="cmd-attach-fail",
    )

    assert ack.accepted_names == ()
    assert ack.rejected == {"robot0": "attach mutation failed"}
    assert dispatcher.command_events[0].accepted_names == ("robot0",)
    assert dispatcher.command_events[0].rejected == {}


def test_dispatcher_execute_action_queues_parsed_actions():
    robot0 = FakeAgent("robot0", 1)
    dispatcher = FleetCommandDispatcher(FakeSim([robot0]))

    ack = dispatcher.execute_action(
        [RobotActionCommand("robot0", "wait", '{"duration": 0.1}')],
        command_id="cmd-action",
    )

    assert ack.ok
    assert len(robot0.action_calls) == 1
    assert type(robot0.action_calls[0]).__name__ == "WaitAction"
    assert dispatcher.command_events[0].command_type == "execute_action"


def test_dispatcher_execute_action_rejects_invalid_actions():
    robot0 = FakeAgent("robot0", 1)
    dispatcher = FleetCommandDispatcher(FakeSim([robot0]))

    ack = dispatcher.execute_action(
        [RobotActionCommand("robot0", "missing_action")],
        command_id="cmd-invalid",
    )

    assert ack.accepted_names == ()
    assert ack.rejected == {"robot0": "invalid action 'missing_action'"}
    assert robot0.action_calls == []


def test_command_ack_and_event_rejected_maps_are_immutable():
    dispatcher = FleetCommandDispatcher(FakeSim([]))

    ack = dispatcher.stop(["missing"], command_id="cmd-5")
    event = dispatcher.command_events[0]

    assert ack.rejected == {"missing": "unknown robot"}
    assert event.rejected == {"missing": "unknown robot"}
    with pytest.raises(TypeError):
        cast(dict[str, str], ack.rejected)["other"] = "mutated"
    with pytest.raises(TypeError):
        cast(dict[str, str], event.rejected)["other"] = "mutated"


def test_named_joint_command_positions_are_immutable():
    command = RobotNamedJointPositionsCommand("robot0", positions={"joint": 1.5})

    assert command.positions == {"joint": 1.5}
    with pytest.raises(TypeError):
        cast(dict[str, float], command.positions)["other"] = 0.1


def test_a_planar_goal_without_z_keeps_the_robots_height():
    robot0 = FakeAgent("robot0", 1, pose=Pose.from_xyz(0.0, 0.0, 0.3))
    robot1 = FakeAgent("robot1", 2, pose=Pose.from_xyz(0.0, 0.0, 0.3))
    dispatcher = FleetCommandDispatcher(FakeSim([robot0, robot1]))
    ack = dispatcher.navigate(
        [RobotGoalCommand2D("robot0", position=(1.0, 0.0)), RobotGoalCommand2D("robot1", position=(1.0, 0.0), z=0.0)]
    )

    assert ack.ok
    assert robot0.goal_calls[0].position[2] == pytest.approx(0.3)
    assert robot1.goal_calls[0].position[2] == 0.0  # an explicit height is kept


def test_dispatcher_reports_a_goal_the_robot_cannot_take_as_rejected():
    """A moving robot refuses a goal it cannot reach within its limits; the
    ack says so and the robot is not touched."""

    class MovingAgent(FakeAgent):
        def path_rejection(self, path):
            return PathRejection("goal is behind the motion", temporary=True)

    robot0 = MovingAgent("robot0", 1)
    robot1 = FakeAgent("robot1", 2)
    dispatcher = FleetCommandDispatcher(FakeSim([robot0, robot1]))
    ack = dispatcher.navigate(
        [RobotGoalCommand2D("robot0", position=(1.0, 0.0)), RobotGoalCommand2D("robot1", position=(2.0, 0.0))]
    )

    assert ack.accepted_names == ("robot1",)
    assert ack.rejected == {"robot0": "goal is behind the motion"}
    assert robot0.goal_calls == []
    assert len(robot1.goal_calls) == 1


def test_a_goal_refused_after_the_command_event_is_not_acknowledged():
    """The command event is emitted before the goals are applied. If a handler
    of it changes a robot so that it refuses its goal, the returned ack and the
    recorded input must not claim the goal was taken."""

    class RefusingAgent(FakeAgent):
        refuse = False

        def set_goal_pose(self, pose):
            if self.refuse:
                return False
            return super().set_goal_pose(pose)

        def path_rejection(self, path):
            return PathRejection("goal is behind the motion", temporary=True) if self.refuse else None

    robot0 = RefusingAgent("robot0", 1)
    robot1 = FakeAgent("robot1", 2)
    sim = FakeSim([robot0, robot1])
    recorded = []
    sim._record_state_input = lambda operation, details: recorded.append(details)
    announced = []

    def on_command(command_event):
        announced.append(command_event.accepted_names)
        robot0.refuse = True  # e.g. a handler that sends robot0 elsewhere

    sim.events.on(FLEET_COMMAND_EVENT, on_command)
    ack = FleetCommandDispatcher(sim).navigate(
        [RobotGoalCommand2D("robot0", position=(1.0, 0.0)), RobotGoalCommand2D("robot1", position=(2.0, 0.0))]
    )

    assert announced == [("robot0", "robot1")]
    assert ack.accepted_names == ("robot1",)
    assert ack.rejected == {"robot0": "goal is behind the motion"}
    assert set(recorded[0]["accepted"]) == {"robot1"}
    assert recorded[0]["rejected"] == {"robot0": "goal is behind the motion"}
