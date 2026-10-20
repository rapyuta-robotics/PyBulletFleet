"""Tests for fleet-level ROS wrappers.

Most tests require generated ``pybullet_fleet_msgs`` Python modules, so only
those tests skip when the bindings are unavailable.
"""

from dataclasses import dataclass, field
from unittest.mock import ANY, MagicMock

import pytest

try:
    from geometry_msgs.msg import Point, Pose as RosPose, Quaternion
    from pybullet_fleet_ros import fleet_ros_interface as fleet_ros_interface_module
    from pybullet_fleet_ros.fleet_ros_interface import FleetRosInterface

    _GEOMETRY_MSGS_IMPORT_ERROR = None
except ImportError as exc:
    Point = None
    RosPose = None
    Quaternion = None
    fleet_ros_interface_module = None
    FleetRosInterface = None
    _GEOMETRY_MSGS_IMPORT_ERROR = exc

from pybullet_fleet.geometry import Pose
from pybullet_fleet_ros.interface_config import FleetApiConfig, ManagerInterfaceConfig


def _require_geometry_msgs():
    if _GEOMETRY_MSGS_IMPORT_ERROR is not None:
        pytest.skip("geometry_msgs not available")


def _fleet_msg_types():
    _require_geometry_msgs()
    msgs = pytest.importorskip("pybullet_fleet_msgs.msg", reason="pybullet_fleet_msgs not generated")
    srvs = pytest.importorskip("pybullet_fleet_msgs.srv", reason="pybullet_fleet_msgs not generated")
    return msgs, srvs


@dataclass
class FakeObject:
    name: str


@dataclass
class FakeAgent:
    name: str
    object_id: int
    pose: Pose = field(default_factory=lambda: Pose.from_xyz(0.0, 0.0, 0.0))
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: float = 0.0
    is_moving: bool = False

    def __post_init__(self):
        self.goal_calls = []
        self.joint_calls = []
        self.stop_calls = 0
        self.attach_calls = []
        self.detach_calls = []
        self.action_calls = []
        self.attached_objects = []
        self.pickable_object = None
        self.search_radii = []
        self.sim_core = None

    def get_pose(self):
        return self.pose

    def set_goal_pose(self, pose):
        self.goal_calls.append(pose)

    def path_rejection(self, path):
        return None  # at rest: every path is taken

    def set_joints_targets_by_name(self, positions):
        self.joint_calls.append(dict(positions))

    def stop(self):
        self.stop_calls += 1

    def find_nearest_pickable(self, search_radius=0.5):
        self.search_radii.append(search_radius)
        return self.pickable_object

    def attach_object(self, obj, parent_link_index="base_link", relative_pose=None):
        self.attach_calls.append((obj, parent_link_index, relative_pose))
        self.attached_objects.append(obj)
        return True

    def detach_object(self, obj):
        self.detach_calls.append(obj)
        self.attached_objects.remove(obj)
        return True

    def get_attached_objects(self):
        return list(self.attached_objects)

    def add_action(self, action):
        self.action_calls.append(action)


class FakeSim:
    def __init__(self, agents):
        self.agents = agents
        self.sim_time = 12.5
        self.sim_objects = []
        self.last_profiling = None
        self.profile_records = []
        for agent in self.agents:
            agent.sim_core = self

    def record_profiling(self, name, value_ms):
        self.profile_records.append((name, value_ms))

    def _record_state_input(self, operation, details):
        pass


def _node():
    node = MagicMock()
    node.get_logger.return_value = MagicMock()
    node.create_publisher.return_value = MagicMock()
    node.create_subscription.return_value = MagicMock()
    node.create_service.return_value = MagicMock()
    return node


def test_disabled_fleet_api_does_not_require_generated_message_bindings(monkeypatch):
    _require_geometry_msgs()
    monkeypatch.setattr(fleet_ros_interface_module, "_FLEET_MSGS_IMPORT_ERROR", ImportError("missing"))

    interface = FleetRosInterface(_node(), FakeSim([]), FleetApiConfig(enabled=False))

    assert interface._state_pub is None


def test_disabled_fleet_api_does_not_construct_provider_or_dispatcher(monkeypatch):
    _require_geometry_msgs()
    state_provider = MagicMock(side_effect=AssertionError("state provider should not be created"))
    dispatcher = MagicMock(side_effect=AssertionError("dispatcher should not be created"))
    monkeypatch.setattr(fleet_ros_interface_module, "FleetStateProvider", state_provider)
    monkeypatch.setattr(fleet_ros_interface_module, "FleetCommandDispatcher", dispatcher)

    FleetRosInterface(_node(), FakeSim([FakeAgent("robot0", 1)]), FleetApiConfig(enabled=False))

    state_provider.assert_not_called()
    dispatcher.assert_not_called()


def test_enabled_fleet_api_requires_generated_message_bindings(monkeypatch):
    _require_geometry_msgs()
    monkeypatch.setattr(fleet_ros_interface_module, "_FLEET_MSGS_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="pybullet_fleet_msgs Python bindings are unavailable"):
        FleetRosInterface(_node(), FakeSim([]), FleetApiConfig(enabled=True, states=True))


def test_fleet_state_publisher_uses_provider():
    _fleet_msg_types()
    node = _node()
    agent = FakeAgent("robot0", 7, pose=Pose.from_yaw(1.0, 2.0, 0.3, 1.57), velocity=(0.4, 0.5, 0.0))
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, states=True))

    interface.post_step()

    published = node.create_publisher.return_value.publish.call_args.args[0]
    assert published.robots[0].name == "robot0"
    assert published.robots[0].object_id == 7
    assert published.robots[0].pose.position.x == pytest.approx(1.0)
    assert published.robots[0].twist.linear.x == pytest.approx(0.4)


def test_fleet_state_publisher_records_core_profile_subphases():
    _fleet_msg_types()
    node = _node()
    sim = FakeSim([FakeAgent("robot0", 7)])
    # ``last_profiling`` is populated after the first profiled core step. The
    # bridge records into the current step only while that standard profiler is
    # active.
    sim.last_profiling = {}
    interface = FleetRosInterface(node, sim, FleetApiConfig(enabled=True, states=True))

    interface.post_step()

    assert [name for name, _ in sim.profile_records] == [
        "fleet_state_collect",
        "fleet_state_message",
        "fleet_state_publish",
    ]
    assert all(value_ms >= 0.0 for _, value_ms in sim.profile_records)


def test_manager_state_publisher_filters_members_and_uses_namespace():
    _fleet_msg_types()
    node = _node()
    node.create_publisher.return_value.get_subscription_count.return_value = 1
    delivery = FakeAgent("delivery_0", 1)
    npc = FakeAgent("npc_0", 2)
    interface = FleetRosInterface(
        node,
        FakeSim([delivery, npc]),
        ManagerInterfaceConfig(manager="delivery", states=True),
        namespace="/fleet/delivery",
        state_names=["delivery_0"],
        command_names=["delivery_0"],
    )

    interface.post_step()

    node.create_publisher.assert_called_once()
    assert node.create_publisher.call_args.args[1] == "/fleet/delivery/states"
    published = node.create_publisher.return_value.publish.call_args.args[0]
    assert [robot.name for robot in published.robots] == ["delivery_0"]


def test_state_publisher_skips_collection_without_subscribers(monkeypatch):
    _fleet_msg_types()
    node = _node()
    node.create_publisher.return_value.get_subscription_count.return_value = 0
    provider = MagicMock()
    monkeypatch.setattr(fleet_ros_interface_module, "FleetStateProvider", provider)
    interface = FleetRosInterface(node, FakeSim([FakeAgent("robot0", 1)]), FleetApiConfig(enabled=True, states=True))

    interface.post_step()

    provider.return_value.get_states_3d.assert_not_called()
    node.create_publisher.return_value.publish.assert_not_called()


def test_manager_state_rate_limit_skips_subscription_query_until_due():
    _fleet_msg_types()
    from pybullet_fleet_ros.conversions import sim_time_to_ros_time

    node = _node()
    node.create_publisher.return_value.get_subscription_count.return_value = 1
    interface = FleetRosInterface(
        node,
        FakeSim([FakeAgent("delivery_0", 1)]),
        ManagerInterfaceConfig(manager="delivery", states=True, state_publish_rate=1.0),
        state_names=["delivery_0"],
    )

    interface.post_step(stamp=sim_time_to_ros_time(0.0))
    node.create_publisher.return_value.get_subscription_count.assert_called_once_with()
    node.create_publisher.return_value.publish.assert_called_once_with(ANY)
    node.create_publisher.return_value.get_subscription_count.reset_mock()
    node.create_publisher.return_value.publish.reset_mock()

    interface.post_step(stamp=sim_time_to_ros_time(0.5))

    node.create_publisher.return_value.get_subscription_count.assert_not_called()
    node.create_publisher.return_value.publish.assert_not_called()


def test_fleet_state_publisher_applies_frame_offset():
    _fleet_msg_types()
    node = _node()
    node.rmf_frame_offset = (100.0, 200.0)
    agent = FakeAgent("robot0", 7, pose=Pose.from_xyz(1.0, 2.0, 0.3))
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, states=True))

    interface.post_step()

    published = node.create_publisher.return_value.publish.call_args.args[0]
    assert published.robots[0].pose.position.x == pytest.approx(101.0)
    assert published.robots[0].pose.position.y == pytest.approx(202.0)


def test_fleet_navigate_service_dispatches_2d_goal():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    agent = FakeAgent("robot0", 1)
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, navigate=True))
    request = srvs.FleetNavigate.Request()
    request.command_id = "cmd-1"
    request.source = "test"
    request.goals_2d = [msgs.RobotGoal2D(name="robot0", position=[1.0, 2.0], yaw=0.5, z=0.0)]
    response = srvs.FleetNavigate.Response()

    result = interface._on_navigate_service(request, response)

    assert result.ack.command_id == "cmd-1"
    assert result.ack.accepted_names == ["robot0"]
    assert agent.goal_calls[0].x == pytest.approx(1.0)
    assert agent.goal_calls[0].yaw == pytest.approx(0.5)


def test_manager_navigate_service_rejects_out_of_scope_target():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    delivery = FakeAgent("delivery_0", 1)
    npc = FakeAgent("npc_0", 2)
    interface = FleetRosInterface(
        node,
        FakeSim([delivery, npc]),
        ManagerInterfaceConfig(manager="delivery", navigate=True),
        namespace="/fleet/delivery",
        command_names=["delivery_0"],
    )
    request = srvs.FleetNavigate.Request()
    request.goals_2d = [msgs.RobotGoal2D(name="npc_0", position=[1.0, 2.0], yaw=0.0, z=0.0)]

    result = interface._on_navigate_service(request, srvs.FleetNavigate.Response())

    assert result.ack.accepted_names == []
    assert result.ack.rejected_names == ["npc_0"]
    assert result.ack.reject_reasons == ["not managed by this interface"]
    assert npc.goal_calls == []


def test_fleet_navigate_service_applies_frame_offset():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    node.rmf_frame_offset = (100.0, 200.0)
    agent = FakeAgent("robot0", 1)
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, navigate=True))
    request = srvs.FleetNavigate.Request()
    request.goals_2d = [msgs.RobotGoal2D(name="robot0", position=[101.0, 202.0], yaw=0.5, z=0.0)]
    response = srvs.FleetNavigate.Response()

    interface._on_navigate_service(request, response)

    assert agent.goal_calls[0].x == pytest.approx(1.0)
    assert agent.goal_calls[0].y == pytest.approx(2.0)


def test_fleet_navigate_service_rejects_unsupported_frame():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    agent = FakeAgent("robot0", 1)
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, navigate=True))
    request = srvs.FleetNavigate.Request()
    request.header.frame_id = "map"
    request.command_id = "cmd-map"
    request.goals_2d = [msgs.RobotGoal2D(name="robot0", position=[1.0, 2.0], yaw=0.5, z=0.0)]
    response = srvs.FleetNavigate.Response()

    result = interface._on_navigate_service(request, response)

    assert result.ack.command_id == "cmd-map"
    assert result.ack.accepted_names == []
    assert result.ack.rejected_names == ["robot0"]
    assert "unsupported frame_id 'map'" in result.ack.reject_reasons[0]
    assert agent.goal_calls == []


def test_fleet_navigate_service_rejects_unsupported_frame_with_empty_command_id():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    agent = FakeAgent("robot0", 1)
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, navigate=True))
    request = srvs.FleetNavigate.Request()
    request.header.frame_id = "map"
    request.goals_2d = [msgs.RobotGoal2D(name="robot0", position=[1.0, 2.0], yaw=0.5, z=0.0)]
    response = srvs.FleetNavigate.Response()

    result = interface._on_navigate_service(request, response)

    assert result.ack.command_id == ""
    assert result.ack.rejected_names == ["robot0"]
    assert agent.goal_calls == []


def test_fleet_navigate_topic_logs_rejected_targets():
    msgs, _ = _fleet_msg_types()
    node = _node()
    interface = FleetRosInterface(node, FakeSim([FakeAgent("robot0", 1)]), FleetApiConfig(enabled=True, navigate=True))
    request = msgs.FleetNavigate()
    request.command_id = "cmd-reject"
    request.goals_2d = [msgs.RobotGoal2D(name="missing", position=[1.0, 2.0], yaw=0.5, z=0.0)]

    interface._on_navigate(request)

    warning = node.get_logger.return_value.warning
    warning.assert_called_once()
    assert "cmd-reject" in warning.call_args.args[0]
    assert "missing: unknown robot" in warning.call_args.args[0]


def test_fleet_navigate_service_dispatches_3d_pose_goal():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    node.rmf_frame_offset = (100.0, 200.0)
    agent = FakeAgent("robot0", 1)
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, navigate=True))
    request = srvs.FleetNavigate.Request()
    request.goals_3d = [
        msgs.RobotGoal3D(
            name="robot0",
            pose=RosPose(
                position=Point(x=101.0, y=202.0, z=0.3),
                orientation=Quaternion(x=0.0, y=0.0, z=0.707, w=0.707),
            ),
        )
    ]
    response = srvs.FleetNavigate.Response()

    result = interface._on_navigate_service(request, response)

    assert result.ack.accepted_names == ["robot0"]
    assert agent.goal_calls[0].position == pytest.approx([1.0, 2.0, 0.3])
    assert agent.goal_calls[0].orientation == pytest.approx([0.0, 0.0, 0.707, 0.707])


def test_fleet_stop_service_dispatches_named_targets():
    _, srvs = _fleet_msg_types()
    node = _node()
    robot0 = FakeAgent("robot0", 1)
    robot1 = FakeAgent("robot1", 2)
    interface = FleetRosInterface(
        node,
        FakeSim([robot0, robot1]),
        FleetApiConfig(enabled=True, stop=True),
    )
    request = srvs.FleetStop.Request()
    request.command_id = "stop-1"
    request.source = "test"
    request.names = ["robot0", "missing"]
    response = srvs.FleetStop.Response()

    result = interface._on_stop_service(request, response)

    assert result.ack.command_id == "stop-1"
    assert result.ack.source == "test"
    assert result.ack.accepted_names == ["robot0"]
    assert result.ack.rejected_names == ["missing"]
    assert robot0.stop_calls == 1
    assert robot1.stop_calls == 0


def test_fleet_attach_service_dispatches_object_commands():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    robot0 = FakeAgent("robot0", 1)
    box = FakeObject("box")
    sim = FakeSim([robot0])
    sim.sim_objects = [box]
    interface = FleetRosInterface(
        node,
        sim,
        FleetApiConfig(enabled=True, attach=True),
    )
    request = srvs.FleetAttach.Request()
    request.command_id = "attach-1"
    request.source = "test"
    command = msgs.RobotAttachCommand()
    command.name = "robot0"
    command.attach = True
    command.object_name = "box"
    command.parent_link = "tool"
    command.offset.position.z = 0.2
    request.commands = [command]
    response = srvs.FleetAttach.Response()

    result = interface._on_attach_service(request, response)

    assert result.ack.command_id == "attach-1"
    assert result.ack.accepted_names == ["robot0"]
    assert result.ack.rejected_names == []
    assert robot0.attach_calls[0][0] is box
    assert robot0.attach_calls[0][1] == "tool"
    assert robot0.attach_calls[0][2].z == pytest.approx(0.2)


def test_fleet_attach_service_defaults_non_positive_search_radius():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    robot0 = FakeAgent("robot0", 1)
    robot0.pickable_object = FakeObject("box")
    sim = FakeSim([robot0])
    interface = FleetRosInterface(
        node,
        sim,
        FleetApiConfig(enabled=True, attach=True),
    )
    request = srvs.FleetAttach.Request()
    command = msgs.RobotAttachCommand()
    command.name = "robot0"
    command.attach = True
    command.search_radius = -1.0
    request.commands = [command]
    response = srvs.FleetAttach.Response()

    result = interface._on_attach_service(request, response)

    assert result.ack.accepted_names == ["robot0"]
    assert robot0.search_radii == [0.5]


def test_fleet_execute_action_service_dispatches_action_commands():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    robot0 = FakeAgent("robot0", 1)
    interface = FleetRosInterface(
        node,
        FakeSim([robot0]),
        FleetApiConfig(enabled=True, execute_action=True),
    )
    request = srvs.FleetExecuteAction.Request()
    request.command_id = "action-1"
    request.source = "test"
    request.commands = [
        msgs.RobotActionCommand(
            name="robot0",
            action_type="wait",
            action_params_json='{"duration": 0.1}',
        )
    ]
    response = srvs.FleetExecuteAction.Response()

    result = interface._on_execute_action_service(request, response)

    assert result.ack.command_id == "action-1"
    assert result.ack.accepted_names == ["robot0"]
    assert result.ack.rejected_names == []
    assert len(robot0.action_calls) == 1
    assert type(robot0.action_calls[0]).__name__ == "WaitAction"


def test_named_joint_service_rejects_mismatched_arrays():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    interface = FleetRosInterface(
        node,
        FakeSim([FakeAgent("robot0", 1)]),
        FleetApiConfig(enabled=True, joint_command=True),
    )
    request = srvs.FleetJointCommand.Request()
    request.command_id = "cmd-2"
    request.named_joint_position_commands = [
        msgs.RobotNamedJointPositionsCommand(name="robot0", joint_names=["joint1", "joint2"], positions=[1.0])
    ]
    response = srvs.FleetJointCommand.Response()

    result = interface._on_joint_command_service(request, response)

    assert result.ack.command_id == "cmd-2"
    assert result.ack.accepted_names == []
    assert result.ack.rejected_names == ["robot0"]
    assert "2 joint names" in result.ack.reject_reasons[0]


def test_named_joint_service_rejects_mismatched_arrays_with_empty_command_id():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    interface = FleetRosInterface(
        node,
        FakeSim([FakeAgent("robot0", 1)]),
        FleetApiConfig(enabled=True, joint_command=True),
    )
    request = srvs.FleetJointCommand.Request()
    request.named_joint_position_commands = [
        msgs.RobotNamedJointPositionsCommand(name="robot0", joint_names=["joint1", "joint2"], positions=[1.0])
    ]
    response = srvs.FleetJointCommand.Response()

    result = interface._on_joint_command_service(request, response)

    assert result.ack.command_id == ""
    assert result.ack.rejected_names == ["robot0"]
    assert "2 joint names" in result.ack.reject_reasons[0]


def test_named_joint_service_rejects_invalid_robot_but_dispatches_valid_command():
    msgs, srvs = _fleet_msg_types()
    node = _node()
    robot0 = FakeAgent("robot0", 1)
    robot1 = FakeAgent("robot1", 2)
    interface = FleetRosInterface(
        node,
        FakeSim([robot0, robot1]),
        FleetApiConfig(enabled=True, joint_command=True),
    )
    request = srvs.FleetJointCommand.Request()
    request.command_id = "cmd-mixed"
    request.named_joint_position_commands = [
        msgs.RobotNamedJointPositionsCommand(name="robot0", joint_names=["joint1"], positions=[1.0]),
        msgs.RobotNamedJointPositionsCommand(name="robot1", joint_names=["joint1", "joint2"], positions=[2.0]),
    ]
    response = srvs.FleetJointCommand.Response()

    result = interface._on_joint_command_service(request, response)

    assert result.ack.accepted_names == ["robot0"]
    assert result.ack.rejected_names == ["robot1"]
    assert robot0.joint_calls == [{"joint1": 1.0}]
    assert robot1.joint_calls == []


def test_named_joint_topic_logs_rejected_request():
    msgs, _ = _fleet_msg_types()
    node = _node()
    interface = FleetRosInterface(
        node,
        FakeSim([FakeAgent("robot0", 1)]),
        FleetApiConfig(enabled=True, joint_command=True),
    )
    request = msgs.FleetJointCommand()
    request.command_id = "cmd-2"
    request.named_joint_position_commands = [
        msgs.RobotNamedJointPositionsCommand(name="robot0", joint_names=["joint1", "joint2"], positions=[1.0])
    ]

    interface._on_joint_command(request)

    warning = node.get_logger.return_value.warning
    warning.assert_called_once()
    assert "cmd-2" in warning.call_args.args[0]
    assert "robot0" in warning.call_args.args[0]
    assert "2 joint names" in warning.call_args.args[0]
