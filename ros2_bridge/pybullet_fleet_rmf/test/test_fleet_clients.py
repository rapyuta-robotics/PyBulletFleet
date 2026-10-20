"""Unit tests for RMF client factory and fleet ROS client."""

import math
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pybullet_fleet_rmf.fleet_clients", reason="ROS 2 / RMF message dependencies not available")

from geometry_msgs.msg import Point, Pose, Quaternion
from pybullet_fleet_msgs.msg import FleetState, RobotState3D


def _node_with_clients(*clients):
    node = MagicMock()
    remaining = list(clients)
    node.create_subscription.side_effect = lambda *args, **kwargs: MagicMock()
    node.create_client.side_effect = lambda *args, **kwargs: remaining.pop(0) if remaining else MagicMock()
    return node


def _fleet_state(robot_name="tinyRobot1", x=1.0, y=2.0, yaw=0.0, battery_soc=0.5):
    state = FleetState()
    robot = RobotState3D()
    robot.name = robot_name
    robot.pose = Pose(
        position=Point(x=x, y=y, z=0.0),
        orientation=Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)),
    )
    robot.has_battery_soc = True
    robot.battery_soc = battery_soc
    state.robots = [robot]
    return state


class _FakeObject:
    def __init__(self, name: str):
        self.name = name


class _FakeAgent:
    def __init__(self, name: str, object_id: int = 1):
        from pybullet_fleet.geometry import Pose as PbfPose

        self.name = name
        self.object_id = object_id
        self.velocity = (0.0, 0.0, 0.0)
        self.angular_velocity = 0.0
        self.is_moving = False
        self.battery_soc = 0.75
        self.is_charging = False
        self.pose = PbfPose.from_xyz(0.0, 0.0, 0.0)
        self.goal_calls = []
        self.stop_calls = 0
        self.attach_calls = []
        self.detach_calls = []
        self.attached = []
        self.search_radii = []
        self.sim_core = None

    def get_pose(self):
        return self.pose

    def set_goal_pose(self, pose):
        self.goal_calls.append(pose)
        self.is_moving = True

    def path_rejection(self, path):
        return None  # at rest: every path is taken

    def stop(self):
        self.stop_calls += 1
        self.is_moving = False

    def attach_object(self, obj, parent_link_index="base_link", relative_pose=None):
        self.attach_calls.append((obj, parent_link_index, relative_pose))
        self.attached.append(obj)
        return True

    def detach_object(self, obj):
        self.detach_calls.append(obj)
        if obj in self.attached:
            self.attached.remove(obj)
        return True

    def get_attached_objects(self):
        return list(self.attached)

    def find_nearest_pickable(self, search_radius=0.5):
        self.search_radii.append(search_radius)
        return self.sim_core.sim_objects[0] if self.sim_core.sim_objects else None

    def set_charging(self, charging: bool):
        self.is_charging = bool(charging)


class _FakeSim:
    def __init__(self, agents, objects=None):
        self.agents = list(agents)
        self.sim_objects = list(objects or [])
        self.sim_time = 1.25
        for agent in self.agents:
            agent.sim_core = self

    def _record_state_input(self, operation, details):
        pass


def test_client_factory_defaults_to_per_robot_ros(mock_node):
    from pybullet_fleet_rmf.fleet_clients import PerRobotRosClientFactory, create_rmf_client_factory
    from pybullet_fleet_rmf.per_robot_ros_client import PerRobotRosClient

    factory = create_rmf_client_factory("", mock_node)
    assert isinstance(factory, PerRobotRosClientFactory)
    assert isinstance(factory.robot("tinyRobot1"), PerRobotRosClient)


def test_client_factory_creates_fleet_ros_client():
    from pybullet_fleet_rmf.fleet_clients import RosRmfFleetClient, create_rmf_client_factory

    nav_client = MagicMock()
    stop_client = MagicMock()
    attach_client = MagicMock()
    node = _node_with_clients(nav_client, stop_client, attach_client)

    factory = create_rmf_client_factory("fleet_ros", node, fleet_namespace="/fleet/tinyRobot")

    assert isinstance(factory, RosRmfFleetClient)
    node.create_subscription.assert_called_once()
    assert node.create_subscription.call_args.args[1] == "/fleet/tinyRobot/states"
    assert node.create_client.call_count == 3
    assert node.create_client.call_args_list[0].args[1] == "/fleet/tinyRobot/navigate"
    assert node.create_client.call_args_list[1].args[1] == "/fleet/tinyRobot/stop"
    assert node.create_client.call_args_list[2].args[1] == "/fleet/tinyRobot/attach"


def test_client_factory_creates_python_fleet_client(mock_node):
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient, create_rmf_client_factory

    sim = _FakeSim([_FakeAgent("tinyRobot1")])

    factory = create_rmf_client_factory("python_fleet", mock_node, sim_core=sim)

    assert isinstance(factory, PythonRmfFleetClient)
    assert not mock_node.create_subscription.called
    assert not mock_node.create_client.called


def test_client_factory_rejects_aliases(mock_node):
    from pybullet_fleet_rmf.fleet_clients import create_rmf_client_factory

    for mode in ("per_robot", "legacy", "legacy_ros", "fleet", "ros_fleet"):
        with pytest.raises(ValueError):
            create_rmf_client_factory(mode, mock_node)


def test_ros_fleet_client_reads_shared_fleet_state():
    from pybullet_fleet_rmf.fleet_clients import RosRmfFleetClient

    nav_client = MagicMock()
    node = _node_with_clients(nav_client, MagicMock(), MagicMock(), MagicMock())
    fleet = RosRmfFleetClient(node, map_name="L1")
    state_cb = node.create_subscription.call_args.args[2]
    robot = fleet.robot("tinyRobot1")

    state_cb(_fleet_state(x=1.5, y=-2.0, yaw=0.25, battery_soc=0.42))
    data = robot.get_data()

    assert data is not None
    assert data.map == "L1"
    assert data.position[0] == pytest.approx(1.5)
    assert data.position[1] == pytest.approx(-2.0)
    assert data.position[2] == pytest.approx(0.25)
    assert data.battery_soc == pytest.approx(0.42)


def test_ros_fleet_client_tracks_map_name_per_robot():
    from pybullet_fleet_rmf.fleet_clients import RosRmfFleetClient

    nav_client = MagicMock()
    node = _node_with_clients(nav_client, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    fleet = RosRmfFleetClient(node, map_name="L1")
    state_cb = node.create_subscription.call_args.args[2]
    first = fleet.robot("tinyRobot1")
    second = fleet.robot("tinyRobot2")

    first.set_map_name("L2")
    state = FleetState()
    state.robots = [
        _fleet_state("tinyRobot1").robots[0],
        _fleet_state("tinyRobot2").robots[0],
    ]
    state_cb(state)

    assert first.get_data().map == "L2"
    assert second.get_data().map == "L1"


def test_ros_fleet_robot_client_navigates_through_fleet_service():
    from pybullet_fleet_rmf.fleet_clients import RosRmfFleetClient

    future = MagicMock()
    nav_client = MagicMock()
    nav_client.service_is_ready.return_value = True
    nav_client.call_async.return_value = future
    node = _node_with_clients(nav_client, MagicMock(), MagicMock(), MagicMock())
    fleet = RosRmfFleetClient(node, map_name="L1")
    state_cb = node.create_subscription.call_args.args[2]
    robot = fleet.robot("tinyRobot1")
    state_cb(_fleet_state(x=0.0, y=0.0))

    assert robot.navigate(12, [1.0, 2.0, 0.3], "L1") is True

    nav_client.call_async.assert_called_once()
    req = nav_client.call_async.call_args.args[0]
    assert req.header.frame_id == "odom"
    assert req.command_id == "12"
    assert req.source == "rmf"
    assert len(req.goals_2d) == 1
    assert req.goals_2d[0].name == "tinyRobot1"
    assert list(req.goals_2d[0].position) == pytest.approx([1.0, 2.0])
    assert req.goals_2d[0].yaw == pytest.approx(0.3)
    future.add_done_callback.assert_called_once()


def test_ros_fleet_robot_client_stops_through_fleet_service():
    from pybullet_fleet_rmf.fleet_clients import RosRmfFleetClient

    future = MagicMock()
    nav_client = MagicMock()
    stop_client = MagicMock()
    stop_client.service_is_ready.return_value = True
    stop_client.call_async.return_value = future
    node = _node_with_clients(nav_client, stop_client)
    fleet = RosRmfFleetClient(node, map_name="L1")
    robot = fleet.robot("tinyRobot1")

    assert robot.stop() is True

    stop_client.call_async.assert_called_once()
    req = stop_client.call_async.call_args.args[0]
    assert req.header.frame_id == "odom"
    assert req.command_id == "1"
    assert req.source == "rmf"
    assert req.names == ["tinyRobot1"]
    future.add_done_callback.assert_called_once()


def test_ros_fleet_robot_client_attaches_through_fleet_service():
    from pybullet_fleet_rmf.fleet_clients import RosRmfFleetClient

    future = MagicMock()
    nav_client = MagicMock()
    stop_client = MagicMock()
    attach_client = MagicMock()
    attach_client.service_is_ready.return_value = True
    attach_client.call_async.return_value = future
    node = _node_with_clients(nav_client, stop_client, attach_client)
    fleet = RosRmfFleetClient(node, map_name="L1")
    robot = fleet.robot("tinyRobot1")

    assert robot.attach_object(
        True,
        7,
        object_name="box",
        parent_link="tool",
        offset_position=(0.0, 0.0, 0.2),
        search_radius=0.75,
    )

    attach_client.call_async.assert_called_once()
    req = attach_client.call_async.call_args.args[0]
    assert req.header.frame_id == "odom"
    assert req.command_id == "7"
    assert req.source == "rmf"
    assert len(req.commands) == 1
    assert req.commands[0].name == "tinyRobot1"
    assert req.commands[0].attach is True
    assert req.commands[0].object_name == "box"
    assert req.commands[0].parent_link == "tool"
    assert req.commands[0].offset.position.z == pytest.approx(0.2)
    assert req.commands[0].search_radius == pytest.approx(0.75)
    future.add_done_callback.assert_called_once()


def test_ros_fleet_robot_client_marks_navigation_complete_from_state():
    from pybullet_fleet_rmf.fleet_clients import RosRmfFleetClient

    nav_client = MagicMock()
    nav_client.service_is_ready.return_value = True
    nav_client.call_async.return_value = MagicMock()
    node = _node_with_clients(nav_client, MagicMock(), MagicMock(), MagicMock())
    fleet = RosRmfFleetClient(node, map_name="L1")
    state_cb = node.create_subscription.call_args.args[2]
    robot = fleet.robot("tinyRobot1")

    state_cb(_fleet_state(x=0.0, y=0.0))
    assert robot.navigate(4, [1.0, 0.0, 0.0], "L1") is True
    assert not robot.get_data().is_command_completed(4)

    state_cb(_fleet_state(x=1.0, y=0.0))
    assert robot.get_data().is_command_completed(4)


def test_python_fleet_client_reads_provider_state_and_tracks_map(mock_node):
    del mock_node
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient

    agent = _FakeAgent("tinyRobot1")
    sim = _FakeSim([agent])
    fleet = PythonRmfFleetClient.from_sim_core(sim, map_name="L1")
    robot = fleet.robot("tinyRobot1")

    robot.set_map_name("L2")
    data = robot.get_data()

    assert data is not None
    assert data.map == "L2"
    assert data.position == pytest.approx([0.0, 0.0, 0.0])
    assert data.battery_soc == pytest.approx(0.75)


def test_python_fleet_client_applies_rmf_frame_offset(mock_node):
    del mock_node
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient

    agent = _FakeAgent("tinyRobot1")
    sim = _FakeSim([agent])
    robot = PythonRmfFleetClient.from_sim_core(
        sim,
        map_name="L1",
        rmf_frame_offset=(100.0, 200.0),
    ).robot("tinyRobot1")

    data = robot.get_data()
    assert data is not None
    assert data.position == pytest.approx([100.0, 200.0, 0.0])

    assert robot.navigate(3, [101.0, 202.0, 0.5], "L1")
    robot._fleet.drain_commands()
    assert agent.goal_calls[0].x == pytest.approx(1.0)
    assert agent.goal_calls[0].y == pytest.approx(2.0)
    assert agent.goal_calls[0].yaw == pytest.approx(0.5)


def test_python_fleet_robot_client_dispatches_navigation_stop_and_attach(mock_node):
    del mock_node
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient

    box = _FakeObject("box")
    agent = _FakeAgent("tinyRobot1")
    sim = _FakeSim([agent], [box])
    fleet = PythonRmfFleetClient.from_sim_core(sim, map_name="L1")
    robot = fleet.robot("tinyRobot1")

    assert robot.navigate(3, [1.0, 2.0, 0.5], "L1")
    fleet.drain_commands()
    assert agent.goal_calls[0].x == pytest.approx(1.0)
    assert agent.goal_calls[0].y == pytest.approx(2.0)
    assert agent.goal_calls[0].yaw == pytest.approx(0.5)

    assert robot.stop()
    fleet.drain_commands()
    assert agent.stop_calls == 1
    assert robot.get_data().last_completed_cmd_id == 4

    assert robot.attach_object(True, 7, object_name="box", parent_link="tool", offset_position=(0.0, 0.0, 0.2))
    fleet.drain_commands()
    assert agent.attach_calls[0][0] is box
    assert agent.attach_calls[0][1] == "tool"
    assert agent.attach_calls[0][2].z == pytest.approx(0.2)
    assert robot.get_data().last_completed_cmd_id == 7


def test_python_fleet_robot_client_defaults_non_positive_attach_search_radius(mock_node):
    del mock_node
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient

    box = _FakeObject("box")
    agent = _FakeAgent("tinyRobot1")
    sim = _FakeSim([agent], [box])
    fleet = PythonRmfFleetClient.from_sim_core(sim, map_name="L1")
    robot = fleet.robot("tinyRobot1")

    assert robot.attach_object(True, 7, search_radius=-1.0)
    robot._fleet.drain_commands()

    assert agent.search_radii == [0.5]
    assert agent.attach_calls[0][0] is box


def test_python_fleet_robot_client_sets_charging_directly(mock_node):
    del mock_node
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient

    agent = _FakeAgent("tinyRobot1")
    sim = _FakeSim([agent])
    robot = PythonRmfFleetClient.from_sim_core(sim, map_name="L1").robot("tinyRobot1")

    assert robot.start_charge(8)
    robot._fleet.drain_commands()
    assert agent.is_charging is True
    assert robot.get_data().last_completed_cmd_id == 8

    assert robot.stop_charge()
    robot._fleet.drain_commands()
    assert agent.is_charging is False


def test_python_fleet_robot_client_rejects_unknown_robot_before_enqueue(mock_node):
    del mock_node
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient

    fleet = PythonRmfFleetClient.from_sim_core(_FakeSim([_FakeAgent("tinyRobot1")]), map_name="L1")
    robot = fleet.robot("missing")

    assert robot.navigate(1, [1.0, 0.0, 0.0], "L1") is False
    assert robot.stop() is False
    assert robot.attach_object(True, 2) is False
    assert robot.start_charge(3) is False
    assert fleet._commands.empty()  # type: ignore[attr-defined]


def test_python_fleet_client_drops_unknown_queued_command(mock_node):
    del mock_node
    from pybullet_fleet_rmf.fleet_clients import PythonRmfFleetClient

    agent = _FakeAgent("tinyRobot1")
    sim = _FakeSim([agent])
    fleet = PythonRmfFleetClient.from_sim_core(sim, map_name="L1")
    fleet._commands.put(("future_command", "tinyRobot1", ()))  # type: ignore[attr-defined]

    fleet.drain_commands()

    assert agent.goal_calls == []
    assert agent.stop_calls == 0
