"""Fleet command callbacks with a StepCommandQueue: accept now, apply at the step boundary."""

import threading

import pytest

from pybullet_fleet_ros.command_queue import StepCommandQueue
from pybullet_fleet_ros.fleet_ros_interface import FleetRosInterface
from pybullet_fleet_ros.interface_config import FleetApiConfig, ManagerInterfaceConfig, resolve_bridge_api_config
from test_fleet_ros_interface import FakeAgent, FakeSim, _fleet_msg_types, _node


def _navigate_request(name="robot0", command_id="cmd-1"):
    msgs, srvs = _fleet_msg_types()
    request = srvs.FleetNavigate.Request()
    request.command_id = command_id
    request.source = "test"
    request.goals_2d = [msgs.RobotGoal2D(name=name, position=[1.0, 2.0], yaw=0.5, z=0.0)]
    return request, srvs.FleetNavigate.Response()


def _active_queue(sim):
    """A queue whose simulation thread is inside a step, so a callback must defer."""
    queue = StepCommandQueue(sim)
    queue.activate()
    queue.begin_step()
    return queue


def _accepted_config(**overrides):
    return FleetApiConfig(enabled=True, navigate=True, ack_mode="accepted", **overrides)


def _ack_publisher(node):
    """The mock publisher created for the command_acks topic (one mock serves every publisher)."""
    topics = [str(call.args[1]) for call in node.create_publisher.call_args_list if len(call.args) > 1]
    assert any(topic.endswith("/command_acks") for topic in topics), topics
    return node.create_publisher.return_value


def test_service_answers_accepted_before_the_goal_is_applied():
    node = _node()
    agent = FakeAgent("robot0", 1)
    sim = FakeSim([agent])
    queue = _active_queue(sim)
    interface = FleetRosInterface(node, sim, _accepted_config(), command_queue=queue)
    request, response = _navigate_request()

    result = interface._on_navigate_service(request, response)

    assert result.ack.command_id == "cmd-1"
    assert result.ack.accepted_names == ["robot0"]
    assert result.ack.rejected_names == []
    assert agent.goal_calls == []
    assert queue.pending_count == 1

    queue.apply_pending()

    assert agent.goal_calls[0].x == pytest.approx(1.0)
    published = _ack_publisher(node).publish.call_args.args[0]
    assert published.command_id == "cmd-1"
    assert published.accepted_names == ["robot0"]


def test_unknown_target_is_rejected_in_the_accepted_answer():
    node = _node()
    sim = FakeSim([FakeAgent("robot0", 1)])
    interface = FleetRosInterface(node, sim, _accepted_config(), command_queue=_active_queue(sim))
    request, response = _navigate_request(name="ghost")

    result = interface._on_navigate_service(request, response)

    assert result.ack.accepted_names == []
    assert result.ack.rejected_names == ["ghost"]
    assert result.ack.reject_reasons == ["unknown robot"]


def test_missing_command_id_is_minted_once_for_answer_and_application():
    node = _node()
    agent = FakeAgent("robot0", 1)
    sim = FakeSim([agent])
    queue = _active_queue(sim)
    interface = FleetRosInterface(node, sim, _accepted_config(), command_queue=queue)
    request, response = _navigate_request(command_id="")

    result = interface._on_navigate_service(request, response)
    queue.apply_pending()

    assert result.ack.command_id
    published = _ack_publisher(node).publish.call_args.args[0]
    assert published.command_id == result.ack.command_id


def test_default_mode_is_applied_and_applies_inline_between_steps():
    node = _node()
    agent = FakeAgent("robot0", 1)
    sim = FakeSim([agent])
    queue = _active_queue(sim)
    queue.end_step()
    interface = FleetRosInterface(node, sim, FleetApiConfig(enabled=True, navigate=True), command_queue=queue)
    request, response = _navigate_request()

    result = interface._on_navigate_service(request, response)

    assert result.ack.accepted_names == ["robot0"]
    assert agent.goal_calls[0].x == pytest.approx(1.0)
    assert queue.pending_count == 0


def test_applied_mode_waits_for_the_step_and_answers_with_its_outcome():
    node = _node()
    agent = FakeAgent("robot0", 1)
    sim = FakeSim([agent])
    queue = _active_queue(sim)
    interface = FleetRosInterface(node, sim, FleetApiConfig(enabled=True, navigate=True, ack_timeout=2.0), command_queue=queue)
    request, response = _navigate_request()
    stepper = threading.Timer(0.05, queue.apply_pending)
    stepper.start()
    try:
        result = interface._on_navigate_service(request, response)
    finally:
        stepper.join()

    assert result.ack.accepted_names == ["robot0"]
    assert agent.goal_calls[0].x == pytest.approx(1.0)


def test_applied_mode_times_out_to_the_accepted_answer_and_keeps_the_command():
    node = _node()
    agent = FakeAgent("robot0", 1)
    sim = FakeSim([agent])
    queue = _active_queue(sim)
    interface = FleetRosInterface(
        node, sim, FleetApiConfig(enabled=True, navigate=True, ack_timeout=0.01), command_queue=queue
    )
    request, response = _navigate_request()

    result = interface._on_navigate_service(request, response)

    assert result.ack.accepted_names == ["robot0"]
    assert agent.goal_calls == []
    node.get_logger.return_value.warning.assert_called()
    queue.apply_pending()
    assert len(agent.goal_calls) == 1


def test_topic_command_never_waits_and_applies_at_the_step():
    node = _node()
    agent = FakeAgent("robot0", 1)
    sim = FakeSim([agent])
    queue = _active_queue(sim)
    interface = FleetRosInterface(node, sim, FleetApiConfig(enabled=True, navigate=True), command_queue=queue)
    msgs, _ = _fleet_msg_types()
    msg = msgs.FleetNavigate()
    msg.command_id = "topic-1"
    msg.goals_2d = [msgs.RobotGoal2D(name="robot0", position=[3.0, 4.0], yaw=0.0, z=0.0)]

    interface._on_navigate(msg)

    assert agent.goal_calls == []
    queue.apply_pending()
    assert agent.goal_calls[0].x == pytest.approx(3.0)


def test_stop_service_is_deferred_too():
    node = _node()
    agent = FakeAgent("robot0", 1)
    sim = FakeSim([agent])
    queue = _active_queue(sim)
    interface = FleetRosInterface(node, sim, FleetApiConfig(enabled=True, stop=True, ack_mode="accepted"), command_queue=queue)
    _, srvs = _fleet_msg_types()
    request = srvs.FleetStop.Request()
    request.command_id = "stop-1"
    request.names = ["robot0"]

    result = interface._on_stop_service(request, srvs.FleetStop.Response())

    assert result.ack.accepted_names == ["robot0"]
    assert agent.stop_calls == 0
    queue.apply_pending()
    assert agent.stop_calls == 1


def test_without_a_queue_the_callback_applies_inline():
    node = _node()
    agent = FakeAgent("robot0", 1)
    interface = FleetRosInterface(node, FakeSim([agent]), FleetApiConfig(enabled=True, navigate=True))
    request, response = _navigate_request()

    interface._on_navigate_service(request, response)

    assert len(agent.goal_calls) == 1


def test_ack_mode_is_read_from_config_and_validated():
    config = resolve_bridge_api_config(
        {"fleet_api": {"enabled": True, "navigate": True, "ack_mode": "accepted", "ack_timeout": 0.5}}
    )
    assert config.fleet_api.ack_mode == "accepted"
    assert config.fleet_api.ack_timeout == 0.5
    assert FleetApiConfig().ack_mode == "applied"
    assert ManagerInterfaceConfig(manager="m").ack_mode == "applied"
    with pytest.raises(ValueError, match="ack_mode"):
        resolve_bridge_api_config({"fleet_api": {"enabled": True, "ack_mode": "later"}})
    with pytest.raises(ValueError, match="ack_timeout"):
        resolve_bridge_api_config({"fleet_api": {"enabled": True, "ack_timeout": 0}})
