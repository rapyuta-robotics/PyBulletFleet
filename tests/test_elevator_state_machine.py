"""Contract tests for the backend-neutral elevator lifecycle."""

from pybullet_fleet.devices.elevator_state_machine import (
    ElevatorRequestPolicy,
    ElevatorRequestResult,
    ElevatorState,
    ElevatorStateMachine,
)


class _Adapter:
    def __init__(self):
        self.target_height = None
        self.moving = False
        self.attached = 0
        self.detached = 0

    def begin_motion(self, target_height):
        self.target_height = target_height
        self.moving = True

    def motion_in_progress(self):
        return self.moving

    def attach_platform_passengers(self):
        self.attached += 1
        return self.attached

    def detach_passengers(self):
        self.detached += 1
        return self.detached


def test_elevator_state_machine_owns_request_lifecycle_without_a_backend():
    adapter = _Adapter()
    machine = ElevatorStateMachine({"L1": 0.0, "L2": 4.0}, "L1", adapter)

    assert machine.state is ElevatorState.IDLE
    assert machine.request_floor("L2") is ElevatorRequestResult.ACCEPTED
    assert adapter.target_height == 4.0
    assert adapter.attached == 1
    assert machine.current_floor == "L1"
    assert machine.target_floor == "L2"
    assert machine.state is ElevatorState.MOVING
    assert machine.request_floor("L2") is ElevatorRequestResult.REJECTED

    adapter.moving = False
    assert machine.update()
    assert machine.current_floor == "L2"
    assert adapter.detached == 1
    assert machine.state is ElevatorState.ARRIVED
    assert not machine.is_moving

    assert not machine.update()
    assert machine.state is ElevatorState.IDLE


def test_elevator_state_machine_rejects_unknown_and_current_floor_requests():
    adapter = _Adapter()
    machine = ElevatorStateMachine({"L1": 0.0}, "L1", adapter)

    assert machine.request_floor("unknown") is ElevatorRequestResult.REJECTED
    assert machine.request_floor("L1") is ElevatorRequestResult.REJECTED
    assert adapter.target_height is None


def test_elevator_state_machine_accepts_a_new_request_while_arrived():
    adapter = _Adapter()
    machine = ElevatorStateMachine({"L1": 0.0, "L2": 4.0, "L3": 8.0}, "L1", adapter)

    assert machine.request_floor("L2") is ElevatorRequestResult.ACCEPTED
    adapter.moving = False
    assert machine.update()
    assert machine.state is ElevatorState.ARRIVED

    assert machine.request_floor("L3") is ElevatorRequestResult.ACCEPTED
    assert machine.state is ElevatorState.MOVING


def test_elevator_state_machine_queues_requests_after_the_current_trip():
    adapter = _Adapter()
    machine = ElevatorStateMachine(
        {"L1": 0.0, "L2": 4.0, "L3": 8.0},
        "L1",
        adapter,
        request_policy=ElevatorRequestPolicy.QUEUE,
    )

    assert machine.request_floor("L2") is ElevatorRequestResult.ACCEPTED
    assert machine.request_floor("L3") is ElevatorRequestResult.QUEUED
    assert machine.pending_floors == ("L3",)

    adapter.moving = False
    assert machine.update()
    assert machine.state is ElevatorState.ARRIVED
    assert machine.current_floor == "L2"
    assert adapter.detached == 1

    assert not machine.update()
    assert machine.state is ElevatorState.MOVING
    assert machine.target_floor == "L3"
    assert machine.pending_floors == ()
    assert adapter.attached == 2


def test_elevator_state_machine_replaces_the_next_request_without_redirecting():
    adapter = _Adapter()
    machine = ElevatorStateMachine(
        {"L1": 0.0, "L2": 4.0, "L3": 8.0},
        "L1",
        adapter,
        request_policy=ElevatorRequestPolicy.REPLACE_NEXT,
    )

    assert machine.request_floor("L2") is ElevatorRequestResult.ACCEPTED
    assert machine.request_floor("L3") is ElevatorRequestResult.QUEUED
    assert machine.request_floor("L1") is ElevatorRequestResult.REPLACED
    assert adapter.target_height == 4.0
    assert machine.target_floor == "L2"
    assert machine.pending_floors == ("L1",)


def test_elevator_state_machine_drops_the_next_request_when_the_current_target_is_asked_again():
    adapter = _Adapter()
    machine = ElevatorStateMachine(
        {"L1": 0.0, "L2": 4.0, "L3": 8.0},
        "L1",
        adapter,
        request_policy=ElevatorRequestPolicy.REPLACE_NEXT,
    )

    assert machine.request_floor("L3") is ElevatorRequestResult.ACCEPTED
    assert machine.request_floor("L2") is ElevatorRequestResult.QUEUED
    # The caller changed its mind back: the cabin must stay at L3 once there.
    assert machine.request_floor("L3") is ElevatorRequestResult.REPLACED
    assert machine.pending_floors == ()
    assert machine.request_floor("L3") is ElevatorRequestResult.REJECTED

    adapter.moving = False
    assert machine.update() is True
    machine.update()
    assert machine.state is ElevatorState.IDLE
    assert machine.current_floor == "L3"


def test_elevator_state_machine_redirects_a_moving_cabin_to_the_latest_request():
    adapter = _Adapter()
    machine = ElevatorStateMachine(
        {"L1": 0.0, "L2": 4.0, "L3": 8.0},
        "L1",
        adapter,
        request_policy=ElevatorRequestPolicy.REDIRECT,
    )

    assert machine.request_floor("L3") is ElevatorRequestResult.ACCEPTED
    assert machine.request_floor("L2") is ElevatorRequestResult.REDIRECTED
    assert adapter.target_height == 4.0
    assert machine.target_floor == "L2"
    assert machine.pending_floors == ()
    assert adapter.attached == 1, "passengers already on board stay attached"
    assert machine.request_floor("L2") is ElevatorRequestResult.REJECTED

    adapter.moving = False
    assert machine.update() is True
    assert machine.current_floor == "L2"
    machine.update()
    assert machine.state is ElevatorState.IDLE
