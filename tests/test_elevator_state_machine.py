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
        self.instant = None

    def begin_motion(self, target_height, instant=False):
        self.target_height = target_height
        self.instant = instant
        # An instant motion is over by the time begin_motion returns.
        self.moving = not instant

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


def test_an_instant_request_arrives_on_the_next_update():
    adapter = _Adapter()
    machine = ElevatorStateMachine({"L1": 0.0, "L2": 4.0}, "L1", adapter)

    assert machine.request_floor("L2", instant=True) is ElevatorRequestResult.ACCEPTED
    assert adapter.instant is True
    assert adapter.target_height == 4.0
    assert adapter.attached == 1
    # Still MOVING until an update observes the arrival, as after any motion.
    assert machine.state is ElevatorState.MOVING

    assert machine.update()
    assert machine.current_floor == "L2"
    assert adapter.detached == 1


def test_an_instant_request_while_moving_takes_the_cabin_there_and_drops_the_queue():
    adapter = _Adapter()
    machine = ElevatorStateMachine(
        {"L1": 0.0, "L2": 4.0, "L3": 8.0}, "L1", adapter, request_policy=ElevatorRequestPolicy.QUEUE
    )
    machine.request_floor("L2")
    assert machine.request_floor("L1") is ElevatorRequestResult.QUEUED

    assert machine.request_floor("L3", instant=True) is ElevatorRequestResult.ACCEPTED

    # The passengers attached for the first trip ride along; none are attached twice.
    assert adapter.attached == 1
    assert adapter.target_height == 8.0
    assert machine.target_floor == "L3"
    assert machine.pending_floors == ()
    assert machine.update()
    assert machine.current_floor == "L3"


def test_an_instant_request_for_the_floor_the_cabin_is_at_is_rejected():
    adapter = _Adapter()
    machine = ElevatorStateMachine({"L1": 0.0, "L2": 4.0}, "L1", adapter)

    assert machine.request_floor("L1", instant=True) is ElevatorRequestResult.REJECTED
    assert adapter.target_height is None
