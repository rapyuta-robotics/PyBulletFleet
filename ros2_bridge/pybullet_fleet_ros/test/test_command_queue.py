"""StepCommandQueue: inline while nothing drives steps, deferred to apply_pending() otherwise."""

import threading
from unittest.mock import patch

import pytest

from pybullet_fleet_ros import command_queue
from pybullet_fleet_ros.command_queue import StepCommandQueue


class _Sim:
    def __init__(self, paused=False):
        self.is_paused = paused


def test_inactive_queue_applies_inline():
    queue = StepCommandQueue(_Sim())
    calls = []
    assert queue.submit(lambda: calls.append(1) or "done") == "done"
    assert calls == [1]
    assert queue.pending_count == 0


def _mid_step_queue(sim=None):
    """An active queue whose simulation thread is inside a step (the step lock is held)."""
    queue = StepCommandQueue(sim or _Sim())
    queue.activate()
    queue.begin_step()
    return queue


def test_active_queue_defers_during_a_step_until_apply_pending():
    queue = _mid_step_queue()
    calls = []
    assert queue.submit(lambda: calls.append(1)) is None
    assert calls == []
    assert queue.pending_count == 1
    assert queue.apply_pending() == 1
    assert calls == [1]
    assert queue.pending_count == 0


def test_active_queue_applies_inline_between_steps():
    queue = _mid_step_queue()
    queue.end_step()
    calls = []
    assert queue.submit(lambda: calls.append(1) or "now") == "now"
    assert calls == [1]
    assert queue.pending_count == 0


def test_begin_step_applies_what_arrived_between_steps_first():
    queue = StepCommandQueue(_Sim())
    queue.activate()
    queue.begin_step()
    calls = []
    queue.submit(lambda: calls.append("queued"))
    queue.end_step()
    assert queue.begin_step() == 1
    assert calls == ["queued"]
    queue.end_step()


def test_paused_simulation_applies_inline_even_mid_step():
    queue = _mid_step_queue(_Sim(paused=True))
    calls = []
    queue.submit(lambda: calls.append(1))
    assert calls == [1]


def test_submissions_apply_in_order():
    queue = _mid_step_queue()
    order = []
    for i in range(3):
        queue.submit(lambda i=i: order.append(i))
    queue.apply_pending()
    assert order == [0, 1, 2]


def test_wait_returns_the_result_once_the_step_applies_it():
    queue = _mid_step_queue()
    stepper = threading.Timer(0.05, queue.apply_pending)
    stepper.start()
    try:
        assert queue.submit(lambda: "applied", wait=True, timeout=2.0) == "applied"
    finally:
        stepper.join()


def test_wait_times_out_and_leaves_the_submission_queued():
    queue = _mid_step_queue()
    calls = []
    assert queue.submit(lambda: calls.append(1), wait=True, timeout=0.01) is None
    assert calls == []
    assert queue.pending_count == 1
    queue.apply_pending()
    assert calls == [1]


def test_waiter_gets_the_exception_and_the_rest_still_applies():
    queue = _mid_step_queue()
    calls = []

    def boom():
        raise RuntimeError("refused")

    stepper = threading.Timer(0.05, queue.apply_pending)
    queue.submit(lambda: calls.append("before"))
    stepper.start()
    try:
        with pytest.raises(RuntimeError, match="refused"):
            queue.submit(boom, wait=True, timeout=2.0)
    finally:
        stepper.join()
    queue.submit(lambda: calls.append("after"))
    queue.apply_pending()
    assert calls == ["before", "after"]


def test_fire_and_forget_exception_is_logged_not_raised():
    queue = _mid_step_queue()

    def boom():
        raise RuntimeError("refused")

    queue.submit(boom)
    with patch.object(command_queue.logger, "error") as error:
        assert queue.apply_pending() == 1
    assert "refused" in str(error.call_args)


def test_deactivate_applies_what_is_pending_and_goes_inline():
    queue = _mid_step_queue()
    calls = []
    queue.submit(lambda: calls.append(1))
    queue.deactivate()
    assert calls == [1]
    assert not queue.deferred
    queue.submit(lambda: calls.append(2))
    assert calls == [1, 2]
