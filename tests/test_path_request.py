"""PathRequest: the one request object Agent.request_path and the controllers take."""

from __future__ import annotations

from dataclasses import replace

import pybullet as p
import pytest

from pybullet_fleet import (
    Agent,
    AgentSpawnParams,
    MotionMode,
    MultiRobotSimulationCore,
    PathRequest,
    PathRequestState,
    Pose,
    SimulationParams,
)
from pybullet_fleet.controller import Controller
from pybullet_fleet.types import MovementDirection

# The set_path overrides below exercise the deprecated compatibility path on
# purpose; the warning itself is tested in test_overriding_set_path_warns_once.
pytestmark = pytest.mark.filterwarnings("ignore:.*overrides set_path\\(\\), which is deprecated:FutureWarning")


@pytest.fixture
def sim():
    sc = MultiRobotSimulationCore(
        SimulationParams(gui=False, monitor=False, physics=False, timestep=0.05, target_rtf=0, monitor_log_interval=0)
    )
    sc.initialize_simulation()
    yield sc
    p.disconnect(sc.client)


def _agent(sim, batch=False):
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            controller={"max_linear_vel": 2.0, "max_linear_accel": 4.0},
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"}).add_object(agent)
    return agent


def test_fields_and_conversion():
    request = PathRequest(path=[Pose.from_xyz(1, 0, 0)], direction="backward")
    assert request.direction is MovementDirection.BACKWARD
    assert isinstance(request.path, tuple)
    assert request.final_orientation_align is True and request.auto_approach is True
    with pytest.raises(ValueError):
        PathRequest(path=[Pose.from_xyz(1, 0, 0)], direction="sideways")


def test_identity_survives_copies_but_not_equal_values():
    request = PathRequest(path=[Pose.from_xyz(1, 0, 0)])
    copy = replace(request, path=[Pose.from_xyz(0, 0, 0), Pose.from_xyz(1, 0, 0)])
    twin = PathRequest(path=[Pose.from_xyz(1, 0, 0)])
    assert request.same_request(copy)
    assert not request.same_request(twin)  # equal fields, another request
    assert not request.same_request(None)


def test_record_round_trip():
    request = PathRequest(path=[Pose.from_xyz(1, 2, 0)], final_orientation_align=False, direction="auto")
    back = PathRequest.from_record(request.to_record())
    assert back.to_record() == request.to_record()
    assert back.direction is MovementDirection.AUTO


@pytest.mark.parametrize("batch", [False, True], ids=["per_agent", "batch"])
def test_request_path_keeps_identity_through_the_approach_waypoint(sim, batch):
    agent = _agent(sim, batch)
    request = PathRequest(path=[Pose.from_xyz(3.0, 0.0, 0.1)], final_orientation_align=False)  # approach added
    assert agent.request_path(request) is True
    assert agent.path_request_state(request) is PathRequestState.ACTIVE
    for _ in range(200):
        sim.step_once()
    assert agent.path_request_state(request) is PathRequestState.FINISHED
    other = PathRequest(path=[Pose.from_xyz(0.0, 0.0, 0.1)])
    assert agent.path_request_state(other) is PathRequestState.GONE


def test_set_path_and_set_goal_pose_go_through_request_path(sim, monkeypatch):
    agent = _agent(sim)
    seen = []
    original = agent.request_path
    monkeypatch.setattr(agent, "request_path", lambda request, **kw: seen.append(request) or original(request, **kw))
    agent.set_path([Pose.from_xyz(1, 0, 0.1)], auto_approach=False, direction="forward")
    agent.set_goal_pose(Pose.from_xyz(2, 0, 0.1))
    assert [r.auto_approach for r in seen] == [False, True]
    assert seen[0].direction is MovementDirection.FORWARD


def test_a_controller_written_against_set_path_still_works(sim):
    class Legacy(Controller):
        def __init__(self):
            super().__init__()
            self.paths = []

        def compute(self, agent, dt):
            return False

        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            self.paths.append((list(path), final_orientation_align))
            return True

    agent = _agent(sim)
    legacy = Legacy()
    agent.set_controller(legacy)
    goal = Pose.from_xyz(1.0, 0.0, 0.1)
    assert agent.request_path(PathRequest(path=[goal], final_orientation_align=False, auto_approach=False)) is True
    assert legacy.paths == [([goal], False)]
    assert agent.path_request_state(PathRequest(path=[goal])) is PathRequestState.UNKNOWN


class _ShiftingOmni:
    """A set_path override that edits the path, as a custom controller might."""

    SHIFT_Y = 1.0

    @staticmethod
    def shifted(path):
        return [Pose.from_xyz(p.position[0], p.position[1] + _ShiftingOmni.SHIFT_Y, p.position[2]) for p in path]


def _shifting_per_agent():
    from pybullet_fleet.controller import OmniController

    class ShiftingOmniController(OmniController):
        calls = 0

        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            type(self).calls += 1
            return super().set_path(agent, _ShiftingOmni.shifted(path), final_orientation_align, direction, preempt)

    return ShiftingOmniController


def _shifting_batch():
    from pybullet_fleet.controllers.batch_omni import BatchOmniController

    class ShiftingBatchOmni(BatchOmniController):
        _registry_name = "test_shifting_batch_omni"
        calls = 0

        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            type(self).calls += 1
            return super().set_path(agent, _ShiftingOmni.shifted(path), final_orientation_align, direction, preempt)

    return ShiftingBatchOmni


@pytest.mark.parametrize("batch", [False, True], ids=["per_agent", "batch"])
def test_a_set_path_override_of_a_kinematic_controller_is_still_called(sim, batch):
    """A subclass that overrides set_path (to edit the path, say) keeps being
    called for every request, and the request it passes on is still tracked."""
    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.geometry import Path
    from pybullet_fleet.types import ActionStatus

    agent = _agent(sim)
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        cls = _shifting_batch()
        AgentManager(sim_core=sim, fleet_controller={"type": "test_shifting_batch_omni"}).add_object(agent)
    else:
        cls = _shifting_per_agent()
        agent.set_controller(cls(agent._controllers[0].params))

    action = MoveAction(path=Path.from_positions([[3.0, 0.0, 0.1]]), auto_approach=False, final_orientation_align=False)
    agent.add_action(action)
    sim.step_once()
    assert cls.calls == 1
    assert agent.path_request_state(action.request) is PathRequestState.ACTIVE
    for _ in range(400):
        sim.step_once()
        if action.status is not ActionStatus.IN_PROGRESS:
            break
    assert action.status is ActionStatus.COMPLETED
    assert list(agent.get_pose().position[:2]) == pytest.approx([3.0, 1.0], abs=1e-6)


@pytest.mark.parametrize("batch", [False, True], ids=["per_agent", "batch"])
def test_brake_does_not_go_through_a_set_path_override(sim, batch):
    """The braking path is the controller's own, not a path the user sent: an
    override that edits paths must not see it. brake() stops on the line of
    motion and the agent stays there."""
    import numpy as np

    agent = _agent(sim)
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        cls = _shifting_batch()
        AgentManager(sim_core=sim, fleet_controller={"type": "test_shifting_batch_omni"}).add_object(agent)
    else:
        cls = _shifting_per_agent()
        agent.set_controller(cls(agent._controllers[0].params))
    # Shifted by the override to (20, 1): the agent heads there in a straight line.
    agent.set_path([Pose.from_xyz(20.0, 0.0, 0.1)], auto_approach=False, final_orientation_align=False)
    for _ in range(15):
        sim.step_once()
    assert agent.is_moving
    calls = cls.calls
    start = np.array(agent.get_pose().position[:2], dtype=float)
    heading = np.array([20.0, 1.0]) - start
    heading /= np.linalg.norm(heading)
    assert agent.brake() is True
    assert cls.calls == calls  # the braking path did not reach the override
    track = []
    for _ in range(200):
        sim.step_once()
        track.append(np.array(agent.get_pose().position[:2], dtype=float))
    assert not agent.is_moving
    rel = np.array(track) - start
    along = rel @ heading
    across = rel - np.outer(along, heading)
    assert np.abs(across).max() < 1e-6  # never left the line of motion
    assert np.all(np.diff(along) >= -1e-9)  # never turned back
    assert track[-1] == pytest.approx(track[-100], abs=1e-9)  # and stayed where it stopped


def test_a_set_path_override_that_replaces_the_request_is_untracked(sim):
    """An override that never passes the request on (it builds its own) cannot
    be tracked: the state is UNKNOWN, so a MoveAction judges by where the agent rests."""
    from pybullet_fleet.controller import OmniController

    class OwnRequest(OmniController):
        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            return self.request_path(agent, PathRequest(path=path, final_orientation_align=False, auto_approach=False))

    agent = _agent(sim)
    agent.set_controller(OwnRequest(agent._controllers[0].params))
    request = PathRequest(path=[Pose.from_xyz(2.0, 0.0, 0.1)], auto_approach=False)
    assert agent.request_path(request) is True
    assert agent.path_request_state(request) is PathRequestState.UNKNOWN


def test_an_override_that_sends_an_edited_path_of_its_own_ends_the_action_cancelled(sim):
    """Documented: untracked, the action is judged against its own path's end,
    which the agent never reaches when the override moved it."""
    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.controller import OmniController
    from pybullet_fleet.geometry import Path
    from pybullet_fleet.types import ActionStatus

    class OwnShiftedRequest(OmniController):
        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            own = PathRequest(path=_ShiftingOmni.shifted(path), final_orientation_align=False, auto_approach=False)
            return self.request_path(agent, own)

    agent = _agent(sim)
    agent.set_controller(OwnShiftedRequest(agent._controllers[0].params))
    action = MoveAction(path=Path.from_positions([[3.0, 0.0, 0.1]]), auto_approach=False, final_orientation_align=False)
    agent.add_action(action)
    for _ in range(400):
        sim.step_once()
        if action.status not in (ActionStatus.NOT_STARTED, ActionStatus.IN_PROGRESS):
            break
    assert list(agent.get_pose().position[:2]) == pytest.approx([3.0, 1.0], abs=1e-6)
    assert action.status is ActionStatus.CANCELLED


def test_every_untracked_request_stays_unknown_after_later_requests(sim):
    """Each request an override did not pass on stays UNKNOWN, also once later
    requests -- untracked or tracked -- were accepted."""
    from pybullet_fleet.controller import OmniController

    class OwnRequestOnce(OmniController):
        own = True

        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            if self.own:
                return self.request_path(agent, PathRequest(path=path, auto_approach=False))
            return super().set_path(agent, path, final_orientation_align, direction, preempt)

    agent = _agent(sim)
    controller = OwnRequestOnce(agent._controllers[0].params)
    agent.set_controller(controller)
    first = PathRequest(path=[Pose.from_xyz(2.0, 0.0, 0.1)], auto_approach=False)
    second = PathRequest(path=[Pose.from_xyz(3.0, 0.0, 0.1)], auto_approach=False)
    assert agent.request_path(first) is True
    assert agent.request_path(second) is True
    controller.own = False
    third = PathRequest(path=[Pose.from_xyz(4.0, 0.0, 0.1)], auto_approach=False)
    assert agent.request_path(third) is True

    assert agent.path_request_state(first) is PathRequestState.UNKNOWN
    assert agent.path_request_state(second) is PathRequestState.UNKNOWN
    assert agent.path_request_state(third) is PathRequestState.ACTIVE


def test_an_untracked_request_nobody_holds_is_forgotten(sim):
    import gc

    from pybullet_fleet.controller import OmniController

    class OwnRequest(OmniController):
        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            return self.request_path(agent, PathRequest(path=path, auto_approach=False))

    agent = _agent(sim)
    agent.set_controller(OwnRequest(agent._controllers[0].params))
    for x in range(2, 6):
        assert agent.request_path(PathRequest(path=[Pose.from_xyz(float(x), 0.0, 0.1)], auto_approach=False))
    gc.collect()
    assert len(agent._untracked_requests) == 0


def test_super_set_path_without_request_path_fails_fast():
    """A direct Controller subclass whose set_path override calls super() but
    implements no request_path() would recurse without end: it raises instead."""

    class OnlySetPath(Controller):
        def compute(self, agent, dt):
            return False

        def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
            return super().set_path(agent, path, final_orientation_align, direction, preempt)

    with pytest.raises(NotImplementedError, match="implement request_path"):
        OnlySetPath().set_path(None, [Pose.from_xyz(1.0, 0.0, 0.0)])


def test_the_interruption_message_says_whether_anything_is_left_to_wait_for(sim, caplog):
    import logging

    from pybullet_fleet.action import MoveAction
    from pybullet_fleet.geometry import Path

    agent = _agent(sim)
    action = MoveAction(path=Path.from_positions([[1.0, 0.0, 0.1]]), auto_approach=False)
    with caplog.at_level(logging.INFO, logger="pybullet_fleet"):
        agent._interrupt_actions(action, "ended early: stopped")
    assert "MoveAction ended early: stopped; no other path is left" in caplog.text

    agent.set_path([Pose.from_xyz(5.0, 0.0, 0.1)], auto_approach=False)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pybullet_fleet"):
        agent._interrupt_actions(action, "ended early: replaced")
    assert "waits until the path from outside it ends" in caplog.text


def test_a_restored_path_is_an_active_request_that_finishes(sim):
    """After restore_navigation_state the path followed is a request of its own:
    ACTIVE while followed, FINISHED at its end. A request sent before is GONE."""
    from pybullet_fleet.controller import OmniController

    agent = _agent(sim)
    before = PathRequest(path=[Pose.from_xyz(3.0, 0.0, 0.1)], auto_approach=False)
    assert agent.request_path(before) is True
    for _ in range(5):
        sim.step_once()
    controller = agent._controllers[0]
    state = controller.capture_navigation_state()

    restored = OmniController(controller.params)
    restored.restore_navigation_state(state, list(agent.get_pose().orientation))
    active = restored._active_request
    assert active is not None
    assert [list(p.position) for p in active.path] == [list(p.position) for p in controller._path]
    assert restored.request_state(agent, active) is PathRequestState.ACTIVE
    assert restored.request_state(agent, before) is PathRequestState.GONE


def test_the_batch_differential_set_path_is_not_an_override():
    """Its set_path only keeps direction third: requests go straight to request_path,
    while a subclass overriding it again is an override as usual."""
    from pybullet_fleet.controller import overrides_set_path
    from pybullet_fleet.controllers.batch_differential import BatchDifferentialController

    class Edited(BatchDifferentialController):
        def set_path(self, agent, path, direction=None, final_orientation_align=True, preempt=False):
            return super().set_path(agent, path, direction, final_orientation_align, preempt)

    assert not overrides_set_path(BatchDifferentialController.__new__(BatchDifferentialController))
    assert overrides_set_path(Edited.__new__(Edited))


def test_overriding_set_path_warns_once_at_class_definition():
    """A subclass that overrides set_path is told to override request_path instead."""
    import warnings

    from pybullet_fleet.controller import OmniController

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        class Edits(OmniController):
            def set_path(self, agent, path, final_orientation_align=True, direction=None, preempt=False):
                return super().set_path(agent, path, final_orientation_align, direction, preempt)

        Edits(None)
        Edits(None)
    deprecations = [w for w in caught if issubclass(w.category, FutureWarning) and "set_path" in str(w.message)]
    assert len(deprecations) == 1, "once, when the class is defined"
    message = str(deprecations[0].message)
    assert "Edits overrides set_path()" in message and "override request_path() instead" in message
    assert deprecations[0].filename == __file__, "the warning points at the class statement"


def test_built_in_controllers_and_request_path_overrides_do_not_warn():
    import warnings

    from pybullet_fleet.controller import OmniController
    from pybullet_fleet.controllers.batch_differential import BatchDifferentialController

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        class Requests(OmniController):
            def request_path(self, agent, request, preempt=False):
                return super().request_path(agent, request, preempt=preempt)

        class Batch(BatchDifferentialController):
            pass

    assert not [w for w in caught if "set_path" in str(w.message)]
