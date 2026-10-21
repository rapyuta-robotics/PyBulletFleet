"""goal_replacement: what a moving agent does with a new goal.

``wait_complete`` holds every goal set while a path is in progress and starts
it once that path is finished. In the other modes a goal the agent can reach
from its current motion starts at once, continuing it; any other goal is
handled by the mode: ``brake`` (the default) brakes along the motion at the
configured deceleration and starts the goal from rest, ``immediate`` starts it
from rest at once, and ``reject`` refuses it. Only the latest held goal is
kept, and dropping one is logged. Per-agent and batch controllers, omni and
differential, behave alike.
"""

from __future__ import annotations

import logging

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet.types import MovementDirection, CollisionMode, GoalReplacement

DT = 0.05
VMAX = 2.0
ACCEL = 4.0
DECEL = 2.0  # differs from ACCEL, so braking at the wrong rate shows
CRUISE_GOAL = (10.0, 0.0)
OFF_LINE = (0.0, 5.0)  # behind and to the side: no mode can continue into it
BRAKING_DISTANCE = VMAX * VMAX / (2.0 * DECEL)


def _world():
    sc = MultiRobotSimulationCore(
        SimulationParams(
            gui=False,
            monitor=False,
            physics=False,
            timestep=DT,
            target_rtf=0,
            collision_check_frequency=0,
            monitor_log_interval=0,
            log_level="warning",
        )
    )
    sc.initialize_simulation()
    return sc


@pytest.fixture
def sim():
    sc = _world()
    yield sc
    p.disconnect(sc.client)


@pytest.fixture(params=[False, True], ids=["per_agent", "batch"])
def batch(request):
    return request.param


@pytest.fixture(params=[MotionMode.OMNIDIRECTIONAL, MotionMode.DIFFERENTIAL], ids=["omni", "differential"])
def motion(request):
    return request.param


def _agent(sim, batch=False, motion=MotionMode.OMNIDIRECTIONAL, mode=None, name=None):
    agent = Agent.from_params(
        AgentSpawnParams(
            name=name,
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=motion,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={
                "max_linear_vel": VMAX,
                "max_linear_accel": ACCEL,
                "max_linear_decel": DECEL,
                "goal_replacement": mode,
            },
        ),
        sim_core=sim,
    )
    if batch:
        from pybullet_fleet.agent_manager import AgentManager

        kind = "batch_differential" if motion == MotionMode.DIFFERENTIAL else "batch_omni"
        AgentManager(sim_core=sim, fleet_controller={"type": kind}).add_object(agent)
    return agent


def _xy(agent):
    return np.array(agent.get_pose().position[:2], dtype=float)


def _pending(agent):
    if agent._batch_controller is not None:
        return agent._batch_controller.pending_path(agent)
    return agent._controllers[0].pending_path()


def _goal(xy):
    return [Pose.from_xyz(xy[0], xy[1], 0.1)]


def _cruise(sim, agent, goal=CRUISE_GOAL):
    """Drive toward ``goal`` until at full speed; return the position then."""
    agent.set_path(_goal(goal), auto_approach=False, final_orientation_align=False)
    for _ in range(int(VMAX / ACCEL / DT) + 5):
        sim.step_once()
    # Still on the way, at full speed: the profile speed, not a batch row's
    # reported last-step average.
    controller = agent._batch_controller or agent._controllers[0]
    assert agent.is_moving
    assert np.linalg.norm(controller.continuation_velocity(agent)) == pytest.approx(VMAX, rel=1e-6)
    return _xy(agent)


def _run(sim, agent, steps=600):
    """Step until the agent is at rest with nothing held; return per-step positions."""
    track = [_xy(agent)]
    for _ in range(steps):
        sim.step_once()
        track.append(_xy(agent))
        if not agent.is_moving and _pending(agent) is None:
            break
    assert not agent.is_moving
    return np.array(track)


def _speeds(track):
    return np.linalg.norm(np.diff(track, axis=0), axis=1) / DT


def _replace(sim, batch, motion, mode, goal=OFF_LINE):
    agent = _agent(sim, batch, motion, mode)
    _cruise(sim, agent)
    start = _xy(agent)
    accepted = agent.set_path(_goal(goal), auto_approach=False, final_orientation_align=False)
    return agent, start, accepted


class TestModes:
    def test_reject_refuses_and_the_motion_runs_on(self, sim, batch, motion):
        agent, start, accepted = _replace(sim, batch, motion, "reject")
        assert accepted is False
        assert _pending(agent) is None
        track = _run(sim, agent)
        assert track[-1] == pytest.approx(CRUISE_GOAL, abs=1e-6)
        assert np.abs(track[:, 1]).max() < 1e-9

    def test_immediate_starts_from_rest_at_once(self, sim, batch, motion):
        agent, start, accepted = _replace(sim, batch, motion, "immediate")
        assert accepted is True
        assert _pending(agent) is None
        sim.step_once()
        # From rest: one step covers at most what the acceleration allows.
        assert np.linalg.norm(_xy(agent) - start) <= 0.5 * ACCEL * DT * DT + 1e-9
        track = _run(sim, agent)
        assert track[-1] == pytest.approx(OFF_LINE, abs=1e-6)

    def test_brake_stops_at_the_deceleration_then_goes(self, sim, batch, motion):
        agent, start, accepted = _replace(sim, batch, motion, "brake")
        assert accepted is True
        assert _pending(agent) is not None
        track = _run(sim, agent)
        speeds = _speeds(track)
        stop = int(np.argmax(speeds < 1e-9))
        assert stop > 0
        # Straight on along the line of motion to the braking point, ...
        braking = track[: stop + 1]
        assert np.abs(braking[:, 1]).max() < 1e-9
        assert braking[-1] == pytest.approx(start + [BRAKING_DISTANCE, 0.0], abs=1e-6)
        # ... never slowing faster than the deceleration ...
        profile = np.r_[VMAX, speeds[:stop]]
        assert np.all(np.diff(profile) <= 1e-6)
        assert np.max(-np.diff(profile)) <= DECEL * DT + 1e-6
        # ... then to the new goal.
        assert track[-1] == pytest.approx(OFF_LINE, abs=1e-6)
        assert _pending(agent) is None

    def test_brake_with_an_infeasible_braking_path_starts_from_rest(self, sim, batch, motion, monkeypatch, caplog):
        """A braking path that cannot continue the motion (it stops short of
        the braking distance) is not started blindly: a warning, then the new
        goal from rest at once."""
        import pybullet_fleet._replan as replan
        from pybullet_fleet._replan import PathRequest
        from pybullet_fleet.types import MovementDirection

        def short_stop(params, position, orientation, velocity):
            ahead = np.asarray(position, dtype=float) + [0.1 * BRAKING_DISTANCE, 0.0, 0.0]
            goal = Pose(position=ahead.tolist(), orientation=list(orientation))
            return PathRequest(path=[goal], final_orientation_align=False, direction=MovementDirection.AUTO)

        monkeypatch.setattr(replan, "braking_request", short_stop)
        with caplog.at_level(logging.WARNING):
            agent, _, accepted = _replace(sim, batch, motion, "brake")
        assert accepted is True
        assert "cannot continue the motion" in caplog.text
        assert _pending(agent) is None  # started at once, nothing held
        track = _run(sim, agent)
        assert track[-1] == pytest.approx(OFF_LINE, abs=1e-6)

    def test_wait_complete_finishes_the_current_path_first(self, sim, batch, motion):
        agent, _, accepted = _replace(sim, batch, motion, "wait_complete")
        assert accepted is True
        assert _pending(agent) is not None
        track = _run(sim, agent)
        reached = int(np.argmin(np.linalg.norm(track - CRUISE_GOAL, axis=1)))
        assert track[reached] == pytest.approx(CRUISE_GOAL, abs=1e-6)
        assert np.abs(track[: reached + 1, 1]).max() < 1e-9
        assert track[-1] == pytest.approx(OFF_LINE, abs=1e-6)

    def test_wait_complete_waits_for_the_final_turn(self, sim, batch, motion):
        agent = _agent(sim, batch, motion, "wait_complete")
        facing_left = Pose.from_yaw(CRUISE_GOAL[0], CRUISE_GOAL[1], 0.1, np.pi / 2)
        agent.set_path([facing_left], auto_approach=False, final_orientation_align=True)
        for _ in range(int(VMAX / ACCEL / DT) + 5):
            sim.step_once()
        assert agent.set_path(_goal(OFF_LINE), auto_approach=False, final_orientation_align=False)
        turned = False
        for _ in range(600):
            sim.step_once()
            at_goal = np.allclose(_xy(agent), CRUISE_GOAL, atol=1e-6)
            if at_goal and np.allclose(agent.get_pose().orientation, facing_left.orientation, atol=1e-6):
                turned = True
            assert turned or at_goal or _xy(agent)[1] < 1e-9  # not off toward the held goal yet
            if not agent.is_moving and _pending(agent) is None:
                break
        assert turned
        assert _xy(agent) == pytest.approx(OFF_LINE, abs=1e-6)

    def test_the_default_is_brake(self, sim, batch, motion):
        default = _replace(sim, batch, motion, None)[0]
        assert ControllerParams()._eff_goal_replacement() is GoalReplacement.BRAKE
        track = _run(sim, default)
        assert np.min(_speeds(track)) < 1e-9  # it stopped before turning off
        assert track[-1] == pytest.approx(OFF_LINE, abs=1e-6)


class TestReachableGoals:
    AHEAD = (20.0, 0.0)

    @pytest.mark.parametrize("mode", ["brake", "reject", "immediate"])
    def test_are_taken_at_once_continuing_the_motion(self, sim, batch, motion, mode):
        agent, _, accepted = _replace(sim, batch, motion, mode, goal=self.AHEAD)
        assert accepted is True
        assert _pending(agent) is None
        track = _run(sim, agent)
        speeds = _speeds(track)
        assert speeds[:-1].min() > 0.1  # never stopped on the way
        assert track[-1] == pytest.approx(self.AHEAD, abs=1e-6)

    def test_a_goal_behind_on_the_line_is_reached_by_reversing(self, sim, batch, motion):
        agent = _agent(sim, batch, motion, "brake")
        start = _cruise(sim, agent)
        orientation = agent.get_pose().orientation
        behind = (-20.0, 0.0)
        # AUTO lets a differential drive back to it without turning.
        assert agent.set_path(_goal(behind), auto_approach=False, final_orientation_align=False, direction="auto")
        assert _pending(agent) is None
        track = _run(sim, agent)
        # It runs on, slowing at the acceleration (the trajectory's turn-around
        # is part of its acceleration phase), then drives back.
        assert track[:, 0].max() == pytest.approx(start[0] + VMAX * VMAX / (2.0 * ACCEL), abs=1e-6)
        assert np.abs(track[:, 1]).max() < 1e-9
        assert track[-1] == pytest.approx(behind, abs=1e-6)
        assert agent.get_pose().orientation == pytest.approx(orientation, abs=1e-9)

    def test_wait_complete_holds_them_for_the_current_path(self, sim, batch, motion):
        agent, _, accepted = _replace(sim, batch, motion, "wait_complete", goal=self.AHEAD)
        assert accepted is True
        assert _pending(agent) is not None
        track = _run(sim, agent)
        speeds = _speeds(track)
        stop = int(np.argmax(speeds < 1e-9))
        assert track[stop] == pytest.approx(CRUISE_GOAL, abs=1e-6)
        assert track[-1] == pytest.approx(self.AHEAD, abs=1e-6)

    def test_a_reachable_goal_while_braking_supersedes_the_held_one(self, sim, batch, motion, caplog):
        agent, _, _ = _replace(sim, batch, motion, "brake")
        sim.step_once()
        with caplog.at_level(logging.INFO, logger="pybullet_fleet"):
            assert agent.set_path(_goal((50.0, 0.0)), auto_approach=False, final_orientation_align=False)
        assert _pending(agent) is None
        assert "dropped: a newer path started at once" in caplog.text
        assert _run(sim, agent)[-1] == pytest.approx((50.0, 0.0), abs=1e-6)


class TestHeldGoal:
    @pytest.mark.parametrize("mode", ["brake", "wait_complete"])
    def test_a_newer_request_replaces_it(self, sim, batch, motion, mode, caplog):
        agent, _, _ = _replace(sim, batch, motion, mode)
        with caplog.at_level(logging.INFO, logger="pybullet_fleet"):
            assert agent.set_path(_goal((0.0, -5.0)), auto_approach=False, final_orientation_align=False)
        assert _pending(agent).path[0].position[:2] == pytest.approx([0.0, -5.0])
        assert "to [0.0, 5.0, 0.1] dropped: a newer path replaced it" in caplog.text
        assert _run(sim, agent)[-1] == pytest.approx((0.0, -5.0), abs=1e-6)

    @pytest.mark.parametrize("mode", ["brake", "wait_complete"])
    def test_stop_drops_it(self, sim, batch, motion, mode, caplog):
        agent, _, _ = _replace(sim, batch, motion, mode)
        with caplog.at_level(logging.INFO, logger="pybullet_fleet"):
            agent.stop()
        assert _pending(agent) is None
        assert "dropped: the agent was stopped" in caplog.text
        before = _xy(agent)
        for _ in range(20):
            sim.step_once()
        assert _xy(agent) == pytest.approx(before)

    def test_brake_does_not_wait_for_the_current_path(self, sim, batch, motion):
        agent = _agent(sim, batch, motion, "wait_complete")
        start = _cruise(sim, agent)
        assert agent.brake() is True
        track = _run(sim, agent)
        assert track[-1] == pytest.approx(start + [BRAKING_DISTANCE, 0.0], abs=1e-6)

    def test_wait_complete_brakes_off_a_velocity_command(self, sim):
        """A velocity command has no path to wait for, so it is braked."""
        agent = _agent(sim, mode="wait_complete")
        agent._controller.set_velocity(vx=VMAX)
        for _ in range(int(VMAX / ACCEL / DT) + 5):
            sim.step_once()
        start = _xy(agent)
        assert agent.set_path(_goal(OFF_LINE), auto_approach=False, final_orientation_align=False)
        track = _run(sim, agent)
        stop = int(np.argmax(_speeds(track) < 1e-9))
        assert track[stop] == pytest.approx(start + [BRAKING_DISTANCE, 0.0], abs=1e-6)
        assert track[-1] == pytest.approx(OFF_LINE, abs=1e-6)


@pytest.mark.parametrize("mode", ["brake", "wait_complete", "immediate"])
def test_per_agent_and_batch_agree(motion, mode):
    tracks = []
    for batch in (False, True):
        # Each in a world of its own, on the same clock.
        world = _world()
        try:
            agent, _, _ = _replace(world, batch, motion, mode)
            tracks.append(_run(world, agent))
        finally:
            p.disconnect(world.client)
    assert len(tracks[0]) == len(tracks[1])
    assert tracks[0] == pytest.approx(tracks[1], abs=1e-6)


class TestConfiguration:
    def test_the_global_default_comes_from_the_environment(self, monkeypatch):
        from pybullet_fleet import _defaults

        snapshot = _defaults._snapshot()
        try:
            monkeypatch.setenv("PBF_CONTROLLER_GOAL_REPLACEMENT", "wait_complete")
            _defaults.reload_defaults()
            assert ControllerParams()._eff_goal_replacement() is GoalReplacement.WAIT_COMPLETE
            # An agent's own setting wins over the global default.
            assert ControllerParams(goal_replacement="reject")._eff_goal_replacement() is GoalReplacement.REJECT
        finally:
            _defaults._restore(snapshot)
        assert ControllerParams()._eff_goal_replacement() is GoalReplacement.BRAKE

    def test_a_manager_sets_it_for_agents_without_their_own(self, sim):
        from pybullet_fleet.agent_manager import AgentManager

        manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni", "goal_replacement": "reject"})
        unset, own = _agent(sim, name="unset"), _agent(sim, mode="immediate", name="own")
        manager.add_object(unset)
        manager.add_object(own)
        assert unset.controller_params._eff_goal_replacement() is GoalReplacement.REJECT
        assert own.controller_params._eff_goal_replacement() is GoalReplacement.IMMEDIATE
        _cruise(sim, unset)
        assert unset.set_path(_goal(OFF_LINE), auto_approach=False) is False

    def test_an_unknown_mode_is_an_error(self):
        with pytest.raises(ValueError, match="Unknown goal_replacement 'later'"):
            ControllerParams(goal_replacement="later")
        with pytest.raises(ValueError, match="Unknown default_direction 'sideways'"):
            ControllerParams(default_direction="sideways")

    @pytest.mark.parametrize("build", ["direct", "from_dict"])
    def test_enum_fields_take_their_string_values(self, build):
        """Both enum fields are stored as the enum however the params are built."""
        values = {"goal_replacement": "reject", "default_direction": "backward"}
        params = ControllerParams(**values) if build == "direct" else ControllerParams.from_dict(values)
        assert params.goal_replacement is GoalReplacement.REJECT
        assert params.default_direction is MovementDirection.BACKWARD
        assert ControllerParams().goal_replacement is None  # unset: a manager or the default decides

    def test_enum_fields_take_their_string_values_when_assigned_later(self, sim, batch):
        """The documented way to change one agent: assign after construction."""
        params = ControllerParams()
        params.goal_replacement = "reject"  # type: ignore[assignment]  # strings convert at run time
        params.default_direction = "backward"  # type: ignore[assignment]
        assert params.goal_replacement is GoalReplacement.REJECT
        assert params.default_direction is MovementDirection.BACKWARD
        with pytest.raises(ValueError, match="Unknown goal_replacement 'later'"):
            params.goal_replacement = "later"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown default_direction 'sideways'"):
            params.default_direction = "sideways"  # type: ignore[assignment]
        params.goal_replacement = None  # back to unset
        assert params.goal_replacement is None

        agent = _agent(sim, batch, MotionMode.OMNIDIRECTIONAL)
        agent.controller_params.goal_replacement = "reject"  # type: ignore[assignment]
        _cruise(sim, agent)
        assert agent.set_path(_goal(OFF_LINE), auto_approach=False) is False  # rejected, not braked

    def test_as_config_writes_plain_values_that_build_the_same_params(self):
        params = ControllerParams(max_linear_vel=1.5, goal_replacement="wait_complete", default_direction="auto")
        config = params.as_config()
        assert config["goal_replacement"] == "wait_complete" and type(config["goal_replacement"]) is str
        assert config["default_direction"] == "auto" and type(config["default_direction"]) is str
        assert ControllerParams.from_dict(config) == params


class TestCheckpoint:
    @pytest.mark.parametrize("mode, after", [("brake", 3), ("wait_complete", 3)])
    def test_a_held_goal_survives_a_checkpoint(self, motion, mode, after):
        from pybullet_fleet.replay.kinematic_profile import capture_agent_state, restore_agent_state

        def world():
            sc = _world()
            return sc, _agent(sc, motion=motion, mode=mode, name="robot")

        source, robot = world()
        try:
            _cruise(source, robot)
            robot.set_path(_goal(OFF_LINE), auto_approach=False, final_orientation_align=False)
            for _ in range(after):
                source.step_once()
            state = capture_agent_state(robot)
            boundary = source.get_completed_step_boundary()
            replacement = state["controller"]["state"]["replacement"]
            assert replacement["pending"]["path"][0]["position"] == pytest.approx([*OFF_LINE, 0.1])
            assert replacement["braking"] is (mode == "brake")
            reference = _run(source, robot)
        finally:
            p.disconnect(source.client)

        target, restored = world()
        try:
            restore_agent_state(restored, state)
            target.restore_completed_step_boundary(*boundary)
            resumed = _run(target, restored)
        finally:
            p.disconnect(target.client)
        assert len(resumed) == len(reference)
        assert resumed == pytest.approx(reference, abs=1e-9)

    def test_an_invalid_held_goal_is_rejected(self, sim):
        from pybullet_fleet.replay.kinematic_profile import _validate_navigation, capture_agent_state

        agent = _agent(sim, mode="brake", name="robot")
        _cruise(sim, agent)
        agent.set_path(_goal(OFF_LINE), auto_approach=False)
        navigation = capture_agent_state(agent)["controller"]["state"]
        _validate_navigation(navigation, differential=False)
        for change, message in [
            (lambda r: r.update(pending=None), "braking state"),
            (lambda r: r["pending"].update(direction="sideways"), "pending direction"),
            (lambda r: r["pending"].update(path=[]), "pending path"),
            (lambda r: r.update(extra=1), "missing or unsupported"),
        ]:
            import copy

            broken = copy.deepcopy(navigation)
            change(broken["replacement"])
            with pytest.raises(ValueError, match=message):
                _validate_navigation(broken, differential=False)


class TestDifferentialHeadingTolerance:
    """A moving differential agent takes a goal whose line is within
    HEADING_TOL of its heading as straight on, as omni does for its cross-track
    tolerance; a larger turn is a turn in place, braked for in brake mode."""

    @pytest.mark.parametrize("turn, continues", [(1e-4, True), (1e-2, False)], ids=["within", "beyond"])
    def test_a_slightly_off_goal(self, sim, batch, turn, continues):
        agent = _agent(sim, batch, MotionMode.DIFFERENTIAL)  # brake, the default
        x, y = _cruise(sim, agent)
        ahead = 6.0
        # Facing along the line to it, so the heading the agent needs is the line's.
        goal = Pose.from_yaw(x + ahead, y + ahead * np.tan(turn), 0.1, turn)
        agent.set_path([goal], auto_approach=False, final_orientation_align=False)
        assert (_pending(agent) is None) is continues  # nothing held: continued at once
        track = [_xy(agent)]
        for _ in range(5):
            sim.step_once()
            track.append(_xy(agent))
        speeds = _speeds(np.array(track))
        if continues:
            assert speeds == pytest.approx([VMAX] * 5, rel=1e-3)
        else:
            assert speeds[-1] < VMAX - DECEL * DT * 4 + 1e-6  # braking


class TestMoveActionOutcome:
    """A MoveAction ends COMPLETED only at its own path's end; stopped elsewhere
    -- its held goal replaced by a newer one -- it ends CANCELLED."""

    @staticmethod
    def _move(agent, xy):
        from pybullet_fleet.action import MoveAction
        from pybullet_fleet.geometry import Path

        action = MoveAction(
            path=Path.from_positions([[xy[0], xy[1], 0.1]]), auto_approach=False, final_orientation_align=False
        )
        agent.add_action(action)
        return action

    @staticmethod
    def _until_done(sim, action, steps=800):
        from pybullet_fleet.types import ActionStatus

        for _ in range(steps):
            sim.step_once()
            if action.status not in (ActionStatus.NOT_STARTED, ActionStatus.IN_PROGRESS):
                return
        raise AssertionError(f"action still {action.status}")

    def test_completes_at_its_end(self, sim, batch, motion):
        from pybullet_fleet.types import ActionStatus

        agent = _agent(sim, batch, motion)
        action = self._move(agent, (3.0, 0.0))
        self._until_done(sim, action)
        assert action.status == ActionStatus.COMPLETED
        assert _xy(agent) == pytest.approx([3.0, 0.0], abs=1e-6)

    def test_a_held_goal_replaced_by_a_newer_one_is_cancelled(self, sim, batch, motion):
        from pybullet_fleet.types import ActionStatus

        agent = _agent(sim, batch, motion)  # brake, the default
        _cruise(sim, agent)
        action = self._move(agent, OFF_LINE)
        sim.step_once()
        assert action.status == ActionStatus.IN_PROGRESS and _pending(agent) is not None  # held while braking
        newer = (0.0, -5.0)
        agent.set_path(_goal(newer), auto_approach=False, final_orientation_align=False)  # drops the held one
        self._until_done(sim, action)
        assert action.status == ActionStatus.CANCELLED
        assert _xy(agent) == pytest.approx(newer, abs=1e-6)


class TestDefaultDirection:
    """A path without a direction uses the agent's default_direction, per-agent
    and batched alike, both to judge a replacement and to drive it."""

    @staticmethod
    def _differential(sim, batch, mode, default_direction="forward"):
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/simple_cube.urdf",
                initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
                motion_mode=MotionMode.DIFFERENTIAL,
                collision_mode=CollisionMode.NORMAL_3D,
                controller={
                    "max_linear_vel": VMAX,
                    "max_linear_accel": ACCEL,
                    "max_linear_decel": DECEL,
                    "goal_replacement": mode,
                    "default_direction": default_direction,
                },
            ),
            sim_core=sim,
        )
        if batch:
            from pybullet_fleet.agent_manager import AgentManager

            AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"}).add_object(agent)
        return agent

    @pytest.mark.parametrize("mode", ["reject", "brake"])
    def test_backing_up_then_a_goal_behind_without_a_direction(self, sim, batch, mode):
        """Backing up under ``direction="auto"``, a new goal further behind with
        no direction is a forward path (the default): it needs a turn, so
        ``reject`` refuses it and the agent keeps backing up, and ``brake``
        stops it at the deceleration before it turns."""
        agent = self._differential(sim, batch, mode)
        agent.set_path([Pose.from_xyz(-10.0, 0.0, 0.1)], direction="auto", auto_approach=False, final_orientation_align=False)
        for _ in range(15):
            sim.step_once()
        assert agent.velocity[0] == pytest.approx(-VMAX)
        x0 = _xy(agent)[0]
        taken = agent.set_path([Pose.from_xyz(-20.0, 0.0, 0.1)], auto_approach=False, final_orientation_align=False)
        assert taken is (mode != "reject")
        xs = [x0]
        for _ in range(10):
            sim.step_once()
            xs.append(_xy(agent)[0])
        speeds = -np.diff(xs) / DT  # backing up: x decreases
        if mode == "reject":
            assert speeds == pytest.approx([VMAX] * 10)
        else:
            assert np.all(np.diff(speeds) <= 1e-9)  # never speeds up
            assert np.max(-np.diff(np.r_[VMAX, speeds])) <= DECEL * DT + 1e-6  # within the deceleration

    def test_a_backward_default_drives_alike_per_agent_and_batched(self):
        tracks = []
        for batch in (False, True):
            sc = _world()
            try:
                agent = self._differential(sc, batch, "brake", default_direction="backward")
                agent.set_path([Pose.from_xyz(3.0, 0.0, 0.1)], auto_approach=False, final_orientation_align=False)
                track = []
                for _ in range(80):
                    sc.step_once()
                    track.append(tuple(agent.get_pose().position[:2]) + (agent.get_pose().yaw,))
                tracks.append(track)
            finally:
                p.disconnect(sc.client)
        assert np.asarray(tracks[1]) == pytest.approx(np.asarray(tracks[0]), abs=1e-6)
        assert tracks[0][-1][0] == pytest.approx(3.0, abs=1e-6)
