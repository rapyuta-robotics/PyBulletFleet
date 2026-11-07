# tests/test_events.py
"""Tests for EventBus class and sim-level event integration."""

import logging

import pybullet as p
import pybullet_data
import pytest

from pybullet_fleet.events import EventBus, SimEvents
from pybullet_fleet import Agent, AgentSpawnParams, Pose
from pybullet_fleet.sim_object import SimObject, ShapeParams
from pybullet_fleet.types import MotionMode, CollisionMode


# === EventBus unit tests ===


class TestEventBusUnit:
    """Unit tests for EventBus class in isolation."""

    def test_on_emit(self):
        bus = EventBus()
        results = []
        bus.on("test", lambda **kw: results.append(kw))
        bus.emit("test", x=1)
        assert results == [{"x": 1}]

    def test_emit_no_handlers(self):
        """Emit with no handlers does not raise."""
        bus = EventBus()
        bus.emit("nonexistent", x=1)

    def test_priority_order(self):
        bus = EventBus()
        order = []
        bus.on("test", lambda: order.append("B"), priority=10)
        bus.on("test", lambda: order.append("A"), priority=0)
        bus.emit("test")
        assert order == ["A", "B"]

    def test_fifo_within_same_priority(self):
        bus = EventBus()
        order = []
        bus.on("test", lambda: order.append("first"), priority=0)
        bus.on("test", lambda: order.append("second"), priority=0)
        bus.emit("test")
        assert order == ["first", "second"]

    def test_off(self):
        bus = EventBus()
        results = []

        def handler(**kw):
            results.append(1)

        bus.on("test", handler)
        bus.off("test", handler)
        bus.emit("test")
        assert results == []

    def test_off_nonexistent_handler(self):
        """Removing a handler that was never registered doesn't raise."""
        bus = EventBus()
        bus.off("test", lambda: None)

    def test_handler_exception_isolated(self, caplog):
        """Exception in one handler is logged but does not block others."""
        bus = EventBus()
        results = []
        bus.on("test", lambda: 1 / 0, priority=0)
        bus.on("test", lambda: results.append("ok"), priority=1)
        with caplog.at_level(logging.ERROR, logger="pybullet_fleet.events"):
            bus.emit("test")
        assert results == ["ok"]
        assert any("handler error" in r.message and "test" in r.message for r in caplog.records)

    def test_clear_all(self):
        bus = EventBus()
        bus.on("a", lambda: None)
        bus.on("b", lambda: None)
        bus.clear()
        assert not bus.has_handlers("a")
        assert not bus.has_handlers("b")

    def test_clear_specific(self):
        bus = EventBus()
        bus.on("a", lambda: None)
        bus.on("b", lambda: None)
        bus.clear("a")
        assert not bus.has_handlers("a")
        assert bus.has_handlers("b")

    def test_has_handlers(self):
        bus = EventBus()
        assert not bus.has_handlers("x")
        bus.on("x", lambda: None)
        assert bus.has_handlers("x")

    def test_multiple_events_independent(self):
        bus = EventBus()
        a_results = []
        b_results = []
        bus.on("a", lambda: a_results.append(1))
        bus.on("b", lambda: b_results.append(2))
        bus.emit("a")
        assert a_results == [1]
        assert b_results == []

    def test_a_handler_added_during_an_emit_runs_from_the_next_one(self):
        bus = EventBus()
        order = []

        def adder():
            order.append("adder")
            if len(order) == 1:
                bus.on("test", lambda: order.append("added"), priority=-10)

        bus.on("test", adder)
        bus.emit("test")
        assert order == ["adder"]
        bus.emit("test")
        assert order == ["adder", "added", "adder"]

    def test_a_handler_removed_during_an_emit_still_runs_in_it(self):
        bus = EventBus()
        order = []

        def second():
            order.append("second")

        def first():
            order.append("first")
            bus.off("test", second)

        bus.on("test", first)
        bus.on("test", second)
        bus.emit("test")
        bus.emit("test")
        assert order == ["first", "second", "first"]


# === SimEvents constants tests ===


class TestSimEvents:
    """Tests for SimEvents pre-defined event name constants."""

    def test_constants_match_expected_names(self):
        """Verify each constant maps to the expected raw string (implies str type)."""
        assert SimEvents.PRE_STEP == "pre_step"
        assert SimEvents.STEP == "step"
        assert SimEvents.POST_STEP == "post_step"
        assert SimEvents.PAUSED == "paused"
        assert SimEvents.RESUMED == "resumed"
        assert SimEvents.OBJECT_SPAWNED == "object_spawned"
        assert SimEvents.OBJECT_REMOVED == "object_removed"
        assert SimEvents.AGENT_SPAWNED == "agent_spawned"
        assert SimEvents.AGENT_REMOVED == "agent_removed"
        assert SimEvents.COLLISION_STARTED == "collision_started"
        assert SimEvents.COLLISION_ENDED == "collision_ended"
        assert SimEvents.PRE_UPDATE == "pre_update"
        assert SimEvents.POST_UPDATE == "post_update"
        assert SimEvents.ACTION_STARTED == "action_started"
        assert SimEvents.ACTION_COMPLETED == "action_completed"

    def test_raw_string_interop(self):
        """Registering with constant and emitting with raw string (and vice versa) works."""
        bus = EventBus()
        results = []
        bus.on(SimEvents.COLLISION_STARTED, lambda **kw: results.append("const"))
        bus.emit("collision_started", obj_a=None, obj_b=None)
        assert results == ["const"]

        results.clear()
        bus2 = EventBus()
        bus2.on("action_completed", lambda **kw: results.append("raw"))
        bus2.emit(SimEvents.ACTION_COMPLETED, action=None, status=None)
        assert results == ["raw"]

    def test_no_duplicate_values(self):
        """All SimEvents constant values are unique."""
        values = []
        for attr in dir(SimEvents):
            if attr.startswith("_"):
                continue
            values.append(getattr(SimEvents, attr))
        assert len(values) == len(set(values)), "Duplicate event name values found"


# === Per-entity EventBus tests ===


@pytest.fixture
def pybullet_direct():
    """Headless PyBullet session for SimObject tests."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    yield cid
    p.disconnect(cid)


class TestPerEntityEventBus:
    """Tests for lazy per-entity EventBus on SimObject."""

    def test_lazy_creation_not_created(self, pybullet_direct):
        """Bus is not created until .events is accessed."""
        from pybullet_fleet.sim_object import SimObject, ShapeParams

        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        )
        assert obj._events is None
        assert not obj._has_entity_events()

    def test_lazy_creation_triggered_on_access(self, pybullet_direct):
        """Accessing .events creates the EventBus."""
        from pybullet_fleet.sim_object import SimObject, ShapeParams

        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        )
        bus = obj.events
        assert bus is not None
        assert obj._has_entity_events()

    def test_per_entity_on_emit(self, pybullet_direct):
        """Per-entity EventBus on/emit works."""
        from pybullet_fleet.sim_object import SimObject, ShapeParams

        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        )
        results = []
        obj.events.on("test_event", lambda **kw: results.append(kw))
        obj.events.emit("test_event", value=42)
        assert results == [{"value": 42}]


# === core_simulation integration tests ===


class TestGlobalEventBus:
    """Tests for Global EventBus on MultiRobotSimulationCore."""

    def test_pre_step_fires(self, sim_core):
        events = []
        sim_core.events.on("pre_step", lambda **kw: events.append(("pre", kw)))
        sim_core.step_once()
        assert len(events) == 1
        assert events[0][0] == "pre"
        assert "dt" in events[0][1]
        assert "sim_time" in events[0][1]

    def test_pre_before_post(self, sim_core):
        order = []
        sim_core.events.on("pre_step", lambda **kw: order.append("pre"))
        sim_core.events.on("post_step", lambda **kw: order.append("post"))
        sim_core.step_once()
        assert order == ["pre", "post"]

    def test_pre_step_skipped_when_paused(self, sim_core):
        events = []
        sim_core.events.on("pre_step", lambda **kw: events.append(1))
        sim_core.pause()
        sim_core.step_once()
        assert events == []


class TestLifecycleEvents:
    """Tests for entity spawn/remove events."""

    def test_object_spawned(self, sim_core):
        spawned = []
        sim_core.events.on("object_spawned", lambda obj: spawned.append(obj))
        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            sim_core=sim_core,
        )
        assert len(spawned) == 1
        assert spawned[0] is obj

    def test_agent_spawned(self, sim_core):
        added = []
        sim_core.events.on("agent_spawned", lambda agent: added.append(agent))
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        assert len(added) == 1
        assert added[0] is agent

    def test_agent_spawned_also_fires_object_spawned(self, sim_core):
        """Agent is a SimObject, so object_spawned should also fire."""
        spawned = []
        sim_core.events.on("object_spawned", lambda obj: spawned.append(obj))
        Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        assert len(spawned) == 1

    def test_object_removed(self, sim_core):
        removed = []
        sim_core.events.on("object_removed", lambda obj: removed.append(obj))
        obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            sim_core=sim_core,
        )
        sim_core.remove_object(obj)
        assert len(removed) == 1
        assert removed[0] is obj

    def test_agent_removed(self, sim_core):
        removed = []
        sim_core.events.on("agent_removed", lambda agent: removed.append(agent))
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        sim_core.remove_object(agent)
        assert len(removed) == 1


class TestCollisionEvents:
    """Tests for collision Enter/Exit events."""

    def test_collision_started_fires(self, sim_core):
        """Two overlapping objects → collision_started fires."""
        started = []
        sim_core.events.on("collision_started", lambda obj_a, obj_b: started.append((obj_a, obj_b)))
        SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0.3, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        sim_core.step_once()
        assert len(started) >= 1

    def test_collision_started_not_repeated(self, sim_core):
        """Sustained collision does NOT re-fire collision_started."""
        started = []
        sim_core.events.on("collision_started", lambda obj_a, obj_b: started.append(1))
        SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0.3, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        sim_core.step_once()
        count_after_first = len(started)
        sim_core.step_once()
        assert len(started) == count_after_first

    def test_collision_ended_fires(self, sim_core):
        """Objects separate → collision_ended fires."""
        ended = []
        sim_core.events.on("collision_ended", lambda obj_a, obj_b: ended.append((obj_a, obj_b)))
        SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        obj_b = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0.3, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        sim_core.step_once()  # collision_started
        obj_b.set_pose(Pose.from_xyz(10, 10, 0.5))
        sim_core.step_once()  # collision_ended
        assert len(ended) >= 1

    def test_per_entity_collision_started(self, sim_core):
        """Per-entity collision_started fires with 'other' kwarg."""
        obj_a = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        obj_b = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            pose=Pose.from_xyz(0.3, 0, 0.5),
            sim_core=sim_core,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        a_collisions = []
        obj_a.events.on("collision_started", lambda other: a_collisions.append(other))
        sim_core.step_once()
        assert len(a_collisions) == 1
        assert a_collisions[0] is obj_b


class TestPauseResumeEvents:
    def test_paused_event(self, sim_core):
        events = []
        sim_core.events.on("paused", lambda: events.append("paused"))
        sim_core.pause()
        assert events == ["paused"]

    def test_resumed_event(self, sim_core):
        events = []
        sim_core.events.on("resumed", lambda: events.append("resumed"))
        sim_core.pause()
        sim_core.resume()
        assert events == ["resumed"]


class TestActionEvents:
    """Tests for action_started / action_completed events."""

    def test_action_started_fires(self, sim_core):
        from pybullet_fleet.action import WaitAction

        started = []
        sim_core.events.on("action_started", lambda agent, action: started.append((agent, action)))
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        wait = WaitAction(duration=0.001)
        agent.add_action(wait)
        sim_core.step_once()
        assert len(started) == 1
        assert started[0][0] is agent
        assert started[0][1] is wait

    def test_action_completed_fires(self, sim_core):
        from pybullet_fleet.action import WaitAction
        from pybullet_fleet.types import ActionStatus

        completed = []
        sim_core.events.on("action_completed", lambda agent, action, status: completed.append(status))
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        agent.add_action(WaitAction(duration=0.001))
        # Step enough for wait to complete
        for _ in range(10):
            sim_core.step_once()
        assert len(completed) >= 1
        assert completed[0] == ActionStatus.COMPLETED

    def test_per_entity_action_started(self, sim_core):
        from pybullet_fleet.action import WaitAction

        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        per_entity = []
        agent.events.on("action_started", lambda action: per_entity.append(action))
        agent.add_action(WaitAction(duration=0.001))
        sim_core.step_once()
        assert len(per_entity) == 1

    def test_per_entity_action_completed(self, sim_core):
        from pybullet_fleet.action import WaitAction
        from pybullet_fleet.types import ActionStatus

        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        per_entity = []
        agent.events.on("action_completed", lambda action, status: per_entity.append((action, status)))
        wait = WaitAction(duration=0.001)
        agent.add_action(wait)
        for _ in range(10):
            sim_core.step_once()
        assert len(per_entity) == 1
        assert per_entity[0][0] is wait
        assert per_entity[0][1] == ActionStatus.COMPLETED


class TestPerEntityUpdateEvents:
    """Tests for per-entity pre_update / post_update events."""

    def test_pre_update_fires(self, sim_core):
        """pre_update fires each step with dt."""
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        events = []
        agent.events.on(SimEvents.PRE_UPDATE, lambda **kw: events.append(kw))
        sim_core.step_once()
        assert len(events) == 1
        assert "dt" in events[0]
        assert events[0]["dt"] == sim_core.params.timestep

    def test_post_update_fires_with_moved(self, sim_core):
        """post_update fires with dt and moved flag."""
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        events = []
        agent.events.on(SimEvents.POST_UPDATE, lambda **kw: events.append(kw))
        sim_core.step_once()
        assert len(events) == 1
        assert "dt" in events[0]
        assert "moved" in events[0]
        assert isinstance(events[0]["moved"], bool)

    def test_pre_before_post_update(self, sim_core):
        """pre_update fires before post_update in same step."""
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        order = []
        agent.events.on(SimEvents.PRE_UPDATE, lambda **kw: order.append("pre"))
        agent.events.on(SimEvents.POST_UPDATE, lambda **kw: order.append("post"))
        sim_core.step_once()
        assert order == ["pre", "post"]

    def test_no_event_bus_no_overhead(self, sim_core):
        """Agent without .events access has no EventBus created."""
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
            ),
            sim_core=sim_core,
        )
        assert agent._events is None
        sim_core.step_once()
        # EventBus still not created — zero overhead
        assert agent._events is None

    def test_post_update_moved_true_when_moving(self, sim_core):
        """post_update reports moved=True when agent is navigating."""
        from pybullet_fleet.action import MoveAction
        from pybullet_fleet.geometry import Path

        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                motion_mode=MotionMode.OMNIDIRECTIONAL,
                controller={"max_linear_vel": 2.0},
            ),
            sim_core=sim_core,
        )
        events = []
        agent.events.on(SimEvents.POST_UPDATE, lambda **kw: events.append(kw))
        agent.add_action(MoveAction(path=Path.from_positions([[5, 5, 0.1]])))
        sim_core.step_once()
        assert len(events) == 1
        assert events[0]["moved"] is True
