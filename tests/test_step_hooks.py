"""Per-step hooks: ``register_callback()``, ``events.on()`` and plugin ``on_step()``.

The first group pins the observable behaviour each mechanism has always had --
order, schedule, arguments and exception handling -- so that driving all three
through the event bus changes none of it. The second group covers what the bus
adds: an event at the ``STEP`` point and frequency-limited subscriptions.
"""

import gc
import logging
import time
import weakref

import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, CallbackPhase, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.events import EventBus, SimEvents
from pybullet_fleet.sim_plugin import SimPlugin


@pytest.fixture
def sim():
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, monitor=False, timestep=0.1, target_rtf=0, monitor_log_interval=0)
    )
    sim.initialize_simulation()
    yield sim
    p.disconnect(sim.client)


class _Recorder(SimPlugin):
    """A plugin that appends ``(name, dt)`` to a shared list on each ``on_step``."""

    def __init__(self, sim_core, name="plugin", log=None, fail=False):
        super().__init__(sim_core)
        self.name = name
        self.log = log if log is not None else []
        self.fail = fail

    def on_step(self, dt):
        self.log.append((self.name, dt))
        if self.fail:
            raise RuntimeError("plugin boom")


def _names(log):
    return [entry[0] if isinstance(entry, tuple) else entry for entry in log]


# ---------------------------------------------------------------------------
# Behaviour that must not change
# ---------------------------------------------------------------------------


class TestOrder:
    def test_one_step_runs_the_hooks_in_a_fixed_order(self, sim):
        """pre_step event, STEP callbacks, plugins, post_step event, POST_STEP callbacks --
        regardless of the order in which they were registered."""
        log = []
        sim.register_callback(lambda sc, dt: log.append("post_cb"), phase=CallbackPhase.POST_STEP)
        sim.register_plugin(_Recorder(sim, "plugin", log))
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("post_event"))
        sim.register_callback(lambda sc, dt: log.append("step_cb"))
        sim.events.on(SimEvents.PRE_STEP, lambda **_: log.append("pre_event"))
        sim.step_once()
        assert _names(log) == ["pre_event", "step_cb", "plugin", "post_event", "post_cb"]

    def test_callbacks_run_in_registration_order(self, sim):
        log = []
        for name in "abc":
            sim.register_callback(lambda sc, dt, name=name: log.append(name))
        sim.step_once()
        assert log == ["a", "b", "c"]

    def test_plugins_run_in_load_order_after_every_step_callback(self, sim):
        log = []
        sim.register_plugin(_Recorder(sim, "p1", log))
        sim.register_plugin(_Recorder(sim, "p2", log))
        sim.register_callback(lambda sc, dt: log.append("cb"))
        sim.step_once()
        assert _names(log) == ["cb", "p1", "p2"]

    def test_post_step_callbacks_follow_every_post_step_handler_whatever_its_priority(self, sim):
        log = []
        sim.register_callback(lambda sc, dt: log.append("cb"), phase="post_step")
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("late"), priority=1000)
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("early"), priority=-1000)
        sim.step_once()
        assert log == ["early", "late", "cb"]

    def test_a_callback_registered_during_a_step_first_runs_in_the_next(self, sim):
        log = []

        def adder(sc, dt):
            if not log:
                sc.register_callback(lambda sc, dt: log.append("added"))
            log.append("adder")

        sim.register_callback(adder)
        sim.step_once()
        assert log == ["adder"]
        sim.step_once()
        assert log == ["adder", "adder", "added"]

    def test_a_callback_unregistered_during_a_step_still_runs_in_it(self, sim):
        log = []

        def second(sc, dt):
            log.append("second")

        def first(sc, dt):
            log.append("first")
            sc.unregister_callback(second)

        sim.register_callback(first)
        sim.register_callback(second)
        sim.step_once()
        sim.step_once()
        assert log == ["first", "second", "first"]

    def test_a_callback_registered_by_a_step_handler_first_runs_in_the_next_step(self, sim):
        """The STEP event calls the subscriptions present when it starts, across its tiers too."""
        log = []

        def handler(**_):
            if not log:
                sim.register_callback(lambda sc, dt: log.append("added"))
            log.append("handler")

        sim.events.on(SimEvents.STEP, handler)
        sim.step_once()
        assert log == ["handler"]
        sim.step_once()
        assert log == ["handler", "handler", "added"]

    def test_a_plugin_registered_by_a_step_callback_first_runs_in_the_next_step(self, sim):
        log = []

        def adder(sc, dt):
            if not log:
                sc.register_plugin(_Recorder(sim, "added", log))
            log.append("adder")

        sim.register_callback(adder)
        sim.step_once()
        assert _names(log) == ["adder"]
        sim.step_once()
        assert _names(log) == ["adder", "adder", "added"]

    def test_a_callback_unregistered_by_a_step_handler_still_runs_in_that_step(self, sim):
        log = []

        def callback(sc, dt):
            log.append("callback")

        def handler(**_):
            log.append("handler")
            sim.unregister_callback(callback)

        sim.register_callback(callback)
        sim.events.on(SimEvents.STEP, handler)
        sim.step_once()
        sim.step_once()
        assert log == ["handler", "callback", "handler"]

    def test_a_post_step_callback_registered_by_a_post_step_handler_first_runs_in_the_next_step(self, sim):
        log = []

        def handler(**_):
            if not log:
                sim.register_callback(lambda sc, dt: log.append("added"), phase=CallbackPhase.POST_STEP)
            log.append("handler")

        sim.events.on(SimEvents.POST_STEP, handler)
        sim.step_once()
        assert log == ["handler"]
        sim.step_once()
        assert log == ["handler", "handler", "added"]


class TestArguments:
    def test_a_callback_gets_the_core_and_the_time_since_it_last_ran(self, sim):
        seen = []
        sim.register_callback(lambda *args, **kwargs: seen.append((args, kwargs)), frequency=5.0)
        for _ in range(4):
            sim.step_once()
        assert len(seen) == 2
        for args, kwargs in seen:
            assert kwargs == {}
            assert args[0] is sim
            assert args[1] == pytest.approx(0.2)

    def test_a_plugin_gets_the_time_since_it_last_ran(self, sim):
        plugin = sim.register_plugin(_Recorder(sim), frequency=5.0)
        for _ in range(4):
            sim.step_once()
        assert [dt for _, dt in plugin.log] == [pytest.approx(0.2), pytest.approx(0.2)]

    def test_step_events_get_the_timestep_and_the_time_they_observe(self, sim):
        seen = []
        sim.events.on(SimEvents.PRE_STEP, lambda **kw: seen.append(("pre", kw)))
        sim.events.on(SimEvents.POST_STEP, lambda **kw: seen.append(("post", kw)))
        sim.step_once()
        assert [name for name, _ in seen] == ["pre", "post"]
        assert seen[0][1] == {"dt": pytest.approx(0.1), "sim_time": pytest.approx(0.0)}
        assert seen[1][1] == {"dt": pytest.approx(0.1), "sim_time": pytest.approx(0.1)}


class TestExceptions:
    def test_a_callback_exception_propagates_and_skips_the_rest_of_the_step(self, sim):
        log = []

        def boom(sc, dt):
            raise RuntimeError("callback boom")

        sim.register_callback(boom)
        sim.register_callback(lambda sc, dt: log.append("later_cb"))
        sim.register_plugin(_Recorder(sim, "plugin", log))
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("post_event"))
        with pytest.raises(RuntimeError, match="callback boom"):
            sim.step_once()
        assert log == []
        assert sim.step_count == 1  # the clock had already advanced
        assert sim._in_step is False

    def test_a_post_step_callback_exception_propagates(self, sim):
        sim.register_callback(lambda sc, dt: 1 / 0, phase="post_step")
        with pytest.raises(ZeroDivisionError):
            sim.step_once()

    def test_a_plugin_exception_is_logged_and_the_step_goes_on(self, sim, caplog):
        log = []
        sim.register_plugin(_Recorder(sim, "bad", log, fail=True))
        sim.register_plugin(_Recorder(sim, "good", log))
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("post_event"))
        with caplog.at_level(logging.ERROR):
            sim.step_once()
        assert _names(log) == ["bad", "good", "post_event"]
        assert "Plugin _Recorder.on_step() failed" in caplog.text

    def test_an_event_handler_exception_is_logged_and_the_step_goes_on(self, sim, caplog):
        log = []

        def boom(**_):
            raise RuntimeError("handler boom")

        sim.events.on(SimEvents.POST_STEP, boom)
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("next"))
        sim.register_callback(lambda sc, dt: log.append("post_cb"), phase="post_step")
        with caplog.at_level(logging.ERROR):
            sim.step_once()
        assert log == ["next", "post_cb"]
        assert "EventBus handler error for event 'post_step'" in caplog.text


class TestRegistry:
    def test_unregister_removes_the_earliest_registration_of_a_function(self, sim):
        log = []

        def cb(sc, dt):
            log.append(sc.sim_time)

        sim.register_callback(cb, phase="post_step")
        sim.register_callback(cb)
        assert sim.unregister_callback(cb) is True
        sim.step_once()
        assert len(log) == 1  # the STEP one is left
        assert len(sim._callbacks) == 1 and sim._callbacks[0]["phase"] == CallbackPhase.STEP
        assert sim.unregister_callback(cb) is True
        assert sim.unregister_callback(cb) is False

    def test_the_registry_lists_function_phase_and_schedule(self, sim):
        def cb(sc, dt):
            pass

        sim.register_callback(cb, frequency=4.0, phase="post_step")
        (info,) = sim._callbacks
        assert info["func"] is cb
        assert info["phase"] == CallbackPhase.POST_STEP
        assert info["gate"].frequency == 4.0

    def test_the_registry_keeps_registration_order_across_phases(self, sim):
        def a(sc, dt):
            pass

        def b(sc, dt):
            pass

        def c(sc, dt):
            pass

        sim.events.on(SimEvents.COLLISION_STARTED, lambda **_: None)
        sim.register_callback(a, phase="post_step")
        sim.events.on(SimEvents.STEP, lambda **_: None)
        sim.register_callback(b)
        sim.register_callback(c, phase="post_step")
        assert [info["func"] for info in sim._callbacks] == [a, b, c]
        assert [info["phase"] for info in sim._callbacks] == [
            CallbackPhase.POST_STEP,
            CallbackPhase.STEP,
            CallbackPhase.POST_STEP,
        ]

    def test_frequency_zero_means_every_step(self, sim):
        log = []
        sim.register_callback(lambda sc, dt: log.append(dt), frequency=0)
        for _ in range(3):
            sim.step_once()
        assert len(log) == 3

    def test_a_negative_frequency_is_rejected(self, sim):
        with pytest.raises(ValueError):
            sim.register_callback(lambda sc, dt: None, frequency=-1.0)
        assert sim._callbacks == []

    def test_clearing_the_event_bus_keeps_callbacks_and_plugins(self, sim):
        log = []
        sim.register_callback(lambda sc, dt: log.append("cb"))
        sim.register_plugin(_Recorder(sim, "plugin", log))
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("event"))
        sim.events.clear()
        sim.step_once()
        assert _names(log) == ["cb", "plugin"]

    def test_a_dropped_simulation_is_freed_without_the_cycle_collector(self):
        """The bus's clock and callbacks do not keep the simulation alive."""
        gc.collect()
        gc.disable()
        try:
            sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, timestep=0.1, target_rtf=0))
            client = sim.client
            sim.register_callback(lambda sc, dt: None, frequency=5.0)
            sim.register_callback(lambda sc, dt: None, phase="post_step")
            sim.events.on(SimEvents.POST_STEP, lambda **_: None, frequency=2.0)
            sim.initialize_simulation()
            sim.step_once()
            ref = weakref.ref(sim)
            del sim
            assert ref() is None
        finally:
            gc.enable()
            if p.isConnected(client):
                p.disconnect(client)

    def test_subscriptions_keep_tier_priority_registration_order_however_they_were_added(self, sim):
        log = []
        sim.register_plugin(_Recorder(sim, "plugin", log))
        sim.register_callback(lambda sc, dt: log.append("cb1"))
        sim.events.on(SimEvents.STEP, lambda **_: log.append("h+5"), priority=5)
        sim.register_callback(lambda sc, dt: log.append("cb2"))
        sim.events.on(SimEvents.STEP, lambda **_: log.append("h0a"))
        sim.events.on(SimEvents.STEP, lambda **_: log.append("h-5"), priority=-5)
        sim.events.on(SimEvents.STEP, lambda **_: log.append("h0b"))
        sim.step_once()
        assert _names(log) == ["h-5", "h0a", "h0b", "h+5", "cb1", "cb2", "plugin"]


class TestProfiling:
    def test_plugin_time_is_not_counted_as_callback_time(self, sim):
        class Slow(SimPlugin):
            def on_step(self, dt):
                time.sleep(0.02)

        sim.register_plugin(Slow(sim))
        profile = sim.step_once(return_profiling=True)
        assert profile["callbacks"] < 20.0
        assert profile["phase1_update"] >= 20.0


# ---------------------------------------------------------------------------
# What the event bus adds
# ---------------------------------------------------------------------------


def _agent(sim, x=0.0):
    params = AgentSpawnParams(urdf_path="simple_cube", initial_pose=Pose.from_xyz(x, 0, 0.1), mass=0.0)
    return Agent.from_params(params, sim)


class TestStepEvent:
    def test_the_step_event_is_named_like_the_callback_phase(self):
        assert SimEvents.STEP == "step" == CallbackPhase.STEP.value

    def test_it_runs_after_the_object_update_and_before_the_step_callbacks(self, sim):
        log = []
        sim.register_plugin(_Recorder(sim, "plugin", log))
        sim.register_callback(lambda sc, dt: log.append("step_cb"))
        sim.events.on(SimEvents.STEP, lambda **_: log.append("step_event"))
        sim.events.on(SimEvents.PRE_STEP, lambda **_: log.append("pre_event"))
        sim.events.on(SimEvents.POST_STEP, lambda **_: log.append("post_event"))
        sim.step_once()
        assert _names(log) == ["pre_event", "step_event", "step_cb", "plugin", "post_event"]

    def test_it_sees_the_new_state_before_the_pose_flush(self, sim):
        agent = _agent(sim)
        agent.set_goal_pose(Pose.from_xyz(5.0, 0.0, 0.1))
        seen = []

        def on_step(dt, sim_time):
            cached = agent.get_pose().position[0]
            in_pybullet = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=sim.client)[0][0]
            seen.append((dt, sim_time, sim._in_step, cached, in_pybullet))

        sim.events.on(SimEvents.STEP, on_step)
        sim.step_once()
        sim.step_once()
        dt, sim_time, in_step, cached, in_pybullet = seen[-1]
        assert dt == pytest.approx(0.1)
        assert sim_time == pytest.approx(0.2)  # the time of the state it sees
        assert in_step is True  # set_pose() here is buffered like in a STEP callback
        assert cached > in_pybullet  # the flush has not happened yet

    def test_it_runs_before_the_collision_check(self, sim, monkeypatch):
        log = []
        real = sim.check_collisions
        monkeypatch.setattr(sim, "check_collisions", lambda **kw: (log.append("collision"), real(**kw))[1])
        sim.events.on(SimEvents.STEP, lambda **_: log.append("step_event"))
        sim.step_once()
        assert log == ["step_event", "collision"]

    def test_handlers_follow_their_priority(self, sim):
        log = []
        sim.events.on(SimEvents.STEP, lambda **_: log.append("b"), priority=5)
        sim.events.on(SimEvents.STEP, lambda **_: log.append("a"), priority=-5)
        sim.step_once()
        assert log == ["a", "b"]

    def test_its_exceptions_are_logged_like_any_event(self, sim, caplog):
        log = []
        sim.events.on(SimEvents.STEP, lambda **_: 1 / 0)
        sim.register_callback(lambda sc, dt: log.append("cb"))
        with caplog.at_level(logging.ERROR):
            sim.step_once()
        assert log == ["cb"]
        assert "EventBus handler error for event 'step'" in caplog.text

    def test_it_is_profiled_apart_from_callbacks(self, sim):
        sim.events.on(SimEvents.STEP, lambda **_: time.sleep(0.02))
        profile = sim.step_once(return_profiling=True)
        assert profile["events_step"] >= 20.0
        assert profile["callbacks"] < 20.0


class TestFrequencyLimitedEvents:
    @pytest.mark.parametrize("event", [SimEvents.STEP, SimEvents.POST_STEP])
    def test_runs_on_the_fixed_grid_with_the_time_since_it_last_ran(self, sim, event):
        seen = []
        sim.events.on(event, lambda dt, sim_time: seen.append((round(sim_time, 9), dt)), frequency=5.0)
        for _ in range(20):
            sim.step_once()
        assert [t for t, _ in seen] == [round(0.2 * k, 9) for k in range(1, 11)]
        assert all(dt == pytest.approx(0.2) for _, dt in seen)

    def test_matches_a_callback_at_the_same_rate_beat_for_beat(self, sim):
        """3 Hz on a 0.1 s timestep is not a whole number of steps; both must still agree."""
        from_event, from_callback = [], []
        sim.events.on(SimEvents.STEP, lambda dt, sim_time: from_event.append((sim.step_count, dt)), frequency=3.0)
        sim.register_callback(lambda sc, dt: from_callback.append((sc.step_count, dt)), frequency=3.0)
        for _ in range(100):
            sim.step_once()
        assert from_event == from_callback
        assert len(from_event) == 30

    def test_pre_step_is_scheduled_on_the_time_it_observes(self, sim):
        seen = []
        sim.events.on(SimEvents.PRE_STEP, lambda dt, sim_time: seen.append(round(sim_time, 9)), frequency=5.0)
        for _ in range(7):
            sim.step_once()
        assert seen == [0.2, 0.4, 0.6]

    def test_a_handler_without_a_frequency_still_gets_every_step(self, sim):
        every, limited = [], []
        sim.events.on(SimEvents.STEP, lambda **kw: every.append(kw["dt"]))
        sim.events.on(SimEvents.STEP, lambda **kw: limited.append(kw["dt"]), frequency=2.0)
        for _ in range(10):
            sim.step_once()
        assert len(every) == 10 and all(dt == pytest.approx(0.1) for dt in every)
        assert limited == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_off_removes_a_frequency_limited_handler(self, sim):
        seen = []

        def handler(**_):
            seen.append(1)

        sim.events.on(SimEvents.STEP, handler, frequency=10.0)
        sim.step_once()
        sim.events.off(SimEvents.STEP, handler)
        sim.step_once()
        assert seen == [1]

    def test_a_restart_restarts_its_schedule(self, sim):
        seen = []
        sim.events.on(SimEvents.STEP, lambda dt, sim_time: seen.append((round(sim_time, 9), round(dt, 9))), frequency=10.0)
        for _ in range(7):
            sim.step_once()
        seen.clear()
        sim.initialize_simulation()
        sim.step_once()
        assert seen == [(0.1, 0.1)]

    @pytest.mark.parametrize("frequency", [-1.0, float("inf"), float("nan")])
    def test_rejects_a_bad_frequency(self, sim, frequency):
        with pytest.raises(ValueError):
            sim.events.on(SimEvents.STEP, lambda **_: None, frequency=frequency)
        assert not sim.events.has_handlers(SimEvents.STEP)

    @pytest.mark.parametrize("frequency", [0, 0.0])
    def test_frequency_zero_means_every_step_as_for_register_callback(self, sim, frequency):
        from_event, from_callback = [], []
        sim.events.on(SimEvents.STEP, lambda dt, sim_time: from_event.append(dt), frequency=frequency)
        sim.register_callback(lambda sc, dt: from_callback.append(dt), frequency=frequency)
        for _ in range(3):
            sim.step_once()
        assert len(from_event) == 3
        assert from_event == pytest.approx(from_callback)

    def test_frequency_zero_needs_no_clock_or_step_event(self):
        """``0`` is no limit at all, so it is accepted wherever ``None`` is."""
        bus = EventBus()
        seen = []
        bus.on(SimEvents.COLLISION_STARTED, lambda **kw: seen.append(kw), frequency=0)
        bus.emit(SimEvents.COLLISION_STARTED, obj_a=1, obj_b=2)
        assert seen == [{"obj_a": 1, "obj_b": 2}]

    def test_is_only_for_the_step_events(self, sim):
        with pytest.raises(ValueError, match="step events"):
            sim.events.on(SimEvents.COLLISION_STARTED, lambda **_: None, frequency=1.0)

    def test_needs_a_bus_with_a_clock(self):
        """A bare or per-entity bus has no simulation clock to schedule on."""
        with pytest.raises(ValueError, match="clock"):
            EventBus().on(SimEvents.POST_STEP, lambda **_: None, frequency=1.0)

    def test_a_bus_with_a_clock_schedules_on_it(self):
        now = [0.0]
        bus = EventBus(clock=lambda: (now[0], 0.25))
        seen = []
        bus.on("post_step", lambda dt, sim_time: seen.append((sim_time, dt)), frequency=1.0)
        for _ in range(9):
            now[0] += 0.25
            bus.emit("post_step", dt=0.25, sim_time=now[0])
        assert seen == [(1.0, 1.0), (2.0, 1.0)]


# ---------------------------------------------------------------------------
# Every phase for callbacks and plugins
# ---------------------------------------------------------------------------


class TestPhases:
    """Callbacks and plugin ``on_step()`` can run at any of the three step events,
    after that event's handlers (callbacks, then plugins), with a frequency."""

    def test_the_phases_are_the_step_events(self):
        assert [phase.value for phase in CallbackPhase] == [
            SimEvents.PRE_STEP,
            SimEvents.STEP,
            SimEvents.POST_STEP,
        ]

    def test_every_tier_in_every_phase_runs_in_order(self, sim):
        log = []
        for phase in ("post_step", "step", "pre_step"):  # registered in reverse
            sim.register_plugin(_Recorder(sim, f"{phase}:plugin", log), phase=phase)
            sim.register_callback(lambda sc, dt, phase=phase: log.append(f"{phase}:callback"), phase=phase)
            sim.events.on(phase, lambda phase=phase, **_: log.append(f"{phase}:handler"))
        sim.step_once()
        assert _names(log) == [
            f"{phase}:{tier}" for phase in ("pre_step", "step", "post_step") for tier in ("handler", "callback", "plugin")
        ]

    def test_pre_step_hooks_see_the_previous_state(self, sim):
        """Before the clock advances and before anything moves."""
        agent = Agent.from_params(
            AgentSpawnParams(urdf_path="robots/simple_cube.urdf", initial_pose=Pose.from_xyz(0.0, 0.0, 0.1)),
            sim_core=sim,
        )
        agent.set_path([Pose.from_xyz(5.0, 0.0, 0.1)], auto_approach=False, final_orientation_align=False)
        for _ in range(5):
            sim.step_once()
        seen = []

        class Watch(SimPlugin):
            def on_step(self, dt):
                seen.append(("plugin", round(self.sim_core.sim_time, 9), agent.get_pose().position[0]))

        sim.register_plugin(Watch, phase="pre_step")
        sim.register_callback(
            lambda sc, dt: seen.append(("callback", round(sc.sim_time, 9), agent.get_pose().position[0])),
            phase=CallbackPhase.PRE_STEP,
        )
        before = (round(sim.sim_time, 9), agent.get_pose().position[0])
        sim.step_once()
        assert [entry[1:] for entry in seen] == [before, before]
        assert agent.get_pose().position[0] > before[1], "it moved after the pre_step hooks"

    def test_a_post_step_plugin_runs_after_the_collision_check(self, sim, monkeypatch):
        log = []
        check = sim.check_collisions

        def checked(*args, **kwargs):
            log.append("collision_check")
            return check(*args, **kwargs)

        monkeypatch.setattr(sim, "check_collisions", checked)
        sim.register_plugin(_Recorder(sim, "plugin", log), phase=CallbackPhase.POST_STEP)
        sim.step_once()
        assert _names(log) == ["collision_check", "plugin"]

    @pytest.mark.parametrize("phase", ["pre_step", "step", "post_step"])
    def test_a_plugin_at_any_phase_keeps_its_frequency_beat_for_beat_with_a_callback(self, sim, phase):
        from_plugin, from_callback = [], []

        class Beat(SimPlugin):
            def on_step(self, dt):
                from_plugin.append((self.sim_core.step_count, round(dt, 9)))

        sim.register_plugin(Beat, frequency=3.0, phase=phase)
        sim.register_callback(lambda sc, dt: from_callback.append((sc.step_count, round(dt, 9))), frequency=3.0, phase=phase)
        for _ in range(100):
            sim.step_once()
        assert from_plugin == from_callback
        # pre_step observes the times before each step, 0.0 to 9.9: one beat fewer.
        assert len(from_plugin) == (29 if phase == "pre_step" else 30)

    def test_a_plugin_class_can_choose_its_phase(self, sim):
        class Late(SimPlugin):
            default_step_phase = "post_step"

        assert sim.register_plugin(Late).step_phase is CallbackPhase.POST_STEP
        assert sim.register_plugin(Late, phase="pre_step").step_phase is CallbackPhase.PRE_STEP  # per instance
        assert sim.register_plugin(_Recorder(sim)).step_phase is CallbackPhase.STEP  # the default

    def test_a_plugin_keeps_its_own_attribute_named_phase(self, sim):
        """``phase`` is a common name for a plugin's own state; only step_phase is the hook's."""
        log = []

        class Loader(SimPlugin):
            phase = None  # its own state

            def __init__(self, sim_core):
                super().__init__(sim_core)
                self.phase = "loading"

            def on_step(self, dt):
                log.append(self.phase)
                self.phase = "idle"

        plugin = sim.register_plugin(Loader)
        sim.step_once()
        sim.step_once()
        assert log == ["loading", "idle"]
        assert plugin.step_phase is CallbackPhase.STEP

    def test_the_phase_is_fixed_once_registered(self, sim):
        plugin = sim.register_plugin(_Recorder(sim))
        with pytest.raises(RuntimeError, match="fixed once the plugin is registered"):
            plugin.step_phase = "post_step"
        assert plugin.step_phase is CallbackPhase.STEP

    def test_the_yaml_entry_takes_a_phase(self, sim):
        from pybullet_fleet.sim_plugin import create_sim_plugin_from_entry

        plugin = create_sim_plugin_from_entry({"class": f"{__name__}._Recorder", "phase": "pre_step", "frequency": 2.0}, sim)
        assert plugin.step_phase is CallbackPhase.PRE_STEP
        assert plugin.frequency == 2.0

    def test_an_unknown_phase_is_refused(self, sim):
        from pybullet_fleet.sim_plugin import create_sim_plugin_from_entry

        with pytest.raises(ValueError, match="phase must be one of"):
            sim.register_plugin(_Recorder(sim), phase="later")
        with pytest.raises(ValueError, match="phase must be one of"):
            create_sim_plugin_from_entry({"class": f"{__name__}._Recorder", "phase": "later"}, sim)
        with pytest.raises(ValueError, match="phase must be one of"):
            sim.register_callback(lambda sc, dt: None, phase="later")
        with pytest.raises(ValueError, match="phase must be one of"):

            class Bad(SimPlugin):
                default_step_phase = "later"

        assert not sim.plugins

    def test_pre_step_callbacks_are_profiled_apart_from_pre_step_handlers(self, sim):
        sim.register_callback(lambda sc, dt: time.sleep(0.002), phase="pre_step")
        breakdown = sim.step_once(return_profiling=True)
        assert breakdown["callbacks_pre_step"] >= 2.0
        assert breakdown["events_pre_step"] < 2.0
