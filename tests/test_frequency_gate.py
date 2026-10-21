"""Frequency-limited handlers run on a fixed grid of simulated time.

Simulated time carries float error (ten steps of 0.1 s add up to
0.9999999999999999 s); due times on a fixed grid with a small tolerance keep
every beat and keep the phase. The core passes the gates a time computed from
the step count, whose error does not grow with the run. Callbacks, plugins,
the collision check and the DataMonitor refresh all share :class:`FrequencyGate`.
"""

import pytest

from pybullet_fleet import CallbackPhase, MultiRobotSimulationCore, SimulationParams
from pybullet_fleet._frequency import FrequencyGate
from pybullet_fleet.events import SimEvents
from pybullet_fleet.sim_plugin import SimPlugin


def _times(timestep, steps):
    """Times accumulated one timestep at a time, float error included."""
    now, out = 0.0, []
    for _ in range(steps):
        out.append(now)
        now += timestep
    return out


def _fires(gate, timestep, steps):
    return [(now, dt) for now in _times(timestep, steps) for dt in [gate.due(now, timestep)] if dt is not None]


class TestFrequencyGate:
    def test_no_skipped_beats_at_the_step_rate(self):
        fires = _fires(FrequencyGate(10.0), 0.1, 101)
        assert len(fires) == 100  # 0.1 .. 10.0 s
        assert all(dt == pytest.approx(0.1) for _, dt in fires)

    def test_no_drift_when_the_interval_is_not_a_whole_number_of_steps(self):
        # 3 Hz on a 0.1 s timestep: due at 1/3, 2/3, 1 ... s, run on the first
        # step at or after each, so exactly 3 runs per simulated second.
        fires = _fires(FrequencyGate(3.0), 0.1, 301)
        assert len(fires) == 90
        assert [round(now, 6) for now, _ in fires[:6]] == [0.4, 0.7, 1.0, 1.4, 1.7, 2.0]

    def test_first_run_is_one_interval_in(self):
        fires = _fires(FrequencyGate(2.0), 0.1, 6)
        assert [round(now, 6) for now, _ in fires] == [0.5]
        assert fires[0][1] == pytest.approx(0.5)

    def test_a_first_call_on_a_due_time_runs_at_once(self):
        """A handler registered at 9.9 s whose first step ends on the 10 Hz
        grid point 10.0 s runs at 10.0, not one interval later at 10.1."""
        gate = FrequencyGate(10.0)
        assert gate.due(10.0, 0.1) == pytest.approx(0.1)
        assert gate.due(10.1, 0.1) == pytest.approx(0.1)

    def test_a_gap_runs_once_not_once_per_missed_beat(self):
        gate = FrequencyGate(10.0)
        assert gate.due(0.0, 0.1) is None
        assert gate.due(0.1, 0.1) == pytest.approx(0.1)
        assert gate.due(5.0, 0.1) == pytest.approx(4.9)
        assert gate.due(5.0 + 0.05, 0.1) is None
        assert gate.due(5.1, 0.1) == pytest.approx(0.1)

    def test_faster_than_the_step_rate_runs_every_step(self):
        fires = _fires(FrequencyGate(100.0), 0.1, 11)
        assert len(fires) == 10
        assert all(dt == pytest.approx(0.1) for _, dt in fires)

    def test_every_step_without_a_frequency(self):
        fires = _fires(FrequencyGate(None), 0.1, 5)
        assert len(fires) == 5

    @pytest.mark.parametrize("frequency", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_a_bad_frequency(self, frequency):
        with pytest.raises(ValueError):
            FrequencyGate(frequency)


@pytest.fixture
def sim():
    import pybullet as p

    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, monitor=False, timestep=0.1, target_rtf=0, monitor_log_interval=0)
    )
    sim.initialize_simulation()
    yield sim
    p.disconnect(sim.client)


class TestTheCoreUsesIt:
    def test_callback_runs_every_beat(self, sim):
        dts = []
        sim.register_callback(lambda sc, dt: dts.append(dt), frequency=10.0)
        for _ in range(100):  # steps ending at 0.1 .. 10.0 s: one beat each
            sim.step_once()
        assert len(dts) == 100
        assert all(dt == pytest.approx(0.1) for dt in dts)

    def test_plugin_runs_every_beat(self, sim):
        class Counter(SimPlugin):
            def __init__(self, sim_core):
                super().__init__(sim_core)
                self.dts = []

            def on_step(self, dt):
                self.dts.append(dt)

        plugin = sim.register_plugin(Counter(sim), frequency=1.0)
        for _ in range(101):
            sim.step_once()
        # Accumulating 0.1 ten times is 0.9999999999999999; every beat still runs.
        assert len(plugin.dts) == 10
        assert all(dt == pytest.approx(1.0) for dt in plugin.dts)

    def test_collision_check_runs_every_beat(self, sim, monkeypatch):
        checks = []
        real = sim.check_collisions
        monkeypatch.setattr(sim, "check_collisions", lambda **kw: (checks.append(sim.sim_time), real(**kw))[1])
        sim.set_collision_check_frequency(10.0)
        for _ in range(100):  # steps ending at 0.1 .. 10.0 s: one beat each
            sim.step_once()
        assert len(checks) == 100

    def test_plugin_back_from_every_step_starts_a_fresh_schedule(self, sim):
        class Counter(SimPlugin):
            def __init__(self, sim_core):
                super().__init__(sim_core)
                self.dts = []

            def on_step(self, dt):
                self.dts.append(dt)

        plugin = sim.register_plugin(Counter(sim), frequency=1.0)
        for _ in range(15):
            sim.step_once()
        plugin.frequency = None
        for _ in range(30):
            sim.step_once()
        plugin.frequency = 1.0
        plugin.dts.clear()
        sim.step_once()
        # Not an immediate run carrying the whole gap since the last one.
        assert plugin.dts == []
        for _ in range(10):
            sim.step_once()
        assert plugin.dts == [pytest.approx(1.0)]

    def test_a_new_collision_frequency_takes_effect_mid_run(self, sim, monkeypatch):
        checks = []
        monkeypatch.setattr(sim, "check_collisions", lambda **kw: checks.append(sim.sim_time) or ([], None))
        sim.set_collision_check_frequency(10.0)
        for _ in range(10):
            sim.step_once()
        sim.set_collision_check_frequency(2.0)
        for _ in range(10):  # 1.1 .. 2.0 s: the 2 Hz grid points 1.5 and 2.0 s
            sim.step_once()
        assert len(checks) == 12
        assert checks[-2:] == [pytest.approx(1.5), pytest.approx(2.0)]

    def test_collision_check_back_at_the_same_rate_starts_a_fresh_schedule(self, sim, monkeypatch):
        checks = []
        monkeypatch.setattr(sim, "check_collisions", lambda **kw: checks.append(round(sim.sim_time, 9)) or ([], None))
        sim.set_collision_check_frequency(2.0)
        for _ in range(5):  # 0.1 .. 0.5 s: the beat at 0.5 s
            sim.step_once()
        sim.set_collision_check_frequency(0)
        for _ in range(12):  # 0.6 .. 1.7 s, past the beats at 1.0 and 1.5 s
            sim.step_once()
        sim.set_collision_check_frequency(2.0)
        for _ in range(3):  # 1.8 .. 2.0 s
            sim.step_once()
        # Not at once at 1.8 s for the missed beats: on the grid again at 2.0 s.
        assert checks == [0.5, 2.0]

    @pytest.mark.parametrize("via", ["pacing", "params"])
    def test_a_timestep_change_keeps_the_schedule(self, sim, via):
        seen = []
        sim.register_callback(lambda sc, dt: seen.append((round(sc.sim_time, 9), dt)), frequency=2.0)
        for _ in range(5):  # 0.1 .. 0.5 s
            sim.step_once()
        if via == "pacing":
            sim._set_simulation_pacing(None, 0.05, source="test")
        else:
            sim.params.timestep = 0.05
        assert sim._schedule_time() == pytest.approx(0.5)  # the past steps keep their timestep
        for _ in range(10):  # 0.55 .. 1.0 s
            sim.step_once()
        assert sim._schedule_time() == pytest.approx(sim.sim_time)
        assert [t for t, _ in seen] == [0.5, 1.0]
        assert seen[-1][1] == pytest.approx(0.5)

    @pytest.mark.parametrize("frequency", [0, -1.0])
    def test_a_non_positive_collision_frequency_disables_checks(self, sim, monkeypatch, frequency):
        checks = []
        monkeypatch.setattr(sim, "check_collisions", lambda **kw: checks.append(1) or ([], None))
        sim.set_collision_check_frequency(frequency)
        for _ in range(5):
            sim.step_once()
        assert checks == []


class TestLongRuns:
    """sim_time adds one timestep per step, so its float error grows with the
    run; the gates read the time from the step count instead."""

    def test_accumulated_time_outgrows_the_tolerance(self):
        # Why the core does not pass sim_time: at 1 ms a few minutes of steps
        # are more than a millionth of a step off.
        now = 0.0
        for _ in range(400_000):
            now += 0.001
        assert abs(now - 400_000 * 0.001) > 1e-6 * 0.001

    def test_a_step_rate_callback_keeps_every_beat_late_in_a_run(self):
        import pybullet as p

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, monitor=False, timestep=0.001, target_rtf=0, monitor_log_interval=0)
        )
        try:
            sim.initialize_simulation()
            sim.restore_completed_step_boundary(200_000_000, 200_000_000 * 0.001)  # 55.6 h into the run
            dts = []
            sim.register_callback(lambda sc, dt: dts.append(dt), frequency=1000.0)
            for _ in range(2000):
                sim.step_once()
            assert len(dts) == 2000
            assert all(dt == pytest.approx(0.001) for dt in dts)
        finally:
            p.disconnect(sim.client)


class TestPostStepPhase:
    def test_runs_after_the_post_step_event(self, sim):
        order = []
        sim.events.on(SimEvents.POST_STEP, lambda **_: order.append("event"))
        sim.register_callback(lambda sc, dt: order.append("step"))
        sim.register_callback(lambda sc, dt: order.append("post_step"), phase="post_step")
        sim.step_once()
        assert order == ["step", "event", "post_step"]

    def test_is_frequency_limited_like_the_rest(self, sim):
        seen = []
        sim.register_callback(lambda sc, dt: seen.append((sc.sim_time, dt)), frequency=5.0, phase="post_step")
        for _ in range(20):
            sim.step_once()
        # Scheduled on the poses' time: runs when they reach 0.2, 0.4 ... 2.0 s.
        assert [round(t, 9) for t, _ in seen] == [round(0.2 * k, 9) for k in range(1, 11)]
        assert all(dt == pytest.approx(0.2) for _, dt in seen)

    def test_both_phases_see_the_time_of_the_state(self, sim):
        """A Phase-1 and a post-step callback in the same step see the same
        sim_time: the end of the step, the time of the poses."""
        seen = []
        sim.register_callback(lambda sc, dt: seen.append(("post", round(sc.sim_time, 9))), phase="post_step")
        sim.register_callback(lambda sc, dt: seen.append(("step", round(sc.sim_time, 9))))
        sim.step_once()
        sim.step_once()
        assert seen == [("step", 0.1), ("post", 0.1), ("step", 0.2), ("post", 0.2)]
        assert sim.sim_time == pytest.approx(0.2)  # outside a step: the completed time

    def test_a_phase_one_schedule_at_the_step_rate_runs_from_the_first_step(self, sim):
        seen = []
        sim.register_callback(lambda sc, dt: seen.append(round(sc.sim_time, 9)), frequency=10.0)
        for _ in range(3):
            sim.step_once()
        assert seen == [0.1, 0.2, 0.3]

    def test_a_post_step_schedule_at_the_step_rate_runs_from_the_first_step(self, sim):
        seen = []
        sim.register_callback(lambda sc, dt: seen.append(round(sc.sim_time, 9)), frequency=10.0, phase="post_step")
        for _ in range(3):
            sim.step_once()
        assert seen == [0.1, 0.2, 0.3]

    def test_profiled_apart_from_the_post_step_event(self, sim):
        import time

        sim.register_callback(lambda sc, dt: time.sleep(0.02), phase="post_step")
        profile = sim.step_once(return_profiling=True)
        assert profile["callbacks_post_step"] >= 20.0
        assert profile["events_post_step"] < 20.0

    def test_rejects_an_unknown_phase(self, sim):
        with pytest.raises(ValueError):
            sim.register_callback(lambda sc, dt: None, phase="later")

    @pytest.mark.parametrize("phase", [CallbackPhase.POST_STEP, "post_step"])
    def test_takes_the_enum_or_its_value(self, sim, phase):
        order = []
        sim.events.on(SimEvents.POST_STEP, lambda **_: order.append("event"))
        sim.register_callback(lambda sc, dt: order.append("callback"), phase=phase)
        sim.step_once()
        assert order == ["event", "callback"]


class TestNonPositiveDt:
    """A gate asked twice at one time, or with the clock moved back, is a caller error."""

    def test_repeated_time_warns_once_and_reports_one_timestep(self, caplog):
        gate = FrequencyGate(None)
        gate.due(0.1, 0.1)
        with caplog.at_level("WARNING", logger="pybullet_fleet._frequency"):
            assert gate.due(0.1, 0.1) == pytest.approx(0.1)  # asked twice at one time
            assert gate.due(0.05, 0.1) == pytest.approx(0.1)  # clock moved back without reset()
            gate.reset()
            gate.due(0.0, 0.1)
            assert gate.due(0.0, 0.1) == pytest.approx(0.1)
        warnings = [r for r in caplog.records if "non-positive dt" in r.getMessage()]
        assert len(warnings) == 1  # once per gate, reset() included

    def test_a_frequency_gate_is_not_due_again_at_the_same_time(self, caplog):
        gate = FrequencyGate(10.0)
        assert gate.due(0.1, 0.1) is not None
        with caplog.at_level("WARNING", logger="pybullet_fleet._frequency"):
            assert gate.due(0.1, 0.1) is None
        assert not caplog.records

    def test_a_normal_run_does_not_warn(self, caplog):
        with caplog.at_level("WARNING", logger="pybullet_fleet._frequency"):
            _fires(FrequencyGate(3.0), 0.1, 100)
            _fires(FrequencyGate(None), 0.1, 100)
        assert not [r for r in caplog.records if "non-positive dt" in r.getMessage()]


class TestSetFrequency:
    """An owner keeps one gate and changes its rate through set_frequency()."""

    def test_the_same_rate_keeps_the_schedule(self):
        gate = FrequencyGate(2.0)
        assert gate.due(0.5, 0.1) is not None  # first due time, 0.5 s
        gate.set_frequency(2.0)
        # Not restarted: the next run reports the time since the last one (0.7 s),
        # not the one-interval dt of a fresh schedule's first run.
        assert gate.due(1.2, 0.1) == pytest.approx(0.7)

    def test_a_new_rate_starts_a_fresh_schedule(self):
        gate = FrequencyGate(2.0)
        assert gate.due(0.5, 0.1) is not None
        gate.set_frequency(10.0)
        assert gate.frequency == 10.0
        # Fresh: the first due time at or after now runs at once, reporting one interval.
        assert gate.due(0.6, 0.1) == pytest.approx(0.1)

    @pytest.mark.parametrize("frequency", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_a_bad_rate_and_keeps_the_old_one(self, frequency):
        gate = FrequencyGate(2.0)
        with pytest.raises(ValueError):
            gate.set_frequency(frequency)
        assert gate.frequency == 2.0

    def test_frequency_is_read_only(self):
        gate = FrequencyGate(2.0)
        with pytest.raises(AttributeError):
            gate.frequency = 3.0  # type: ignore[misc]


class TestPluginFrequency:
    def test_setting_plugin_frequency_reschedules_its_gate(self, sim):
        class Counter(SimPlugin):
            def __init__(self, sim_core):
                super().__init__(sim_core)
                self.dts = []

            def on_step(self, dt):
                self.dts.append(dt)

        plugin = sim.register_plugin(Counter(sim), frequency=None)
        gate = plugin._frequency_gate
        sim.step_once()
        assert plugin.dts == [pytest.approx(0.1)]  # every step: one timestep since the last run
        plugin.frequency = 2.0
        assert plugin._frequency_gate is gate and gate.frequency == 2.0
        plugin.dts.clear()
        for _ in range(10):
            sim.step_once()
        assert plugin.dts == [pytest.approx(0.5), pytest.approx(0.5)]
        with pytest.raises(ValueError):
            plugin.frequency = -1.0
        assert plugin.frequency == 2.0

    def test_an_invalid_frequency_does_not_register_the_plugin(self, sim):
        with pytest.raises(ValueError):
            sim.register_plugin(SimPlugin(sim), frequency=-1.0)
        assert sim.plugins == []
