"""`monitor_log_interval`: the monitor's figures in the log, without the monitor.

The DataMonitor exists to feed a window, and its own ``[MONITOR]`` line is
logged at DEBUG, so a run had no record of its real-time factor or collision
count. ``monitor_log_interval`` logs that line at INFO on a wall-clock interval
whatever ``monitor`` and ``gui`` are set to -- by default as often as the
monitor itself refreshes without a GUI.
"""

import logging

import pytest

from pybullet_fleet import MultiRobotSimulationCore, SimulationParams


def _make(**overrides):
    params = dict(gui=False, monitor=False, enable_monitor_gui=False, target_rtf=0, log_level="info")
    params.update(overrides)
    sim = MultiRobotSimulationCore(SimulationParams(**params))
    sim.initialize_simulation()
    return sim


@pytest.fixture
def make_sim():
    import pybullet as p

    made = []

    def make(**overrides):
        sim = _make(**overrides)
        made.append(sim)
        return sim

    yield make
    for sim in made:
        try:
            p.disconnect(sim.client)
        except p.error:
            pass


def _monitor_lines(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO and record.getMessage().startswith("[MONITOR]")
    ]


def test_logs_without_the_monitor(make_sim, caplog):
    sim = make_sim(monitor_log_interval=60.0)
    with caplog.at_level(logging.INFO):
        for _ in range(5):
            sim.step_once()
    lines = _monitor_lines(caplog)
    # The first step logs; the next four fall inside the same 60 s interval.
    assert len(lines) == 1
    for field in ("rtf=", "target_rtf=0", "total_collisions=0", "active_collisions=0", "steps="):
        assert field in lines[0]


def test_logs_once_per_interval(make_sim, caplog, monkeypatch):
    import pybullet_fleet.core_simulation as core

    clock = {"now": 1000.0}
    monkeypatch.setattr(core.time, "monotonic", lambda: clock["now"])
    sim = make_sim(monitor_log_interval=1.0)
    with caplog.at_level(logging.INFO):
        for _ in range(10):  # 0.25 s apart: 2.25 s in all
            sim.step_once()
            clock["now"] += 0.25
    # At 0.0, 1.0 and 2.0 s.
    assert len(_monitor_lines(caplog)) == 3


def test_on_by_default_once_a_second(make_sim, caplog, monkeypatch):
    import pybullet_fleet.core_simulation as core

    assert SimulationParams().monitor_log_interval == 1.0
    clock = {"now": 1000.0}
    monkeypatch.setattr(core.time, "monotonic", lambda: clock["now"])
    sim = make_sim()
    with caplog.at_level(logging.INFO):
        for _ in range(10):  # 0.25 s apart: 2.25 s in all
            sim.step_once()
            clock["now"] += 0.25
    assert len(_monitor_lines(caplog)) == 3


def test_a_fresh_run_logs_its_first_step(make_sim, caplog):
    """The interval does not carry over into the next run: a short run would
    otherwise end without a single line."""
    sim = make_sim(monitor_log_interval=60.0)
    with caplog.at_level(logging.INFO):
        sim.step_once()
        sim.initialize_simulation()
        sim.step_once()
    assert len(_monitor_lines(caplog)) == 2


def test_a_sparse_interval_still_reports_the_rtf(make_sim, caplog, monkeypatch):
    """With the monitor off, the log is the only thing sampling the speed. An
    interval longer than the monitor's 10 s window must still measure across
    it rather than report rtf=0.00."""
    import pybullet_fleet.core_simulation as core

    clock = {"now": 1000.0}
    monkeypatch.setattr(core.time, "monotonic", lambda: clock["now"])
    sim = make_sim(monitor_log_interval=30.0, timestep=0.1)
    with caplog.at_level(logging.INFO):
        for _ in range(11):  # 3 s apart: 30 real seconds per simulated second
            sim.step_once()
            clock["now"] += 3.0
    lines = _monitor_lines(caplog)
    assert len(lines) == 2
    assert "rtf=0.03," in lines[1], lines[1]


def _monitor_updates(make_sim, monkeypatch, steps=40, **params):
    calls = []
    sim = make_sim(monitor=True, monitor_log_interval=0, timestep=0.1, **params)
    monkeypatch.setattr(sim, "update_monitor", lambda: calls.append(sim.sim_time))
    for _ in range(steps):
        sim.step_once()
    return len(calls)


def test_the_monitor_refresh_is_set_separately(make_sim, monkeypatch):
    """monitor_update_interval paces the DataMonitor; monitor_log_interval
    paces the log. Neither moves the other."""
    headless_default = _monitor_updates(make_sim, monkeypatch)
    twice_as_often = _monitor_updates(make_sim, monkeypatch, monitor_update_interval=0.5)
    every_step = _monitor_updates(make_sim, monkeypatch, monitor_update_interval=0)
    assert SimulationParams().monitor_update_interval is None
    # 40 steps of 0.1 s end at 0.1 .. 4.0 s: once a second headless by
    # default (1, 2, 3, 4 s), every 0.5 s at 0.5, and on every step at 0.
    assert headless_default == 4
    assert twice_as_often == 8
    assert every_step == 40


def test_zero_turns_it_off(make_sim, caplog):
    sim = make_sim(monitor_log_interval=0)
    with caplog.at_level(logging.INFO):
        for _ in range(5):
            sim.step_once()
    assert _monitor_lines(caplog) == []


def test_logging_does_not_refresh_the_monitor(make_sim, caplog, monkeypatch):
    """A DataMonitor frame is written by update_monitor(); the log must not
    go through it, or monitor_update_interval would follow the log's."""
    calls = []
    sim = make_sim(monitor=True, monitor_update_interval=1000.0, monitor_log_interval=0.001, timestep=0.1)
    monkeypatch.setattr(sim, "update_monitor", lambda: calls.append(sim.sim_time))
    with caplog.at_level(logging.INFO):
        for _ in range(20):
            sim.step_once()
    assert calls == []
    assert len(_monitor_lines(caplog)) >= 1


def test_nothing_is_computed_when_info_is_off(make_sim):
    sim = make_sim(log_level="warning", monitor_log_interval=0.001)
    for _ in range(5):
        sim.step_once()
    assert len(sim._speed_history) == 0


def test_record_forced_headless_keeps_the_headless_monitor_rate(monkeypatch, tmp_path):
    """setup_pybullet() turns the GUI off under RECORD; the monitor's default
    rate must follow the final mode (once a second), not the requested GUI's
    every step."""
    import pybullet as p

    monkeypatch.setenv("RECORD", str(tmp_path / "unused.gif"))
    monkeypatch.delenv("RECORD_GUI", raising=False)
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=True, monitor=True, enable_monitor_gui=False, target_rtf=0, timestep=0.1, monitor_log_interval=0)
    )
    try:
        assert sim.params.gui is False
        sim.initialize_simulation()
        calls = []
        monkeypatch.setattr(sim, "update_monitor", lambda: calls.append(sim.sim_time))
        for _ in range(40):
            sim.step_once()
        assert len(calls) == 4  # at 1, 2, 3 and 4 s
    finally:
        p.disconnect(sim.client)


def test_the_default_refresh_follows_the_monitor_window(make_sim, monkeypatch):
    """The default refresh follows the window (enable_monitor_gui, or gui when
    that is None), not gui alone: 0.25 s with it, 1 s without."""
    assert _monitor_updates(make_sim, monkeypatch, enable_monitor_gui=True) == 16
    assert _monitor_updates(make_sim, monkeypatch, enable_monitor_gui=None) == 4
