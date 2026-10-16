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


def test_on_by_default_at_the_monitor_interval(make_sim, caplog, monkeypatch):
    import pybullet_fleet.core_simulation as core

    assert SimulationParams().monitor_log_interval is None
    clock = {"now": 1000.0}
    monkeypatch.setattr(core.time, "monotonic", lambda: clock["now"])
    sim = make_sim()
    with caplog.at_level(logging.INFO):
        for _ in range(10):  # 0.25 * MONITOR_UPDATE_INTERVAL apart
            sim.step_once()
            clock["now"] += 0.25 * core.MONITOR_UPDATE_INTERVAL
    assert len(_monitor_lines(caplog)) == 3


def test_zero_turns_it_off(make_sim, caplog):
    sim = make_sim(monitor_log_interval=0)
    with caplog.at_level(logging.INFO):
        for _ in range(5):
            sim.step_once()
    assert _monitor_lines(caplog) == []
