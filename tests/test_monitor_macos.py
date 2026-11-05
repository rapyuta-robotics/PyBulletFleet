"""Check macOS monitor IPC without opening native windows."""

from dataclasses import asdict
from multiprocessing import Pipe
from unittest.mock import Mock

import pybullet as p
import pytest

from pybullet_fleet import core_simulation, data_monitor
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.gui_commands import GuiCommand, GuiCommandType

# conftest disables start() for the rest of the suite.
_start_monitor = data_monitor.DataMonitor.start


def _set_platform(monkeypatch, platform):
    for module in (data_monitor, core_simulation):
        monkeypatch.setattr(module, "IS_MACOS", platform == "darwin")


def test_mac_monitor_uses_process_without_creating_tk(monkeypatch):
    _set_platform(monkeypatch, "darwin")
    monitor = data_monitor.DataMonitor(enable_gui=False)
    monitor.enable_gui = True
    monitor._start_process = Mock()
    monitor._run_monitor = Mock()
    _start_monitor(monitor)
    monitor._start_process.assert_called_once()
    monitor._run_monitor.assert_not_called()
    assert monitor.monitor_thread is None


def test_monitor_subprocess_can_resume_paused_simulation(monkeypatch):
    _set_platform(monkeypatch, "darwin")
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    monitor = data_monitor.DataMonitor(enable_gui=False)
    parent, child = Pipe()
    monitor._command_pipe = parent
    monitor.running = True
    monitor.set_command_sink(sim.enqueue_gui_command)
    try:
        sim.initialize_simulation()
        sim._data_monitor = monitor
        sim.pause()
        child.send(asdict(GuiCommand(GuiCommandType.RESUME)))
        sim.step_once()
        assert not sim.is_paused
        assert sim.step_count == 1
        child.send(asdict(GuiCommand(GuiCommandType.SET_SIMULATION_PACING, target_rtf=2.0, timestep=0.05)))
        sim.step_once()
        assert sim.params.target_rtf == 2.0
        assert sim.params.timestep == 0.05
        child.close()
        sim.step_once()
        assert not monitor.running
        assert monitor._command_pipe is None
    finally:
        monitor.stop()
        child.close()
        p.disconnect(sim.client)


def test_monitor_stop_closes_pipe_and_reaps_child(monkeypatch):
    _set_platform(monkeypatch, "darwin")
    monitor = data_monitor.DataMonitor(enable_gui=False)
    parent, child = Pipe()
    monitor._command_pipe = parent
    process = Mock()
    monitor._monitor_process = process
    try:
        monitor.stop()
        assert parent.closed
        process.wait.assert_called_once_with(timeout=3)
        assert monitor._monitor_process is None
        monitor.stop()
        process.wait.assert_called_once()
    finally:
        child.close()


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_non_mac_monitor_keeps_threaded_startup(monkeypatch, platform):
    _set_platform(monkeypatch, platform)
    monitor = data_monitor.DataMonitor(enable_gui=False)
    monitor.enable_gui = True
    monitor._start_process = Mock()
    thread_factory = Mock()
    # The thread would report its window attempt; start() waits for that.
    thread_factory.return_value.start.side_effect = lambda: monitor._window_attempted.set()
    monkeypatch.setattr(data_monitor.threading, "Thread", thread_factory)

    _start_monitor(monitor)

    thread_factory.assert_called_once_with(target=monitor._run_monitor, daemon=True)
    thread_factory.return_value.start.assert_called_once()
    monitor._start_process.assert_not_called()


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_simulation_only_polls_and_stops_monitor_on_mac(monkeypatch, platform):
    _set_platform(monkeypatch, platform)
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False, target_rtf=0, timestep=0.1))
    monitor = Mock()
    sim._data_monitor = monitor
    try:
        sim.run_simulation(duration=0.1)
        assert sim.step_count == 1
        if platform == "darwin":
            monitor.process_events.assert_called_once()
            monitor.stop.assert_called_once()
        else:
            monitor.process_events.assert_not_called()
            monitor.stop.assert_not_called()
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_standalone_monitor_changes_only_on_mac(monkeypatch, platform):
    _set_platform(monkeypatch, platform)
    monitor = Mock()
    monitor.running = True
    monkeypatch.setattr(data_monitor, "DataMonitor", Mock(return_value=monitor))
    sleep = Mock(side_effect=lambda _: setattr(monitor, "running", False))
    monkeypatch.setattr(data_monitor.time, "sleep", sleep)

    data_monitor.main()

    if platform == "darwin":
        monitor._run_monitor.assert_called_once()
        monitor.start.assert_not_called()
        sleep.assert_not_called()
    else:
        monitor.start.assert_called_once()
        monitor._run_monitor.assert_not_called()
        sleep.assert_called_once_with(1)
