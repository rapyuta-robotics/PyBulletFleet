"""The monitor falls back to its file where Tk is installed but no display is.

``import tkinter`` works on such a host (a CI runner, an SSH session), but
``tk.Tk()`` raises ``TclError``. Each case runs in a fresh interpreter with
``DISPLAY`` removed, as the suite's conftest turns ``DataMonitor.start`` off.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import tkinter  # noqa: F401

    _HAS_TK = True
except ImportError:
    _HAS_TK = False

pytestmark = [
    pytest.mark.skipif(not _HAS_TK, reason="needs tkinter"),
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="X11 display check"),
]


def _run(body: str, timeout: float = 60) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, env=env, timeout=timeout
    )


def test_starting_the_window_without_a_display_falls_back_to_the_file():
    result = _run(
        """
        from pybullet_fleet.data_monitor import DataMonitor

        monitor = DataMonitor(enable_gui=True)
        monitor.start()
        if monitor.monitor_thread is not None:
            monitor.monitor_thread.join(timeout=10)
        assert monitor.enable_gui is False, "enable_gui"
        assert monitor.running is False, "running"
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.splitlines()
    assert "Traceback" not in result.stderr
    assert "could not open the monitor window" in result.stderr
    assert "no display name" in result.stderr  # the reason
    assert "file-only mode" in result.stderr


def test_a_simulation_asking_for_the_window_without_a_display_refreshes_at_the_headless_rate():
    result = _run(
        """
        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams
        from pybullet_fleet._defaults import MONITOR_REFRESH_HEADLESS

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, monitor=True, enable_monitor_gui=True, log_level="warning")
        )
        assert sim._data_monitor.enable_gui is False
        assert sim._monitor_update_gate.frequency == 1.0 / MONITOR_REFRESH_HEADLESS, sim._monitor_update_gate.frequency
        sim.step_once()
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.splitlines()
    assert "Traceback" not in result.stderr


def test_the_standalone_monitor_exits_without_a_display():
    result = _run(
        """
        from pybullet_fleet.data_monitor import main

        main()
        """,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "could not open the monitor window" in result.stderr
