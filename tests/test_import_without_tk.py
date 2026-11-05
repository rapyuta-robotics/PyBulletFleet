"""pybullet_fleet imports and runs headless where Tk cannot be loaded.

A slim container image has the ``tkinter`` package but not the Tk shared
library, and then ``import tkinter`` raises a plain ``ImportError`` (for
example ``libtk8.6.so: cannot open shared object file``), not
``ModuleNotFoundError``. Each case runs in a fresh interpreter, because the
package and ``tkinter`` are already imported in this one.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make `import tkinter` fail the given way, before anything else imports it.
_BLOCK_TK = """
import sys

class _NoTk:
    def find_spec(self, name, path=None, target=None):
        if name in ("tkinter", "_tkinter") or name.startswith("tkinter."):
            raise {error}("libtk8.6.so: cannot open shared object file: No such file or directory")
        return None

sys.meta_path.insert(0, _NoTk())

# Count what pybullet_fleet.data_monitor logs, whatever the format or stream.
import logging

class _Count(logging.Handler):
    records = 0

    def emit(self, record):
        _Count.records += 1

logging.getLogger("pybullet_fleet.data_monitor").addHandler(_Count())
"""


def _run(error: str, body: str, timeout: float = 120) -> subprocess.CompletedProcess:
    script = _BLOCK_TK.format(error=error) + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=timeout)


@pytest.mark.parametrize("error", ["ImportError", "ModuleNotFoundError"])
def test_the_package_imports_and_steps_with_the_monitor_off(error):
    result = _run(
        error,
        """
        import pybullet_fleet
        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, log_level="warning"))
        sim.step_once()
        print("data_monitor records:", _Count.records)
        print("ok")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.splitlines()  # PyBullet prints its own lines too
    # Nothing to say when the window was never asked for.
    assert "data_monitor records: 0" in result.stdout.splitlines()


@pytest.mark.parametrize("error", ["ImportError", "ModuleNotFoundError"])
def test_asking_for_the_window_falls_back_to_the_file_and_says_why(error):
    result = _run(
        error,
        """
        from pybullet_fleet.data_monitor import DataMonitor

        monitor = DataMonitor(enable_gui=True)
        assert monitor.enable_gui is False
        print("ok")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.splitlines()  # PyBullet prints its own lines too
    assert "tkinter is unavailable" in result.stderr
    assert f"{error}: libtk8.6.so: cannot open shared object file" in result.stderr  # the reason
    assert "python3-tk" in result.stderr  # how to fix it


@pytest.mark.parametrize("error", ["ImportError", "ModuleNotFoundError"])
def test_a_simulation_asking_for_the_window_refreshes_at_the_headless_rate(error):
    # Through the simulation (setup_monitor() and start()), and with no window
    # to read the file, the monitor refreshes as it does headless.
    result = _run(
        error,
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
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.splitlines()


@pytest.mark.parametrize("error", ["ImportError", "ModuleNotFoundError"])
def test_the_standalone_monitor_exits_without_tk(error):
    result = _run(
        error,
        """
        from pybullet_fleet.data_monitor import main

        main()
        """,
        timeout=30,  # it used to wait forever for a window that never opened
    )
    assert result.returncode != 0
    assert "tkinter is unavailable" in result.stderr
