"""DataMonitor.stop() closes the window from the Tk thread and returns.

Tk is not thread-safe: a Tk call made from another thread is handed to the
Tk thread and waits for it, which hangs once its main loop has ended. stop()
therefore only asks the Tk thread to quit; the Tk thread polls for that with
``after()`` and closes the window itself.

The first tests run the real stop logic against a fake Tk root, so they need
no display. The last one opens a real window and skips without a display.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from collections import deque

import pytest

from pybullet_fleet import data_monitor
from pybullet_fleet.data_monitor import DataMonitor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeRoot:
    """A Tk root that runs ``after()`` callbacks in the thread that made it.

    Every call records the thread it came from, so a test can check that no
    call came from outside the Tk thread. ``ignore_quit`` makes the main loop
    keep running, as a Tk thread that does not react would.
    """

    def __init__(self, ignore_quit: bool = False) -> None:
        self.owner = threading.get_ident()
        self.calls: list = []
        self._pending: deque = deque()
        self._quit = threading.Event()
        self.ignore_quit = ignore_quit
        self.destroyed = False

    def _record(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))

    def after(self, _ms: int, callback) -> None:
        self._record("after")
        self._pending.append(callback)

    def quit(self) -> None:
        self._record("quit")
        if not self.ignore_quit:
            self._quit.set()

    def destroy(self) -> None:
        self._record("destroy")
        self.destroyed = True

    def mainloop(self) -> None:
        self._record("mainloop")
        while not self._quit.is_set():
            if self._pending:
                self._pending.popleft()()
            time.sleep(0.005)

    def foreign_calls(self) -> list:
        return [name for name, ident in self.calls if ident != self.owner]


def _serve_fake_window(monitor: DataMonitor, **root_kwargs) -> FakeRoot:
    """Run ``_serve_window()`` on a fake root in a monitor thread, as start() would."""
    created = threading.Event()
    holder: dict = {}

    def run() -> None:
        holder["root"] = monitor.window = FakeRoot(**root_kwargs)
        created.set()
        monitor._serve_window()

    monitor.running = True
    monitor.monitor_thread = threading.Thread(target=run, daemon=True)
    monitor.monitor_thread.start()
    assert created.wait(5)
    return holder["root"]


def test_stop_from_another_thread_closes_the_window_in_the_tk_thread():
    monitor = DataMonitor(enable_gui=False)
    root = _serve_fake_window(monitor)

    monitor.stop()

    assert not monitor.monitor_thread.is_alive()
    assert root.foreign_calls() == []
    assert "quit" in [name for name, _ in root.calls]
    assert root.destroyed
    assert monitor.running is False


def test_stop_logs_and_returns_when_the_tk_thread_does_not_end(monkeypatch, caplog):
    monkeypatch.setattr(data_monitor, "_STOP_JOIN_TIMEOUT", 0.3)
    monitor = DataMonitor(enable_gui=False)
    root = _serve_fake_window(monitor, ignore_quit=True)

    started = time.monotonic()
    with caplog.at_level("WARNING"):
        monitor.stop()

    assert time.monotonic() - started < 3
    assert monitor.monitor_thread.is_alive()
    assert "did not end" in caplog.text
    assert root.foreign_calls() == []
    root._quit.set()  # let the fake thread end
    monitor.monitor_thread.join(5)


def test_stop_on_the_tk_thread_quits_the_loop_directly():
    """The macOS child process calls stop() from its own Tk callbacks."""
    monitor = DataMonitor(enable_gui=False)
    monitor.running = True
    root = FakeRoot()
    monitor.window = root
    root.after(0, monitor.stop)

    monitor._serve_window()  # this thread is the Tk thread

    assert root.foreign_calls() == []
    assert "quit" in [name for name, _ in root.calls]
    assert root.destroyed
    assert monitor.running is False


def test_stop_without_a_window_returns():
    monitor = DataMonitor(enable_gui=False)  # never started
    monitor.stop()
    assert monitor.running is False


# --- a real window -----------------------------------------------------------


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _can_open_a_window() -> bool:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import tkinter; tkinter.Tk().destroy()"],
            capture_output=True,
            env=_env(),
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="threaded monitor path")
@pytest.mark.parametrize("open_for", [0.0, 1.0])
def test_stop_returns_while_a_real_window_is_open(open_for):
    if not _can_open_a_window():
        pytest.skip("needs a display that Tk can open a window on")
    body = f"""
        import faulthandler, time
        faulthandler.dump_traceback_later(20, exit=True)
        from pybullet_fleet.data_monitor import DataMonitor

        monitor = DataMonitor("stop test")
        monitor.start()
        assert monitor.enable_gui, "the window did not open"
        time.sleep({open_for})
        started = time.monotonic()
        monitor.stop()
        print("stop took", round(time.monotonic() - started, 2))
        assert not monitor.monitor_thread.is_alive(), "Tk thread still running"
        print("ok")
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, env=_env(), timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout.splitlines()
    assert "Traceback" not in result.stderr
