"""The DataMonitor stays cheap in a large scene.

The entity list is rebuilt only when objects come, go or are renamed, its JSON
is serialized once per change, and the file the monitor window reads is
written at most once per ``monitor_write_interval`` -- except for changes the
window shows (selection, follow, pause, the entity list) and the final frame,
which go out at once. The JSON a reader sees is what it always was.
"""

import json
from dataclasses import asdict

import pybullet as p
import pytest

from pybullet_fleet import data_monitor as data_monitor_module
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.data_monitor import DataMonitor
from pybullet_fleet.geometry import Pose
from pybullet_fleet.gui_commands import MonitorEntity, MonitorFrame
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionMode


@pytest.fixture
def sim(tmp_path):
    def make(write_interval=0.25, update_interval=0.0):
        core = MultiRobotSimulationCore(
            SimulationParams(
                gui=False,
                monitor=True,
                enable_monitor_gui=False,
                enable_floor=False,
                timestep=0.1,
                target_rtf=0,
                monitor_update_interval=update_interval,
                monitor_write_interval=write_interval,
                monitor_log_interval=0,
                log_level="warning",
            )
        )
        core._data_monitor.data_file = str(tmp_path / "monitor.json")
        core.initialize_simulation()
        made.append(core)
        return core

    made = []
    yield make
    for core in made:
        if p.isConnected(core.client):  # run_simulation() closes its own
            p.disconnect(core.client)


def _box(core, x, name=None):
    return SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        pose=Pose.from_xyz(x, 0.0, 0.1),
        mass=0.0,
        sim_core=core,
        collision_mode=CollisionMode.STATIC,
        name=name,
    )


def _read(core):
    with open(core._data_monitor.data_file) as f:
        return json.load(f)


def _legacy_json(frame: MonitorFrame) -> dict:
    """What write_frame wrote before the entity list was cached."""
    data = asdict(frame)
    data["time_step"] = data.pop("timestep")
    data["physics"] = "enabled" if data.pop("physics_enabled") else "disabled"
    data["frequency"] = 1 / frame.timestep
    return json.loads(json.dumps(data))


class _Writes:
    """Counts the file writes a DataMonitor makes."""

    def __init__(self, monitor, monkeypatch):
        self.count = 0
        original = monitor._write_text

        def counted(text):
            self.count += 1
            original(text)

        monkeypatch.setattr(monitor, "_write_text", counted)


class TestTheJson:
    def test_is_what_it_was(self, sim):
        core = sim()
        a, b = _box(core, 0.0, "shelf"), _box(core, 1.0)
        core._set_selected_entity(a.object_id, "test")
        core.step_once()
        core.update_monitor(force_write=True)
        assert _read(core) == _legacy_json(core.last_monitor_frame)
        names = [e["name"] for e in _read(core)["entities"]]
        assert names == ["shelf", "entity-{}".format(b.object_id)]

    def test_an_empty_entity_list(self, tmp_path):
        monitor = DataMonitor(enable_gui=False, write_interval=0)
        monitor.data_file = str(tmp_path / "m.json")
        frame = MonitorFrame(0.0, 0.0, 1.0, 1.0, 0.1, False, 0, 0, 0, 0, 0, False)
        assert monitor.write_frame(frame)
        with open(monitor.data_file) as f:
            assert json.load(f) == _legacy_json(frame)


class TestTheEntityList:
    def test_follows_objects_added_removed_and_renamed(self, sim):
        core = sim()
        a = _box(core, 0.0, "a")
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["a"]
        b = _box(core, 1.0, "b")
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["a", "b"]
        b.name = "renamed"
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["a", "renamed"]
        core.remove_object(a)
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["renamed"]

    def test_is_rebuilt_only_on_a_change(self, sim, monkeypatch):
        """Performance guard: no per-step rebuild or re-serialization of a static list."""
        core = sim(write_interval=0)
        for i in range(50):
            _box(core, float(i))
        built, dumped = [0], [0]
        original_entity = MonitorEntity.__init__
        original_dumps = data_monitor_module.json.dumps

        def counting_entity(self, *args, **kwargs):
            built[0] += 1
            original_entity(self, *args, **kwargs)

        def counting_dumps(obj, *args, **kwargs):
            if isinstance(obj, list):
                dumped[0] += 1
            return original_dumps(obj, *args, **kwargs)

        monkeypatch.setattr(MonitorEntity, "__init__", counting_entity)
        monkeypatch.setattr(data_monitor_module.json, "dumps", counting_dumps)
        for _ in range(20):
            core.step_once()
        assert built[0] == 50  # one rebuild, for the 50 boxes
        assert dumped[0] == 1  # one serialization of that list
        _box(core, 100.0)
        for _ in range(5):
            core.step_once()
        assert built[0] == 50 + 51
        assert dumped[0] == 2


class TestThrottle:
    def test_writes_at_most_once_per_interval(self, sim, monkeypatch):
        core = sim(write_interval=3600.0)
        writes = _Writes(core._data_monitor, monkeypatch)
        for _ in range(30):
            core.step_once()
        assert writes.count == 1  # the first refresh; the rest are within the hour
        assert core.last_monitor_frame.steps == 30  # refreshes still happen every step

    def test_zero_writes_every_refresh(self, sim, monkeypatch):
        core = sim(write_interval=0)
        writes = _Writes(core._data_monitor, monkeypatch)
        for _ in range(10):
            core.step_once()
        assert writes.count == 10

    def test_the_final_frame_is_written(self, sim):
        core = sim(write_interval=3600.0)
        core.run_simulation(duration=1.0)
        assert _read(core)["steps"] == core.step_count

    def test_a_change_the_window_shows_is_written_at_once(self, sim, monkeypatch):
        core = sim(write_interval=3600.0)
        box = _box(core, 0.0, "picked")
        core.step_once()
        writes = _Writes(core._data_monitor, monkeypatch)
        core.step_once()
        assert writes.count == 0
        core._set_selected_entity(box.object_id, "test")
        core.step_once()
        assert writes.count == 1
        assert _read(core)["selected_entity"]["entity"]["name"] == "picked"
        core.pause()
        core.update_monitor()
        assert writes.count == 2
        assert _read(core)["paused"] is True


class _Clock:
    """Stands in for the DataMonitor's wall clock, so a throttle window can pass in a test."""

    def __init__(self, monkeypatch):
        import time as real_time
        import types

        self.now = 1000.0
        fake = types.SimpleNamespace(
            **{name: getattr(real_time, name) for name in dir(real_time) if not name.startswith("__")}
        )
        fake.monotonic = lambda: self.now
        monkeypatch.setattr(data_monitor_module, "time", fake)

    def advance(self, seconds):
        self.now += seconds


class TestNothingIsLeftUnwritten:
    """A throttled frame reaches the file once the window passes, or on a flush."""

    def test_a_single_step_while_paused(self, sim, monkeypatch):
        clock = _Clock(monkeypatch)
        core = sim(write_interval=1.0)
        core.step_once()
        core.pause()
        core.step_once()  # paused: writes the pause at once
        assert _read(core)["paused"] is True
        assert core.request_single_step()
        core.step_once()  # the single step's frame falls inside the window
        steps = core.step_count
        clock.advance(2.0)
        core.step_once()  # still paused: nothing else will come
        assert _read(core)["steps"] == steps

    def test_a_selection_while_paused(self, sim, monkeypatch):
        _Clock(monkeypatch)
        core = sim(write_interval=1.0)
        box = _box(core, 0.0, "picked")
        core.step_once()
        core.pause()
        core.step_once()
        core._set_selected_entity(box.object_id, "test")
        core.step_once()  # paused, but the window shows the selection
        assert _read(core)["selected_entity"]["entity"]["name"] == "picked"

    def test_the_end_of_a_step_once_loop(self, sim, monkeypatch):
        _Clock(monkeypatch)
        core = sim(write_interval=1.0)
        for _ in range(10):
            core.step_once()
        assert _read(core)["steps"] == 1  # the rest fell inside the window
        assert core.flush_monitor()
        assert _read(core)["steps"] == 10
        assert not core.flush_monitor()  # nothing left to write

    def test_close_flushes(self, sim, monkeypatch):
        _Clock(monkeypatch)
        core = sim(write_interval=1.0)
        for _ in range(10):
            core.step_once()
        data_file = core._data_monitor.data_file
        core.close()
        with open(data_file) as f:
            assert json.load(f)["steps"] == 10

    def test_close_after_an_external_disconnect_does_not_raise(self, sim, monkeypatch):
        _Clock(monkeypatch)
        core = sim(write_interval=1.0)
        box = _box(core, 0.0, "picked")
        core.step_once()
        core._set_selected_entity(box.object_id, "test")  # the flush needs PyBullet for this
        p.disconnect(core.client)
        core.close()

    def test_a_running_loop_writes_the_trailing_frame_once_the_window_passes(self, sim, monkeypatch):
        """With refreshes sparser than steps, the next step after the window writes the last frame."""
        clock = _Clock(monkeypatch)
        core = sim(write_interval=1.0, update_interval=0.5)
        core.step_once()
        for _ in range(4):
            core.step_once()
        refreshed = core.last_monitor_frame.steps
        clock.advance(2.0)
        core.step_once()
        assert _read(core)["steps"] >= refreshed


class TestOtherWaysTheSetOfObjectsChanges:
    def test_reset_empties_the_entity_list(self, sim):
        core = sim(write_interval=0)
        _box(core, 0.0, "a")
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["a"]
        core.reset()
        core.update_monitor(force_write=True)
        assert _read(core)["entities"] == []

    def test_set_name_and_a_loader_rename_go_through_the_setter(self, sim):
        core = sim(write_interval=0)
        box = _box(core, 0.0, "a")
        core.step_once()
        box.set_name("b")
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["b"]
        box.name = "{}/{}".format("model", box.name)  # what sdf_loader does
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["model/b"]
