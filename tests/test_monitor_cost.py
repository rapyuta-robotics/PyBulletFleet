"""The DataMonitor stays cheap in a large scene.

The entity list is rebuilt only when objects come, go or are renamed, and its
JSON is serialized once per change. A frame is built and written on each
refresh -- every ``monitor_update_interval`` simulated seconds, by default
0.25 s with the monitor window and 1 s without -- and, while paused, at once
for a change the window shows or a single step. The JSON a reader sees is what
it always was.
"""

import json
import os
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
def sim(tmp_path, monkeypatch):
    def make(update_interval=0.0, window=False):
        # The window flag only decides the default interval here; the Tk
        # window itself is never started.
        monkeypatch.setattr(DataMonitor, "start", lambda self: None)
        core = MultiRobotSimulationCore(
            SimulationParams(
                gui=False,
                monitor=True,
                enable_monitor_gui=window,
                enable_floor=False,
                timestep=0.1,
                target_rtf=0,
                monitor_update_interval=update_interval,
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

        def counted(*parts):
            self.count += 1
            original(*parts)

        monkeypatch.setattr(monitor, "_write_text", counted)


class TestTheJson:
    def test_is_what_it_was(self, sim):
        core = sim()
        a, b = _box(core, 0.0, "shelf"), _box(core, 1.0)
        core._set_selected_entity(a.object_id, "test")
        core.step_once()
        core.update_monitor()
        assert _read(core) == _legacy_json(core.last_monitor_frame)
        names = [e["name"] for e in _read(core)["entities"]]
        assert names == ["shelf", "entity-{}".format(b.object_id)]

    def test_an_empty_entity_list(self, tmp_path):
        monitor = DataMonitor(enable_gui=False)
        monitor.data_file = str(tmp_path / "m.json")
        frame = MonitorFrame(0.0, 0.0, 1.0, 1.0, 0.1, False, 0, 0, 0, 0, 0, False)
        monitor.write_frame(frame)
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
        core = sim()
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


class TestTheRefreshInterval:
    """``monitor_update_interval``: every refresh builds and writes one frame."""

    @pytest.mark.parametrize(
        "window, expected",
        [(True, 16), (False, 4)],
        ids=["with_the_window_every_quarter_second", "without_it_every_second"],
    )
    def test_the_default(self, sim, monkeypatch, window, expected):
        core = sim(update_interval=None, window=window)
        writes = _Writes(core._data_monitor, monkeypatch)
        for _ in range(40):  # 4 simulated seconds at dt=0.1
            core.step_once()
        assert expected - 1 <= writes.count <= expected + 1

    def test_zero_is_every_step(self, sim, monkeypatch):
        core = sim(update_interval=0.0)
        writes = _Writes(core._data_monitor, monkeypatch)
        for _ in range(10):
            core.step_once()
        assert writes.count == 10

    def test_every_frame_built_is_written(self, sim, monkeypatch):
        core = sim(update_interval=0.5)
        writes = _Writes(core._data_monitor, monkeypatch)
        built = [0]
        original = core.update_monitor

        def counted():
            built[0] += 1
            original()

        monkeypatch.setattr(core, "update_monitor", counted)
        for _ in range(30):
            core.step_once()
        assert built[0] == writes.count > 0


class TestTheWindowIsNeverLeftBehind:
    """Paused, a change the window shows or a single step is written at once; the end of a run is written."""

    def test_the_final_frame_of_a_run(self, sim):
        core = sim(update_interval=10.0)
        core.run_simulation(duration=1.0)
        assert _read(core)["steps"] == core.step_count

    def test_a_pause(self, sim):
        core = sim(update_interval=10.0)
        core.step_once()
        core.pause()
        core.step_once()
        assert _read(core)["paused"] is True

    def test_a_single_step_while_paused(self, sim):
        core = sim(update_interval=10.0)
        core.step_once()
        core.pause()
        core.step_once()
        assert core.request_single_step()
        core.step_once()
        assert _read(core)["steps"] == core.step_count

    def test_a_selection_while_paused(self, sim):
        core = sim(update_interval=10.0)
        box = _box(core, 0.0, "picked")
        core.step_once()
        core.pause()
        core.step_once()
        core._set_selected_entity(box.object_id, "test")
        core.step_once()  # paused, but the window shows the selection
        assert _read(core)["selected_entity"]["entity"]["name"] == "picked"

    def test_the_end_of_a_step_once_loop(self, sim):
        core = sim(update_interval=10.0)
        for _ in range(10):
            core.step_once()
        assert not os.path.exists(core._data_monitor.data_file)  # the first refresh is 10 s away
        assert core.flush_monitor()
        assert _read(core)["steps"] == 10
        assert not core.flush_monitor()  # nothing left to write

    def test_close_flushes(self, sim):
        core = sim(update_interval=10.0)
        for _ in range(10):
            core.step_once()
        data_file = core._data_monitor.data_file
        core.close()
        with open(data_file) as f:
            assert json.load(f)["steps"] == 10

    def test_close_after_an_external_disconnect_does_not_raise(self, sim):
        core = sim(update_interval=10.0)
        box = _box(core, 0.0, "picked")
        core.step_once()
        core._set_selected_entity(box.object_id, "test")  # the flush needs PyBullet for this
        p.disconnect(core.client)
        core.close()


class TestOtherWaysTheSetOfObjectsChanges:
    def test_reset_empties_the_entity_list(self, sim):
        core = sim()
        _box(core, 0.0, "a")
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["a"]
        core.reset()
        core.update_monitor()
        assert _read(core)["entities"] == []

    def test_set_name_and_a_loader_rename_go_through_the_setter(self, sim):
        core = sim()
        box = _box(core, 0.0, "a")
        core.step_once()
        box.set_name("b")
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["b"]
        box.name = "{}/{}".format("model", box.name)  # what sdf_loader does
        core.step_once()
        assert [e["name"] for e in _read(core)["entities"]] == ["model/b"]
