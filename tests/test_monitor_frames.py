"""MonitorFrames on its own: the entity-list cache and the "window is behind" state."""

from types import SimpleNamespace

from pybullet_fleet._monitor_frames import MonitorFrames
from pybullet_fleet.gui_commands import MonitorFrame


class _Obj(SimpleNamespace):
    pass


class _Box(_Obj):
    pass


def _objects(*names):
    return [_Box(object_id=i, name=name) for i, name in enumerate(names)]


def _frame(steps):
    return MonitorFrame(
        sim_time=0.0,
        real_time=0.0,
        target_rtf=1.0,
        actual_rtf=1.0,
        timestep=0.1,
        physics_enabled=False,
        agents=0,
        objects=0,
        active_collisions=0,
        collisions=0,
        steps=steps,
        paused=False,
        entities=(),
        selected_entity=None,
        follow_enabled=False,
    )


class TestTheEntityList:
    def test_is_sorted_by_id_and_names_the_unnamed(self):
        objects = [_Box(object_id=2, name="b"), _Box(object_id=1, name=None)]
        entities = MonitorFrames().entities(objects)
        assert [(e.object_id, e.name, e.kind) for e in entities] == [(1, "entity-1", "_Box"), (2, "b", "_Box")]

    def test_is_reused_until_an_object_changes(self):
        frames = MonitorFrames()
        objects = _objects("a", "b")
        first = frames.entities(objects)
        assert frames.entities(objects) is first
        objects.append(_Box(object_id=2, name="c"))
        assert frames.entities(objects) is first  # not told yet
        frames.objects_changed()
        rebuilt = frames.entities(objects)
        assert rebuilt is not first and len(rebuilt) == 3

    def test_a_view_change_keeps_it(self):
        frames = MonitorFrames()
        objects = _objects("a")
        first = frames.entities(objects)
        frames.view_changed()
        assert frames.entities(objects) is first


class TestTheWindowIsBehind:
    def test_until_the_first_frame(self):
        frames = MonitorFrames()
        assert frames.stale and frames.due_while_paused() and frames.behind(0)

    def test_not_once_a_frame_shows_the_current_step(self):
        frames = MonitorFrames()
        frames.built(_frame(steps=5))
        assert not frames.stale and not frames.due_while_paused()
        assert not frames.behind(5)
        assert frames.last_frame.steps == 5

    def test_when_the_simulation_stepped_past_the_last_frame(self):
        frames = MonitorFrames()
        frames.built(_frame(steps=5))
        assert frames.behind(6)
        assert not frames.due_while_paused()  # running, the next refresh shows it

    def test_on_a_view_or_object_change(self):
        for change in ("view_changed", "objects_changed"):
            frames = MonitorFrames()
            frames.built(_frame(steps=1))
            getattr(frames, change)()
            assert frames.due_while_paused() and frames.behind(1), change

    def test_reset_forgets_the_last_frame(self):
        frames = MonitorFrames()
        frames.built(_frame(steps=3))
        frames.reset()
        assert frames.last_frame is None and frames.stale


class TestTheDefaultRefreshIsConfigurable:
    """monitor_update_interval=None takes the window/headless defaults, which can be overridden."""

    def _writes_over_4_seconds(self, tmp_path, monkeypatch, **params):
        import pybullet as p

        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams
        from pybullet_fleet.data_monitor import DataMonitor

        monkeypatch.setattr(DataMonitor, "start", lambda self: None)
        core = MultiRobotSimulationCore(
            SimulationParams(
                gui=False,
                monitor=True,
                enable_floor=False,
                timestep=0.1,
                target_rtf=0,
                monitor_update_interval=None,
                monitor_log_interval=0,
                log_level="warning",
                **params,
            )
        )
        core._data_monitor.data_file = str(tmp_path / "monitor.json")
        core.initialize_simulation()
        count = [0]
        original = core._data_monitor._write_text

        def counted(*parts):
            count[0] += 1
            original(*parts)

        monkeypatch.setattr(core._data_monitor, "_write_text", counted)
        try:
            for _ in range(40):  # 4 simulated seconds at dt=0.1
                core.step_once()
        finally:
            p.disconnect(core.client)
        return count[0]

    def test_with_the_window(self, tmp_path, monkeypatch):
        import pybullet_fleet.core_simulation as core_simulation

        monkeypatch.setattr(core_simulation, "MONITOR_REFRESH_WITH_WINDOW", 0.5)
        n = self._writes_over_4_seconds(tmp_path, monkeypatch, enable_monitor_gui=True)
        assert 7 <= n <= 9

    def test_without_it(self, tmp_path, monkeypatch):
        import pybullet_fleet.core_simulation as core_simulation

        monkeypatch.setattr(core_simulation, "MONITOR_REFRESH_HEADLESS", 2.0)
        n = self._writes_over_4_seconds(tmp_path, monkeypatch, enable_monitor_gui=False)
        assert 1 <= n <= 3

    def test_they_are_not_parameters(self):
        from pybullet_fleet import SimulationParams

        assert not hasattr(SimulationParams(), "monitor_update_interval_window")
        assert not hasattr(SimulationParams(), "monitor_update_interval_headless")
