"""Tests for SimulationRecorder — headless simulation recording."""

import os

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.recorder import SimulationRecorder


def _has_imageio() -> bool:
    """Check if imageio is available for MP4 tests."""
    try:
        import imageio  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_core():
    """Headless sim_core with one robot for recording tests."""
    params = SimulationParams(gui=False, monitor=False, physics=False, duration=0, target_rtf=0)
    sim = MultiRobotSimulationCore(params)
    yield sim
    try:
        p.disconnect(sim.client)
    except Exception:
        pass


@pytest.fixture
def sim_core_with_robot(sim_core):
    """sim_core with a mobile robot spawned."""
    from pybullet_fleet.agent import Agent, AgentSpawnParams
    from pybullet_fleet.geometry import Pose

    spawn = AgentSpawnParams(
        urdf_path="robots/mobile_robot.urdf",
        initial_pose=Pose.from_xyz(0, 0, 0.3),
        use_fixed_base=False,
    )
    Agent.from_params(spawn, sim_core)
    return sim_core


@pytest.fixture
def tmp_gif(tmp_path):
    """Temporary GIF output path."""
    return str(tmp_path / "test_output.gif")


@pytest.fixture
def tmp_mp4(tmp_path):
    """Temporary MP4 output path."""
    return str(tmp_path / "test_output.mp4")


# ---------------------------------------------------------------------------
# Unit Tests — SimulationRecorder
# ---------------------------------------------------------------------------


class TestRecorderInit:
    """Test SimulationRecorder constructor defaults."""

    def test_default_params(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        assert rec.is_recording is False
        assert rec.frame_count == 0
        assert rec._width == 800
        assert rec._height == 600
        assert rec._fps == 15
        assert rec._duration is None
        assert rec._camera_mode == "auto"

    def test_custom_params(self, sim_core, tmp_gif):
        rec = SimulationRecorder(
            sim_core,
            output=tmp_gif,
            width=640,
            height=480,
            fps=10,
            duration=3.0,
            camera_mode="orbit",
            orbit_degrees=45.0,
        )
        assert rec._width == 640
        assert rec._height == 480
        assert rec._fps == 10
        assert rec._duration == 3.0
        assert rec._camera_mode == "orbit"
        assert rec._orbit_degrees == 45.0


class TestRecorderStartStop:
    """Test start/stop lifecycle."""

    def test_start_registers_callback(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        initial_count = len(sim_core._callbacks)
        rec.start()
        assert rec.is_recording is True
        assert len(sim_core._callbacks) == initial_count + 1

    def test_stop_unregisters_callback(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        initial_count = len(sim_core._callbacks)
        rec.start()
        rec.stop()
        assert rec.is_recording is False
        assert len(sim_core._callbacks) == initial_count

    def test_double_start_is_safe(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        rec.start()
        rec.start()  # should not double-register
        assert rec.is_recording is True

    def test_stop_without_start_is_safe(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        rec.stop()  # should not raise
        assert rec.is_recording is False


class TestRecorderCapture:
    """Test frame capture."""

    def test_capture_single_frame(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120)
        rec.start()
        # Manually invoke the capture callback
        rec._capture_frame(sim_core, 1.0 / 15)
        assert rec.frame_count == 1
        assert rec._frames[0].shape == (120, 160, 3)
        assert rec._frames[0].dtype == np.uint8

    def test_capture_multiple_frames(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120)
        rec.start()
        for i in range(5):
            rec._capture_frame(sim_core, 1.0 / 15)
        assert rec.frame_count == 5

    def test_capture_not_recording_is_noop(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        # Don't call start
        rec._capture_frame(sim_core, 1.0 / 15)
        assert rec.frame_count == 0

    def test_auto_stop_on_duration(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif, duration=1.0, width=160, height=120)
        rec.start()  # at sim time 0

        # Capture within duration
        rec._capture_at(0.5)
        assert rec.is_recording is True
        assert rec.frame_count == 1

        # Capture at duration boundary
        rec._capture_at(1.1)
        assert rec.is_recording is False
        # Frame should NOT be captured after duration exceeded
        assert rec.frame_count == 1


class TestRecorderSave:
    """Test GIF output."""

    def test_save_produces_gif(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120, fps=10)
        rec.start()
        for _ in range(5):
            rec._capture_frame(sim_core, 0.1)
        rec.stop()
        path = rec.save()
        assert path == tmp_gif
        assert os.path.exists(tmp_gif)
        assert os.path.getsize(tmp_gif) > 0

    def test_save_no_frames_raises(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        with pytest.raises(ValueError, match="No frames"):
            rec.save()

    def test_save_creates_parent_dirs(self, tmp_path, sim_core):
        nested_path = str(tmp_path / "sub" / "dir" / "output.gif")
        rec = SimulationRecorder(sim_core, output=nested_path, width=80, height=60)
        rec.start()
        rec._capture_frame(sim_core, 0.1)
        rec.stop()
        path = rec.save()
        assert os.path.exists(path)


class TestRecorderMP4:
    """Test MP4 output format."""

    @pytest.mark.skipif(
        not _has_imageio(),
        reason="imageio not installed",
    )
    def test_save_produces_mp4(self, sim_core, tmp_mp4):
        rec = SimulationRecorder(sim_core, output=tmp_mp4, width=160, height=120, fps=10)
        rec.start()
        for _ in range(5):
            rec._capture_frame(sim_core, 0.1)
        rec.stop()
        path = rec.save()
        assert path == tmp_mp4
        assert os.path.exists(tmp_mp4)
        assert os.path.getsize(tmp_mp4) > 0

    @pytest.mark.skipif(
        not _has_imageio(),
        reason="imageio not installed",
    )
    def test_mp4_creates_parent_dirs(self, tmp_path, sim_core):
        nested_path = str(tmp_path / "sub" / "dir" / "output.mp4")
        rec = SimulationRecorder(sim_core, output=nested_path, width=80, height=60)
        rec.start()
        rec._capture_frame(sim_core, 0.1)
        rec.stop()
        path = rec.save()
        assert os.path.exists(path)

    def test_mp4_without_imageio_raises(self, sim_core, tmp_mp4, monkeypatch):
        """MP4 save raises RuntimeError when imageio is not available."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("imageio"):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        rec = SimulationRecorder(sim_core, output=tmp_mp4, width=80, height=60)
        rec.start()
        rec._capture_frame(sim_core, 0.1)
        rec.stop()

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(RuntimeError, match="imageio"):
            rec.save()


class TestRecorderCamera:
    """Test camera mode behavior."""

    def test_auto_camera_no_objects(self, sim_core, tmp_gif):
        """Auto camera with empty scene still produces valid frame."""
        # Remove all sim_objects (only plane exists)
        rec = SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120, camera_mode="auto")
        rec.start()
        rec._capture_frame(sim_core, 0.1)
        assert rec.frame_count == 1

    def test_auto_camera_with_robot(self, sim_core_with_robot, tmp_gif):
        """Auto camera frames the robot."""
        rec = SimulationRecorder(sim_core_with_robot, output=tmp_gif, width=160, height=120, camera_mode="auto")
        rec.start()
        rec._capture_frame(sim_core_with_robot, 0.1)
        assert rec.frame_count == 1
        # Verify frame is not all-black (camera sees something)
        assert rec._frames[0].sum() > 0

    def test_orbit_camera_rotates(self, sim_core, tmp_gif):
        """Orbit mode produces different frames over time."""
        rec = SimulationRecorder(
            sim_core, output=tmp_gif, width=160, height=120, camera_mode="orbit", duration=2.0, orbit_degrees=90.0
        )
        rec.start()  # at sim time 0
        rec._capture_frame(sim_core, 0.1)
        frame1 = rec._frames[0].copy()

        rec._capture_at(1.0)
        frame2 = rec._frames[1].copy()

        # Frames should differ (camera moved)
        assert not np.array_equal(frame1, frame2)

    def test_manual_camera(self, sim_core, tmp_gif):
        """Manual camera uses provided params."""
        rec = SimulationRecorder(
            sim_core,
            output=tmp_gif,
            width=160,
            height=120,
            camera_mode="manual",
            camera_params={
                "camera_target": [0, 0, 0],
                "camera_distance": 5.0,
                "camera_yaw": 0,
                "camera_pitch": -45,
            },
        )
        rec.start()
        rec._capture_frame(sim_core, 0.1)
        assert rec.frame_count == 1

    def test_gui_camera_captures_without_view_matrix(self, sim_core, tmp_gif, monkeypatch):
        """GUI mode calls getCameraImage without viewMatrix/projectionMatrix (uses GUI camera directly)."""
        monkeypatch.setattr(sim_core.params, "gui", True)

        # Track what arguments getCameraImage receives
        captured_kwargs = {}
        dummy_rgba = np.zeros((120, 160, 4), dtype=np.uint8)
        original_getCameraImage = p.getCameraImage

        def mock_getCameraImage(*args, **kwargs):
            captured_kwargs.update(kwargs)
            captured_kwargs["_positional"] = args
            return (160, 120, dummy_rgba.flatten().tolist(), [0.0] * (160 * 120), [0] * (160 * 120))

        monkeypatch.setattr(p, "getCameraImage", mock_getCameraImage)

        rec = SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120, camera_mode="gui")
        rec.start()
        rec._capture_frame(sim_core, 0.1)

        # gui mode should NOT pass viewMatrix or projectionMatrix
        assert "viewMatrix" not in captured_kwargs
        assert "projectionMatrix" not in captured_kwargs
        assert rec.frame_count == 1

    def test_gui_camera_uses_opengl_renderer(self, sim_core, tmp_gif, monkeypatch):
        """GUI mode uses ER_BULLET_HARDWARE_OPENGL for GPU rendering (matches GUI appearance)."""
        monkeypatch.setattr(sim_core.params, "gui", True)

        captured_kwargs = {}
        dummy_rgba = np.zeros((120, 160, 4), dtype=np.uint8)

        def mock_getCameraImage(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return (160, 120, dummy_rgba.flatten().tolist(), [0.0] * (160 * 120), [0] * (160 * 120))

        monkeypatch.setattr(p, "getCameraImage", mock_getCameraImage)

        rec = SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120, camera_mode="gui")
        rec.start()
        rec._capture_frame(sim_core, 0.1)

        assert captured_kwargs["renderer"] == p.ER_BULLET_HARDWARE_OPENGL

    def test_gui_camera_raises_in_direct_mode(self, sim_core, tmp_gif):
        """camera_mode='gui' raises ValueError when sim is not in GUI mode."""
        # sim_core fixture uses gui=False (p.DIRECT)
        with pytest.raises(ValueError, match="gui.*DIRECT"):
            SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120, camera_mode="gui")

    def test_gui_camera_default_when_gui_mode(self, sim_core, tmp_gif, monkeypatch):
        """When params.gui=True and no camera_mode specified, default to 'gui'."""
        monkeypatch.setattr(sim_core.params, "gui", True)

        rec = SimulationRecorder(sim_core, output=tmp_gif, width=160, height=120, camera_mode="auto")
        # When gui=True, auto should be promoted to gui
        assert rec._camera_mode == "gui"


# ---------------------------------------------------------------------------
# Integration Tests — core_simulation API
# ---------------------------------------------------------------------------


class TestRecorderTimeBase:
    """Test time_base='sim' (default) vs time_base='real' capture modes."""

    def test_default_time_base_is_sim(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif)
        assert rec._time_base == "sim"

    def test_invalid_time_base_raises(self, sim_core, tmp_gif):
        with pytest.raises(ValueError, match="time_base"):
            SimulationRecorder(sim_core, output=tmp_gif, time_base="invalid")

    def test_sim_mode_registers_with_frequency(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif, fps=10, time_base="sim")
        rec.start()
        # sim mode: callback registered with frequency=fps
        cb = sim_core._callbacks[-1]
        assert cb["frequency"] == 10.0
        rec.stop()

    def test_real_mode_registers_every_step(self, sim_core, tmp_gif):
        rec = SimulationRecorder(sim_core, output=tmp_gif, fps=10, time_base="real")
        rec.start()
        # real mode: callback registered with frequency=None (every step)
        cb = sim_core._callbacks[-1]
        assert cb["frequency"] is None
        rec.stop()

    def test_real_mode_throttles_by_wall_clock(self, sim_core, tmp_gif, monkeypatch):
        """In real mode, frames are captured at wall-clock intervals, not sim-time."""
        import pybullet_fleet.recorder as rec_module

        # Mock time.monotonic to control wall clock
        wall_clock = [0.0]

        def mock_monotonic():
            return wall_clock[0]

        monkeypatch.setattr(rec_module.time, "monotonic", mock_monotonic)

        rec = SimulationRecorder(sim_core, output=tmp_gif, width=80, height=60, fps=10, time_base="real")
        rec.start()

        # First call at t=0 should capture (last_wall_capture initialized to 0)
        wall_clock[0] = 0.0
        rec._capture_frame(sim_core, 0.01)
        assert rec.frame_count == 1

        # Second call at t=0.05 (< 0.1 interval) should skip
        wall_clock[0] = 0.05
        rec._capture_frame(sim_core, 0.01)
        assert rec.frame_count == 1

        # Third call at t=0.11 (>= 0.1 interval) should capture
        wall_clock[0] = 0.11
        rec._capture_frame(sim_core, 0.01)
        assert rec.frame_count == 2

    def test_real_mode_duration_uses_wall_clock(self, sim_core, tmp_gif, monkeypatch):
        """In real mode, duration is measured in wall-clock seconds."""
        import pybullet_fleet.recorder as rec_module

        wall_clock = [0.0]

        def mock_monotonic():
            return wall_clock[0]

        monkeypatch.setattr(rec_module.time, "monotonic", mock_monotonic)

        rec = SimulationRecorder(sim_core, output=tmp_gif, width=80, height=60, fps=10, duration=1.0, time_base="real")
        rec.start()

        # Capture within duration
        wall_clock[0] = 0.5
        rec._capture_frame(sim_core, 0.01)
        assert rec.is_recording is True
        assert rec.frame_count == 1

        # Wall clock exceeds duration → stop
        wall_clock[0] = 1.1
        rec._capture_frame(sim_core, 0.01)
        assert rec.is_recording is False
        assert rec.frame_count == 1

    def test_sim_mode_ignores_wall_clock(self, sim_core, tmp_gif):
        """In sim mode (default), capture frequency is based on sim_time, not wall clock."""
        rec = SimulationRecorder(sim_core, output=tmp_gif, width=80, height=60, fps=10, duration=1.0, time_base="sim")
        rec.start()  # at sim time 0

        # Capture within duration
        rec._capture_at(0.5)
        assert rec.is_recording is True
        assert rec.frame_count == 1

        # Sim time exceeds duration → stop
        rec._capture_at(1.1)
        assert rec.is_recording is False
        assert rec.frame_count == 1


class TestCoreRecordingAPI:
    """Test start_recording/stop_recording on MultiRobotSimulationCore."""

    def test_start_stop_recording(self, sim_core, tmp_gif):
        rec = sim_core.start_recording(output=tmp_gif, width=160, height=120)
        assert isinstance(rec, SimulationRecorder)
        assert rec.is_recording is True
        assert sim_core._recorder is rec

        # Manually capture a frame
        rec._capture_frame(sim_core, 0.1)

        path = sim_core.stop_recording()
        assert path == tmp_gif
        assert os.path.exists(tmp_gif)
        assert sim_core._recorder is None

    def test_stop_recording_when_not_recording(self, sim_core):
        result = sim_core.stop_recording()
        assert result is None

    def test_run_simulation_auto_saves(self, sim_core_with_robot, tmp_gif):
        """Recording auto-saved when run_simulation() ends."""
        sim_core_with_robot.start_recording(output=tmp_gif, width=160, height=120, fps=15, duration=0.5)
        sim_core_with_robot.run_simulation(duration=1.0)
        # Should have auto-saved
        assert sim_core_with_robot._recorder is None
        assert os.path.exists(tmp_gif)
        assert os.path.getsize(tmp_gif) > 0

    def test_record_env_var(self, sim_core_with_robot, tmp_gif, monkeypatch):
        """RECORD env var triggers automatic recording."""
        monkeypatch.setenv("RECORD", tmp_gif)
        monkeypatch.setenv("RECORD_DURATION", "0.5")
        sim_core_with_robot.run_simulation(duration=1.0)
        assert os.path.exists(tmp_gif)
        assert os.path.getsize(tmp_gif) > 0
        assert sim_core_with_robot._recorder is None

    def test_no_record_env_var_no_recording(self, sim_core_with_robot, monkeypatch):
        """Without RECORD env var, no recording happens."""
        monkeypatch.delenv("RECORD", raising=False)
        sim_core_with_robot.run_simulation(duration=0.5)
        assert sim_core_with_robot._recorder is None

    def test_start_recording_with_time_base(self, sim_core, tmp_gif):
        """time_base parameter is passed through to recorder."""
        rec = sim_core.start_recording(output=tmp_gif, width=80, height=60, time_base="real")
        assert rec._time_base == "real"
        sim_core.stop_recording()

    def test_record_env_var_time_base(self, sim_core_with_robot, tmp_gif, monkeypatch):
        """RECORD_TIME_BASE env var sets time_base on recorder."""
        monkeypatch.setenv("RECORD", tmp_gif)
        monkeypatch.setenv("RECORD_DURATION", "0.5")
        monkeypatch.setenv("RECORD_TIME_BASE", "real")
        sim_core_with_robot.run_simulation(duration=1.0)
        assert os.path.exists(tmp_gif)
        assert os.path.getsize(tmp_gif) > 0

    def test_record_env_forces_headless_by_default(self, tmp_gif, monkeypatch):
        """RECORD without RECORD_GUI forces gui=False in setup_pybullet."""
        monkeypatch.setenv("RECORD", tmp_gif)
        monkeypatch.delenv("RECORD_GUI", raising=False)
        params = SimulationParams(gui=True, monitor=False, physics=False, duration=0, target_rtf=0)
        sim = MultiRobotSimulationCore(params)
        try:
            # setup_pybullet should have forced gui=False
            assert sim.params.gui is False
        finally:
            p.disconnect(sim.client)

    def test_record_gui_env_keeps_gui(self, tmp_gif, monkeypatch):
        """RECORD + RECORD_GUI=1 preserves gui=True in setup_pybullet."""
        monkeypatch.setenv("RECORD", tmp_gif)
        monkeypatch.setenv("RECORD_GUI", "1")
        # Intercept p.connect to force DIRECT (CI is headless, p.GUI crashes).
        # We only care that params.gui stays True after setup_pybullet logic.
        original_connect = p.connect
        monkeypatch.setattr(p, "connect", lambda *a, **kw: original_connect(p.DIRECT))
        params = SimulationParams(gui=True, monitor=False, physics=False, duration=0, target_rtf=0)
        sim = MultiRobotSimulationCore(params)
        try:
            assert sim.params.gui is True
        finally:
            p.disconnect(sim.client)

    def test_record_gui_env_noop_in_direct_mode(self, sim_core_with_robot, tmp_gif, monkeypatch):
        """RECORD_GUI has no effect when sim was created with gui=False.

        The env var only prevents setup_pybullet from *forcing* gui off;
        it does not override a sim that was headless from the start.
        camera_mode stays 'auto' (recorder auto-promotes only when params.gui).
        """
        monkeypatch.setenv("RECORD", tmp_gif)
        monkeypatch.setenv("RECORD_DURATION", "0.5")
        monkeypatch.setenv("RECORD_GUI", "1")
        sim_core_with_robot.run_simulation(duration=1.0)
        assert os.path.exists(tmp_gif)
        assert os.path.getsize(tmp_gif) > 0


class TestUnregisterCallback:
    """Test the new unregister_callback method."""

    def test_unregister_existing(self, sim_core):
        def my_cb(sim_core, dt):
            pass

        sim_core.register_callback(my_cb, frequency=10.0)
        assert len(sim_core._callbacks) == 1
        removed = sim_core.unregister_callback(my_cb)
        assert removed is True
        assert len(sim_core._callbacks) == 0

    def test_unregister_nonexistent(self, sim_core):
        def my_cb(sim_core, dt):
            pass

        removed = sim_core.unregister_callback(my_cb)
        assert removed is False

    def test_unregister_specific_callback(self, sim_core):
        def cb1(sim_core, dt):
            pass

        def cb2(sim_core, dt):
            pass

        sim_core.register_callback(cb1)
        sim_core.register_callback(cb2)
        assert len(sim_core._callbacks) == 2
        sim_core.unregister_callback(cb1)
        assert len(sim_core._callbacks) == 1
        assert sim_core._callbacks[0]["func"] is cb2
