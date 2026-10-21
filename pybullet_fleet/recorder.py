"""
recorder.py — SimulationRecorder for headless/GUI simulation recording.

Captures frames via ``getCameraImage`` and saves as GIF or MP4.
Uses GPU rendering (``ER_BULLET_HARDWARE_OPENGL``) in GUI mode,
CPU rendering (``ER_TINY_RENDERER``) for headless/custom cameras.
Integrates with ``MultiRobotSimulationCore`` via the callback system.

Output format is determined by file extension:

- ``.gif`` — Animated GIF via Pillow (always available)
- ``.mp4`` — H.264 MP4 via imageio + pyav (requires ``pip install imageio[pyav]``)

Usage::

    # GIF (default)
    sim.start_recording("output.gif", duration=4.0, fps=15)
    sim.run_simulation(duration=5.0)

    # MP4 (smaller files, better quality)
    sim.start_recording("output.mp4", duration=4.0, fps=30)
    sim.run_simulation(duration=5.0)

    # Via environment variable (zero code changes)
    # RECORD=output.mp4 python pybullet_fleet/examples/scale/100robots_grid_demo.py

    # Direct instantiation
    rec = SimulationRecorder(sim_core, output="output.mp4", duration=4.0)
    rec.start()
    sim_core.run_simulation(duration=5.0)
    rec.stop()
    rec.save()
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pybullet as p

from pybullet_fleet.logging_utils import get_lazy_logger

if TYPE_CHECKING:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore

logger = get_lazy_logger(__name__)


class SimulationRecorder:
    """Record simulation frames and save as animated GIF or MP4.

    Captures RGB frames using ``p.getCameraImage()``. Uses
    ``ER_TINY_RENDERER`` (CPU) for headless/custom cameras, or
    ``ER_BULLET_HARDWARE_OPENGL`` (GPU) in ``"gui"`` mode for
    faster rendering that matches the GUI window exactly.

    Output format is auto-detected from the file extension:

    - ``.gif`` — Animated GIF via Pillow (always available)
    - ``.mp4`` — H.264 MP4 via imageio + pyav (requires ``pip install imageio[pyav]``)

    Camera modes:

    - ``"auto"``  — Compute scene AABB, position camera overhead with auto-framing.
      Automatically promoted to ``"gui"`` when running in ``p.GUI`` mode.
    - ``"gui"``   — Record exactly what the GUI camera shows, using
      ``ER_BULLET_HARDWARE_OPENGL`` (GPU). Mouse pan/zoom/rotate is
      captured in real time. Only valid in ``p.GUI`` mode.
    - ``"orbit"`` — Same as auto + camera orbits horizontally over the recording.
    - ``"manual"`` — Use explicit camera parameters from ``camera_params``.

    Time bases:

    - ``"sim"``  — (default) Capture at fixed sim-time intervals. Playback always
      looks like 1× speed regardless of how fast the simulation ran.
    - ``"real"`` — Capture at wall-clock intervals. Playback reflects actual
      execution speed (e.g. a 10× sim produces a 10× fast video).

    Args:
        sim_core: The simulation core instance to record.
        output: Output file path (``.gif`` or ``.mp4``). Directory is created automatically.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Capture frequency and playback FPS.
        duration: Recording duration in simulation seconds (``time_base="sim"``) or
            wall-clock seconds (``time_base="real"``). ``None`` for manual stop.
        camera_mode: ``"auto"`` | ``"gui"`` | ``"orbit"`` | ``"manual"``.
            ``"auto"`` is promoted to ``"gui"`` when ``params.gui`` is True.
        camera_params: Camera parameters for ``"manual"`` mode. Keys:
            ``camera_target``, ``camera_distance``, ``camera_yaw``, ``camera_pitch``.
        orbit_degrees: Total horizontal orbit angle in degrees (for ``"orbit"`` mode).
        time_base: ``"sim"`` (default) or ``"real"``. Controls capture timing and
            duration measurement. See **Time bases** above.
    """

    def __init__(
        self,
        sim_core: "MultiRobotSimulationCore",
        output: str = "recording.gif",
        width: int = 800,
        height: int = 600,
        fps: int = 15,
        duration: Optional[float] = None,
        camera_mode: str = "auto",
        camera_params: Optional[Dict[str, Any]] = None,
        orbit_degrees: float = 60.0,
        time_base: str = "sim",
    ) -> None:
        if time_base not in ("sim", "real"):
            raise ValueError(f"time_base must be 'sim' or 'real', got '{time_base}'")

        # Auto-promote "auto" → "gui" when running with a GUI window
        is_gui = sim_core.params.gui
        if camera_mode == "auto" and is_gui:
            camera_mode = "gui"

        if camera_mode == "gui" and not is_gui:
            raise ValueError(
                "camera_mode='gui' requires p.GUI connection (params.gui=True). "
                "In p.DIRECT mode, use 'auto', 'orbit', or 'manual'."
            )

        self._sim_core = sim_core
        self._output = output
        self._width = width
        self._height = height
        self._fps = fps
        self._duration = duration
        self._camera_mode = camera_mode
        self._camera_params = camera_params or {}
        self._orbit_degrees = orbit_degrees
        self._time_base = time_base

        self._frames: List[np.ndarray] = []
        self._recording: bool = False
        self._start_sim_time: float = 0.0
        self._start_wall_time: float = 0.0
        self._last_wall_capture: float = 0.0

        self._proj_matrix = p.computeProjectionMatrixFOV(fov=50, aspect=width / height, nearVal=0.1, farVal=200)

        # Cached auto-camera values (computed once at start)
        self._auto_center: List[float] = [0.0, 0.0, 0.0]
        self._auto_extent: float = 10.0
        self._start_yaw: float = math.radians(45)

        # Bound callback reference (stored so unregister_callback can match by identity)
        self._callback_ref = self._capture_frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin recording. Registers a capture callback on sim_core."""
        if self._recording:
            return  # already recording
        self._recording = True
        self._start_sim_time = self._sim_core.sim_time
        self._start_wall_time = time.monotonic()
        self._last_wall_capture = self._start_wall_time - (1.0 / self._fps)  # ensure first capture fires
        self._frames.clear()
        self._compute_auto_camera()

        if self._time_base == "real":
            # Register at high frequency; wall-clock throttling in _capture_frame.
            # Use None (every step) to not miss wall-clock deadlines.
            self._sim_core.register_callback(self._callback_ref, frequency=None)
        else:
            self._sim_core.register_callback(self._callback_ref, frequency=float(self._fps))

        logger.info(
            "Recording started: %s (%dx%d @ %dfps, time_base=%s)",
            self._output,
            self._width,
            self._height,
            self._fps,
            self._time_base,
        )

    def stop(self) -> None:
        """Stop recording. Unregisters the capture callback."""
        if not self._recording:
            return
        self._recording = False
        self._sim_core.unregister_callback(self._callback_ref)
        logger.info("Recording stopped: %d frames captured", len(self._frames))

    def save(self) -> str:
        """Write captured frames to file (GIF or MP4, auto-detected from extension).

        Returns:
            The output file path.

        Raises:
            ValueError: If no frames have been captured.
            RuntimeError: If MP4 is requested but ``imageio`` is not installed.
        """
        if not self._frames:
            raise ValueError("No frames captured — call start() and run simulation before save()")

        os.makedirs(os.path.dirname(os.path.abspath(self._output)), exist_ok=True)

        ext = os.path.splitext(self._output)[1].lower()
        if ext == ".mp4":
            self._save_mp4()
        else:
            self._save_gif()

        file_size_mb = os.path.getsize(self._output) / (1024 * 1024)
        logger.info("Saved: %s (%d frames, %.1f MB)", self._output, len(self._frames), file_size_mb)
        return self._output

    def _save_gif(self) -> None:
        """Save frames as animated GIF using Pillow."""
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in self._frames]
        duration_ms = int(1000 / self._fps)
        pil_frames[0].save(
            self._output,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )

        file_size_mb = os.path.getsize(self._output) / (1024 * 1024)
        if file_size_mb > 5.0:
            logger.warning("GIF exceeds 5 MB (%.1f MB). Consider using MP4 or reducing fps/duration/resolution.", file_size_mb)

    def _save_mp4(self) -> None:
        """Save frames as H.264 MP4 using imageio + pyav.

        Raises:
            RuntimeError: If imageio is not installed.
        """
        try:
            import imageio
        except ImportError:
            raise RuntimeError("MP4 output requires imageio: pip install imageio[pyav]") from None

        # Ensure frame dimensions are divisible by macro_block_size (16)
        # to avoid ffmpeg warnings and ensure codec compatibility.
        macro_block = 16
        h, w = self._frames[0].shape[:2]
        pad_h = (macro_block - h % macro_block) % macro_block
        pad_w = (macro_block - w % macro_block) % macro_block

        if pad_h or pad_w:
            frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") for f in self._frames]
        else:
            frames = self._frames

        # Use get_writer (works with imageio v2.x and v3.x)
        writer = imageio.get_writer(self._output, fps=self._fps, macro_block_size=None)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """Whether the recorder is currently capturing frames."""
        return self._recording

    @property
    def frame_count(self) -> int:
        """Number of frames captured so far."""
        return len(self._frames)

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _capture_frame(self, sim_core: "MultiRobotSimulationCore", dt: float) -> None:
        """Callback: capture one RGB frame at the simulation's current time."""
        self._capture_at(sim_core.sim_time)

    def _capture_at(self, sim_time: float) -> None:
        """Capture one RGB frame of the scene, taken at simulated time ``sim_time``."""
        if not self._recording:
            return
        elapsed = sim_time - self._start_sim_time

        if self._time_base == "real":
            now = time.monotonic()
            wall_elapsed = now - self._start_wall_time

            # Duration check (wall-clock)
            if self._duration is not None and wall_elapsed >= self._duration:
                self.stop()
                return

            # Throttle: skip if wall-clock interval hasn't elapsed
            interval = 1.0 / self._fps
            if now - self._last_wall_capture < interval:
                return
            self._last_wall_capture = now
        else:
            # Duration check (sim-time)
            if self._duration is not None and elapsed >= self._duration:
                self.stop()
                return

        view_matrix = self._compute_view_matrix(elapsed)

        if self._camera_mode == "gui":
            # GUI mode: let PyBullet use the GUI camera directly.
            # No viewMatrix/projectionMatrix → same view as the GUI window.
            # ER_BULLET_HARDWARE_OPENGL → GPU render, matching GUI appearance (shadows, lighting).
            _, _, rgba, _, _ = p.getCameraImage(
                self._width,
                self._height,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self._sim_core.client,
            )
        else:
            _, _, rgba, _, _ = p.getCameraImage(
                self._width,
                self._height,
                viewMatrix=view_matrix,
                projectionMatrix=self._proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self._sim_core.client,
            )
        rgb = np.array(rgba, dtype=np.uint8).reshape(self._height, self._width, 4)[:, :, :3]
        self._frames.append(rgb)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _compute_auto_camera(self) -> None:
        """Compute scene center and extent from sim_object positions.

        Delegates to :meth:`MultiRobotSimulationCore.compute_scene_bounds`
        so that GUI camera and recorder share the same scene-framing logic.
        """
        center, extent = self._sim_core.compute_scene_bounds()

        if not any(extent):
            # No objects or all co-located — use safe defaults
            self._auto_center = [center[0], center[1], 0.3]
            self._auto_extent = 10.0
            return

        max_extent = max(extent[0], extent[1], 2.0)
        self._auto_center = [center[0], center[1], 0.3]
        self._auto_extent = max_extent

    def _compute_view_matrix(self, elapsed: float) -> List[float]:
        """Compute the view matrix based on camera_mode, ``elapsed`` seconds into the recording.

        Note: ``"gui"`` mode is handled directly in :meth:`_capture_frame`
        (skips view/projection matrices entirely).
        """
        if self._camera_mode == "manual":
            return self._compute_manual_view()
        elif self._camera_mode == "orbit":
            return self._compute_orbit_view(elapsed)
        else:  # auto
            return self._compute_auto_view()

    def _compute_auto_view(self) -> List[float]:
        """Fixed overhead camera based on auto-computed scene bounds."""
        cx, cy, cz = self._auto_center
        radius = self._auto_extent * 0.9
        cam_height = self._auto_extent * 0.7
        yaw = self._start_yaw

        eye_x = cx + radius * math.cos(yaw)
        eye_y = cy + radius * math.sin(yaw)

        return list(
            p.computeViewMatrix(
                cameraEyePosition=[eye_x, eye_y, cam_height],
                cameraTargetPosition=self._auto_center,
                cameraUpVector=[0, 0, 1],
            )
        )

    def _compute_orbit_view(self, elapsed: float) -> List[float]:
        """Auto camera + horizontal orbit, ``elapsed`` seconds into the recording."""
        cx, cy, cz = self._auto_center
        radius = self._auto_extent * 0.9
        cam_height = self._auto_extent * 0.7

        total_duration = self._duration if self._duration else 10.0
        progress = min(elapsed / total_duration, 1.0) if total_duration > 0 else 0.0
        yaw = self._start_yaw + math.radians(self._orbit_degrees) * progress

        eye_x = cx + radius * math.cos(yaw)
        eye_y = cy + radius * math.sin(yaw)

        return list(
            p.computeViewMatrix(
                cameraEyePosition=[eye_x, eye_y, cam_height],
                cameraTargetPosition=self._auto_center,
                cameraUpVector=[0, 0, 1],
            )
        )

    def _compute_manual_view(self) -> List[float]:
        """Camera from explicit user-provided parameters."""
        target = self._camera_params.get("camera_target", [0, 0, 0])
        distance = self._camera_params.get("camera_distance", 10.0)
        yaw_deg = self._camera_params.get("camera_yaw", 0)
        pitch_deg = self._camera_params.get("camera_pitch", -45)

        # Convert spherical to cartesian
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        eye_x = target[0] + distance * math.cos(pitch) * math.cos(yaw)
        eye_y = target[1] + distance * math.cos(pitch) * math.sin(yaw)
        eye_z = target[2] - distance * math.sin(pitch)

        return list(
            p.computeViewMatrix(
                cameraEyePosition=[eye_x, eye_y, eye_z],
                cameraTargetPosition=target,
                cameraUpVector=[0, 0, 1],
            )
        )
