#!/usr/bin/env python3
"""
Real-time data monitor window for PyBullet simulation
Shows simulation statistics in a separate tkinter window
"""

import atexit
import json
import math
import os
import sys
import subprocess
import threading
import time
from dataclasses import asdict, replace
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Optional

from pybullet_fleet._platform import IS_MACOS
from pybullet_fleet._defaults import SIMULATION as _SIM_D
from pybullet_fleet.gui_commands import GuiCommand, GuiCommandType, MonitorFrame
from pybullet_fleet.logging_utils import get_lazy_logger

logger = get_lazy_logger(__name__)

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:
    tk = None
    ttk = None
    logger.warning(
        "tkinter is not installed; DataMonitor GUI will be disabled. "
        "Install it (Debian/Ubuntu: `sudo apt-get install python3-tk`, "
        "or the matching `python3.X-tk` package for your interpreter) to enable the monitor window."
    )


class DataMonitor:
    """Real-time data monitoring window"""

    def __init__(
        self,
        title: str = "Simulation Monitor",
        enable_gui: bool = True,
        width: int = _SIM_D["monitor_width"],
        height: int = _SIM_D["monitor_height"],
        x: int = _SIM_D["monitor_x"],
        y: int = _SIM_D["monitor_y"],
        initial_target_rtf: Optional[float] = None,
        initial_timestep: Optional[float] = None,
    ) -> None:
        self.title: str = title
        self.enable_gui: bool = enable_gui
        self.running: bool = False
        self.window: Optional[Any] = None
        self.labels: Dict[str, Any] = {}
        self.data_file: str = "/tmp/pybullet_sim_data.json"
        self.update_interval: float = 0.5  # Update every 500ms
        # The entity list's JSON, kept for the tuple it was made from: the
        # list only changes when objects come and go, and serializing a few
        # thousand entries on every write would cost more than the step.
        self._entities_source: Optional[tuple] = None
        self._entities_json: str = "[]"
        self.status_label: Optional[Any] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self._monitor_process: Optional[subprocess.Popen] = None
        self._command_pipe: Optional[Connection] = None
        self._parent_poll: Optional[Callable[[], None]] = None
        self.command_sink: Optional[Callable[[GuiCommand], bool]] = None
        self._entity_rows: list[int] = []
        self._all_entities: list[Dict[str, Any]] = []
        self._follow_enabled: bool = False
        self._current_selected_entity_id: Optional[int] = None
        self.initial_target_rtf: Optional[float] = initial_target_rtf
        self.initial_timestep: Optional[float] = initial_timestep
        if tk is None and self.enable_gui:
            logger.warning(
                "DataMonitor GUI was requested but tkinter is unavailable; "
                "falling back to file-only mode (data still written to %s).",
                self.data_file,
            )
            self.enable_gui = False
        # If x/y are -1, omit position so the window manager places the window
        # on the primary display. Explicit x/y use absolute virtual-screen coords.
        if x >= 0 and y >= 0:
            self.geometry_string: str = f"{width}x{height}+{x}+{y}"
        else:
            self.geometry_string: str = f"{width}x{height}"

    def start(self) -> None:
        """Start Tk in a separate process on macOS, or a thread elsewhere."""
        if self.running:
            return

        self.running = True

        # Only start GUI thread if GUI is enabled
        if self.enable_gui:
            if IS_MACOS:
                self._start_process()
            else:
                self.monitor_thread = threading.Thread(target=self._run_monitor, daemon=True)
                self.monitor_thread.start()

    def _start_process(self) -> None:
        # PyBullet and Tk both own NSApplication on macOS. Even main-thread Tk
        # cannot share PyBullet's process. Use a fresh interpreter, not fork or
        # multiprocessing spawn (which would re-execute unguarded demo scripts).
        parent, child = Pipe()
        settings = json.dumps(
            {
                "title": self.title,
                "geometry": self.geometry_string,
                "data_file": self.data_file,
                "initial_target_rtf": self.initial_target_rtf,
                "initial_timestep": self.initial_timestep,
            }
        )
        try:
            self._monitor_process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from pybullet_fleet.data_monitor import _process_main; _process_main()",
                    str(child.fileno()),
                    settings,
                ],
                pass_fds=(child.fileno(),),
            )
        except Exception:
            parent.close()
            self.running = False
            raise
        finally:
            child.close()
        self._command_pipe = parent
        atexit.register(self.stop)

    def process_events(self) -> None:
        """Forward subprocess commands at the simulation boundary, even paused."""
        pipe = self._command_pipe
        if pipe is None:
            return
        try:
            for _ in range(100):
                if not pipe.poll():
                    break
                command = GuiCommand(**pipe.recv())
                if self.command_sink is not None and not self.command_sink(command):
                    logger.warning("Monitor command queue is full; request was not applied")
        except (EOFError, OSError):
            self.stop()

    def stop(self) -> None:
        """Stop the monitor window"""
        self.running = False
        if IS_MACOS and self._command_pipe is not None:
            self._command_pipe.close()
            self._command_pipe = None
        if IS_MACOS and self._monitor_process is not None:
            try:
                self._monitor_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._monitor_process.terminate()
                self._monitor_process.wait(timeout=3)
            self._monitor_process = None
            atexit.unregister(self.stop)
        if self.window:
            try:
                self.window.quit()
                self.window.destroy()
            except Exception:
                pass
            if IS_MACOS:
                self.window = None

    def _run_monitor(self) -> None:
        """Run the tkinter monitor window"""
        if tk is None or ttk is None:
            self.running = False
            return

        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.geometry(self.geometry_string)
        self.window.configure(bg="black")
        if IS_MACOS:
            self.window.protocol("WM_DELETE_WINDOW", self.stop)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        # Create main frame  (padding: top right bottom left)
        main_frame = ttk.Frame(self.window, padding="4 5 4 2")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure style for better visibility
        style = ttk.Style()
        style.configure("Monitor.TLabel", foreground="lime", background="black", font=("Courier", 9))

        # Keep compact run statistics beside a vertically expandable entity
        # inspector.  Large fleets need the latter to use the window height.
        metrics = ttk.Frame(main_frame)
        metrics.grid(row=0, column=0, sticky=(tk.N, tk.W), padx=(0, 8))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create labels for different data fields
        self.labels = {}
        fields = [
            "Sim Time",
            "Real Time",
            "Target RTF",
            "Actual RTF",
            "Time Step",
            "Physics",
            "Agents",
            "Objects",
            "Collisions",
            "Tot Collis.",
            "Steps",
        ]

        for i, field in enumerate(fields):
            label = ttk.Label(metrics, text=f"{field}: --", style="Monitor.TLabel")
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            self.labels[field] = label

        # Playback controls submit a request to the simulation thread.  They
        # never call core/PyBullet methods directly from tkinter's thread.
        controls = ttk.Frame(metrics)
        controls.grid(row=len(fields), column=0, sticky=tk.W, pady=(8, 2))
        ttk.Button(controls, text="Pause", command=lambda: self._submit_command(GuiCommandType.PAUSE)).grid(
            row=0, column=0, padx=(0, 3)
        )
        ttk.Button(controls, text="Resume", command=lambda: self._submit_command(GuiCommandType.RESUME)).grid(
            row=0, column=1, padx=3
        )
        ttk.Button(controls, text="Step", command=lambda: self._submit_command(GuiCommandType.SINGLE_STEP)).grid(
            row=0, column=2, padx=(3, 0)
        )

        pacing = ttk.LabelFrame(metrics, text="Pacing", padding="3 2 3 2")
        pacing.grid(row=len(fields) + 1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Label(pacing, text="RTF").grid(row=0, column=0, sticky=tk.W)
        self.target_rtf_var = tk.StringVar(value=f"{self.initial_target_rtf:g}" if self.initial_target_rtf is not None else "")
        self.target_rtf_entry = ttk.Entry(pacing, textvariable=self.target_rtf_var, width=7)
        self.target_rtf_entry.grid(row=0, column=1, padx=(3, 6))
        ttk.Label(pacing, text="dt").grid(row=0, column=2, sticky=tk.W)
        self.timestep_var = tk.StringVar(value=f"{self.initial_timestep:g}" if self.initial_timestep is not None else "")
        self.timestep_entry = ttk.Entry(pacing, textvariable=self.timestep_var, width=7)
        self.timestep_entry.grid(row=0, column=3, padx=(3, 6))
        ttk.Button(pacing, text="Apply", command=self._apply_pacing).grid(row=0, column=4)

        inspector = ttk.LabelFrame(main_frame, text="Entities", padding="3 3 3 3")
        inspector.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        inspector.columnconfigure(0, weight=1)
        inspector.rowconfigure(2, weight=1)
        ttk.Label(inspector, text="Search name / ID / type").grid(row=0, column=0, sticky=tk.W)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._apply_entity_filter)
        ttk.Entry(inspector, textvariable=self.search_var, width=28).grid(row=1, column=0, sticky=tk.W, pady=(0, 3))
        self.entity_list = tk.Listbox(inspector, height=6, exportselection=False)
        self.entity_list.grid(row=2, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.entity_list.bind("<<ListboxSelect>>", self._on_entity_selected)
        scrollbar = ttk.Scrollbar(inspector, orient=tk.VERTICAL, command=self.entity_list.yview)
        scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))
        self.entity_list.configure(yscrollcommand=scrollbar.set)
        selection_controls = ttk.Frame(inspector)
        selection_controls.grid(row=3, column=0, sticky=tk.W, pady=(3, 0))
        self.follow_button = ttk.Button(selection_controls, text="Follow selected", command=self._toggle_follow)
        self.follow_button.grid(row=0, column=0, padx=(0, 3))
        ttk.Button(selection_controls, text="Clear selection", command=self._clear_selection).grid(row=0, column=1)
        self.inspector_label = ttk.Label(inspector, text="No entity selected", style="Monitor.TLabel", justify=tk.LEFT)
        self.inspector_label.grid(row=4, column=0, sticky=tk.W)

        pose_editor = ttk.LabelFrame(inspector, text="Move selected (paused)", padding="3 2 3 2")
        pose_editor.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(4, 0))
        self.pose_vars = {axis: tk.StringVar() for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.pose_entries: Dict[str, Any] = {}
        for row, axes in enumerate((("x", "y", "z"), ("roll", "pitch", "yaw"))):
            for column, axis in enumerate(axes):
                ttk.Label(pose_editor, text=f"{axis}{'°' if row else ''}").grid(row=row, column=column * 2)
                entry = ttk.Entry(pose_editor, textvariable=self.pose_vars[axis], width=7)
                entry.grid(row=row, column=column * 2 + 1, padx=(2, 4))
                self.pose_entries[axis] = entry
        ttk.Button(pose_editor, text="Move", command=self._apply_selected_pose).grid(
            row=2, column=0, columnspan=6, pady=(3, 0)
        )

        # Add status label
        self.status_label = ttk.Label(metrics, text="Status: Waiting for data...", style="Monitor.TLabel")
        self.status_label.grid(row=len(fields) + 2, column=0, sticky=tk.W, pady=10)

        # Start periodic update
        self.window.after(int(self.update_interval * 1000), self._update_display)
        if self._parent_poll is not None:
            self.window.after(100, self._parent_poll)

        # Start the tkinter main loop
        try:
            self.window.mainloop()
        except Exception:
            pass
        finally:
            self.running = False

    def _update_display(self) -> None:
        """Update the display with latest data"""
        if not self.running:
            return

        try:
            # Read data from shared file
            if os.path.exists(self.data_file):
                with open(self.data_file, "r") as f:
                    data = json.load(f)

                # Update labels
                self.labels["Sim Time"].config(text=f"Sim Time : {data.get('sim_time', 0):.1f}s")
                self.labels["Real Time"].config(text=f"Real Time: {data.get('real_time', 0):.1f}s")
                self.labels["Target RTF"].config(text=f"Tgt RTF  : {data.get('target_rtf', 0):.1f}x")
                self.labels["Actual RTF"].config(text=f"Act RTF  : {data.get('actual_rtf', 0):.1f}x")
                self.labels["Time Step"].config(text=f"dt={data.get('time_step', 0):.4f}s ({data.get('frequency', 0):.0f}Hz)")
                self.labels["Physics"].config(text=f"Physics  : {data.get('physics', '?')}")
                self.labels["Agents"].config(text=f"Agents   : {data.get('agents', 0)}")
                self.labels["Objects"].config(text=f"Objects  : {data.get('objects', 0)}")
                self.labels["Collisions"].config(text=f"Collis.  : {data.get('active_collisions', 0)}")
                self.labels["Tot Collis."].config(text=f"Tot Col. : {data.get('collisions', 0)}")
                self.labels["Steps"].config(text=f"Steps    : {data.get('steps', 0)}")
                if self.window.focus_get() is not self.target_rtf_entry:
                    self.target_rtf_var.set(f"{data.get('target_rtf', 0):g}")
                if self.window.focus_get() is not self.timestep_entry:
                    self.timestep_var.set(f"{data.get('time_step', 0):g}")

                state = "PAUSED" if data.get("paused") else "PLAYING"
                self.status_label.config(text=f"Status: {state} (Updated: {time.strftime('%H:%M:%S')})")
                self._update_inspector(data)
            else:
                self.status_label.config(text="Status: No data file found")

        except Exception as e:
            self.status_label.config(text=f"Status: Error reading data - {str(e)}")

        # Schedule next update
        if self.running:
            self.window.after(int(self.update_interval * 1000), self._update_display)

    def write_data(self, sim_data):
        """Write simulation data to shared file (called from main simulation).

        Uses atomic write (write to temp file + os.replace) to prevent the
        reader from seeing a truncated / partially-written JSON file.
        """
        self._write_text(json.dumps(sim_data))

    def _write_text(self, text: str) -> None:
        tmp_path = self.data_file + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                f.write(text)
            os.replace(tmp_path, self.data_file)  # atomic on POSIX
        except Exception:
            pass  # Ignore errors in data writing

    def write_frame(self, frame: MonitorFrame) -> None:
        """Write a :class:`MonitorFrame` through the legacy JSON monitor path.

        The JSON (keys and values) is what it has always been; only the
        entity list is serialized once per change of ``frame.entities``
        rather than on every write.
        """
        if frame.entities is not self._entities_source:
            self._entities_json = json.dumps([asdict(entity) for entity in frame.entities])
            self._entities_source = frame.entities
        data = asdict(replace(frame, entities=()))
        del data["entities"]
        # Keep the established JSON keys stable for external file readers.
        data["time_step"] = data.pop("timestep")
        data["physics"] = "enabled" if data.pop("physics_enabled") else "disabled"
        data["frequency"] = 1 / frame.timestep
        text = json.dumps(data)
        # Splice the cached list in as the "entities" member.
        self._write_text(text[:-1] + (", " if data else "") + '"entities": ' + self._entities_json + "}")

    def set_command_sink(self, sink: Callable[[GuiCommand], bool]) -> None:
        """Set the thread-safe callable that accepts monitor commands."""
        self.command_sink = sink

    def _submit_command(self, command: GuiCommandType, **kwargs: Any) -> bool:
        """Send a UI request without accessing simulation state from tkinter."""
        if self.command_sink is None:
            return False
        accepted = self.command_sink(GuiCommand(command=command, **kwargs))
        if not accepted and self.status_label is not None:
            self.status_label.config(text="Status: Command queue is full; request was not applied")
        return accepted

    def _on_entity_selected(self, _event: Any) -> None:
        selection = self.entity_list.curselection()
        if selection:
            self._submit_command(GuiCommandType.SELECT_ENTITY, entity_id=self._entity_rows[selection[0]])

    def _toggle_follow(self) -> None:
        self._submit_command(GuiCommandType.SET_FOLLOW, enabled=not self._follow_enabled)

    def _clear_selection(self) -> None:
        self._submit_command(GuiCommandType.SELECT_ENTITY, entity_id=None)

    def _apply_pacing(self) -> None:
        """Validate monitor inputs and request an atomic pacing change."""
        try:
            target_rtf = float(self.target_rtf_var.get())
            timestep = float(self.timestep_var.get())
        except ValueError:
            self.status_label.config(text="Status: RTF and dt must be numbers")
            return
        if target_rtf < 0 or timestep <= 0:
            self.status_label.config(text="Status: RTF must be >= 0 and dt must be > 0")
            return
        self._submit_command(GuiCommandType.SET_SIMULATION_PACING, target_rtf=target_rtf, timestep=timestep)

    def _apply_selected_pose(self) -> None:
        """Request a paused-only teleport using XYZ and yaw in degrees."""
        try:
            position = tuple(float(self.pose_vars[axis].get()) for axis in ("x", "y", "z"))
            rpy_radians = tuple(math.radians(float(self.pose_vars[axis].get())) for axis in ("roll", "pitch", "yaw"))
        except ValueError:
            self.status_label.config(text="Status: X, Y, Z, roll, pitch, and yaw must be numbers")
            return
        selected = self._selected_entity_id()
        if selected is None:
            self.status_label.config(text="Status: Select an entity before moving it")
            return
        self._submit_command(
            GuiCommandType.SET_ENTITY_POSE,
            entity_id=selected,
            position=position,
            rpy_radians=rpy_radians,
        )

    def _selected_entity_id(self) -> Optional[int]:
        return self._current_selected_entity_id

    def _update_inspector(self, data: Dict[str, Any]) -> None:
        entities = data.get("entities", [])
        if entities != self._all_entities:
            self._all_entities = entities
            self._apply_entity_filter()
        selected = data.get("selected_entity")
        self._follow_enabled = bool(data.get("follow_enabled"))
        self.follow_button.config(text="Stop following" if self._follow_enabled else "Follow selected")
        if selected:
            self._current_selected_entity_id = selected["entity"]["object_id"]
            position = ", ".join(f"{value:.2f}" for value in selected["position"])
            orientation = ", ".join(f"{value:.2f}" for value in selected["orientation"])
            action = selected["action_type"] or "idle"
            if selected["action_status"]:
                action = f"{action} ({selected['action_status']})"
            collision = "yes" if selected["active_collisions"] else "no"
            self.inspector_label.config(
                text=(
                    f"ID: {selected['entity']['object_id']}  {selected['entity']['name']}\n"
                    f"Type: {selected['entity']['kind']}\n"
                    f"Position: ({position})\nOrientation: ({orientation})\n"
                    f"Action: {action}; queued: {selected['queued_actions']}\n"
                    f"Attached: {selected['attached_objects']}; active collision: {collision}"
                )
            )
            if not any(entry.focus_get() is entry for entry in self.pose_entries.values()):
                roll = math.degrees(
                    math.atan2(
                        2
                        * (
                            selected["orientation"][3] * selected["orientation"][0]
                            + selected["orientation"][1] * selected["orientation"][2]
                        ),
                        1 - 2 * (selected["orientation"][0] ** 2 + selected["orientation"][1] ** 2),
                    )
                )
                pitch = math.degrees(
                    math.asin(
                        max(
                            -1.0,
                            min(
                                1.0,
                                2
                                * (
                                    selected["orientation"][3] * selected["orientation"][1]
                                    - selected["orientation"][2] * selected["orientation"][0]
                                ),
                            ),
                        )
                    )
                )
                yaw = math.degrees(
                    math.atan2(
                        2
                        * (
                            selected["orientation"][3] * selected["orientation"][2]
                            + selected["orientation"][0] * selected["orientation"][1]
                        ),
                        1 - 2 * (selected["orientation"][1] ** 2 + selected["orientation"][2] ** 2),
                    )
                )
                for axis, value in zip(("x", "y", "z"), selected["position"]):
                    self.pose_vars[axis].set(f"{value:g}")
                for axis, value in zip(("roll", "pitch", "yaw"), (roll, pitch, yaw)):
                    self.pose_vars[axis].set(f"{value:g}")
        else:
            self._current_selected_entity_id = None
            self.inspector_label.config(text="No entity selected")

    def _apply_entity_filter(self, *_args: Any) -> None:
        """Apply the case-insensitive inspector filter without touching sim state."""
        query = self.search_var.get().casefold().strip()
        self.entity_list.delete(0, tk.END)
        self._entity_rows = []
        for entry in self._all_entities:
            label = f"{entry['object_id']}: {entry['name']} ({entry['kind']})"
            if query and query not in label.casefold():
                continue
            self._entity_rows.append(entry["object_id"])
            self.entity_list.insert(tk.END, label)


def _process_main() -> None:
    """macOS child entry point: Tk owns this process's main thread."""
    pipe = Connection(int(sys.argv[1]))
    settings = json.loads(sys.argv[2])
    monitor = DataMonitor(
        settings["title"],
        initial_target_rtf=settings["initial_target_rtf"],
        initial_timestep=settings["initial_timestep"],
    )
    monitor.geometry_string = settings["geometry"]
    monitor.data_file = settings["data_file"]

    def send_command(command: GuiCommand) -> bool:
        try:
            pipe.send(asdict(command))
            return True
        except (BrokenPipeError, EOFError, OSError):
            return False

    def check_parent() -> None:
        try:
            if pipe.poll():
                pipe.recv()  # EOF when the parent exits or stops the monitor.
        except (EOFError, OSError):
            monitor.stop()
            return
        if monitor.running and monitor.window is not None:
            monitor.window.after(100, check_parent)

    monitor.set_command_sink(send_command)
    monitor._parent_poll = check_parent
    monitor.running = True
    try:
        monitor._run_monitor()
    finally:
        pipe.close()


# Standalone monitor launcher
def main():
    """Run standalone data monitor"""
    monitor = DataMonitor("PyBullet Simulation Monitor")
    print("Starting data monitor window...")
    print("Press Ctrl+C to exit")

    try:
        if IS_MACOS:
            monitor.running = True
            monitor._run_monitor()
        else:
            monitor.start()
            while monitor.running:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()


if __name__ == "__main__":
    main()
    # This file has been moved to the core directory
