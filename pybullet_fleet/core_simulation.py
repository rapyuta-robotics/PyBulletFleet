"""
core_simulation.py
Reusable core simulation logic for multi-robot PyBullet environments.
Integrates generation, management, transport, attach/detach, collision detection,
coordinate conversion, occupied judgment, transport path generation, debugging,
monitoring, and log control for various robots, pallets, and meshes.
"""

import copy
import logging
import math
import numbers
import os
import statistics
import time
import tracemalloc  # Memory profiling (imported once at module level)
import warnings
import weakref
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

# --- All imports at the top for PEP8 compliance ---
import pybullet as p
import pybullet_data

import yaml

from pybullet_fleet._frequency import FrequencyGate
from pybullet_fleet._monitor_frames import MonitorFrames
from pybullet_fleet._platform import IS_MACOS
from pybullet_fleet.camera_controller import CameraController
from pybullet_fleet.collision_store import (
    MODE_DISABLED,
    MODE_NORMAL_2D,
    MODE_NORMAL_3D,
    MODE_STATIC,
    CollisionStore,
)
from pybullet_fleet.collision_visualizer import CollisionVisualizer  # deprecated, unused
from pybullet_fleet.config_utils import load_yaml_config
from pybullet_fleet.data_monitor import DataMonitor
from pybullet_fleet.events import _TIER_CALLBACK, _TIER_HANDLER, _TIER_PLUGIN, EventBus, SimEvents
from pybullet_fleet.geometry import Pose
from pybullet_fleet.gui_commands import (
    GuiCommand,
    GuiCommandType,
    MonitorEntity,
    MonitorFrame,
    SelectedEntityMonitorState,
)
from pybullet_fleet.logging_utils import get_lazy_logger
from pybullet_fleet.sim_object import SimObject
from pybullet_fleet.agent import Agent
from pybullet_fleet.agent_manager import SimObjectManager
from pybullet_fleet.behavior_tree_core import BehaviorTree
from pybullet_fleet.types import CallbackPhase, SpatialHashCellSizeMode, CollisionDetectionMethod, CollisionMode
from pybullet_fleet._defaults import MONITOR_REFRESH_HEADLESS, MONITOR_REFRESH_WITH_WINDOW
from pybullet_fleet._defaults import SIMULATION as _SIM_D

if TYPE_CHECKING:
    from pybullet_fleet.sim_plugin import SimPlugin

_MODE_CODE = {
    CollisionMode.NORMAL_3D: MODE_NORMAL_3D,
    CollisionMode.NORMAL_2D: MODE_NORMAL_2D,
    CollisionMode.STATIC: MODE_STATIC,
    CollisionMode.DISABLED: MODE_DISABLED,
}

#: ``fn(obj_a, obj_b) -> bool for :meth:`MultiRobotSimulationCore.set_collision_ignore_filter`:
#: ``True`` when the two touching objects are meant to touch and should not count as a collision.
CollisionIgnoreFilter = Callable[[SimObject, SimObject], bool]

# Generic plugin type so register_plugin() returns the concrete subclass type.
_SimPluginT = TypeVar("_SimPluginT", bound="SimPlugin")

# Global log_level (default: 'info')
GLOBAL_LOG_LEVEL = "INFO"
if "PYBULLET_LOG_LEVEL" in os.environ:
    GLOBAL_LOG_LEVEL = os.environ["PYBULLET_LOG_LEVEL"].upper()


#: The package's own logger. Everything below configures this rather than the
#: root logger: importing a library must not decide how the application that
#: imported it formats or filters its logs. A program that wants
#: PyBulletFleet's output on screen and has configured nothing of its own gets
#: it from the handler installed here; one that has called basicConfig() keeps
#: its own, because the handler is only added when this logger has none.
def _install_default_handler(package_logger: logging.Logger, root_logger: logging.Logger) -> bool:
    """Give the package a handler only when nothing else is handling records.

    Records propagate to the root logger, so installing one here while the
    host application has its own would print every PyBulletFleet line twice.
    A program that has configured nothing still gets the output it used to get
    from ``basicConfig()``.

    Returns whether a handler was added, which is what makes this testable
    without re-importing the package.
    """
    if package_logger.handlers or root_logger.handlers:
        return False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    package_logger.addHandler(handler)
    # Propagation is left on. Turning it off would stop the one case this
    # cannot catch -- a host that imports PyBulletFleet before configuring
    # logging, then adds a root handler, and sees each line twice -- but it
    # also cuts these records off from every handler the host adds later,
    # pytest's caplog among them. Measured: 17 tests in this suite capture
    # PyBulletFleet's own records through the root logger. The duplicate is
    # cosmetic and the host can remove this handler; losing the records is
    # not recoverable from the outside.
    return True


_PACKAGE_LOGGER = logging.getLogger(__name__.split(".")[0])
_PACKAGE_LOGGER.setLevel(logging.getLevelName(GLOBAL_LOG_LEVEL))
_install_default_handler(_PACKAGE_LOGGER, logging.getLogger())

# Create module logger (LazyLogger: avoids expensive f-string evaluation when log level is disabled)
logger = get_lazy_logger(__name__)

# --- run_simulation real-time-loop diagnostics ----------------------------
# (The sleep clamp is configurable via SimulationParams.max_sleep_frames.)
# Wall-vs-monotonic divergence (seconds) worth flagging as a clock jump. WARNING
# log only; does not affect pacing.
_CLOCK_JUMP_LOG_THRESHOLD_S = 0.5

# Real seconds between consecutive run_simulation() loop iterations worth
# flagging as a stall (host overloaded, GUI event processing stuck, etc.).
# WARNING log only. Unlike the clock-jump diagnostic above, this fires in
# every pacing mode including target_rtf<=0 (run-as-fast-as-possible), which
# otherwise has no loop-timing instrumentation at all.
_STEP_STALL_LOG_THRESHOLD_S = 2.0

# PyBullet does not expose these mouse constants in every wheel build.  Keep
# them aligned with CameraController's documented event tuple values.
_MOUSE_BUTTON_EVENT = 2
_MOUSE_BUTTON_LEFT = 0
_MOUSE_MOVE_EVENT = 1
_SELECTED_NAMEPLATE_UPDATE_INTERVAL_S = 0.25
_VIEWPORT_CLICK_MAX_DISTANCE_PX = 8

#: A collision clearance: one distance for every axis, or one per world axis (x, y, z).
CollisionMargin = Union[float, Tuple[float, float, float]]


def _is_scalar_margin(value: Any) -> bool:
    """A real number (numpy scalars included), not a bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _normalize_margin(value: Any) -> CollisionMargin:
    """A ``collision_margin`` as a validated float, or as a validated per-axis tuple of floats."""
    if _is_scalar_margin(value):
        margin = float(value)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError(f"collision_margin must be finite and non-negative, got {value!r}")
        return margin
    return _clearance_triple(value)


def _clearance_triple(value: Iterable[float]) -> Tuple[float, float, float]:
    """Validate a per-axis ``collision_margin`` and return it as a tuple of floats."""
    shape_error = ValueError(f"collision_margin must be a number or an (x, y, z) triple, got {value!r}")
    # A string is iterable ("005" would read as 0, 0, 5) and a bool is a number; neither is a clearance.
    if isinstance(value, (str, bytes, bool)):
        raise shape_error
    try:
        items = list(value)
    except TypeError:
        raise shape_error from None
    if len(items) != 3 or not all(_is_scalar_margin(v) for v in items):
        raise shape_error
    triple = tuple(float(v) for v in items)
    if not all(math.isfinite(v) and v >= 0 for v in triple):
        raise ValueError(f"collision_margin components must be finite and non-negative, got {value!r}")
    return cast(Tuple[float, float, float], triple)


@dataclass(frozen=True)
class CollisionPairObservation:
    """Last narrow-phase measurement for an active object pair."""

    object_ids: Tuple[int, int]
    detection_method: CollisionDetectionMethod
    signed_distance: Optional[float]
    sample_step: int
    sample_time: float


@dataclass(frozen=True)
class CollisionObservation:
    """Active pairs as of a completed collision check, not an event history."""

    step: int
    sim_time: float
    configured_method: CollisionDetectionMethod
    margin: CollisionMargin
    pairs: Tuple[CollisionPairObservation, ...]


@dataclass
class SimulationParams:
    """Simulation configuration parameters.

    All fields have sensible defaults for headless kinematics-based simulation.
    Use ``from_config`` / ``from_dict`` to load from YAML files.
    """

    target_rtf: float = _SIM_D["target_rtf"]  # target_rtf=0 means maximum speed (no sleep)
    timestep: float = _SIM_D["timestep"]
    duration: float = _SIM_D["duration"]
    gui: bool = _SIM_D["gui"]
    physics: bool = _SIM_D["physics"]
    #: When PyBullet receives a kinematic body's pose. ``"lazy"`` (the
    #: default): only when something reads it there, which the simulation
    #: does itself for a collision pair, an AABB query, an accessor such as
    #: ``SimObject.body_id`` or ``sim.client``, a recording or a GUI.
    #: ``"always"``: after every step, for code that calls PyBullet with a
    #: body id it keeps. See :meth:`MultiRobotSimulationCore.sync_mirror`.
    mirror: str = _SIM_D["mirror"]
    monitor: bool = _SIM_D["monitor"]
    #: Whether the DataMonitor opens a window. ``None`` (the default) follows
    #: ``gui``: a headless simulation collects monitor data without putting a
    #: Tk window on screen. ``True`` or ``False`` is honoured as given, so a
    #: monitor window without the PyBullet viewer is still available.
    enable_monitor_gui: Optional[bool] = _SIM_D["enable_monitor_gui"]
    #: Simulated seconds between DataMonitor refreshes while ``monitor`` is
    #: on. A refresh builds :attr:`last_monitor_frame` and writes the file the
    #: monitor window reads (every 0.5 wall seconds). ``None`` (the default)
    #: is 0.25 s while the monitor window is open (``enable_monitor_gui``, or
    #: ``gui`` when that is ``None``) and 1 s otherwise; ``0`` is every step.
    #: Simulated time: at a real-time factor above 1 a short interval means
    #: more writes per wall second. While paused, a selection, follow or pause
    #: change, and a single step, refresh at once. Read once, when the
    #: simulation is created.
    monitor_update_interval: Optional[float] = _SIM_D["monitor_update_interval"]
    #: Wall-clock seconds between ``[MONITOR]`` lines logged at INFO -- RTF,
    #: collision totals, step count -- independent of ``monitor``, ``gui`` and
    #: ``monitor_update_interval``. ``monitor`` drives the DataMonitor, whose
    #: line is logged at DEBUG; this keeps those figures in the log whether or
    #: not anything is on screen. Default 1 s; ``0`` logs none. Wall clock
    #: rather than simulated time, so the log's volume does not scale with the
    #: real-time factor.
    monitor_log_interval: float = _SIM_D["monitor_log_interval"]
    collision_check_frequency: Optional[float] = None
    log_level: str = _SIM_D["log_level"]
    max_steps_per_frame: int = _SIM_D["max_steps_per_frame"]
    # Real-time pacing: when the loop runs ahead of schedule, never sleep more
    # than this many "normal" frame intervals (max(min_sleep, timestep/target_rtf)).
    # Bounds a residual anomaly so the sim cannot freeze; raise for slower-than-1
    # rtf if needed. The catch-up counterpart is max_steps_per_frame.
    max_sleep_frames: float = _SIM_D["max_sleep_frames"]
    gui_min_fps: int = _SIM_D["gui_min_fps"]
    # Visualizer settings
    enable_collision_shapes: bool = _SIM_D["enable_collision_shapes"]
    enable_structure_transparency: bool = _SIM_D["enable_structure_transparency"]
    enable_shadows: bool = _SIM_D["enable_shadows"]
    enable_gui_panel: bool = _SIM_D["enable_gui_panel"]
    ignore_static_collision: bool = _SIM_D["ignore_static_collision"]
    #: Do not count contact between objects on one attachment tree
    #: (``attach_object``): a carrier, its loads and their loads touch one
    #: another by design. Contact with anything outside the tree is reported.
    ignore_attached_collision: bool = _SIM_D["ignore_attached_collision"]
    enable_time_profiling: bool = _SIM_D["enable_time_profiling"]
    profiling_interval: int = _SIM_D["profiling_interval"]
    enable_memory_profiling: bool = _SIM_D["enable_memory_profiling"]
    enable_collision_color_change: bool = _SIM_D["enable_collision_color_change"]
    spatial_hash_cell_size_mode: SpatialHashCellSizeMode = SpatialHashCellSizeMode(_SIM_D["spatial_hash_cell_size_mode"])
    spatial_hash_cell_size: Optional[float] = None  # Fixed cell size (for mode=CONSTANT)
    collision_detection_method: Optional[CollisionDetectionMethod] = None  # Auto-select based on physics
    #: Clearance, in metres, within which CLOSEST_POINTS (and the kinematic
    #: pairs of HYBRID) report a pair that does not touch; CONTACT_POINTS
    #: ignores it. A number applies to every direction; an ``(x, y, z)``
    #: triple (a list in YAML) gives one clearance per world axis, for
    #: example ``(0.05, 0.05, 0.0)`` to report bodies side by side but not one
    #: passing close above or below another. A triple acts in the broad phase:
    #: each AABB axis is enlarged by that axis's clearance, so two bodies whose
    #: AABBs are farther apart than that on any axis are never reported. The
    #: narrow phase judges the straight-line gap against the largest
    #: component, so ``(m, m, m)`` is the same as ``m``, and ``(m, m, 0)``
    #: reports what ``CollisionMode.NORMAL_2D`` with ``m`` did, for every pair.
    collision_margin: CollisionMargin = _SIM_D["collision_margin"]
    multi_cell_threshold: float = _SIM_D["multi_cell_threshold"]  # Accepted for compatibility; not used
    enable_floor: bool = _SIM_D["enable_floor"]
    camera_config: Optional[Dict[str, Any]] = None  # Camera configuration from config file
    lighting_config: Optional[Dict[str, Any]] = None  # PyBullet GUI main-light configuration from config file
    model_paths: Optional[List[str]] = None  # Additional directories to scan for URDF/mesh models
    # Window size parameters
    window_width: int = _SIM_D["window_width"]
    window_height: int = _SIM_D["window_height"]
    # DataMonitor window parameters
    monitor_width: int = _SIM_D["monitor_width"]
    monitor_height: int = _SIM_D["monitor_height"]
    monitor_x: int = _SIM_D["monitor_x"]
    monitor_y: int = _SIM_D["monitor_y"]

    def __post_init__(self) -> None:
        if self.multi_cell_threshold != _SIM_D["multi_cell_threshold"]:
            warnings.warn(
                "multi_cell_threshold no longer has any effect: every body is registered in each "
                "spatial-hash cell it reaches. It will be removed in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )
        # A per-axis clearance (a list from YAML) is stored as a validated tuple.
        self.collision_margin = _normalize_margin(self.collision_margin)
        if self.mirror not in ("lazy", "always"):
            raise ValueError(f"mirror must be 'lazy' or 'always', got {self.mirror!r}")
        # Coerce a string (e.g. from YAML: collision_detection_method: "closest_points")
        # into the CollisionDetectionMethod enum. Without this the value stays a plain
        # str and the `== CollisionDetectionMethod.X` checks in check_collisions() never
        # match, silently disabling all collision detection.
        if isinstance(self.collision_detection_method, str):
            try:
                self.collision_detection_method = CollisionDetectionMethod(self.collision_detection_method)
            except ValueError:
                logger.warning(
                    "Unknown collision_detection_method %r; falling back to physics-based default. Valid values: %s",
                    self.collision_detection_method,
                    [m.value for m in CollisionDetectionMethod],
                )
                self.collision_detection_method = None
        # Auto-select collision detection method based on physics mode if not explicitly set
        if self.collision_detection_method is None:
            # Physics ON: use CONTACT_POINTS (actual contact manifold)
            # Physics OFF: use CLOSEST_POINTS (distance-based, kinematics-safe)
            self.collision_detection_method = (
                CollisionDetectionMethod.CONTACT_POINTS if self.physics else CollisionDetectionMethod.CLOSEST_POINTS
            )
        # Ensure camera_config is always a dict (avoid mutable default)
        if self.camera_config is None:
            self.camera_config = {}
        # Ensure lighting_config is always a dict (avoid mutable default)
        if self.lighting_config is None:
            self.lighting_config = {}
        # Ensure model_paths is always a list (avoid mutable default)
        if self.model_paths is None:
            self.model_paths = []

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "SimulationParams":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Support nested layout: {simulation: {...}, world: {...}, robots: [...]}
        sim_config = config.get("simulation", config) if isinstance(config, dict) else config
        return cls.from_dict(sim_config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SimulationParams":
        """
        Create SimulationParams from a configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            SimulationParams instance
        """
        from pybullet_fleet.config_utils import dataclass_from_dict

        return dataclass_from_dict(
            cls,
            config,
            aliases={"camera_config": "camera", "lighting_config": "lighting"},
        )


#: Tiers of the step events emitted by ``step_once()`` and the time-profiling
#: key of each (``None``: counted only in ``phase1_update``).
_PRE_STEP_EVENT_TIERS = (
    (_TIER_HANDLER, "events_pre_step"),
    (_TIER_CALLBACK, "callbacks_pre_step"),
    (_TIER_PLUGIN, None),
)
_STEP_EVENT_TIERS = ((_TIER_HANDLER, "events_step"), (_TIER_CALLBACK, "callbacks"), (_TIER_PLUGIN, None))
_POST_STEP_EVENT_TIERS = (
    (_TIER_HANDLER, "events_post_step"),
    (_TIER_CALLBACK, "callbacks_post_step"),
    (_TIER_PLUGIN, None),
)


def _weak_hook_clock(sim_ref: "weakref.ref[MultiRobotSimulationCore]") -> Callable[[], Tuple[float, float]]:
    """The ``events`` clock of the simulation ``sim_ref`` refers to, without a strong reference to it."""

    def clock() -> Tuple[float, float]:
        return sim_ref()._hook_clock()  # type: ignore[union-attr]  # the bus lives no longer than its simulation

    return clock


class MultiRobotSimulationCore:
    _EMPTY_SET: frozenset = frozenset()  # Reusable empty set for dict.get() default (avoids per-call allocation)

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "MultiRobotSimulationCore":
        """Create and initialise a simulation from a YAML config file.

        Supports two YAML layouts:

        **Nested** (recommended) — ``simulation``, ``world``, and
        ``entities`` sections::

            simulation:
              gui: true
              physics: false
            world:
              world_file: "/path/to/office.world"
            entities:
              - name: robot0
                urdf_path: "robots/mobile_robot.urdf"
                pose: [0, 0, 0.05]
              - type: sim_object
                name: box0
                visual_shape:
                  shape_type: box
                  half_extents: [0.3, 0.3, 0.1]
                pose: [3, 0, 0.1]

        **Flat** (legacy) — all keys at the top level::

            gui: true
            physics: false
            timestep: 0.1

        **Custom entity types** — register classes via dotted Python
        paths so ``type:`` dispatch works without any Python code::

            entity_classes:
              forklift: "my_ros_pkg.entities.ForkliftAgent"
              conveyor: "warehouse_sim.conveyor.ConveyorObject"

            entities:
              - type: forklift
                urdf_path: robots/forklift.urdf
                pose: [0, 0, 0.05]

        When a ``world`` section is present and ``enable_floor`` is not
        explicitly set in the simulation config, the floor plane is
        automatically disabled (the world geometry replaces it).

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            Fully initialised ``MultiRobotSimulationCore`` with world
            loaded and robots spawned (if configured).
        """
        config = load_yaml_config(yaml_path)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MultiRobotSimulationCore":
        """Create MultiRobotSimulationCore from a configuration dictionary.

        Supports both flat simulation-params dicts and nested dicts with
        ``simulation``, ``world``, and ``entities`` sections.
        When nested keys are present they are processed the same way
        as :meth:`from_yaml`.

        An optional ``entity_classes`` section maps names to dotted
        Python paths for dynamic class registration (see
        :meth:`from_yaml` for details).

        Args:
            config: Configuration dictionary.

        Returns:
            MultiRobotSimulationCore instance.
        """
        # Register custom entity classes before spawning
        entity_classes = config.get("entity_classes")
        if entity_classes:
            from pybullet_fleet.entity_registry import register_entity_classes_from_config

            register_entity_classes_from_config(entity_classes)

        # Extract each section independently
        sim_config = config.get("simulation")
        world_config = config.get("world", {})
        entities_config = list(config.get("entities", []))

        # Flat layout (legacy): no nested keys, entire config is sim params
        if sim_config is None and not world_config and not entities_config:
            params = SimulationParams.from_dict(config)
            return cls(params)

        sim_config = sim_config or {}

        if world_config and "enable_floor" not in sim_config:
            sim_config = dict(sim_config)
            sim_config["enable_floor"] = False

        params = SimulationParams.from_dict(sim_config)
        sim = cls(params)

        # managers: must be processed before entities: so that entity groups
        # can reference them by name via the ``manager:`` key.
        managers_config = list(config.get("managers", []))
        if managers_config:
            sim._init_named_managers(managers_config)

        if world_config:
            entity_names = [e.get("name", "") for e in entities_config if e.get("name")]
            skip = list(set(world_config.get("skip_models", []) + entity_names))
            sim.load_world(world_config, skip_models=skip)
        if entities_config:
            sim.spawn_robots_from_config(entities_config)

        # Load and initialise plugins
        sim._load_and_init_plugins(config)

        return sim

    def __init__(self, params: SimulationParams, collision_color: Optional[List[float]] = None) -> None:
        # Initialize log level from params, on this package's logger only.
        # Setting it on the root logger silenced the host application's own
        # output: log_level defaults to "warn", so merely constructing a
        # simulation turned off every INFO line in the process.
        level_str = getattr(params, "log_level", GLOBAL_LOG_LEVEL)
        level_name = str(level_str).upper()
        _PACKAGE_LOGGER.setLevel(logging.getLevelName(level_name))

        # --- Public (read-only via @property, internal via self._X) ---
        self._client: Optional[int] = None
        #: Set by close(), so its shutdown callbacks run at most once.
        self._closed: bool = False
        self._sim_objects: List[SimObject] = []  # List of all simulation objects (Agent, SimObject, etc.)
        self._sim_objects_dict: Dict[int, SimObject] = {}  # Dict for O(1) lookup: object_id -> SimObject
        self._agents: List[Agent] = []  # List of Agent instances only for O(M) iteration in step_once()
        self._robot_original_colors: Dict[int, List[float]] = {}  # body_id: rgbaColor
        self._collision_count: int = 0
        self._step_count: int = 0
        # The simulation clock: the one source of time; ``sim_time`` reads it.
        self._elapsed_sim_time: float = 0.0
        # The step, time and timestep _schedule_time() counts whole timesteps from.
        self._schedule_origin: Tuple[int, float, float] = (0, 0.0, params.timestep)
        self._enable_time_profiling: bool = params.enable_time_profiling

        # --- Fully private (no external access) ---
        self._start_time: Optional[float] = None
        self._next_object_id: int = 0  # Counter for unique object IDs (read/incremented by SimObject.__init__)
        self._last_logged_collision_count: int = 0  # Track last logged collision count
        self._monitor_enabled: bool = params.monitor
        self._last_monitor_log: Optional[float] = None  # time.monotonic() of the last INFO [MONITOR] line
        self._behavior_tree_callbacks: List[Tuple[BehaviorTree, Callable]] = []
        self._data_monitor: Optional[DataMonitor] = None
        self._gui_commands: Queue[GuiCommand] = Queue(maxsize=32)
        self._single_step_requested: bool = False
        # Entity-list cache and "window is behind" state of the DataMonitor
        # (see pybullet_fleet._monitor_frames).
        self._monitor_frames = MonitorFrames()
        self._selected_entity_id: Optional[int] = None
        self._gui_follow_enabled: bool = False
        self._pacing_rebase_requested: bool = False
        self._selected_nameplate_id: Optional[int] = None
        self._last_selected_nameplate_update: float = 0.0
        self._viewport_left_click_start: Optional[Tuple[int, int]] = None
        self._viewport_drag_entity_id: Optional[int] = None
        self._collision_check_frequency: Optional[float] = params.collision_check_frequency  # If None, check every step
        self._collision_check_gate = FrequencyGate()  # follows _collision_check_frequency while enabled
        self._log_level: str = params.log_level
        self._params: SimulationParams = params
        self._collision_color: List[float] = collision_color if collision_color is not None else [0, 0, 1, 1]
        self._rendering_enabled: bool = False  # Track rendering state
        self._batch_spawning: bool = False  # True inside batch_spawn() context
        self._collision_visualizer: CollisionVisualizer = CollisionVisualizer()  # Collision shape visualizer

        self._keyboard_events_registered: bool = False  # Track if keyboard events are registered
        self._camera_controller: Optional[CameraController] = None  # Interactive camera control
        self._camera_configured: bool = False  # True after explicit setup_camera() — prevents configure_visualizer overwrite
        self._lighting_control_ids: Optional[Dict[str, int]] = None
        self._lighting_control_values: Optional[Tuple[float, ...]] = None
        self._original_visual_colors: Dict[Tuple[int, int], List[float]] = (
            {}
        )  # Store original colors: (body_id, link_index) -> rgba
        self._structure_transparent: bool = False  # Track if structure is transparent
        self._ignore_static_collision: bool = (
            params.ignore_static_collision
        )  # If True, skip collision checks with static objects
        self._simulation_paused: bool = False  # Track if simulation is paused

        # --- Profiling configuration ---
        self._profiling_interval: int = params.profiling_interval  # Report average every N steps
        self._enable_memory_profiling: bool = params.enable_memory_profiling  # Memory profiling switch

        # --- Profiling statistics ---
        # Two-phase step keys (spec.md):
        #   phase1_update          — agent.update + step hooks: step event, callbacks, plugin on_step (buffered writes)
        #   phase2_pose_flush      — _flush_pending_poses: PyBullet pose writes
        #   phase3_aabb_grid_flush — _flush_aabb_and_grid: kinematic AABB + spatial-grid refresh
        # `agent_update` is kept for backward compatibility (legacy per-object update loop only).
        self._profiling_stats: Dict[str, List[float]] = {
            "agent_update": [],
            "phase1_update": [],
            "phase2_pose_flush": [],
            "phase3_aabb_grid_flush": [],
            "events_step": [],
            "callbacks": [],
            "step_simulation": [],
            "collision_check": [],
            "events_pre_step": [],
            "callbacks_pre_step": [],
            "events_post_step": [],
            "callbacks_post_step": [],
            "events_collision": [],
            "monitor_update": [],
            "total": [],
        }
        # Snapshot of the most recently completed profiled step.  This is
        # intentionally separate from _profiling_stats, which is an internal
        # interval accumulator and may be cleared after a periodic log report.
        self._last_profiling: Optional[Mapping[str, Any]] = None

        # --- EventBus ---
        # The one hook mechanism of a step: event handlers, register_callback()
        # callbacks and plugin on_step() all subscribe here, and frequency-limited
        # ones are scheduled on the simulation clock.
        # The bus holds the simulation weakly (clock, callbacks), so a dropped
        # simulation is still freed by reference counting.
        self._weak_self = weakref.ref(self)
        self.events: EventBus = EventBus(clock=_weak_hook_clock(self._weak_self))
        self._prev_collision_pairs: Set[Tuple[int, int]] = set()

        # --- Memory profiling statistics (O(1) memory, aggregator-based) ---
        self._memory_stats: Dict[str, Union[int, float]] = {
            "count": 0,  # Number of samples
            "current_sum": 0.0,  # Sum of current memory (for average)
            "current_min": float("inf"),  # Minimum current memory
            "current_max": 0.0,  # Maximum current memory
            "current_first": 0.0,  # First sample (for growth calculation)
            "current_last": 0.0,  # Last sample (for growth calculation)
            "peak_sum": 0.0,  # Sum of peak memory (for average)
            "peak_max": 0.0,  # Maximum peak memory
        }
        self._memory_tracemalloc_started: bool = False  # Track if WE started tracemalloc

        # --- Monitor speed history (deque for O(1) popleft) ---
        self._speed_history: deque = deque()  # [(real_time, sim_time)]

        # --- Movement tracking and collision optimization ---
        self._moved_this_step: set = set()  # object_ids that moved this step
        self._physics_objects: set = set()  # object_ids with physics enabled (can move via collisions)
        self._kinematic_objects: set = set()  # object_ids that move only via set_pose/update
        self._static_collision_objects: set = set()  # object_ids with CollisionMode.STATIC
        self._disabled_collision_objects: set = set()  # object_ids with collision completely disabled

        # --- Collision detection cache ---
        self._cached_collision_modes: Dict[int, CollisionMode] = {}  # object_id -> CollisionMode
        self._cached_cell_size: Optional[float] = None  # Cached cell size for spatial hashing
        self._dynamic_collision_objects: set = set()  # object_ids with NORMAL_3D or NORMAL_2D (excludes STATIC and DISABLED)

        # Every object's collision columns (AABB, local box, mode, physics)
        # and the spatial hash over them. Rows are the store's; everything
        # here refers to objects by object_id.
        self._store = CollisionStore()
        # The links (-1 = base) whose boxes make up each body's AABB: those
        # with a collision shape, read once from PyBullet. Empty for a body
        # without child links (or without any collider): its AABB is one
        # getAABB() call. An articulated body's is the union over these links.
        self._aabb_links: Dict[int, Tuple[int, ...]] = {}
        # Pickable objects whose base origin lies outside their cached AABB,
        # with the largest per-axis distance from the origin to the box. Lets
        # a search by base origin (pickable_candidates_near) go through the
        # spatial hash, which holds AABBs, without missing a shape set far off
        # its origin. Only pickable objects: nothing else is searched by origin,
        # and the AABB update of every moved object stays free of it.
        self._origin_gaps: Dict[int, float] = {}
        # The broadphase margin the grid's registrations were made with. A pair
        # within the margin is found only if both bodies were registered with
        # it, so the grid is rebuilt whenever the effective margin differs --
        # set_collision_margin() does so at once; a change made on ``params``
        # directly is caught by filter_aabb_pairs().
        self._grid_margin: Optional[Tuple[float, float, float]] = None
        # Half of _grid_margin per axis: what each body's AABB is enlarged by.
        self._grid_half_margin: Optional[Tuple[float, float, float]] = None
        self._aabb_cache_valid: bool = False  # Flag indicating if AABB cache is valid
        self._last_ignore_static: Optional[bool] = None  # Track last ignore_static value to detect mode changes
        self._last_ignore_attached: Optional[bool] = None  # the same for ignore_attached_collision

        # Incremental collision tracking
        self._active_collision_pairs: Set[Tuple[int, int]] = set()  # Currently colliding object_id pairs (sorted: i < j)
        self._active_pairs_by_object: Dict[int, Set[Tuple[int, int]]] = {}
        self._collision_pair_observations: Dict[Tuple[int, int], CollisionPairObservation] = {}
        self._last_collision_check_metadata: Optional[Tuple[int, float, CollisionDetectionMethod, CollisionMargin]] = None
        self._recheck_stationary_pairs = False
        self._collision_ignore_filter: Optional[CollisionIgnoreFilter] = None
        # Broad-phase candidate pairs the ignore rules excluded last time they
        # were checked. Their narrow phase was skipped, so whether they touch
        # is unknown; they are re-evaluated on every check, so a pair that
        # stops being ignored without either body moving (an in-place detach,
        # a filter reading run-time state) is still found.
        self._ignored_collision_pairs: Set[Tuple[int, int]] = set()

        # --- Recording ---
        self._recorder: Optional[Any] = None  # Optional[SimulationRecorder] (lazy import to avoid circular)
        self._state_recorder: Optional[Any] = None  # Supported execution-state recorder

        # --- Registered managers (optional overlays for batch API) ---
        self._registered_managers: List[SimObjectManager] = []

        # --- Two-phase step state ---
        # _in_step is True while step_once() is executing. SimObject._set_pose_internal
        # checks this flag to decide between buffered (Phase 2 flush) and immediate writes.
        # _pending_pose_ids holds object_ids whose _pending_pose buffer is non-empty.
        # See docs/design/two-phase-step/spec.md.
        self._in_step: bool = False
        self._in_post_step: bool = False
        # True only while PRE_STEP is emitted: the one point of a step where the
        # clock and step counter still hold the previous step (see step_once()).
        self._in_pre_step: bool = False
        self._resume_ready: bool = False  # Set only by a validated completed-step restore.
        self._pending_pose_ids: Set[int] = set()
        # Kinematic objects whose links moved inside the step without a pose
        # write (a joint): their AABB and grid cells are refreshed in Phase 3
        # with the flushed poses. See notify_geometry_changed().
        self._pending_geometry_ids: Set[int] = set()
        # Parent object ids that have link-level kinematic attachments.
        # Used to avoid scanning every flushed object in the hot path when no
        # link attachments exist.
        self._link_attachment_parent_ids: Set[int] = set()
        # Base-link attachments: carrier object_id -> the object_ids riding on
        # its base. Their poses follow the carrier's as arrays in set_poses().
        self._base_children: Dict[int, List[int]] = {}
        self._base_parent_of: Dict[int, int] = {}  # rider object_id -> carrier object_id
        # Plugins and callbacks that asked for every pose in PyBullet after each step.
        self._mirror_requests: int = 0
        # The objects with a per-step update() (agents, devices): the step
        # visits these, not every body in the scene.
        self._updatable: List[SimObject] = []
        # The objects that left _updatable since the update loop began: the
        # loop runs over a snapshot and skips these (an update() may remove
        # another object, or clear its _needs_update).
        self._updatable_left: Set[SimObject] = set()
        #: Bumped when an object is added, removed or renamed: readers that
        #: cache the fleet's rows and names (the state plane) compare it.
        self._fleet_revision: int = 0
        # One batch per agent-plugin class that offers one (None when it
        # does not): it updates all of that class's plugins in one call.
        self._plugin_batches: Dict[type, Any] = {}

        # --- Plugins ---
        self._plugins: List["SimPlugin"] = []

        self.setup_pybullet()
        self.setup_monitor()
        # DataMonitor refresh: monitor_update_interval, else
        # MONITOR_REFRESH_WITH_WINDOW while its window is open
        # (enable_monitor_gui, or gui when that is None; the window reads every
        # 0.5 wall seconds) and MONITOR_REFRESH_HEADLESS otherwise.
        # Set after setup_pybullet(), as RECORD may turn the GUI off, and
        # after setup_monitor(), as the window stays shut without Tk.
        interval = params.monitor_update_interval
        if interval is None:
            window = self._data_monitor is not None and self._data_monitor.enable_gui
            interval = MONITOR_REFRESH_WITH_WINDOW if window else MONITOR_REFRESH_HEADLESS
        self._monitor_update_gate = FrequencyGate(1.0 / interval if interval > 0 else None)

    def setup_monitor(self) -> None:
        # If monitor: true and console_monitor: false, start DataMonitor

        if self._params.monitor:
            # None means "follow gui": a headless run should not open a
            # window. It used to open one regardless, so a program that
            # builds several simulations -- a test suite, a sweep -- got a Tk
            # window per simulation on a machine with no viewer in sight.
            enable_gui = self._params.enable_monitor_gui
            if enable_gui is None:
                enable_gui = self._params.gui
            self._data_monitor = DataMonitor(
                "PyBullet Simulation Monitor",
                enable_gui=enable_gui,
                width=self._params.monitor_width,
                height=self._params.monitor_height,
                x=self._params.monitor_x,
                y=self._params.monitor_y,
                initial_target_rtf=self._params.target_rtf,
                initial_timestep=self._params.timestep,
            )
            self._data_monitor.set_command_sink(self.enqueue_gui_command)
            self._data_monitor.start()
        else:
            self._data_monitor = None

    # --- Read-only public properties ---

    @property
    def client(self) -> Optional[int]:
        """PyBullet physics client ID, with every pending pose written to PyBullet first.

        Code that calls PyBullet through this id sees the current poses.
        Internally the simulation uses the id without this sync.
        """
        self.sync_mirror()
        return self._client

    @property
    def sim_objects(self) -> List[SimObject]:
        """List of all simulation objects (Agent, SimObject, etc.)."""
        return self._sim_objects

    @property
    def last_monitor_frame(self) -> Optional[MonitorFrame]:
        """Return the most recently published toolkit-neutral monitor frame."""
        return self._monitor_frames.last_frame

    @property
    def selected_entity_id(self) -> Optional[int]:
        """Return the entity selected by the monitor, if any."""
        return self._selected_entity_id

    @property
    def gui_follow_enabled(self) -> bool:
        """Return whether the viewport follows the selected entity."""
        return self._gui_follow_enabled

    def enqueue_gui_command(self, command: GuiCommand) -> bool:
        """Queue a monitor request for the next simulation-step boundary.

        This method is safe to call from a tkinter callback.  It deliberately
        does not mutate simulation state or make PyBullet calls on the caller's
        thread.  Returns ``False`` only when the bounded queue is full.
        """
        if not isinstance(command, GuiCommand):
            raise TypeError("command must be a GuiCommand")
        try:
            self._gui_commands.put_nowait(command)
        except Full:
            logger.warning("Dropping GUI command because the monitor command queue is full")
            return False
        return True

    def request_single_step(self, source: str = "api") -> bool:
        """Request one step while paused, without re-entering :meth:`step_once`.

        The request is consumed by the outer ``step_once`` invocation.  A
        request made while running is ignored because normal stepping already
        advances the simulation.
        """
        if not self.is_paused:
            return False
        self._single_step_requested = True
        self.events.emit("gui.single_step_requested", source=source)
        return True

    def _consume_gui_commands(self) -> None:
        """Apply queued monitor requests on the simulation thread."""
        while True:
            try:
                command = self._gui_commands.get_nowait()
            except Empty:
                return
            if command.command is GuiCommandType.PAUSE:
                if not self.is_paused:
                    self.pause()
            elif command.command is GuiCommandType.RESUME:
                if self.is_paused:
                    self.resume()
            elif command.command is GuiCommandType.SINGLE_STEP:
                self.request_single_step(source=command.source)
            elif command.command is GuiCommandType.SELECT_ENTITY:
                self._set_selected_entity(command.entity_id, source=command.source)
            elif command.command is GuiCommandType.SET_FOLLOW:
                self._set_gui_follow(bool(command.enabled))
            elif command.command is GuiCommandType.SET_SIMULATION_PACING:
                self._set_simulation_pacing(command.target_rtf, command.timestep, source=command.source)
            elif command.command is GuiCommandType.SET_ENTITY_POSE:
                self._set_entity_pose(
                    command.entity_id,
                    command.position,
                    command.yaw_radians,
                    command.rpy_radians,
                    source=command.source,
                )

    def _set_simulation_pacing(self, target_rtf: Optional[float], timestep: Optional[float], source: str) -> None:
        """Apply live pacing settings at a simulation-step boundary.

        A currently running simulation is paused internally while the physics
        timestep and scheduler settings are changed, then immediately resumed.
        The outer run loop observes ``_pacing_rebase_requested`` and discards
        its old wall-clock baseline before its next pacing decision.
        """
        if target_rtf is None and timestep is None:
            return
        invalid_values = []
        if target_rtf is not None and target_rtf < 0:
            invalid_values.append(f"target_rtf={target_rtf} (must be >= 0)")
        if timestep is not None and timestep <= 0:
            invalid_values.append(f"timestep={timestep} (must be > 0)")
        if invalid_values:
            logger.warning("Ignoring invalid GUI pacing change: %s", "; ".join(invalid_values))
            return

        was_running = not self.is_paused
        if was_running:
            self.pause()
        if target_rtf is not None:
            self._params.target_rtf = target_rtf
        if timestep is not None:
            self._params.timestep = timestep
            p.setTimeStep(timestep, physicsClientId=self._client)
        self._pacing_rebase_requested = True
        self.events.emit(
            "gui.simulation_pacing_changed",
            target_rtf=self._params.target_rtf,
            timestep=self._params.timestep,
            source=source,
        )
        if was_running:
            self.resume()

    def _set_entity_pose(
        self,
        entity_id: Optional[int],
        position: Optional[Tuple[float, float, float]],
        yaw_radians: Optional[float],
        rpy_radians: Optional[Tuple[float, float, float]],
        source: str,
    ) -> bool:
        """Teleport one entity while paused, preserving roll and pitch.

        GUI pose editing deliberately has paused-only semantics: it cannot race
        an agent controller or physics integration.  Dynamic bodies also have
        their velocity cleared so a pre-edit velocity cannot move them away
        immediately after resume.

        The monitor shows the new pose at once, but collision state is not
        recomputed: the paused world is stopped, so contact counts (the
        frame's ``active_collisions``) stay as of the last step until the
        simulation steps again.
        """
        if not self.is_paused:
            logger.warning("Ignoring GUI pose edit while simulation is running")
            return False
        if entity_id is None or entity_id not in self._sim_objects_dict:
            logger.warning("Ignoring GUI pose edit for missing entity %s", entity_id)
            return False
        if position is None or len(position) != 3 or not all(math.isfinite(value) for value in position):
            logger.warning("Ignoring invalid GUI position %s", position)
            return False
        if yaw_radians is not None and not math.isfinite(yaw_radians):
            logger.warning("Ignoring invalid GUI yaw %s", yaw_radians)
            return False
        if rpy_radians is not None and (len(rpy_radians) != 3 or not all(math.isfinite(value) for value in rpy_radians)):
            logger.warning("Ignoring invalid GUI roll/pitch/yaw %s", rpy_radians)
            return False

        entity = self._sim_objects_dict[entity_id]
        orientation = list(entity.get_pose().orientation)
        if rpy_radians is not None:
            orientation = list(p.getQuaternionFromEuler(rpy_radians))
        elif yaw_radians is not None:
            roll, pitch, _ = p.getEulerFromQuaternion(orientation)
            orientation = list(p.getQuaternionFromEuler([roll, pitch, yaw_radians]))
        moved = entity.set_pose_raw(list(position), orientation, preserve_velocity=False)
        if moved:
            # set_pose_raw() marked the monitor window behind.
            self.events.emit("gui.entity_pose_changed", entity_id=entity_id, source=source)
        return moved

    def _set_selected_entity(self, entity_id: Optional[int], source: str) -> None:
        if entity_id is not None and entity_id not in self._sim_objects_dict:
            logger.warning("Ignoring GUI selection for missing entity %s", entity_id)
            return
        if self._selected_entity_id == entity_id:
            return
        self._selected_entity_id = entity_id
        self._monitor_frames.view_changed()
        if entity_id is None:
            self._gui_follow_enabled = False
        self._update_selected_nameplate(force=True)
        self.events.emit("gui.selection_changed", entity_id=entity_id, source=source)

    def _set_gui_follow(self, enabled: bool) -> None:
        enabled = enabled and self._selected_entity_id is not None
        if self._gui_follow_enabled == enabled:
            return
        self._gui_follow_enabled = enabled
        self._monitor_frames.view_changed()
        self.events.emit("gui.follow_changed", entity_id=self._selected_entity_id, enabled=enabled)

    def _clear_selected_nameplate(self) -> None:
        """Remove the one viewport label owned by the current selection."""
        if self._selected_nameplate_id is None:
            return
        try:
            p.removeUserDebugItem(self._selected_nameplate_id, physicsClientId=self._client)
        except p.error:
            pass
        self._selected_nameplate_id = None
        self._last_selected_nameplate_update = 0.0

    def _update_selected_nameplate(self, force: bool = False) -> None:
        """Place one nameplate above the selected entity, when a GUI is active.

        Debug-text parenting to a base link is not consistent across PyBullet
        builds, so the label uses the entity's world pose.  Only the selected
        label is refreshed, at most four times per real-time second.  It
        remains visualizer state rather than a SimObject attribute, and
        therefore does not affect simulation state or snapshots.
        """
        if not self._params.gui or self._selected_entity_id is None:
            self._clear_selected_nameplate()
            return
        selected = self._sim_objects_dict.get(self._selected_entity_id)
        if selected is None or selected.body_id is None:
            self._clear_selected_nameplate()
            return
        now = time.monotonic()
        if not force and now - self._last_selected_nameplate_update < _SELECTED_NAMEPLATE_UPDATE_INTERVAL_S:
            return
        position = list(selected.get_pose().position)
        position[2] += 0.3
        try:
            self._selected_nameplate_id = p.addUserDebugText(
                selected.name or f"entity-{selected.object_id}",
                position,
                textColorRGB=[1, 1, 0],
                textSize=1.2,
                replaceItemUniqueId=self._selected_nameplate_id if self._selected_nameplate_id is not None else -1,
                physicsClientId=self._client,
            )
            self._last_selected_nameplate_update = now
        except p.error:
            self._selected_nameplate_id = None

    def _viewport_ray_at(self, mouse_x: int, mouse_y: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return world-space near/far points for a viewport coordinate."""
        camera = p.getDebugVisualizerCamera(physicsClientId=self._client)
        width, height = camera[0], camera[1]
        if width <= 0 or height <= 0:
            return None
        view = np.asarray(camera[2], dtype=float).reshape((4, 4), order="F")
        projection = np.asarray(camera[3], dtype=float).reshape((4, 4), order="F")
        inverse = np.linalg.inv(projection @ view)
        x = 2.0 * mouse_x / width - 1.0
        y = 1.0 - 2.0 * mouse_y / height
        near = inverse @ np.array([x, y, -1.0, 1.0])
        far = inverse @ np.array([x, y, 1.0, 1.0])
        near /= near[3]
        far /= far[3]
        return near[:3], far[:3]

    def _pick_entity_at(self, mouse_x: int, mouse_y: int) -> Optional[int]:
        """Return the entity under a viewport coordinate, if PyBullet reports one."""
        ray = self._viewport_ray_at(mouse_x, mouse_y)
        if ray is None:
            return None
        near, far = ray
        hit = p.rayTest(near[:3], far[:3], physicsClientId=self._client)[0]
        body_id = hit[0]
        for obj in self._sim_objects:
            if obj.body_id == body_id:
                return obj.object_id
        return None

    def _drag_selected_entity_to(self, mouse_x: int, mouse_y: int) -> None:
        """Move the selected paused entity to the mouse ray at its current Z."""
        if self._selected_entity_id is None:
            return
        entity = self._sim_objects_dict.get(self._selected_entity_id)
        if entity is None:
            return
        ray = self._viewport_ray_at(mouse_x, mouse_y)
        if ray is None:
            return
        near, far = ray
        direction = far - near
        if abs(direction[2]) < 1e-9:
            return
        current_pose = entity.get_pose()
        scale = (current_pose.position[2] - near[2]) / direction[2]
        if scale < 0:
            return
        point = near + scale * direction
        self._set_entity_pose(
            entity.object_id,
            (float(point[0]), float(point[1]), current_pose.position[2]),
            None,
            None,
            source="viewport_drag",
        )

    def _select_viewport_entity(self, mouse_events: List[Tuple[int, int, int, int, int]]) -> None:
        """Select an entity on a left click, while leaving left-drag for camera control."""
        for event_type, mouse_x, mouse_y, button, state in mouse_events:
            if event_type == _MOUSE_MOVE_EVENT and self._viewport_drag_entity_id is not None:
                self._drag_selected_entity_to(mouse_x, mouse_y)
                continue
            if event_type != _MOUSE_BUTTON_EVENT or button != _MOUSE_BUTTON_LEFT:
                continue
            if state != p.KEY_WAS_RELEASED:
                self._viewport_left_click_start = (mouse_x, mouse_y)
                continue

            if self._viewport_drag_entity_id is not None:
                self._viewport_drag_entity_id = None
                self._viewport_left_click_start = None
                continue

            start = self._viewport_left_click_start
            self._viewport_left_click_start = None
            if start is not None and max(abs(mouse_x - start[0]), abs(mouse_y - start[1])) > _VIEWPORT_CLICK_MAX_DISTANCE_PX:
                continue
            try:
                selected_id = self._pick_entity_at(mouse_x, mouse_y)
            except p.error:
                return
            self._set_selected_entity(selected_id, source="viewport")

    @property
    def agents(self) -> List[Agent]:
        """List of Agent instances."""
        return self._agents

    def find_objects_by_name(self, name: str) -> List[SimObject]:
        """Return all registered simulation objects whose name matches.

        Names are not required to be unique. The result follows registration
        order and includes Agents, which are also SimObjects.
        """
        return [obj for obj in self._sim_objects if obj.name == name]

    def find_agents_by_name(self, name: str) -> List[Agent]:
        """Return matching Agents in registration order."""
        return [agent for agent in self._agents if agent.name == name]

    def get_unique_object_by_name(self, name: str) -> SimObject:
        """Return one named object, or raise if absent or ambiguous."""
        matches = self.find_objects_by_name(name)
        if len(matches) != 1:
            raise LookupError(f"Expected one simulation object named {name!r}; found {len(matches)}")
        return matches[0]

    def get_unique_agent_by_name(self, name: str) -> Agent:
        """Return one named Agent, or raise if absent or ambiguous."""
        matches = self.find_agents_by_name(name)
        if len(matches) != 1:
            raise LookupError(f"Expected one Agent named {name!r}; found {len(matches)}")
        return matches[0]

    @property
    def params(self) -> SimulationParams:
        """Simulation parameters."""
        return self._params

    @property
    def step_count(self) -> int:
        """Current simulation step count."""
        return self._step_count

    @property
    def sim_time(self) -> float:
        """Simulated time of the state you see (read-only).

        Outside a step it is the completed step's time, the same as
        :meth:`get_completed_step_boundary`. Inside :meth:`step_once` the clock
        advances right after the ``pre_step`` event, so from the object update
        on -- callbacks, plugins, the collision check, ``post_step`` -- it is the
        time of the state that step computes. A ``pre_step`` subscriber sees the
        previous step's time. To move the clock (resuming from a checkpoint),
        use :meth:`restore_completed_step_boundary`.
        """
        return self._elapsed_sim_time

    def _schedule_time(self) -> float:
        """:attr:`sim_time` without its accumulated float error, for frequency gates.

        ``sim_time`` adds one timestep per step, so its error grows with the
        run: at a 1 ms timestep it passes a millionth of a timestep within
        minutes, enough to move a due time by a step. This counts whole
        timesteps from an origin that moves when the clock is restored and,
        at the next step, when the timestep has changed -- through pacing or
        by assigning ``params.timestep`` -- so its error stays at the rounding
        of one product.
        """
        step, time_at_step, timestep = self._schedule_origin
        return time_at_step + (self._step_count - step) * timestep

    def _emit_step_hooks(self, event: str, tiers: Tuple[Tuple[int, Optional[str]], ...], measure_timing: bool) -> None:
        """Emit a step event one tier at a time, timing each tier under its profiling key.

        One event on the bus; the tiers are emitted apart only so the time
        profile keeps handlers, callbacks and plugins separate. Every tier is
        taken from one snapshot read before any subscription runs, so, as for
        any emit, a subscription added or removed by a subscription takes
        effect from the next step. With no subscriber at all this is a single
        lookup.
        """
        snapshot = self.events._tier_snapshot(event)
        if snapshot is None:
            return
        kwargs = {"dt": self._params.timestep, "sim_time": self.sim_time}
        dispatch = self.events._dispatch
        for tier, profiling_key in tiers:
            subscriptions = snapshot[tier]
            if not subscriptions:
                continue
            if measure_timing and profiling_key is not None:
                t0 = time.perf_counter()
                dispatch(event, subscriptions, kwargs)
                self._profiling_stats[profiling_key][-1] = (time.perf_counter() - t0) * 1000
            else:
                dispatch(event, subscriptions, kwargs)

    def _hook_clock(self) -> Tuple[float, float]:
        """The time and timestep that frequency-limited step hooks schedule on (``events`` clock)."""
        return self._schedule_time(), self._params.timestep

    def get_completed_step_boundary(self) -> tuple[int, float]:
        """Return the completed step and canonical elapsed time outside a step."""
        if self._in_step or self._in_post_step:
            raise RuntimeError("A completed-step boundary is unavailable during step_once()")
        return self._step_count, self._elapsed_sim_time

    def configure_state_recording(
        self,
        *,
        profile: Any,
        output: Optional[str] = None,
        checkpoint_every_steps: int = 1,
        records: tuple[Any, ...] = (),
    ) -> Any:
        """Record a supported execution profile during ordinary simulation runs.

        State recording is distinct from camera/video recording. The profile
        declares what can be restored; named records capture caller-owned data
        at the same completed boundary. Unsupported live state fails at capture.
        """
        if self._state_recorder is not None:
            raise RuntimeError("State recording is already configured")
        from pybullet_fleet.replay.state_recording import StateRecorder

        recorder = StateRecorder(self, output, profile, records, checkpoint_every_steps)
        self._state_recorder = recorder
        return recorder

    def _record_state_input(self, operation: str, details: dict) -> None:
        """Forward an effective input when state recording is configured."""
        if self._state_recorder is not None:
            try:
                self._state_recorder.record_input(operation, details)
            except Exception as exc:
                self._abort_state_recording(exc)

    def _abort_state_recording(self, exc: Exception) -> None:
        """Keep an optional recorder failure from interrupting a valid simulation."""
        recorder = self._state_recorder
        self._state_recorder = None
        logger.warning("State recording stopped after an error: %s", exc)
        if recorder is not None:
            recorder.abort(exc)

    def _record_state_spawn(self, obj: SimObject, spawn_params: Any) -> None:
        """Retain construction data and record a spawned entity if enabled."""
        if self._state_recorder is None:
            return
        try:
            obj._checkpoint_spawn_params = copy.deepcopy(spawn_params)
            self._state_recorder.record_spawn(obj, spawn_params)
        except Exception as exc:
            self._abort_state_recording(exc)

    def restore_completed_step_boundary(self, step: int, elapsed_time: float) -> None:
        """Restore the clock of a fresh, initialized fixed-step simulation."""
        if self._in_step or self._in_post_step or self._step_count != 0 or self._elapsed_sim_time != 0.0:
            raise RuntimeError("Completed-step restore requires a fresh initialized simulation")
        if type(step) is not int or step < 1 or not math.isfinite(elapsed_time):
            raise ValueError("Invalid completed-step clock")
        expected = step * self._params.timestep
        if not math.isclose(elapsed_time, expected, rel_tol=0.0, abs_tol=self._params.timestep * 1e-9):
            raise ValueError("Completed-step time does not match the fixed timestep")
        self._step_count = step
        self._elapsed_sim_time = elapsed_time
        self._schedule_origin = (step, elapsed_time, self._params.timestep)
        self._resume_ready = True

    @property
    def collision_count(self) -> int:
        """Total number of collisions detected."""
        return self._collision_count

    @property
    def enable_memory_profiling(self) -> bool:
        """Whether memory profiling is enabled."""
        return self._enable_memory_profiling

    @property
    def last_profiling(self) -> Optional[Mapping[str, Any]]:
        """Timing values from the most recently completed profiled step.

        The mapping has the same timing fields (in milliseconds) as
        :meth:`step_once` with ``return_profiling=True``, including custom
        fields added by :meth:`record_profiling`.  It is updated only when
        timing is measured: either time profiling is enabled or that step was
        called with ``return_profiling=True``.  Otherwise it is ``None``.

        Simulation plugins and callbacks execute before the current step's
        later phases have completed, so they observe the previous completed
        step through this property.  The returned mapping is read-only.
        """
        return self._last_profiling

    # ------------------------------------------------------------------
    # Plugin system
    # ------------------------------------------------------------------

    def _request_mirror(self, target: Any) -> None:
        """Count a plugin or callback that keeps body ids for PyBullet calls (``needs_mirror``).

        The poses earlier lazy steps left pending are written now, so the
        requester reads the current poses from its first run on.
        """
        if getattr(target, "needs_mirror", False):
            self._mirror_requests += 1
            self.sync_mirror()

    def _release_mirror(self, target: Any) -> None:
        """The plugin or callback counted by :meth:`_request_mirror` is gone."""
        if getattr(target, "needs_mirror", False) and self._mirror_requests > 0:
            self._mirror_requests -= 1

    def register_plugin(
        self,
        plugin_or_cls: "Union[_SimPluginT, Type[_SimPluginT]]",
        config: Optional[Dict[str, Any]] = None,
        frequency: Optional[float] = None,
        phase: Optional[Union[CallbackPhase, str]] = None,
    ) -> "_SimPluginT":
        """Register a simulation plugin.

        Args:
            plugin_or_cls: Either a :class:`SimPlugin` instance or a
                :class:`SimPlugin` subclass (will be instantiated).
            config: Config dict (only used when *plugin_or_cls* is a class).
            frequency: Step frequency in Hz (``None`` = every step).
            phase: When in the step :meth:`SimPlugin.on_step` runs, a
                :class:`CallbackPhase` or its value (``"pre_step"``, ``"step"``,
                ``"post_step"``), as for :meth:`register_callback`. ``None``
                keeps the plugin's own :attr:`SimPlugin.step_phase` (``"step"``
                unless the class sets another).

        Returns:
            The registered :class:`SimPlugin` instance.

        Raises:
            TypeError: If *plugin_or_cls* is not a SimPlugin.
            ValueError: If ``frequency`` or ``phase`` is not valid.
        """
        from pybullet_fleet.sim_plugin import SimPlugin

        if isinstance(plugin_or_cls, SimPlugin):
            plugin = plugin_or_cls
        elif isinstance(plugin_or_cls, type) and issubclass(plugin_or_cls, SimPlugin):
            plugin = plugin_or_cls.from_config(self, config or {})
        else:
            raise TypeError(f"Expected SimPlugin instance or subclass, got {plugin_or_cls!r}")

        plugin.frequency = frequency  # validates before the plugin is registered
        if phase is not None:
            plugin.step_phase = phase  # validates too
        self._add_plugin(plugin)
        return plugin

    @property
    def plugins(self) -> "List[SimPlugin]":
        """Read-only list of active plugins."""
        return list(self._plugins)

    def _load_and_init_plugins(self, config: Dict[str, Any]) -> None:
        """Load plugins from *config* and call ``on_init()`` on each.

        Shared helper used by :meth:`from_yaml` and :meth:`from_dict`.
        """
        plugin_configs = config.get("plugins", [])
        if plugin_configs:
            from pybullet_fleet.sim_plugin import _load_plugins_from_config

            plugins = _load_plugins_from_config(plugin_configs, self)
            for plugin in plugins:
                self._add_plugin(plugin)
        self._init_plugins()

    def _init_plugins(self) -> None:
        """Call ``on_init()`` on all loaded plugins (after spawn)."""
        for plugin in self._plugins:
            plugin.on_init()

    def _add_plugin(self, plugin: "SimPlugin") -> None:
        """Keep ``plugin`` and subscribe its ``on_step()`` to its phase's step event.

        It runs on the event named by :attr:`SimPlugin.step_phase` (``step`` unless
        set otherwise), in the plugin tier, after that event's handlers and
        callbacks, in load order, on the plugin's own schedule
        (:attr:`SimPlugin.frequency`). Its exceptions are logged, not raised.
        """

        def on_step(dt: float) -> None:
            try:
                plugin.on_step(dt)
            except Exception:
                logger.error("Plugin %s.on_step() failed", type(plugin).__name__, exc_info=True)

        self._plugins.append(plugin)
        self._request_mirror(plugin)
        self.events._subscribe(
            plugin.step_phase.value, plugin, on_step, tier=_TIER_PLUGIN, gate=plugin._frequency_gate, takes_dt=True
        )
        plugin._subscribed = True

    def _shutdown_plugins(self) -> None:
        """Call ``on_shutdown()`` on all plugins."""
        for plugin in self._plugins:
            try:
                plugin.on_shutdown()
            except Exception:
                logger.error("Plugin %s.on_shutdown() failed", type(plugin).__name__, exc_info=True)

    def register_callback(
        self,
        callback: Callable,
        frequency: Optional[float] = None,
        phase: Union[CallbackPhase, str] = CallbackPhase.STEP,
    ) -> None:
        """
        Register a callback function to be called during simulation.

        Args:
            callback: Function with signature callback(sim_core, dt) -> None
                     - sim_core: Reference to MultiRobotSimulationCore instance
                     - dt: Simulated time since this callback last ran (seconds)
            frequency: Callback frequency in Hz. If None (or 0), called every
                simulation step. Due times are a fixed grid of ``k / frequency``
                seconds of simulated time, so a callback neither skips beats nor
                drifts; it runs at most once per step, and the first time one
                interval into the run.
            phase: When in the step the callback runs, a :class:`CallbackPhase`
                or its value as a string.

                - ``CallbackPhase.PRE_STEP`` / ``"pre_step"``: at the start of the step, right after the
                  ``pre_step`` event's handlers, before the clock advances and before anything
                  moves. ``sim_core.sim_time`` is still the previous state's time. Use it to
                  apply commands this step should act on.
                - ``CallbackPhase.STEP`` / ``"step"`` (default): after the objects have been updated, before
                  their new poses are written to PyBullet and before the collision
                  check.
                - ``CallbackPhase.POST_STEP`` / ``"post_step"``: at the end of the step, right after the
                  ``post_step`` event: poses are written, and when the step ran
                  a collision check (see ``collision_check_frequency``) the
                  active pairs are those of the new poses. Use it to publish or
                  record the step's outcome.

                In every phase ``sim_core.sim_time`` is the time of the state the
                callback sees: the previous state's in ``PRE_STEP``, the end of the
                step in ``STEP`` and ``POST_STEP``; stamp anything read from the
                bodies with it. All three schedule their ``frequency`` on it.

        Raises:
            ValueError: ``frequency`` is negative or not finite, or ``phase`` is
                not a :class:`CallbackPhase`.

        A callback's exception is not caught (unlike a plugin's): it propagates
        out of :meth:`step_once`, and the rest of the step is skipped. What has
        happened by then depends on the phase:

        - ``PRE_STEP``: the clock has not advanced (``sim_time`` and
          ``step_count`` stay at the previous state) and nothing has moved; the
          later ``pre_step`` callbacks and plugins and the whole update are
          skipped.
        - ``STEP``: the clock has advanced and the objects have moved; later
          callbacks, the step's plugins, the pose flush, the collision check,
          the ``post_step`` event and the ``POST_STEP`` callbacks are skipped.
        - ``POST_STEP``: the step is complete except for later callbacks, the
          ``post_step`` plugins and what follows them.

        In every phase the state recorder's completed step and the DataMonitor
        refresh are skipped.

        A callback is a subscription to the ``phase`` event on :attr:`events`
        (the event of the same name, :attr:`SimEvents.PRE_STEP`,
        :attr:`SimEvents.STEP` or :attr:`SimEvents.POST_STEP`) in a tier of its own: it runs after that
        event's handlers, whatever their priority, and callbacks run in
        registration order. ``events.on(phase, handler, frequency=...)`` is the
        keyword-argument form of the same hook, with the same schedule; its
        exceptions are logged instead of raised.

        Example::

            def my_callback(sim_core, dt):
                for obj in sim_core.sim_objects:
                    if isinstance(obj, Agent):
                        # Update agent logic
                        pass

            sim_core.register_callback(my_callback, frequency=10.0)  # 10 Hz

            def publish(sim_core, dt):
                stamp = sim_core.sim_time
                ...

            sim_core.register_callback(publish, frequency=10.0, phase=CallbackPhase.POST_STEP)
        """
        try:
            phase = CallbackPhase(phase)
        except ValueError:
            raise ValueError("phase must be one of {}, got {!r}".format([p.value for p in CallbackPhase], phase)) from None
        self._request_mirror(callback)
        # Every callback has a gate, also without a frequency, so its dt is the
        # time since it last ran, measured the same way at every rate.
        gate = FrequencyGate(frequency or None)
        self.events._subscribe(
            phase.value,
            callback,
            callback,
            tier=_TIER_CALLBACK,
            gate=gate,
            propagate=True,
            takes_dt=True,
            owner=self._weak_self,
        )

    @property
    def _callbacks(self) -> List[Dict[str, Any]]:
        """The registered callbacks in registration order, as ``{"func", "phase", "gate"}`` records."""
        return [
            {"func": sub.handler, "phase": CallbackPhase(event), "gate": sub.gate}
            for event, sub in self.events._subscriptions(_TIER_CALLBACK)
        ]

    def register_behavior_tree(self, tree: BehaviorTree) -> BehaviorTree:
        """Register a behavior tree to tick once per simulation step.

        ``tree`` must provide ``tick(sim_core, dt)``. Registration is
        idempotent by object identity, and the returned tree makes inline use
        convenient. The tree's ``tick`` method remains public for deterministic
        unit tests and custom stepping loops.
        """
        callback = getattr(tree, "tick", None)
        if not callable(callback):
            raise TypeError("Behavior tree must provide a callable tick(sim_core, dt) method")
        if any(registered_tree is tree for registered_tree, _ in self._behavior_tree_callbacks):
            return tree
        self.register_callback(callback)
        self._behavior_tree_callbacks.append((tree, callback))
        return tree

    def unregister_behavior_tree(self, tree: BehaviorTree) -> bool:
        """Stop automatically ticking a tree previously registered with this simulation."""
        for index, (registered_tree, callback) in enumerate(self._behavior_tree_callbacks):
            if registered_tree is tree:
                self._behavior_tree_callbacks.pop(index)
                self.unregister_callback(callback)
                return True
        return False

    @property
    def behavior_trees(self) -> Tuple[BehaviorTree, ...]:
        """Behavior trees automatically ticked by this simulation."""
        return tuple(tree for tree, _ in self._behavior_tree_callbacks)

    def register_manager(self, manager: "SimObjectManager") -> None:
        """Register a :class:`SimObjectManager` (or subclass) with this simulation.

        Registered managers are automatically synchronised when objects are
        removed via :meth:`remove_object` or the simulation is :meth:`reset`.
        The manager's ``sim_core`` attribute is set to *self*.

        A manager can only be registered once; duplicate calls are ignored.

        Args:
            manager: The manager instance to register.

        Example::

            from pybullet_fleet.agent_manager import AgentManager

            mgr = AgentManager()
            sim.register_manager(mgr)
            mgr.spawn_agents_grid(10, grid_params, spawn_params)
            # When sim.remove_object(agent) is called later,
            # mgr.objects is kept in sync automatically.
        """
        if manager in self._registered_managers:
            return
        manager.sim_core = self
        self._registered_managers.append(manager)
        logger.debug("Registered manager %r (total: %d)", manager, len(self._registered_managers))

    def unregister_manager(self, manager: "SimObjectManager") -> bool:
        """Remove a manager from the registry.

        Returns:
            True if the manager was found and removed, False otherwise.
        """
        try:
            self._registered_managers.remove(manager)
            logger.debug("Unregistered manager %r", manager)
            return True
        except ValueError:
            return False

    def get_manager(self, name: str) -> Optional[Any]:
        """Return the registered :class:`~pybullet_fleet.agent_manager.AgentManager`
        with the given name, or ``None``.

        Args:
            name: The name assigned to the manager (``name:`` field in YAML, or
                the ``name=`` argument passed to :class:`AgentManager`).

        Example::

            sim = MultiRobotSimulationCore.from_yaml("config.yaml")
            fleet = sim.get_manager("delivery_fleet")
            for agent in fleet.objects:
                agent.set_path([goal])
        """
        for mgr in self._registered_managers:
            if getattr(mgr, "name", None) == name:
                return mgr
        return None

    def _init_named_managers(self, managers_config: List[Dict[str, Any]]) -> None:
        """Create :class:`~pybullet_fleet.agent_manager.AgentManager` instances
        from the ``managers:`` config section and register them.

        Each entry must have a ``name:`` field.  Its optional
        ``fleet_controller:`` mapping combines manager-level execution and
        shared :class:`~pybullet_fleet.controller_params.ControllerParams`
        defaults. ``fleet_controller.type`` is either ``"batch_omni"`` or
        ``"batch_differential"`` to enable that batch controller; omit it for
        per-agent execution. Remaining fields (for example
        ``max_linear_vel`` or ``navigation_2d``) are applied only where an
        agent did not specify that controller parameter itself.
        * ``update_frequency`` — callback Hz (default ``10.0``).
        """
        from pybullet_fleet.agent_manager import AgentManager

        for cfg in managers_config:
            name = cfg.get("name")
            if not name:
                logger.warning("managers: entry missing required 'name' field, skipping")
                continue
            if self.get_manager(name) is not None:
                logger.warning("managers: duplicate name %r, skipping second entry", name)
                continue
            fleet_controller = dict(cfg.get("fleet_controller") or {})
            update_frequency = float(cfg.get("update_frequency", 10.0))
            mgr = AgentManager(
                sim_core=self,
                name=name,
                update_frequency=update_frequency,
                fleet_controller=fleet_controller or None,
            )
            logger.info(
                "Created manager %r (batch_controller=%r, fleet_controller=%r, update_frequency=%s Hz)",
                name,
                type(mgr.batch_controller).__name__ if mgr.batch_controller is not None else None,
                mgr._fleet_controller,
                update_frequency,
            )

    def unregister_callback(self, callback_func: Callable) -> bool:
        """Remove a registered callback by function reference.

        Args:
            callback_func: The callback function to remove (matched by identity).

        Returns:
            True if the callback was found and removed, False otherwise.
        """
        for event, sub in self.events._subscriptions(_TIER_CALLBACK):
            if sub.handler is callback_func:
                self.events._unsubscribe(event, sub)
                self._release_mirror(callback_func)
                return True
        return False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(
        self,
        output: str = "recording.gif",
        width: int = 800,
        height: int = 600,
        fps: int = 15,
        duration: Optional[float] = None,
        camera_mode: str = "auto",
        time_base: str = "sim",
        **kwargs: Any,
    ) -> Any:
        """Start recording the simulation as an animated GIF or MP4.

        Frames are captured via ``getCameraImage(ER_TINY_RENDERER)`` at the
        specified *fps* frequency. Works in both ``p.DIRECT`` and ``p.GUI``.

        Output format is auto-detected from the file extension:

        - ``.gif`` — Animated GIF via Pillow (always available)
        - ``.mp4`` — H.264 MP4 via imageio + pyav (requires ``pip install imageio[pyav]``)

        Time bases:

        - ``"sim"`` (default) — Capture at fixed sim-time intervals.
          Playback always looks like 1× speed.
        - ``"real"`` — Capture at wall-clock intervals.
          Playback reflects actual execution speed (e.g. 10× sim → 10× video).

        Args:
            output: Output file path (``.gif`` or ``.mp4``). Parent directories created automatically.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Capture frequency and playback FPS.
            duration: Recording duration in sim-seconds (``time_base="sim"``) or
                wall-clock seconds (``time_base="real"``). ``None`` for manual stop.
            camera_mode: ``"auto"`` | ``"gui"`` | ``"orbit"`` | ``"manual"``.
                ``"auto"`` is promoted to ``"gui"`` when running in GUI mode.
            time_base: ``"sim"`` or ``"real"``. Controls capture timing.
            **kwargs: Passed to ``SimulationRecorder`` (e.g. ``orbit_degrees``,
                     ``camera_params``).

        Returns:
            :class:`~pybullet_fleet.recorder.SimulationRecorder` instance.

        Example::

            sim.start_recording("output.gif", duration=4.0)
            sim.start_recording("output.mp4", duration=4.0, fps=30)
            sim.start_recording("perf.mp4", duration=4.0, time_base="real")
            sim.run_simulation(duration=5.0)
            # File is auto-saved when run_simulation ends
        """
        from pybullet_fleet.recorder import SimulationRecorder

        if self._recorder is not None:
            self.stop_recording()  # stop existing recording
        self._recorder = SimulationRecorder(
            sim_core=self,
            output=output,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            camera_mode=camera_mode,
            time_base=time_base,
            **kwargs,
        )
        self._recorder.start()
        logger.info("Recording started: %s", output)
        return self._recorder

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save the output file.

        Returns:
            Output file path if a recording was active, ``None`` otherwise.
        """
        if self._recorder is None:
            return None
        recorder = self._recorder
        recorder.stop()
        try:
            path = recorder.save()
            logger.info("Recording saved: %s (%d frames)", path, recorder.frame_count)
        except ValueError:
            logger.warning("Recording stopped but no frames were captured")
            path = None
        self._recorder = None
        return path

    def set_collision_check_frequency(self, frequency: Optional[float] = None) -> None:
        """
        Set the frequency (Hz, number of times per second) for collision detection.

        Args:
            frequency: Collision check frequency in Hz (times per second)
                      - None: Check every step (highest accuracy, highest cost)
                      - 0: Disable collision checking (no collision detection)
                      - > 0: Check at specified frequency (e.g., 1.0 = once per second)

        Example::

            sim.set_collision_check_frequency(None)  # Every step
            sim.set_collision_check_frequency(1.0)   # 1 Hz (once per second)
            sim.set_collision_check_frequency(10.0)  # 10 Hz
            sim.set_collision_check_frequency(0)     # Disable

        Note: Higher frequency increases computational cost but improves collision detection accuracy.
        """
        self._collision_check_frequency = frequency

        # Log the setting for clarity
        if frequency is None:
            logger.info("Collision check frequency set to: EVERY STEP (highest accuracy)")
        elif frequency == 0:
            logger.info("Collision check frequency set to: DISABLED (no collision detection)")
        else:
            logger.info(f"Collision check frequency set to: {frequency} Hz ({1.0/frequency:.3f}s interval)")

    def setup_pybullet(self) -> None:
        """Initialize PyBullet with GUI panels hidden."""
        # Force headless mode when RECORD env var is set (enables zero-code-change recording).
        # RECORD_GUI=1 overrides this to keep the GUI window open so the recorder
        # can capture via ER_BULLET_HARDWARE_OPENGL (GPU-quality frames).
        if os.environ.get("RECORD") and self._params.gui and not os.environ.get("RECORD_GUI"):
            logger.info("RECORD env var detected: forcing gui=False for headless recording")
            self._params.gui = False

        options = ""
        if self._params.gui and self._params.window_width > 0 and self._params.window_height > 0:
            options = f"--width={self._params.window_width} --height={self._params.window_height}"
        self._client = p.connect(p.GUI if self._params.gui else p.DIRECT, options=options)
        # A reused client id may still have cache entries from a connection
        # that was disconnected without close() (p.disconnect() by the caller).
        SimObject._forget_shared_shapes(self._client)

        # Hide all debug UI panels immediately after connection
        if self._params.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81 if self._params.physics else 0, physicsClientId=self._client)
        p.setTimeStep(self._params.timestep, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)

        # Physics engine parameter tuning
        self._apply_physics_engine_params()
        if self._params.enable_floor:
            p.loadURDF("plane.urdf", physicsClientId=self._client)

        # Register user-supplied model search paths.
        # NOTE: add_search_path() mutates the global _user_search_paths registry
        # in robot_models, so paths registered here persist across sim instances.
        if self._params.model_paths:
            from pybullet_fleet.robot_models import add_search_path

            for path in self._params.model_paths:
                add_search_path(path)

        # Disable rendering during setup for better performance
        if self._params.gui:
            self.disable_rendering()

    def _apply_physics_engine_params(self) -> None:
        """Apply physics engine parameter tuning.

        Shared by ``setup_pybullet()`` and ``reset()`` to ensure
        consistent engine configuration after ``p.resetSimulation()``.
        """
        p.setPhysicsEngineParameter(enableFileCaching=True, physicsClientId=self._client)
        if self._params.physics:
            # Physics ON: deterministic broadphase + CCD for reproducibility and tunneling prevention
            p.setPhysicsEngineParameter(
                deterministicOverlappingPairs=True,
                allowedCcdPenetration=0.01,
                physicsClientId=self._client,
            )
        else:
            # Physics OFF (kinematics only): minimize solver overhead
            p.setPhysicsEngineParameter(
                numSubSteps=1,
                numSolverIterations=1,
                enableConeFriction=False,
                physicsClientId=self._client,
            )

    def disable_rendering(self) -> None:
        """Disable rendering during object spawning for better performance."""
        if self._params.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._client)
            self._rendering_enabled = False
            logger.info("Rendering disabled for setup/spawning phase")

    def enable_rendering(self) -> None:
        """Enable rendering before starting simulation."""
        if self._params.gui and not self._rendering_enabled:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._client)
            # Toggling rendering can reset this panel on some PyBullet builds.
            # Lighting controls intentionally use the native Parameters panel.
            show_panel = self._params.enable_gui_panel or self._lighting_controls_enabled
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if show_panel else 0, physicsClientId=self._client)
            self._rendering_enabled = True
            logger.info("Rendering enabled for simulation")

    @contextmanager
    def batch_spawn(self):
        """Context manager for optimised batch spawning.

        Disables rendering and defers spatial-grid rebuild until
        the context exits.  Use when spawning many objects at once.

        Rendering state is saved on entry and restored on exit.
        If rendering was already disabled (e.g. during the initial
        setup phase before ``run_simulation()``), it stays disabled
        after ``batch_spawn()`` exits.

        Example::

            with sim_core.batch_spawn():
                for params in params_list:
                    Agent.from_params(params, sim_core)
            # rendering restored to previous state, spatial grid rebuilt once
        """
        was_adaptive = self._params.spatial_hash_cell_size_mode == SpatialHashCellSizeMode.AUTO_ADAPTIVE
        was_rendering = self._rendering_enabled
        self._batch_spawning = True
        self.disable_rendering()
        try:
            yield
        finally:
            self._batch_spawning = False
            if was_adaptive:
                # set_collision_spatial_hash_cell_size_mode() already calls _rebuild_spatial_grid()
                self.set_collision_spatial_hash_cell_size_mode()
            else:
                self._rebuild_spatial_grid()
            if was_rendering:
                self.enable_rendering()

    def _calculate_cell_size_from_aabbs(
        self, aabbs: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = None
    ) -> Optional[float]:
        """
        Calculate optimal cell size from AABBs using median extent.

        Args:
            aabbs: List of AABB tuples ((min_x, min_y, min_z), (max_x, max_y, max_z)).
                   If None, uses the AABBs the collision system holds (default: None)

        Returns:
            Calculated cell_size, or None if no AABBs available

        Example::

            # Use cached AABBs
            cell_size = sim._calculate_cell_size_from_aabbs()

            # Use custom AABBs
            custom_aabbs = [p.getAABB(body_id) for body_id in body_ids]
            cell_size = sim._calculate_cell_size_from_aabbs(custom_aabbs)
        """
        if aabbs is None:
            median_extent = self._store.median_extent()
            return None if median_extent is None else max(median_extent * 2.0, 0.5)

        if not aabbs:
            return None

        extents = []
        for aabb in aabbs:
            extent_x = aabb[1][0] - aabb[0][0]
            extent_y = aabb[1][1] - aabb[0][1]
            extent_z = aabb[1][2] - aabb[0][2]
            extents.append(max(extent_x, extent_y, extent_z))

        median_extent = sorted(extents)[len(extents) // 2]
        cell_size = max(median_extent * 2.0, 0.5)

        return cell_size

    def _rebuild_spatial_grid(self) -> None:
        """Register every body again with the current cell size and margin (on the hash's next sync)."""
        self._set_grid_margin(self._collision_broadphase_margin())
        if self._cached_cell_size is not None:
            self._store.set_hash(self._cached_cell_size, cast(Tuple[float, float, float], self._grid_half_margin))
            logger.debug("Collision spatial hash: rebuild scheduled")

    def _mark_object_moved(self, object_id: int) -> None:
        """
        Mark an object as moved in the current step.
        Called by SimObject.set_pose() when movement is detected.

        Args:
            object_id: The object_id that moved
        """
        self._moved_this_step.add(object_id)
        logger.debug(f"Object {object_id} marked as moved")

    def _add_plugin_to_batch(self, plugin: Any) -> bool:
        """Add ``plugin`` to its class's batch; False when it is updated plugin by plugin.

        That is the case when the class has no batch, or when the batch does
        not take this plugin (:meth:`PluginBatch.add` returned False).
        """
        cls = type(plugin)
        if cls not in self._plugin_batches:
            self._plugin_batches[cls] = cls.create_batch(self)
        batch = self._plugin_batches[cls]
        if batch is None:
            return False
        return batch.add(plugin) is not False

    def _note_needs_update(self, obj: SimObject) -> None:
        """``obj._needs_update`` changed: it joins or leaves the step's update loop at once."""
        if obj.object_id not in self._sim_objects_dict:
            return
        if obj._needs_update:
            if obj not in self._updatable:
                self._updatable.append(obj)
        else:
            try:
                self._updatable.remove(obj)
            except ValueError:
                pass
            else:
                self._updatable_left.add(obj)

    def _remove_plugin_from_batch(self, plugin: Any) -> None:
        batch = self._plugin_batches.get(type(plugin))
        if batch is not None:
            batch.remove(plugin)

    def _bind_pose_rows(self) -> None:
        """Point every object's pose cache at its row of the pose columns (after a reallocation)."""
        store = self._store
        for obj in self._sim_objects:
            row = store.row_of.get(obj.object_id)
            if row is not None:
                obj._bind_rows(store, row)

    def _note_preserve_velocity(self, object_id: int, preserve_velocity: bool) -> None:
        """Remember, for the Phase 2 flush, whether the object's pending pose write keeps its velocity."""
        row = self._store.row_of.get(object_id)
        if row is not None:
            self._store.preserve_velocity[row] = preserve_velocity

    def _set_per_object_pose(self, obj: SimObject) -> None:
        """Flag ``obj`` and its carriers when a buffered pose write must go through the object (see ``per_object_pose``)."""
        store = self._store
        carrier: Optional[SimObject] = obj
        while carrier is not None:
            row = store.row_of.get(carrier.object_id)
            if row is not None:
                store.per_object_pose[row] = bool(carrier.attached_objects) and not self._carries_as_arrays(carrier)
            carrier = carrier._attached_to

    def _note_base_attachment(self, child_id: int, parent_id: Optional[int], offset: Optional[Pose]) -> None:
        """Record that ``child_id`` rides on ``parent_id``'s base at ``offset`` (``None``: it no longer does).

        A rider that leaves its carrier counts as moved, so a pair the two
        made is checked on the next collision check even if neither moves.
        """
        store = self._store
        row = store.row_of.get(child_id)
        former_parent = self._base_parent_of.pop(child_id, None)
        if former_parent is not None:
            riders = self._base_children.get(former_parent)
            if riders is not None:
                try:
                    riders.remove(child_id)
                except ValueError:
                    pass
                if not riders:
                    del self._base_children[former_parent]
        if parent_id is None or offset is None:
            if row is not None:
                store.attach_pos[row] = 0.0
                store.attach_orn[row] = (0.0, 0.0, 0.0, 1.0)
                store.parent_row[row] = -1
            if former_parent is not None:
                self._moved_this_step.add(child_id)
                former = self._sim_objects_dict.get(former_parent)
                if former is not None:
                    self._set_per_object_pose(former)
            return
        self._base_children.setdefault(parent_id, []).append(child_id)
        self._base_parent_of[child_id] = parent_id
        parent_row = store.row_of.get(parent_id)
        if row is not None:
            store.attach_pos[row] = offset.position
            store.attach_orn[row] = offset.orientation
            store.parent_row[row] = -1 if parent_row is None else parent_row
        parent = self._sim_objects_dict.get(parent_id)
        if parent is not None:
            self._set_per_object_pose(parent)

    def _note_attachment_tree(self, obj: SimObject, left: bool = False) -> None:
        """Record the root of the attachment tree ``obj`` is in, for every member (``attach_root``).

        With ``left``, ``obj``'s tree has just left another one (a detach, a
        removed carrier): every member whose root changes counts as moved,
        so a pair it made with the old tree (a rider's own rider against the
        old carrier, say), no pair while they shared a root, is checked on
        the next collision check, as a pair of a detached rider is.
        """
        store = self._store
        root = obj._attachment_root()
        root_id = root.object_id
        moved = self._moved_this_step
        pending = [root]
        seen = set()
        while pending:
            member = pending.pop()
            if id(member) in seen:
                continue
            seen.add(id(member))
            row = store.row_of.get(member.object_id)
            if row is not None and store.attach_root[row] != root_id:
                store.attach_root[row] = root_id
                if left:
                    moved.add(member.object_id)
            pending.extend(member.attached_objects)

    @property
    def _mirror_always(self) -> bool:
        """Whether every pending pose is written to PyBullet in Phase 2.

        With ``physics`` on, PyBullet integrates and resolves contacts against
        the kinematic bodies in ``stepSimulation()``, so it must hold their
        poses at every step: the mirror is always written. Otherwise it is
        written for a GUI, ``mirror: always``, or a plugin or callback that
        asked for it (``needs_mirror``).
        """
        return (
            bool(self._params.physics)
            or bool(self._params.gui)
            or self._params.mirror == "always"
            or self._mirror_requests > 0
        )

    def sync_mirror(self, object_ids: Optional[Iterable[int]] = None) -> None:
        """Write every pending pose (or those of ``object_ids``) to PyBullet.

        With ``mirror: lazy`` a kinematic body's buffered pose stays in the
        simulation's pose columns until PyBullet needs it. The simulation
        syncs by itself before anything of its own reads PyBullet: the
        narrow phase of a collision pair, an AABB query, a recording, the
        accessors ``SimObject.body_id`` and ``sim.client``. Call this before
        PyBullet calls made with a body id kept from earlier, or run with
        ``mirror: always``.

        Args:
            object_ids: The objects to sync; every object when ``None``.
        """
        store = self._store
        n = len(store)
        if n == 0 or not p.isConnected(physicsClientId=self._client):
            return  # nothing to write, or no PyBullet to write to (after disconnect)
        if object_ids is None:
            dirty = store.mirror_dirty[:n]
            if not dirty.any():
                return
            rows = np.flatnonzero(dirty)
        else:
            rows = store.rows(object_ids)
            if len(rows) == 0:
                return
            rows = rows[store.mirror_dirty[rows]]
            if len(rows) == 0:
                return
        self._write_rows_to_pybullet(rows)

    def _write_rows_to_pybullet(self, rows: np.ndarray) -> None:
        """Write the rows' poses to their PyBullet bodies and mark them mirrored."""
        store = self._store
        objects = self._sim_objects_dict
        client = self._client
        ids = store.object_id[rows].tolist()
        positions = store.pos[rows].tolist()
        orientations = store.orn[rows].tolist()
        preserve_flags = store.preserve_velocity[rows].tolist()
        for object_id, position, orientation, preserve_velocity in zip(ids, positions, orientations, preserve_flags):
            obj = objects.get(object_id)
            if obj is None:
                continue
            body_id = obj._body_id
            # Match the legacy immediate path exactly: velocity preservation for
            # non-kinematic objects, fast path for kinematics.
            if preserve_velocity and not obj.is_kinematic:
                linear_vel, angular_vel = p.getBaseVelocity(body_id, physicsClientId=client)
                p.resetBasePositionAndOrientation(body_id, position, orientation, physicsClientId=client)
                p.resetBaseVelocity(body_id, linear_vel, angular_vel, physicsClientId=client)
            else:
                p.resetBasePositionAndOrientation(body_id, position, orientation, physicsClientId=client)
        store.mirror_dirty[rows] = False

    def _sync_object_mirror_if_dirty(self, obj: SimObject) -> None:
        """Write ``obj``'s pose to PyBullet if a lazy flush left it pending (``SimObject.body_id`` calls this).

        A pose set inside a step is buffered until Phase 2 and is not pending
        in this sense: PyBullet receives it when the step flushes it, as it
        always did.
        """
        row = self._store.row_of.get(obj.object_id)
        if row is not None and self._store.mirror_dirty[row]:
            self._write_rows_to_pybullet(np.array([row], dtype=np.int64))

    def _note_mirrored(self, object_id: int) -> None:
        """The object's pose was just written to PyBullet outside the flush."""
        row = self._store.row_of.get(object_id)
        if row is not None:
            self._store.mirror_dirty[row] = False

    def _set_needs_mirror(self, object_id: int) -> None:
        """Whether PyBullet must hold the object's pose after every step.

        A physics body, an articulated body or a link-attachment parent.
        """
        row = self._store.row_of.get(object_id)
        if row is None:
            return
        needs = (
            object_id in self._physics_objects
            or self._aabb_links.get(object_id, ()) != ()
            or object_id in self._link_attachment_parent_ids
        )
        self._store.needs_mirror[row] = needs

    # ----------------------------------------------------------------------
    # Public batch pose API (D8: two-phase-step spec)
    # ----------------------------------------------------------------------
    def set_poses(
        self,
        object_ids: Sequence[int],
        positions: np.ndarray,
        orientations: np.ndarray,
        *,
        preserve_velocity: bool = True,
    ) -> None:
        """Set poses for N objects in one call (batch / high-throughput path).

        Use this when poses are already in array form (ROS ``PoseArray``,
        vectorized controllers, NumPy pipelines). For single-object writes
        prefer :meth:`SimObject.set_pose` with a :class:`Pose` object — this
        method intentionally takes raw ndarrays to avoid per-object ``Pose``
        allocations on the hot path.

        Internally honours the same two-phase semantics as ``SimObject.set_pose``:
        inside ``step_once()`` the writes are buffered to be flushed in Phase 2,
        outside they propagate to PyBullet immediately. ``get_pose`` /
        ``get_poses`` always reflect the new values regardless of phase.

        Args:
            object_ids: N object ids to write to (in matching order).
            positions: ``(N, 3)`` float array of XYZ positions.
            orientations: ``(N, 4)`` float array of XYZW quaternions.
            preserve_velocity: If True, preserve current velocities (dynamic
                objects only; ignored for kinematic objects).

        Raises:
            ValueError: if shapes don't match ``(N, 3)`` / ``(N, 4)`` for
                ``N = len(object_ids)``.
            KeyError: if any ``object_id`` is not registered in this sim.
        """
        n = len(object_ids)
        if n == 0:
            return
        if positions.shape != (n, 3):
            raise ValueError(f"positions must have shape ({n}, 3), got {positions.shape}")
        if orientations.shape != (n, 4):
            raise ValueError(f"orientations must have shape ({n}, 4), got {orientations.shape}")

        # Resolve every row up front so we raise before any partial write.
        row_of = self._store.row_of
        try:
            rows = np.fromiter(map(row_of.__getitem__, object_ids), np.int64, count=n)
        except KeyError as exc:
            raise KeyError(f"object_id {exc.args[0]} not registered in this simulation") from None

        if self._in_step:
            self._set_poses_buffered(rows, positions, orientations, preserve_velocity)
            return
        objects = self._sim_objects_dict
        for i, oid in enumerate(object_ids):
            objects[oid].set_pose_raw(positions[i], orientations[i], preserve_velocity=preserve_velocity)

    def _set_poses_buffered(
        self,
        rows: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        preserve_velocity: bool,
    ) -> None:
        """The buffered path of :meth:`set_poses`: the rows that moved are written as arrays.

        Movement is judged as ``SimObject.set_pose`` judges it (a squared
        distance above 1e-12 in position, or in orientation up to the sign
        of the quaternion). A STATIC object does not move, as there; an
        object with attachments or a physics body takes the per-object path,
        which propagates to the children and stamps the cache.
        """
        store = self._store
        delta = positions - store.pos[rows]
        moved = (delta * delta).sum(axis=1) > 1e-12
        old_orn = store.orn[rows]
        diff = orientations - old_orn
        orn_diff = (diff * diff).sum(axis=1)
        flipped = orn_diff > 1e-12
        if flipped.any():
            total = orientations + old_orn
            orn_sum = (total * total).sum(axis=1)
            orn_diff = np.where(orn_sum < orn_diff, orn_sum, orn_diff)
        moved |= orn_diff > 1e-12
        if not moved.any():
            return
        static = store.mode[rows] == MODE_STATIC
        if (moved & static).any():
            logger.warning(
                "Attempting to move static objects. Static objects (CollisionMode.STATIC) should not be moved. "
                "If these objects need to move, change collision_mode to NORMAL_3D."
            )
            moved &= ~static
        # Rows the array path cannot finish: a physics body (its cache is
        # stamped per object), an object with link attachments, or a carrier
        # whose riders are not all kinematic base attachments.
        special = moved & (~store.kinematic[rows] | store.per_object_pose[rows])
        if special.any():
            objects = self._sim_objects_dict
            for k in np.flatnonzero(special).tolist():
                obj = objects[int(store.object_id[rows[k]])]
                obj.set_pose_raw(positions[k], orientations[k], preserve_velocity=preserve_velocity)
            moved &= ~special
        base_children = self._base_children
        if not moved.any():
            return
        moved_rows = rows[moved]
        store.pos[moved_rows] = positions[moved]
        store.orn[moved_rows] = orientations[moved]
        store.preserve_velocity[moved_rows] = preserve_velocity
        moved_ids = store.object_id[moved_rows].tolist()
        self._pending_pose_ids.update(moved_ids)
        self._moved_this_step.update(moved_ids)
        if base_children:
            self._carry_as_arrays(moved_rows, preserve_velocity)

    def _carries_as_arrays(self, obj: SimObject) -> bool:
        """Whether every object riding on ``obj`` is a kinematic base attachment the array path can carry."""
        kinematic = self._kinematic_objects
        pending = [obj]
        while pending:
            carrier = pending.pop()
            for child in carrier.attached_objects:
                if child._attached_link_index >= 0 or child.object_id not in kinematic:
                    return False
                if child.attached_objects:
                    pending.append(child)
        return True

    def _carry_as_arrays(self, carrier_rows: np.ndarray, preserve_velocity: bool) -> None:
        """Move the base attachments of the carrier rows that just moved, level by level, as arrays.

        Each rider's world pose is its carrier's pose composed with its
        attachment offset, as ``p.multiplyTransforms`` composes them; the
        rows are written, and the riders are pending and moved like their
        carriers. A rider that carries something is a carrier of the next
        level.
        """
        store = self._store
        n = len(store)
        parent_col = store.parent_row[:n]
        level = carrier_rows
        while len(level):
            child_rows = np.flatnonzero(np.isin(parent_col, level))
            if len(child_rows) == 0:
                return
            parent_rows = parent_col[child_rows]
            pos, orn = store.attached_poses(child_rows, parent_rows)
            store.pos[child_rows] = pos
            store.orn[child_rows] = orn
            store.preserve_velocity[child_rows] = preserve_velocity
            child_ids = store.object_id[child_rows].tolist()
            self._pending_pose_ids.update(child_ids)
            self._moved_this_step.update(child_ids)
            level = child_rows

    def get_poses(self, object_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(positions, orientations)`` arrays for N objects (batch read).

        Counterpart of :meth:`set_poses`. Returns raw ndarrays for direct use
        in vectorized math, ROS message construction, or controller state.
        For single-object reads prefer :meth:`SimObject.get_pose` which returns
        a :class:`Pose` object.

        Reads each object's cached pose (no PyBullet API call), so values are
        always consistent with the two-phase semantics — buffered writes made
        during the current step are visible immediately.

        Args:
            object_ids: N object ids to read (output order matches input).

        Returns:
            Tuple ``(positions (N, 3), orientations (N, 4))`` as float64 arrays.

        Raises:
            KeyError: if any ``object_id`` is not registered in this sim.
        """
        row_of = self._store.row_of
        try:
            rows = np.fromiter((row_of[oid] for oid in object_ids), np.int64, count=len(object_ids))
        except KeyError as exc:
            raise KeyError(f"object_id {exc.args[0]} not registered in this simulation") from None
        return self._store.pos[rows].copy(), self._store.orn[rows].copy()

    # ------------------------------------------------------------------ #
    # Batch controller registration (Phase B opt-in)
    # ------------------------------------------------------------------ #

    def _flush_pending_poses(self, ids: Set[int]) -> None:
        """
        Two-phase step Phase 2: write the pending poses PyBullet must hold.

        Called once per step_once() right after the agent/plugin update loop and
        before stepSimulation(). The poses are the rows of the pose columns.
        With ``mirror: lazy`` only the rows PyBullet needs after every step
        are written (physics bodies, articulated bodies, link-attachment
        parents); the others are marked pending in PyBullet and written when
        something reads them there (:meth:`sync_mirror`). With ``physics``
        on, ``mirror: always``, a GUI or a plugin that asked for it, every
        row is written (:attr:`_mirror_always`).

        Args:
            ids: Snapshot of object_ids to flush. The caller owns the lifecycle
                of ``_pending_pose_ids`` (snapshot → reset → pass here); this
                method neither reads nor mutates that field, so call order
                between the two flush helpers is explicit at the call site.

        See docs/design/two-phase-step/spec.md (D2/D3b).
        """
        if not ids:
            return
        store = self._store
        rows = store.rows(ids)
        if len(rows) == 0:
            return
        if self._mirror_always:
            self._write_rows_to_pybullet(rows)
            return
        needs = store.needs_mirror[rows]
        store.mirror_dirty[rows[~needs]] = True
        if needs.any():
            self._write_rows_to_pybullet(rows[needs])

    def _register_link_attachment_parent(self, object_id: int) -> None:
        """Track an object whose link-level attachments need post-flush sync."""
        self._link_attachment_parent_ids.add(object_id)
        self._set_needs_mirror(object_id)
        obj = self._sim_objects_dict.get(object_id)
        if obj is not None:
            self._set_per_object_pose(obj)

    def _unregister_link_attachment_parent(self, object_id: int) -> None:
        """Stop tracking an object with no remaining link-level attachments."""
        self._link_attachment_parent_ids.discard(object_id)
        self._set_needs_mirror(object_id)
        obj = self._sim_objects_dict.get(object_id)
        if obj is not None:
            self._set_per_object_pose(obj)

    def _sync_link_attachments_after_pose_flush(self, ids: Set[int]) -> Set[int]:
        """
        Recompute link-level attachments after parent poses have been flushed.

        Link attachments read parent link poses via PyBullet FK. During
        ``step_once()``, parent base poses are buffered until Phase 2, so the
        normal ``Agent.update()`` call can see a one-step-old link transform
        when the base moved in the same step. Running this immediately after
        the parent pose flush keeps EE-attached objects visually locked to the
        current-frame link pose while preserving the two-phase write path.

        Returns:
            Object ids whose poses were updated by the attachment sync.
        """
        if not ids or not self._link_attachment_parent_ids:
            return set()
        all_synced_ids: Set[int] = set()
        to_sync = ids & self._link_attachment_parent_ids
        # Link-level attachment chains can require multiple passes:
        # parent Agent pose flush -> child Agent link sync -> grandchild sync.
        # Base-link descendants are handled by SimObject._propagate_to_attached()
        # within each pose update, so this loop only needs to revisit objects
        # that were newly flushed by link-level FK sync.
        while to_sync:
            for object_id in to_sync:
                obj = self._sim_objects_dict.get(object_id)
                if not isinstance(obj, Agent) or not obj.attached_objects:
                    continue
                obj.update_attached_objects_kinematics()

            synced_ids = self._pending_pose_ids
            self._pending_pose_ids = set()
            if not synced_ids:
                break
            self._flush_pending_poses(synced_ids)
            all_synced_ids.update(synced_ids)
            to_sync = synced_ids & self._link_attachment_parent_ids
        return all_synced_ids

    def _flush_aabb_and_grid(self, ids: Set[int], geometry_ids: Set[int] = frozenset()) -> None:
        """
        Two-phase step Phase 3: refresh the AABBs of the buffered objects.

        Runs after stepSimulation() and before collision_check() so that
        collision detection sees up-to-date geometry. Only kinematic objects
        are refreshed here — physics objects are handled by the dedicated
        post-stepSimulation loop in step_once() that walks ``_physics_objects``.

        A single-body object's box is computed from its pose, all of them in
        one array operation; an articulated body's is read from PyBullet. The
        spatial hash picks the new boxes up on its next sync (the collision
        check's).

        Args:
            ids: Snapshot of object_ids to refresh (same set passed to
                ``_flush_pending_poses``). Independent of ``_pending_pose_ids``
                so the two flush helpers have no implicit ordering coupling.
            geometry_ids: Objects whose links moved in the step
                (``notify_geometry_changed``): refreshed too, kinematic or
                ``STATIC``.
        """
        if geometry_ids:
            ids = ids | geometry_ids
        if not ids:
            return
        store = self._store
        rows = store.rows(ids)
        if len(rows) == 0:
            return
        refresh = store.kinematic[rows]
        if geometry_ids:
            refresh = refresh | np.isin(rows, store.rows(geometry_ids))
        rows = rows[refresh]
        if len(rows) == 0:
            return
        # Single kinematic bodies from their poses, as arrays; the rest one by one from PyBullet.
        from_pose = ~store.pybullet_aabb[rows] & ~np.isnan(store.local_half[rows, 0])
        for object_id in store.object_id[rows[~from_pose]].tolist():
            self._update_object_aabb(object_id)
        row_array = rows[from_pose]
        if len(row_array):
            pos = store.pos[row_array]
            store.set_aabbs_from_poses(row_array, pos, store.orn[row_array])
            self._record_origin_gaps(row_array, pos)

    def _body_aabb(self, obj: SimObject) -> Tuple[Sequence[float], Sequence[float]]:
        """The AABB of every link of ``obj``'s body, as PyBullet has them now.

        ``p.getAABB(body)`` alone is the base link's box. A body without child
        links (a mobile robot box, a tile, a pallet) costs that one call; an
        articulated body's box is the union of its link boxes, one call per
        link. A link without a collision shape (an elevator shaft, a tool or
        sensor frame) is a point at its frame for PyBullet and is left out:
        nothing there can collide. The links are read from PyBullet once per
        object.
        """
        body_id = obj.body_id
        client = self._client
        links = self._aabb_links.get(obj.object_id)
        if links is None:
            links = self._collider_links(body_id)
            self._aabb_links[obj.object_id] = links
        if not links:
            return p.getAABB(body_id, physicsClientId=client)
        (x0, y0, z0), (x1, y1, z1) = p.getAABB(body_id, links[0], physicsClientId=client)
        for link in links[1:]:
            (a0, b0, c0), (a1, b1, c1) = p.getAABB(body_id, link, physicsClientId=client)
            if a0 < x0:
                x0 = a0
            if b0 < y0:
                y0 = b0
            if c0 < z0:
                z0 = c0
            if a1 > x1:
                x1 = a1
            if b1 > y1:
                y1 = b1
            if c1 > z1:
                z1 = c1
        return (x0, y0, z0), (x1, y1, z1)

    def _collider_links(self, body_id: int) -> Tuple[int, ...]:
        """The links (-1 = base) of ``body_id`` that have a collision shape, for ``_body_aabb``.

        Empty when the body has no child links, or no link with a collider:
        its AABB is then the base link's, one ``getAABB()`` call, as before.
        """
        client = self._client
        num_links = p.getNumJoints(body_id, physicsClientId=client)
        if not num_links:
            return ()
        links = tuple(link for link in range(-1, num_links) if p.getCollisionShapeData(body_id, link, physicsClientId=client))
        return links if links != (-1,) else ()

    def notify_geometry_changed(self, obj: SimObject) -> None:
        """Tell the simulation that ``obj``'s links moved without its base pose being set.

        A joint moved by :meth:`Agent.set_joint_position`, by the joint
        interpolation of a kinematic agent or by restoring a checkpoint calls
        this already. Call it after moving a body's joints directly through
        PyBullet (``p.resetJointState``).

        The body counts as moved for the next collision check, and its
        broadphase AABB (the union of its link boxes) and spatial-hash cells
        follow the links. Outside a step that happens at once. Inside a step
        a kinematic or ``STATIC`` body is refreshed with the flushed poses
        before the collision check; a physics body is re-read after
        ``stepSimulation()`` anyway.

        Args:
            obj: The object whose links moved.
        """
        object_id = obj.object_id
        if object_id not in self._sim_objects_dict:
            return
        self._moved_this_step.add(object_id)
        if self.cached_aabb(object_id) is None:
            # Collision DISABLED (no cached box) or not yet initialised: the
            # box is read from PyBullet when it is needed.
            return
        if self._in_step:
            if object_id not in self._physics_objects:
                self._pending_geometry_ids.add(object_id)
            return
        self._update_object_aabb(object_id)

    def _update_object_aabb(self, object_id: int, update_grid: bool = True) -> None:
        """
        Recompute the AABB of one object now.

        A kinematic single-body object's box is computed from its pose and its
        local box (read from PyBullet once, at identity orientation, the first
        time). A physics body's and an articulated body's box is read from
        PyBullet. The spatial hash picks the change up on its next sync;
        ``update_grid`` is accepted for the callers that still pass it.

        Args:
            object_id: The object_id to update
            update_grid: Ignored: the hash follows the AABBs on its next sync.
        """
        obj = self._sim_objects_dict.get(object_id)
        store = self._store
        row = store.row_of.get(object_id)
        if obj is None or row is None:
            logger.warning(f"Cannot update AABB for object {object_id}: object not found")
            return

        try:
            if store.pybullet_aabb[row]:
                if self._aabb_links.get(object_id) == ():
                    # Hot path, one getAABB() call (no child links): _body_aabb() inlined
                    # on purpose, to save a method call for every moved object every
                    # step. Both read the same _aabb_links entry, so the two cannot
                    # disagree.
                    aabb = p.getAABB(obj.body_id, physicsClientId=self._client)
                else:
                    aabb = self._body_aabb(obj)
                store.set_aabb(row, aabb)
            else:
                if not store.local_box_known(row):
                    self._read_local_box(obj, row)
                pose = obj._cached_pose
                store.set_aabb_from_pose(row, pose.position, pose.orientation)
                aabb = store.aabb(row)
            self._record_origin_gap(obj, aabb)
            logger.debug(f"Updated AABB for object {object_id}")
        except (p.error, Exception):
            # In kinematics mode (physics=false), getAABB() may fail for newly
            # spawned bodies because PyBullet has not yet computed collision info.
            # This is harmless — the AABB will be populated on the first
            # performCollisionDetection() / stepSimulation() call.
            logger.debug(f"Deferred AABB init for object {object_id} (body {obj.body_id})")

    def _read_local_box(self, obj: SimObject, row: int) -> None:
        """Read the body's box in its own frame from PyBullet, at identity orientation, into the store.

        PyBullet's AABB of a body at any pose is this box transformed by the
        element-wise absolute rotation (``btTransformAabb``), a sphere's
        apart, whose extents do not rotate. A body not at identity
        orientation is put there for the read and back at once, velocity
        kept. The orientation it is put back at is the one read back, which
        is its rotation matrix turned into a quaternion again, about an ulp
        from the one it was given; that is why a simulation with physics on
        reads every box from PyBullet instead (``_register_object_movement_type``).
        """
        client = self._client
        body = obj.body_id
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
        at_identity = abs(orn[0]) + abs(orn[1]) + abs(orn[2]) < 1e-12
        if at_identity:
            aabb = p.getAABB(body, physicsClientId=client)
        else:
            velocity = p.getBaseVelocity(body, physicsClientId=client)
            p.resetBasePositionAndOrientation(body, pos, (0.0, 0.0, 0.0, 1.0), physicsClientId=client)
            try:
                aabb = p.getAABB(body, physicsClientId=client)
            finally:
                p.resetBasePositionAndOrientation(body, pos, orn, physicsClientId=client)
                p.resetBaseVelocity(body, velocity[0], velocity[1], physicsClientId=client)
        local_min = [aabb[0][k] - pos[k] for k in range(3)]
        local_max = [aabb[1][k] - pos[k] for k in range(3)]
        shapes = p.getCollisionShapeData(body, -1, physicsClientId=client)
        invariant = (
            len(shapes) == 1
            and shapes[0][2] == p.GEOM_SPHERE
            and tuple(shapes[0][5]) == (0.0, 0.0, 0.0)
            and tuple(shapes[0][6]) == (0.0, 0.0, 0.0, 1.0)
        )
        self._store.set_local_box(row, local_min, local_max, invariant)

    def _record_origin_gap(self, obj: SimObject, aabb: Tuple[Sequence[float], Sequence[float]]) -> None:
        """Note how far a pickable ``obj``'s base origin lies outside its AABB, if at all.

        A kinematic object's pose is its cached one; a physics object's is read
        from PyBullet, as its cache may still hold the pose of the last step.
        Objects that are not pickable are not tracked (see
        :meth:`_on_pickable_changed` for one that becomes pickable).
        """
        if not obj.pickable:
            self._origin_gaps.pop(obj.object_id, None)
            return
        if obj.is_kinematic:
            x, y, z = obj.get_pose().position
        else:
            (x, y, z), _ = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self._client)
        (x0, y0, z0), (x1, y1, z1) = aabb
        gap = max(x0 - x, x - x1, y0 - y, y - y1, z0 - z, z - z1)
        if gap > 0.0:
            self._origin_gaps[obj.object_id] = gap
        else:
            self._origin_gaps.pop(obj.object_id, None)

    def _record_origin_gaps(self, rows: np.ndarray, pos: np.ndarray) -> None:
        """:meth:`_record_origin_gap` for the rows just computed from ``pos``, as arrays."""
        store = self._store
        gaps = self._origin_gaps
        pickable = store.pickable[rows]
        if not pickable.any():
            if gaps:
                for object_id in store.object_id[rows].tolist():
                    gaps.pop(object_id, None)
            return
        gap = np.maximum(store.aabb_min[rows] - pos, pos - store.aabb_max[rows]).max(axis=1)
        outside = pickable & (gap > 0.0)
        for object_id, value in zip(store.object_id[rows[outside]].tolist(), gap[outside].tolist()):
            gaps[object_id] = value
        if gaps:
            for object_id in store.object_id[rows[~outside]].tolist():
                gaps.pop(object_id, None)

    def _on_pickable_changed(self, obj: SimObject) -> None:
        """Start or stop tracking ``obj``'s origin gap when its ``pickable`` flag changes.

        Called by the ``SimObject.pickable`` setter. An object that becomes
        pickable while standing still would otherwise have no gap recorded
        until it next moves.
        """
        object_id = getattr(obj, "object_id", None)
        if object_id is None:
            return
        row = self._store.row_of.get(object_id)
        if row is not None:
            self._store.pickable[row] = bool(obj.pickable)
        aabb = self.cached_aabb(object_id)
        if aabb is None:
            self._origin_gaps.pop(object_id, None)
            return
        self._record_origin_gap(obj, aabb)

    def _remove_object_aabb(self, object_id: int, update_grid: bool = True) -> None:
        """
        Forget the AABB of an object: it leaves the spatial hash on the next sync.
        Used when an object's collision is disabled or it is removed.

        Args:
            object_id: The object_id whose AABB to forget
            update_grid: Ignored: the hash follows the rows on its next sync.
        """
        self._origin_gaps.pop(object_id, None)
        row = self._store.row_of.get(object_id)
        if row is not None and self._store.has_aabb[row]:
            self._store.has_aabb[row] = False
            self._store.hash_dirty[row] = True
            logger.debug(f"Removed object {object_id} from AABB cache")

    def _coord_to_cell(self, coord: float) -> int:
        """Convert a world coordinate to a spatial grid cell index.

        Uses floor division so that negative coordinates map toward -inf
        (e.g. -0.1 / 2.0 → cell -1).

        Args:
            coord: World-space coordinate value (x, y, or z)

        Returns:
            Integer cell index along the given axis
        """
        # _get_overlapping_cells inlines this rounding on its hot path; change both together.
        return int(math.floor(coord / cast(float, self._cached_cell_size)))

    def set_collision_ignore_filter(self, ignore: Optional[CollisionIgnoreFilter]) -> None:
        """Exclude the touching pairs ``ignore(obj_a, obj_b)`` returns ``True`` for from collision results.

        An ignored pair is treated as not touching everywhere a collision
        shows up: it is not an active pair, not counted, not recoloured, not
        in the collision observation and raises no ``COLLISION_STARTED``. A
        pair that becomes ignored while active is resolved (``COLLISION_ENDED``)
        on the next collision check, and an ignored pair that stops being
        ignored is checked again on the next one, whether or not either body
        moved. So the filter may depend on state that changes at run time --
        which object carries which, say. It is applied on top of
        ``ignore_attached_collision``. ``None`` removes it.

        Ignored pairs skip the narrow phase, so the filter is called, on each
        collision check, for every broad-phase candidate pair involving a
        moved object, every active pair and every pair it ignored last time
        until they move apart; keep it cheap.
        """
        self._collision_ignore_filter = ignore
        # Stationary pairs are otherwise not rechecked until one of them moves.
        self._moved_this_step.update(self._sim_objects_dict)
        self._recheck_stationary_pairs = True

    def _is_ignored_collision(self, obj_a: SimObject, obj_b: SimObject) -> bool:
        """Whether two touching objects are excluded from collision results."""
        # Every object on one attachment tree -- an agent, its loads, a load's
        # loads -- belongs to one body: no contact among them is a collision.
        if self._params.ignore_attached_collision and obj_a._attachment_root() is obj_b._attachment_root():
            return True
        # An object's own rule for its contacts (an elevator ignores the
        # bodies standing on its platform).
        if obj_a._filters_own_contacts and obj_a._ignores_contact_with(obj_b):
            return True
        if obj_b._filters_own_contacts and obj_b._ignores_contact_with(obj_a):
            return True
        return self._collision_ignore_filter is not None and bool(self._collision_ignore_filter(obj_a, obj_b))

    def _collision_broadphase_margin(self) -> Tuple[float, float, float]:
        """Match the clearance used by the configured narrow phase, per world axis (x, y, z)."""
        if self._params.collision_detection_method in (
            CollisionDetectionMethod.CLOSEST_POINTS,
            CollisionDetectionMethod.HYBRID,
        ):
            margin = self._params.collision_margin
            if _is_scalar_margin(margin):
                margin = max(0.0, float(margin))
                return (margin, margin, margin)
            return _clearance_triple(margin)
        return (0.0, 0.0, 0.0)

    def _narrow_phase_clearance(self) -> float:
        """The getClosestPoints query distance: the margin, or the largest component of a per-axis one.

        A per-axis clearance acts in the broad phase only (each AABB axis is
        enlarged by that axis's clearance); the narrow phase judges the
        straight-line gap, so ``(m, m, m)`` behaves as ``m``.
        """
        margin = self._params.collision_margin
        if _is_scalar_margin(margin):
            return float(margin)
        return max(_clearance_triple(margin))

    def _aabbs_within_clearance(self, obj_id_a: int, obj_id_b: int, clearance: Tuple[float, float, float]) -> bool:
        """Whether two kinematic bodies' AABBs are within ``clearance`` of each other on every axis.

        The pair test of filter_aabb_pairs(), for a pair the narrow phase
        checks again without the broad phase offering it (an active pair, or
        any pair after a margin change). No z clearance with a NORMAL_2D body.
        """
        aabb_a = self.cached_aabb(obj_id_a)
        aabb_b = self.cached_aabb(obj_id_b)
        if aabb_a is None or aabb_b is None:
            return True  # nothing to judge by: leave it to the narrow phase
        modes = self._cached_collision_modes
        two_d = modes.get(obj_id_a) == CollisionMode.NORMAL_2D or modes.get(obj_id_b) == CollisionMode.NORMAL_2D
        for axis in range(3):
            margin = 0.0 if two_d and axis == 2 else clearance[axis]
            if aabb_a[1][axis] + margin < aabb_b[0][axis] or aabb_a[0][axis] - margin > aabb_b[1][axis]:
                return False
        return True

    def _closest_points_within_clearance(
        self,
        obj_id_a: int,
        obj_id_b: int,
        query_distance: float,
        recheck_clearance: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[bool, Optional[float]]:
        """Whether two bodies touch or are within the clearance, and the smallest signed distance found.

        ``recheck_clearance`` is given for a pair the broad phase did not
        offer (an active pair checked again): its AABBs must then pass the
        broad phase's per-axis test first, so a pair the broad phase would
        drop resolves here too. A pair the broad phase offered passed it.
        """
        if recheck_clearance is not None and not self._aabbs_within_clearance(obj_id_a, obj_id_b, recheck_clearance):
            return False, None
        body_a, body_b = self._sim_objects_dict[obj_id_a].body_id, self._sim_objects_dict[obj_id_b].body_id
        closest_points = p.getClosestPoints(body_a, body_b, distance=query_distance, physicsClientId=self._client)
        if not closest_points:
            return False, None
        distances = (float(point[8]) for point in closest_points)
        return True, min((d for d in distances if math.isfinite(d)), default=None)

    def set_collision_margin(self, margin: Union[float, Sequence[float]]) -> None:
        """Change the clearance and recheck stationary pairs on the next collision check.

        Args:
            margin: Metres, for every direction, or an ``(x, y, z)`` sequence
                with one clearance per world axis (see
                :attr:`SimulationParams.collision_margin`).

        Raises:
            ValueError: If the margin, or any component, is negative or not
                finite, or a sequence does not have three components.
        """
        margin = _normalize_margin(margin)
        if margin == self._params.collision_margin:
            return
        old_effective_margin = self._collision_broadphase_margin()
        self._params.collision_margin = margin
        if self._collision_broadphase_margin() == old_effective_margin:
            return
        self._rebuild_spatial_grid()
        self._moved_this_step.update(self._sim_objects_dict)
        self._recheck_stationary_pairs = True

    def _sync_grid_margin(self) -> bool:
        """Rebuild the grid if the effective margin changed without set_collision_margin().

        A margin changed on ``params`` directly (collision_margin, or a
        collision_detection_method with a different clearance) bypasses
        set_collision_margin(): rebuild with it, as that does, and mark every
        body moved so every pair -- stationary ones included -- is checked
        again. Returns whether it rebuilt.
        """
        if self._grid_margin is None or self._collision_broadphase_margin() == self._grid_margin:
            return False
        self._rebuild_spatial_grid()
        self._moved_this_step.update(self._sim_objects_dict)
        self._recheck_stationary_pairs = True
        return True

    def _set_grid_margin(self, margin: Tuple[float, float, float]) -> None:
        self._grid_margin = margin
        self._grid_half_margin = (margin[0] / 2, margin[1] / 2, margin[2] / 2)

    def _get_overlapping_cells(
        self, object_id: int, margin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> List[Tuple[int, int, int]]:
        """
        Get all grid cells that overlap with an object's AABB plus margin.

        A NORMAL_2D body's z extent is not enlarged: a pair with one needs
        real z overlap, and the cell holding that overlap is one both register.

        Args:
            object_id: The object_id to get overlapping cells for
            margin: Per-side AABB enlargement in meters for each world axis
                (x, y, z); z is not enlarged for a NORMAL_2D body

        Returns:
            List of cell tuples (x, y, z) that overlap with the object's AABB;
            empty before the hash has a cell size or the object an AABB.
        """
        row = self._store.row_of.get(object_id)
        if row is None:
            return []
        return self._store.cells_of(row, margin)

    def _register_object_movement_type(self, obj: SimObject) -> None:
        """
        Register (or re-register) an object's movement type for optimization.

        This method is **idempotent**: it first clears any existing movement-type
        entries for the object, then classifies it afresh.  Safe to call both on
        initial add and on subsequent mode changes.

        Movement types:
        - STATIC: Never moves (registered via obj.is_static)
        - PHYSICS: Moves via physics engine (mass > 0, not kinematic)
        - KINEMATIC: Moves via set_pose() (mass == 0 or is_kinematic=True)

        Note: Collision mode (DISABLED) is handled in collision system methods,
              not here (this only handles movement classification).

        Args:
            obj: The SimObject instance to register
        """
        object_id = obj.object_id

        # Idempotent: clear previous classification (no-op on first call)
        self._physics_objects.discard(object_id)
        self._kinematic_objects.discard(object_id)

        # Register based on movement type (not collision mode)
        # Note: _static_collision_objects is managed by collision-domain methods,
        # not here. This only handles _physics_objects / _kinematic_objects.
        is_physics = False
        if obj.is_static:
            # STATIC: Never moves (no movement tracking needed)
            logger.debug(f"Object {object_id} registered as static (movement type)")
        else:
            # Dynamic objects: Determine if physics-enabled or kinematic
            is_physics = obj.mass > 0 and not obj.is_kinematic

            if is_physics:
                self._physics_objects.add(object_id)
                logger.debug(f"Object {object_id} registered as physics-enabled (movement type)")
            else:
                self._kinematic_objects.add(object_id)
                logger.debug(f"Object {object_id} registered as kinematic (movement type)")

        # A physics body's AABB is read from PyBullet after each step, an
        # articulated body's from its links; a kinematic single body's is
        # computed from its pose. With physics on, every body's is read from
        # PyBullet: reading a rotated body's local box means putting it at
        # identity and back, which leaves its orientation an ulp away, and a
        # contact with a dynamic body would amplify that.
        row = self._store.row_of.get(object_id)
        if row is not None:
            self._store.physics[row] = is_physics
            self._store.kinematic[row] = object_id in self._kinematic_objects
            self._store.pybullet_aabb[row] = (
                is_physics or bool(self._params.physics) or self._aabb_links.get(object_id, ()) != ()
            )
            self._set_needs_mirror(object_id)
            carrier = obj._attached_to
            if carrier is not None:
                self._set_per_object_pose(carrier)

    def _update_object_collision_mode(self, object_id: int, old_mode: CollisionMode, new_mode: CollisionMode) -> None:
        """
        Update collision system when object's collision mode changes.

        Called by SimObject.set_collision_mode() when mode is changed.
        Handles all cache updates (AABB, spatial grid, movement tracking).

        Args:
            object_id: Object ID
            old_mode: Previous CollisionMode
            new_mode: New CollisionMode
        """
        # Early return if mode hasn't changed
        if old_mode == new_mode:
            return

        obj = self._sim_objects_dict.get(object_id)
        if obj is None:
            logger.warning(f"Cannot update collision mode for object {object_id}: not found")
            return

        # Re-register movement type (physics / kinematic classification)
        self._register_object_movement_type(obj)

        # Update collision mode cache
        self._cached_collision_modes[object_id] = new_mode
        row = self._store.row_of.get(object_id)
        if row is not None:
            # The registration (which cells, and whether at all) follows the mode on the next sync.
            self._store.mode[row] = _MODE_CODE[new_mode]
            self._store.hash_dirty[row] = True

        # Handle collision system updates based on mode transition
        # _static_collision_objects and _dynamic_collision_objects are managed here (collision domain)
        # Structure: First check old_mode, then check new_mode within each case

        if old_mode == CollisionMode.DISABLED:
            # DISABLED -> *: Classification + AABB/grid handled inside
            self._add_object_to_collision_system(object_id)

        elif old_mode == CollisionMode.STATIC:
            # STATIC -> *: STATIC already has AABB/grid
            self._static_collision_objects.discard(object_id)
            if new_mode == CollisionMode.DISABLED:
                # STATIC -> DISABLED: Remove from collision system
                self._remove_object_from_collision_system(object_id)
            else:
                # STATIC -> NORMAL_3D/2D: Add to dynamic set, re-register (a
                # NORMAL_2D body registers different z cells), mark as moved
                self._dynamic_collision_objects.add(object_id)
                self._moved_this_step.add(object_id)

        else:
            # NORMAL_3D/2D -> *: Currently in dynamic set
            if new_mode == CollisionMode.DISABLED:
                # NORMAL -> DISABLED: Remove from collision system and dynamic set
                self._dynamic_collision_objects.discard(object_id)
                self._remove_object_from_collision_system(object_id)
            elif new_mode == CollisionMode.STATIC:
                # NORMAL -> STATIC: Remove from dynamic set, add to static set,
                # keep the AABB. Re-register: a NORMAL_2D body registers
                # different z cells from a STATIC one. (A NORMAL_3D body's
                # registration is the same, so for it this changes nothing.)
                # Mark it moved, as STATIC -> NORMAL does: a NORMAL_2D body
                # now gets the z clearance, so a body just above or below can
                # become a pair without either of them moving.
                self._dynamic_collision_objects.discard(object_id)
                self._static_collision_objects.add(object_id)
                self._moved_this_step.add(object_id)
            else:
                # NORMAL_3D <-> NORMAL_2D: the two register different z cells,
                # so re-register (the mode change above did); mark as moved
                self._moved_this_step.add(object_id)

        logger.info(f"Updated collision mode for object {object_id}: {old_mode.value} -> {new_mode.value}")

    def _add_object_to_collision_system(self, object_id: int) -> None:
        """
        Add object to collision detection system.

        Classifies the object by its current ``collision_mode`` into the
        appropriate collision set / dict, then initialises AABB cache,
        spatial grid entry, and marks the object as moved.

        Called when an object is first added (``add_object``) or when its
        collision mode transitions from DISABLED to an enabled mode.

        Note: Must NOT be called for DISABLED collision mode objects.
              The caller is responsible for handling DISABLED separately.

        Args:
            object_id: Object ID to add
        """
        obj = self._sim_objects_dict.get(object_id)
        if obj is None:
            logger.warning(f"Cannot add object {object_id} to collision system: not found")
            return

        # Remove from disabled set (transitioning from DISABLED to enabled mode)
        self._disabled_collision_objects.discard(object_id)

        # Classify into collision-domain set based on current mode
        mode = obj.collision_mode
        if mode == CollisionMode.STATIC:
            self._static_collision_objects.add(object_id)
        else:
            # NORMAL_3D or NORMAL_2D
            self._dynamic_collision_objects.add(object_id)

        self._update_object_aabb(object_id, update_grid=True)
        self._moved_this_step.add(object_id)
        logger.debug(f"Added object {object_id} to collision system (mode={mode.value})")

    def _remove_object_from_collision_system(self, object_id: int) -> None:
        """
        Remove object from collision detection system.

        Removes AABB cache, spatial grid entry, and movement tracking.
        If object is transitioning to DISABLED mode, adds to _disabled_collision_objects.
        If object is being completely removed, cleans up _disabled_collision_objects.

        Used when object is removed or when collision mode changes to disable collision.

        Args:
            object_id: Object ID to remove
        """
        obj = self._sim_objects_dict.get(object_id)
        if obj is not None and obj.collision_mode == CollisionMode.DISABLED:
            # Object exists and is DISABLED: Add to disabled set
            self._disabled_collision_objects.add(object_id)
            logger.debug(f"Removed object {object_id} from collision system (now DISABLED)")
        else:
            # Object removed or transitioning to non-DISABLED mode: Remove from disabled set
            self._disabled_collision_objects.discard(object_id)
            logger.debug(f"Removed object {object_id} from collision system")

        self._remove_object_aabb(object_id)
        self._moved_this_step.discard(object_id)

        # The reverse index limits removal to this object's active neighbors.
        for pair in tuple(self._active_pairs_by_object.get(object_id, ())):
            self._discard_active_collision_pair(pair)
        self._last_collision_check_metadata = None

    def _add_active_collision_pair(self, pair: Tuple[int, int]) -> None:
        self._active_collision_pairs.add(pair)
        for object_id in pair:
            self._active_pairs_by_object.setdefault(object_id, set()).add(pair)

    def _discard_active_collision_pair(self, pair: Tuple[int, int]) -> None:
        self._active_collision_pairs.discard(pair)
        self._collision_pair_observations.pop(pair, None)
        for object_id in pair:
            pairs = self._active_pairs_by_object.get(object_id)
            if pairs is not None:
                pairs.discard(pair)
                if not pairs:
                    del self._active_pairs_by_object[object_id]

    def _clear_active_collision_state(self) -> None:
        """Clear active pairs, their indexes, and the last check result together."""
        self._active_collision_pairs.clear()
        self._active_pairs_by_object.clear()
        self._collision_pair_observations.clear()
        self._ignored_collision_pairs.clear()
        self._last_collision_check_metadata = None

    def set_collision_spatial_hash_cell_size_mode(
        self, mode: Optional[SpatialHashCellSizeMode] = None, cell_size: Optional[float] = None
    ) -> float:
        """
        Set collision detection spatial hash cell size mode and recalculate/rebuild grid.

        This method allows dynamic mode switching and always recalculates the cell_size
        based on the spatial_hash_cell_size_mode setting (or override via parameters).

        Args:
            mode: Override spatial_hash_cell_size_mode for this calculation (optional)
            cell_size: Override spatial_hash_cell_size when mode=CONSTANT (optional)

        Returns:
            The calculated cell_size (always > 0, falls back to 1.0 if no AABBs available)

        Example::

            # Recalculate with current mode
            sim_core.set_collision_spatial_hash_cell_size_mode()

            # Switch to constant mode with specific size
            sim_core.set_collision_spatial_hash_cell_size_mode(
                mode=SpatialHashCellSizeMode.CONSTANT,
                cell_size=3.0
            )

            # Switch to auto_adaptive mode
            sim_core.set_collision_spatial_hash_cell_size_mode(
                mode=SpatialHashCellSizeMode.AUTO_ADAPTIVE
            )
        """
        # Update params if mode or cell_size is provided
        if mode is not None:
            old_mode = self._params.spatial_hash_cell_size_mode
            self._params.spatial_hash_cell_size_mode = mode
            logger.debug(f"Updated mode: {old_mode} -> {mode}")

        if cell_size is not None:
            old_size = self._params.spatial_hash_cell_size
            self._params.spatial_hash_cell_size = cell_size

            # Warn if cell_size is provided but mode is not CONSTANT
            current_mode = self._params.spatial_hash_cell_size_mode
            if current_mode != SpatialHashCellSizeMode.CONSTANT:
                logger.warning(
                    f"cell_size parameter ({cell_size}) will be ignored in {current_mode.value} mode. "
                    "To use a fixed cell_size, set mode=SpatialHashCellSizeMode.CONSTANT"
                )
            else:
                logger.debug(f"Updated cell_size: {old_size} -> {cell_size}")

        mode = self._params.spatial_hash_cell_size_mode
        old_cell_size = self._cached_cell_size

        # Calculate cell_size based on mode
        if mode == SpatialHashCellSizeMode.CONSTANT:
            # CONSTANT mode: use user-provided value
            if self._params.spatial_hash_cell_size is None:
                logger.error(
                    "spatial_hash_cell_size_mode=CONSTANT requires spatial_hash_cell_size to be set. "
                    "Falling back to default 1.0m"
                )
                cell_size = 1.0
            else:
                cell_size = self._params.spatial_hash_cell_size
            self._cached_cell_size = cell_size
            logger.info(f"Collision spatial hash: Using constant cell size: {cell_size:.3f}m")

        elif mode in [SpatialHashCellSizeMode.AUTO_ADAPTIVE, SpatialHashCellSizeMode.AUTO_INITIAL]:
            # AUTO modes: Calculate from median AABB extent
            cell_size = self._calculate_cell_size_from_aabbs()
            if cell_size is not None:
                self._cached_cell_size = cell_size
                logger.info(
                    f"Collision spatial hash: {'Recalculated' if old_cell_size else 'Calculated'} cell size ({mode.value}): "
                    f"{cell_size:.3f}m (num_objects: {len(self._store)})"
                )
            else:
                cell_size = 1.0
                self._cached_cell_size = cell_size
                logger.warning("Collision spatial hash: No AABBs available, using default cell size: 1.0m")
        else:
            logger.error(f"Unknown spatial_hash_cell_size_mode: {mode}, using default 1.0m")
            cell_size = 1.0
            self._cached_cell_size = cell_size

        # Rebuild spatial grid with new cell_size
        self._rebuild_spatial_grid()

        return cell_size

    def add_object(self, obj: SimObject) -> None:
        """
        Add an object to simulation and register it with all necessary caches.

        This method centralizes object registration logic for consistency and maintainability.
        Called automatically by SimObject.__init__() when sim_core is provided.

        Args:
            obj: The SimObject instance to add

        Example::

            # Manual addition (not recommended - SimObject does this automatically)
            obj = SimObject(body_id=body_id, sim_core=None)
            sim_core.add_object(obj)

            # Automatic addition (recommended)
            obj = SimObject(body_id=body_id, sim_core=sim_core)  # Calls add_object() internally
        """
        # Add to sim_objects list and dict
        if obj in self._sim_objects:
            logger.warning(f"Object {obj.object_id} already added to simulation")
            return

        self._sim_objects.append(obj)
        self._sim_objects_dict[obj.object_id] = obj
        self._note_monitor_entities_changed()
        links = self._collider_links(obj.body_id)
        self._aabb_links[obj.object_id] = links
        generation = self._store.generation
        row = self._store.add(
            obj.object_id,
            mode=_MODE_CODE[obj.collision_mode],
            physics=False,  # set by _register_object_movement_type below
            pickable=bool(obj.pickable),
            pybullet_aabb=links != () or bool(self._params.physics),
            pos=obj._cached_pose.position,
            orn=obj._cached_pose.orientation,
        )
        if self._store.generation != generation:
            self._bind_pose_rows()  # the columns were reallocated: every view is stale
        else:
            obj._bind_rows(self._store, row)
        if obj._needs_update:
            self._updatable.append(obj)
        if isinstance(obj, Agent):
            obj._bind_plugins_to_batches()

        # Add to agents list if this is an Agent instance (for O(M) iteration)
        if isinstance(obj, Agent):
            self._agents.append(obj)

        # Store collision mode in cache
        self._cached_collision_modes[obj.object_id] = obj.collision_mode

        # Register movement type based on collision mode
        self._register_object_movement_type(obj)

        # Add to collision system based on collision mode
        if obj.collision_mode == CollisionMode.DISABLED:
            # DISABLED: No collision detection at all
            self._disabled_collision_objects.add(obj.object_id)
        else:
            # STATIC / NORMAL_3D / NORMAL_2D: classification + AABB/grid
            self._add_object_to_collision_system(obj.object_id)

        # Trigger cell_size recalculation in auto_adaptive mode (deferred during batch spawn)
        if self._params.spatial_hash_cell_size_mode == SpatialHashCellSizeMode.AUTO_ADAPTIVE:
            if not self._batch_spawning:
                self.set_collision_spatial_hash_cell_size_mode()

        # Save original color once when object is added (for collision visualization)
        if obj.body_id not in self._robot_original_colors:
            try:
                # Try to get current visual color, fallback to default if not available
                visual_data = p.getVisualShapeData(obj.body_id, physicsClientId=self._client)
                if visual_data:
                    self._robot_original_colors[obj.body_id] = list(visual_data[0][7])  # RGBA color
                else:
                    self._robot_original_colors[obj.body_id] = [0.0, 0.0, 0.0, 1.0]
            except (p.error, IndexError):
                self._robot_original_colors[obj.body_id] = [0.0, 0.0, 0.0, 1.0]

        logger.debug(f"Added object {obj.object_id} (body {obj.body_id}) to simulation")

        # --- Lifecycle events ---
        # OBJECT_SPAWNED is safe here: the base SimObject is fully constructed by
        # the time add_object() runs. AGENT_SPAWNED is emitted at the END of
        # Agent.__init__ instead — add_object() runs mid-Agent-construction (from
        # super().__init__()), so emitting it here would expose a half-built Agent.
        self.events.emit(SimEvents.OBJECT_SPAWNED, obj=obj)

    def remove_object(self, obj: SimObject) -> None:
        """
        Remove an object from simulation and clean up all caches.

        Args:
            obj: The SimObject instance to remove

        Example::

            obj = sim_core.spawn_robot(...)
            # ... later ...
            sim_core.remove_object(obj)
        """
        obj_id = obj.object_id

        # Remove from sim_objects list and dict
        try:
            self._sim_objects.remove(obj)
        except ValueError:
            logger.warning(f"Object {obj_id} not found in sim_objects")
            return

        self._sim_objects_dict.pop(obj_id, None)
        self._note_monitor_entities_changed()
        if self._selected_entity_id == obj_id:
            self._set_selected_entity(None, source="system")

        # Remove from agents list if this is an Agent instance
        if isinstance(obj, Agent):
            try:
                self._agents.remove(obj)
            except ValueError:
                pass  # Already removed or not in list
            for tree, _ in list(self._behavior_tree_callbacks):
                if getattr(tree, "agent", None) is obj:
                    self.unregister_behavior_tree(tree)

        # Remove from movement tracking sets
        self._moved_this_step.discard(obj_id)
        self._physics_objects.discard(obj_id)
        self._kinematic_objects.discard(obj_id)
        self._static_collision_objects.discard(obj_id)
        self._disabled_collision_objects.discard(obj_id)
        self._origin_gaps.pop(obj_id, None)
        self._aabb_links.pop(obj_id, None)
        self._pending_geometry_ids.discard(obj_id)

        # Remove from collision mode cache
        self._cached_collision_modes.pop(obj_id, None)

        # Remove from collision system (AABB, spatial grid)
        self._remove_object_from_collision_system(obj_id)
        # It leaves its attachment tree: whatever rode on it is detached, in
        # place, at the top of its own tree (two former riders are a pair for
        # the broad phase again), and its carrier no longer lists it.
        for rider in list(obj.attached_objects):
            obj.detach_object(rider)
        carrier = obj._attached_to
        if carrier is not None:
            carrier.detach_object(obj)
        for rider_id in self._base_children.pop(obj_id, []):
            self._base_parent_of.pop(rider_id, None)
            rider_row = self._store.row_of.get(rider_id)
            if rider_row is not None:
                self._store.parent_row[rider_row] = -1
        if obj_id in self._base_parent_of:
            self._note_base_attachment(obj_id, None, None)
        if obj_id in self._store.row_of:
            obj._unbind_rows()
            relocated = self._store.remove(obj_id)
            if relocated is not None:
                moved_obj = self._sim_objects_dict.get(relocated)
                if moved_obj is not None:
                    moved_obj._bind_rows(self._store, self._store.row_of[relocated])
        try:
            self._updatable.remove(obj)  # whatever its flag says now
        except ValueError:
            pass
        else:
            self._updatable_left.add(obj)
        if isinstance(obj, Agent):
            obj._release_plugins_from_batches()

        # Remove from dynamic collision set
        self._dynamic_collision_objects.discard(obj_id)

        # --- Lifecycle events ---
        self.events.emit(SimEvents.OBJECT_REMOVED, obj=obj)
        if isinstance(obj, Agent):
            self.events.emit(SimEvents.AGENT_REMOVED, agent=obj)

        # Remove original color cache (prevents stale color on body_id reuse)
        self._robot_original_colors.pop(obj.body_id, None)

        # Remove PyBullet body
        if obj.body_id is not None:
            try:
                p.removeBody(obj.body_id, physicsClientId=self._client)
            except p.error:
                logger.warning(f"Failed to remove PyBullet body {obj.body_id}")

        # Sync registered managers: remove object if tracked
        for mgr in self._registered_managers:
            if obj_id in mgr.object_ids:
                mgr.remove_object(obj)

        # Trigger cell_size recalculation in auto_adaptive mode
        if self._params.spatial_hash_cell_size_mode == SpatialHashCellSizeMode.AUTO_ADAPTIVE:
            self.set_collision_spatial_hash_cell_size_mode()

        logger.info(f"Removed object {obj_id} (body {obj.body_id}) from simulation")
        if self._state_recorder is not None:
            self._record_state_input("remove_object", {"key": obj.name})

    def configure_visualizer(
        self,
        enable_structure_transparency: Optional[bool] = None,
        enable_shadows: Optional[bool] = None,
    ) -> None:
        """
        Configure PyBullet visualizer settings with keyboard control.

        Args:
            enable_structure_transparency: Initial state for structure transparency (None=use config)
            enable_shadows: Enable shadows (None=use config)

        Custom keyboard shortcuts (active during simulation):
        - Press SPACE to pause/play simulation
        - Press 't' to toggle structure transparency ON/OFF

        Camera controls (always active in GUI mode):
        - Right-drag to pan camera
        - Press arrow keys to pan camera horizontally
        - Press '=' to zoom in, '-' to zoom out
        - Press 'o' for top-down view

        PyBullet built-in keyboard shortcuts (always available):
        - Press 'w' to toggle wireframe / collision shape display
        - Press 'g' to toggle grid display
        - Press 'j' to toggle joint axes display
        """
        if not self._params.gui:
            return

        # Use config values if parameters are None
        if enable_structure_transparency is None:
            enable_structure_transparency = self._params.enable_structure_transparency
        if enable_shadows is None:
            enable_shadows = self._params.enable_shadows

        # Save original colors of all visual shapes ONCE
        if not self._original_visual_colors:
            self._save_original_visual_colors()

        # Store initial states
        self._structure_transparent = enable_structure_transparency

        # Configure shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if enable_shadows else 0, physicsClientId=self._client)
        self.configure_gui_lighting()
        if self._lighting_controls_enabled:
            self._install_gui_lighting_controls()

        # Apply initial collision shape visibility from config
        if self._params.enable_collision_shapes:
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1, physicsClientId=self._client)

        # Apply initial structure transparency
        self._set_structure_transparency(self._structure_transparent)

        # Enable keyboard event handling in step_once()
        self._keyboard_events_registered = True

        # Auto-setup camera from params (creates CameraController if interactive)
        if self._params.camera_config and not self._camera_configured:
            self.setup_camera()

        # Always create CameraController in GUI mode so right-drag/zoom/topdown
        # work even when no camera_config is specified in YAML.
        if self._camera_controller is None:
            self._camera_controller = CameraController(client_id=self._client)

        logger.info(
            "Visualizer configured: transparency=%s, shadows=%s",
            enable_structure_transparency,
            enable_shadows,
        )
        logger.info("Keyboard controls registered: SPACE=pause, t=transparency, arrows/right-drag=pan, +/-=zoom, o=top-down")
        print("\n[KEYBOARD CONTROLS]")
        print("  Custom:")
        print("    SPACE       pause / resume simulation")
        print("    t           toggle structure transparency " f"(current: {'ON' if self._structure_transparent else 'OFF'})")
        if len(self._static_collision_objects) > 100:
            print(
                f"             ⚠️  Warning: {len(self._static_collision_objects)} static objects detected - "
                "toggling may be slow (not recommended)"
            )
        print("  Camera:")
        print("    right-drag  pan camera")
        print("    arrows      pan camera horizontally")
        print("    = / -       zoom in / out")
        print("    o           top-down view")
        print("  PyBullet built-in:")
        print("    w           toggle wireframe / collision shapes")
        print("    g           toggle grid")
        print("    j           toggle joint axes")

    @property
    def _lighting_controls_enabled(self) -> bool:
        """Whether native PyBullet sliders are enabled for GUI lighting."""
        return bool(self._params.lighting_config.get("enable_controls", False))

    def configure_gui_lighting(self, lighting_config: Optional[Dict[str, Any]] = None) -> None:
        """Apply optional PyBullet GUI main-light and shadow-map settings.

        ``lighting_config`` accepts ``light_position`` (three coordinates),
        ``shadow_map_world_size`` and ``shadow_map_resolution``. The normal
        PyBullet GUI has one main light; this does not model USD light prims.
        """
        if not self._params.gui:
            return
        config = self._params.lighting_config if lighting_config is None else lighting_config
        kwargs: Dict[str, Any] = {"physicsClientId": self._client}
        position, world_size, resolution = self._validated_gui_lighting_values(config)
        if position is not None:
            kwargs["lightPosition"] = position
        if world_size is not None:
            kwargs["shadowMapWorldSize"] = world_size
        if resolution is not None:
            kwargs["shadowMapResolution"] = resolution
        if len(kwargs) > 1:
            p.configureDebugVisualizer(**kwargs)

    @staticmethod
    def _validated_gui_lighting_values(
        config: Dict[str, Any], *, defaults: bool = False
    ) -> tuple[list[float] | None, int | None, int | None]:
        """Return validated GUI-lighting values shared by setup and sliders."""
        position = config.get("light_position", [4.0, -4.0, 8.0] if defaults else None)
        if position is None and defaults:
            position = [4.0, -4.0, 8.0]
        if position is not None:
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                raise ValueError("lighting.light_position must be a three-element list")
            position = [float(value) for value in position]

        def positive_integer(key: str, default: int | None) -> int | None:
            # An explicit null takes the default too, as light_position does.
            value = config.get(key)
            if value is None:
                value = default
            if value is None:
                return None
            try:
                value = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"lighting.{key} must be a positive integer") from error
            if value <= 0:
                raise ValueError(f"lighting.{key} must be positive")
            return value

        return (
            position,
            positive_integer("shadow_map_world_size", 12 if defaults else None),
            positive_integer("shadow_map_resolution", 2048 if defaults else None),
        )

    def _install_gui_lighting_controls(self) -> Dict[str, int]:
        """Create the optional built-in-panel sliders once per PyBullet world; return their ids."""
        if self._lighting_control_ids is not None:
            return self._lighting_control_ids
        config = self._params.lighting_config
        position, world_size, resolution = self._validated_gui_lighting_values(config, defaults=True)
        if position is None or world_size is None or resolution is None:
            raise ValueError("GUI lighting controls need a light position and shadow map size and resolution")
        self._lighting_control_ids = {
            "x": p.addUserDebugParameter("Light X", -30.0, 30.0, float(position[0]), physicsClientId=self._client),
            "y": p.addUserDebugParameter("Light Y", -30.0, 30.0, float(position[1]), physicsClientId=self._client),
            "z": p.addUserDebugParameter("Light Z", 0.5, 30.0, float(position[2]), physicsClientId=self._client),
            "shadow_size": p.addUserDebugParameter(
                "Shadow map world size", 2.0, 60.0, float(world_size), physicsClientId=self._client
            ),
            "shadow_resolution": p.addUserDebugParameter(
                "Shadow map resolution", 256.0, 4096.0, float(resolution), physicsClientId=self._client
            ),
        }
        self._lighting_control_values = None
        return self._lighting_control_ids

    def _update_gui_lighting_controls(self) -> None:
        """Apply changed native lighting sliders on the simulation thread."""
        if not self._params.gui or not self._lighting_controls_enabled:
            return
        control_ids = self._install_gui_lighting_controls()
        values = tuple(
            p.readUserDebugParameter(control_id, physicsClientId=self._client) for control_id in control_ids.values()
        )
        if values == self._lighting_control_values:
            return
        self._lighting_control_values = values
        self.configure_gui_lighting(
            {
                "light_position": values[:3],
                "shadow_map_world_size": int(values[3]),
                "shadow_map_resolution": int(values[4]),
            }
        )

    def _save_original_visual_colors(self) -> None:
        """
        Save original colors of all visual shapes for fast restoration.
        Called once during configure_visualizer().
        """
        num_bodies = p.getNumBodies(physicsClientId=self._client)
        for body_id in range(num_bodies):
            visual_data = p.getVisualShapeData(body_id, physicsClientId=self._client)
            for shape in visual_data:
                link_index = shape[1]
                rgba = shape[7]  # Original RGBA color
                key = (body_id, link_index)
                self._original_visual_colors[key] = rgba

        logger.info(f"Saved original colors for {len(self._original_visual_colors)} visual shapes")

    def _handle_keyboard_events(self) -> None:
        """
        Handle keyboard events for pausing simulation and toggling transparency.
        Called during simulation step.
        """
        if not self._params.gui:
            return

        try:
            # Get keyboard events
            keys = p.getKeyboardEvents(physicsClientId=self._client)
        except p.error:
            # PyBullet disconnected (window closed), stop handling events
            return

        # Space key (ASCII 32) - toggle pause/play
        if ord(" ") in keys and keys[ord(" ")] & p.KEY_WAS_TRIGGERED:
            if self.is_paused:
                self.resume()
            else:
                self.pause()
            print(f"\n[PAUSE] Simulation: {'PAUSED' if self.is_paused else 'PLAYING'}")

        # '.' advances exactly one outer simulation step while paused.  It
        # records a request only; it never calls step_once() recursively.
        if ord(".") in keys and keys[ord(".")] & p.KEY_WAS_TRIGGERED:
            self.request_single_step(source="keyboard")

        # Camera controller (interactive mode)
        if self._camera_controller is not None:
            try:
                mouse_events = p.getMouseEvents(physicsClientId=self._client)
            except p.error:
                mouse_events = []
            self._camera_controller.update(keys, mouse_events=mouse_events)
            if self._gui_follow_enabled and self._selected_entity_id is not None:
                selected = self._sim_objects_dict.get(self._selected_entity_id)
                if selected is not None:
                    self._camera_controller.follow_target(tuple(selected.get_pose().position))
            control_state = keys.get(p.B3G_CONTROL, 0)
            control_down = bool(control_state & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED))
            if control_down and self.is_paused and self._selected_entity_id is not None:
                for event_type, _mouse_x, _mouse_y, button, state in mouse_events:
                    if event_type == _MOUSE_BUTTON_EVENT and button == _MOUSE_BUTTON_LEFT and state != p.KEY_WAS_RELEASED:
                        self._viewport_drag_entity_id = self._selected_entity_id
                        break
            self._select_viewport_entity(mouse_events)
        # 't' key (ASCII 116) - toggle structure transparency
        if ord("t") in keys and keys[ord("t")] & p.KEY_WAS_TRIGGERED:
            num_structures = len(self._static_collision_objects)
            if num_structures > 100:
                print(f"\n[TOGGLE] ⚠️  Warning: Toggling transparency for {num_structures} static objects...")
                print("         This may take several seconds (not recommended for large scenes)")
                print("         Consider setting 'enable_structure_transparency' in config.yaml instead")

            self._structure_transparent = not self._structure_transparent
            self._set_structure_transparency(self._structure_transparent)
            print(f"\n[TOGGLE] Structure transparency: {'ON' if self._structure_transparent else 'OFF'}")

    def _set_structure_transparency(self, transparent: bool) -> None:
        """
        Set transparency of static bodies (structures).
        Uses pre-saved colors with modified alpha channel.
        Disables rendering during batch update for ~175x speedup in GUI mode.

        Args:
            transparent: True to make static objects semi-transparent, False for opaque
        """
        if not self._params.gui:
            return

        alpha = 0.3 if transparent else 1.0

        # Use _static_collision_objects — world-loaded objects default to
        # CollisionMode.STATIC, so this covers walls, floors, furniture, etc.
        structure_body_ids = {
            self._sim_objects_dict[oid].body_id for oid in self._static_collision_objects if oid in self._sim_objects_dict
        }
        logger.info(f"[TRANSPARENCY] Applying alpha={alpha} to {len(structure_body_ids)} static objects...")

        processed = 0
        # Disable rendering during batch update to avoid per-call OpenGL re-render
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._client)
        try:
            for key, rgba in self._original_visual_colors.items():
                body_id, link_index = key

                # Only process structure bodies (skip agents/robots)
                if body_id not in structure_body_ids:
                    continue

                try:
                    # Apply new alpha to the original color
                    p.changeVisualShape(
                        body_id, link_index, rgbaColor=[rgba[0], rgba[1], rgba[2], alpha], physicsClientId=self._client
                    )
                    processed += 1
                except Exception:
                    pass
        finally:
            # Always re-enable rendering, even if an error occurred
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._client)
            # Re-assert GUI panel hidden
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)

        logger.info(f"[TRANSPARENCY] Complete: {processed} visual shapes updated")
        logger.info(f"Static objects transparency {'enabled (alpha=0.3)' if transparent else 'disabled (alpha=1.0)'}")

    def compute_scene_bounds(self) -> Tuple[List[float], List[float]]:
        """Compute scene bounding box from all sim_object positions.

        Returns a ``(center, extent)`` tuple where *center* is the mean position
        ``[cx, cy, cz]`` and *extent* is the axis-aligned size ``[ex, ey, ez]``
        (i.e. ``max - min`` per axis).

        When the scene is empty both lists are ``[0.0, 0.0, 0.0]``.

        This is used by :meth:`setup_camera` (GUI) and
        :class:`~pybullet_fleet.recorder.SimulationRecorder` (offscreen) so that
        both share the same scene-framing logic.

        Returns:
            Tuple of ``([cx, cy, cz], [ex, ey, ez])``.
        """
        positions: List[List[float]] = []
        for obj in self._sim_objects:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self._client)
                positions.append(list(pos))
            except Exception:
                continue

        if not positions:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        arr = np.array(positions)
        center = arr.mean(axis=0)
        extent = arr.max(axis=0) - arr.min(axis=0)
        return [float(center[0]), float(center[1]), float(center[2])], [
            float(extent[0]),
            float(extent[1]),
            float(extent[2]),
        ]

    def setup_camera(
        self, camera_config: Optional[Dict[str, Any]] = None, entity_positions: Optional[List[List[float]]] = None
    ) -> None:
        """
        Set up camera view based on configuration.

        Valid ``camera_mode`` values:

        - ``"auto"`` (default) — Auto-calculates camera distance and target
          from entity positions or ``sim_objects``.
        - ``"manual"`` — Uses explicit ``camera_distance``, ``camera_yaw``,
          ``camera_pitch``, and ``camera_target`` from the config.

        Interactive controls (right-drag/arrow-key pan, +/- zoom, o=top-down) are
        always enabled in GUI mode regardless of camera_mode.

        Args:
            camera_config: Dictionary with camera settings (from yaml config)
            entity_positions: List of [x, y, z] positions for auto camera calculation
        """
        if not self._params.gui:
            return  # No camera setup needed without GUI

        # Mark that setup_camera was called (prevents configure_visualizer overwrite)
        self._camera_configured = True

        # Use self._params.camera_config if camera_config is not provided
        if camera_config is None:
            camera_config = self._params.camera_config

        camera_mode = camera_config.get("camera_mode", "auto")

        valid_modes = ("manual", "auto")
        if camera_mode not in valid_modes:
            logger.warning(f"Unknown camera_mode '{camera_mode}'. Valid modes: {valid_modes}")
            return

        if camera_mode == "manual":
            # Use manual camera settings
            distance = camera_config.get("camera_distance", 10.0)
            yaw = camera_config.get("camera_yaw", 0)
            pitch = camera_config.get("camera_pitch", -89)
            target = camera_config.get("camera_target", [0, 0, 0])

            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=target,
                physicsClientId=self._client,
            )

            logger.info(f"Camera set to manual mode: distance={distance:.2f}m, yaw={yaw}°, pitch={pitch}°, target={target}")

        elif camera_mode == "auto":
            # Calculate camera from entity positions
            if entity_positions is not None and len(entity_positions) > 0:
                # Use explicitly provided positions
                positions_array = np.array(entity_positions)
                center = list(positions_array.mean(axis=0).astype(float))
                extent = list((positions_array.max(axis=0) - positions_array.min(axis=0)).astype(float))
            else:
                # Use shared bounds computation from sim_objects
                center, extent = self.compute_scene_bounds()
                if center == [0.0, 0.0, 0.0] and extent == [0.0, 0.0, 0.0] and len(self._sim_objects) == 0:
                    logger.warning("Auto camera mode requested but no objects in simulation")
                    return
                logger.info(f"Auto camera: calculated from {len(self._sim_objects)} objects")

            # Get camera settings
            view_type = camera_config.get("camera_view_type", "top_down")
            auto_scale = camera_config.get("camera_auto_scale", 0.8)

            _MIN_CAMERA_DISTANCE = 1.0  # Prevent distance=0 when all objects are co-located
            if view_type == "top_down":
                # Top-down orthographic view
                distance = max(extent[0], extent[1]) * auto_scale
                yaw = 0
                pitch = -89
            else:
                # Perspective view
                distance = max(extent[0], extent[1], extent[2]) * auto_scale
                yaw = 45
                pitch = -30

            distance = max(distance, _MIN_CAMERA_DISTANCE)
            target = [center[0], center[1], center[2]]

            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=target,
                physicsClientId=self._client,
            )

            logger.info(f"Camera set to auto mode ({view_type}): distance={distance:.2f}m, target={target}")
            logger.info(f"  Structure extent: [{extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f}] meters")
            logger.info(f"  Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
            logger.info(f"  Yaw: {yaw}°, Pitch: {pitch}°")

        # Always enable interactive camera controller in GUI mode
        if self._params.gui and self._camera_controller is None:
            self._camera_controller = CameraController(client_id=self._client)
            logger.info("Interactive camera controls enabled: right-drag/arrows=pan, +/-=zoom, o=top-down")

    def query_aabb(self, aabb_min: Sequence[float], aabb_max: Sequence[float]) -> List[SimObject]:
        """Objects whose AABB overlaps the box ``[aabb_min, aabb_max]``.

        Answers from the AABBs the collision system holds: through its
        spatial hash when the box spans few cells (tens of microseconds,
        whatever the number of objects), by testing every object's AABB in
        one array operation when it spans many, or before the hash exists.
        Kinematic and physics objects' boxes are refreshed on every step
        whatever the collision check frequency, so the answer is for the
        current step.

        Objects with collision ``DISABLED`` hold no box; they are tested
        directly against their PyBullet AABB. A pose written outside a step
        is applied at once and answered for. Inside a step (from a callback or
        a controller), a written pose is buffered and reaches PyBullet only
        when the step flushes it: until then such an object is tested against
        the AABB of the pose it had when the step began. Joints are not
        buffered: an object whose links moved in the step
        (``notify_geometry_changed``) is tested against its links as they
        are now, at the base pose PyBullet holds.

        Args:
            aabb_min: ``(x, y, z)`` lower corner of the box, world frame.
            aabb_max: ``(x, y, z)`` upper corner of the box, world frame.

        Returns:
            The overlapping objects, in no particular order.

        Example::

            # Everything within 0.5 m of a point, as a box
            nearby = sim.query_aabb((x - 0.5, y - 0.5, z - 0.5), (x + 0.5, y + 0.5, z + 0.5))
        """
        lo = tuple(float(v) for v in aabb_min)
        hi = tuple(float(v) for v in aabb_max)

        def overlaps(box: Tuple[Sequence[float], Sequence[float]]) -> bool:
            (x0, y0, z0), (x1, y1, z1) = box
            return x0 <= hi[0] and x1 >= lo[0] and y0 <= hi[1] and y1 >= lo[1] and z0 <= hi[2] and z1 >= lo[2]

        objects = self._sim_objects_dict
        store = self._store
        stale: Set[int] = self._pending_pose_ids
        if self._pending_geometry_ids:
            stale = stale | self._pending_geometry_ids

        found = []
        for object_id in store.object_id[store.query_overlapping(lo, hi)].tolist():
            if object_id in stale:
                continue
            obj = objects.get(object_id)
            if obj is not None:
                found.append(obj)

        # Objects without a stored box, and those with a pose or joints
        # written in this step: judged by PyBullet's box as it is now.
        n = len(store)
        live = set(store.object_id[:n][~store.has_aabb[:n]].tolist())
        live.update(stale)
        for object_id in live:
            obj = objects.get(object_id)
            if obj is None:
                continue
            try:
                box = self._body_aabb(obj)
            except p.error:
                continue
            if overlaps(box):
                found.append(obj)
        return found

    def pickable_candidates_near(self, position: Sequence[float], radius: float) -> List[SimObject]:
        """Every pickable object whose base origin may be within ``radius`` of ``position``.

        A superset, for a search by base origin such as
        :meth:`SimObject.find_nearest_pickable`: it may also hold objects
        farther away or not pickable, so the caller measures the distance and
        checks ``pickable`` itself. The objects come through the spatial hash,
        as for :meth:`query_aabb`, plus those the hash cannot place by their
        origin: a pickable shape set more than a cell off its origin, objects
        with collision ``DISABLED``, and objects with a pose written in the
        current step and not yet flushed (their cached pose is already the new
        one). Before the hash exists, every object.

        Args:
            position: ``(x, y, z)`` to search around, world frame.
            radius: Search radius in metres.

        Returns:
            The candidate objects, in no particular order.
        """
        cell = self._cached_cell_size
        if cell is None:
            return list(self._sim_objects)
        # An origin a little outside its AABB (a mesh with its origin at a
        # corner, say) is still within the box widened by one cell.
        reach = radius + cell
        found = {obj.object_id: obj for obj in self.query_aabb([c - reach for c in position], [c + reach for c in position])}
        objects = self._sim_objects_dict
        extra = [object_id for object_id, gap in self._origin_gaps.items() if gap > cell]
        extra.extend(self._disabled_collision_objects)
        extra.extend(self._pending_pose_ids)
        for object_id in extra:
            if object_id not in found:
                obj = objects.get(object_id)
                if obj is not None:
                    found[object_id] = obj
        return list(found.values())

    def get_aabbs(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get the AABB of every simulation object, each covering all of the body's links."""
        return [self._body_aabb(obj) for obj in self._sim_objects]

    def cached_aabb(self, object_id: int) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """The AABB the collision system holds for ``object_id``, as of the last step or pose write.

        ``None`` for an object with collision ``DISABLED``, one not yet
        initialised, or an unknown id.
        """
        row = self._store.row_of.get(object_id)
        if row is None or not self._store.has_aabb[row]:
            return None
        return self._store.aabb(row)

    def grid_cells_of(self, object_id: int) -> List[Tuple[int, int, int]]:
        """The spatial hash cells ``object_id`` is registered in, in x, y, z order; empty when it is in none."""
        row = self._store.row_of.get(object_id)
        if row is None or not self._store.registered(np.array([row]))[0]:
            return []
        return self._store.cells_of(row)

    def _spatial_hash_contents(self) -> Dict[Tuple[int, int, int], Set[int]]:
        """The spatial hash as ``{cell: {object_id, ...}}``, synced, for tests and demos."""
        return self._store.hash_contents()

    def filter_aabb_pairs(
        self, ignore_static: Optional[bool] = None, moved_objects: Optional[Set[int]] = None, return_profiling: bool = False
    ) -> Tuple[List[Tuple[int, int]], Optional[Dict[str, float]]]:
        """
        The broad phase: candidate pairs of the moved objects, from the spatial hash.

        Each moved body is paired with the bodies sharing one of its cells
        whose AABBs are within the collision clearance of its own on every
        axis (no z clearance with a NORMAL_2D body, none at all for a pair
        with a physics body under HYBRID). Every body is in each cell its
        AABB, enlarged by half the clearance, touches, so a pair within the
        clearance always shares a cell.

        Args:
            ignore_static: If True, skip collision checks with static objects
            moved_objects: Set of object IDs that moved since last check.
                          If None, uses self._moved_this_step (default behavior)
            return_profiling: If True, include timing information in return value

        Returns:
            Tuple of (pairs, timings):
            - pairs: List of (obj_id_i, obj_id_j) tuples for candidate collision pairs
            - timings: dict with get_aabbs, spatial_hashing, aabb_filtering (in ms) if return_profiling=True, else None

        Example::

            # Without profiling
            pairs, _ = sim.filter_aabb_pairs()

            # With profiling
            pairs, timings = sim.filter_aabb_pairs(return_profiling=True)
            print(f"AABB filtering took {timings['total']:.2f}ms")
        """
        # Profiling timings (always create dict for consistent return type)
        timings: Dict[str, float] = {}

        if ignore_static is None:
            ignore_static = self._ignore_static_collision
        clearance = self._collision_broadphase_margin()

        # 0. A margin changed on ``params`` directly: rebuild, and check every
        # body. check_collisions() has already done this before taking its
        # moved set; this covers a direct call.
        if self._sync_grid_margin() and moved_objects is not None:
            moved_objects = set(self._sim_objects_dict.keys())

        # 1. Handle mode changes
        # Check if ignore_static mode has changed (skip if this is the first call)
        if self._last_ignore_static is not None and self._last_ignore_static != ignore_static:
            # Mode changed: force full recalculation regardless of moved_objects argument
            # Override moved_objects directly so the current call scans all objects
            # (setting self._moved_this_step alone would only affect the *next* call)
            moved_objects = set(self._sim_objects_dict.keys())
            # Clear existing collision pairs (will be recalculated with new filter rules)
            self._clear_active_collision_state()
            logger.debug(f"ignore_static mode changed to {ignore_static}, resetting collision state and marking all as moved")

        ignore_attached = bool(self._params.ignore_attached_collision)
        if self._last_ignore_attached is not None and self._last_ignore_attached != ignore_attached:
            # The same for ignore_attached_collision: a carrier and its load
            # that overlap are a pair again (or no longer one) from this check.
            moved_objects = set(self._sim_objects_dict.keys())
            self._clear_active_collision_state()
            logger.debug(f"ignore_attached_collision changed to {ignore_attached}, checking every pair again")

        # Update last mode (do this after the check)
        self._last_ignore_static = ignore_static
        self._last_ignore_attached = ignore_attached

        # 2. AABBs: kept up to date by add_object(), set_pose(), the step's
        # flushes and the physics loop. Computed for every object once, here,
        # when the collision system starts.
        if return_profiling:
            t0 = time.perf_counter()

        if not self._aabb_cache_valid:
            # In kinematics mode, call performCollisionDetection() first to ensure
            # PyBullet's broadphase has initialised collision info for all bodies.
            # Without this, getAABB() emits noisy C++ b3Warning messages.
            if not self._params.physics:
                p.performCollisionDetection(physicsClientId=self._client)
            logger.debug(f"Full AABB rebuild for {len(self._sim_objects)} objects")
            disabled = self._disabled_collision_objects
            for obj in self._sim_objects:
                if obj.object_id not in disabled:  # a DISABLED object holds no box (query_aabb reads PyBullet)
                    self._update_object_aabb(obj.object_id, update_grid=False)
            # Mark cache as valid after update
            self._aabb_cache_valid = True

        if return_profiling:
            timings["get_aabbs"] = (time.perf_counter() - t0) * 1000  # ms

        # 3. The spatial hash: its cell size on first use, then the rows whose
        # AABB or mode changed since the last check.
        if return_profiling:
            t0 = time.perf_counter()

        if self._cached_cell_size is None:
            self.set_collision_spatial_hash_cell_size_mode()
        self._store.sync_hash()

        if return_profiling:
            timings["spatial_hashing"] = (time.perf_counter() - t0) * 1000  # ms

        # 4. The pairs of the moved bodies, as arrays
        if return_profiling:
            t0 = time.perf_counter()

        if moved_objects is None:
            moved_objects = self._moved_this_step
        hybrid = self._params.collision_detection_method == CollisionDetectionMethod.HYBRID
        ids_a, ids_b = self._store.candidate_pairs(
            self._store.rows(moved_objects),
            clearance,
            ignore_static=ignore_static,
            hybrid=hybrid,
            ignore_attached=ignore_attached,
        )
        pairs = [(a, b) if a < b else (b, a) for a, b in zip(ids_a.tolist(), ids_b.tolist())]

        if return_profiling:
            timings["aabb_filtering"] = (time.perf_counter() - t0) * 1000  # ms
            return pairs, timings
        else:
            return pairs, None

    @staticmethod
    def _stop_on_collision(obj_a: SimObject, obj_b: SimObject) -> None:
        """Stop the agents a new contact involves, when their ``stop_on_collision`` is set.

        An object attached to an agent (a carried load) stands for that agent,
        so a load running into something stops its carrier. A contact between
        an agent and its own load is not a collision with anything else. An
        agent that is not moving, such as one struck while it stands, is left
        alone: there is no motion to stop, and an action it is running goes on.
        """

        root_a, root_b = obj_a._attachment_root(), obj_b._attachment_root()
        if root_a is root_b:
            # Only reached with ignore_attached_collision off, where such a
            # pair is still reported; it is never a reason to stop.
            return
        for root, other in ((root_a, obj_b), (root_b, obj_a)):
            if isinstance(root, Agent) and root.is_moving and root.controller_params._eff_stop_on_collision():
                logger.info(f"{root.name or root.object_id} stopped on collision with {other.name or other.object_id}")
                root.stop()

    def check_collisions(
        self,
        collision_color: Optional[List[float]] = None,
        ignore_static: Optional[bool] = None,
        return_profiling: bool = False,
    ) -> Tuple[List[Tuple[int, int]], Optional[Dict[str, float]]]:
        """
        Check for collisions between robots using incremental updates.

        Only re-checks collision pairs involving moved objects for efficiency.
        Maintains _active_collision_pairs as persistent state.

        Args:
            collision_color: RGB color for collision visualization
            ignore_static: If True, ignore structure collisions
            return_profiling: If True, include timing information in return value

        Returns:
            Tuple of (pairs, timings):
            - pairs: List of currently active collision pairs (obj_id_i, obj_id_j)
            - timings: dict with get_aabbs, spatial_hashing, aabb_filtering, contact_points, total (in ms)
              if return_profiling=True, else None

        Example::

            # Without profiling
            active_pairs, _ = sim.check_collisions()

            # With profiling
            active_pairs, timings = sim.check_collisions(return_profiling=True)
            print(f"Collision check took {timings['total']:.2f}ms")
        """
        detection_method = self._params.collision_detection_method
        if detection_method is None:
            raise ValueError("collision_detection_method must be set before checking collisions")
        # A margin set on ``params`` directly is validated only here: reject a
        # bad one before anything below changes the collision state.
        margin = _normalize_margin(self._params.collision_margin)

        t0 = time.perf_counter()

        # Profiling timings (always create dict for consistent return type)
        timings: Dict[str, float] = {}

        # A margin changed on ``params`` directly marks every body moved; done
        # before the moved set is taken, so the active pairs below are checked
        # again under the new margin in this call, not the next.
        self._sync_grid_margin()

        # Clear movement tracking from previous collision check
        # This accumulates all movements since the last collision check,
        # which is important when collision_check_frequency < step_frequency
        moved_since_last_check = self._moved_this_step.copy()
        self._moved_this_step.clear()

        if collision_color is None:
            collision_color = self._collision_color

        # Determine ignore_static flag
        if ignore_static is None:
            ignore_static = self._ignore_static_collision

        try:
            # Get candidate pairs (moved-based optimization)
            # Pass moved_since_last_check as argument instead of using self._moved_this_step
            if return_profiling:
                candidate_pairs, filter_timings = self.filter_aabb_pairs(
                    ignore_static=ignore_static, moved_objects=moved_since_last_check, return_profiling=True
                )
                if filter_timings:
                    timings.update(filter_timings)
            else:
                candidate_pairs, _ = self.filter_aabb_pairs(
                    ignore_static=ignore_static, moved_objects=moved_since_last_check, return_profiling=False
                )

            if return_profiling:
                t_contact0 = time.perf_counter()

            # --- Incremental collision detection ---
            # candidate_pairs already contains only pairs involving moved objects (filtered in filter_aabb_pairs)
            # We just need to add existing collision pairs for resolution detection
            pairs_to_check = set(candidate_pairs)
            offered_pairs = frozenset(pairs_to_check)

            # Also re-check existing collision pairs involving moved objects
            # (to detect when collisions are resolved)
            # Skip static-static pairs (they never change after initial check)
            for obj_id_i, obj_id_j in list(self._active_collision_pairs):
                # Re-check a pair that is now ignored, so that it resolves even
                # when neither object moved (e.g. one was just attached). Ahead
                # of the static shortcut: the rule can change for any pair.
                obj_i = self._sim_objects_dict.get(obj_id_i)
                obj_j = self._sim_objects_dict.get(obj_id_j)
                if obj_i is not None and obj_j is not None and self._is_ignored_collision(obj_i, obj_j):
                    pairs_to_check.add((obj_id_i, obj_id_j))
                    continue

                # A pair with a body switched to STATIC while static
                # collisions are ignored: the broad phase no longer offers it,
                # so resolve it here, even if neither body moved.
                if ignore_static and (
                    obj_id_i in self._static_collision_objects or obj_id_j in self._static_collision_objects
                ):
                    pairs_to_check.add((obj_id_i, obj_id_j))
                    continue

                # Skip if both objects are static (no need to re-check)
                if (
                    not self._recheck_stationary_pairs
                    and obj_id_i in self._static_collision_objects
                    and obj_id_j in self._static_collision_objects
                ):
                    continue

                # Re-check if at least one object moved
                if obj_id_i in moved_since_last_check or obj_id_j in moved_since_last_check:
                    pairs_to_check.add((obj_id_i, obj_id_j))

            # And the pairs the ignore rules excluded last time. A moved pair
            # the broad phase no longer offers is apart and is dropped; the
            # rest are checked again, ignored or not.
            for pair in list(self._ignored_collision_pairs):
                obj_id_i, obj_id_j = pair
                moved = obj_id_i in moved_since_last_check or obj_id_j in moved_since_last_check
                if moved and pair not in pairs_to_check:
                    self._ignored_collision_pairs.discard(pair)
                else:
                    pairs_to_check.add(pair)

            # The narrow phase reads PyBullet: the bodies of the pairs it
            # judges must hold their poses there.
            if pairs_to_check:
                self.sync_mirror({object_id for pair in pairs_to_check for object_id in pair})

            # Check contact points for all pairs. A pair the broad phase did not
            # offer (an active one checked again) first gets the broad phase's
            # per-axis AABB test, so it is judged as the broad phase would judge
            # it; the closest points are then queried at the largest component.
            query_distance = self._narrow_phase_clearance()
            clearance = self._collision_broadphase_margin()
            new_collisions = set()
            resolved_collisions = set()

            # The clock advances before a step's update, so the counters are the
            # sampled state's step and time whether or not a step is under way.
            sample_step = self._step_count
            sample_time = self._elapsed_sim_time

            for obj_id_i, obj_id_j in pairs_to_check:
                obj_i = self._sim_objects_dict.get(obj_id_i)
                obj_j = self._sim_objects_dict.get(obj_id_j)
                if obj_i is None or obj_j is None:
                    # Object was removed — clean up stale pair
                    self._discard_active_collision_pair((obj_id_i, obj_id_j))
                    self._ignored_collision_pairs.discard((obj_id_i, obj_id_j))
                    continue

                # Select collision detection method based on configuration
                has_collision = False
                signed_distance: Optional[float] = None
                pair_method = detection_method

                self._ignored_collision_pairs.discard((obj_id_i, obj_id_j))
                if self._is_ignored_collision(obj_i, obj_j):
                    # Treated as not touching: resolves the pair if it was
                    # active. Remembered, so it is re-evaluated next check.
                    self._ignored_collision_pairs.add((obj_id_i, obj_id_j))

                elif ignore_static and (
                    obj_id_i in self._static_collision_objects or obj_id_j in self._static_collision_objects
                ):
                    # Static collisions are ignored: not touching, which
                    # resolves the pair if it was active.
                    pass

                elif detection_method == CollisionDetectionMethod.CONTACT_POINTS:
                    # Method 1: getContactPoints (physics mode, actual contact manifold)
                    # Best for: physics simulation after a collision-detection pass.
                    # Note: Cache/solver-dependent for kinematic-kinematic pairs.
                    contact_points = p.getContactPoints(obj_i.body_id, obj_j.body_id, physicsClientId=self._client)
                    has_collision = len(contact_points) > 0

                elif detection_method == CollisionDetectionMethod.CLOSEST_POINTS:
                    # Method 2: getClosestPoints (kinematics mode, distance-based)
                    # Best for: kinematics motion, safety margin detection
                    # Works with: Physics ON/OFF, stable with resetBasePositionAndOrientation
                    has_collision, signed_distance = self._closest_points_within_clearance(
                        obj_id_i, obj_id_j, query_distance, None if (obj_id_i, obj_id_j) in offered_pairs else clearance
                    )

                elif detection_method == CollisionDetectionMethod.HYBRID:
                    # Method 3: Hybrid (physics uses getContactPoints, kinematic uses getClosestPoints)
                    # Best for: Mixed physics/kinematics with different detection needs
                    is_physics_i = obj_id_i in self._physics_objects
                    is_physics_j = obj_id_j in self._physics_objects

                    if is_physics_i or is_physics_j:
                        pair_method = CollisionDetectionMethod.CONTACT_POINTS
                        # At least one is physics object - use getContactPoints
                        contact_points = p.getContactPoints(obj_i.body_id, obj_j.body_id, physicsClientId=self._client)
                        has_collision = len(contact_points) > 0
                    else:
                        pair_method = CollisionDetectionMethod.CLOSEST_POINTS
                        # Both are kinematic - use getClosestPoints with safety margin
                        has_collision, signed_distance = self._closest_points_within_clearance(
                            obj_id_i,
                            obj_id_j,
                            query_distance,
                            None if (obj_id_i, obj_id_j) in offered_pairs else clearance,
                        )

                pair = (obj_id_i, obj_id_j)

                if has_collision:
                    self._collision_pair_observations[pair] = CollisionPairObservation(
                        pair, pair_method, signed_distance, sample_step, sample_time
                    )
                    # Collision detected
                    if pair not in self._active_collision_pairs:
                        # New collision
                        new_collisions.add(pair)
                        self._add_active_collision_pair(pair)
                        logger.info(
                            f"NEW COLLISION: object {obj_id_i} (body {obj_i.body_id}) "
                            f"<-> object {obj_id_j} (body {obj_j.body_id})"
                        )
                else:
                    # No collision
                    if pair in self._active_collision_pairs:
                        # Collision resolved
                        resolved_collisions.add(pair)
                        self._discard_active_collision_pair(pair)
                        logger.info(
                            f"COLLISION RESOLVED: object {obj_id_i} (body {obj_i.body_id}) "
                            f"<-> object {obj_id_j} (body {obj_j.body_id})"
                        )

            self._last_collision_check_metadata = (
                sample_step,
                sample_time,
                detection_method,
                margin,
            )

            if return_profiling:
                timings["contact_points"] = (time.perf_counter() - t_contact0) * 1000  # ms

            # Update collision count (track total new collisions)
            self._collision_count += len(new_collisions)

            # Color update (collision: blue, normal: original) only if enabled
            if self._params.enable_collision_color_change:
                # Get all currently colliding objects
                currently_colliding = set()
                for obj_id_i, obj_id_j in self._active_collision_pairs:
                    currently_colliding.add(obj_id_i)
                    currently_colliding.add(obj_id_j)

                # Update colors for new collisions (set to collision_color)
                for obj_id_i, obj_id_j in new_collisions:
                    for obj_id in (obj_id_i, obj_id_j):
                        obj = self._sim_objects_dict.get(obj_id)
                        if obj is not None:
                            p.changeVisualShape(obj.body_id, -1, rgbaColor=collision_color, physicsClientId=self._client)

                # Update colors for resolved collisions (restore original color)
                # Only restore if object is not involved in any other collision
                for obj_id_i, obj_id_j in resolved_collisions:
                    for obj_id in (obj_id_i, obj_id_j):
                        if obj_id not in currently_colliding:
                            obj = self._sim_objects_dict.get(obj_id)
                            if obj is not None:
                                orig_color = self._robot_original_colors.get(obj.body_id, [0, 0, 0, 1])
                                p.changeVisualShape(obj.body_id, -1, rgbaColor=orig_color, physicsClientId=self._client)

            # --- Collision Enter/Exit events ---
            if new_collisions or resolved_collisions:
                measure_ev = self._enable_time_profiling or return_profiling
                if measure_ev:
                    t_ev_col0 = time.perf_counter()

                for obj_id_i, obj_id_j in new_collisions:
                    obj_a = self._sim_objects_dict.get(obj_id_i)
                    obj_b = self._sim_objects_dict.get(obj_id_j)
                    if obj_a is not None and obj_b is not None:
                        self.events.emit(SimEvents.COLLISION_STARTED, obj_a=obj_a, obj_b=obj_b)
                        # Per-entity events
                        if obj_a._has_entity_events():
                            obj_a.events.emit(SimEvents.COLLISION_STARTED, other=obj_b)
                        if obj_b._has_entity_events():
                            obj_b.events.emit(SimEvents.COLLISION_STARTED, other=obj_a)
                        # After the events, so handlers see the motion that led to the contact.
                        self._stop_on_collision(obj_a, obj_b)

                for obj_id_i, obj_id_j in resolved_collisions:
                    obj_a = self._sim_objects_dict.get(obj_id_i)
                    obj_b = self._sim_objects_dict.get(obj_id_j)
                    if obj_a is not None and obj_b is not None:
                        self.events.emit(SimEvents.COLLISION_ENDED, obj_a=obj_a, obj_b=obj_b)
                        if obj_a._has_entity_events():
                            obj_a.events.emit(SimEvents.COLLISION_ENDED, other=obj_b)
                        if obj_b._has_entity_events():
                            obj_b.events.emit(SimEvents.COLLISION_ENDED, other=obj_a)

                if measure_ev:
                    ev_col_ms = (time.perf_counter() - t_ev_col0) * 1000
                    if self._profiling_stats["events_collision"]:
                        self._profiling_stats["events_collision"][-1] += ev_col_ms

        except p.error:
            # PyBullet disconnected, skip collision checking
            pass

        self._recheck_stationary_pairs = False

        if return_profiling:
            timings["total"] = (time.perf_counter() - t0) * 1000  # ms
            return list(self._active_collision_pairs), timings
        else:
            return list(self._active_collision_pairs), None

    def get_active_collision_pairs(self) -> List[Tuple[int, int]]:
        """
        Get currently active collision pairs.

        Returns:
            List of (object_id_i, object_id_j) tuples for all active collisions.
            Pairs are sorted such that object_id_i < object_id_j.

        Example::

            # Check if any collisions are active
            if sim_core.get_active_collision_pairs():
                print("Collisions detected!")

            # Get detailed collision info
            for obj_id_i, obj_id_j in sim_core.get_active_collision_pairs():
                obj_i = sim_core._sim_objects_dict[obj_id_i]
                obj_j = sim_core._sim_objects_dict[obj_id_j]
                print(f"Collision: {obj_i} <-> {obj_j}")
        """
        return list(self._active_collision_pairs)

    def get_collision_observation(self) -> Optional[CollisionObservation]:
        """Return the latest completed collision check, or ``None`` before one runs.

        A skipped step does not update the check timestamp. Each pair also has
        its own sample timestamp because incremental checks may leave an active
        pair unmeasured during a later completed check.
        """
        if self._last_collision_check_metadata is None:
            return None
        step, sim_time, configured_method, margin = self._last_collision_check_metadata
        return CollisionObservation(
            step,
            sim_time,
            configured_method,
            margin,
            tuple(self._collision_pair_observations[pair] for pair in sorted(self._active_collision_pairs)),
        )

    def _maybe_log_monitor_line(self) -> None:
        """Log the monitor figures at INFO once every ``monitor_log_interval`` wall seconds.

        Computes them afresh rather than reading the last monitor frame: with
        ``monitor`` off there is none, and with it on it advances only every
        ``monitor_update_interval``. It does not go through
        ``update_monitor()``, which would also write a DataMonitor frame and
        so tie the window's refresh to the log's.
        """
        if not logger.isEnabledFor(logging.INFO):
            return
        now = time.monotonic()
        if self._last_monitor_log is not None and now - self._last_monitor_log < self._params.monitor_log_interval:
            return
        self._last_monitor_log = now
        sim_time = self._elapsed_sim_time
        num_agents = len(self._agents)
        logger.info(
            "[MONITOR] sim_time=%.2f, real_time=%.2f, rtf=%.2f, target_rtf=%g, agents=%d, objects=%d, "
            "active_collisions=%d, total_collisions=%d, steps=%d",
            sim_time,
            now - self._start_time if self._start_time else 0,
            self._sample_speed(now, sim_time),
            self._params.target_rtf,
            num_agents,
            len(self._sim_objects) - num_agents,
            len(self._active_collision_pairs),
            self._collision_count,
            self._step_count,
        )

    def _sample_speed(self, now: float, sim_time: float) -> float:
        """Record a (real, simulated) time sample and return the real-time factor.

        Measured over a 10-second sliding window, plus the newest sample older
        than that as the baseline: when samples are sparser than the window (a
        long monitor_log_interval with the monitor off), dropping it would
        leave one sample and report 0.
        """
        self._speed_history.append((now, sim_time))
        while len(self._speed_history) > 1 and now - self._speed_history[1][0] > 10.0:
            self._speed_history.popleft()
        if len(self._speed_history) < 2:
            return 0.0
        rt0, st0 = self._speed_history[0]
        rt1, st1 = self._speed_history[-1]
        elapsed = rt1 - rt0
        return (st1 - st0) / elapsed if elapsed > 0 else 0.0

    def _note_monitor_entities_changed(self) -> None:
        """An object was added, removed or renamed: rebuild the monitor's entity list."""
        self._monitor_frames.objects_changed()
        self._fleet_revision += 1

    def update_monitor(self) -> None:
        """Update simulation monitor with current statistics.

        Calculates actual simulation speed from a 10-second sliding window,
        logs monitor data via logger, and writes to DataMonitor if enabled.
        """
        now = time.monotonic()  # monotonic: RTF/elapsed stay correct across wall-clock jumps
        sim_time = self._elapsed_sim_time
        actual_rtf = self._sample_speed(now, sim_time)
        elapsed_time = now - self._start_time if self._start_time else 0

        # Use len(self._agents) for O(1) agent count instead of O(N) iteration
        num_agents = len(self._agents)
        num_objects = len(self._sim_objects) - num_agents

        entities = self._monitor_frames.entities(self._sim_objects)
        selected_state = None
        selected = self._sim_objects_dict.get(self._selected_entity_id) if self._selected_entity_id is not None else None
        if selected is not None:
            pose = selected.get_pose()
            action = selected.get_current_action() if isinstance(selected, Agent) else None
            selected_state = SelectedEntityMonitorState(
                entity=MonitorEntity(
                    selected.object_id, selected.name or f"entity-{selected.object_id}", type(selected).__name__
                ),
                position=tuple(pose.position),
                orientation=tuple(pose.orientation),
                action_type=type(action).__name__ if action is not None else None,
                action_status=action.status.value if action is not None else None,
                queued_actions=selected.get_action_queue_size() if isinstance(selected, Agent) else 0,
                attached_objects=len(selected.attached_objects),
                active_collisions=sum(selected.object_id in pair for pair in self._active_collision_pairs),
            )
        elif self._selected_entity_id is not None:
            self._set_selected_entity(None, source="system")

        frame = MonitorFrame(
            sim_time=sim_time,
            real_time=elapsed_time,
            target_rtf=self._params.target_rtf,
            actual_rtf=actual_rtf,
            timestep=self._params.timestep,
            physics_enabled=self._params.physics,
            agents=num_agents,
            objects=num_objects,
            active_collisions=len(self._active_collision_pairs),
            collisions=self._collision_count,
            steps=self._step_count,
            paused=self.is_paused,
            entities=entities,
            selected_entity=selected_state,
            follow_enabled=self._gui_follow_enabled,
        )
        logger.debug(
            "[MONITOR] sim_time=%.2f, real_time=%.2f, rtf=%.2f, agents=%d, objects=%d, "
            "active_collisions=%d, total_collisions=%d, steps=%d",
            sim_time,
            elapsed_time,
            actual_rtf,
            num_agents,
            num_objects,
            len(self._active_collision_pairs),
            self._collision_count,
            self._step_count,
        )
        # Log only when collision count increases (new collision detected)
        if self._collision_count > self._last_logged_collision_count:
            logger.info(f"NEW COLLISION: total={self._collision_count} at sim_time={sim_time:.2f}")
            self._last_logged_collision_count = self._collision_count
        # Also output to DataMonitor window
        self._monitor_frames.built(frame)
        if self._data_monitor:
            self._data_monitor.write_frame(frame)

    def flush_monitor(self) -> bool:
        """Build and write a monitor frame now if the window is behind; return whether it did.

        Running, the next refresh (``monitor_update_interval``) shows any
        change, and :meth:`close` flushes, so only a program that stops
        calling ``step_once()`` and keeps the simulation open needs this.
        """
        if not self._data_monitor:
            return False
        if self._monitor_frames.behind(self._step_count):
            self.update_monitor()
            return True
        return False

    def initialize_simulation(self) -> None:
        """
        Initialize simulation state before running.

        This method prepares the simulation for execution by:
        - Resetting simulation counters (step_count, collision_count, sim_time)
        - Clearing movement tracking state
        - Configuring visualizer settings (transparency, collision shapes)
        - Enabling rendering for GUI mode

        Called automatically by run_simulation(), but can also be called manually
        for custom simulation loops or when restarting simulation.

        Example::

            # Automatic (recommended)
            sim.run_simulation(duration=10.0)

            # Manual initialization for custom loop
            sim.initialize_simulation()
            while custom_condition:
                sim.step_once()
        """
        # Reset simulation state
        self._start_time = time.monotonic()  # monotonic: elapsed/RTF immune to wall-clock jumps
        self._step_count = 0
        self._collision_count = 0
        self._elapsed_sim_time = 0.0
        self._schedule_origin = (0, 0.0, self._params.timestep)
        self._resume_ready = False
        self._collision_check_gate.reset()
        self._monitor_update_gate.reset()
        self._last_monitor_log = None  # a fresh run logs its first step
        self._last_logged_collision_count = 0
        self._monitor_frames.reset()
        self._single_step_requested = False
        self._pacing_rebase_requested = False
        self._clear_selected_nameplate()
        self._selected_entity_id = None
        self._gui_follow_enabled = False

        # Reset pause state (#1: prevents stuck-paused on re-run)
        self.resume()

        # Clear movement tracking from any previous simulation
        self._moved_this_step.clear()

        # Clear active collision pairs (fresh start)
        self._clear_active_collision_state()

        # Reset filter_aabb_pairs mode-change detection (#6: fresh start)
        self._last_ignore_static = None
        self._last_ignore_attached = None
        self._recheck_stationary_pairs = False

        # Clear speed history (#2: prevents stale data in update_monitor)
        self._speed_history.clear()

        # Restart the schedules of callbacks, plugins and frequency-limited
        # event handlers (#3: prevents stale dt on restart)
        self.events._reset_schedules()

        # Clear profiling statistics (#4: prevents mixing data across runs)
        for key in self._profiling_stats:
            self._profiling_stats[key].clear()
        self._last_profiling = None

        # Reset memory statistics (#5: prevents stale growth/average data)
        self._memory_stats = {
            "count": 0,
            "current_sum": 0.0,
            "current_min": float("inf"),
            "current_max": 0.0,
            "current_first": 0.0,
            "current_last": 0.0,
            "peak_sum": 0.0,
            "peak_max": 0.0,
        }

        # Configure visualizer after all objects are spawned
        # This ensures transparency and other visual settings are applied correctly
        self.configure_visualizer()

        # Enable rendering before starting simulation
        self.enable_rendering()

        # Start memory profiling if enabled (using is_tracing() for robustness)
        if self._enable_memory_profiling and not tracemalloc.is_tracing():
            # Start with 10 frames for traceback (useful for debugging memory issues)
            tracemalloc.start(10)
            self._memory_tracemalloc_started = True
            logger.info("Memory profiling started using tracemalloc (10 frames)")

        logger.info(f"Simulation initialized: {len(self._sim_objects)} objects, timestep={self._params.timestep}s")

    def reset(self) -> None:
        """Reset simulation to a clean state (no objects, fresh PyBullet world).

        Removes all objects from the simulation, resets PyBullet, re-applies
        physics-engine parameters, optionally reloads the ground plane (only
        when ``enable_floor`` is *True*), and re-initialises counters via
        :meth:`initialize_simulation`.

        After calling ``reset()`` the simulation is ready for new objects to be
        spawned.  Callbacks registered via :meth:`register_callback` are **preserved**
        (their schedules restart in ``initialize_simulation``).

        Example::

            sim.reset()
            # All objects removed, PyBullet world cleared
            agent = Agent.from_params(params, sim)
            sim.step_once()
        """
        # 1. Clear automatic behavior trees before their owning agents vanish.
        for tree, _ in list(self._behavior_tree_callbacks):
            self.unregister_behavior_tree(tree)

        # 2. Clear all Python-side object tracking. An object kept across the
        # reset gets its own pose storage back (its row is reused), and its
        # plugins come back from the batches.
        for obj in self._sim_objects:
            obj._unbind_rows()
            if isinstance(obj, Agent):
                obj._release_plugins_from_batches()
        self._sim_objects.clear()
        self._sim_objects_dict.clear()
        self._agents.clear()
        self._note_monitor_entities_changed()

        # Movement / collision caches
        self._moved_this_step.clear()
        self._physics_objects.clear()
        self._kinematic_objects.clear()
        self._static_collision_objects.clear()
        self._disabled_collision_objects.clear()
        self._dynamic_collision_objects.clear()
        self._cached_collision_modes.clear()
        self._store.clear()
        self._base_children.clear()
        self._base_parent_of.clear()
        self._updatable.clear()
        self._updatable_left.clear()
        self._plugin_batches.clear()
        self._aabb_links.clear()
        self._pending_geometry_ids.clear()
        self._origin_gaps.clear()
        self._cached_cell_size = None
        self._aabb_cache_valid = False
        self._clear_active_collision_state()
        self._robot_original_colors.clear()
        self._original_visual_colors.clear()

        # Reset camera flag so configure_visualizer can re-apply camera settings
        self._camera_configured = False

        # 1b. Clear registered managers
        for mgr in self._registered_managers:
            mgr.clear()

        # 1c. Notify plugins of reset
        for plugin in self._plugins:
            try:
                plugin.on_reset()
            except Exception:
                logger.error("Plugin %s.on_reset() failed", type(plugin).__name__, exc_info=True)

        # 2. Reset PyBullet world (removes all bodies including ground plane)
        p.resetSimulation(physicsClientId=self._client)
        SimObject._forget_shared_shapes(self._client)  # resetSimulation() removed the shapes too
        self._lighting_control_ids = None
        self._lighting_control_values = None

        # 3. Re-configure PyBullet (gravity, timestep, physics params, ground plane)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81 if self._params.physics else 0, physicsClientId=self._client)
        p.setTimeStep(self._params.timestep, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        self._apply_physics_engine_params()
        if self._params.enable_floor:
            p.loadURDF("plane.urdf", physicsClientId=self._client)

        # 4. Reset object ID counter so IDs start fresh
        self._next_object_id = 0

        # 5. Re-initialise counters, profiling, and rendering
        self.initialize_simulation()

        logger.info("Simulation reset: all objects removed, world reloaded")

    # ------------------------------------------------------------------
    # Config-driven world & robot loading
    # ------------------------------------------------------------------

    def load_world(
        self,
        world_config: Dict[str, Any],
        skip_models: Optional[List[str]] = None,
    ) -> List[SimObject]:
        """Load world environment from a configuration dictionary.

        Supports three sources (checked in priority order):

        1. ``world_file`` — a Gazebo ``.world`` file with ``<include>``
           elements (buildings, furniture, props).
        2. ``sdf`` — a single SDF model file (building geometry).
        3. ``mesh_dir`` — a directory of OBJ/STL mesh files.

        Args:
            world_config: Dictionary with world loading options::

                {
                    "world_file": "/path/to/office.world",
                    "skip_models": ["TinyRobot", "TeleportDispenser"],
                }

                or::

                    {"sdf": "/path/to/model.sdf"}

                or::

                    {"mesh_dir": "/path/to/meshes/"}

            skip_models: Model names to skip when loading a ``.world``
                file (e.g. robot names that are spawned separately).

        Returns:
            List of :class:`SimObject` instances created from the world.

        Example::

            sim = MultiRobotSimulationCore(SimulationParams(enable_floor=False))
            sim.initialize_simulation()
            objs = sim.load_world({"world_file": "/maps/office.world"})
        """
        from pybullet_fleet.sdf_loader import (
            load_mesh_directory,
            load_sdf_world,
            load_sdf_world_file,
        )

        world_file = world_config.get("world_file", "")
        world_sdf = world_config.get("sdf", "")
        world_mesh_dir = world_config.get("mesh_dir", "")
        skip = skip_models or world_config.get("skip_models", [])
        # Optional flat colour for world meshes — flat-shades multi-material
        # meshes PyBullet can't texture correctly (e.g. the campus env OBJ).
        force_color = world_config.get("force_color")

        if world_file:
            objs = load_sdf_world_file(world_file, sim_core=self, skip_models=skip, force_color=force_color)
            logger.info("Loaded %d objects from world file %s", len(objs), world_file)
            return objs

        if world_sdf:
            objs = load_sdf_world(world_sdf, sim_core=self)
            logger.info("Loaded %d links from SDF %s", len(objs), world_sdf)
            return objs

        if world_mesh_dir:
            if not os.path.isabs(world_mesh_dir):
                world_mesh_dir = os.path.join(os.getcwd(), world_mesh_dir)
            if os.path.isdir(world_mesh_dir):
                objs = load_mesh_directory(world_mesh_dir, sim_core=self)
                logger.info("Loaded %d meshes from %s", len(objs), world_mesh_dir)
                return objs
            logger.warning("World mesh directory not found: %s", world_mesh_dir)

        return []

    def spawn_robots_from_config(self, robots_config: List[Dict[str, Any]]) -> List[SimObject]:
        """Spawn robots/entities from a YAML-style configuration list.

        Each entry in *robots_config* is a dict understood by
        :meth:`AgentSpawnParams.from_dict` (or the appropriate
        ``SpawnParams`` for the entity ``type``).  SDF models are
        automatically converted to temporary URDF files via
        :func:`resolve_sdf_to_urdf`, and model names are resolved
        through :func:`resolve_model`.

        **Grid spawning**: If an entry contains a ``grid`` key, the
        entry is expanded into multiple entities placed on a grid.
        The ``grid`` dict is parsed by
        :meth:`~pybullet_fleet.agent_manager.GridSpawnParams.from_dict`
        and supports two formats:

        *Count-based* (convenience)::

            {
                "urdf_path": "robots/mobile_robot.urdf",
                "controller": {"type": "omni"},
                "grid": {
                    "count": 50,
                    "spacing": [3.0, 3.0],
                    "offset": [0, 0, 0.05],
                    "columns": 10,          # optional
                },
            }

        *Range-based* (explicit grid extents)::

            {
                "urdf_path": "robots/mobile_robot.urdf",
                "grid": {
                    "x_min": 0, "x_max": 9,
                    "y_min": 0, "y_max": 4,
                    "spacing": [3.0, 3.0, 0.0],
                    "offset": [0, 0, 0.05],
                },
            }

        Grid entries create a manager internally and register it with this
        simulation, so objects are automatically synchronised on
        :meth:`remove_object` and :meth:`reset`.

        **Batch processing**: Declare a named manager with
        ``fleet_controller.type`` and route the entity group to it. All
        entities in the group are added to that manager::

              "managers": [{
                  "name": "delivery_fleet",
                  "fleet_controller": {"type": "batch_differential"},
              }]
              "entities": [{
                  "urdf_path": "robots/simple_cube.urdf",
                  "controller": {"type": "differential"},
                  "manager": "delivery_fleet",
                  "grid": {"count": 100, "spacing": [2, 2]},
              }]

        **Entity type dispatch**: If an entry contains a ``type`` key
        (e.g. ``"agent"``, ``"sim_object"``, or a custom registered
        type), the corresponding class from the entity registry is
        used.  Default is ``"agent"``.

        Args:
            robots_config: List of robot/entity parameter dicts.

        Returns:
            List of spawned :class:`SimObject` (or subclass) instances.

        Example::

            sim.spawn_robots_from_config([
                # Individual robot
                {"urdf_path": "robots/mobile_robot.urdf", "pose": [0, 0, 0.05]},
                # Grid routed to a named batch manager
                {
                    "urdf_path": "robots/mobile_robot.urdf",
                    "controller": {"type": "omni"},
                    "manager": "delivery_fleet",
                    "grid": {"count": 20, "spacing": [3, 3]},
                },
                # Grid assigned to a named manager
                {
                    "urdf_path": "robots/mobile_robot.urdf",
                    "manager": "delivery_fleet",
                    "grid": {"count": 50, "spacing": [2, 2]},
                },
                # Custom entity type
                {"type": "forklift", "urdf_path": "robots/forklift.urdf", "pose": [10, 0, 0]},
            ])
        """
        from pybullet_fleet.agent_manager import GridSpawnParams, SimObjectManager
        from pybullet_fleet.entity_registry import ENTITY_REGISTRY
        from pybullet_fleet.robot_models import resolve_model
        from pybullet_fleet.sdf_loader import resolve_sdf_to_urdf

        spawned: List[SimObject] = []

        # --- Helper: resolve SDF→URDF and model paths on a dict --------
        def _resolve_paths(d: Dict[str, Any]) -> None:
            # Optional mesh yaw correction (radians) baked into the URDF visual/
            # collision origins — use when a model's mesh "front" is not aligned
            # with the body +X axis (e.g. a human mesh that walks sideways under
            # a differential controller). Popped so it is not forwarded to
            # SpawnParams.from_dict.
            yaw_off = float(d.pop("model_yaw_offset", 0.0))
            if "sdf_path" in d and "urdf_path" not in d:
                try:
                    d["urdf_path"] = resolve_sdf_to_urdf(d["sdf_path"], model_yaw_offset=yaw_off)
                    logger.info("Converted SDF %s → %s", d["sdf_path"], d["urdf_path"])
                except Exception as exc:
                    logger.error("Failed to convert SDF %s: %s", d.get("sdf_path"), exc)
            if "urdf_path" in d:
                try:
                    d["urdf_path"] = resolve_model(d["urdf_path"])
                except FileNotFoundError as exc:
                    logger.warning("Could not resolve '%s': %s", d["urdf_path"], exc)

        idx_counter = 0  # for auto-naming

        for d in robots_config:
            d = dict(d)  # avoid mutating caller's data
            grid_cfg = d.pop("grid", None)
            entity_type = d.pop("type", "agent")

            if entity_type not in ENTITY_REGISTRY:
                raise KeyError(f"Unknown entity type: {entity_type!r}. " f"Available: {list(ENTITY_REGISTRY)}")
            entity_cls = ENTITY_REGISTRY[entity_type]

            _resolve_paths(d)

            # Entity groups are routed only through a named manager. Batch
            # execution and shared controller parameters belong to that
            # manager's ``fleet_controller:`` mapping, not this spawn entry.
            if "batch_controller" in d or "fleet_controller" in d:
                raise ValueError(
                    "entities[].batch_controller and entities[].fleet_controller are unsupported; "
                    "declare a named manager with fleet_controller.type and route the entity with manager:"
                )
            manager_name = d.pop("manager", None)

            if grid_cfg is None:
                # --- Individual entity ----------------------------------
                if "name" not in d:
                    d["name"] = f"entity_{len(self._sim_objects) + idx_counter}"
                    idx_counter += 1

                with self.batch_spawn():
                    obj = entity_cls.from_dict(d, sim_core=self)

                # Route individual entity into a named manager if requested.
                if manager_name is not None:
                    named_mgr = self.get_manager(manager_name)
                    if named_mgr is None:
                        raise KeyError(
                            f"manager {manager_name!r} not found. " "Declare it in the 'managers:' section before 'entities:'."
                        )
                    named_mgr.add_object(obj)
                spawned.append(obj)
            else:
                # --- Grid spawn -----------------------------------------
                from pybullet_fleet.agent_manager import AgentManager

                grid_params = GridSpawnParams.from_dict(grid_cfg)
                count = grid_cfg.get("count", grid_params.total_cells)

                name_base = d.get("name", d.get("urdf_path", "robot"))
                d.pop("pose", None)
                d["name"] = name_base

                spawn_params = entity_cls._spawn_params_cls.from_dict(d)

                # Resolve the optional named manager.
                if manager_name is not None:
                    mgr = self.get_manager(manager_name)
                    if mgr is None:
                        raise KeyError(
                            f"manager {manager_name!r} not found. " "Declare it in the 'managers:' section before 'entities:'."
                        )
                else:
                    mgr = SimObjectManager(sim_core=self, object_class=entity_cls)
                    self.register_manager(mgr)

                grid_objects = mgr.spawn_grid_mixed(count, grid_params, [(spawn_params, 1.0)], name_prefix=name_base)

                spawned.extend(grid_objects)

                logger.info(
                    "Grid-spawned '%s': %d entities (type=%s, grid=%s)",
                    name_base,
                    len(grid_objects),
                    entity_type,
                    f"{grid_params.x_max - grid_params.x_min + 1}x" f"{grid_params.y_max - grid_params.y_min + 1}",
                )

        logger.info("Spawned %d entities from config", len(spawned))
        return spawned

    def run_simulation(self, duration: Optional[float] = None, *, resume: bool = False) -> None:
        """
        Run the simulation for a specified duration.

        Args:
            duration: Absolute simulation-time cutoff in seconds (not wall time).
                     If None, uses self._params.duration. If duration <= 0, runs indefinitely.
            resume: Continue an explicitly restored completed-step state without
                    resetting its clock. A core returned by the supported
                    recording loader resumes on its first plain call too.

        Example::

            # Run for 10 seconds (simulation time)
            sim.run_simulation(duration=10.0)

            # Run indefinitely (until Ctrl+C or GUI closed)
            sim.run_simulation(duration=0)
        """
        if duration is None:
            duration = self._params.duration

        # Auto-recording from RECORD environment variable (zero-code-change recording)
        record_path = os.environ.get("RECORD")
        if record_path and self._recorder is None:
            record_duration = float(os.environ.get("RECORD_DURATION", "4.0"))
            record_fps = int(os.environ.get("RECORD_FPS", "15"))
            record_width = int(os.environ.get("RECORD_WIDTH", "800"))
            record_height = int(os.environ.get("RECORD_HEIGHT", "600"))
            record_time_base = os.environ.get("RECORD_TIME_BASE", "sim")
            self.start_recording(
                output=record_path,
                duration=record_duration,
                fps=record_fps,
                width=record_width,
                height=record_height,
                time_base=record_time_base,
            )

        # A completed-step restore has already initialized this core. Treat its
        # first plain run as continuation; a fresh core still initializes as
        # before. Explicit resume=True remains useful for lower-level callers.
        resume = resume or self._resume_ready

        # Initialize simulation state (counters, visualizer, rendering)
        # Note: initialize_simulation() already starts tracemalloc if needed
        if resume:
            if not self._resume_ready:
                raise RuntimeError("resume=True requires a restored completed-step state")
            self._resume_ready = False
        else:
            self.initialize_simulation()

        try:
            # Pace against the MONOTONIC clock: it never jumps backwards and is
            # immune to NTP / host-suspend / WSL2 wall-clock corrections, which
            # previously desynced this loop into multi-second freezes.
            pacing_now = time.monotonic()
            if resume:
                # Monitor Real Time measures this process's resumed run, while
                # Sim Time continues from the restored completed-step clock.
                self._start_time = pacing_now
            start_time = pacing_now
            if resume and self._params.target_rtf > 0:
                start_time -= self._elapsed_sim_time / self._params.target_rtf
            last_step_process_time = 0.0  # Track processing time excluding sleep
            last_pause_state = False  # Track pause state to detect resume
            # Diagnostics: previous wall/monotonic readings to spot clock jumps.
            # Resume rebases pacing, but clock-jump diagnostics start now.
            diag_prev_wall = time.time()
            diag_prev_mono = pacing_now
            last_iter_mono = pacing_now

            while True:
                iter_now = time.monotonic()
                stall = iter_now - last_iter_mono
                if stall > _STEP_STALL_LOG_THRESHOLD_S:
                    logger.warning(
                        "run_simulation: %.2fs since the previous loop iteration started (step %d) -- "
                        "the host was likely overloaded, or GUI event processing stalled; this is real "
                        "time elapsed between steps, not a jump in the simulation's own clock.",
                        stall,
                        self._step_count,
                    )
                last_iter_mono = iter_now

                current_sim_time = self._elapsed_sim_time

                # Check duration based on simulation time (not real time)
                if duration > 0 and (
                    current_sim_time >= duration
                    or math.isclose(current_sim_time, duration, rel_tol=0.0, abs_tol=self._params.timestep * 1e-9)
                ):
                    break

                # Check if PyBullet connection is still active (e.g., GUI window not closed)
                try:
                    p.getConnectionInfo(physicsClientId=self._client)
                except p.error:
                    logger.info("PyBullet connection lost (GUI window closed)")
                    break

                if self._pacing_rebase_requested:
                    if self._params.target_rtf > 0:
                        pacing_now = time.monotonic()
                        start_time = pacing_now - current_sim_time / self._params.target_rtf
                        diag_prev_mono = pacing_now
                    last_step_process_time = 0.0
                    self._pacing_rebase_requested = False

                # target_rtf=0: Run as fast as possible (no synchronization, no sleep)
                if self._params.target_rtf <= 0:
                    self.step_once()
                else:
                    # Speed>0: Synchronize with real time using absolute time calculation.
                    # One monotonic read serves as both the iteration start and "now".
                    loop_start = current_time = time.monotonic()

                    # Diagnostics (WARNING level): detect wall-clock jumps (NTP /
                    # host suspend / WSL2 drift). Pacing uses the monotonic clock,
                    # so a jump no longer affects the sim — we only log it so the
                    # cause is visible. Guarded by isEnabledFor and confined to the
                    # rtf>0 branch to keep the max-speed (rtf<=0) loop call-free.
                    if logger.isEnabledFor(logging.WARNING):
                        now_wall = time.time()
                        wall_d, mono_d = now_wall - diag_prev_wall, current_time - diag_prev_mono
                        if abs(wall_d - mono_d) > _CLOCK_JUMP_LOG_THRESHOLD_S:
                            logger.warning(
                                "run_simulation: wall clock jumped %.2fs while monotonic advanced %.2fs "
                                "(delta %.2fs) at step %d — pacing uses the monotonic clock, so this does "
                                "not affect the simulation.",
                                wall_d,
                                mono_d,
                                wall_d - mono_d,
                                self._step_count,
                            )
                        diag_prev_wall, diag_prev_mono = now_wall, current_time

                    # Detect pause state change: reset start_time on resume
                    if last_pause_state and not self.is_paused:
                        # Just resumed from pause: reset start_time to avoid jump
                        start_time = current_time - current_sim_time / self._params.target_rtf
                        logger.info("Resumed from pause: start_time reset for smooth continuation")
                    last_pause_state = self.is_paused

                    # Calculate target sim_time using absolute time (stable control)
                    elapsed_time = current_time - start_time
                    target_sim_time = elapsed_time * self._params.target_rtf
                    time_diff = target_sim_time - current_sim_time

                    actual_sleep = 0.0

                    if time_diff > 0:
                        # Behind target: execute multiple steps to catch up
                        steps_needed = int(time_diff / self._params.timestep)
                        steps_needed = max(1, min(steps_needed, self._params.max_steps_per_frame))
                        for _ in range(steps_needed):
                            self.step_once()
                    else:
                        # Ahead of schedule: wait in REAL time. We are
                        # abs(time_diff) sim-seconds ahead; at target_rtf× speed
                        # that is abs(time_diff) / target_rtf real-seconds (the
                        # /target_rtf conversion the old code was missing — it
                        # over-slept by a factor of rtf, e.g. 3× at rtf=3).
                        real_ahead = abs(time_diff) / self._params.target_rtf
                        sleep_time = real_ahead - last_step_process_time

                        # Determine minimum sleep for GUI responsiveness
                        if self._params.gui:
                            min_sleep = 1.0 / self._params.gui_min_fps
                        else:
                            # For non-GUI, use timestep as minimum
                            min_sleep = self._params.timestep

                        # Sleep strategy based on how much we're ahead/behind
                        if sleep_time > 0:
                            # Ahead of target: sleep exactly the calculated time (no forced minimum)
                            actual_sleep = sleep_time
                        elif sleep_time > -min_sleep:
                            # Ahead of target but less than min_sleep: sleep minimum for GUI responsiveness
                            actual_sleep = min_sleep

                        # Safety clamp relative to the intended per-step real
                        # interval, so a residual anomaly can never freeze the sim
                        # while still allowing legitimate slow-motion (rtf < 1),
                        # where each step is meant to take timestep/rtf seconds.
                        # max(0.0, ...) guards against a negative (misconfigured)
                        # max_sleep_frames, which would otherwise make time.sleep raise.
                        sleep_cap = max(
                            0.0,
                            max(min_sleep, self._params.timestep / self._params.target_rtf) * self._params.max_sleep_frames,
                        )
                        if actual_sleep > sleep_cap:
                            logger.warning(
                                "run_simulation: clamping sleep %.2fs -> %.2fs at step %d "
                                "(target_rtf=%s, time_diff=%.2f, last_step_process=%.2f) — "
                                "unexpectedly large; the simulation will not freeze.",
                                actual_sleep,
                                sleep_cap,
                                self._step_count,
                                self._params.target_rtf,
                                time_diff,
                                last_step_process_time,
                            )
                            actual_sleep = sleep_cap
                        time.sleep(actual_sleep)

                    # Record processing time for this iteration (excluding sleep)
                    last_step_process_time = time.monotonic() - loop_start - actual_sleep

        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
        except RuntimeError as e:
            if "PyBullet disconnected" in str(e):
                logger.info("PyBullet connection lost (GUI window closed)")
            else:
                raise
        except p.error:
            logger.info("PyBullet connection lost (GUI window closed)")
        self.update_monitor()
        self.close()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release everything this simulation owns, including its client.

        ``run_simulation()`` has always done this on the way out, but a
        program that drives ``step_once()`` itself, or a test that only builds
        a scene, never reached that code -- and nothing else disconnected. An
        undisconnected client leaks for the life of the process, and because
        PyBullet's module-level calls resolve against a default client, the
        *next* simulation in the same process then finds bodies and joints
        "not found" rather than failing where the leak happened.

        Safe to call more than once, and safe on a simulation that was never
        run. A second call does nothing at all: shutdown callbacks run once,
        so a plugin that closes a file or a socket in ``on_shutdown()`` is not
        asked to do it twice by ``with sim:`` around a ``run_simulation()``
        that already closed.

        Releasing the client is unconditional. Finalising a recording can
        raise -- saving an MP4 without ``imageio``, for one -- and letting
        that escape before the disconnect would leak the very client this
        exists to release.
        """
        if self._closed:
            return
        self._closed = True

        def _finalise_recording() -> None:
            if self._recorder is not None:
                self.stop_recording()
            if self._state_recorder is not None:
                try:
                    self._state_recorder.close()
                except Exception as exc:
                    self._abort_state_recording(exc)

        def _stop_monitor() -> None:
            # Kept as it was: on other platforms the monitor is not stopped
            # here, which tests/test_monitor_macos.py pins.
            if IS_MACOS and self._data_monitor:
                self._data_monitor.stop()

        # Each step runs even if an earlier one raised, so one failure cannot
        # strand the resources the later steps own -- and since _closed is
        # already set, a retry would be a no-op. The first exception is kept
        # and re-raised once everything has been given its chance.
        def _flush_monitor() -> None:
            # A program driving step_once() itself may stop between refreshes;
            # leave the file at the last state it reached.
            if self._monitor_enabled:
                try:
                    self.flush_monitor()
                except Exception:  # noqa: BLE001 - a stale monitor file must not fail close()
                    logger.debug("Final DataMonitor flush failed", exc_info=True)

        first_error: Optional[BaseException] = None
        for step in (_flush_monitor, _finalise_recording, self._shutdown_plugins, _stop_monitor):
            try:
                step()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if first_error is None:
                    first_error = exc
        try:
            # PyBullet reuses client ids, so disconnecting twice could name a
            # later, unrelated simulation -- but the _closed guard above means
            # the disconnect runs at most once, so `client` stays readable
            # after close() as it always has. run_simulation() ends by calling
            # this, and callers do read it afterwards.
            if self._client is not None:
                try:
                    p.disconnect(self._client)
                except p.error:
                    pass  # already disconnected
                SimObject._forget_shared_shapes(self._client)
        finally:
            if first_error is not None:
                raise first_error

    def __enter__(self) -> "MultiRobotSimulationCore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Custom Profiling Fields
    # ------------------------------------------------------------------

    def record_profiling(self, name: str, value_ms: float) -> None:
        """Record a custom profiling measurement (in milliseconds).

        Call this from within Agent.update() or callback code to accumulate
        timing data. Multiple calls per step are summed (e.g., 100 agents each
        recording their custom_logic time). Auto-registers the field on first use.

        The recorded field appears in both _print_profiling_summary() output and
        step_once(return_profiling=True) results alongside built-in fields.

        Args:
            name: Field name (e.g., 'custom_logic', 'planner').
                  Auto-registered on first call — no separate setup needed.
            value_ms: Time measurement in milliseconds

        Example::

            # Inside your custom Agent.update():
            t0 = time.perf_counter()
            self._do_custom_logic(dt)
            self.sim_core.record_profiling('custom_logic', (time.perf_counter() - t0) * 1000)
        """
        data = self._profiling_stats.get(name)
        if data is None:
            # Auto-register: create with initial 0.0 slot for current step
            self._profiling_stats[name] = [value_ms]
            return
        if data:
            data[-1] += value_ms

    def pause(self) -> None:
        """Pause the simulation. step_once will be skipped while paused."""
        self._simulation_paused = True
        self._monitor_frames.view_changed()
        self.events.emit(SimEvents.PAUSED)

    def resume(self) -> None:
        """Resume the simulation after a pause."""
        self._simulation_paused = False
        self._monitor_frames.view_changed()
        self.events.emit(SimEvents.RESUMED)

    @property
    def is_paused(self) -> bool:
        """Return True if the simulation is currently paused."""
        return self._simulation_paused

    def step_once(self, return_profiling: bool = False) -> Optional[Dict[str, Any]]:
        """
        Execute one simulation step.

        Performs agent updates, callbacks, physics step, collision detection, and monitoring.
        If enable_time_profiling is True, detailed timing information is logged.

        Args:
            return_profiling: If True, return timing breakdown dictionary instead of logging.
                             Useful for external profiling tools.

        Returns:
            If return_profiling=True, returns dict with timing breakdown in milliseconds::

                {
                    'phase1_update': float,        # agent.update + callbacks + plugin on_step
                    'phase2_pose_flush': float,    # flush buffered poses to PyBullet
                    'phase3_aabb_grid_flush': float, # kinematic AABB + spatial-grid refresh
                    'agent_update': float,         # per-object update loop (subset of phase1; compat)
                    'callbacks': float,            # CallbackPhase.STEP callbacks (not plugins)
                    'step_simulation': float,      # p.stepSimulation(); non-zero only if physics=True
                    'collision_check': float,
                    'collision_breakdown': {       # present only when collision ran
                        'get_aabbs': float,
                        'spatial_hashing': float,
                        'aabb_filtering': float,
                        'contact_points': float,   # narrow phase; non-zero only if physics=True
                        'total': float,
                    },
                    'events_pre_step': float,      # handlers of the pre_step event
                    'callbacks_pre_step': float,   # CallbackPhase.PRE_STEP callbacks
                    'events_step': float,          # handlers of the step event (before callbacks)
                    'events_post_step': float,
                    'callbacks_post_step': float,  # CallbackPhase.POST_STEP callbacks
                    'events_collision': float,
                    'monitor_update': float,
                    'total': float,
                }

            Otherwise returns None.

        Performance note: time.perf_counter() calls have negligible overhead (<0.1% for 10k objects).
        The profiling measurements themselves do not significantly impact simulation performance.
        """

        # Profiling: step start time (measure even if return_profiling=True)
        measure_timing = self._enable_time_profiling or return_profiling
        if not measure_timing:
            # Do not expose a stale result after profiling has been disabled.
            self._last_profiling = None
        if measure_timing:
            t_step = time.perf_counter()

        # Check if PyBullet is still connected
        try:
            p.getConnectionInfo(physicsClientId=self._client)
        except p.error:
            # PyBullet disconnected, skip this step
            return

        # Monitor controls are submitted from tkinter and consumed only by this
        # simulation thread.  Do this before GUI input so both sources share
        # the same step boundary.
        if IS_MACOS and self._data_monitor:
            self._data_monitor.process_events()
        self._consume_gui_commands()
        self._update_gui_lighting_controls()

        # Handle keyboard events for visual/collision shape toggling and pause
        if self._keyboard_events_registered:
            self._handle_keyboard_events()

        # Skip physics simulation and callbacks if paused
        if self.is_paused and not self._single_step_requested:
            if self._monitor_enabled and self._data_monitor and self._monitor_frames.due_while_paused():
                # Nothing refreshes the monitor while paused: still show a
                # selection, follow or pause change, or a pose set from the GUI or user code.
                self.update_monitor()
            return
        # A single step taken while paused is shown at once, refresh due or not.
        single_step = self.is_paused
        self._single_step_requested = False

        # Initialize profiling accumulators for this step (append 0.0 slot for ALL fields)
        # Placed after pause/disconnect checks so paused steps don't accumulate stale entries
        if measure_timing:
            for data in self._profiling_stats.values():
                data.append(0.0)

        # --- Two-phase step boundary ---
        # _in_step gates SimObject._set_pose_internal between buffered and immediate paths.
        # try/finally guarantees the flag is cleared even if a callback / plugin raises.
        self._in_step = True
        if measure_timing:
            t_phase1 = time.perf_counter()
        try:
            # Physics objects are always considered as potentially moved (conservative approach)
            # This avoids expensive pose comparison every step
            # Note: _moved_this_step is cleared in check_collisions() to accumulate movements
            # across multiple steps when collision_check_frequency < step_frequency
            self._moved_this_step.update(self._physics_objects)

            # --- pre_step event ---
            # The last point of the step that sees the previous state's time.
            self._in_pre_step = True
            try:
                self._emit_step_hooks(SimEvents.PRE_STEP, _PRE_STEP_EVENT_TIERS, measure_timing)
            finally:
                self._in_pre_step = False

            # --- Advance the clock ---
            # Everything from here on computes or observes the state at the end
            # of this step, so the clock and counter move first: sim_time is the
            # time of the state any later code sees (object update, callbacks,
            # plugins, collision check, POST_STEP, and outside the step).
            if self._params.timestep != self._schedule_origin[2]:
                # A new timestep: the schedule counts its steps from here.
                self._schedule_origin = (self._step_count, self._schedule_time(), self._params.timestep)
            self._elapsed_sim_time += self._params.timestep
            self._step_count += 1
            # Poses written from pre_step are still pending (buffered).
            self._restamp_pending_pose_cache()

            # Update the objects with a per-step update (_updatable: agents,
            # devices, a subclass with _wants_update = True or that set its
            # _needs_update). The loop runs over a snapshot of the list and
            # skips the objects that leave it meanwhile: an update() may
            # remove another object or clear its _needs_update.
            if measure_timing:
                t0 = time.perf_counter()

            # Batch controllers run BEFORE the per-object update loop. They
            # write poses for their managed agents in one batched call via
            # set_poses() (no per-agent Python dispatch). Agents managed by a
            # batch controller still appear in self._sim_objects and have
            # _needs_update=True, but their own controller is IDLE so the
            # per-object loop is a cheap no-op for them.
            # BCs are owned by AgentManagers registered with this sim_core.
            for mgr in self._registered_managers:
                bc = getattr(mgr, "batch_controller", None)
                if bc is not None:
                    bc.synchronized_batch_advance(self._params.timestep)

            kinematic = self._kinematic_objects
            moved_ids = self._moved_this_step
            left = self._updatable_left
            left.clear()
            dt = self._params.timestep
            for obj in tuple(self._updatable):
                if left and obj in left:
                    continue
                if obj.update(dt) and obj.object_id in kinematic:
                    moved_ids.add(obj.object_id)
            # Agent plugins with a batch: every plugin of the class in one call.
            for batch in self._plugin_batches.values():
                if batch is not None:
                    batch.update(self._params.timestep)

            if measure_timing:
                t1 = time.perf_counter()
                # agent_update covers the per-object update loop only (legacy metric).
                self._profiling_stats["agent_update"][-1] = (t1 - t0) * 1000

            # --- step event: handlers, then callbacks, then plugin on_step ---
            self._emit_step_hooks(SimEvents.STEP, _STEP_EVENT_TIERS, measure_timing)

            # --- Two-phase step Phase 2: flush pending pose writes to PyBullet ---
            # Snapshot + reset the buffer here so re-entrant set_pose calls after this
            # point (e.g. from post_step events or the physics-objects loop below) start
            # a fresh buffer. The snapshot is threaded explicitly into the Phase 3 flush
            # to avoid hidden ordering coupling between the two helpers.
            if measure_timing:
                t_phase1_end = time.perf_counter()
                self._profiling_stats["phase1_update"][-1] = (t_phase1_end - t_phase1) * 1000
                t_phase2 = time.perf_counter()
            flushed_ids = self._pending_pose_ids
            self._pending_pose_ids = set()
            self._flush_pending_poses(flushed_ids)
            synced_link_attachment_ids = self._sync_link_attachments_after_pose_flush(flushed_ids)
            flushed_ids = flushed_ids | synced_link_attachment_ids
            if measure_timing:
                self._profiling_stats["phase2_pose_flush"][-1] = (time.perf_counter() - t_phase2) * 1000

            # Check if PyBullet is still connected before stepping
            if not p.isConnected(physicsClientId=self._client):
                raise RuntimeError("PyBullet disconnected (GUI window closed)")

            if measure_timing:
                t_sim0 = time.perf_counter()

            # stepSimulation() control based on physics mode
            # Physics ON: Call stepSimulation() every step (rigid body integration, contact resolution)
            # Physics OFF: Skip stepSimulation() for pure kinematics (position updates via reset API)
            if self._params.physics:
                p.stepSimulation(physicsClientId=self._client)
            # Note: Even in Physics OFF mode, collision detection still works via getClosestPoints()
            # which queries geometry directly without requiring stepSimulation()

            # Update AABBs and spatial grid for physics objects AFTER stepSimulation()
            # This ensures collision detection uses current-frame positions, not stale ones.
            # Kinematic objects update their AABBs and spatial grid in set_pose() for immediate consistency.
            for obj_id in self._physics_objects:
                # Safeguard against stale _physics_objects entries
                obj = self._sim_objects_dict.get(obj_id)
                if obj is None:
                    continue
                # The body's pose as the physics step left it, into its row
                # (and so into its pose cache), stamped as this step's.
                position, orientation = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self._client)
                obj._cached_pose.position[:] = position
                obj._cached_pose.orientation[:] = orientation
                obj._cached_pose_sim_time = self._elapsed_sim_time
                self._update_object_aabb(obj_id, update_grid=False)

            # step_simulation covers p.stepSimulation() + physics-object AABB loop only.
            # Phase 3 (kinematic AABB flush) is measured separately to avoid double-counting.
            if measure_timing:
                t_sim1 = time.perf_counter()
                self._profiling_stats["step_simulation"][-1] = (t_sim1 - t_sim0) * 1000

            # --- Two-phase step Phase 3: flush AABB + spatial grid for buffered kinematic objects ---
            # Uses the snapshot captured at Phase 2 — no dependency on _pending_pose_ids,
            # so the two flush helpers can be reordered or invoked independently if needed.
            if measure_timing:
                t_phase3 = time.perf_counter()
            geometry_ids = self._pending_geometry_ids
            if geometry_ids:
                self._pending_geometry_ids = set()
            self._flush_aabb_and_grid(flushed_ids, geometry_ids)
            if measure_timing:
                self._profiling_stats["phase3_aabb_grid_flush"][-1] = (time.perf_counter() - t_phase3) * 1000

            # Collision check frequency control
            if measure_timing:
                t_col0 = time.perf_counter()

            collision_breakdown = None  # per-phase breakdown (only when return_profiling=True)
            freq = self._collision_check_frequency  # None: every step, <= 0: disabled
            if freq is None or freq > 0:
                self._collision_check_gate.set_frequency(freq)  # no-op unless it changed
                if self._collision_check_gate.due(self._schedule_time(), self._params.timestep) is not None:
                    _, collision_breakdown = self.check_collisions(return_profiling=return_profiling)
            else:
                # Re-enabled later, even at the same rate, it starts a fresh schedule.
                self._collision_check_gate.reset()

            if measure_timing:
                t_col1 = time.perf_counter()
                self._profiling_stats["collision_check"][-1] = (t_col1 - t_col0) * 1000

            # Clear _in_step before POST_STEP so that set_pose() calls made by
            # POST_STEP subscribers are written directly to PyBullet in this step
            # rather than being buffered for the next step.
            self._in_step = False
            self._in_post_step = True

            # --- post_step event: handlers, then POST_STEP callbacks ---
            self._emit_step_hooks(SimEvents.POST_STEP, _POST_STEP_EVENT_TIERS, measure_timing)

            self._update_selected_nameplate()
            self._in_post_step = False
            # A durable recorder must see the completed clock and propagate
            # failures; EventBus deliberately swallows subscriber exceptions.
            if self._state_recorder is not None:
                try:
                    self._state_recorder.on_completed_step()
                except Exception as exc:
                    self._abort_state_recording(exc)
            # Monitor: every step if GUI enabled, otherwise every second
            if measure_timing:
                t_mon0 = time.perf_counter()

            if self._monitor_enabled:
                if self._monitor_update_gate.due(self._schedule_time(), self._params.timestep) is not None or (
                    single_step and self._data_monitor
                ):
                    self.update_monitor()

            if self._params.monitor_log_interval > 0:
                self._maybe_log_monitor_line()

            if measure_timing:
                t_mon1 = time.perf_counter()
                t_end = time.perf_counter()
                self._profiling_stats["monitor_update"][-1] = (t_mon1 - t_mon0) * 1000
                self._profiling_stats["total"][-1] = (t_end - t_step) * 1000

                # Keep a stable public snapshot before return_profiling pops
                # values or periodic logging clears the interval accumulator.
                result: Dict[str, Any] = {name: data[-1] for name, data in self._profiling_stats.items() if data}
                if collision_breakdown is not None:
                    result["collision_breakdown"] = collision_breakdown
                snapshot = dict(result)
                if collision_breakdown is not None:
                    snapshot["collision_breakdown"] = MappingProxyType(dict(collision_breakdown))
                self._last_profiling = MappingProxyType(snapshot)

            # Return profiling data if requested (pop values from _profiling_stats)
            if return_profiling:
                for data in self._profiling_stats.values():
                    if data:
                        data.pop()
                return result

            # Profiling output (only when time profiling is enabled and not returning data)
            if self._enable_time_profiling:
                # All values already written to _profiling_stats via [-1] assignment/accumulation

                # Print average statistics every N steps
                if self._step_count % self._profiling_interval == 0 and len(self._profiling_stats["total"]) > 0:
                    self._print_profiling_summary()

            # Memory profiling output (O(1) aggregator-based statistics)
            if self._enable_memory_profiling and tracemalloc.is_tracing():
                # Get memory usage (no import needed - already at top)
                current, peak = tracemalloc.get_traced_memory()
                current_mb = current / 1024 / 1024  # Convert to MB
                peak_mb = peak / 1024 / 1024  # Convert to MB

                # Update aggregators (O(1) memory)
                count = self._memory_stats["count"] + 1
                self._memory_stats["count"] = count
                self._memory_stats["current_sum"] += current_mb
                self._memory_stats["current_min"] = min(self._memory_stats["current_min"], current_mb)
                self._memory_stats["current_max"] = max(self._memory_stats["current_max"], current_mb)
                self._memory_stats["peak_sum"] += peak_mb
                self._memory_stats["peak_max"] = max(self._memory_stats["peak_max"], peak_mb)

                # Track first and last samples for growth calculation
                if count == 1:
                    self._memory_stats["current_first"] = current_mb
                self._memory_stats["current_last"] = current_mb

                # Print average statistics every N steps
                if self._step_count % self._profiling_interval == 0 and count > 0:
                    self._print_memory_profiling_summary()
        finally:
            self._in_step = False
            self._in_post_step = False
            self._in_pre_step = False

    def _restamp_pending_pose_cache(self) -> None:
        """Keep buffered poses' cache valid across a change of ``sim_time``.

        A dynamic body's cached pose is valid for one ``sim_time``. A pose
        written but not yet flushed to PyBullet lives only in that cache, so
        when the clock advances after ``pre_step`` the pending bodies are
        restamped; otherwise ``get_pose()`` would reread the unflushed PyBullet
        state.
        """
        for object_id in self._pending_pose_ids:
            pending = self._sim_objects_dict.get(object_id)
            if pending is not None:
                pending._cached_pose_sim_time = self.sim_time

    def _print_profiling_summary(self) -> None:
        """Print profiling statistics summary (average over last N steps).

        Prints all fields in _profiling_stats (built-in + custom) in a single line.
        Custom fields recorded via record_profiling() appear alongside
        built-in components automatically.
        """
        if not self._profiling_stats["total"]:
            return

        total_avg = statistics.mean(self._profiling_stats["total"])
        num_samples = len(self._profiling_stats["total"])

        # Print all fields from _profiling_stats dict (preserves insertion order)
        # Built-in fields come first (defined in __init__), custom fields follow
        stats_info = []
        for name, data in self._profiling_stats.items():
            if not data:
                continue
            avg = statistics.mean(data)
            percentage = (avg / total_avg * 100) if total_avg > 0 else 0
            stats_info.append(f"{name}={avg:.2f}ms ({percentage:.1f}%)")

        logger.info(f"[PROFILING] Last {num_samples} steps average: " + ", ".join(stats_info))

        # Clear all statistics for next interval
        for data in self._profiling_stats.values():
            data.clear()

    def _print_memory_profiling_summary(self) -> None:
        """
        Print memory profiling statistics summary (aggregator-based, O(1) memory).

        Note: Reports Python heap memory tracked by tracemalloc, not RSS (process memory).
        For RSS, use psutil.Process().memory_info().rss separately.
        """
        count = self._memory_stats["count"]
        if count == 0:
            return

        # Calculate statistics from aggregators
        current_avg = self._memory_stats["current_sum"] / count
        current_min = self._memory_stats["current_min"]
        current_max = self._memory_stats["current_max"]
        peak_avg = self._memory_stats["peak_sum"] / count
        peak_max = self._memory_stats["peak_max"]

        # Calculate memory growth (difference between first and last sample)
        memory_growth = self._memory_stats["current_last"] - self._memory_stats["current_first"]
        growth_str = f", growth={memory_growth:+.2f}MB"

        logger.info(
            f"[MEMORY] Last {count} steps: "
            f"current={current_avg:.2f}MB (min={current_min:.2f}, max={current_max:.2f}), "
            f"peak={peak_avg:.2f}MB (max={peak_max:.2f}){growth_str}"
        )

        # Reset aggregators for next interval (O(1) memory)
        self._memory_stats["count"] = 0
        self._memory_stats["current_sum"] = 0.0
        self._memory_stats["current_min"] = float("inf")
        self._memory_stats["current_max"] = 0.0
        self._memory_stats["current_first"] = 0.0
        self._memory_stats["current_last"] = 0.0
        self._memory_stats["peak_sum"] = 0.0
        self._memory_stats["peak_max"] = 0.0

        # Reset peak memory for interval-based peak tracking (requires Python 3.9+)
        # This makes "peak" meaningful for each interval rather than cumulative
        if tracemalloc.is_tracing() and hasattr(tracemalloc, "reset_peak"):
            tracemalloc.reset_peak()
        # Note: Without reset_peak (Python <3.9), peak becomes cumulative over entire run

    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """
        Get current memory usage (requires enable_memory_profiling=True).

        Returns:
            Dictionary with current and peak memory usage in MB, or None if not enabled.
            Keys: 'current_mb', 'peak_mb'

        Note:

            - Returns **Python heap memory** tracked by tracemalloc, NOT RSS (process memory).
            - tracemalloc tracks Python object allocations, not C extensions or OS-level memory.
            - For total process memory (RSS), use: psutil.Process().memory_info().rss
            - Peak is cumulative since last reset_peak() call (interval-based in profiling).

        Example::

            >>> sim = MultiRobotSimulationCore.from_dict(config)
            >>> # ... run simulation with enable_memory_profiling=True ...
            >>> mem = sim.get_memory_usage()
            >>> if mem:
            >>>     print(f"Python heap: {mem['current_mb']:.2f} MB, Peak: {mem['peak_mb']:.2f} MB")
        """
        if not self._enable_memory_profiling or not tracemalloc.is_tracing():
            return None

        current, peak = tracemalloc.get_traced_memory()
        return {"current_mb": current / 1024 / 1024, "peak_mb": peak / 1024 / 1024}
