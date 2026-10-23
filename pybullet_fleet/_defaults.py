"""Single source of truth for all default parameter values.

Environment variables ``PBF_{SECTION}_{KEY}`` override defaults at
module load time (e.g. ``PBF_SIMULATION_TIMESTEP=0.05``).

If ``python-dotenv`` is installed, a ``.env`` file in the working
directory is loaded automatically.  This is optional — the module
works without it.
"""

import os
from typing import Any, Dict

# Optional: load .env if python-dotenv is installed (no hard dependency)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Raw defaults — the ONLY place default values are defined
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "simulation": {
        "target_rtf": 1.0,
        "timestep": 0.1,
        "duration": 0,
        "gui": True,
        "physics": False,
        "monitor": True,
        # None: follow `gui`, so a headless run opens no window.
        "enable_monitor_gui": None,
        # Simulated seconds between DataMonitor refreshes (build and write a
        # frame). None: monitor_update_interval_window with the monitor
        # window, monitor_update_interval_headless without it.
        "monitor_update_interval": None,
        # The refresh used when monitor_update_interval is None: the window
        # reads the file every 0.5 wall seconds, so 0.25 s keeps it current at
        # a real-time factor of 1; without a window, a frame a second is
        # enough for last_monitor_frame.
        "monitor_update_interval_window": 0.25,
        "monitor_update_interval_headless": 1.0,
        # Wall seconds between INFO-level [MONITOR] lines; 0 keeps them off.
        "monitor_log_interval": 1.0,
        "log_level": "warn",
        "max_steps_per_frame": 10,
        "max_sleep_frames": 4.0,  # real-time sleep clamp: at most this many normal frame intervals
        "gui_min_fps": 30,
        # Visualizer settings
        "enable_collision_shapes": False,
        "enable_structure_transparency": False,
        "enable_shadows": True,
        "enable_gui_panel": False,
        "ignore_static_collision": True,
        "ignore_attached_collision": True,
        "enable_time_profiling": False,
        "profiling_interval": 100,
        "enable_memory_profiling": False,
        "enable_collision_color_change": False,
        "spatial_hash_cell_size_mode": "auto_initial",
        "collision_margin": 0.02,
        "multi_cell_threshold": 1.5,
        "enable_floor": True,
        # Window size
        "window_width": 1024,
        "window_height": 768,
        # Monitor window
        "monitor_width": 680,
        "monitor_height": 480,
        # monitor_x / monitor_y: -1 means "let the window manager decide"
        # (tkinter geometry string omits +x+y). Set to >= 0 for absolute
        # virtual-screen coordinates.
        "monitor_x": -1,
        "monitor_y": -1,
        # --- None sentinels (not env-overridable) ---
        # Listed here for documentation: these are valid SimulationParams
        # fields whose default is None (meaning "auto" or "disabled").
        # None = check every step (no throttling)
        "collision_check_frequency": None,
        # None = auto-select based on physics mode
        # (physics ON → CONTACT_POINTS, physics OFF → CLOSEST_POINTS)
        "collision_detection_method": None,
        # None = auto-calculate from largest entity AABB
        "spatial_hash_cell_size": None,
        # Default: interactive camera (Shift+drag pan, +/- zoom, o top-down)
        "camera_config": None,
        # Optional PyBullet GUI main-light and shadow-map settings.
        "lighting_config": None,
        # None = no additional model search paths
        "model_paths": None,
    },
    "agent": {
        "motion_mode": "differential",
        "use_fixed_base": False,
        "global_scaling": 1.0,
        "joint_tolerance": None,
        # Not centralized (complex / per-instance):
        #   urdf_path   — str, per-robot URDF path
        #   ik_params   — IKParams dataclass (see "ik" section for scalars)
        #   controller  — dict, controller type + ControllerParams fields
        #                 (kinematic limits, navigation_2d, default_direction, …)
    },
    "controller": {
        # Kinematic limits — used as ControllerParams defaults.
        # ``math.inf`` in an explicit ControllerParams instance still means
        # "no limit"; only unspecified fields fall back to these values.
        "max_linear_vel": 2.0,
        "max_linear_accel": 5.0,
        "max_angular_vel": 3.0,
        "max_angular_accel": 10.0,
        "cmd_vel_timeout": 0.0,
        "navigation_2d": False,
        # What a moving agent does with a goal it cannot reach from its current
        # motion within its limits: "brake", "reject", "wait_complete" or
        # "immediate". See ControllerParams.goal_replacement.
        "goal_replacement": "brake",
        # Not centralized (complex enum):
        #   default_direction — MovementDirection enum (forward / backward / …)
    },
    "sim_object": {
        "mass": 0.0,
        "pickable": True,
        "collision_mode": "normal_3d",
        # Not centralized (complex / per-instance):
        #   name                 — str, required per-object identity
        #   initial_pose         — Pose object
        #   visual_shape         — ShapeParams (mesh/box/sphere/cylinder)
        #   collision_shape      — ShapeParams
        #   urdf_path            — str, per-object model name or path
        #   visual_frame_pose    — Pose offset for visual shape
        #   collision_frame_pose — Pose offset for collision shape
        #   user_data            — dict, arbitrary per-object metadata
    },
    "shape": {
        "radius": 0.5,
        "height": 1.0,
        "rgba_color": (0.8, 0.8, 0.8, 1.0),
        # Not centralized (mutable list defaults — use field(default_factory)):
        #   mesh_scale    — [1.0, 1.0, 1.0]
        #   half_extents  — [0.5, 0.5, 0.5]
    },
    "ik": {
        "max_outer_iterations": 5,
        "convergence_threshold": 0.01,
        "max_inner_iterations": 200,
        "residual_threshold": 1e-4,
        "reachability_tolerance": 0.02,
        # Not centralized (complex / per-robot):
        #   seed_quartiles  — tuple of floats (default: 0.25, 0.5, 0.75)
        #   ik_joint_names  — optional tuple of joint name strings
    },
}

# ---------------------------------------------------------------------------
# Type coercion for env var string → Python type
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    bool: lambda v: v.lower() in ("true", "1", "yes"),
    int: int,
    float: float,
    str: str,
}


def _apply_env_overrides() -> None:
    """Apply ``PBF_{SECTION}_{KEY}`` env-var overrides to ``_DEFAULTS``."""
    for section, params in _DEFAULTS.items():
        for key, default_val in params.items():
            if default_val is None:
                continue  # None-typed fields are not env-overridable
            env_key = f"PBF_{section.upper()}_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                coerce = _TYPE_MAP.get(type(default_val), str)
                params[key] = coerce(env_val)


# Immutable baseline — captured BEFORE env-override pass so that
# ``reload_defaults()`` resets to pure code defaults, not env-tainted values.
import copy as _copy

_BASELINE: Dict[str, Dict[str, Any]] = _copy.deepcopy(_DEFAULTS)

_apply_env_overrides()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get(section: str, key: str) -> Any:
    """Return the (possibly env-overridden) default for *section*.*key*."""
    return _DEFAULTS[section][key]


def reload_defaults() -> None:
    """Re-apply env overrides from a clean baseline.

    Resets all values to the immutable baseline captured at import time,
    then applies current ``PBF_*`` environment variables on top.  This
    ensures that *removed* env vars revert to their original defaults.

    Useful after ``monkeypatch.setenv`` / ``monkeypatch.delenv`` in tests.
    """
    _restore(_copy.deepcopy(_BASELINE))
    _apply_env_overrides()


def _snapshot() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy of the current defaults (for test save/restore)."""
    import copy

    return copy.deepcopy(_DEFAULTS)


def _restore(snapshot: Dict[str, Dict[str, Any]]) -> None:
    """Restore defaults from a snapshot captured by :func:`_snapshot`."""
    for section, params in snapshot.items():
        _DEFAULTS[section].update(params)


#: Convenience accessors — import these in consuming modules.
SIMULATION: Dict[str, Any] = _DEFAULTS["simulation"]
AGENT: Dict[str, Any] = _DEFAULTS["agent"]
CONTROLLER: Dict[str, Any] = _DEFAULTS["controller"]
SIM_OBJECT: Dict[str, Any] = _DEFAULTS["sim_object"]
SHAPE: Dict[str, Any] = _DEFAULTS["shape"]
IK: Dict[str, Any] = _DEFAULTS["ik"]

#: Keys in ``simulation`` whose default is ``None`` (not env-overridable).
NONE_SENTINEL_KEYS = frozenset(k for k, v in _DEFAULTS["simulation"].items() if v is None)
