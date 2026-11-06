"""Robot model resolution and auto-detection.

Provides a 3-tier model resolution system (URDF and SDF):

- **Tier 1** (``pybullet_data``): Always available with ``pip install pybullet``.
- **Tier 2** (ROS packages): Available when ROS description packages are installed.
- **Tier 3** (``robot_descriptions``): Available via ``pip install robot_descriptions``.

Usage::

    from pybullet_fleet.robot_models import resolve_model

    path = resolve_model("panda")           # Tier 1: pybullet_data
    path = resolve_model("robots/arm.urdf") # Direct path: returned as-is
    path = resolve_model("kiva_shelf")      # SDF model from pybullet_data
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pybullet as p
import pybullet_data

from pybullet_fleet.logging_utils import get_lazy_logger

logger = get_lazy_logger(__name__)


# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """Registry entry for a known robot model."""

    rel_path: str
    tier: str  # "pybullet_data", "ros", "robot_descriptions"
    install_hint: str = ""
    xacro_args: str = ""  # e.g. "ur_type:=ur5e name:=ur5e"


# ---------------------------------------------------------------------------
# Known models registry
# ---------------------------------------------------------------------------

KNOWN_MODELS: Dict[str, ModelEntry] = {
    # Tier 0: Local project URDFs (robots/ directory)
    "mobile_robot": ModelEntry("robots/mobile_robot.urdf", "local"),
    "arm_robot": ModelEntry("robots/arm_robot.urdf", "local"),
    "simple_cube": ModelEntry("robots/simple_cube.urdf", "local"),
    "mobile_manipulator": ModelEntry("robots/mobile_manipulator.urdf", "local"),
    "rail_arm_robot": ModelEntry("robots/rail_arm_robot.urdf", "local"),
    # Tier 1: pybullet_data (always available with pybullet)
    "panda": ModelEntry("franka_panda/panda.urdf", "pybullet_data"),
    "kuka_iiwa": ModelEntry("kuka_iiwa/model.urdf", "pybullet_data"),
    "husky": ModelEntry("husky/husky.urdf", "pybullet_data"),
    "racecar": ModelEntry("racecar/racecar.urdf", "pybullet_data"),
    "table": ModelEntry("table/table.urdf", "pybullet_data"),
    "tray": ModelEntry("tray/tray.urdf", "pybullet_data"),
    "plane": ModelEntry("plane.urdf", "pybullet_data"),
    "a1": ModelEntry("a1/a1.urdf", "pybullet_data"),
    "laikago": ModelEntry("laikago/laikago.urdf", "pybullet_data"),
    "aliengo": ModelEntry("aliengo/aliengo.urdf", "pybullet_data"),
    "mini_cheetah": ModelEntry("mini_cheetah/mini_cheetah.urdf", "pybullet_data"),
    "minitaur": ModelEntry("quadruped/minitaur.urdf", "pybullet_data"),
    "xarm6": ModelEntry("xarm/xarm6_with_gripper.urdf", "pybullet_data"),
    "wsg50_gripper": ModelEntry("gripper/wsg50_one_motor_gripper_new.sdf", "pybullet_data"),
    # pybullet_data: warehouse / factory objects
    "kiva_shelf": ModelEntry("kiva_shelf/model.sdf", "pybullet_data"),
    "table_square": ModelEntry("table_square/table_square.urdf", "pybullet_data"),
    "domino": ModelEntry("domino/domino.urdf", "pybullet_data"),
    "jenga": ModelEntry("jenga/jenga.urdf", "pybullet_data"),
    "lego": ModelEntry("lego/lego.urdf", "pybullet_data"),
    "mug": ModelEntry("objects/mug.urdf", "pybullet_data"),
    # Tier 2: ROS packages
    "ur5e": ModelEntry(
        "ur_description/urdf/ur.urdf.xacro",
        "ros",
        "apt install ros-${ROS_DISTRO}-ur-description",
        xacro_args="ur_type:=ur5e name:=ur5e safety_limits:=true safety_pos_margin:=0.15 safety_k_position:=20",
    ),
    "turtlebot3_burger": ModelEntry(
        "turtlebot3_description/urdf/turtlebot3_burger.urdf",
        "ros",
        "apt install ros-${ROS_DISTRO}-turtlebot3-description",
    ),
    "turtlebot3_waffle": ModelEntry(
        "turtlebot3_description/urdf/turtlebot3_waffle.urdf",
        "ros",
        "apt install ros-${ROS_DISTRO}-turtlebot3-description",
    ),
    "fetch": ModelEntry(
        "fetch_description",
        "ros",
        "apt install ros-${ROS_DISTRO}-fetch-description",
    ),
    # Tier 3: robot_descriptions (pip)
    "tiago": ModelEntry(
        "tiago_description",
        "robot_descriptions",
        "pip install robot_descriptions",
    ),
    "pr2": ModelEntry(
        "pr2_description",
        "robot_descriptions",
        "pip install robot_descriptions",
    ),
}


# ---------------------------------------------------------------------------
# User search paths
# ---------------------------------------------------------------------------

_user_search_paths: List[str] = []


def add_search_path(directory: str) -> None:
    """Register a directory to search for URDF/SDF files by name.

    User search paths are checked **before** the :data:`KNOWN_MODELS` registry,
    allowing users to shadow built-in models with custom versions.

    Example::

        from pybullet_fleet.robot_models import add_search_path, resolve_model

        add_search_path("/opt/company_robots")
        path = resolve_model("my_agv")  # → /opt/company_robots/my_agv.urdf

    Args:
        directory: Absolute or relative path to a directory containing
            ``.urdf`` or ``.sdf`` files.

    Raises:
        ValueError: If the directory does not exist.
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise ValueError(f"Search path does not exist: {directory}")
    if directory not in _user_search_paths:
        _user_search_paths.append(directory)
        logger.info("Added URDF search path: %s", directory)


def remove_search_path(directory: str) -> None:
    """Remove a previously registered search directory.

    No-op if the directory was not registered.

    Args:
        directory: The path that was passed to :func:`add_search_path`.
    """
    directory = os.path.abspath(directory)
    try:
        _user_search_paths.remove(directory)
        logger.info("Removed URDF search path: %s", directory)
    except ValueError:
        pass


def get_search_paths() -> List[str]:
    """Return the current list of user search paths (read-only copy)."""
    return list(_user_search_paths)


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------


def register_model(name: str, path_or_entry: Union[str, ModelEntry], *, force: bool = False) -> None:
    """Register a custom model so it can be resolved by name.

    This is the recommended way to make any URDF/SDF loadable via
    ``resolve_model(name)`` — including models from ``pybullet_data``,
    ROS packages, or ``robot_descriptions`` that are not in the built-in
    :data:`KNOWN_MODELS` registry.

    Example::

        from pybullet_fleet.robot_models import register_model, resolve_model

        # Simple: register an absolute path
        register_model("my_agv", "/opt/robots/my_agv.urdf")

        # Or register a pybullet_data model by relative path + tier
        register_model("sphere", ModelEntry("sphere2.urdf", "pybullet_data"))

        path = resolve_model("my_agv")

    Args:
        name: Short model name (e.g. ``"my_agv"``).
        path_or_entry: Either an absolute path to a URDF/SDF file, or a
            :class:`ModelEntry` for full tier metadata.
        force: If ``True``, overwrite an existing entry.  By default,
            raises :class:`ValueError` to prevent accidental overwrites.

    Raises:
        ValueError: If *name* is already registered and *force* is ``False``.
    """
    if name in KNOWN_MODELS and not force:
        raise ValueError(
            f"Model '{name}' is already registered (tier={KNOWN_MODELS[name].tier}). " f"Use force=True to overwrite."
        )
    if isinstance(path_or_entry, ModelEntry):
        entry = path_or_entry
    else:
        entry = ModelEntry(rel_path=path_or_entry, tier="user")
    KNOWN_MODELS[name] = entry
    logger.info("Registered model '%s' → %s (tier=%s)", name, entry.rel_path, entry.tier)


def unregister_model(name: str) -> None:
    """Remove a model from :data:`KNOWN_MODELS`.

    No-op if the name is not registered.

    Args:
        name: The model name to remove.
    """
    removed = KNOWN_MODELS.pop(name, None)
    if removed is not None:
        logger.info("Unregistered model '%s'", name)


def _resolve_from_search_paths(name: str) -> Optional[str]:
    """Try to find ``name.urdf`` or ``name.sdf`` in user search paths."""
    for directory in _user_search_paths:
        for ext in (".urdf", ".sdf"):
            candidate = os.path.join(directory, name + ext)
            if os.path.isfile(candidate):
                return candidate
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_model(name_or_path: str) -> str:
    """Resolve a model name or file path to an absolute URDF/SDF path.

    Resolution order:

    1. If ``name_or_path`` contains a path separator (``os.sep`` or ``/``) or
       ends with ``.urdf`` / ``.sdf``, treat as a direct file path: absolute paths and
       paths that exist relative to the CWD are returned as-is; a bundled
       relative path (e.g. ``"robots/mobile_robot.urdf"``) otherwise resolves to
       the packaged asset (absolute), falling back to the raw value if not found.
    2. Search user-registered directories (see :func:`add_search_path`).
    3. Look up in :data:`KNOWN_MODELS` registry and resolve by tier.
    4. **Auto-discovery fallback:** scan ``pybullet_data`` files and
       ``robot_descriptions`` modules for a matching name.
    5. Raise :class:`FileNotFoundError` with install instructions.

    Auto-discovery does **not** modify :data:`KNOWN_MODELS`.

    Args:
        name_or_path: Model name (e.g. ``"panda"``, ``"kiva_shelf"``) or file path.

    Returns:
        Absolute path to the model file (URDF or SDF).

    Raises:
        FileNotFoundError: If the model cannot be resolved.
    """
    # Direct path. Absolute or found relative to the CWD → use as-is. Otherwise
    # try resolving a bundled relative path (e.g. "robots/mobile_robot.urdf")
    # against the packaged asset root so pip-installed users (CWD != repo) and
    # repo users both work; fall back to the raw value if still not found.
    if os.sep in name_or_path or "/" in name_or_path or name_or_path.endswith((".urdf", ".sdf")):
        if os.path.isabs(name_or_path) or os.path.isfile(name_or_path):
            return name_or_path
        bundled = os.path.normpath(os.path.join(_PROJECT_ROOT, name_or_path))
        return bundled if os.path.isfile(bundled) else name_or_path

    # User search paths (highest priority for name-based lookup)
    user_hit = _resolve_from_search_paths(name_or_path)
    if user_hit is not None:
        return user_hit

    # Known model lookup
    if name_or_path in KNOWN_MODELS:
        entry = KNOWN_MODELS[name_or_path]
        return _resolve_by_tier(name_or_path, entry)

    # Fallback: auto-discover from installed packages
    fallback = _fallback_discover(name_or_path)
    if fallback is not None:
        return fallback

    # Unknown name
    available = ", ".join(sorted(KNOWN_MODELS.keys()))
    raise FileNotFoundError(f"Unknown robot model '{name_or_path}'. Available models: {available}")


def resolve_urdf(name_or_path: str) -> str:
    """Backward-compatible alias for :func:`resolve_model`.

    .. deprecated::
        Use :func:`resolve_model` instead.  ``resolve_urdf`` will be
        removed in a future release.
    """
    return resolve_model(name_or_path)


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

# Cache to avoid rescanning the filesystem on every resolve_model() miss.
_pybullet_data_cache: Optional[Dict[str, str]] = None


def _scan_pybullet_data() -> Dict[str, str]:
    """Scan ``pybullet_data`` directory for all URDF/SDF files.

    Returns a dict mapping stem name → absolute path.  When multiple files
    share a stem (rare — only 'model' and 'mug'), the first match wins
    (directory walk order, typically alphabetical).

    Results are cached in :data:`_pybullet_data_cache`.
    """
    global _pybullet_data_cache
    if _pybullet_data_cache is not None:
        return _pybullet_data_cache

    result: Dict[str, str] = {}
    data_path = pybullet_data.getDataPath()
    for root, _dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith((".urdf", ".sdf")):
                stem = os.path.splitext(f)[0]
                if stem not in result:
                    result[stem] = os.path.join(root, f)
    _pybullet_data_cache = result
    logger.debug("Scanned pybullet_data: %d models found", len(result))
    return result


def _scan_robot_descriptions() -> Dict[str, str]:
    """Scan ``robot_descriptions`` package for modules with ``URDF_PATH``.

    Returns a dict mapping a short name (e.g. ``"anymal_b"``) → absolute URDF
    path.  Only modules whose ``URDF_PATH`` points to an existing file are
    included.  MuJoCo-only modules (``*_mj_description``) are skipped.

    .. note:: Importing each module may trigger a git clone on first access
       (``robot_descriptions`` lazy-downloads).  This function is only called
       as a fallback, not on every ``resolve_model()``.
    """
    result: Dict[str, str] = {}
    try:
        import importlib
        import pkgutil

        import robot_descriptions

        for mod_info in pkgutil.iter_modules(robot_descriptions.__path__):
            mod_name = mod_info.name
            if not mod_name.endswith("_description"):
                continue
            if mod_name.endswith("_mj_description"):
                continue  # MuJoCo-only, no URDF
            if mod_name.startswith("_"):
                continue  # internal
            # Derive a short name: "panda_description" → "panda"
            short_name = mod_name[: -len("_description")] if mod_name.endswith("_description") else mod_name
            try:
                mod = importlib.import_module(f"robot_descriptions.{mod_name}")
                urdf_path = getattr(mod, "URDF_PATH", None)
                if urdf_path and os.path.isfile(urdf_path):
                    result[short_name] = urdf_path
            except Exception:
                continue
    except ImportError:
        pass  # robot_descriptions not installed
    return result


def _fallback_discover(name: str) -> Optional[str]:
    """Try to find *name* via auto-discovery (pybullet_data, robot_descriptions).

    Called by :func:`resolve_model` when *name* is not in :data:`KNOWN_MODELS`.
    Does **not** modify :data:`KNOWN_MODELS`.

    Returns:
        Absolute path to the URDF/SDF, or ``None`` if not found.
    """
    # 1. pybullet_data scan (cached, fast)
    pb_models = _scan_pybullet_data()
    if name in pb_models:
        logger.info("Auto-discovered '%s' in pybullet_data", name)
        return pb_models[name]

    # 2. robot_descriptions scan (slower, imports modules)
    rd_models = _scan_robot_descriptions()
    if name in rd_models:
        logger.info("Auto-discovered '%s' in robot_descriptions", name)
        return rd_models[name]

    return None


def discover_models(tier: str) -> Dict[str, str]:
    """Scan a tier and return all discoverable models.

    Unlike :data:`KNOWN_MODELS` (curated subset), this scans the actual
    installed package and returns every model found.

    Args:
        tier: ``"pybullet_data"`` or ``"robot_descriptions"``.

    Returns:
        Dict mapping short model name → absolute URDF/SDF path.

    Raises:
        ValueError: If *tier* is not a supported scan target.
    """
    if tier == "pybullet_data":
        return dict(_scan_pybullet_data())
    if tier == "robot_descriptions":
        return _scan_robot_descriptions()
    raise ValueError(f"Unknown tier '{tier}'. Supported: 'pybullet_data', 'robot_descriptions'")


def list_all_models() -> Dict[str, dict]:
    """Check availability of all known models.

    Returns:
        Dict mapping model name to ``{tier, available, path}``
        or ``{tier, available, error}``.
    """
    result: Dict[str, dict] = {}
    for name, entry in KNOWN_MODELS.items():
        try:
            path = resolve_model(name)
            result[name] = {
                "tier": entry.tier,
                "available": os.path.isfile(path),
                "path": path,
            }
        # Optional model packages can fail while downloading or importing an
        # asset.  A catalog should report that model as unavailable rather
        # than make ``--list`` and catalog demos fail as a whole.
        except Exception as e:
            result[name] = {
                "tier": entry.tier,
                "available": False,
                "error": str(e),
            }
    return result


# ---------------------------------------------------------------------------
# Tier resolvers (private)
# ---------------------------------------------------------------------------


# Root for bundled assets (robots/, config/, mesh/). These ship *inside* the
# package, so this is the package directory itself — works both from a repo
# checkout and from a pip-installed wheel.
_PROJECT_ROOT = os.path.dirname(__file__)


def _resolve_by_tier(name: str, entry: ModelEntry) -> str:
    """Resolve a ModelEntry based on its tier."""
    if entry.tier == "user":
        return entry.rel_path

    if entry.tier == "local":
        path = os.path.normpath(os.path.join(_PROJECT_ROOT, entry.rel_path))
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Local URDF not found: {path}")

    if entry.tier == "pybullet_data":
        return os.path.join(pybullet_data.getDataPath(), entry.rel_path)

    if entry.tier == "ros":
        return _resolve_ros(name, entry)

    if entry.tier == "robot_descriptions":
        return _resolve_robot_descriptions(name, entry)

    raise FileNotFoundError(f"Unknown tier '{entry.tier}' for model '{name}'")


def _resolve_ros(name: str, entry: ModelEntry) -> str:
    """Resolve via ROS package share directory.

    For xacro files, processes the xacro and caches the result.
    For plain URDF files containing ``package://`` URIs, creates a
    cached copy with URIs resolved to absolute paths so that PyBullet
    can find mesh files.
    """
    raw_path = _find_ros_urdf(entry)
    if raw_path is None:
        raise FileNotFoundError(f"Robot '{name}' requires a ROS package. Install with: {entry.install_hint}")

    # xacro: process and cache (already handles package:// resolution)
    if raw_path.endswith(".xacro"):
        return _resolve_xacro(name, raw_path, entry.xacro_args)

    # Plain URDF: check for package:// URIs that PyBullet can't resolve
    with open(raw_path) as f:
        content = f.read()
    if "package://" not in content:
        return raw_path  # No package:// URIs, safe to use directly

    # Cache a copy with package:// URIs resolved to absolute paths
    return _cache_resolved_urdf(name, raw_path, content)


def _find_ros_urdf(entry: ModelEntry) -> Optional[str]:
    """Locate the URDF/xacro file from a ROS package."""
    for get_pkg_dir in _ros_pkg_resolvers():
        try:
            pkg_name = entry.rel_path.split("/")[0]
            pkg_dir = get_pkg_dir(pkg_name)
            parts = entry.rel_path.split("/")[1:]
            if parts:
                full_path = os.path.join(pkg_dir, *parts)
                if os.path.isfile(full_path):
                    return full_path
            # Search for .urdf files in the package
            for root, _dirs, files in os.walk(pkg_dir):
                for f in files:
                    if f.endswith(".urdf"):
                        return os.path.join(root, f)
        except Exception:
            continue
    return None


def _ros_pkg_resolvers():
    """Yield callables that map pkg_name → pkg_dir."""
    try:
        from ament_index_python.packages import get_package_share_directory

        yield get_package_share_directory
    except ImportError:
        pass
    try:
        import rospkg

        yield rospkg.RosPack().get_path
    except ImportError:
        pass


def _cache_resolved_urdf(name: str, source_path: str, content: str) -> str:
    """Cache a URDF with package:// URIs replaced by absolute paths.

    Also strips unresolved xacro variables like ``${namespace}`` that
    some ROS packages (e.g. turtlebot3_description) leave in plain
    ``.urdf`` files.
    """
    import tempfile

    cache_dir = os.path.join(tempfile.gettempdir(), "pybullet_fleet_urdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"{name}.urdf")

    # Return cached if newer than source
    if os.path.isfile(cached) and os.path.getmtime(cached) >= os.path.getmtime(source_path):
        logger.debug("Using cached URDF for '%s': %s", name, cached)
        return cached

    resolved = _resolve_package_uris(content)
    resolved = _strip_xacro_vars(resolved)
    with open(cached, "w") as f:
        f.write(resolved)
    logger.info("Cached resolved URDF for '%s': %s", name, cached)
    return cached


def _resolve_xacro(name: str, xacro_path: str, xacro_args: str) -> str:
    """Process a xacro file and return a cached plain URDF path.

    Uses ``xacro`` Python API first, falls back to subprocess.
    The generated URDF is cached in a temp directory and re-used
    if the cache is newer than the source xacro.

    ``package://`` URIs in the output are resolved to absolute filesystem
    paths so that PyBullet can find mesh files.

    The cache key includes a hash of ``xacro_args`` so that different
    parameter sets (e.g. ``ur_type:=ur5e`` vs ``ur_type:=ur10e``) produce
    separate cached files.
    """
    import hashlib
    import tempfile

    cache_dir = os.path.join(tempfile.gettempdir(), "pybullet_fleet_urdf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Include xacro_args in the cache filename to avoid collisions
    if xacro_args:
        args_hash = hashlib.md5(xacro_args.encode()).hexdigest()[:8]
        cached_urdf = os.path.join(cache_dir, f"{name}_{args_hash}.urdf")
    else:
        cached_urdf = os.path.join(cache_dir, f"{name}.urdf")

    # Return cached URDF if it exists and is newer than the xacro source
    if os.path.isfile(cached_urdf) and os.path.isfile(xacro_path):
        if os.path.getmtime(cached_urdf) >= os.path.getmtime(xacro_path):
            logger.debug("Using cached URDF for '%s': %s", name, cached_urdf)
            return cached_urdf

    # Parse xacro_args string into a mappings dict
    mappings: Dict[str, str] = {}
    if xacro_args:
        for token in xacro_args.split():
            if ":=" in token:
                k, v = token.split(":=", 1)
                mappings[k] = v

    urdf_xml: Optional[str] = None

    # Try xacro Python API
    try:
        import xacro

        doc = xacro.process_file(xacro_path, mappings=mappings)
        urdf_xml = doc.toxml()
    except ImportError:
        pass

    # Fallback: subprocess
    if urdf_xml is None:
        try:
            import subprocess

            cmd = ["xacro", xacro_path]
            for k, v in mappings.items():
                cmd.append(f"{k}:={v}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                urdf_xml = result.stdout
            else:
                logger.error("xacro failed for '%s': %s", name, result.stderr)
        except Exception as e:
            logger.error("xacro subprocess failed for '%s': %s", name, e)

    if urdf_xml is None:
        raise FileNotFoundError(f"Failed to process xacro file: {xacro_path}")

    # Replace package:// URIs with absolute paths for PyBullet
    urdf_xml = _resolve_package_uris(urdf_xml)

    with open(cached_urdf, "w") as f:
        f.write(urdf_xml)
    logger.info("Generated URDF via xacro for '%s': %s", name, cached_urdf)
    return cached_urdf


def _resolve_package_uris(urdf_xml: str) -> str:
    """Replace ``package://pkg/path`` URIs with absolute filesystem paths.

    PyBullet cannot resolve ``package://`` URIs.  This helper resolves
    them via ``ament_index_python`` (ROS 2) so that mesh files are found.
    Unresolvable URIs are left unchanged (PyBullet will warn).
    """
    import re

    _PKG_URI_RE = re.compile(r"package://([^/]+)/([\S]*?)([\"\'])")

    def _replace(m: re.Match) -> str:
        pkg_name = m.group(1)
        rel_path = m.group(2)
        quote = m.group(3)
        try:
            from ament_index_python.packages import get_package_share_directory

            pkg_dir = get_package_share_directory(pkg_name)
            return os.path.join(pkg_dir, rel_path) + quote
        except Exception:
            pass
        try:
            import rospkg

            pkg_dir = rospkg.RosPack().get_path(pkg_name)
            return os.path.join(pkg_dir, rel_path) + quote
        except Exception:
            pass
        # Return original if resolution fails
        return m.group(0)

    return _PKG_URI_RE.sub(_replace, urdf_xml)


def _strip_xacro_vars(urdf_xml: str) -> str:
    """Remove unresolved xacro variable references from URDF XML.

    Some ROS packages (e.g. ``turtlebot3_description``) ship plain ``.urdf``
    files that still contain xacro constructs like ``${namespace}`` with a
    default empty-string value.  PyBullet loads these literally, causing
    link/joint names like ``${namespace}base_link`` instead of ``base_link``.

    This strips:
    - ``${varname}`` references (replaced with empty string)
    - ``<xacro:arg .../>`` and ``<xacro:property .../>`` elements
    - ``$(arg ...)`` and ``$(find ...)`` references
    """
    import re

    # Remove ${varname} references (default is typically empty)
    urdf_xml = re.sub(r"\$\{[^}]+\}", "", urdf_xml)
    # Remove $(arg ...) and $(find ...) references
    urdf_xml = re.sub(r"\$\((?:arg|find)\s+[^)]+\)", "", urdf_xml)
    # Remove <xacro:arg .../> and <xacro:property .../> elements
    urdf_xml = re.sub(r"<xacro:\w+[^>]*/>\s*", "", urdf_xml)
    return urdf_xml


def _resolve_robot_descriptions(name: str, entry: ModelEntry) -> str:
    """Resolve via ``robot_descriptions`` pip package."""
    try:
        import importlib

        mod = importlib.import_module(f"robot_descriptions.{entry.rel_path}")
        urdf_path = getattr(mod, "URDF_PATH", None)
        if urdf_path and os.path.isfile(urdf_path):
            return urdf_path
        raise FileNotFoundError(f"URDF_PATH not found in robot_descriptions.{entry.rel_path}")
    except ImportError:
        raise FileNotFoundError(
            f"Robot '{name}' requires the robot_descriptions package. " f"Install with: {entry.install_hint}"
        )


# ---------------------------------------------------------------------------
# Robot profile auto-detection
# ---------------------------------------------------------------------------


@dataclass
class RobotProfile:
    """Auto-detected robot profile from URDF introspection.

    This is a **pre-spawn, read-only snapshot** intended for external
    tooling (GUIs, catalog browsers, spawn helpers).  It is **not** used
    by :class:`~pybullet_fleet.agent.Agent` at runtime — the Agent
    maintains its own joint info, IK state, and velocity limits
    independently.  Modifying a ``RobotProfile`` instance has no effect
    on a running simulation.

    All fields can be overridden after detection or via
    ``auto_detect_profile(..., **overrides)``.
    """

    urdf_path: str
    robot_type: str = "unknown"  # "arm", "mobile", "mobile_manipulator"
    ee_link_name: Optional[str] = None
    movable_joint_names: List[str] = field(default_factory=list)
    joint_lower_limits: List[float] = field(default_factory=list)
    joint_upper_limits: List[float] = field(default_factory=list)
    joint_max_velocities: List[float] = field(default_factory=list)
    num_joints: int = 0
    # Mapping from joint name → index in the full PyBullet joint array
    _movable_joint_indices: List[int] = field(default_factory=list)
    # Approximate max reach from base to EE (metres). ``None`` for non-arm robots.
    arm_reach: Optional[float] = None
    # Resolved EE link index in the PyBullet joint array. -1 if no EE.
    ee_link_index: int = -1

    def make_joint_targets(self, fractions: List[float], agent: object) -> List[float]:
        """Build a full joint-target list from per-movable-joint *fractions*.

        Each fraction in ``[-1, 1]`` is mapped to the joint's limit range::

            target = mid + fraction * half_range

        Fixed joints and unmapped joints keep their current position.

        Args:
            fractions: One value per movable joint (same order as
                :attr:`movable_joint_names`).  Length must match.
            agent: An :class:`~pybullet_fleet.agent.Agent` instance whose
                ``joint_info`` determines the full joint count.

        Returns:
            A list with one target per joint (``len(agent.joint_info)``),
            suitable for ``agent.set_all_joints_targets()``.
        """
        joint_info = getattr(agent, "joint_info", [])
        n_total = len(joint_info)
        if n_total == 0:
            return []

        # Start from current positions
        targets: List[float] = [0.0] * n_total

        # Fill movable joints using fractions
        for idx, (frac, lo, hi) in enumerate(zip(fractions, self.joint_lower_limits, self.joint_upper_limits)):
            if idx < len(self._movable_joint_indices):
                ji = self._movable_joint_indices[idx]
                mid = (lo + hi) / 2.0
                half = (hi - lo) / 2.0
                targets[ji] = mid + frac * half

        return targets


_EE_KEYWORDS = ("hand", "ee", "end_effector", "tool", "gripper", "tcp")
_LEG_KEYWORDS = ("hip", "thigh", "calf", "knee", "foot", "shin", "ankle", "shank", "abduct")
_LEG_GROUP_KEYWORDS = (
    "fr",
    "fl",
    "rr",
    "rl",
    "hr",
    "hl",  # hind-right / hind-left (mini_cheetah)
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
    "lf",
    "rf",
    "lb",
    "rb",
)


def detect_robot_type(body_id: int, physics_client: int) -> str:
    """Detect robot type from an already-loaded PyBullet body.

    Unlike :func:`auto_detect_profile`, this does **not** load or remove
    any bodies — it introspects a body that is already in the simulation.
    This makes it safe to call in GUI mode without corrupting GPU resources.

    .. note:: **Best-effort heuristic.**  The classification relies on
       joint types and link-name keywords, which does not cover all robot
       designs.  Known limitations:

       * A differential-drive robot with a passive caster (bounded
         revolute) is mis-classified as ``"mobile_manipulator"``.
       * Humanoids and hexapods do not match the quadruped keyword set
         and fall back to ``"arm"`` or ``"unknown"``.
       * Grippers with only bounded joints are classified as ``"arm"``.

       Use ``auto_detect_profile(..., robot_type="mobile")`` or the
       ``**overrides`` parameter to correct mis-detections.

    Args:
        body_id: PyBullet body ID (already loaded).
        physics_client: PyBullet physics client ID.

    Returns:
        One of ``"object"``, ``"arm"``, ``"mobile"``,
        ``"mobile_manipulator"``, ``"quadruped"``, or ``"unknown"``.
    """
    n = p.getNumJoints(body_id, physicsClientId=physics_client)
    if n == 0:
        return "object"

    has_continuous = False
    has_bounded = False
    link_names: List[str] = []

    for i in range(n):
        info = p.getJointInfo(body_id, i, physicsClientId=physics_client)
        jtype: int = info[2]
        lower: float = info[8]
        upper: float = info[9]
        link_names.append(info[12].decode("utf-8"))

        if jtype == p.JOINT_REVOLUTE:
            if lower >= upper:
                has_continuous = True
            else:
                has_bounded = True
        elif jtype == p.JOINT_PRISMATIC:
            has_bounded = True

    # Quadruped: leg keywords + ≥4 leg groups
    if any(any(kw in ln.lower() for kw in _LEG_KEYWORDS) for ln in link_names):
        groups: set = set()
        for ln in link_names:
            ll = ln.lower()
            for kw in _LEG_GROUP_KEYWORDS:
                if kw in ll:
                    groups.add(kw)
        if len(groups) >= 4:
            return "quadruped"

    if has_bounded and not has_continuous:
        return "arm"
    if has_continuous and not has_bounded:
        return "mobile"
    if has_continuous and has_bounded:
        return "mobile_manipulator"
    return "unknown"


def auto_detect_profile(
    urdf_path_or_body_id: Union[str, int],
    physics_client: int,
    **overrides: object,
) -> RobotProfile:
    """Introspect a URDF/SDF or an already-loaded PyBullet body.

    When *urdf_path_or_body_id* is a **string**, the URDF/SDF is loaded
    temporarily and removed after introspection.  When it is an **int**
    (PyBullet body ID), the already-loaded body is introspected in place
    — no load or remove occurs.  Use the ``int`` form in GUI mode to
    avoid GPU resource corruption from ``removeBody``.

    Robot type detection is heuristic-based (see :func:`detect_robot_type`
    for known limitations).  Pass ``robot_type=`` in *overrides* to
    correct any mis-detection.

    Args:
        urdf_path_or_body_id: Absolute path to URDF/SDF **or** an
            already-loaded PyBullet body ID (int).
        physics_client: PyBullet physics client ID.
        **overrides: Override any auto-detected field.

    Returns:
        RobotProfile with detected (or overridden) values.
    """
    if isinstance(urdf_path_or_body_id, int):
        # Already-loaded body — introspect without load/remove
        body_id = urdf_path_or_body_id
        # Recover URDF path from PyBullet body info (best-effort)
        body_info = p.getBodyInfo(body_id, physicsClientId=physics_client)
        urdf_path = body_info[0].decode("utf-8") if body_info else "unknown"
        profile = _introspect_body(body_id, urdf_path, physics_client)
    else:
        import warnings

        warnings.warn(
            "auto_detect_profile(str) loads and removes a temporary body, "
            "which can corrupt GPU resources in GUI mode. "
            "Prefer passing a body_id (int) for already-loaded bodies.",
            stacklevel=2,
        )
        urdf_path = urdf_path_or_body_id
        body_ids_to_remove: list[int] = []
        if urdf_path.endswith(".sdf"):
            bodies = p.loadSDF(urdf_path, physicsClientId=physics_client)
            if not bodies:
                raise FileNotFoundError(f"SDF file loaded no bodies: {urdf_path}")
            body_id = bodies[0]
            body_ids_to_remove = list(bodies)
        else:
            body_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=physics_client)
            body_ids_to_remove = [body_id]
        try:
            profile = _introspect_body(body_id, urdf_path, physics_client)
        finally:
            for bid in body_ids_to_remove:
                p.removeBody(bid, physicsClientId=physics_client)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(profile, key):
            object.__setattr__(profile, key, value)
        else:
            logger.warning("Unknown RobotProfile field: %s", key)

    return profile


def _introspect_body(body_id: int, urdf_path: str, physics_client: int) -> RobotProfile:
    """Extract joint info, detect EE link and robot type.

    Robot type classification is delegated to :func:`detect_robot_type`
    (single source of truth).
    """
    n_joints = p.getNumJoints(body_id, physicsClientId=physics_client)

    movable_names: List[str] = []
    movable_indices: List[int] = []
    lower_limits: List[float] = []
    upper_limits: List[float] = []
    max_velocities: List[float] = []
    has_revolute_or_prismatic = False
    ee_link: Optional[str] = None

    # Build parent→children map for leaf detection
    child_indices: set[int] = set()
    for i in range(n_joints):
        info = p.getJointInfo(body_id, i, physicsClientId=physics_client)
        parent_idx: int = info[16]
        if parent_idx >= 0:
            child_indices.add(parent_idx)

    for i in range(n_joints):
        info = p.getJointInfo(body_id, i, physicsClientId=physics_client)
        joint_type: int = info[2]
        joint_name: str = info[1].decode("utf-8")
        link_name: str = info[12].decode("utf-8")
        lower: float = info[8]
        upper: float = info[9]
        max_vel: float = info[11]

        # Collect bounded (controllable) joints for the profile
        if joint_type == p.JOINT_REVOLUTE and lower < upper:
            has_revolute_or_prismatic = True
            movable_names.append(joint_name)
            movable_indices.append(i)
            lower_limits.append(float(lower))
            upper_limits.append(float(upper))
            max_velocities.append(float(max_vel))
        elif joint_type == p.JOINT_PRISMATIC:
            has_revolute_or_prismatic = True
            movable_names.append(joint_name)
            movable_indices.append(i)
            lower_limits.append(float(lower))
            upper_limits.append(float(upper))
            max_velocities.append(float(max_vel))

        # EE detection: keyword match
        if ee_link is None:
            for kw in _EE_KEYWORDS:
                if kw in link_name.lower():
                    ee_link = link_name
                    break

    # Fallback: leaf link (no children) among movable links
    if ee_link is None and has_revolute_or_prismatic:
        for i in range(n_joints - 1, -1, -1):
            if i not in child_indices:
                info = p.getJointInfo(body_id, i, physicsClientId=physics_client)
                jtype: int = info[2]
                jlower: float = info[8]
                jupper: float = info[9]
                # Bounded revolute or prismatic (skip continuous/fixed)
                is_bounded_revolute = jtype == p.JOINT_REVOLUTE and jlower < jupper
                if is_bounded_revolute or jtype == p.JOINT_PRISMATIC:
                    ee_link = info[12].decode("utf-8")
                    break

    # Delegate type detection to the single-source-of-truth function
    robot_type = detect_robot_type(body_id, physics_client)

    # For mobile/quadruped: no EE
    if robot_type in ("mobile", "quadruped"):
        ee_link = None

    # Arm reach estimation — test several joint configurations via FK
    #
    # WARNING: This is a **rough approximation**.  It samples only 6 fixed
    # joint configurations and measures the max Euclidean distance from
    # base origin to EE.  Known limitations:
    #   • Underestimates the true workspace envelope (continuous space
    #     sampled with only 6 discrete points).
    #   • Distance is measured from the base frame origin, not the
    #     shoulder joint — includes the base-to-shoulder offset.
    #   • For mobile_manipulators the base moves, making the value
    #     ambiguous (it reflects the arm reach at the initial base pose).
    arm_reach: Optional[float] = None
    if robot_type in ("arm", "mobile_manipulator") and ee_link is not None:
        import math

        # Resolve EE link index
        ee_link_idx = n_joints - 1
        for i in range(n_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=physics_client)
            if info[12].decode("utf-8") == ee_link:
                ee_link_idx = i
                break

        base_pos = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)[0]
        max_dist = 0.0

        # Save joint states so probing does not change a live body's pose
        saved_states = [p.getJointState(body_id, ji, physicsClientId=physics_client)[:2] for ji in movable_indices]

        # Sample configs: zeros, upper limits, lower limits, midpoints, alternating
        configs: list[list[float]] = []
        n_mov = len(movable_indices)
        configs.append([0.0] * n_mov)  # all zeros
        configs.append([(lo + hi) / 2.0 for lo, hi in zip(lower_limits, upper_limits)])  # midpoints
        configs.append(list(upper_limits))  # all upper
        configs.append(list(lower_limits))  # all lower
        # Alternating upper/lower
        configs.append([hi if k % 2 == 0 else lo for k, (lo, hi) in enumerate(zip(lower_limits, upper_limits))])
        configs.append([lo if k % 2 == 0 else hi for k, (lo, hi) in enumerate(zip(lower_limits, upper_limits))])

        try:
            for cfg in configs:
                for idx_in_mov, ji in enumerate(movable_indices):
                    if idx_in_mov < len(cfg):
                        p.resetJointState(body_id, ji, cfg[idx_in_mov], physicsClientId=physics_client)
                ee_pos = p.getLinkState(body_id, ee_link_idx, physicsClientId=physics_client)[0]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ee_pos, base_pos)))
                if dist > max_dist:
                    max_dist = dist
        finally:
            # Restore the joint states found before probing, even if probing raised
            for ji, (pos, vel) in zip(movable_indices, saved_states):
                p.resetJointState(body_id, ji, pos, vel, physicsClientId=physics_client)

        if max_dist > 0.0:
            arm_reach = max_dist

    return RobotProfile(
        urdf_path=urdf_path,
        robot_type=robot_type,
        ee_link_name=ee_link,
        movable_joint_names=movable_names,
        joint_lower_limits=lower_limits,
        joint_upper_limits=upper_limits,
        joint_max_velocities=max_velocities,
        num_joints=n_joints,
        _movable_joint_indices=movable_indices,
        arm_reach=arm_reach,
        ee_link_index=ee_link_idx if (arm_reach is not None) else -1,
    )
