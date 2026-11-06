# pybullet_fleet/sdf_loader.py
"""SDF / mesh world loader for PyBulletFleet.

**Why self-implemented?**  PyBullet's ``loadSDF()`` crashes on multi-root-link
models (segfault) and rejects Gazebo SDF joints without ``<dynamics>`` defaults.
No usable Python SDF→URDF library exists — ``sdformat_urdf`` is C++-only with
no Python bindings, ``pysdf`` (PyPI) fails to build on Python 3.12, and
``libsdformat14`` is a C++ template API not callable via ctypes.

This module provides:

- :func:`load_mesh_directory` — OBJ/STL mesh directory loader (from world_loader).
- :func:`load_sdf_world` — Load a single SDF ``<model>`` (building) as meshes.
- :func:`load_sdf_world_file` — Load a ``.world`` file, resolving ``<include>``
  elements (Gazebo Fuel URIs, ``model://`` local refs) into SimObjects.
- :func:`resolve_sdf_to_urdf` — Convert a robot SDF into a temporary URDF.

Backward-compatible aliases:

- :func:`load_rmf_world` → :func:`load_mesh_directory`
"""

import glob
import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import pybullet as p

from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import SimObject, ShapeParams, _physics_client_of
from pybullet_fleet.types import CollisionMode

logger = logging.getLogger(__name__)

_FUEL_CACHE = os.path.expanduser("~/.gz/fuel")
_FUEL_HOSTS = ("fuel.gazebosim.org", "fuel.ignitionrobotics.org")

#: Default color for scenery objects without material info.
#: Matches Gazebo Sim (Harmonic) default — a neutral light grey
#: that renders well under PyBullet's default lighting.
DEFAULT_SCENERY_COLOR = [0.7, 0.7, 0.7, 1.0]


def _is_fuel_uri(uri: str) -> bool:
    """Return whether *uri* uses a supported Gazebo Fuel host."""
    return urlparse(uri).hostname in _FUEL_HOSTS


def _fuel_cache_hosts(uri: str) -> Tuple[str, ...]:
    """Return preferred and compatible Fuel cache hosts for *uri*."""
    host = urlparse(uri).hostname
    if host not in _FUEL_HOSTS:
        return ()
    return (host, *(candidate for candidate in _FUEL_HOSTS if candidate != host))


def _fuel_path_parts(uri: str) -> List[str]:
    """Return Fuel API path parts without its optional ``1.0`` prefix."""
    parts = [unquote(part) for part in urlparse(uri).path.split("/") if part]
    if parts[:1] == ["1.0"]:
        parts = parts[1:]
    return parts


# ---------------------------------------------------------------------------
# model:// URI resolution
# ---------------------------------------------------------------------------


def _resolve_model_uri(uri: str, model_dir: str, model_name: str) -> str:
    """Resolve a ``model://ModelName/path`` URI to an absolute filesystem path.

    Also handles Gazebo Fuel HTTPS URIs like
    ``https://fuel.gazebosim.org/1.0/owner/models/name/ver/files/path``
    by mapping them to the local Fuel cache (``~/.gz/fuel/...``).

    Args:
        uri: Raw URI from SDF.
        model_dir: Absolute directory containing the SDF file.
        model_name: The ``<model name="...">`` attribute from the SDF.

    Returns:
        Absolute filesystem path.
    """
    if _is_fuel_uri(uri):
        # Fuel API URI: /1.0/owner/models/name/ver/files/rest
        # Cache URI:     ~/.gz/fuel/<host>/owner/models/name/ver/rest
        parts = _fuel_path_parts(uri)
        if parts:
            parts[0] = parts[0].lower()  # owner
        if len(parts) >= 3:
            parts[2] = parts[2].lower()  # model name
        if "files" in parts:
            parts.remove("files")
        for host in _fuel_cache_hosts(uri):
            local = os.path.join(_FUEL_CACHE, host, *parts)
            if os.path.isfile(local):
                return local
        return os.path.join(_FUEL_CACHE, _fuel_cache_hosts(uri)[0], *parts)

    if not uri.startswith("model://"):
        # Plain relative path (e.g. "meshes/suv.obj") — resolve relative to model_dir
        if not os.path.isabs(uri):
            return os.path.join(model_dir, uri)
        return uri

    # model://ModelName/rest/of/path  OR  model://other_model/...
    rest = uri[len("model://") :]
    parts = rest.split("/", 1)
    if len(parts) == 2 and parts[0].lower() == model_name.lower():
        return os.path.join(model_dir, parts[1])
    # Fallback: try relative to model_dir's parent (co-located models)
    return os.path.join(os.path.dirname(model_dir), rest)


# ---------------------------------------------------------------------------
# <submesh> extraction
# ---------------------------------------------------------------------------
#
# Gazebo's ``<mesh><submesh><name>X</name></submesh></mesh>`` renders only the
# named part of a multi-part mesh file (e.g. the Caddy's ``polaris.dae`` packs
# the chassis + 4 wheels + pedals + steering wheel in a single DAE, and each
# link selects its own submesh). PyBullet cannot select a submesh, so without
# this every link would load the *whole* model and the parts would overlap.
#
# We extract the requested geometry (with its scene-node transform baked in, to
# match Gazebo/assimp) into a standalone OBJ, cached on disk, and reference that
# instead. Requires the optional ``sdf`` extra (``pip install
# 'pybullet-fleet[sdf]'`` → trimesh + pycollada); without it we fall back to the
# full mesh and warn once.

_SUBMESH_CACHE_DIR = os.path.join(tempfile.gettempdir(), "pybullet_fleet_submesh")
_submesh_lib_warned = False


def _collada_unit_meter(path: str) -> float:
    """Return a COLLADA file's ``<unit meter="X">`` scale (metres per unit).

    COLLADA assets are often authored in non-metre units (the Caddy's
    ``polaris.dae`` is in inches → ``meter="0.0254"``). PyBullet applies this
    unit when it loads a ``.dae`` directly, but trimesh exports raw coordinates
    to a unit-less OBJ, so we must bake the conversion in ourselves to keep the
    extracted submesh in metres. Returns ``1.0`` for non-COLLADA files or when
    no ``<unit>`` is present.
    """
    if not path.lower().endswith(".dae"):
        return 1.0
    try:
        root = ET.parse(path).getroot()
        unit_el = root.find(".//{*}unit")  # COLLADA is namespaced
        if unit_el is not None and unit_el.get("meter"):
            return float(unit_el.get("meter"))
    except Exception:  # noqa: B902
        pass
    return 1.0


def _select_submesh_geometry(scene, name: str):
    """Return a ``Trimesh`` for the submesh *name*, node transform baked in.

    Matches *name* against scene node and geometry names, preferring an exact
    (case-insensitive) match before falling back to suffix / substring. Returns
    ``None`` if nothing matches.
    """
    want = name.strip().lower()
    loose = None  # (transform, geometry) for a non-exact match, used as fallback
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        for cand in (str(node).lower(), str(geom_name).lower()):
            if cand == want:
                mesh = scene.geometry[geom_name].copy()
                mesh.apply_transform(transform)
                return mesh
            if loose is None and (cand.endswith(want) or want in cand):
                loose = (transform, geom_name)
    if loose is not None:
        transform, geom_name = loose
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        return mesh
    # Last resort: direct geometry-key match without node transform.
    for gname, geom in scene.geometry.items():
        if gname.lower() == want or gname.lower().endswith(want) or want in gname.lower():
            return geom.copy()
    return None


def _extract_submesh(mesh_path: str, submesh_name: str, center: bool) -> Optional[str]:
    """Extract a named submesh from *mesh_path* into a cached standalone OBJ.

    Args:
        mesh_path: Absolute path to the combined mesh file (e.g. a ``.dae``).
        submesh_name: Name of the geometry/node to extract.
        center: If ``True``, recenter the extracted mesh on its AABB centre
            (Gazebo ``<center>true</center>``; used e.g. for wheels so they
            pivot about their own centre and the link pose positions them).

    Returns:
        Absolute path to the cached OBJ, or ``None`` if the optional ``sdf``
        extra is unavailable or extraction fails (caller falls back to the
        full mesh).
    """
    global _submesh_lib_warned
    try:
        import trimesh
    except ImportError:
        if not _submesh_lib_warned:
            logger.warning(
                "SDF <submesh> '%s' in %s needs the 'sdf' extra "
                "(pip install 'pybullet-fleet[sdf]'); loading the full mesh "
                "instead — multi-part models (e.g. Caddy) may overlap.",
                submesh_name,
                os.path.basename(mesh_path),
            )
            _submesh_lib_warned = True
        return None

    if not os.path.isfile(mesh_path):
        return None
    try:
        mtime = os.path.getmtime(mesh_path)
    except OSError:
        return None

    unit_m = _collada_unit_meter(mesh_path)
    # ``v2`` salts the cache key so meshes cached before the unit-scaling fix
    # are not reused.
    key = hashlib.md5(f"v2|{mesh_path}|{mtime}|{submesh_name}|{center}|{unit_m}".encode()).hexdigest()[:16]
    safe_name = "".join(c if c.isalnum() else "_" for c in submesh_name)
    out_path = os.path.join(_SUBMESH_CACHE_DIR, f"{safe_name}_{key}.obj")
    if os.path.isfile(out_path):
        return out_path

    try:
        scene = trimesh.load(mesh_path, force="scene")
        mesh = _select_submesh_geometry(scene, submesh_name)
        if mesh is None:
            logger.warning("Submesh '%s' not found in %s", submesh_name, mesh_path)
            return None
        # Convert authored units (e.g. inches) to metres so the unit-less OBJ
        # is metric — PyBullet would otherwise treat raw coords as metres.
        if unit_m != 1.0:
            mesh.apply_scale(unit_m)
        if center:
            mesh.apply_translation(-mesh.bounding_box.centroid)
        os.makedirs(_SUBMESH_CACHE_DIR, exist_ok=True)
        mesh.export(out_path)
        return out_path
    except Exception as exc:  # noqa: B902
        logger.warning("Failed to extract submesh '%s' from %s: %s", submesh_name, mesh_path, exc)
        return None


def _resolve_submesh(mesh_el: ET.Element, abs_path: str) -> Optional[str]:
    """If *mesh_el* has a ``<submesh>``, return the extracted OBJ path, else ``None``."""
    sub = mesh_el.find("submesh")
    if sub is None:
        return None
    name_el = sub.find("name")
    if name_el is None or not name_el.text:
        return None
    center_el = sub.find("center")
    center = center_el is not None and (center_el.text or "").strip().lower() in ("true", "1")
    return _extract_submesh(abs_path, name_el.text.strip(), center)


def _obj_without_materials(obj_path: str) -> str:
    """Return a cached copy of an OBJ with its material library stripped.

    PyBullet applies ``rgbaColor`` only as a *tint over* a mesh's MTL textures,
    so a flat colour can't override them. To truly flat-shade a multi-material
    OBJ (e.g. the campus env, which PyBullet can't texture correctly), we drop
    the ``mtllib``/``usemtl`` lines so no textures load and ``rgbaColor`` shows.
    Returns the original path on any failure.
    """
    if not obj_path.lower().endswith(".obj"):
        return obj_path
    try:
        mtime = os.path.getmtime(obj_path)
    except OSError:
        return obj_path
    key = hashlib.md5(f"nomtl|{obj_path}|{mtime}".encode()).hexdigest()[:16]
    out_path = os.path.join(_SUBMESH_CACHE_DIR, f"{os.path.basename(obj_path)[:-4]}_{key}.obj")
    if os.path.isfile(out_path):
        return out_path
    try:
        os.makedirs(_SUBMESH_CACHE_DIR, exist_ok=True)
        with open(obj_path, "r", encoding="utf-8", errors="replace") as src, open(out_path, "w") as dst:
            for line in src:
                s = line.lstrip()
                if s.startswith("mtllib") or s.startswith("usemtl"):
                    continue
                dst.write(line)
        return out_path
    except Exception as exc:  # noqa: B902
        logger.warning("Failed to strip materials from %s: %s", obj_path, exc)
        return obj_path


def _parse_pose(
    pose_text: Optional[str],
) -> Tuple[float, float, float, float, float, float]:
    """Parse an SDF ``<pose>`` string ``'x y z roll pitch yaw'``.

    Returns:
        ``(x, y, z, roll, pitch, yaw)`` — always a 6-tuple.
        Missing orientation components default to ``0.0``.
    """
    if not pose_text or not pose_text.strip():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parts = pose_text.strip().split()
    x = float(parts[0]) if len(parts) > 0 else 0.0
    y = float(parts[1]) if len(parts) > 1 else 0.0
    z = float(parts[2]) if len(parts) > 2 else 0.0
    roll = float(parts[3]) if len(parts) > 3 else 0.0
    pitch = float(parts[4]) if len(parts) > 4 else 0.0
    yaw = float(parts[5]) if len(parts) > 5 else 0.0
    return (x, y, z, roll, pitch, yaw)


# ---------------------------------------------------------------------------
# Mesh directory loader  (moved from world_loader.py)
# ---------------------------------------------------------------------------


def load_mesh_directory(
    mesh_dir: str,
    sim_core=None,
    collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
    mesh_scale: Optional[List[float]] = None,
    rgba_color: Optional[List[float]] = None,
    pattern: str = "*.obj",
) -> List[SimObject]:
    """Load mesh files from a directory as static SimObjects.

    Scans ``mesh_dir`` for files matching ``pattern`` and creates a static
    SimObject for each one.  Object names are derived from filenames (stem).

    Originally written for rmf_demos environments where
    ``rmf_building_map_tools`` generates OBJ meshes with world-space
    vertices, but works with any directory of mesh files.

    Args:
        mesh_dir: Directory containing mesh files
        sim_core: Simulation core for registration
        collision_mode: Collision mode for environment objects
        mesh_scale: Optional scale override [sx, sy, sz] (default: [1,1,1])
        rgba_color: Optional color override [r,g,b,a] (default: grey)
        pattern: Glob pattern for mesh files (default: ``'*.obj'``)

    Returns:
        List of SimObject instances (one per mesh file)

    Raises:
        FileNotFoundError: If ``mesh_dir`` does not exist.
    """
    mesh_dir = os.path.expanduser(mesh_dir)
    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, pattern)))
    if not mesh_files:
        logger.warning("No %s files found in %s", pattern, mesh_dir)
        return []

    scale = mesh_scale or [1.0, 1.0, 1.0]
    objects: List[SimObject] = []

    def _load_meshes() -> None:
        for mesh_path in mesh_files:
            name = Path(mesh_path).stem
            visual = ShapeParams(
                shape_type="mesh",
                mesh_path=mesh_path,
                mesh_scale=scale,
                rgba_color=rgba_color or list(DEFAULT_SCENERY_COLOR),
            )
            collision = ShapeParams(
                shape_type="mesh",
                mesh_path=mesh_path,
                mesh_scale=scale,
            )
            obj = SimObject.from_mesh(
                visual_shape=visual,
                collision_shape=collision,
                pose=Pose.from_xyz(0, 0, 0),
                mass=0.0,
                sim_core=sim_core,
                collision_mode=collision_mode,
                name=name,
            )
            objects.append(obj)

    if sim_core is not None:
        with sim_core.batch_spawn():
            _load_meshes()
    else:
        _load_meshes()

    logger.info("Loaded %d environment objects from %s", len(objects), mesh_dir)
    return objects


def load_rmf_world(*args, **kwargs) -> List[SimObject]:
    """Backward-compatible alias for :func:`load_mesh_directory`.

    .. deprecated::
        Use :func:`load_mesh_directory` instead.
    """
    return load_mesh_directory(*args, **kwargs)


# ---------------------------------------------------------------------------
# World (static building) loader
# ---------------------------------------------------------------------------


def load_sdf_world(
    sdf_path: str,
    sim_core=None,
    collision_mode: CollisionMode = CollisionMode.STATIC,
    rgba_color: Optional[List[float]] = None,
    mass: float = 0.0,
    pickable: bool = False,
    strip_materials: bool = False,
) -> List[SimObject]:
    """Load building/prop models from an SDF file.

    Parses all ``<model>`` elements, iterates over their ``<link>`` children,
    extracts mesh ``<uri>`` from the *first* ``<visual>`` and the
    link-level ``<pose>``, then creates a :class:`SimObject` for each.

    Supports SDF files containing multiple top-level ``<model>`` elements.

    This correctly handles rmf_building_map_tools output where floor
    meshes have world-baked vertices (pose = 0) and wall meshes have
    local vertices with per-link pose offsets.

    Args:
        sdf_path: Absolute path to ``model.sdf``.
        sim_core: Simulation core for object registration.
        collision_mode: Collision mode for environment objects
            (default: ``STATIC`` — static scenery participates in
            collision detection without moving).
        rgba_color: Optional colour override ``[r, g, b, a]``.
        mass: Mass for the created objects (default: 0.0 for static).
        pickable: Whether objects can be picked up (default: False).

    Returns:
        List of created SimObject instances.

    Example::

        from pybullet_fleet.sdf_loader import load_sdf_world

        objects = load_sdf_world(
            "/rmf_demos_ws/.../building_L1/model.sdf",
            sim_core=sim,
        )
    """
    sdf_path = os.path.expanduser(sdf_path)
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    model_dir = os.path.dirname(sdf_path)
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    model_els = root.findall(".//model")
    if not model_els:
        raise ValueError(f"No <model> element found in {sdf_path}")

    color = rgba_color  # None → use mesh native colours (DAE materials)
    objects: List[SimObject] = []

    def _load() -> None:
        for model_el in model_els:
            model_name = model_el.get("name", "")
            for link in model_el.findall("link"):
                link_name = link.get("name", "unknown")

                # Extract pose
                pose_el = link.find("pose")
                x, y, z, roll, pitch, yaw = _parse_pose(pose_el.text if pose_el is not None else None)

                # Find first visual mesh URI, scale, and per-visual material color
                mesh_uri = None
                visual_scale = [1.0, 1.0, 1.0]
                per_visual_color = None  # SDF <material><diffuse> RGBA if present
                for visual in link.findall("visual"):
                    mesh_el = visual.find(".//mesh/uri")
                    if mesh_el is not None and mesh_el.text:
                        mesh_uri = mesh_el.text
                        scale_el = visual.find(".//mesh/scale")
                        if scale_el is not None and scale_el.text:
                            parts = scale_el.text.strip().split()
                            if len(parts) >= 3:
                                visual_scale = [float(parts[0]), float(parts[1]), float(parts[2])]
                        # Extract per-visual SDF material color (including alpha)
                        mat_color_str = _extract_sdf_material_color(visual)
                        if mat_color_str:
                            mat_parts = mat_color_str.split()
                            if len(mat_parts) >= 4:
                                per_visual_color = [
                                    float(mat_parts[0]),
                                    float(mat_parts[1]),
                                    float(mat_parts[2]),
                                    float(mat_parts[3]),
                                ]
                        break

                if mesh_uri is None:
                    logger.debug("Link '%s' has no visual mesh, skipping", link_name)
                    continue

                mesh_path = _resolve_model_uri(mesh_uri, model_dir, model_name)
                if not os.path.isfile(mesh_path):
                    logger.warning("Mesh file not found: %s (from %s)", mesh_path, mesh_uri)
                    continue
                # Flat-shade request: drop the OBJ's materials so rgba_color
                # (not the bleeding texture) is what PyBullet renders.
                if strip_materials:
                    mesh_path = _obj_without_materials(mesh_path)

                # Find collision mesh and scale (use visual mesh as fallback)
                # Skip collision shape entirely when collision is disabled — many
                # Gazebo Fuel .dae meshes fail createCollisionShape but render
                # fine as visual-only bodies.
                coll_shape_for_link: Optional[ShapeParams] = None
                if collision_mode != CollisionMode.DISABLED:
                    collision_uri = None
                    collision_scale = list(visual_scale)
                    for coll in link.findall("collision"):
                        coll_mesh = coll.find(".//mesh/uri")
                        if coll_mesh is not None and coll_mesh.text:
                            collision_uri = coll_mesh.text
                            coll_scale_el = coll.find(".//mesh/scale")
                            if coll_scale_el is not None and coll_scale_el.text:
                                parts = coll_scale_el.text.strip().split()
                                if len(parts) >= 3:
                                    collision_scale = [float(parts[0]), float(parts[1]), float(parts[2])]
                            break
                    collision_path = _resolve_model_uri(collision_uri, model_dir, model_name) if collision_uri else mesh_path
                    coll_shape_for_link = ShapeParams(
                        shape_type="mesh",
                        mesh_path=collision_path if os.path.isfile(collision_path) else mesh_path,
                        mesh_scale=collision_scale,
                    )

                # Use per-visual SDF material color if available, else fallback
                effective_color = per_visual_color if per_visual_color else color

                visual_shape = ShapeParams(
                    shape_type="mesh",
                    mesh_path=mesh_path,
                    mesh_scale=visual_scale,
                    rgba_color=effective_color,
                )

                obj = SimObject.from_mesh(
                    visual_shape=visual_shape,
                    collision_shape=coll_shape_for_link,
                    pose=Pose.from_euler(x, y, z, roll, pitch, yaw),
                    mass=mass,
                    sim_core=sim_core,
                    collision_mode=collision_mode,
                    name=link_name,
                    pickable=pickable,
                )

                # Apply per-visual alpha post-load if alpha < 1.0.
                # PyBullet changeVisualShape is needed because the RGBA set
                # during shape creation may be overridden when meshes have
                # their own materials.
                if effective_color and len(effective_color) >= 4 and effective_color[3] < 1.0:
                    pid = _physics_client_of(sim_core)
                    try:
                        p.changeVisualShape(
                            obj.body_id,
                            -1,
                            rgbaColor=effective_color,
                            physicsClientId=pid,
                        )
                    except Exception:
                        pass

                objects.append(obj)

    if sim_core is not None:
        with sim_core.batch_spawn():
            _load()
    else:
        _load()

    logger.info("Loaded %d links from SDF %s", len(objects), sdf_path)
    return objects


# ---------------------------------------------------------------------------
# .world file loader  (handles <include> + Gazebo Fuel URIs)
# ---------------------------------------------------------------------------


def _resolve_fuel_uri(uri: str) -> Optional[str]:
    """Resolve a Gazebo Fuel URI to a local model directory.

    Downloads the model with ``gz fuel download`` if not already cached.
    Returns the local directory containing ``model.sdf``, or *None* on failure.
    """
    if not _is_fuel_uri(uri):
        return None

    # URL structure: https://fuel.<host>/1.0/<owner>/models/<model>
    # Local cache:   ~/.gz/fuel/<host>/<owner>/models/<model>/<ver>/
    parts = _fuel_path_parts(uri)
    if len(parts) < 3:  # owner/models/ModelName
        return None
    owner = parts[0].lower()
    model_name = parts[2].lower()

    def cached_model_dir() -> Optional[str]:
        for host in _fuel_cache_hosts(uri):
            cache_dir = os.path.join(_FUEL_CACHE, host, owner, "models", model_name)
            if os.path.isdir(cache_dir):
                versions = sorted(os.listdir(cache_dir))
                if versions:
                    candidate = os.path.join(cache_dir, versions[-1])
                    if os.path.isfile(os.path.join(candidate, "model.sdf")):
                        _ensure_texture_symlinks(candidate)
                        return candidate
        return None

    cached = cached_model_dir()
    if cached:
        return cached

    # Download via gz fuel
    if shutil.which("gz") is None:
        logger.warning("'gz' CLI not found — cannot download Fuel model: %s", uri)
        return None

    try:
        subprocess.run(
            ["gz", "fuel", "download", "-u", uri],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except Exception as e:
        logger.warning("Failed to download Fuel model %s: %s", uri, e)
        return None

    cached = cached_model_dir()
    if cached:
        return cached

    logger.warning("Fuel model not found after download: %s", uri)
    return None


def _ensure_texture_symlinks(model_version_dir: str) -> None:
    """Create symlinks in ``meshes/`` for textures in ``materials/textures/``.

    Gazebo Fuel models store texture images in ``materials/textures/`` but
    DAE files reference them by bare filename.  PyBullet resolves textures
    relative to the DAE file's directory (``meshes/``), so it cannot find them.
    This function creates relative symlinks to bridge the gap.
    """
    tex_dir = os.path.join(model_version_dir, "materials", "textures")
    if not os.path.isdir(tex_dir):
        return

    meshes_dir = os.path.join(model_version_dir, "meshes")
    if not os.path.isdir(meshes_dir):
        return

    for fname in os.listdir(tex_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".tga", ".bmp"):
            continue
        link_path = os.path.join(meshes_dir, fname)
        if os.path.exists(link_path):
            continue
        target = os.path.join("..", "materials", "textures", fname)
        try:
            os.symlink(target, link_path)
        except OSError:
            pass  # read-only filesystem or permission issue — skip silently


def _resolve_include_uri(
    uri: str,
    resource_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve ``<uri>`` from a ``<include>`` element to a local model directory.

    Handles:
    - Gazebo Fuel HTTPS URIs — download/cache lookup
    - ``model://ModelName`` — search ``GZ_SIM_RESOURCE_PATH`` and *resource_paths*
    """
    if _is_fuel_uri(uri):
        return _resolve_fuel_uri(uri)

    if uri.startswith("model://"):
        model_name = uri[len("model://") :]
        paths = list(resource_paths or [])
        env_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
        if env_path:
            paths.extend(env_path.split(":"))
        for base in paths:
            candidate = os.path.join(base, model_name)
            if os.path.isfile(os.path.join(candidate, "model.sdf")):
                return candidate
        # rmf_demos Office maps use model:// for furniture that Gazebo resolves
        # from OpenRobotics Fuel on first use. Match that fallback after local
        # model directories have been checked.
        resolved = _resolve_fuel_uri(f"https://fuel.gazebosim.org/1.0/OpenRobotics/models/{quote(model_name)}")
        if resolved:
            return resolved
        logger.warning("model://%s not found in resource paths", model_name)
        return None

    # Absolute or relative path
    if os.path.isdir(uri) and os.path.isfile(os.path.join(uri, "model.sdf")):
        return uri

    return None


def load_sdf_world_file(
    world_path: str,
    sim_core=None,
    collision_mode: CollisionMode = CollisionMode.STATIC,
    skip_models: Optional[List[str]] = None,
    resource_paths: Optional[List[str]] = None,
    force_color: Optional[List[float]] = None,
) -> List[SimObject]:
    """Load a Gazebo ``.world`` file with ``<include>`` elements.

    Parses the ``<world>`` element, resolves each ``<include>`` to a local
    model directory (via Fuel download or ``model://`` search), then calls
    :func:`load_sdf_world` for the building model and creates SimObjects
    for furniture/prop models.

    Args:
        world_path: Absolute path to the ``.world`` file.
        sim_core: Simulation core for registration.
        collision_mode: Collision mode for loaded objects.
        skip_models: Model names to skip (e.g. robot names like ``TinyRobot``).
        resource_paths: Additional directories to search for ``model://`` URIs.

    Returns:
        All SimObjects loaded from the world file.
    """
    world_path = os.path.expanduser(world_path)
    if not os.path.isfile(world_path):
        raise FileNotFoundError(f"World file not found: {world_path}")

    tree = ET.parse(world_path)
    root = tree.getroot()

    world_el = root.find(".//world")
    if world_el is None:
        raise ValueError(f"No <world> element found in {world_path}")

    # Auto-discover model:// resource paths from the world file's directory.
    # rmf_demos_maps convention: <map_dir>/models/ contains building SDF models.
    effective_paths = list(resource_paths or [])
    world_dir = os.path.dirname(world_path)
    models_dir = os.path.join(world_dir, "models")
    if os.path.isdir(models_dir) and models_dir not in effective_paths:
        effective_paths.insert(0, models_dir)

    skip = {s.lower() for s in (skip_models or [])}
    all_objects: List[SimObject] = []

    # --- Phase 1: prefetch ------------------------------------------------
    # Resolve all include URIs (downloading Fuel models as needed) BEFORE
    # touching PyBullet.  This avoids long network waits while the GUI
    # rendering thread is running, which can crash on Intel GPUs.

    @dataclass
    class _IncludeInfo:
        name: str
        model_type: str
        sdf_path: str
        ix: float
        iy: float
        iz: float
        iroll: float
        ipitch: float
        iyaw: float
        is_fuel: bool = False  # True → Gazebo Fuel model (.dae with embedded colours).
        # Currently only Fuel URIs are detected.  With DAE→OBJ conversion
        # (see roadmap) this could become a generic "has_native_materials" flag.
        is_static: bool = True  # True → static scenery (default); False → movable/pickable object.
        model_mass: float = 0.0  # Total mass from SDF <inertial><mass> (used when physics=True).

    resolved: List[_IncludeInfo] = []

    for include in world_el.findall("include"):
        name_el = include.find("name")
        uri_el = include.find("uri")
        if uri_el is None or not uri_el.text:
            continue

        include_name = name_el.text.strip() if name_el is not None else ""
        uri = uri_el.text.strip()

        # Extract model type name from URI for skip check
        model_type = uri.rstrip("/").split("/")[-1]
        if model_type.lower() in skip or include_name.lower() in skip:
            logger.debug("Skipping model: %s (%s)", include_name, model_type)
            # Record the skipped model's position on sim_core so that
            # downstream components (like WorkcellHandler) can look up
            # workcell positions even though the visual model was not
            # loaded into PyBullet.  This mirrors how Gazebo's
            # TeleportDispenser uses its own model <pose> for spatial
            # lookups.
            if sim_core is not None and include_name:
                pose_el = include.find("pose")
                pose_text = pose_el.text if pose_el is not None else None
                sx, sy, sz, _, _, _ = _parse_pose(pose_text)
                if not hasattr(sim_core, "skipped_model_positions"):
                    sim_core.skipped_model_positions = {}
                sim_core.skipped_model_positions[include_name] = (sx, sy, sz)
                logger.debug(
                    "Recorded skipped model position: %s -> (%.2f, %.2f, %.2f)",
                    include_name,
                    sx,
                    sy,
                    sz,
                )
            continue

        # Resolve URI to local directory (may download from Fuel)
        # NOTE: is_fuel is currently a simple URL prefix check, meaning it only
        # handles Gazebo Fuel models.  If DAE→OBJ auto-conversion is added
        # (see roadmap), the is_fuel flag could be replaced with a mesh-format
        # check (e.g. "does the model contain .dae files?") to handle
        # arbitrary DAE models from any source generically.
        is_fuel = _is_fuel_uri(uri)
        model_dir = _resolve_include_uri(uri, effective_paths)
        if model_dir is None:
            logger.warning("Cannot resolve include URI: %s", uri)
            continue

        sdf_path = os.path.join(model_dir, "model.sdf")

        # Parse include-level pose
        pose_el = include.find("pose")
        pose_text = pose_el.text if pose_el is not None else None
        ix, iy, iz, iroll, ipitch, iyaw = _parse_pose(pose_text)

        # Parse include-level <static> tag (default: True — scenery).
        # SDF convention: <static>false</static> marks movable objects.
        static_el = include.find("static")
        is_static = True
        if static_el is not None and static_el.text:
            is_static = static_el.text.strip().lower() != "false"

        # Extract total mass from model SDF for non-static objects.
        model_mass = 0.0
        if not is_static and os.path.isfile(sdf_path):
            try:
                model_tree = ET.parse(sdf_path)
                model_root = model_tree.getroot()
                for link_el in model_root.findall(".//model/link"):
                    mass_el = link_el.find(".//inertial/mass")
                    if mass_el is not None and mass_el.text:
                        model_mass += float(mass_el.text.strip())
            except Exception:
                pass

        resolved.append(
            _IncludeInfo(
                name=include_name,
                model_type=model_type,
                sdf_path=sdf_path,
                ix=ix,
                iy=iy,
                iz=iz,
                iroll=iroll,
                ipitch=ipitch,
                iyaw=iyaw,
                is_fuel=is_fuel,
                is_static=is_static,
                model_mass=model_mass,
            )
        )

    logger.info("Prefetched %d models, loading into PyBullet...", len(resolved))

    # --- Phase 2: load into PyBullet --------------------------------------
    for info in resolved:
        # Fuel models (.dae) have embedded materials → rgba_color=None lets
        # PyBullet read native colours.  Local building models (.obj from
        # rmf_building_map_tools) lack material info → use default grey.
        # force_color overrides everything: used to flat-shade a multi-material
        # mesh that PyBullet can't texture correctly (e.g. the campus env, a
        # single 50-material OBJ — PyBullet would tile one wrong texture).
        if force_color is not None:
            model_color = list(force_color)
        else:
            model_color = None if info.is_fuel else list(DEFAULT_SCENERY_COLOR)

        # Ensure texture symlinks exist for Fuel models so PyBullet can
        # resolve DAE <init_from> texture references (textures live in
        # materials/textures/ but DAE refs use bare filenames).
        if info.is_fuel:
            _ensure_texture_symlinks(os.path.dirname(info.sdf_path))

        # Determine collision_mode, mass and pickable based on <static> tag.
        # Non-static objects are movable/pickable with NORMAL_3D collision.
        # Mass is taken from SDF <inertial> when physics is enabled,
        # otherwise 0.0 (kinematic mode teleports objects).
        if info.is_static:
            obj_collision_mode = collision_mode
            obj_mass = 0.0
            obj_pickable = False
        else:
            obj_collision_mode = CollisionMode.NORMAL_3D
            obj_pickable = True
            physics_enabled = sim_core is not None and hasattr(sim_core, "params") and sim_core.params.physics
            if physics_enabled:
                # Use SDF mass; fall back to 0.1 if <inertial> was absent.
                obj_mass = info.model_mass if info.model_mass > 0 else 0.1
            else:
                obj_mass = 0.0

        try:
            model_objs = load_sdf_world(
                info.sdf_path,
                sim_core=sim_core,
                collision_mode=obj_collision_mode,
                rgba_color=model_color,
                mass=obj_mass,
                pickable=obj_pickable,
                strip_materials=force_color is not None,
            )
        except Exception as e:
            logger.warning("Failed to load model SDF %s: %s", info.sdf_path, e)
            continue

        # Offset each object by the include-level pose.
        # Use resetBasePositionAndOrientation directly instead of set_pose()
        # because set_pose() rejects STATIC objects (by design — static
        # objects should not move during simulation).  This is an initial
        # placement, not a runtime move.
        cos_y, sin_y = math.cos(info.iyaw), math.sin(info.iyaw)
        pid = _physics_client_of(sim_core)
        for obj in model_objs:
            obj_pose = obj.get_pose()
            ox, oy, oz = obj_pose.position
            rx = ox * cos_y - oy * sin_y + info.ix
            ry = ox * sin_y + oy * cos_y + info.iy
            rz = oz + info.iz
            new_pose = Pose.from_euler(rx, ry, rz, info.iroll, info.ipitch, info.iyaw)
            pos, orn = new_pose.as_position_orientation()
            p.resetBasePositionAndOrientation(obj.body_id, pos, orn, physicsClientId=pid)
            obj._cached_pose = new_pose
            # Update AABB/spatial-grid so collision system sees
            # the relocated position (not the pre-offset origin).
            if sim_core is not None:
                sim_core._update_object_aabb(obj.object_id, update_grid=True)
            base_name = info.name or info.model_type
            obj.name = f"{base_name}/{obj.name}"

        all_objects.extend(model_objs)

    logger.info("Loaded %d objects from world %s", len(all_objects), world_path)
    return all_objects


# ---------------------------------------------------------------------------
# Robot SDF → temporary URDF converter
# ---------------------------------------------------------------------------


def resolve_sdf_to_urdf(sdf_path: str, model_yaw_offset: float = 0.0) -> str:
    """Convert an SDF robot model to a temporary URDF file.

    PyBullet cannot reliably load Gazebo SDF files (``model://`` URIs,
    missing joint dynamics defaults). This function:

    1. Parses the SDF XML.
    2. Resolves ``model://`` mesh URIs to absolute filesystem paths.
    3. Adds missing ``<dynamics>`` defaults that PyBullet requires.
    4. Writes a minimal URDF to a temporary file.

    The caller is responsible for deleting the temp file when done
    (or it will be cleaned up on process exit by the OS).

    The base link's ``<pose>`` yaw is automatically detected and applied
    to visual/collision origins, so ``model_yaw_offset`` is normally not
    needed.

    Args:
        sdf_path: Absolute path to the robot ``model.sdf``.
        model_yaw_offset: Additional rotation offset (radians) applied on
            top of the auto-detected base-link ``<pose>`` yaw.  Defaults
            to ``0.0``.  Only needed if the auto-detected value is wrong.

    Returns:
        Absolute path to the generated temporary URDF file.

    Example::

        from pybullet_fleet.sdf_loader import resolve_sdf_to_urdf

        urdf_path = resolve_sdf_to_urdf("/path/to/TinyRobot/model.sdf")
        agent = Agent.from_urdf(urdf_path, ...)
    """
    sdf_path = os.path.expanduser(sdf_path)
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    model_dir = os.path.dirname(sdf_path)
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    model_el = root.find(".//model")
    if model_el is None:
        raise ValueError(f"No <model> element found in {sdf_path}")
    model_name = model_el.get("name", "unknown")

    # Collect links and joints
    links: Dict[str, ET.Element] = {}
    joints: List[ET.Element] = []
    for link in model_el.findall("link"):
        links[link.get("name", "")] = link
    for joint in model_el.findall("joint"):
        joints.append(joint)

    # Identify the base link (the link that is NOT a child of any joint).
    # Only the base link's visuals/collisions get the model_yaw_offset applied.
    child_link_names: set = set()
    for joint in model_el.findall("joint"):
        child_el = joint.find("child")
        if child_el is not None and child_el.text:
            child_link_names.add(child_el.text.strip())

    # Build URDF string
    lines: List[str] = [
        '<?xml version="1.0"?>',
        f'<robot name="{model_name}">',
    ]

    for link_name, link_el in links.items():
        lines.append(f'  <link name="{link_name}">')

        # Only apply yaw offset to the base link (not child links like wheels).
        # Auto-detect the base link's own <pose> yaw from SDF and combine
        # with the explicit model_yaw_offset (default 0.0).
        is_base_link = link_name not in child_link_names
        effective_yaw_offset = model_yaw_offset
        if is_base_link:
            base_pose_el = link_el.find("pose")
            if base_pose_el is not None and base_pose_el.text:
                base_pose_parts = base_pose_el.text.strip().split()
                if len(base_pose_parts) >= 6:
                    effective_yaw_offset += float(base_pose_parts[5])
        apply_yaw_offset = abs(effective_yaw_offset) > 1e-9 and is_base_link

        # Inertial
        inertial = link_el.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            mass_val = mass_el.text.strip() if mass_el is not None else "1.0"
            pose_el = inertial.find("pose")
            ipose = pose_el.text.strip().split() if pose_el is not None else ["0"] * 6
            origin_xyz = " ".join(ipose[:3])
            origin_rpy = " ".join(ipose[3:6]) if len(ipose) >= 6 else "0 0 0"

            inertia_el = inertial.find("inertia")
            ixx = iyy = izz = "0.01"
            ixy = ixz = iyz = "0"
            if inertia_el is not None:
                for tag in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                    el = inertia_el.find(tag)
                    if el is not None:
                        locals()[tag] = el.text.strip()

            lines.append("    <inertial>")
            lines.append(f'      <mass value="{mass_val}"/>')
            lines.append(f'      <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>')
            lines.append(f'      <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>')
            lines.append("    </inertial>")

        # Visuals
        for visual in link_el.findall("visual"):
            vis_name = visual.get("name", "visual")
            pose_el = visual.find("pose")
            vpose = pose_el.text.strip().split() if pose_el is not None else ["0"] * 6
            xyz_parts = list(vpose[:3])
            rpy_parts = vpose[3:6] if len(vpose) >= 6 else ["0", "0", "0"]
            if apply_yaw_offset:
                # Rotate the XYZ position around Z axis
                x, y = float(xyz_parts[0]), float(xyz_parts[1])
                cos_a = math.cos(effective_yaw_offset)
                sin_a = math.sin(effective_yaw_offset)
                xyz_parts[0] = str(x * cos_a - y * sin_a)
                xyz_parts[1] = str(x * sin_a + y * cos_a)
                # Add yaw offset to RPY
                rpy_parts = list(rpy_parts)
                rpy_parts[2] = str(float(rpy_parts[2]) + effective_yaw_offset)
            origin_xyz = " ".join(xyz_parts)
            origin_rpy = " ".join(rpy_parts)

            # Extract color from SDF <material>
            mat_color = _extract_sdf_material_color(visual)

            mesh_el = visual.find(".//mesh")
            if mesh_el is not None:
                uri_el = mesh_el.find("uri")
                scale_el = mesh_el.find("scale")
                if uri_el is not None and uri_el.text:
                    abs_path = _resolve_model_uri(uri_el.text, model_dir, model_name)
                    scale = scale_el.text.strip() if scale_el is not None else "1 1 1"
                    # Honour <submesh>: extract just the named part so multi-part
                    # DAEs (e.g. Caddy's polaris.dae) don't load the whole model
                    # per link. Falls back to the full mesh if the 'sdf' extra
                    # is missing.
                    submesh_path = _resolve_submesh(mesh_el, abs_path)
                    use_path = submesh_path or abs_path
                    # If no SDF <material>, try extracting color from DAE mesh.
                    # Skip for submeshes — the extracted OBJ carries its own
                    # material, and the whole-DAE diffuse would mis-color parts.
                    if (
                        submesh_path is None
                        and not mat_color
                        and abs_path.lower().endswith(".dae")
                        and os.path.isfile(abs_path)
                    ):
                        mat_color = _extract_dae_diffuse_color(abs_path)
                    lines.append(f'    <visual name="{vis_name}">')
                    lines.append(f'      <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>')
                    lines.append(f'      <geometry><mesh filename="{use_path}" scale="{scale}"/></geometry>')
                    if mat_color and submesh_path is None:
                        lines.append(f'      <material name="{vis_name}_mat"><color rgba="{mat_color}"/></material>')
                    lines.append("    </visual>")
            else:
                # Check for primitive shapes
                geom = _sdf_geometry_to_urdf(visual.find("geometry"))
                if geom:
                    lines.append(f'    <visual name="{vis_name}">')
                    lines.append(f'      <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>')
                    lines.append(f"      {geom}")
                    if mat_color:
                        lines.append(f'      <material name="{vis_name}_mat"><color rgba="{mat_color}"/></material>')
                    lines.append("    </visual>")

        # Collisions
        for coll in link_el.findall("collision"):
            pose_el = coll.find("pose")
            cpose = pose_el.text.strip().split() if pose_el is not None else ["0"] * 6
            xyz_parts = list(cpose[:3])
            rpy_parts = cpose[3:6] if len(cpose) >= 6 else ["0", "0", "0"]
            if apply_yaw_offset:
                x, y = float(xyz_parts[0]), float(xyz_parts[1])
                cos_a = math.cos(effective_yaw_offset)
                sin_a = math.sin(effective_yaw_offset)
                xyz_parts[0] = str(x * cos_a - y * sin_a)
                xyz_parts[1] = str(x * sin_a + y * cos_a)
                rpy_parts = list(rpy_parts)
                rpy_parts[2] = str(float(rpy_parts[2]) + effective_yaw_offset)
            origin_xyz = " ".join(xyz_parts)
            origin_rpy = " ".join(rpy_parts)

            geom = _sdf_geometry_to_urdf(coll.find("geometry"), model_dir, model_name)
            if geom:
                lines.append("    <collision>")
                lines.append(f'      <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>')
                lines.append(f"      {geom}")
                lines.append("    </collision>")

        lines.append("  </link>")

    # Joints
    for joint_el in joints:
        jname = joint_el.get("name", "joint")
        jtype = joint_el.get("type", "fixed")
        limit_el = joint_el.find(".//limit")

        # Map SDF joint type → URDF joint type
        urdf_type = jtype
        if jtype == "revolute":
            # Revolute with explicit finite limits → URDF "revolute"
            # Revolute without limits or with ±inf → URDF "continuous"
            if limit_el is not None:
                lower = limit_el.find("lower")
                upper = limit_el.find("upper")
                if lower is not None and upper is not None:
                    lo = float(lower.text.strip())
                    hi = float(upper.text.strip())
                    if abs(lo) > 1e10 or abs(hi) > 1e10:
                        urdf_type = "continuous"
                else:
                    urdf_type = "continuous"
            else:
                # No <limit> in SDF → unlimited rotation
                urdf_type = "continuous"

        child = joint_el.find("child")
        parent = joint_el.find("parent")
        if child is None or parent is None:
            continue
        child_name = child.text.strip()
        parent_name = parent.text.strip()

        # --- Joint origin: child link pose RELATIVE to parent link ---
        # In SDF, link poses are specified in the model frame.
        # In URDF, joint origins are relative to the parent link frame.
        # For base-link parents (model frame origin), child model-frame pose
        # is already the correct joint origin.
        # For chained joints (non-base parent), we subtract the parent's
        # model-frame pose from the child's to get the relative offset.
        child_link = links.get(child_name)
        parent_link = links.get(parent_name)

        child_pose_el = child_link.find("pose") if child_link is not None else None
        cpose = child_pose_el.text.strip().split() if child_pose_el is not None else ["0"] * 6

        parent_pose_el = parent_link.find("pose") if parent_link is not None else None
        ppose = parent_pose_el.text.strip().split() if parent_pose_el is not None else ["0"] * 6

        # Check if parent is a non-base link (child of another joint)
        parent_is_base = parent_name not in child_link_names

        if parent_is_base:
            # Base link frame in URDF = model frame (no transformation needed)
            origin_xyz = " ".join(cpose[:3])
            origin_rpy = " ".join(cpose[3:6]) if len(cpose) >= 6 else "0 0 0"
        else:
            # Compute relative pose: child_model - parent_model
            cx, cy, cz = float(cpose[0]), float(cpose[1]), float(cpose[2])
            px, py, pz = float(ppose[0]), float(ppose[1]), float(ppose[2])
            dx, dy, dz = cx - px, cy - py, cz - pz

            # If parent has rotation, transform delta into parent frame
            if len(ppose) >= 6:
                p_yaw = float(ppose[5])
                if abs(p_yaw) > 1e-9:
                    cos_a = math.cos(-p_yaw)
                    sin_a = math.sin(-p_yaw)
                    dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a

            # RPY relative
            c_rpy = [float(cpose[i]) if len(cpose) > i else 0.0 for i in (3, 4, 5)]
            p_rpy = [float(ppose[i]) if len(ppose) > i else 0.0 for i in (3, 4, 5)]
            dr, dp, dyaw = c_rpy[0] - p_rpy[0], c_rpy[1] - p_rpy[1], c_rpy[2] - p_rpy[2]

            origin_xyz = f"{dx} {dy} {dz}"
            origin_rpy = f"{dr} {dp} {dyaw}"

        axis_el = joint_el.find(".//axis/xyz")
        axis = axis_el.text.strip() if axis_el is not None else "0 0 1"

        lines.append(f'  <joint name="{jname}" type="{urdf_type}">')
        lines.append(f'    <parent link="{parent_name}"/>')
        lines.append(f'    <child link="{child_name}"/>')
        lines.append(f'    <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>')
        lines.append(f'    <axis xyz="{axis}"/>')

        # Add dynamics (PyBullet requires damping/friction for revolute/continuous)
        lines.append('    <dynamics damping="0.0" friction="0.0"/>')

        # URDF <limit> is REQUIRED for revolute and prismatic joints.
        # Without it PyBullet rejects the URDF ("joint xml is not initialized correctly").
        if urdf_type == "revolute":
            lo_val, hi_val = "-3.14159", "3.14159"
            if limit_el is not None:
                lower = limit_el.find("lower")
                upper = limit_el.find("upper")
                if lower is not None:
                    lo_val = lower.text.strip()
                if upper is not None:
                    hi_val = upper.text.strip()
            lines.append(f'    <limit lower="{lo_val}" upper="{hi_val}" effort="100" velocity="10"/>')
        elif urdf_type == "prismatic":
            lo_val, hi_val = "-1.0", "1.0"
            if limit_el is not None:
                lower = limit_el.find("lower")
                upper = limit_el.find("upper")
                if lower is not None:
                    lo_val = lower.text.strip()
                if upper is not None:
                    hi_val = upper.text.strip()
            lines.append(f'    <limit lower="{lo_val}" upper="{hi_val}" effort="100" velocity="10"/>')

        lines.append("  </joint>")

    lines.append("</robot>")

    # Write to temp file
    urdf_content = "\n".join(lines)
    fd, tmp_path = tempfile.mkstemp(suffix=".urdf", prefix=f"sdf_{model_name}_")
    with os.fdopen(fd, "w") as f:
        f.write(urdf_content)

    logger.info("Converted SDF %s → temp URDF %s (%d links, %d joints)", sdf_path, tmp_path, len(links), len(joints))
    return tmp_path


def _extract_sdf_material_color(visual_el: ET.Element) -> Optional[str]:
    """Extract RGBA color string from SDF ``<material>`` inside a ``<visual>``.

    Looks for ``<diffuse>``, ``<ambient>``, or ``<emissive>`` (in that order).
    Also reads the SDF ``<transparency>`` tag (child of ``<visual>``, **not**
    ``<material>``) which ``building_map_tools`` generates per-wall.

    ``<transparency>`` is a float 0.0 (opaque) → 1.0 (invisible).  When
    present it overrides the alpha channel:  ``alpha = 1.0 - transparency``.

    Returns ``"r g b a"`` string suitable for URDF ``<color rgba="..."/>``
    or *None* if no color information is found.
    """
    # --- SDF <transparency> on the visual element --------------------------
    transparency_el = visual_el.find("transparency")
    transparency: Optional[float] = None
    if transparency_el is not None and transparency_el.text:
        try:
            transparency = float(transparency_el.text.strip())
        except ValueError:
            pass

    mat = visual_el.find("material")

    # Extract base RGBA from <material> (diffuse/ambient/emissive)
    base_color: Optional[str] = None
    if mat is not None:
        for tag in ("diffuse", "ambient", "emissive"):
            el = mat.find(tag)
            if el is not None and el.text:
                parts = el.text.strip().split()
                if len(parts) >= 3:
                    r, g, b = parts[0], parts[1], parts[2]
                    a = parts[3] if len(parts) >= 4 else "1"
                    base_color = f"{r} {g} {b} {a}"
                    break

    # Apply <transparency> override
    if transparency is not None and transparency > 0.0:
        alpha = 1.0 - transparency
        if base_color:
            parts = base_color.split()
            return f"{parts[0]} {parts[1]} {parts[2]} {alpha}"
        else:
            # No material — use white with the transparency-derived alpha
            return f"1 1 1 {alpha}"

    # No transparency — return whatever we found (may be None)
    return base_color


def _extract_dae_diffuse_color(dae_path: str) -> Optional[str]:
    """Extract the first diffuse RGBA color from a COLLADA (``.dae``) file.

    PyBullet sometimes ignores colours embedded in DAE meshes when loading
    via URDF.  This helper reads the ``<diffuse><color>`` from the first
    ``<phong>``/``<lambert>`` effect and returns an ``"r g b a"`` string
    that can be injected into a URDF ``<material><color rgba="..."/>``.

    Returns *None* if the file cannot be parsed or contains no colour data.
    """
    try:
        tree = ET.parse(dae_path)
    except Exception:
        return None

    root = tree.getroot()
    # COLLADA uses a namespace — strip it for simpler searches
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Search for <diffuse><color> inside <phong> or <lambert>
    for technique in root.iter(f"{ns}technique"):
        for shader in ("phong", "lambert", "blinn"):
            shader_el = technique.find(f"{ns}{shader}")
            if shader_el is None:
                continue
            diffuse_el = shader_el.find(f"{ns}diffuse")
            if diffuse_el is None:
                continue
            color_el = diffuse_el.find(f"{ns}color")
            if color_el is not None and color_el.text:
                parts = color_el.text.strip().split()
                if len(parts) >= 3:
                    r, g, b = parts[0], parts[1], parts[2]
                    a = parts[3] if len(parts) >= 4 else "1"
                    return f"{r} {g} {b} {a}"
    return None


def _sdf_geometry_to_urdf(
    geom_el: Optional[ET.Element],
    model_dir: str = "",
    model_name: str = "",
) -> str:
    """Convert an SDF ``<geometry>`` element to a URDF geometry string."""
    if geom_el is None:
        return ""

    box = geom_el.find("box/size")
    if box is not None:
        return f'<geometry><box size="{box.text.strip()}"/></geometry>'

    sphere = geom_el.find("sphere/radius")
    if sphere is not None:
        return f'<geometry><sphere radius="{sphere.text.strip()}"/></geometry>'

    cylinder = geom_el.find("cylinder")
    if cylinder is not None:
        r = cylinder.find("radius")
        length = cylinder.find("length")
        if r is not None and length is not None:
            return f'<geometry><cylinder radius="{r.text.strip()}" length="{length.text.strip()}"/></geometry>'

    mesh = geom_el.find("mesh")
    if mesh is not None:
        uri_el = mesh.find("uri")
        if uri_el is not None and uri_el.text:
            abs_path = _resolve_model_uri(uri_el.text, model_dir, model_name)
            scale_el = mesh.find("scale")
            scale = scale_el.text.strip() if scale_el is not None else "1 1 1"
            # Honour <submesh> (see _resolve_submesh) so multi-part meshes
            # don't load the whole model for every link.
            use_path = _resolve_submesh(mesh, abs_path) or abs_path
            return f'<geometry><mesh filename="{use_path}" scale="{scale}"/></geometry>'

    return ""
