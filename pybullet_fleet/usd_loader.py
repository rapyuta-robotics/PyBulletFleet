"""Optional OpenUSD static-world importer.

The importer deliberately handles only composed, local OpenUSD mesh scenes.
It converts supported meshes into temporary OBJ files and creates static
``SimObject`` instances through the normal PyBulletFleet mesh path.  Isaac and
Omniverse runtime schemas are not executed.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from struct import pack
from tempfile import gettempdir
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionMode

_TEXTURE_CACHE: dict[tuple[int, str, float], int] = {}


@dataclass(frozen=True)
class UsdImportDiagnostic:
    """A JSON-serializable diagnostic emitted while importing a USD stage."""

    severity: str
    code: str
    message: str
    prim_path: str | None = None
    schema_type: str | None = None
    asset_path: str | None = None


@dataclass(frozen=True)
class UsdImportOptions:
    """Controls for :func:`load_usd_world`.

    PointInstancer expansion is bounded before any PyBullet bodies are created,
    preventing a composed warehouse stage from allocating an unbounded number
    of static meshes.
    """

    collision: bool = True
    collision_required: bool = False
    include_invisible: bool = False
    include_point_instances: bool = True
    point_instance_warning: int = 1_000
    max_point_instances: int = 5_000
    max_collision_triangles_per_mesh: int = 50_000
    max_collision_triangles_total: int = 500_000
    texture_brightness: float = 1.0


@dataclass
class UsdImportReport:
    """Result and source metadata for one USD world import."""

    source_stage: str
    source_up_axis: str
    source_meters_per_unit: float
    target_convention: str = "Z-up metres"
    stage_to_pbf_transform: str = "identity"
    created_object_ids: dict[str, int] = field(default_factory=dict)
    normalized_object_poses: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    normalized_world_transforms: dict[str, list[list[float]]] = field(default_factory=dict)
    object_transform_modes: dict[str, str] = field(default_factory=dict)
    diagnostics: list[UsdImportDiagnostic] = field(default_factory=list)

    @property
    def objects_created(self) -> int:
        """Number of static objects created by the import."""
        return len(self.created_object_ids)


@dataclass(frozen=True)
class _CollisionBudget:
    """Preflight result used to keep collision-required imports atomic."""

    enabled_paths: frozenset[str]


def load_usd_world(
    usd_path: str | Path,
    *,
    sim_core: Any,
    options: UsdImportOptions | None = None,
) -> UsdImportReport:
    """Load supported OpenUSD meshes as static objects in ``sim_core``.

    Supported rigid or uniformly-scaled transforms become the corresponding
    ``SimObject.pose`` and mesh scale. Other affine transforms are baked into
    emitted OBJ vertices. Both paths normalize visual and collision geometry
    through the same single stage-to-PyBullet transform.

    Args:
        usd_path: Local ``.usd``, ``.usda``, ``.usdc``, or ``.usdz`` stage.
        sim_core: Connected :class:`MultiRobotSimulationCore` to receive the
            imported static objects. It may be initialized later by
            ``run_simulation()``. A custom-step loop should initialize after
            loading and before its first step.
        options: Import controls.  Defaults to :class:`UsdImportOptions`.

    Returns:
        Import report including object IDs keyed by stable USD prim path.

    Raises:
        ImportError: If the optional ``usd`` dependency is not installed.
        FileNotFoundError: If ``usd_path`` does not exist.
        ValueError: If stage metadata is unsupported or cannot be opened.
    """
    Usd, UsdGeom, UsdShade = _require_openusd()
    resolved_path = Path(usd_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"USD stage not found: {resolved_path}")

    opts = options or UsdImportOptions()
    if opts.point_instance_warning < 0 or opts.max_point_instances < 1:
        raise ValueError("PointInstancer limits must be non-negative / positive respectively")
    if opts.point_instance_warning > opts.max_point_instances:
        raise ValueError("point_instance_warning must not exceed max_point_instances")
    if opts.max_collision_triangles_per_mesh < 1 or opts.max_collision_triangles_total < 1:
        raise ValueError("Collision triangle limits must be positive")
    if opts.texture_brightness <= 0.0:
        raise ValueError("texture_brightness must be positive")

    stage = Usd.Stage.Open(str(resolved_path))
    if stage is None:
        raise ValueError(f"Unable to open USD stage: {resolved_path}")

    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if up_axis not in {"Y", "Z"}:
        raise ValueError(f"Unsupported USD upAxis {up_axis!r}; expected 'Y' or 'Z'")
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if metres_per_unit <= 0.0:
        raise ValueError(f"USD metersPerUnit must be positive, got {metres_per_unit}")

    report = UsdImportReport(
        source_stage=str(resolved_path),
        source_up_axis=up_axis,
        source_meters_per_unit=metres_per_unit,
        stage_to_pbf_transform=_normalization_description(up_axis, metres_per_unit),
    )
    xform_cache = UsdGeom.XformCache()
    point_instancers = [UsdGeom.PointInstancer(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.PointInstancer)]
    _validate_point_instance_budget(point_instancers, report, opts)
    prototype_paths = {str(target) for instancer in point_instancers for target in instancer.GetPrototypesRel().GetTargets()}
    collision_budget = _plan_collision_budget(
        stage=stage,
        usd_geom=UsdGeom,
        point_instancers=point_instancers,
        prototype_paths=prototype_paths,
        options=opts,
        report=report,
    )
    spawn_context = sim_core.batch_spawn() if sim_core is not None else nullcontext()

    with spawn_context:
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.PointInstancer):
                _import_point_instancer(
                    instancer=UsdGeom.PointInstancer(prim),
                    stage=stage,
                    usd=Usd,
                    usd_geom=UsdGeom,
                    usd_shade=UsdShade,
                    metres_per_unit=metres_per_unit,
                    up_axis=up_axis,
                    sim_core=sim_core,
                    options=opts,
                    report=report,
                    collision_budget=collision_budget,
                )
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if str(prim.GetPath()) in prototype_paths:
                continue
            if not _is_renderable(UsdGeom, prim, opts.include_invisible):
                continue
            _import_mesh(
                mesh=UsdGeom.Mesh(prim),
                xform_cache=xform_cache,
                metres_per_unit=metres_per_unit,
                up_axis=up_axis,
                sim_core=sim_core,
                options=opts,
                report=report,
                collision_budget=collision_budget,
                usd_shade=UsdShade,
            )

    return report


def _require_openusd() -> tuple[Any, Any, Any]:
    try:
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError as exc:
        raise ImportError(
            "OpenUSD support requires usd-core. From this source checkout run "
            "`python -m pip install -e '.[usd]'`; for a released version that "
            "does not yet provide the extra, run `python -m pip install usd-core`."
        ) from exc
    return Usd, UsdGeom, UsdShade


def _is_invisible(usd_geom: Any, prim: Any) -> bool:
    return usd_geom.Imageable(prim).ComputeVisibility() == usd_geom.Tokens.invisible


def _is_renderable(usd_geom: Any, prim: Any, include_invisible: bool) -> bool:
    """Exclude non-rendering guide/proxy geometry from the visual import."""
    imageable = usd_geom.Imageable(prim)
    if not include_invisible and imageable.ComputeVisibility() == usd_geom.Tokens.invisible:
        return False
    return imageable.ComputePurpose() in {usd_geom.Tokens.default_, usd_geom.Tokens.render}


def _validate_point_instance_budget(instancers: list[Any], report: UsdImportReport, options: UsdImportOptions) -> None:
    if not options.include_point_instances:
        return
    instance_count = sum(len(instancer.GetProtoIndicesAttr().Get() or []) for instancer in instancers)
    if instance_count > options.max_point_instances:
        raise ValueError(
            "point_instance_limit_exceeded: "
            f"stage expands to {instance_count} instances (limit {options.max_point_instances})"
        )
    if instance_count >= options.point_instance_warning:
        report.diagnostics.append(
            UsdImportDiagnostic(
                severity="warning",
                code="point_instance_warning",
                message=f"Stage expands to {instance_count} PointInstancer instances.",
            )
        )


def _plan_collision_budget(
    *,
    stage: Any,
    usd_geom: Any,
    point_instancers: list[Any],
    prototype_paths: set[str],
    options: UsdImportOptions,
    report: UsdImportReport,
) -> _CollisionBudget:
    """Select visual-mesh collision candidates before any bodies are created."""
    if not options.collision:
        return _CollisionBudget(frozenset())

    candidates: list[tuple[str, int]] = []
    for prim in stage.Traverse():
        if not prim.IsA(usd_geom.Mesh) or str(prim.GetPath()) in prototype_paths:
            continue
        if not _is_renderable(usd_geom, prim, options.include_invisible):
            continue
        candidates.append((str(prim.GetPath()), _mesh_triangle_count(usd_geom.Mesh(prim))))

    if options.include_point_instances:
        for instancer in point_instancers:
            paths = instancer.GetPrototypesRel().GetTargets()
            for index, prototype_index in enumerate(instancer.GetProtoIndicesAttr().Get() or []):
                if prototype_index < 0 or prototype_index >= len(paths):
                    continue
                prototype = stage.GetPrimAtPath(paths[prototype_index])
                if prototype.IsA(usd_geom.Mesh) and _is_renderable(usd_geom, prototype, options.include_invisible):
                    candidates.append(
                        (f"{instancer.GetPrim().GetPath()}[{index}]", _mesh_triangle_count(usd_geom.Mesh(prototype)))
                    )

    total = 0
    enabled: set[str] = set()
    unavailable: list[str] = []
    for prim_path, triangles in candidates:
        if triangles > options.max_collision_triangles_per_mesh or total + triangles > options.max_collision_triangles_total:
            unavailable.append(prim_path)
            continue
        enabled.add(prim_path)
        total += triangles

    for prim_path in unavailable:
        report.diagnostics.append(
            UsdImportDiagnostic(
                severity="warning",
                code="collision_mesh_unavailable",
                message=("Visual mesh exceeds the configured collision triangle budget; " "it will be imported visual-only."),
                prim_path=prim_path,
                schema_type="Mesh",
            )
        )
    if options.collision_required and unavailable:
        raise ValueError(
            "collision_mesh_unavailable: collision_required=True but "
            f"{len(unavailable)} mesh(es) exceed the configured collision triangle budget"
        )
    return _CollisionBudget(frozenset(enabled))


def _import_point_instancer(
    *,
    instancer: Any,
    stage: Any,
    usd: Any,
    usd_geom: Any,
    usd_shade: Any,
    metres_per_unit: float,
    up_axis: str,
    sim_core: Any,
    options: UsdImportOptions,
    report: UsdImportReport,
    collision_budget: _CollisionBudget,
) -> None:
    if not options.include_point_instances:
        return
    proto_indices = [int(value) for value in (instancer.GetProtoIndicesAttr().Get() or [])]
    transforms = instancer.ComputeInstanceTransformsAtTime(usd.TimeCode.Default(), usd.TimeCode.Default())
    prototype_paths = instancer.GetPrototypesRel().GetTargets()
    instancer_path = str(instancer.GetPrim().GetPath())
    if len(proto_indices) != len(transforms):
        report.diagnostics.append(
            UsdImportDiagnostic(
                severity="warning",
                code="point_instance_transform_mismatch",
                message="PointInstancer prototype indices and computed transforms differ in length.",
                prim_path=instancer_path,
                schema_type=instancer.GetPrim().GetTypeName(),
            )
        )

    for index, (prototype_index, transform) in enumerate(zip(proto_indices, transforms)):
        instance_path = f"{instancer_path}[{index}]"
        if prototype_index < 0 or prototype_index >= len(prototype_paths):
            report.diagnostics.append(
                UsdImportDiagnostic(
                    severity="warning",
                    code="point_instance_invalid_prototype",
                    message=f"Prototype index {prototype_index} is outside the prototype relation.",
                    prim_path=instance_path,
                    schema_type=instancer.GetPrim().GetTypeName(),
                )
            )
            continue
        prototype = stage.GetPrimAtPath(prototype_paths[prototype_index])
        if not prototype.IsA(usd_geom.Mesh):
            report.diagnostics.append(
                UsdImportDiagnostic(
                    severity="warning",
                    code="point_instance_unsupported_prototype",
                    message="Only Mesh PointInstancer prototypes are supported.",
                    prim_path=instance_path,
                    schema_type=prototype.GetTypeName(),
                )
            )
            continue
        _import_mesh(
            mesh=usd_geom.Mesh(prototype),
            world_transform=transform,
            source_prim_path=instance_path,
            metres_per_unit=metres_per_unit,
            up_axis=up_axis,
            sim_core=sim_core,
            options=options,
            report=report,
            collision_budget=collision_budget,
            usd_shade=usd_shade,
        )


def _import_mesh(
    *,
    mesh: Any,
    xform_cache: Any | None = None,
    world_transform: Any | None = None,
    source_prim_path: str | None = None,
    metres_per_unit: float,
    up_axis: str,
    sim_core: Any,
    options: UsdImportOptions,
    report: UsdImportReport,
    collision_budget: _CollisionBudget,
    usd_shade: Any,
) -> None:
    prim = mesh.GetPrim()
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    prim_path = source_prim_path or str(prim.GetPath())
    if not points or not counts or not indices:
        report.diagnostics.append(
            UsdImportDiagnostic(
                severity="warning",
                code="mesh_without_geometry",
                message="Mesh has no points or faces and was skipped.",
                prim_path=prim_path,
                schema_type=prim.GetTypeName(),
            )
        )
        return

    world = world_transform if world_transform is not None else xform_cache.GetLocalToWorldTransform(prim)
    normalized_transform = _normalized_affine_transform(world, metres_per_unit, up_axis)
    pose_transform = _decompose_normalized_transform(normalized_transform)
    if pose_transform is None:
        vertices = [_normalize_point(world.Transform(point), metres_per_unit, up_axis) for point in points]
        pose = Pose.from_xyz(0.0, 0.0, 0.0)
        mesh_scale = [1.0, 1.0, 1.0]
        transform_mode = "baked"
        report.diagnostics.append(
            UsdImportDiagnostic(
                severity="warning",
                code="transform_baked",
                message=(
                    "USD transform contains non-uniform scale, reflection, or shear; " "it was baked into mesh vertices."
                ),
                prim_path=prim_path,
                schema_type=prim.GetTypeName(),
            )
        )
    else:
        pose, uniform_scale = pose_transform
        vertices = [(float(point[0]), float(point[1]), float(point[2])) for point in points]
        mesh_scale = [uniform_scale, uniform_scale, uniform_scale]
        transform_mode = "pose"
    triangles = _triangulate_faces([int(value) for value in counts], [int(value) for value in indices])
    if not triangles:
        report.diagnostics.append(
            UsdImportDiagnostic(
                severity="warning",
                code="mesh_without_triangles",
                message="Mesh contains no valid polygon faces and was skipped.",
                prim_path=prim_path,
                schema_type=prim.GetTypeName(),
            )
        )
        return

    texture_coordinates = _mesh_texture_coordinates(mesh)
    normals = _mesh_local_normals(mesh) if transform_mode == "pose" else _mesh_normals(mesh, world, up_axis)
    if texture_coordinates is None:
        expanded = _expand_face_varying_uvs(
            mesh,
            vertices,
            triangles,
            normals,
            [int(value) for value in counts],
            [int(value) for value in indices],
        )
        if expanded is not None:
            vertices, triangles, texture_coordinates, normals = expanded
    mesh_path = _write_obj(vertices, triangles, prim_path, texture_coordinates, normals)
    color, texture_path = _material_appearance(mesh, usd_shade)
    visual = ShapeParams(shape_type="mesh", mesh_path=str(mesh_path), mesh_scale=mesh_scale, rgba_color=color)
    collision_enabled = prim_path in collision_budget.enabled_paths
    collision = ShapeParams(shape_type="mesh", mesh_path=str(mesh_path), mesh_scale=mesh_scale) if collision_enabled else None
    obj = SimObject.from_mesh(
        visual_shape=visual,
        collision_shape=collision,
        pose=pose,
        mass=0.0,
        pickable=False,
        sim_core=sim_core,
        collision_mode=CollisionMode.STATIC if collision_enabled else CollisionMode.DISABLED,
        name=prim.GetName(),
        user_data={
            "usd_prim_path": prim_path,
            "usd_transform_mode": transform_mode,
            "usd_normalized_world_transform": normalized_transform.tolist(),
        },
    )
    if texture_path is not None and texture_coordinates is not None:
        _apply_texture(obj.body_id, texture_path, sim_core.client, options.texture_brightness)
    report.created_object_ids[prim_path] = obj.object_id
    if transform_mode == "pose":
        pose_data = _pose_data(pose)
        obj.user_data["usd_normalized_world_pose"] = pose_data
        report.normalized_object_poses[prim_path] = pose_data
    report.normalized_world_transforms[prim_path] = normalized_transform.tolist()
    report.object_transform_modes[prim_path] = transform_mode


def _normalize_point(point: Any, metres_per_unit: float, up_axis: str) -> tuple[float, float, float]:
    x = float(point[0]) * metres_per_unit
    y = float(point[1]) * metres_per_unit
    z = float(point[2]) * metres_per_unit
    if up_axis == "Y":
        return (x, -z, y)
    return (x, y, z)


def _normalized_affine_transform(world: Any, metres_per_unit: float, up_axis: str) -> np.ndarray:
    """Return the exact stage-to-PyBullet affine transform as a 4×4 matrix."""
    origin = np.asarray(_normalize_point(world.Transform((0.0, 0.0, 0.0)), metres_per_unit, up_axis), dtype=float)
    columns = [
        np.asarray(_normalize_point(world.Transform(axis), metres_per_unit, up_axis), dtype=float) - origin
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]
    transform = np.identity(4)
    transform[:3, :3] = np.column_stack(columns)
    transform[:3, 3] = origin
    return transform


def _decompose_normalized_transform(transform: np.ndarray) -> tuple[Pose, float] | None:
    """Return a PyBullet pose and uniform mesh scale for a representable USD transform.

    PyBullet applies a mesh scale in local coordinates followed by a rigid body
    pose. That represents translation, rotation, and positive uniform scale;
    non-uniform scale, reflection, and shear intentionally use the baked path.
    """
    origin = transform[:3, 3]
    linear = transform[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    if np.any(scales <= 1e-12) or not np.allclose(scales, scales[0], rtol=1e-6, atol=1e-9):
        return None
    rotation_matrix = linear / scales[0]
    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.identity(3), rtol=1e-6, atol=1e-9):
        return None
    if np.linalg.det(rotation_matrix) <= 0.0:
        return None
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat().tolist()
    return Pose(position=origin.tolist(), orientation=quaternion), float(scales[0])


def _pose_data(pose: Pose) -> dict[str, list[float]]:
    """Return JSON-serializable pose data for reports and object metadata."""
    return {"position": list(pose.position), "orientation": list(pose.orientation)}


def _normalization_description(up_axis: str, metres_per_unit: float) -> str:
    axis = "(x, y, z) -> (x, -z, y)" if up_axis == "Y" else "identity"
    return f"{axis}; scale={metres_per_unit:g} metres per stage unit"


def _triangulate_faces(counts: list[int], indices: list[int]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        if len(face) < 3:
            continue
        for offset in range(1, len(face) - 1):
            triangles.append((face[0], face[offset], face[offset + 1]))
    return triangles


def _mesh_triangle_count(mesh: Any) -> int:
    """Return the fan-triangulated face count without allocating triangle tuples."""
    return sum(max(0, int(count) - 2) for count in (mesh.GetFaceVertexCountsAttr().Get() or []))


def _material_appearance(mesh: Any, usd_shade: Any) -> tuple[list[float], str | None]:
    """Return a PyBullet approximation of a USD Preview Surface or MDL material."""
    prim = mesh.GetPrim()
    material, _ = usd_shade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material:
        material = usd_shade.Material(material)
        preview = material.GetSurfaceOutput()
        if preview and preview.HasConnectedSource():
            shader = usd_shade.Shader(preview.GetConnectedSource()[0].GetPrim())
            if shader.GetIdAttr().Get() == "UsdPreviewSurface":
                return _shader_appearance(shader, ("diffuseColor",), ("file",), _display_color(prim))
        mdl = material.GetSurfaceOutput("mdl")
        if mdl and mdl.HasConnectedSource():
            shader = usd_shade.Shader(mdl.GetConnectedSource()[0].GetPrim())
            return _shader_appearance(
                shader,
                ("ColorAlbedo", "diffuse_color", "base_color"),
                ("AlbedoTexture", "diffuse_texture", "base_color_texture"),
                _display_color(prim),
                ignore_near_black_texture_tint=True,
            )
    return _display_color(prim), None


def _shader_appearance(
    shader: Any,
    color_names: tuple[str, ...],
    texture_names: tuple[str, ...],
    fallback_color: list[float],
    *,
    ignore_near_black_texture_tint: bool = False,
) -> tuple[list[float], str | None]:
    color = fallback_color
    has_explicit_color = False
    for color_name in color_names:
        color_input = shader.GetInput(color_name)
        if color_input and color_input.Get() is not None:
            value = color_input.Get()
            color = [float(value[0]), float(value[1]), float(value[2]), float(value[3]) if len(value) > 3 else 1.0]
            has_explicit_color = True
            break
    texture = None
    for texture_name in texture_names:
        texture_input = shader.GetInput(texture_name)
        if texture_input and texture_input.Get() is not None:
            texture = texture_input.Get()
            break
    resolved_path = getattr(texture, "resolvedPath", "") if texture else ""
    texture_path = resolved_path if resolved_path and Path(resolved_path).is_file() else None
    # PyBullet multiplies a visual shape's RGBA by its texture. Isaac MDL
    # sometimes authors a near-black placeholder ColorAlbedo next to a real
    # albedo map (for example its floor), but uses meaningful orange/red tint
    # values for shelves, walls, and forklifts. Ignore only the placeholder.
    if texture_path is not None and (not has_explicit_color or (ignore_near_black_texture_tint and max(color[:3]) < 0.1)):
        color = [1.0, 1.0, 1.0, 1.0]
    return color, texture_path


def _display_color(prim: Any) -> list[float]:
    value = prim.GetAttribute("primvars:displayColor").Get()
    if value:
        color = value[0]
        return [float(color[0]), float(color[1]), float(color[2]), 1.0]
    return [0.55, 0.58, 0.62, 1.0]


def _mesh_texture_coordinates(mesh: Any) -> list[tuple[float, float]] | None:
    """Return vertex UVs from the common USD ``st`` or ``st_0`` primvar."""
    try:
        from pxr import UsdGeom
    except ImportError:  # pragma: no cover - guarded by the importer dependency
        return None
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None
    for name in ("st", "st_0"):
        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar(name)
        if not primvar or primvar.GetInterpolation() not in {"vertex", "varying"}:
            continue
        values = primvar.Get()
        if values and len(values) == len(points):
            return [(float(value[0]), float(value[1])) for value in values]
    return None


def _expand_face_varying_uvs(
    mesh: Any,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    normals: list[tuple[float, float, float]] | None,
    counts: list[int],
    face_indices: list[int],
) -> (
    tuple[
        list[tuple[float, float, float]],
        list[tuple[int, int, int]],
        list[tuple[float, float]],
        list[tuple[float, float, float]] | None,
    ]
    | None
):
    """Split vertices as needed to preserve indexed USD ``faceVarying`` UVs."""
    try:
        from pxr import UsdGeom
    except ImportError:  # pragma: no cover - guarded by the importer dependency
        return None
    primvars = UsdGeom.PrimvarsAPI(mesh)
    for name in ("st", "st_0"):
        primvar = primvars.GetPrimvar(name)
        if not primvar or primvar.GetInterpolation() != "faceVarying":
            continue
        values = primvar.Get()
        if not values:
            continue
        uv_indices = [int(value) for value in primvar.GetIndices()]
        corner_count = len(face_indices)
        if uv_indices and len(uv_indices) != corner_count:
            continue
        if not uv_indices:
            uv_indices = list(range(corner_count))
        if any(index < 0 or index >= len(values) for index in uv_indices):
            continue
        corner_triangles = _triangulate_face_corners(counts)
        if len(corner_triangles) != len(triangles):
            return None
        remapped_vertices: list[tuple[float, float, float]] = []
        remapped_uvs: list[tuple[float, float]] = []
        remapped_normals: list[tuple[float, float, float]] | None = [] if normals is not None else None
        remapped_triangles: list[tuple[int, int, int]] = []
        remap: dict[tuple[int, int], int] = {}
        for triangle, corners in zip(triangles, corner_triangles):
            remapped_triangle: list[int] = []
            for point_index, corner_index in zip(triangle, corners):
                key = (point_index, uv_indices[corner_index])
                output_index = remap.get(key)
                if output_index is None:
                    output_index = len(remapped_vertices)
                    remap[key] = output_index
                    remapped_vertices.append(vertices[point_index])
                    uv = values[key[1]]
                    remapped_uvs.append((float(uv[0]), float(uv[1])))
                    if remapped_normals is not None and normals is not None:  # set together
                        remapped_normals.append(normals[point_index])
                remapped_triangle.append(output_index)
            remapped_triangles.append(tuple(remapped_triangle))
        return remapped_vertices, remapped_triangles, remapped_uvs, remapped_normals
    return None


def _triangulate_face_corners(counts: list[int]) -> list[tuple[int, int, int]]:
    """Return source face-corner indices matching :func:`_triangulate_faces`."""
    corners: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        if count >= 3:
            for offset in range(1, count - 1):
                corners.append((cursor, cursor + offset, cursor + offset + 1))
        cursor += count
    return corners


def _mesh_normals(mesh: Any, world_transform: Any, up_axis: str) -> list[tuple[float, float, float]] | None:
    """Return transformed vertex normals when USD authors one normal per point."""
    values = mesh.GetNormalsAttr().Get()
    points = mesh.GetPointsAttr().Get()
    if not values or not points or len(values) != len(points) or mesh.GetNormalsInterpolation() != "vertex":
        return None
    normals = []
    for value in values:
        direction = world_transform.TransformDir(value)
        x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
        if up_axis == "Y":
            x, y, z = x, -z, y
        length = (x * x + y * y + z * z) ** 0.5
        if length == 0.0:
            return None
        normals.append((x / length, y / length, z / length))
    return normals


def _mesh_local_normals(mesh: Any) -> list[tuple[float, float, float]] | None:
    """Return normalized vertex normals for a mesh whose transform is a body pose."""
    values = mesh.GetNormalsAttr().Get()
    points = mesh.GetPointsAttr().Get()
    if not values or not points or len(values) != len(points) or mesh.GetNormalsInterpolation() != "vertex":
        return None
    normals = []
    for value in values:
        x, y, z = float(value[0]), float(value[1]), float(value[2])
        length = (x * x + y * y + z * z) ** 0.5
        if length == 0.0:
            return None
        normals.append((x / length, y / length, z / length))
    return normals


def _apply_texture(body_id: int, texture_path: str, physics_client_id: int, brightness: float) -> None:
    """Apply a cached texture, reloading if a reused client ID made it stale."""
    import pybullet as p

    cache_key = (physics_client_id, texture_path, brightness)
    texture_id = _TEXTURE_CACHE.get(cache_key)
    if texture_id is None:
        texture_id = p.loadTexture(_brightened_texture_path(texture_path, brightness), physicsClientId=physics_client_id)
        _TEXTURE_CACHE[cache_key] = texture_id
    if texture_id < 0:
        return
    try:
        p.changeVisualShape(body_id, -1, textureUniqueId=texture_id, physicsClientId=physics_client_id)
    except p.error:
        # PyBullet may reuse a client ID after disconnecting.  Texture IDs
        # belong to the old connection in that case, so reload once.
        _TEXTURE_CACHE.pop(cache_key, None)
        texture_id = p.loadTexture(_brightened_texture_path(texture_path, brightness), physicsClientId=physics_client_id)
        _TEXTURE_CACHE[cache_key] = texture_id
        if texture_id >= 0:
            p.changeVisualShape(body_id, -1, textureUniqueId=texture_id, physicsClientId=physics_client_id)


def _brightened_texture_path(texture_path: str, brightness: float) -> str:
    """Return a cached brightness-adjusted copy, preserving the source alpha."""
    if brightness == 1.0:
        return texture_path
    from PIL import Image, ImageEnhance

    source = Path(texture_path)
    stat = source.stat()
    digest = sha256(f"{source}|{stat.st_mtime_ns}|{stat.st_size}|{brightness:.6g}".encode("utf-8")).hexdigest()
    output = Path(gettempdir()) / "pybullet_fleet_usd_textures" / f"{digest}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return str(output)
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        rgb = ImageEnhance.Brightness(rgba.convert("RGB")).enhance(brightness)
        rgb.putalpha(alpha)
        rgb.save(output)
    return str(output)


def _write_obj(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    prim_path: str,
    texture_coordinates: list[tuple[float, float]] | None = None,
    normals: list[tuple[float, float, float]] | None = None,
) -> Path:
    digest = _mesh_cache_digest(vertices, triangles, texture_coordinates, normals)
    cache_dir = Path(gettempdir()) / "pybullet_fleet_usd_meshes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}.obj"
    if path.exists():
        return path
    lines = [f"# Generated from USD prim {prim_path}"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    if texture_coordinates is not None:
        lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in texture_coordinates)
    if normals is not None:
        lines.extend(f"vn {x:.9g} {y:.9g} {z:.9g}" for x, y, z in normals)
    if texture_coordinates is not None and normals is not None:
        lines.extend(f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}" for a, b, c in triangles)
    elif texture_coordinates is not None:
        lines.extend(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in triangles)
    elif normals is not None:
        lines.extend(f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}" for a, b, c in triangles)
    else:
        lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mesh_cache_digest(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    texture_coordinates: list[tuple[float, float]] | None,
    normals: list[tuple[float, float, float]] | None,
) -> str:
    """Hash mesh content incrementally without materializing one large repr."""
    digest = sha256()

    def update_rows(tag: bytes, rows: list[tuple[Any, ...]] | None, row_format: str) -> None:
        digest.update(tag)
        if rows is None:
            digest.update(b"\0")
            return
        digest.update(b"\1")
        digest.update(pack("<Q", len(rows)))
        for row in rows:
            digest.update(pack(row_format, *row))

    update_rows(b"vertices", vertices, "<3d")
    update_rows(b"triangles", triangles, "<3q")
    update_rows(b"texture_coordinates", texture_coordinates, "<2d")
    update_rows(b"normals", normals, "<3d")
    return digest.hexdigest()
