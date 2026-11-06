"""Focused tests for the optional OpenUSD static-world importer."""

from pathlib import Path
from types import SimpleNamespace

import pybullet as p
import pytest

pytest.importorskip("pxr")

from pybullet_fleet import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.usd_loader import (
    UsdImportOptions,
    _expand_face_varying_uvs,
    _mesh_cache_digest,
    _mesh_texture_coordinates,
    _shader_appearance,
    load_usd_world,
)
import pybullet_fleet.usd_loader as usd_loader


def test_mesh_cache_digest_is_deterministic_and_includes_optional_attributes():
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    triangles = [(0, 1, 2)]

    digest = _mesh_cache_digest(vertices, triangles, None, None)

    assert digest == _mesh_cache_digest(list(vertices), list(triangles), None, None)
    assert digest != _mesh_cache_digest(vertices, triangles, [(0.0, 0.0)] * 3, None)
    assert digest != _mesh_cache_digest(vertices, [(0, 2, 1)], None, None)


def test_apply_texture_reloads_a_stale_cached_id_after_client_reconnect(monkeypatch):
    texture_path = "/tmp/texture.png"
    cache_key = (7, texture_path, 1.0)
    usd_loader._TEXTURE_CACHE[cache_key] = 13
    changes = []

    def change_visual_shape(*_args, **kwargs):
        changes.append(kwargs["textureUniqueId"])
        if len(changes) == 1:
            raise p.error("stale texture ID")

    try:
        monkeypatch.setattr("pybullet.loadTexture", lambda *_args, **_kwargs: 29)
        monkeypatch.setattr("pybullet.changeVisualShape", change_visual_shape)
        monkeypatch.setattr(usd_loader, "_brightened_texture_path", lambda path, _brightness: path)

        usd_loader._apply_texture(3, texture_path, physics_client_id=7, brightness=1.0)

        assert changes == [13, 29]
        assert usd_loader._TEXTURE_CACHE[cache_key] == 29
    finally:
        usd_loader._TEXTURE_CACHE.pop(cache_key, None)


@pytest.fixture
def sim_core_params():
    return {"enable_floor": False}


def test_import_normalizes_y_up_units_and_preserves_world_pose(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "axis_units.usda"

    report = load_usd_world(stage, sim_core=sim_core)

    assert report.objects_created == 1
    assert report.source_up_axis == "Y"
    assert report.source_meters_per_unit == pytest.approx(0.01)
    obj = sim_core.sim_objects[0]
    assert obj.user_data["usd_transform_mode"] == "pose"
    assert obj.user_data["usd_normalized_world_pose"]["position"] == pytest.approx([1.0, -3.0, 2.0])
    assert report.normalized_object_poses["/Scene/Triangle"]["position"] == pytest.approx([1.0, -3.0, 2.0])
    assert report.object_transform_modes["/Scene/Triangle"] == "pose"
    body_position, _ = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=sim_core.client)
    assert body_position == pytest.approx((1.0, -3.0, 2.0))
    aabb_min, aabb_max = p.getAABB(obj.body_id, physicsClientId=sim_core.client)
    # USD (100, 200, 300) cm converts once to PBF (1, -3, 2) metres.
    # PyBullet expands concave-mesh collision AABBs by its small collision
    # margin. The tolerance is far below one centimetre and still detects a
    # missing/double unit conversion or the wrong Y-up rotation.
    assert aabb_min[0] == pytest.approx(1.0, abs=0.003)
    assert aabb_min[1] == pytest.approx(-4.0, abs=0.003)
    assert aabb_min[2] == pytest.approx(2.0, abs=0.003)
    assert aabb_max[0] == pytest.approx(2.0, abs=0.003)
    assert aabb_max[1] == pytest.approx(-3.0, abs=0.003)
    assert aabb_max[2] == pytest.approx(2.0, abs=0.003)


def test_import_can_precede_run_simulation_initialization():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        stage = Path(__file__).parent / "fixtures" / "usd" / "axis_units.usda"
        report = load_usd_world(stage, sim_core=sim)

        assert report.objects_created == 1
        assert len(sim.sim_objects) == 1
    finally:
        p.disconnect(sim.client)


def test_import_has_stable_prim_source_metadata(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "axis_units.usda"

    report = load_usd_world(stage, sim_core=sim_core, options=UsdImportOptions(collision=False))

    assert "/Scene/Triangle" in report.created_object_ids
    assert sim_core.sim_objects[0].user_data["usd_prim_path"] == "/Scene/Triangle"


def test_nonuniform_transform_uses_baked_geometry_and_records_its_exact_transform(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "nonuniform_scale.usda"

    report = load_usd_world(stage, sim_core=sim_core, options=UsdImportOptions(collision=False))

    path = "/Scene/StretchedTriangle"
    obj = sim_core.sim_objects[0]
    assert obj.user_data["usd_transform_mode"] == "baked"
    assert "usd_normalized_world_pose" not in obj.user_data
    assert obj.user_data["usd_normalized_world_transform"][0][3] == pytest.approx(2.0)
    assert obj.user_data["usd_normalized_world_transform"][1][3] == pytest.approx(3.0)
    assert report.object_transform_modes[path] == "baked"
    assert path not in report.normalized_object_poses
    assert report.normalized_world_transforms[path][2][3] == pytest.approx(4.0)
    assert any(diagnostic.code == "transform_baked" for diagnostic in report.diagnostics)


def test_import_rejects_non_positive_texture_brightness(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "axis_units.usda"

    with pytest.raises(ValueError, match="texture_brightness must be positive"):
        load_usd_world(stage, sim_core=sim_core, options=UsdImportOptions(texture_brightness=0.0))


def test_import_uses_preview_surface_diffuse_color(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "preview_surface_color.usda"

    load_usd_world(stage, sim_core=sim_core, options=UsdImportOptions(collision=False))

    visual = p.getVisualShapeData(sim_core.sim_objects[0].body_id, physicsClientId=sim_core.client)[0]
    assert visual[7] == pytest.approx((0.1, 0.3, 0.8, 1.0))


def test_mesh_texture_coordinates_supports_isaac_st_0_primvar():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(Path(__file__).parent / "fixtures" / "usd" / "preview_surface_color.usda"))

    assert _mesh_texture_coordinates(UsdGeom.Mesh(stage.GetPrimAtPath("/Scene/Triangle"))) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    ]


def test_texture_only_material_keeps_its_authored_brightness():
    texture = SimpleNamespace(resolvedPath=str(Path(__file__)))

    class Shader:
        def GetInput(self, name):
            if name == "albedo":
                return SimpleNamespace(Get=lambda: texture)
            return None

    color, texture_path = _shader_appearance(Shader(), ("base_color",), ("albedo",), [0.2, 0.3, 0.4, 1.0])

    assert color == [1.0, 1.0, 1.0, 1.0]
    assert texture_path == str(Path(__file__))


def test_isaac_mdl_texture_ignores_a_near_black_placeholder_color_input():
    texture = SimpleNamespace(resolvedPath=str(Path(__file__)))

    class Shader:
        def GetInput(self, name):
            values = {"base_color": (0.01, 0.02, 0.03, 0.0), "albedo": texture}
            return SimpleNamespace(Get=lambda: values[name]) if name in values else None

    color, _ = _shader_appearance(
        Shader(),
        ("base_color",),
        ("albedo",),
        [0.2, 0.3, 0.4, 1.0],
        ignore_near_black_texture_tint=True,
    )

    assert color == [1.0, 1.0, 1.0, 1.0]


def test_isaac_mdl_texture_preserves_a_meaningful_color_tint():
    texture = SimpleNamespace(resolvedPath=str(Path(__file__)))

    class Shader:
        def GetInput(self, name):
            values = {"base_color": (0.488, 0.106, 0.0, 0.0), "albedo": texture}
            return SimpleNamespace(Get=lambda: values[name]) if name in values else None

    color, _ = _shader_appearance(
        Shader(),
        ("base_color",),
        ("albedo",),
        [0.2, 0.3, 0.4, 1.0],
        ignore_near_black_texture_tint=True,
    )

    assert color == pytest.approx([0.488, 0.106, 0.0, 0.0])


def test_face_varying_uvs_are_remapped_to_obj_compatible_vertices():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(Path(__file__).parent / "fixtures" / "usd" / "face_varying_uv.usda"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Scene/Quad"))
    result = _expand_face_varying_uvs(
        mesh,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2), (0, 2, 3)],
        None,
        [4],
        [0, 1, 2, 3],
    )

    assert result is not None
    vertices, triangles, texture_coordinates, normals = result
    assert len(vertices) == 4
    assert triangles == [(0, 1, 2), (0, 2, 3)]
    assert texture_coordinates == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert normals is None


def test_import_skips_guide_purpose_collision_meshes(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "purpose_filter.usda"

    report = load_usd_world(stage, sim_core=sim_core, options=UsdImportOptions(collision=False))

    assert set(report.created_object_ids) == {"/Scene/VisibleTriangle"}


def test_point_instancer_expands_mesh_prototypes_with_stable_instance_paths(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "point_instances.usda"

    report = load_usd_world(stage, sim_core=sim_core, options=UsdImportOptions(point_instance_warning=1))

    assert report.objects_created == 2
    assert set(report.created_object_ids) == {"/Scene/Pallets[0]", "/Scene/Pallets[1]"}
    assert any(diagnostic.code == "point_instance_warning" for diagnostic in report.diagnostics)
    first = next(obj for obj in sim_core.sim_objects if obj.object_id == report.created_object_ids["/Scene/Pallets[0]"])
    assert first.user_data["usd_prim_path"] == "/Scene/Pallets[0]"
    aabb_min, _ = p.getAABB(first.body_id, physicsClientId=sim_core.client)
    assert aabb_min[:2] == pytest.approx((2.0, 3.0), abs=0.003)


def test_point_instancer_limit_rejects_before_creating_bodies(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "point_instances.usda"

    with pytest.raises(ValueError, match="point_instance_limit_exceeded"):
        load_usd_world(
            stage,
            sim_core=sim_core,
            options=UsdImportOptions(point_instance_warning=1, max_point_instances=1),
        )

    assert sim_core.sim_objects == []


def test_collision_triangle_budget_imports_visual_only_with_diagnostic(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "point_instances.usda"

    report = load_usd_world(
        stage,
        sim_core=sim_core,
        options=UsdImportOptions(max_collision_triangles_per_mesh=1),
    )

    assert report.objects_created == 2
    assert all(obj.collision_mode.name == "DISABLED" for obj in sim_core.sim_objects)
    assert [diagnostic.code for diagnostic in report.diagnostics].count("collision_mesh_unavailable") == 2


def test_collision_required_rejects_before_creating_bodies_when_budget_exceeded(sim_core):
    stage = Path(__file__).parent / "fixtures" / "usd" / "point_instances.usda"

    with pytest.raises(ValueError, match="collision_mesh_unavailable"):
        load_usd_world(
            stage,
            sim_core=sim_core,
            options=UsdImportOptions(max_collision_triangles_per_mesh=1, collision_required=True),
        )

    assert sim_core.sim_objects == []
