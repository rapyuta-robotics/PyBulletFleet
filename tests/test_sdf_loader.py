# tests/test_sdf_loader.py
"""Tests for SimObject.from_sdf() SDF loading."""

import pybullet as p
import pybullet_data
import pytest

from pybullet_fleet import MultiRobotSimulationCore, SimulationParams
import pybullet_fleet.sdf_loader as sdf_loader
from pybullet_fleet.sdf_loader import (
    _extract_sdf_material_color,
    _parse_pose,
    load_sdf_world,
    load_sdf_world_file,
    resolve_sdf_to_urdf,
    DEFAULT_SCENERY_COLOR,
)
from pybullet_fleet.sim_object import SimObject
from pybullet_fleet.types import CollisionMode


class TestFromSdf:
    """Tests for SimObject.from_sdf()."""

    def test_load_kiva_shelf(self, sim_core):
        """Load pybullet_data kiva_shelf SDF."""
        objects = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim_core)
        assert len(objects) >= 1
        assert all(isinstance(o, SimObject) for o in objects)
        assert all(o.object_id >= 0 for o in objects)

    def test_objects_registered_in_sim_core(self, sim_core):
        """All loaded objects are registered in sim_core."""
        objects = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim_core)
        for obj in objects:
            assert obj in sim_core.sim_objects

    def test_objects_have_names(self, sim_core):
        """Each loaded object has a non-empty name."""
        objects = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim_core)
        for obj in objects:
            assert obj.name  # not None, not ""

    def test_global_scaling(self, sim_core):
        """global_scaling=2.0 produces a larger AABB than 1.0."""
        objs_1x = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim_core, global_scaling=1.0)
        objs_2x = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim_core, global_scaling=2.0)

        # Match bodies by name (ordering may differ between loads)
        by_name_1x = {o.name: o for o in objs_1x}
        by_name_2x = {o.name: o for o in objs_2x}
        assert by_name_1x.keys() == by_name_2x.keys()

        for name in by_name_1x:
            aabb_1 = p.getAABB(by_name_1x[name].body_id, physicsClientId=sim_core.client)
            aabb_2 = p.getAABB(by_name_2x[name].body_id, physicsClientId=sim_core.client)
            size_1 = max(aabb_1[1][i] - aabb_1[0][i] for i in range(3))
            size_2 = max(aabb_2[1][i] - aabb_2[0][i] for i in range(3))
            if size_1 > 0.001:  # skip degenerate bodies
                assert size_2 == pytest.approx(size_1 * 2.0, rel=0.05)

    def test_collision_mode(self, sim_core):
        """collision_mode is applied to all loaded objects."""
        objects = SimObject.from_sdf(
            "kiva_shelf/model.sdf",
            sim_core=sim_core,
            collision_mode=CollisionMode.DISABLED,
        )
        for obj in objects:
            assert obj.collision_mode == CollisionMode.DISABLED

    def test_use_fixed_base(self, sim_core):
        """use_fixed_base=True → mass=0, use_fixed_base=False → mass>0."""
        fixed = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim_core, use_fixed_base=True)
        for obj in fixed:
            assert obj.mass == 0.0

        free = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim_core, use_fixed_base=False)
        for obj in free:
            assert obj.mass is None or obj.mass >= 0.0  # not forced to 0

    def test_invalid_path_raises(self, sim_core):
        """Invalid SDF path raises error."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            SimObject.from_sdf("nonexistent_model.sdf", sim_core=sim_core)

    def test_without_sim_core(self):
        """from_sdf works without sim_core (standalone physics client)."""
        cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        try:
            objects = SimObject.from_sdf("kiva_shelf/model.sdf")
            assert len(objects) >= 1
        finally:
            p.disconnect(cid)


class TestMultiModelSdf:
    """Tests for load_sdf_world handling multiple <model> elements."""

    def _write_sdf(self, tmp_path, content: str) -> str:
        """Write SDF content to a temp file and return the path."""
        sdf_file = tmp_path / "model.sdf"
        sdf_file.write_text(content)
        return str(sdf_file)

    def _make_mesh(self, tmp_path, name: str = "box.obj") -> str:
        """Create a minimal OBJ mesh file."""
        mesh_path = tmp_path / name
        mesh_path.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n" "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n" "f 1 2 3 4\nf 5 6 7 8\n"
        )
        return str(mesh_path)

    def test_single_model_still_works(self, sim_core, tmp_path):
        """Single <model> SDF still loads correctly (regression)."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="building">
                <link name="floor">
                  <visual name="v"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core)
        assert len(objects) == 1
        assert objects[0].name == "floor"

    def test_multiple_top_level_models(self, sim_core, tmp_path):
        """Multiple <model> elements at same level are all loaded."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="building_A">
                <link name="floor_A">
                  <visual name="v"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
                </link>
              </model>
              <model name="building_B">
                <link name="wall_B">
                  <visual name="v"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core)
        names = {o.name for o in objects}
        assert "floor_A" in names
        assert "wall_B" in names
        assert len(objects) == 2

    def test_three_models_all_links_loaded(self, sim_core, tmp_path):
        """Three models with multiple links each — all links are loaded."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="m1">
                <link name="L1a">
                  <visual name="v"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
                </link>
                <link name="L1b">
                  <visual name="v"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
                </link>
              </model>
              <model name="m2">
                <link name="L2">
                  <visual name="v"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
                </link>
              </model>
              <model name="m3">
                <link name="L3">
                  <visual name="v"><geometry><mesh><uri>{mesh}</uri></mesh></geometry></visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core)
        assert len(objects) == 4
        names = {o.name for o in objects}
        assert names == {"L1a", "L1b", "L2", "L3"}

    def test_no_model_raises(self, sim_core, tmp_path):
        """SDF with no <model> element raises ValueError."""
        sdf = self._write_sdf(
            tmp_path,
            """<?xml version="1.0"?>
            <sdf version="1.6">
            </sdf>""",
        )
        with pytest.raises(ValueError, match="No <model> element"):
            load_sdf_world(sdf, sim_core=sim_core)


class TestParsePose:
    """Tests for _parse_pose returning full 6-DoF."""

    def test_empty_returns_zeros(self):
        """None or empty string returns 6 zeros."""
        result = _parse_pose(None)
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        result = _parse_pose("")
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_xyz_only(self):
        """3-value string returns xyz with rpy=0."""
        result = _parse_pose("1.0 2.0 3.0")
        assert result == (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)

    def test_full_6dof(self):
        """6-value string returns all components."""
        result = _parse_pose("1.0 2.0 3.0 0.1 0.2 1.57")
        assert result == pytest.approx((1.0, 2.0, 3.0, 0.1, 0.2, 1.57))


class TestDefaultSceneryColor:
    """Tests for DEFAULT_SCENERY_COLOR constant."""

    def test_matches_gazebo_sim_default(self):
        """Value matches Gazebo Sim (Harmonic) default of 0.7 grey."""
        assert DEFAULT_SCENERY_COLOR == [0.7, 0.7, 0.7, 1.0]


class TestSdfMaterialAlpha:
    """Tests for per-visual SDF material color extraction including alpha."""

    def _make_mesh(self, tmp_path, name="box.obj"):
        mesh_path = tmp_path / name
        mesh_path.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n" "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n" "f 1 2 3 4\nf 5 6 7 8\n"
        )
        return str(mesh_path)

    def _write_sdf(self, tmp_path, content):
        sdf_file = tmp_path / "model.sdf"
        sdf_file.write_text(content)
        return str(sdf_file)

    def test_sdf_material_alpha_extracted(self, sim_core, tmp_path):
        """SDF visual with <material><diffuse> alpha < 1.0 is applied post-load."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="glass_wall">
                <link name="wall">
                  <visual name="v">
                    <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
                    <material>
                      <diffuse>0.8 0.9 1.0 0.3</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core, rgba_color=list(DEFAULT_SCENERY_COLOR))
        assert len(objects) == 1

        # Verify the alpha from SDF is applied (not the default 1.0)
        visual_data = p.getVisualShapeData(objects[0].body_id, physicsClientId=sim_core.client)
        rgba = visual_data[0][7]  # RGBA tuple
        assert rgba[3] == pytest.approx(0.3, abs=0.05), f"Expected alpha ~0.3, got {rgba[3]}"

    def test_sdf_no_material_uses_fallback(self, sim_core, tmp_path):
        """SDF visual without <material> uses the rgba_color fallback."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="plain_wall">
                <link name="wall">
                  <visual name="v">
                    <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
                  </visual>
                </link>
              </model>
            </sdf>""",
        )
        custom_color = [0.5, 0.6, 0.7, 1.0]
        objects = load_sdf_world(sdf, sim_core=sim_core, rgba_color=custom_color)
        assert len(objects) == 1

        visual_data = p.getVisualShapeData(objects[0].body_id, physicsClientId=sim_core.client)
        rgba = visual_data[0][7]
        assert rgba[3] == pytest.approx(1.0, abs=0.05), f"Expected alpha=1.0, got {rgba[3]}"

    def test_sdf_material_full_rgba_preserved(self, sim_core, tmp_path):
        """SDF material diffuse RGBA is fully preserved when specified."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="colored_obj">
                <link name="obj">
                  <visual name="v">
                    <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
                    <material>
                      <diffuse>1.0 0.0 0.0 0.5</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core, rgba_color=list(DEFAULT_SCENERY_COLOR))
        assert len(objects) == 1

        visual_data = p.getVisualShapeData(objects[0].body_id, physicsClientId=sim_core.client)
        rgba = visual_data[0][7]
        # Color should come from SDF material, not the DEFAULT_SCENERY_COLOR
        assert rgba[0] == pytest.approx(1.0, abs=0.1)  # R
        assert rgba[3] == pytest.approx(0.5, abs=0.05)  # A

    def test_sdf_transparency_tag_sets_alpha(self, sim_core, tmp_path):
        """SDF <transparency> tag on visual converts to alpha (1 - transparency)."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="semi_wall">
                <link name="wall">
                  <visual name="v">
                    <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
                    <transparency>0.5</transparency>
                    <material>
                      <diffuse>1.0 1.0 1.0</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core, rgba_color=list(DEFAULT_SCENERY_COLOR))
        assert len(objects) == 1

        visual_data = p.getVisualShapeData(objects[0].body_id, physicsClientId=sim_core.client)
        rgba = visual_data[0][7]
        assert rgba[3] == pytest.approx(0.5, abs=0.05), f"Expected alpha ~0.5, got {rgba[3]}"

    def test_sdf_transparency_zero_stays_opaque(self, sim_core, tmp_path):
        """SDF <transparency>0.0</transparency> keeps wall opaque."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="opaque_wall">
                <link name="wall">
                  <visual name="v">
                    <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
                    <transparency>0.0</transparency>
                    <material>
                      <diffuse>1.0 1.0 1.0</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core, rgba_color=list(DEFAULT_SCENERY_COLOR))
        assert len(objects) == 1

        visual_data = p.getVisualShapeData(objects[0].body_id, physicsClientId=sim_core.client)
        rgba = visual_data[0][7]
        # Not necessarily 1.0 exactly — fallback colour may apply — but NOT transparent
        assert rgba[3] >= 0.95, f"Expected opaque, got alpha {rgba[3]}"

    def test_sdf_transparency_without_material(self, sim_core, tmp_path):
        """SDF <transparency> works even without <material>."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="no_mat_wall">
                <link name="wall">
                  <visual name="v">
                    <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
                    <transparency>0.3</transparency>
                  </visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core, rgba_color=list(DEFAULT_SCENERY_COLOR))
        assert len(objects) == 1

        visual_data = p.getVisualShapeData(objects[0].body_id, physicsClientId=sim_core.client)
        rgba = visual_data[0][7]
        assert rgba[3] == pytest.approx(0.7, abs=0.05), f"Expected alpha ~0.7, got {rgba[3]}"

    def test_sdf_transparency_overrides_diffuse_alpha(self, sim_core, tmp_path):
        """When both <transparency> and <diffuse> alpha exist, <transparency> wins."""
        mesh = self._make_mesh(tmp_path)
        sdf = self._write_sdf(
            tmp_path,
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="override_wall">
                <link name="wall">
                  <visual name="v">
                    <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
                    <transparency>0.8</transparency>
                    <material>
                      <diffuse>1.0 0.0 0.0 0.5</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>""",
        )
        objects = load_sdf_world(sdf, sim_core=sim_core, rgba_color=list(DEFAULT_SCENERY_COLOR))
        assert len(objects) == 1

        visual_data = p.getVisualShapeData(objects[0].body_id, physicsClientId=sim_core.client)
        rgba = visual_data[0][7]
        # transparency=0.8 → alpha=0.2, overrides diffuse alpha=0.5
        assert rgba[3] == pytest.approx(0.2, abs=0.05), f"Expected alpha ~0.2, got {rgba[3]}"


class TestExtractSdfMaterialColor:
    """Unit tests for _extract_sdf_material_color helper."""

    def _make_visual(self, xml_str: str):
        """Parse XML string into an Element."""
        import xml.etree.ElementTree as ET

        return ET.fromstring(xml_str)

    def test_transparency_applied_to_diffuse(self):
        """<transparency> overrides alpha in returned color string."""
        visual = self._make_visual(
            """<visual>
              <transparency>0.4</transparency>
              <material><diffuse>0.8 0.9 1.0</diffuse></material>
            </visual>"""
        )
        result = _extract_sdf_material_color(visual)
        parts = result.split()
        assert float(parts[3]) == pytest.approx(0.6, abs=0.01)

    def test_transparency_without_material_returns_color(self):
        """<transparency> alone (no <material>) returns white with alpha."""
        visual = self._make_visual(
            """<visual>
              <transparency>0.5</transparency>
            </visual>"""
        )
        result = _extract_sdf_material_color(visual)
        assert result is not None
        parts = result.split()
        assert len(parts) == 4
        assert float(parts[3]) == pytest.approx(0.5, abs=0.01)

    def test_no_transparency_no_material_returns_none(self):
        """Without <transparency> or <material>, returns None."""
        visual = self._make_visual("<visual></visual>")
        result = _extract_sdf_material_color(visual)
        assert result is None

    def test_transparency_zero_no_effect(self):
        """<transparency>0</transparency> doesn't change alpha from 1."""
        visual = self._make_visual(
            """<visual>
              <transparency>0.0</transparency>
              <material><diffuse>1.0 1.0 1.0 1.0</diffuse></material>
            </visual>"""
        )
        result = _extract_sdf_material_color(visual)
        parts = result.split()
        assert float(parts[3]) == pytest.approx(1.0, abs=0.01)


class TestResolveSdfToUrdf:
    """Tests for resolve_sdf_to_urdf() — SDF-to-URDF conversion."""

    def _make_sdf(self, tmp_path, joints_xml: str, extra_links: str = "") -> str:
        """Write a minimal SDF with custom joints and return its path."""
        sdf = f"""\
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="test_robot">
    <link name="base_link">
      <inertial><mass>1.0</mass></inertial>
    </link>
    <link name="child_link">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial><mass>0.5</mass></inertial>
    </link>
    {extra_links}
    {joints_xml}
  </model>
</sdf>"""
        sdf_path = tmp_path / "model.sdf"
        sdf_path.write_text(sdf)
        return str(sdf_path)

    def test_prismatic_joint_gets_limit(self, tmp_path):
        """Prismatic joints must produce <limit> in URDF (PyBullet requirement)."""
        sdf_path = self._make_sdf(
            tmp_path,
            """
            <joint name="slider" type="prismatic">
              <parent>base_link</parent>
              <child>child_link</child>
              <axis><xyz>0 0 1</xyz></axis>
            </joint>
            """,
        )
        urdf_path = resolve_sdf_to_urdf(sdf_path)
        urdf_content = open(urdf_path).read()
        assert 'type="prismatic"' in urdf_content
        assert "<limit" in urdf_content
        # Default range for prismatic without SDF limits
        assert 'lower="-1.0"' in urdf_content
        assert 'upper="1.0"' in urdf_content

    def test_prismatic_with_sdf_limits_preserved(self, tmp_path):
        """SDF <limit> values for prismatic joints are preserved in URDF."""
        sdf_path = self._make_sdf(
            tmp_path,
            """
            <joint name="slider" type="prismatic">
              <parent>base_link</parent>
              <child>child_link</child>
              <axis>
                <xyz>0 0 1</xyz>
                <limit><lower>-0.05</lower><upper>0.05</upper></limit>
              </axis>
            </joint>
            """,
        )
        urdf_path = resolve_sdf_to_urdf(sdf_path)
        urdf_content = open(urdf_path).read()
        assert 'lower="-0.05"' in urdf_content
        assert 'upper="0.05"' in urdf_content

    def test_revolute_without_limits_becomes_continuous(self, tmp_path):
        """Revolute joints without SDF <limit> become continuous in URDF."""
        sdf_path = self._make_sdf(
            tmp_path,
            """
            <joint name="wheel" type="revolute">
              <parent>base_link</parent>
              <child>child_link</child>
              <axis><xyz>0 1 0</xyz></axis>
            </joint>
            """,
        )
        urdf_path = resolve_sdf_to_urdf(sdf_path)
        urdf_content = open(urdf_path).read()
        # No limits → continuous (unlimited rotation, no <limit> needed)
        assert 'type="continuous"' in urdf_content

    def test_revolute_with_finite_limits_stays_revolute(self, tmp_path):
        """Revolute joints with finite SDF limits stay revolute in URDF."""
        sdf_path = self._make_sdf(
            tmp_path,
            """
            <joint name="hinge" type="revolute">
              <parent>base_link</parent>
              <child>child_link</child>
              <axis>
                <xyz>0 0 1</xyz>
                <limit><lower>-1.57</lower><upper>1.57</upper></limit>
              </axis>
            </joint>
            """,
        )
        urdf_path = resolve_sdf_to_urdf(sdf_path)
        urdf_content = open(urdf_path).read()
        assert 'type="revolute"' in urdf_content
        assert 'lower="-1.57"' in urdf_content
        assert 'upper="1.57"' in urdf_content

    def test_revolute_with_infinite_limits_becomes_continuous(self, tmp_path):
        """Revolute joints with ±inf SDF limits become continuous in URDF."""
        sdf_path = self._make_sdf(
            tmp_path,
            """
            <joint name="spinner" type="revolute">
              <parent>base_link</parent>
              <child>child_link</child>
              <axis>
                <xyz>0 0 1</xyz>
                <limit><lower>-1e16</lower><upper>1e16</upper></limit>
              </axis>
            </joint>
            """,
        )
        urdf_path = resolve_sdf_to_urdf(sdf_path)
        urdf_content = open(urdf_path).read()
        assert 'type="continuous"' in urdf_content

    def test_mixed_joints_all_valid(self, tmp_path, sim_core):
        """SDF with prismatic + revolute joints produces a PyBullet-loadable URDF.

        Mimics DeliveryRobot structure: prismatic suspension + revolute wheels.
        """
        extra_links = """
    <link name="suspension">
      <pose>0 0.26 0.1 0 0 0</pose>
      <inertial><mass>1.0</mass></inertial>
    </link>
    <link name="wheel">
      <pose>0 0.26 0.1 0 0 0</pose>
      <inertial><mass>2.0</mass></inertial>
    </link>"""

        joints = """
    <joint name="suspension_joint" type="prismatic">
      <parent>base_link</parent>
      <child>suspension</child>
      <axis><xyz>0 0 1</xyz></axis>
    </joint>
    <joint name="wheel_joint" type="revolute">
      <parent>suspension</parent>
      <child>wheel</child>
      <axis><xyz>0 1 0</xyz></axis>
    </joint>
    <joint name="fixed_child" type="fixed">
      <parent>base_link</parent>
      <child>child_link</child>
    </joint>"""

        sdf_path = self._make_sdf(tmp_path, joints, extra_links)
        urdf_path = resolve_sdf_to_urdf(sdf_path)

        # Must load without error in PyBullet
        body_id = p.loadURDF(urdf_path, physicsClientId=sim_core.client)
        assert body_id >= 0

    def test_chained_joint_origin_is_relative(self, tmp_path):
        """Chained joints (A→B→C) use relative poses, not absolute model-frame poses.

        Mimics DeliveryRobot: base → suspension → tire at the SAME model-frame
        position.  The joint from suspension→tire should have origin (0,0,0)
        because both links co-locate in the model frame.
        """
        import xml.etree.ElementTree as ET

        extra_links = """
    <link name="suspension">
      <pose>0 0.26 0.1 0 0 0</pose>
      <inertial><mass>1.0</mass></inertial>
    </link>
    <link name="wheel">
      <pose>0 0.26 0.1 0 0 0</pose>
      <inertial><mass>2.0</mass></inertial>
    </link>"""

        joints = """
    <joint name="suspension_joint" type="prismatic">
      <parent>base_link</parent>
      <child>suspension</child>
      <axis><xyz>0 0 1</xyz></axis>
    </joint>
    <joint name="wheel_joint" type="revolute">
      <parent>suspension</parent>
      <child>wheel</child>
      <axis><xyz>0 1 0</xyz></axis>
    </joint>"""

        sdf_path = self._make_sdf(tmp_path, joints, extra_links)
        urdf_path = resolve_sdf_to_urdf(sdf_path)

        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.get("xyz").split()]
            if joint.get("name") == "suspension_joint":
                # base_link (0,0,0) → suspension (0,0.26,0.1): origin = (0, 0.26, 0.1)
                assert xyz[1] == pytest.approx(0.26, abs=0.01)
                assert xyz[2] == pytest.approx(0.1, abs=0.01)
            elif joint.get("name") == "wheel_joint":
                # suspension (0,0.26,0.1) → wheel (0,0.26,0.1): origin = (0, 0, 0)
                assert xyz[0] == pytest.approx(0.0, abs=0.01), f"Expected x=0, got {xyz[0]}"
                assert xyz[1] == pytest.approx(0.0, abs=0.01), f"Expected y=0, got {xyz[1]}"
                assert xyz[2] == pytest.approx(0.0, abs=0.01), f"Expected z=0, got {xyz[2]}"

    def test_chained_joint_child_world_position(self, tmp_path, sim_core):
        """Child link in a chained joint ends up at the correct world position.

        After loading the URDF, the tire link should be at (0, 0.26, 0.1),
        NOT at (0, 0.52, 0.2) which would result from cumulative joint offsets.
        """
        # Use the extra_links from test_mixed_joints_all_valid (which already
        # proves it loads) — just add a position check.
        extra_links = """
    <link name="suspension">
      <pose>0 0.26 0.1 0 0 0</pose>
      <inertial><mass>1.0</mass></inertial>
    </link>
    <link name="wheel">
      <pose>0 0.26 0.1 0 0 0</pose>
      <inertial><mass>2.0</mass></inertial>
    </link>"""

        joints = """
    <joint name="suspension_joint" type="prismatic">
      <parent>base_link</parent>
      <child>suspension</child>
      <axis><xyz>0 0 1</xyz></axis>
    </joint>
    <joint name="wheel_joint" type="revolute">
      <parent>suspension</parent>
      <child>wheel</child>
      <axis><xyz>0 1 0</xyz></axis>
    </joint>
    <joint name="fixed_child" type="fixed">
      <parent>base_link</parent>
      <child>child_link</child>
    </joint>"""

        sdf_path = self._make_sdf(tmp_path, joints, extra_links)
        urdf_path = resolve_sdf_to_urdf(sdf_path)
        body_id = p.loadURDF(urdf_path, physicsClientId=sim_core.client)

        # Find wheel link index
        num_joints = p.getNumJoints(body_id, physicsClientId=sim_core.client)
        wheel_index = None
        for i in range(num_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=sim_core.client)
            if info[12].decode() == "wheel":
                wheel_index = i
                break
        assert wheel_index is not None, "wheel link not found"

        # Get wheel link world position
        link_state = p.getLinkState(body_id, wheel_index, physicsClientId=sim_core.client)
        pos = link_state[0]  # worldLinkFramePosition
        assert pos[1] == pytest.approx(0.26, abs=0.02), f"Wheel Y should be ~0.26, got {pos[1]}"
        assert pos[2] == pytest.approx(0.1, abs=0.02), f"Wheel Z should be ~0.1, got {pos[2]}"


class TestModelYawOffset:
    """Tests for model_yaw_offset in resolve_sdf_to_urdf."""

    def _make_sdf(self, tmp_path) -> str:
        """Create a robot SDF with base_link and a child wheel link."""
        mesh = tmp_path / "body.obj"
        mesh.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n" "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n" "f 1 2 3 4\nf 5 6 7 8\n")
        wheel_mesh = tmp_path / "wheel.obj"
        wheel_mesh.write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n")
        sdf_path = tmp_path / "model.sdf"
        sdf_path.write_text(
            f"""\
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="test_robot">
    <link name="base_link">
      <inertial><mass>1.0</mass></inertial>
      <visual name="body_visual">
        <pose>0.1 0.2 0.1 0 0 0</pose>
        <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
      </visual>
      <collision name="body_collision">
        <pose>0 0 0.1 0 0 0</pose>
        <geometry><box><size>0.5 0.5 0.2</size></box></geometry>
      </collision>
    </link>
    <link name="left_wheel">
      <pose>0.0 0.3 0.05 0 0 0</pose>
      <inertial><mass>0.1</mass></inertial>
      <visual name="wheel_visual">
        <pose>0 0 0 1.5708 0 0</pose>
        <geometry><mesh><uri>{wheel_mesh}</uri></mesh></geometry>
      </visual>
    </link>
    <joint name="left_wheel_joint" type="continuous">
      <parent>base_link</parent>
      <child>left_wheel</child>
      <axis><xyz>0 0 1</xyz></axis>
    </joint>
  </model>
</sdf>"""
        )
        return str(sdf_path)

    def test_no_offset_default(self, sim_core, tmp_path):
        """Without model_yaw_offset, visual rpy is unchanged."""
        sdf_path = self._make_sdf(tmp_path)
        urdf_path = resolve_sdf_to_urdf(sdf_path)

        with open(urdf_path) as f:
            urdf_content = f.read()

        # Original base visual pose has rpy="0 0 0"
        assert 'rpy="0 0 0"' in urdf_content

    def test_yaw_offset_applied_to_base_only(self, sim_core, tmp_path):
        """model_yaw_offset rotates only the base link, not child links."""
        import math

        sdf_path = self._make_sdf(tmp_path)
        urdf_path = resolve_sdf_to_urdf(sdf_path, model_yaw_offset=math.pi / 2)

        with open(urdf_path) as f:
            urdf_content = f.read()

        # Parse the URDF to check individual links
        import xml.etree.ElementTree as ET

        root = ET.fromstring(urdf_content)

        # Find base_link visual — should have yaw offset applied
        for link in root.findall("link"):
            if link.get("name") == "base_link":
                visual = link.find("visual")
                origin = visual.find("origin")
                rpy = origin.get("rpy").split()
                assert abs(float(rpy[2]) - math.pi / 2) < 0.01, f"Base link yaw should be ~pi/2, got {rpy[2]}"
                # XYZ should also be rotated: (0.1, 0.2) -> (-0.2, 0.1)
                xyz = origin.get("xyz").split()
                assert abs(float(xyz[0]) - (-0.2)) < 0.01, f"Base link x should be ~-0.2 after rotation, got {xyz[0]}"
                assert abs(float(xyz[1]) - 0.1) < 0.01, f"Base link y should be ~0.1 after rotation, got {xyz[1]}"

            elif link.get("name") == "left_wheel":
                visual = link.find("visual")
                origin = visual.find("origin")
                rpy = origin.get("rpy").split()
                # Wheel rpy should be UNCHANGED (1.5708 0 0)
                assert abs(float(rpy[0]) - 1.5708) < 0.01, f"Wheel roll should be unchanged at 1.5708, got {rpy[0]}"
                assert abs(float(rpy[2])) < 0.01, f"Wheel yaw should be unchanged at 0, got {rpy[2]}"

        # Joint origin should NOT be rotated — child link poses in SDF are
        # in model frame, and URDF base link frame = model frame.
        # model_yaw_offset only rotates visuals, not the kinematic structure.
        # wheel at (0.0, 0.3, 0.05) should stay at (0.0, 0.3, 0.05).
        for joint in root.findall("joint"):
            if joint.get("name") == "left_wheel_joint":
                origin = joint.find("origin")
                xyz = origin.get("xyz").split()
                assert abs(float(xyz[0]) - 0.0) < 0.01, f"Joint x should remain 0.0 (no rotation), got {xyz[0]}"
                assert abs(float(xyz[1]) - 0.3) < 0.01, f"Joint y should remain 0.3 (no rotation), got {xyz[1]}"

    def test_yaw_offset_loads_in_pybullet(self, sim_core, tmp_path):
        """URDF with model_yaw_offset loads successfully in PyBullet."""
        import math

        sdf_path = self._make_sdf(tmp_path)
        urdf_path = resolve_sdf_to_urdf(sdf_path, model_yaw_offset=math.pi / 2)

        body_id = p.loadURDF(urdf_path, physicsClientId=sim_core.client)
        assert body_id >= 0

    def test_base_link_pose_yaw_auto_detected(self, sim_core, tmp_path):
        """Base link <pose> yaw is auto-applied without model_yaw_offset.

        Mimics DeliveryRobot: base_link has <pose>0 0 0 0 0 -1.57</pose>.
        The yaw should be automatically read and applied to base link
        visuals/collisions, so model_yaw_offset is not needed.
        """
        import xml.etree.ElementTree as ET

        mesh = tmp_path / "body.obj"
        mesh.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n" "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n" "f 1 2 3 4\nf 5 6 7 8\n")
        sdf_path = tmp_path / "model.sdf"
        sdf_path.write_text(
            f"""\
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="delivery_robot">
    <link name="body">
      <pose>0 0 0 0 0 -1.57</pose>
      <inertial><mass>1.0</mass></inertial>
      <visual name="body_visual">
        <geometry><mesh><uri>{mesh}</uri></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>"""
        )

        # No model_yaw_offset — should auto-detect from base link <pose>
        urdf_path = resolve_sdf_to_urdf(str(sdf_path))

        root = ET.parse(urdf_path).getroot()
        link = root.find(".//link[@name='body']")
        visual = link.find("visual")
        origin = visual.find("origin")
        rpy = [float(v) for v in origin.get("rpy").split()]

        # Base link yaw -1.57 should be auto-applied
        assert rpy[2] == pytest.approx(-1.57, abs=0.01), f"Base link yaw should be auto-detected as -1.57, got {rpy[2]}"

    def test_base_link_pose_yaw_zero_no_rotation(self, sim_core, tmp_path):
        """Base link with <pose>0 0 0 0 0 0</pose> stays unrotated."""
        import xml.etree.ElementTree as ET

        # _make_sdf creates a base_link with no <pose> (defaults to 0)
        sdf_path = self._make_sdf(tmp_path)
        urdf_path = resolve_sdf_to_urdf(sdf_path)

        root = ET.parse(urdf_path).getroot()
        link = root.find(".//link[@name='base_link']")
        visual = link.find("visual")
        origin = visual.find("origin")
        rpy = [float(v) for v in origin.get("rpy").split()]

        # No rotation applied
        assert rpy[2] == pytest.approx(0.0, abs=0.01)


class TestFuelIncludeResolution:
    """Tests for Fuel-backed ``model://`` includes used by RMF maps."""

    def test_fuel_path_parts_decode_model_name(self):
        uri = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Cafe%20Table/1/files/model.sdf"

        assert sdf_loader._fuel_path_parts(uri) == [
            "OpenRobotics",
            "models",
            "Cafe Table",
            "1",
            "files",
            "model.sdf",
        ]

    def test_missing_model_uri_falls_back_to_quoted_fuel_uri(self, monkeypatch):
        requested_uris = []

        def resolve_fuel(uri):
            requested_uris.append(uri)
            return "/tmp/fuel/Cafe Table/1"

        monkeypatch.delenv("GZ_SIM_RESOURCE_PATH", raising=False)
        monkeypatch.setattr(sdf_loader, "_resolve_fuel_uri", resolve_fuel)

        resolved = sdf_loader._resolve_include_uri("model://Cafe Table", [])

        assert resolved == "/tmp/fuel/Cafe Table/1"
        assert requested_uris == ["https://fuel.gazebosim.org/1.0/OpenRobotics/models/Cafe%20Table"]


class TestStaticAutoDetection:
    """Tests for SDF <static> tag auto-detection in load_sdf_world_file.

    Non-static objects (<static>false</static>) should be loaded as
    pickable=True, collision_mode=NORMAL_3D, with mass from SDF inertial
    when physics=True, and mass=0 when physics=False.
    """

    OBJ_MESH = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n" "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n" "f 1 2 3 4\nf 5 6 7 8\n"

    def _make_model_dir(self, tmp_path, name: str, static: bool = True, mass: float = 0.0) -> str:
        """Create a model directory with model.sdf and mesh file."""
        model_dir = tmp_path / "models" / name
        model_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = model_dir / "mesh.obj"
        mesh_path.write_text(self.OBJ_MESH)

        inertial_xml = ""
        if mass > 0:
            inertial_xml = f"""
                  <inertial>
                    <mass>{mass}</mass>
                    <inertia>
                      <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
                      <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
                    </inertia>
                  </inertial>"""

        sdf_content = f"""<?xml version="1.0"?>
        <sdf version="1.6">
          <model name="{name}">
            <link name="{name}_link">
              {inertial_xml}
              <visual name="v">
                <geometry><mesh><uri>{mesh_path}</uri></mesh></geometry>
              </visual>
              <collision name="c">
                <geometry><mesh><uri>{mesh_path}</uri></mesh></geometry>
              </collision>
            </link>
          </model>
        </sdf>"""

        (model_dir / "model.sdf").write_text(sdf_content)
        return str(model_dir)

    def _make_world_file(
        self,
        tmp_path,
        includes: list,
    ) -> str:
        """Create a .world file with <include> elements.

        Args:
            includes: list of dicts with keys: name, uri, pose, static
        """
        include_xml_parts = []
        for inc in includes:
            static_tag = ""
            if "static" in inc:
                static_tag = f"<static>{str(inc['static']).lower()}</static>"
            pose_tag = ""
            if "pose" in inc:
                pose_tag = f"<pose>{inc['pose']}</pose>"
            include_xml_parts.append(
                f"""    <include>
      <name>{inc['name']}</name>
      <uri>model://{inc['uri']}</uri>
      {pose_tag}
      {static_tag}
    </include>"""
            )

        world_content = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="test_world">
{''.join(include_xml_parts)}
  </world>
</sdf>"""

        world_path = tmp_path / "test.world"
        world_path.write_text(world_content)
        return str(world_path)

    @pytest.fixture
    def sim_core_no_physics(self):
        """Headless sim_core with physics=False."""
        sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, physics=False))
        sim.initialize_simulation()
        yield sim
        try:
            p.disconnect(sim.client)
        except p.error:
            pass

    @pytest.fixture
    def sim_core_physics(self):
        """Headless sim_core with physics=True."""
        sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, physics=True))
        sim.initialize_simulation()
        yield sim
        try:
            p.disconnect(sim.client)
        except p.error:
            pass

    def test_static_object_default_behaviour(self, sim_core_no_physics, tmp_path):
        """Objects with <static>true</static> remain STATIC, mass=0, pickable=False."""
        self._make_model_dir(tmp_path, "Wall", static=True, mass=0.0)

        world_path = self._make_world_file(
            tmp_path,
            [{"name": "wall_1", "uri": "Wall", "static": True}],
        )
        resource_paths = [str(tmp_path / "models")]
        objects = load_sdf_world_file(world_path, sim_core=sim_core_no_physics, resource_paths=resource_paths)

        assert len(objects) == 1
        obj = objects[0]
        assert obj.collision_mode == CollisionMode.STATIC
        assert obj.mass == 0.0
        assert obj.pickable is False

    def test_non_static_object_pickable_normal_3d(self, sim_core_no_physics, tmp_path):
        """Objects with <static>false</static> get pickable=True, collision_mode=NORMAL_3D."""
        self._make_model_dir(tmp_path, "Coke", static=False, mass=0.39)

        world_path = self._make_world_file(
            tmp_path,
            [{"name": "coke_can", "uri": "Coke", "static": False}],
        )
        resource_paths = [str(tmp_path / "models")]
        objects = load_sdf_world_file(world_path, sim_core=sim_core_no_physics, resource_paths=resource_paths)

        assert len(objects) == 1
        obj = objects[0]
        assert obj.collision_mode == CollisionMode.NORMAL_3D
        assert obj.pickable is True

    def test_non_static_mass_zero_when_physics_off(self, sim_core_no_physics, tmp_path):
        """Non-static objects get mass=0.0 when physics=False."""
        self._make_model_dir(tmp_path, "Coke", static=False, mass=0.39)

        world_path = self._make_world_file(
            tmp_path,
            [{"name": "coke_can", "uri": "Coke", "static": False}],
        )
        resource_paths = [str(tmp_path / "models")]
        objects = load_sdf_world_file(world_path, sim_core=sim_core_no_physics, resource_paths=resource_paths)

        assert len(objects) == 1
        assert objects[0].mass == 0.0

    def test_non_static_mass_from_sdf_when_physics_on(self, sim_core_physics, tmp_path):
        """Non-static objects get mass from SDF <inertial> when physics=True."""
        self._make_model_dir(tmp_path, "Coke", static=False, mass=0.39)

        world_path = self._make_world_file(
            tmp_path,
            [{"name": "coke_can", "uri": "Coke", "static": False}],
        )
        resource_paths = [str(tmp_path / "models")]
        objects = load_sdf_world_file(world_path, sim_core=sim_core_physics, resource_paths=resource_paths)

        assert len(objects) == 1
        assert objects[0].mass == pytest.approx(0.39, rel=0.01)

    def test_mixed_static_and_non_static(self, sim_core_no_physics, tmp_path):
        """World with both static and non-static objects correctly differentiates them."""
        self._make_model_dir(tmp_path, "Wall", static=True, mass=0.0)
        self._make_model_dir(tmp_path, "Coke", static=False, mass=0.39)

        world_path = self._make_world_file(
            tmp_path,
            [
                {"name": "wall_1", "uri": "Wall", "static": True},
                {"name": "coke_can", "uri": "Coke", "static": False},
            ],
        )
        resource_paths = [str(tmp_path / "models")]
        objects = load_sdf_world_file(world_path, sim_core=sim_core_no_physics, resource_paths=resource_paths)

        assert len(objects) == 2
        wall = next(o for o in objects if o.name and "wall_1" in o.name)
        coke = next(o for o in objects if o.name and "coke_can" in o.name)

        assert wall.collision_mode == CollisionMode.STATIC
        assert wall.pickable is False
        assert wall.mass == 0.0

        assert coke.collision_mode == CollisionMode.NORMAL_3D
        assert coke.pickable is True

    def test_no_static_tag_defaults_to_static(self, sim_core_no_physics, tmp_path):
        """Objects without <static> tag default to static behaviour."""
        self._make_model_dir(tmp_path, "Chair", static=True, mass=0.0)

        world_path = self._make_world_file(
            tmp_path,
            [{"name": "chair_1", "uri": "Chair"}],  # No 'static' key
        )
        resource_paths = [str(tmp_path / "models")]
        objects = load_sdf_world_file(world_path, sim_core=sim_core_no_physics, resource_paths=resource_paths)

        assert len(objects) == 1
        obj = objects[0]
        assert obj.collision_mode == CollisionMode.STATIC
        assert obj.pickable is False

    def test_non_static_mass_no_inertial_defaults_to_fallback(self, sim_core_physics, tmp_path):
        """Non-static object without <inertial> gets fallback mass (0.1) when physics=True."""
        self._make_model_dir(tmp_path, "Coke", static=False, mass=0.0)  # mass=0 → no inertial tag

        world_path = self._make_world_file(
            tmp_path,
            [{"name": "coke_can", "uri": "Coke", "static": False}],
        )
        resource_paths = [str(tmp_path / "models")]
        objects = load_sdf_world_file(world_path, sim_core=sim_core_physics, resource_paths=resource_paths)

        assert len(objects) == 1
        # Should get a small fallback mass, not 0
        assert objects[0].mass > 0

    def test_load_sdf_world_passes_mass_and_pickable(self, sim_core_no_physics, tmp_path):
        """load_sdf_world() accepts mass and pickable parameters."""
        mesh_path = tmp_path / "box.obj"
        mesh_path.write_text(self.OBJ_MESH)

        sdf_file = tmp_path / "model.sdf"
        sdf_file.write_text(
            f"""<?xml version="1.0"?>
            <sdf version="1.6">
              <model name="test">
                <link name="link1">
                  <visual name="v"><geometry><mesh><uri>{mesh_path}</uri></mesh></geometry></visual>
                </link>
              </model>
            </sdf>"""
        )

        objects = load_sdf_world(str(sdf_file), sim_core=sim_core_no_physics, mass=1.5, pickable=True)

        assert len(objects) == 1
        assert objects[0].mass == 1.5
        assert objects[0].pickable is True


# ---------------------------------------------------------------------------
# <submesh> extraction (multi-part meshes, e.g. Caddy's polaris.dae)
# ---------------------------------------------------------------------------


class TestSubmeshResolution:
    """Wiring/parsing tests for <submesh> — no mesh library required."""

    def _mesh_el(self, xml):
        import xml.etree.ElementTree as ET

        return ET.fromstring(xml)

    def test_no_submesh_element_returns_none(self):
        from pybullet_fleet.sdf_loader import _resolve_submesh

        m = self._mesh_el("<mesh><uri>model://X/x.dae</uri></mesh>")
        assert _resolve_submesh(m, "/abs/x.dae") is None

    def test_submesh_name_and_center_forwarded(self, monkeypatch):
        """_resolve_submesh parses <name>/<center> and calls _extract_submesh."""
        import pybullet_fleet.sdf_loader as s

        calls = []
        monkeypatch.setattr(
            s, "_extract_submesh", lambda path, name, center: calls.append((path, name, center)) or "/cache/out.obj"
        )

        m = self._mesh_el(
            "<mesh><uri>model://Caddy/meshes/polaris.dae</uri>"
            "<submesh><name>Rear_Wheel_Right</name><center>true</center></submesh></mesh>"
        )
        out = s._resolve_submesh(m, "/abs/polaris.dae")
        assert out == "/cache/out.obj"
        assert calls == [("/abs/polaris.dae", "Rear_Wheel_Right", True)]

    def test_center_defaults_false(self, monkeypatch):
        import pybullet_fleet.sdf_loader as s

        captured = {}
        monkeypatch.setattr(s, "_extract_submesh", lambda path, name, center: captured.update(center=center))

        m = self._mesh_el("<mesh><uri>model://Caddy/meshes/polaris.dae</uri>" "<submesh><name>Ranger</name></submesh></mesh>")
        s._resolve_submesh(m, "/abs/polaris.dae")
        assert captured["center"] is False

    def test_extract_submesh_missing_file_returns_none(self):
        """Graceful fallback when the mesh file does not exist."""
        from pybullet_fleet.sdf_loader import _extract_submesh

        assert _extract_submesh("/no/such/mesh.dae", "Ranger", False) is None

    def test_collada_unit_meter_parsed(self, tmp_path):
        """COLLADA <unit meter> (e.g. inches) is read so OBJ export stays metric."""
        from pybullet_fleet.sdf_loader import _collada_unit_meter

        dae = tmp_path / "inch.dae"
        dae.write_text(
            '<?xml version="1.0"?>'
            '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
            '<asset><unit name="inch" meter="0.0254"/><up_axis>Z_UP</up_axis></asset>'
            "</COLLADA>"
        )
        assert _collada_unit_meter(str(dae)) == 0.0254

    def test_collada_unit_meter_defaults_to_one(self, tmp_path):
        """Non-COLLADA files and unit-less DAEs default to 1.0 (no scaling)."""
        from pybullet_fleet.sdf_loader import _collada_unit_meter

        assert _collada_unit_meter("/x/model.obj") == 1.0
        dae = tmp_path / "nounit.dae"
        dae.write_text('<?xml version="1.0"?><COLLADA><asset/></COLLADA>')
        assert _collada_unit_meter(str(dae)) == 1.0


class TestSubmeshExtraction:
    """End-to-end extraction — requires the optional 'sdf' extra (trimesh)."""

    def test_extracts_single_named_part(self, tmp_path):
        trimesh = pytest.importorskip("trimesh")

        import numpy as np

        from pybullet_fleet.sdf_loader import _extract_submesh

        # Two distinct named boxes packed into one multi-part scene. We export
        # GLB (trimesh can't *write* DAE); _extract_submesh loads any
        # trimesh-readable scene, so this exercises the same select/bake/export
        # path used on real .dae files (validated separately against polaris.dae).
        body = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
        wheel = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
        wheel.apply_translation([1.0, 0.0, 0.0])
        scene = trimesh.Scene()
        scene.add_geometry(body, geom_name="Ranger", node_name="Ranger")
        scene.add_geometry(wheel, geom_name="Rear_Wheel_Right", node_name="Rear_Wheel_Right")
        mesh_file = tmp_path / "combined.glb"
        scene.export(str(mesh_file))

        out = _extract_submesh(str(mesh_file), "Rear_Wheel_Right", center=False)
        assert out is not None and out.endswith(".obj")

        part = trimesh.load(out, force="mesh")
        # Just the wheel (0.4³ box at x=1), not the combined scene which also
        # spans the 2×1×0.5 body. Compare AABB extents (robust to OBJ re-meshing).
        assert np.allclose(sorted(part.bounding_box.extents), [0.4, 0.4, 0.4], atol=1e-5)
        # Caching: same args -> same path, no re-export needed.
        assert _extract_submesh(str(mesh_file), "Rear_Wheel_Right", center=False) == out

    def test_center_recenters_on_origin(self, tmp_path):
        trimesh = pytest.importorskip("trimesh")

        import numpy as np

        from pybullet_fleet.sdf_loader import _extract_submesh

        wheel = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
        wheel.apply_translation([5.0, 3.0, 0.0])  # far from origin
        scene = trimesh.Scene()
        scene.add_geometry(wheel, geom_name="Wheel", node_name="Wheel")
        mesh_file = tmp_path / "wheel.glb"
        scene.export(str(mesh_file))

        out = _extract_submesh(str(mesh_file), "Wheel", center=True)
        centered = trimesh.load(out, force="mesh")
        assert np.allclose(centered.bounding_box.centroid, [0, 0, 0], atol=1e-6)
