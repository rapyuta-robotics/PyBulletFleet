"""
Tests for SimObject class.

This module tests:
- SimObject creation (from_mesh, from_params)
- Shape parameters (ShapeParams) and PyBullet collision geometry verification
- Pose get/set operations
- Static vs dynamic objects
- Collision modes
- Object properties (name, pickable, mass)
"""

import os
from typing import Any

import pybullet as p
import pytest

from pybullet_fleet.sim_object import SimObject, SimObjectSpawnParams, ShapeParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.types import CollisionMode


MESH_PATH = os.path.join(os.path.dirname(__file__), "..", "pybullet_fleet", "mesh", "cube.obj")


_SKIP = object()  # Sentinel to indicate "skip this check"


def assert_object_properties(
    obj,
    *,
    mass=0.0,
    pickable=True,
    name=None,
    position=(0, 0, 0),
    orientation=(0, 0, 0, 1),
    is_kinematic=True,
    collision_mode=CollisionMode.NORMAL_3D,
    collision_shape_type=_SKIP,
    collision_dimensions: Any = _SKIP,
    pos_tolerance=0.01,
    orient_tolerance=0.01,
    dim_tolerance=0.001,
):
    """
    Assert multiple SimObject properties in one call.

    By default, checks against SimObject's default values (mass=0.0, pickable=True,
    name=None, position=(0,0,0), orientation=(0,0,0,1), is_kinematic=True,
    collision_mode=NORMAL_3D).
    Pass _SKIP sentinel to skip a specific check.

    Args:
        obj: SimObject instance to verify
        mass: Expected mass (default 0.0, _SKIP to skip)
        pickable: Expected pickable flag (default True, _SKIP to skip)
        name: Expected name (default None, _SKIP to skip)
        position: Expected (x, y, z) position tuple (default (0,0,0), _SKIP to skip)
        orientation: Expected (x, y, z, w) quaternion (default (0,0,0,1), _SKIP to skip)
        is_kinematic: Expected is_kinematic flag (default True, _SKIP to skip)
        collision_mode: Expected CollisionMode (default NORMAL_3D, _SKIP to skip)
        collision_shape_type: Expected PyBullet geometry type, e.g. p.GEOM_BOX (_SKIP to skip)
        collision_dimensions: Expected collision shape dimensions tuple (_SKIP to skip)
            BOX: (full_x, full_y, full_z) = half_extents * 2
            SPHERE: (radius, radius, radius)
            CYLINDER: (height, radius, 0.0)
        pos_tolerance: Position comparison tolerance (default 0.01)
        orient_tolerance: Orientation comparison tolerance (default 0.01)
        dim_tolerance: Collision dimensions comparison tolerance (default 0.001)
    """
    assert obj is not None
    assert obj.body_id >= 0
    if mass is not _SKIP:
        assert obj.mass == mass, f"Expected mass={mass}, got {obj.mass}"
    if pickable is not _SKIP:
        assert obj.pickable is pickable, f"Expected pickable={pickable}, got {obj.pickable}"
    if name is not _SKIP:
        assert obj.name == name, f"Expected name={name!r}, got {obj.name!r}"
    pose = None
    if position is not _SKIP or orientation is not _SKIP:
        pose = obj.get_pose()
    if position is not _SKIP:
        for axis, (actual, expected) in enumerate(zip(pose.position, position)):
            assert abs(actual - expected) < pos_tolerance, f"Position axis {axis}: expected {expected}, got {actual}"
    if orientation is not _SKIP:
        for axis, (actual, expected) in enumerate(zip(pose.orientation, orientation)):
            assert abs(actual - expected) < orient_tolerance, f"Orientation axis {axis}: expected {expected}, got {actual}"
    if is_kinematic is not _SKIP:
        assert obj.is_kinematic is is_kinematic, f"Expected is_kinematic={is_kinematic}, got {obj.is_kinematic}"
    if collision_mode is not _SKIP:
        assert obj.collision_mode == collision_mode, f"Expected collision_mode={collision_mode}, got {obj.collision_mode}"
    if collision_shape_type is not _SKIP or collision_dimensions is not _SKIP:
        shape_data: Any = p.getCollisionShapeData(obj.body_id, -1)[0]
        if collision_shape_type is not _SKIP:
            actual_type = shape_data[2]
            assert (
                actual_type == collision_shape_type
            ), f"Expected collision shape type={collision_shape_type}, got {actual_type}"
        if collision_dimensions is not _SKIP:
            actual_dims: Any = shape_data[3]
            for i, (actual, expected) in enumerate(zip(actual_dims, collision_dimensions)):
                assert abs(actual - expected) < dim_tolerance, f"Collision dimension[{i}]: expected {expected}, got {actual}"


def create_mesh_object(pose=None, mass=0.0, pickable=True, collision_mode=CollisionMode.NORMAL_3D, name=None):
    """
    Create a SimObject from mesh with sensible defaults for testing.

    Reduces boilerplate for the common case of cube.obj + box collision.
    Pass pose=None (default) to let SimObject.from_mesh() use its own default (origin).
    """
    return SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="mesh", mesh_path=MESH_PATH),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        pose=pose,
        mass=mass,
        pickable=pickable,
        collision_mode=collision_mode,
        name=name,
    )


class TestSimObjectCreationFromMesh:
    """Test SimObject creation from mesh files"""

    def test_create_from_mesh_basic(self, pybullet_env):
        """Test basic mesh object creation"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)
        assert_object_properties(obj, mass=1.0, pickable=True, position=(0, 0, 1), is_kinematic=False)

    def test_create_from_mesh_default_pose(self, pybullet_env):
        """Test that pose=None defaults to origin"""
        obj = create_mesh_object()
        assert_object_properties(obj)

    def test_create_static_object(self, pybullet_env):
        """Test creating static (mass=0) object"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 0), mass=0.0)
        assert_object_properties(obj, mass=0.0, is_kinematic=True)
        # is_static checks CollisionMode.STATIC, not mass
        assert obj.is_static is False

    def test_create_static_collision_mode(self, pybullet_env):
        """Test creating object with STATIC collision mode"""
        obj = create_mesh_object(collision_mode=CollisionMode.STATIC)
        assert_object_properties(obj, collision_mode=CollisionMode.STATIC)
        assert obj.is_static is True

    def test_create_non_pickable(self, pybullet_env):
        """Test creating object with pickable=False"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0, pickable=False)
        assert_object_properties(obj, mass=1.0, pickable=False, position=(0, 0, 1), is_kinematic=False)


class TestSimObjectCreationFromParams:
    """Test ShapeParams and SimObject creation from spawn parameters.

    Verifies that ShapeParams attributes are correctly set AND that
    the shape parameters are properly passed through to PyBullet
    (collision shape type and dimensions).
    """

    def test_create_from_params_box(self, pybullet_env):
        """Test box shape: ShapeParams attributes and PyBullet collision geometry"""
        half_extents = [0.5, 0.5, 0.5]
        collision_shape = ShapeParams(shape_type="box", half_extents=half_extents)
        # Verify ShapeParams attributes
        assert collision_shape.shape_type == "box"
        assert collision_shape.half_extents == half_extents

        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=half_extents, rgba_color=[1, 0, 0, 1]),
            collision_shape=collision_shape,
            initial_pose=Pose.from_xyz(0, 0, 1),
            mass=1.0,
            pickable=True,
        )
        obj = SimObject.from_params(params)
        # BOX dimensions = half_extents * 2
        assert_object_properties(
            obj,
            mass=1.0,
            pickable=True,
            position=(0, 0, 1),
            is_kinematic=False,
            collision_shape_type=p.GEOM_BOX,
            collision_dimensions=(1.0, 1.0, 1.0),
        )

    def test_create_from_params_sphere(self, pybullet_env):
        """Test sphere shape: ShapeParams attributes and PyBullet collision geometry"""
        radius = 0.3
        collision_shape = ShapeParams(shape_type="sphere", radius=radius)
        # Verify ShapeParams attributes
        assert collision_shape.shape_type == "sphere"
        assert collision_shape.radius == radius

        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="sphere", radius=radius, rgba_color=[0, 1, 0, 1]),
            collision_shape=collision_shape,
            initial_pose=Pose.from_xyz(1, 1, 1),
            mass=0.5,
        )
        obj = SimObject.from_params(params)
        # SPHERE dimensions = (radius, radius, radius)
        assert_object_properties(
            obj,
            mass=0.5,
            position=(1, 1, 1),
            is_kinematic=False,
            collision_shape_type=p.GEOM_SPHERE,
            collision_dimensions=(radius, radius, radius),
        )

    def test_create_from_params_cylinder(self, pybullet_env):
        """Test cylinder shape: ShapeParams attributes and PyBullet collision geometry"""
        radius = 0.2
        height = 0.8
        collision_shape = ShapeParams(shape_type="cylinder", radius=radius, height=height)
        # Verify ShapeParams attributes
        assert collision_shape.shape_type == "cylinder"
        assert collision_shape.radius == radius
        assert collision_shape.height == height

        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="cylinder", radius=radius, height=height, rgba_color=[0, 0, 1, 1]),
            collision_shape=collision_shape,
            initial_pose=Pose.from_xyz(0, 0, 0.5),
            mass=2.0,
        )
        obj = SimObject.from_params(params)
        # CYLINDER dimensions = (height, radius, 0.0)
        assert_object_properties(
            obj,
            mass=2.0,
            position=(0, 0, 0.5),
            is_kinematic=False,
            collision_shape_type=p.GEOM_CYLINDER,
            collision_dimensions=(height, radius, 0.0),
        )

    def test_create_from_params_with_name(self, pybullet_env):
        """Test creation with optional name parameter"""
        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
            initial_pose=Pose.from_xyz(0, 0, 1),
            mass=1.0,
            name="TestObject",
        )
        obj = SimObject.from_params(params)
        assert_object_properties(obj, mass=1.0, name="TestObject", position=(0, 0, 1), is_kinematic=False)

    def test_shape_params_mesh(self):
        """Test mesh shape parameters (no PyBullet needed)"""
        shape = ShapeParams(shape_type="mesh", mesh_path="test.obj", mesh_scale=[2.0, 2.0, 2.0])
        assert shape.shape_type == "mesh"
        assert shape.mesh_path == "test.obj"
        assert shape.mesh_scale == [2.0, 2.0, 2.0]

    def test_shape_params_with_frame_pose(self):
        """Test shape with frame offset (no PyBullet needed)"""
        pose = Pose.from_xyz(0.5, 0.5, 0.5)
        shape = ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5], frame_pose=pose)
        assert shape.frame_pose is not None
        assert shape.frame_pose.position[0] == 0.5

    def test_shape_params_rgba_color(self):
        """Test rgba_color attribute on visual shape (no PyBullet needed)"""
        shape = ShapeParams(shape_type="sphere", radius=0.5, rgba_color=[1, 0, 0, 1])
        assert shape.rgba_color == [1, 0, 0, 1]

    def test_from_dict_full(self):
        """from_dict creates ShapeParams from a complete dict."""
        d = {
            "shape_type": "box",
            "half_extents": [0.3, 0.3, 0.1],
            "rgba_color": [1.0, 0.0, 0.0, 1.0],
        }
        shape = ShapeParams.from_dict(d)
        assert shape.shape_type == "box"
        assert shape.half_extents == [0.3, 0.3, 0.1]
        assert shape.rgba_color == [1.0, 0.0, 0.0, 1.0]

    def test_from_dict_defaults(self):
        """from_dict fills defaults for missing keys."""
        shape = ShapeParams.from_dict({"shape_type": "sphere"})
        assert shape.radius == 0.5
        assert shape.mesh_scale == [1.0, 1.0, 1.0]
        assert shape.rgba_color == [0.8, 0.8, 0.8, 1.0]

    def test_from_dict_mesh(self):
        """from_dict handles mesh shape with path and scale."""
        d = {"shape_type": "mesh", "mesh_path": "obj/tote.obj", "mesh_scale": [2, 2, 2]}
        shape = ShapeParams.from_dict(d)
        assert shape.mesh_path == "obj/tote.obj"
        assert shape.mesh_scale == [2, 2, 2]


class TestSimObjectSpawnParamsFromDict:
    """SimObjectSpawnParams.from_dict() creates params from raw YAML dicts."""

    def test_minimal_defaults(self):
        """Minimal dict with only name produces correct defaults."""
        result = SimObjectSpawnParams.from_dict({"name": "box0"})
        assert result.name == "box0"
        assert result.mass == 0.0
        assert result.pickable is True
        assert result.visual_shape is None
        assert result.collision_shape is None
        assert result.initial_pose.x == pytest.approx(0.0)

    def test_pose_and_yaw(self):
        """pose list + yaw are converted to Pose."""
        result = SimObjectSpawnParams.from_dict({"name": "b", "pose": [3.0, 1.0, 0.15], "yaw": 1.57})
        assert result.initial_pose.x == pytest.approx(3.0)
        assert result.initial_pose.y == pytest.approx(1.0)
        assert result.initial_pose.z == pytest.approx(0.15)
        assert result.initial_pose.yaw == pytest.approx(1.57)

    def test_shape_dicts_parsed(self):
        """visual_shape / collision_shape raw dicts become ShapeParams."""
        d = {
            "name": "crate",
            "visual_shape": {"shape_type": "box", "half_extents": [0.5, 0.3, 0.2]},
            "collision_shape": {"shape_type": "sphere", "radius": 0.3},
        }
        result = SimObjectSpawnParams.from_dict(d)
        assert isinstance(result.visual_shape, ShapeParams)
        assert result.visual_shape.shape_type == "box"
        assert isinstance(result.collision_shape, ShapeParams)
        assert result.collision_shape.radius == pytest.approx(0.3)

    def test_missing_name_raises(self):
        """Missing name raises ValueError."""
        with pytest.raises(ValueError, match="missing required 'name'"):
            SimObjectSpawnParams.from_dict({"mass": 1.0})

    def test_all_collision_modes(self):
        """All collision mode strings correctly converted to enum."""
        for mode_str, expected in [
            ("normal_3d", CollisionMode.NORMAL_3D),
            ("static", CollisionMode.STATIC),
            ("disabled", CollisionMode.DISABLED),
        ]:
            result = SimObjectSpawnParams.from_dict({"name": "obj", "collision_mode": mode_str})
            assert result.collision_mode == expected
        # Deprecated, but still parsed.
        with pytest.warns(FutureWarning, match="NORMAL_2D is deprecated"):
            result = SimObjectSpawnParams.from_dict({"name": "obj", "collision_mode": "normal_2d"})
        assert result.collision_mode == CollisionMode.NORMAL_2D

    def test_initial_pose_takes_precedence(self):
        """initial_pose Pose object takes precedence over pose list."""
        result = SimObjectSpawnParams.from_dict(
            {
                "name": "obj",
                "pose": [99, 99, 99],
                "initial_pose": Pose.from_xyz(1.0, 2.0, 3.0),
            }
        )
        assert result.initial_pose.x == pytest.approx(1.0)

    def test_pose_too_short_raises(self):
        """pose list with fewer than 3 elements raises ValueError."""
        with pytest.raises(ValueError, match="[Pp]ose"):
            SimObjectSpawnParams.from_dict({"name": "bad", "pose": [1.0, 2.0]})

    def test_pose_wrong_type_raises(self):
        """pose as a non-sequence (e.g. scalar) raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            SimObjectSpawnParams.from_dict({"name": "bad", "pose": 42})

    def test_visual_shape_bad_type_raises(self):
        """visual_shape as a string raises TypeError."""
        with pytest.raises(TypeError, match="visual_shape"):
            SimObjectSpawnParams.from_dict({"name": "bad", "visual_shape": "box"})

    def test_collision_shape_bad_type_raises(self):
        """collision_shape as a string raises TypeError."""
        with pytest.raises(TypeError, match="collision_shape"):
            SimObjectSpawnParams.from_dict({"name": "bad", "collision_shape": 123})


class TestSimObjectFromDict:
    """SimObject.from_dict creates instance directly from config dict."""

    def test_subclass_returns_subclass(self, pybullet_env):
        """Subclass.from_dict returns an instance of the subclass."""

        class CustomObject(SimObject):
            pass

        config = {
            "name": "custom0",
            "visual_shape": {"shape_type": "box", "half_extents": [0.1, 0.1, 0.1]},
            "collision_shape": {"shape_type": "box", "half_extents": [0.1, 0.1, 0.1]},
        }
        obj = CustomObject.from_dict(config)
        assert isinstance(obj, CustomObject)


class TestSimObjectPose:
    """Test pose get/set operations"""

    def test_get_pose(self, pybullet_env):
        """Test getting object pose"""
        obj = create_mesh_object(pose=Pose.from_xyz(1, 2, 3), mass=1.0)
        assert_object_properties(obj, mass=1.0, position=(1, 2, 3), is_kinematic=False)

    def test_pending_preserve_velocity_default(self, pybullet_env):
        """Two-phase step: _pending_preserve_velocity defaults to True before any buffered write."""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)
        assert obj._pending_preserve_velocity is True

    def test_set_pose(self, pybullet_env):
        """Test setting object pose"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)

        obj.set_pose(Pose.from_xyz(5, 5, 5))
        assert_object_properties(obj, mass=1.0, position=(5, 5, 5), is_kinematic=False)

    def test_set_pose_with_orientation(self, pybullet_env):
        """Test setting pose with specific orientation"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)

        new_pose = Pose.from_euler(3, 3, 3, roll=0, pitch=0, yaw=1.57)
        obj.set_pose(new_pose)
        assert_object_properties(
            obj,
            mass=1.0,
            position=(3, 3, 3),
            orientation=(0, 0, 0.706825, 0.707388),
            is_kinematic=False,
        )

    def test_set_pose_returns_true_on_move(self, pybullet_env):
        """Test that set_pose returns True when position changes"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)
        result = obj.set_pose(Pose.from_xyz(5, 5, 5))
        assert result is True

    def test_set_pose_returns_false_on_same_position(self, pybullet_env):
        """Test that set_pose returns False when position doesn't change"""
        obj = create_mesh_object(pose=Pose.from_xyz(1, 2, 3), mass=1.0)
        result = obj.set_pose(Pose.from_xyz(1, 2, 3))
        assert result is False

    def test_set_pose_static_rejected(self, pybullet_env):
        """Test that set_pose on STATIC object returns False and doesn't move"""
        obj = create_mesh_object(collision_mode=CollisionMode.STATIC)
        result = obj.set_pose(Pose.from_xyz(5, 5, 5))
        assert result is False
        # Object should remain at original position
        assert_object_properties(obj, collision_mode=CollisionMode.STATIC)

    def test_set_pose_kinematic(self, pybullet_env):
        """Test set_pose on kinematic object updates both cache and PyBullet"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        assert obj.is_kinematic is True

        obj.set_pose(Pose.from_xyz(3, 4, 5))
        # Verify cache (get_pose returns cached value for kinematic)
        assert_object_properties(obj, position=(3, 4, 5))
        # Verify PyBullet state directly (bypassing cache)
        pybullet_pos, _ = p.getBasePositionAndOrientation(obj.body_id)
        assert abs(pybullet_pos[0] - 3) < 0.01
        assert abs(pybullet_pos[1] - 4) < 0.01
        assert abs(pybullet_pos[2] - 5) < 0.01

    def test_set_pose_returns_true_on_orientation_change(self, pybullet_env):
        """Test that set_pose returns True when only orientation changes"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)
        new_pose = Pose.from_euler(0, 0, 1, roll=0, pitch=0, yaw=1.57)
        result = obj.set_pose(new_pose)
        assert result is True

    def test_get_pose_kinematic_returns_cache(self, pybullet_env):
        """Test that kinematic get_pose returns cached value, ignoring direct PyBullet changes.

        Kinematic objects (mass=0) cache their pose and never query PyBullet.
        Even if PyBullet state is changed directly, get_pose() returns the cache.
        """
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1))
        assert obj.is_kinematic is True

        # Directly manipulate PyBullet state (bypass set_pose)
        p.resetBasePositionAndOrientation(obj.body_id, [9, 9, 9], [0, 0, 0, 1])

        # get_pose should return CACHED value, not the PyBullet state
        assert_object_properties(obj, position=(0, 0, 1))
        # Confirm PyBullet state is actually different
        pybullet_pos, _ = p.getBasePositionAndOrientation(obj.body_id)
        assert abs(pybullet_pos[0] - 9) < 0.01

    def test_get_pose_dynamic_queries_pybullet(self, pybullet_env):
        """Test that dynamic get_pose queries PyBullet, reflecting direct changes.

        Dynamic objects (mass>0, sim_core=None) always query PyBullet for current pose.
        Direct PyBullet state changes are immediately visible via get_pose().
        """
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)
        assert obj.is_kinematic is False

        # Directly manipulate PyBullet state (bypass set_pose)
        p.resetBasePositionAndOrientation(obj.body_id, [9, 9, 9], [0, 0, 0, 1])

        # get_pose should return the UPDATED PyBullet state
        assert_object_properties(obj, mass=1.0, position=(9, 9, 9), is_kinematic=False)


class TestSimObjectAttachDetach:
    """Test attach/detach operations (sim_core not required)"""

    def test_attach_basic(self, pybullet_env):
        """Test basic attach: object is added to attached list"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        result = parent.attach_object(child)
        assert result is True
        assert child in parent.get_attached_objects()
        assert child.is_attached() is True

    def test_attach_non_pickable_rejected(self, pybullet_env):
        """Test that attach is rejected for non-pickable objects"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0), pickable=False)

        result = parent.attach_object(child)
        assert result is False
        assert child not in parent.get_attached_objects()
        assert child.is_attached() is False

    def test_attach_already_attached_rejected(self, pybullet_env):
        """Test that attaching an already-attached object is rejected"""
        parent1 = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent2 = create_mesh_object(pose=Pose.from_xyz(2, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        assert parent1.attach_object(child) is True
        # Try to attach to a different parent
        assert parent2.attach_object(child) is False
        assert child in parent1.get_attached_objects()
        assert child not in parent2.get_attached_objects()

    def test_attach_duplicate_rejected(self, pybullet_env):
        """Test that attaching the same object twice to the same parent is rejected"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        assert parent.attach_object(child) is True
        # Detach first, then try duplicate via internal state
        # Actually, since is_attached() returns True, second attach is rejected
        assert parent.attach_object(child) is False
        assert len(parent.get_attached_objects()) == 1

    def test_detach_basic(self, pybullet_env):
        """Test basic detach: object is removed from attached list"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        parent.attach_object(child)
        result = parent.detach_object(child)
        assert result is True
        assert child not in parent.get_attached_objects()
        assert child.is_attached() is False

    def test_detach_resets_state(self, pybullet_env):
        """Test that detach resets attachment state on the child"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        parent.attach_object(child, relative_pose=Pose.from_xyz(0.5, 0, 0))
        parent.detach_object(child)

        assert child._attached_to is None
        assert child._attached_link_index == -1
        assert child._constraint_id is None

    def test_detach_not_attached_rejected(self, pybullet_env):
        """Test that detaching a non-attached object returns False"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        result = parent.detach_object(child)
        assert result is False

    def test_attach_follows_parent_set_pose(self, pybullet_env):
        """Test that attached child follows parent on set_pose"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child)
        parent.set_pose(Pose.from_xyz(3, 4, 5))

        # Child should follow to same position (default relative_pose is zero offset)
        assert_object_properties(child, position=(3, 4, 5))

    def test_attach_with_relative_pose(self, pybullet_env):
        """Test that attached child maintains relative offset on parent move"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        offset = Pose.from_xyz(1.0, 0, 0)
        parent.attach_object(child, relative_pose=offset)

        parent.set_pose(Pose.from_xyz(5, 5, 5))
        # Child should be at parent position + offset
        assert_object_properties(child, position=(6, 5, 5))

    def test_attach_with_relative_pose_and_rotation(self, pybullet_env):
        """Test that relative offset rotates correctly with parent orientation.

        Child is attached at offset (1, 0, 0) in parent frame.
        When parent rotates yaw=90° (pi/2), the offset rotates from +x to +y.
        Parent at (5, 5, 0) with yaw=90° → child at (5, 6, 0) with same orientation.
        """
        import math

        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        offset = Pose.from_xyz(1.0, 0, 0)
        parent.attach_object(child, relative_pose=offset)

        # Move parent with 90° yaw rotation
        parent.set_pose(Pose.from_euler(5, 5, 0, roll=0, pitch=0, yaw=math.pi / 2))

        # Offset (1,0,0) rotated by yaw=90° becomes (0,1,0)
        # Child position = parent (5,5,0) + rotated offset (0,1,0) = (5,6,0)
        # Child orientation inherits parent's rotation
        assert_object_properties(
            child,
            position=(5, 6, 0),
            orientation=(0, 0, 0.707107, 0.707107),
        )

    def test_attach_with_orientation_offset(self, pybullet_env):
        """Test attach with both position and orientation offset in relative_pose.

        Child is attached at offset (1, 0, 0) with roll=90° in parent frame.
        When parent is at origin with no rotation, child has the offset orientation.
        When parent rotates yaw=90°, child orientation = parent_yaw90 * offset_roll90.
        """
        import math

        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        # Offset: position (1,0,0) + orientation roll=90°
        offset = Pose.from_euler(1.0, 0, 0, roll=math.pi / 2, pitch=0, yaw=0)
        parent.attach_object(child, relative_pose=offset)

        # Parent at origin, no rotation → child gets offset as-is
        parent.set_pose(Pose.from_xyz(0, 0, 0))
        assert_object_properties(
            child,
            position=(1, 0, 0),
            orientation=(0.707107, 0, 0, 0.707107),  # roll=90°
        )

        # Parent at (5,5,0) with yaw=90°
        # Position: (5,5,0) + yaw90°*(1,0,0) = (5,6,0)
        # Orientation: yaw90° * roll90° = (0.5, 0.5, 0.5, 0.5)
        parent.set_pose(Pose.from_euler(5, 5, 0, roll=0, pitch=0, yaw=math.pi / 2))
        assert_object_properties(
            child,
            position=(5, 6, 0),
            orientation=(0.5, 0.5, 0.5, 0.5),
        )

    def test_attach_detach_reattach(self, pybullet_env):
        """Test that an object can be re-attached after detach"""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        assert parent.attach_object(child) is True
        assert parent.detach_object(child) is True
        assert child.is_attached() is False

        # Re-attach should succeed
        assert parent.attach_object(child) is True
        assert child.is_attached() is True

    def test_attach_keep_world_pose(self, pybullet_env):
        """keep_world_pose=True computes relative offset from current positions."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(2, 3, 0))

        result = parent.attach_object(child, keep_world_pose=True)
        assert result is True

        # Child should stay at (2, 3, 0) — offset auto-computed
        assert_object_properties(child, position=(2, 3, 0))

        # Move parent → child should maintain its relative offset
        parent.set_pose(Pose.from_xyz(10, 10, 0))
        assert_object_properties(child, position=(12, 13, 0))

    def test_attach_keep_world_pose_with_rotation(self, pybullet_env):
        """keep_world_pose preserves world orientation of the child."""
        import math

        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_euler(1, 0, 0, roll=0, pitch=0, yaw=math.pi / 2))

        parent.attach_object(child, keep_world_pose=True)

        # Move parent with yaw=90° — child's world position
        # should be parent's new transform applied to the relative offset
        parent.set_pose(Pose.from_euler(5, 5, 0, roll=0, pitch=0, yaw=math.pi / 2))
        # Offset (1,0,0) rotated by yaw=90° → (0,1,0)
        # Child world position = (5,5,0) + (0,1,0) = (5,6,0)
        child_pos = child.get_pose().position
        assert abs(child_pos[0] - 5.0) < 0.01
        assert abs(child_pos[1] - 6.0) < 0.01
        assert abs(child_pos[2] - 0.0) < 0.01

    def test_attach_keep_world_pose_conflicts_with_relative_pose(self, pybullet_env):
        """keep_world_pose=True and relative_pose cannot both be specified."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(1, 0, 0))

        with pytest.raises(ValueError, match="relative_pose.*keep_world_pose"):
            parent.attach_object(
                child,
                relative_pose=Pose.from_xyz(1, 0, 0),
                keep_world_pose=True,
            )

    # ========================================
    # Chained attachment tests (A → B → C)
    # ========================================

    def test_chained_attach_basic(self, pybullet_env):
        """Test chained attachment: grandparent → parent → child (A → B → C)"""
        grandparent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        assert grandparent.attach_object(parent) is True
        assert parent.attach_object(child) is True

        # Verify attachment state
        assert parent in grandparent.get_attached_objects()
        assert child in parent.get_attached_objects()
        assert child not in grandparent.get_attached_objects()  # Child is NOT directly on grandparent
        assert parent.is_attached() is True
        assert child.is_attached() is True
        assert parent._attached_to is grandparent
        assert child._attached_to is parent

    def test_chained_attach_follows_grandparent_move(self, pybullet_env):
        """Test that moving grandparent cascades to parent and child (zero offsets)"""
        grandparent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        grandparent.attach_object(parent)
        parent.attach_object(child)

        # Move grandparent → both parent and child should follow
        grandparent.set_pose(Pose.from_xyz(10, 20, 30))

        assert_object_properties(parent, position=(10, 20, 30))
        assert_object_properties(child, position=(10, 20, 30))

    def test_chained_attach_with_offsets(self, pybullet_env):
        """Test chained attachment with relative offsets at each level.

        grandparent at (0,0,0)
          └── parent at offset (1, 0, 0)
                └── child at offset (0, 2, 0)

        Move grandparent to (5, 5, 0):
          parent → (5+1, 5, 0) = (6, 5, 0)
          child  → parent_pos + offset = (6, 5+2, 0) = (6, 7, 0)
        """
        grandparent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        grandparent.attach_object(parent, relative_pose=Pose.from_xyz(1, 0, 0))
        parent.attach_object(child, relative_pose=Pose.from_xyz(0, 2, 0))

        grandparent.set_pose(Pose.from_xyz(5, 5, 0))

        assert_object_properties(parent, position=(6, 5, 0))
        assert_object_properties(child, position=(6, 7, 0))

    def test_chained_attach_with_rotation(self, pybullet_env):
        """Test chained attachment with grandparent rotation cascades through offsets.

        grandparent at origin
          └── parent at offset (1, 0, 0)
                └── child at offset (1, 0, 0)

        Grandparent rotates yaw=90° at (0, 0, 0):
          parent offset (1,0,0) rotated by yaw=90° → (0, 1, 0) → parent at (0, 1, 0)
          child offset (1,0,0) rotated by parent's yaw=90° → (0, 1, 0) → child at (0, 2, 0)
        """
        import math

        grandparent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        grandparent.attach_object(parent, relative_pose=Pose.from_xyz(1, 0, 0))
        parent.attach_object(child, relative_pose=Pose.from_xyz(1, 0, 0))

        # Rotate grandparent yaw=90° at origin
        grandparent.set_pose(Pose.from_euler(0, 0, 0, roll=0, pitch=0, yaw=math.pi / 2))

        # parent: (0,0,0) + yaw90°*(1,0,0) = (0, 1, 0), orientation inherits yaw=90°
        assert_object_properties(
            parent,
            position=(0, 1, 0),
            orientation=(0, 0, 0.707107, 0.707107),
        )
        # child: (0,1,0) + yaw90°*(1,0,0) = (0, 1, 0) + (0, 1, 0) = (0, 2, 0), orientation inherits yaw=90°
        assert_object_properties(
            child,
            position=(0, 2, 0),
            orientation=(0, 0, 0.707107, 0.707107),
        )

    def test_chained_attach_detach_middle(self, pybullet_env):
        """Test detaching the middle object (parent) from grandparent.

        After detach, moving grandparent should NOT affect parent or child.
        But parent → child link is still active.
        """
        grandparent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        grandparent.attach_object(parent)
        parent.attach_object(child)

        # Detach parent from grandparent
        assert grandparent.detach_object(parent) is True
        assert parent.is_attached() is False
        assert child.is_attached() is True  # child is still attached to parent

        # Move grandparent — parent and child should NOT follow
        grandparent.set_pose(Pose.from_xyz(100, 0, 0))
        assert_object_properties(parent, position=(0, 0, 0))
        assert_object_properties(child, position=(0, 0, 0))

        # Move parent — child should still follow
        parent.set_pose(Pose.from_xyz(5, 5, 0))
        assert_object_properties(child, position=(5, 5, 0))

    def test_chained_attach_detach_leaf(self, pybullet_env):
        """Test detaching the leaf object (child) from parent.

        grandparent → parent link is still active.
        child is now independent.
        """
        grandparent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        grandparent.attach_object(parent)
        parent.attach_object(child)

        # Detach child from parent
        assert parent.detach_object(child) is True
        assert child.is_attached() is False
        assert parent.is_attached() is True

        # Move grandparent — parent follows, child does NOT
        grandparent.set_pose(Pose.from_xyz(10, 10, 0))
        assert_object_properties(parent, position=(10, 10, 0))
        assert_object_properties(child, position=(0, 0, 0))  # Unchanged

    def test_chained_attach_three_levels_with_offsets_and_rotation(self, pybullet_env):
        """Test 3-level chain with offsets and rotation for correctness.

        grandparent at (5, 0, 0) with yaw=90°
          └── parent at offset (2, 0, 0) → world pos = (5, 0, 0) + yaw90°*(2, 0, 0) = (5, 2, 0)
                └── child at offset (0, 0, 3) → world pos = (5, 2, 0) + yaw90°*(0, 0, 3) = (5, 2, 3)

        Note: z-offset is unaffected by yaw rotation.
        """
        import math

        grandparent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        grandparent.attach_object(parent, relative_pose=Pose.from_xyz(2, 0, 0))
        parent.attach_object(child, relative_pose=Pose.from_xyz(0, 0, 3))

        grandparent.set_pose(Pose.from_euler(5, 0, 0, roll=0, pitch=0, yaw=math.pi / 2))

        # parent: (5,0,0) + yaw90°*(2,0,0) = (5, 2, 0), orientation inherits yaw=90°
        assert_object_properties(
            parent,
            position=(5, 2, 0),
            orientation=(0, 0, 0.707107, 0.707107),
        )
        # child: (5,2,0) + yaw90°*(0,0,3) = (5, 2, 3), orientation inherits yaw=90°
        assert_object_properties(
            child,
            position=(5, 2, 3),
            orientation=(0, 0, 0.707107, 0.707107),
        )

    def test_chained_attach_multiple_children_on_same_parent(self, pybullet_env):
        """Test parent with multiple children, one of which also has a child.

        parent at origin
          ├── child_a at offset (1, 0, 0)
          └── child_b at offset (0, 1, 0)
                └── grandchild at offset (0, 0, 1)

        Move parent to (10, 10, 10):
          child_a → (11, 10, 10)
          child_b → (10, 11, 10)
          grandchild → (10, 11, 11)
        """
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child_a = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child_b = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        grandchild = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child_a, relative_pose=Pose.from_xyz(1, 0, 0))
        parent.attach_object(child_b, relative_pose=Pose.from_xyz(0, 1, 0))
        child_b.attach_object(grandchild, relative_pose=Pose.from_xyz(0, 0, 1))

        parent.set_pose(Pose.from_xyz(10, 10, 10))

        assert_object_properties(child_a, position=(11, 10, 10))
        assert_object_properties(child_b, position=(10, 11, 10))
        assert_object_properties(grandchild, position=(10, 11, 11))

    def test_chained_attach_reattach_leaf_to_different_parent(self, pybullet_env):
        """Test detaching leaf and reattaching to a different parent.

        Initial: A → B → C
        Detach C from B, attach C to A directly.
        Then A → B (offset 1,0,0) and A → C (offset 0,1,0).
        """
        a = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        b = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        c = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        a.attach_object(b, relative_pose=Pose.from_xyz(1, 0, 0))
        b.attach_object(c, relative_pose=Pose.from_xyz(0, 1, 0))

        # Detach C from B, attach C directly to A
        assert b.detach_object(c) is True
        assert a.attach_object(c, relative_pose=Pose.from_xyz(0, 1, 0)) is True

        # Move A → B and C both follow A, not B
        a.set_pose(Pose.from_xyz(5, 5, 0))
        assert_object_properties(b, position=(6, 5, 0))  # A(5,5,0) + (1,0,0)
        assert_object_properties(c, position=(5, 6, 0))  # A(5,5,0) + (0,1,0)

    def test_chained_attach_four_levels(self, pybullet_env):
        """Test 4-level deep chain: A → B → C → D with offsets.

        A at (0,0,0)
          └── B at offset (1, 0, 0)
                └── C at offset (1, 0, 0)
                      └── D at offset (1, 0, 0)

        Move A to (10, 0, 0):
          B → (11, 0, 0), C → (12, 0, 0), D → (13, 0, 0)
        """
        a = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        b = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        c = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        d = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        a.attach_object(b, relative_pose=Pose.from_xyz(1, 0, 0))
        b.attach_object(c, relative_pose=Pose.from_xyz(1, 0, 0))
        c.attach_object(d, relative_pose=Pose.from_xyz(1, 0, 0))

        a.set_pose(Pose.from_xyz(10, 0, 0))

        assert_object_properties(b, position=(11, 0, 0))
        assert_object_properties(c, position=(12, 0, 0))
        assert_object_properties(d, position=(13, 0, 0))

    # ------------------------------------------------------------------
    # Immediate teleport on attach / detach
    # ------------------------------------------------------------------

    def test_attach_teleports_child_immediately(self, pybullet_env):
        """attach_object() should immediately teleport child to the correct
        world position based on parent pose + relative_pose."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(99, 99, 99))  # far away

        offset = Pose.from_xyz(1.0, 0, 0)
        parent.attach_object(child, relative_pose=offset)

        # Child should be at parent(0,0,0) + offset(1,0,0) = (1,0,0)
        # immediately — no set_pose on parent needed.
        assert_object_properties(child, position=(1, 0, 0))

    def test_attach_teleports_child_with_parent_offset(self, pybullet_env):
        """Teleport accounts for parent's current position."""
        parent = create_mesh_object(pose=Pose.from_xyz(5, 5, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        offset = Pose.from_xyz(2.0, 0, 0)
        parent.attach_object(child, relative_pose=offset)

        # 5+2=7
        assert_object_properties(child, position=(7, 5, 0))

    def test_attach_teleports_child_with_parent_rotation(self, pybullet_env):
        """Teleport accounts for parent's orientation."""
        import math

        parent = create_mesh_object(pose=Pose.from_euler(5, 5, 0, yaw=math.pi / 2))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        offset = Pose.from_xyz(1.0, 0, 0)
        parent.attach_object(child, relative_pose=offset)

        # yaw 90° rotates (1,0,0) → (0,1,0)
        # parent(5,5,0) + (0,1,0) = (5,6,0)
        # child orientation inherits parent's yaw=90°
        assert_object_properties(
            child,
            position=(5, 6, 0),
            orientation=(0, 0, 0.707107, 0.707107),
        )

    def test_attach_zero_offset_teleports_to_parent(self, pybullet_env):
        """Default zero offset teleports child to parent origin."""
        parent = create_mesh_object(pose=Pose.from_xyz(3, 4, 5))
        child = create_mesh_object(pose=Pose.from_xyz(99, 99, 99))

        parent.attach_object(child)  # default relative_pose = zero offset
        assert_object_properties(child, position=(3, 4, 5))

    def test_attach_keep_world_pose_no_teleport(self, pybullet_env):
        """keep_world_pose=True: child stays in place (no unexpected teleport)."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(2, 3, 0))

        parent.attach_object(child, keep_world_pose=True)
        # Child should remain at (2,3,0)
        assert_object_properties(child, position=(2, 3, 0))

    def test_detach_with_drop_pose(self, pybullet_env):
        """detach_object(obj, drop_pose=...) teleports child to drop_pose."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child)
        result = parent.detach_object(child, drop_pose=Pose.from_xyz(10, 20, 0.1))
        assert result is True
        assert_object_properties(child, position=(10, 20, 0.1))

    def test_detach_with_drop_pose_orientation(self, pybullet_env):
        """drop_pose orientation is applied to the detached object."""
        import math

        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child)
        drop = Pose.from_euler(5, 5, 0, yaw=math.pi / 2)
        parent.detach_object(child, drop_pose=drop)
        assert_object_properties(
            child,
            position=(5, 5, 0),
            orientation=(0, 0, 0.707107, 0.707107),
        )

    def test_detach_without_drop_pose_keeps_position(self, pybullet_env):
        """Backward compat: no drop_pose means child stays where it was."""
        parent = create_mesh_object(pose=Pose.from_xyz(5, 5, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child, relative_pose=Pose.from_xyz(1, 0, 0))
        # After attach, child at (6,5,0) due to teleport
        parent.detach_object(child)
        # Should stay at (6,5,0) — no teleport
        assert_object_properties(child, position=(6, 5, 0))

    def test_detach_with_drop_relative_pose(self, pybullet_env):
        """drop_relative_pose offsets from current (pre-detach) object position."""
        parent = create_mesh_object(pose=Pose.from_xyz(5, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child, relative_pose=Pose.from_xyz(1, 0, 0))
        # child is now at (6, 0, 0) due to immediate teleport

        # Detach with relative offset (0, 0, -0.5) → final = (6, 0, -0.5)
        result = parent.detach_object(child, drop_relative_pose=Pose.from_xyz(0, 0, -0.5))
        assert result is True
        assert_object_properties(child, position=(6, 0, -0.5))

    def test_detach_with_drop_relative_pose_rotation(self, pybullet_env):
        """drop_relative_pose applies rotation relative to current object orientation."""
        import math

        parent = create_mesh_object(pose=Pose.from_euler(0, 0, 0, yaw=math.pi / 2))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child)
        # child is at (0,0,0) with yaw=90° (inherited from parent via teleport)

        # Relative offset (1,0,0) rotated by child's yaw=90° → (0,1,0)
        result = parent.detach_object(child, drop_relative_pose=Pose.from_xyz(1, 0, 0))
        assert result is True
        assert_object_properties(
            child,
            position=(0, 1, 0),
            orientation=(0, 0, 0.707107, 0.707107),  # retains yaw=90°
        )

    def test_detach_drop_pose_and_drop_relative_pose_conflict(self, pybullet_env):
        """Cannot specify both drop_pose and drop_relative_pose."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        child = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))

        parent.attach_object(child)
        with pytest.raises(ValueError, match="drop_pose.*drop_relative_pose"):
            parent.detach_object(
                child,
                drop_pose=Pose.from_xyz(1, 0, 0),
                drop_relative_pose=Pose.from_xyz(0, 1, 0),
            )


class TestFindNearestPickable:
    """Test SimObject.find_nearest_pickable() utility method."""

    @pytest.fixture
    def sim_env(self, pybullet_env):
        """Create a MockSimCore and register objects."""
        from tests.conftest import MockSimCore

        sim = MockSimCore()
        sim._client = pybullet_env
        return sim

    def test_finds_nearest(self, sim_env):
        """Returns the closest pickable, unattached object."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent.sim_core = sim_env
        sim_env.sim_objects.append(parent)

        far_obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(3, 0, 0),
            pickable=True,
        )
        sim_env.sim_objects.append(far_obj)

        near_obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(0.5, 0, 0),
            pickable=True,
        )
        sim_env.sim_objects.append(near_obj)

        result = parent.find_nearest_pickable(search_radius=5.0)
        assert result is near_obj

    def test_excludes_self(self, sim_env):
        """Doesn't return self even if pickable."""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        obj.sim_core = sim_env
        sim_env.sim_objects.append(obj)

        result = obj.find_nearest_pickable(search_radius=5.0)
        assert result is None

    def test_excludes_attached(self, sim_env):
        """Doesn't return objects already attached to something."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent.sim_core = sim_env
        sim_env.sim_objects.append(parent)

        child = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(0.5, 0, 0),
            pickable=True,
        )
        sim_env.sim_objects.append(child)

        # Attach child to something else
        other = create_mesh_object(pose=Pose.from_xyz(10, 10, 0))
        other.attach_object(child)

        result = parent.find_nearest_pickable(search_radius=5.0)
        assert result is None

    def test_excludes_non_pickable(self, sim_env):
        """Doesn't return non-pickable objects."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent.sim_core = sim_env
        sim_env.sim_objects.append(parent)

        wall = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(0.5, 0, 0),
            pickable=False,
        )
        sim_env.sim_objects.append(wall)

        result = parent.find_nearest_pickable(search_radius=5.0)
        assert result is None

    def test_respects_radius(self, sim_env):
        """Returns None if nothing within search_radius."""
        parent = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        parent.sim_core = sim_env
        sim_env.sim_objects.append(parent)

        far_obj = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(10, 0, 0),
            pickable=True,
        )
        sim_env.sim_objects.append(far_obj)

        result = parent.find_nearest_pickable(search_radius=1.0)
        assert result is None

    def test_no_sim_core_returns_none(self, pybullet_env):
        """Returns None if sim_core is not set."""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 0))
        # obj.sim_core is None by default with create_mesh_object
        result = obj.find_nearest_pickable(search_radius=5.0)
        assert result is None


class TestCollisionMode:
    """Test collision mode settings"""

    def test_collision_mode_normal(self, pybullet_env):
        """Test NORMAL_3D collision mode"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0, collision_mode=CollisionMode.NORMAL_3D)
        assert_object_properties(obj, mass=1.0, collision_mode=CollisionMode.NORMAL_3D, position=(0, 0, 1), is_kinematic=False)

    def test_collision_mode_disabled(self, pybullet_env):
        """Test DISABLED collision mode"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0, collision_mode=CollisionMode.DISABLED)
        assert_object_properties(obj, mass=1.0, collision_mode=CollisionMode.DISABLED, position=(0, 0, 1), is_kinematic=False)

    def test_set_collision_mode(self, pybullet_env):
        """Test changing collision mode after creation"""
        obj = create_mesh_object(pose=Pose.from_xyz(0, 0, 1), mass=1.0)

        obj.set_collision_mode(CollisionMode.DISABLED)
        assert_object_properties(obj, mass=1.0, collision_mode=CollisionMode.DISABLED, position=(0, 0, 1), is_kinematic=False)

        with pytest.warns(FutureWarning, match="NORMAL_2D is deprecated"):
            obj.set_collision_mode(CollisionMode.NORMAL_2D)
        assert_object_properties(obj, mass=1.0, collision_mode=CollisionMode.NORMAL_2D, position=(0, 0, 1), is_kinematic=False)


class TestSimObjectSpawnParams:
    """Test SimObjectSpawnParams dataclass"""

    def test_spawn_params_creation(self):
        """Test creating spawn parameters"""
        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=[1, 1, 1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[1, 1, 1]),
            initial_pose=Pose.from_xyz(0, 0, 0),
            mass=1.0,
            pickable=True,
        )

        assert params.mass == 1.0
        assert params.pickable is True
        assert params.visual_shape.shape_type == "box"
        assert params.collision_shape.shape_type == "box"

    def test_spawn_params_with_name(self):
        """Test spawn params with name"""
        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
            collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
            initial_pose=Pose.from_xyz(0, 0, 1),
            mass=2.0,
            name="TestPallet",
        )

        assert params.name == "TestPallet"
        assert params.mass == 2.0

    def test_spawn_params_minimal(self):
        """Test spawn params with minimal configuration"""
        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="box"),
            collision_shape=None,  # No collision shape
            initial_pose=Pose.from_xyz(0, 0, 0),
            mass=1.0,
        )

        assert params.visual_shape is not None
        assert params.collision_shape is None
        assert params.pickable is True  # Default value


class TestSimObjectMultiple:
    """Test multiple objects interaction"""

    def test_create_multiple_objects(self, pybullet_env):
        """Test creating multiple objects"""
        obj1 = create_mesh_object(pose=Pose.from_xyz(0, 0, 1))
        obj2 = create_mesh_object(pose=Pose.from_xyz(2, 0, 1))
        obj3 = create_mesh_object(pose=Pose.from_xyz(4, 0, 1))

        assert obj1.body_id != obj2.body_id
        assert obj2.body_id != obj3.body_id
        assert obj1.body_id != obj3.body_id

    def test_objects_with_different_shapes(self, pybullet_env):
        """Test objects with different shape types"""
        box_obj = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                initial_pose=Pose.from_xyz(0, 0, 1),
                mass=1.0,
            )
        )

        sphere_obj = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="sphere", radius=0.5),
                collision_shape=ShapeParams(shape_type="sphere", radius=0.5),
                initial_pose=Pose.from_xyz(2, 0, 1),
                mass=1.0,
            )
        )

        mesh_obj = create_mesh_object(pose=Pose.from_xyz(4, 0, 1))

        assert box_obj.body_id >= 0
        assert sphere_obj.body_id >= 0
        assert mesh_obj.body_id >= 0


class TestSimObjectVisualOnly:
    """Test SimObject with visual-only or collision-only shapes"""

    def test_visual_only_object(self, pybullet_env):
        """Test creating object with visual shape but no collision"""
        obj = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                collision_shape=None,
                initial_pose=Pose.from_xyz(0, 0, 1),
                mass=1.0,
            )
        )
        assert_object_properties(obj, mass=1.0, position=(0, 0, 1), is_kinematic=False)

    def test_collision_only_object(self, pybullet_env):
        """Test creating object with collision shape but no visual"""
        obj = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=None,
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                initial_pose=Pose.from_xyz(0, 0, 1),
                mass=1.0,
            )
        )
        assert_object_properties(obj, mass=1.0, position=(0, 0, 1), is_kinematic=False)


class TestSimObjectKinematic:
    """Test kinematic (mass=0) object behavior"""

    def test_kinematic_flag(self, pybullet_env):
        """Test is_kinematic flag is set correctly"""
        # Kinematic object (mass=0)
        kinematic_obj = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                initial_pose=Pose.from_xyz(0, 0, 1),
                mass=0.0,
            )
        )
        assert_object_properties(kinematic_obj, mass=0.0, is_kinematic=True, position=(0, 0, 1))

        # Dynamic object (mass>0)
        dynamic_obj = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5]),
                initial_pose=Pose.from_xyz(2, 0, 1),
                mass=1.5,
            )
        )
        assert_object_properties(dynamic_obj, mass=1.5, is_kinematic=False, position=(2, 0, 1))


class TestSharedShapesCache:
    """Test SimObject._shared_shapes caching behaviour.

    Mesh shapes are expensive to create, so ``create_shared_shapes`` caches
    them by their ShapeParams key.  Primitive shapes (box, sphere, …) are
    cheap and should NOT be cached.
    """

    def test_mesh_shapes_are_cached(self, pybullet_env):
        """Two calls with identical mesh params should return the same shape IDs."""
        mesh_params = ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.2, 0.2, 0.2],
            rgba_color=[1, 0, 0, 1],
        )
        col_params = ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1])

        vid1, cid1 = SimObject.create_shared_shapes(
            visual_shape=mesh_params, collision_shape=col_params, physics_client_id=pybullet_env
        )
        vid2, cid2 = SimObject.create_shared_shapes(
            visual_shape=mesh_params, collision_shape=col_params, physics_client_id=pybullet_env
        )

        # Same IDs → cache was hit
        assert vid1 == vid2
        assert cid1 == cid2
        # Exactly one entry should have been added to the cache
        assert len(SimObject._shared_shapes) == 1

    def test_different_mesh_params_create_new_shapes(self, pybullet_env):
        """Different mesh params should produce different shape IDs."""
        base = dict(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2])
        red = ShapeParams(**base, rgba_color=[1, 0, 0, 1])
        green = ShapeParams(**base, rgba_color=[0, 1, 0, 1])

        vid_r, _ = SimObject.create_shared_shapes(visual_shape=red, physics_client_id=pybullet_env)
        vid_g, _ = SimObject.create_shared_shapes(visual_shape=green, physics_client_id=pybullet_env)

        assert vid_r != vid_g
        assert len(SimObject._shared_shapes) == 2

    def test_primitive_shapes_are_not_cached(self, pybullet_env):
        """Box / sphere / cylinder shapes should not be cached."""
        box = ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.5])

        SimObject.create_shared_shapes(visual_shape=box, collision_shape=box, physics_client_id=pybullet_env)
        SimObject.create_shared_shapes(visual_shape=box, collision_shape=box, physics_client_id=pybullet_env)

        # Cache should remain empty — primitives are never stored
        assert len(SimObject._shared_shapes) == 0

    def test_a_mesh_collision_shape_that_fails_is_cached_visual_only_and_warned_once(self, pybullet_env, monkeypatch, caplog):
        """A failed mesh collision shape gives a visual-only body. The result is
        cached, so later spawns with the same shape reuse it without another
        attempt or warning: the same mesh would fail the same way each time."""
        import logging

        attempts = []

        def failing(shape, client):
            attempts.append(shape.mesh_path)
            raise RuntimeError("cannot build this mesh")

        monkeypatch.setattr(SimObject, "_create_collision_shape", staticmethod(failing))
        visual = ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2])
        collision = ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2])

        with caplog.at_level(logging.WARNING, logger="pybullet_fleet.sim_object"):
            first = SimObject.create_shared_shapes(visual, collision, physics_client_id=pybullet_env)
            second = SimObject.create_shared_shapes(visual, collision, physics_client_id=pybullet_env)

        assert first == second and first[0] >= 0 and first[1] == -1
        assert attempts == [MESH_PATH], "built once, then reused from the cache"
        warnings = [r.getMessage() for r in caplog.records if "Failed to create collision shape" in r.getMessage()]
        assert len(warnings) == 1
        assert "reuse it without another attempt" in warnings[0]


# -------------------------------------------------------------------
# SimObject._needs_update and update() hook
# -------------------------------------------------------------------


class TestSimObjectUpdate:
    """Tests for SimObject._needs_update flag and update() hook."""

    @pytest.fixture
    def sim_core(self, pybullet_env):
        from tests.conftest import MockSimCore

        sc = MockSimCore()
        sc._client = pybullet_env
        return sc

    def test_needs_update_default_false(self, sim_core):
        """SimObject._needs_update defaults to False."""
        params = SimObjectSpawnParams(
            visual_shape=None,
            collision_shape=None,
            initial_pose=Pose.from_xyz(0, 0, 0),
        )
        obj = SimObject.from_params(params, sim_core)
        assert obj._needs_update is False

    def test_update_returns_false_by_default(self, sim_core):
        """SimObject.update() returns False (no-op)."""
        params = SimObjectSpawnParams(
            visual_shape=None,
            collision_shape=None,
            initial_pose=Pose.from_xyz(0, 0, 0),
        )
        obj = SimObject.from_params(params, sim_core)
        assert obj.update(0.01) is False

    def test_subclass_with_needs_update(self, sim_core):
        """Subclass with _needs_update=True gets update() called in unified loop."""
        call_count = 0

        class UpdatingObject(SimObject):
            _wants_update = True  # the class default of _needs_update, as Agent sets it

            def update(self, dt):
                nonlocal call_count
                call_count += 1
                return True

        params = SimObjectSpawnParams(
            visual_shape=None,
            collision_shape=None,
            initial_pose=Pose.from_xyz(0, 0, 0),
        )
        obj = UpdatingObject.from_params(params, sim_core)
        sim_core.tick(5)
        assert call_count == 5
