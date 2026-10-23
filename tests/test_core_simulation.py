"""
Unit tests for MultiRobotSimulationCore.

Tests object lifecycle, simulation step, callback system, and configuration
using a real PyBullet environment (DIRECT mode).
"""

import math
import time
from unittest.mock import Mock

import pytest
import pybullet as p

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.sim_object import SimObject, ShapeParams
from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.types import (
    CollisionMode,
    CollisionDetectionMethod,
    SpatialHashCellSizeMode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_core():
    """Headless kinematics sim with collision enabled."""
    params = SimulationParams(
        gui=False,
        physics=False,
        monitor=False,
        collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
        collision_margin=0.02,
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        ignore_static_collision=False,
        log_level="warning",
    )
    sc = MultiRobotSimulationCore(params)
    sc.set_collision_spatial_hash_cell_size_mode()
    yield sc
    p.disconnect(sc.client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_box(
    sim_core,
    position,
    size=0.25,
    collision_mode=CollisionMode.NORMAL_3D,
    mass=0.0,
    name=None,
):
    """Create a box SimObject registered to sim_core."""
    return SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="box", half_extents=[size, size, size]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[size, size, size]),
        pose=Pose.from_xyz(*position),
        mass=mass,
        sim_core=sim_core,
        collision_mode=collision_mode,
        name=name,
    )


def make_agent(sim_core, position=(0, 0, 0), name=None):
    """Create an Agent registered to sim_core."""
    return Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(*position),
            collision_mode=CollisionMode.NORMAL_3D,
            name=name,
        ),
        sim_core=sim_core,
    )


def obj_cells(sim_core, obj):
    """Return the set of cells that *obj* is registered in."""
    return set(sim_core._cached_object_to_cell[obj.object_id])


class SpyGrid(dict):
    """A dict subclass that records every key passed to .get().

    Replaces sim_core._cached_spatial_grid to observe which cells
    filter_aabb_pairs actually queries: a body's own cells.
    """

    def __init__(self, original):
        super().__init__(original)
        self.queried_keys = set()

    def get(self, key, default=None):
        self.queried_keys.add(key)
        return super().get(key, default)


def was_searched(spy_grid, sim_core, obj):
    """Return True if ANY cell occupied by *obj* was queried during search."""
    return bool(obj_cells(sim_core, obj) & spy_grid.queried_keys)


# ============================================================================
# SimulationParams configuration
# ============================================================================


class TestSimulationParamsCoercion:
    """SimulationParams normalises config-derived values into proper types."""

    def test_collision_detection_method_string_coerced_to_enum(self):
        """A string (e.g. from YAML) is coerced to the CollisionDetectionMethod enum.

        Regression: when left as a plain str, the ``== CollisionDetectionMethod.X``
        checks in check_collisions() never match, silently disabling collision detection.
        """
        params = SimulationParams(physics=False, collision_detection_method="closest_points")
        assert params.collision_detection_method is CollisionDetectionMethod.CLOSEST_POINTS

    def test_collision_detection_method_enum_passthrough(self):
        params = SimulationParams(collision_detection_method=CollisionDetectionMethod.CONTACT_POINTS)
        assert params.collision_detection_method is CollisionDetectionMethod.CONTACT_POINTS

    def test_collision_detection_method_none_auto_selected(self):
        assert SimulationParams(physics=False).collision_detection_method is CollisionDetectionMethod.CLOSEST_POINTS
        assert SimulationParams(physics=True).collision_detection_method is CollisionDetectionMethod.CONTACT_POINTS

    def test_invalid_collision_detection_method_falls_back_to_default(self):
        """An unknown string warns and falls back to the physics-based default."""
        params = SimulationParams(physics=False, collision_detection_method="not_a_method")
        assert params.collision_detection_method is CollisionDetectionMethod.CLOSEST_POINTS
        params_phys = SimulationParams(physics=True, collision_detection_method="not_a_method")
        assert params_phys.collision_detection_method is CollisionDetectionMethod.CONTACT_POINTS


# ============================================================================
# Object Lifecycle
# ============================================================================


class TestObjectLifecycle:
    """Test add_object / remove_object with real sim_core caches."""

    def test_add_object_registers_in_dict(self, sim_core):
        obj = make_box(sim_core, [0, 0, 0])
        assert obj.object_id in sim_core._sim_objects_dict
        assert obj in sim_core.sim_objects

    def test_add_object_sets_collision_mode_cache(self, sim_core):
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        assert sim_core._cached_collision_modes[obj.object_id] == CollisionMode.NORMAL_3D

    def test_add_object_registers_aabb(self, sim_core):
        obj = make_box(sim_core, [0, 0, 0])
        assert obj.object_id in sim_core._cached_aabbs_dict

    def test_add_object_registers_spatial_grid_cell(self, sim_core):
        obj = make_box(sim_core, [0, 0, 0])
        assert obj.object_id in sim_core._cached_object_to_cell
        cells = sim_core._cached_object_to_cell[obj.object_id]
        assert len(cells) >= 1

    def test_add_agent_registers_in_agents_list(self, sim_core):
        agent = make_agent(sim_core)
        assert agent in sim_core.agents
        assert agent in sim_core.sim_objects

    def test_name_lookup_preserves_duplicates_and_requires_unique_results(self, sim_core):
        robot = make_agent(sim_core, name="shared")
        box_a = make_box(sim_core, [0, 0, 0], name="shared")
        box_b = make_box(sim_core, [2, 0, 0], name="shared")
        unique_box = make_box(sim_core, [4, 0, 0], name="unique-box")

        assert sim_core.find_objects_by_name("shared") == [robot, box_a, box_b]
        assert sim_core.find_agents_by_name("shared") == [robot]
        assert sim_core.get_unique_agent_by_name("shared") is robot
        assert sim_core.get_unique_object_by_name("unique-box") is unique_box
        with pytest.raises(LookupError, match="found 3"):
            sim_core.get_unique_object_by_name("shared")
        with pytest.raises(LookupError, match="found 0"):
            sim_core.get_unique_object_by_name("missing")
        with pytest.raises(LookupError, match="found 0"):
            sim_core.get_unique_agent_by_name("missing")

    def test_add_disabled_object_skips_collision_system(self, sim_core):
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.DISABLED)
        # Collision domain: disabled only
        assert obj.object_id in sim_core._disabled_collision_objects
        assert obj.object_id not in sim_core._static_collision_objects
        assert obj.object_id not in sim_core._dynamic_collision_objects
        assert obj.object_id not in sim_core._cached_aabbs_dict
        # Movement type: DISABLED mass=0 is kinematic (not STATIC, so not skipped)
        assert obj.object_id in sim_core._kinematic_objects
        assert obj.object_id not in sim_core._physics_objects

    def test_add_static_object_registers_in_static_collision_set(self, sim_core):
        """STATIC object should be in _static_collision_objects (collision domain),
        not in _dynamic_collision_objects or _disabled_collision_objects."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        assert obj.object_id in sim_core._static_collision_objects
        assert obj.object_id not in sim_core._dynamic_collision_objects
        assert obj.object_id not in sim_core._disabled_collision_objects

    def test_add_normal_object_registers_in_dynamic_collision_set(self, sim_core):
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        assert obj.object_id in sim_core._dynamic_collision_objects
        assert obj.object_id not in sim_core._static_collision_objects
        assert obj.object_id not in sim_core._disabled_collision_objects

    def test_static_collision_objects_not_managed_by_movement_type(self, sim_core):
        """_register_object_movement_type should NOT touch _static_collision_objects.
        The collision set is managed exclusively by collision-domain methods."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        # Remove from collision set manually, then re-register movement type
        sim_core._static_collision_objects.discard(obj.object_id)
        sim_core._register_object_movement_type(obj)
        # Movement type registration should NOT re-add to collision set
        assert obj.object_id not in sim_core._static_collision_objects

    def test_register_object_movement_type_is_idempotent(self, sim_core):
        """_register_object_movement_type is idempotent.

        Calling it twice (initial add + re-registration on mode change) must
        produce exactly the same state with no duplicates or stale entries.
        """
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id

        # Initially kinematic (mass=0)
        assert oid in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

        # Calling again should be a safe no-op
        sim_core._register_object_movement_type(obj)

        assert oid in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects
        # set size unchanged (no duplicate entries)
        assert sum(1 for x in sim_core._kinematic_objects if x == oid) == 1

    def test_static_object_in_neither_movement_set(self, sim_core):
        """STATIC object should be in neither _kinematic_objects nor _physics_objects."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        oid = obj.object_id
        # Collision domain: static only
        assert oid in sim_core._static_collision_objects
        assert oid not in sim_core._dynamic_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        # Movement type: neither (STATIC)
        assert oid not in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_kinematic_object_in_kinematic_set_only(self, sim_core):
        """Kinematic object (mass=0, non-STATIC) should be in _kinematic_objects,
        not in _physics_objects."""
        obj = make_box(sim_core, [0, 0, 0], mass=0.0, collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id
        # Collision domain: dynamic only
        assert oid in sim_core._dynamic_collision_objects
        assert oid not in sim_core._static_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        # Movement type: kinematic only
        assert oid in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_physics_object_in_physics_set_only(self, sim_core):
        """Physics object (mass > 0) should be in _physics_objects,
        not in _kinematic_objects."""
        obj = make_box(sim_core, [0, 0, 0], mass=1.0, collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id
        # Collision domain: dynamic only
        assert oid in sim_core._dynamic_collision_objects
        assert oid not in sim_core._static_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        # Movement type: physics only
        assert oid in sim_core._physics_objects
        assert oid not in sim_core._kinematic_objects

    def test_transition_normal_to_static_maintains_exclusivity(self, sim_core):
        """NORMAL_3D → STATIC: collision domain and movement type exclusivity."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id
        # Pre-condition
        assert oid in sim_core._dynamic_collision_objects
        assert oid in sim_core._kinematic_objects

        obj.set_collision_mode(CollisionMode.STATIC)

        # Collision domain: static only
        assert oid in sim_core._static_collision_objects
        assert oid not in sim_core._dynamic_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        # Movement type: neither (STATIC)
        assert oid not in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_transition_normal_to_disabled_maintains_exclusivity(self, sim_core):
        """NORMAL_3D → DISABLED: collision domain and movement type exclusivity."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id
        # Pre-condition
        assert oid in sim_core._dynamic_collision_objects
        assert oid in sim_core._kinematic_objects

        obj.set_collision_mode(CollisionMode.DISABLED)

        # Collision domain: disabled only
        assert oid in sim_core._disabled_collision_objects
        assert oid not in sim_core._dynamic_collision_objects
        assert oid not in sim_core._static_collision_objects
        # Movement type: DISABLED mass=0 → kinematic
        assert oid in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_transition_static_to_normal_maintains_exclusivity(self, sim_core):
        """STATIC → NORMAL_3D: collision domain and movement type exclusivity."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        oid = obj.object_id
        # Pre-condition
        assert oid in sim_core._static_collision_objects
        assert oid not in sim_core._kinematic_objects

        obj.set_collision_mode(CollisionMode.NORMAL_3D)

        # Collision domain: dynamic only
        assert oid in sim_core._dynamic_collision_objects
        assert oid not in sim_core._static_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        # Movement type: mass=0 → kinematic
        assert oid in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_transition_static_to_disabled_maintains_exclusivity(self, sim_core):
        """STATIC → DISABLED: collision domain and movement type exclusivity."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        oid = obj.object_id
        # Pre-condition
        assert oid in sim_core._static_collision_objects
        assert oid not in sim_core._kinematic_objects

        obj.set_collision_mode(CollisionMode.DISABLED)

        # Collision domain: disabled only
        assert oid in sim_core._disabled_collision_objects
        assert oid not in sim_core._static_collision_objects
        assert oid not in sim_core._dynamic_collision_objects
        # Movement type: DISABLED mass=0 → kinematic
        assert oid in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_transition_disabled_to_normal_maintains_exclusivity(self, sim_core):
        """DISABLED → NORMAL_3D: collision domain and movement type exclusivity."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.DISABLED)
        oid = obj.object_id
        # Pre-condition
        assert oid in sim_core._disabled_collision_objects
        assert oid in sim_core._kinematic_objects

        obj.set_collision_mode(CollisionMode.NORMAL_3D)

        # Collision domain: dynamic only
        assert oid in sim_core._dynamic_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        assert oid not in sim_core._static_collision_objects
        # Movement type: mass=0 → kinematic
        assert oid in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_transition_disabled_to_static_maintains_exclusivity(self, sim_core):
        """DISABLED → STATIC: collision domain and movement type exclusivity."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.DISABLED)
        oid = obj.object_id
        # Pre-condition
        assert oid in sim_core._disabled_collision_objects
        assert oid in sim_core._kinematic_objects

        obj.set_collision_mode(CollisionMode.STATIC)

        # Collision domain: static only
        assert oid in sim_core._static_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        assert oid not in sim_core._dynamic_collision_objects
        # Movement type: neither (STATIC)
        assert oid not in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_add_object_to_collision_system_classifies_static(self, sim_core):
        """_add_object_to_collision_system should register a STATIC object
        into _static_collision_objects (collision-domain classification)."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        oid = obj.object_id
        # Clear the set to test classification by _add_object_to_collision_system
        sim_core._static_collision_objects.discard(oid)
        assert oid not in sim_core._static_collision_objects

        sim_core._add_object_to_collision_system(oid)
        assert oid in sim_core._static_collision_objects

    def test_add_object_to_collision_system_classifies_normal(self, sim_core):
        """_add_object_to_collision_system should register a NORMAL_3D object
        into _dynamic_collision_objects (collision-domain classification)."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        oid = obj.object_id
        # Clear the set entry to test classification by _add_object_to_collision_system
        sim_core._dynamic_collision_objects.discard(oid)
        assert oid not in sim_core._dynamic_collision_objects

        sim_core._add_object_to_collision_system(oid)
        assert oid in sim_core._dynamic_collision_objects

    def test_add_duplicate_is_noop(self, sim_core):
        obj = make_box(sim_core, [0, 0, 0])
        count_before = len(sim_core.sim_objects)
        sim_core.add_object(obj)  # duplicate
        assert len(sim_core.sim_objects) == count_before

    def test_remove_object_clears_all_caches(self, sim_core):
        """remove_object should clear all caches: dict, collision mode,
        AABB, spatial grid, collision domain sets, and movement type sets."""
        obj = make_box(sim_core, [0, 0, 0])
        obj_id = obj.object_id
        sim_core.remove_object(obj)

        # Core registries
        assert obj_id not in sim_core._sim_objects_dict
        assert obj not in sim_core.sim_objects
        assert obj_id not in sim_core._cached_collision_modes
        assert obj_id not in sim_core._cached_aabbs_dict
        assert obj_id not in sim_core._cached_object_to_cell
        # Collision domain sets (mutually exclusive)
        assert obj_id not in sim_core._dynamic_collision_objects
        assert obj_id not in sim_core._static_collision_objects
        assert obj_id not in sim_core._disabled_collision_objects
        # Movement type sets (mutually exclusive)
        assert obj_id not in sim_core._kinematic_objects
        assert obj_id not in sim_core._physics_objects
        # Movement tracking
        assert obj_id not in sim_core._moved_this_step

    def test_remove_agent_clears_agents_list(self, sim_core):
        """remove_object(agent) should clear agents list and all caches,
        same as remove_object for a normal SimObject."""
        agent = make_agent(sim_core)
        oid = agent.object_id
        sim_core.remove_object(agent)

        # Agent-specific
        assert agent not in sim_core.agents
        # Core registries (same as test_remove_object_clears_all_caches)
        assert agent not in sim_core.sim_objects
        assert oid not in sim_core._sim_objects_dict
        assert oid not in sim_core._cached_collision_modes
        assert oid not in sim_core._cached_aabbs_dict
        assert oid not in sim_core._cached_object_to_cell
        # Collision domain sets
        assert oid not in sim_core._dynamic_collision_objects
        assert oid not in sim_core._static_collision_objects
        assert oid not in sim_core._disabled_collision_objects
        # Movement type sets
        assert oid not in sim_core._kinematic_objects
        assert oid not in sim_core._physics_objects

    def test_remove_nonexistent_is_noop(self, sim_core, caplog):
        """Removing an already-removed object should not raise,
        and should log a warning."""
        import logging

        obj = make_box(sim_core, [0, 0, 0])
        sim_core.remove_object(obj)
        # Second remove should not raise, but should warn
        with caplog.at_level(logging.WARNING, logger="pybullet_fleet.core_simulation"):
            sim_core.remove_object(obj)

        assert any(
            "not found" in r.message for r in caplog.records
        ), "Second remove_object should log a warning about object not found"

    def test_remove_object_clears_robot_original_colors(self, sim_core):
        """remove_object should delete body_id entry from _robot_original_colors."""
        obj = make_box(sim_core, [0, 0, 0])
        body_id = obj.body_id
        # add_object should have stored the color
        assert body_id in sim_core._robot_original_colors

        sim_core.remove_object(obj)
        # After removal the stale entry must be gone
        assert body_id not in sim_core._robot_original_colors

    def test_remove_clears_active_collision_pairs(self, sim_core):
        obj1 = make_box(sim_core, [0, 0, 0])
        obj2 = make_box(sim_core, [0.3, 0, 0])
        sim_core._moved_this_step.add(obj2.object_id)
        sim_core.check_collisions()

        # Verify collision exists before removal
        expected = (
            min(obj1.object_id, obj2.object_id),
            max(obj1.object_id, obj2.object_id),
        )
        assert expected in sim_core._active_collision_pairs, "Pre-condition: objects should be colliding before removal"

        sim_core.remove_object(obj2)
        # No pair should reference removed object
        for pair in sim_core._active_collision_pairs:
            assert obj2.object_id not in pair


# ============================================================================
# SimulationParams
# ============================================================================


class TestSimulationParams:
    """Test SimulationParams construction and from_dict."""

    def test_defaults(self):
        params = SimulationParams(gui=False, monitor=False)
        assert params.physics is False
        assert params.timestep == pytest.approx(0.1)
        assert params.collision_margin == 0.02
        assert params.multi_cell_threshold == 1.5

    def test_auto_select_closest_points_for_kinematics(self):
        params = SimulationParams(gui=False, physics=False, monitor=False)
        assert params.collision_detection_method == CollisionDetectionMethod.CLOSEST_POINTS

    def test_auto_select_contact_points_for_physics(self):
        params = SimulationParams(gui=False, physics=True, monitor=False)
        assert params.collision_detection_method == CollisionDetectionMethod.CONTACT_POINTS

    def test_explicit_method_overrides_auto(self):
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.HYBRID,
        )
        assert params.collision_detection_method == CollisionDetectionMethod.HYBRID

    def test_from_dict(self):
        cfg = {
            "gui": False,
            "physics": True,
            "monitor": False,
            "timestep": 0.01,
            "collision_margin": 0.05,
        }
        params = SimulationParams.from_dict(cfg)
        assert params.physics is True
        assert params.timestep == 0.01
        assert params.collision_margin == 0.05

    def test_lighting_config_alias_from_dict(self):
        params = SimulationParams.from_dict(
            {
                "gui": False,
                "monitor": False,
                "lighting": {"light_position": [8, -8, 15], "enable_controls": True},
            }
        )
        assert params.lighting_config == {"light_position": [8, -8, 15], "enable_controls": True}

    def test_configure_gui_lighting_applies_initial_values(self, sim_core, monkeypatch):
        sim_core._params.gui = True
        sim_core._params.lighting_config = {
            "light_position": [8, -8, 15],
            "shadow_map_world_size": 20,
            "shadow_map_resolution": 2048,
        }
        configure = Mock()
        monkeypatch.setattr("pybullet_fleet.core_simulation.p.configureDebugVisualizer", configure)

        sim_core.configure_gui_lighting()

        configure.assert_called_once_with(
            lightPosition=[8.0, -8.0, 15.0],
            shadowMapWorldSize=20,
            shadowMapResolution=2048,
            physicsClientId=sim_core.client,
        )

    def test_gui_lighting_controls_validate_position_before_creating_sliders(self, sim_core):
        sim_core._params.lighting_config = {
            "light_position": [8, -8],
            "enable_controls": True,
        }

        with pytest.raises(ValueError, match="light_position must be a three-element list"):
            sim_core._install_gui_lighting_controls()

    def test_gui_lighting_defaults_fill_explicit_nulls(self):
        """A null shadow setting takes the default, as a null light_position does."""
        config = {"light_position": None, "shadow_map_world_size": None, "shadow_map_resolution": None}
        values = MultiRobotSimulationCore._validated_gui_lighting_values(config, defaults=True)
        assert values == ([4.0, -4.0, 8.0], 12, 2048)
        assert MultiRobotSimulationCore._validated_gui_lighting_values(config) == (None, None, None)

    def test_from_yaml(self, tmp_path):
        import yaml

        cfg = {
            "gui": False,
            "physics": False,
            "monitor": False,
            "timestep": 0.05,
            "target_rtf": 2.0,
        }
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml.dump(cfg))
        params = SimulationParams.from_config(str(yaml_file))
        assert params.timestep == 0.05
        assert params.target_rtf == 2.0

    def test_enable_floor_default_true(self):
        """Default enable_floor should be True for backward compatibility."""
        params = SimulationParams(gui=False, monitor=False)
        assert params.enable_floor is True

    def test_enable_floor_false(self):
        """enable_floor=False should be accepted."""
        params = SimulationParams(gui=False, monitor=False, enable_floor=False)
        assert params.enable_floor is False

    def test_enable_floor_from_dict(self):
        """enable_floor should round-trip through from_dict."""
        params = SimulationParams.from_dict({"gui": False, "monitor": False, "enable_floor": False})
        assert params.enable_floor is False

    # --- Window size / monitor position params ---

    def test_window_size_defaults(self):
        """Window size params default to 1024x768."""
        params = SimulationParams(gui=False, monitor=False)
        assert params.window_width == 1024
        assert params.window_height == 768

    def test_monitor_size_defaults(self):
        """Monitor size/position defaults: size 680x480, position -1,-1 (WM-managed)."""
        params = SimulationParams(gui=False, monitor=False)
        assert params.monitor_width == 680
        assert params.monitor_height == 480
        assert params.monitor_x == -1
        assert params.monitor_y == -1

    def test_window_size_from_dict(self):
        """Window size params should load from dict."""
        cfg = {
            "gui": False,
            "monitor": False,
            "window_width": 1280,
            "window_height": 720,
        }
        params = SimulationParams.from_dict(cfg)
        assert params.window_width == 1280
        assert params.window_height == 720

    def test_monitor_params_from_dict(self):
        """Monitor size/position should load from dict."""
        cfg = {
            "gui": False,
            "monitor": False,
            "monitor_width": 300,
            "monitor_height": 400,
            "monitor_x": 50,
            "monitor_y": 100,
        }
        params = SimulationParams.from_dict(cfg)
        assert params.monitor_width == 300
        assert params.monitor_height == 400
        assert params.monitor_x == 50
        assert params.monitor_y == 100

    def test_window_size_from_dict_defaults(self):
        """Omitted window/monitor keys should use defaults."""
        cfg = {"gui": False, "monitor": False}
        params = SimulationParams.from_dict(cfg)
        assert params.window_width == 1024
        assert params.window_height == 768
        assert params.monitor_width == 680
        assert params.monitor_height == 480
        assert params.monitor_x == -1
        assert params.monitor_y == -1


class TestDataMonitorParams:
    """Test DataMonitor accepts size/position params."""

    def test_default_geometry(self):
        """Default geometry string omits position when x/y are -1 (WM-managed)."""
        from pybullet_fleet.data_monitor import DataMonitor

        monitor = DataMonitor("Test")
        assert monitor.geometry_string == "680x480"

    def test_custom_geometry(self):
        """Custom size/position should produce correct geometry string."""
        from pybullet_fleet.data_monitor import DataMonitor

        monitor = DataMonitor("Test", width=300, height=400, x=50, y=100)
        assert monitor.geometry_string == "300x400+50+100"

    def test_setup_monitor_passes_params(self):
        """setup_monitor() should pass monitor_* params to DataMonitor."""
        params = SimulationParams(
            gui=False,
            monitor=True,
            enable_monitor_gui=False,
            monitor_width=300,
            monitor_height=400,
            monitor_x=50,
            monitor_y=100,
        )
        sc = MultiRobotSimulationCore(params)
        try:
            assert sc._data_monitor is not None
            assert sc._data_monitor.geometry_string == "300x400+50+100"
        finally:
            p.disconnect(sc.client)


class TestFloorLoading:
    """Test that floor param controls ground-plane loading."""

    def test_enable_floor_loads_body(self):
        """Default enable_floor=True loads plane.urdf (body_id 0)."""
        params = SimulationParams(gui=False, monitor=False, enable_floor=True)
        sc = MultiRobotSimulationCore(params)
        try:
            # plane.urdf is body 0 — getBodyInfo should succeed
            info = p.getBodyInfo(0, physicsClientId=sc.client)
            assert info is not None
        finally:
            p.disconnect(sc.client)

    def test_disable_floor_skips_plane(self):
        """enable_floor=False should NOT load any ground plane body."""
        params = SimulationParams(gui=False, monitor=False, enable_floor=False)
        sc = MultiRobotSimulationCore(params)
        try:
            # No bodies should exist
            num = p.getNumBodies(physicsClientId=sc.client)
            assert num == 0, f"Expected 0 bodies with enable_floor=False, got {num}"
        finally:
            p.disconnect(sc.client)

    def test_reset_respects_disable_floor(self):
        """After reset(), enable_floor=False should still not load plane."""
        params = SimulationParams(gui=False, monitor=False, enable_floor=False)
        sc = MultiRobotSimulationCore(params)
        try:
            sc.reset()
            num = p.getNumBodies(physicsClientId=sc.client)
            assert num == 0, f"Expected 0 bodies after reset with enable_floor=False, got {num}"
        finally:
            p.disconnect(sc.client)


# ============================================================================
# Initialization
# ============================================================================


class TestInitialization:
    """Test MultiRobotSimulationCore initialization."""

    def test_pybullet_connected(self, sim_core):
        assert sim_core.client is not None
        assert p.isConnected(physicsClientId=sim_core.client)

    def test_empty_initial_state(self, sim_core):
        assert len(sim_core.sim_objects) == 0
        assert len(sim_core.agents) == 0
        assert sim_core.step_count == 0
        assert sim_core.sim_time == 0.0

    def test_from_dict(self):
        cfg = {"gui": False, "physics": False, "monitor": False}
        sc = MultiRobotSimulationCore.from_dict(cfg)
        try:
            assert sc.params.gui is False
            assert p.isConnected(physicsClientId=sc.client)
        finally:
            p.disconnect(sc.client)


# ============================================================================
# step_once
# ============================================================================


class TestStepOnce:
    """Test step_once execution."""

    def test_step_increments_count(self, sim_core):
        sim_core.step_once()
        assert sim_core.step_count == 1

    def test_step_advances_sim_time(self, sim_core):
        # The clock advances with the step counter, so outside a step
        # sim_time is the completed time: after 2 steps, 2*dt.
        dt = sim_core.params.timestep
        sim_core.step_once()
        sim_core.step_once()
        assert sim_core.sim_time == pytest.approx(2 * dt, abs=1e-9)

    def test_step_updates_agent(self, sim_core):
        agent = make_agent(sim_core, position=(0, 0, 0))
        goal = Pose.from_xyz(5, 0, 0)
        agent.set_goal_pose(goal)

        for _ in range(10):
            sim_core.step_once()

        pos = agent.get_pose().position
        # Agent should have moved towards goal
        assert pos[0] > 0.0, "Agent should move towards goal"

    def test_step_return_profiling(self, sim_core):
        # Enable time profiling on the sim_core
        sim_core._enable_time_profiling = True
        result = sim_core.step_once(return_profiling=True)
        assert result is not None
        assert "total" in result
        assert "agent_update" in result
        assert "collision_check" in result
        # Top-level float values should be non-negative
        for key, val in result.items():
            if isinstance(val, (int, float)):
                assert val >= 0, f"{key} should be >= 0"
        # collision_breakdown should be present and contain per-phase timings
        assert "collision_breakdown" in result
        breakdown = result["collision_breakdown"]
        assert isinstance(breakdown, dict)
        for phase in ("get_aabbs", "spatial_hashing", "aabb_filtering", "contact_points", "total"):
            assert phase in breakdown
            assert breakdown[phase] >= 0

    def test_last_profiling_exposes_completed_step_snapshot(self, sim_core):
        """Plugins and callbacks can inspect the preceding completed step."""
        sim_core._enable_time_profiling = True
        assert sim_core.last_profiling is None

        sim_core.step_once()
        first = sim_core.last_profiling
        assert first is not None
        assert first["total"] >= 0
        with pytest.raises(TypeError):
            first["total"] = 0

        seen = []

        def inspect_previous(sc, dt):
            seen.append(sc.last_profiling)

        sim_core.register_callback(inspect_previous)
        sim_core.step_once()
        assert seen == [first]

    def test_last_profiling_matches_returned_measurement(self, sim_core):
        """A return_profiling step also updates the public snapshot."""
        result = sim_core.step_once(return_profiling=True)
        assert result is not None
        assert sim_core.last_profiling is not None
        assert sim_core.last_profiling["total"] == result["total"]

    def test_multiple_steps_accumulate(self, sim_core):
        n = 50
        dt = sim_core.params.timestep
        for _ in range(n):
            sim_core.step_once()
        assert sim_core.step_count == n
        # sim_time advances with step_count: after n steps it is n * dt.
        assert sim_core.sim_time == pytest.approx(n * dt, abs=1e-6)

    def test_step_triggers_collision_check(self, sim_core):
        obj1 = make_box(sim_core, [0, 0, 0])
        obj2 = make_box(sim_core, [0.3, 0, 0])
        sim_core._moved_this_step.add(obj1.object_id)

        sim_core.step_once()

        # After step, active_collision_pairs should be populated
        pairs = sim_core.get_active_collision_pairs()
        expected = (
            min(obj1.object_id, obj2.object_id),
            max(obj1.object_id, obj2.object_id),
        )
        assert expected in pairs

    def test_step_adds_moving_agent_to_moved_this_step(self, sim_core):
        """step_once should add a kinematic agent to _moved_this_step
        when agent.update() returns True (agent moved)."""
        agent = make_agent(sim_core, position=(0, 0, 0))
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0))

        # Disable collision check so _moved_this_step is not cleared
        sim_core._collision_check_frequency = 0
        # Clear any _moved_this_step entries from add_object
        sim_core._moved_this_step.clear()

        sim_core.step_once()

        assert (
            agent.object_id in sim_core._moved_this_step
        ), "Moving kinematic agent should be in _moved_this_step after step_once"

    def test_step_does_not_add_stationary_agent_to_moved_this_step(self, sim_core):
        """step_once should NOT add an agent to _moved_this_step
        when the agent has no goal (not moving)."""
        agent = make_agent(sim_core, position=(0, 0, 0))
        # No goal set → agent.update() should return False

        # Disable collision check so _moved_this_step is not cleared
        sim_core._collision_check_frequency = 0
        # Clear any _moved_this_step entries from add_object
        sim_core._moved_this_step.clear()

        sim_core.step_once()

        assert agent.object_id not in sim_core._moved_this_step, "Stationary agent should NOT be in _moved_this_step"

    def test_step_adds_physics_objects_to_moved_this_step(self):
        """step_once should unconditionally add all physics objects
        to _moved_this_step (conservative approach)."""
        # Need physics=True so that mass>0 objects are in _physics_objects
        params = SimulationParams(
            gui=False,
            physics=True,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.CONTACT_POINTS,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=2.0,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        sc.set_collision_spatial_hash_cell_size_mode()
        try:
            obj = make_box(sc, [0, 0, 0], mass=1.0, collision_mode=CollisionMode.NORMAL_3D)
            assert obj.object_id in sc._physics_objects

            # Disable collision check so _moved_this_step is not cleared
            sc._collision_check_frequency = 0
            # Clear any _moved_this_step entries from add_object
            sc._moved_this_step.clear()

            sc.step_once()

            assert obj.object_id in sc._moved_this_step, "Physics object should always be in _moved_this_step after step_once"
        finally:
            p.disconnect(sc.client)

    def test_static_object_in_moved_this_step_on_add(self, sim_core):
        """STATIC object should be in _moved_this_step after add_object
        so it participates in the first collision check."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)

        assert obj.object_id in sim_core._moved_this_step, "STATIC object should be in _moved_this_step after add_object"

    def test_static_object_not_readded_to_moved_this_step_by_step_once(self, sim_core):
        """After initial add, step_once should NOT re-add a STATIC object
        to _moved_this_step (it's not in _physics_objects and not a
        kinematic agent)."""
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)

        # Disable collision check so _moved_this_step is not cleared
        sim_core._collision_check_frequency = 0
        # Clear to isolate what step_once adds
        sim_core._moved_this_step.clear()

        sim_core.step_once()

        assert obj.object_id not in sim_core._moved_this_step, (
            "STATIC object should NOT be re-added to _moved_this_step " "by step_once"
        )

    def test_step_clears_moved_this_step_after_collision_check(self, sim_core):
        """_moved_this_step should be cleared after check_collisions runs."""
        agent = make_agent(sim_core, position=(0, 0, 0))
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0))

        # collision_check_frequency=None means check every step
        assert sim_core._collision_check_frequency is None

        sim_core.step_once()

        # After collision check, _moved_this_step should be cleared
        assert len(sim_core._moved_this_step) == 0, "_moved_this_step should be cleared after collision check"


# ============================================================================
# Callback System
# ============================================================================


class TestCallbackSystem:
    """Test register_callback and execution."""

    def test_callback_called_every_step(self, sim_core):
        calls = []
        sim_core.register_callback(lambda sc, dt: calls.append(dt), frequency=None)
        for _ in range(5):
            sim_core.step_once()
        assert len(calls) == 5

    def test_callback_frequency_control(self, sim_core):
        """Callback at 1 Hz should fire ~once per second of sim_time."""
        calls = []
        sim_core.register_callback(lambda sc, dt: calls.append(sc.sim_time), frequency=1.0)

        dt = sim_core.params.timestep
        # Run 2 seconds of sim time
        steps = int(2.0 / dt)
        for _ in range(steps):
            sim_core.step_once()

        # Should fire approximately 2 times (at t≈1.0 and t≈2.0)
        assert 1 <= len(calls) <= 3, f"Expected ~2 calls at 1Hz over 2s, got {len(calls)}"

    def test_multiple_callbacks(self, sim_core):
        calls_a, calls_b = [], []
        sim_core.register_callback(lambda sc, dt: calls_a.append(1), frequency=None)
        sim_core.register_callback(lambda sc, dt: calls_b.append(1), frequency=None)
        sim_core.step_once()
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_unregister_during_iteration_safe(self, sim_core):
        """Unregistering a callback from within a callback must not skip others."""
        calls = []

        def self_removing_cb(sc, dt):
            """Callback that unregisters itself on first call."""
            sc.unregister_callback(self_removing_cb)

        def tracking_cb(sc, dt):
            calls.append(1)

        # Register self-removing first, tracking second
        sim_core.register_callback(self_removing_cb, frequency=None)
        sim_core.register_callback(tracking_cb, frequency=None)

        # Should not raise and tracking_cb should still fire
        sim_core.step_once()
        assert len(calls) == 1, f"tracking_cb should fire once, got {len(calls)}"


# ============================================================================
# ignore_static_collision
# ============================================================================


class TestIgnoreStaticCollision:
    """Test ignore_static_collision flag behaviour."""

    @pytest.fixture
    def sim_ignore_static(self):
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
            spatial_hash_cell_size=2.0,
            ignore_static_collision=True,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        sc.set_collision_spatial_hash_cell_size_mode()
        yield sc
        p.disconnect(sc.client)

    def test_static_collision_ignored(self, sim_ignore_static):
        static = make_box(sim_ignore_static, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        normal = make_box(sim_ignore_static, [0.3, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        sim_ignore_static._moved_this_step.add(normal.object_id)

        pairs, _ = sim_ignore_static.check_collisions()
        pair = (min(static.object_id, normal.object_id), max(static.object_id, normal.object_id))
        assert pair not in pairs, "Static collision should be ignored"

    def test_static_collision_detected_when_not_ignored(self, sim_core):
        static = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        normal = make_box(sim_core, [0.3, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        sim_core._moved_this_step.add(normal.object_id)

        pairs, _ = sim_core.check_collisions()
        pair = (min(static.object_id, normal.object_id), max(static.object_id, normal.object_id))
        assert pair in pairs, "Static collision should be detected when ignore_static=False"


# ============================================================================
# Collision Check Frequency
# ============================================================================


class TestCollisionCheckFrequency:
    """Test collision_check_frequency parameter."""

    def test_frequency_zero_skips_checks(self):
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            collision_check_frequency=0,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        try:
            obj1 = make_box(sc, [0, 0, 0])
            obj2 = make_box(sc, [0, 0, 0])
            sc._moved_this_step.add(obj1.object_id)

            sc.step_once()

            # With frequency=0, collision check is skipped
            assert len(sc.get_active_collision_pairs()) == 0
        finally:
            p.disconnect(sc.client)

    def test_frequency_none_checks_every_step(self, sim_core):
        obj1 = make_box(sim_core, [0, 0, 0])
        obj2 = make_box(sim_core, [0, 0, 0])
        sim_core._moved_this_step.add(obj1.object_id)

        sim_core.step_once()
        assert len(sim_core.get_active_collision_pairs()) > 0


# ============================================================================
# configure_visualizer
# ============================================================================


class TestSetStructureTransparencyRendering:
    """Test _set_structure_transparency disables rendering during batch update."""

    def test_disables_rendering_during_batch_update(self, sim_core, monkeypatch):
        """Should disable rendering before changing shapes and re-enable after."""
        import pybullet as pb

        sim_core._params.gui = True
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        sim_core._original_visual_colors[(obj.body_id, -1)] = [1.0, 0.0, 0.0, 1.0]

        calls = []
        monkeypatch.setattr(
            pb,
            "configureDebugVisualizer",
            lambda flag, enable, **kw: calls.append(("configDbgVis", flag, enable)),
        )
        monkeypatch.setattr(
            pb,
            "changeVisualShape",
            lambda *args, **kwargs: calls.append(("changeVis",)),
        )

        sim_core._set_structure_transparency(True)

        # Rendering must be disabled (COV_ENABLE_RENDERING=0) before any changeVisualShape
        # and re-enabled (COV_ENABLE_RENDERING=1) after all changeVisualShape calls.
        # After re-enabling rendering, COV_ENABLE_GUI is reasserted to 0 to prevent
        # the side panel from re-appearing.
        assert len(calls) >= 4, f"Expected at least 4 calls, got {len(calls)}"

        # First call: disable rendering
        assert calls[0] == ("configDbgVis", pb.COV_ENABLE_RENDERING, 0), f"First call should disable rendering, got {calls[0]}"
        # Second-to-last call: re-enable rendering
        assert calls[-2] == (
            "configDbgVis",
            pb.COV_ENABLE_RENDERING,
            1,
        ), f"Second-to-last call should re-enable rendering, got {calls[-2]}"
        # Last call: reassert GUI hidden
        assert calls[-1] == (
            "configDbgVis",
            pb.COV_ENABLE_GUI,
            0,
        ), f"Last call should hide GUI panel, got {calls[-1]}"
        # All middle calls should be changeVisualShape
        for c in calls[1:-2]:
            assert c == ("changeVis",), f"Middle calls should be changeVisualShape, got {c}"

    def test_rendering_restored_on_exception(self, sim_core, monkeypatch):
        """Rendering must be re-enabled even if changeVisualShape raises."""
        import pybullet as pb

        sim_core._params.gui = True
        obj = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        sim_core._original_visual_colors[(obj.body_id, -1)] = [1.0, 0.0, 0.0, 1.0]
        # Add a second entry that will cause exception
        sim_core._original_visual_colors[(9999, -1)] = [0.0, 1.0, 0.0, 1.0]
        sim_core._static_collision_objects.add("fake_static_9999")
        # Patch _sim_objects_dict to map fake_static_9999 to body_id 9999
        from unittest.mock import MagicMock

        fake_obj = MagicMock()
        fake_obj.body_id = 9999
        sim_core._sim_objects_dict["fake_static_9999"] = fake_obj

        rendering_states = []

        def fake_configure(flag, enable, **kw):
            if flag == pb.COV_ENABLE_RENDERING:
                rendering_states.append(enable)

        call_count = [0]

        def fake_change(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated GPU error")

        monkeypatch.setattr(pb, "configureDebugVisualizer", fake_configure)
        monkeypatch.setattr(pb, "changeVisualShape", fake_change)

        # Should not raise despite internal error
        sim_core._set_structure_transparency(True)

        # Rendering must be re-enabled (last state = 1)
        assert rendering_states[-1] == 1, f"Rendering should be re-enabled after exception, states: {rendering_states}"


class TestConfigureVisualizerCameraGuard:
    """configure_visualizer must not overwrite an explicit setup_camera call."""

    def test_explicit_setup_camera_not_overwritten(self, sim_core, monkeypatch):
        """If the user called setup_camera() before run_simulation, configure_visualizer must not re-call it."""
        call_count = 0

        def counting_setup_camera(camera_config=None, entity_positions=None):
            nonlocal call_count
            call_count += 1

        sim_core._params.gui = True
        # Populate camera_config so the guard `if self._params.camera_config:` is truthy
        sim_core._params.camera_config = {"camera_mode": "manual", "camera_distance": 20.0}

        # Simulate user explicitly calling setup_camera
        sim_core.setup_camera(camera_config={"camera_mode": "manual", "camera_distance": 2.0})

        # Now monkeypatch setup_camera to count re-calls
        monkeypatch.setattr(sim_core, "setup_camera", counting_setup_camera)

        # configure_visualizer should NOT call setup_camera again
        sim_core.configure_visualizer()

        assert call_count == 0, f"configure_visualizer should not re-call setup_camera, but called it {call_count} time(s)"

    def test_no_explicit_setup_camera_allows_configure_visualizer(self, sim_core, monkeypatch):
        """If user never called setup_camera, configure_visualizer should call it."""
        call_count = 0

        def counting_setup_camera(camera_config=None, entity_positions=None):
            nonlocal call_count
            call_count += 1

        sim_core._params.gui = True
        sim_core._params.camera_config = {"camera_mode": "manual", "camera_distance": 20.0}

        monkeypatch.setattr(sim_core, "setup_camera", counting_setup_camera)

        sim_core.configure_visualizer()

        assert call_count == 1, f"configure_visualizer should call setup_camera once, but called it {call_count} time(s)"


# ---------------------------------------------------------------------------
# setup_camera
# ---------------------------------------------------------------------------


class TestSetupCamera:
    """Test setup_camera behavior."""

    def test_unknown_camera_mode_warns(self, sim_core, caplog, monkeypatch):
        """Unknown camera_mode should log a warning."""
        import logging

        sim_core._params.gui = True

        with caplog.at_level(logging.WARNING, logger="pybullet_fleet.core_simulation"):
            sim_core.setup_camera(camera_config={"camera_mode": "bogus"})

        assert any("bogus" in r.message for r in caplog.records), "Should warn about unknown camera_mode"

    def test_auto_mode_minimum_distance(self, sim_core, monkeypatch):
        """auto mode should enforce a minimum distance when all objects are co-located."""
        import pybullet as pb

        captured_kwargs = {}

        def fake_reset(**kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(pb, "resetDebugVisualizerCamera", fake_reset)
        sim_core._params.gui = True

        # All objects at the same position → extent = [0,0,0]
        sim_core.setup_camera(
            camera_config={"camera_mode": "auto"},
            entity_positions=[[3, 3, 0], [3, 3, 0], [3, 3, 0]],
        )

        assert (
            captured_kwargs.get("cameraDistance", 0) > 0
        ), f"Camera distance should never be 0, got {captured_kwargs.get('cameraDistance')}"


# ============================================================================
# reset
# ============================================================================


class TestReset:
    """Test MultiRobotSimulationCore.reset() clears all state and reloads world."""

    def test_clears_all_objects(self, sim_core):
        """Objects and agents are removed from all tracking structures."""
        make_box(sim_core, [0, 0, 0.5])
        make_box(sim_core, [2, 0, 0.5])
        assert len(sim_core.sim_objects) == 2

        sim_core.reset()

        assert len(sim_core.sim_objects) == 0
        assert len(sim_core.agents) == 0
        assert len(sim_core._sim_objects_dict) == 0

    def test_clears_agents(self, sim_core):
        """Agents list is cleared after reset."""
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_robot.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0),
            ),
            sim_core,
        )
        assert len(sim_core.agents) == 1

        sim_core.reset()
        assert len(sim_core.agents) == 0

    def test_resets_object_id_counter(self, sim_core):
        """_next_object_id resets to 0 so new objects get fresh IDs."""
        make_box(sim_core, [0, 0, 0.5])
        make_box(sim_core, [1, 0, 0.5])
        assert sim_core._next_object_id >= 2

        sim_core.reset()
        assert sim_core._next_object_id == 0

    def test_allows_respawn_after_reset(self, sim_core):
        """New objects can be created after reset."""
        make_box(sim_core, [0, 0, 0.5])
        sim_core.reset()

        obj = make_box(sim_core, [5, 5, 0.5])
        assert len(sim_core.sim_objects) == 1
        assert obj.object_id == 0  # IDs restart from 0

    def test_resets_counters(self, sim_core):
        """Step count, collision count, and sim_time are zeroed."""
        for _ in range(3):
            sim_core.step_once()
        sim_core._collision_count = 50
        assert sim_core.step_count == 3 and sim_core.sim_time > 0.0

        sim_core.reset()

        assert sim_core.step_count == 0
        assert sim_core._collision_count == 0
        assert sim_core.sim_time == 0.0

    def test_clears_collision_caches(self, sim_core):
        """All collision-related caches are emptied."""
        sim_core._cached_collision_modes[42] = CollisionMode.NORMAL_3D
        sim_core._cached_aabbs_dict[42] = ((0, 0, 0), (1, 1, 1))
        sim_core._add_active_collision_pair((1, 2))

        sim_core.reset()

        assert len(sim_core._cached_collision_modes) == 0
        assert len(sim_core._cached_aabbs_dict) == 0
        assert len(sim_core._active_collision_pairs) == 0
        assert len(sim_core._active_pairs_by_object) == 0

    def test_clears_movement_caches(self, sim_core):
        """Movement tracking sets are emptied."""
        sim_core._moved_this_step.add(1)
        sim_core._physics_objects.add(2)
        sim_core._kinematic_objects.add(3)

        sim_core.reset()

        assert len(sim_core._moved_this_step) == 0
        assert len(sim_core._physics_objects) == 0
        assert len(sim_core._kinematic_objects) == 0

    def test_ground_plane_reloaded(self, sim_core):
        """Ground plane is present after reset (body count >= 1)."""
        sim_core.reset()
        # PyBullet should have at least one body (the ground plane)
        num_bodies = p.getNumBodies(physicsClientId=sim_core.client)
        assert num_bodies >= 1

    def test_preserves_callbacks(self, sim_core):
        """Registered callbacks survive reset, and their schedule restarts."""
        calls = []

        def cb(sc, dt):
            calls.append((round(sc.sim_time, 9), round(dt, 9)))

        sim_core.register_callback(cb, frequency=10.0)
        sim_core.initialize_simulation()
        for _ in range(25):
            sim_core.step_once()
        calls.clear()

        sim_core.reset()
        # The first step ends at one timestep: run exactly one interval.
        for _ in range(int(round(0.1 / sim_core.params.timestep))):
            sim_core.step_once()

        assert len(sim_core._callbacks) == 1
        # One interval into the new run, with a dt measured from its start.
        assert calls == [(0.1, 0.1)]

    def test_step_works_after_reset(self, sim_core):
        """step_once() runs cleanly on a fresh world after reset."""
        make_box(sim_core, [0, 0, 0.5])
        sim_core.reset()

        # Should not raise
        sim_core.step_once()
        assert sim_core._step_count == 1

    def test_double_reset(self, sim_core):
        """Calling reset() twice in a row does not crash."""
        make_box(sim_core, [0, 0, 0.5])
        sim_core.reset()
        sim_core.reset()
        assert len(sim_core.sim_objects) == 0
        assert sim_core._next_object_id == 0


# ============================================================================
# initialize_simulation
# ============================================================================


class TestInitializeSimulation:
    """Test initialize_simulation resets all stale state for a fresh run."""

    def test_resets_simulation_paused(self, sim_core):
        """#1: pause state must be reset so step_once doesn't skip."""
        sim_core.pause()
        sim_core.initialize_simulation()
        assert sim_core.is_paused is False

    def test_clears_speed_history(self, sim_core):
        """#2: _speed_history deque must be emptied for accurate speed calculation."""
        import time

        sim_core._speed_history.append((time.time(), 1.0))
        sim_core._speed_history.append((time.time(), 2.0))
        assert len(sim_core._speed_history) > 0

        sim_core.initialize_simulation()
        assert len(sim_core._speed_history) == 0

    def test_resets_callback_schedule(self, sim_core):
        """#3: a restart must not carry the previous run's schedule (stale dt)."""
        called = []

        def cb(sc, dt):
            called.append((round(sc.sim_time, 9), round(dt, 9)))

        sim_core.register_callback(cb, frequency=10.0)
        sim_core.initialize_simulation()
        for _ in range(37):
            sim_core.step_once()
        called.clear()

        sim_core.initialize_simulation()
        # The first step ends at one timestep: run exactly one interval.
        for _ in range(int(round(0.1 / sim_core.params.timestep))):
            sim_core.step_once()
        assert called == [(0.1, 0.1)]

    def test_clears_profiling_stats(self, sim_core):
        """#4: _profiling_stats lists must be emptied to avoid mixing runs."""
        sim_core._profiling_stats["total"].append(1.23)
        sim_core._profiling_stats["agent_update"].append(0.5)

        sim_core.initialize_simulation()
        for key, vals in sim_core._profiling_stats.items():
            assert len(vals) == 0, f"_profiling_stats['{key}'] not cleared"

    def test_resets_memory_stats(self, sim_core):
        """#5: _memory_stats must be reset to initial values."""
        sim_core._memory_stats["count"] = 100
        sim_core._memory_stats["current_max"] = 999.0
        sim_core._memory_stats["peak_max"] = 888.0

        sim_core.initialize_simulation()
        assert sim_core._memory_stats["count"] == 0
        assert sim_core._memory_stats["current_max"] == 0.0
        assert sim_core._memory_stats["current_min"] == float("inf")

    def test_resets_last_ignore_static(self, sim_core):
        """#6: _last_ignore_static must be None for fresh mode-change detection."""
        sim_core._last_ignore_static = True

        sim_core.initialize_simulation()
        assert sim_core._last_ignore_static is None

    def test_counters_reset(self, sim_core):
        """Basic counters should all be zeroed."""
        for _ in range(3):
            sim_core.step_once()
        sim_core._collision_count = 50
        assert sim_core.step_count == 3 and sim_core.sim_time > 0.0

        sim_core.initialize_simulation()
        assert sim_core.step_count == 0
        assert sim_core._collision_count == 0
        assert sim_core.sim_time == 0.0


# ============================================================================
# run_simulation
# ============================================================================


class TestRunSimulation:
    """Test run_simulation loop behavior."""

    @pytest.fixture
    def fast_sim(self):
        """Headless sim with target_rtf=0 (no sleep) for deterministic testing."""
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            target_rtf=0,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        yield sc
        try:
            p.disconnect(sc.client)
        except p.error:
            pass  # already disconnected by run_simulation on connection check

    # -- basic duration -------------------------------------------------------

    def test_stops_after_duration(self, fast_sim):
        """Stops after ceil(duration / timestep) steps when duration=N."""
        fast_sim.run_simulation(duration=0.5)
        # Loop breaks when step_count * timestep >= duration → step_count = ceil(duration / timestep)
        expected = math.ceil(0.5 / fast_sim.params.timestep)
        assert fast_sim.step_count == expected

    def test_uses_params_duration_when_none(self, fast_sim):
        """duration=None falls back to params.duration."""
        fast_sim._params.duration = 0.1
        fast_sim.run_simulation()  # duration=None
        expected = math.ceil(0.1 / fast_sim.params.timestep)
        assert fast_sim.step_count == expected

    def test_initializes_before_run(self, fast_sim):
        """run_simulation resets step_count at the start."""
        fast_sim._step_count = 999
        fast_sim.run_simulation(duration=0.01)
        # step_count should reflect duration, not the stale 999
        expected = math.ceil(0.01 / fast_sim.params.timestep)
        assert fast_sim.step_count == expected

    # -- target_rtf ----------------------------------------------------------------

    def test_speed_zero_runs_without_sleep(self, fast_sim, monkeypatch):
        """target_rtf=0 never calls time.sleep (runs at maximum speed)."""
        sleep_calls = []
        original_sleep = time.sleep

        def spy_sleep(seconds):
            sleep_calls.append(seconds)
            original_sleep(seconds)

        monkeypatch.setattr(time, "sleep", spy_sleep)
        fast_sim._params.target_rtf = 0
        fast_sim.run_simulation(duration=0.05)
        assert len(sleep_calls) == 0, f"target_rtf=0 should never call time.sleep, got {len(sleep_calls)} calls"

    def test_speed_positive_calls_sleep(self, monkeypatch):
        """target_rtf>0 calls time.sleep for real-time synchronization."""
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            target_rtf=1.0,
            timestep=1.0 / 240.0,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        try:
            sleep_calls = []
            original_sleep = time.sleep

            def spy_sleep(seconds):
                sleep_calls.append(seconds)
                # Keep actual sleep minimal to avoid slow tests
                if seconds > 0.001:
                    original_sleep(0.001)

            monkeypatch.setattr(time, "sleep", spy_sleep)
            sc.run_simulation(duration=0.05)
            assert len(sleep_calls) > 0, "target_rtf=1.0 should call time.sleep for sync"
        finally:
            if p.isConnected(physicsClientId=sc.client):
                p.disconnect(sc.client)

    def test_higher_speed_executes_more_steps_per_frame(self, monkeypatch):
        """Both target_rtf=1 and target_rtf=10 produce the same step_count for the same
        sim-time duration, since duration is measured in simulation time."""
        step_counts = {}
        for rtf in (1.0, 10.0):
            params = SimulationParams(
                gui=False,
                physics=False,
                monitor=False,
                target_rtf=rtf,
                log_level="warning",
            )
            sc = MultiRobotSimulationCore(params)
            try:
                # Stub out sleep so the loop runs without real-time delay
                monkeypatch.setattr(time, "sleep", lambda s: None)
                sc.run_simulation(duration=0.1)
                step_counts[rtf] = sc.step_count
            finally:
                if p.isConnected(physicsClientId=sc.client):
                    p.disconnect(sc.client)
        # Same sim-time duration=0.1 → same step_count regardless of target_rtf
        # (duration is measured in simulation time, not wall-clock time)
        expected = math.ceil(0.1 / SimulationParams(gui=False).timestep)
        assert step_counts[1.0] == expected
        assert step_counts[10.0] == expected

    def test_higher_speed_completes_in_less_wall_clock_time(self, monkeypatch):
        """target_rtf=10 should accumulate ~10x less total sleep than target_rtf=1,
        confirming faster wall-clock completion.

        Uses a fake clock where time.time() advances only by sleep amounts,
        giving deterministic control over the real-time sync loop.
        """
        total_sleeps = {}
        for rtf in (1.0, 10.0):
            params = SimulationParams(
                gui=False,
                physics=False,
                monitor=False,
                target_rtf=rtf,
                log_level="warning",
            )
            sc = MultiRobotSimulationCore(params)
            try:
                # Fake clock: only sleep advances time; step_once computation
                # is treated as zero-cost.  This is intentional — both speeds
                # share the same simplification, so the *ratio* of accumulated
                # sleep is a valid proxy for relative wall-clock duration.
                base = time.time()
                clock = base

                def fake_time():
                    return clock

                def fake_sleep(seconds):
                    # nonlocal required: closures can read outer scalars,
                    # but reassignment (+=) needs nonlocal to avoid UnboundLocalError.
                    nonlocal clock
                    clock += seconds

                monkeypatch.setattr(time, "time", fake_time)
                monkeypatch.setattr(time, "monotonic", fake_time)  # loop paces on monotonic
                monkeypatch.setattr(time, "sleep", fake_sleep)
                sc.run_simulation(duration=0.5)
                # clock - base == total sleep accumulated (computation is zero-cost)
                total_sleeps[rtf] = clock - base
            finally:
                if p.isConnected(physicsClientId=sc.client):
                    p.disconnect(sc.client)

        # target_rtf=10 should sleep roughly 10x less than target_rtf=1
        assert total_sleeps[10.0] < total_sleeps[1.0] * 0.5, (
            f"target_rtf=10 should complete in much less wall-clock time than target_rtf=1: "
            f"sleep[10]={total_sleeps[10.0]:.4f}s vs sleep[1]={total_sleeps[1.0]:.4f}s"
        )

    # -- behind target (catch-up) ---------------------------------------------

    def test_behind_target_executes_multiple_steps(self, monkeypatch):
        """With extreme target_rtf, the catch-up loop batches multiple step_once
        calls per iteration.

        We spy on step_once directly and detect iteration boundaries via
        time.time() calls to measure per-iteration batch sizes, then
        verify at least some batches contain more than one step.
        """
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            # The fake clock advances per time.monotonic() call; the [MONITOR]
            # log's own call would shift the pacing these tests count.
            monitor_log_interval=0,
            target_rtf=10000.0,  # extreme: always far behind after any sleep
            max_steps_per_frame=10,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        try:
            base = time.time()
            clock = base

            # Detect iteration boundaries: time.time() is called between
            # step_once batches (loop_start, current_time, last_step_process_time).
            # We set a flag each call; the step_once spy flushes the current
            # batch counter when it sees the flag.
            new_iter_flag = False
            batches = []
            batch_counter = 0

            original_step_once = sc.step_once

            def spy_step_once(**kwargs):
                nonlocal new_iter_flag, batch_counter
                if new_iter_flag:
                    if batch_counter > 0:
                        batches.append(batch_counter)
                    batch_counter = 0
                    new_iter_flag = False
                batch_counter += 1
                return original_step_once(**kwargs)

            def fake_time():
                nonlocal new_iter_flag
                new_iter_flag = True
                return clock

            def fake_sleep(seconds):
                nonlocal clock
                clock += seconds

            monkeypatch.setattr(sc, "step_once", spy_step_once)
            monkeypatch.setattr(time, "time", fake_time)
            monkeypatch.setattr(time, "monotonic", fake_time)  # loop paces on monotonic
            monkeypatch.setattr(time, "sleep", fake_sleep)
            sc.run_simulation(duration=0.5)
            # Flush the last batch
            if batch_counter > 0:
                batches.append(batch_counter)

            # At least some iterations executed multiple step_once calls
            assert any(b > 1 for b in batches), (
                f"Expected multi-step catch-up batches, " f"got max={max(batches)}, batches={batches[:10]}"
            )
        finally:
            if p.isConnected(physicsClientId=sc.client):
                p.disconnect(sc.client)

    def test_max_steps_per_frame_caps_catchup(self, monkeypatch):
        """max_steps_per_frame caps step_once calls per loop iteration.

        With extreme target_rtf every iteration is behind target.  We spy on
        step_once and detect iteration boundaries via time.time() calls,
        recording per-iteration batch sizes.  No batch may exceed
        max_steps_per_frame, and extreme target_rtf should hit the cap.
        """
        max_steps = 5
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            # The fake clock advances per time.monotonic() call; the [MONITOR]
            # log's own call would shift the pacing these tests count.
            monitor_log_interval=0,
            target_rtf=10000.0,
            timestep=1.0 / 240.0,
            max_steps_per_frame=max_steps,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        try:
            base = time.time()
            clock = base

            new_iter_flag = False
            batches = []
            batch_counter = 0

            original_step_once = sc.step_once

            def spy_step_once(**kwargs):
                nonlocal new_iter_flag, batch_counter
                if new_iter_flag:
                    if batch_counter > 0:
                        batches.append(batch_counter)
                    batch_counter = 0
                    new_iter_flag = False
                batch_counter += 1
                return original_step_once(**kwargs)

            def fake_time():
                nonlocal new_iter_flag
                new_iter_flag = True
                return clock

            def fake_sleep(seconds):
                nonlocal clock
                clock += seconds

            monkeypatch.setattr(sc, "step_once", spy_step_once)
            monkeypatch.setattr(time, "time", fake_time)
            monkeypatch.setattr(time, "monotonic", fake_time)  # loop paces on monotonic
            monkeypatch.setattr(time, "sleep", fake_sleep)
            sc.run_simulation(duration=0.5)
            if batch_counter > 0:
                batches.append(batch_counter)

            expected_total = math.ceil(0.5 / params.timestep)
            assert sc.step_count == expected_total, f"Expected {expected_total} total steps, got {sc.step_count}"

            # No batch exceeds max_steps_per_frame
            assert all(b <= max_steps for b in batches), (
                f"Batch exceeded max_steps_per_frame={max_steps}: " f"max={max(batches)}, batches={batches[:10]}"
            )
            # Extreme target_rtf keeps the loop behind target → cap is reached
            assert any(b == max_steps for b in batches), (
                f"Expected some batches to hit cap={max_steps}: " f"batches={batches[:10]}"
            )
        finally:
            if p.isConnected(physicsClientId=sc.client):
                p.disconnect(sc.client)

    # -- ahead of target (sleep) ----------------------------------------------

    def test_ahead_of_target_sleeps(self, monkeypatch):
        """When ahead of real-time target, sleeps to wait.

        Uses a fake clock starting at 0.0 (not real time) to avoid
        floating-point precision loss from large unix timestamps.
        With target_rtf=1.0 and zero-cost step_once, each sleep is exactly
        one timestep.
        """
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            target_rtf=1.0,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        try:
            # Start clock at 0.0 to avoid float precision issues with
            # large unix timestamps (≈1.77e9 loses ~7 decimal digits).
            clock = 0.0
            sleep_durations = []

            def fake_time():
                return clock

            def fake_sleep(seconds):
                nonlocal clock
                sleep_durations.append(seconds)
                clock += seconds

            monkeypatch.setattr(time, "time", fake_time)
            monkeypatch.setattr(time, "monotonic", fake_time)  # loop paces on monotonic
            monkeypatch.setattr(time, "sleep", fake_sleep)
            sc.run_simulation(duration=0.02)

            dt = params.timestep
            # With zero-cost step_once, the loop alternates: ahead→sleep(dt), behind→step.
            # Every sleep should be approximately one timestep (target_rtf=1.0).
            positive_sleeps = [s for s in sleep_durations if s > 0]
            assert len(positive_sleeps) > 0, "Should have positive sleeps when ahead of target"
            for s in positive_sleeps:
                assert dt * 0.5 <= s <= dt * 2, (
                    f"Sleep {s:.6f}s outside expected range " f"[{dt*0.5:.6f}, {dt*2:.6f}] (timestep={dt:.6f})"
                )
        finally:
            if p.isConnected(physicsClientId=sc.client):
                p.disconnect(sc.client)

    # -- pause / resume -------------------------------------------------------

    def test_pause_skips_step_once(self, fast_sim):
        """While paused, step_once does not increment step_count."""
        fast_sim.initialize_simulation()
        fast_sim.pause()
        fast_sim.step_once()
        assert fast_sim.step_count == 0, "step_once should be skipped when paused"

    def test_resume_continues_stepping(self, fast_sim):
        """After resume, step_once executes again."""
        fast_sim.initialize_simulation()
        fast_sim.pause()
        fast_sim.step_once()
        assert fast_sim.step_count == 0

        fast_sim.resume()
        fast_sim.step_once()
        assert fast_sim.step_count == 1, "step_once should execute after resume"

    def test_pause_resume_preserves_sim_time_continuity(self, monkeypatch):
        """After pause→resume, sim_time advances continuously (no jump).

        Verifies that resetting start_time on resume prevents wall-clock
        time elapsed during pause from causing step skips or time jumps.
        """
        params = SimulationParams(
            gui=False,
            physics=False,
            monitor=False,
            target_rtf=1.0,
            max_steps_per_frame=5,
            log_level="warning",
        )
        sc = MultiRobotSimulationCore(params)
        try:
            dt = params.timestep

            # Control time.time() to simulate wall-clock time passing during pause
            wall_clock = 0.0
            base = time.time()

            def fake_time():
                # Closure only reads wall_clock; no nonlocal needed because
                # reassignment happens in the enclosing (same) scope.
                return base + wall_clock

            monkeypatch.setattr(time, "time", fake_time)
            monkeypatch.setattr(time, "monotonic", fake_time)  # loop paces on monotonic
            monkeypatch.setattr(time, "sleep", lambda s: None)

            sc.initialize_simulation()

            steps_before_pause = 5
            steps_during_pause = 2  # paused → all skipped
            steps_after_resume = 3

            # --- Phase 1: Execute normal steps before pause ---
            for _ in range(steps_before_pause):
                wall_clock += dt  # advance wall-clock by one timestep
                sc.step_once()
            assert sc.step_count == steps_before_pause

            sim_time_at_pause = sc.sim_time

            # --- Phase 2: Pause (simulate 10s of wall-clock time passing) ---
            sc.pause()
            wall_clock += 10.0  # 10s passes during pause
            for _ in range(steps_during_pause):
                sc.step_once()  # paused → skipped
            assert sc.step_count == steps_before_pause, "step_count should not change while paused"

            # --- Phase 3: Resume ---
            sc.resume()

            for _ in range(steps_after_resume):
                wall_clock += dt
                sc.step_once()
            assert sc.step_count == steps_before_pause + steps_after_resume, "Should resume stepping after unpause"

            # sim_time should advance continuously from pre-pause value
            expected_sim_time = sim_time_at_pause + steps_after_resume * dt
            assert (
                sc.sim_time > sim_time_at_pause
            ), f"sim_time should advance after resume: {sc.sim_time} > {sim_time_at_pause}"
            # No 10s jump — should advance by steps_after_resume timesteps only
            assert sc.sim_time == pytest.approx(expected_sim_time, abs=dt), (
                f"sim_time should advance by ~{steps_after_resume} step(s), not jump: "
                f"got {sc.sim_time}, expected ~{expected_sim_time}"
            )
        finally:
            if p.isConnected(physicsClientId=sc.client):
                p.disconnect(sc.client)


# ============================================================================
# filter_aabb_pairs
# ============================================================================


class TestFilterAabbPairs:
    """Test filter_aabb_pairs collision candidate filtering."""

    def test_overlapping_objects_returned_as_pair(self, sim_core):
        """Two boxes placed close together should be returned as a candidate pair."""
        obj1 = make_box(sim_core, [0, 0, 0])
        obj2 = make_box(sim_core, [0.3, 0, 0])
        sim_core._moved_this_step.add(obj1.object_id)

        pairs, _ = sim_core.filter_aabb_pairs()

        expected = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert expected in pairs

    def test_distant_objects_not_returned(self, sim_core):
        """Two boxes far apart should NOT appear in candidate pairs."""
        obj1 = make_box(sim_core, [0, 0, 0])
        obj2 = make_box(sim_core, [100, 100, 0])
        sim_core._moved_this_step.add(obj1.object_id)

        pairs, _ = sim_core.filter_aabb_pairs()

        pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert pair not in pairs

    def test_no_moved_objects_returns_empty(self, sim_core):
        """If no objects moved, filter should return no pairs."""
        make_box(sim_core, [0, 0, 0])
        make_box(sim_core, [0.3, 0, 0])
        sim_core._moved_this_step.clear()

        pairs, _ = sim_core.filter_aabb_pairs()
        assert pairs == []

    def test_disabled_objects_excluded(self, sim_core):
        """DISABLED collision mode objects should never appear in pairs."""
        normal = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        disabled = make_box(sim_core, [0.3, 0, 0], collision_mode=CollisionMode.DISABLED)
        sim_core._moved_this_step.add(normal.object_id)

        pairs, _ = sim_core.filter_aabb_pairs()

        for a, b in pairs:
            assert a != disabled.object_id and b != disabled.object_id

    def test_ignore_static_skips_static_pairs(self, sim_core):
        """With ignore_static=True, static objects should be excluded from pairs."""
        static = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        normal = make_box(sim_core, [0.3, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        sim_core._moved_this_step.add(normal.object_id)

        pairs, _ = sim_core.filter_aabb_pairs(ignore_static=True)

        pair = (min(static.object_id, normal.object_id), max(static.object_id, normal.object_id))
        assert pair not in pairs

    def test_ignore_static_false_includes_static(self, sim_core):
        """With ignore_static=False, static+normal overlapping pair should be included."""
        static = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        normal = make_box(sim_core, [0.3, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        sim_core._moved_this_step.add(normal.object_id)

        pairs, _ = sim_core.filter_aabb_pairs(ignore_static=False)

        pair = (min(static.object_id, normal.object_id), max(static.object_id, normal.object_id))
        assert pair in pairs

    def test_2d_mode_rejects_z_separated_same_cell(self, sim_core):
        """NORMAL_2D with Z-separated objects in the SAME Z-cell: rejected.

        Both objects share the same XY position and fall in the same Z-cell
        (Z=0 → cell 0, Z=1.5 → cell 0 with cell_size=2.0), so they share a
        cell and the search finds them.  However their Z-axis AABBs (size=0.25) do NOT
        overlap: [−0.125, 0.125] vs [1.375, 1.625].

        NORMAL_2D changes registration only (no z enlargement by the margin).
        AABB overlap is always checked on all 3 axes, so this pair is rejected
        — same as NORMAL_3D for the same geometry."""
        obj1 = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_2D)
        obj2 = make_box(sim_core, [0, 0, 1.5], collision_mode=CollisionMode.NORMAL_2D)

        # Only obj1 is "moved" so the spy observes obj1's search only
        sim_core._moved_this_step = {obj1.object_id}
        spy = SpyGrid(sim_core._cached_spatial_grid)
        sim_core._cached_spatial_grid = spy
        pairs, _ = sim_core.filter_aabb_pairs()

        # obj2's cell WAS queried (same Z-cell) → search hit
        assert was_searched(spy, sim_core, obj2), "obj2's cell should have been queried (same Z-cell)"
        # but pair is rejected by AABB Z-overlap check
        pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert pair not in pairs, "Search found obj2, but AABB Z-overlap should reject the pair"

    def test_2d_mode_detects_overlapping_objects(self, sim_core):
        """NORMAL_2D with actually overlapping objects should be detected.

        Both objects are at the same position → AABBs fully overlap on all
        3 axes.  They share a cell, so the search finds them, and the full
        XYZ AABB check passes."""
        obj1 = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_2D)
        obj2 = make_box(sim_core, [0.3, 0, 0], collision_mode=CollisionMode.NORMAL_2D)

        sim_core._moved_this_step = {obj1.object_id}
        spy = SpyGrid(sim_core._cached_spatial_grid)
        sim_core._cached_spatial_grid = spy
        pairs, _ = sim_core.filter_aabb_pairs()

        assert was_searched(spy, sim_core, obj2), "obj2's cell should have been queried"
        pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert pair in pairs, "2D mode should detect overlapping objects"

    def test_3d_rejects_z_separated_same_cell(self, sim_core):
        """Mirror of test_2d_mode_rejects_z_separated_same_cell with NORMAL_3D.

        Same geometry (Z=0 vs Z=1.5, same Z-cell) but NORMAL_3D mode.
        AABB overlap requires Z-axis overlap, which fails here
        (Z AABBs [−0.125,0.125] vs [1.375,1.625] don't overlap)."""
        obj1 = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        obj2 = make_box(sim_core, [0, 0, 1.5], collision_mode=CollisionMode.NORMAL_3D)

        sim_core._moved_this_step = {obj1.object_id}
        spy = SpyGrid(sim_core._cached_spatial_grid)
        sim_core._cached_spatial_grid = spy
        pairs, _ = sim_core.filter_aabb_pairs()

        # obj2's cell WAS queried (same Z-cell) → search hit
        assert was_searched(spy, sim_core, obj2), "obj2's cell should have been queried (same Z-cell)"
        pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert pair not in pairs, "Search found obj2, but AABB Z-overlap should reject the pair"

    def test_2d_search_skips_2d_body_in_different_z_cell(self, sim_core):
        """Two NORMAL_2D bodies in different Z-cells are never a candidate.

        Z=0 → cell 0, Z=3.0 → cell 1 (floor(3.0/2.0)=1). A NORMAL_2D body is
        registered in every z cell it touches, so two that overlap share one;
        the search may look at the next layer (for NORMAL_3D bodies there)
        but skips NORMAL_2D bodies outside its own layer."""
        obj1 = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_2D)
        obj2 = make_box(sim_core, [0, 0, 3.0], collision_mode=CollisionMode.NORMAL_2D)
        assert {c[2] for c in sim_core._cached_object_to_cell[obj1.object_id]}.isdisjoint(
            c[2] for c in sim_core._cached_object_to_cell[obj2.object_id]
        )

        sim_core._moved_this_step = {obj1.object_id}
        pairs, _ = sim_core.filter_aabb_pairs()

        pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert pair not in pairs

    def test_3d_search_miss_z_cell_too_far(self, sim_core):
        """A body in Z-cell 2 shares no cell with one in Z-cell 0 (search miss).

        Z=0 → cell 0, Z=5 → cell 2 (floor(5/2.0)=2).
        Each body's cells are only the ones its AABB (plus half the margin) touches,
        so obj1's cells never include Z-cell 2.

        This tests the search boundary, NOT the AABB overlap check.
        See test_3d_aabb_rejects_adjacent_z_cell for the AABB rejection path."""
        obj1 = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        obj2 = make_box(sim_core, [0, 0, 5], collision_mode=CollisionMode.NORMAL_3D)

        sim_core._moved_this_step = {obj1.object_id}
        spy = SpyGrid(sim_core._cached_spatial_grid)
        sim_core._cached_spatial_grid = spy
        pairs, _ = sim_core.filter_aabb_pairs()

        # obj2's cell was NEVER queried → genuine search miss
        assert not was_searched(spy, sim_core, obj2), "obj2's Z-cell should NOT have been queried (not one of obj1's cells)"
        pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert pair not in pairs

    def test_3d_bodies_in_adjacent_z_cells_share_no_cell(self, sim_core):
        """Z=0 (AABB -0.125..0.125) and Z=2.5 (2.375..2.625), cell 2.0.

        Each body is registered only in the cells its AABB touches, and the
        search looks only there, so obj2's cell is never queried and the pair
        is not a candidate."""
        obj1 = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        obj2 = make_box(sim_core, [0, 0, 2.5], collision_mode=CollisionMode.NORMAL_3D)

        sim_core._moved_this_step = {obj1.object_id}
        spy = SpyGrid(sim_core._cached_spatial_grid)
        sim_core._cached_spatial_grid = spy
        pairs, _ = sim_core.filter_aabb_pairs()

        assert not was_searched(spy, sim_core, obj2)
        pair = (min(obj1.object_id, obj2.object_id), max(obj1.object_id, obj2.object_id))
        assert pair not in pairs

    def test_return_profiling_includes_timings(self, sim_core):
        """return_profiling=True should return a dict with timing keys."""
        make_box(sim_core, [0, 0, 0])
        sim_core._moved_this_step.add(list(sim_core._sim_objects_dict.keys())[0])

        pairs, timings = sim_core.filter_aabb_pairs(return_profiling=True)

        assert timings is not None
        assert "get_aabbs" in timings
        assert "spatial_hashing" in timings
        assert "aabb_filtering" in timings

    def test_return_profiling_false_returns_none(self, sim_core):
        """return_profiling=False should return None for timings."""
        make_box(sim_core, [0, 0, 0])

        _, timings = sim_core.filter_aabb_pairs(return_profiling=False)
        assert timings is None

    def test_mode_change_forces_full_recalculation(self, sim_core):
        """Changing ignore_static triggers a full rescan of ALL objects.

        Setup: static + normal1 + normal2 (all overlapping).

        1st call (ignore_static=False): only normal1 is in _moved_this_step.
        Then _moved_this_step is cleared to simulate "nothing moved".

        2nd call (ignore_static=True): mode change should override the empty
        _moved_this_step with all object IDs → full recalculation.

        We prove recalculation happened by asserting normal1-normal2 pair IS
        found (impossible with an empty moved set).  We prove the new mode is
        applied by asserting static pairs are excluded.
        """
        static = make_box(sim_core, [0, 0, 0], collision_mode=CollisionMode.STATIC)
        normal1 = make_box(sim_core, [0.1, 0, 0], collision_mode=CollisionMode.NORMAL_3D)
        normal2 = make_box(sim_core, [0.2, 0, 0], collision_mode=CollisionMode.NORMAL_3D)

        # First call with ignore_static=False — sets _last_ignore_static
        sim_core._moved_this_step = {normal1.object_id}
        sim_core.filter_aabb_pairs(ignore_static=False)

        # Clear moved set to simulate "nothing moved"
        sim_core._moved_this_step.clear()

        # Switch to ignore_static=True — should force full recalculation
        pairs, _ = sim_core.filter_aabb_pairs(ignore_static=True)

        # Proof of recalculation: normal1-normal2 pair found despite empty
        # _moved_this_step (only possible if all objects were re-scanned)
        dyn_pair = (
            min(normal1.object_id, normal2.object_id),
            max(normal1.object_id, normal2.object_id),
        )
        assert dyn_pair in pairs, (
            "normal1-normal2 pair should be found — proves full recalculation "
            "happened even though _moved_this_step was empty"
        )

        # Proof of new mode applied: static pairs excluded
        for a, b in pairs:
            assert a != static.object_id and b != static.object_id, "static object should be excluded under ignore_static=True"


# ============================================================================
# _calculate_cell_size_from_aabbs
# ============================================================================


class TestCalculateCellSizeFromAabbs:
    """Test _calculate_cell_size_from_aabbs cell size calculation."""

    def test_returns_none_for_empty_aabbs(self, sim_core):
        """Empty AABB list should return None."""
        result = sim_core._calculate_cell_size_from_aabbs([])
        assert result is None

    def test_returns_none_when_no_cached_aabbs(self, sim_core):
        """No objects in sim → None when called without explicit aabbs."""
        assert len(sim_core._cached_aabbs_dict) == 0
        result = sim_core._calculate_cell_size_from_aabbs()
        assert result is None

    def test_single_object_returns_valid_cell_size(self, sim_core):
        """Single AABB should return a positive cell size."""
        aabbs = [((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))]
        result = sim_core._calculate_cell_size_from_aabbs(aabbs)
        assert result is not None
        assert result > 0

    def test_cell_size_at_least_min_floor(self, sim_core):
        """Cell size should be at least 0.5 even for tiny objects."""
        tiny_aabbs = [((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))]
        result = sim_core._calculate_cell_size_from_aabbs(tiny_aabbs)
        assert result >= 0.5, f"Cell size {result} should be >= 0.5"

    def test_cell_size_scales_with_object_size(self, sim_core):
        """Larger objects should produce larger cell sizes."""
        small_aabbs = [((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))]
        large_aabbs = [((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))]

        small_size = sim_core._calculate_cell_size_from_aabbs(small_aabbs)
        large_size = sim_core._calculate_cell_size_from_aabbs(large_aabbs)

        assert large_size > small_size

    def test_uses_cached_aabbs_when_none(self, sim_core):
        """When aabbs=None, should use cached AABBs from sim objects."""
        obj = make_box(sim_core, [0, 0, 0], size=1.0)
        assert len(sim_core._cached_aabbs_dict) > 0

        result = sim_core._calculate_cell_size_from_aabbs()
        assert result is not None
        assert result > 0


# ============================================================================
# get_aabbs
# ============================================================================


class TestGetAabbs:
    """Test get_aabbs returns correct AABB data."""

    def test_returns_empty_for_no_objects(self, sim_core):
        """No objects should return empty list."""
        result = sim_core.get_aabbs()
        assert result == []

    def test_returns_aabb_per_object(self, sim_core):
        """Should return one AABB per registered object."""
        make_box(sim_core, [0, 0, 0])
        make_box(sim_core, [5, 5, 0])
        make_box(sim_core, [10, 10, 0])

        result = sim_core.get_aabbs()
        assert len(result) == 3

    def test_aabb_format_is_min_max_tuple(self, sim_core):
        """Each AABB should be ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
        make_box(sim_core, [0, 0, 0])
        result = sim_core.get_aabbs()
        assert len(result) == 1
        aabb = result[0]
        assert len(aabb) == 2  # (min, max)
        assert len(aabb[0]) == 3  # min_x, min_y, min_z
        assert len(aabb[1]) == 3  # max_x, max_y, max_z
        # min should be <= max for all axes
        for i in range(3):
            assert aabb[0][i] <= aabb[1][i]


# ============================================================================
# update_monitor — collision logging
# ============================================================================


class TestCollisionLogging:
    """Test NEW COLLISION log output from update_monitor."""

    def test_logs_new_collision_when_count_increases(self, sim_core, caplog):
        """Should log NEW COLLISION message when collision_count increases."""
        import logging

        sim_core._start_time = __import__("time").time()
        sim_core._collision_count = 5
        sim_core._last_logged_collision_count = 0

        with caplog.at_level(logging.INFO, logger="pybullet_fleet.core_simulation"):
            sim_core.update_monitor()

        assert any("NEW COLLISION" in r.message for r in caplog.records)

    def test_no_log_when_count_unchanged(self, sim_core, caplog):
        """Should NOT log NEW COLLISION when count doesn't change."""
        import logging

        sim_core._start_time = __import__("time").time()
        sim_core._collision_count = 5
        sim_core._last_logged_collision_count = 5

        with caplog.at_level(logging.INFO, logger="pybullet_fleet.core_simulation"):
            sim_core.update_monitor()

        assert not any("NEW COLLISION" in r.message for r in caplog.records)


# ============================================================================
# disable_rendering / enable_rendering
# ============================================================================


class TestRenderingControl:
    """Test disable_rendering and enable_rendering."""

    def test_disable_sets_flag(self, sim_core, monkeypatch):
        """disable_rendering should set _rendering_enabled to False."""
        import pybullet as pb

        monkeypatch.setattr(pb, "configureDebugVisualizer", lambda *a, **kw: None)
        sim_core._params.gui = True
        sim_core._rendering_enabled = True

        sim_core.disable_rendering()
        assert sim_core._rendering_enabled is False

    def test_enable_sets_flag(self, sim_core, monkeypatch):
        """enable_rendering should set _rendering_enabled to True."""
        import pybullet as pb

        monkeypatch.setattr(pb, "configureDebugVisualizer", lambda *a, **kw: None)
        sim_core._params.gui = True
        sim_core._rendering_enabled = False

        sim_core.enable_rendering()
        assert sim_core._rendering_enabled is True

    def test_enable_noop_when_already_enabled(self, sim_core, monkeypatch):
        """enable_rendering should be a no-op if already enabled."""
        calls = []

        import pybullet as pb

        monkeypatch.setattr(pb, "configureDebugVisualizer", lambda *a, **kw: calls.append(1))
        sim_core._params.gui = True
        sim_core._rendering_enabled = True

        sim_core.enable_rendering()
        assert len(calls) == 0, "Should not call pybullet API when already enabled"

    def test_disable_noop_when_not_gui(self, sim_core):
        """disable_rendering should be a no-op in headless mode."""
        sim_core._params.gui = False
        sim_core._rendering_enabled = True  # artificially set

        sim_core.disable_rendering()
        # Should not crash, and state should not change (gui guard)
        assert sim_core._rendering_enabled is True

    def test_enable_noop_when_not_gui(self, sim_core):
        """enable_rendering should be a no-op in headless mode."""
        sim_core._params.gui = False
        sim_core._rendering_enabled = False

        sim_core.enable_rendering()
        assert sim_core._rendering_enabled is False


# ---------------------------------------------------------------------------
# Custom Profiling Fields
# ---------------------------------------------------------------------------


class TestCustomProfilingFields:
    """Tests for record_profiling API."""

    def test_record_profiling_auto_registers(self, sim_core):
        """First record_profiling call auto-registers the field in _profiling_stats."""
        sim_core.record_profiling("custom_logic", 1.0)

        assert "custom_logic" in sim_core._profiling_stats
        assert sim_core._profiling_stats["custom_logic"] == [1.0]

    def test_record_profiling_duplicate_no_extra_keys(self, sim_core):
        """Multiple record_profiling calls don't create duplicate keys."""
        sim_core.record_profiling("my_field", 1.0)
        sim_core.record_profiling("my_field", 2.0)

        assert list(sim_core._profiling_stats.keys()).count("my_field") == 1

    def test_record_profiling_accumulates_within_step(self, sim_core):
        """Multiple record_profiling calls within one step accumulate."""
        sim_core._enable_time_profiling = True

        # First call auto-registers with initial value
        sim_core.record_profiling("work", 1.5)
        # Second call accumulates into same slot
        sim_core.record_profiling("work", 2.0)

        assert sim_core._profiling_stats["work"][-1] == pytest.approx(3.5)

    def test_record_profiling_resets_each_step(self, sim_core):
        """Accumulators reset at the start of each step_once()."""
        sim_core._enable_time_profiling = True

        # Use a callback to record profiling DURING step_once
        def record_cb(sc, dt):
            sc.record_profiling("work", 5.0)

        sim_core.register_callback(record_cb)

        # First step: records 5.0
        sim_core.step_once()

        # Second step: new 0.0 slot is created, then callback records 5.0
        # The previous step's value (5.0) should be in _profiling_stats history,
        # not contaminating the new step
        sim_core.step_once()

        # Each step should independently have 5.0 (not accumulated 10.0)
        assert sim_core._profiling_stats["work"][-1] == pytest.approx(5.0)

    def test_custom_fields_in_return_profiling(self, sim_core):
        """Custom fields appear in step_once(return_profiling=True) as top-level keys."""

        # Use a callback to record profiling DURING step_once (simulates Agent.update())
        def record_cb(sc, dt):
            sc.record_profiling("my_compute", 0.42)

        sim_core.register_callback(record_cb)

        result = sim_core.step_once(return_profiling=True)

        assert result is not None
        assert "my_compute" in result
        assert result["my_compute"] == pytest.approx(0.42)

    def test_custom_fields_flushed_to_profiling_stats(self, sim_core):
        """Custom fields are appended to _profiling_stats during enable_time_profiling."""
        sim_core._enable_time_profiling = True

        # Use a callback to record profiling DURING step_once
        def record_cb(sc, dt):
            sc.record_profiling("sensor_read", 1.23)

        sim_core.register_callback(record_cb)
        sim_core.step_once()

        # The accumulated value should have been flushed to _profiling_stats
        assert len(sim_core._profiling_stats["sensor_read"]) == 1
        assert sim_core._profiling_stats["sensor_read"][0] == pytest.approx(1.23)

    def test_custom_fields_in_print_summary(self, sim_core, caplog):
        """Custom fields appear in _print_profiling_summary output."""
        import logging

        sim_core._enable_time_profiling = True

        # Manually populate stats to trigger summary
        sim_core._profiling_stats["planner"] = [0.5]
        sim_core._profiling_stats["total"].append(2.0)
        sim_core._profiling_stats["agent_update"].append(1.0)
        sim_core._profiling_stats["callbacks"].append(0.1)
        sim_core._profiling_stats["step_simulation"].append(0.1)
        sim_core._profiling_stats["collision_check"].append(0.2)
        sim_core._profiling_stats["monitor_update"].append(0.1)

        # Scoped to the package logger: PyBulletFleet sets its level there
        # rather than on the root logger, so raising the root level alone no
        # longer lets its INFO records through.
        with caplog.at_level(logging.INFO, logger="pybullet_fleet"):
            sim_core._print_profiling_summary()

        # Custom field "planner" should appear in the log output
        assert any("planner" in record.message for record in caplog.records)

    def test_initialize_simulation_clears_custom_stats(self, sim_core):
        """initialize_simulation() should clear custom profiling stats."""
        sim_core.record_profiling("work", 1.0)  # auto-registers

        sim_core.initialize_simulation()

        assert sim_core._profiling_stats["work"] == []
        # Field registration persists across initialize_simulation
        assert "work" in sim_core._profiling_stats


# ---------------------------------------------------------------------------
# batch_spawn context manager
# ---------------------------------------------------------------------------


class TestBatchSpawn:
    """Tests for MultiRobotSimulationCore.batch_spawn() context manager."""

    def test_sets_and_clears_flag(self, sim_core):
        """_batch_spawning is True inside context, False after."""
        assert sim_core._batch_spawning is False
        with sim_core.batch_spawn():
            assert sim_core._batch_spawning is True
        assert sim_core._batch_spawning is False

    def test_disables_rendering_during_context(self, sim_core, monkeypatch):
        """Rendering disabled on enter, re-enabled on exit."""
        import pybullet as pb

        monkeypatch.setattr(pb, "configureDebugVisualizer", lambda *a, **kw: None)
        sim_core._params.gui = True
        sim_core._rendering_enabled = True

        with sim_core.batch_spawn():
            assert sim_core._rendering_enabled is False
        assert sim_core._rendering_enabled is True

    def test_defers_spatial_grid_rebuild(self, sim_core, monkeypatch):
        """In AUTO_ADAPTIVE mode, set_collision_spatial_hash_cell_size_mode is not
        called per add_object inside batch_spawn, but is called once on exit."""
        import pybullet as pb

        monkeypatch.setattr(pb, "configureDebugVisualizer", lambda *a, **kw: None)
        sim_core._params.gui = True
        sim_core._rendering_enabled = True

        sim_core._params.spatial_hash_cell_size_mode = SpatialHashCellSizeMode.AUTO_ADAPTIVE

        rebuild_calls = []
        original_set = sim_core.set_collision_spatial_hash_cell_size_mode

        def spy_set(*args, **kwargs):
            rebuild_calls.append(1)
            return original_set(*args, **kwargs)

        monkeypatch.setattr(sim_core, "set_collision_spatial_hash_cell_size_mode", spy_set)

        with sim_core.batch_spawn():
            # Add 3 objects inside context
            make_box(sim_core, [0, 0, 0.5])
            make_box(sim_core, [2, 0, 0.5])
            make_box(sim_core, [4, 0, 0.5])
            # No rebuilds during batch
            assert len(rebuild_calls) == 0

        # One rebuild on exit
        assert len(rebuild_calls) == 1

    def test_reenables_on_exception(self, sim_core, monkeypatch):
        """Rendering re-enabled and flag cleared even if exception occurs."""
        import pybullet as pb

        monkeypatch.setattr(pb, "configureDebugVisualizer", lambda *a, **kw: None)
        sim_core._params.gui = True
        sim_core._rendering_enabled = True

        with pytest.raises(RuntimeError):
            with sim_core.batch_spawn():
                raise RuntimeError("boom")

        assert sim_core._batch_spawning is False
        assert sim_core._rendering_enabled is True

    def test_noop_in_headless(self, sim_core):
        """batch_spawn works fine in headless mode (no gui)."""
        sim_core._params.gui = False
        with sim_core.batch_spawn():
            make_box(sim_core, [0, 0, 0.5])
        # Should not crash; object created successfully
        assert len(sim_core.sim_objects) >= 1


# ============================================================================
# Model Paths Wiring
# ============================================================================


class TestModelPathsWiring:
    """SimulationParams.model_paths registers search paths on setup."""

    def test_model_paths_registered_on_setup(self, tmp_path):
        """model_paths from SimulationParams are added as robot_models search paths."""
        from pybullet_fleet.robot_models import get_search_paths, remove_search_path

        model_dir = tmp_path / "my_models"
        model_dir.mkdir()
        params = SimulationParams(gui=False, monitor=False, model_paths=[str(model_dir)])
        sc = MultiRobotSimulationCore(params)
        try:
            assert str(model_dir) in get_search_paths()
        finally:
            remove_search_path(str(model_dir))
            p.disconnect(sc.client)

    def test_model_paths_empty_by_default(self):
        """No extra search paths added if model_paths is empty."""
        from pybullet_fleet.robot_models import get_search_paths

        before = get_search_paths()
        params = SimulationParams(gui=False, monitor=False)
        sc = MultiRobotSimulationCore(params)
        try:
            after = get_search_paths()
            assert after == before
        finally:
            p.disconnect(sc.client)


# ---------------------------------------------------------------------------
# compute_scene_bounds
# ---------------------------------------------------------------------------


class TestComputeSceneBounds:
    """Test MultiRobotSimulationCore.compute_scene_bounds()."""

    def test_no_objects_returns_defaults(self, sim_core):
        """Empty scene returns origin center and default extent."""
        center, extent = sim_core.compute_scene_bounds()
        assert center == [0.0, 0.0, 0.0]
        assert extent == [0.0, 0.0, 0.0]

    def test_single_object(self, sim_core):
        """Single object: center = its position, extent = [0,0,0]."""
        make_box(sim_core, [3.0, 4.0, 0.5])
        center, extent = sim_core.compute_scene_bounds()
        assert abs(center[0] - 3.0) < 0.1
        assert abs(center[1] - 4.0) < 0.1
        # extent should be ~0 in X/Y since single object
        assert extent[0] < 0.5
        assert extent[1] < 0.5

    def test_multiple_objects(self, sim_core):
        """Multiple objects: center is mean, extent is max-min."""
        make_box(sim_core, [0.0, 0.0, 0.5])
        make_box(sim_core, [10.0, 0.0, 0.5])
        make_box(sim_core, [0.0, 6.0, 0.5])
        center, extent = sim_core.compute_scene_bounds()
        # center ~ mean([0,10,0], [0,0,6], ...) ≈ (3.3, 2.0, 0.5)
        assert abs(center[0] - 10.0 / 3.0) < 0.5
        assert abs(center[1] - 6.0 / 3.0) < 0.5
        # extent X ~ 10, extent Y ~ 6
        assert extent[0] > 8.0
        assert extent[1] > 4.0

    def test_returns_lists_of_floats(self, sim_core):
        """Return type should be plain lists of float, not numpy arrays."""
        make_box(sim_core, [1.0, 2.0, 0.5])
        center, extent = sim_core.compute_scene_bounds()
        assert isinstance(center, list)
        assert isinstance(extent, list)
        assert all(isinstance(v, float) for v in center)
        assert all(isinstance(v, float) for v in extent)
