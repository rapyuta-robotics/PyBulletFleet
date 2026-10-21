"""Tests for WorkcellPlugin — headless, no ROS dependency.

Covers: item discovery, dispense, ingest, nearest-robot lookup,
pending action completion callbacks, return-home teleporting.
"""

import math

import pybullet as p

from pybullet_fleet.action import PickAction, DropAction
from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.plugins.workcell_plugin import WorkcellPlugin
from pybullet_fleet.sim_object import SimObject, SimObjectSpawnParams, ShapeParams
from pybullet_fleet.types import ActionStatus, CollisionMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim():
    """Headless simulation with physics off."""
    return MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, physics=False))


def _spawn_agent(sim, name: str, x: float = 0.0, y: float = 0.0) -> Agent:
    """Spawn a minimal mobile robot agent."""
    params = AgentSpawnParams(
        urdf_path="robots/mobile_robot.urdf",
        initial_pose=Pose.from_xyz(x, y, 0.05),
        name=name,
        collision_mode=CollisionMode.DISABLED,
    )
    return Agent.from_params(params, sim)


def _spawn_item(sim, name: str, x: float, y: float, pickable: bool = True) -> SimObject:
    """Spawn a small pickable box."""
    params = SimObjectSpawnParams(
        visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
        initial_pose=Pose.from_xyz(x, y, 0.1),
        mass=0.5,
        pickable=pickable,
        name=name,
        collision_mode=CollisionMode.DISABLED,
    )
    return SimObject.from_params(params, sim)


# ---------------------------------------------------------------------------
# Tests: Plugin construction
# ---------------------------------------------------------------------------


class TestPluginConstruction:
    def test_default_config(self):
        sim = _make_sim()
        plugin = sim.register_plugin(WorkcellPlugin)
        assert plugin._item_search_radius == 1.0
        assert plugin._return_home is True
        # Default item shape is a box
        assert plugin._default_item_visual.shape_type == "box"
        assert plugin._default_item_visual.half_extents == [0.15, 0.15, 0.1]
        assert plugin._default_item_visual.rgba_color == [0.8, 0.5, 0.2, 1.0]
        assert plugin._default_item_mass == 0.5
        p.disconnect(sim.client)

    def test_custom_config(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={"item_search_radius": 2.5, "return_home_delay": 0},
        )
        assert plugin._item_search_radius == 2.5
        assert plugin._return_home is False
        p.disconnect(sim.client)

    def test_custom_item_shape(self):
        """Top-level item_shape overrides default box shape."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_shape": {
                    "shape_type": "box",
                    "half_extents": [0.3, 0.3, 0.2],
                    "rgba_color": [1.0, 0.0, 0.0, 1.0],
                },
                "item_mass": 2.0,
            },
        )
        assert plugin._default_item_visual.half_extents == [0.3, 0.3, 0.2]
        assert plugin._default_item_visual.rgba_color == [1.0, 0.0, 0.0, 1.0]
        assert plugin._default_item_mass == 2.0
        p.disconnect(sim.client)

    def test_item_shape_infers_mesh_type(self):
        """shape_type is inferred as 'mesh' when mesh_path is present."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_shape": {
                    "mesh_path": "robots/simple_cube.urdf",
                    "mesh_scale": [0.5, 0.5, 0.5],
                },
            },
        )
        assert plugin._default_item_visual.shape_type == "mesh"
        assert plugin._default_item_visual.mesh_path == "robots/simple_cube.urdf"
        p.disconnect(sim.client)

    def test_item_shape_infers_box_type(self):
        """shape_type defaults to 'box' when not specified and no mesh_path."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_shape": {"half_extents": [0.2, 0.2, 0.1]},
            },
        )
        assert plugin._default_item_visual.shape_type == "box"
        assert plugin._default_item_visual.half_extents == [0.2, 0.2, 0.1]
        p.disconnect(sim.client)

    def test_spawn_fallback_default_enabled(self):
        sim = _make_sim()
        plugin = sim.register_plugin(WorkcellPlugin)
        assert plugin._spawn_fallback is True
        p.disconnect(sim.client)

    def test_spawn_fallback_disabled(self):
        sim = _make_sim()
        plugin = sim.register_plugin(WorkcellPlugin, config={"spawn_fallback": False})
        assert plugin._spawn_fallback is False
        p.disconnect(sim.client)

    def test_return_home_disabled_via_flag(self):
        sim = _make_sim()
        plugin = sim.register_plugin(WorkcellPlugin, config={"return_home": False})
        assert plugin._return_home is False
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Per-workcell config overrides
# ---------------------------------------------------------------------------


class TestWorkcellOverrides:
    def test_overrides_pre_resolved(self):
        """Per-workcell overrides are pre-resolved into WorkcellConfig."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_search_radius": 1.0,
                "overrides": {"dispenser_1": {"item_search_radius": 3.0}},
            },
        )
        assert plugin._workcells["dispenser_1"].search_radius == 3.0
        p.disconnect(sim.client)

    def test_dispense_uses_override_search_radius(self):
        """dispense() uses per-workcell item_search_radius override."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_search_radius": 0.1,  # global: very small
                "overrides": {"wc": {"position": [0.0, 0.0], "item_search_radius": 5.0}},
            },
        )
        _spawn_item(sim, "cargo", 2.0, 0.0)  # 2m away — outside 0.1, inside 5.0
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        _, pick = plugin.dispense("wc", robot)
        assert isinstance(pick, PickAction)
        assert pick.search_radius == 5.0  # override applied to PickAction
        p.disconnect(sim.client)

    def test_dispense_per_workcell_spawn_fallback_disabled(self):
        """Per-workcell spawn_fallback=false prevents fallback for that workcell only."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "spawn_fallback": True,  # global: enabled
                "overrides": {
                    "wc_no_spawn": {"position": [0.0, 0.0], "spawn_fallback": False},
                    "wc_spawn": {"position": [5.0, 0.0]},
                },
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        # wc_no_spawn: override disables fallback
        _, pick1 = plugin.dispense("wc_no_spawn", robot)
        assert pick1 is None

        # wc_spawn: uses global default (enabled), spawns fallback
        _, pick2 = plugin.dispense("wc_spawn", robot)
        assert isinstance(pick2, PickAction)
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Workcell position lookup
# ---------------------------------------------------------------------------


class TestWorkcellPosition:
    def test_position_from_config_2d(self):
        """2-element position gets z=0 default."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={"overrides": {"wc2d": {"position": [8.5, -0.5]}}},
        )
        pos = plugin.get_workcell_position("wc2d")
        assert pos == (8.5, -0.5, 0.0)
        p.disconnect(sim.client)

    def test_position_from_config_3d(self):
        """3-element position preserves z for multi-floor."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={"overrides": {"wc3d": {"position": [1.0, 2.0, 5.0]}}},
        )
        pos = plugin.get_workcell_position("wc3d")
        assert pos == (1.0, 2.0, 5.0)
        p.disconnect(sim.client)

    def test_position_not_found(self):
        sim = _make_sim()
        plugin = sim.register_plugin(WorkcellPlugin)
        assert plugin.get_workcell_position("nonexistent") is None
        p.disconnect(sim.client)

    def test_nearest_robot_uses_3d_distance(self):
        """3D distance matters: robot closer in Z wins over robot closer in XY.

        Without the dz term the XY-closer robot would win — this layout
        ensures the test *fails* if only 2D distance is used.
        """
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc_floor2": {"position": [0.0, 0.0, 10.0]}},
            },
        )
        # close_xy: near in XY (1m) but on ground floor (z≈0.05)
        #   2D dist = 1.0,  3D dist = sqrt(1 + 0 + ~100) ≈ 10.05
        # close_z:  far in XY (5m) but teleported near workcell floor (z=9.5)
        #   2D dist = 5.0,  3D dist = sqrt(25 + 0 + 0.25) ≈ 5.02
        # With 2D only → close_xy wins (1 < 5)  ← WRONG
        # With 3D     → close_z  wins (5.02 < 10.05) ← CORRECT
        close_xy = _spawn_agent(sim, "close_xy", 1.0, 0.0)
        close_z = _spawn_agent(sim, "close_z", 5.0, 0.0)
        # Teleport close_z to near the workcell's floor (use set_pose_raw
        # so the kinematic pose cache is updated)
        close_z.set_pose_raw([5.0, 0.0, 9.5], [0, 0, 0, 1])

        result = plugin.find_nearest_robot("wc_floor2", [close_xy, close_z])
        assert result.name == "close_z"
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: has_pickable_near (fallback cargo check)
# ---------------------------------------------------------------------------


class TestHasPickableNear:
    def test_true_when_pickable_exists(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_search_radius": 5.0,
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        # Item near workcell
        _spawn_item(sim, "near_item", 0.5, 0.0)

        assert plugin._has_pickable_near("wc") is True
        p.disconnect(sim.client)

    def test_false_when_attached(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_search_radius": 5.0,
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        item = _spawn_item(sim, "item", 0.5, 0.0)

        # Attach item to a robot
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)
        robot.attach_object(item)

        assert plugin._has_pickable_near("wc") is False
        p.disconnect(sim.client)

    def test_false_when_outside_radius(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_search_radius": 0.5,
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        _spawn_item(sim, "far_item", 5.0, 0.0)

        assert plugin._has_pickable_near("wc") is False
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Nearest robot / carrier
# ---------------------------------------------------------------------------


class TestNearestRobot:
    def test_finds_nearest(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )

        far_robot = _spawn_agent(sim, "far", 10.0, 0.0)
        near_robot = _spawn_agent(sim, "near", 1.0, 0.0)

        result = plugin.find_nearest_robot("wc", [far_robot, near_robot])
        assert result is near_robot
        p.disconnect(sim.client)

    def test_empty_candidates(self):
        sim = _make_sim()
        plugin = sim.register_plugin(WorkcellPlugin)
        assert plugin.find_nearest_robot("wc", []) is None
        p.disconnect(sim.client)

    def test_carrier_preferred(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )

        # Near robot without cargo
        near = _spawn_agent(sim, "near", 1.0, 0.0)
        # Far robot with cargo
        far = _spawn_agent(sim, "far", 5.0, 0.0)
        cargo = _spawn_item(sim, "cargo", 5.0, 0.0)
        far.attach_object(cargo)

        result = plugin.find_nearest_carrier("wc", [near, far])
        assert result is far  # carrier preferred even if farther
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Dispense
# ---------------------------------------------------------------------------


class TestDispense:
    def test_dispense_existing_item(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_search_radius": 5.0,
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        item = _spawn_item(sim, "cargo", 0.5, 0.0)
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        result_item, pick = plugin.dispense("wc", robot, key="req1")
        assert result_item is None  # item resolved lazily by PickAction
        assert isinstance(pick, PickAction)
        assert pick.target_position is not None  # uses target_position mode
        assert pick.search_radius == 5.0
        assert "req1" in plugin.pending_actions
        p.disconnect(sim.client)

    def test_dispense_spawns_fallback(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        result_item, pick = plugin.dispense("wc", robot)
        # Item resolved lazily by PickAction; fallback cargo spawned
        assert result_item is None
        assert isinstance(pick, PickAction)
        # Verify fallback cargo was spawned (a pickable object named robot0_cargo)
        cargo_names = [o.name for o in sim.sim_objects if getattr(o, "pickable", False)]
        assert "robot0_cargo" in cargo_names
        p.disconnect(sim.client)

    def test_dispense_spawn_offset(self):
        """spawn_offset applies 6-DOF offset to fallback cargo position."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {
                    "wc": {
                        "position": [1.0, 2.0, 3.0],
                        "spawn_offset": [0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
                    },
                },
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        plugin.dispense("wc", robot)
        cargo = [o for o in sim.sim_objects if o.name == "robot0_cargo"]
        assert len(cargo) == 1
        pose = cargo[0].get_pose()
        assert abs(pose.position[0] - 1.5) < 0.01  # 1.0 + 0.5
        assert abs(pose.position[1] - 1.5) < 0.01  # 2.0 + (-0.5)
        assert abs(pose.position[2] - 4.0) < 0.01  # 3.0 + 1.0
        p.disconnect(sim.client)

    def test_dispense_fallback_uses_item_config_box(self):
        """Per-workcell item_shape overrides global shape and color."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {
                    "wc": {
                        "position": [0.0, 0.0],
                        "item_shape": {
                            "half_extents": [0.3, 0.3, 0.2],
                            "rgba_color": [1.0, 0.0, 0.0, 1.0],
                        },
                        "item_mass": 1.5,
                    },
                },
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        plugin.dispense("wc", robot)
        cargo = [o for o in sim.sim_objects if o.name == "robot0_cargo"]
        assert len(cargo) == 1
        assert cargo[0].mass == 1.5
        p.disconnect(sim.client)

    def test_dispense_fallback_uses_item_config_mesh(self):
        """Smoke test: mesh item_shape spawns without error and applies mass.

        Verifies that the mesh code path in _spawn_fallback_cargo (which
        calls p.createVisualShape with GEOM_MESH) doesn't crash.  Shape
        attributes (mesh_path, mesh_scale) are not accessible from the
        spawned SimObject, so only mass is checked — same as the box test.
        """
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {
                    "wc": {
                        "position": [0.0, 0.0],
                        "item_shape": {
                            "mesh_path": "robots/simple_cube.urdf",
                            "mesh_scale": [0.5, 0.5, 0.5],
                            "rgba_color": [0.0, 1.0, 0.0, 1.0],
                        },
                        "item_mass": 2.5,
                    },
                },
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        _, pick = plugin.dispense("wc", robot)
        assert isinstance(pick, PickAction)
        cargo = [o for o in sim.sim_objects if o.name == "robot0_cargo"]
        assert len(cargo) == 1
        assert cargo[0].mass == 2.5
        p.disconnect(sim.client)

    def test_dispense_no_fallback_when_disabled(self):
        """When spawn_fallback=False, dispense returns (None, None) if no item exists."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "spawn_fallback": False,
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        result_item, pick = plugin.dispense("wc", robot)
        assert result_item is None
        assert pick is None  # no fallback spawned, no PickAction created
        p.disconnect(sim.client)

    def test_dispense_auto_key(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        plugin.dispense("wc", robot)
        plugin.dispense("wc", robot)

        assert len(plugin.pending_actions) == 2
        assert "0" in plugin.pending_actions
        assert "1" in plugin.pending_actions
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_creates_drop_action(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"ingestor_1": {"position": [5.0, 5.0]}},
            },
        )
        robot = _spawn_agent(sim, "robot0", 5.0, 5.0)

        drop = plugin.ingest("ingestor_1", robot, key="drop1")
        assert isinstance(drop, DropAction)
        assert "drop1" in plugin.pending_actions
        p.disconnect(sim.client)

    def test_ingest_unknown_workcell_returns_none(self):
        """When ingestor has no configured position, ingest returns None."""
        sim = _make_sim()
        plugin = sim.register_plugin(WorkcellPlugin)
        robot = _spawn_agent(sim, "robot0", 4.0, 6.0)

        drop = plugin.ingest("unknown_ingestor", robot, key="d1")
        assert drop is None
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Pending action completion + callback
# ---------------------------------------------------------------------------


class TestPendingCompletion:
    def test_callback_on_completion(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        completions = []

        def on_done(success, pending):
            completions.append((success, pending.workcell_name))

        plugin.dispense("wc", robot, key="r1", on_complete=on_done)

        # Simulate action completion
        pa = plugin._pending_actions["r1"]
        pa.action.status = ActionStatus.COMPLETED

        plugin.on_step(0.1)

        assert len(completions) == 1
        assert completions[0] == (True, "wc")
        assert "r1" not in plugin._pending_actions
        p.disconnect(sim.client)

    def test_failed_action_callback(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        completions = []
        plugin.dispense("wc", robot, key="r2", on_complete=lambda s, p: completions.append(s))

        pa = plugin._pending_actions["r2"]
        pa.action.status = ActionStatus.FAILED

        plugin.on_step(0.1)

        assert completions == [False]
        assert "r2" not in plugin._pending_actions
        p.disconnect(sim.client)

    def test_initial_pose_recorded_from_workcell_position(self):
        """_item_initial_poses records item's actual position (pre-pick snapshot)."""
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "item_search_radius": 5.0,
                "overrides": {"wc": {"position": [3.0, 4.0]}},
            },
        )
        item = _spawn_item(sim, "cargo", 3.5, 4.5)  # item near workcell but not at same pos
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        _, pick = plugin.dispense("wc", robot, key="d1")

        # Simulate PickAction resolving target
        pick._target_object = item
        pick.status = ActionStatus.COMPLETED

        plugin.on_step(0.1)

        # Initial pose should be the item's actual spawn position (snapshotted
        # in dispense() before the PickAction runs), matching Gazebo's
        # send_ingested_item_home() which returns items to their spawn pose.
        assert item.body_id in plugin._item_initial_poses
        pos, orn = plugin._item_initial_poses[item.body_id]
        assert abs(pos[0] - 3.5) < 0.1  # item's actual x
        assert abs(pos[1] - 4.5) < 0.1  # item's actual y
        assert len(orn) == 4  # quaternion preserved
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Return home
# ---------------------------------------------------------------------------


class TestReturnHome:
    def test_return_home_after_delay_preserves_orientation(self):
        """Full lifecycle: dispense → pick → ingest → drop → return-home.

        Verifies:
        1. Item does NOT teleport before the scheduled delay.
        2. Item teleports back to its original position after the delay.
        3. Original orientation (non-identity) is preserved.
        """
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "return_home_delay": 1.0,
                "item_search_radius": 5.0,
                "overrides": {"wc": {"position": [2.0, 3.0]}},
            },
        )

        item = _spawn_item(sim, "cargo", 2.5, 3.5)
        # Give the item a non-identity orientation (90° yaw)
        import pybullet as pb

        yaw_quat = list(pb.getQuaternionFromEuler([0, 0, math.pi / 2]))
        item.set_pose_raw([2.5, 3.5, 0.1], yaw_quat)

        robot = _spawn_agent(sim, "robot0", 2.0, 3.0)

        # Dispense and simulate pick completion
        _, pick = plugin.dispense("wc", robot, key="d1")
        pick._target_object = item
        pick.status = ActionStatus.COMPLETED
        plugin.on_step(0.1)

        # Ingest (drop) and simulate completion — this schedules return-home
        drop = plugin.ingest("wc", robot, key="i1")
        drop._target_object = item
        drop.status = ActionStatus.COMPLETED
        plugin.on_step(0.1)  # schedules the return home at sim time 0 + delay (1.0)

        # Move item away with different orientation
        item.set_pose_raw([10.0, 10.0, 0.1], [0, 0, 0, 1])

        # 1. Before delay expires — item should NOT have returned
        plugin._process_return_home(0.5)
        pos = item.get_pose().position
        assert abs(pos[0] - 10.0) < 0.1, "Item should still be at 10,10 before delay"

        # 2. After delay expires — item should return
        plugin._process_return_home(2.0)

        pose = item.get_pose()
        # Position should be item's original spawn position
        assert abs(pose.position[0] - 2.5) < 0.1  # item x
        assert abs(pose.position[1] - 3.5) < 0.1  # item y

        # 3. Orientation should be the original (90° yaw), not identity
        restored_quat = pose.orientation
        for i in range(4):
            assert (
                abs(restored_quat[i] - yaw_quat[i]) < 0.01
            ), f"Orientation component {i}: expected {yaw_quat[i]}, got {restored_quat[i]}"
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_on_reset_clears_state(self):
        sim = _make_sim()
        plugin = sim.register_plugin(
            WorkcellPlugin,
            config={
                "overrides": {"wc": {"position": [0.0, 0.0]}},
            },
        )
        robot = _spawn_agent(sim, "robot0", 0.0, 0.0)

        plugin.dispense("wc", robot, key="r1")
        plugin._item_initial_poses[99] = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))

        plugin.on_reset()

        assert len(plugin._pending_actions) == 0
        assert len(plugin._item_initial_poses) == 0
        assert len(plugin._return_home_queue) == 0
        p.disconnect(sim.client)
