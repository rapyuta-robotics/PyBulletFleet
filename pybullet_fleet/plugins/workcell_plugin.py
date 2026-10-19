"""WorkcellPlugin — ROS-free dispenser/ingestor simulation logic.

Extracts the pure simulation logic from ``WorkcellHandler`` so that
dispense/ingest cycles can run without ROS.  The ROS bridge
(``WorkcellHandler``) delegates to this plugin and handles only
messaging (subscriptions, publishers, result/state messages).

**Dispenser** flow:

1. ``dispense(workcell_name, robot)`` checks for nearby pickable
   SimObjects (like Gazebo TeleportDispenser's ``fill_dispenser``).
   If none found, a fallback cargo box is spawned.
2. A ``PickAction(target_position=...)`` is queued on the robot.
   The action itself resolves the nearest pickable object.
3. ``on_step()`` monitors pending actions, records item initial
   positions on completion, and emits completion events.

**Ingestor** flow:

1. ``ingest(workcell_name, robot)`` queues a ``DropAction`` on the
   carrier robot.
2. On completion the item is scheduled for return-home (teleport back
   to its dispenser position after a configurable delay).

**Config keys** (passed via ``config`` dict)::

    item_shape:                              # ShapeParams-compatible dict (default: box)
      shape_type: box
      half_extents: [0.15, 0.15, 0.1]
      rgba_color: [0.8, 0.5, 0.2, 1.0]
    item_mass: 0.5
    item_search_radius: 1.0
    attach_offset: [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]  # [dx, dy, dz, roll, pitch, yaw]
    spawn_fallback: true
    spawn_offset: [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]  # [dx, dy, dz, roll, pitch, yaw]
    return_home: true
    return_home_delay: 5.0
    overrides:              # per-workcell config (optional)
      dispenser_1:
        position: [8.5, -0.5]       # z defaults to 0
        item_search_radius: 2.0
        spawn_fallback: false
      dispenser_2:
        position: [1.0, 2.0, 5.0]   # multi-floor (z=5)
        item_shape:                  # ShapeParams dict (shape_type inferred if omitted)
          mesh_path: "mesh/coke_can.obj"
          mesh_scale: [0.01, 0.01, 0.01]
          rgba_color: [1.0, 0.0, 0.0, 1.0]
        item_mass: 0.3

**Usage (standalone, no ROS)**::

    plugin = sim.register_plugin(WorkcellPlugin, config={
        "item_search_radius": 1.0,
    })
    # After sim init:
    item, pick = plugin.dispense("dispenser_1", robot)
    # ... sim steps until pick completes ...
    drop = plugin.ingest("ingestor_1", robot)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from pybullet_fleet.logging_utils import get_lazy_logger
from pybullet_fleet.sim_plugin import SimPlugin

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore
    from pybullet_fleet.sim_object import ShapeParams, SimObject

logger = get_lazy_logger(__name__)

# Default delay (sim-seconds) before returning an item to its home position.
_DEFAULT_RETURN_HOME_DELAY: float = 5.0

# Default fallback cargo shape (box, orange).
_DEFAULT_ITEM_HALF_EXTENTS: List[float] = [0.15, 0.15, 0.1]
_DEFAULT_ITEM_COLOR: List[float] = [0.8, 0.5, 0.2, 1.0]
_DEFAULT_ITEM_MASS: float = 0.5


# ------------------------------------------------------------------
# Internal state containers
# ------------------------------------------------------------------


class WorkcellKind(Enum):
    """Type of workcell operation."""

    DISPENSER = "dispenser"
    INGESTOR = "ingestor"


@dataclass
class PendingAction:
    """A pick/drop action waiting for completion."""

    action: Any  # PickAction or DropAction
    cargo: Optional["SimObject"]  # dispensed item (dispenser) or None (ingestor)
    kind: WorkcellKind
    robot: "Agent"
    workcell_name: str
    on_complete: Optional[Callable] = None  # callback(success: bool, pending: PendingAction)


@dataclass
class _ReturnHomeEntry:
    """An item scheduled to teleport back to its initial position."""

    body_id: int
    home_position: Tuple[float, float, float]
    home_orientation: Tuple[float, float, float, float]
    scheduled_time: float


@dataclass
class WorkcellConfig:
    """Resolved per-workcell configuration."""

    position: Optional[Tuple[float, float, float]] = None
    search_radius: float = 1.0
    spawn_fallback: bool = True
    spawn_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    item_visual: Optional["ShapeParams"] = None  # Pre-resolved visual ShapeParams
    item_mass: float = _DEFAULT_ITEM_MASS


# ------------------------------------------------------------------
# WorkcellPlugin
# ------------------------------------------------------------------


class WorkcellPlugin(SimPlugin):
    """ROS-free workcell (dispenser/ingestor) simulation logic.

    Provides public methods for dispense/ingest operations and
    monitors pending actions in ``on_step()``.  Completion is
    reported via an optional ``on_complete`` callback passed to
    each operation.
    """

    _registry_name = "workcell"

    def __init__(
        self,
        sim_core: "MultiRobotSimulationCore",
        item_shape: Optional[Dict[str, Any]] = None,
        item_mass: float = _DEFAULT_ITEM_MASS,
        item_search_radius: float = 1.0,
        attach_offset: Optional[List[float]] = None,
        spawn_fallback: bool = True,
        spawn_offset: Optional[List[float]] = None,
        return_home_delay: float = _DEFAULT_RETURN_HOME_DELAY,
        return_home: bool = True,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(sim_core)

        from pybullet_fleet.sim_object import ShapeParams

        # Config — default item shape (resolved to ShapeParams at init time)
        self._default_item_visual: ShapeParams = self._parse_item_shape(item_shape)
        self._default_item_mass: float = item_mass
        self._item_search_radius: float = item_search_radius
        self._attach_offset: List[float] = attach_offset or [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
        self._spawn_fallback: bool = spawn_fallback
        self._spawn_offset: List[float] = spawn_offset or [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
        self._return_home_delay: float = return_home_delay
        self._return_home: bool = return_home and self._return_home_delay > 0

        # Pre-built per-workcell config (resolved from overrides at init time)
        overrides = overrides or {}
        self._workcells: Dict[str, WorkcellConfig] = {}
        for wc_name, wc_cfg in overrides.items():
            position: Optional[Tuple[float, float, float]] = None
            if "position" in wc_cfg:
                pos = wc_cfg["position"]
                z = float(pos[2]) if len(pos) > 2 else 0.0
                position = (float(pos[0]), float(pos[1]), z)
            # Per-workcell item shape: if item_shape dict provided, resolve to ShapeParams
            wc_item_shape = wc_cfg.get("item_shape")
            item_visual = self._parse_item_shape(wc_item_shape) if wc_item_shape else None
            wc_item_mass = float(wc_cfg["item_mass"]) if "item_mass" in wc_cfg else self._default_item_mass
            self._workcells[wc_name] = WorkcellConfig(
                position=position,
                search_radius=(
                    float(wc_cfg["item_search_radius"]) if "item_search_radius" in wc_cfg else self._item_search_radius
                ),
                spawn_fallback=bool(wc_cfg["spawn_fallback"]) if "spawn_fallback" in wc_cfg else self._spawn_fallback,
                spawn_offset=list(wc_cfg["spawn_offset"]) if "spawn_offset" in wc_cfg else list(self._spawn_offset),
                item_visual=item_visual,
                item_mass=wc_item_mass,
            )

        # Pending actions (key is caller-chosen, e.g. RMF GUID or sequential id)
        self._pending_actions: Dict[str, PendingAction] = {}

        # Item initial poses for return-home (body_id → (position, orientation))
        self._item_initial_poses: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = {}

        # Items scheduled to teleport back home after drop
        self._return_home_queue: List[_ReturnHomeEntry] = []

        # Auto-incrementing key for pending actions (when caller doesn't provide one)
        self._next_key: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispense(
        self,
        workcell_name: str,
        robot: "Agent",
        key: Optional[str] = None,
        on_complete: Optional[Callable] = None,
    ) -> Tuple[Optional["SimObject"], Any]:
        """Queue a PickAction on *robot* to pick the nearest item at *workcell_name*.

        Uses ``PickAction(target_position=...)`` so the action itself resolves
        the nearest pickable object within ``item_search_radius``.  If no
        pickable item exists near the workcell, a fallback cargo box is
        spawned first.

        Args:
            workcell_name: Name of the dispenser workcell.
            robot: The Agent to pick the item.
            key: Unique key for tracking (auto-generated if ``None``).
            on_complete: Callback ``(success: bool, pending: PendingAction)``
                called when the PickAction completes or fails.

        Returns:
            ``(None, pick_action)`` on success (item resolved lazily by
            PickAction), ``(None, None)`` if workcell position unknown.
        """
        from pybullet_fleet.action import PickAction
        from pybullet_fleet.geometry import Pose as PbfPose

        workcell_pos = self.get_workcell_position(workcell_name)
        if workcell_pos is None:
            logger.error("Dispense '%s': workcell position unknown — add it to overrides", workcell_name)
            return None, None

        # Ensure at least one pickable object exists near the workcell
        wc = self._workcells[workcell_name]
        search_radius = wc.search_radius
        spawn_fallback = wc.spawn_fallback

        if not self._has_pickable_near(workcell_name, search_radius):
            if not spawn_fallback:
                logger.warning("Dispense '%s': no pickable item near workcell (spawn_fallback disabled)", workcell_name)
                return None, None
            spawned = self._spawn_fallback_cargo(robot, workcell_name)
            if spawned is None:
                return None, None

        # Pre-record initial poses of pickable objects near the workcell
        # so return-home can restore them after delivery.  Recorded here
        # (before PickAction runs) because attach teleports the object.
        self._snapshot_pickable_poses(workcell_name, search_radius)

        # Let PickAction resolve the nearest pickable object itself.
        # move_skip_threshold=search_radius: the robot is already at the
        # RMF waypoint near the item — no need to drive to the item's
        # exact position (matches Gazebo's teleport-attach behaviour).
        # use_2d_search=False: workcell position now includes correct z,
        # so 3D distance is appropriate.
        pick = PickAction(
            target_position=[workcell_pos[0], workcell_pos[1], workcell_pos[2]],
            search_radius=search_radius,
            use_2d_search=False,
            use_approach=False,
            pick_offset=0.0,
            move_skip_threshold=search_radius,
            attach_link=-1,
            attach_relative_pose=PbfPose.from_euler(*self._attach_offset),
        )
        robot.add_action(pick)

        # Track pending
        if key is None:
            key = str(self._next_key)
            self._next_key += 1
        self._pending_actions[key] = PendingAction(
            action=pick,
            cargo=None,  # resolved lazily via pick._target_object
            kind=WorkcellKind.DISPENSER,
            robot=robot,
            workcell_name=workcell_name,
            on_complete=on_complete,
        )

        logger.info(
            "Dispense '%s': PickAction(target_position) for robot '%s' [key=%s]",
            workcell_name,
            robot.name,
            key,
        )
        return None, pick

    def ingest(
        self,
        workcell_name: str,
        robot: "Agent",
        key: Optional[str] = None,
        on_complete: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Queue a DropAction on *robot* at the ingestor position.

        Args:
            workcell_name: Name of the ingestor workcell.
            robot: The Agent carrying cargo.
            key: Unique key for tracking (auto-generated if ``None``).
            on_complete: Callback ``(success: bool, pending: PendingAction)``
                called when the DropAction completes or fails.

        Returns:
            DropAction on success, ``None`` on failure.
        """
        from pybullet_fleet.action import DropAction
        from pybullet_fleet.geometry import Pose as PbfPose

        ingestor_pos = self.get_workcell_position(workcell_name)
        if ingestor_pos is None:
            logger.error("Ingest '%s': workcell position unknown — add it to overrides", workcell_name)
            return None
        drop_x, drop_y, drop_z = ingestor_pos

        # Robot is already at the RMF waypoint near the ingestor.
        # Use move_skip_threshold so the robot doesn't drive to the
        # exact ingestor position (matches Gazebo teleport behaviour).
        robot_pose = robot.get_pose()
        dist = math.sqrt((robot_pose.x - drop_x) ** 2 + (robot_pose.y - drop_y) ** 2)

        drop = DropAction(
            drop_pose=PbfPose.from_xyz(drop_x, drop_y, drop_z),
            use_approach=False,
            drop_offset=0.0,
            move_skip_threshold=max(dist + 0.1, 1.0),  # skip MOVING_TO_DROP; robot already at waypoint
        )
        robot.add_action(drop)

        # Track pending
        if key is None:
            key = str(self._next_key)
            self._next_key += 1
        self._pending_actions[key] = PendingAction(
            action=drop,
            cargo=None,
            kind=WorkcellKind.INGESTOR,
            robot=robot,
            workcell_name=workcell_name,
            on_complete=on_complete,
        )

        logger.info(
            "Ingest '%s': DropAction for robot '%s' [key=%s]",
            workcell_name,
            robot.name,
            key,
        )
        return drop

    def find_nearest_robot(
        self,
        workcell_name: str,
        candidates: List["Agent"],
    ) -> Optional["Agent"]:
        """Return the candidate Agent nearest to *workcell_name*.

        Args:
            workcell_name: Workcell to measure distance from.
            candidates: List of Agent instances to search.

        Returns:
            Nearest Agent, or ``None`` if *candidates* is empty.
        """
        if not candidates:
            return None
        return self._nearest_to_workcell(candidates, workcell_name)

    def find_nearest_carrier(
        self,
        workcell_name: str,
        candidates: List["Agent"],
    ) -> Optional["Agent"]:
        """Return the nearest candidate carrying cargo (attached objects).

        Falls back to the nearest candidate if no carrier is found.

        Args:
            workcell_name: Workcell to measure distance from.
            candidates: List of Agent instances to search.

        Returns:
            Nearest carrier Agent, or nearest candidate as fallback.
        """
        if not candidates:
            return None
        carriers = [a for a in candidates if a.get_attached_objects()]
        if carriers:
            return self._nearest_to_workcell(carriers, workcell_name)
        # Fallback: nearest regardless of cargo
        return self._nearest_to_workcell(candidates, workcell_name)

    def get_workcell_position(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Look up a workcell's XYZ position.

        Positions are defined in config overrides::

            overrides:
              dispenser_1:
                position: [8.5, -0.5]       # z defaults to 0
              dispenser_2:
                position: [1.0, 2.0, 5.0]   # multi-floor

        Returns:
            ``(x, y, z)`` tuple, or ``None`` if not found.
        """
        return self._workcells[name].position if name in self._workcells else None

    @property
    def pending_actions(self) -> Dict[str, PendingAction]:
        """Read-only view of pending actions (key → PendingAction)."""
        return dict(self._pending_actions)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_step(self, dt: float) -> None:
        """Check pending actions and process return-home queue."""
        self._check_pending_actions()
        self._process_return_home(self.sim_core.sim_time)

    def on_reset(self) -> None:
        """Clear all pending actions and return-home queue."""
        self._pending_actions.clear()
        self._return_home_queue.clear()
        self._item_initial_poses.clear()

    # ------------------------------------------------------------------
    # Internal — item discovery
    # ------------------------------------------------------------------

    def _has_pickable_near(self, workcell_name: str, search_radius: Optional[float] = None) -> bool:
        """Check if any pickable, unattached SimObject exists within search radius.

        Args:
            workcell_name: Workcell to search around.
            search_radius: Override search radius (uses global default if ``None``).
        """
        workcell_pos = self.get_workcell_position(workcell_name)
        if workcell_pos is None:
            return False

        radius = search_radius if search_radius is not None else self._item_search_radius

        wx, wy, _wz = workcell_pos
        for obj in self.sim_core.sim_objects:
            if not getattr(obj, "pickable", False):
                continue
            if obj.is_attached():
                continue
            if getattr(obj, "name", None) == workcell_name:
                continue

            pose = obj.get_pose()
            dx = pose.position[0] - wx
            dy = pose.position[1] - wy
            # Use 2D (XY) distance — items may be on tables/shelves
            # at a different z height than the floor-level waypoint.
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < radius:
                return True

        return False

    def _snapshot_pickable_poses(self, workcell_name: str, search_radius: float) -> None:
        """Pre-record initial poses of pickable objects near a workcell.

        Called in ``dispense()`` **before** the PickAction runs so that
        the object's true spawn position (e.g. on a table at z=1.08) is
        captured.  After attach the object's pose reflects the parent
        link, making the original position unrecoverable.

        Only records objects not already tracked (idempotent for
        repeated dispenses of the same item).
        """
        workcell_pos = self.get_workcell_position(workcell_name)
        if workcell_pos is None:
            return

        wx, wy, _wz = workcell_pos
        for obj in self.sim_core.sim_objects:
            if obj.body_id in self._item_initial_poses:
                continue
            if not getattr(obj, "pickable", False) or obj.is_attached():
                continue
            pose = obj.get_pose()
            dx = pose.position[0] - wx
            dy = pose.position[1] - wy
            if math.sqrt(dx * dx + dy * dy) < search_radius:
                self._item_initial_poses[obj.body_id] = (
                    tuple(pose.position),  # type: ignore[arg-type]
                    tuple(pose.orientation),  # type: ignore[arg-type]
                )

    def _spawn_fallback_cargo(self, robot: "Agent", workcell_name: str) -> Optional["SimObject"]:
        """Spawn a fallback cargo item when no pre-placed item exists.

        Shape and mass can be customized per-workcell via ``item_shape``
        (a ``ShapeParams``-compatible dict) and ``item_mass``::

            overrides:
              dispenser_1:
                item_shape:
                  shape_type: box
                  half_extents: [0.3, 0.3, 0.2]
                  rgba_color: [1.0, 0.0, 0.0, 1.0]
                item_mass: 1.5
              dispenser_2:
                item_shape:
                  mesh_path: "mesh/coke_can.obj"
                  mesh_scale: [0.01, 0.01, 0.01]
        """
        from dataclasses import replace as dc_replace

        from pybullet_fleet.geometry import Pose as PbfPose
        from pybullet_fleet.sim_object import SimObject, SimObjectSpawnParams

        wc = self._workcells.get(workcell_name)
        offset = wc.spawn_offset if wc else self._spawn_offset
        dx, dy, dz = offset[0], offset[1], offset[2]
        roll = offset[3] if len(offset) > 3 else 0.0
        pitch = offset[4] if len(offset) > 4 else 0.0
        yaw = offset[5] if len(offset) > 5 else 0.0

        workcell_pos = self.get_workcell_position(workcell_name)
        if workcell_pos is not None:
            spawn_x = workcell_pos[0] + dx
            spawn_y = workcell_pos[1] + dy
            spawn_z = workcell_pos[2] + dz
        else:
            robot_pose = robot.get_pose()
            spawn_x = robot_pose.x + dx
            spawn_y = robot_pose.y + dy
            spawn_z = robot_pose.z + dz

        spawn_pose = PbfPose.from_euler(spawn_x, spawn_y, spawn_z, roll, pitch, yaw)

        # Use per-workcell ShapeParams if available, otherwise global default
        visual = wc.item_visual if wc and wc.item_visual else self._default_item_visual
        mass = wc.item_mass if wc else self._default_item_mass
        collision = dc_replace(visual, rgba_color=None)

        cargo_params = SimObjectSpawnParams(
            visual_shape=visual,
            collision_shape=collision,
            initial_pose=spawn_pose,
            mass=mass,
            pickable=True,
            name=f"{robot.name}_cargo",
        )
        try:
            return SimObject.from_params(cargo_params, self.sim_core)
        except Exception as e:
            logger.error("Failed to spawn fallback cargo: %s", e)
            return None

    @staticmethod
    def _parse_item_shape(shape_dict: Optional[Dict[str, Any]] = None) -> "ShapeParams":
        """Parse a dict into ``ShapeParams``, inferring ``shape_type`` if omitted.

        If *shape_dict* is ``None``, returns the default box shape
        (0.15 x 0.15 x 0.1, orange).  When ``shape_type`` is not
        specified, it is inferred: ``"mesh"`` if ``mesh_path`` is
        present, otherwise ``"box"``.
        """
        from pybullet_fleet.sim_object import ShapeParams

        if shape_dict is None:
            return ShapeParams(
                shape_type="box",
                half_extents=list(_DEFAULT_ITEM_HALF_EXTENTS),
                rgba_color=list(_DEFAULT_ITEM_COLOR),
            )
        # Infer shape_type when not explicitly provided
        d = dict(shape_dict)  # shallow copy — don't mutate caller's dict
        if "shape_type" not in d:
            d["shape_type"] = "mesh" if "mesh_path" in d else "box"
        return ShapeParams.from_dict(d)

    # ------------------------------------------------------------------
    # Internal — robot lookup
    # ------------------------------------------------------------------

    def _nearest_to_workcell(self, candidates: List["Agent"], workcell_name: str) -> Optional["Agent"]:
        """Return the candidate nearest to *workcell_name*."""
        if not candidates:
            return None

        workcell_pos = self.get_workcell_position(workcell_name)
        if workcell_pos is None:
            logger.warning(
                "Workcell '%s' position unknown — cannot determine nearest robot. " "Configure position in overrides to fix.",
                workcell_name,
            )
            return None

        nearest = None
        nearest_dist = float("inf")
        for agent in candidates:
            pose = agent.get_pose()
            dx = pose.position[0] - workcell_pos[0]
            dy = pose.position[1] - workcell_pos[1]
            dz = pose.position[2] - workcell_pos[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = agent
        return nearest

    # ------------------------------------------------------------------
    # Internal — pending action tracking
    # ------------------------------------------------------------------

    def _check_pending_actions(self) -> None:
        """Check if pending pick/drop actions have completed."""
        from pybullet_fleet.types import ActionStatus

        completed_keys: List[str] = []

        for key, pa in self._pending_actions.items():
            action = pa.action
            if action is None:
                completed_keys.append(key)
                continue

            if action.status in (ActionStatus.NOT_STARTED, ActionStatus.IN_PROGRESS):
                continue

            success = action.status == ActionStatus.COMPLETED

            # Record workcell position + item orientation as return-home destination
            # NOTE: orientation is captured after attach, not at spawn time.
            # This is acceptable for kinematic mode but may need revisiting
            # if physics-based pick alters item orientation significantly.
            if pa.kind == WorkcellKind.DISPENSER and success:
                target_obj = getattr(action, "_target_object", None)
                if target_obj is not None:
                    pa.cargo = target_obj
                    # _item_initial_poses was pre-recorded in dispense()
                    # via _snapshot_pickable_poses() before the PickAction ran.

            # Schedule return-home for ingestor drops
            if pa.kind == WorkcellKind.INGESTOR and success and self._return_home:
                target_obj = getattr(action, "_target_object", None)
                if target_obj is not None:
                    home = self._item_initial_poses.get(target_obj.body_id)
                    if home is not None:
                        home_pos, home_orn = home
                        entry = _ReturnHomeEntry(
                            body_id=target_obj.body_id,
                            home_position=home_pos,
                            home_orientation=home_orn,
                            scheduled_time=self.sim_core.sim_time + self._return_home_delay,
                        )
                        self._return_home_queue.append(entry)

            # Notify caller
            if pa.on_complete is not None:
                try:
                    pa.on_complete(success, pa)
                except Exception:
                    logger.error("on_complete callback failed for key=%s", key, exc_info=True)

            logger.info(
                "%s '%s': %s (robot '%s') [key=%s]",
                pa.kind.value.capitalize(),
                pa.workcell_name,
                "SUCCESS" if success else "FAILED",
                pa.robot.name if pa.robot else "?",
                key,
            )
            completed_keys.append(key)

        for key in completed_keys:
            del self._pending_actions[key]

    # ------------------------------------------------------------------
    # Internal — item return-home
    # ------------------------------------------------------------------

    def _process_return_home(self, sim_time: float) -> None:
        """Teleport items whose return time has come, at simulated time ``sim_time``, back to their initial position.

        Mirrors Gazebo TeleportIngestor ``send_ingested_item_home()``.
        """
        if not self._return_home_queue:
            return

        still_waiting: List[_ReturnHomeEntry] = []

        for entry in self._return_home_queue:
            if sim_time < entry.scheduled_time:
                still_waiting.append(entry)
                continue

            # Find the object
            obj = None
            for o in self.sim_core.sim_objects:
                if o.body_id == entry.body_id:
                    obj = o
                    break

            if obj is None:
                logger.warning("Item %d no longer exists — cannot return home", entry.body_id)
                continue

            if obj.is_attached():
                still_waiting.append(entry)
                continue

            hx, hy, hz = entry.home_position
            obj.set_pose_raw([hx, hy, hz], list(entry.home_orientation))
            logger.info("Item %d returned home to (%.2f, %.2f, %.2f)", entry.body_id, hx, hy, hz)

        self._return_home_queue = still_waiting
