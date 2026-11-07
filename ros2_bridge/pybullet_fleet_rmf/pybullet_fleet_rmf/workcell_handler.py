"""Workcell handler — RMF dispenser/ingestor protocol bridge.

Bridges RMF ``DispenserRequest`` and ``IngestorRequest`` messages to
PyBulletFleet simulation.  Runs inside BridgeNode with direct access
to ``sim_core`` for robot lookup and item management.

Unlike DoorHandler and LiftHandler (which are per-agent handlers bound
to a device via ``handler_map``), WorkcellHandler is a **singleton**
instantiated once by BridgeNode.  It manages all dispensers and
ingestors across the simulation.

**Architecture split:**

The actual simulation logic (item discovery, PickAction / DropAction
creation, return-home teleporting) lives in
:class:`~pybullet_fleet.plugins.workcell_plugin.WorkcellPlugin`
(a :class:`SimPlugin` in ``pybullet_fleet`` core — no ROS dependency).

This handler is the *thin ROS bridge*:

- Subscribes to RMF request topics and publishes state/result responses.
- Implements the RMF protocol layer (GUID tracking, duplicate detection,
  ACKNOWLEDGED → SUCCESS/FAILED lifecycle).
- Delegates dispense/ingest operations to ``WorkcellPlugin``.
- Translates plugin completion callbacks into ROS result messages.

Config (bridge YAML) — this handler is loaded as a bridge plugin, so its options
live under that plugin's ``config:`` (not a top-level ``workcell:`` section)::

    bridge_plugins:
      - class: pybullet_fleet_rmf.workcell_handler.WorkcellHandler
        config:
          item_shape:                  # ShapeParams-compatible dict
            shape_type: box
            half_extents: [0.15, 0.15, 0.1]
            rgba_color: [0.8, 0.5, 0.2, 1.0]
          item_mass: 0.5
          item_search_radius: 1.0
          attach_offset: [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]  # [dx, dy, dz, roll, pitch, yaw]
          return_home_delay: 5.0
          fail_on_no_robot: true     # default: true (Gazebo behaviour). When no
                                     # robot/carrier is at the workcell, report
                                     # FAILED; set false to report SUCCESS so an
                                     # RMF task doesn't stall on a missing robot.

The handler auto-discovers workcell names from incoming requests
(no pre-configuration required).
"""

import logging
from typing import Dict, List, Optional

from rmf_dispenser_msgs.msg import DispenserRequest, DispenserResult, DispenserState
from rmf_fleet_msgs.msg import FleetState
from rmf_ingestor_msgs.msg import IngestorRequest, IngestorResult, IngestorState

from pybullet_fleet.plugins.workcell_plugin import WorkcellPlugin, PendingAction

from pybullet_fleet_ros.bridge_plugin import BridgePlugin
from pybullet_fleet_ros.conversions import sim_time_to_ros_time

logger = logging.getLogger(__name__)


class _WorkcellInfo:
    """Internal state for a single workcell (dispenser or ingestor)."""

    __slots__ = ("name", "mode", "request_guid", "last_result")

    def __init__(self, name: str):
        self.name: str = name
        self.mode: int = DispenserState.IDLE
        self.request_guid: str = ""
        self.last_result: Optional[int] = None  # cached result for duplicate requests


class WorkcellHandler(BridgePlugin):
    """Singleton ROS bridge for all dispensers and ingestors.

    Instantiated once by BridgeNode.  Manages the RMF workcell protocol:
    subscribes to request topics, publishes state and results.

    All simulation logic (item discovery, action creation, return-home)
    is delegated to :class:`WorkcellPlugin`.  This class handles only:

    - ROS subscriptions and publishers
    - RMF protocol (GUID tracking, duplicate detection, state machine)
    - Fleet robot name → Agent resolution via ``/fleet_states``
    - Result/state message publishing
    """

    _registry_name = "workcell"

    # Prefixes for _past_requests keys so a dispenser and an ingestor sharing the
    # same RMF request GUID don't collide in the dedup cache.
    _DISPENSER_PREFIX = "d:"
    _INGESTOR_PREFIX = "i:"

    def __init__(self, node, sim_core):
        super().__init__(node, sim_core)
        # Backward-compat aliases used internally
        self._node = node
        self._sim_core = sim_core

        # Tracked workcells (auto-discovered from requests)
        self._dispensers: Dict[str, _WorkcellInfo] = {}
        self._ingestors: Dict[str, _WorkcellInfo] = {}

        # Past request GUIDs → success status (for duplicate detection)
        self._past_requests: Dict[str, int] = {}

        # Fleet states cache (fleet_name → list of robot names)
        self._fleet_robots: Dict[str, List[str]] = {}

        # ── Dispenser topics ──
        self._dispenser_req_sub = node.create_subscription(
            DispenserRequest, "dispenser_requests", self._on_dispenser_request, 10
        )
        self._dispenser_result_pub = node.create_publisher(DispenserResult, "dispenser_results", 10)
        self._dispenser_state_pub = node.create_publisher(DispenserState, "dispenser_states", 10)

        # ── Ingestor topics ──
        self._ingestor_req_sub = node.create_subscription(IngestorRequest, "ingestor_requests", self._on_ingestor_request, 10)
        self._ingestor_result_pub = node.create_publisher(IngestorResult, "ingestor_results", 10)
        self._ingestor_state_pub = node.create_publisher(IngestorState, "ingestor_states", 10)

        # ── Fleet states subscription ──
        self._fleet_state_sub = node.create_subscription(FleetState, "fleet_states", self._on_fleet_state, 10)

        # ── Periodic state publish (every 2s of sim time) ──
        self._last_state_publish: float = -1.0
        self._state_publish_interval: float = 2.0

        # When a request can't be fulfilled (no robot/carrier nearby, or the
        # action can't be created), publish FAILED — matching Gazebo's
        # TeleportDispenser/Ingestor. Set workcell.fail_on_no_robot: false to
        # instead report SUCCESS so an RMF task never stalls on a missing robot
        # (the legacy kinematic-bridge behaviour). Overridden in on_init().
        self._fail_on_no_robot: bool = True

    def on_init(self) -> None:
        """Register WorkcellPlugin and pre-register workcells from config.

        Called by BridgeNode after all plugins are loaded.
        ``self.config`` is available here (set by ``from_config_introspect``
        after ``__init__`` returns).
        """
        # Register the WorkcellPlugin on sim_core (passes config through)
        self._plugin: WorkcellPlugin = self._sim_core.register_plugin(WorkcellPlugin, config=self.config)

        # Pre-register workcells from overrides so RMF publishes requests
        self._pre_register_workcells(self.config)

        # Gazebo-faithful FAILED on no-robot by default; opt out via config.
        self._fail_on_no_robot = bool(self.config.get("fail_on_no_robot", True))

        logger.info(
            "WorkcellHandler started (plugin=%s, fail_on_no_robot=%s)",
            type(self._plugin).__name__,
            self._fail_on_no_robot,
        )

    # ------------------------------------------------------------------
    # Fleet states tracking
    # ------------------------------------------------------------------

    def _on_fleet_state(self, msg: FleetState):
        """Cache fleet → robot name mapping from /fleet_states."""
        self._fleet_robots[msg.name] = [rs.name for rs in msg.robots]

    def _pre_register_workcells(self, cfg: dict) -> None:
        """Pre-register workcells from config overrides.

        RMF's delivery task expects workcell State messages to be
        published *before* the robot arrives at the pickup/dropoff
        waypoint.  If no state is published, RMF never publishes
        the corresponding DispenserRequest / IngestorRequest and
        the task completes without pick/drop.

        Workcell type is inferred from the name:
        - Names containing ``ingestor`` → registered as ingestor
        - All others → registered as dispenser

        Can also be set explicitly per-workcell::

            overrides:
              my_workcell:
                type: "ingestor"  # or "dispenser"
                position: [x, y]
        """
        overrides = cfg.get("overrides", {})
        for name, wc_cfg in overrides.items():
            # Determine type from explicit config or name
            wc_type = wc_cfg.get("type", "") if isinstance(wc_cfg, dict) else ""
            if not wc_type:
                wc_type = "ingestor" if "ingestor" in name.lower() else "dispenser"

            if wc_type == "ingestor":
                if name not in self._ingestors:
                    self._ingestors[name] = _WorkcellInfo(name)
                    logger.info("Pre-registered ingestor '%s'", name)
            else:
                if name not in self._dispensers:
                    self._dispensers[name] = _WorkcellInfo(name)
                    logger.info("Pre-registered dispenser '%s'", name)

    # ------------------------------------------------------------------
    # Dispenser
    # ------------------------------------------------------------------

    def _on_dispenser_request(self, msg: DispenserRequest):
        """Handle a dispenser request from RMF.

        Like Gazebo's TeleportDispenser:
        1. Find the nearest pickable SimObject within search radius.
        2. If not found, spawn a fallback cargo box.
        3. Queue PickAction on the nearest fleet robot.
        4. Defer SUCCESS until the PickAction completes (checked in
           plugin's ``on_step``).
        """
        guid = msg.request_guid
        target = msg.target_guid
        fleet = msg.transporter_type
        dedup_key = f"{self._DISPENSER_PREFIX}{guid}"  # separate from ingestor guids

        # Duplicate detection
        if dedup_key in self._past_requests:
            self._publish_dispenser_result(guid, target, self._past_requests[dedup_key])
            return

        # Already pending (RMF retransmits requests)
        if dedup_key in self._plugin.pending_actions:
            self._publish_dispenser_result(guid, target, DispenserResult.ACKNOWLEDGED)
            return

        # Register workcell if new
        if target not in self._dispensers:
            self._dispensers[target] = _WorkcellInfo(target)

        info = self._dispensers[target]
        info.mode = DispenserState.BUSY
        info.request_guid = guid

        self._node.get_logger().info(f"[WorkcellHandler] Dispenser '{target}': request {guid} from fleet '{fleet}'")

        # Publish ACKNOWLEDGED
        self._publish_dispenser_result(guid, target, DispenserResult.ACKNOWLEDGED)

        # Find nearest robot from the requesting fleet
        candidates = self._get_fleet_candidates(fleet)
        self._node.get_logger().info(
            f"[WorkcellHandler] Dispenser '{target}': {len(candidates)} candidates from fleet '{fleet}'"
        )
        robot = self._plugin.find_nearest_robot(target, candidates)
        if robot is None:
            # No robot at the workcell. Gazebo's TeleportDispenser reports FAILED
            # here (default); workcell.fail_on_no_robot: false reports SUCCESS so
            # the RMF task doesn't stall.
            status = DispenserResult.FAILED if self._fail_on_no_robot else DispenserResult.SUCCESS
            self._publish_dispenser_result(guid, target, status)
            self._past_requests[dedup_key] = status
            info.mode = DispenserState.IDLE
            info.request_guid = ""
            self._node.get_logger().warning(
                f"[WorkcellHandler] Dispenser '{target}': no robot found -> "
                f"{'FAILED' if self._fail_on_no_robot else 'SUCCESS'}"
            )
            return

        # Delegate to plugin — dispense item + queue PickAction
        item, pick_action = self._plugin.dispense(
            target,
            robot,
            key=dedup_key,  # prefixed: keeps dispenser/ingestor in-flight tracking disjoint
            on_complete=self._on_dispenser_complete,
        )
        if pick_action is None:
            # Couldn't create the PickAction (no workcell position / fallback).
            status = DispenserResult.FAILED if self._fail_on_no_robot else DispenserResult.SUCCESS
            self._publish_dispenser_result(guid, target, status)
            self._past_requests[dedup_key] = status
            info.mode = DispenserState.IDLE
            info.request_guid = ""
            # error when we actually fail; warn when configured to report SUCCESS.
            log = self._node.get_logger().error if self._fail_on_no_robot else self._node.get_logger().warn
            log(
                f"[WorkcellHandler] Dispenser '{target}': failed to create PickAction -> "
                f"{'FAILED' if self._fail_on_no_robot else 'SUCCESS'}"
            )
            return

        self._node.get_logger().info(f"[WorkcellHandler] Dispenser '{target}': PickAction queued for robot '{robot.name}'")

    def _on_dispenser_complete(self, success: bool, pending: PendingAction):
        """Callback from WorkcellPlugin when a dispenser PickAction completes."""
        guid = self._find_guid_for_workcell(pending.workcell_name, "dispenser")

        status = DispenserResult.SUCCESS if success else DispenserResult.FAILED
        if guid:
            self._publish_dispenser_result(guid, pending.workcell_name, status)
            self._past_requests[f"{self._DISPENSER_PREFIX}{guid}"] = status

        info = self._dispensers.get(pending.workcell_name)
        if info:
            info.mode = DispenserState.IDLE
            info.request_guid = ""

        self._node.get_logger().info(
            f"[WorkcellHandler] Dispenser '{pending.workcell_name}': "
            f"{'SUCCESS' if success else 'FAILED'} (robot '{pending.robot.name if pending.robot else '?'}')"
        )

    def _publish_dispenser_result(self, guid: str, source: str, status: int):
        """Publish a DispenserResult message."""
        result = DispenserResult()
        result.time = self._node.get_clock().now().to_msg()
        result.request_guid = guid
        result.source_guid = source
        result.status = status
        self._dispenser_result_pub.publish(result)

    # ------------------------------------------------------------------
    # Ingestor
    # ------------------------------------------------------------------

    def _on_ingestor_request(self, msg: IngestorRequest):
        """Handle an ingestor request from RMF.

        Queues a DropAction on the nearest robot.  SUCCESS is deferred
        until the DropAction completes (checked in plugin's ``on_step``).
        """
        guid = msg.request_guid
        target = msg.target_guid
        fleet = msg.transporter_type
        dedup_key = f"{self._INGESTOR_PREFIX}{guid}"  # separate from dispenser guids

        # Duplicate detection
        if dedup_key in self._past_requests:
            self._publish_ingestor_result(guid, target, self._past_requests[dedup_key])
            return

        # Already pending
        if dedup_key in self._plugin.pending_actions:
            self._publish_ingestor_result(guid, target, IngestorResult.ACKNOWLEDGED)
            return

        # Register workcell if new
        if target not in self._ingestors:
            self._ingestors[target] = _WorkcellInfo(target)

        info = self._ingestors[target]
        info.mode = IngestorState.BUSY
        info.request_guid = guid

        self._node.get_logger().info(f"[WorkcellHandler] Ingestor '{target}': request {guid} from fleet '{fleet}'")

        # Publish ACKNOWLEDGED
        self._publish_ingestor_result(guid, target, IngestorResult.ACKNOWLEDGED)

        # Find nearest robot that is carrying cargo
        candidates = self._get_fleet_candidates(fleet)
        robot = self._plugin.find_nearest_carrier(target, candidates)
        if robot is None:
            # No carrier robot. Gazebo's TeleportIngestor reports FAILED (default);
            # workcell.fail_on_no_robot: false reports SUCCESS to avoid stalling.
            status = IngestorResult.FAILED if self._fail_on_no_robot else IngestorResult.SUCCESS
            self._publish_ingestor_result(guid, target, status)
            self._past_requests[dedup_key] = status
            info.mode = IngestorState.IDLE
            info.request_guid = ""
            self._node.get_logger().warning(
                f"[WorkcellHandler] Ingestor '{target}': no carrier robot found -> "
                f"{'FAILED' if self._fail_on_no_robot else 'SUCCESS'}"
            )
            return

        # Delegate to plugin — queue DropAction
        drop_action = self._plugin.ingest(
            target,
            robot,
            key=dedup_key,  # prefixed: keeps dispenser/ingestor in-flight tracking disjoint
            on_complete=self._on_ingestor_complete,
        )
        if drop_action is None:
            status = IngestorResult.FAILED if self._fail_on_no_robot else IngestorResult.SUCCESS
            self._publish_ingestor_result(guid, target, status)
            self._past_requests[dedup_key] = status
            info.mode = IngestorState.IDLE
            info.request_guid = ""
            # error when we actually fail; warn when configured to report SUCCESS.
            log = self._node.get_logger().error if self._fail_on_no_robot else self._node.get_logger().warn
            log(
                f"[WorkcellHandler] Ingestor '{target}': failed to create DropAction -> "
                f"{'FAILED' if self._fail_on_no_robot else 'SUCCESS'}"
            )
            return

        self._node.get_logger().info(
            f"[WorkcellHandler] Ingestor '{target}': queued DropAction for robot '{robot.name}'"
            f" (attached={len(robot.get_attached_objects())})"
        )

    def _on_ingestor_complete(self, success: bool, pending: PendingAction):
        """Callback from WorkcellPlugin when an ingestor DropAction completes."""
        guid = self._find_guid_for_workcell(pending.workcell_name, "ingestor")

        status = IngestorResult.SUCCESS if success else IngestorResult.FAILED
        if guid:
            self._publish_ingestor_result(guid, pending.workcell_name, status)
            self._past_requests[f"{self._INGESTOR_PREFIX}{guid}"] = status

        info = self._ingestors.get(pending.workcell_name)
        if info:
            info.mode = IngestorState.IDLE
            info.request_guid = ""

        self._node.get_logger().info(
            f"[WorkcellHandler] Ingestor '{pending.workcell_name}': "
            f"{'SUCCESS' if success else 'FAILED'} (robot '{pending.robot.name if pending.robot else '?'}')"
        )

    def _publish_ingestor_result(self, guid: str, source: str, status: int):
        """Publish an IngestorResult message."""
        result = IngestorResult()
        result.time = self._node.get_clock().now().to_msg()
        result.request_guid = guid
        result.source_guid = source
        result.status = status
        self._ingestor_result_pub.publish(result)

    # ------------------------------------------------------------------
    # Robot lookup
    # ------------------------------------------------------------------

    def _get_fleet_candidates(self, fleet_name: str):
        """Return agents matching *fleet_name* from fleet_states.

        Returns:
            List of Agent, or empty list.
        """
        if self._sim_core is None:
            return []

        fleet_robot_names = self._fleet_robots.get(fleet_name, [])

        candidates = []
        for agent in self._sim_core.agents:
            if agent.name in fleet_robot_names:
                candidates.append(agent)

        if not candidates:
            logger.warning(
                "No agents found for fleet '%s' (known fleets: %s, agents: %s)",
                fleet_name,
                list(self._fleet_robots.keys()),
                [a.name for a in self._sim_core.agents],
            )
        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_guid_for_workcell(self, workcell_name: str, kind: str) -> Optional[str]:
        """Recover the RMF GUID for a workcell that just completed.

        The ``on_complete`` callback receives the PendingAction but we
        need the GUID to publish the ROS result.  Since we pass the GUID
        as ``key=`` to the plugin, we can recover it from
        ``_WorkcellInfo.request_guid``.
        """
        registry = self._dispensers if kind == "dispenser" else self._ingestors
        info = registry.get(workcell_name)
        if info and info.request_guid:
            return info.request_guid
        return None

    # ------------------------------------------------------------------
    # Periodic state publishing
    # (called from BridgeNode POST_STEP)
    # ------------------------------------------------------------------

    def post_step(self, sim_time: float):
        """Publish periodic workcell state messages.

        Pending action checks and return-home processing are handled
        automatically by ``WorkcellPlugin.on_step()`` (run by the
        simulation's ``step`` event).  This method only publishes ROS
        state messages periodically.
        """
        if sim_time - self._last_state_publish < self._state_publish_interval:
            return
        self._last_state_publish = sim_time

        # Use sim_time directly — with use_sim_time=true, get_clock().now()
        # returns the same value from /clock, but this avoids the extra call.
        # now = self._node.get_clock().now().to_msg()
        now = sim_time_to_ros_time(sim_time)

        for info in self._dispensers.values():
            state = DispenserState()
            state.time = now
            state.guid = info.name
            state.mode = info.mode
            state.request_guid_queue = [info.request_guid] if info.request_guid else []
            state.seconds_remaining = 0.0
            self._dispenser_state_pub.publish(state)

        for info in self._ingestors.values():
            state = IngestorState()
            state.time = now
            state.guid = info.name
            state.mode = info.mode
            state.request_guid_queue = [info.request_guid] if info.request_guid else []
            state.seconds_remaining = 0.0
            self._ingestor_state_pub.publish(state)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Clean up ROS resources."""
        self._node.destroy_subscription(self._dispenser_req_sub)
        self._node.destroy_subscription(self._ingestor_req_sub)
        self._node.destroy_subscription(self._fleet_state_sub)
        self._node.destroy_publisher(self._dispenser_result_pub)
        self._node.destroy_publisher(self._dispenser_state_pub)
        self._node.destroy_publisher(self._ingestor_result_pub)
        self._node.destroy_publisher(self._ingestor_state_pub)
        logger.info("WorkcellHandler destroyed")
