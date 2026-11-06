"""ROS wrappers for the transport-neutral fleet API."""

from __future__ import annotations

import time
from typing import Iterable

from rclpy.qos import QoSProfile
from rclpy.serialization import serialize_message
from std_msgs.msg import Header

from pybullet_fleet.commands import (
    CommandAck as PbfCommandAck,
    DEFAULT_ATTACH_SEARCH_RADIUS,
    RobotActionCommand,
    RobotAttachCommand,
    RobotGoalCommand2D,
    RobotGoalCommand3D,
    RobotJointPositionsCommand,
    RobotNamedJointPositionsCommand,
)
from pybullet_fleet.fleet_api import (
    FleetCommandDispatcher,
    FleetStateProvider,
    RobotState3D,
)
from pybullet_fleet.geometry import Pose as PbfPose

try:
    from pybullet_fleet_msgs.msg import (
        CommandAck,
        FleetAttach,
        FleetExecuteAction,
        FleetJointCommand,
        FleetNavigate,
        FleetState,
        FleetStop,
        RobotGoal2D,
        RobotGoal3D,
        RobotJointPositionsCommand as RobotJointPositionsCommandMsg,
        RobotNamedJointPositionsCommand as RobotNamedJointPositionsCommandMsg,
        RobotState3D as RobotState3DMsg,
        TransportTiming,
    )
    from pybullet_fleet_msgs.srv import FleetAttach as FleetAttachSrv
    from pybullet_fleet_msgs.srv import FleetExecuteAction as FleetExecuteActionSrv
    from pybullet_fleet_msgs.srv import FleetJointCommand as FleetJointCommandSrv
    from pybullet_fleet_msgs.srv import FleetNavigate as FleetNavigateSrv
    from pybullet_fleet_msgs.srv import FleetStop as FleetStopSrv

    _FLEET_MSGS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    CommandAck = None
    FleetAttach = None
    FleetExecuteAction = None
    FleetJointCommand = None
    FleetNavigate = None
    FleetState = None
    FleetStop = None
    RobotGoal2D = None
    RobotGoal3D = None
    RobotJointPositionsCommandMsg = None
    RobotNamedJointPositionsCommandMsg = None
    RobotState3DMsg = None
    TransportTiming = None
    FleetAttachSrv = None
    FleetExecuteActionSrv = None
    FleetJointCommandSrv = None
    FleetNavigateSrv = None
    FleetStopSrv = None
    _FLEET_MSGS_IMPORT_ERROR = exc

from .interface_config import FleetApiConfig, ManagerInterfaceConfig
from .fleet_endpoints import FLEET_API_NAMESPACE, fleet_endpoint, normalize_fleet_namespace
from .conversions import ros_time_to_seconds


def _require_fleet_msgs() -> None:
    if _FLEET_MSGS_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "fleet_api is enabled but pybullet_fleet_msgs Python bindings are unavailable. "
        "Build/source the ROS workspace so pybullet_fleet_msgs is generated."
    ) from _FLEET_MSGS_IMPORT_ERROR


def _fleet_state_qos(config: FleetApiConfig | ManagerInterfaceConfig) -> QoSProfile:
    return QoSProfile(
        history=config.state_qos.history,
        depth=config.state_qos.depth,
        reliability=config.state_qos.reliability,
        durability=config.state_qos.durability,
    )


class FleetRosInterface:
    """Expose one global or manager-scoped fleet endpoint set over ROS."""

    def __init__(
        self,
        node,
        sim_core,
        config: FleetApiConfig | ManagerInterfaceConfig,
        *,
        namespace: str = FLEET_API_NAMESPACE,
        state_names: Iterable[str] | None = None,
        command_names: Iterable[str] | None = None,
    ) -> None:
        """Create one global or manager-scoped Fleet ROS endpoint set.

        Args:
            node: ROS node that owns this endpoint set.
            sim_core: Simulation used by the state provider and command
                dispatcher.
            config: Global Fleet API or manager-scoped endpoint configuration.
            namespace: Base namespace, such as ``/fleet`` or
                ``/fleet/delivery``.
            state_names: Agent names included in the state stream. ``None``
                means no filter, so the provider publishes every Agent. This
                preserves the global ``/fleet/states`` compatibility stream;
                manager endpoints receive their resolved Agent-name set.
            command_names: Agent names accepted by commands. ``None`` leaves
                commands unrestricted. It is intentionally separate from
                ``state_names``: a global endpoint may publish a filtered
                state stream while retaining compatibility commands for every
                Agent.
        """
        self.node = node
        self._sim_core = sim_core
        self.config = config
        self._namespace = normalize_fleet_namespace(namespace)
        # ``None`` deliberately means no state filter, i.e. every sim Agent.
        # A manager endpoint receives an explicit frozen set instead.
        self._state_names = frozenset(state_names) if state_names is not None else None
        self._state_publish_rate = getattr(config, "state_publish_rate", None)
        self._last_state_publish_time = float("-inf")

        self._state_pub = None
        self._navigate_sub = None
        self._navigate_srv = None
        self._stop_sub = None
        self._stop_srv = None
        self._attach_sub = None
        self._attach_srv = None
        self._execute_action_sub = None
        self._execute_action_srv = None
        self._joint_sub = None
        self._joint_srv = None
        self._transport_probe_pub = None
        self._transport_probe_sequence = 0

        if not getattr(config, "enabled", True):
            return
        _require_fleet_msgs()

        self.state_provider = FleetStateProvider(sim_core)
        self.command_dispatcher = FleetCommandDispatcher(sim_core, allowed_names=command_names)
        self._rmf_frame_offset = _node_frame_offset(node)

        if config.states:
            self._state_pub = node.create_publisher(FleetState, self._endpoint("states"), _fleet_state_qos(config))
        if getattr(config, "transport_probe", False):
            self._transport_probe_pub = node.create_publisher(TransportTiming, self._endpoint("transport_timing"), 10)
        if config.navigate:
            self._navigate_sub = node.create_subscription(FleetNavigate, self._endpoint("navigate"), self._on_navigate, 10)
            self._navigate_srv = node.create_service(FleetNavigateSrv, self._endpoint("navigate"), self._on_navigate_service)
        if config.stop:
            self._stop_sub = node.create_subscription(FleetStop, self._endpoint("stop"), self._on_stop, 10)
            self._stop_srv = node.create_service(FleetStopSrv, self._endpoint("stop"), self._on_stop_service)
        if config.attach:
            self._attach_sub = node.create_subscription(FleetAttach, self._endpoint("attach"), self._on_attach, 10)
            self._attach_srv = node.create_service(FleetAttachSrv, self._endpoint("attach"), self._on_attach_service)
        if config.execute_action:
            self._execute_action_sub = node.create_subscription(
                FleetExecuteAction,
                self._endpoint("execute_action"),
                self._on_execute_action,
                10,
            )
            self._execute_action_srv = node.create_service(
                FleetExecuteActionSrv,
                self._endpoint("execute_action"),
                self._on_execute_action_service,
            )
        if config.joint_command:
            self._joint_sub = node.create_subscription(
                FleetJointCommand,
                self._endpoint("joint_command"),
                self._on_joint_command,
                10,
            )
            self._joint_srv = node.create_service(
                FleetJointCommandSrv,
                self._endpoint("joint_command"),
                self._on_joint_command_service,
            )

    def post_step(self, stamp=None) -> None:
        """Publish fleet state after a simulation step."""
        if self._state_pub is None:
            return
        sim_time = ros_time_to_seconds(stamp)
        if (
            self._state_publish_rate is not None
            and sim_time is not None
            and sim_time - self._last_state_publish_time < 1.0 / self._state_publish_rate
        ):
            return
        # Query only when this endpoint is due. This avoids a ROS graph query
        # on every bridge tick for lower-rate manager streams, and remains
        # immediately before the expensive state collection/message creation.
        subscription_count = self._state_pub.get_subscription_count()
        if isinstance(subscription_count, int) and subscription_count == 0:
            return

        # Bridge state publication runs inside the simulation POST_STEP event.
        # Once core profiling has produced its first step snapshot, record the
        # endpoint work in its current per-step accumulator.  Keep the fields
        # fleet-wide: manager endpoints are deliberately summed rather than
        # creating an unbounded profiling field for each manager name.
        profile_enabled = self._sim_core.last_profiling is not None
        if profile_enabled:
            collect_started_ns = time.perf_counter_ns()
        # FleetStateProvider interprets ``names=None`` as an all-Agent query.
        states = self.state_provider.get_states_3d(names=self._state_names)
        if profile_enabled:
            self._record_profile("fleet_state_collect", collect_started_ns)
            message_started_ns = time.perf_counter_ns()
        msg = fleet_state_to_msg(states, stamp=stamp, xy_offset=self._rmf_frame_offset)
        if profile_enabled:
            self._record_profile("fleet_state_message", message_started_ns)
            publish_started_ns = time.perf_counter_ns()
        publish_ns = time.monotonic_ns()
        self._state_pub.publish(msg)
        if profile_enabled:
            self._record_profile("fleet_state_publish", publish_started_ns)
        if sim_time is not None:
            self._last_state_publish_time = sim_time
        if self._transport_probe_pub is not None:
            self._publish_transport_timing(
                channel=self._endpoint("states"),
                source_sim_time=stamp,
                source_monotonic_ns=publish_ns,
                item_count=len(msg.robots),
                payload_bytes=len(serialize_message(msg)),
            )

    def _record_profile(self, name: str, started_ns: int) -> None:
        """Add a bridge sub-phase to the active core profiling step."""
        self._sim_core.record_profiling(name, (time.perf_counter_ns() - started_ns) / 1_000_000)

    def _endpoint(self, name: str) -> str:
        return fleet_endpoint(self._namespace, name)

    def destroy(self) -> None:
        """Destroy ROS entities created by this wrapper."""
        for attr, destroy in (
            ("_state_pub", self.node.destroy_publisher),
            ("_navigate_sub", self.node.destroy_subscription),
            ("_navigate_srv", self.node.destroy_service),
            ("_stop_sub", self.node.destroy_subscription),
            ("_stop_srv", self.node.destroy_service),
            ("_attach_sub", self.node.destroy_subscription),
            ("_attach_srv", self.node.destroy_service),
            ("_execute_action_sub", self.node.destroy_subscription),
            ("_execute_action_srv", self.node.destroy_service),
            ("_joint_sub", self.node.destroy_subscription),
            ("_joint_srv", self.node.destroy_service),
        ):
            entity = getattr(self, attr)
            if entity is not None:
                destroy(entity)
                setattr(self, attr, None)
        if self._transport_probe_pub is not None:
            self.node.destroy_publisher(self._transport_probe_pub)
            self._transport_probe_pub = None

    def _on_navigate(self, msg: FleetNavigate) -> None:
        receive_ns = time.monotonic_ns()
        ack = self._dispatch_navigate(msg)
        self._log_rejections(ack)
        if self._transport_probe_pub is not None:
            self._publish_transport_timing(
                channel=self._endpoint("navigate_topic"),
                correlation_id=msg.command_id,
                source_sim_time=ack.sim_time,
                source_monotonic_ns=receive_ns,
                end_monotonic_ns=time.monotonic_ns(),
                item_count=len(msg.goals_2d) + len(msg.goals_3d),
                payload_bytes=len(serialize_message(msg)),
            )

    def _on_navigate_service(self, request, response):
        receive_ns = time.monotonic_ns()
        response.ack = command_ack_to_msg(self._dispatch_navigate(request))
        if self._transport_probe_pub is not None:
            self._publish_transport_timing(
                channel=self._endpoint("navigate"),
                correlation_id=request.command_id,
                source_sim_time=response.ack.sim_time,
                source_monotonic_ns=receive_ns,
                end_monotonic_ns=time.monotonic_ns(),
                item_count=len(request.goals_2d) + len(request.goals_3d),
                payload_bytes=len(serialize_message(request)),
            )
        return response

    def _publish_transport_timing(
        self,
        *,
        channel: str,
        source_sim_time,
        source_monotonic_ns: int,
        correlation_id: str = "",
        end_monotonic_ns: int = 0,
        item_count: int = 0,
        payload_bytes: int = 0,
    ) -> None:
        if self._transport_probe_pub is None:
            return
        probe = TransportTiming()
        probe.channel = channel
        probe.correlation_id = correlation_id
        probe.sequence = self._transport_probe_sequence
        self._transport_probe_sequence += 1
        if source_sim_time is not None:
            if hasattr(source_sim_time, "sec"):
                probe.source_sim_time.sec = int(source_sim_time.sec)
                probe.source_sim_time.nanosec = int(source_sim_time.nanosec)
            else:
                seconds = int(float(source_sim_time))
                nanoseconds = int(round((float(source_sim_time) - seconds) * 1e9))
                if nanoseconds >= 1_000_000_000:
                    seconds += 1
                    nanoseconds -= 1_000_000_000
                probe.source_sim_time.sec = seconds
                probe.source_sim_time.nanosec = nanoseconds
        probe.source_monotonic_ns = int(source_monotonic_ns)
        probe.end_monotonic_ns = int(end_monotonic_ns)
        probe.item_count = int(item_count)
        probe.payload_bytes = int(payload_bytes)
        self._transport_probe_pub.publish(probe)

    def _dispatch_navigate(self, msg: FleetNavigate) -> PbfCommandAck:
        frame_rejections = _navigation_frame_rejections(msg)
        if frame_rejections:
            return PbfCommandAck(
                command_id=msg.command_id,
                source=_source_from_msg(msg),
                sim_time=float(getattr(self.command_dispatcher.sim_core, "sim_time", 0.0)),
                rejected=frame_rejections,
            )
        return self.command_dispatcher.navigate(
            _navigation_goals_from_msg(msg, xy_offset=self._rmf_frame_offset),
            source=_source_from_msg(msg),
            command_id=msg.command_id or None,
        )

    def _on_stop(self, msg: FleetStop) -> None:
        self._log_rejections(self._dispatch_stop(msg))

    def _on_stop_service(self, request, response):
        response.ack = command_ack_to_msg(self._dispatch_stop(request))
        return response

    def _dispatch_stop(self, msg: FleetStop) -> PbfCommandAck:
        return self.command_dispatcher.stop(
            tuple(msg.names),
            source=_source_from_msg(msg),
            command_id=msg.command_id or None,
        )

    def _on_attach(self, msg: FleetAttach) -> None:
        self._log_rejections(self._dispatch_attach(msg))

    def _on_attach_service(self, request, response):
        response.ack = command_ack_to_msg(self._dispatch_attach(request))
        return response

    def _dispatch_attach(self, msg: FleetAttach) -> PbfCommandAck:
        return self.command_dispatcher.attach(
            _attach_commands_from_msg(msg),
            source=_source_from_msg(msg),
            command_id=msg.command_id or None,
        )

    def _on_execute_action(self, msg: FleetExecuteAction) -> None:
        self._log_rejections(self._dispatch_execute_action(msg))

    def _on_execute_action_service(self, request, response):
        response.ack = command_ack_to_msg(self._dispatch_execute_action(request))
        return response

    def _dispatch_execute_action(self, msg: FleetExecuteAction) -> PbfCommandAck:
        return self.command_dispatcher.execute_action(
            _action_commands_from_msg(msg),
            source=_source_from_msg(msg),
            command_id=msg.command_id or None,
        )

    def _on_joint_command(self, msg: FleetJointCommand) -> None:
        self._log_rejections(self._dispatch_joint_command(msg))

    def _on_joint_command_service(self, request, response):
        response.ack = command_ack_to_msg(self._dispatch_joint_command(request))
        return response

    def _dispatch_joint_command(self, msg: FleetJointCommand) -> PbfCommandAck:
        source = _source_from_msg(msg)
        command_id = msg.command_id or None
        commands, validation_rejected = _joint_commands_from_msg(msg)
        if not commands:
            return PbfCommandAck(
                command_id=msg.command_id,
                source=source,
                sim_time=float(getattr(self.command_dispatcher.sim_core, "sim_time", 0.0)),
                rejected=validation_rejected,
            )
        ack = self.command_dispatcher.joint_command(commands, source=source, command_id=command_id)
        if not validation_rejected:
            return ack
        return PbfCommandAck(
            command_id=ack.command_id,
            source=ack.source,
            sim_time=ack.sim_time,
            accepted_names=ack.accepted_names,
            rejected={**dict(ack.rejected), **validation_rejected},
        )

    def _log_rejections(self, ack: PbfCommandAck) -> None:
        if not ack.rejected:
            return
        details = ", ".join(f"{name}: {reason}" for name, reason in ack.rejected.items())
        self.node.get_logger().warning(f"Fleet command '{ack.command_id}' rejected targets: {details}")


def fleet_state_to_msg(
    states: Iterable[RobotState3D],
    stamp=None,
    frame_id: str = "odom",
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> FleetState:
    """Convert Python fleet state values into a ROS fleet state message."""
    _require_fleet_msgs()
    msg = FleetState()
    msg.header = Header(frame_id=frame_id)
    if stamp is not None:
        msg.header.stamp = stamp
    msg.robots = [robot_state3d_to_msg(state, xy_offset=xy_offset) for state in states]
    return msg


def robot_state3d_to_msg(state: RobotState3D, xy_offset: tuple[float, float] = (0.0, 0.0)) -> RobotState3DMsg:
    """Convert one 3D robot state into its ROS message encoding."""
    _require_fleet_msgs()
    msg = RobotState3DMsg()
    msg.name = state.name
    msg.object_id = int(state.object_id)
    # Generated ROS Python messages eagerly allocate their nested defaults.
    # Populate those objects instead of constructing a second Pose/Twist tree.
    position = msg.pose.position
    position.x = state.position[0] + xy_offset[0]
    position.y = state.position[1] + xy_offset[1]
    position.z = state.position[2]
    orientation = msg.pose.orientation
    orientation.x = state.orientation[0]
    orientation.y = state.orientation[1]
    orientation.z = state.orientation[2]
    orientation.w = state.orientation[3]
    linear = msg.twist.linear
    linear.x = state.linear_velocity[0]
    linear.y = state.linear_velocity[1]
    linear.z = state.linear_velocity[2]
    angular = msg.twist.angular
    angular.x = state.angular_velocity[0]
    angular.y = state.angular_velocity[1]
    angular.z = state.angular_velocity[2]
    msg.is_moving = bool(state.is_moving)
    msg.has_battery_soc = state.battery_soc is not None
    msg.battery_soc = float(state.battery_soc or 0.0)
    msg.has_is_charging = state.is_charging is not None
    msg.is_charging = bool(state.is_charging)
    return msg


def snapshot_to_fleet_state_msg(
    snapshot,
    stamp=None,
    frame_id: str = "odom",
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> FleetState:
    """Convert a state-plane :class:`~pybullet_fleet.state_plane.Snapshot` into a FleetState message.

    Same encoding as :func:`fleet_state_to_msg`, built from the snapshot's
    arrays: one ``RobotState3D`` per row, in row order.
    """
    _require_fleet_msgs()
    msg = FleetState()
    msg.header = Header(frame_id=frame_id)
    if stamp is not None:
        msg.header.stamp = stamp
    n = snapshot.n_rows
    pos = snapshot.pos.tolist()
    orn = snapshot.orn.tolist()
    lin = snapshot.lin_vel.tolist()
    ang = snapshot.ang_vel.tolist()
    moving = snapshot.is_moving.tolist()
    battery = snapshot.battery_soc.tolist()
    charging = snapshot.is_charging.tolist()
    ids = snapshot.object_id.tolist()
    dx, dy = xy_offset
    robots = []
    for i in range(n):
        robot = RobotState3DMsg()
        robot.name = snapshot.names[i] if i < len(snapshot.names) else ""
        robot.object_id = int(ids[i])
        position = robot.pose.position
        position.x = pos[i][0] + dx
        position.y = pos[i][1] + dy
        position.z = pos[i][2]
        orientation = robot.pose.orientation
        orientation.x, orientation.y, orientation.z, orientation.w = orn[i]
        linear = robot.twist.linear
        linear.x, linear.y, linear.z = lin[i]
        angular = robot.twist.angular
        angular.x, angular.y, angular.z = ang[i]
        robot.is_moving = bool(moving[i])
        soc = battery[i]
        has_battery = not (soc != soc)  # NaN marks a robot without a battery
        robot.has_battery_soc = has_battery
        robot.battery_soc = float(soc) if has_battery else 0.0
        robot.has_is_charging = has_battery
        robot.is_charging = bool(charging[i])
        robots.append(robot)
    msg.robots = robots
    return msg


def command_ack_to_msg(ack: PbfCommandAck) -> CommandAck:
    """Convert a Python command acknowledgement into a ROS message."""
    _require_fleet_msgs()
    msg = CommandAck()
    msg.command_id = ack.command_id
    msg.source = ack.source
    msg.sim_time = ack.sim_time
    msg.accepted_names = list(ack.accepted_names)
    msg.rejected_names = list(ack.rejected.keys())
    msg.reject_reasons = list(ack.rejected.values())
    return msg


def _navigation_goals_from_msg(
    msg: FleetNavigate,
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> list[RobotGoalCommand2D | RobotGoalCommand3D]:
    return [_goal_2d_from_msg(goal, xy_offset=xy_offset) for goal in msg.goals_2d] + [
        _goal_3d_from_msg(goal, xy_offset=xy_offset) for goal in msg.goals_3d
    ]


def _navigation_frame_rejections(msg: FleetNavigate) -> dict[str, str]:
    frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
    if frame_id in ("", "odom"):
        return {}
    names = [goal.name for goal in msg.goals_2d] + [goal.name for goal in msg.goals_3d]
    return {name: f"unsupported frame_id {frame_id!r}; expected 'odom'" for name in names}


def _attach_commands_from_msg(msg: FleetAttach) -> list[RobotAttachCommand]:
    commands = []
    for command in msg.commands:
        offset = command.offset
        offset_orientation = [
            float(offset.orientation.x),
            float(offset.orientation.y),
            float(offset.orientation.z),
            float(offset.orientation.w),
        ]
        if all(v == 0.0 for v in offset_orientation):
            offset_orientation = [0.0, 0.0, 0.0, 1.0]
        search_radius = float(command.search_radius)
        if search_radius <= 0.0:
            search_radius = DEFAULT_ATTACH_SEARCH_RADIUS
        commands.append(
            RobotAttachCommand(
                name=command.name,
                attach=bool(command.attach),
                object_name=command.object_name,
                parent_link=command.parent_link or "base_link",
                offset=PbfPose(
                    position=[
                        float(offset.position.x),
                        float(offset.position.y),
                        float(offset.position.z),
                    ],
                    orientation=offset_orientation,
                ),
                search_radius=search_radius,
            )
        )
    return commands


def _action_commands_from_msg(msg: FleetExecuteAction) -> list[RobotActionCommand]:
    return [
        RobotActionCommand(
            name=command.name,
            action_type=command.action_type,
            action_params_json=command.action_params_json,
            command_id=msg.command_id or None,
        )
        for command in msg.commands
    ]


def _goal_2d_from_msg(msg: RobotGoal2D, xy_offset: tuple[float, float] = (0.0, 0.0)) -> RobotGoalCommand2D:
    return RobotGoalCommand2D(
        name=msg.name,
        position=(float(msg.position[0]) - xy_offset[0], float(msg.position[1]) - xy_offset[1]),
        yaw=float(msg.yaw),
        z=float(msg.z),
    )


def _goal_3d_from_msg(msg: RobotGoal3D, xy_offset: tuple[float, float] = (0.0, 0.0)) -> RobotGoalCommand3D:
    return RobotGoalCommand3D(
        name=msg.name,
        position=(
            float(msg.pose.position.x) - xy_offset[0],
            float(msg.pose.position.y) - xy_offset[1],
            float(msg.pose.position.z),
        ),
        orientation=(
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        ),
    )


def _joint_commands_from_msg(
    msg: FleetJointCommand,
) -> tuple[list[RobotJointPositionsCommand | RobotNamedJointPositionsCommand], dict[str, str]]:
    commands: list[RobotJointPositionsCommand | RobotNamedJointPositionsCommand] = [
        _joint_positions_from_msg(command) for command in msg.joint_position_commands
    ]
    rejected: dict[str, str] = {}
    for command in msg.named_joint_position_commands:
        try:
            commands.append(_named_joint_positions_from_msg(command))
        except ValueError as exc:
            rejected[command.name] = str(exc)
    return commands, rejected


def _joint_positions_from_msg(msg: RobotJointPositionsCommandMsg) -> RobotJointPositionsCommand:
    return RobotJointPositionsCommand(name=msg.name, positions=tuple(float(value) for value in msg.positions))


def _named_joint_positions_from_msg(msg: RobotNamedJointPositionsCommandMsg) -> RobotNamedJointPositionsCommand:
    if len(msg.joint_names) != len(msg.positions):
        raise ValueError(
            f"Named joint command for '{msg.name}' has {len(msg.joint_names)} joint names "
            f"but {len(msg.positions)} positions"
        )
    return RobotNamedJointPositionsCommand(
        name=msg.name,
        positions={name: float(position) for name, position in zip(msg.joint_names, msg.positions)},
    )


def _source_from_msg(msg) -> str:
    return msg.source or "ros2"


def _node_frame_offset(node) -> tuple[float, float]:
    value = getattr(node, "rmf_frame_offset", (0.0, 0.0))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return (0.0, 0.0)
