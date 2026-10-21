"""Transport-neutral fleet state and command API.

The classes in this module intentionally avoid ROS imports. ROS bridge wrappers,
RMF plugin paths, and future replay tools can share these Python data structures
and call sites without depending on a specific transport.
"""

from __future__ import annotations

from collections import Counter
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from pybullet_fleet.commands import (
    DEFAULT_ATTACH_SEARCH_RADIUS,
    FLEET_COMMAND_EVENT,
    CommandAck,
    CommandEvent,
    Quat,
    RobotActionCommand,
    RobotAttachCommand,
    RobotGoalCommand2D,
    RobotGoalCommand3D,
    RobotJointPositionsCommand,
    RobotNamedJointPositionsCommand,
    Vec3,
)
from pybullet_fleet.states import RobotState2D, RobotState3D
from pybullet_fleet.logging_utils import get_lazy_logger

logger = get_lazy_logger(__name__)


class FleetStateProvider:
    """Generate deterministic fleet state values from a simulation core."""

    def __init__(self, sim_core: Any) -> None:
        self.sim_core = sim_core

    def get_states(self) -> list[RobotState3D]:
        """Return 3D states for all agents.

        ``get_states()`` uses the more general 3D representation by default.
        Call ``get_states_2d()`` when a planar transport wants a smaller model.
        """
        return self.get_states_3d()

    def get_states_2d(self, names: Iterable[str] | None = None) -> list[RobotState2D]:
        """Return planar states for all or selected agents."""
        selected = _normalize_names(names)
        states = []
        for agent in _iter_agents(self.sim_core):
            name = _agent_name(agent)
            if selected is not None and name not in selected:
                continue
            pose = agent.get_pose()
            velocity = _vec3(getattr(agent, "velocity", (0.0, 0.0, 0.0)))
            states.append(
                RobotState2D(
                    name=name,
                    object_id=_agent_object_id(agent),
                    position=(float(pose.x), float(pose.y)),
                    yaw=float(pose.yaw),
                    linear_velocity=(velocity[0], velocity[1]),
                    angular_velocity=_yaw_rate(getattr(agent, "angular_velocity", 0.0)),
                    is_moving=bool(getattr(agent, "is_moving", False)),
                    battery_soc=_optional_float(getattr(agent, "battery_soc", None)),
                    is_charging=_optional_bool(getattr(agent, "is_charging", None)),
                )
            )
        return states

    def get_states_3d(self, names: Iterable[str] | None = None) -> list[RobotState3D]:
        """Return 3D states for all or selected agents."""
        selected = _normalize_names(names)
        states = []
        for agent in _iter_agents(self.sim_core):
            name = _agent_name(agent)
            if selected is not None and name not in selected:
                continue
            pose = agent.get_pose()
            states.append(
                RobotState3D(
                    name=name,
                    object_id=_agent_object_id(agent),
                    position=_vec3(pose.position),
                    orientation=_quat(pose.orientation),
                    linear_velocity=_vec3(getattr(agent, "velocity", (0.0, 0.0, 0.0))),
                    angular_velocity=_angular_velocity_vec3(getattr(agent, "angular_velocity", (0.0, 0.0, 0.0))),
                    is_moving=bool(getattr(agent, "is_moving", False)),
                    battery_soc=_optional_float(getattr(agent, "battery_soc", None)),
                    is_charging=_optional_bool(getattr(agent, "is_charging", None)),
                )
            )
        return states


class FleetCommandDispatcher:
    """Validate and apply fleet commands against a simulation core."""

    def __init__(
        self, sim_core: Any, *, allowed_names: Iterable[str] | None = None, retain_command_events: bool = True
    ) -> None:
        self.sim_core = sim_core
        self.command_events: list[CommandEvent] = []
        self._retain_command_events = retain_command_events
        self.allowed_names = frozenset(allowed_names) if allowed_names is not None else None
        self._name_index: dict[str, Any | None] = {}
        self.refresh_name_index()

    def refresh_name_index(self) -> None:
        """Rebuild the robot-name index used for command dispatch."""
        index: dict[str, Any | None] = {}
        for agent in _iter_agents(self.sim_core):
            explicit_name = _explicit_agent_name(agent)
            if explicit_name is None:
                continue
            index[explicit_name] = None if explicit_name in index else agent
        self._name_index = index

    def navigate(
        self,
        goals: Iterable[RobotGoalCommand2D | RobotGoalCommand3D],
        *,
        source: str = "python",
        command_id: str | None = None,
    ) -> CommandAck:
        """Apply one or more navigation goals by robot name.

        A target robot provides ``get_pose()``, ``set_goal_pose()`` and
        ``path_rejection()``, as :class:`~pybullet_fleet.agent.Agent` does.
        """
        commands = tuple(goals)
        resolved_id = _resolve_command_id(command_id, commands)
        accepted, rejected = self._resolve_targets(command.name for command in commands)
        by_name = _first_command_by_name(commands)
        # A planar goal without a height keeps the robot's present one.
        goal_poses = {name: by_name[name].to_pose(_current_z(agent)) for name, agent in accepted.items()}
        # A moving robot only takes a goal it can reach within its limits (see
        # Agent.set_path); report the others as rejected before acting.
        for name in list(accepted):
            rejection = accepted[name].path_rejection([goal_poses[name]])
            if rejection is not None:
                rejected[name] = rejection.reason
                del accepted[name]
        ack = self._ack("navigate", resolved_id, source, tuple(command.name for command in commands), accepted, rejected)
        # The command event is emitted before the goals are applied. A handler
        # of it can change a robot so that it then refuses its goal; the event
        # cannot be taken back, but the returned ack and the recorded input
        # only claim the goals that were applied.
        refused: dict[str, str] = {}
        for name in ack.accepted_names:
            agent = accepted[name]
            if agent.set_goal_pose(goal_poses[name]) is False:
                rejection = agent.path_rejection([goal_poses[name]])
                refused[name] = rejection.reason if rejection is not None else "goal refused"
        if refused:
            logger.warning(
                "navigate %s: goals refused after the command event announced them as accepted: %s", resolved_id, refused
            )
            ack = CommandAck(
                command_id=ack.command_id,
                source=ack.source,
                sim_time=ack.sim_time,
                accepted_names=tuple(name for name in ack.accepted_names if name not in refused),
                rejected={**ack.rejected, **refused},
            )
        self._record_effective_input(
            "navigate",
            ack,
            {
                name: {
                    "position": list(goal_poses[name].position),
                    "orientation": list(goal_poses[name].orientation),
                    "auto_approach": True,
                    "final_orientation_align": True,
                }
                for name in ack.accepted_names
            },
        )
        return ack

    def joint_command(
        self,
        commands: Iterable[RobotJointPositionsCommand | RobotNamedJointPositionsCommand],
        *,
        source: str = "python",
        command_id: str | None = None,
    ) -> CommandAck:
        """Apply one or more joint target commands by robot name."""
        command_tuple = tuple(commands)
        resolved_id = _resolve_command_id(command_id, command_tuple)
        accepted, rejected = self._resolve_targets(command.name for command in command_tuple)
        ack = self._ack(
            "joint_command",
            resolved_id,
            source,
            tuple(command.name for command in command_tuple),
            accepted,
            rejected,
        )
        by_name = _first_command_by_name(command_tuple)
        for name in ack.accepted_names:
            command = by_name[name]
            agent = accepted[name]
            if isinstance(command, RobotJointPositionsCommand):
                agent.set_all_joints_targets(list(command.positions))
            else:
                agent.set_joints_targets_by_name(dict(command.positions))
        self._record_effective_input(
            "joint_command",
            ack,
            {
                name: (
                    dict(by_name[name].positions)
                    if isinstance(by_name[name], RobotNamedJointPositionsCommand)
                    else list(by_name[name].positions)
                )
                for name in ack.accepted_names
            },
        )
        return ack

    def stop(
        self,
        names: Iterable[str],
        *,
        source: str = "python",
        command_id: str | None = None,
    ) -> CommandAck:
        """Stop one or more robots by name."""
        target_names = tuple(names)
        resolved_id = command_id or _new_command_id()
        accepted, rejected = self._resolve_targets(target_names)
        ack = self._ack("stop", resolved_id, source, target_names, accepted, rejected)
        for name in ack.accepted_names:
            accepted[name].stop()
        self._record_effective_input("stop", ack, {name: {} for name in ack.accepted_names})
        return ack

    def attach(
        self,
        commands: Iterable[RobotAttachCommand],
        *,
        source: str = "python",
        command_id: str | None = None,
    ) -> CommandAck:
        """Attach or detach objects for one or more robots by name."""
        command_tuple = tuple(commands)
        resolved_id = _resolve_command_id(command_id, command_tuple)
        accepted, rejected = self._resolve_targets(command.name for command in command_tuple)
        by_name = _first_command_by_name(command_tuple)
        targets: dict[str, Any] = {}
        for name, agent in tuple(accepted.items()):
            target, reason = _resolve_attach_target(agent, by_name[name])
            if reason is not None:
                rejected[name] = reason
                del accepted[name]
                continue
            targets[name] = target

        event = self._emit_command_event(
            "attach",
            resolved_id,
            source,
            tuple(command.name for command in command_tuple),
            accepted,
            rejected,
        )
        for name, agent in tuple(accepted.items()):
            command = by_name[name]
            target = targets[name]
            if command.attach:
                ok = agent.attach_object(
                    target,
                    parent_link_index=command.parent_link or "base_link",
                    relative_pose=command.offset,
                )
            else:
                ok = agent.detach_object(target)
            if not ok:
                rejected[name] = "attach mutation failed" if command.attach else "detach mutation failed"
                del accepted[name]

        ack = self._command_ack(event, accepted, rejected)
        self._record_effective_input(
            "attach",
            ack,
            {
                name: {
                    "attach": by_name[name].attach,
                    "object_name": by_name[name].object_name,
                    "parent_link": by_name[name].parent_link,
                    "offset_position": list(by_name[name].offset.position),
                    "offset_orientation": list(by_name[name].offset.orientation),
                }
                for name in ack.accepted_names
            },
        )
        return ack

    def _record_effective_input(self, operation: str, ack: CommandAck, effective: dict) -> None:
        record = getattr(self.sim_core, "_record_state_input", None)
        if record is None:
            return
        record(
            operation,
            {
                "command_id": ack.command_id,
                "accepted": effective,
                "rejected": dict(ack.rejected),
            },
        )

    def execute_action(
        self,
        commands: Iterable[RobotActionCommand],
        *,
        source: str = "python",
        command_id: str | None = None,
    ) -> CommandAck:
        """Queue generic PyBulletFleet actions for one or more robots."""
        command_tuple = tuple(commands)
        resolved_id = _resolve_command_id(command_id, command_tuple)
        accepted, rejected = self._resolve_targets(command.name for command in command_tuple)
        by_name = _first_command_by_name(command_tuple)
        actions: dict[str, Any] = {}
        for name in tuple(accepted.keys()):
            action = _build_action_command(by_name[name])
            if action is None:
                rejected[name] = f"invalid action '{by_name[name].action_type}'"
                del accepted[name]
                continue
            actions[name] = action

        ack = self._ack(
            "execute_action",
            resolved_id,
            source,
            tuple(command.name for command in command_tuple),
            accepted,
            rejected,
        )
        for name in ack.accepted_names:
            accepted[name].add_action(actions[name])
        return ack

    def _resolve_targets(self, names: Iterable[str]) -> tuple[dict[str, Any], dict[str, str]]:
        self.refresh_name_index()
        accepted: dict[str, Any] = {}
        rejected: dict[str, str] = {}
        target_names = tuple(names)
        counts = Counter(target_names)
        for name in target_names:
            if name in accepted or name in rejected:
                continue
            if counts[name] > 1:
                rejected[name] = "duplicate target"
                continue
            if name not in self._name_index:
                rejected[name] = "unknown robot"
                continue
            if self.allowed_names is not None and name not in self.allowed_names:
                rejected[name] = "not managed by this interface"
                continue
            agent = self._name_index[name]
            if agent is None:
                rejected[name] = "ambiguous robot name"
                continue
            accepted[name] = agent
        return accepted, rejected

    def _ack(
        self,
        command_type: str,
        command_id: str,
        source: str,
        target_names: tuple[str, ...],
        accepted: Mapping[str, Any],
        rejected: Mapping[str, str],
    ) -> CommandAck:
        event = self._emit_command_event(command_type, command_id, source, target_names, accepted, rejected)
        return self._command_ack(event, accepted, rejected)

    def _emit_command_event(
        self,
        command_type: str,
        command_id: str,
        source: str,
        target_names: tuple[str, ...],
        accepted: Mapping[str, Any],
        rejected: Mapping[str, str],
    ) -> CommandEvent:
        sim_time = float(getattr(self.sim_core, "sim_time", 0.0))
        accepted_names = tuple(accepted.keys())
        event = CommandEvent(
            command_id=command_id,
            source=source,
            sim_time=sim_time,
            command_type=command_type,
            target_names=target_names,
            accepted_names=accepted_names,
            rejected=dict(rejected),
        )
        if self._retain_command_events:
            self.command_events.append(event)
        events = getattr(self.sim_core, "events", None)
        if events is not None and hasattr(events, "emit"):
            events.emit(FLEET_COMMAND_EVENT, command_event=event)
        return event

    def _command_ack(
        self,
        event: CommandEvent,
        accepted: Mapping[str, Any],
        rejected: Mapping[str, str],
    ) -> CommandAck:
        return CommandAck(
            command_id=event.command_id,
            source=event.source,
            sim_time=event.sim_time,
            accepted_names=tuple(accepted.keys()),
            rejected=dict(rejected),
        )


def _iter_agents(sim_core: Any) -> Sequence[Any]:
    return tuple(getattr(sim_core, "agents", ()))


def _agent_name(agent: Any) -> str:
    name = _explicit_agent_name(agent)
    if name is not None:
        return name
    return f"agent_{getattr(agent, 'object_id', 'unknown')}"


def _explicit_agent_name(agent: Any) -> str | None:
    name = getattr(agent, "name", None)
    if name:
        return str(name)
    return None


def _agent_object_id(agent: Any) -> int:
    return int(getattr(agent, "object_id", -1))


def _normalize_names(names: Iterable[str] | None) -> set[str] | None:
    if names is None:
        return None
    return {str(name) for name in names}


def _vec3(value: Any) -> Vec3:
    if isinstance(value, Real):
        return (float(value), 0.0, 0.0)
    seq = tuple(value)
    if len(seq) == 0:
        return (0.0, 0.0, 0.0)
    if len(seq) == 1:
        return (float(seq[0]), 0.0, 0.0)
    if len(seq) == 2:
        return (float(seq[0]), float(seq[1]), 0.0)
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _quat(value: Any) -> Quat:
    seq = tuple(value)
    if len(seq) != 4:
        raise ValueError(f"Expected quaternion with 4 values, got {len(seq)}")
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


def _yaw_rate(value: Any) -> float:
    if isinstance(value, Real):
        return float(value)
    seq = tuple(value)
    if len(seq) == 0:
        return 0.0
    return float(seq[-1])


def _angular_velocity_vec3(value: Any) -> Vec3:
    if isinstance(value, Real):
        return (0.0, 0.0, float(value))
    return _vec3(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _resolve_command_id(command_id: str | None, commands: Iterable[Any]) -> str:
    if command_id:
        return command_id
    for command in commands:
        item_id = getattr(command, "command_id", None)
        if item_id:
            return str(item_id)
    return _new_command_id()


def _new_command_id() -> str:
    return uuid4().hex


def _current_z(agent: Any) -> float:
    """The robot's present height, for a planar goal that does not name one."""
    return float(agent.get_pose().position[2])


def _first_command_by_name(commands: Iterable[Any]) -> dict[str, Any]:
    by_name: dict[str, Any] = {}
    for command in commands:
        by_name.setdefault(command.name, command)
    return by_name


def _resolve_attach_target(agent: Any, command: RobotAttachCommand) -> tuple[Any | None, str | None]:
    if command.attach:
        if command.object_name:
            obj = _find_sim_object(agent, command.object_name)
            if obj is None:
                return None, f"object '{command.object_name}' not found"
            return obj, None
        finder = getattr(agent, "find_nearest_pickable", None)
        if finder is None:
            return None, "robot does not support nearest pickable search"
        search_radius = float(command.search_radius)
        if search_radius <= 0.0:
            search_radius = DEFAULT_ATTACH_SEARCH_RADIUS
        obj = finder(search_radius=search_radius)
        if obj is None:
            return None, f"no pickable object within {search_radius:.3g}m"
        return obj, None

    attached_getter = getattr(agent, "get_attached_objects", None)
    if attached_getter is None:
        return None, "robot does not expose attached objects"
    attached = tuple(attached_getter())
    if command.object_name:
        for obj in attached:
            if getattr(obj, "name", None) == command.object_name:
                return obj, None
        return None, f"object '{command.object_name}' not attached"
    if not attached:
        return None, "no attached object to detach"
    return attached[0], None


def _find_sim_object(agent: Any, object_name: str) -> Any | None:
    sim_core = getattr(agent, "sim_core", None)
    if sim_core is None:
        return None
    for obj in getattr(sim_core, "sim_objects", ()):
        if getattr(obj, "name", None) == object_name:
            return obj
    return None


def _build_action_command(command: RobotActionCommand) -> Any | None:
    from pybullet_fleet.action_parser import parse_action_goal

    return parse_action_goal(command.action_type, command.action_params_json)
