"""Configuration helpers for fleet and per-robot ROS interfaces.

BridgeNode uses this module to translate explicit ``fleet_api`` /
``per_robot_api`` sections into one normalized configuration object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from pybullet_fleet.config_utils import config_get_bool, config_get_str_list

try:
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
except ImportError:  # pragma: no cover - exercised by core-only configuration tests
    # Keep configuration parsing importable in the normal, non-ROS test
    # environment. A running bridge always imports the rclpy policy enums.
    class ReliabilityPolicy(Enum):
        RELIABLE = auto()
        BEST_EFFORT = auto()

    class HistoryPolicy(Enum):
        KEEP_LAST = auto()
        KEEP_ALL = auto()

    class DurabilityPolicy(Enum):
        VOLATILE = auto()
        TRANSIENT_LOCAL = auto()


@dataclass(frozen=True)
class FleetStateQosConfig:
    """Internal QoS settings used by the FleetState scale-check profiles."""

    reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    history: HistoryPolicy = HistoryPolicy.KEEP_LAST
    depth: int = 10
    durability: DurabilityPolicy = DurabilityPolicy.VOLATILE


FLEET_STATE_QOS_PRESETS = {
    "fleet_state_reliable": FleetStateQosConfig(),
    "fleet_state_best_effort": FleetStateQosConfig(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1),
}


class StateScope(str, Enum):
    """Membership policy for the global ``/fleet/states`` stream."""

    ALL_AGENTS = "all_agents"
    MANAGERS = "managers"


@dataclass(frozen=True)
class FleetApiConfig:
    """Fleet-level API group switches."""

    enabled: bool = False
    states: bool = False
    navigate: bool = False
    joint_command: bool = False
    stop: bool = False
    execute_action: bool = False
    attach: bool = False
    charging: bool = False
    transport_probe: bool = False
    state_scope: StateScope = StateScope.ALL_AGENTS
    state_include_managers: tuple[str, ...] = field(default_factory=tuple)
    # The scale checker writes this internal setting into its temporary bridge
    # config. It is not yet a documented general bridge configuration API.
    state_qos: FleetStateQosConfig = field(default_factory=FleetStateQosConfig)


@dataclass(frozen=True)
class ManagerInterfaceConfig:
    """One named manager's fleet-level ROS endpoint set."""

    manager: str
    states: bool = True
    state_publish_rate: float | None = None
    state_qos: FleetStateQosConfig = field(default_factory=FleetStateQosConfig)
    # Measurement-only setting used by the scale checker. It is intentionally
    # not a documented application-facing endpoint option.
    transport_probe: bool = False
    navigate: bool = False
    joint_command: bool = False
    stop: bool = False
    execute_action: bool = False
    attach: bool = False


@dataclass(frozen=True)
class PerRobotApiConfig:
    """Per-robot API group switches."""

    enabled: bool = True
    state_publishers: bool = True
    tf: bool = True
    command_topics: bool = True
    services: bool = True
    actions: bool = True
    include_robots: tuple[str, ...] = field(default_factory=tuple)
    exclude_robots: tuple[str, ...] = field(default_factory=tuple)

    def robot_enabled(self, name: str) -> bool:
        """Return whether any per-robot API may be created for *name*."""
        if not self.enabled:
            return False
        if self.include_robots and name not in self.include_robots:
            return False
        if name in self.exclude_robots:
            return False
        return self.any_group_enabled

    @property
    def any_group_enabled(self) -> bool:
        """Whether at least one per-robot interface group is enabled."""
        return any(
            (
                self.state_publishers,
                self.tf,
                self.command_topics,
                self.services,
                self.actions,
            )
        )


@dataclass(frozen=True)
class BridgeApiConfig:
    """Normalized bridge API configuration."""

    fleet_api: FleetApiConfig = field(default_factory=FleetApiConfig)
    per_robot_api: PerRobotApiConfig = field(default_factory=PerRobotApiConfig)
    manager_interfaces: tuple[ManagerInterfaceConfig, ...] = field(default_factory=tuple)


def _fleet_api_from_dict(config: Mapping[str, Any], base: FleetApiConfig) -> FleetApiConfig:
    state_qos = _fleet_state_qos_from_dict(config.get("state_qos"), base.state_qos)
    state_include_managers = tuple(config_get_str_list(config, "state_include_managers", list(base.state_include_managers)))
    state_scope = _state_scope_from_dict(config, state_include_managers, base.state_scope)
    return FleetApiConfig(
        enabled=config_get_bool(config, "enabled", base.enabled),
        states=config_get_bool(config, "states", base.states),
        navigate=config_get_bool(config, "navigate", base.navigate),
        joint_command=config_get_bool(config, "joint_command", base.joint_command),
        stop=config_get_bool(config, "stop", base.stop),
        execute_action=config_get_bool(config, "execute_action", base.execute_action),
        attach=config_get_bool(config, "attach", base.attach),
        charging=config_get_bool(config, "charging", base.charging),
        transport_probe=config_get_bool(config, "transport_probe", base.transport_probe),
        state_scope=state_scope,
        state_include_managers=state_include_managers,
        state_qos=state_qos,
    )


def _state_scope_from_dict(config: Mapping[str, Any], managers: tuple[str, ...], default: StateScope) -> StateScope:
    raw_scope = config.get("state_scope")
    if raw_scope is None:
        # Retain the pre-state_scope manager-filter behavior for configurations
        # that already supplied an inclusion list.
        return StateScope.MANAGERS if managers else default
    if not isinstance(raw_scope, str):
        raise ValueError("fleet_api.state_scope must be 'all_agents' or 'managers'")
    try:
        scope = StateScope(raw_scope.strip().lower())
    except ValueError as exc:
        raise ValueError("fleet_api.state_scope must be 'all_agents' or 'managers'") from exc
    if scope is StateScope.MANAGERS and not managers:
        raise ValueError("fleet_api.state_scope 'managers' requires state_include_managers")
    if scope is StateScope.ALL_AGENTS and managers:
        raise ValueError("fleet_api.state_scope 'all_agents' cannot use state_include_managers")
    return scope


def _fleet_state_qos_from_dict(config: Any, base: FleetStateQosConfig) -> FleetStateQosConfig:
    if config is None:
        return base
    if not isinstance(config, Mapping):
        raise ValueError("fleet_api.state_qos must be a mapping")
    preset = str(config.get("preset", "fleet_state_reliable")).strip().lower()
    if preset not in FLEET_STATE_QOS_PRESETS:
        choices = ", ".join(sorted(FLEET_STATE_QOS_PRESETS))
        raise ValueError(f"fleet_api.state_qos.preset must be one of: {choices}")
    selected = FLEET_STATE_QOS_PRESETS[preset]
    reliability = _qos_policy_from_dict(
        config,
        "reliability",
        selected.reliability,
        ReliabilityPolicy,
        (ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT),
    )
    history = _qos_policy_from_dict(
        config,
        "history",
        selected.history,
        HistoryPolicy,
        (HistoryPolicy.KEEP_LAST, HistoryPolicy.KEEP_ALL),
    )
    depth = config.get("depth", selected.depth)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError("fleet_api.state_qos.depth must be a positive integer")
    durability = _qos_policy_from_dict(
        config,
        "durability",
        selected.durability,
        DurabilityPolicy,
        (DurabilityPolicy.VOLATILE, DurabilityPolicy.TRANSIENT_LOCAL),
    )
    return FleetStateQosConfig(reliability=reliability, history=history, depth=depth, durability=durability)


def _qos_policy_from_dict(config: Mapping[str, Any], key: str, default, policy_type, supported):
    raw_value = config.get(key, default)
    if isinstance(raw_value, policy_type):
        value = raw_value
    else:
        try:
            value = policy_type[str(raw_value).strip().upper()]
        except KeyError as exc:
            choices = " or ".join(repr(item.name.lower()) for item in supported)
            raise ValueError(f"fleet_api.state_qos.{key} must be {choices}") from exc
    if value not in supported:
        choices = " or ".join(repr(item.name.lower()) for item in supported)
        raise ValueError(f"fleet_api.state_qos.{key} must be {choices}")
    return value


def _per_robot_api_from_dict(config: Mapping[str, Any], base: PerRobotApiConfig) -> PerRobotApiConfig:
    return PerRobotApiConfig(
        enabled=config_get_bool(config, "enabled", base.enabled),
        state_publishers=config_get_bool(config, "state_publishers", base.state_publishers),
        tf=config_get_bool(config, "tf", base.tf),
        command_topics=config_get_bool(config, "command_topics", base.command_topics),
        services=config_get_bool(config, "services", base.services),
        actions=config_get_bool(config, "actions", base.actions),
        include_robots=tuple(config_get_str_list(config, "include_robots", list(base.include_robots))),
        exclude_robots=tuple(config_get_str_list(config, "exclude_robots", list(base.exclude_robots))),
    )


def _manager_interfaces_from_dict(config: Mapping[str, Any]) -> tuple[ManagerInterfaceConfig, ...]:
    raw_entries = config.get("manager_interfaces", [])
    if not isinstance(raw_entries, list):
        raise ValueError("fleet_api.manager_interfaces must be a list")
    entries = []
    seen = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"fleet_api.manager_interfaces[{index}] must be a mapping")
        manager = raw_entry.get("manager")
        if not isinstance(manager, str) or not manager.strip():
            raise ValueError(f"fleet_api.manager_interfaces[{index}].manager must be a non-empty string")
        manager = manager.strip()
        if "/" in manager:
            raise ValueError(f"fleet_api.manager_interfaces[{index}].manager must be one ROS namespace segment")
        if manager in seen:
            raise ValueError(f"fleet_api.manager_interfaces contains duplicate manager {manager!r}")
        seen.add(manager)
        rate = raw_entry.get("state_publish_rate")
        if rate is not None:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
                raise ValueError(f"fleet_api.manager_interfaces[{index}].state_publish_rate must be positive")
            rate = float(rate)
        entries.append(
            ManagerInterfaceConfig(
                manager=manager,
                states=config_get_bool(raw_entry, "states", True),
                state_publish_rate=rate,
                state_qos=_fleet_state_qos_from_dict(raw_entry.get("state_qos"), FleetStateQosConfig()),
                transport_probe=config_get_bool(raw_entry, "transport_probe", False),
                navigate=config_get_bool(raw_entry, "navigate", False),
                joint_command=config_get_bool(raw_entry, "joint_command", False),
                stop=config_get_bool(raw_entry, "stop", False),
                execute_action=config_get_bool(raw_entry, "execute_action", False),
                attach=config_get_bool(raw_entry, "attach", False),
            )
        )
    return tuple(entries)


def _optional_mapping_section(config: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    if key not in config:
        return None
    value = config[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected '{key}' to be a mapping, got {type(value).__name__}")
    return value


def resolve_bridge_api_config(bridge_config: Mapping[str, Any]) -> BridgeApiConfig:
    """Resolve explicit bridge API config into one object."""
    fleet_api = FleetApiConfig()
    per_robot_api = PerRobotApiConfig()

    explicit_fleet = _optional_mapping_section(bridge_config, "fleet_api")
    if explicit_fleet is not None:
        fleet_api = _fleet_api_from_dict(explicit_fleet, fleet_api)

    explicit_per_robot = _optional_mapping_section(bridge_config, "per_robot_api")
    if explicit_per_robot is not None:
        per_robot_api = _per_robot_api_from_dict(explicit_per_robot, per_robot_api)

    manager_interfaces = _manager_interfaces_from_dict(explicit_fleet) if explicit_fleet is not None else ()
    return BridgeApiConfig(
        fleet_api=fleet_api,
        per_robot_api=per_robot_api,
        manager_interfaces=manager_interfaces,
    )


@dataclass(frozen=True)
class StatePlaneConfig:
    """The ``state_plane`` section: publish fleet state through shared memory for another process.

    When enabled, the bridge writes every step's fleet state to the plane and
    stops publishing ``/clock`` and the global ``/fleet/states`` itself; the
    ``state_publisher`` node reads the plane and publishes both.
    """

    enabled: bool = False
    name: str = "pbf_state"
    mode: str = "every_step"
    slots: int = 8
    wait_timeout: float = 1.0
    capacity: int | None = None


def resolve_state_plane_config(bridge_config: Mapping[str, Any]) -> StatePlaneConfig:
    """Read the optional ``state_plane`` section of a bridge config."""
    section = _optional_mapping_section(bridge_config, "state_plane")
    if section is None:
        return StatePlaneConfig()
    mode = section.get("mode", "every_step")
    if mode not in ("every_step", "latest"):
        raise ValueError(f"state_plane.mode must be 'every_step' or 'latest', got {mode!r}")
    slots = section.get("slots", 8)
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 2:
        raise ValueError(f"state_plane.slots must be an integer of at least 2, got {slots!r}")
    wait_timeout = section.get("wait_timeout", 1.0)
    if isinstance(wait_timeout, bool) or not isinstance(wait_timeout, (int, float)) or wait_timeout <= 0:
        raise ValueError(f"state_plane.wait_timeout must be a positive number of seconds, got {wait_timeout!r}")
    capacity = section.get("capacity")
    if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1):
        raise ValueError(f"state_plane.capacity must be a positive integer, got {capacity!r}")
    return StatePlaneConfig(
        enabled=config_get_bool(section, "enabled", False),
        name=str(section.get("name", "pbf_state")),
        mode=str(mode),
        slots=int(slots),
        wait_timeout=float(wait_timeout),
        capacity=capacity,
    )
