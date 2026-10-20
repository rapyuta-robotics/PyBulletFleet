"""Transport-neutral fleet command and acknowledgement types."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from pybullet_fleet.geometry import Pose


FLEET_COMMAND_EVENT = "fleet_command"
DEFAULT_ATTACH_SEARCH_RADIUS = 0.5

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotGoalCommand2D:
    """Planar navigation command for one robot.

    ``z`` is the goal's height. ``None`` (the default) keeps the robot's
    present height, so a planar goal never moves it up or down; a goal at
    another height is a 3D move, which a moving robot can only continue into
    along its line of motion (see :meth:`Agent.set_path`).
    """

    name: str
    position: Vec2
    yaw: float = 0.0
    z: float | None = None
    command_id: str | None = None

    def to_pose(self, current_z: float = 0.0) -> Pose:
        """Convert this command to a simulation ``Pose``.

        ``current_z`` is the robot's present height, used when ``z`` is ``None``.
        """
        z = current_z if self.z is None else self.z
        return Pose.from_yaw(self.position[0], self.position[1], z, self.yaw)


@dataclass(frozen=True)
class RobotGoalCommand3D:
    """3D navigation command for one robot."""

    name: str
    position: Vec3
    orientation: Quat = (0.0, 0.0, 0.0, 1.0)
    command_id: str | None = None

    def to_pose(self, current_z: float = 0.0) -> Pose:
        """Convert this command to a simulation ``Pose``.

        ``current_z`` is accepted for symmetry with
        :meth:`RobotGoalCommand2D.to_pose` and unused: a 3D goal names its height.
        """
        return Pose(position=list(self.position), orientation=list(self.orientation))


@dataclass(frozen=True)
class RobotJointPositionsCommand:
    """Ordered joint target command for one robot."""

    name: str
    positions: tuple[float, ...]


@dataclass(frozen=True)
class RobotNamedJointPositionsCommand:
    """Named joint target command for one robot."""

    name: str
    positions: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))


@dataclass(frozen=True)
class RobotAttachCommand:
    """Attach or detach a simulation object for one robot."""

    name: str
    attach: bool
    object_name: str = ""
    parent_link: str = "base_link"
    offset: Pose = field(default_factory=lambda: Pose.from_xyz(0.0, 0.0, 0.0))
    search_radius: float = DEFAULT_ATTACH_SEARCH_RADIUS


@dataclass(frozen=True)
class RobotActionCommand:
    """Generic PyBulletFleet action command for one robot."""

    name: str
    action_type: str
    action_params_json: str = ""
    command_id: str | None = None


@dataclass(frozen=True)
class CommandAck:
    """Result of accepting or rejecting a fleet command."""

    command_id: str
    source: str
    sim_time: float
    accepted_names: tuple[str, ...] = field(default_factory=tuple)
    rejected: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected", MappingProxyType(dict(self.rejected)))

    @property
    def ok(self) -> bool:
        """Whether at least one target was accepted and none were rejected."""
        return bool(self.accepted_names) and not self.rejected


@dataclass(frozen=True)
class CommandEvent:
    """Command trace event emitted before mutating simulation state."""

    command_id: str
    source: str
    sim_time: float
    command_type: str
    target_names: tuple[str, ...]
    accepted_names: tuple[str, ...]
    rejected: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected", MappingProxyType(dict(self.rejected)))
