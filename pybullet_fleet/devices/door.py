"""Door device — Agent subclass with URDF joint control."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, cast

from pybullet_fleet.action import JointAction
from pybullet_fleet.agent import Agent, AgentSpawnParams, IKParams  # noqa: F401 - resolves inherited API type hints
from pybullet_fleet.controller import Controller  # noqa: F401 - resolves inherited API type hints
from pybullet_fleet.controller_params import ControllerParams  # noqa: F401 - resolves inherited API type hints
from pybullet_fleet.logging_utils import get_lazy_logger
from pybullet_fleet.types import ActionStatus, DoorState, _normal_2d_warning_already_given

logger = get_lazy_logger(__name__)


@dataclass
class DoorParams(AgentSpawnParams):
    """Spawn parameters for a Door device.

    Extends AgentSpawnParams with door-specific joint targets.

    Attributes:
        open_positions: Joint name → target angle/position for the "open" state.
            Required — ``None`` raises TypeError in ``__post_init__``.
        close_positions: Joint name → target angle/position for the "closed" state.
    """

    open_positions: Optional[Dict[str, float]] = None
    close_positions: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.open_positions is None:
            raise TypeError("DoorParams requires open_positions (Dict[str, float])")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DoorParams":
        """Create DoorParams from a config dict.

        Delegates base-field parsing to ``AgentSpawnParams.from_dict``
        and adds door-specific fields (``open_positions``, ``close_positions``).
        """
        with _normal_2d_warning_already_given():  # cls(...) below warns about a deprecated collision mode
            base = AgentSpawnParams.from_dict(config)
        base_kwargs = {f.name: getattr(base, f.name) for f in fields(base)}
        return cls(
            **base_kwargs,
            open_positions=config.get("open_positions"),
            close_positions=config.get("close_positions", {}),
        )


class Door(Agent):
    """Door controlled via URDF revolute or prismatic joint.

    Uses JointAction for open/close transitions. Door state is derived from
    the current JointAction status and joint positions.

    Config (YAML)::

        type: door
        urdf_path: robots/door_hinge.urdf
        open_positions: {hinge: 1.57}
        close_positions: {hinge: 0.0}
    """

    _spawn_params_cls = DoorParams
    _entity_type_name = "door"

    # Instance attributes populated in from_params (declared for type checking).
    _open_positions: Dict[str, float]
    _close_positions: Dict[str, float]

    @classmethod
    def from_params(cls, spawn_params: "DoorParams", sim_core=None) -> "Door":
        """Create a Door from DoorParams.

        Raises:
            TypeError: If *spawn_params* is not a ``DoorParams`` instance
                or ``open_positions`` is missing.
        """
        if not isinstance(spawn_params, DoorParams):
            raise TypeError(f"Door.from_params requires DoorParams, got {type(spawn_params).__name__}")
        agent = cast("Door", super().from_params(spawn_params, sim_core))
        agent._open_positions = spawn_params.open_positions  # type: ignore[assignment]
        agent._close_positions = spawn_params.close_positions
        return agent

    @property
    def door_state(self) -> DoorState:
        """Derived door state."""
        action = self.get_current_action()
        if action and isinstance(action, JointAction) and action.status == ActionStatus.IN_PROGRESS:
            # Determine direction from action targets
            if action.target_joint_positions == self._open_positions:
                return DoorState.OPENING
            return DoorState.CLOSING
        if self._open_positions and self.are_joints_at_targets(self._open_positions):
            return DoorState.OPEN
        return DoorState.CLOSED

    def request_open(self) -> None:
        """Open the door (enqueue JointAction to open positions)."""
        if self.door_state in (DoorState.CLOSED, DoorState.CLOSING):
            self.clear_actions()
            self.add_action(JointAction(target_joint_positions=self._open_positions))

    def request_close(self) -> None:
        """Close the door (enqueue JointAction to close positions)."""
        if self.door_state in (DoorState.OPEN, DoorState.OPENING):
            self.clear_actions()
            self.add_action(JointAction(target_joint_positions=self._close_positions))
