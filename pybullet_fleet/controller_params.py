"""Shared parameter dataclass for kinematic controllers.

``ControllerParams`` carries all kinematic limits and behaviour flags that are
common to every :class:`KinematicController` subclass (omni, differential, …).
It is stored on the per-agent controller as ``controller.params`` and is the
single authoritative source consulted by both per-agent control loops and the
batched controllers (``BatchOmniController`` / ``BatchDifferentialController``).

Design rationale
----------------
* **Single source of truth** — eliminates duplication of ``max_linear_vel``,
  ``navigation_2d``, etc. across :class:`AgentSpawnParams`, :class:`Agent`, and
  the controllers themselves.
* **Per-axis kinematic limits** — ``max_*_vel/accel`` accept either a scalar
  (uniform limit, magnitude clamp semantics) or a 3-element sequence
  (per-axis limit, per-axis clamp / direction-projected TPI). Velocity-mode
  clamping and pose-mode TPI both branch on the runtime form.
* **Symmetry with batched controllers** — batch controllers read
  ``agent.controller.params`` during ``register_agent`` and pack the fields
  into ``(N,)`` arrays. Per-agent and batch share the exact same dataclass.
* **Pure data, no logic** — kept separate from ``controller.py`` (which holds
  the control-loop logic) to avoid bloating that module and to keep imports
  cycle-free between ``agent.py``, ``controller.py`` and ``controllers/*``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

import numpy as np

from pybullet_fleet._defaults import CONTROLLER as _CTRL_D
from pybullet_fleet.geometry import (
    ScalarOrAxes,
    as_axes,
    clamp_vec_to_limit,
    is_scalar,
    projected_axis_limit,
)
from pybullet_fleet.types import GoalReplacement, MovementDirection


@dataclass
class ControllerParams:
    """Kinematic limits and behaviour flags shared by all ``KinematicController`` subclasses.

    All numeric fields default to ``None`` ("not explicitly set").  Controllers
    resolve ``None`` to the framework default from
    :data:`pybullet_fleet._defaults.CONTROLLER` at the point of use, so a
    ``ControllerParams()`` with all-``None`` fields behaves identically to the
    old all-default instance.  The ``None`` sentinel enables fleet-level
    defaults in :class:`~pybullet_fleet.agent_manager.AgentManager`: only
    fields that are still ``None`` on an agent are filled in from the fleet
    config; explicitly set values are never overwritten.

    Attributes:
        max_linear_vel: Maximum linear velocity in m/s. Scalar = magnitude
            clamp (Euclidean norm). 3-element sequence ``[vx, vy, vz]`` =
            per-axis limit in body frame. ``None`` → framework default.
        max_angular_vel: Maximum angular velocity in rad/s. Same semantics
            as ``max_linear_vel``. ``None`` → framework default.
        max_linear_accel: Maximum linear acceleration in m/s² (used by TPI).
            ``None`` → framework default.
        max_linear_decel: Maximum linear deceleration in m/s² (used by TPI).
            Same scalar/per-axis semantics as ``max_linear_accel``.
            ``None`` → mirror ``max_linear_accel`` (symmetric trapezoid),
            which is the behaviour of every release before this field
            existed. Set it for vehicles that brake harder than they
            accelerate.
        max_angular_accel: Maximum angular acceleration in rad/s² (used by TPI).
            ``None`` → framework default.
        cmd_vel_timeout: Watchdog timeout in seconds; ``0.0`` or ``None``
            disables the watchdog. ``None`` → framework default (``0.0``).
        navigation_2d: When ``True``, pose-control trajectories preserve the
            current ``z`` coordinate (XY-plane navigation). ``None`` = not
            explicitly set (treated as ``False`` by all controllers).
        default_direction: Default movement direction for ``set_path``, a
            :class:`~pybullet_fleet.types.MovementDirection` or its string
            value. Only meaningful for differential-drive controllers.
        goal_replacement: What a moving agent does with a new path it cannot
            reach from its current motion within these limits -- a
            :class:`~pybullet_fleet.types.GoalReplacement` or its string value
            (``"brake"``, ``"reject"``, ``"wait_complete"``, ``"immediate"``).
            ``None`` → framework default, ``"brake"`` unless changed (see
            :data:`pybullet_fleet._defaults.CONTROLLER`, overridable with the
            ``PBF_CONTROLLER_GOAL_REPLACEMENT`` environment variable).
    """

    max_linear_vel: Optional[ScalarOrAxes] = None
    max_angular_vel: Optional[ScalarOrAxes] = None
    max_linear_accel: Optional[ScalarOrAxes] = None
    max_linear_decel: Optional[ScalarOrAxes] = None
    max_angular_accel: Optional[ScalarOrAxes] = None
    cmd_vel_timeout: Optional[float] = None
    navigation_2d: Optional[bool] = None
    default_direction: MovementDirection = MovementDirection.FORWARD
    goal_replacement: Optional[GoalReplacement] = None

    def __setattr__(self, name: str, value: Any) -> None:
        # The enum fields take the enum or its string value on every
        # assignment -- construction (directly, from_dict, YAML) and later,
        # e.g. ``agent.controller_params.goal_replacement = "reject"`` -- and
        # are stored as the enum; an unknown value fails there, not later.
        # as_config() writes the values.
        if name == "default_direction":
            value = _parse_enum(MovementDirection, value, name)
        elif name == "goal_replacement" and value is not None:
            value = parse_goal_replacement(value)
        super().__setattr__(name, value)

    def as_config(self) -> Dict[str, Any]:
        """The fields as plain data -- enums as their string values -- for
        YAML, checkpoints and :meth:`from_dict`."""
        config = asdict(self)
        for name, value in config.items():
            if isinstance(value, Enum):
                config[name] = value.value
        return config

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ControllerParams":
        """Create from a dict, ignoring unknown keys.

        Unknown keys are silently dropped so that subclass-specific extras
        (e.g. ``wheel_separation``) can coexist with shared params inside a
        single YAML block. Enum fields take their string values.
        """
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in config.items() if k in valid}
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # per-axis helpers
    # ------------------------------------------------------------------

    def _eff_linear_vel(self) -> ScalarOrAxes:
        return self.max_linear_vel if self.max_linear_vel is not None else _CTRL_D["max_linear_vel"]

    def _eff_angular_vel(self) -> ScalarOrAxes:
        return self.max_angular_vel if self.max_angular_vel is not None else _CTRL_D["max_angular_vel"]

    def _eff_linear_accel(self) -> ScalarOrAxes:
        return self.max_linear_accel if self.max_linear_accel is not None else _CTRL_D["max_linear_accel"]

    def _eff_linear_decel(self) -> ScalarOrAxes:
        # Unset means "brakes as hard as it accelerates", so this mirrors the
        # effective acceleration rather than carrying its own framework default.
        return self.max_linear_decel if self.max_linear_decel is not None else self._eff_linear_accel()

    def _eff_angular_accel(self) -> ScalarOrAxes:
        return self.max_angular_accel if self.max_angular_accel is not None else _CTRL_D["max_angular_accel"]

    def _eff_goal_replacement(self) -> GoalReplacement:
        if self.goal_replacement is not None:
            return self.goal_replacement
        return parse_goal_replacement(_CTRL_D["goal_replacement"])

    def _eff_cmd_vel_timeout(self) -> float:
        return self.cmd_vel_timeout if self.cmd_vel_timeout is not None else _CTRL_D["cmd_vel_timeout"]

    def linear_vel_along_direction(self, direction_unit: np.ndarray) -> float:
        """Maximum scalar speed along *direction_unit* respecting per-axis limits."""
        return projected_axis_limit(self._eff_linear_vel(), direction_unit)

    def linear_accel_along_direction(self, direction_unit: np.ndarray) -> float:
        """Maximum scalar acceleration along *direction_unit* respecting per-axis limits."""
        return projected_axis_limit(self._eff_linear_accel(), direction_unit)

    def linear_decel_along_direction(self, direction_unit: np.ndarray) -> float:
        """Maximum scalar deceleration along *direction_unit* respecting per-axis limits."""
        return projected_axis_limit(self._eff_linear_decel(), direction_unit)

    def clamp_linear_vec(self, vec: np.ndarray) -> np.ndarray:
        """Clamp a body-frame linear velocity vector ``(3,)``."""
        return clamp_vec_to_limit(vec, self._eff_linear_vel())

    def clamp_angular_vec(self, vec: np.ndarray) -> np.ndarray:
        """Clamp a body-frame angular velocity vector ``(3,)``."""
        return clamp_vec_to_limit(vec, self._eff_angular_vel())

    def scalar_max_linear_vel(self) -> float:
        """Forward-axis scalar ``max_linear_vel`` (for differential drive)."""
        v = self._eff_linear_vel()
        return float(v) if is_scalar(v) else float(as_axes(v)[0])  # type: ignore[arg-type]

    def scalar_max_linear_accel(self) -> float:
        """Forward-axis scalar ``max_linear_accel`` (for differential drive)."""
        a = self._eff_linear_accel()
        return float(a) if is_scalar(a) else float(as_axes(a)[0])  # type: ignore[arg-type]

    def scalar_max_linear_decel(self) -> float:
        """Forward-axis scalar ``max_linear_decel`` (for differential drive)."""
        d = self._eff_linear_decel()
        return float(d) if is_scalar(d) else float(as_axes(d)[0])  # type: ignore[arg-type]

    def scalar_max_angular_vel(self) -> float:
        """Yaw-axis scalar ``max_angular_vel`` (for differential drive)."""
        w = self._eff_angular_vel()
        return float(w) if is_scalar(w) else float(as_axes(w)[2])  # type: ignore[arg-type]

    def scalar_max_angular_accel(self) -> float:
        """Yaw-axis scalar ``max_angular_accel`` (for differential drive)."""
        a = self._eff_angular_accel()
        return float(a) if is_scalar(a) else float(as_axes(a)[2])  # type: ignore[arg-type]


_EnumT = TypeVar("_EnumT", bound=Enum)


def _parse_enum(kind: Type[_EnumT], value: Any, name: str) -> _EnumT:
    """``kind`` from itself or its string value; ``ValueError`` naming the accepted ones."""
    try:
        return kind(value)
    except ValueError:
        accepted = ", ".join(repr(member.value) for member in kind)
        raise ValueError(f"Unknown {name} {value!r}; expected one of {accepted}") from None


def parse_goal_replacement(value: Any) -> GoalReplacement:
    """A :class:`~pybullet_fleet.types.GoalReplacement` from itself or its string value.

    Raises:
        ValueError: for any other value, naming the accepted ones.
    """
    return _parse_enum(GoalReplacement, value, "goal_replacement")


__all__: List[str] = ["ControllerParams", "ScalarOrAxes", "parse_goal_replacement"]
