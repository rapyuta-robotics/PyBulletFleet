"""Unified controller hierarchy + built-in controllers + registry.

Each kinematic model (omni, differential) gets one controller class that handles
both velocity commands (``set_velocity``) and pose commands (``set_goal_pose`` /
``set_path``) via an internal mode state machine.

Hierarchy
---------
::

    Controller (ABC)            — pure compute + hooks
    └── KinematicController     — mode state machine + velocity & pose dispatch
        ├── OmniController      — 6-DoF velocity + omnidirectional TPI pose
        └── DifferentialController — unicycle velocity + rotate-then-forward TPI pose

Mode state machine (B3 — input-based + interruption)
-----------------------------------------------------
::

    IDLE ──set_velocity()──→ VELOCITY
    IDLE ──set_goal_pose()──→ POSE
    VELOCITY ──set_goal_pose()──→ POSE  (cancel velocity)
    VELOCITY ──watchdog timeout──→ IDLE
    POSE ──set_velocity()──→ VELOCITY  (cancel pose/TPI)
    POSE ──trajectory complete──→ IDLE

Registry
--------
::

    from pybullet_fleet.controller import create_controller

    ctrl = create_controller("omni")
    ctrl = create_controller("differential", {"max_linear_vel": 1.5})
"""

import inspect
import math
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar

import numpy as np

from pybullet_fleet.geometry import (
    Pose,
    quat_angle_between,
    quat_from_rotvec,
    quat_multiply,
    quat_slerp,
    quat_slerp_precompute,
    rotate_vector,
)
from pybullet_fleet.logging_utils import get_lazy_logger
from pybullet_fleet.plugin_utils import PluginRegistry
from pybullet_fleet.types import ControllerMode, MovementDirection, PosePhase

from two_point_interpolation import TwoPointInterpolation

from pybullet_fleet._tpi import build_tpi, extract_phase_params
from pybullet_fleet._tpi_state import capture_tpi_state, restore_tpi_state
from pybullet_fleet._motion_planning import align_x_axis_quat
from pybullet_fleet.controller_params import ControllerParams

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent

logger = get_lazy_logger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: PluginRegistry["Controller"] = PluginRegistry("controller")


def register_controller(name: str, cls: Type["Controller"]) -> None:
    """Register a controller class under *name*."""
    _registry.register(name, cls)


def create_controller(name: str, config: Optional[dict] = None) -> "Controller":
    """Create a controller by registered *name*, optionally passing *config*.

    Convenience wrapper around :func:`create_controller_from_entry`.

    Raises:
        KeyError: If *name* is not registered.
    """
    return create_controller_from_entry({"type": name, "config": config or {}})


def create_controller_from_entry(entry: Dict[str, Any]) -> "Controller":
    """Create a controller from a YAML-style entry dict.

    Supports two formats:

    - ``{"type": "omni", "config": {"max_linear_vel": 2.0}}`` — registry lookup.
    - ``{"class": "my_pkg.MyController", "config": {...}}`` — dotted path.

    Both formats use a nested ``config`` sub-dict for parameters.

    Returns:
        Instantiated :class:`Controller`.

    Raises:
        KeyError: If ``type`` is not in the registry.
        TypeError: If ``class`` does not resolve to a Controller subclass.
        ValueError: If neither ``type`` nor ``class`` is present.
    """
    cls = _registry.resolve_from_entry(entry, Controller)
    config = entry.get("config", {})
    return cls.from_config(config)


def list_controllers() -> Dict[str, Type["Controller"]]:
    """Return a copy of the controller registry for inspection."""
    return _registry.items()


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


_ControllerT = TypeVar("_ControllerT", bound="Controller")


class Controller(ABC):
    """Agent movement control interface (Strategy pattern).

    Pure base class: only ``compute()`` + lifecycle hooks.
    All built-in controllers are **kinematic**: they update agent pose directly
    via ``agent.set_pose()``.  No physics forces or torques are applied.

    Subclasses with a ``_registry_name`` class attribute are auto-registered
    via ``_registry_name`` class attribute (e.g. ``_registry_name = "omni"``).
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _registry.auto_register_subclass(cls)

    @abstractmethod
    def compute(self, agent: "Agent", dt: float) -> bool:
        """Compute one step of control.

        Returns:
            ``True`` if the agent moved, ``False`` otherwise.
        """
        ...

    def __init__(self, params: Optional["ControllerParams"] = None) -> None:
        from pybullet_fleet.controller_params import ControllerParams

        self.params: ControllerParams = params if params is not None else ControllerParams()

    def _construction_config(self) -> Optional[dict]:
        """Registered type and shared settings, separate from execution state.

        Subclass-specific constructor arguments are outside this descriptor;
        profiles must explicitly support and validate those controllers.
        """
        from pybullet_fleet.controller_params import ControllerParams

        kind = type(self).__dict__.get("_registry_name")
        if kind is None or not isinstance(self.params, ControllerParams):
            return None
        config = {"type": kind, **asdict(self.params)}
        direction = config.get("default_direction")
        if isinstance(direction, MovementDirection):
            config["default_direction"] = direction.value
        return config

    @classmethod
    def from_config(cls: Type[_ControllerT], config: dict) -> "_ControllerT":
        """Create from a config dict.

        The dict may contain:

        * Any field of :class:`~pybullet_fleet.controller_params.ControllerParams`
          (e.g. ``max_linear_vel``, ``navigation_2d``, …). Unknown keys are
          dropped silently by ``ControllerParams.from_dict``.
        * Subclass-specific keys that match the subclass ``__init__`` signature
          (e.g. ``wheel_separation`` for ``DifferentialController``).
        * A ``type`` key naming the controller; ignored here (consumed by
          :func:`create_controller`).
        """
        from pybullet_fleet.controller_params import ControllerParams

        sig = inspect.signature(cls.__init__)
        param_names = {p for p in sig.parameters if p != "self"}
        kwargs: dict = {"params": ControllerParams.from_dict(config)}
        extra_keys = param_names - {"params"}
        kwargs.update({k: v for k, v in config.items() if k in extra_keys})
        return cls(**kwargs)

    def on_stop(self, agent: "Agent") -> None:
        """Called when ``stop()`` is invoked on the agent."""

    def reset(self) -> None:
        """Reset internal controller state."""


# ---------------------------------------------------------------------------
# KinematicController — unified ABC for velocity + pose control
# ---------------------------------------------------------------------------


class KinematicController(Controller):
    """Unified base for kinematic controllers handling both velocity and pose.

    Internal mode state machine:
        IDLE     — no active command
        VELOCITY — processing velocity input (body-frame)
        POSE     — processing goal pose via TPI trajectory

    Input-based mode switching with interruption (B3):
        - ``set_velocity()`` → switch to VELOCITY (cancels POSE if active)
        - ``set_goal_pose()`` / ``set_path()`` → switch to POSE (cancels VELOCITY)
        - Watchdog timeout → IDLE
        - Trajectory complete → IDLE
    """

    def __init__(self, params: Optional[ControllerParams] = None) -> None:
        super().__init__(params)
        self._time_since_last_set_velocity: float = 0.0
        self._velocity_ever_set: bool = False
        self._linear_velocity = np.zeros(3)  # body-frame [vx, vy, vz]
        self._angular_velocity = np.zeros(3)  # body-frame [wx, wy, wz]
        self._mode: ControllerMode = ControllerMode.IDLE

        # Path-following state (owned by controller, not agent)
        self._path: List[Pose] = []
        self._current_waypoint_index: int = 0
        self._goal_pose: Optional[Pose] = None
        self._movement_direction: MovementDirection = MovementDirection.FORWARD
        self._align_final_orientation: bool = False
        self._final_target_orientation: Optional[np.ndarray] = None
        self._is_final_orientation_aligning: bool = False

        # Rotation slerp state (shared by Omni and Differential controllers)
        self._tpi_rotation_angle: Optional[TwoPointInterpolation] = None
        self._slerp_precomp = None
        self._rotation_start_quat: Optional[np.ndarray] = None
        self._rotation_target_quat: Optional[np.ndarray] = None
        self._rotation_total_angle: float = 0.0

        # Forward (straight-line) movement state (shared by Omni and Differential)
        self._tpi_forward: Optional[TwoPointInterpolation] = None
        self._forward_start_pos: Optional[np.ndarray] = None
        self._forward_direction_3d: Optional[np.ndarray] = None
        self._forward_direction_unit: Optional[np.ndarray] = None
        # Body-frame copy of ``_forward_direction_unit`` used only for
        # direction-projected kinematic cap computation in
        # ``_init_forward_trajectory``. ``max_linear_vel`` / ``max_linear_accel``
        # per-axis limits are defined in body frame, so the projection must
        # use the body-frame direction (not world).
        self._forward_direction_unit_body: Optional[np.ndarray] = None
        self._forward_total_distance_3d: float = 0.0

        # Pose phase state machine: ROTATE → FORWARD
        self._pose_phase: PosePhase = PosePhase.FORWARD

    @property
    def default_direction(self) -> MovementDirection:
        """Default movement direction used when ``set_path`` is called without explicit direction."""
        return self.params.default_direction

    @default_direction.setter
    def default_direction(self, value: MovementDirection) -> None:
        if isinstance(value, str):
            value = MovementDirection(value)
        self.params.default_direction = value

    @property
    def mode(self) -> ControllerMode:
        """Current controller mode."""
        return self._mode

    @property
    def goal_pose(self) -> Optional[Pose]:
        """Current goal pose (active waypoint)."""
        return self._goal_pose

    @property
    def current_waypoint_index(self) -> int:
        """Current waypoint index in the path (0-based)."""
        return self._current_waypoint_index

    @property
    def path(self) -> List[Pose]:
        """Current path (list of waypoints). Empty when idle."""
        return self._path

    # -- Template method -----------------------------------------------

    def compute(self, agent: "Agent", dt: float) -> bool:
        """Template: dispatch to velocity or pose handler based on mode."""
        if self._mode == ControllerMode.VELOCITY:
            is_idle = self._check_velocity_timeout(dt) or (
                np.allclose(self._linear_velocity, 0.0) and np.allclose(self._angular_velocity, 0.0)
            )
            if is_idle:
                self._zero_velocity()
                self._mode = ControllerMode.IDLE
                agent._current_velocity[:] = 0.0
                agent._current_angular_velocity = 0.0
                agent._reset_pybullet_velocity()
                return False
            return self._apply_velocity(agent, dt)
        elif self._mode == ControllerMode.POSE:
            return self._apply_pose(agent, dt)
        # IDLE
        return False

    def on_stop(self, agent: "Agent") -> None:
        """Zero all velocity commands, cancel pose, and clear path state."""
        self._zero_velocity()
        self._mode = ControllerMode.IDLE
        self._cancel_pose_internal()
        self._reset_path_state()

    def reset(self) -> None:
        """Reset all internal state."""
        self._zero_velocity()
        self._mode = ControllerMode.IDLE
        self._time_since_last_set_velocity = 0.0
        self._velocity_ever_set = False
        self._cancel_pose_internal()
        self._reset_path_state()

    # -- Velocity interface --------------------------------------------

    @abstractmethod
    def _apply_velocity(self, agent: "Agent", dt: float) -> bool:
        """Apply stored velocity to *agent* for one time step."""
        ...

    def set_velocity(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
    ) -> None:
        """Set velocity command (body frame, 6-DoF). Override in subclasses for clamping."""
        if self._mode == ControllerMode.POSE:
            self._cancel_pose_internal()
        self._mode = ControllerMode.VELOCITY
        self._linear_velocity[:] = [vx, vy, vz]
        self._angular_velocity[:] = [wx, wy, wz]
        self._time_since_last_set_velocity = 0.0
        self._velocity_ever_set = True

    def _zero_velocity(self) -> None:
        """Zero all velocity state."""
        self._linear_velocity[:] = 0.0
        self._angular_velocity[:] = 0.0

    def _check_velocity_timeout(self, dt: float) -> bool:
        """Check if velocity command has timed out."""
        timeout = self.params._eff_cmd_vel_timeout()
        if timeout <= 0.0 or not self._velocity_ever_set:
            return False
        self._time_since_last_set_velocity += dt
        return self._time_since_last_set_velocity > timeout

    # -- Pose interface ------------------------------------------------

    @abstractmethod
    def _init_pose_trajectory(self, agent: "Agent", goal: Pose, **kwargs) -> None:
        """Initialize TPI trajectory for a single waypoint goal."""
        ...

    def cancel_pose(self) -> None:
        """Cancel active pose navigation and switch to IDLE."""
        self._cancel_pose_internal()
        self._mode = ControllerMode.IDLE

    def _cancel_pose_internal(self) -> None:
        """Reset shared pose-mode TPI state (rotation + forward + phase).

        Subclasses should call ``super()._cancel_pose_internal()`` and
        reset their own additional state.
        """
        self._reset_rotation_state()
        self._reset_forward_state()
        self._pose_phase = PosePhase.FORWARD

    def _reset_rotation_state(self) -> None:
        """Zero all rotation/slerp TPI state."""
        self._tpi_rotation_angle = None
        self._slerp_precomp = None
        self._rotation_start_quat = None
        self._rotation_target_quat = None
        self._rotation_total_angle = 0.0

    def _reset_forward_state(self) -> None:
        """Zero all forward (straight-line) TPI state."""
        self._tpi_forward = None
        self._forward_start_pos = None
        self._forward_direction_3d = None
        self._forward_direction_unit = None
        self._forward_direction_unit_body = None
        self._forward_total_distance_3d = 0.0

    def set_path(
        self,
        agent: "Agent",
        path: List[Pose],
        final_orientation_align: bool = True,
        direction: Optional[MovementDirection] = None,
    ) -> None:
        """Set a waypoint path for the agent to follow.

        The controller owns all path-navigation state: waypoint index, movement
        direction, final-orientation alignment.  ``Agent.set_path()`` handles
        validation, auto-approach, and visualisation, then delegates here.

        Args:
            agent: Agent to navigate.
            path: Non-empty list of waypoint Poses.
            final_orientation_align: Rotate to match last waypoint orientation
                after reaching its position.
            direction: ``None`` (use ``default_direction``), ``FORWARD``, ``AUTO``,
                or ``BACKWARD`` (differential only).
        """
        effective_direction = direction if direction is not None else self.params.default_direction
        self._path = list(path)
        self._current_waypoint_index = 0
        self._goal_pose = path[0]
        self._movement_direction = effective_direction
        self._original_direction = effective_direction  # Preserve for AUTO re-evaluation per waypoint
        self._is_final_orientation_aligning = False

        if final_orientation_align and path:
            self._final_target_orientation = np.array(path[-1].orientation)
            self._align_final_orientation = True
        else:
            self._final_target_orientation = None
            self._align_final_orientation = False

        # Start trajectory for first waypoint
        if self._mode == ControllerMode.VELOCITY:
            self._zero_velocity()
        self._mode = ControllerMode.POSE
        self._init_pose_trajectory(agent, path[0], direction=effective_direction)

    def _init_forward_trajectory(self, agent: "Agent", distance: float) -> None:
        """Initialize a single distance TPI for straight-line forward motion.

        Uses direction-projected scalar limits derived from
        :attr:`ControllerParams.max_linear_vel` / ``max_linear_accel``. When
        the limit is scalar (uniform magnitude clamp), it is used directly;
        when per-axis, the largest scalar speed compatible with every axis
        along ``self._forward_direction_unit_body`` (body frame) is used.
        Per-axis caps are defined in body frame, so the projection must use
        the body-frame direction, not the world-frame travel direction.
        After calling this, ``_apply_forward()`` can interpolate position
        along the cached world-frame direction vector.
        """
        t0 = agent.sim_core.sim_time if agent.sim_core is not None else 0.0

        direction_unit_body = self._forward_direction_unit_body
        if direction_unit_body is None or not np.any(direction_unit_body):
            # Zero-length trajectory or no cached direction — fall back to a
            # safe scalar form (forward-axis limits).
            vmax = self.params.scalar_max_linear_vel()
            amax = self.params.scalar_max_linear_accel()
            dmax = self.params.scalar_max_linear_decel()
        else:
            vmax = self.params.linear_vel_along_direction(direction_unit_body)
            amax = self.params.linear_accel_along_direction(direction_unit_body)
            dmax = self.params.linear_decel_along_direction(direction_unit_body)

        self._tpi_forward = build_tpi(p0=0.0, pe=distance, vmax=vmax, accel=amax, t0=t0, decel=dmax)

    def _init_rotation_tpi(
        self,
        agent: "Agent",
        start_quat: np.ndarray,
        target_quat: np.ndarray,
        t0: float,
    ) -> bool:
        """Set up rotation state + TPI for slerp between two quaternions.

        Computes the rotation angle, creates a TPI trajectory, and
        pre-computes the slerp data.  On success the five rotation state
        variables are populated; on failure they are reset.

        Returns:
            True if a rotation trajectory was created.
            False if the angle was negligible (< 1e-6 rad) or TPI creation
            failed — rotation state is reset via ``_reset_rotation_state()``.
        """
        rotation_angle = quat_angle_between(tuple(start_quat), tuple(target_quat))

        if rotation_angle <= 1e-6:
            self._reset_rotation_state()
            return False

        self._rotation_start_quat = np.asarray(start_quat)
        self._rotation_target_quat = np.asarray(target_quat)
        self._rotation_total_angle = float(rotation_angle)

        angular_vel = self.params.scalar_max_angular_vel()
        angular_accel = self.params.scalar_max_angular_accel()

        self._tpi_rotation_angle = build_tpi(p0=0.0, pe=rotation_angle, vmax=angular_vel, accel=angular_accel, t0=t0)
        _, _, t_tot, _, _ = extract_phase_params(self._tpi_rotation_angle)
        if t_tot <= 0.0:
            # build_tpi produced a degenerate zero-duration trajectory (velocity or
            # accel limits are too small to plan the rotation). Treat as failure so
            # the caller can fall back gracefully instead of silently snapping.
            self._reset_rotation_state()
            return False
        self._slerp_precomp = quat_slerp_precompute(self._rotation_start_quat, self._rotation_target_quat)
        return True

    def _apply_pose(self, agent: "Agent", dt: float) -> bool:
        """Unified TPI pose control: ROTATE phase → FORWARD phase.

        **Path-following lifecycle** (driven by ``_handle_waypoint_reached``)::

            For each waypoint:
                [ROTATE] → [FORWARD] → waypoint reached
            After last waypoint (if final-orientation enabled):
                [ROTATE] → (forward distance = 0, completes immediately)

        Omni skips the initial ROTATE (starts at FORWARD); Differential
        always starts at ROTATE to align heading toward the next waypoint.

        The ``_init_pose_trajectory`` sets the initial ``_pose_phase``:

        - Omni (translation): ``FORWARD`` (no rotation)
        - Omni (final alignment): ``ROTATE``, forward distance = 0
        - Differential: ``ROTATE``, then ``FORWARD``
        """
        current_pose = agent.get_pose()
        current_time = agent.sim_core.sim_time if agent.sim_core is not None else 0.0

        if self._pose_phase == PosePhase.ROTATE:
            if self._apply_rotation(agent, current_time, current_pose):
                self._pose_phase = PosePhase.FORWARD
                self._init_forward_trajectory(agent, self._forward_total_distance_3d)
            return True

        if self._pose_phase == PosePhase.FORWARD:
            return self._apply_forward(agent, current_time, current_pose)

        return False

    def _apply_forward(self, agent: "Agent", current_time: float, current_pose: Pose) -> bool:
        """Interpolate position along the straight-line direction using the distance TPI.

        Returns True when trajectory is complete (caller should handle waypoint arrival).
        Returns False if no forward TPI is active.
        """
        if self._tpi_forward is None:
            return False

        distance_traveled, forward_vel, _ = self._tpi_forward.get_point(current_time)
        trajectory_complete = current_time >= self._tpi_forward.get_end_time()

        if trajectory_complete:
            self._handle_waypoint_reached(agent, current_pose)
            return True

        if self._forward_start_pos is None or self._forward_direction_3d is None or self._forward_direction_unit is None:
            return False

        total_distance = self._forward_total_distance_3d
        ratio = distance_traveled / total_distance if total_distance > 1e-6 else 0.0
        new_pos = self._forward_start_pos + self._forward_direction_3d * ratio

        # Orientation: use rotation target if set, else preserve current
        if self._rotation_target_quat is not None:
            orientation = self._rotation_target_quat.tolist()
        else:
            orientation = current_pose.orientation

        np.multiply(self._forward_direction_unit, forward_vel, out=agent._current_velocity)
        agent.set_pose_raw(new_pos.tolist(), orientation)
        agent._current_angular_velocity = 0.0
        return True

    def _apply_rotation(self, agent: "Agent", current_time: float, current_pose: Pose) -> bool:
        """Rotate in place via slerp TPI.

        Returns True when rotation is complete (or no rotation TPI is active).
        On completion, snaps orientation to the target and zeros velocity.
        Returns False while still rotating (intermediate orientation applied).
        """
        slerp_result = self._compute_slerp(current_time)
        if slerp_result is None:
            return True

        orientation, rot_complete = slerp_result
        if rot_complete:
            if self._rotation_target_quat is not None:
                agent.set_pose_raw(current_pose.position, self._rotation_target_quat.tolist())
            agent._current_velocity[:] = 0.0
            agent._current_angular_velocity = 0.0
            return True

        agent.set_pose_raw(current_pose.position, orientation)
        agent._current_velocity[:] = 0.0
        if self._tpi_rotation_angle is not None:
            _, angular_vel, _ = self._tpi_rotation_angle.get_point(current_time)
            agent._current_angular_velocity = angular_vel
        else:
            agent._current_angular_velocity = 0.0
        return False

    def _compute_slerp(self, current_time: float):
        """Compute slerp orientation from rotation TPI.

        Returns:
            ``(orientation_list, rot_complete)`` if rotation TPI is active,
            ``None`` if no rotation TPI is set.
        """
        if self._tpi_rotation_angle is None or self._slerp_precomp is None:
            return None

        rot_angle_val, _, _ = self._tpi_rotation_angle.get_point(current_time)
        rot_complete = current_time >= self._tpi_rotation_angle.get_end_time()

        if self._rotation_total_angle > 1e-9:
            t_fraction = max(0.0, min(rot_angle_val / self._rotation_total_angle, 1.0))
        else:
            t_fraction = 1.0

        new_quat = quat_slerp(
            self._rotation_start_quat,
            self._rotation_target_quat,
            t_fraction,
            self._slerp_precomp,
        )
        return list(new_quat), rot_complete

    def _handle_waypoint_reached(self, agent: "Agent", current_pose: Pose) -> None:
        """Handle waypoint arrival: snap, advance, or complete path.

        Called from ``_apply_pose()`` when the TPI trajectory for the current
        waypoint completes.  Manages the full waypoint-advance → final-orientation
        → path-complete state machine internally.
        """
        # Snap to exact goal position
        snap_position = list(self._goal_pose.position)
        if self.params.navigation_2d:
            snap_position[2] = current_pose.position[2]
        agent.set_pose_raw(snap_position, current_pose.orientation, preserve_velocity=False)

        logger.debug("Reached waypoint at position %s", self._goal_pose.position[:2])

        # Zero velocity on agent
        agent._current_velocity[:] = 0.0
        agent._current_angular_velocity = 0.0
        agent._is_moving = False

        if self._path:
            self._current_waypoint_index += 1
            if self._current_waypoint_index < len(self._path):
                logger.debug("Moving to next waypoint %d", self._current_waypoint_index)
                self._goal_pose = self._path[self._current_waypoint_index]
                agent._is_moving = True
                self._init_pose_trajectory(agent, self._goal_pose, direction=self._original_direction)
            else:
                # All waypoints visited
                if self._align_final_orientation and self._final_target_orientation is not None:
                    logger.info("Path complete, aligning to final orientation...")
                    self._start_final_orientation(agent)
                else:
                    self._complete_path(agent)
        else:
            # Path empty → final orientation alignment just completed
            self._is_final_orientation_aligning = False
            self._goal_pose = None
            self._mode = ControllerMode.IDLE
            agent._reset_pybullet_velocity()
            logger.info("Final orientation alignment complete")

    def _start_final_orientation(self, agent: "Agent") -> None:
        """Begin rotation-in-place after reaching the last waypoint."""
        current_pose = agent.get_pose()
        final_goal = Pose(
            position=current_pose.position,
            orientation=self._final_target_orientation.tolist(),
        )
        self._goal_pose = final_goal
        agent._is_moving = True

        # Polymorphic — each controller knows its rotation strategy
        self._init_final_orientation_trajectory(agent, final_goal)

        self._path = []
        self._align_final_orientation = False
        self._is_final_orientation_aligning = True

    def _init_final_orientation_trajectory(self, agent: "Agent", goal: Pose) -> None:
        """Initialize trajectory for final orientation alignment (rotation in place).

        Called by ``_start_final_orientation()`` when all waypoints have been
        reached and the agent needs to rotate to match the goal orientation.

        Default implementation delegates to ``_init_pose_trajectory``.
        Subclasses may override to use a rotation-only strategy.
        """
        self._init_pose_trajectory(agent, goal)

    def _complete_path(self, agent: "Agent") -> None:
        """Clean up after all waypoints (and optional final orientation) are done."""
        self._path = []
        self._goal_pose = None
        self._is_final_orientation_aligning = False
        self._mode = ControllerMode.IDLE
        agent._reset_pybullet_velocity()
        logger.info("Path complete")

    def _reset_path_state(self) -> None:
        """Reset path-following state to defaults."""
        self._path = []
        self._current_waypoint_index = 0
        self._goal_pose = None
        self._movement_direction = self.params.default_direction
        self._original_direction = self.params.default_direction
        self._align_final_orientation = False
        self._final_target_orientation = None
        self._is_final_orientation_aligning = False

    def capture_navigation_state(self) -> dict:
        """Capture the active kinematic pose path, including waypoint and turn phases."""
        if self._mode is not ControllerMode.POSE or self._goal_pose is None:
            raise ValueError("Checkpoint requires active pose navigation")

        def pose_value(pose: Pose) -> dict:
            return {"position": list(pose.position), "orientation": list(pose.orientation)}

        forward = None
        if self._pose_phase is PosePhase.FORWARD:
            if self._tpi_forward is None or (self._forward_start_pos is None and not self._is_final_orientation_aligning):
                raise ValueError("Forward trajectory is incomplete")
            forward_tpi = capture_tpi_state(self._tpi_forward)
            forward = {
                "origin": self._forward_start_pos.tolist() if self._forward_start_pos is not None else None,
                "t0": forward_tpi["t0"],
                "vmax": forward_tpi["vmax"],
                "accel": forward_tpi["accel"],
                "decel": forward_tpi["decel"],
                "orientation_after_rotation": (
                    self._rotation_target_quat.tolist() if self._rotation_target_quat is not None else None
                ),
            }
        rotation = None
        if self._pose_phase is PosePhase.ROTATE and self._tpi_rotation_angle is not None:
            if self._rotation_start_quat is None or self._rotation_target_quat is None:
                raise ValueError("Rotation trajectory is incomplete")
            rotation_tpi = capture_tpi_state(self._tpi_rotation_angle)
            rotation = {
                "start_orientation": self._rotation_start_quat.tolist(),
                "target_orientation": self._rotation_target_quat.tolist(),
                "t0": rotation_tpi["t0"],
                "vmax": rotation_tpi["vmax"],
                "accel": rotation_tpi["accel"],
                "decel": rotation_tpi["decel"],
            }
        return {
            "path": [pose_value(pose) for pose in self._path],
            "waypoint_index": self._current_waypoint_index,
            "goal": pose_value(self._goal_pose),
            "align_final_orientation": self._align_final_orientation,
            "final_target_orientation": (
                self._final_target_orientation.tolist() if self._final_target_orientation is not None else None
            ),
            "final_alignment_in_progress": self._is_final_orientation_aligning,
            "phase": self._pose_phase.value,
            "forward": forward,
            "rotation": rotation,
        }

    def restore_navigation_state(self, state: dict, orientation: list[float]) -> None:
        """Restore a kinematic pose path without reissuing its navigation command."""
        if self._mode is not ControllerMode.IDLE:
            raise ValueError("Navigation restore requires an idle controller")

        def pose_from(value: dict) -> Pose:
            return Pose(position=list(value["position"]), orientation=list(value["orientation"]))

        self._path = [pose_from(value) for value in state["path"]]
        self._current_waypoint_index = state["waypoint_index"]
        self._goal_pose = pose_from(state["goal"])
        self._align_final_orientation = state["align_final_orientation"]
        target = state["final_target_orientation"]
        self._final_target_orientation = np.asarray(target, dtype=float) if target is not None else None
        self._is_final_orientation_aligning = state["final_alignment_in_progress"]
        self._mode = ControllerMode.POSE
        self._pose_phase = PosePhase(state["phase"])

        if self._pose_phase is PosePhase.FORWARD:
            forward = state["forward"]
            origin = np.asarray(forward["origin"], dtype=float) if forward["origin"] is not None else None
            if origin is not None:
                goal_position = np.asarray(self._goal_pose.position, dtype=float)
                if self.params.navigation_2d:
                    goal_position[2] = origin[2]
                displacement = goal_position - origin
                distance = float(np.linalg.norm(displacement))
            else:
                displacement = np.zeros(3)
                distance = 0.0
            self._forward_start_pos = origin
            self._forward_direction_3d = displacement
            self._forward_total_distance_3d = distance
            self._forward_direction_unit = displacement / distance if distance > 1e-9 else np.zeros(3)
            qx, qy, qz, qw = orientation
            self._forward_direction_unit_body = np.asarray(
                rotate_vector(tuple(self._forward_direction_unit), (-qx, -qy, -qz, qw))
            )
            after_rotation = forward["orientation_after_rotation"]
            self._rotation_target_quat = np.asarray(after_rotation, dtype=float) if after_rotation is not None else None
            self._tpi_forward = restore_tpi_state(
                {
                    "p0": 0.0,
                    "pe": distance,
                    "v0": 0.0,
                    "ve": 0.0,
                    "vmax": forward["vmax"],
                    "accel": forward["accel"],
                    "decel": forward["decel"],
                    "t0": forward["t0"],
                }
            )
        else:
            self._reset_forward_state()
            rotation = state["rotation"]
            if rotation is not None:
                start = np.asarray(rotation["start_orientation"], dtype=float)
                target = np.asarray(rotation["target_orientation"], dtype=float)
                angle = quat_angle_between(tuple(start), tuple(target))
                self._rotation_start_quat = start
                self._rotation_target_quat = target
                self._rotation_total_angle = float(angle)
                self._tpi_rotation_angle = restore_tpi_state(
                    {
                        "p0": 0.0,
                        "pe": angle,
                        "v0": 0.0,
                        "ve": 0.0,
                        "vmax": rotation["vmax"],
                        "accel": rotation["accel"],
                        "decel": rotation["decel"],
                        "t0": rotation["t0"],
                    }
                )
                self._slerp_precomp = quat_slerp_precompute(start, target)


# ---------------------------------------------------------------------------
# OmniController (6-DoF velocity + omnidirectional TPI pose)
# ---------------------------------------------------------------------------


class OmniController(KinematicController):
    """Unified omnidirectional controller: velocity commands + TPI pose navigation.

    **Velocity mode (VELOCITY):**
        6-DoF body-frame velocity (vx, vy, vz, wx, wy, wz) → quaternion-based
        world-frame integration. Linear and angular magnitudes are clamped.

    **Pose mode (POSE):**
        Per-axis TPI with distance-ratio velocity scaling for straight-line
        motion from start to goal.
    """

    _registry_name = "omni"

    def __init__(self, params: Optional[ControllerParams] = None) -> None:
        super().__init__(params)

    # -- Velocity kinematics -------------------------------------------

    def set_velocity(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
    ) -> None:
        """Set velocity in **body frame** (6-DoF) with kinematic clamping.

        Scalar limits ⇒ magnitude clamp (preserves direction).
        Per-axis limits ⇒ per-component clamp.
        """
        lin = np.array([vx, vy, vz], dtype=float)
        ang = np.array([wx, wy, wz], dtype=float)

        lin = self.params.clamp_linear_vec(lin)
        ang = self.params.clamp_angular_vec(ang)

        super().set_velocity(vx=lin[0], vy=lin[1], vz=lin[2], wx=ang[0], wy=ang[1], wz=ang[2])

    def _apply_velocity(self, agent: "Agent", dt: float) -> bool:
        pose = agent.get_pose()

        qx, qy, qz, qw = pose.orientation

        # Body → world rotation for linear velocity (SciPy-free hot path)
        world_vel = rotate_vector(tuple(self._linear_velocity), (qx, qy, qz, qw))

        new_pos = [
            pose.x + world_vel[0] * dt,
            pose.y + world_vel[1] * dt,
            pose.z + world_vel[2] * dt,
        ]

        # Angular velocity integration via quaternion multiplication (SciPy-free)
        ang_zero = np.allclose(self._angular_velocity, 0.0)
        if ang_zero:
            new_quat = [qx, qy, qz, qw]
        else:
            omega = self._angular_velocity
            rotvec = (omega[0] * dt, omega[1] * dt, omega[2] * dt)
            delta_q = quat_from_rotvec(rotvec)
            new_q = quat_multiply((qx, qy, qz, qw), delta_q)  # body-frame
            new_quat = list(new_q)

        new_pose = Pose(position=new_pos, orientation=new_quat)
        agent.set_pose(new_pose)

        agent._current_velocity[:] = world_vel
        agent._current_angular_velocity = self._angular_velocity[2]
        return True

    # -- Pose (TPI) kinematics ----------------------------------------

    def _init_pose_trajectory(self, agent: "Agent", goal: Pose, **kwargs) -> None:
        """Initialize single-distance TPI for omnidirectional straight-line motion.

        Orientation is NOT handled here — omni agents keep orientation constant
        during translation.  Sets ``_pose_phase = FORWARD`` to skip rotation.
        For rotation-only (final alignment), use
        ``_init_rotation_only_trajectory`` instead.
        """
        current_pose = agent.get_pose()
        current_pos = np.array(current_pose.position)
        goal_pos = np.array(goal.position)
        if self.params.navigation_2d:
            goal_pos[2] = current_pos[2]

        # Reset rotation state (position-only trajectory)
        self._reset_rotation_state()

        # Skip rotation phase — omni moves in any direction
        self._pose_phase = PosePhase.FORWARD

        # Cache direction for straight-line interpolation
        displacement = goal_pos - current_pos
        total_dist = float(np.linalg.norm(displacement))

        self._forward_start_pos = current_pos.copy()
        self._forward_total_distance_3d = total_dist
        if total_dist > 1e-9:
            self._forward_direction_3d = displacement.copy()
            self._forward_direction_unit = displacement / total_dist
            # Body-frame direction for cap projection: rotate world unit
            # vector by the conjugate (= inverse for unit quat) of the
            # current orientation.  ``max_linear_vel`` per-axis caps are
            # body-frame, so projection must happen in body frame.
            qx, qy, qz, qw = current_pose.orientation
            dir_body = rotate_vector(tuple(self._forward_direction_unit), (-qx, -qy, -qz, qw))
            self._forward_direction_unit_body = np.array(dir_body)
        else:
            self._forward_direction_3d = np.zeros(3)
            self._forward_direction_unit = np.zeros(3)
            self._forward_direction_unit_body = np.zeros(3)

        self._init_forward_trajectory(agent, total_dist)

    def _init_final_orientation_trajectory(self, agent: "Agent", goal: Pose) -> None:
        """Rotation-only final alignment (no position movement)."""
        self._init_rotation_only_trajectory(agent, goal)

    def _init_rotation_only_trajectory(self, agent: "Agent", goal: Pose) -> None:
        """Initialize slerp+TPI for rotation-only (final alignment).

        Sets ``_pose_phase = ROTATE`` and resets forward state
        so the unified ``_apply_pose`` will rotate then immediately complete
        the zero-distance forward phase.
        """
        current_pose = agent.get_pose()
        t0 = agent.sim_core.sim_time if agent.sim_core is not None else 0.0

        self._pose_phase = PosePhase.ROTATE
        self._reset_forward_state()
        self._init_rotation_tpi(
            agent,
            np.array(current_pose.orientation),
            np.array(goal.orientation),
            t0,
        )


# ---------------------------------------------------------------------------
# DifferentialController (unicycle velocity + rotate-then-forward TPI pose)
# ---------------------------------------------------------------------------


class DifferentialController(KinematicController):
    """Unified differential-drive controller: velocity commands + TPI pose.

    **Velocity mode (VELOCITY):**
        Unicycle model — only forward (``vx``) and yaw rate (``wz``).
        Lateral velocity and other DoF are ignored.

    **Pose mode (POSE):**
        Two-phase TPI: ROTATE to face target, then FORWARD to reach it.
        Uses TPI + quaternion slerp for smooth rotation.
    """

    _registry_name = "differential"

    def __init__(
        self,
        params: Optional[ControllerParams] = None,
        wheel_separation: float = 0.0,
    ) -> None:
        super().__init__(params)
        self._wheel_separation = wheel_separation

    def capture_navigation_state(self) -> dict:
        """Capture the common pose path plus differential heading and direction."""
        state = super().capture_navigation_state()
        if self._forward_start_pos is None and not self._is_final_orientation_aligning:
            raise ValueError("Differential navigation origin is missing")
        state["movement_direction"] = self._movement_direction.value
        state["requested_direction"] = self._original_direction.value
        state["forward_origin"] = self._forward_start_pos.tolist() if self._forward_start_pos is not None else None
        return state

    def restore_navigation_state(self, state: dict, orientation: list[float]) -> None:
        """Restore a turn or straight segment without issuing another command."""
        super().restore_navigation_state(state, orientation)
        self._movement_direction = MovementDirection(state["movement_direction"])
        self._original_direction = MovementDirection(state["requested_direction"])
        if self._movement_direction is MovementDirection.AUTO:
            raise ValueError("Differential checkpoint direction must be resolved")
        origin = np.asarray(state["forward_origin"], dtype=float) if state["forward_origin"] is not None else None
        if self._pose_phase is PosePhase.ROTATE and origin is not None:
            goal = np.asarray(self._goal_pose.position, dtype=float)
            if self.params.navigation_2d:
                goal[2] = origin[2]
            displacement = goal - origin
            distance = float(np.linalg.norm(displacement))
            self._forward_start_pos = origin
            self._forward_direction_3d = displacement
            self._forward_total_distance_3d = distance
            self._forward_direction_unit = displacement / distance if distance > 1e-9 else np.zeros(3)
        sign = -1.0 if self._movement_direction is MovementDirection.BACKWARD else 1.0
        self._forward_direction_unit_body = np.array([sign, 0.0, 0.0])

    def set_velocity(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
    ) -> None:
        """Set velocity (body frame). Only ``vx`` and ``wz`` are used.

        Per-axis limits are reduced to scalars via X (forward) / yaw axes.
        """
        vmax = self.params.scalar_max_linear_vel()
        wmax = self.params.scalar_max_angular_vel()
        clamped_vx = max(-vmax, min(vx, vmax))
        clamped_wz = max(-wmax, min(wz, wmax))
        super().set_velocity(vx=clamped_vx, wz=clamped_wz)

    # -- Velocity kinematics (unicycle) --------------------------------

    def _apply_velocity(self, agent: "Agent", dt: float) -> bool:
        v = self._linear_velocity[0]
        wz = self._angular_velocity[2]
        pose = agent.get_pose()
        yaw = pose.yaw

        world_vx = v * math.cos(yaw)
        world_vy = v * math.sin(yaw)

        new_pose = Pose.from_yaw(
            pose.x + world_vx * dt,
            pose.y + world_vy * dt,
            pose.z,
            yaw + wz * dt,
        )
        agent.set_pose(new_pose)

        agent._current_velocity[:] = [world_vx, world_vy, 0.0]
        agent._current_angular_velocity = wz
        return True

    # -- Pose (TPI) kinematics ----------------------------------------

    def _init_pose_trajectory(self, agent: "Agent", goal: Pose, **kwargs) -> None:
        """Initialize rotation trajectory for differential drive (ROTATE phase)."""
        self._movement_direction = kwargs.get("direction", MovementDirection.FORWARD)
        if isinstance(self._movement_direction, str):
            self._movement_direction = MovementDirection(self._movement_direction)

        current_pose = agent.get_pose()
        current_pos = np.array(current_pose.position)
        goal_pos = np.array(goal.position)
        if self.params.navigation_2d:
            goal_pos[2] = current_pos[2]

        direction_vec = goal_pos - current_pos

        # AUTO direction: choose FORWARD/BACKWARD based on yaw delta
        if self._movement_direction == MovementDirection.AUTO:
            dist_2d = float(np.linalg.norm(direction_vec[:2]))
            if dist_2d > 1e-6:
                goal_heading = math.atan2(direction_vec[1], direction_vec[0])
                current_heading = current_pose.yaw
                delta = goal_heading - current_heading
                # Normalize to [-pi, pi]
                delta = (delta + math.pi) % (2 * math.pi) - math.pi
                self._movement_direction = (
                    MovementDirection.BACKWARD if abs(delta) > math.pi / 2 else MovementDirection.FORWARD
                )
            else:
                self._movement_direction = MovementDirection.FORWARD

        # Cache forward direction for straight-line phase
        self._forward_start_pos = current_pos.copy()
        self._forward_total_distance_3d = float(np.linalg.norm(direction_vec))

        if self._forward_total_distance_3d > 1e-6:
            forward_direction_3d = goal_pos - current_pos
            self._forward_direction_3d = forward_direction_3d
            self._forward_direction_unit = forward_direction_3d / self._forward_total_distance_3d
        else:
            self._forward_direction_3d = np.zeros(3)
            self._forward_direction_unit = np.zeros(3)

        # Differential drive always rotates to face the target before the
        # FORWARD phase, so by construction the travel direction is body +X
        # (or -X for BACKWARD). No need for a world→body rotation.
        if self._forward_total_distance_3d > 1e-6:
            sign = -1.0 if self._movement_direction == MovementDirection.BACKWARD else 1.0
            self._forward_direction_unit_body = np.array([sign, 0.0, 0.0])
        else:
            self._forward_direction_unit_body = np.zeros(3)

        t0 = agent.sim_core.sim_time if agent.sim_core is not None else 0.0

        self._pose_phase = PosePhase.ROTATE

        start_quat = np.array(current_pose.orientation)

        # Calculate target orientation via shared helper
        if self._forward_total_distance_3d > 1e-6:
            movement_direction_vec = direction_vec / self._forward_total_distance_3d
            if self._movement_direction == MovementDirection.BACKWARD:
                x_axis_target = -movement_direction_vec
            else:
                x_axis_target = movement_direction_vec
            target_quat = align_x_axis_quat(start_quat, np.array(goal.orientation), x_axis_target)
        else:
            target_quat = np.array(goal.orientation)

        if not self._init_rotation_tpi(agent, start_quat, target_quat, t0):
            # Negligible or failed rotation — snap heading and skip to forward
            agent.set_pose_raw(current_pos.tolist(), target_quat.tolist(), preserve_velocity=False)
            self._pose_phase = PosePhase.FORWARD
            self._init_forward_trajectory(agent, self._forward_total_distance_3d)


# ---------------------------------------------------------------------------
# Import high-level controllers so __init_subclass__ auto-registration fires
# ---------------------------------------------------------------------------
from pybullet_fleet.controllers.patrol_controller import PatrolController  # noqa: E402, F401
from pybullet_fleet.controllers.random_walk_controller import RandomWalkController  # noqa: E402, F401
