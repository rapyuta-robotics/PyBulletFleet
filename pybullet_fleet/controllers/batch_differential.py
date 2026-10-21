"""Vectorized differential-drive batch controller.

Manages many differential-drive agents at once using NumPy arrays. Each
registered agent runs the standard ROTATE → FORWARD lifecycle per waypoint:

1. ROTATE: trapezoidal-velocity TPI on rotation angle + quaternion slerp.
2. FORWARD: trapezoidal-velocity TPI on straight-line distance.

A single ``batch_advance(dt)`` call evaluates both phases for all agents
without per-agent Python dispatch and writes the resulting poses via
``sim_core.set_poses``.

Scope
-----
- Pose mode only (path following). No ``set_velocity``.
- ``MovementDirection.FORWARD``, ``BACKWARD``, and ``AUTO`` are supported.
- ``final_orientation_align`` performs an in-place rotation to match
  ``path[-1].orientation`` after reaching the last waypoint.
- 2D navigation friendly (``navigation_2d=True`` flattens goal z).

Numerical equivalence with ``DifferentialController`` for the supported
scope is within ~1e-6 per step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from pybullet_fleet._replan import HEADING_TOL, REST_SPEED, PathRequest, replacement_rejection
from pybullet_fleet._motion_planning import heading_toward
from pybullet_fleet._tpi import (
    build_tpi,
    build_tpi_from_speed,
    extract_phase_params,
    trapezoid_distance,
    trapezoid_speed,
)
from pybullet_fleet.controllers.batch_base import BatchKinematicController
from pybullet_fleet.geometry import Pose, quat_angle_between, quat_slerp_batch
from pybullet_fleet.logging_utils import get_lazy_logger
from pybullet_fleet.types import MovementDirection

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent

logger = get_lazy_logger(__name__)


# Phase tags (kept as int8 in arrays for cheap masking).
_PHASE_IDLE = 0
_PHASE_ROTATE = 1
_PHASE_FORWARD = 2
_PHASE_FINAL_ROTATE = 3  # in-place rotation to match path[-1].orientation


class BatchDifferentialController(BatchKinematicController):
    """Batched differential-drive pose controller."""

    _registry_name = "batch_differential"

    def __init__(self) -> None:
        super().__init__()

        # Phase per agent.
        self._phase: np.ndarray = np.zeros((0,), dtype=np.int8)

        # ROTATE-phase state (also used for _PHASE_FINAL_ROTATE).
        self._rot_t_start: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_start_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        self._rot_target_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        # Slerp precompute (per agent): cached dot (always >= 0) and theta_0/sin_theta_0.
        self._rot_dot: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_theta0: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_sin_theta0: np.ndarray = np.zeros((0,), dtype=np.float64)
        # Rotation angle TPI params.
        self._rot_total_angle: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_t_accel: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_t_const: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_t_total: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_accel: np.ndarray = np.zeros((0,), dtype=np.float64)
        # Un-flipped target for snap-to at rotation end (stores original sign).
        self._rot_snap_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)

        # FORWARD-phase state.
        self._fwd_t_start: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._fwd_p_start: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._fwd_displacement: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._fwd_total_distance: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._fwd_t_accel: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._fwd_t_const: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._fwd_t_total: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._fwd_accel: np.ndarray = np.zeros((0,), dtype=np.float64)
        #: Braking scalar for the forward phase; see BatchOmniController.
        self._fwd_decel: np.ndarray = np.zeros((0,), dtype=np.float64)
        #: Speed each forward phase started at: a path that continues the
        #: motion starts from it (see build_tpi_from_speed). Zero from rest.
        self._fwd_v0: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._fwd_target_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)

        # Per-agent direction and final-orientation state (Python lists — rare updates).
        self._original_direction: List[Optional[MovementDirection]] = []
        self._align_final_orient: np.ndarray = np.zeros((0,), dtype=bool)
        self._final_orient_target_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)

        # Multi-waypoint path state (Python lists — waypoint switching is rare).
        self._paths: List[List[Pose]] = []
        self._wp_index: List[int] = []

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def _resize_subclass(self, n: int) -> None:
        self._phase = self._resize_rows(self._phase, n)
        # ROTATE
        self._rot_t_start = self._resize_rows(self._rot_t_start, n)
        self._rot_start_quat = self._resize_rows(self._rot_start_quat, n)
        self._rot_target_quat = self._resize_rows(self._rot_target_quat, n)
        self._rot_snap_quat = self._resize_rows(self._rot_snap_quat, n)
        self._rot_dot = self._resize_rows(self._rot_dot, n)
        self._rot_theta0 = self._resize_rows(self._rot_theta0, n)
        self._rot_sin_theta0 = self._resize_rows(self._rot_sin_theta0, n)
        self._rot_total_angle = self._resize_rows(self._rot_total_angle, n)
        self._rot_t_accel = self._resize_rows(self._rot_t_accel, n)
        self._rot_t_const = self._resize_rows(self._rot_t_const, n)
        self._rot_t_total = self._resize_rows(self._rot_t_total, n)
        self._rot_accel = self._resize_rows(self._rot_accel, n)
        # FORWARD
        self._fwd_t_start = self._resize_rows(self._fwd_t_start, n)
        self._fwd_p_start = self._resize_rows(self._fwd_p_start, n)
        self._fwd_displacement = self._resize_rows(self._fwd_displacement, n)
        self._fwd_total_distance = self._resize_rows(self._fwd_total_distance, n)
        self._fwd_t_accel = self._resize_rows(self._fwd_t_accel, n)
        self._fwd_t_const = self._resize_rows(self._fwd_t_const, n)
        self._fwd_t_total = self._resize_rows(self._fwd_t_total, n)
        self._fwd_accel = self._resize_rows(self._fwd_accel, n)
        self._fwd_decel = self._resize_rows(self._fwd_decel, n)
        self._fwd_v0 = self._resize_rows(self._fwd_v0, n)
        self._fwd_target_quat = self._resize_rows(self._fwd_target_quat, n)
        # Final-orientation
        self._align_final_orient = self._resize_rows(self._align_final_orient, n)
        self._final_orient_target_quat = self._resize_rows(self._final_orient_target_quat, n)

    def _on_agent_registered(self, idx: int, agent: "Agent") -> None:
        self._resize_subclass(len(self._agents))
        self._phase[idx] = _PHASE_IDLE
        pose = agent.get_pose()
        # Default identity targets so an idle agent's row is well-defined.
        self._rot_start_quat[idx] = pose.orientation
        self._rot_target_quat[idx] = pose.orientation
        self._rot_snap_quat[idx] = pose.orientation
        self._fwd_target_quat[idx] = pose.orientation
        self._final_orient_target_quat[idx] = pose.orientation
        self._align_final_orient[idx] = False
        while len(self._paths) <= idx:
            self._paths.append([])
            self._wp_index.append(0)
            self._original_direction.append(None)
        self._paths[idx] = []
        self._wp_index[idx] = 0
        self._original_direction[idx] = None

    def _on_agent_unregistered(self, idx: int, agent: "Agent") -> None:
        n = len(self._agents)
        self._resize_subclass(n)
        while len(self._paths) > n:
            self._paths.pop()
            self._wp_index.pop()
            self._original_direction.pop()

    def _swap_rows(self, i: int, j: int) -> None:
        super()._swap_rows(i, j)
        for arr in (
            self._phase,
            self._rot_t_start,
            self._rot_dot,
            self._rot_theta0,
            self._rot_sin_theta0,
            self._rot_total_angle,
            self._rot_t_accel,
            self._rot_t_const,
            self._rot_t_total,
            self._rot_accel,
            self._fwd_t_start,
            self._fwd_total_distance,
            self._fwd_t_accel,
            self._fwd_t_const,
            self._fwd_t_total,
            self._fwd_accel,
            self._fwd_decel,
            self._fwd_v0,
            self._align_final_orient,
        ):
            arr[[i, j]] = arr[[j, i]]
        for arr2d in (
            self._rot_start_quat,
            self._rot_target_quat,
            self._rot_snap_quat,
            self._fwd_p_start,
            self._fwd_displacement,
            self._fwd_target_quat,
            self._final_orient_target_quat,
        ):
            arr2d[[i, j]] = arr2d[[j, i]]
        self._paths[i], self._paths[j] = self._paths[j], self._paths[i]
        self._wp_index[i], self._wp_index[j] = self._wp_index[j], self._wp_index[i]
        self._original_direction[i], self._original_direction[j] = (
            self._original_direction[j],
            self._original_direction[i],
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_path(
        self,
        agent: "Agent",
        path: List[Pose],
        direction: Optional[MovementDirection] = None,
        final_orientation_align: bool = True,
        **kwargs,
    ) -> bool:
        """Set a waypoint path.

        Args:
            agent: A registered agent.
            path: Non-empty list of goal poses.
            direction: ``FORWARD`` (default), ``BACKWARD``, or ``AUTO``
                (re-evaluated per waypoint based on heading delta).
                ``None`` uses the agent's ``controller_params.default_direction``
                (``FORWARD`` unless configured), as the per-agent controller does.
            final_orientation_align: If ``True`` (default), rotate to match
                ``path[-1].orientation`` after reaching the last waypoint.

        Returns:
            ``False`` when the path is refused (``goal_replacement="reject"``
            only); ``True`` once it is set or held to start later. See
            :meth:`KinematicController.set_path <pybullet_fleet.controller.KinematicController.set_path>`.
        """
        if self._sim_core is None:
            raise RuntimeError("BatchDifferentialController must be attached to a sim_core before set_path.")
        if not path:
            raise ValueError("path must be a non-empty list of Poses.")
        if self._agent_index.get(id(agent)) is None:
            raise KeyError(f"Agent {agent} is not registered with this batch controller.")
        if isinstance(direction, str):
            direction = MovementDirection(direction)
        return self._replace_path(
            agent,
            PathRequest(path=list(path), final_orientation_align=final_orientation_align, direction=direction),
            preempt=bool(kwargs.get("preempt", False)),
        )

    def _start_row_path(self, agent: "Agent", request: PathRequest, continue_motion: bool) -> None:
        idx = self._agent_index[id(agent)]
        path = request.path
        assert self._sim_core is not None
        now = float(self._sim_core.sim_time)
        initial_velocity = None
        if continue_motion and self._phase[idx] == _PHASE_FORWARD:
            initial_velocity = self._row_velocity(idx, now)
        self._paths[idx] = list(path)
        self._wp_index[idx] = 0
        # Unset means the agent's default_direction, as the per-agent
        # controller resolves it -- not the previous path's direction.
        direction = request.direction if request.direction is not None else agent.controller_params.default_direction
        self._original_direction[idx] = direction
        if request.final_orientation_align:
            self._align_final_orient[idx] = True
            self._final_orient_target_quat[idx] = np.asarray(path[-1].orientation, dtype=np.float64)
        else:
            self._align_final_orient[idx] = False
        self._begin_waypoint(idx, agent, path[0], now, direction=direction, initial_velocity=initial_velocity)

    def continuation_velocity(self, agent: "Agent") -> Optional[np.ndarray]:
        """The row's forward profile velocity while it drives straight, else ``None``.

        A turn in place has no linear velocity to continue. See
        :meth:`BatchOmniController.continuation_velocity
        <pybullet_fleet.controllers.batch_omni.BatchOmniController.continuation_velocity>`.
        """
        idx = self._agent_index.get(id(agent))
        if idx is None or self._sim_core is None or self._phase[idx] != _PHASE_FORWARD:
            return None
        return self._row_velocity(idx, float(self._sim_core.sim_time))

    def _replacement_obstacle(
        self, agent: "Agent", path: List[Pose], direction: Optional[MovementDirection] = None
    ) -> Optional[str]:
        """Why ``path`` cannot continue the current motion, or ``None`` -- the same rule as
        :meth:`DifferentialController._replacement_obstacle
        <pybullet_fleet.controller.DifferentialController._replacement_obstacle>`.
        """
        idx = self._agent_index.get(id(agent))
        if idx is None or not path or self._sim_core is None or self._phase[idx] != _PHASE_FORWARD:
            return None
        now = float(self._sim_core.sim_time)
        velocity = self._row_velocity(idx, now)
        if float(np.linalg.norm(velocity)) < REST_SPEED:
            return None
        # The direction the new path will use (see _start_row_path), not the
        # current path's.
        requested = direction if direction is not None else agent.controller_params.default_direction
        _, target_quat, _, distance = heading_toward(
            agent.get_pose(), path[0], requested, agent.controller_params.navigation_2d
        )
        current = agent.get_pose()
        if distance > 1e-6:
            turn = quat_angle_between(np.asarray(current.orientation, dtype=np.float64), target_quat)
            if turn > HEADING_TOL:
                return f"goal needs a {math.degrees(turn):.3g} degree turn in place"
        return replacement_rejection(
            agent.controller_params, current.position, current.orientation, velocity, path[0].position, now
        )

    def _row_velocity(self, idx: int, now: float) -> np.ndarray:
        """World-frame velocity of row ``idx``'s forward profile at ``now``."""
        total = float(self._fwd_total_distance[idx])
        if total <= 1e-9:
            return np.zeros(3)
        row = slice(idx, idx + 1)
        speed = trapezoid_speed(
            np.maximum(now - self._fwd_t_start[row], 0.0),
            self._fwd_t_accel[row],
            self._fwd_t_const[row],
            self._fwd_t_total[row],
            self._fwd_accel[row],
            decel=self._fwd_decel[row],
            v0=self._fwd_v0[row],
        )[0]
        return self._fwd_displacement[idx] / total * float(speed)

    # ------------------------------------------------------------------ #
    # Waypoint lifecycle (per-agent, called rarely)
    # ------------------------------------------------------------------ #

    def _begin_waypoint(
        self,
        idx: int,
        agent: "Agent",
        goal: Pose,
        sim_time: float,
        direction: Optional[MovementDirection] = None,
        initial_velocity: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise the ROTATE phase for a waypoint.

        Mirrors ``DifferentialController._init_pose_trajectory`` with
        FORWARD / BACKWARD / AUTO directions supported; navigation_2d honoured.
        ``initial_velocity`` (a path continuing the motion) is kept as the
        forward phase's starting speed when no turn is needed.
        """
        # Resolve direction: explicit arg > stored original > FORWARD
        if direction is None:
            direction = self._original_direction[idx]
        direction, target_quat, direction_vec, total_distance = heading_toward(
            agent.get_pose(), goal, direction, agent.controller_params.navigation_2d
        )
        current = agent.get_pose()
        start_pos = np.asarray(current.position, dtype=np.float64)
        start_quat = np.asarray(current.orientation, dtype=np.float64)

        # Cache forward state up front (will be used after ROTATE completes).
        self._fwd_p_start[idx] = start_pos
        self._fwd_displacement[idx] = direction_vec
        self._fwd_total_distance[idx] = total_distance

        self._fwd_target_quat[idx] = target_quat

        # Compute rotation angle and TPI; if negligible -- or, continuing the
        # motion, within HEADING_TOL -- snap heading and skip to FORWARD
        # immediately (matching DifferentialController behaviour).
        angle = quat_angle_between(start_quat, target_quat)
        if angle <= 1e-6 or (initial_velocity is not None and angle <= HEADING_TOL):
            v0 = 0.0
            if initial_velocity is not None and total_distance > 1e-6:
                # Signed: negative is moving away from the goal, which the
                # trajectory reverses (see build_tpi_from_speed).
                v0 = float(np.dot(initial_velocity, direction_vec / total_distance))
            self._begin_forward(idx, agent, target_quat, sim_time, v0=v0)
            return

        # Precompute slerp constants (with shortest-path correction).
        dot = float(np.dot(start_quat, target_quat))
        if dot < 0.0:
            target_for_slerp = -target_quat
            dot = -dot
        else:
            target_for_slerp = target_quat.copy()
        dot = min(1.0, max(0.0, dot))
        theta0 = math.acos(dot)
        sin_theta0 = math.sin(theta0)

        ang_vel = float(agent.max_angular_vel[0])
        ang_accel = float(agent.max_angular_accel[0])
        tpi = build_tpi(p0=0.0, pe=angle, vmax=ang_vel, accel=ang_accel, t0=sim_time)
        # Rotation is symmetric: angular limits carry no separate decel.
        # See #67.
        t_acc, t_cst, t_tot, accel_eff, _ = extract_phase_params(tpi)

        self._phase[idx] = _PHASE_ROTATE
        self._rot_t_start[idx] = sim_time
        self._rot_start_quat[idx] = start_quat
        self._rot_target_quat[idx] = target_for_slerp
        self._rot_snap_quat[idx] = target_quat  # un-flipped for snap
        self._rot_dot[idx] = dot
        self._rot_theta0[idx] = theta0
        self._rot_sin_theta0[idx] = sin_theta0
        self._rot_total_angle[idx] = angle
        self._rot_t_accel[idx] = t_acc
        self._rot_t_const[idx] = t_cst
        self._rot_t_total[idx] = t_tot
        self._rot_accel[idx] = accel_eff

    def _begin_forward(self, idx: int, agent: "Agent", target_quat: np.ndarray, sim_time: float, v0: float = 0.0) -> None:
        """Switch to FORWARD phase: snap heading, init distance TPI from speed ``v0`` (0 from rest)."""
        self._pos_buf[idx] = self._fwd_p_start[idx]
        self._orn_buf[idx] = target_quat

        distance = float(self._fwd_total_distance[idx])
        # Differential drive travels along its forward (body-x) axis — use the
        # forward-axis limit (ControllerParams scalar helper), not the axis mean.
        vmax = agent.controller_params.scalar_max_linear_vel()
        amax = agent.controller_params.scalar_max_linear_accel()
        dmax = agent.controller_params.scalar_max_linear_decel()
        tpi = build_tpi_from_speed(p0=0.0, pe=distance, vmax=vmax, accel=amax, t0=sim_time, v0=v0, decel=dmax)
        # The forward phase is the one with a configurable braking rate.
        t_acc, t_cst, t_tot, accel_eff, decel_eff = extract_phase_params(tpi)

        self._phase[idx] = _PHASE_FORWARD
        self._fwd_t_start[idx] = sim_time
        self._fwd_t_accel[idx] = t_acc
        self._fwd_t_const[idx] = t_cst
        self._fwd_t_total[idx] = t_tot
        self._fwd_accel[idx] = accel_eff
        self._fwd_decel[idx] = decel_eff
        self._fwd_v0[idx] = float(tpi.v0)

    def _begin_final_rotate(self, idx: int, sim_time: float) -> None:
        """Start in-place rotation to ``_final_orient_target_quat[idx]``, then go IDLE."""
        agent = self._agents[idx]
        start_quat = np.asarray(agent.get_pose().orientation, dtype=np.float64)
        target_quat = self._final_orient_target_quat[idx].copy()

        angle = quat_angle_between(start_quat, target_quat)
        if angle <= 1e-6:
            # Already at target; snap and go idle.
            self._orn_buf[idx] = target_quat
            self._moved_mask[idx] = True
            self._phase[idx] = _PHASE_IDLE
            self._paths[idx] = []
            self._wp_index[idx] = 0
            self._agents[idx]._is_moving = False
            self._row_finished(idx)
            return

        dot = float(np.dot(start_quat, target_quat))
        if dot < 0.0:
            target_for_slerp = -target_quat
            dot = -dot
        else:
            target_for_slerp = target_quat.copy()
        dot = min(1.0, max(0.0, dot))
        theta0 = math.acos(dot)
        sin_theta0 = math.sin(theta0)

        ang_vel = float(agent.max_angular_vel[0])
        ang_accel = float(agent.max_angular_accel[0])
        tpi = build_tpi(p0=0.0, pe=angle, vmax=ang_vel, accel=ang_accel, t0=sim_time)
        # Rotation is symmetric: angular limits carry no separate decel.
        # See #67.
        t_acc, t_cst, t_tot, accel_eff, _ = extract_phase_params(tpi)

        # Reuse the ROTATE state arrays (same slerp logic; different exit action).
        self._phase[idx] = _PHASE_FINAL_ROTATE
        self._rot_t_start[idx] = sim_time
        self._rot_start_quat[idx] = start_quat
        self._rot_target_quat[idx] = target_for_slerp
        self._rot_snap_quat[idx] = target_quat  # un-flipped for snap
        self._fwd_target_quat[idx] = target_quat  # also store unflipped here
        self._rot_dot[idx] = dot
        self._rot_theta0[idx] = theta0
        self._rot_sin_theta0[idx] = sin_theta0
        self._rot_total_angle[idx] = angle
        self._rot_t_accel[idx] = t_acc
        self._rot_t_const[idx] = t_cst
        self._rot_t_total[idx] = t_tot
        self._rot_accel[idx] = accel_eff
        # Hold position at current location.
        self._fwd_p_start[idx] = np.asarray(agent.get_pose().position, dtype=np.float64)
        self._paths[idx] = []
        self._wp_index[idx] = 0

    def _on_cancel_path(self, idx: int) -> None:
        """Reset phase to IDLE immediately (called by agent.stop())."""
        self._phase[idx] = _PHASE_IDLE
        self._paths[idx] = []
        self._wp_index[idx] = 0
        self._align_final_orient[idx] = False

    def _advance_waypoint(self, idx: int, sim_time: float) -> None:
        path = self._paths[idx]
        if not path:
            self._phase[idx] = _PHASE_IDLE
            return
        self._wp_index[idx] += 1
        if self._wp_index[idx] < len(path):
            direction = self._original_direction[idx] if idx < len(self._original_direction) else None
            self._begin_waypoint(idx, self._agents[idx], path[self._wp_index[idx]], sim_time, direction=direction)
        else:
            # End of path: optionally do a final in-place rotation.
            if self._align_final_orient[idx]:
                self._align_final_orient[idx] = False
                self._begin_final_rotate(idx, sim_time)
            else:
                self._phase[idx] = _PHASE_IDLE
                self._paths[idx] = []
                self._wp_index[idx] = 0
                self._agents[idx]._is_moving = False
                self._row_finished(idx)

    # ------------------------------------------------------------------ #
    # Vectorized step
    # ------------------------------------------------------------------ #

    def batch_advance(self, dt: float) -> np.ndarray:
        n = len(self._agents)
        self._moved_mask.fill(False)
        if n == 0:
            return self._moved_mask
        sim_core = self._sim_core
        if sim_core is None:
            return self._moved_mask

        phase = self._phase
        rotate_mask = (phase == _PHASE_ROTATE) | (phase == _PHASE_FINAL_ROTATE)
        forward_mask = phase == _PHASE_FORWARD
        if not (rotate_mask.any() or forward_mask.any()):
            # Nothing active. Still flush once if agents moved last step, so
            # _apply_phase1 can zero the velocity of any that just stopped.
            if self._was_moved.any():
                self._apply_phase1(dt)
            return self._moved_mask

        now = float(sim_core.sim_time)

        # ----- ROTATE / FINAL_ROTATE: slerp orientation -----
        completed_rotate = np.zeros(n, dtype=bool)
        if rotate_mask.any():
            rm = rotate_mask
            tau = np.maximum(now - self._rot_t_start[rm], 0.0)
            angle_traveled = trapezoid_distance(
                tau,
                self._rot_t_accel[rm],
                self._rot_t_const[rm],
                self._rot_t_total[rm],
                self._rot_accel[rm],
                self._rot_total_angle[rm],
            )
            total_angle_safe = np.where(self._rot_total_angle[rm] > 1e-9, self._rot_total_angle[rm], 1.0)
            t_frac = np.clip(angle_traveled / total_angle_safe, 0.0, 1.0)
            new_quat = quat_slerp_batch(
                self._rot_start_quat[rm],
                self._rot_target_quat[rm],
                t_frac,
                self._rot_dot[rm],
                self._rot_theta0[rm],
                self._rot_sin_theta0[rm],
            )
            self._orn_buf[rm] = new_quat
            # Hold position during rotation.
            self._pos_buf[rm] = self._fwd_p_start[rm]
            self._moved_mask[rm] = True
            # Completed rotations.
            done = tau >= self._rot_t_total[rm]
            if done.any():
                rotate_idx = np.flatnonzero(rm)
                completed_rotate[rotate_idx[done]] = True
                # Snap to un-flipped target to avoid one-frame sign discontinuity.
                self._orn_buf[rotate_idx[done]] = self._rot_snap_quat[rotate_idx[done]]

        # ----- FORWARD: trapezoid distance + position interpolation -----
        completed_forward = np.zeros(n, dtype=bool)
        if forward_mask.any():
            fm = forward_mask
            tau = np.maximum(now - self._fwd_t_start[fm], 0.0)
            distance = trapezoid_distance(
                tau,
                self._fwd_t_accel[fm],
                self._fwd_t_const[fm],
                self._fwd_t_total[fm],
                self._fwd_accel[fm],
                self._fwd_total_distance[fm],
                self._fwd_decel[fm],
                self._fwd_v0[fm],
            )
            total_dist_safe = np.where(self._fwd_total_distance[fm] > 1e-9, self._fwd_total_distance[fm], 1.0)
            # No lower clip: a forward phase started moving away from its goal
            # (negative v0) runs behind its start before turning around.
            ratio = np.minimum(distance / total_dist_safe, 1.0)
            new_pos = self._fwd_p_start[fm] + ratio[:, None] * self._fwd_displacement[fm]
            self._pos_buf[fm] = new_pos
            self._orn_buf[fm] = self._fwd_target_quat[fm]
            self._moved_mask[fm] = True
            done = tau >= self._fwd_t_total[fm]
            if done.any():
                fwd_idx = np.flatnonzero(fm)
                done_rows = fwd_idx[done]
                completed_forward[done_rows] = True
                # Snap to exact goal position.
                self._pos_buf[done_rows] = self._fwd_p_start[done_rows] + self._fwd_displacement[done_rows]

        # Handle ROTATE→FORWARD transitions (or FINAL_ROTATE→IDLE) BEFORE
        # writing to sim so the snapped pose reaches PyBullet in the same step.
        if completed_rotate.any():
            final_rot_phase = phase == _PHASE_FINAL_ROTATE
            for i in np.flatnonzero(completed_rotate):
                ii = int(i)
                if final_rot_phase[ii]:
                    # Final rotation done: go IDLE.
                    self._phase[ii] = _PHASE_IDLE
                    self._agents[ii]._is_moving = False
                    self._row_finished(ii)
                else:
                    # Normal rotation done: start FORWARD phase.
                    self._begin_forward(ii, self._agents[ii], self._rot_snap_quat[ii], now)

        # Write rows to sim (and refresh per-agent reported velocities).
        self._apply_phase1(dt)

        # Handle waypoint advance for completed forwards.
        if completed_forward.any():
            for i in np.flatnonzero(completed_forward):
                self._advance_waypoint(int(i), now)

        return self._moved_mask


# ---------------------------------------------------------------------------
# Module-level helpers kept for backwards-compatibility with any code that
# imported them directly from this module.  New code should import from
# batch_base instead.
# ---------------------------------------------------------------------------
