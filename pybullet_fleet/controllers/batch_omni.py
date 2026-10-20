"""Vectorized omnidirectional batch controller.

Manages many omni agents at once using NumPy arrays. For each registered
agent we store a single straight-line trapezoidal-velocity trajectory; one
``batch_advance(dt)`` evaluates all trajectories without per-agent Python
dispatch and writes the resulting poses via ``sim_core.set_poses``.

Scope
-----
- Pose mode only (path following). Velocity commands are not supported in
  the batched path; agents needing ``set_velocity`` should use the per-agent
  ``OmniController`` instead.
- Multi-waypoint paths are supported.
- ``final_orientation_align=True`` (default) performs an in-place slerp
  rotation to match ``path[-1].orientation`` after the last waypoint.

Numerical equivalence with ``OmniController`` for the supported scope is
within ~1e-6 per step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from pybullet_fleet._replan import replacement_rejection
from pybullet_fleet._tpi import (
    build_tpi,
    build_tpi_from_speed,
    extract_phase_params,
    trapezoid_distance,
    trapezoid_speed,
)
from pybullet_fleet.controllers.batch_base import BatchKinematicController
from pybullet_fleet.geometry import Pose, quat_angle_between, quat_slerp_batch, rotate_vector
from pybullet_fleet.logging_utils import get_lazy_logger

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent

logger = get_lazy_logger(__name__)

# Phase tags.
_PHASE_IDLE = 0
_PHASE_FORWARD = 1
_PHASE_FINAL_ROTATE = 2  # in-place rotation after last waypoint


class BatchOmniController(BatchKinematicController):
    """Batched omnidirectional pose controller."""

    _registry_name = "batch_omni"

    def __init__(self) -> None:
        super().__init__()

        # TRANSLATE-phase state arrays.
        self._phase: np.ndarray = np.zeros((0,), dtype=np.int8)
        self._t_start: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._p_start: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._displacement: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._total_distance: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._t_accel: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._t_const: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._t_total: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._accel_buf: np.ndarray = np.zeros((0,), dtype=np.float64)
        #: Braking scalar per row, kept apart from _accel_buf so a profile
        #: that brakes at a different rate is integrated with its own value.
        self._decel_buf: np.ndarray = np.zeros((0,), dtype=np.float64)
        #: Speed each row's translation started at: a goal set mid-motion
        #: continues from it (see build_tpi_from_speed). Zero from rest.
        self._v0_buf: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._target_orientation: np.ndarray = np.zeros((0, 4), dtype=np.float64)

        # FINAL_ROTATE-phase state arrays (slerp rotation at end of path).
        self._rot_t_start: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_start_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        self._rot_target_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        self._rot_snap_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        self._rot_dot: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_theta0: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_sin_theta0: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_total_angle: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_t_accel: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_t_const: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_t_total: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_accel: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._rot_p_hold: np.ndarray = np.zeros((0, 3), dtype=np.float64)

        # Per-agent final-orientation state.
        self._align_final_orient: np.ndarray = np.zeros((0,), dtype=bool)
        self._final_orient_target_quat: np.ndarray = np.zeros((0, 4), dtype=np.float64)

        # Multi-waypoint path state (Python lists are fine: waypoint switching
        # is rare relative to TPI evaluation).
        self._paths: List[List[Pose]] = []
        self._wp_index: List[int] = []

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def _resize_subclass(self, n: int) -> None:
        self._phase = self._resize_rows(self._phase, n)
        self._t_start = self._resize_rows(self._t_start, n)
        self._total_distance = self._resize_rows(self._total_distance, n)
        self._t_accel = self._resize_rows(self._t_accel, n)
        self._t_const = self._resize_rows(self._t_const, n)
        self._t_total = self._resize_rows(self._t_total, n)
        self._accel_buf = self._resize_rows(self._accel_buf, n)
        self._decel_buf = self._resize_rows(self._decel_buf, n)
        self._v0_buf = self._resize_rows(self._v0_buf, n)
        self._p_start = self._resize_rows(self._p_start, n)
        self._displacement = self._resize_rows(self._displacement, n)
        self._target_orientation = self._resize_rows(self._target_orientation, n)
        # Rotation
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
        self._rot_p_hold = self._resize_rows(self._rot_p_hold, n)
        # Final-orientation
        self._align_final_orient = self._resize_rows(self._align_final_orient, n)
        self._final_orient_target_quat = self._resize_rows(self._final_orient_target_quat, n)

    def _on_agent_registered(self, idx: int, agent: "Agent") -> None:
        self._resize_subclass(len(self._agents))
        self._phase[idx] = _PHASE_IDLE
        self._t_total[idx] = 0.0
        pose = agent.get_pose()
        self._target_orientation[idx] = pose.orientation
        self._final_orient_target_quat[idx] = pose.orientation
        self._rot_start_quat[idx] = pose.orientation
        self._rot_target_quat[idx] = pose.orientation
        self._rot_snap_quat[idx] = pose.orientation
        self._align_final_orient[idx] = False
        while len(self._paths) <= idx:
            self._paths.append([])
            self._wp_index.append(0)
        self._paths[idx] = []
        self._wp_index[idx] = 0

    def _on_agent_unregistered(self, idx: int, agent: "Agent") -> None:
        n = len(self._agents)
        self._resize_subclass(n)
        if idx < len(self._paths):
            while len(self._paths) > n:
                self._paths.pop()
                self._wp_index.pop()

    def _swap_rows(self, i: int, j: int) -> None:
        super()._swap_rows(i, j)
        for arr in (
            self._phase,
            self._t_start,
            self._total_distance,
            self._t_accel,
            self._t_const,
            self._t_total,
            self._accel_buf,
            self._decel_buf,
            self._v0_buf,
            self._align_final_orient,
            self._rot_t_start,
            self._rot_dot,
            self._rot_theta0,
            self._rot_sin_theta0,
            self._rot_total_angle,
            self._rot_t_accel,
            self._rot_t_const,
            self._rot_t_total,
            self._rot_accel,
        ):
            arr[[i, j]] = arr[[j, i]]
        for arr2d in (
            self._p_start,
            self._displacement,
            self._target_orientation,
            self._final_orient_target_quat,
            self._rot_start_quat,
            self._rot_target_quat,
            self._rot_snap_quat,
            self._rot_p_hold,
        ):
            arr2d[[i, j]] = arr2d[[j, i]]
        self._paths[i], self._paths[j] = self._paths[j], self._paths[i]
        self._wp_index[i], self._wp_index[j] = self._wp_index[j], self._wp_index[i]

    # ------------------------------------------------------------------ #
    # Public API — set a path for an already-registered agent
    # ------------------------------------------------------------------ #

    def path_rejection(self, agent: "Agent", path: List[Pose]) -> Optional[str]:
        """Why ``path`` cannot be taken now, or ``None`` -- the same rule as
        :meth:`OmniController.path_rejection <pybullet_fleet.controller.OmniController.path_rejection>`.

        The motion is the row's own profile velocity, and only while a
        translation is under way: the agent's reported velocity is the last
        step's average and, right after arrival, still non-zero.
        """
        idx = self._agent_index.get(id(agent))
        if idx is None or not path or self._sim_core is None or self._phase[idx] != _PHASE_FORWARD:
            return None
        now = float(self._sim_core.sim_time)
        current = agent.get_pose()
        return replacement_rejection(
            agent.controller_params,
            current.position,
            current.orientation,
            self._row_velocity(idx, now),
            path[0].position,
            now,
        )

    def set_path(self, agent: "Agent", path: List[Pose], final_orientation_align: bool = True, **kwargs) -> bool:
        """Set a waypoint path.

        Args:
            agent: A registered agent.
            path: Non-empty list of goal poses.
            final_orientation_align: If ``True`` (default), rotate to match
                ``path[-1].orientation`` after reaching the last waypoint.

        Returns:
            ``False``, with the current motion and path untouched, when
            :meth:`path_rejection` refuses the path; ``True`` once it is set.
        """
        if self._sim_core is None:
            raise RuntimeError("BatchOmniController must be attached to a sim_core before set_path.")
        if not path:
            raise ValueError("path must be a non-empty list of Poses.")
        idx = self._agent_index.get(id(agent))
        if idx is None:
            raise KeyError(f"Agent {agent} is not registered with this batch controller.")

        reason = self.path_rejection(agent, path)
        if reason is not None:
            logger.debug(f"Path rejected for {agent.name or agent.object_id}: {reason}")
            return False

        # Continue from the agent's current motion, as OmniController does --
        # but only while a translation is under way. The speed is the row's own
        # profile speed at the time it was last evaluated, which is what
        # OmniController keeps: the agent's reported velocity here is the last
        # step's average (low while accelerating, high while braking) and,
        # right after arrival, still non-zero.
        now = float(self._sim_core.sim_time)
        initial_velocity = None
        if self._phase[idx] == _PHASE_FORWARD:
            initial_velocity = self._row_velocity(idx, now)

        self._paths[idx] = list(path)
        self._wp_index[idx] = 0
        if final_orientation_align:
            self._align_final_orient[idx] = True
            self._final_orient_target_quat[idx] = np.asarray(path[-1].orientation, dtype=np.float64)
        else:
            self._align_final_orient[idx] = False
        self._begin_waypoint(idx, agent, path[0], now, initial_velocity)
        return True

    def _row_velocity(self, idx: int, now: float) -> np.ndarray:
        """World-frame velocity of row ``idx``'s translation profile at ``now``."""
        total = float(self._total_distance[idx])
        if total <= 1e-9:
            return np.zeros(3)
        row = slice(idx, idx + 1)
        speed = trapezoid_speed(
            np.maximum(now - self._t_start[row], 0.0),
            self._t_accel[row],
            self._t_const[row],
            self._t_total[row],
            self._accel_buf[row],
            decel=self._decel_buf[row],
            v0=self._v0_buf[row],
        )[0]
        return self._displacement[idx] / total * float(speed)

    def _begin_waypoint(
        self,
        idx: int,
        agent: "Agent",
        goal: Pose,
        sim_time: float,
        initial_velocity: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise trajectory state at row ``idx`` for a single goal pose.

        ``initial_velocity`` (world frame) is the agent's motion when a new
        path replaces the current one; the translation starts from its
        component toward the goal. ``None`` (later waypoints) starts from rest.
        """
        current = agent.get_pose()
        start = np.asarray(current.position, dtype=np.float64)
        goal_pos = np.asarray(goal.position, dtype=np.float64)
        if agent.controller_params.navigation_2d:
            goal_pos = goal_pos.copy()
            goal_pos[2] = start[2]

        disp = goal_pos - start
        total = float(np.linalg.norm(disp))

        # Project the per-axis linear limits onto the body-frame travel direction
        # (the caps are body-frame), matching OmniController. Averaging the axes
        # would ignore per-axis limits like max_linear_vel: [0.3, 3.0, 0.0].
        params = agent.controller_params
        if total > 1e-9:
            qx, qy, qz, qw = current.orientation
            dir_body = np.asarray(rotate_vector(tuple(disp / total), (-qx, -qy, -qz, qw)))
            vmax = params.linear_vel_along_direction(dir_body)
            amax = params.linear_accel_along_direction(dir_body)
            dmax = params.linear_decel_along_direction(dir_body)
        else:
            vmax = params.scalar_max_linear_vel()
            amax = params.scalar_max_linear_accel()
            dmax = params.scalar_max_linear_decel()

        v0 = 0.0
        if initial_velocity is not None and total > 1e-9:
            # Signed, as OmniController: negative is moving away from the goal.
            v0 = float(np.dot(initial_velocity, disp / total))
        tpi = build_tpi_from_speed(p0=0.0, pe=total, vmax=vmax, accel=amax, t0=sim_time, v0=v0, decel=dmax)
        t_accel, t_const, t_total_rel, accel_eff, decel_eff = extract_phase_params(tpi)

        self._phase[idx] = _PHASE_FORWARD
        self._t_start[idx] = sim_time
        self._p_start[idx] = start
        self._displacement[idx] = disp
        self._total_distance[idx] = total
        self._t_accel[idx] = t_accel
        self._t_const[idx] = t_const
        self._t_total[idx] = t_total_rel
        self._accel_buf[idx] = accel_eff
        self._decel_buf[idx] = decel_eff
        self._v0_buf[idx] = float(tpi.v0)
        # Omni does not rotate during translation: hold current orientation.
        self._target_orientation[idx] = current.orientation

    def _begin_final_rotate(self, idx: int, sim_time: float) -> None:
        """Start in-place slerp rotation to ``_final_orient_target_quat[idx]``."""
        agent = self._agents[idx]
        start_quat = np.asarray(agent.get_pose().orientation, dtype=np.float64)
        target_quat = self._final_orient_target_quat[idx].copy()

        angle = quat_angle_between(start_quat, target_quat)
        if angle <= 1e-6:
            # Already at target orientation; snap and go idle.
            self._orn_buf[idx] = target_quat
            self._moved_mask[idx] = True
            self._phase[idx] = _PHASE_IDLE
            agent._is_moving = False
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
        # Angular limits carry no separate deceleration, so rotation is
        # symmetric and the fifth value equals the fourth. See #67.
        t_acc, t_cst, t_tot, accel_eff, _ = extract_phase_params(tpi)

        self._phase[idx] = _PHASE_FINAL_ROTATE
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
        self._rot_p_hold[idx] = np.asarray(agent.get_pose().position, dtype=np.float64)

    def _advance_waypoint(self, idx: int, sim_time: float) -> None:
        path = self._paths[idx]
        if not path:
            self._phase[idx] = _PHASE_IDLE
            return
        self._wp_index[idx] += 1
        if self._wp_index[idx] < len(path):
            agent = self._agents[idx]
            self._begin_waypoint(idx, agent, path[self._wp_index[idx]], sim_time)
        else:
            if self._align_final_orient[idx]:
                self._align_final_orient[idx] = False
                self._begin_final_rotate(idx, sim_time)
            else:
                self._phase[idx] = _PHASE_IDLE
                self._paths[idx] = []
                self._wp_index[idx] = 0
                self._agents[idx]._is_moving = False

    def _on_cancel_path(self, idx: int) -> None:
        """Reset phase to IDLE immediately (called by agent.stop())."""
        self._phase[idx] = _PHASE_IDLE
        self._paths[idx] = []
        self._wp_index[idx] = 0
        self._align_final_orient[idx] = False

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
        translate_mask = phase == _PHASE_FORWARD
        final_rotate_mask = phase == _PHASE_FINAL_ROTATE

        if not (translate_mask.any() or final_rotate_mask.any()):
            # Nothing active. Still flush once if agents moved last step, so
            # _apply_phase1 can zero the velocity of any that just stopped.
            if self._was_moved.any():
                self._apply_phase1(dt)
            return self._moved_mask

        # Match OmniController._apply_pose semantics: evaluates at sim_core.sim_time,
        # the time of the state this step computes.
        now = float(sim_core.sim_time)

        # ----- TRANSLATE phase -----
        completed = np.zeros(n, dtype=bool)
        if translate_mask.any():
            active = translate_mask
            tau = np.maximum(now - self._t_start, 0.0)
            distance = trapezoid_distance(
                tau,
                self._t_accel,
                self._t_const,
                self._t_total,
                self._accel_buf,
                self._total_distance,
                self._decel_buf,
                self._v0_buf,
            )
            total_dist_safe = np.where(self._total_distance > 1e-9, self._total_distance, 1.0)
            # No lower clip: a trajectory started moving away from its goal
            # (negative v0) runs behind its start before turning around.
            ratio = np.minimum(distance / total_dist_safe, 1.0)
            new_pos = self._p_start + ratio[:, None] * self._displacement

            completed_trans = active & (tau >= self._t_total)
            if completed_trans.any():
                new_pos[completed_trans] = self._p_start[completed_trans] + self._displacement[completed_trans]
                completed[completed_trans] = True

            self._pos_buf[active] = new_pos[active]
            self._orn_buf[active] = self._target_orientation[active]
            self._moved_mask[active] = True

        # ----- FINAL_ROTATE phase -----
        completed_rot = np.zeros(n, dtype=bool)
        if final_rotate_mask.any():
            rm = final_rotate_mask
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
            self._pos_buf[rm] = self._rot_p_hold[rm]
            self._moved_mask[rm] = True
            done = tau >= self._rot_t_total[rm]
            if done.any():
                rot_idx = np.flatnonzero(rm)
                done_rows = rot_idx[done]
                # Snap to un-flipped target.
                self._orn_buf[done_rows] = self._rot_snap_quat[done_rows]
                completed_rot[done_rows] = True

        # Write to sim (and refresh per-agent reported velocities).
        self._apply_phase1(dt)

        # Advance translate completions.
        if completed.any():
            for i in np.flatnonzero(completed):
                self._advance_waypoint(int(i), now)

        # Clear final-rotate completions → IDLE.
        if completed_rot.any():
            for i in np.flatnonzero(completed_rot):
                ii = int(i)
                self._phase[ii] = _PHASE_IDLE
                self._agents[ii]._is_moving = False

        return self._moved_mask

    # ------------------------------------------------------------------ #
    # Backward-compatibility shim: _active property
    # ------------------------------------------------------------------ #

    @property
    def _active(self) -> np.ndarray:
        """Legacy boolean view: True for any non-IDLE agent.

        Code that read ``_active`` before the phase refactor can use this
        property without changes; code that *wrote* ``_active`` should be
        updated to use ``_phase`` directly.
        """
        return self._phase != _PHASE_IDLE
