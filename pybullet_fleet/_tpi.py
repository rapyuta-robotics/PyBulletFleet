"""Trapezoidal-velocity profile primitives shared by per-agent and batched controllers.

:func:`build_tpi` is the canonical factory wrapping the external
``TwoPointInterpolation`` package — used by both per-agent
``KinematicController`` and the vectorised ``BatchKinematicController``
subclasses to construct a single trajectory. :func:`try_build_tpi` is the
same construction without the fallback, for callers that have to tell an
infeasible request apart from a trajectory that simply finishes early.

:func:`extract_phase_params` lifts the symmetric trapezoidal phase scalars
(``t_accel``, ``t_const``, ``t_total``, ``accel``) out of a constructed
``TwoPointInterpolation`` so the batched controllers can store them in
``(N,)`` arrays for vectorised evaluation.

:func:`trapezoid_distance` evaluates the distance traveled at time ``tau``
for ``(N,)`` arrays of such trajectories — the batched eval hot path — and
:func:`trapezoid_speed` the speed, its time derivative.

:func:`build_tpi_from_speed` starts a trajectory from a body already moving
along the travel axis, for a goal replaced mid-motion; :func:`try_build_tpi`
with the same ``v0`` says whether it can stop in time.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from two_point_interpolation import TwoPointInterpolation


#: Fraction below ``vmax`` an initial speed is clamped to. The solver treats an
#: initial speed equal to ``vmax`` as exceeding it (and warns), so a body that
#: is cruising at the cap starts a hair below it instead.
_V0_CAP = 1.0 - 1e-9

#: Relative excess over ``vmax`` that still counts as cruising at the cap (float
#: noise). A larger initial speed -- vmax lowered during the motion -- is left
#: to the solver, whose trajectory first slows down to ``vmax``.
_V0_OVER_VMAX_TOL = 1e-6


def build_tpi(
    p0: float,
    pe: float,
    vmax: float,
    accel: float,
    t0: float,
    v0: float = 0.0,
    ve: float = 0.0,
    decel: Optional[float] = None,
) -> TwoPointInterpolation:
    """Construct a ready-to-evaluate :class:`TwoPointInterpolation`.

    Wraps the ``init() + calc_trajectory()`` pair with a zero-distance
    fallback (returns a degenerate ``p0 → p0`` trajectory if the requested
    motion is infeasible). This is the single canonical entry point for
    building a TPI in both per-agent and batched controllers.

    Args:
        p0: Start position along the travel axis.
        pe: End position along the same axis.
        vmax: Speed cap for the cruise phase.
        accel: Acceleration magnitude for the ramp up.
        t0: Simulation time the trajectory starts at; evaluation is relative
            to it.
        v0: Speed at ``p0`` (default: at rest).
        ve: Speed to arrive at ``pe`` with (default: at rest).
        decel: Deceleration magnitude for the ramp down. ``None`` (the
            default) reuses ``accel``, giving the symmetric trapezoid that was
            the only profile this helper could build previously. Pass a
            different value for vehicles that brake harder, or softer, than
            they accelerate.

    Returns:
        A ``TwoPointInterpolation`` with its trajectory already calculated.
    """
    tpi = try_build_tpi(p0=p0, pe=pe, vmax=vmax, accel=accel, t0=t0, v0=v0, ve=ve, decel=decel)
    if tpi is not None:
        return tpi
    dec = accel if decel is None else decel
    tpi = TwoPointInterpolation()
    tpi.init(p0=p0, pe=p0, acc_max=accel, vmax=vmax, t0=t0, v0=0.0, ve=0.0, dec_max=dec)
    tpi.calc_trajectory()
    return tpi


#: A solve kept for the next build: ``(inputs, result)``, set only by a call
#: with ``keep_for_build=True``. A goal set on a moving agent is checked
#: (:func:`pybullet_fleet._replan.replacement_rejection`, which keeps its solve
#: and discards it) and then built with the very same solver inputs; that build
#: takes the kept solve instead of solving twice. A kept solve is handed out at
#: most once and only to an exact match of every input, so the trajectory it
#: returns belongs to that one caller -- no two trajectories ever share a
#: ``TwoPointInterpolation``. One entry, swapped as a whole (atomic under the GIL).
_kept_solve: Optional[Tuple[tuple, Optional[TwoPointInterpolation]]] = None


def try_build_tpi(
    p0: float,
    pe: float,
    vmax: float,
    accel: float,
    t0: float,
    v0: float = 0.0,
    ve: float = 0.0,
    decel: Optional[float] = None,
    keep_for_build: bool = False,
) -> Optional[TwoPointInterpolation]:
    """Build a :class:`TwoPointInterpolation`, or ``None`` if there is none.

    Same arguments and same solver as :func:`build_tpi`, without its
    fallback. A request is infeasible when ``v0`` is too high to reach ``pe``
    at ``ve`` within ``decel`` — the gap between ``p0`` and ``pe`` is shorter
    than the braking distance the speed demands.

    ``v0`` is signed along ``p0 -> pe``: negative is moving away from ``pe``,
    and the trajectory then slows, reverses and drives to ``pe`` (that first
    slowing uses ``accel``, the solver's first-phase limit). A ``v0`` at
    ``vmax`` up to float noise starts a hair below it; from one really above
    ``vmax`` (lowered during the motion) the trajectory first slows down to
    it, again at ``accel``. No limit is ever raised to make one fit.

    :func:`build_tpi` answers that with a degenerate ``p0 → p0`` trajectory,
    which is the right default for a controller that re-plans from where it
    stands. A caller that reads "trajectory over" as "target reached" needs
    the two cases kept apart, and gets ``None`` here instead.

    ``keep_for_build`` is for a feasibility check that discards the result:
    the solve is kept so the next call with exactly the same inputs takes it
    instead of solving again. Any other call solves unless it finds such a
    kept solve, and never keeps its own.

    Returns:
        The calculated trajectory, or ``None`` when the solver rejects the
        request as infeasible.
    """
    global _kept_solve
    dec = accel if decel is None else decel
    if abs(v0) <= vmax * (1.0 + _V0_OVER_VMAX_TOL):
        # The solver treats v0 == vmax as exceeding it.
        v0 = max(-vmax * _V0_CAP, min(float(v0), vmax * _V0_CAP))
    key = (p0, pe, vmax, accel, t0, v0, ve, dec)
    kept = _kept_solve
    if not keep_for_build and kept is not None and kept[0] == key:
        _kept_solve = None
        return kept[1]
    tpi: Optional[TwoPointInterpolation] = TwoPointInterpolation()
    try:
        tpi.init(p0=p0, pe=pe, acc_max=accel, vmax=vmax, t0=t0, v0=v0, ve=ve, dec_max=dec)
        tpi.calc_trajectory()
    except ValueError:
        tpi = None
    if keep_for_build:
        _kept_solve = (key, tpi)
    return tpi


def build_tpi_from_speed(
    p0: float,
    pe: float,
    vmax: float,
    accel: float,
    t0: float,
    v0: float,
    decel: Optional[float] = None,
) -> TwoPointInterpolation:
    """A trajectory to ``pe`` for a body already moving at ``v0`` toward it.

    For a goal replaced mid-motion: the body keeps its (signed) speed instead
    of dropping to rest and accelerating again. From rest this is
    :func:`build_tpi`.

    Raises:
        ValueError: no trajectory within the limits exists (see
            :func:`try_build_tpi`). Callers check first with
            :func:`pybullet_fleet._replan.replacement_rejection`; no limit is
            ever raised to make a goal fit.
    """
    tpi = try_build_tpi(p0=p0, pe=pe, vmax=vmax, accel=accel, t0=t0, v0=v0, decel=decel)
    if tpi is None:
        raise ValueError(
            f"no trajectory over {abs(pe - p0):.6g} from speed {v0:.6g} within accel "
            f"{accel:.6g} / decel {accel if decel is None else decel:.6g}"
        )
    return tpi


def extract_phase_params(tpi: TwoPointInterpolation) -> Tuple[float, float, float, float, float]:
    """Read trapezoidal phase scalars from a constructed TPI.

    Returns ``(t_accel, t_const, t_total, accel, decel)`` relative to
    ``tpi.t0`` in the form expected by :func:`trapezoid_distance` for batched
    eval.

    Handles all three TPI cases:

    - ``case == -1`` (degenerate / zero distance): all zeros.
    - ``case == 0`` (triangle, ``vmax`` not reached): ``t_const = 0``.
    - ``case == 1`` (full trapezoid): three populated phases.

    An asymmetric profile needs nothing special here. ``tpi.dt`` already holds
    the phase durations ``TwoPointInterpolation`` computed from both scalars;
    only ``decel`` had to be carried out alongside ``accel`` for
    :func:`trapezoid_distance` to integrate the braking phase with it.
    """
    dt = tpi.dt
    accel = float(tpi.amax_accel)
    decel = float(tpi.amax_decel)
    if len(dt) == 0:
        return 0.0, 0.0, 0.0, accel, decel
    if len(dt) == 2:
        t_accel = float(dt[0])
        t_decel = float(dt[1])
        return t_accel, 0.0, t_accel + t_decel, accel, decel
    # len == 3 (full trapezoid)
    t_accel = float(dt[0])
    t_const = float(dt[1])
    t_total = float(dt[0] + dt[1] + dt[2])
    return t_accel, t_const, t_total, accel, decel


def trapezoid_distance(
    tau: np.ndarray,
    t_accel: np.ndarray,
    t_const: np.ndarray,
    t_total: np.ndarray,
    accel: np.ndarray,
    total_distance: np.ndarray,
    decel: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorised distance-traveled lookup for trapezoidal profiles.

    All inputs are ``(N,)`` arrays; returns ``(N,)``.

    ``v0`` is the speed each trajectory starts at (``None``: at rest), as built
    by :func:`build_tpi_from_speed`; the ramp up then starts from it.

    ``decel`` is the braking scalar for the final phase. ``None`` reuses
    ``accel``, which is the symmetric profile and what every caller got before
    this parameter existed.
    """
    if decel is None:
        decel = accel
    out = np.zeros_like(tau)
    # phase masks
    in_const = (tau > t_accel) & (tau <= t_accel + t_const)
    in_decel = (tau > t_accel + t_const) & (tau < t_total)
    done = tau >= t_total
    in_accel = (tau > 0.0) & ~in_const & ~in_decel & ~done

    if v0 is None:
        v0 = np.zeros_like(tau)
    # accel phase: v0 * t + 0.5 * a * t^2
    if in_accel.any():
        t = tau[in_accel]
        out[in_accel] = v0[in_accel] * t + 0.5 * accel[in_accel] * t * t
    # const phase: v0 * t_accel + 0.5 * a * t_accel^2 + v_peak * (t - t_accel),
    # where v_peak = v0 + a * t_accel
    if in_const.any():
        ta = t_accel[in_const]
        a = accel[in_const]
        u = v0[in_const]
        out[in_const] = u * ta + 0.5 * a * ta * ta + (u + a * ta) * (tau[in_const] - ta)
    # decel phase: total_distance - 0.5 * a * (t_total - t)^2
    if in_decel.any():
        d = decel[in_decel]
        t_left = t_total[in_decel] - tau[in_decel]
        out[in_decel] = total_distance[in_decel] - 0.5 * d * t_left * t_left
    if done.any():
        out[done] = total_distance[done]
    return out


def trapezoid_speed(
    tau: np.ndarray,
    t_accel: np.ndarray,
    t_const: np.ndarray,
    t_total: np.ndarray,
    accel: np.ndarray,
    decel: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorised speed lookup for the profiles of :func:`trapezoid_distance`.

    Takes the same ``(N,)`` arrays (without ``total_distance``) and returns
    the speed along the travel axis at ``tau``, the time derivative of
    :func:`trapezoid_distance`: signed, so a trajectory that starts moving
    away from its target (negative ``v0``) is negative until it turns. ``v0``
    at ``tau <= 0``, ``0`` from ``t_total`` on.
    """
    if decel is None:
        decel = accel
    if v0 is None:
        v0 = np.zeros_like(tau)
    out = np.zeros_like(tau)
    in_const = (tau > t_accel) & (tau <= t_accel + t_const)
    in_decel = (tau > t_accel + t_const) & (tau < t_total)
    done = tau >= t_total
    in_accel = ~in_const & ~in_decel & ~done
    # accel phase: v0 + a * t (t clipped at 0: before the start it is v0)
    if in_accel.any():
        out[in_accel] = v0[in_accel] + accel[in_accel] * np.maximum(tau[in_accel], 0.0)
    # const phase: v_peak = v0 + a * t_accel
    if in_const.any():
        out[in_const] = v0[in_const] + accel[in_const] * t_accel[in_const]
    # decel phase: d * (t_total - t)
    if in_decel.any():
        out[in_decel] = decel[in_decel] * (t_total[in_decel] - tau[in_decel])
    return out
