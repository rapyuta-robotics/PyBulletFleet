"""Primitive, JSON-compatible state for an active TPI trajectory."""

from __future__ import annotations

import math

from two_point_interpolation import TwoPointInterpolation

from pybullet_fleet._tpi import try_build_tpi


def capture_tpi_state(tpi: TwoPointInterpolation) -> dict[str, float]:
    """Capture solver inputs, including asymmetric braking, without pickling TPI."""
    return {
        "p0": float(tpi.p0),
        "pe": float(tpi.pe),
        "v0": float(tpi.v0),
        "ve": float(tpi.ve),
        "vmax": float(tpi.vmax),
        "accel": float(tpi.amax_accel),
        "decel": float(tpi.amax_decel),
        "t0": float(tpi.t0),
    }


def restore_tpi_state(state: dict[str, float]) -> TwoPointInterpolation:
    """Validate primitive inputs and reconstruct the same solved trajectory."""
    required = {"p0", "pe", "v0", "ve", "vmax", "accel", "decel", "t0"}
    if (
        not isinstance(state, dict)
        or set(state) != required
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in state.values()
        )
        or any(state[key] <= 0 for key in ("vmax", "accel", "decel"))
    ):
        raise ValueError("Invalid TPI checkpoint state")
    trajectory = try_build_tpi(**state)
    if trajectory is None:
        raise ValueError("TPI checkpoint trajectory cannot be rebuilt")
    return trajectory
