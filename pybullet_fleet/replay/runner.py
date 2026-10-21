"""Fresh re-execution and meaningful, streaming comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from pybullet_fleet.geometry import quat_angle_between

from .artifact import ReplayArtifact, assets, environment
from .schema import ReplayError, ReplayInput, clone, number
from .session import ReplaySession


@dataclass(frozen=True)
class Comparison:
    status: str
    first_difference: dict | None = None
    conditions: dict = field(default_factory=dict)


def reexecute(
    artifact: ReplayArtifact | str | Path,
    output: str | Path,
    *,
    pbf_overrides: dict | None = None,
    allow_environment_change: bool = False,
) -> ReplayArtifact:
    """Execute recorded effective inputs, never overwrite results with recorded poses.

    Config overrides/environment changes explicitly create a variant. They are
    recorded as provenance rather than being treated as reproduction success.
    """
    reference = ReplayArtifact.open(artifact.path if isinstance(artifact, ReplayArtifact) else artifact)
    if reference.manifest["assets"] != assets():
        raise ReplayError("unsupported", "missing or mismatched bundled asset")
    current_environment = environment()
    if reference.manifest["environment"] != current_environment and not allow_environment_change:
        raise ReplayError("unsupported", "environment mismatch; opt into a variant explicitly")
    initial = clone(reference.initial)
    if pbf_overrides is not None:
        for key, value in pbf_overrides.items():
            if key == "limits":
                initial["pbf"]["limits"].update(value)
            else:
                initial["pbf"][key] = value
    provenance = {
        "source_run_id": reference.manifest["run_id"],
        "pbf_overrides": clone(pbf_overrides or {}),
        "environment_changed": reference.manifest["environment"] != current_environment,
    }
    intents = (r for r in reference.journal() if r["record_type"] == "command_intent")
    upcoming = next(intents, None)
    with ReplaySession.create(
        initial, output=output, observation_interval=reference.manifest["observation_interval"], provenance=provenance
    ) as session:
        for step in range(reference.completion["final_step"]):
            inputs = []
            while upcoming is not None and upcoming["step"] == step:
                inputs.append(ReplayInput.from_record(upcoming["input"]))
                upcoming = next(intents, None)
            session.step(inputs)
    return ReplayArtifact.open(output)


def _either(left: dict | None, right: dict | None) -> dict:
    """The record of a ``zip_longest`` pair that is present (at least one is)."""
    if left is not None:
        return left
    if right is None:
        raise ValueError("Both records of a compared pair are missing")
    return right


def _difference(left: Any, right: Any, path: str, tolerance: float) -> dict | None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return {"field": path, "expected": sorted(left), "actual": sorted(right)}
        for key in sorted(left):
            diff = _difference(left[key], right[key], f"{path}.{key}", tolerance)
            if diff:
                return diff
        return None
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        if path.endswith(".orientation"):
            angle = quat_angle_between(np.asarray(left), np.asarray(right))
            if angle <= tolerance:
                return None
            return {"field": path, "expected": left, "actual": right, "angular_error": float(angle), "tolerance": tolerance}
        for i, (a, b) in enumerate(zip(left, right)):
            diff = _difference(a, b, f"{path}[{i}]", tolerance)
            if diff:
                return diff
        return None
    # Only observation floating state gets tolerance. Discrete timing/order and
    # journal fields are compared exactly by callers.
    if (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        if abs(left - right) <= tolerance:
            return None
    elif left == right:
        return None
    return {"field": path, "expected": left, "actual": right, "tolerance": tolerance}


def compare(
    reference: ReplayArtifact | str | Path, candidate: ReplayArtifact | str | Path, *, tolerance: float = 1e-6
) -> Comparison:
    """Report first recorded difference; never infer divergence in unsampled steps."""
    number(tolerance, "tolerance")
    if tolerance < 0:
        raise ReplayError("invalid", "tolerance must be nonnegative")
    try:
        left = ReplayArtifact.open(reference.path if isinstance(reference, ReplayArtifact) else reference)
        right = ReplayArtifact.open(candidate.path if isinstance(candidate, ReplayArtifact) else candidate)
    except ReplayError as exc:
        return Comparison(exc.status, {"reason": str(exc)})
    conditions = {}
    for key, a, b in (
        ("initial", left.initial, right.initial),
        ("environment", left.manifest["environment"], right.manifest["environment"]),
        ("assets", left.manifest["assets"], right.manifest["assets"]),
    ):
        if a != b:
            conditions[key] = {"reference": a, "candidate": b}
    if left.initial["world"] != right.initial["world"]:
        return Comparison("unsupported", {"reason": "different initial worlds/identity mappings"}, conditions)
    if left.manifest["observation_interval"] != right.manifest["observation_interval"]:
        return Comparison("unsupported", {"reason": "different observation schedules"}, conditions)
    candidates = []
    for a, b in zip_longest(left.journal(), right.journal()):
        aa = None if a is None else {k: v for k, v in a.items() if k not in ("run_id", "record_seq")}
        bb = None if b is None else {k: v for k, v in b.items() if k not in ("run_id", "record_seq")}
        diff = _difference(aa, bb, "journal", 0.0)
        if diff:
            existing = _either(a, b)
            step = min(r["step"] for r in (a, b) if r is not None)
            phase = existing["phase"]
            candidates.append(((step, 0 if phase == "input" else 1), {"step": step, "phase": phase, **diff}))
            break
    for a, b in zip_longest(left.observations(), right.observations()):
        if a is None or b is None or a["state_step"] != b["state_step"]:
            existing = _either(a, b)
            step = existing["state_step"]
            diff = {"field": "observation.state_step", "expected": a and a["state_step"], "actual": b and b["state_step"]}
        else:
            step = a["state_step"]
            diff = _difference(a["entities"], b["entities"], "entities", tolerance)
        if diff:
            candidates.append(((step - 1, 2), {"state_step": step, "phase": "observation", **diff}))
            break
    if candidates:
        return Comparison("different", min(candidates, key=lambda item: item[0])[1], conditions)
    return Comparison("matched", conditions=conditions)
