"""PBF-owned, deliberately restricted navigation replay data contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

SCHEMA_VERSION = 1
PROFILE = {"id": "pbf.kinematic_navigation", "version": 1}
FRAME = {"position": "world", "up": "z", "length": "m", "time": "s", "orientation": "xyzw"}
LIMITS = {"max_linear_vel": 2.0, "max_linear_accel": 5.0, "max_angular_vel": 2.0, "max_angular_accel": 5.0}


class ReplayError(ValueError):
    """An artifact/profile failure, distinct from a valid run with different results."""

    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(f"{status}: {message}")


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def clone(value: Any) -> Any:
    try:
        return json.loads(encode(value))
    except (ValueError, TypeError) as exc:
        raise ReplayError("invalid", f"expected finite JSON data: {exc}") from exc


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def keys(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ReplayError("invalid", f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ReplayError("unsupported", f"{label}: unknown fields {sorted(unknown)}")


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError("invalid", f"{label} must be finite")
    if positive and value <= 0:
        raise ReplayError("invalid", f"{label} must be positive")
    return float(value)


def vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ReplayError("invalid", f"{label} requires {size} values")
    return [number(v, label) for v in value]


def name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayError("invalid", f"{label} must be a nonempty string")
    return value


def positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ReplayError("invalid", f"{label} must be a positive integer")
    return value


def normalize_initial(value: dict) -> dict:
    """Resolve v1 defaults once; reject unknown fields rather than discarding them."""
    value = clone(value)
    keys(value, {"world", "pbf"}, "initial state")
    world = value.get("world", {})
    keys(world, {"entities", "frame", "state_step", "sim_time"}, "world")
    if world.get("frame", FRAME) != FRAME or world.get("state_step", 0) != 0 or world.get("sim_time", 0) != 0:
        raise ReplayError("unsupported", "initial world must be S_0 in the v1 frame")
    entries = world.get("entities")
    if not isinstance(entries, list) or not entries:
        raise ReplayError("invalid", "world.entities must be a nonempty list")
    entities, ids, names = [], set(), set()
    for entry in entries:
        keys(entry, {"entity_id", "name", "kind", "model", "position", "yaw", "half_extents"}, "entity")
        entity_id = name(entry.get("entity_id", str(uuid4())), "entity_id")
        api_name = name(entry.get("name"), "name")
        if entity_id in ids or api_name in names:
            raise ReplayError("invalid", "duplicate entity_id or Fleet API name")
        ids.add(entity_id)
        names.add(api_name)
        kind = entry.get("kind", "robot")
        if kind not in ("robot", "static_box"):
            raise ReplayError("unsupported", f"entity kind {kind}")
        entity = {
            "entity_id": entity_id,
            "name": api_name,
            "kind": kind,
            "position": vector(entry.get("position", [0, 0, 0.1]), 3, "position"),
            "yaw": number(entry.get("yaw", 0), "yaw"),
        }
        if kind == "robot":
            if entry.get("model", "simple_cube") != "simple_cube" or "half_extents" in entry:
                raise ReplayError("unsupported", "v1 robots use bundled simple_cube")
            entity["model"] = "simple_cube"
        else:
            if "model" in entry:
                raise ReplayError("unsupported", "static_box has no external model")
            extent = vector(entry.get("half_extents", [0.5, 0.5, 0.5]), 3, "half_extents")
            if min(extent) <= 0:
                raise ReplayError("invalid", "half_extents must be positive")
            entity["half_extents"] = extent
        entities.append(entity)
    execution = value.get("pbf", {})
    keys(
        execution,
        {
            "controller",
            "timestep",
            "limits",
            "collision_frequency",
            "collision_margin",
            "position_tolerance",
            "angle_tolerance",
        },
        "pbf",
    )
    controller = execution.get("controller", "omni")
    if controller not in ("omni", "batch_omni"):
        raise ReplayError("unsupported", f"controller {controller}")
    supplied_limits = execution.get("limits", {})
    keys(supplied_limits, set(LIMITS), "limits")
    limits = {k: number(supplied_limits.get(k, v), k, positive=True) for k, v in LIMITS.items()}
    frequency = execution.get("collision_frequency")
    if frequency is not None:
        frequency = number(frequency, "collision_frequency")
        if frequency < 0:
            raise ReplayError("invalid", "collision_frequency must be nonnegative")
    margin = number(execution.get("collision_margin", 0.02), "collision_margin")
    if margin < 0:
        raise ReplayError("invalid", "collision_margin must be nonnegative")
    return {
        "world": {"entities": entities, "frame": dict(FRAME), "state_step": 0, "sim_time": 0.0},
        "pbf": {
            "controller": controller,
            "timestep": number(execution.get("timestep", 0.1), "timestep", positive=True),
            "limits": limits,
            "collision_frequency": frequency,
            "collision_margin": margin,
            "position_tolerance": number(execution.get("position_tolerance", 0.001), "position_tolerance", positive=True),
            "angle_tolerance": number(execution.get("angle_tolerance", 0.001), "angle_tolerance", positive=True),
        },
    }


@dataclass(frozen=True)
class ReplayInput:
    """Effective PBF input, after producer-specific translation; producer is only provenance.

    The payload is copied and validated at the input boundary. Repeated command IDs
    are allowed: (step, order) identifies each application, not command_id.
    """

    command_type: str
    payload: dict
    source: str = "python"
    command_id: str = field(default_factory=lambda: str(uuid4()))
    allowed_names: tuple[str, ...] | None = None

    @classmethod
    def navigate(
        cls, target: str, position: Iterable[float], *, yaw: float = 0.0, z: float | None = None, **kwargs
    ) -> ReplayInput:
        return cls("navigate", {"goals": [{"name": target, "position": list(position), "yaw": yaw, "z": z}]}, **kwargs)

    @classmethod
    def stop(cls, targets: Iterable[str], **kwargs) -> ReplayInput:
        return cls("stop", {"names": list(targets)}, **kwargs)

    def to_record(self) -> dict:
        payload = clone(self.payload)
        name(self.source, "source")
        name(self.command_id, "command_id")
        if self.command_type == "navigate":
            keys(payload, {"goals"}, "navigate")
            if not isinstance(payload.get("goals"), list):
                raise ReplayError("invalid", "navigate.goals must be a list")
            goals = []
            for goal in payload["goals"]:
                keys(goal, {"name", "position", "yaw", "z", "command_id"}, "goal")
                item = {
                    "name": name(goal.get("name"), "goal.name"),
                    "position": vector(goal.get("position"), 2, "goal.position"),
                    "yaw": number(goal.get("yaw", 0), "goal.yaw"),
                    # None (or absent) keeps the robot's present height.
                    "z": None if goal.get("z") is None else number(goal["z"], "goal.z"),
                }
                item["command_id"] = None if goal.get("command_id") is None else name(goal["command_id"], "goal.command_id")
                goals.append(item)
            payload = {"goals": goals}
        elif self.command_type == "stop":
            keys(payload, {"names"}, "stop")
            if not isinstance(payload.get("names"), list):
                raise ReplayError("invalid", "stop.names must be a list")
            payload = {"names": [name(n, "target") for n in payload["names"]]}
        else:
            raise ReplayError("unsupported", f"command {self.command_type}")
        if self.allowed_names is not None and not isinstance(self.allowed_names, (tuple, list)):
            raise ReplayError("invalid", "allowed_names must be a sequence of names, not a string")
        allowed = None if self.allowed_names is None else [name(n, "allowed name") for n in self.allowed_names]
        return {
            "command_type": self.command_type,
            "payload": payload,
            "source": self.source,
            "command_id": self.command_id,
            "allowed_names": allowed,
        }

    @classmethod
    def from_record(cls, record: dict) -> ReplayInput:
        keys(record, {"command_type", "payload", "source", "command_id", "allowed_names"}, "input")
        try:
            result = cls(**record)
            result.to_record()
            return result
        except TypeError as exc:
            raise ReplayError("invalid", f"malformed command: {exc}") from exc
