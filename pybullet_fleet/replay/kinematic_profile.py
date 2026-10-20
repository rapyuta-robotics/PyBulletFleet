"""Declared physics-off owner handlers for supported execution checkpoints.

This profile restores PBF-owned kinematic state. Application callbacks and
their mutable state are supplied by separately registered DataRecord restorers.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import pybullet as p

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.controller import DifferentialController, OmniController
from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import PRIMITIVE_SHAPE_TYPES, ShapeParams, SimObject, SimObjectSpawnParams
from pybullet_fleet.types import CollisionMode, ControllerMode, MotionMode, MovementDirection

from .state_recording import (
    ResultPlayback,
    _keys,
    _pose_from_record,
    _pose_record,
    _read_json,
    _sha256,
    _vector,
    register_recording_profile,
    restore_supported_simulation,
)

PROFILE_ID = "pbf.kinematic.v4"
PROFILE_VERSION = 4
_SUPPORTED_CONTROLLER_NAMES_BY_TYPE = {
    OmniController: OmniController._registry_name,
    DifferentialController: DifferentialController._registry_name,
}
_SUPPORTED_CONTROLLER_TYPES = frozenset(_SUPPORTED_CONTROLLER_NAMES_BY_TYPE)
_SUPPORTED_CONTROLLER_NAMES = frozenset(_SUPPORTED_CONTROLLER_NAMES_BY_TYPE.values())


def _asset_fingerprint(path: str) -> str:
    asset = Path(path)
    if not asset.is_file():
        raise ValueError(f"Checkpoint asset is missing: {path}")
    return hashlib.sha256(asset.read_bytes()).hexdigest()


def _controller_spec(agent: Agent) -> dict:
    controller = agent.controller
    if not agent.is_moving and controller._mode is not ControllerMode.IDLE:
        raise ValueError(f"Unobserved controller execution for {agent.name!r}")
    kind = _SUPPORTED_CONTROLLER_NAMES_BY_TYPE.get(type(controller))
    if kind is None:
        raise ValueError(f"Unsupported controller execution for {agent.name!r}")
    config = controller._construction_config()
    if config is None:
        raise ValueError(f"Unsupported controller construction for {agent.name!r}")
    return {"type": kind, "parameters": {key: value for key, value in config.items() if key != "type"}}


def _validate_controller_parameters(parameters: dict) -> None:
    if not isinstance(parameters, dict) or set(parameters) != {field.name for field in fields(ControllerParams)}:
        raise ValueError("Unsupported controller parameters")
    for name in ("max_linear_vel", "max_angular_vel", "max_linear_accel", "max_linear_decel", "max_angular_accel"):
        value = parameters[name]
        if value is None:
            continue
        if isinstance(value, list):
            if len(value) != 3:
                raise ValueError(f"Invalid controller limit: {name}")
            values = value
        else:
            values = [value]
        if any(type(item) not in (int, float) or not math.isfinite(item) or item <= 0 for item in values):
            raise ValueError(f"Invalid controller limit: {name}")
    timeout = parameters["cmd_vel_timeout"]
    if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0):
        raise ValueError("Invalid controller timeout")
    if parameters["navigation_2d"] is not None and type(parameters["navigation_2d"]) is not bool:
        raise ValueError("Invalid controller navigation_2d")
    if not isinstance(parameters["default_direction"], str) or parameters["default_direction"] not in {
        direction.value for direction in MovementDirection
    }:
        raise ValueError("Invalid controller default direction")


def _check_agent_owners(agent: Agent) -> None:
    if (
        type(agent) is not Agent
        or agent.plugins
        or agent.callbacks
        or agent._batch_controller is not None
        or not agent.is_action_queue_empty()
    ):
        raise ValueError(f"Unsupported Agent execution state: {agent.name!r}")
    if len(agent.controllers) != 1 or (agent._events is not None and agent._events._handlers):
        raise ValueError(f"Undeclared Agent controller or event state: {agent.name!r}")


def capture_agent_state(agent: Agent) -> dict:
    """Capture supported base, joint and controller execution from an Agent."""
    _check_agent_owners(agent)
    spec = _controller_spec(agent)
    navigation = agent.controller.capture_navigation_state() if agent.is_moving else None
    joints = agent.capture_kinematic_joint_execution() if agent.joint_info else []
    return {
        "pose": _pose_record(agent.get_pose()),
        "velocity": agent.get_velocity().tolist(),
        "angular_velocity": float(agent.angular_velocity),
        "moving": agent.is_moving,
        "controller": {"type": spec["type"], "version": 1, "state": navigation},
        "joints": joints,
    }


def restore_agent_state(agent: Agent, value: dict) -> None:
    """Apply supported Agent state without reissuing a movement command."""
    pose = _pose_from_record(value["pose"])
    agent.restore_motion_state(pose, value["velocity"], value["angular_velocity"], moving=value["moving"])
    # Base reset can invalidate PyBullet's articulated-link transforms. Apply
    # joints afterwards, so the first resumed attachment update uses restored FK.
    if agent.joint_info:
        agent.restore_kinematic_joint_execution(value["joints"])
    elif value["joints"]:
        raise ValueError("Jointless Agent has checkpoint joints")
    navigation = value["controller"]["state"]
    if navigation is not None:
        if type(agent.controller) not in _SUPPORTED_CONTROLLER_TYPES:
            raise ValueError("Unsupported navigation execution")
        agent.controller.restore_navigation_state(navigation, pose.orientation)
    elif value["moving"]:
        raise ValueError("Moving Agent lacks controller state")


def _capture_primitive_shape(shape: ShapeParams | None, *, visual: bool) -> dict | None:
    if shape is None:
        return None
    if shape.shape_type not in PRIMITIVE_SHAPE_TYPES:
        raise ValueError(f"Unsupported primitive shape: {shape.shape_type!r}")
    value = {"type": shape.shape_type, "frame_pose": _pose_record(shape.frame_pose) if shape.frame_pose else None}
    if shape.shape_type == "box":
        value["half_extents"] = list(shape.half_extents)
    else:
        value["radius"] = shape.radius
        if shape.shape_type == "cylinder":
            value["height"] = shape.height
    if visual:
        value["rgba_color"] = list(shape.rgba_color) if shape.rgba_color is not None else None
    return value


def _restore_primitive_shape(value: dict | None) -> ShapeParams | None:
    if value is None:
        return None
    frame = value["frame_pose"]
    kwargs = {key: item for key, item in value.items() if key not in {"type", "frame_pose"}}
    return ShapeParams(
        shape_type=value["type"],
        frame_pose=_pose_from_record(frame) if frame is not None else None,
        **kwargs,
    )


def capture_primitive_state(obj: SimObject) -> dict:
    """Capture a constructed, physics-off primitive and its attachment."""
    if type(obj) is not SimObject or obj.callbacks or (obj._events is not None and obj._events._handlers):
        raise ValueError(f"Unsupported object execution state: {obj.name!r}")
    params = obj._checkpoint_spawn_params
    if (
        params is None
        or params.urdf_path is not None
        or (params.visual_shape is None and params.collision_shape is None)
        or obj.mass != 0
        or obj.user_data
        or obj.pickable != params.pickable
    ):
        raise ValueError(f"Object lacks supported primitive construction: {obj.name!r}")
    return {
        "pose": _pose_record(obj.get_pose()),
        "attachment": obj.get_kinematic_attachment_state(),
        "construction": {
            "visual_shape": _capture_primitive_shape(params.visual_shape, visual=True),
            "collision_shape": _capture_primitive_shape(params.collision_shape, visual=False),
            "pickable": params.pickable,
            "collision_mode": obj.collision_mode.value,
        },
    }


def restore_primitive(sim: MultiRobotSimulationCore, name: str, value: dict) -> SimObject:
    """Construct a checkpoint primitive; attachments follow owner creation."""
    c = value["construction"]
    return SimObject.from_params(
        SimObjectSpawnParams(
            name=name,
            initial_pose=_pose_from_record(value["pose"]),
            mass=0.0,
            pickable=c["pickable"],
            collision_mode=CollisionMode(c["collision_mode"]),
            visual_shape=_restore_primitive_shape(c["visual_shape"]),
            collision_shape=_restore_primitive_shape(c["collision_shape"]),
        ),
        sim,
    )


def _remove_playback_primitive(sim: MultiRobotSimulationCore, obj: SimObject) -> None:
    attachment = obj.get_kinematic_attachment_state()
    if attachment is not None:
        parent = sim.get_unique_agent_by_name(attachment["parent_name"])
        if not parent.detach_object(obj):
            raise ValueError("Playback attachment detachment failed")
    sim.remove_object(obj)


def restore_attachment(sim: MultiRobotSimulationCore, obj: SimObject, value: dict) -> None:
    """Apply a named parent/link relation after Agent and primitive construction."""
    attachment = value["attachment"]
    if attachment is None:
        return
    parent = sim.get_unique_agent_by_name(attachment["parent_name"])
    link_names = {
        p.getJointInfo(parent.body_id, i, physicsClientId=sim.client)[12].decode("utf-8")
        for i in range(p.getNumJoints(parent.body_id, physicsClientId=sim.client))
    }
    if attachment["parent_link"] != "base_link" and attachment["parent_link"] not in link_names:
        raise ValueError("Checkpoint attachment link is absent")
    if not parent.attach_object(
        obj,
        parent_link_index=attachment["parent_link"],
        relative_pose=Pose(position=attachment["relative_position"], orientation=attachment["relative_orientation"]),
    ):
        raise ValueError("Checkpoint attachment restoration failed")


def _validate_agent(name: str, construction: dict, entry: dict) -> None:
    _keys(entry, {"type", "version", "state"}, "Agent entry")
    if entry["type"] != "kinematic_agent" or entry["version"] != 2:
        raise ValueError(f"Unsupported Agent state type: {name!r}")
    value = _keys(entry["state"], {"pose", "velocity", "angular_velocity", "moving", "controller", "joints"}, "Agent state")
    _vector(value["pose"]["position"], 3, "Agent position")
    _quaternion(value["pose"]["orientation"], "Agent orientation")
    _vector(value["velocity"], 3, "Agent velocity")
    angular_velocity = value["angular_velocity"]
    if (
        type(value["moving"]) is not bool
        or isinstance(angular_velocity, bool)
        or not isinstance(angular_velocity, (int, float))
        or not math.isfinite(angular_velocity)
    ):
        raise ValueError("Invalid Agent movement state")
    controller = _keys(value["controller"], {"type", "version", "state"}, "controller")
    if (
        controller["version"] != 1
        or controller["type"] != construction["controller"]["type"]
        or controller["type"] not in _SUPPORTED_CONTROLLER_NAMES
        or value["moving"] != (controller["state"] is not None)
    ):
        raise ValueError(f"Unsupported controller state: {name!r}")
    if controller["state"] is not None:
        _validate_navigation(controller["state"], differential=controller["type"] == DifferentialController._registry_name)
    if not isinstance(value["joints"], list) or len({j["name"] for j in value["joints"]}) != len(value["joints"]):
        raise ValueError("Invalid joint roster")
    for joint in value["joints"]:
        _keys(
            joint,
            {"name", "position", "target", "max_velocity", "max_accel", "max_decel", "speed", "braking", "trajectory"},
            "joint",
        )
        if not isinstance(joint["name"], str) or not joint["name"]:
            raise ValueError("Invalid joint identity")
        for field in ("position", "target", "max_velocity", "max_accel", "max_decel", "speed"):
            number = joint[field]
            if number is not None and (
                isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)
            ):
                raise ValueError("Invalid joint value")
        if joint["position"] is None:
            raise ValueError("Joint position is missing")
        if joint["speed"] is None or type(joint["braking"]) is not bool:
            raise ValueError("Invalid joint execution")
        if any(joint[field] is not None and joint[field] <= 0 for field in ("max_velocity", "max_accel", "max_decel")):
            raise ValueError("Invalid joint motion limits")
        if joint["max_decel"] is not None and joint["max_accel"] is None:
            raise ValueError("Joint deceleration requires acceleration")
        if joint["max_accel"] is None and (joint["speed"] != 0 or joint["braking"] or joint["trajectory"] is not None):
            raise ValueError("Unconfigured joint has ramp execution")
        trajectory = joint["trajectory"]
        if trajectory is not None:
            _keys(
                trajectory,
                {"p0", "pe", "v0", "ve", "vmax", "accel", "decel", "t0_offset", "target"},
                "joint trajectory",
            )
            if joint["braking"] or any(
                isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)
                for number in trajectory.values()
            ):
                raise ValueError("Invalid joint trajectory")
            if any(trajectory[field] <= 0 for field in ("vmax", "accel", "decel")):
                raise ValueError("Invalid joint trajectory limits")
            if joint["target"] is None or trajectory["target"] != joint["target"]:
                raise ValueError("Joint trajectory target differs from joint target")


def _quaternion(value: object, label: str) -> None:
    components = _vector(value, 4, label)
    if not math.isclose(sum(component * component for component in components), 1.0, rel_tol=0, abs_tol=1e-4):
        raise ValueError(f"Invalid {label} quaternion")


def _validate_navigation(value: dict, *, differential: bool) -> None:
    """Check the supported pose-path fields before constructing a core."""
    navigation = _keys(
        value,
        {
            "path",
            "waypoint_index",
            "goal",
            "align_final_orientation",
            "final_target_orientation",
            "final_alignment_in_progress",
            "phase",
            "forward",
            "rotation",
        }
        | ({"movement_direction", "requested_direction", "forward_origin"} if differential else set()),
        "navigation",
    )
    if differential:
        if navigation["movement_direction"] not in {MovementDirection.FORWARD.value, MovementDirection.BACKWARD.value}:
            raise ValueError("Invalid differential movement direction")
        if navigation["requested_direction"] not in {direction.value for direction in MovementDirection}:
            raise ValueError("Invalid differential requested direction")
        if navigation["forward_origin"] is None:
            if not navigation["final_alignment_in_progress"]:
                raise ValueError("Differential navigation origin is missing")
        else:
            _vector(navigation["forward_origin"], 3, "differential origin")
    if not isinstance(navigation["path"], list):
        raise ValueError("Invalid navigation path")
    for pose in [*navigation["path"], navigation["goal"]]:
        _keys(pose, {"position", "orientation"}, "navigation pose")
        _vector(pose["position"], 3, "navigation position")
        _quaternion(pose["orientation"], "navigation orientation")
    index = navigation["waypoint_index"]
    final = navigation["final_alignment_in_progress"]
    if (
        type(index) is not int
        or type(final) is not bool
        or type(navigation["align_final_orientation"]) is not bool
        or (not final and (not navigation["path"] or not 0 <= index < len(navigation["path"])))
        or (final and navigation["path"])
    ):
        raise ValueError("Invalid navigation progress")
    if not final and navigation["goal"] != navigation["path"][index]:
        raise ValueError("Navigation goal differs from active waypoint")
    target = navigation["final_target_orientation"]
    if target is not None:
        _quaternion(target, "final orientation")
    elif navigation["align_final_orientation"]:
        raise ValueError("Missing final orientation target")
    phase = navigation["phase"]
    if phase == "forward":
        if navigation["rotation"] is not None:
            raise ValueError("Forward phase has rotation state")
        trajectory = _keys(
            navigation["forward"], {"origin", "t0", "vmax", "accel", "decel", "orientation_after_rotation"}, "forward"
        )
        if trajectory["origin"] is not None:
            _vector(trajectory["origin"], 3, "forward origin")
        elif not final:
            raise ValueError("Forward origin is missing")
        if trajectory["orientation_after_rotation"] is not None:
            _quaternion(trajectory["orientation_after_rotation"], "forward orientation")
    elif phase == "rotate":
        if navigation["forward"] is not None:
            raise ValueError("Rotation phase has forward state")
        trajectory = navigation["rotation"]
        if trajectory is not None:
            trajectory = _keys(
                trajectory, {"start_orientation", "target_orientation", "t0", "vmax", "accel", "decel"}, "rotation"
            )
            _quaternion(trajectory["start_orientation"], "rotation start")
            _quaternion(trajectory["target_orientation"], "rotation target")
    else:
        raise ValueError("Unsupported navigation phase")
    if trajectory is not None:
        for field in ("t0", "vmax", "accel", "decel"):
            number = trajectory[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError("Invalid navigation trajectory")
        if trajectory["vmax"] <= 0 or trajectory["accel"] <= 0 or trajectory["decel"] <= 0:
            raise ValueError("Invalid navigation limits")


def _validate_primitive_shape(value: dict | None, *, visual: bool) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("Invalid primitive shape")
    shape = value
    kind = shape.get("type")
    if not isinstance(kind, str) or kind not in PRIMITIVE_SHAPE_TYPES:
        raise ValueError("Unsupported primitive shape")
    fields = {"type", "frame_pose", "half_extents" if kind == "box" else "radius"}
    if kind == "cylinder":
        fields.add("height")
    if visual:
        fields.add("rgba_color")
    _keys(shape, fields, "primitive shape")
    if kind == "box":
        if any(number <= 0 for number in _vector(shape["half_extents"], 3, "primitive half extents")):
            raise ValueError("Invalid primitive dimensions")
    else:
        dimensions = ("radius", "height") if kind == "cylinder" else ("radius",)
        if any(
            isinstance(shape[field], bool)
            or not isinstance(shape[field], (int, float))
            or not math.isfinite(shape[field])
            or shape[field] <= 0
            for field in dimensions
        ):
            raise ValueError("Invalid primitive dimensions")
    if shape["frame_pose"] is not None:
        _keys(shape["frame_pose"], {"position", "orientation"}, "primitive frame")
        _vector(shape["frame_pose"]["position"], 3, "primitive frame position")
        _quaternion(shape["frame_pose"]["orientation"], "primitive frame orientation")
    if visual and shape["rgba_color"] is not None:
        _vector(shape["rgba_color"], 4, "primitive color")


def _validate_primitive(name: str, entry: dict, agent_names: set[str]) -> None:
    _keys(entry, {"type", "version", "state"}, "object entry")
    if entry["type"] != "kinematic_primitive" or entry["version"] != 1:
        raise ValueError(f"Unsupported object state type: {name!r}")
    value = _keys(entry["state"], {"pose", "attachment", "construction"}, "primitive state")
    _vector(value["pose"]["position"], 3, "primitive position")
    _quaternion(value["pose"]["orientation"], "primitive orientation")
    c = _keys(
        value["construction"],
        {"visual_shape", "collision_shape", "pickable", "collision_mode"},
        "primitive construction",
    )
    if c["visual_shape"] is None and c["collision_shape"] is None:
        raise ValueError("Primitive has no shape")
    _validate_primitive_shape(c["visual_shape"], visual=True)
    _validate_primitive_shape(c["collision_shape"], visual=False)
    if (
        type(c["pickable"]) is not bool
        or not isinstance(c["collision_mode"], str)
        or c["collision_mode"] not in {mode.value for mode in CollisionMode}
    ):
        raise ValueError("Invalid primitive construction")
    attachment = value["attachment"]
    if attachment is not None:
        _keys(attachment, {"parent_name", "parent_link", "relative_position", "relative_orientation"}, "attachment")
        if attachment["parent_name"] not in agent_names or not isinstance(attachment["parent_link"], str):
            raise ValueError("Invalid attachment parent")
        _vector(attachment["relative_position"], 3, "attachment offset")
        _quaternion(attachment["relative_orientation"], "attachment orientation")


class KinematicProfile:
    """Supported URDF Agents and constructed primitives in a physics-off world."""

    profile_id = PROFILE_ID
    version = PROFILE_VERSION
    coverage = {
        "inputs": "only explicitly instrumented effective PBF operations",
        "results": "each completed step of declared physics-off kinematic owners",
        "checkpoints": (
            "URDF Agent base and joint motion, active omni/differential pose navigation, "
            "constructed primitives and attachments"
        ),
        "unsupported": ("Actions, BTs, plugins, collision state, physics and undeclared callbacks"),
    }

    def __init__(
        self, *, allowed_callbacks: tuple[Callable, ...] = (), allowed_event_handlers: tuple[Callable, ...] = ()
    ) -> None:
        self.allowed_callbacks = allowed_callbacks
        self.allowed_event_handlers = allowed_event_handlers
        self._recorded_agent_modes: dict[str, tuple[str, str]] = {}
        self._recorded_agent_specs: dict[str, dict] = {}
        self._asset_cache: dict[str, tuple[str, int, int]] = {}

    def _check_asset(self, path: str, expected: str, *, force: bool = False) -> None:
        asset = Path(path)
        try:
            stat = asset.stat()
        except FileNotFoundError as exc:
            raise ValueError(f"Checkpoint asset is missing: {path}") from exc
        cached = self._asset_cache.get(path)
        if force or cached != (expected, stat.st_mtime_ns, stat.st_size):
            if _asset_fingerprint(path) != expected:
                raise ValueError("Agent asset changed since recording")
            self._asset_cache[path] = (expected, stat.st_mtime_ns, stat.st_size)

    @classmethod
    def from_manifest(cls, manifest: dict) -> "KinematicProfile":
        return cls()

    def _check_live_extensions(self, sim: MultiRobotSimulationCore) -> None:
        if sim._plugins or sim._behavior_tree_callbacks or sim._registered_managers:
            raise ValueError("Plugins, BTs or managers are outside this checkpoint profile")
        allowed = {id(callback) for callback in self.allowed_callbacks}
        if any(id(info["func"]) not in allowed or info["gate"].frequency is not None for info in sim._callbacks):
            raise ValueError("Undeclared simulation callback is not checkpointable")
        event_allowed = {id(handler) for handler in self.allowed_event_handlers}
        if any(id(handler) not in event_allowed for handlers in sim.events._handlers.values() for _, handler in handlers):
            raise ValueError("Undeclared event handler is not checkpointable")

    def construction(self, sim: MultiRobotSimulationCore) -> dict:
        if sim.params.physics or sim._collision_check_frequency != 0:
            raise ValueError("Kinematic recording requires physics and collision checks disabled")
        self._check_live_extensions(sim)
        self._recorded_agent_modes = {}
        self._recorded_agent_specs = {}
        self._asset_cache = {}
        agents = {}
        names = set()
        for agent in sim.agents:
            _check_agent_owners(agent)
            params = agent._checkpoint_spawn_params
            if (
                type(params) is not AgentSpawnParams
                or not isinstance(agent.name, str)
                or not agent.name
                or agent.name in names
                or not params.urdf_path
                or params.user_data
                or params.pickable
            ):
                raise ValueError("Supported Agents require effective URDF spawn parameters and unique names")
            if params.global_scaling != 1.0 or params.ik_params is not None or params.joint_tolerance is not None:
                raise ValueError(f"Unsupported Agent spawn options for kinematic checkpoint: {agent.name!r}")
            names.add(agent.name)
            spec = _controller_spec(agent)
            asset = str(Path(params.urdf_path).resolve())
            digest = _asset_fingerprint(asset)
            stat = Path(asset).stat()
            self._asset_cache[asset] = (digest, stat.st_mtime_ns, stat.st_size)
            self._recorded_agent_modes[agent.name] = (agent.motion_mode.value, agent.collision_mode.value)
            self._recorded_agent_specs[agent.name] = spec
            agents[agent.name] = {
                "asset": asset,
                "asset_sha256": digest,
                "initial_pose": _pose_record(params.initial_pose),
                "mass": params.mass,
                "fixed_base": params.use_fixed_base,
                "collision_mode": agent.collision_mode.value,
                "motion_mode": agent.motion_mode.value,
                "controller": spec,
            }
        for obj in sim.sim_objects:
            if obj not in sim.agents:
                if not isinstance(obj.name, str) or not obj.name or obj.name in names:
                    raise ValueError("Supported objects require unique names")
                names.add(obj.name)
                capture_primitive_state(obj)
        return {
            "timestep": sim.params.timestep,
            "physics": False,
            "enable_floor": sim.params.enable_floor,
            "collision_check_frequency": 0,
            "agents": agents,
        }

    def spawn_record(self, obj: SimObject, params: SimObjectSpawnParams) -> dict:
        if not obj.name:
            raise ValueError("Runtime primitives require a stable name")
        capture_primitive_state(obj)
        return {"key": obj.name, "pose": _pose_record(params.initial_pose)}

    def capture(self, sim: MultiRobotSimulationCore) -> dict:
        if sim.params.physics or sim._collision_check_frequency != 0:
            raise ValueError("Kinematic recording requires physics and collision checks disabled")
        self._check_live_extensions(sim)
        step, elapsed = sim.get_completed_step_boundary()
        agents = {}
        objects = {}
        names = set()
        for agent in sim.agents:
            if not agent.name or agent.name in names:
                raise ValueError("Duplicate or missing Agent checkpoint name")
            params = agent._checkpoint_spawn_params
            if type(params) is not AgentSpawnParams or agent.motion_mode != params.motion_mode:
                raise ValueError(f"Agent motion mode changed after construction: {agent.name!r}")
            expected_modes = self._recorded_agent_modes.get(agent.name)
            if expected_modes is not None and expected_modes != (agent.motion_mode.value, agent.collision_mode.value):
                raise ValueError(f"Agent mode changed after recording setup: {agent.name!r}")
            if (
                agent.user_data
                or agent.pickable != params.pickable
                or (
                    agent.name in self._recorded_agent_specs
                    and _controller_spec(agent) != self._recorded_agent_specs[agent.name]
                )
            ):
                raise ValueError(f"Unsupported Agent metadata or controller change: {agent.name!r}")
            names.add(agent.name)
            agents[agent.name] = {"type": "kinematic_agent", "version": 2, "state": capture_agent_state(agent)}
        for obj in sim.sim_objects:
            if obj in sim.agents:
                continue
            if not obj.name or obj.name in names:
                raise ValueError("Duplicate or missing object checkpoint name")
            names.add(obj.name)
            objects[obj.name] = {"type": "kinematic_primitive", "version": 1, "state": capture_primitive_state(obj)}
        return {
            "sim": {"type": "kinematic", "version": 1, "step": step, "elapsed_time": elapsed},
            "agents": agents,
            "objects": objects,
        }

    def validate_checkpoint(self, manifest: dict, state: dict) -> None:
        c = manifest["construction"]
        if state["sim"]["type"] != "kinematic" or state["sim"]["version"] != 1:
            raise ValueError("Unsupported kinematic checkpoint state")
        _keys(c, {"timestep", "physics", "enable_floor", "collision_check_frequency", "agents"}, "construction")
        if (
            c["physics"] is not False
            or c["collision_check_frequency"] != 0
            or type(c["enable_floor"]) is not bool
            or not isinstance(c["agents"], dict)
        ):
            raise ValueError("Unsupported kinematic construction")
        if set(state["agents"]) != set(c["agents"]) or set(state["objects"]) & set(c["agents"]):
            raise ValueError("Checkpoint entity roster differs from construction")
        for name, construction in c["agents"].items():
            if not name or not isinstance(name, str):
                raise ValueError("Invalid Agent key")
            _keys(
                construction,
                {"asset", "asset_sha256", "initial_pose", "mass", "fixed_base", "collision_mode", "motion_mode", "controller"},
                "Agent construction",
            )
            _keys(construction["initial_pose"], {"position", "orientation"}, "Agent initial pose")
            _vector(construction["initial_pose"]["position"], 3, "Agent initial position")
            _quaternion(construction["initial_pose"]["orientation"], "Agent initial orientation")
            if not isinstance(construction["motion_mode"], str) or construction["motion_mode"] not in {
                mode.value for mode in MotionMode
            }:
                raise ValueError("Unsupported Agent motion mode")
            mass = construction["mass"]
            if mass is not None and (type(mass) not in (int, float) or not math.isfinite(mass) or mass < 0):
                raise ValueError("Invalid Agent mass")
            if type(construction["fixed_base"]) is not bool:
                raise ValueError("Invalid Agent fixed_base")
            if not isinstance(construction["collision_mode"], str) or construction["collision_mode"] not in {
                mode.value for mode in CollisionMode
            }:
                raise ValueError("Invalid Agent collision mode")
            if (
                not isinstance(construction["asset"], str)
                or not Path(construction["asset"]).is_absolute()
                or not isinstance(construction["asset_sha256"], str)
                or len(construction["asset_sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in construction["asset_sha256"])
            ):
                raise ValueError("Invalid Agent asset identifier")
            self._check_asset(construction["asset"], construction["asset_sha256"])
            controller = _keys(construction["controller"], {"type", "parameters"}, "controller construction")
            parameters = controller["parameters"]
            if not isinstance(controller["type"], str) or controller["type"] not in _SUPPORTED_CONTROLLER_NAMES:
                raise ValueError("Unsupported controller construction")
            _validate_controller_parameters(parameters)
            _validate_agent(name, construction, state["agents"][name])
        for name, entry in state["objects"].items():
            if not name or not isinstance(name, str):
                raise ValueError("Invalid object key")
            _validate_primitive(name, entry, set(c["agents"]))

    def restore(self, manifest: dict, state: dict, *, gui: bool, target_rtf: float) -> MultiRobotSimulationCore:
        self.validate_checkpoint(manifest, state)
        c = manifest["construction"]
        for construction in c["agents"].values():
            self._check_asset(construction["asset"], construction["asset_sha256"], force=True)
        sim = MultiRobotSimulationCore(
            SimulationParams(
                gui=gui,
                physics=False,
                enable_floor=c["enable_floor"],
                timestep=c["timestep"],
                target_rtf=target_rtf,
                collision_check_frequency=0,
                monitor=False,
                enable_monitor_gui=False,
                log_level="warning",
            )
        )
        try:
            for name, construction in c["agents"].items():
                controller = construction["controller"]
                Agent.from_params(
                    AgentSpawnParams(
                        name=name,
                        urdf_path=construction["asset"],
                        initial_pose=_pose_from_record(construction["initial_pose"]),
                        mass=construction["mass"],
                        use_fixed_base=construction["fixed_base"],
                        collision_mode=CollisionMode(construction["collision_mode"]),
                        motion_mode=MotionMode(construction["motion_mode"]),
                        controller={
                            "type": controller["type"],
                            **controller["parameters"],
                        },
                    ),
                    sim,
                )
            sim.initialize_simulation()
            primitives = {}
            for name, entry in state["objects"].items():
                primitives[name] = restore_primitive(sim, name, entry["state"])
            for name, entry in state["agents"].items():
                restore_agent_state(sim.get_unique_agent_by_name(name), entry["state"])
            for name, entry in state["objects"].items():
                restore_attachment(sim, primitives[name], entry["state"])
            sim.restore_completed_step_boundary(state["sim"]["step"], state["sim"]["elapsed_time"])
            return sim
        except BaseException:
            if p.isConnected(sim.client):
                p.disconnect(sim.client)
            raise

    def play_gui(
        self,
        playback: ResultPlayback,
        *,
        rtf: float,
        hold: bool,
        camera_config: dict | None = None,
    ) -> None:
        """Display saved kinematic frames without advancing the simulation."""
        manifest = playback.manifest
        checkpoint_steps = sorted(int(step) for step in manifest["checkpoint_sha256"])
        if not checkpoint_steps:
            raise ValueError("GUI playback requires a checkpoint")
        first_step = checkpoint_steps[0]
        first_path = playback.path / "checkpoints" / f"{first_step:09d}.json"
        first_state = _read_json(first_path)
        if _sha256(first_path) != manifest["checkpoint_sha256"].get(str(first_step)):
            raise ValueError("First checkpoint integrity check failed")
        sim = restore_supported_simulation(manifest, first_state, profile=self, gui=True)
        try:
            if camera_config is not None:
                sim.setup_camera(camera_config=camera_config)
            for frame in playback.frames[first_step - 1 :]:
                for name, entry in frame["agents"].items():
                    agent = sim.get_unique_agent_by_name(name)
                    state = entry["state"]
                    agent.set_pose(_pose_from_record(state["pose"]))
                    if agent.joint_info:
                        agent.restore_kinematic_joint_execution(state["joints"])
                live_objects = {obj.name: obj for obj in sim.sim_objects if obj not in sim.agents}
                recorded_objects = frame["objects"]
                for name in live_objects.keys() - recorded_objects.keys():
                    _remove_playback_primitive(sim, live_objects[name])
                    del live_objects[name]
                for name, entry in recorded_objects.items():
                    state = entry["state"]
                    obj = live_objects.get(name)
                    if obj is not None and capture_primitive_state(obj)["construction"] != state["construction"]:
                        _remove_playback_primitive(sim, obj)
                        obj = None
                    if obj is None:
                        obj = restore_primitive(sim, name, state)
                        live_objects[name] = obj
                    current = obj.get_kinematic_attachment_state()
                    desired = state["attachment"]
                    if current != desired:
                        if current is not None:
                            parent = sim.get_unique_agent_by_name(current["parent_name"])
                            if not parent.detach_object(obj):
                                raise ValueError("Playback attachment detachment failed")
                        if desired is not None:
                            restore_attachment(sim, obj, state)
                    obj.set_pose(_pose_from_record(state["pose"]))
                time.sleep(manifest["construction"]["timestep"] / rtf)
            if hold:
                while p.isConnected(sim.client):
                    time.sleep(0.1)
        finally:
            if p.isConnected(sim.client):
                p.disconnect(sim.client)


register_recording_profile(PROFILE_ID, PROFILE_VERSION, KinematicProfile.from_manifest)
