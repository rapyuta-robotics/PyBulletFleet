"""Owned fresh-instance execution for the v1 navigation profile."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
import pybullet as p

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.agent_manager import AgentManager
from pybullet_fleet.commands import CommandAck, RobotGoalCommand2D
from pybullet_fleet.core_simulation import MultiRobotSimulationCore
from pybullet_fleet.fleet_api import FleetCommandDispatcher
from pybullet_fleet.geometry import Pose, quat_angle_between
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams
from pybullet_fleet.types import CollisionMode

from .artifact import PACKAGE, ArtifactWriter, assets, environment
from .schema import PROFILE, SCHEMA_VERSION, ReplayError, ReplayInput, clone, normalize_initial, positive_int


class ReplaySession:
    """Coordinate a supported initial state, ordered effective inputs and observations.

    Use :meth:`create` and the context manager. The owned core is deliberately
    private: calls outside the session input boundary are not captured or
    guaranteed to replay.
    """

    @classmethod
    def create(
        cls,
        initial_definition: dict,
        *,
        output: str | Path | None = None,
        observation_interval: int = 1,
        provenance: dict | None = None,
    ) -> ReplaySession:
        initial = normalize_initial(initial_definition)
        interval = positive_int(observation_interval, "observation_interval")
        provenance = clone(provenance or {})
        if output is not None and Path(output).exists():
            raise ReplayError("invalid", "output directory already exists")
        return cls(initial, output, interval, provenance)

    def __init__(self, initial: dict, output, interval: int, provenance: dict):
        self._initial = initial
        self.run_id = str(uuid4())
        self._writer: ArtifactWriter | None = None
        self._closed = False
        self._failed = False
        self._stepping = False
        self._acks: list[CommandAck] = []
        self._interval = interval
        self._last_observed = -1
        self._active: dict[str, dict] = {}
        self._pairs: set[tuple[str, ...]] = set()
        self._objects: dict[str, SimObject] = {}
        self._agents: dict[str, Agent] = {}
        self._id_by_name: dict[str, str] = {}
        self._id_by_object: dict[int, str] = {}
        self._expected_step = 0
        config = initial["pbf"]
        self._sim = MultiRobotSimulationCore.from_dict(
            {
                "simulation": {
                    "gui": False,
                    "monitor": False,
                    "enable_monitor_gui": False,
                    "physics": False,
                    "enable_floor": False,
                    "target_rtf": 0,
                    "timestep": config["timestep"],
                    "collision_check_frequency": config["collision_frequency"],
                    "collision_margin": config["collision_margin"],
                    "collision_detection_method": "closest_points",
                    "ignore_static_collision": False,
                    "spatial_hash_cell_size_mode": "auto_initial",
                    "log_level": "warning",
                }
            }
        )
        try:
            manager = AgentManager(
                self._sim,
                name="replay_fleet",
                fleet_controller={"type": "batch_omni"} if config["controller"] == "batch_omni" else None,
            )
            with self._sim.batch_spawn():
                for entity in initial["world"]["entities"]:
                    pose = Pose.from_yaw(entity["position"][0], entity["position"][1], entity["position"][2], entity["yaw"])
                    if entity["kind"] == "robot":
                        obj = Agent.from_params(
                            AgentSpawnParams(
                                name=entity["name"],
                                urdf_path=str(PACKAGE / "robots/simple_cube.urdf"),
                                initial_pose=pose,
                                mass=0.0,
                                pickable=False,
                                collision_mode=CollisionMode.NORMAL_2D,
                                controller={
                                    "type": "omni",
                                    **config["limits"],
                                    "navigation_2d": True,
                                    "cmd_vel_timeout": 0.0,
                                    "default_direction": "forward",
                                },
                            ),
                            self._sim,
                        )
                        manager.add_object(obj)
                        self._agents[entity["entity_id"]] = obj
                    else:
                        shape = ShapeParams(shape_type="box", half_extents=entity["half_extents"])
                        obj = SimObject.from_params(
                            SimObjectSpawnParams(
                                name=entity["name"],
                                initial_pose=pose,
                                mass=0.0,
                                pickable=False,
                                visual_shape=shape,
                                collision_shape=shape,
                                collision_mode=CollisionMode.STATIC,
                            ),
                            self._sim,
                        )
                    self._objects[entity["entity_id"]] = obj
                    self._id_by_name[entity["name"]] = entity["entity_id"]
                    self._id_by_object[obj.object_id] = entity["entity_id"]
            self._sim.initialize_simulation()
            self._dispatcher = FleetCommandDispatcher(self._sim, retain_command_events=False)
            self._params_signature = asdict(self._sim.params)
            self._collision_frequency = self._sim._collision_check_frequency
            self._entity_signature = self._signature()
            self._manager = manager
            self._manager_controller = manager.batch_controller
            if output is not None:
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "profile": dict(PROFILE),
                    "run_id": self.run_id,
                    "environment": environment(),
                    "assets": assets(),
                    "provenance": provenance,
                    "observation_interval": interval,
                    "velocity_semantics": (
                        "linear: world-frame m/s; angular: controller-reported "
                        "yaw rate/magnitude in rad/s (not solver velocity)"
                    ),
                    "outcome_semantics": "latest accepted request: stopped, or arrived within position/angle tolerance",
                }
                self._writer = ArtifactWriter(Path(output), manifest, initial)
                self._observe()
        except Exception as exc:
            if self._writer:
                self._writer.abort()
            p.disconnect(self._sim.client)
            if isinstance(exc, OSError):
                raise ReplayError("incomplete", f"recording initialization failed: {exc}") from exc
            raise

    @property
    def initial(self) -> dict:
        """Return a detached copy of the normalized initial definition."""
        return clone(self._initial)

    @property
    def step_count(self) -> int:
        return self._expected_step

    def _signature(self) -> tuple:
        return tuple(
            (
                id(obj),
                obj.name,
                obj.collision_mode,
                tuple(obj.callbacks),
                tuple(id(controller) for controller in obj._controllers) if isinstance(obj, Agent) else None,
                # V1 admits scalar limits only. Snapshot those immutable values
                # without a recursive dataclass deepcopy on every agent/step.
                tuple(vars(obj.controller_params).items()) if isinstance(obj, Agent) else None,
            )
            for obj in self._sim.sim_objects
        )

    def _check_runtime(self) -> None:
        if (
            self._sim.step_count != self._expected_step
            or asdict(self._sim.params) != self._params_signature
            or self._sim._collision_check_frequency != self._collision_frequency
            or self._signature() != self._entity_signature
            or self._sim._callbacks
            or self._sim.plugins
            or self._sim.behavior_trees
            or self._sim.events._handlers
            or self._sim.is_paused
            or self._manager.batch_controller is not self._manager_controller
        ):
            self._reject("runtime configuration/entity/callback mutation")
        for agent in self._agents.values():
            if not agent.is_action_queue_empty() or agent.plugins or (agent._events and agent._events._handlers):
                self._reject("action/plugin/event callback mutation")

    def _reject(self, message: str) -> None:
        self._failed = True
        raise ReplayError("unsupported", message)

    def _apply_inputs(self, step_inputs: list[dict]) -> None:
        # Between steps the clock is the completed step's time. Replay snaps it
        # to the canonical step * timestep so accumulated rounding cannot drift
        # from the recording; ordinary simulations keep their accumulated clock.
        self._sim._elapsed_sim_time = self._expected_step * self._initial["pbf"]["timestep"]
        for order, command in enumerate(step_inputs):
            self._journal("command_intent", order=order, input=command)
            self._dispatcher.allowed_names = None if command["allowed_names"] is None else frozenset(command["allowed_names"])
            kwargs = {"source": command["source"], "command_id": command["command_id"]}
            if command["command_type"] == "navigate":
                ack = self._dispatcher.navigate([RobotGoalCommand2D(**goal) for goal in command["payload"]["goals"]], **kwargs)
            else:
                ack = self._dispatcher.stop(command["payload"]["names"], **kwargs)
            self._acks.append(ack)
            self._journal(
                "command_result",
                order=order,
                ack={
                    "command_id": ack.command_id,
                    "source": ack.source,
                    "sim_time": ack.sim_time,
                    "accepted_names": list(ack.accepted_names),
                    "rejected": dict(ack.rejected),
                },
            )
            goals = {g["name"]: g for g in command["payload"].get("goals", [])}
            for target in ack.accepted_names:
                self._active[self._id_by_name[target]] = {
                    "step": self._expected_step,
                    "order": order,
                    "command_id": ack.command_id,
                    "type": command["command_type"],
                    "goal": goals.get(target),
                }

    def _journal(self, kind: str, **payload) -> None:
        if self._writer:
            self._writer.append(
                "journal",
                {
                    "record_type": kind,
                    "run_id": self.run_id,
                    "step": self._expected_step,
                    "sim_time": self._expected_step * self._initial["pbf"]["timestep"],
                    "phase": "input" if kind.startswith("command_") else "after_update",
                    **payload,
                },
            )

    def _after_step(self) -> None:
        pairs = {
            tuple(sorted((self._id_by_object[a], self._id_by_object[b]))) for a, b in self._sim.get_active_collision_pairs()
        }
        for kind, changed in (("collision_started", pairs - self._pairs), ("collision_ended", self._pairs - pairs)):
            for pair in sorted(changed):
                self._journal("event", event={"type": kind, "entities": list(pair)})
        self._pairs = pairs
        config = self._initial["pbf"]
        for entity_id, request in list(self._active.items()):
            agent = self._agents[entity_id]
            if agent.is_moving:
                continue
            outcome = "stopped"
            if request["type"] == "navigate":
                pose, goal = agent.get_pose(), request["goal"]
                target = Pose.from_yaw(goal["position"][0], goal["position"][1], pose.z, goal["yaw"])
                if (
                    np.linalg.norm(np.asarray(pose.position[:2]) - goal["position"]) > config["position_tolerance"]
                    or quat_angle_between(pose.orientation, target.orientation) > config["angle_tolerance"]
                ):
                    continue
                outcome = "arrived"
            self._journal(
                "event",
                event={
                    "type": "outcome",
                    "entity_id": entity_id,
                    "outcome": outcome,
                    "command_id": request["command_id"],
                    "input_step": request["step"],
                    "input_order": request["order"],
                },
            )
            del self._active[entity_id]
        self._expected_step += 1
        if self._writer and self._expected_step % self._interval == 0:
            self._observe()

    def _observe(self) -> None:
        if self._writer is None or self._last_observed == self._expected_step:
            return
        entities = {}
        for entity_id, obj in self._objects.items():
            pose = obj.get_pose()
            agent = self._agents.get(entity_id)
            entities[entity_id] = {
                "position": list(pose.position),
                "orientation": list(pose.orientation),
                "linear_velocity": agent.velocity.tolist() if agent else [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, float(agent.angular_velocity)] if agent else [0.0, 0.0, 0.0],
                "is_moving": bool(agent.is_moving) if agent else False,
            }
        self._writer.append(
            "observations",
            {
                "run_id": self.run_id,
                "state_step": self._expected_step,
                "sim_time": self._expected_step * self._initial["pbf"]["timestep"],
                "entities": entities,
            },
        )
        self._last_observed = self._expected_step

    def step(self, inputs: Iterable[ReplayInput] = ()) -> tuple[CommandAck, ...]:
        if self._closed or self._failed or self._stepping:
            raise ReplayError("incomplete", "session is closed, failed or already stepping")
        try:
            step_inputs = [item.to_record() for item in inputs]
            self._acks = []
            self._stepping = True
            before = self._expected_step
            self._check_runtime()
            self._apply_inputs(step_inputs)
            self._sim.step_once()
            if self._sim.step_count != before + 1:
                raise ReplayError("incomplete", "simulation did not complete a step")
            self._after_step()
            return tuple(self._acks)
        except BaseException as exc:
            self._failed = True
            if isinstance(exc, OSError):
                raise ReplayError("incomplete", f"recording I/O failed: {exc}") from exc
            raise
        finally:
            self._stepping = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if not self._failed:
                self._check_runtime()
                if self._writer:
                    self._observe()
                    self._writer.finish(self._expected_step)
        except OSError as exc:
            self._failed = True
            raise ReplayError("incomplete", f"finalization failed: {exc}") from exc
        finally:
            self._closed = True
            if self._writer:
                self._writer.abort()
            if p.isConnected(self._sim.client):
                p.disconnect(self._sim.client)

    def __enter__(self) -> ReplaySession:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self._failed = True
        self.close()
