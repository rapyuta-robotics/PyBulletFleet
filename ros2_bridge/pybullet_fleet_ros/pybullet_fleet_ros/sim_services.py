"""simulation_interfaces service and action server implementations.

Implements the simulation_interfaces standard for PyBulletFleet.
See: https://github.com/ros-simulation/simulation_interfaces
"""

import logging
import os
from typing import TYPE_CHECKING

import pybullet as p

from pybullet_fleet import AgentSpawnParams
from pybullet_fleet.robot_models import resolve_model

from .constants import DEFAULT_SPAWN_URDF, FALLBACK_SPAWN_URDF, SUPPORTED_FEATURES, SUPPORTED_SPAWN_FORMATS
from .conversions import pbf_pose_to_ros, ros_pose_to_pbf
from .uri_utils import resolve_uri

if TYPE_CHECKING:
    from .command_queue import StepCommandQueue
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore
    from rclpy.node import Node

    from .bridge_node import BridgeNode

logger = logging.getLogger(__name__)

from geometry_msgs.msg import Vector3
from rclpy.action import ActionServer
from simulation_interfaces.action import SimulateSteps
from simulation_interfaces.msg import EntityState, Result, SimulationState, Spawnable
from simulation_interfaces.srv import (
    DeleteEntity,
    GetEntities,
    GetEntitiesStates,
    GetEntityBounds,
    GetEntityInfo,
    GetEntityState,
    GetSimulationState,
    GetSimulatorFeatures,
    GetSpawnables,
    ResetSimulation,
    SetEntityState,
    SetSimulationState,
    SpawnEntity,
    StepSimulation,
)

# Result code constants for readability
_OK = Result.RESULT_OK
_NOT_FOUND = Result.RESULT_NOT_FOUND
_FAILED = Result.RESULT_OPERATION_FAILED


class SimServices:
    """Implements simulation_interfaces services and actions.

    Services (14)::

        /sim/get_simulator_features
        /sim/spawn_entity
        /sim/delete_entity
        /sim/get_entity_state
        /sim/set_entity_state
        /sim/get_entities
        /sim/get_entities_states
        /sim/get_entity_info
        /sim/get_entity_bounds
        /sim/step_simulation
        /sim/get_simulation_state
        /sim/set_simulation_state
        /sim/reset_simulation
        /sim/get_spawnables

    Actions (1)::

        /sim/simulate_steps
    """

    def __init__(
        self,
        node: "Node",
        sim: "MultiRobotSimulationCore",
        bridge: "BridgeNode",
        command_queue: "StepCommandQueue | None" = None,
    ) -> None:
        self._node = node
        self._sim = sim
        self._bridge = bridge
        # Pose writes go to the simulation thread at the step boundary when a
        # loop is running; without a queue they apply inline, as before.
        self._command_queue = command_queue

        # Create services
        node.create_service(GetSimulatorFeatures, "/sim/get_simulator_features", self._get_features)
        node.create_service(SpawnEntity, "/sim/spawn_entity", self._spawn_entity)
        node.create_service(DeleteEntity, "/sim/delete_entity", self._delete_entity)
        node.create_service(GetEntityState, "/sim/get_entity_state", self._get_entity_state)
        node.create_service(SetEntityState, "/sim/set_entity_state", self._set_entity_state)
        node.create_service(GetEntities, "/sim/get_entities", self._get_entities)
        node.create_service(GetEntitiesStates, "/sim/get_entities_states", self._get_entities_states)
        node.create_service(GetEntityInfo, "/sim/get_entity_info", self._get_entity_info)
        node.create_service(GetEntityBounds, "/sim/get_entity_bounds", self._get_entity_bounds)
        node.create_service(StepSimulation, "/sim/step_simulation", self._step_sim)
        node.create_service(GetSimulationState, "/sim/get_simulation_state", self._get_sim_state)
        node.create_service(SetSimulationState, "/sim/set_simulation_state", self._set_sim_state)
        node.create_service(ResetSimulation, "/sim/reset_simulation", self._reset_sim)
        node.create_service(GetSpawnables, "/sim/get_spawnables", self._get_spawnables)

        # Action server
        self._simulate_steps_server = ActionServer(
            node,
            SimulateSteps,
            "/sim/simulate_steps",
            self._simulate_steps_cb,
        )

        logger.info("SimServices: all simulation_interfaces registered")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_by_name(self, name: str):
        """Find SimObject or Agent by name."""
        for obj in self._sim.sim_objects:
            if obj.name == name:
                return obj
        return None

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def _get_features(self, request, response):
        """Report supported simulation features."""
        response.features.features = list(SUPPORTED_FEATURES)
        response.features.spawn_formats = list(SUPPORTED_SPAWN_FORMATS)
        return response

    def _spawn_entity(self, request, response):
        """Spawn a new entity.

        Supports ``package://`` URIs (resolved via ament_index).
        If the requested URDF does not exist on disk, falls back to
        ``simple_cube.urdf`` (a minimal mesh-less body) so the entity
        still appears in the simulation.

        Duplicate name handling:
        - ``allow_renaming == False`` (default): reject if name already exists.
        - ``allow_renaming == True``: auto-suffix ``_1``, ``_2``, ... until unique.
        """
        try:
            name = request.name
            # simulation_interfaces 2.1.0 (Jazzy): flat `uri` field (older code
            # targeted the unreleased main API's nested `entity_resource.uri`).
            requested = request.uri or DEFAULT_SPAWN_URDF

            # Resolve to a filesystem path: package:// URIs first, then known model
            # names (Tier 1/2/3, e.g. "ur5e" -> ur_description) via resolve_model.
            # Without the model step a bare name like "ur5e" is not a file and would
            # silently fall back to the placeholder URDF.
            urdf_path = resolve_uri(requested)
            if not os.path.isfile(urdf_path):
                try:
                    urdf_path = resolve_model(requested)
                except FileNotFoundError:
                    pass

            if not os.path.isfile(urdf_path):
                logger.warning(
                    "SpawnEntity: could not resolve '%s', falling back to '%s'",
                    requested,
                    FALLBACK_SPAWN_URDF,
                )
                urdf_path = FALLBACK_SPAWN_URDF

            # Duplicate name check
            if name and self._find_by_name(name) is not None:
                if not request.allow_renaming:
                    response.result.result = _FAILED
                    response.result.error_message = f"Entity '{name}' already exists. Set allow_renaming=true to auto-rename."
                    return response
                # Auto-rename: append _1, _2, ... until unique
                base_name = name
                suffix = 1
                while self._find_by_name(name) is not None:
                    name = f"{base_name}_{suffix}"
                    suffix += 1
                logger.info("SpawnEntity: renamed '%s' → '%s' (allow_renaming)", base_name, name)

            initial_pose = ros_pose_to_pbf(request.initial_pose.pose)

            params = AgentSpawnParams(
                urdf_path=urdf_path,
                initial_pose=initial_pose,
                controller={"type": "omni"},
                name=name or None,
            )
            agent = self._bridge.spawn_robot(params)
            response.result.result = _OK
            response.result.error_message = f"Spawned '{agent.name}' (id={agent.object_id})"
            response.entity_name = agent.name
        except Exception as e:
            response.result.result = _FAILED
            response.result.error_message = str(e)
            logger.error("SpawnEntity failed: %s", e)
        return response

    def _delete_entity(self, request, response):
        """Delete an entity by name."""
        obj = self._find_by_name(request.entity)
        if obj is None:
            response.result.result = _NOT_FOUND
            response.result.error_message = f"Entity '{request.entity}' not found"
            return response
        self._bridge.remove_robot(obj.object_id)
        response.result.result = _OK
        return response

    def _get_entity_state(self, request, response):
        """Get pose and velocity of an entity."""
        obj = self._find_by_name(request.entity)
        if obj is None:
            response.result.result = _NOT_FOUND
            response.result.error_message = f"Entity '{request.entity}' not found"
            return response
        pose = obj.get_pose()
        response.state.pose = pbf_pose_to_ros(pose)
        response.result.result = _OK
        return response

    def _set_entity_state(self, request, response):
        """Set pose of an entity."""
        obj = self._find_by_name(request.entity)
        if obj is None:
            response.result.result = _NOT_FOUND
            response.result.error_message = f"Entity '{request.entity}' not found"
            return response
        if getattr(request, "set_pose", True):
            new_pose = ros_pose_to_pbf(request.state.pose)
            if self._command_queue is None:
                obj.set_pose(new_pose)
            elif self._command_queue.submit(lambda: obj.set_pose(new_pose) or True, wait=True) is None:
                response.result.result = _FAILED
                response.result.error_message = f"Pose of '{request.entity}' queued but not applied within the step timeout"
                return response
        response.result.result = _OK
        return response

    def _get_entities(self, request, response):
        """Return list of all entities."""
        for obj in self._sim.sim_objects:
            name = obj.name or f"object_{obj.object_id}"
            response.entities.append(name)
        response.result.result = _OK
        return response

    def _get_entities_states(self, request, response):
        """Return states for all entities."""
        for obj in self._sim.sim_objects:
            state = EntityState()
            pose = obj.get_pose()
            state.pose = pbf_pose_to_ros(pose)
            response.states.append(state)
            response.entities.append(obj.name or f"object_{obj.object_id}")
        response.result.result = _OK
        return response

    def _get_entity_info(self, request, response):
        """Return metadata about an entity."""
        obj = self._find_by_name(request.entity)
        if obj is None:
            response.result.result = _NOT_FOUND
            response.result.error_message = f"Entity '{request.entity}' not found"
            return response
        response.result.result = _OK
        return response

    def _get_entity_bounds(self, request, response):
        """Get AABB bounds of an entity."""
        obj = self._find_by_name(request.entity)
        if obj is None:
            response.result.result = _NOT_FOUND
            response.result.error_message = f"Entity '{request.entity}' not found"
            return response
        try:
            aabb_min, aabb_max = p.getAABB(obj.body_id, physicsClientId=self._sim.client)
            response.bounds.type = 1  # Bounds.TYPE_BOX
            response.bounds.points = [
                Vector3(x=float(aabb_min[0]), y=float(aabb_min[1]), z=float(aabb_min[2])),
                Vector3(x=float(aabb_max[0]), y=float(aabb_max[1]), z=float(aabb_max[2])),
            ]
            response.result.result = _OK
        except Exception as e:
            response.result.result = _FAILED
            response.result.error_message = str(e)
        return response

    def _step_sim(self, request, response):
        """Execute N simulation steps.

        If the simulation is paused, steps are still executed (Gazebo-compatible
        behaviour) and the simulation returns to paused state afterwards.
        """
        num_steps = request.steps
        was_paused = self._sim.is_paused
        if was_paused:
            self._sim.resume()
        for _ in range(num_steps):
            self._sim.step_once()
        if was_paused:
            self._sim.pause()
        response.result.result = _OK
        return response

    def _get_sim_state(self, request, response):
        """Return current simulation state (paused/playing)."""
        if self._sim.is_paused:
            response.state.state = SimulationState.STATE_PAUSED
        else:
            response.state.state = SimulationState.STATE_PLAYING
        response.result.result = _OK
        return response

    def _set_sim_state(self, request, response):
        """Set simulation state (pause/resume)."""
        target_state = request.state.state
        if target_state == SimulationState.STATE_PAUSED:
            self._sim.pause()
        elif target_state == SimulationState.STATE_PLAYING:
            self._sim.resume()
        response.result.result = _OK
        return response

    def _reset_sim(self, request, response):
        """Reset simulation to initial state."""
        self._bridge.reset()
        response.result.result = _OK
        response.result.error_message = "Simulation reset. Re-spawn entities as needed."
        return response

    def _get_spawnables(self, request, response):
        """List available robot URDFs from all configured model paths."""
        # Collect search directories: model_paths from SimulationParams + default "robots/"
        search_dirs = list(self._sim.params.model_paths)
        default_dir = "robots"
        if default_dir not in search_dirs:
            search_dirs.append(default_dir)

        # 1. Collect unique URIs across all directories
        seen_uris: set = set()
        for directory in search_dirs:
            if not os.path.isdir(directory):
                continue
            for f in sorted(os.listdir(directory)):
                if f.endswith(".urdf"):
                    seen_uris.add(os.path.join(directory, f))

        # 2. Build spawnable list from deduplicated URIs
        for uri in sorted(seen_uris):
            spawnable = Spawnable()
            spawnable.uri = uri  # 2.1.0 flat field (not the main API's entity_resource)
            spawnable.description = os.path.basename(uri).replace(".urdf", "")
            response.spawnables.append(spawnable)
        response.result.result = _OK
        return response

    # ------------------------------------------------------------------
    # Action server
    # ------------------------------------------------------------------

    def _simulate_steps_cb(self, goal_handle):
        """Execute N simulation steps with feedback.

        If the simulation is paused, steps are still executed (Gazebo-compatible
        behaviour) and the simulation returns to paused state afterwards.
        """
        num_steps = goal_handle.request.steps
        feedback = SimulateSteps.Feedback()

        was_paused = self._sim.is_paused
        if was_paused:
            self._sim.resume()

        for i in range(num_steps):
            if goal_handle.is_cancel_requested:
                if was_paused:
                    self._sim.pause()
                goal_handle.canceled()
                result = SimulateSteps.Result()
                result.result.result = _FAILED
                result.result.error_message = "Cancelled"
                return result

            self._sim.step_once()
            feedback.completed_steps = i + 1
            feedback.remaining_steps = num_steps - (i + 1)
            goal_handle.publish_feedback(feedback)

        if was_paused:
            self._sim.pause()

        goal_handle.succeed()
        result = SimulateSteps.Result()
        result.result.result = _OK
        return result
