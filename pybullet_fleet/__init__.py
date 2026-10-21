"""
pybullet_fleet package
General-purpose PyBullet simulation library for multi-robot fleets
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("pybullet-fleet")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

# Type definitions
from pybullet_fleet.types import (  # noqa: F401
    ActionStatus,
    CallbackPhase,
    ControllerMode,
    DoorState,
    GoalReplacement,
    MotionMode,
    MovementDirection,
    PosePhase,
    SpatialHashCellSizeMode,
)

# Geometry primitives
from pybullet_fleet.geometry import Path, Pose

# Fleet command types
from pybullet_fleet.commands import (
    DEFAULT_ATTACH_SEARCH_RADIUS,
    FLEET_COMMAND_EVENT,
    CommandAck,
    CommandEvent,
    RobotActionCommand,
    RobotAttachCommand,
    RobotGoalCommand2D,
    RobotGoalCommand3D,
    RobotJointPositionsCommand,
    RobotNamedJointPositionsCommand,
)

# Fleet API
from pybullet_fleet.fleet_api import (
    FleetCommandDispatcher,
    FleetStateProvider,
)
from pybullet_fleet.states import RobotState2D, RobotState3D

# Toolkit-neutral monitor UI boundary
from pybullet_fleet.gui_commands import (
    GuiCommand,
    GuiCommandType,
    MonitorEntity,
    MonitorFrame,
    SelectedEntityMonitorState,
)

# Robot/Agent management
from pybullet_fleet.agent import Agent, AgentSpawnParams, IKParams
from pybullet_fleet.plugins.battery_plugin import BatteryPlugin
from pybullet_fleet.agent_manager import AgentManager, GridSpawnParams

# Core simulation classes
from pybullet_fleet.core_simulation import (
    MultiRobotSimulationCore,
    SimulationParams,
)

# EventBus
from pybullet_fleet.events import EventBus, SimEvents  # noqa: F401

# World / SDF loading  (load_rmf_world kept as backward-compatible alias)
from pybullet_fleet.sdf_loader import load_mesh_directory, load_rmf_world  # noqa: F401
from pybullet_fleet.sdf_loader import load_sdf_world, load_sdf_world_file  # noqa: F401
from pybullet_fleet.sdf_loader import resolve_sdf_to_urdf  # noqa: F401

# Optional OpenUSD static-world import. The module lazy-imports ``pxr`` so this
# public import remains safe without installing the ``usd`` extra.
from pybullet_fleet.usd_loader import UsdImportDiagnostic, UsdImportOptions, UsdImportReport, load_usd_world  # noqa: F401

from pybullet_fleet.behavior_tree import (  # noqa: F401
    AgentActionId,
    AgentBehaviorTree,
    BehaviorTree,
    BehaviorTreeNodeType,
    NavigationAdapter,
    NavigationStatus,
    PyBulletFleetNavigationAdapter,
    TickResult,
    WorkerBehaviorTree,
    WorkerActionId,
)

# Robot model resolution  (resolve_urdf kept as backward-compatible alias)
from pybullet_fleet.robot_models import resolve_model  # noqa: F401

# Recorder
from pybullet_fleet.recorder import SimulationRecorder

# Base classes
from pybullet_fleet.sim_object import SimObject

# Controller system
from pybullet_fleet.controller import (
    Controller,
    KinematicController,
    DifferentialController,
    OmniController,
    create_controller,
    create_controller_from_entry,
    register_controller,
)

# Geometry (quaternion utilities)
from pybullet_fleet.geometry import (  # noqa: F401,F811
    quat_angle_between,
    quat_from_rotvec,
    quat_multiply,
    quat_slerp_batch,
    quat_to_rot_matrix,
    rotate_vector,
)

# Utilities
from pybullet_fleet.tools import (
    body_to_world_velocity_2d,
    body_to_world_velocity_3d,
    grid_to_world,
    world_to_grid,
    normalize_vector_param,
)

# Config utilities
from pybullet_fleet.config_utils import load_yaml_config

# Entity registry
from pybullet_fleet.entity_registry import register_entity_class

# Plugin system
from pybullet_fleet.sim_plugin import (
    SimPlugin,
    create_sim_plugin,
    create_sim_plugin_from_entry,
    list_sim_plugins,
    register_sim_plugin,
)
from pybullet_fleet.agent_plugin import (
    AgentPlugin,
    create_agent_plugin,
    list_agent_plugins,
    register_agent_plugin,
)
from pybullet_fleet.plugin_utils import PluginRegistry

# Device classes
from pybullet_fleet.devices.door import Door, DoorParams
from pybullet_fleet.devices.elevator import Elevator, ElevatorParams

# High-level controllers
from pybullet_fleet.controllers import PatrolController, RandomWalkController

# Robot model resolution
from pybullet_fleet.robot_models import (
    ModelEntry,
    RobotProfile,
    add_search_path,
    auto_detect_profile,
    detect_robot_type,
    discover_models,
    get_search_paths,
    list_all_models,
    register_model,
    remove_search_path,
    resolve_urdf,
    unregister_model,
)

__all__ = [
    "__version__",
    # Type definitions
    "ActionStatus",
    "CallbackPhase",
    "GoalReplacement",
    "MotionMode",
    "MovementDirection",
    "PosePhase",
    # Geometry
    "Pose",
    "Path",
    # Fleet API
    "FLEET_COMMAND_EVENT",
    "DEFAULT_ATTACH_SEARCH_RADIUS",
    "RobotState2D",
    "RobotState3D",
    "RobotGoalCommand2D",
    "RobotGoalCommand3D",
    "RobotAttachCommand",
    "RobotActionCommand",
    "RobotJointPositionsCommand",
    "RobotNamedJointPositionsCommand",
    "CommandAck",
    "CommandEvent",
    "FleetStateProvider",
    "FleetCommandDispatcher",
    # Core simulation
    "MultiRobotSimulationCore",
    "SimulationParams",
    "SpatialHashCellSizeMode",
    "SimulationRecorder",
    "SimObject",
    # OpenUSD static-world import
    "UsdImportDiagnostic",
    "UsdImportOptions",
    "UsdImportReport",
    "load_usd_world",
    "WorkerBehaviorTree",
    "WorkerActionId",
    "AgentBehaviorTree",
    "AgentActionId",
    "BehaviorTree",
    "BehaviorTreeNodeType",
    "NavigationAdapter",
    "NavigationStatus",
    "PyBulletFleetNavigationAdapter",
    "TickResult",
    # Events
    "EventBus",
    "SimEvents",
    # Agent management
    "Agent",
    "AgentSpawnParams",
    "BatteryPlugin",
    "IKParams",
    "AgentManager",
    "GridSpawnParams",
    # Controller system
    "Controller",
    "ControllerMode",
    "KinematicController",
    "OmniController",
    "DifferentialController",
    "create_controller",
    "register_controller",
    # Utilities
    "body_to_world_velocity_2d",
    "body_to_world_velocity_3d",
    "grid_to_world",
    "world_to_grid",
    "normalize_vector_param",
    "quat_to_rot_matrix",
    "quat_angle_between",
    "quat_from_rotvec",
    "quat_multiply",
    "quat_slerp_batch",
    "rotate_vector",
    # Config utilities
    "load_yaml_config",
    # Entity registry
    "register_entity_class",
    # Plugin system
    "SimPlugin",
    "AgentPlugin",
    "BatteryPlugin",
    "create_agent_plugin",
    "list_agent_plugins",
    "register_agent_plugin",
    # Device classes
    "Door",
    "DoorParams",
    "Elevator",
    "ElevatorParams",
    # High-level controllers
    "PatrolController",
    "RandomWalkController",
    # Robot models
    "resolve_model",
    "resolve_urdf",
    "list_all_models",
    "ModelEntry",
    "RobotProfile",
    "auto_detect_profile",
    "detect_robot_type",
    "register_model",
    "unregister_model",
    "discover_models",
    "add_search_path",
    "remove_search_path",
    "get_search_paths",
]
