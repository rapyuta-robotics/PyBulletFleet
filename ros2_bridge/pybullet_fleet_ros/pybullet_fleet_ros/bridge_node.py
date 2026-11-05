"""Main bridge node — wraps MultiRobotSimulationCore as a ROS 2 node.

The simulation loop (``run_simulation()``) runs on a background thread.
Bridge publishes state via EventBus callbacks (``POST_STEP``).

Usage::

    ros2 run pybullet_fleet_ros bridge_node \
        --ros-args -p config_yaml:=/path/to/config.yaml
"""

import inspect
import logging
import threading
from typing import Any, Dict, Iterable, List, Type

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster

from pybullet_fleet import (
    Agent,
    AgentSpawnParams,
    MultiRobotSimulationCore,
    SimEvents,
)
from pybullet_fleet.config_utils import load_yaml_config
from pybullet_fleet.controller import OmniController
from pybullet_fleet.types import MotionMode

from .bridge_plugin import BridgePlugin
from .command_queue import StepCommandQueue
from .conversions import sim_time_to_ros_time
from .fleet_ros_interface import FleetRosInterface
from .fleet_endpoints import manager_fleet_namespace
from .handler_registry import HandlerMap, load_handler_map_from_config, resolve_handler_classes
from .interface_config import BridgeApiConfig, StateScope, resolve_bridge_api_config
from .param_utils import get_bool_param, get_float_param
from .robot_handler import RobotHandler
from .robot_handler_base import RobotHandlerBase
from .uri_utils import resolve_package_uris

logger = logging.getLogger(__name__)


def _handler_accepts_interface_config(handler_cls: Type[RobotHandlerBase]) -> bool:
    """Return whether *handler_cls* can accept ``interface_config``."""
    try:
        params = inspect.signature(handler_cls).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(param.name == "interface_config" or param.kind is inspect.Parameter.VAR_KEYWORD for param in params)


def _is_robot_handler_class(value: object) -> bool:
    """Return whether *value* is a RobotHandler class or subclass."""
    return isinstance(value, type) and issubclass(value, RobotHandler)


def _handler_display_name(value: object) -> str:
    """Return a readable name for handler class or factory logging."""
    return getattr(value, "__name__", value.__class__.__name__)


def _handler_needs_pre_step(handler: RobotHandlerBase) -> bool:
    # RobotHandlerBase defines this property; getattr keeps older test doubles
    # and non-standard handlers compatible.
    return bool(getattr(handler, "needs_pre_step", True))


def _handler_needs_post_step(handler: RobotHandlerBase) -> bool:
    # Default is every-step post_step for custom handler compatibility.
    return bool(getattr(handler, "needs_post_step", True))


def _handler_throttles_post_step(handler: RobotHandlerBase) -> bool:
    # Only publish-oriented handlers should opt into bridge publish_rate
    # throttling; custom handlers default to every-step post_step.
    return bool(getattr(handler, "throttle_post_step", False))


def _register_step_handler(
    handler: RobotHandlerBase,
    pre_step_handlers: List[RobotHandlerBase],
    post_step_handlers: List[RobotHandlerBase],
    throttled_post_step_handlers: List[RobotHandlerBase],
) -> None:
    """Register a handler in exactly the step dispatch lists it needs."""
    if _handler_needs_pre_step(handler):
        pre_step_handlers.append(handler)
    if _handler_needs_post_step(handler):
        if _handler_throttles_post_step(handler):
            throttled_post_step_handlers.append(handler)
        else:
            post_step_handlers.append(handler)


def _manager_agent_names(sim_core, managers: Iterable[str]) -> dict[str, frozenset[str]]:
    """Resolve named AgentManagers to their current Agent names.

    Manager membership is intentionally resolved after simulation construction:
    it is the v1 source of truth for manager-scoped Fleet endpoints.
    """
    sim_agent_ids = {id(agent) for agent in sim_core.agents}
    resolved = {}
    for name in managers:
        manager = sim_core.get_manager(name)
        if manager is None:
            raise ValueError(f"fleet_api manager {name!r} is unknown")
        names = frozenset(
            agent.name
            for agent in getattr(manager, "objects", ())
            if id(agent) in sim_agent_ids and isinstance(getattr(agent, "name", None), str) and agent.name
        )
        if not names:
            logger.warning(
                "fleet_api manager %r has no Agent members; its manager-scoped endpoint will be empty",
                name,
            )
        resolved[name] = names
    return resolved


class BridgeNode(Node):
    """ROS 2 node that drives PyBulletFleet simulation and exposes per-robot topics.

    Parameters (ROS):
        config_yaml (str): Path to PyBulletFleet YAML config file (required).
        publish_rate (float): State publish rate in Hz (default: 1/timestep, i.e. every step).
        gui (bool): Enable PyBullet GUI window (overrides YAML).
        physics (bool): Enable physics simulation (overrides YAML).
        target_rtf (float): Target real-time factor (overrides YAML).
        enable_sim_services (bool): Create simulation_interfaces services.
    """

    #: Default handler class(es) for all robots.  Override in subclasses
    #: or pass ``handler_map`` to use per-robot handlers.
    handler_class: Type[RobotHandler] = RobotHandler

    def __init__(self, handler_map: HandlerMap | None = None) -> None:
        super().__init__("pybullet_fleet_bridge")
        self._handler_map: HandlerMap = handler_map or {}

        # -- Step 1: Read config_yaml and load YAML config --
        self.declare_parameter("config_yaml", "")
        config_yaml = self.get_parameter("config_yaml").get_parameter_value().string_value

        if not config_yaml:
            self.get_logger().info("config_yaml not set, starting with defaults")
            bridge_config: Dict[str, Any] = {}
        else:
            bridge_config = load_yaml_config(config_yaml)
        bridge_config = resolve_package_uris(bridge_config)

        sim_cfg = bridge_config.get("simulation", {})

        # -- Step 2: Declare params with YAML values as defaults --
        # ROS 2 parameter overrides (from launch file or -p flag)
        # automatically take precedence over these defaults.
        # Priority: code default → YAML config → command-line parameter.
        from rcl_interfaces.msg import ParameterDescriptor

        _dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("publish_rate", float(sim_cfg.get("publish_rate", 0.0)), _dyn)
        self.declare_parameter("gui", sim_cfg.get("gui", False), _dyn)
        self.declare_parameter("physics", sim_cfg.get("physics", False), _dyn)
        self.declare_parameter("target_rtf", float(sim_cfg.get("target_rtf", 1.0)), _dyn)
        self.declare_parameter("enable_sim_services", True, _dyn)
        # "" (empty) is a sentinel meaning "not set" — lets us tell an explicit
        # enable_monitor_gui:=false from the unset case (so we can follow gui by default).
        self.declare_parameter("enable_monitor_gui", "", _dyn)

        # -- Step 3: Read final resolved values --
        publish_rate_param = get_float_param(self, "publish_rate", 0.0)
        enable_sim_services = get_bool_param(self, "enable_sim_services", True)
        effective_gui = get_bool_param(self, "gui", sim_cfg.get("gui", False))
        effective_physics = get_bool_param(self, "physics", sim_cfg.get("physics", False))
        effective_rtf = get_float_param(self, "target_rtf", float(sim_cfg.get("target_rtf", 1.0)))

        # Monitor GUI (tkinter window). Precedence: explicit enable_monitor_gui
        # param → YAML enable_monitor_gui → follow gui (headless gui:=false hides it).
        monitor_gui_raw = self.get_parameter("enable_monitor_gui").value
        if isinstance(monitor_gui_raw, str) and monitor_gui_raw == "":
            effective_monitor_gui = bool(sim_cfg.get("enable_monitor_gui", effective_gui))
        else:
            effective_monitor_gui = get_bool_param(self, "enable_monitor_gui", effective_gui)

        # Apply resolved values into the config dict
        sim_overrides = dict(sim_cfg)
        sim_overrides["gui"] = effective_gui
        sim_overrides["physics"] = effective_physics
        sim_overrides["target_rtf"] = effective_rtf
        sim_overrides["enable_monitor_gui"] = effective_monitor_gui

        # Load handler_map from config (if present) and merge with
        # programmatic handler_map.  Config entries have lower priority
        # (programmatic handler_map wins on key conflict).
        config_handler_map = bridge_config.get("handler_map", {})
        if config_handler_map:
            config_map = load_handler_map_from_config(config_handler_map)
            # Programmatic handler_map takes priority over config
            config_map.update(self._handler_map)
            self._handler_map = config_map

        # Delegate world loading & robot spawning to core
        full_config = dict(bridge_config)
        full_config["simulation"] = sim_overrides
        self.sim = MultiRobotSimulationCore.from_dict(full_config)

        # Offset between the sim (world/local) frame and the RMF (nav-graph/map)
        # frame. Georeferenced maps (e.g. campus) place the nav graph far from
        # the world origin, so robots spawn in local coords but RMF expects map
        # coords. RobotHandler shifts published poses by +offset (sim→RMF) and
        # incoming nav goals by -offset (RMF→sim). Default (0, 0) = same frame.
        off = bridge_config.get("rmf_frame_offset", [0.0, 0.0])
        self.rmf_frame_offset = (float(off[0]), float(off[1]))

        # Resolve publish_rate: 0 (default) = every simulation step
        timestep = self.sim.params.timestep
        publish_rate = publish_rate_param if publish_rate_param > 0 else 1.0 / timestep

        # TF broadcaster (shared by all handlers)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Robot handlers (object_id → list of RobotHandlerBase instances)
        self._handler_lock = threading.RLock()
        self._handler_callback_lock = threading.RLock()
        # Serialize ROS publisher destruction with callbacks from the simulation
        # thread. Without this, Ctrl-C can destroy rclpy handles while POST_STEP
        # is publishing /clock or robot state.
        self._shutdown_lock = threading.RLock()
        self._shutdown_requested = False
        self._handlers: Dict[int, List["RobotHandlerBase"]] = {}
        self._pre_step_handlers: List["RobotHandlerBase"] = []
        self._post_step_handlers: List["RobotHandlerBase"] = []
        self._throttled_post_step_handlers: List["RobotHandlerBase"] = []

        # /clock publisher and throttle state
        self._clock_pub = self.create_publisher(Clock, "/clock", 10)
        self._last_clock_time: float = -1.0
        self._clock_min_interval: float = 1.0 / publish_rate  # throttle /clock
        self._last_publish_time: float = -1.0
        self._publish_min_interval: float = 1.0 / publish_rate

        # Interface selection from explicit fleet_api/per_robot_api sections.
        self._api_config: BridgeApiConfig = resolve_bridge_api_config(bridge_config)
        # ``None`` is the unfiltered/global convention used by
        # FleetStateProvider: it publishes every Agent for ``all_agents``.
        global_state_names = None
        if self._api_config.fleet_api.state_scope is StateScope.MANAGERS:
            included = _manager_agent_names(self.sim, self._api_config.fleet_api.state_include_managers)
            global_state_names = frozenset().union(*included.values())
        agents_by_manager = _manager_agent_names(
            self.sim,
            (entry.manager for entry in self._api_config.manager_interfaces),
        )
        claimed_names = set()
        for manager, names in agents_by_manager.items():
            overlap = claimed_names.intersection(names)
            if overlap:
                formatted = ", ".join(sorted(overlap))
                raise ValueError(f"Agent(s) {formatted} belong to more than one exposed manager interface")
            claimed_names.update(names)
        # Commands from ROS callbacks are applied on the simulation thread at
        # PRE_STEP; the callbacks only validate and answer. See command_queue.py.
        self.command_queue = StepCommandQueue(self.sim)
        self._fleet_ros = FleetRosInterface(
            self,
            self.sim,
            self._api_config.fleet_api,
            # Preserve ``None`` for the all-Agent global compatibility stream.
            state_names=global_state_names,
            command_queue=self.command_queue,
        )
        self._manager_fleet_ros = [
            FleetRosInterface(
                self,
                self.sim,
                entry,
                namespace=manager_fleet_namespace(entry.manager),
                state_names=agents_by_manager[entry.manager],
                command_names=agents_by_manager[entry.manager],
                command_queue=self.command_queue,
            )
            for entry in self._api_config.manager_interfaces
        ]
        if not self._api_config.per_robot_api.enabled:
            self.get_logger().warning("per_robot_api disabled: no per-robot ROS handlers will be created")
        elif not self._api_config.per_robot_api.any_group_enabled:
            self.get_logger().warning(
                "per_robot_api enabled but all per-robot groups are disabled: " "no per-robot ROS handlers will be created"
            )

        # Register ROS handlers for all enabled agents already in the sim.
        for agent in self.sim.agents:
            if self._should_create_per_robot_handler(agent):
                self._register_robot_handler(agent)
        actual_count = len(self.sim.agents)

        # Bridge plugins (singleton handlers loaded from config)
        self._bridge_plugins: List[BridgePlugin] = self._load_bridge_plugins(bridge_config)
        for bp in self._bridge_plugins:
            bp.on_init()

        # Auto-register/unregister handlers when agents spawn/despawn
        self.sim.events.on(SimEvents.AGENT_SPAWNED, self._on_agent_spawned)
        self.sim.events.on(SimEvents.AGENT_REMOVED, self._on_agent_removed)

        # Drive publish from simulation step
        self.sim.events.on(SimEvents.PRE_STEP, self._on_pre_step)
        self.sim.events.on(SimEvents.POST_STEP, self._on_post_step)

        # simulation_interfaces services
        self._sim_services = None
        if enable_sim_services:
            from .sim_services import SimServices

            self._sim_services = SimServices(self, self.sim, self, command_queue=self.command_queue)

        # Start simulation loop on a daemon thread.
        # rclpy.spin() runs on the main thread for action server callbacks.
        # From here on, ROS callbacks hand their commands to the loop's PRE_STEP.
        self.command_queue.activate()
        self._sim_thread = threading.Thread(target=self._run_simulation_loop, daemon=True, name="sim_loop")
        self._sim_thread.start()

        self.get_logger().info(
            f"BridgeNode started: {actual_count} robots, "
            f"dt={timestep:.4f}s, target_rtf={effective_rtf}, "
            f"publish_rate={publish_rate:.1f}Hz, gui={effective_gui}, "
            f"enable_monitor_gui={effective_monitor_gui}, physics={effective_physics}"
        )

    # ------------------------------------------------------------------
    # Bridge plugins (singleton handlers loaded from config)
    # ------------------------------------------------------------------

    def _load_bridge_plugins(self, bridge_config: Dict[str, Any]) -> List[BridgePlugin]:
        """Dynamically load bridge plugins from config.

        Config format::

            bridge_plugins:
              - class: my_pkg.workcell_handler.WorkcellHandler  # cross-package
                config:
                  item_search_radius: 1.0
              - type: my_plugin        # registry shorthand (same package only)
                config: {}

        Each plugin class must be a :class:`BridgePlugin` subclass.

        Note: ``type:`` shorthand only works when the plugin class module
        has already been imported (so ``__init_subclass__`` can register it).
        For cross-package plugins, use the ``class:`` dotted-path format.
        """
        from .bridge_plugin import create_bridge_plugin_from_entry

        entries: List[Dict[str, Any]] = list(bridge_config.get("bridge_plugins", []))

        plugins: List[BridgePlugin] = []
        for entry in entries:
            if not (entry.get("type") or entry.get("class")):
                continue
            label = entry.get("type") or entry.get("class", "?")
            try:
                plugin = create_bridge_plugin_from_entry(entry, self, self.sim)
                plugins.append(plugin)
                self.get_logger().info(f"Loaded bridge plugin: {label}")
            except ImportError as e:
                self.get_logger().warning(f"Bridge plugin {label} not available: {e}")
            except Exception as e:
                self.get_logger().error(f"Failed to load bridge plugin {label}: {e}", exc_info=True)

        return plugins

    # ------------------------------------------------------------------
    # Robot handler registration
    # ------------------------------------------------------------------

    def _should_create_per_robot_handler(self, agent: Agent) -> bool:
        """Return whether the current config allows a per-robot handler."""
        return self._api_config.per_robot_api.robot_enabled(agent.name)

    def _register_robot_handler(self, agent: Agent) -> None:
        """Create a RobotHandler for *agent* and register it.

        If the agent already has a KinematicController (e.g. from
        ``controller``), it is passed to RobotHandler so that
        ``cmd_vel`` messages drive the controller.
        If no controller is present, an :class:`OmniController`
        is attached **only for omnidirectional agents**. Differential-drive
        agents use TPI-based navigation directly and do not need a controller
        for ``cmd_vel`` (they use ``goal_pose`` / ``path`` topics instead).
        """
        motion_ctrl = agent._controller
        if motion_ctrl is None and agent._motion_mode == MotionMode.OMNIDIRECTIONAL:
            motion_ctrl = OmniController()
            agent.set_controller(motion_ctrl)
        classes = resolve_handler_classes(agent, self._handler_map, [self.handler_class])
        handlers = []
        for cls in classes:
            kwargs = {"tf_broadcaster": self._tf_broadcaster}
            if _is_robot_handler_class(cls) and _handler_accepts_interface_config(cls):
                kwargs["interface_config"] = self._api_config.per_robot_api
            handler = cls(self, agent, **kwargs)
            handlers.append(handler)
            if cls is not RobotHandler:
                logger.info("Using custom handler %s for '%s'", _handler_display_name(cls), agent.name)
        with self._handler_lock:
            if agent.object_id in self._handlers:
                duplicate_handlers = handlers
                handlers = []
            else:
                duplicate_handlers = []
            for handler in handlers:
                _register_step_handler(
                    handler,
                    self._pre_step_handlers,
                    self._post_step_handlers,
                    self._throttled_post_step_handlers,
                )
            if handlers:
                self._handlers[agent.object_id] = handlers
        if duplicate_handlers:
            self._destroy_handlers(duplicate_handlers)

    def _unregister_step_handlers(self, handlers: List["RobotHandlerBase"]) -> None:
        """Remove handlers from per-step dispatch lists."""
        with self._handler_lock:
            for handler in handlers:
                if handler in self._pre_step_handlers:
                    self._pre_step_handlers.remove(handler)
                if handler in self._post_step_handlers:
                    self._post_step_handlers.remove(handler)
                if handler in self._throttled_post_step_handlers:
                    self._throttled_post_step_handlers.remove(handler)

    def _destroy_handlers(self, handlers: List["RobotHandlerBase"]) -> None:
        """Destroy handlers after they are unreachable from step dispatch lists."""
        with self._handler_callback_lock:
            for handler in handlers:
                handler.destroy()

    def spawn_robot(self, spawn_params: AgentSpawnParams) -> Agent:
        """Spawn a robot dynamically (e.g., via SpawnEntity service)."""
        agent = Agent.from_params(spawn_params, sim_core=self.sim)
        # from_params emits AGENT_SPAWNED at the end of construction, which
        # _on_agent_spawned already turns into a RobotHandler. Register here only
        # if that didn't happen (idempotent — avoids a double registration).
        if self._should_create_per_robot_handler(agent):
            self._register_robot_handler(agent)
        self.get_logger().info(f"Spawned robot '{agent.name}' (id={agent.object_id})")
        return agent

    def remove_robot(self, object_id: int) -> bool:
        """Remove a robot and its ROS interfaces."""
        with self._handler_lock:
            handlers = self._handlers.pop(object_id, None)
            if handlers:
                self._unregister_step_handlers(handlers)
        if handlers:
            agent = handlers[0].agent
        else:
            agent = next((a for a in self.sim.agents if a.object_id == object_id), None)
        if agent is None:
            return False
        if handlers:
            self._destroy_handlers(handlers)
        self.sim.remove_object(agent)
        self.get_logger().info(f"Removed robot '{agent.name}' (id={object_id})")
        return True

    def _on_agent_spawned(self, agent, **kwargs):
        """Auto-register RobotHandler when a new agent spawns."""
        if not self._should_create_per_robot_handler(agent):
            return
        with self._handler_lock:
            if agent.object_id in self._handlers:
                return
        self._register_robot_handler(agent)
        with self._handler_lock:
            n = len(self._handlers[agent.object_id])
        self.get_logger().info(f"Registered {n} handler(s) for {agent.name}")

    def _on_agent_removed(self, agent, **kwargs):
        """Auto-cleanup when agent is removed."""
        with self._handler_lock:
            handlers = self._handlers.pop(agent.object_id, None)
            if handlers is not None:
                self._unregister_step_handlers(handlers)
        if handlers is not None:
            self._destroy_handlers(handlers)
            self.get_logger().info(f"Removed handler(s) for {agent.name}")

    def _run_simulation_loop(self) -> None:
        """Run ``sim.run_simulation()`` on a daemon thread."""
        try:
            self.sim.run_simulation()
        except Exception as e:
            if "Not connected" in str(e) or "physics server" in str(e):
                self.get_logger().warn("PyBullet disconnected, sim thread exiting")
            else:
                self.get_logger().error(f"Simulation error: {e}")

    def _on_pre_step(self, dt, sim_time, **kwargs) -> None:
        """PRE_STEP — delegate to handler.pre_step()."""
        if getattr(self, "_shutdown_requested", False):
            return
        with self._shutdown_lock:
            if self._shutdown_requested:
                return
            # Commands accepted by ROS callbacks since the last step, applied
            # before any handler or the step itself reads them. The step lock
            # stays held until POST_STEP, so a callback arriving mid-step queues.
            self.command_queue.begin_step()
            stamp = sim_time_to_ros_time(sim_time)
            with self._handler_lock:
                handlers = list(self._pre_step_handlers)
            with self._handler_callback_lock:
                for h in handlers:
                    h.pre_step(dt=dt, stamp=stamp)

    def _on_post_step(self, dt, sim_time, **kwargs) -> None:
        """POST_STEP — publish /clock and delegate to handler.post_step()."""
        # A stub without __init__ (tests) has no queue; a real node always does.
        queue = getattr(self, "command_queue", None)
        try:
            if getattr(self, "_shutdown_requested", False):
                return
            with self._shutdown_lock:
                if self._shutdown_requested:
                    return
                self._on_post_step_active(dt, sim_time)
        finally:
            if queue is not None:
                queue.end_step()

    def _on_post_step_active(self, dt, sim_time) -> None:
        """Publish post-step state while ROS handles are known to be live."""
        try:
            stamp = sim_time_to_ros_time(sim_time)

            # Publish /clock and publish-oriented post-step handlers (throttled)
            should_run_throttled_post_step = sim_time - self._last_publish_time >= self._publish_min_interval
            if sim_time - self._last_clock_time >= self._clock_min_interval:
                clock_msg = Clock()
                clock_msg.clock = stamp
                self._clock_pub.publish(clock_msg)
                self._last_clock_time = sim_time

            # Per-robot update (odom, TF, joint_states, diagnostics)
            if should_run_throttled_post_step:
                self._fleet_ros.post_step(stamp=stamp)
                for interface in self._manager_fleet_ros:
                    interface.post_step(stamp=stamp)
                with self._handler_lock:
                    throttled_handlers = list(self._throttled_post_step_handlers)
                with self._handler_callback_lock:
                    for h in throttled_handlers:
                        h.post_step(dt=dt, stamp=stamp)
                self._last_publish_time = sim_time
            with self._handler_lock:
                post_handlers = list(self._post_step_handlers)
            with self._handler_callback_lock:
                for h in post_handlers:
                    h.post_step(dt=dt, stamp=stamp)

            # Bridge plugins (workcell handler, etc.)
            for bp in self._bridge_plugins:
                bp.post_step(sim_time)
        except Exception as e:
            if "Not connected" in str(e) or "physics server" in str(e):
                return
            raise

    def destroy_node(self):
        """Stop simulation callbacks before rclpy destroys publisher handles."""
        with self._shutdown_lock:
            if self._shutdown_requested:
                return False
            self._shutdown_requested = True
        # No step will apply the queue any more; release anyone waiting on it.
        self.command_queue.deactivate()
        return super().destroy_node()

    def reset(self) -> None:
        """Destroy all robot handlers and reset the simulation."""
        with self._handler_lock:
            handlers_to_destroy = [handler for handlers in self._handlers.values() for handler in handlers]
            self._handlers.clear()
            self._pre_step_handlers.clear()
            self._post_step_handlers.clear()
            self._throttled_post_step_handlers.clear()
        self._destroy_handlers(handlers_to_destroy)
        self._last_clock_time = -1.0
        self._last_publish_time = -1.0
        for bp in self._bridge_plugins:
            bp.on_reset()
            bp.destroy()
        self._bridge_plugins.clear()
        self.sim.reset()
        self.get_logger().info("BridgeNode reset: all handlers destroyed, simulation reset")

    @property
    def handlers(self) -> Dict[int, List["RobotHandlerBase"]]:
        """Access robot handlers (used by SimServices for spawn/delete)."""
        return self._handlers


def main(args=None):
    # Enable Python logging so that logger.info() from WorkcellHandler,
    # DoorHandler, etc. is visible in docker logs.
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    rclpy.init(args=args)
    node = BridgeNode()
    # MultiThreadedExecutor allows action server execute callbacks
    # (which use time.sleep polling) to run on separate threads without
    # blocking the simulation step timer.
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # SIGINT may already have shut down the default context before the
        # executor exits. Avoid turning an otherwise clean Ctrl-C into RCLError.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
