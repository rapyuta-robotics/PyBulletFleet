"""
core/agent.py
Agent class for goal-based position control with max velocity and acceleration constraints.
Supports both mobile (use_fixed_base=False) and fixed (use_fixed_base=True) robots.
Supports both Mesh and URDF loading.
"""

import copy
import logging
import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import pybullet as p
from .geometry import Pose, as_axes
from .sim_object import ShapeParams, SimObject, SimObjectSpawnParams, load_urdf_body
from .action import Action
from .types import ControllerMode, MotionMode, MovementDirection, ActionStatus, CollisionMode
from .tools import normalize_vector_param  # noqa: F401  (re-exported for legacy callers)
from pybullet_fleet._tpi import try_build_tpi
from pybullet_fleet.joint import KinematicJoint
from pybullet_fleet.tools import resolve_joint_index, resolve_link_index
from .logging_utils import get_lazy_logger
from pybullet_fleet.events import SimEvents
from pybullet_fleet._defaults import AGENT as _AGT_D, IK as _IK_D
from pybullet_fleet.config_utils import forward_spawn_params as _forward_spawn_params
from pybullet_fleet.agent_plugin import AgentPlugin

if TYPE_CHECKING:
    from pybullet_fleet._replan import PathRejection
    from pybullet_fleet.controller import Controller
    from pybullet_fleet.controller_params import ControllerParams

# Create logger for this module
logger = logging.getLogger(__name__)
lazy_logger = get_lazy_logger(__name__)

# Generic plugin type so get_plugin() returns the concrete subclass type.
_PluginT = TypeVar("_PluginT", bound=AgentPlugin)


@dataclass
class AgentSpawnParams(SimObjectSpawnParams):
    """
    Agent spawn parameters extending SimObjectSpawnParams.

    These parameters define the physical properties, appearance, and initial state of an agent.
    They are used by Agent.from_params() and AgentManager.spawn_agents_grid().

    Supports three types of robots:
    1. Mesh robots: Specify visual_shape (and optionally collision_shape)
    2. URDF robots: Specify urdf_path (for robots with joints)
    3. Virtual agents: Specify neither (invisible, no collision, useful for tracking/planning)

    Attributes:
        urdf_path: Path to robot URDF file (for URDF-based robots with joints)
        motion_mode: MotionMode.OMNIDIRECTIONAL (move in any direction) or MotionMode.DIFFERENTIAL (rotate then move forward)
        use_fixed_base: If True, robot base is fixed and doesn't move (default: False)
        plugins: List of plugin config dicts for per-agent plugins.
            Each entry is ``{"type": "battery", "config": {...}}``
            or ``{"class": "my.Module", "config": {...}}``.
        controller: Movement controller selector / configuration. Accepts a
            registry-name string (``"omni"``), a flat dict, a list of dicts, a
            :class:`ControllerParams`, or a pre-built :class:`Controller`.
            Each dict entry uses either ``type:`` (registry name) or ``class:``
            (dotted import path), plus controller-specific config inline.
            KinematicController entries (omni, differential) replace the base
            controller; high-level entries (patrol, random_walk) are appended on
            top of it. A lone high-level entry auto-creates the base from
            ``motion_mode``.
            Example single: ``{"type": "patrol", "waypoints": [...], "loop": True}``
            Example custom: ``{"class": "my_pkg.MyController", "max_linear_vel": 1.5}``
            Example list: ``[{"type": "differential", ...}, {"type": "patrol", ...}]``

    See ``Agent.__init__`` for the ``controller`` argument (selector / configuration for the
    Agent's movement controller; supports str / dict / list / Controller / ControllerParams).

    Inherited from SimObjectSpawnParams:
        name: Optional string name for human-readable identification.
              Duplicates allowed - use for debugging/filtering, not unique lookup.
              For unique identification, use object_id after spawning.

    Note:
        AgentManager.spawn_agents_grid() calculates positions automatically and may override initial_pose.
    """

    #: Agents are not pickable by default -- a robot is usually the one doing
    #: the picking -- which narrows SimObjectSpawnParams' default of True.
    #: Redeclared rather than set elsewhere so the value a caller passes is
    #: the value the agent gets; Agent.__init__ used to overwrite it
    #: unconditionally, which made the field inert.
    pickable: bool = False
    motion_mode: Union[MotionMode, str] = MotionMode(_AGT_D["motion_mode"])
    use_fixed_base: bool = _AGT_D["use_fixed_base"]
    ik_params: Optional["IKParams"] = None
    controller: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]], "ControllerParams", "Controller"]] = None
    plugins: Optional[List[Dict[str, Any]]] = None
    global_scaling: float = _AGT_D["global_scaling"]
    joint_tolerance: Optional[Union[float, list, dict]] = _AGT_D["joint_tolerance"]

    def __post_init__(self):
        """Validate agent spawn parameters."""
        has_visual = self.visual_shape is not None
        has_urdf = self.urdf_path is not None

        if not has_visual and not has_urdf:
            logger.warning(
                "AgentSpawnParams: Neither visual_shape nor urdf_path specified. "
                "This will create a virtual agent (invisible, no collision)."
            )
        if has_visual and has_urdf:
            logger.warning(
                "AgentSpawnParams: Both visual_shape and urdf_path specified. "
                "visual_shape will be ignored - URDF will be used."
            )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """
        Create AgentSpawnParams from configuration dictionary.

        Args:
            config: Dictionary with robot parameters including:

                - ``visual_shape`` / ``collision_shape`` or ``urdf_path``
                - ``initial_pose``: ``Pose`` object (optional)
                - ``controller``: dict / str / ``None`` — see :attr:`controller`
                  and the *Controller Configuration* how-to
                - ``use_fixed_base``, ``motion_mode``, ``ik_params`` (optional)

        Returns:
            AgentSpawnParams instance

        Example::

            config = {
                'visual_shape': ShapeParams(
                    shape_type='mesh',
                    mesh_path='robot.obj',
                    mesh_scale=[1.0, 1.0, 1.0],
                    rgba_color=[0.2, 0.2, 0.2, 1.0]
                ),
                'collision_shape': ShapeParams(
                    shape_type='box',
                    half_extents=[0.2, 0.1, 0.2]
                ),
                'initial_pose': Pose.from_xyz(1.0, 2.0, 0.0),
                'controller': {'max_linear_vel': 2.0, 'max_linear_accel': 5.0},
                'use_fixed_base': False
            }
            params = AgentSpawnParams.from_dict(config)

            # Or for URDF:
            config = {
                'urdf_path': 'robots/mobile_robot.urdf',
                'initial_pose': Pose.from_xyz(1.0, 2.0, 0.0),
                'use_fixed_base': False
            }
            params = AgentSpawnParams.from_dict(config)
        """
        # Get base fields from parent
        base = SimObjectSpawnParams.from_dict(config)
        base_kwargs = {f.name: getattr(base, f.name) for f in fields(base)}

        # ...except pickable, where this class narrows the parent's default to
        # False. SimObjectSpawnParams.from_dict() has already resolved an
        # absent key to its own True, and splatting that in would reinstate it
        # for every config-driven agent -- the common path -- while the direct
        # constructor said False.
        if "pickable" not in config:
            base_kwargs.pop("pickable", None)

        # Get motion_mode and convert string to enum if needed
        motion_mode_value = config.get("motion_mode", _AGT_D["motion_mode"])
        if isinstance(motion_mode_value, str):
            motion_mode_value = MotionMode(motion_mode_value)

        # Parse ik_params from nested dict if provided
        ik_params_value = config.get("ik_params")
        if isinstance(ik_params_value, dict):
            ik_params_value = IKParams(**ik_params_value)

        # ``controller`` is passed through as-is; string ("omni"), dict
        # ({"type": "omni", ...} or {"class": "...", ...}) and list-of-dicts
        # forms are all accepted and wired by :meth:`Agent._init_controller_chain`.
        controller_value = config.get("controller")

        # Parse plugins list (pass through as-is; resolved in from_params)
        plugins_value = list(config.get("plugins") or [])

        # urdf_path is not repeated here: SimObjectSpawnParams carries it now,
        # so it already arrives in base_kwargs.
        return cls(
            **base_kwargs,
            motion_mode=motion_mode_value,
            use_fixed_base=config.get("use_fixed_base", _AGT_D["use_fixed_base"]),
            ik_params=ik_params_value,
            controller=controller_value,
            plugins=plugins_value or None,
            global_scaling=config.get("global_scaling", 1.0),
            joint_tolerance=config.get("joint_tolerance"),
        )


@dataclass(frozen=True)
class IKParams:
    """Inverse kinematics solver configuration.

    Controls the multi-seed iterative IK solver used by
    :meth:`Agent._solve_ik` and :meth:`Agent.move_end_effector`.

    Attributes:
        max_outer_iterations: Maximum refinement iterations per seed.
        convergence_threshold: EE-to-target distance (m) below which
            the IK solution is accepted early.
        max_inner_iterations: ``maxNumIterations`` passed to
            ``pybullet.calculateInverseKinematics``.
        residual_threshold: ``residualThreshold`` passed to
            ``pybullet.calculateInverseKinematics``.
        reachability_tolerance: Maximum EE distance (m) for
            :meth:`Agent.move_end_effector` to report *reachable*.
        seed_quartiles: Tuple of joint-range fractions (0–1) used to
            build diversified rest-pose seeds.  Each value ``q`` produces
            a seed at ``lower + q * range`` for every joint.
        ik_joint_names: Explicit tuple of joint **names** that the IK
            solver is allowed to move.  All other movable joints are
            locked at their current position.  When ``None`` (default),
            the solver falls back to the automatic heuristic (continuous
            joints are locked, everything else is free).

    Example::

        cfg = IKParams(max_outer_iterations=10, seed_quartiles=(0.1, 0.5, 0.9))
        arm = Agent.from_urdf("arm.urdf", ik_params=cfg)

        # Mobile manipulator — solve only for arm joints
        cfg = IKParams(ik_joint_names=(
            "mount_to_shoulder", "shoulder_to_elbow",
            "elbow_to_wrist", "wrist_to_end",
        ))
        robot = Agent.from_urdf("mobile_manipulator.urdf", ik_params=cfg)
    """

    max_outer_iterations: int = _IK_D["max_outer_iterations"]
    convergence_threshold: float = _IK_D["convergence_threshold"]
    max_inner_iterations: int = _IK_D["max_inner_iterations"]
    residual_threshold: float = _IK_D["residual_threshold"]
    reachability_tolerance: float = _IK_D["reachability_tolerance"]
    seed_quartiles: Tuple[float, ...] = (0.25, 0.5, 0.75)
    ik_joint_names: Optional[Tuple[str, ...]] = None


def validate_joint_motion_profile(
    max_velocity: Optional[float] = None,
    max_accel: Optional[float] = None,
    max_decel: Optional[float] = None,
    current_accel: Optional[float] = None,
) -> None:
    """Refuse an unusable joint motion profile from the values alone.

    :meth:`Agent.set_joint_motion_profile` needs a body to resolve the joint
    against, which is too late for a caller that spawns the body and then
    configures it: a profile refused at that point leaves the half-configured
    object registered with the simulation. This checks everything that does
    not need the body, so such a caller can refuse before spawning.

    Args:
        max_velocity: Speed cap, or ``None``.
        max_accel: Acceleration, or ``None``.
        max_decel: Deceleration, or ``None``.
        current_accel: Acceleration the joint already has, for a profile being
            changed rather than set. ``max_decel`` on its own is valid when
            there is one.

    Raises:
        ValueError: If a value given is not a finite number above zero, or if
            ``max_decel`` is given for a joint that has no acceleration.
    """
    for label, value in (("max_velocity", max_velocity), ("max_accel", max_accel), ("max_decel", max_decel)):
        if value is None:
            continue
        # isfinite before the comparison: NaN and inf both slip past
        # `value <= 0.0`, and would then be stored and produce a nan or
        # infinite step.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be a finite number > 0, got {value!r}")
    if max_decel is not None and max_accel is None and current_accel is None:
        # Braking is the second half of a ramp; without an acceleration there
        # is no ramp to brake out of, and the joint would take the
        # constant-speed path and ignore this entirely. Refused rather than
        # accepted as a setting that does nothing.
        raise ValueError(
            "max_decel needs max_accel: a joint with no acceleration runs at "
            "constant speed, so there is no braking phase for it to apply to."
        )


class Agent(SimObject):
    """
    Agent class with goal-based position control.

    Inherits from SimObject to get attach/detach and callback functionality.
    Supports both mobile robots (use_fixed_base=False) and fixed robots (use_fixed_base=True).
    Supports both Mesh and URDF loading.

    Class Attributes:
        _spawn_params_cls: The SpawnParams dataclass used by config-driven
            grid spawning (``AgentSpawnParams``).

    Class Constants:
        _KINEMATIC_JOINT_FALLBACK_VELOCITY: Default max joint velocity for
            revolute joints (rad/s) used when the URDF
            ``<limit velocity="...">`` is 0 or missing.
        _KINEMATIC_PRISMATIC_FALLBACK_VELOCITY: Default max joint velocity
            for prismatic joints (m/s) used when the URDF limit is 0 or
            missing.
        _DEFAULT_JOINT_TOLERANCE: Default tolerance (rad / m) for joint
            target comparison methods.

    Features:
    - Accepts Pose goals (compatible with ROS2 geometry_msgs/Pose)
    - Max velocity and acceleration constraints (for mobile robots)
    - Current pose query API
    - Shared shape optimization for fast spawning
    - Fixed robot support (non-moving objects)
    - URDF support with joint control
    - Attach/detach objects (inherited from SimObject)

    Object Identification (inherited from SimObject):
    - object_id: Unique integer ID (assigned by sim_core, use for lookups)
    - name: Optional string name (duplicates allowed, use for debugging/filtering)

    This class is designed to be eventually controlled by ROS2 nodes,
    so the API follows ROS message conventions.
    """

    _needs_update = True  # Agents always participate in the step_once() update loop

    _entity_type_name = "agent"  # Auto-registers in ENTITY_REGISTRY via SimObject.__init_subclass__

    _KINEMATIC_JOINT_FALLBACK_VELOCITY: float = 2.0  # rad/s (revolute)
    _KINEMATIC_PRISMATIC_FALLBACK_VELOCITY: float = 0.5  # m/s (prismatic)
    _DEFAULT_JOINT_TOLERANCE: float = 0.01  # rad (or m for prismatic)

    # Override parent: use AgentSpawnParams for config-driven grid spawning
    _spawn_params_cls = AgentSpawnParams
    _kinematic_joints_physics_off_logged: bool = False  # Log physics=False fallback only once

    def __init__(
        self,
        body_id: int,
        urdf_path: Optional[str] = None,
        motion_mode: Union[MotionMode, str] = MotionMode(_AGT_D["motion_mode"]),
        use_fixed_base: bool = _AGT_D["use_fixed_base"],
        collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
        sim_core=None,
        name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
        mass: Optional[float] = None,
        ik_params: Optional[IKParams] = None,
        joint_tolerance: Optional[Union[float, list, dict]] = None,
        controller: Optional[Union[str, Dict[str, Any], "ControllerParams", "Controller"]] = None,
        notify_spawn: bool = True,
        # Agents default to not pickable -- a robot is usually the thing doing
        # the picking -- but an agent can be cargo too: a lift car's platform,
        # or a pallet with a joint. attach_object() refuses a non-pickable
        # body, so it has to be settable at the spawn rather than only after.
        pickable: bool = False,
    ):
        """
        Initialize Agent.

        Args:
            body_id: PyBullet body ID
            urdf_path: Path to robot URDF file (if URDF-based)
            motion_mode: MotionMode.OMNIDIRECTIONAL or MotionMode.DIFFERENTIAL drive.
                Used as the default controller selector when ``controller``
                has no ``"type"`` key.  If ``controller={"type": ...}`` is
                given and disagrees with ``motion_mode``, a warning is logged
                and ``controller.type`` wins.

                .. deprecated::
                    Prefer ``controller="omni"`` / ``controller="differential"``
                    (or a full ``controller={...}`` dict).  ``motion_mode`` is
                    retained for backward compatibility and as a coarse
                    "2D mobile base shape" hint for batched controllers.
            use_fixed_base: If True, robot base is fixed and doesn't move
            collision_mode: Collision detection mode (default: NORMAL_3D)
            sim_core: Reference to simulation core (optional)
            ik_params: IK solver configuration (default: ``IKParams()``).
            joint_tolerance: Default joint tolerance for convergence checks.
                Accepts ``float`` (all joints), ``list`` (per-joint), or
                ``dict`` (``{joint_name: tol}``).
                If ``None``, uses the class default (0.01).
                For mixed prismatic / revolute chains, a dict is
                recommended so prismatic joints (metres) can use a
                tighter tolerance than revolute joints (radians).
            controller: Controller selector / configuration.  Accepts:

                * ``None`` — framework defaults; ``motion_mode`` selects
                  ``OmniController`` or ``DifferentialController``.
                * ``str`` — registry name, e.g. ``"omni"``,
                  ``"differential"``, ``"patrol"``.  Equivalent to
                  ``{"type": <str>}``.
                * ``dict`` — ``{"type": "...", "max_linear_vel": 1.5,
                  "wheel_separation": 0.3, ...}``.  ``ControllerParams``
                  fields populate :class:`ControllerParams`; impl-specific
                  extras (e.g. ``wheel_separation``) flow through to the
                  controller's ``__init__``.  **Recommended for YAML /
                  config-file workflows.**
                * :class:`~pybullet_fleet.controller.Controller` instance —
                  any pre-built controller (built-in or custom).  Installed
                  as-is; ``self.controller_params`` is taken from
                  ``controller.params``.  **Recommended for Python
                  workflows that need custom controllers or want
                  type-safe construction.**
                * :class:`~pybullet_fleet.controller_params.ControllerParams`
                  instance — params only; ``motion_mode`` still picks the
                  Omni/Diff class.  Use this only when you want to tweak
                  shared params but keep the default class.

        Note:
            For spawning robots from AgentSpawnParams, use Agent.from_params() instead.
            For mesh robots, use Agent.from_mesh() factory method.
            For URDF robots, use Agent.from_urdf() factory method.
        """
        # Initialize SimObject base class (collision_mode is forwarded so
        # add_object receives the correct mode directly – no post-hoc transition)
        # pickable goes in through super(), not assigned afterwards: SimObject
        # registers the body and emits OBJECT_SPAWNED from there, so a late
        # overwrite let every listener see the default rather than what the
        # agent was configured with.
        super().__init__(
            body_id,
            sim_core=sim_core,
            collision_mode=collision_mode,
            name=name,
            user_data=user_data,
            mass=mass,
            pickable=pickable,
        )

        self.urdf_path = urdf_path

        self.use_fixed_base = use_fixed_base

        # IK solver configuration
        self._ik_params: IKParams = ik_params if ik_params is not None else IKParams()

        # Instance-level joint tolerance (None → use class default)
        self._joint_tolerance: Optional[Union[float, list, dict]] = joint_tolerance

        # URDF-specific: joint information
        self.joint_info = []
        if urdf_path is not None:
            num_joints = p.getNumJoints(body_id, physicsClientId=self._pid)
            self.joint_info = [p.getJointInfo(body_id, j, physicsClientId=self._pid) for j in range(num_joints)]

        # Cached movable (non-fixed) joint indices — avoids recomputing in
        # _solve_ik on every call.  Built once at construction time since
        # joint topology does not change after loading.
        self._cached_movable_indices: list = [i for i, info in enumerate(self.joint_info) if info[2] != p.JOINT_FIXED]

        # Goal tracking (only used for non-static robots) - Private internal state
        self._current_velocity = np.array([0.0, 0.0, 0.0])
        self._current_angular_velocity = 0.0  # rad/s (for differential drive)
        self._is_moving = False  # Private: use property for read-only access

        # Set by BatchKinematicController.register_agent(); cleared on unregister.
        # When set, agent.update() skips ctrl.compute() (batch_advance handles
        # movement) but still runs _update_actions() for action queue processing.
        self._batch_controller = None  # type: Optional[Any]

        # Controller chain — low-level controller at index 0, higher-level
        # controllers appended via add_controller().  update() iterates in
        # reversed order (high → low) so that patrol/random-walk controllers
        # can set goals before the base navigation controller executes.
        self._controllers: List[Any] = []

        # Action queue system for high-level task execution
        self._action_queue: List[Action] = []
        self._current_action: Optional[Action] = None

        #: Per-joint kinematic state, {joint_index: KinematicJoint}: where the
        #: joint is, where it was last told to go, how it is configured to get
        #: there, and how that is going. Created on first use, and seeded
        #: below for every joint of a kinematic URDF body so that a position
        #: can be read without a PyBullet call. A target persists after
        #: arrival, for are_joints_at_targets() to be asked about later.
        self._joints: Dict[int, KinematicJoint] = {}
        #: The subset of _joints that has a target, which is what the step
        #: loop has to visit. An index into _joints rather than a copy of it:
        #: the entries are the same objects. Worth keeping because a jointed
        #: body usually has one joint under command and six not, and because
        #: an agent that has never commanded one leaves the loop at once.
        self._active_joints: List[KinematicJoint] = []
        #: Time base for joint trajectories. Kept here rather than read from
        #: sim_core so an agent without one still ramps.
        self._joint_clock: float = 0.0

        # Cached flag: True when joints must use kinematic interpolation
        # (mass=0 OR sim_core has physics disabled).  Computed once to avoid
        # per-step property overhead.
        self._use_kinematic_joints: bool = self._compute_use_kinematic_joints()

        # Seeding the positions here avoids a per-step p.getJointState() call.
        # Only for kinematic joints; in physics mode PyBullet owns them, and
        # an entry is created on demand for a target instead.
        if self._use_kinematic_joints and self.joint_info:
            indices = list(range(len(self.joint_info)))
            states = p.getJointStates(body_id, indices, physicsClientId=self._pid)
            self._joints = {i: KinematicJoint(index=i, position=states[i][0]) for i in indices}

        # Per-agent plugin system (e.g. BatteryPlugin)
        self._plugins: List[AgentPlugin] = []

        # Path visualization settings
        self.path_visualize: bool = False  # Enable/disable path visualization
        self.path_visualize_color: Optional[List[float]] = None  # RGB color (None = default [0, 1, 1])
        self.path_visualize_width: float = 2.0  # Line width
        self._path_debug_ids: List[int] = []  # Store debug line IDs for cleanup

        self._motion_mode = motion_mode if isinstance(motion_mode, MotionMode) else MotionMode(motion_mode)

        from pybullet_fleet.controller import Controller
        from pybullet_fleet.controller_params import ControllerParams

        if isinstance(controller, Controller):
            preset_params = getattr(controller, "params", None)
            self.controller_params: ControllerParams = (
                preset_params if isinstance(preset_params, ControllerParams) else ControllerParams()
            )
            self.set_controller(controller)
        elif isinstance(controller, ControllerParams):
            self.controller_params = controller
            self.set_controller(self._make_default_controller())
        else:
            self._init_controller_chain(controller)

        # Update log prefix to include Agent class name (SimObject sets it initially)
        self._update_log_prefix()

        # Notify observers that the agent spawned — but only once it is fully
        # constructed. add_object() (called mid-construction from super().__init__)
        # only emits OBJECT_SPAWNED; emitting AGENT_SPAWNED there would expose a
        # half-built Agent (no _controllers / _current_action). from_params()
        # attaches plugins AFTER __init__, so it constructs with notify_spawn=False
        # and emits the event itself once plugins are attached — otherwise a
        # dynamically spawned battery robot would be observed (and its RobotHandler
        # built) before its BatteryPlugin exists.
        if notify_spawn:
            self._emit_agent_spawned()

    def _emit_agent_spawned(self) -> None:
        """Emit AGENT_SPAWNED for this fully-constructed agent."""
        if self.sim_core is not None:
            self.sim_core.events.emit(SimEvents.AGENT_SPAWNED, agent=self)

    # ------------------------------------------------------------------
    # Kinematic-limit read-only accessors.
    # These delegate to ``self.controller_params`` (the single source of
    # truth) and always return a normalised ``np.ndarray`` of shape ``(3,)``
    # — matching the legacy ``Agent.max_*`` attribute shape so existing
    # callers / tests can read without migration.  Writes must go through
    # ``self.controller_params`` directly.
    # ------------------------------------------------------------------
    @property
    def max_linear_vel(self) -> np.ndarray:
        return as_axes(self.controller_params._eff_linear_vel())

    @property
    def max_linear_accel(self) -> np.ndarray:
        return as_axes(self.controller_params._eff_linear_accel())

    @property
    def max_linear_decel(self) -> np.ndarray:
        """Effective deceleration limit; mirrors :attr:`max_linear_accel` when unset."""
        return as_axes(self.controller_params._eff_linear_decel())

    @property
    def max_angular_vel(self) -> np.ndarray:
        return as_axes(self.controller_params._eff_angular_vel())

    @property
    def max_angular_accel(self) -> np.ndarray:
        return as_axes(self.controller_params._eff_angular_accel())

    @property
    def joint_tolerance(self) -> Union[float, list, dict]:
        """Default joint tolerance used by convergence checks.

        Returns the instance-level tolerance if set, otherwise the class
        default (``_DEFAULT_JOINT_TOLERANCE = 0.01``).

        For robots with mixed prismatic (metres) and revolute (radians)
        joints, set this to a ``dict`` so each joint type can have an
        appropriate tolerance::

            agent = Agent.from_urdf(
                ...,
                joint_tolerance={"rail_joint": 0.005, "elbow_to_wrist": 0.05},
            )
        """
        if self._joint_tolerance is not None:
            return self._joint_tolerance
        return self._DEFAULT_JOINT_TOLERANCE

    @joint_tolerance.setter
    def joint_tolerance(self, value: Optional[Union[float, list, dict]]) -> None:
        self._joint_tolerance = value

    def _resolve_joint_tolerance(
        self,
        tolerance: Optional[Union[float, list, dict]],
        joint_index: int,
    ) -> float:
        """Resolve a possibly-compound tolerance to a scalar for one joint.

        Handles the full fallback chain:

        1. If *tolerance* is ``None``, fall back to ``self.joint_tolerance``.
        2. If the result is a ``dict``, look up by joint name
           (falling back to ``_DEFAULT_JOINT_TOLERANCE`` if the name is absent).
        3. If the result is a ``list``/``tuple``, index by *joint_index*
           (falling back to ``_DEFAULT_JOINT_TOLERANCE`` when out of range).
        4. Otherwise return the scalar as-is.

        Args:
            tolerance: ``float``, ``list``, ``dict``, or ``None``.
            joint_index: 0-based joint index.

        Returns:
            Scalar tolerance (``float``).
        """
        tol = tolerance if tolerance is not None else self.joint_tolerance
        if isinstance(tol, dict):
            name = self.joint_info[joint_index][1].decode("utf-8")
            return tol.get(name, self._DEFAULT_JOINT_TOLERANCE)
        if isinstance(tol, (list, tuple)):
            return tol[joint_index] if joint_index < len(tol) else self._DEFAULT_JOINT_TOLERANCE
        return float(tol)

    @property
    def is_moving(self) -> bool:
        """
        Read-only property: Check if robot is currently moving.

        To change movement state, use set_goal_pose(), set_path(), or stop().

        Returns:
            True if robot is moving, False otherwise
        """
        return self._is_moving

    # -- Battery properties (delegate to BatteryPlugin) ----------------------

    def _get_battery_plugin(self):
        """Return the BatteryPlugin instance, or None."""
        from pybullet_fleet.plugins.battery_plugin import BatteryPlugin

        return self.get_plugin(BatteryPlugin)

    @property
    def battery_soc(self) -> float:
        """Current state of charge [0.0, 1.0].

        Always returns ``1.0`` when no :class:`BatteryPlugin` is attached.
        """
        bp = self._get_battery_plugin()
        return bp.soc if bp is not None else 1.0

    @property
    def is_charging(self) -> bool:
        """Whether the agent is currently charging."""
        bp = self._get_battery_plugin()
        return bp.is_charging if bp is not None else False

    @property
    def battery_plugin(self):
        """The :class:`~pybullet_fleet.plugins.battery_plugin.BatteryPlugin` instance, or ``None``."""
        return self._get_battery_plugin()

    def set_charging(self, charging: bool) -> None:
        """Start or stop charging.

        Args:
            charging: ``True`` to start, ``False`` to stop.
        """
        bp = self._get_battery_plugin()
        if bp is None:
            return
        bp.set_charging(charging)

    # -- Agent plugin system -------------------------------------------------

    @property
    def plugins(self) -> List[AgentPlugin]:
        """Read-only list of active per-agent plugins."""
        return list(self._plugins)

    def add_plugin(self, plugin: AgentPlugin) -> None:
        """Attach a per-agent plugin.

        Args:
            plugin: An :class:`AgentPlugin` instance bound to this agent.

        Raises:
            ValueError: If *plugin* is not bound to this agent.
        """
        if plugin.agent is not self:
            raise ValueError(f"Plugin {plugin!r} is bound to a different agent")
        self._plugins.append(plugin)

    def remove_plugin(self, plugin_or_cls) -> None:
        """Remove a plugin by instance or class.

        When a **class** is given, the first instance of that class is
        removed.  When an **instance** is given, that exact instance is
        removed.

        Args:
            plugin_or_cls: An :class:`AgentPlugin` instance or subclass.
        """
        if isinstance(plugin_or_cls, type):
            for i, plugin in enumerate(self._plugins):
                if isinstance(plugin, plugin_or_cls):
                    self._plugins.pop(i)
                    return
        else:
            self._plugins.remove(plugin_or_cls)

    def get_plugin(self, cls: Type[_PluginT]) -> Optional[_PluginT]:
        """Return the first plugin matching *cls*, or ``None``."""
        for plugin in self._plugins:
            if isinstance(plugin, cls):
                return plugin
        return None

    def _update_plugins(self, dt: float) -> None:
        """Run all per-agent plugins.  Called from :meth:`update`."""
        for plugin in self._plugins:
            plugin.on_update(dt)

    # -- Motion properties --------------------------------------------------

    @property
    def motion_mode(self) -> MotionMode:
        """
        Read-only property: Get current motion mode.

        To change motion mode, use set_motion_mode() method.

        Returns:
            Current MotionMode (OMNIDIRECTIONAL or DIFFERENTIAL)
        """
        return self._motion_mode

    @property
    def velocity(self) -> np.ndarray:
        """
        Read-only property: Get current velocity vector.

        Returns:
            Velocity [vx, vy, vz] in m/s (copy of internal state)
        """
        return self._current_velocity.copy()

    @property
    def angular_velocity(self) -> float:
        """
        Read-only property: Get current angular velocity.

        Returns:
            Angular velocity in rad/s
        """
        return self._current_angular_velocity

    @property
    def goal_pose(self) -> Optional[Pose]:
        """
        Read-only property: Get current goal pose.

        To set a new goal, use set_goal_pose() or set_path().

        Returns:
            Current goal Pose, or None if no goal is set
        """
        if self._controllers:
            return self._controllers[0].goal_pose
        return None

    @property
    def current_waypoint_index(self) -> int:
        """
        Read-only property: Get current waypoint index in the path.

        Returns:
            Current waypoint index (0-based), or 0 if no path is set
        """
        if self._controllers:
            return self._controllers[0].current_waypoint_index
        return 0

    @property
    def path(self) -> List[Pose]:
        """Read-only property: Current waypoint path. Empty when idle."""
        if self._controllers:
            return self._controllers[0].path
        return []

    @classmethod
    def from_params(cls, spawn_params: AgentSpawnParams, sim_core=None) -> "Agent":
        """
        Create an Agent from AgentSpawnParams.

        This is the recommended way to spawn robots when using AgentSpawnParams.
        Automatically detects whether to use mesh, URDF, or virtual agent based on spawn_params.

        Args:
            spawn_params: AgentSpawnParams instance with all robot parameters
            sim_core: Reference to simulation core (optional)

        Returns:
            Agent instance

        The **recommended** way to choose a controller is to set
        ``controller={"type": "omni" | "differential" | ...}`` explicitly.
        ``motion_mode`` still works as a shortcut and as the default when
        ``controller.type`` is omitted, but if both are given,
        ``controller.type`` wins and a warning is emitted on mismatch.

        Example (Mesh, omni — recommended explicit form)::

            params = AgentSpawnParams(
                visual_shape=ShapeParams(
                    shape_type="mesh",
                    mesh_path="robot.obj",
                    mesh_scale=[1.0, 1.0, 1.0],
                    rgba_color=[0.2, 0.2, 0.2, 1.0]
                ),
                collision_shape=ShapeParams(
                    shape_type="box",
                    half_extents=[0.2, 0.1, 0.2]
                ),
                initial_pose=Pose.from_xyz(1.0, 2.0, 0.0),
                controller={"type": "omni", "max_linear_vel": 3.0},
            )
            robot = Agent.from_params(spawn_params=params)

        Example (URDF)::

            params = AgentSpawnParams(
                urdf_path="arm_robot.urdf",
                initial_pose=Pose.from_xyz(0.0, 0.0, 0.0),
                use_fixed_base=True,
            )
            robot = Agent.from_params(spawn_params=params)

        Example (Virtual Agent - invisible, no collision)::

            params = AgentSpawnParams(
                initial_pose=Pose.from_xyz(0.0, 0.0, 0.0),
                controller={"type": "omni", "max_linear_vel": 2.0},
            )
            virtual_agent = Agent.from_params(spawn_params=params)
        """
        # Forward spawn_params.controller verbatim — Agent.__init__ does
        # the single canonical parse (type / ControllerParams / impl extras)
        # and the motion_mode-vs-controller.type mismatch warning.
        # notify_spawn=False: defer AGENT_SPAWNED until plugins are attached below,
        # so observers (e.g. the ROS bridge) never see a plugin-less agent.
        extra: Dict[str, Any] = {"sim_core": sim_core, "controller": spawn_params.controller, "notify_spawn": False}
        if spawn_params.urdf_path is not None:
            agent = cls.from_urdf(
                **_forward_spawn_params(
                    cls.from_urdf,
                    spawn_params,
                    aliases={"initial_pose": "pose"},
                    extra_kwargs=extra,
                )
            )
        else:
            agent = cls.from_mesh(
                **_forward_spawn_params(
                    cls.from_mesh,
                    spawn_params,
                    aliases={"initial_pose": "pose"},
                    extra_kwargs=extra,
                )
            )

        # Load per-agent plugins from YAML config (if any)
        if spawn_params.plugins:
            from pybullet_fleet.agent_plugin import create_agent_plugin_from_entry

            for entry in spawn_params.plugins:
                agent.add_plugin(create_agent_plugin_from_entry(entry, agent))
            if isinstance(agent._checkpoint_spawn_params, AgentSpawnParams):
                agent._checkpoint_spawn_params.plugins = list(spawn_params.plugins)

        # Agent is now fully constructed (controllers + plugins) — notify observers.
        agent._emit_agent_spawned()
        agent._record_checkpoint_spawn(sim_core)
        return agent

    def _record_checkpoint_spawn(self, sim_core) -> None:
        """Report runtime Agent creation after its effective owner state exists."""
        record_spawn = getattr(sim_core, "_record_state_spawn", None) if sim_core is not None else None
        if record_spawn is not None:
            record_spawn(self, self._checkpoint_spawn_params)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], sim_core=None) -> "Agent":
        """Create an Agent from a raw config dict.

        Combines :meth:`AgentSpawnParams.from_dict` and :meth:`from_params`
        in a single call.  Uses ``cls._spawn_params_cls`` so subclasses
        (e.g. Door, Elevator) automatically use their own SpawnParams.

        Args:
            config: Entity definition dict (as parsed from YAML).
            sim_core: Reference to simulation core (optional).

        Returns:
            ``cls`` instance.
        """
        params = cls._spawn_params_cls.from_dict(config)
        return cls.from_params(params, sim_core)

    @classmethod
    def from_mesh(
        cls,
        visual_shape: Optional[ShapeParams] = None,
        collision_shape: Optional[ShapeParams] = None,
        pose: Pose = None,
        mass: float = 0.0,
        motion_mode: Union[MotionMode, str] = MotionMode(_AGT_D["motion_mode"]),
        use_fixed_base: bool = _AGT_D["use_fixed_base"],
        collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
        sim_core=None,
        name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
        controller: Optional[Union[str, Dict[str, Any], "ControllerParams", "Controller"]] = None,
        notify_spawn: bool = True,
        pickable: bool = False,
    ) -> "Agent":
        """
        Create a mesh-based Agent with flexible shape control.

        Args:
            visual_shape: ShapeParams for visual geometry
            collision_shape: ShapeParams for collision geometry
            pose: Initial Pose (position and orientation)
            mass: Robot mass (kg), 0.0 for kinematic control
            motion_mode: MotionMode.OMNIDIRECTIONAL or MotionMode.DIFFERENTIAL
            use_fixed_base: If True, robot base is fixed and doesn't move
            sim_core: Reference to simulation core
            controller: Controller selector / config.  See :meth:`Agent.__init__`
                for accepted forms (``None`` / ``str`` / ``dict`` /
                :class:`~pybullet_fleet.controller_params.ControllerParams`).

        Returns:
            Agent instance
        """
        if pose is None:
            pose = Pose.from_xyz(0.0, 0.0, 0.0)

        # Use parent class's create_shared_shapes() for optimization
        # This dramatically reduces OpenGL object count when spawning many robots
        visual_id, collision_id = cls.create_shared_shapes(
            visual_shape=visual_shape,
            collision_shape=collision_shape,
        )

        # Create multibody using helper method
        body_id = cls._create_body_from_shapes(
            visual_id=visual_id,
            collision_id=collision_id,
            pose=pose,
            mass=mass,
        )

        # Create agent instance with Agent-specific parameters
        # collision_mode is passed through __init__ -> super().__init__() -> add_object
        # so the object is registered with the correct mode directly (no post-hoc transition)
        agent = cls(
            body_id=body_id,
            motion_mode=motion_mode,
            use_fixed_base=use_fixed_base,
            collision_mode=collision_mode,
            sim_core=sim_core,
            name=name,
            user_data=user_data,
            mass=mass,
            controller=controller,
            notify_spawn=notify_spawn,
            pickable=pickable,
        )

        agent._checkpoint_spawn_params = AgentSpawnParams(
            name=agent.name,
            visual_shape=copy.deepcopy(visual_shape),
            collision_shape=copy.deepcopy(collision_shape),
            initial_pose=copy.deepcopy(pose),
            mass=agent.mass,
            pickable=agent.pickable,
            collision_mode=agent.collision_mode,
            user_data=dict(agent.user_data),
            motion_mode=agent._motion_mode,
            use_fixed_base=agent.use_fixed_base,
            controller=copy.deepcopy(agent.controller._construction_config()),
        )

        if notify_spawn:
            agent._record_checkpoint_spawn(sim_core)

        return agent

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        pose: Pose = None,
        mass: Optional[float] = None,
        motion_mode: Union[MotionMode, str] = MotionMode(_AGT_D["motion_mode"]),
        use_fixed_base: bool = _AGT_D["use_fixed_base"],
        collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
        sim_core=None,
        name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
        ik_params: Optional[IKParams] = None,
        joint_tolerance: Optional[Union[float, list, dict]] = None,
        controller: Optional[Union[str, Dict[str, Any], "ControllerParams", "Controller"]] = None,
        notify_spawn: bool = True,
        global_scaling: float = 1.0,
        pickable: bool = False,
    ) -> "Agent":
        """
        Create a URDF-based Agent (with joints).

        Args:
            urdf_path: Robot model name (e.g. ``"panda"``) or path to URDF file.
                Names are resolved via :func:`~pybullet_fleet.robot_models.resolve_model`
            pose: Initial Pose (position and orientation). Defaults to origin
            mass: Mass override.
                - None (default): Use URDF file's mass values as-is
                - 0.0: Override all links to mass=0 for kinematic control (no physics)
            motion_mode: MotionMode.OMNIDIRECTIONAL or MotionMode.DIFFERENTIAL
            use_fixed_base: If True, robot base is fixed in space
            global_scaling: Uniform scale factor applied when loading.
            pickable: Whether the agent may be attached to another body
                (default: False -- a robot is usually the one doing the
                picking). ``attach_object()`` refuses a non-pickable body.
            sim_core: Reference to simulation core
            ik_params: IK solver configuration (default: ``IKParams()``).
            joint_tolerance: Default joint tolerance for convergence.
                See :attr:`joint_tolerance` property for details.
            controller: Controller selector / config.  See :meth:`Agent.__init__`
                for accepted forms (``None`` / ``str`` / ``dict`` /
                :class:`~pybullet_fleet.controller_params.ControllerParams`).

        Returns:
            Agent instance

        Note:
            - mass=None (default): Uses URDF file's mass values for physics simulation
            - mass=0.0: Override all links to mass=0 for kinematic control (no physics)

        Example::

            robot = Agent.from_urdf(urdf_path="arm_robot.urdf")
            robot = Agent.from_urdf(urdf_path="panda", use_fixed_base=True)
            robot = Agent.from_urdf(urdf_path="arm_robot.urdf", mass=0.0)
        """
        # Shared with SimObject.from_urdf(): the load, the mass resolution and
        # the URDF's own name are the same work either way, and the two had
        # drifted -- only one of them totalled the links' mass or turned an
        # unreadable file into a FileNotFoundError.
        loaded = load_urdf_body(
            urdf_path,
            pose,
            mass=mass,
            use_fixed_base=use_fixed_base,
            global_scaling=global_scaling,
            physics_client_id=sim_core.client if sim_core is not None else 0,
        )
        # The resolved path comes back with the body: resolving again would
        # repeat an uncached robot_descriptions scan for an auto-discovered
        # name, and a second resolution that disagreed would leave the body
        # just loaded orphaned.
        body_id, resolved_mass, urdf_path = loaded.body_id, loaded.mass, loaded.path

        # Create agent instance (SimObject.__init__ handles auto-registration)
        # collision_mode is passed through __init__ -> super().__init__() -> add_object
        agent = cls(
            body_id=body_id,
            urdf_path=urdf_path,
            motion_mode=motion_mode,
            use_fixed_base=use_fixed_base,
            collision_mode=collision_mode,
            sim_core=sim_core,
            name=name,
            user_data=user_data,
            mass=resolved_mass,
            ik_params=ik_params,
            joint_tolerance=joint_tolerance,
            controller=controller,
            notify_spawn=notify_spawn,
            pickable=pickable,
        )

        # Retain effective construction parameters regardless of whether a
        # particular checkpoint profile supports them. Profile-specific
        # validation belongs to that profile, not to normal Agent creation.
        # Keep a configuration descriptor, never the live controller (whose
        # execution state changes as the simulation runs). Controllers without
        # a registered name cannot be reconstructed by this descriptor.
        controller_config = agent.controller._construction_config()
        agent._checkpoint_spawn_params = AgentSpawnParams(
            name=agent.name,
            urdf_path=agent.urdf_path,
            initial_pose=copy.deepcopy(pose) if pose is not None else Pose.from_xyz(0.0, 0.0, 0.0),
            mass=agent.mass,
            pickable=agent.pickable,
            collision_mode=agent.collision_mode,
            user_data=dict(agent.user_data),
            motion_mode=agent._motion_mode,
            use_fixed_base=agent.use_fixed_base,
            controller=copy.deepcopy(controller_config),
            ik_params=ik_params,
            global_scaling=global_scaling,
            joint_tolerance=joint_tolerance,
        )

        if notify_spawn:
            agent._record_checkpoint_spawn(sim_core)

        return agent

    def set_goal_pose(
        self,
        goal: Pose,
        direction: Union[MovementDirection, str, None] = None,
    ) -> bool:
        """
        Set goal pose for the robot to move to.

        Note: This is ignored if the robot has a fixed base.

        Args:
            goal: Target Pose (compatible with ROS2 geometry_msgs/Pose)
            direction: Movement direction override, or ``None`` to use
                the controller's ``default_direction``.

        Returns:
            Whether the goal was taken; see :meth:`set_path`.
        """
        # Single goal is just a path with one waypoint
        return self.set_path([goal], direction=direction)

    def brake(self) -> bool:
        """Come to rest along the current direction of travel at the configured deceleration.

        A deceleration-limited counterpart of :meth:`stop`, which zeroes the
        velocity at once (an emergency stop). The agent keeps its heading and
        its line of motion and stops at the braking point,
        ``|v|^2 / (2 * decel)`` ahead, decelerating at exactly the configured
        rate along that direction (the same projected limits a trajectory
        uses). The remaining path and any queued waypoints are dropped; a
        velocity command (the controller's ``set_velocity``) is ended the same way.

        The action queue is left alone. A :class:`~pybullet_fleet.action.MoveAction`
        still executing completes once the agent is at rest; to abandon it
        instead, call :meth:`clear_actions` first. A new path in another
        direction is refused until the agent has stopped (see
        :meth:`set_path`), so a caller that retries temporary refusals -- as
        ``MoveAction`` does -- simply starts once the braking ends.

        Returns:
            ``True`` when a braking trajectory was set. ``False`` at rest, for
            a fixed-base robot, and for controllers that do not continue a
            motion into a new path (differential drive, which restarts every
            path from rest; see issue #78) -- use :meth:`stop` there.
        """
        from pybullet_fleet._replan import braking_position

        if self.use_fixed_base:
            return False
        controller = self._batch_controller
        if controller is not None:
            # Hold the batch lock across reading the velocity and setting the
            # path (it is re-entrant), so a concurrent step cannot change the
            # motion the braking point was computed for.
            with controller._state_lock:
                return self._brake_with(controller, braking_position)
        if not self._controllers:
            return False
        return self._brake_with(self._controllers[0], braking_position)

    def _brake_with(self, controller, braking_position) -> bool:
        velocity = controller.continuation_velocity(self)
        if velocity is None:
            return False
        pose = self.get_pose()
        # The limits the path will be judged and built with: the per-agent
        # controller's own, or the agent's for a batch row.
        params = self.controller_params if controller is self._batch_controller else controller.params
        target = braking_position(params, pose.position, pose.orientation, velocity)
        if target is None:
            return False
        goal = Pose(position=[float(c) for c in target], orientation=list(pose.orientation))
        accepted = self.set_path([goal], auto_approach=False, final_orientation_align=False)
        if accepted is False:
            self._log.warning(f"Braking path refused: {self.path_rejection([goal])}")
        return accepted is not False

    def path_rejection(self, path: List[Pose]) -> "Optional[PathRejection]":
        """Why :meth:`set_path` would refuse ``path`` right now, or ``None``.

        A :class:`~pybullet_fleet._replan.PathRejection`: ``reason`` says why,
        ``temporary`` whether it lasts only until the current motion ends (a
        fixed-base robot and an empty path are permanent).

        A moving omni agent continues its motion into a new path, so it can
        only take one whose straight-line trajectory from its current
        velocity exists within its configured limits; at rest every path is
        taken. Asking first lets a caller report a refusal before acting.
        """
        from pybullet_fleet._replan import PathRejection

        permanent = self._permanent_path_rejection(path)
        if permanent is not None:
            return permanent
        reason = None
        if self._batch_controller is not None:
            reason = self._batch_controller.path_rejection(self, path)
        elif self._controllers:
            reason = self._controllers[0].path_rejection(self, path)
        return PathRejection(reason, temporary=True) if reason is not None else None

    def _permanent_path_rejection(self, path: List[Pose]) -> "Optional[PathRejection]":
        """The refusals that do not depend on the motion: a fixed base, an empty path."""
        from pybullet_fleet._replan import PathRejection

        if self.use_fixed_base:
            return PathRejection("fixed-base robot", temporary=False)
        if len(path) == 0:
            return PathRejection("empty path", temporary=False)
        return None

    def _make_default_controller(self):
        """Build the default Omni/Diff controller from ``self._motion_mode``.

        Used by ``__init__`` and ``set_motion_mode`` so the mapping
        ``motion_mode -> controller class`` lives in one place.
        """
        from pybullet_fleet.controller import DifferentialController, OmniController

        if self._motion_mode == MotionMode.OMNIDIRECTIONAL:
            return OmniController(self.controller_params)
        if self._motion_mode == MotionMode.DIFFERENTIAL:
            return DifferentialController(self.controller_params)
        return None

    def _init_controller_chain(self, controller) -> None:
        """Wire the controller chain from the unified ``controller=`` value.

        Handles the ``str`` / ``dict`` / ``list`` forms of
        :attr:`AgentSpawnParams.controller`:

        * :class:`KinematicController` entries (``omni``, ``differential``, or a
          custom kinematic controller via ``class:``) **replace** the base
          controller at index 0.
        * High-level entries (``patrol``, ``random_walk``, or any non-kinematic
          :class:`Controller` via ``class:``) are **appended** on top of the
          base — matching :meth:`add_controller`, so they set goals that the
          base navigation controller then executes.
        * If no base controller is supplied, one is auto-created from
          ``motion_mode`` (so a lone ``patrol`` entry still moves).

        Every entry accepts both a registry name (``type:``) and a dotted import
        path (``class:``).
        """
        from pybullet_fleet.controller import KinematicController
        from pybullet_fleet.controller_params import ControllerParams

        # Normalize to a list of entries (each a str or dict).
        if controller is None:
            entries: List[Any] = []
        elif isinstance(controller, (list, tuple)):
            entries = list(controller)
        else:
            entries = [controller]

        base_ctrl = None
        high_level: List[Any] = []
        seed_cfg: Dict[str, Any] = {}

        for entry in entries:
            ctrl = self._make_controller_from_entry(entry)
            entry_cfg = {} if isinstance(entry, str) else dict(entry)
            if ctrl is None:
                # Entry names neither type nor class — treat its keys as base
                # ControllerParams (e.g. ``controller={"max_linear_vel": 0.5}``).
                if not seed_cfg:
                    seed_cfg = entry_cfg
                continue
            if isinstance(ctrl, KinematicController):
                base_ctrl = ctrl
                seed_cfg = entry_cfg
            elif not seed_cfg:
                # First high-level entry seeds the base's kinematic limits
                # (e.g. a patrol entry's ``max_linear_vel``).
                seed_cfg = entry_cfg

            if not isinstance(ctrl, KinematicController):
                high_level.append(ctrl)

        # controller_params is the single source of truth for kinematic limits.
        # Prefer the base controller's own params; otherwise derive from the
        # seed config so the auto-created base picks up any limits given on a
        # high-level entry.
        base_params = getattr(base_ctrl, "params", None)
        if base_ctrl is not None and isinstance(base_params, ControllerParams):
            self.controller_params = base_params
        else:
            self.controller_params = ControllerParams.from_dict(seed_cfg)

        self.set_controller(base_ctrl if base_ctrl is not None else self._make_default_controller())
        for hlc in high_level:
            self.add_controller(hlc)

    def _make_controller_from_entry(self, entry):
        """Resolve one ``controller=`` entry to a :class:`Controller` instance.

        *entry* is a registry-name string, or a flat dict carrying either a
        ``type:`` (registry name) or ``class:`` (dotted import path) key plus
        controller-specific config.  Returns ``None`` when the entry names
        neither — the caller then falls back to the motion-mode default.

        Raises:
            ValueError: If ``type`` names a batch controller (those must be
                driven through :class:`~pybullet_fleet.agent_manager.AgentManager`).
        """
        from pybullet_fleet.controller import create_controller, create_controller_from_entry
        from pybullet_fleet.controllers.batch_base import BATCH_CONTROLLER_REGISTRY

        if isinstance(entry, str):
            entry = {"type": entry}
        cfg = dict(entry)

        dotted = cfg.pop("class", None)
        if dotted is not None:
            return create_controller_from_entry({"class": dotted, "config": cfg})

        ctrl_type = cfg.get("type")
        if ctrl_type:
            if ctrl_type.lower() in BATCH_CONTROLLER_REGISTRY:
                raise ValueError(
                    f"controller.type={ctrl_type!r} is a batch controller type. "
                    "Batch controllers must be used through AgentManager: "
                    "AgentManager(sim_core=sim, fleet_controller={'type': 'batch_differential'})."
                )
            return create_controller(ctrl_type, cfg)
        return None

    def set_motion_mode(self, mode: Union[MotionMode, str]) -> bool:
        """
        Set motion mode for the robot.

        Replaces the active controller with the matching default
        (OmniController or DifferentialController).  Any custom
        controller previously installed via ``controller.type`` is lost.

        Note: Motion mode cannot be changed while the robot is moving.

        .. deprecated::
            Prefer ``agent.set_controller(create_controller(name, cfg))``
            for runtime controller swaps.  ``set_motion_mode`` is kept for
            backward compatibility and only handles the built-in
            Omni / Differential defaults.

        Args:
            mode: MotionMode.OMNIDIRECTIONAL or MotionMode.DIFFERENTIAL (or string)

        Returns:
            True if mode was changed successfully, False if robot is moving
        """
        if self._is_moving:
            self._log.warning("Cannot change motion mode while robot is moving. Call stop() first.")
            return False

        if isinstance(mode, str):
            mode = MotionMode(mode)
        self._motion_mode = mode
        self.set_controller(self._make_default_controller())
        return True

    def _clear_path_visualization(self):
        """Clear all path visualization debug lines."""
        for debug_id in self._path_debug_ids:
            try:
                p.removeUserDebugItem(debug_id, physicsClientId=self._pid)
            except Exception:  # noqa: B902
                pass  # Ignore errors if item was already removed
        self._path_debug_ids.clear()

    def _visualize_path(self, path: List[Pose]):
        """
        Visualize a path as debug lines in PyBullet.

        Args:
            path: List of waypoints to visualize
        """
        if not self.path_visualize or len(path) < 2:
            return

        # Default color: cyan [0, 1, 1]
        color = self.path_visualize_color if self.path_visualize_color is not None else [0, 1, 1]
        width = self.path_visualize_width

        # Draw lines between consecutive waypoints
        for i in range(len(path) - 1):
            start_pos = path[i].position
            end_pos = path[i + 1].position

            debug_id = p.addUserDebugLine(
                start_pos,
                end_pos,
                lineColorRGB=color,
                lineWidth=width,
                lifeTime=0,  # Permanent until manually removed
                physicsClientId=self._pid,
            )
            self._path_debug_ids.append(debug_id)

    def set_path(
        self,
        path: List[Pose],
        auto_approach: bool = True,
        final_orientation_align: bool = True,
        direction: Union[MovementDirection, str, None] = None,
    ) -> bool:
        """
        Set a path (list of waypoints) for the robot to follow.

        A path set while an omnidirectional agent is moving continues the
        motion: the straight-line trajectory to the first waypoint starts from
        the agent's current velocity (signed along the line, so a goal behind
        is reached by slowing and reversing). When no such trajectory exists
        within the configured limits -- a goal inside the braking distance,
        where the agent is, or off the line of motion -- the path is refused:
        nothing changes, the current motion continues, and this returns
        ``False`` (the reason is logged and available from
        :meth:`path_rejection`). At rest every path is taken.

        Args:
            path: List of Pose waypoints to follow in sequence
            auto_approach: If True, automatically add straight-line approach from current position
                          to first waypoint if distance > threshold (default: True)
            final_orientation_align: If True, add final rotation to match last waypoint's orientation
                                    after reaching the position (default: True)
            direction: Movement direction override, or ``None`` to use the controller's
                      ``default_direction``.  Can be MovementDirection enum or string
                      ("auto", "forward", "backward").
                      Only used in differential drive mode (MotionMode.DIFFERENTIAL).
                      In omnidirectional mode, this parameter is ignored.

        Returns:
            ``True`` when the path was taken; ``False`` for a fixed-base robot,
            an empty path, or a path the controller refused.
        """
        permanent = self._permanent_path_rejection(path)
        if permanent is not None:
            self._log.warning(f"Path refused: {permanent.reason}")
            return False

        # Normalize direction string → enum (None passes through to controller)
        if isinstance(direction, str):
            direction = MovementDirection(direction)

        # Get current position
        current_pose = self.get_pose()
        current_pos = np.array(current_pose.position)
        first_waypoint_pos = np.array(path[0].position)

        # Check if we need to add approach waypoint
        distance_to_first = np.linalg.norm(first_waypoint_pos - current_pos)
        approach_threshold = 0.5  # meters - add approach if farther than this

        final_path = []

        # Add approach waypoint if needed. Not while the agent is under way: it
        # is the current pose, a zero-length first segment, and would stop the
        # agent dead before the new path -- a new goal mid-motion continues the
        # motion instead. Under way is a path in progress (navigation state, not
        # the reported velocity: a batch controller reports the arrival step's
        # average speed until its next flush, after the agent has stopped) or a
        # velocity command still moving it.
        from pybullet_fleet.controller import KinematicController

        base = self._controllers[0] if self._batch_controller is None and self._controllers else None
        under_velocity_command = (
            isinstance(base, KinematicController)
            and base.mode is ControllerMode.VELOCITY
            and bool(np.any(self._current_velocity))
        )
        under_way = self.is_moving or under_velocity_command
        if auto_approach and distance_to_first > approach_threshold and not under_way:
            # Add current position as approach waypoint
            # This creates a path: current_pose -> path[0] -> path[1] -> ...
            final_path.append(current_pose)
            self._log.debug(
                lambda: f"Added approach waypoint at current position (distance to first: {distance_to_first:.2f}m)"
            )

        # Add all path waypoints
        final_path.extend(path)

        # Delegate first, and only call the agent moving once the controller
        # has accepted the path. A controller is free to refuse one -- an
        # unregistered agent, a profile it cannot represent, a path its own
        # validation rejects -- and setting the flag beforehand would leave an
        # idle agent reporting is_moving with no trajectory behind it, and the
        # previous path's visualization already cleared.
        if self._batch_controller is not None:
            # synchronized_set_path() sets _is_moving itself, under the same
            # lock as the array writes. Setting it here instead would race a
            # concurrent step_once() that completes a zero-distance path and
            # clears the flag between the lock releasing and this line.
            accepted = self._batch_controller.synchronized_set_path(
                self,
                final_path,
                final_orientation_align=final_orientation_align,
                direction=direction,
            )
        else:
            accepted = self._controllers[0].set_path(
                self,
                final_path,
                final_orientation_align=final_orientation_align,
                direction=direction,
            )
            if accepted is not False:
                self._is_moving = True
        if accepted is False:
            return False

        # Clear previous path visualization and visualize new path
        self._clear_path_visualization()
        self._visualize_path(final_path)
        return True

    def _reset_pybullet_velocity(self) -> None:
        """Zero residual physics velocity (prevents drift in kinematic mode)."""
        p.resetBaseVelocity(self.body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._pid)

    # ============================================================================
    # Action Queue System
    # ============================================================================

    def add_action(self, action: Action):
        """
        Add an action to the execution queue.

        Actions are executed sequentially in the order they are added.

        Args:
            action: Action instance to add to queue

        Example::

            agent.add_action(MoveAction(path=my_path))
            agent.add_action(WaitAction(duration=5.0))
        """
        self._action_queue.append(action)
        self._log.info(f"Added {action.__class__.__name__} to queue (queue size: {len(self._action_queue)})")

    def add_action_sequence(self, actions: List[Action]):
        """
        Add multiple actions to the execution queue.

        Args:
            actions: List of Action instances to add

        Example::

            agent.add_action_sequence([
                MoveAction(path=path1),
                PickAction(target_object_id=obj.body_id),
                MoveAction(path=path2),
                DropAction(drop_pose=Pose(position=[10, 5, 0]))
            ])
        """
        self._action_queue.extend(actions)
        self._log.info(f"Added {len(actions)} actions to queue (queue size: {len(self._action_queue)})")

    def clear_actions(self):
        """
        Clear all actions from the queue and cancel current action.
        """
        self._action_queue.clear()
        if self._current_action is not None:
            self._current_action.cancel()
            self._current_action = None
        self._log.info("Cleared all actions")

    def get_current_action(self) -> Optional[Action]:
        """
        Get the currently executing action.

        Returns:
            Current Action instance, or None if no action is executing
        """
        return self._current_action

    def get_action_queue_size(self) -> int:
        """
        Get the number of actions waiting in the queue.

        Returns:
            Number of actions in queue
        """
        return len(self._action_queue)

    def is_action_queue_empty(self) -> bool:
        """
        Check if action queue is empty and no action is currently executing.

        Returns:
            True if no actions are queued or executing, False otherwise
        """
        return len(self._action_queue) == 0 and self._current_action is None

    def _update_actions(self, dt: float):
        """
        Update action execution system.

        This is called internally by update() to process the action queue.

        Args:
            dt: Time step (seconds)
        """
        # If no current action, try to get next one from queue
        if self._current_action is None:
            if self._action_queue:
                self._current_action = self._action_queue.pop(0)
                self._log.info(
                    f"Starting {self._current_action.__class__.__name__} " f"(remaining in queue: {len(self._action_queue)})"
                )
                # --- action_started event ---
                if self.sim_core is not None:
                    self.sim_core.events.emit(SimEvents.ACTION_STARTED, agent=self, action=self._current_action)
                if self._has_entity_events():
                    self.events.emit(SimEvents.ACTION_STARTED, action=self._current_action)
            else:
                return  # No actions to execute

        # Execute current action
        is_complete = self._current_action.execute(self, dt)

        if is_complete:
            action_name = self._current_action.__class__.__name__
            status = self._current_action.status

            if status == ActionStatus.COMPLETED:
                self._log.info(f"{action_name} completed successfully")
            elif status == ActionStatus.FAILED:
                self._log.error(f"{action_name} failed: {self._current_action.error_message}")
            elif status == ActionStatus.CANCELLED:
                self._log.warning(f"{action_name} was cancelled")

            # --- action_completed event ---
            finished_action = self._current_action
            if self.sim_core is not None:
                self.sim_core.events.emit(SimEvents.ACTION_COMPLETED, agent=self, action=finished_action, status=status)
            if self._has_entity_events():
                self.events.emit(SimEvents.ACTION_COMPLETED, action=finished_action, status=status)

            # Move to next action
            self._current_action = None

    def _compute_use_kinematic_joints(self) -> bool:
        """Compute whether joints need kinematic interpolation.

        True when:
        - The agent itself is kinematic (mass=0), OR
        - The simulation has physics disabled (sim_core.physics=False),
          meaning p.stepSimulation() is never called so motor control
          (setJointMotorControl2) would have no effect.

        Called once at init; result cached in ``_use_kinematic_joints``.
        """
        if self.is_kinematic:
            return True
        if self.sim_core is not None:
            if not hasattr(self.sim_core, "_params"):
                self._log.warning(
                    "sim_core has no '_params' attribute; " "assuming physics=True (joints will use motor control)"
                )
                return False
            if not self.sim_core._params.physics:
                if not Agent._kinematic_joints_physics_off_logged:
                    Agent._kinematic_joints_physics_off_logged = True
                    self._log.info(
                        "sim_core has physics=False; URDF joints will use kinematic "
                        "interpolation instead of motor control (resetJointState "
                        "instead of setJointMotorControl2)"
                    )
                return True
        return False

    def set_joint_motion_profile(
        self,
        joint: Union[int, str],
        max_velocity: Optional[float] = None,
        max_accel: Optional[float] = None,
        max_decel: Optional[float] = None,
    ) -> None:
        """Override how fast a kinematic joint moves, and how it gets there.

        Without this, a kinematic joint runs at the URDF's
        ``<limit velocity="...">`` and reaches it instantly: the speed is a
        property of the model file, so changing it means editing or
        regenerating the URDF, and ``changeDynamics(maxJointVelocity=...)``
        does not help because ``getJointInfo()`` keeps reporting the original.

        Giving ``max_accel`` turns the motion into a trapezoid -- ramp up,
        cruise, ramp down -- which is what a real lift, hoist or linear axis
        does. A 0.6 m travel at 0.4 m/s takes 1.5 s at a constant speed and
        1.9 s with 1.0 m/s ramps, so the difference is not cosmetic when
        something downstream is timing the move.

        Both are optional and independent. With neither, nothing changes; with
        ``max_velocity`` alone the joint still steps instantly to that speed,
        exactly as before.

        Every argument left at ``None`` keeps what the joint already has, so
        this can be called again to change one of them. A ramp cannot be
        taken off a joint once it has one.

        Can be called while the joint is moving. The speed it is carrying is
        kept and the next step re-plans from it under the new limits, so the
        change is continuous; if the new limits can no longer stop the joint
        at its target it brakes and comes back, as for any other target it
        cannot reach.

        Args:
            joint: Joint index, or joint name as spelled in the URDF.
            max_velocity: Speed cap, replacing the URDF limit. ``None`` keeps
                whatever the joint already uses.
            max_accel: Acceleration, in units/s^2. ``None`` keeps the
                acceleration the joint already has, and means no ramp only
                when it has none.
            max_decel: Deceleration. ``None`` keeps the current one, or
                follows ``max_accel`` when that is given. On its own it
                changes the braking rate of a joint that already ramps.

        Raises:
            KeyError: If *joint* names a joint this body does not have.
            IndexError: If *joint* is an index this body does not have.
            ValueError: If any value given is not positive, or if ``max_decel``
                is given for a joint with no acceleration.
        """
        if isinstance(joint, int):
            # Checked here rather than left to the first use: a negative index
            # would be stored under a key _update_kinematic_joints() never
            # visits, so the profile would be silently ignored, and an
            # out-of-range one would fail much later inside
            # _urdf_joint_velocity().
            if not 0 <= joint < len(self.joint_info):
                raise IndexError(
                    f"joint index {joint} is out of range for body {self.body_id} " f"({len(self.joint_info)} joints)"
                )
            index = joint
        else:
            index = self._joint_index_by_name(joint)
        state = self._joint(index)
        validate_joint_motion_profile(max_velocity, max_accel, max_decel, current_accel=state.max_accel)

        if max_velocity is not None:
            state.max_velocity = float(max_velocity)
        if max_accel is not None:
            # A new acceleration resets the braking rate to match it, unless
            # this call gives one of its own.
            state.max_accel = float(max_accel)
            state.max_decel = float(max_accel)
        if max_decel is not None:
            state.max_decel = float(max_decel)
        # The trajectory goes, because it was solved against the old limits.
        # The speed stays: it is where the joint actually is, not a plan, and
        # dropping it would restart the next trajectory from zero and step
        # straight back up to speed past the acceleration just configured.
        # _update_kinematic_joints() re-plans from it under the new limits on
        # the next step, and brakes if they no longer reach the target.
        state.clear_trajectory()

    def _joint_index_by_name(self, name: str) -> int:
        for index, info in enumerate(self.joint_info):
            if info[1].decode("utf-8") == name:
                return index
        raise KeyError(f"no joint named {name!r} on body {self.body_id}")

    def _urdf_joint_velocity(self, joint_index: int) -> float:
        """The joint's speed cap from the URDF, or the fallback for its type."""
        max_vel = self.joint_info[joint_index][11]
        if max_vel > 0:
            return float(max_vel)
        if self.joint_info[joint_index][2] == p.JOINT_PRISMATIC:
            return self._KINEMATIC_PRISMATIC_FALLBACK_VELOCITY
        return self._KINEMATIC_JOINT_FALLBACK_VELOCITY

    def _update_kinematic_joints(self, dt: float) -> bool:
        """Interpolate joints toward targets for kinematic robots (mass=0).

        Called from update() each step. Mirrors the role of stepSimulation()
        for physics-mode motor control.

        Each joint moves at most ``velocity_limit * dt`` per step, where
        velocity_limit comes from the URDF ``<limit velocity="...">`` attribute.
        Falls back to 2.0 rad/s for revolute or 0.5 m/s for prismatic
        if the URDF limit is 0 or missing.
        :meth:`set_joint_motion_profile` overrides either, and adds an
        acceleration ramp when one is configured.

        Visits every joint and skips those with no target and those that have
        already reached theirs (``abs(diff) < 1e-7``).  A target is **never**
        deleted, so that ``are_joints_at_targets()`` can query it later.

        Returns:
            True if any joint position changed this step, False otherwise.
        """
        if not self._active_joints:
            return False

        step_start, now = self._joint_clock, self._joint_clock + dt
        self._joint_clock = now

        any_moved = False
        for state in self._active_joints:
            target = state.target
            assert target is not None  # _active_joints is kept to those that have one
            diff = target - state.position
            if abs(diff) < 1e-7 and not state.speed:
                # Already at target with nothing to wind down -- skip. A
                # ramped joint still carrying speed is not: a target set to
                # where the joint happens to be is inside its braking
                # distance like any other, so stopping it here would ignore
                # the profile's deceleration and report the motion over while
                # the joint was still moving. It takes the braking path below.
                state.clear_motion()
                continue
            max_vel = state.max_velocity
            if max_vel is None:
                max_vel = self._urdf_joint_velocity(state.index)
            if not state.is_ramped:
                # Constant speed, reached instantly: what every kinematic
                # joint did before motion profiles existed, and what a profile
                # that sets only a speed still gets.
                step = math.copysign(max_vel * dt, diff)
                new_pos = target if abs(step) >= abs(diff) else state.position + step
            else:
                new_pos = self._ramped_joint_position(state, target, step_start, now, max_vel)
            p.resetJointState(self.body_id, state.index, new_pos, physicsClientId=self._pid)
            state.position = new_pos
            any_moved = True
        return any_moved

    def is_joint_moving(self, joint: Union[int, str]) -> bool:
        """Whether a ramped joint is still in motion.

        An action reports complete once the joint is inside its tolerance,
        which a ramped joint reaches while still braking through the last
        fraction of a millimetre. Anything that has to know the joint has
        actually stopped -- a lift releasing its passengers, say -- needs
        this rather than the action's status.

        True while the joint is following a trajectory, while it is braking
        with none to follow, and while it is between the two: changing a
        motion profile voids the trajectory it was solved against but not the
        speed the joint is carrying, and the replacement is not planned until
        the next step. A joint that is only carrying speed is as much in
        motion as one with a plan, which is why this asks about the motion
        rather than about the trajectory.
        """
        index = joint if isinstance(joint, int) else self._joint_index_by_name(joint)
        state = self._joints.get(index)
        return state is not None and state.is_moving

    def _joint(self, index: int) -> KinematicJoint:
        """The state for one joint, created on first use.

        Seeded for every joint of a kinematic body, so this only creates one
        in physics mode, where a target is recorded but no position is
        cached.
        """
        state = self._joints.get(index)
        if state is None:
            state = self._joints[index] = KinematicJoint(index=index)
        return state

    def _refresh_active_joints(self) -> None:
        """Re-index the joints the step loop visits, after a target changed."""
        self._active_joints = [state for state in self._joints.values() if state.target is not None]

    def _ramped_joint_position(
        self,
        state: KinematicJoint,
        target: float,
        step_start: float,
        now: float,
        max_vel: float,
    ) -> float:
        """Where a ramped joint is at ``now``, from a TwoPointInterpolation.

        The same solver the linear controllers use, rather than a trapezoid
        integrated by hand: it handles triangle, trapezoid and degenerate
        exactly, and starts from the speed the joint is already carrying.

        One trajectory is built per target change, not per step, and ``t0`` is
        the start of the step so the first evaluation already covers ``dt`` of
        travel.

        What the solver will not plan -- a target the joint is moving away
        from, or one closer than its braking distance -- goes to
        :meth:`_braking_joint_step` until it will.
        """
        accel = state.max_accel
        decel = state.decel
        assert accel is not None and decel is not None  # is_ramped, checked by the caller

        tpi = state.trajectory if state.trajectory_target == target else None
        if tpi is None:
            # A joint travelling away from its target is braked here rather
            # than by the solver: asked to turn a joint around inside the
            # distance it needs to stop, the solver returns a trajectory that
            # runs away from the target at many times vmax and then jumps back
            # onto it, instead of reporting the request as infeasible.
            heading_away = state.speed * (target - state.position) < 0.0
            tpi = (
                None
                if heading_away
                else try_build_tpi(
                    p0=state.position,
                    pe=target,
                    vmax=max_vel,
                    accel=accel,
                    t0=step_start,
                    v0=state.speed,
                    decel=decel,
                )
            )
            if tpi is None:
                return self._braking_joint_step(state, decel, now - step_start)
            state.trajectory = tpi
            state.trajectory_target = target
            state.braking = False

        if now >= tpi.get_end_time():
            state.clear_motion()
            return target

        position, velocity, _ = tpi.get_point(now)
        if abs(target - position) < 1e-9:
            # Close enough that the remainder is float residue. Snapping here
            # rather than waiting for get_end_time() keeps the invariant that
            # a joint at its target carries no speed and no trajectory.
            state.clear_motion()
            return target
        state.speed = float(velocity)
        return float(position)

    def _braking_joint_step(self, state: KinematicJoint, decel: float, dt: float) -> float:
        """One step of braking for a joint with no trajectory to follow.

        Covers the two states a trapezoid cannot be fitted to: a target that
        has moved inside the distance the joint needs to stop, which the
        solver refuses, and a joint travelling away from its target, which the
        solver mis-solves. Both used to end with the joint at its target one
        step later -- the first because an unplannable request comes back as a
        zero-length trajectory that reads as "already arrived", the second
        because the mis-solved trajectory ends on the target however far it
        wandered first.

        Instead the joint sheds speed at ``decel`` for this step and the
        trajectory is planned again on the next one. Braking alone does not
        close the gap on a target ahead -- the distance left and the distance
        needed shrink at the same rate -- so the joint runs past it, and from
        there every step is braking towards a target behind until the joint is
        at rest, which the solver can plan from. It takes |v| / decel seconds,
        and the joint never moves faster than ``decel`` allows.
        """
        # Only as much of the step as the joint is still moving for. Averaging
        # the speeds over the whole dt when it comes to rest part-way through
        # would carry it past the distance decel allows.
        moving_for = min(dt, abs(state.speed) / decel)
        slowed = state.speed - math.copysign(decel * moving_for, state.speed)
        if slowed * state.speed <= 0.0:
            slowed = 0.0  # braked through rest inside this step
        position = state.position + 0.5 * (state.speed + slowed) * moving_for
        state.speed = slowed
        state.trajectory = None
        state.trajectory_target = None
        # Keeps is_moving true even on the final step, where the speed reaches
        # exactly zero but the joint is not at its target.
        state.braking = True
        return float(position)

    def update(self, dt: float) -> bool:
        """
        Update robot position towards goal with velocity/acceleration constraints.

        Supports two motion modes:
        - MotionMode.OMNIDIRECTIONAL: Move in any direction without rotating first (smooth trajectories)
        - MotionMode.DIFFERENTIAL: Rotate to face target, then move forward (smooth trajectories)

        Also processes action queue for high-level task execution.

        Note: This does nothing if the robot has a fixed base.

        Args:
            dt: Time step (seconds)

        Returns:
            True if the robot moved (position, orientation, or joint state changed),
            False otherwise
        """
        # Per-entity pre_update event (inline check — ~30ns when no handlers)
        if self._events is not None:
            self._events.emit(SimEvents.PRE_UPDATE, dt=dt)

        # Process action queue first (actions may set goals/paths)
        self._update_actions(dt)

        # Kinematic joint interpolation (mirrors stepSimulation role for mass=0 URDF robots
        # or when physics is disabled in sim_core)
        joints_moved = False
        if self.is_urdf_robot() and self._use_kinematic_joints:
            joints_moved = self._update_kinematic_joints(dt)

        moved = False
        if not self.use_fixed_base:
            if self._batch_controller is not None:
                # Movement is driven by batch_advance() in Phase 1; skip
                # per-agent compute() to avoid double-updating pose.
                pass
            elif self._controllers:
                # Execute controllers in reversed order (high-level first)
                for ctrl in reversed(self._controllers):
                    ctrl_moved = ctrl.compute(self, dt)
                    moved = moved or ctrl_moved
            elif self._is_moving:
                self._log.warning("No controller set — movement ignored. Use set_controller().")
                moved = False

        # Always update attached objects for URDF robots AFTER base motion
        # so that objects attached to links track correctly both when
        # the base moves and when only arm joints move.
        if self.is_urdf_robot() and self.attached_objects:
            self.update_attached_objects_kinematics()

        result = moved or joints_moved

        # Per-agent plugins (battery, sensors, etc.)
        self._update_plugins(dt)

        # Per-entity post_update event (inline check — ~30ns when no handlers)
        if self._events is not None:
            self._events.emit(SimEvents.POST_UPDATE, dt=dt, moved=result)

        # Return True if any geometry changed (base moved OR joints moved)
        # so that sim_core adds this agent to _moved_this_step for collision checks.
        return result

    def get_velocity(self) -> np.ndarray:
        """
        Get current velocity vector.

        Returns:
            Velocity [vx, vy, vz] in m/s
        """
        return self._current_velocity.copy()

    def restore_motion_state(
        self, pose: Pose, linear_velocity: Sequence[float], angular_velocity: float, *, moving: bool
    ) -> None:
        """Restore Agent-owned motion fields; restore controller execution separately."""
        velocity = np.asarray(linear_velocity, dtype=float)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)) or not math.isfinite(angular_velocity):
            raise ValueError("Motion restore requires finite linear and angular velocity")
        if type(moving) is not bool:
            raise ValueError("Motion restore requires a boolean moving flag")
        self.set_pose(pose, preserve_velocity=False)
        self._current_velocity[:] = velocity
        self._current_angular_velocity = angular_velocity
        self._is_moving = moving

    def set_controller(self, controller: Optional[Any] = None) -> None:
        """Set or replace the base (index-0) movement controller.

        When a controller is set, :meth:`update` delegates movement to
        ``controller.compute(agent, dt)``.

        A controller that carries its own
        :class:`~pybullet_fleet.controller_params.ControllerParams` also
        becomes this agent's :attr:`controller_params`, which is what
        ``Agent.__init__`` does when the controller is passed at construction.
        Without that the two diverge silently: the controller moves the agent
        by its own limits while ``controller_params`` -- the single source of
        truth that :attr:`max_linear_vel` and its siblings delegate to, and
        that a batch controller builds its trajectory from -- still holds
        whatever was there before, usually the defaults.

        Note: A default controller is always assigned in ``from_params()``.
        Setting ``None`` disables all movement and leaves
        ``controller_params`` as it is, since there are then no limits to
        take.

        Args:
            controller: A :class:`~pybullet_fleet.controller.Controller`
                instance, or ``None`` to disable movement.
        """
        from pybullet_fleet.controller_params import ControllerParams

        if controller is None:
            self._controllers.clear()
            return
        if self._controllers:
            self._controllers[0] = controller
        else:
            self._controllers.append(controller)
        params = getattr(controller, "params", None)
        if isinstance(params, ControllerParams):
            self.controller_params = params

    def add_controller(self, controller) -> None:
        """Append a high-level controller to the chain.

        High-level controllers (patrol, random walk) are executed *before*
        the base navigation controller in :meth:`update`.

        Args:
            controller: A :class:`~pybullet_fleet.controller.Controller` instance.
        """
        self._controllers.append(controller)

    def remove_controller(self, controller_or_cls) -> None:
        """Remove a non-base controller by instance or class.

        The base controller (index 0) cannot be removed via this method.
        Use :meth:`set_controller` to replace it.

        When a **class** is given, the first instance of that class
        (excluding the base controller) is removed.  When an **instance**
        is given, that exact instance is removed.

        Args:
            controller_or_cls: A Controller instance or subclass.
        """
        if isinstance(controller_or_cls, type):
            for i, c in enumerate(self._controllers[1:], start=1):
                if isinstance(c, controller_or_cls):
                    self._controllers.pop(i)
                    return
        else:
            if controller_or_cls in self._controllers[1:]:
                self._controllers.remove(controller_or_cls)

    def get_controller_by_type(self, cls: type) -> Optional[Any]:
        """Return the first controller matching *cls*, or ``None``.

        Searches the entire chain (including the base controller).

        Args:
            cls: Controller class to search for.

        Returns:
            The first matching controller instance, or ``None``.
        """
        for c in self._controllers:
            if isinstance(c, cls):
                return c
        return None

    @property
    def controllers(self) -> List[Any]:
        """Read-only list of active controllers (copy)."""
        return list(self._controllers)

    @property
    def controller(self):
        """Return the base (index-0) controller, or ``None`` if the chain is empty."""
        return self._controllers[0] if self._controllers else None

    @property
    def _controller(self):
        """Backward-compatible alias for the base controller."""
        return self._controllers[0] if self._controllers else None

    @_controller.setter
    def _controller(self, value):
        """Backward-compatible setter — delegates to set_controller()."""
        self.set_controller(value)

    def stop(self):
        """Stop robot movement and clear goal and path."""
        self._is_moving = False
        self._current_velocity[:] = 0.0
        self._current_angular_velocity = 0.0
        self._reset_pybullet_velocity()
        self._clear_path_visualization()
        if self._batch_controller is not None:
            # Cancel the in-progress batch path so batch_advance() goes IDLE.
            self._batch_controller.synchronized_cancel_path(self)
        elif self._controllers:
            # Notify base controller to reset all internal state (velocity, pose, path)
            self._controllers[0].on_stop(self)

    # ========================================
    # URDF-specific methods (joint control)
    # ========================================

    def is_urdf_robot(self) -> bool:
        """Check if this robot is URDF-based (has joints)."""
        return self.urdf_path is not None

    def get_num_joints(self) -> int:
        """Get number of joints (0 for mesh robots)."""
        return len(self.joint_info)

    def get_joint_state(self, joint_index: int) -> Tuple[float, float]:
        """
        Get joint state (position and velocity).

        For kinematic robots (mass=0), returns the cached position and
        velocity=0.0, avoiding a PyBullet C call.  For physics robots
        the value is read from ``p.getJointState()``.

        Args:
            joint_index: Joint index (0-based)

        Returns:
            (position, velocity) tuple

        Example::

            pos, vel = robot.get_joint_state(0)
        """
        if not self.is_urdf_robot():
            self._log.warning("get_joint_state() only works for URDF robots")
            return (0.0, 0.0)

        if self._use_kinematic_joints and joint_index in self._joints:
            return (self._joints[joint_index].position, 0.0)

        joint_state = p.getJointState(self.body_id, joint_index, physicsClientId=self._pid)
        return (joint_state[0], joint_state[1])  # position, velocity

    def get_joint_state_by_name(self, joint_name: str) -> tuple:
        """
        Get (position, velocity) for a single joint by name.
        Returns (position, velocity) or (0.0, 0.0) if not found.
        """
        idx = resolve_joint_index(self.body_id, joint_name)
        if idx == -1:
            self._log.warning(f"Joint name '{joint_name}' not found.")
            return (0.0, 0.0)
        return self.get_joint_state(idx)

    def capture_kinematic_joint_execution(self) -> list[dict]:
        """Capture every joint's position, command, motion limits and execution.

        A solved trajectory is represented by its primitive inputs and start
        time relative to the joint clock. Rebuilding it preserves the phase
        reached at this completed-step boundary without serializing the TPI
        instance itself.
        """
        if not self._use_kinematic_joints or not self.joint_info:
            raise ValueError("Kinematic joint checkpoint requires a jointed kinematic URDF robot")
        return [
            self._joints[index].capture_execution(info[1].decode("utf-8"), self._joint_clock)
            for index, info in enumerate(self.joint_info)
        ]

    def restore_kinematic_joint_execution(self, states: list[dict]) -> None:
        """Restore joint commands and execution at a completed-step boundary."""
        if not self._use_kinematic_joints or not isinstance(states, list) or len(states) != len(self.joint_info):
            raise ValueError("Joint checkpoint does not match the kinematic URDF robot")
        parsed = []
        for index, (state, info) in enumerate(zip(states, self.joint_info)):
            if not isinstance(state, dict) or state.get("name") != info[1].decode("utf-8"):
                raise ValueError("Joint checkpoint names or order differ from the kinematic URDF")
            parsed.append(KinematicJoint.from_execution(index, state, self._joint_clock))
        for joint in parsed:
            p.resetJointState(self.body_id, joint.index, joint.position, physicsClientId=self._pid)
            self._joints[joint.index] = joint
        self._refresh_active_joints()

    def get_all_joints_state(self) -> list:
        """
        Return a list of (position, velocity) for all joints.
        Example: [(pos0, vel0), (pos1, vel1), ...]
        """
        if not self.is_urdf_robot():
            self._log.warning("get_all_joints_state() only works for URDF robots")
            return []
        return [self.get_joint_state(i) for i in range(self.get_num_joints())]

    def get_all_joints_state_by_name(self) -> dict:
        """
        Return a dict of {joint_name: (position, velocity)} for all joints.
        Example: {joint_name: (pos, vel), ...}
        """
        if not self.is_urdf_robot():
            self._log.warning("get_all_joints_state_by_name() only works for URDF robots")
            return {}
        joint_names = [self.joint_info[i][1].decode("utf-8") for i in range(self.get_num_joints())]
        result = {}
        for name in joint_names:
            result[name] = self.get_joint_state_by_name(name)
        return result

    def get_joints_state_by_name(self, joint_names: list) -> dict:
        """
        Return a dict of {joint_name: (position, velocity)} for the specified joint_names.
        Example: {joint_name: (pos, vel), ...}
        """
        result = {}
        for name in joint_names:
            idx = resolve_joint_index(self.body_id, name)
            if idx == -1:
                self._log.warning(f"Joint name '{name}' not found.")
                continue
            result[name] = self.get_joint_state(idx)
        return result

    def set_joint_target(self, joint_index: int, target_position: float, max_force: float = 500.0):
        """
        Set target position for a joint (for URDF robots only).

        Args:
            joint_index: Joint index (0-based)
            target_position: Target position (radians for revolute, meters for prismatic)
            max_force: Maximum force to apply

        Example::

            robot.set_joint_target(0, 1.57)  # Move first joint to 90 degrees
        """
        if not self.is_urdf_robot():
            self._log.warning("set_joint_target() only works for URDF robots")
            return

        if joint_index >= len(self.joint_info):
            self._log.warning(f"joint_index {joint_index} out of range (max: {len(self.joint_info)-1})")
            return

        # Skip fixed joints — they cannot be driven
        if self.joint_info[joint_index][2] == p.JOINT_FIXED:
            return

        # Always record for are_joints_at_targets() / PoseAction completion
        state = self._joint(joint_index)
        newly_targeted = state.target is None
        state.target = target_position
        if newly_targeted:
            self._refresh_active_joints()

        if not self._use_kinematic_joints:
            # Physics mode (mass>0 with stepSimulation): motor control
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=max_force,
                physicsClientId=self._pid,
            )

    def set_joint_target_by_name(self, joint_name: str, target_position: float, max_force: float = 500.0):
        """
        Set target position for a joint by joint name.
        """
        joint_index = resolve_joint_index(self.body_id, joint_name)
        if joint_index == -1:
            self._log.warning(f"Joint name '{joint_name}' not found.")
            return
        self.set_joint_target(joint_index, target_position, max_force)

    def set_all_joints_targets(self, target_positions: List[float], max_force: float = 500.0):
        """
        Set target positions for all joints at once.

        Args:
            target_positions: List of target positions for each joint
            max_force: Maximum force to apply

        Example::

            robot.set_all_joints_targets([0.0, 1.57, -1.57, 0.0])
        """
        if not self.is_urdf_robot():
            self._log.warning("set_all_joints_targets() only works for URDF robots")
            return

        if len(target_positions) != len(self.joint_info):
            self._log.warning(f"Expected {len(self.joint_info)} targets, got {len(target_positions)}")
            return

        for i, target in enumerate(target_positions):
            self.set_joint_target(i, target, max_force)

    def set_joints_targets_by_name(self, joint_targets: dict, max_force: float = 500.0):
        """
        Set multiple joint targets by dict {joint_name: target_position} (partial update allowed).
        """
        for joint_name, target in joint_targets.items():
            joint_index = resolve_joint_index(self.body_id, joint_name)
            if joint_index == -1:
                self._log.warning(f"Joint name '{joint_name}' not found.")
                continue
            self.set_joint_target(joint_index, target, max_force)

    def set_joints_targets(self, targets: Union[list, dict], max_force: float = 500.0):
        """
        Set joint targets (accepts both list and dict).

        Args:
            targets: List of target positions for all joints, or dict {joint_name: position}
            max_force: Maximum force to apply

        Example::

            # Using list
            robot.set_joints_targets([0.0, 1.57, -1.57, 0.0])

            # Using dict
            robot.set_joints_targets({"joint1": 1.57, "joint2": -1.57})
        """
        if isinstance(targets, dict):
            self.set_joints_targets_by_name(targets, max_force)
        else:
            self.set_all_joints_targets(targets, max_force)

    def is_joint_at_target(
        self, joint_index: int, target: float, tolerance: Optional[Union[float, list, dict]] = None
    ) -> bool:
        """
        Check if a single joint (by index) is within tolerance of the target position.

        Args:
            joint_index: Joint index (0-based).
            target: Target position (radians for revolute, metres for prismatic).
            tolerance: Position tolerance — ``float``, ``list``, ``dict``,
                or ``None`` (→ ``self.joint_tolerance``).  Resolved via
                ``_resolve_joint_tolerance(tolerance, joint_index)``.
        """
        if not self.is_urdf_robot():
            self._log.warning("is_joint_at_target() only works for URDF robots")
            return False
        if joint_index < 0 or joint_index >= len(self.joint_info):
            self._log.warning(f"joint_index {joint_index} out of range (max: {len(self.joint_info)-1})")
            return False
        tol = self._resolve_joint_tolerance(tolerance, joint_index)
        pos, _ = self.get_joint_state(joint_index)
        return abs(pos - target) <= tol

    def is_joint_at_target_by_name(
        self, joint_name: str, target: float, tolerance: Optional[Union[float, list, dict]] = None
    ) -> bool:
        """
        Check if a single joint (by name) is within tolerance of the target position.

        Args:
            joint_name: Joint name string.
            target: Target position (radians for revolute, metres for prismatic).
            tolerance: Position tolerance — ``float``, ``list``, ``dict``,
                or ``None`` (→ ``self.joint_tolerance``).  Resolved via
                ``_resolve_joint_tolerance()`` using the joint's index.
        """
        joint_index = resolve_joint_index(self.body_id, joint_name)
        if joint_index == -1:
            return False
        return self.is_joint_at_target(joint_index, target, tolerance)

    @property
    def last_joint_targets(self) -> Dict[int, float]:
        """Read-only copy of last commanded joint targets."""
        return {i: s.target for i, s in self._joints.items() if s.target is not None}

    def are_all_joints_at_targets(
        self, target_positions: Optional[list] = None, tolerance: Optional[Union[float, list, dict]] = None
    ) -> bool:
        """Check if joints are at target positions.

        Args:
            target_positions: List of target positions for each joint.
                If None, uses the last commanded targets.
            tolerance: ``float``, ``list``, ``dict``, or ``None``.
                If None, uses ``self.joint_tolerance``.
                Passed through to ``_resolve_joint_tolerance()`` per joint.

                .. note::

                    A scalar tolerance applies the same value to all joints.
                    For mixed prismatic (metres) / revolute (radians) robots,
                    consider a ``dict`` keyed by joint name or set
                    ``self.joint_tolerance`` to a dict.

        Returns:
            True if all joints are within tolerance of their targets.
        """
        if not self.is_urdf_robot():
            self._log.warning("are_all_joints_at_targets() only works for URDF robots")
            return False

        if target_positions is None:
            targets = self.last_joint_targets
        else:
            targets = {i: t for i, t in enumerate(target_positions)}

        if not targets:
            if target_positions is None:
                self._log.warning("are_all_joints_at_targets() called but no targets have been set")
            return True  # Nothing to check → vacuously true

        for idx, target in targets.items():
            if not self.is_joint_at_target(idx, target, tolerance):
                return False
        return True

    def are_joints_at_targets_by_name(self, joint_targets: dict, tolerance: Optional[Union[float, list, dict]] = None) -> bool:
        """
        Check if all specified joints (by name) are within tolerance.

        Args:
            joint_targets: ``{joint_name: target}``
            tolerance: ``float`` (all), ``dict`` (``{joint_name: tol}``),
                or ``None`` (→ ``self.joint_tolerance``).

                A ``list`` is also accepted but is resolved by **absolute
                joint index** (not by iteration order of *joint_targets*).
                For named-joint subsets, prefer a ``dict`` to avoid
                unexpected index-based fallback.

                For mixed prismatic / revolute chains, use a ``dict`` so
                prismatic joints (metres) can have a tighter tolerance
                than revolute joints (radians).

        Returns:
            True if all joints are within tolerance, False otherwise.
        """
        for joint_name, target in joint_targets.items():
            joint_index = resolve_joint_index(self.body_id, joint_name)
            if joint_index == -1:
                return False
            if not self.is_joint_at_target(joint_index, target, tolerance):
                return False
        return True

    def are_joints_at_targets(
        self,
        targets: Union[list, dict, None] = None,
        tolerance: Optional[Union[float, list, dict]] = None,
    ) -> bool:
        """Check if joints are at target positions.

        Args:
            targets: List of target positions for all joints,
                dict ``{joint_name: position}``, or **None** to use the
                last commanded targets.
            tolerance: Tolerance value(s) — ``float``, ``list``, ``dict``,
                or **None** (→ ``self.joint_tolerance``).

                For mixed prismatic / revolute chains, use a ``dict`` so
                prismatic joints (metres) can have a tighter tolerance
                than revolute joints (radians)::

                    agent.are_joints_at_targets(
                        targets,
                        tolerance={"rail_joint": 0.005, "elbow_to_wrist": 0.05},
                    )

        Returns:
            True if all joints are within tolerance, False otherwise.

        Example::

            # No args — check last commanded targets with agent's default tolerance
            if robot.are_joints_at_targets():
                print("All joints settled")

            # Using list
            if robot.are_joints_at_targets([0.0, 1.57, -1.57, 0.0], tolerance=0.01):
                print("Reached target")

            # Using dict
            if robot.are_joints_at_targets({"joint1": 1.57, "joint2": -1.57}, tolerance=0.01):
                print("Reached target")
        """
        if targets is None:
            return self.are_all_joints_at_targets(None, tolerance)
        if isinstance(targets, dict):
            return self.are_joints_at_targets_by_name(targets, tolerance)
        return self.are_all_joints_at_targets(targets, tolerance)

    # ========================================
    # Inverse Kinematics (IK)
    # ========================================

    def _get_end_effector_link_index(
        self,
        end_effector_link: Union[int, str, None] = None,
    ) -> int:
        """Resolve end-effector link index.

        Args:
            end_effector_link: Link index (int), link name (str), or None.
                If None, returns the last link index (last joint's child link).

        Returns:
            Link index (int). Returns -1 if robot has no joints.
        """
        if end_effector_link is not None:
            return resolve_link_index(self.body_id, end_effector_link)
        if not self.joint_info:
            self._log.warning("_get_end_effector_link_index: robot has no joints")
            return -1
        return len(self.joint_info) - 1

    @property
    def ik_params(self) -> IKParams:
        """Read-only access to the IK solver configuration."""
        return self._ik_params

    def _solve_ik(
        self,
        target_position: List[float],
        target_orientation: Optional[Tuple[float, float, float, float]] = None,
        ee_link_index: Optional[int] = None,
    ) -> List[float]:
        """Solve inverse kinematics for the given end-effector target.

        Uses a multi-seed iterative scheme on top of PyBullet's
        ``calculateInverseKinematics``.  The following seed strategies
        are tried:

        1. **Current joint positions** — works well when the robot is
           already close to the desired configuration.
        2. **Quartile seeds** — joint-range fractions specified by
           ``ik_params.seed_quartiles`` (default 25 %, 50 %, and 75 %),
           covering diverse regions of the joint space.

        For each seed, up to ``ik_params.max_outer_iterations`` refinement
        passes are performed (feeding the previous IK solution back as
        ``restPoses``).  The loop returns early when the FK end-effector
        position is within ``ik_params.convergence_threshold`` of the target.

        Args:
            target_position: Target EE position [x, y, z] in world frame.
            target_orientation: Target EE orientation as quaternion
                [qx, qy, qz, qw].  If None, only position is constrained.
            ee_link_index: Resolved end-effector link index (int).
                If None, auto-detects (last link).

        Returns:
            List of joint angles (one per joint).  Empty list if IK
            cannot be attempted (non-URDF robot, invalid EE link).
        """
        if not self.is_urdf_robot() or not self.joint_info:
            self._log.warning("_solve_ik: only works for URDF robots with joints")
            return []

        ee_index = ee_link_index if ee_link_index is not None else self._get_end_effector_link_index()
        if ee_index < 0:
            self._log.warning("_solve_ik: invalid end-effector link index")
            return []

        num_joints = len(self.joint_info)
        saved_positions = [self.get_joint_state(i)[0] for i in range(num_joints)]
        cfg = self._ik_params

        # Build movable-joint mapping.  PyBullet's calculateInverseKinematics
        # returns values only for movable joints (skips JOINT_FIXED).
        movable_indices = self._cached_movable_indices

        # Build limit arrays for movable joints only.
        # Joints NOT in the IK "free set" are locked at their current
        # position via a tiny range around the saved value.
        _LOCK_EPS = 1e-4  # tiny range to effectively freeze the joint

        # Determine which movable joints the IK solver is allowed to move.
        ik_names = cfg.ik_joint_names
        if ik_names is not None:
            # Resolve names to a set of joint indices for O(1) lookup.
            ik_free_indices: Optional[set] = set()
            for name in ik_names:
                idx = resolve_joint_index(self.body_id, name)
                if idx >= 0:
                    ik_free_indices.add(idx)
                else:
                    self._log.warning("_solve_ik: ik_joint_names entry '%s' not found", name)
        else:
            ik_free_indices = None  # auto-detect mode

        lower_limits: List[float] = []
        upper_limits: List[float] = []
        joint_ranges: List[float] = []
        for idx in movable_indices:
            lo, hi = self.joint_info[idx][8], self.joint_info[idx][9]
            if ik_free_indices is not None:
                # Explicit mode: lock if not in the free set
                should_lock = idx not in ik_free_indices
            else:
                # Auto mode: lock continuous joints (lower >= upper)
                should_lock = lo >= hi
            if should_lock:
                cur = saved_positions[idx]
                lo, hi = cur - _LOCK_EPS, cur + _LOCK_EPS
            lower_limits.append(lo)
            upper_limits.append(hi)
            joint_ranges.append(hi - lo)

        # Seeds in movable-joint space
        movable_saved = [saved_positions[i] for i in movable_indices]
        seed_strategies: List[List[float]] = [list(movable_saved)]
        for q in cfg.seed_quartiles:
            seed_strategies.append([lo + q * r for lo, r in zip(lower_limits, joint_ranges)])

        target_arr = np.array(target_position)
        best_angles: List[float] = []
        best_distance = float("inf")

        for seed in seed_strategies:
            rest_poses = list(seed)
            prev_angles: Optional[List[float]] = None
            for _ in range(cfg.max_outer_iterations):
                ik_kwargs = dict(
                    bodyUniqueId=self.body_id,
                    endEffectorLinkIndex=ee_index,
                    targetPosition=target_position,
                    lowerLimits=lower_limits,
                    upperLimits=upper_limits,
                    jointRanges=joint_ranges,
                    restPoses=rest_poses,
                    maxNumIterations=cfg.max_inner_iterations,
                    residualThreshold=cfg.residual_threshold,
                    physicsClientId=self._pid,
                )
                if target_orientation is not None:
                    ik_kwargs["targetOrientation"] = target_orientation

                joint_angles = list(p.calculateInverseKinematics(**ik_kwargs))

                # Early exit: solution unchanged from previous iteration
                if prev_angles is not None and joint_angles == prev_angles:
                    break

                # Map movable-joint IK output to actual joint indices
                for k, angle in enumerate(joint_angles):
                    p.resetJointState(self.body_id, movable_indices[k], angle, physicsClientId=self._pid)
                link_state = p.getLinkState(self.body_id, ee_index, computeForwardKinematics=1, physicsClientId=self._pid)
                distance = float(np.linalg.norm(np.array(link_state[0]) - target_arr))

                if distance < best_distance:
                    best_distance = distance
                    best_angles = joint_angles

                if distance < cfg.convergence_threshold:
                    break

                prev_angles = joint_angles
                rest_poses = joint_angles

            if best_distance < cfg.convergence_threshold:
                break

        # Restore original joint positions
        for i, pos in enumerate(saved_positions):
            p.resetJointState(self.body_id, i, pos, physicsClientId=self._pid)

        # Map movable-joint result back to full joint list
        full_angles = list(saved_positions)
        for k, angle in enumerate(best_angles):
            full_angles[movable_indices[k]] = angle
        return full_angles

    def _check_ee_pose(
        self,
        ee_link_index: int,
        target_position: List[float],
        target_orientation: Optional[Tuple[float, float, float, float]] = None,
        tolerance: float = 0.02,
        orientation_tolerance: float = 0.15,
    ) -> bool:
        """Core FK check: compare current EE pose against a target.

        Reads the EE link state via forward kinematics and checks whether
        the position (and optionally orientation) is within tolerance.

        This is the shared implementation used by both
        :meth:`_check_ik_reachability` (with temporary joint states) and
        :meth:`are_ee_at_target` (with current joint states).

        Args:
            ee_link_index: End-effector link index.  Returns False if < 0.
            target_position: Desired EE position [x, y, z].
            target_orientation: Desired EE orientation as quaternion
                [qx, qy, qz, qw].  If None, only position is checked.
            tolerance: Maximum Euclidean distance (m) to consider reached.
            orientation_tolerance: Maximum ``1 - |dot|`` between actual and
                target quaternions to consider orientation achieved (default 0.15).

        Returns:
            True if the EE is within *tolerance* of the target.
        """
        if ee_link_index < 0:
            return False

        link_state = p.getLinkState(self.body_id, ee_link_index, computeForwardKinematics=1, physicsClientId=self._pid)
        actual_pos = np.array(link_state[0])
        distance = float(np.linalg.norm(actual_pos - np.array(target_position)))

        if distance > tolerance:
            return False

        if target_orientation is not None:
            actual_orn = np.array(link_state[1])
            target_orn = np.array(target_orientation, dtype=float)
            # Normalize to handle non-unit quaternions from callers
            norm = np.linalg.norm(target_orn)
            if norm > 0:
                target_orn = target_orn / norm
            dot = float(min(abs(np.dot(actual_orn, target_orn)), 1.0))
            if (1.0 - dot) > orientation_tolerance:
                return False

        return True

    def _check_ik_reachability(
        self,
        joint_angles: List[float],
        target_position: List[float],
        ee_link_index: int,
        tolerance: float = 0.02,
        target_orientation: Optional[Tuple[float, float, float, float]] = None,
        orientation_tolerance: float = 0.15,
    ) -> bool:
        """Check whether an IK solution actually reaches the target.

        Temporarily applies ``joint_angles`` via ``resetJointState``,
        delegates to :meth:`_check_ee_pose` for the FK comparison, then
        restores the original joint positions.

        Args:
            joint_angles: IK solution (one angle per joint).
            target_position: Desired EE position [x, y, z].
            ee_link_index: End-effector link index.
            tolerance: Maximum Euclidean distance (m) to consider reachable.
            target_orientation: Desired EE orientation as quaternion
                [qx, qy, qz, qw].  If provided, orientation is also checked.
            orientation_tolerance: Maximum ``1 - |dot|`` between actual and
                target quaternions to consider orientation achieved (default 0.15).

        Returns:
            True if EE position (and optionally orientation) is within tolerance.
        """
        if ee_link_index < 0:
            return False

        # Save current positions
        num_joints = len(self.joint_info)
        saved = [self.get_joint_state(i)[0] for i in range(num_joints)]

        # Apply IK solution temporarily
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.body_id, i, angle, physicsClientId=self._pid)

        result = self._check_ee_pose(ee_link_index, target_position, target_orientation, tolerance, orientation_tolerance)

        # Restore original positions
        for i, pos in enumerate(saved):
            p.resetJointState(self.body_id, i, pos, physicsClientId=self._pid)

        return result

    def move_end_effector(
        self,
        target_position: List[float],
        target_orientation: Optional[Tuple[float, float, float, float]] = None,
        end_effector_link: Union[int, str, None] = None,
        max_force: float = 500.0,
        tolerance: Optional[float] = None,
    ) -> bool:
        """Set end-effector target position via inverse kinematics.

        Solves IK, checks reachability, and calls
        ``set_all_joints_targets()`` internally.  Actual movement happens
        in ``update()`` (fire-and-forget, like ``set_joint_target()``).

        When the solver returns an IK candidate, its joint targets are set
        even when the target is unreachable (best-effort). If no candidate
        is available, no new joint target is issued. The return value tells
        the caller whether the IK solution actually reaches the desired
        position.

        Args:
            target_position: Target EE position [x, y, z] in world frame.
            target_orientation: Target EE orientation as quaternion [qx, qy, qz, qw].
            end_effector_link: End-effector link (int, str, or None for auto-detect).
            max_force: Maximum force for motor control (physics mode).
            tolerance: Euclidean distance (m) for reachability check.
                If None, uses ``ik_params.reachability_tolerance``.

        Returns:
            True if the IK solution places the EE within *tolerance* of
            the target (reachable). False otherwise. A best-effort candidate,
            when available, is still commanded.

        Example::

            reachable = arm.move_end_effector([0.3, 0.0, 0.5])
            if not reachable:
                print("Target may not be fully reachable")
        """
        if not self.is_urdf_robot():
            self._log.warning("move_end_effector() only works for URDF robots")
            return False

        ee_index = self._get_end_effector_link_index(end_effector_link)

        # Disable rendering during IK solve + reachability check to prevent
        # visual flicker from temporary resetJointState calls.
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._pid)
        except Exception:
            pass  # DIRECT mode or already disabled

        try:
            joint_angles = self._solve_ik(target_position, target_orientation, ee_index)
            if not joint_angles:
                return False

            tol = tolerance if tolerance is not None else self._ik_params.reachability_tolerance
            reachable = self._check_ik_reachability(
                joint_angles,
                target_position,
                ee_index,
                tol,
                target_orientation=target_orientation,
            )
        finally:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._pid)
            except Exception:
                pass

        self.set_all_joints_targets(joint_angles, max_force=max_force)
        return reachable

    def are_ee_at_target(
        self,
        target_position: List[float],
        target_orientation: Optional[Tuple[float, float, float, float]] = None,
        end_effector_link: Union[int, str, None] = None,
        tolerance: float = 0.02,
        orientation_tolerance: float = 0.15,
    ) -> bool:
        """Check whether the end-effector is near the target pose.

        Resolves the EE link and delegates to :meth:`_check_ee_pose`.

        Args:
            target_position: Target EE position [x, y, z] in world frame.
            target_orientation: Target EE orientation as quaternion
                [qx, qy, qz, qw].  If None, only position is checked.
            end_effector_link: Link index, name, or None (auto-detect).
            tolerance: Maximum Euclidean distance (m) to consider reached.
            orientation_tolerance: Maximum ``1 - |dot|`` between actual and
                target quaternions (default 0.15).

        Returns:
            True if the EE is within *tolerance* of the target.
        """
        if not self.is_urdf_robot():
            self._log.warning("are_ee_at_target() only works for URDF robots")
            return False

        ee_index = self._get_end_effector_link_index(end_effector_link)
        return self._check_ee_pose(ee_index, target_position, target_orientation, tolerance, orientation_tolerance)

    def update_attached_objects_kinematics(self):
        """
        For URDF robots: update attached objects' pose to follow the parent link (kinematics attach).
        This is called every step for kinematic (mass=0) attached objects.
        """
        for obj in self.attached_objects:
            # Only update if attached to a link (not base) and no constraint (mass=0)
            if obj._attached_link_index >= 0 and getattr(obj, "_constraint_id", None) is None:
                # Get parent link's world coordinates
                link_state = p.getLinkState(
                    self.body_id, obj._attached_link_index, computeForwardKinematics=1, physicsClientId=self._pid
                )
                parent_pos, parent_orn = link_state[0], link_state[1]
                # Apply relative offset
                new_pos, new_orn = p.multiplyTransforms(
                    parent_pos, parent_orn, obj._attach_offset.position, obj._attach_offset.orientation
                )
                obj.set_pose_raw(new_pos, new_orn)

    def __repr__(self):
        pose = self.get_pose()
        fixed_str = " [FIXED]" if self.use_fixed_base else ""
        urdf_str = " [URDF]" if self.is_urdf_robot() else " [MESH]"
        return (
            f"Agent(id={self.body_id}{fixed_str}{urdf_str}, "
            f"pos={pose.position[:2]}, "
            f"vel={self._current_velocity[:2].tolist()}, "
            f"moving={self._is_moving})"
        )
