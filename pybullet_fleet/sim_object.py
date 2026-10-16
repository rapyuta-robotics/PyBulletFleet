"""
sim_object.py
Base class for simulation objects with attachment support.
"""

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import logging

import pybullet as p

from .geometry import Pose
from .logging_utils import get_lazy_logger, get_named_lazy_logger
from .types import CollisionMode
from .tools import resolve_link_index
from pybullet_fleet._defaults import SIM_OBJECT as _OBJ_D, SHAPE as _SHP_D
from pybullet_fleet.config_utils import forward_spawn_params as _forward_spawn_params

if TYPE_CHECKING:
    from pybullet_fleet.events import EventBus

# Standard logger for info/warning/error
logger = logging.getLogger(__name__)

# Lazy logger for debug messages (avoids expensive string formatting when disabled)
lazy_logger = get_lazy_logger(__name__)

# Shapes supported by SimObject spawning. Recording currently supports only
# the primitive subset; mesh construction needs separate checkpoint handling.
PRIMITIVE_SHAPE_TYPES = frozenset({"box", "sphere", "cylinder"})
SUPPORTED_SHAPE_TYPES = PRIMITIVE_SHAPE_TYPES | {"mesh"}


@dataclass
class ShapeParams:
    """
    Parameters for visual or collision shape.

    Similar to URDF's <visual> and <collision> tags, this allows full control
    over shape type, geometry, appearance, and frame offset.

    Attributes:
        shape_type: Shape type - "box", "sphere", "cylinder", "mesh", or None (no shape)
        mesh_path: Path to mesh file (required if shape_type="mesh")
        mesh_scale: Mesh scaling [sx, sy, sz] (for mesh shapes)
        half_extents: Box half extents [hx, hy, hz] (for box shapes)
        radius: Sphere/cylinder radius (for sphere/cylinder shapes)
        height: Cylinder height (for cylinder shapes)
        rgba_color: RGBA color [r, g, b, a] (for visual shapes, ignored for collision)
        frame_pose: Shape offset as Pose object (relative to body frame)

    Example::

        # Box visual
        visual = ShapeParams(
            shape_type="box",
            half_extents=[0.5, 0.3, 0.2],
            rgba_color=[0.8, 0.2, 0.2, 1.0]
        )

        # Mesh collision
        collision = ShapeParams(
            shape_type="mesh",
            mesh_path="collision.obj",
            mesh_scale=[1.0, 1.0, 1.0]
        )

        # Sphere with offset
        sphere = ShapeParams(
            shape_type="sphere",
            radius=0.5,
            rgba_color=[0.0, 1.0, 0.0, 1.0],
            frame_pose=Pose.from_xyz(0, 0, 0.5)
        )
    """

    shape_type: Optional[str] = None  # "box", "sphere", "cylinder", "mesh", or None
    mesh_path: Optional[str] = None
    mesh_scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    half_extents: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    radius: float = _SHP_D["radius"]
    height: float = _SHP_D["height"]
    rgba_color: Optional[List[float]] = field(default_factory=lambda: list(_SHP_D["rgba_color"]))
    frame_pose: Optional[Pose] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShapeParams":
        """Create ShapeParams from a configuration dictionary.

        Recognised keys match the dataclass fields: ``shape_type``,
        ``mesh_path``, ``mesh_scale``, ``half_extents``, ``radius``,
        ``height``, ``rgba_color``.  Missing keys use dataclass defaults.

        Args:
            d: Configuration dictionary.

        Returns:
            ShapeParams instance.
        """
        from pybullet_fleet.config_utils import dataclass_from_dict

        return dataclass_from_dict(cls, d)


class LoadedUrdf(NamedTuple):
    """What both ``from_urdf`` factories need out of a loaded URDF."""

    body_id: int
    mass: Optional[float]
    name: Optional[str]
    path: str


def load_urdf_body(
    urdf_path: str,
    pose: Optional[Pose] = None,
    *,
    mass: Optional[float] = None,
    use_fixed_base: bool = True,
    global_scaling: float = 1.0,
    physics_client_id: int = 0,
) -> LoadedUrdf:
    """Load a URDF and settle the three things both factories need from it.

    Shared by :meth:`SimObject.from_urdf` and :meth:`Agent.from_urdf`, which
    otherwise differ only in what they wrap the body in.

    Args:
        urdf_path: Model name or path, resolved via
            :func:`~pybullet_fleet.robot_models.resolve_model`.
        pose: Where to place the base (default: origin).
        mass: ``0.0`` overrides every link to mass 0 for kinematic use;
            ``None`` keeps the URDF's own values; anything else is taken as
            given.
        use_fixed_base: If True, the base is fixed in space.
        global_scaling: Uniform scale factor.
        physics_client_id: Client to load into.

    Returns:
        A :class:`LoadedUrdf`. The mass is totalled across every link rather
        than read from the base alone, because ``use_fixed_base`` zeroes the
        base: a multi-link URDF would otherwise report mass 0 while its own
        links are still dynamic. The name is the URDF's own robot name, for a
        caller with nothing better. The path is the resolved one, so a caller
        that needs it does not pay for ``resolve_model()`` twice -- for an
        auto-discovered name that is an uncached ``robot_descriptions`` scan,
        and a second resolution that disagreed would orphan the body just
        loaded.

    Raises:
        FileNotFoundError: If the URDF cannot be loaded.
    """
    from pybullet_fleet.robot_models import resolve_model

    resolved_path = resolve_model(urdf_path)
    if pose is None:
        pose = Pose.from_xyz(0.0, 0.0, 0.0)
    position, orientation = pose.as_position_orientation()

    try:
        body_id = p.loadURDF(
            resolved_path,
            position,
            orientation,
            useFixedBase=use_fixed_base,
            globalScaling=global_scaling,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
            physicsClientId=physics_client_id,
        )
    except p.error as exc:
        raise FileNotFoundError(f"Failed to load URDF: {resolved_path}") from exc

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    if mass == 0.0:
        # Every link, not just the base, or the articulated links keep their
        # inertia and gravity still acts on them.
        p.changeDynamics(body_id, -1, mass=0.0, physicsClientId=physics_client_id)
        for joint_idx in range(num_joints):
            p.changeDynamics(body_id, joint_idx, mass=0.0, physicsClientId=physics_client_id)
        resolved_mass: Optional[float] = 0.0
    elif mass is None:
        resolved_mass = p.getDynamicsInfo(body_id, -1, physicsClientId=physics_client_id)[0]
        for joint_idx in range(num_joints):
            resolved_mass += p.getDynamicsInfo(body_id, joint_idx, physicsClientId=physics_client_id)[0]
    else:
        resolved_mass = mass

    body_info = p.getBodyInfo(body_id, physicsClientId=physics_client_id)
    name = body_info[1].decode("utf-8") if body_info[1] else None
    return LoadedUrdf(body_id=body_id, mass=resolved_mass, name=name, path=resolved_path)


@dataclass
class SimObjectSpawnParams:
    """
    Parameters for spawning a SimObject.

    Attributes:
        visual_shape: Visual shape parameters (ShapeParams or None)
        collision_shape: Collision shape parameters (ShapeParams or None)
        urdf_path: Model name or path to a URDF file, as an alternative to
            ``visual_shape`` / ``collision_shape``.  Mutually exclusive with
            them; the URDF wins if both are given.
        initial_pose: Initial Pose (position and orientation) in world coordinates (default: origin)
        mass: Object mass in kg (0.0 for static objects, >0 for dynamic)
        pickable: Whether this object can be picked by robots (default: True)
        name: Optional string name for human-readable identification (default: None).
              Duplicates allowed - multiple objects can share the same name.
              Use for debugging, logging, filtering (e.g., "LeftRobot", "Pallet_A").
              For unique identification, use object_id instead.
        user_data: Optional dictionary for custom metadata (default: empty dict)

    Note:
        Managers calculate positions automatically and may override initial_pose.

    Example::

        # Box visual + sphere collision with name
        params = SimObjectSpawnParams(
            visual_shape=ShapeParams(
                shape_type="box",
                half_extents=[0.5, 0.3, 0.2],
                rgba_color=[1.0, 0.0, 0.0, 1.0]
            ),
            collision_shape=ShapeParams(
                shape_type="sphere",
                radius=0.3
            ),
            initial_pose=Pose.from_xyz(0, 0, 1),
            mass=1.0,
            name="Pallet_A"  # Human-readable name (duplicates allowed)
        )
        obj = SimObject.from_params(params, sim_core)

        # Later: Use object_id for unique lookup
        retrieved_obj = sim_core.get_object_by_id(obj.object_id)

        # Or filter by name (may return multiple objects)
        pallets = [o for o in sim_core.objects if o.name == "Pallet_A"]
    """

    visual_shape: Optional[ShapeParams] = None
    collision_shape: Optional[ShapeParams] = None
    initial_pose: Pose = field(default_factory=lambda: Pose.from_xyz(0.0, 0.0, 0.0))
    mass: float = _OBJ_D["mass"]
    pickable: bool = _OBJ_D["pickable"]
    name: Optional[str] = None  # Human-readable name (duplicates allowed, use object_id for unique lookup)
    visual_frame_pose: Optional[Pose] = None
    collision_frame_pose: Optional[Pose] = None
    collision_mode: CollisionMode = CollisionMode.NORMAL_3D  # Collision detection mode
    user_data: Dict[str, Any] = field(default_factory=dict)  # Custom metadata storage
    # Declared last so the positional order of the existing fields -- and of
    # AgentSpawnParams and ElevatorParams, which inherit them -- is unchanged.
    urdf_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SimObjectSpawnParams":
        """Create SimObjectSpawnParams from a configuration dictionary.

        Accepts both programmatic dicts (with ``ShapeParams`` / ``Pose``
        objects) and raw YAML dicts (with ``pose`` list, ``yaw``, and
        shape sub-dicts).

        YAML-style keys:
            name (str, required): Object name.
            visual_shape (dict | ShapeParams, optional): Visual shape definition.
            collision_shape (dict | ShapeParams, optional): Collision shape.
            urdf_path (str, optional): URDF model name or path, instead of the shape definitions.
            pose (list): ``[x, y, z]`` position.  Alternative to ``initial_pose``.
            yaw (float): Yaw angle in radians (used with ``pose``).
            initial_pose (Pose): Pose object (takes precedence over ``pose``).
            mass (float): Default ``0.0`` (static).
            pickable (bool): Default ``True``.
            collision_mode (str | CollisionMode): Default ``"normal_3d"``.
            user_data (dict): Default ``{}``.

        Args:
            config: Configuration dictionary.

        Returns:
            SimObjectSpawnParams instance.

        Raises:
            ValueError: If ``name`` is missing.
        """
        if "name" not in config:
            raise ValueError(f"SimObject definition missing required 'name' field: {config}")

        # Resolve pose: prefer initial_pose object, fall back to pose+yaw
        initial_pose = config.get("initial_pose")
        if initial_pose is None:
            pose = config.get("pose", [0.0, 0.0, 0.0])
            if not isinstance(pose, (list, tuple)) or len(pose) < 3:
                raise ValueError(f"'pose' must be a list/tuple of at least 3 elements [x, y, z], got: {pose!r}")
            yaw = config.get("yaw", 0.0)
            initial_pose = Pose.from_yaw(pose[0], pose[1], pose[2], yaw)

        # Resolve collision_mode
        collision_mode_value = config.get("collision_mode", _OBJ_D["collision_mode"])
        if isinstance(collision_mode_value, str):
            collision_mode_value = CollisionMode(collision_mode_value)

        # Resolve shape params from raw dicts or pass through ShapeParams instances
        vs = config.get("visual_shape")
        cs = config.get("collision_shape")
        if vs is not None and not isinstance(vs, ShapeParams):
            if not isinstance(vs, dict):
                raise TypeError(f"'visual_shape' must be a dict or ShapeParams, got {type(vs).__name__}: {vs!r}")
            vs = ShapeParams.from_dict(vs)
        if cs is not None and not isinstance(cs, ShapeParams):
            if not isinstance(cs, dict):
                raise TypeError(f"'collision_shape' must be a dict or ShapeParams, got {type(cs).__name__}: {cs!r}")
            cs = ShapeParams.from_dict(cs)

        return cls(
            visual_shape=vs,
            collision_shape=cs,
            urdf_path=config.get("urdf_path"),
            initial_pose=initial_pose,
            mass=config.get("mass", _OBJ_D["mass"]),
            pickable=config.get("pickable", _OBJ_D["pickable"]),
            name=config.get("name"),
            collision_mode=collision_mode_value,
            user_data=config.get("user_data", {}),
        )


@dataclass
class _UrdfSpawnDescriptor(SimObjectSpawnParams):
    """Effective URDF-only options retained for declared recording handlers."""

    use_fixed_base: bool = True
    global_scaling: float = 1.0


class SimObject:
    """
    Base class for simulation objects (robots, pallets, obstacles).
    Supports attachment/detachment and kinematic teleportation.

    Object Identification:
        - object_id: Unique integer ID assigned by sim_core (used for lookups and internal tracking).
                    This is the primary identifier for programmatic access.
        - name: Optional string name for human-readable identification (used for debugging, logging, filtering).
               Multiple objects can share the same name (duplicates allowed).
               Not guaranteed to be unique - use object_id for unique identification.

    Args:
        body_id: PyBullet body ID
        sim_core: Reference to simulation core (optional)
        pickable: Whether this object can be picked by robots (default: True)

    Attributes:
        object_id: Unique integer ID (assigned by sim_core, use for lookups)
        name: Optional string name (duplicates allowed, use for debugging/filtering)
        body_id: PyBullet body ID
        mass: Object mass in kg
        pickable: Whether this object can be picked by robots
        collision_mode: Collision detection mode (NORMAL_3D, NORMAL_2D, STATIC, DISABLED)
        user_data: Dictionary for custom metadata (default: empty dict)
    """

    # SpawnParams class used by config-driven grid spawning
    _spawn_params_cls = SimObjectSpawnParams

    # Subclasses override to True for per-step update() calls in step_once().
    _needs_update: bool = False

    # Class-level shared shapes cache (for optimization)
    _shared_shapes: Dict[str, Tuple[int, int]] = {}

    # --- Auto-registration via __init_subclass__ ---
    # Subclasses that set ``_entity_type_name`` are automatically registered
    # in :data:`~pybullet_fleet.entity_registry.ENTITY_REGISTRY`.
    _entity_type_name: Optional[str] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only register if _entity_type_name is defined directly on this class
        # (not inherited from a parent).
        name = cls.__dict__.get("_entity_type_name")
        if name is not None:
            from pybullet_fleet.entity_registry import register_entity_class

            register_entity_class(name, cls)

    def __init__(
        self,
        body_id: int,
        sim_core=None,
        pickable: bool = True,
        mass: Optional[float] = None,
        collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
        name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ):
        self.body_id = body_id
        self.sim_core = sim_core

        self.attached_objects: List["SimObject"] = []
        self.callbacks: List[dict] = []
        self.pickable = pickable
        self.collision_mode = collision_mode  # Collision detection mode

        # User-extensible data storage (custom metadata)
        self.user_data: Dict[str, Any] = dict(user_data) if user_data else {}

        # Optional name for human-readable identification (debugging, logging, filtering)
        # Note: Unlike object_id, name is NOT unique - multiple objects can share the same name.
        # For programmatic lookup/search, use object_id instead.
        # Examples: "LeftRobot_1", "Pallet_A", "Obstacle_Wall"
        self.name: Optional[str] = name

        # Per-entity EventBus (lazy — created on first .events access)
        self._events: Optional["EventBus"] = None

        # Assign unique ID from sim_core if available
        # Note: object_id is UNIQUE and should be used for programmatic object identification.
        # Use sim_core.get_object_by_id(object_id) for lookups.
        if sim_core is not None and hasattr(sim_core, "_next_object_id"):
            self.object_id = sim_core._next_object_id
            sim_core._next_object_id += 1
        else:
            self.object_id = -1  # No ID if not registered to sim_core

        # Instance-level logger with prefix for identification in multi-object scenarios.
        # Subclasses (Agent) can update the prefix via self._log.set_prefix(...).
        self._log = get_named_lazy_logger(__name__)
        self._update_log_prefix()

        # Attachment state (initialized with default zero offset)
        self._attach_offset: Pose = Pose(position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0])
        self._checkpoint_spawn_params: Optional[SimObjectSpawnParams] = None
        self._constraint_id: Optional[int] = None
        self._attached_to: Optional["SimObject"] = None  # Which object is this attached to
        self._attached_link_index: int = -1  # Which link is this attached to

        # Mass (for optimization - query from PyBullet if not provided)
        if mass is not None:
            self.mass = mass
        else:
            # Query from PyBullet (getDynamicsInfo returns mass at index 0)
            self.mass = p.getDynamicsInfo(self.body_id, -1, physicsClientId=self._pid)[0]

        # Kinematic flag (mass=0 means kinematic/static object)
        self.is_kinematic = self.mass == 0.0

        # Pose caching for performance optimization (also used for movement detection)
        initial_pos, initial_orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._pid)
        self._cached_pose: Pose = Pose.from_pybullet(initial_pos, initial_orn)
        self._cached_pose_sim_time: float = -1.0  # Simulation time when pose was cached

        # Two-phase step buffer: when sim_core._in_step is True, _set_pose_internal()
        # stores `preserve_velocity` here and registers this object's id with
        # sim_core._pending_pose_ids. `_cached_pose` is the source of truth for the
        # pending (pos, orn) write — Phase 2 flush reads it back. Default True so an
        # accidental read before any buffered write still matches caller intent.
        # See docs/design/two-phase-step/spec.md (D2/D3b).
        self._pending_preserve_velocity: bool = True

        # Disable PyBullet physics collision if collision_mode is DISABLED
        if self.collision_mode == CollisionMode.DISABLED:
            self._set_pybullet_collision_enabled(False)
            self._log.debug(f"Disabled PyBullet collision (body {self.body_id})")

        # Auto-register to sim_core if provided
        if sim_core is not None:
            # Centralized registration via add_object() for consistency
            sim_core.add_object(self)

    # ------------------------------------------------------------------
    # Per-step update hook (overridden by Agent, Device subclasses)
    # ------------------------------------------------------------------

    def update(self, dt: float) -> bool:
        """Called every simulation step for objects with ``_needs_update = True``.

        Override in subclasses to implement per-step behavior (e.g. device
        state machines, animated objects).

        Args:
            dt: Time step in seconds.

        Returns:
            ``True`` if the object moved or changed state, ``False`` otherwise.
        """
        return False

    @property
    def _pid(self) -> int:
        """PyBullet physicsClientId (falls back to 0 if no sim_core)."""
        return self.sim_core._client if self.sim_core is not None else 0

    @property
    def events(self) -> "EventBus":
        """Per-entity EventBus. Created on first access (lazy).

        Use in subclasses to hook into entity-specific events::

            class WarehouseRobot(Agent):
                def __init__(self, ...):
                    super().__init__(...)
                    from pybullet_fleet.events import SimEvents

                    self.events.on(SimEvents.COLLISION_STARTED, self._on_bump)
                    self.events.on(SimEvents.ACTION_COMPLETED, self._on_task_done)
        """
        if self._events is None:
            from pybullet_fleet.events import EventBus

            self._events = EventBus()
        return self._events

    def _has_entity_events(self) -> bool:
        """Fast check: True if per-entity EventBus has been created."""
        return self._events is not None

    def set_name(self, name: Optional[str]) -> None:
        """
        Set the object name and update the log prefix accordingly.

        Use this instead of directly setting ``self.name`` to ensure
        the log prefix stays in sync.

        Args:
            name: New human-readable name (None to clear)
        """
        if self.name != name:
            self.name = name
            self._update_log_prefix()

    def _update_log_prefix(self):
        """
        Update the instance logger prefix based on current identification.

        Called automatically during __init__. Subclasses (Agent) should call
        this again if they add extra identification info (e.g. "Agent" class name).

        Format: ``[ClassName:object_id:name]`` or ``[ClassName:object_id]`` if no name.
        """
        cls_name = self.__class__.__name__
        if self.name:
            self._log.set_prefix(f"[{cls_name}:{self.object_id}:{self.name}] ")
        else:
            self._log.set_prefix(f"[{cls_name}:{self.object_id}] ")

    @classmethod
    def create_shared_shapes(
        cls,
        visual_shape: Optional[ShapeParams] = None,
        collision_shape: Optional[ShapeParams] = None,
        physics_client_id: int = 0,
    ) -> Tuple[int, int]:
        """
        Create shared visual and collision shapes for fast object spawning.

        This method caches mesh shapes by their parameters, so multiple objects with
        identical mesh parameters will reuse the same PyBullet shape IDs.

        Note: Primitive shapes (box, sphere, cylinder) are NOT cached, as PyBullet
        handles them very efficiently and caching adds unnecessary overhead.

        Args:
            visual_shape: ShapeParams for visual geometry
            collision_shape: ShapeParams for collision geometry
            physics_client_id: PyBullet physics client ID

        Returns:
            (visual_id, collision_id) tuple

        Example::

            visual_id, collision_id = create_shared_shapes(
                visual_shape=ShapeParams(
                    shape_type="mesh",
                    mesh_path="visual.obj",
                    mesh_scale=[1.0, 1.0, 1.0],
                    rgba_color=[0.8, 0.2, 0.2, 1.0]
                ),
                collision_shape=ShapeParams(
                    shape_type="box",
                    half_extents=[0.5, 0.3, 0.2]
                ),
                physics_client_id=sim_core.client
            )
        """
        # Only cache mesh shapes (primitives are cheap to create)
        visual_needs_cache = visual_shape and visual_shape.shape_type == "mesh"
        collision_needs_cache = collision_shape and collision_shape.shape_type == "mesh"

        # Create cache key only for mesh shapes
        if visual_needs_cache or collision_needs_cache:
            visual_key = cls._shape_params_to_key(visual_shape) if visual_needs_cache else "none"
            collision_key = cls._shape_params_to_key(collision_shape) if collision_needs_cache else "none"
            shape_key = f"v:{visual_key}_c:{collision_key}"

            # Check cache
            if shape_key in cls._shared_shapes:
                return cls._shared_shapes[shape_key]

        # Create visual shape
        visual_id = (
            cls._create_visual_shape(visual_shape, physics_client_id) if visual_shape and visual_shape.shape_type else -1
        )

        # Create collision shape (fall back to -1 if mesh can't be loaded)
        collision_id = -1
        if collision_shape and collision_shape.shape_type:
            try:
                collision_id = cls._create_collision_shape(collision_shape, physics_client_id)
            except Exception:
                logger.warning(
                    "Failed to create collision shape for %s — using visual-only body",
                    collision_shape.mesh_path or collision_shape.shape_type,
                )
                collision_id = -1

        # Cache only if at least one is a mesh shape
        if visual_needs_cache or collision_needs_cache:
            cls._shared_shapes[shape_key] = (visual_id, collision_id)

        return visual_id, collision_id

    @staticmethod
    def _shape_params_to_key(shape: ShapeParams) -> str:
        """Convert ShapeParams to a unique cache key string."""
        frame_pos = shape.frame_pose.position if shape.frame_pose else [0, 0, 0]
        frame_orn = shape.frame_pose.orientation if shape.frame_pose else [0, 0, 0, 1]

        color_key = tuple(shape.rgba_color) if shape.rgba_color is not None else "native"
        return (
            f"{shape.shape_type}_{shape.mesh_path}_{tuple(shape.mesh_scale)}_"
            f"{tuple(shape.half_extents)}_{shape.radius}_{shape.height}_"
            f"{color_key}_{tuple(frame_pos)}_{tuple(frame_orn)}"
        )

    @staticmethod
    def _create_visual_shape(shape: ShapeParams, physics_client_id: int) -> int:
        """Create a PyBullet visual shape from ShapeParams."""
        if shape.shape_type not in SUPPORTED_SHAPE_TYPES:
            raise ValueError(f"Unknown visual shape_type: {shape.shape_type}")
        # Get frame offset
        frame_position = [0.0, 0.0, 0.0]
        frame_orientation = [0.0, 0.0, 0.0, 1.0]
        if shape.frame_pose:
            frame_position, frame_orientation = shape.frame_pose.as_position_orientation()

        if shape.shape_type == "mesh":
            if not shape.mesh_path:
                raise ValueError("mesh_path is required for shape_type='mesh'")
            kwargs: dict = dict(
                shapeType=p.GEOM_MESH,
                fileName=shape.mesh_path,
                meshScale=shape.mesh_scale,
                visualFramePosition=frame_position,
                visualFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
            # Only pass rgbaColor if explicitly set — omitting it lets
            # PyBullet read native colours from DAE/OBJ materials.
            if shape.rgba_color is not None:
                kwargs["rgbaColor"] = shape.rgba_color
            return p.createVisualShape(**kwargs)
        elif shape.shape_type == "box":
            return p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=shape.half_extents,
                rgbaColor=shape.rgba_color or list(_SHP_D["rgba_color"]),
                visualFramePosition=frame_position,
                visualFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
        elif shape.shape_type == "sphere":
            return p.createVisualShape(
                p.GEOM_SPHERE,
                radius=shape.radius,
                rgbaColor=shape.rgba_color or list(_SHP_D["rgba_color"]),
                visualFramePosition=frame_position,
                visualFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
        elif shape.shape_type == "cylinder":
            return p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=shape.radius,
                length=shape.height,
                rgbaColor=shape.rgba_color or list(_SHP_D["rgba_color"]),
                visualFramePosition=frame_position,
                visualFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
        else:
            raise ValueError(f"Unknown visual shape_type: {shape.shape_type}")

    @staticmethod
    def _create_collision_shape(shape: ShapeParams, physics_client_id: int) -> int:
        """Create a PyBullet collision shape from ShapeParams."""
        if shape.shape_type not in SUPPORTED_SHAPE_TYPES:
            raise ValueError(f"Unknown collision shape_type: {shape.shape_type}")
        # Get frame offset
        frame_position = [0.0, 0.0, 0.0]
        frame_orientation = [0.0, 0.0, 0.0, 1.0]
        if shape.frame_pose:
            frame_position, frame_orientation = shape.frame_pose.as_position_orientation()

        if shape.shape_type == "mesh":
            if not shape.mesh_path:
                raise ValueError("mesh_path is required for shape_type='mesh'")
            return p.createCollisionShape(
                p.GEOM_MESH,
                fileName=shape.mesh_path,
                meshScale=shape.mesh_scale,
                collisionFramePosition=frame_position,
                collisionFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
        elif shape.shape_type == "box":
            return p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=shape.half_extents,
                collisionFramePosition=frame_position,
                collisionFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
        elif shape.shape_type == "sphere":
            return p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=shape.radius,
                collisionFramePosition=frame_position,
                collisionFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
        elif shape.shape_type == "cylinder":
            return p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=shape.radius,
                height=shape.height,
                collisionFramePosition=frame_position,
                collisionFrameOrientation=frame_orientation,
                physicsClientId=physics_client_id,
            )
        else:
            raise ValueError(f"Unknown collision shape_type: {shape.shape_type}")

    @classmethod
    def _create_body_from_shapes(
        cls,
        visual_id: int,
        collision_id: int,
        pose: Pose,
        mass: float = 0.0,
        physics_client_id: int = 0,
    ) -> int:
        """
        Create a PyBullet multibody from pre-created visual and collision shapes.

        This is a helper method to reduce code duplication between SimObject and Agent.

        Args:
            visual_id: Visual shape ID from create_shared_shapes()
            collision_id: Collision shape ID from create_shared_shapes()
            pose: Initial Pose (position and orientation)
            mass: Body mass (kg)
            physics_client_id: PyBullet physics client ID

        Returns:
            PyBullet body ID
        """
        position, orientation = pose.as_position_orientation()
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=physics_client_id,
        )
        return body_id

    @classmethod
    def from_mesh(
        cls,
        visual_shape: Optional[ShapeParams] = None,
        collision_shape: Optional[ShapeParams] = None,
        pose: Pose = None,
        mass: float = 0.0,
        pickable: bool = True,
        sim_core=None,
        collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
        name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> "SimObject":
        """
        Create a SimObject with optional visual and collision shapes.

        Args:
            visual_shape: ShapeParams for visual geometry
            collision_shape: ShapeParams for collision geometry
            pose: Initial Pose (position and orientation, default: origin)
            mass: Object mass (0.0 for static)
            pickable: Whether object can be picked by robots
            sim_core: Reference to simulation core
            name: Optional human-readable name (duplicates allowed)

        Returns:
            SimObject instance

        Example::

            # Box visual + sphere collision
            obj = SimObject.from_mesh(
                visual_shape=ShapeParams(
                    shape_type="box",
                    half_extents=[0.5, 0.3, 0.2],
                    rgba_color=[0.8, 0.2, 0.2, 1.0]
                ),
                collision_shape=ShapeParams(
                    shape_type="sphere",
                    radius=0.4
                ),
                pose=Pose.from_xyz(0, 0, 1),
                mass=1.0,
                sim_core=sim_core
            )
        """
        if pose is None:
            pose = Pose.from_xyz(0.0, 0.0, 0.0)

        # Get physics client ID from sim_core
        physics_client_id = sim_core.client if sim_core is not None else 0

        # Use create_shared_shapes() to get or create cached shapes
        visual_id, collision_id = cls.create_shared_shapes(
            visual_shape=visual_shape,
            collision_shape=collision_shape,
            physics_client_id=physics_client_id,
        )

        # Create multibody using helper method
        body_id = cls._create_body_from_shapes(
            visual_id=visual_id,
            collision_id=collision_id,
            pose=pose,
            mass=mass,
            physics_client_id=physics_client_id,
        )

        obj = cls(
            body_id=body_id,
            sim_core=sim_core,
            pickable=pickable,
            mass=mass,
            collision_mode=collision_mode,
            name=name,
            user_data=user_data,
        )
        # Keep the effective factory inputs for declared kinematic checkpoint
        # handlers. Direct body-id construction remains unsupported.
        obj._checkpoint_spawn_params = SimObjectSpawnParams(
            name=name,
            visual_shape=copy.deepcopy(visual_shape),
            collision_shape=copy.deepcopy(collision_shape),
            initial_pose=copy.deepcopy(pose),
            mass=mass,
            pickable=pickable,
            collision_mode=collision_mode,
            user_data=dict(user_data) if user_data else {},
        )
        if sim_core is not None:
            record_spawn = getattr(sim_core, "_record_state_spawn", None)
            if record_spawn is not None:
                record_spawn(obj, obj._checkpoint_spawn_params)
        return obj

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        pose: Optional[Pose] = None,
        sim_core=None,
        mass: Optional[float] = 0.0,
        pickable: bool = False,
        collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
        use_fixed_base: bool = True,
        global_scaling: float = 1.0,
        name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> "SimObject":
        """Load a URDF file and wrap the body in a SimObject.

        The counterpart of :meth:`from_sdf` for single-body URDF models, and
        the non-agent counterpart of :meth:`Agent.from_urdf`.

        Use this for static structure -- walls, fixtures, shelving, fencing -- that has
        a URDF but no behaviour.  Such a body does not belong in the per-step
        update loop: ``SimObject._needs_update`` is ``False``, so
        ``step_once()`` skips it entirely, where an ``Agent`` is visited every
        step to walk an empty action queue, check joints it does not have and
        sweep plugins it does not own.  Measured on a scene of 2376
        fixed-base, jointless, controller-less bodies: 15.98 ms per step as
        Agents against 0.78 ms as SimObjects.

        Without this method, a URDF-defined static body has to be created as
        an Agent, because ``urdf_path`` exists only on ``AgentSpawnParams``.

        Args:
            urdf_path: Model name (e.g. ``"panda"``) or path to a URDF file.
                Resolved via :func:`~pybullet_fleet.robot_models.resolve_model`.
            pose: Initial Pose (default: origin).
            sim_core: Simulation core for registration.
            mass: ``0.0`` (the default) overrides every link to mass 0 for
                kinematic use; ``None`` keeps the URDF's own mass values.
            pickable: Whether the object can be picked up (default: False,
                since scenery is the usual case).
            collision_mode: Collision detection mode.
            use_fixed_base: If True, the base is fixed in space.
            global_scaling: Uniform scale factor.
            name: Human-readable name (default: the URDF's own robot name).
            user_data: Custom metadata.

        Returns:
            SimObject instance.

        Raises:
            FileNotFoundError: If the URDF cannot be loaded.

        Example::

            tile = SimObject.from_urdf(
                "frame_tile.urdf",
                pose=Pose.from_xyz(1.0, 2.0, 0.0),
                sim_core=sim,
                collision_mode=CollisionMode.STATIC,
            )
        """
        loaded = load_urdf_body(
            urdf_path,
            pose,
            mass=mass,
            use_fixed_base=use_fixed_base,
            global_scaling=global_scaling,
            physics_client_id=sim_core.client if sim_core is not None else 0,
        )
        if name is None:
            name = loaded.name

        obj = cls(
            body_id=loaded.body_id,
            sim_core=sim_core,
            pickable=pickable,
            mass=loaded.mass,
            collision_mode=collision_mode,
            name=name,
            user_data=user_data,
        )
        # Keep the complete effective construction for future declared URDF
        # handlers. The built-in kinematic profile still rejects URDF objects.
        obj._checkpoint_spawn_params = _UrdfSpawnDescriptor(
            urdf_path=loaded.path,
            name=name,
            initial_pose=copy.deepcopy(pose) if pose is not None else Pose.from_xyz(0.0, 0.0, 0.0),
            mass=loaded.mass,
            pickable=pickable,
            collision_mode=collision_mode,
            user_data=dict(user_data) if user_data else {},
            use_fixed_base=use_fixed_base,
            global_scaling=global_scaling,
        )
        if sim_core is not None:
            record_spawn = getattr(sim_core, "_record_state_spawn", None)
            if record_spawn is not None:
                record_spawn(obj, obj._checkpoint_spawn_params)
        lazy_logger.info(lambda: f"Loaded SimObject from URDF: {loaded.path}")
        return obj

    @classmethod
    def from_sdf(
        cls,
        sdf_path: str,
        sim_core=None,
        collision_mode: CollisionMode = CollisionMode.NORMAL_3D,
        global_scaling: float = 1.0,
        name_prefix: Optional[str] = None,
        pickable: bool = False,
        use_fixed_base: bool = True,
    ) -> List["SimObject"]:
        """Load an SDF file and wrap each body in a SimObject.

        Uses PyBullet's ``p.loadSDF()`` to load the file and creates a
        SimObject for each body. All objects are auto-registered to
        ``sim_core`` if provided.

        Note:
            PyBullet's SDF loader does NOT support ``<world>`` tags,
            ``<include>`` tags, or ``model://`` URIs. Use for individual
            SDF models (e.g., pybullet_data's kiva_shelf, wsg50_gripper).

        Args:
            sdf_path: Path to SDF file (resolved via resolve_model())
            sim_core: Simulation core for registration
            collision_mode: Collision detection mode for all loaded objects
            global_scaling: Uniform scale factor
            name_prefix: Prefix for names (default: SDF filename stem)
            pickable: Whether objects can be picked up
            use_fixed_base: If True, fix base (mass=0, kinematic)

        Returns:
            List of SimObject instances (one per body in the SDF)

        Example::

            shelves = SimObject.from_sdf(
                'kiva_shelf/model.sdf', sim_core=sim
            )
            for shelf in shelves:
                print(f'{shelf.name}: {shelf.get_pose()}')
        """
        from pathlib import Path as PathLib

        from pybullet_fleet.robot_models import resolve_model

        resolved_path = resolve_model(sdf_path)
        pid = sim_core._client if sim_core is not None else 0

        # Determine name prefix from filename if not provided
        if name_prefix is None:
            name_prefix = PathLib(resolved_path).stem

        def _load_and_wrap() -> List["SimObject"]:
            try:
                body_ids = p.loadSDF(
                    resolved_path,
                    globalScaling=global_scaling,
                    physicsClientId=pid,
                )
            except p.error as exc:
                raise FileNotFoundError(f"Failed to load SDF: {resolved_path}") from exc

            if not body_ids:
                raise RuntimeError(f"SDF loaded 0 bodies from: {resolved_path}")

            objs: List["SimObject"] = []
            for i, bid in enumerate(body_ids):
                # Try to get model name from PyBullet
                body_info = p.getBodyInfo(bid, physicsClientId=pid)
                body_name = body_info[1].decode("utf-8") if body_info[1] else f"{name_prefix}_{i}"

                if use_fixed_base:
                    # Set mass to 0 for static/kinematic behavior
                    p.changeDynamics(bid, -1, mass=0.0, physicsClientId=pid)

                obj = cls(
                    body_id=bid,
                    sim_core=sim_core,
                    pickable=pickable,
                    mass=0.0 if use_fixed_base else None,
                    collision_mode=collision_mode,
                    name=body_name,
                )
                objs.append(obj)
            return objs

        # Wrap loadSDF + registration in batch_spawn (disables rendering for GUI speedup)
        if sim_core is not None:
            with sim_core.batch_spawn():
                objects = _load_and_wrap()
        else:
            objects = _load_and_wrap()

        lazy_logger.info(lambda: f"Loaded {len(objects)} objects from SDF: " f"{resolved_path}")
        return objects

    @classmethod
    def from_params(cls, spawn_params: SimObjectSpawnParams, sim_core=None) -> "SimObject":
        """
        Create a SimObject from SimObjectSpawnParams.

        Args:
            spawn_params: SimObjectSpawnParams instance
            sim_core: Reference to simulation core

        Returns:
            SimObject instance

        Example::

            params = SimObjectSpawnParams(
                visual_shape=ShapeParams(
                    shape_type="mesh",
                    mesh_path="pallet.obj",
                    mesh_scale=[0.5, 0.5, 0.5],
                    rgba_color=[0.8, 0.6, 0.4, 1.0]
                ),
                collision_shape=ShapeParams(
                    shape_type="box",
                    half_extents=[0.5, 0.4, 0.1]
                ),
                initial_pose=Pose.from_xyz(0, 0, 0.1),
                mass=1.0
            )
            obj = SimObject.from_params(params, sim_core)
        """
        factory = cls.from_urdf if spawn_params.urdf_path is not None else cls.from_mesh
        return factory(
            **_forward_spawn_params(
                factory,
                spawn_params,
                aliases={"initial_pose": "pose"},
                extra_kwargs={"sim_core": sim_core},
            )
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any], sim_core=None) -> "SimObject":
        """Create a SimObject from a raw config dict.

        Combines :meth:`SimObjectSpawnParams.from_dict` and :meth:`from_params`
        in a single call.  Uses ``cls._spawn_params_cls`` so subclasses
        automatically use their own SpawnParams.

        Args:
            config: Entity definition dict (as parsed from YAML).
            sim_core: Reference to simulation core (optional).

        Returns:
            ``cls`` instance.
        """
        params = cls._spawn_params_cls.from_dict(config)
        return cls.from_params(params, sim_core)

    @property
    def is_static(self) -> bool:
        """
        Whether this object is static (never moves).

        Returns True if collision_mode is STATIC.
        This property is read-only. To change collision mode,
        use set_collision_mode() method.
        """
        return self.collision_mode == CollisionMode.STATIC

    def set_collision_mode(self, mode: CollisionMode) -> None:
        """
        Change collision detection mode for this object.

        This method updates the collision mode and notifies sim_core to
        update all collision-related caches (AABB, spatial grid, etc.).

        Args:
            mode: New CollisionMode

        Example::

            # Change to static
            obj.set_collision_mode(CollisionMode.STATIC)

            # Disable collision
            obj.set_collision_mode(CollisionMode.DISABLED)

            # Enable 2D collision
            obj.set_collision_mode(CollisionMode.NORMAL_2D)
        """
        if self.collision_mode == mode:
            return  # No change

        old_mode = self.collision_mode
        self.collision_mode = mode

        # Update PyBullet collision filter if switching to/from DISABLED
        if mode == CollisionMode.DISABLED:
            self._set_pybullet_collision_enabled(False)
        elif old_mode == CollisionMode.DISABLED:
            self._set_pybullet_collision_enabled(True)

        # Notify sim_core to update collision system
        if self.sim_core is not None:
            self.sim_core._update_object_collision_mode(self.object_id, old_mode, mode)

        self._log.info(f"collision_mode changed from {old_mode.value} -> {mode.value}")

    def _set_pybullet_collision_enabled(self, enabled: bool) -> None:
        """Filter this body in or out of PyBullet's own collision detection.

        Every link, not only the base. ``DISABLED`` is documented as turning
        PyBullet collision off for the whole object, but the filter was
        applied to link -1 alone -- so an articulated body kept every child
        link colliding physically while being excluded from PyBulletFleet's
        own checks. Single-link bodies are unaffected, there being nothing
        else to set.

        ``setCollisionFilterGroupMask(body, link, group, mask)``: mask 0
        collides with nothing; group 1 and mask -1 are PyBullet's documented
        defaults.

        Re-enabling writes those defaults, which is not always what the link
        started with: a fixed base loaded from URDF is filtered more tightly
        than ``(1, -1)``, so a round trip through ``DISABLED`` can leave it
        colliding where it did not before. That is pre-existing behaviour for
        the base link, now applied consistently to the rest; restoring the
        true original would need a getter PyBullet does not expose.
        """
        group, mask = (1, -1) if enabled else (0, 0)
        for link_index in range(-1, p.getNumJoints(self.body_id, physicsClientId=self._pid)):
            p.setCollisionFilterGroupMask(self.body_id, link_index, group, mask, physicsClientId=self._pid)

    def get_pose(self) -> Pose:
        """
        Return current position and orientation as Pose object.

        Performance optimization:

        - Kinematic objects (mass=0): Always returns cached pose since position
          only changes via set_pose(), which updates the cache.
        - Dynamic objects: Caches within the same simulation timestep to avoid
          redundant PyBullet API calls. Cache is invalidated when sim_time advances.
        """
        # Kinematic objects never move on their own - cache is always valid
        # (position only changes via set_pose(), which updates the cache)
        if self.is_kinematic:
            return self._cached_pose

        # Dynamic objects: check if cache is valid for current sim time
        current_sim_time = self.sim_core.sim_time if self.sim_core is not None else -1.0

        if self._cached_pose is not None and current_sim_time >= 0 and self._cached_pose_sim_time == current_sim_time:
            return self._cached_pose

        # Cache miss: Query PyBullet and update cache
        position, orientation = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._pid)
        self._cached_pose = Pose.from_pybullet(position, orientation)
        self._cached_pose_sim_time = current_sim_time

        return self._cached_pose

    def set_pose(self, pose: Pose, preserve_velocity: bool = True) -> bool:
        """
        Set position and orientation from a Pose object.

        Two-phase step behaviour (see ``docs/design/two-phase-step/spec.md``):

        - **Outside ``step_once()``** (immediate path): writes to PyBullet via
          ``resetBasePositionAndOrientation`` and, for kinematic objects, refreshes
          the AABB + spatial grid immediately. Same behaviour as legacy callers.
        - **Inside ``step_once()``** (buffered path, ``sim_core._in_step == True``):
          updates only the internal pose cache and registers this object for the
          Phase 2 / Phase 3 batch flush. ``get_pose()`` returns the new pose
          immediately (cache is the source of truth), but PyBullet's view, AABBs,
          and the spatial grid are not refreshed until the flush runs later in the
          same ``step_once()`` invocation. A no-op set_pose (same pose as cache)
          is skipped entirely with zero PyBullet cost.

        Attached children (base-link attachments) are propagated recursively
        through the same code path \u2014 so they take the buffered path inside the
        step and the immediate path outside it.

        Args:
            pose: Pose object containing position and orientation
            preserve_velocity: If True, preserve current velocity after setting pose.
                             If False, velocity is reset to zero (default behavior of PyBullet).
                             For kinematic objects (mass=0), velocity preservation has no effect,
                             so this is automatically set to False for performance.

        Returns:
            True if the object moved (position or orientation changed), False otherwise
        """
        position, orientation = pose.as_position_orientation()
        moved = self._set_pose_internal(position, orientation, preserve_velocity)
        return moved

    def set_pose_raw(
        self,
        position,
        orientation,
        preserve_velocity: bool = True,
    ) -> bool:
        """
        Set position and orientation from raw lists/tuples (no Pose allocation).

        This is the fast path for internal use where position and orientation
        are already available as lists/tuples, avoiding Pose object creation overhead.

        Args:
            position: [x, y, z] position (list, tuple, or any sequence)
            orientation: [x, y, z, w] quaternion (list, tuple, or any sequence)
            preserve_velocity: If True, preserve current velocity after setting pose.

        Returns:
            True if the object moved, False otherwise
        """
        return self._set_pose_internal(position, orientation, preserve_velocity)

    def _set_pose_internal(
        self,
        position,
        orientation,
        preserve_velocity: bool = True,
    ) -> bool:
        """
        Internal implementation for setting position and orientation.

        Shared by set_pose() and set_pose_raw() to avoid code duplication.

        Args:
            position: [x, y, z] position
            orientation: [x, y, z, w] quaternion
            preserve_velocity: If True, preserve current velocity after setting pose.

        Returns:
            True if the object moved, False otherwise
        """
        # Static objects should never move - warn and return early
        if self.collision_mode == CollisionMode.STATIC:
            self._log.warning(
                "Attempting to move static object. "
                "Static objects (CollisionMode.STATIC) should not be moved. "
                "If this object needs to move, change collision_mode to NORMAL_3D/NORMAL_2D."
            )
            return False

        # Detect movement using cached pose (no PyBullet API call needed)
        # Use manual distance calculation instead of np.linalg.norm for ~10x speedup
        last_pos = self._cached_pose.position
        last_orn = self._cached_pose.orientation
        dx = position[0] - last_pos[0]
        dy = position[1] - last_pos[1]
        dz = position[2] - last_pos[2]
        pos_diff_sq = dx * dx + dy * dy + dz * dz

        # Quaternion angular distance: min(|q1 - q2|², |q1 + q2|²) to handle q = -q equivalence
        dq0 = orientation[0] - last_orn[0]
        dq1 = orientation[1] - last_orn[1]
        dq2 = orientation[2] - last_orn[2]
        dq3 = orientation[3] - last_orn[3]
        orn_diff_sq = dq0 * dq0 + dq1 * dq1 + dq2 * dq2 + dq3 * dq3
        # Check q = -q equivalence only if diff is significant
        if orn_diff_sq > 1e-12:
            sq0 = orientation[0] + last_orn[0]
            sq1 = orientation[1] + last_orn[1]
            sq2 = orientation[2] + last_orn[2]
            sq3 = orientation[3] + last_orn[3]
            orn_sum_sq = sq0 * sq0 + sq1 * sq1 + sq2 * sq2 + sq3 * sq3
            if orn_sum_sq < orn_diff_sq:
                orn_diff_sq = orn_sum_sq

        moved = pos_diff_sq > 1e-12 or orn_diff_sq > 1e-12

        # ----- Update pose cache (only when actually moved — saves ~70ns/call) -----
        # The `moved` check is free because we already computed it above.
        # Slice assignment is ~2x faster than 7 individual element writes (single C memcpy).
        if moved:
            self._cached_pose.position[:] = position
            self._cached_pose.orientation[:] = orientation

        # Timestamp is always refreshed so get_pose() cache validity (D2) stays correct,
        # even when the new pose equals the old one.
        self._cached_pose_sim_time = self.sim_core.sim_time if self.sim_core is not None else -1.0

        # ----- Two-phase step buffered path -----
        # Inside step_once(), defer PyBullet write + AABB/grid refresh to the
        # Phase-2 flush. Kinematic agent updates (the hot path) take this branch.
        # `_cached_pose` is the source of truth for the pending write — we only need
        # to remember per-call `preserve_velocity`. See docs/design/two-phase-step/spec.md (D2/D3b).
        #
        # No-op set_pose (moved=False) is dropped entirely: nothing changed, so neither
        # the PyBullet write nor the AABB/grid refresh is needed. Idle agents that keep
        # calling set_pose every tick (post-goal hover, stationary fleet) get a clean
        # zero-cost path. This matches the immediate path, which also gates the AABB/grid
        # refresh on `moved` (the unconditional reset call there is a legacy artefact).
        if self.sim_core is not None and self.sim_core._in_step:
            if moved:
                self._pending_preserve_velocity = preserve_velocity
                self.sim_core._pending_pose_ids.add(self.object_id)
                self.sim_core._mark_object_moved(self.object_id)
                # Attached-object propagation still recurses through _set_pose_internal,
                # so child writes also get buffered (each child sees _in_step True).
                # Skipped when parent didn't move — children's relative pose is unchanged.
                self._propagate_to_attached(position, orientation, preserve_velocity)
            return moved

        # ----- Immediate path (legacy / outside step_once) -----
        # Use cached is_kinematic flag to avoid PyBullet API calls
        if preserve_velocity and not self.is_kinematic:
            # Get current velocities to preserve them (for dynamic objects only)
            linear_vel, angular_vel = p.getBaseVelocity(self.body_id, physicsClientId=self._pid)
            p.resetBasePositionAndOrientation(self.body_id, position, orientation, physicsClientId=self._pid)
            p.resetBaseVelocity(self.body_id, linear_vel, angular_vel, physicsClientId=self._pid)
        else:
            # Fast path: Just set position without velocity operations
            # (kinematic objects or explicitly requested to skip velocity preservation)
            p.resetBasePositionAndOrientation(self.body_id, position, orientation, physicsClientId=self._pid)

        # Notify sim_core if movement detected
        if moved and self.sim_core is not None:
            self.sim_core._mark_object_moved(self.object_id)

            # Update AABB and spatial grid immediately for kinematic objects (optimization)
            # Physics objects are updated in step_once() to batch PyBullet calls
            if self.object_id in self.sim_core._kinematic_objects:
                self.sim_core._update_object_aabb(self.object_id)
                # Update spatial grid if cell_size is initialized
                if self.sim_core._cached_cell_size is not None:
                    self.sim_core._update_object_spatial_grid(self.object_id)

        # Recursively apply the same coordinates and velocity to attached_objects.
        # Skipped when the parent didn't move — children's relative pose is unchanged.
        if moved:
            self._propagate_to_attached(position, orientation, preserve_velocity)

        return moved  # Return movement detection result

    def _propagate_to_attached(self, position, orientation, preserve_velocity: bool) -> None:
        """Recursively forward this object's pose to attached children (base-link attachments only).

        Link-level attachments are handled by Agent.update_attached_objects_kinematics().
        Each recursive set_pose_raw() naturally takes the same code path
        (buffered inside step_once, immediate outside) as the parent call.
        """
        if not self.attached_objects:
            return
        self._log.debug(lambda: f"set_pose: attached_objects(before)={[o.body_id for o in self.attached_objects]}")
        for obj in self.attached_objects:
            # Skip objects attached to a specific link (not base link).
            # These are handled by Agent.update_attached_objects_kinematics()
            # which uses the actual link position from PyBullet.
            if obj._attached_link_index >= 0:
                continue
            # Follow using relative position and orientation from attachment
            # _attach_offset is always initialized with default (0,0,0) if not set
            new_pos, new_orn = p.multiplyTransforms(
                position, orientation, obj._attach_offset.position, obj._attach_offset.orientation
            )
            obj.set_pose_raw(new_pos, new_orn, preserve_velocity=preserve_velocity)
            self._log.debug(
                lambda obj=obj: f"attached_object set pose: obj_body_id={obj.body_id} "
                f"position={obj._attach_offset.position} orientation={obj._attach_offset.orientation}"
            )

    def attach_object(
        self,
        obj: "SimObject",
        parent_link_index: Union[int, str] = -1,
        relative_pose: Optional[Pose] = None,
        joint_type: int = p.JOINT_FIXED,
        keep_world_pose: bool = False,
    ) -> bool:
        """
        Attach another SimObject to this object.

        Args:
            obj: Object to attach
            parent_link_index: Link index (int) or link name (str) to attach to.
                              -1 or ``"base_link"`` for the base link.
            relative_pose: Position and orientation offset in parent link's frame as Pose object
                          (default: None, which uses Pose(position=[0, 0, 0], orientation=[0, 0, 0, 1]))
            joint_type: PyBullet joint type (default: JOINT_FIXED)
            keep_world_pose: If True, automatically compute ``relative_pose``
                            so that the child keeps its current world position
                            and orientation.  Cannot be combined with an
                            explicit ``relative_pose``.

        Returns:
            True if attachment successful, False otherwise

        Raises:
            ValueError: If both ``relative_pose`` and ``keep_world_pose=True``
                        are specified.

        Example::

            # Attach to base link at 0.5m in front
            agent.attach_object(pallet, relative_pose=Pose.from_xyz(0.5, 0, 0))

            # Keep the child where it currently is in the world
            agent.attach_object(box, parent_link_index=2, keep_world_pose=True)

            # Attach to specific link by name
            agent.attach_object(box, parent_link_index="end_effector")
        """
        # Resolve link name to index
        parent_link_index = resolve_link_index(self.body_id, parent_link_index)

        if keep_world_pose and relative_pose is not None:
            raise ValueError("relative_pose and keep_world_pose=True cannot both be specified")

        if keep_world_pose:
            # Compute relative pose from current world positions
            if parent_link_index == -1:
                par_pos, par_orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._pid)
            else:
                ls = p.getLinkState(self.body_id, parent_link_index, computeForwardKinematics=1, physicsClientId=self._pid)
                par_pos, par_orn = ls[0], ls[1]
            inv_pos, inv_orn = p.invertTransform(par_pos, par_orn)
            obj_pose = obj.get_pose()
            obj_pos, obj_orn = obj_pose.as_position_orientation()
            rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, obj_pos, obj_orn)
            relative_pose = Pose.from_pybullet(rel_pos, rel_orn)

        # Default to zero offset if not specified
        if relative_pose is None:
            relative_pose = Pose(position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0])

        # Check if object is pickable

        if not obj.pickable:
            self._log.info(f"Cannot attach: object {obj.body_id} is not pickable")
            return False

        # Prevent attaching an object that is already attached to another SimObject

        if obj.is_attached():
            self._log.info(
                f"Cannot attach: object {obj.body_id} is already attached to "
                f"another SimObject (body_id={obj._attached_to.body_id if obj._attached_to else None})"
            )
            return False

        if obj in self.attached_objects:
            self._log.info(f"Object {obj.body_id} already attached")
            return False

        # Add to attached list
        self.attached_objects.append(obj)
        self._log.debug(
            lambda: (
                f"attach_object: obj={obj.body_id} relative_pose={relative_pose.position} "
                f"orientation={relative_pose.orientation}"
            )
        )

        # Get parent link position and orientation
        if parent_link_index == -1:
            parent_pos, parent_orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._pid)
        else:
            link_state = p.getLinkState(self.body_id, parent_link_index, computeForwardKinematics=1, physicsClientId=self._pid)
            parent_pos, parent_orn = link_state[0], link_state[1]

        # Save attachment state with user-specified relative pose
        obj._attach_offset = relative_pose
        obj._attached_to = self
        obj._attached_link_index = parent_link_index
        if parent_link_index >= 0 and self.sim_core is not None and hasattr(self, "update_attached_objects_kinematics"):
            self.sim_core._register_link_attachment_parent(self.object_id)
        self._log.debug(
            lambda: f"attach_object: obj={obj.body_id} _attach_offset.position={obj._attach_offset.position} "
            f"_attach_offset.orientation={obj._attach_offset.orientation}"
        )

        # Create constraint if object has mass (physics-based attachment)
        if obj.mass > 0:
            obj._constraint_id = p.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=parent_link_index,
                childBodyUniqueId=obj.body_id,
                childLinkIndex=-1,
                jointType=joint_type,
                jointAxis=[0, 0, 0],
                parentFramePosition=relative_pose.position,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=relative_pose.orientation,
                childFrameOrientation=[0, 0, 0, 1],
                physicsClientId=self._pid,
            )

        # Immediately teleport child to the correct world position so that
        # callers never see a 1-frame lag where the object is still at its
        # old location.  Uses the parent link pose already fetched above.
        new_pos, new_orn = p.multiplyTransforms(parent_pos, parent_orn, relative_pose.position, relative_pose.orientation)
        obj.set_pose_raw(new_pos, new_orn)

        self._log.info(f"Attached object {obj.body_id} to link {parent_link_index}")
        return True

    def detach_object(
        self,
        obj: "SimObject",
        drop_pose: Optional[Pose] = None,
        drop_relative_pose: Optional[Pose] = None,
    ) -> bool:
        """
        Detach an object from this object.

        After detaching, the object can optionally be teleported:

        - ``drop_pose`` — absolute world pose.
        - ``drop_relative_pose`` — offset applied to the object's current
          (pre-detach) world pose (position **and** orientation are composed
          via ``multiplyTransforms``).
        - Neither — the object stays at its current world position.

        These two parameters are mutually exclusive.

        Args:
            obj: Object to detach
            drop_pose: Absolute world pose to teleport the object to after
                      detaching.  Mutually exclusive with ``drop_relative_pose``.
            drop_relative_pose: Pose offset relative to the object's current
                      (pre-detach) world pose.  Mutually exclusive with
                      ``drop_pose``.

        Returns:
            True if detachment successful, False otherwise

        Raises:
            ValueError: If both ``drop_pose`` and ``drop_relative_pose`` are
                        specified.

        Example::

            # Detach and leave in place
            agent.detach_object(pallet)

            # Detach and place at absolute world position
            agent.detach_object(pallet, drop_pose=Pose.from_xyz(10, 5, 0.1))

            # Detach and offset from current position (e.g. lower by 0.5m)
            agent.detach_object(pallet, drop_relative_pose=Pose.from_xyz(0, 0, -0.5))
        """
        if drop_pose is not None and drop_relative_pose is not None:
            raise ValueError("drop_pose and drop_relative_pose cannot both be specified")

        if obj not in self.attached_objects:
            self._log.info(f"Object {obj.body_id} is not attached")
            return False

        # Remove from attached list
        self.attached_objects.remove(obj)
        if (
            obj._attached_link_index >= 0
            and self.sim_core is not None
            and not any(attached._attached_link_index >= 0 for attached in self.attached_objects)
        ):
            self.sim_core._unregister_link_attachment_parent(self.object_id)

        # Debug: show attached_objects after removal
        self._log.debug(
            lambda: f"detach_object: detached={obj.body_id} "
            f"attached_objects(after)={[o.body_id for o in self.attached_objects]}"
        )

        # Remove constraint if it exists
        if obj._constraint_id is not None:
            p.removeConstraint(obj._constraint_id, physicsClientId=self._pid)
            obj._constraint_id = None

        # Reset attachment state to default values
        obj._attach_offset = Pose(position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0])
        obj._attached_to = None
        obj._attached_link_index = -1

        # Teleport to final pose if requested
        if drop_relative_pose is not None:
            # Compute world pose from current position + relative offset
            cur_pos, cur_orn = obj.get_pose().as_position_orientation()
            final_pos, final_orn = p.multiplyTransforms(
                cur_pos, cur_orn, drop_relative_pose.position, drop_relative_pose.orientation
            )
            obj.set_pose_raw(final_pos, final_orn)
        elif drop_pose is not None:
            obj.set_pose(drop_pose)

        self._log.info(f"Detached object {obj.body_id}")
        return True

    def get_attached_objects(self) -> List["SimObject"]:
        """
        Return list of objects attached to this object.

        Returns:
            List of attached SimObject instances
        """
        return self.attached_objects.copy()

    def get_kinematic_attachment_state(self) -> Optional[dict]:
        """Describe a supported physics-off attachment without runtime body IDs."""
        if self._attached_to is None:
            return None
        if self.mass > 0 or self._constraint_id is not None:
            raise ValueError("Physics attachment is outside the kinematic checkpoint profile")
        return {
            "parent_name": self._attached_to.name,
            "parent_link": (
                p.getJointInfo(self._attached_to.body_id, self._attached_link_index, physicsClientId=self._pid)[12].decode(
                    "utf-8"
                )
                if self._attached_link_index >= 0
                else "base_link"
            ),
            "relative_position": list(self._attach_offset.position),
            "relative_orientation": list(self._attach_offset.orientation),
        }

    def is_attached(self) -> bool:
        """
        Check if this object is attached to another object.

        Returns:
            True if attached to another object, False otherwise
        """
        return self._attached_to is not None

    def find_nearest_pickable(self, search_radius: float = 1.0) -> Optional["SimObject"]:
        """Find the nearest pickable, unattached SimObject within radius.

        Convenience wrapper around :func:`tools.find_nearest` that
        searches from this object's current position with the standard
        ``pickable and not attached`` filter.

        Args:
            search_radius: Maximum search distance in metres (default: 1.0)

        Returns:
            Nearest SimObject or ``None`` if nothing found.
        """
        from . import tools

        if self.sim_core is None:
            return None

        return tools.find_nearest(
            self.sim_core.sim_objects,
            position=self.get_pose().position,
            search_radius=search_radius,
            predicate=lambda o: o.pickable and not o.is_attached(),
            exclude=self,
        )

    def register_callback(self, callback: Callable, frequency: float = 0.25):
        """Register a callback function to be executed periodically."""
        self.callbacks.append({"func": callback, "frequency": frequency, "last_executed": 0.0})

    def execute_callbacks(self, current_time: float):
        """Execute registered callbacks based on their frequency."""
        for cbinfo in self.callbacks:
            if current_time - cbinfo["last_executed"] >= cbinfo["frequency"]:
                cbinfo["func"](self)
                cbinfo["last_executed"] = current_time
