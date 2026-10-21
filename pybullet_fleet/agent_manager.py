"""
core/agent_manager.py
Management classes for simulation objects and agents.
- SimObjectManager: Base manager for all simulation objects
- AgentManager: Extended manager with movement control for agents
"""

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from .agent import Agent, AgentSpawnParams, Pose
from .sim_object import SimObject, SimObjectSpawnParams
from .tools import grid_to_world

logger = logging.getLogger(__name__)


@contextmanager
def _nullctx():
    """Trivial context manager (no-op) for fallback when sim_core has no batch_spawn."""
    yield


# Type variable for generic SimObjectManager
T = TypeVar("T", bound=SimObject)


@dataclass
class GridSpawnParams:
    """
    Grid parameters for object spawning in 3D space.

    Attributes:
        x_min: Minimum X grid index
        x_max: Maximum X grid index
        y_min: Minimum Y grid index
        y_max: Maximum Y grid index
        spacing: Grid spacing [spacing_x, spacing_y, spacing_z] in meters
        offset: Grid offset [offset_x, offset_y, offset_z] in meters
        z_min: Minimum Z grid index (defaults to 0)
        z_max: Maximum Z grid index (defaults to 0 for 2D grids)

    Note:
        Objects spawn in X → Y → Z order.
        For 2D grids, set z_min=z_max=0 (default).
        For 3D grids, set z_max > z_min to enable vertical layers.
    """

    x_min: int
    x_max: int
    y_min: int
    y_max: int
    spacing: List[float]  # [spacing_x, spacing_y, spacing_z]
    offset: List[float]  # [offset_x, offset_y, offset_z]
    z_min: int = 0
    z_max: int = 0

    @property
    def total_cells(self) -> int:
        """Total number of grid cells: (x range) * (y range) * (z range)."""
        return (self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1) * (self.z_max - self.z_min + 1)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GridSpawnParams":
        """Create GridSpawnParams from a configuration dictionary.

        Supports two formats:

        **Explicit ranges** (original) — requires ``x_min``, ``x_max``,
        ``y_min``, ``y_max``, ``spacing``, ``offset``::

            grid:
              x_min: 0
              x_max: 9
              y_min: 0
              y_max: 4
              spacing: [3.0, 3.0, 0.0]
              offset: [0, 0, 0.05]

        **Count-based** (convenience) — requires ``count`` and
        ``spacing``; ``columns`` and ``offset`` are optional::

            grid:
              count: 50
              spacing: [3.0, 3.0]
              offset: [0, 0, 0.05]
              columns: 10

        When ``count`` is present, :meth:`from_count` is used to
        compute the grid ranges.

        Args:
            config: Configuration dictionary.

        Returns:
            GridSpawnParams instance.

        Raises:
            KeyError: If a required key is missing.
        """
        if "count" in config:
            import math

            count = config["count"]
            spacing_raw = config["spacing"]
            spacing = [spacing_raw[0], spacing_raw[1], 0.0] if len(spacing_raw) == 2 else list(spacing_raw)
            offset = list(config.get("offset", [0.0, 0.0, 0.0]))
            columns = config.get("columns", math.ceil(math.sqrt(count)))
            return cls.from_count(count=count, columns=columns, spacing=spacing, offset=offset)

        return cls(
            x_min=config["x_min"],
            x_max=config["x_max"],
            y_min=config["y_min"],
            y_max=config["y_max"],
            spacing=config["spacing"],
            offset=config["offset"],
            z_min=config.get("z_min", 0),
            z_max=config.get("z_max", 0),
        )

    @classmethod
    def from_count(
        cls,
        count: int,
        columns: Optional[int] = None,
        spacing: Optional[List[float]] = None,
        offset: Optional[List[float]] = None,
    ) -> "GridSpawnParams":
        """Create GridSpawnParams from a total entity count.

        Computes ``x_max`` and ``y_max`` from *count* and *columns*
        so that the grid has exactly enough cells.  Useful when you
        know "I want 50 robots" rather than "x_max=9, y_max=4".

        Args:
            count: Total number of entities to place.
            columns: Grid width (X cells) before wrapping to the next
                row.  Defaults to ``ceil(sqrt(count))``.
            spacing: ``[sx, sy, sz]`` in metres.  Default ``[1,1,0]``.
            offset: ``[ox, oy, oz]`` origin offset.  Default ``[0,0,0]``.

        Returns:
            GridSpawnParams with ``x_min=0``, ``y_min=0``,
            ``x_max=columns-1``, ``y_max=ceil(count/columns)-1``.
        """
        import math

        if columns is None:
            columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)

        return cls(
            x_min=0,
            x_max=columns - 1,
            y_min=0,
            y_max=rows - 1,
            spacing=spacing if spacing is not None else [1.0, 1.0, 0.0],
            offset=offset if offset is not None else [0.0, 0.0, 0.0],
        )


class SimObjectManager(Generic[T]):
    """
    Base manager class for simulation objects (SimObject and Agent).

    This class provides centralized control of multiple simulation objects.

    Features:
    - Grid-based object spawning
    - Query all object poses
    - Get object count

    Type Parameter:
        T: Type of objects managed (must be SimObject or its subclass)
    """

    def __init__(self, sim_core=None, object_class: type = SimObject, enable_profiling: bool = False):
        """
        Initialize SimObjectManager.

        Args:
            sim_core: Reference to simulation core (optional)
            object_class: Default class to instantiate when spawning
                          (SimObject or a subclass such as Agent).
            enable_profiling: If ``True``, spawn methods log elapsed time via
                ``logger.info``.  Default ``False``.
                Can be overridden later: ``mgr.enable_profiling = True``.
        """
        self.sim_core = sim_core
        self._object_class: type = object_class
        self.enable_profiling: bool = enable_profiling
        self.objects: List[T] = []
        self.body_ids: Dict[int, T] = {}  # body_id -> object mapping (PyBullet layer)
        self.object_ids: Dict[int, T] = {}  # object_id -> object mapping (Manager layer)

    def add_object(self, obj: T) -> None:
        """
        Add an object to the manager.

        Args:
            obj: SimObject or Agent instance to add

        Returns:
            None.  The object is silently skipped (with a warning) when its
            ``object_id`` is invalid (< 0), which typically means it was
            created without a sim_core.

        Note:
            Objects are automatically registered to sim_core.sim_objects
            via SimObject.__init__(), so we don't register them here again.
        """
        if obj.object_id < 0:
            logger.warning(
                "object (body_id=%d) has invalid object_id=%d. " "Was it created without sim_core?  Skipping add_object().",
                obj.body_id,
                obj.object_id,
            )
            return
        self.objects.append(obj)
        self.body_ids[obj.body_id] = obj
        self.object_ids[obj.object_id] = obj

    def remove_object(self, obj: T) -> bool:
        """Remove an object from the manager's tracking lists.

        This only removes the object from the *manager's* bookkeeping
        (``objects``, ``body_ids``, ``object_ids``).  It does **not**
        touch ``sim_core`` or PyBullet — that is handled by
        :meth:`MultiRobotSimulationCore.remove_object`.

        Called automatically by ``sim_core.remove_object()`` when this
        manager is registered via ``sim_core.register_manager()``.

        Args:
            obj: The object to remove.

        Returns:
            True if the object was found and removed, False otherwise.
        """
        if obj.object_id not in self.object_ids:
            return False
        try:
            self.objects.remove(obj)
        except ValueError:
            pass
        self.body_ids.pop(obj.body_id, None)
        self.object_ids.pop(obj.object_id, None)
        return True

    def clear(self) -> None:
        """Remove all objects from the manager's tracking lists.

        This is a manager-only operation — it does **not** remove objects
        from ``sim_core`` or PyBullet.  Typically called by
        ``sim_core.reset()`` to keep the manager in sync after a full
        simulation reset.
        """
        self.objects.clear()
        self.body_ids.clear()
        self.object_ids.clear()

    def spawn_objects_grid(
        self,
        num_objects: int,
        grid_params: GridSpawnParams,
        spawn_params: SimObjectSpawnParams,
        *,
        name_prefix: Optional[str] = None,
    ) -> List[T]:
        """Spawn multiple objects in a grid pattern.

        This is a convenience wrapper around ``spawn_grid_counts`` for spawning
        a single object type with a specified count.

        Args:
            num_objects: Number of objects to spawn
            grid_params: GridSpawnParams instance with grid configuration
            spawn_params: SimObjectSpawnParams instance with object parameters
            name_prefix: Optional prefix for stable generated names
                (``<prefix>_0``, ``<prefix>_1``, ...).

        Returns:
            List of spawned object instances
        """
        return self.spawn_grid_counts(
            grid_params=grid_params,
            spawn_params_count_list=[(spawn_params, num_objects)],
            name_prefix=name_prefix,
        )

    def spawn_grid_mixed(
        self,
        num_objects: int,
        grid_params: GridSpawnParams,
        spawn_params_list: List[Tuple[SimObjectSpawnParams, float]],
        *,
        name_prefix: Optional[str] = None,
    ) -> List[T]:
        """
        Spawn multiple SimObjects in a grid pattern with mixed types.

        This method allows spawning different types of objects with specified probabilities.
        If probabilities don't sum to 1.0, some grid positions may remain empty.

        Args:
            num_objects: Maximum number of objects to spawn
            grid_params: GridSpawnParams instance with grid configuration
            spawn_params_list: List of (spawn_params, probability) tuples
                             - spawn_params: SimObjectSpawnParams for this type
                             - probability: Spawn probability (0.0 to 1.0)
                             Note: Probabilities don't need to sum to 1.0
            name_prefix: Optional prefix for stable generated names
                (``<prefix>_0``, ``<prefix>_1``, ...).

        Returns:
            List of spawned SimObject instances

        Example::

            # Spawn 100 objects: 60% typeA, 30% typeB, 10% empty
            spawn_params_list = [
                (typeA_params, 0.6),
                (typeB_params, 0.3),
            ]
            manager.spawn_grid_mixed(100, grid_params, spawn_params_list)
        """

        return self._spawn_grid_mixed_impl(
            num_objects=num_objects,
            grid_params=grid_params,
            spawn_params_list=spawn_params_list,
            name_prefix=name_prefix,
        )

    def _spawn_grid_mixed_impl(
        self,
        num_objects: int,
        grid_params: GridSpawnParams,
        spawn_params_list: List[Tuple[Any, float]],
        *,
        name_prefix: Optional[str] = None,
    ) -> List[Any]:
        """
        Internal implementation for spawn_grid_mixed.

        This method contains the common logic for spawning mixed object types in a grid.

        Args:
            num_objects: Maximum number of objects to spawn
            grid_params: GridSpawnParams instance with grid configuration
            spawn_params_list: List of (spawn_params, probability) tuples
            name_prefix: Optional prefix for stable generated names
                (``<prefix>_0``, ``<prefix>_1``, ...).

        Returns:
            List of spawned object instances
        """
        start = time.perf_counter() if self.enable_profiling else None

        # Normalize probabilities if total > 1.0
        total_prob = sum(prob for _, prob in spawn_params_list)
        if total_prob > 1.0:
            logger.warning("Total probability %.2f > 1.0, normalizing to 1.0", total_prob)
            # Normalize: divide each probability by total
            spawn_params_list = [(params, prob / total_prob) for params, prob in spawn_params_list]
            total_prob = 1.0

        # Extract grid parameters
        x_min = grid_params.x_min
        x_max = grid_params.x_max
        y_min = grid_params.y_min
        y_max = grid_params.y_max
        z_min = grid_params.z_min
        z_max = grid_params.z_max
        spacing = grid_params.spacing
        offset = grid_params.offset

        cls = self._object_class

        # Spawn objects sequentially
        spawned_objects = []
        current_x = x_min
        current_y = y_min
        current_z = z_min

        ctx = self.sim_core.batch_spawn() if self.sim_core and hasattr(self.sim_core, "batch_spawn") else _nullctx()
        with ctx:
            for i in range(num_objects):
                # Calculate grid position
                spawn_grid = [current_x, current_y, current_z]
                spawn_pos = grid_to_world(spawn_grid, spacing, offset)

                # Randomly select spawn params based on probabilities
                rand = random.random()
                cumulative_prob = 0.0
                selected_params = None

                for spawn_params, prob in spawn_params_list:
                    cumulative_prob += prob
                    if rand < cumulative_prob:
                        selected_params = spawn_params
                        break

                # If selected, spawn object
                if selected_params is not None:
                    spawn_pose = self._make_spawn_pose(selected_params, spawn_pos)

                    # Create a copy of spawn_params with the calculated pose
                    grid_spawn_params = replace(selected_params, initial_pose=spawn_pose)

                    # Spawn object using from_params()
                    obj = cls.from_params(spawn_params=grid_spawn_params, sim_core=self.sim_core)
                    self._assign_grid_name(obj, name_prefix, len(spawned_objects))

                    # Track object
                    self.add_object(obj)
                    spawned_objects.append(obj)

                # Increment grid position: X → Y → Z
                current_x += 1
                if current_x > x_max:
                    current_x = x_min
                    current_y += 1
                    if current_y > y_max:
                        current_y = y_min
                        current_z += 1
                        if current_z > z_max:
                            # Grid exhausted
                            if i < num_objects - 1:
                                logger.warning(
                                    "Grid exhausted after %d iterations (requested %d)",
                                    i + 1,
                                    num_objects,
                                )
                            break

        if start is not None:
            elapsed = time.perf_counter() - start
            logger.info(
                "spawn_grid_mixed: %d objects in %.3f sec",
                len(spawned_objects),
                elapsed,
            )
        return spawned_objects

    def _assign_grid_name(self, obj: T, name_prefix: Optional[str], index: int) -> None:
        """Assign an API-generated name when grid spawning requested one."""
        if name_prefix is None:
            return
        obj.set_name(f"{name_prefix}_{index}")

    def _make_spawn_pose(self, params: SimObjectSpawnParams, spawn_pos: List[float]) -> Pose:
        """Create a Pose for grid spawning, preserving orientation from *params* if set."""
        if params.initial_pose is not None:
            return Pose(position=spawn_pos, orientation=params.initial_pose.orientation)
        return Pose.from_xyz(*spawn_pos)

    def spawn_grid_counts(
        self,
        grid_params: GridSpawnParams,
        spawn_params_count_list: List[Tuple[SimObjectSpawnParams, int]],
        *,
        name_prefix: Optional[str] = None,
    ) -> List[T]:
        """
        Spawn objects on a grid according to the specified count for each type.

        Each type is spawned exactly the requested number of times. Grid
        positions are randomly shuffled unless ``name_prefix`` is set; prefixed
        names keep deterministic grid placement so ``<prefix>_i`` maps to a
        stable position across runs.

        Args:
            grid_params: GridSpawnParams
            spawn_params_count_list: List of (spawn_params, count)
            name_prefix: Optional prefix for stable generated names
                (``<prefix>_0``, ``<prefix>_1``, ...).

        Returns:
            List of spawned objects

        Raises:
            ValueError: If the total count exceeds the number of grid cells.
        """
        # --- Validate --------------------------------------------------
        total_count = sum(count for _, count in spawn_params_count_list)
        grid_x = grid_params.x_max - grid_params.x_min + 1
        grid_y = grid_params.y_max - grid_params.y_min + 1
        grid_z = grid_params.z_max - grid_params.z_min + 1
        grid_num = grid_x * grid_y * grid_z
        if total_count > grid_num:
            raise ValueError(f"Total spawn count ({total_count}) exceeds grid cell count ({grid_num})")

        start = time.perf_counter() if self.enable_profiling else None

        # --- Build assignment list: one (params, grid_coord) per object -
        # Flatten spawn_params_count_list into a list of params, one entry
        # per object to spawn.  e.g. [(paramsA,3),(paramsB,2)]
        #   → [paramsA, paramsA, paramsA, paramsB, paramsB]
        flat_params: List[SimObjectSpawnParams] = []
        for params, count in spawn_params_count_list:
            flat_params.extend([params] * count)
        if name_prefix is None:
            random.shuffle(flat_params)

        # Build grid coordinates. Keep prefixed names and their params
        # deterministic; otherwise shuffle positions so exact-count mixed types
        # are spatially distributed.
        grid_coords = [
            [x, y, z]
            for z in range(grid_params.z_min, grid_params.z_max + 1)
            for y in range(grid_params.y_min, grid_params.y_max + 1)
            for x in range(grid_params.x_min, grid_params.x_max + 1)
        ]
        if name_prefix is None:
            random.shuffle(grid_coords)

        # --- Spawn: zip flat_params with grid_coords 1-to-1 -----------
        # len(flat_params) <= len(grid_coords) is guaranteed by the
        # validation above, so zip stops after all objects are placed.
        # Remaining grid cells are simply left empty.
        cls = self._object_class
        spawned_objects: List[T] = []
        ctx = self.sim_core.batch_spawn() if self.sim_core and hasattr(self.sim_core, "batch_spawn") else _nullctx()
        with ctx:
            for params, coord in zip(flat_params, grid_coords):
                spawn_pos = grid_to_world(coord, grid_params.spacing, grid_params.offset)
                spawn_pose = self._make_spawn_pose(params, spawn_pos)
                grid_spawn_params = replace(params, initial_pose=spawn_pose)
                obj = cls.from_params(spawn_params=grid_spawn_params, sim_core=self.sim_core)
                self._assign_grid_name(obj, name_prefix, len(spawned_objects))
                self.add_object(obj)
                spawned_objects.append(obj)

        if start is not None:
            elapsed = time.perf_counter() - start
            logger.info(
                "spawn_grid_counts: %d objects in %.3f sec",
                len(spawned_objects),
                elapsed,
            )
        return spawned_objects

    def spawn_objects_batch(self, params_list: List[SimObjectSpawnParams]) -> List[T]:
        """
        Batch API to spawn multiple object instances at once.

        When *sim_core* provides :meth:`batch_spawn`, rendering is disabled
        and the spatial grid rebuild is deferred until all objects are created.

        Args:
            params_list: List of spawn params for each object.

        Returns:
            List of created object instances.
        """
        start = time.perf_counter() if self.enable_profiling else None
        objects = []
        ctx = self.sim_core.batch_spawn() if self.sim_core and hasattr(self.sim_core, "batch_spawn") else _nullctx()
        with ctx:
            for params in params_list:
                obj = self._object_class.from_params(params, sim_core=self.sim_core)
                self.add_object(obj)
                objects.append(obj)
        if start is not None:
            elapsed = time.perf_counter() - start
            logger.info("spawn_objects_batch: %d objects in %.3f sec", len(params_list), elapsed)
        return objects

    # ------------------------------------------------------------------
    # Config-driven spawning
    # ------------------------------------------------------------------
    def spawn_from_config(self, entities_yaml: List[Dict[str, Any]]) -> List[T]:
        """Spawn entities from a list of config dicts.

        Dispatches to :meth:`entity_cls.from_dict` for each entity, which
        internally calls the appropriate ``SpawnParams.from_dict`` and
        ``from_params``.  Custom entity classes can override ``from_dict``
        to use their own ``SpawnParams``.

        If a dict has no ``type`` field, ``"agent"`` is assumed for backward
        compatibility.  Supported types: ``"agent"``, ``"sim_object"``, or any
        custom type registered via :func:`register_entity_class`.

        Args:
            entities_yaml: List of entity definition dicts (from YAML).
                Each dict may omit ``type`` (defaults to ``"agent"``).

        Returns:
            Flat list of all spawned objects in definition order.

        Raises:
            KeyError: If ``type`` is not registered in the entity registry.
        """
        from .entity_registry import ENTITY_REGISTRY

        enriched = [{**d, "type": d.get("type", "agent")} for d in entities_yaml]

        result: List[T] = []
        ctx = self.sim_core.batch_spawn() if self.sim_core and hasattr(self.sim_core, "batch_spawn") else _nullctx()
        with ctx:
            for entity_def in enriched:
                entity_type = entity_def["type"]
                if entity_type not in ENTITY_REGISTRY:
                    raise KeyError(f"Unknown entity type: {entity_type!r}. " f"Available: {list(ENTITY_REGISTRY)}")
                entity_cls = ENTITY_REGISTRY[entity_type]

                obj = entity_cls.from_dict(entity_def, sim_core=self.sim_core)
                self.add_object(obj)
                result.append(obj)  # type: ignore[arg-type]

        return result

    def spawn_from_yaml(self, yaml_path: str, key: str = "entities") -> List[T]:
        """Load a YAML file and spawn entities from the specified section.

        Reads *yaml_path*, extracts the *key* section (default ``"entities"``),
        and delegates to :meth:`spawn_from_config`.

        Args:
            yaml_path: Path to YAML config file.
            key: Top-level YAML key containing the entity list.
                 Default ``"entities"``.

        Returns:
            List of spawned objects (empty if *key* is missing).

        Raises:
            FileNotFoundError: If *yaml_path* does not exist.
        """
        from .config_utils import load_yaml_config

        config = load_yaml_config(yaml_path)
        section = config.get(key, [])
        return self.spawn_from_config(section)

    def get_pose(self, object_index: int) -> Optional[Pose]:
        """
        Get current pose of a specific object.

        Args:
            object_index: Index of object in self.objects list

        Returns:
            Current Pose or None if invalid index
        """
        if 0 <= object_index < len(self.objects):
            return self.objects[object_index].get_pose()
        else:
            logger.warning("Invalid object index %d", object_index)
            return None

    def get_all_poses(self) -> List[Pose]:
        """
        Get current poses of all objects.

        Returns:
            List of Pose objects for all objects
        """
        return [obj.get_pose() for obj in self.objects]

    def get_poses_dict(self) -> Dict[int, Pose]:
        """
        Get current poses of all objects as a dictionary.

        Returns:
            Dictionary mapping body_id to Pose
        """
        return {obj.body_id: obj.get_pose() for obj in self.objects}

    def set_pose_all(self, pose_factory: Callable[[T], Pose]) -> None:
        """
        Set pose for all objects using a factory function.

        Applies *pose_factory* to each object and immediately calls ``set_pose``
        with the returned value.  Works for both SimObject and Agent instances.

        Args:
            pose_factory: Function that takes an object and returns a target Pose.

        Example::

            # Shift every object 1 m along X
            manager.set_pose_all(lambda obj: Pose.from_xyz(
                obj.get_pose().x + 1.0,
                obj.get_pose().y,
                obj.get_pose().z,
            ))
        """
        for obj in self.objects:
            obj.set_pose(pose_factory(obj))

    def get_object_count(self) -> int:
        """Get total number of managed objects."""
        return len(self.objects)

    def __repr__(self):
        return f"SimObjectManager(total={len(self.objects)})"


class AgentManager(SimObjectManager[Agent]):
    """
    Extended manager class for agents with update callback system.

    This class extends SimObjectManager with agent-specific features:
    - Grid-based agent spawning with AgentSpawnParams
    - Pose goal assignment to individual agents
    - Callback system for custom update logic (goals, state tracking, etc.)
    - Query moving/stopped agents
    - Optional batch controller for vectorized kinematic updates (``fleet_controller={"type": ...}`` or :meth:`enable_batch`)

    Key Features:
    - Auto-registers update callback to simulation loop
    - Callback receives (agents, manager, dt) for easy agent operations
    - Configurable update frequency
    - Direct access to manager methods within callback

    Note:
    - Agent.update() is automatically called by MultiRobotSimulationCore every step
    - AgentManager callbacks are for high-level logic (goals, coordination, etc.)
    - Agent-specific state can be stored in each Agent.user_data dict
    """

    def __init__(
        self,
        sim_core=None,
        name: Optional[str] = None,
        update_frequency: float = 10.0,
        object_class: type = Agent,
        enable_profiling: bool = False,
        fleet_controller: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AgentManager.

        Args:
            sim_core: Reference to simulation core (optional)
            name: Optional name for this manager.  When set, the manager can be
                retrieved later via :meth:`~pybullet_fleet.core_simulation
                .MultiRobotSimulationCore.get_manager`.
            update_frequency: Default update callback frequency in Hz (default: 10.0)
            object_class: Class to instantiate when spawning (default: Agent).
                          Pass an Agent subclass to manage custom agent types.
            enable_profiling: If ``True``, spawn methods log elapsed time.
                Default ``False``.
            fleet_controller: Manager controller configuration. ``type`` may
                be ``"batch_omni"`` or ``"batch_differential"`` to enable
                vectorized movement for every managed agent; omit it for
                per-agent execution. The remaining keys are shared
                ``ControllerParams`` defaults applied only where an agent did
                not set that field in its own ``controller:`` config.
        """
        super().__init__(sim_core, object_class=object_class, enable_profiling=enable_profiling)
        self.name: Optional[str] = name
        self._callbacks: List[Dict[str, Any]] = []  # List of registered callbacks
        self._update_frequency: float = update_frequency  # Default callback frequency in Hz
        self._batch_controller: Optional[Any] = None
        fleet_config = dict(fleet_controller or {})
        controller_type = fleet_config.pop("type", None)
        self._fleet_controller: Optional[Dict[str, Any]] = fleet_config or None

        # Always register so sim_core.remove_object() and sim_core.reset() keep
        # the manager's tracking lists in sync, regardless of batch mode.
        if self.sim_core is not None:
            self.sim_core.register_manager(self)

        if controller_type is not None:
            self.enable_batch(controller_type)

    # ------------------------------------------------------------------
    # Batch controller lifecycle
    # ------------------------------------------------------------------

    @property
    def agents(self) -> List[Agent]:
        """The agents this manager holds, skipping anything that is not one.

        ``spawn_from_config()`` dispatches on each entry's ``type``, so a
        config may mix ``agent`` and ``sim_object`` and ``objects`` can hold
        both. Everything on this class that commands movement, reads motion
        state or queues actions is agent-only, and reads this rather than
        ``objects``.
        """
        return [obj for obj in self.objects if isinstance(obj, Agent)]

    @property
    def batch_controller(self) -> Optional[Any]:
        """The fleet-wide :class:`BatchKinematicController`, or ``None``."""
        return self._batch_controller

    def enable_batch(self, mode: str) -> Any:
        """Create and attach a batch controller for *mode*, replacing any existing one.

        Any agents already managed by this instance that are not yet
        batch-controlled are registered immediately.  Agents spawned
        afterwards are registered automatically via :meth:`add_object`.

        Args:
            mode: Batch controller registry name (e.g. ``"batch_omni"``,
                ``"batch_differential"``) or a dotted import path to a custom
                :class:`BatchKinematicController` subclass
                (e.g. ``"my_pkg.MyBatchController"``).

        Returns:
            The newly-created :class:`BatchKinematicController`.

        Raises:
            ValueError: If *mode* is not recognised.
        """
        from pybullet_fleet.controllers.batch_base import resolve_batch_controller_key

        cls = resolve_batch_controller_key(mode)  # raises ValueError before any state change
        if self._batch_controller is not None:
            self.disable_batch()
        bc = cls()
        self._batch_controller = bc
        # Register agents already in the manager; see the `agents` property
        # for why `objects` may hold things that are not.
        for agent in self.agents:
            if agent._batch_controller is None:
                bc.register_agent(agent)
        return bc

    def disable_batch(self) -> None:
        """Unregister all managed agents from the batch controller and detach it."""
        bc = self._batch_controller
        if bc is None:
            return
        for agent in self.agents:
            if agent._batch_controller is bc:
                bc.unregister_agent(agent)
        self._batch_controller = None

    # ------------------------------------------------------------------
    # add/remove_object overrides — keep batch registration in sync
    # ------------------------------------------------------------------

    def add_object(self, obj: Agent) -> None:
        """Add an object, wiring up the agent-only parts only for agents.

        An ``AgentManager`` can legitimately hold a plain :class:`SimObject`:
        :meth:`spawn_from_config` dispatches on each entry's ``type`` and a
        config may mix ``agent`` and ``sim_object``. Everything below is
        agent-only, though -- the fleet-controller defaults read
        ``controller_params`` and the batch controller reads
        ``_batch_controller``, neither of which a ``SimObject`` has -- so a
        mixed manager used to raise ``AttributeError`` from inside this
        method as soon as either controller was attached. That could be long
        after the object was added, and it read as a fault in
        :meth:`enable_batch` rather than in the mix.
        """
        super().add_object(obj)
        if not isinstance(obj, Agent):
            return
        if self._fleet_controller:
            from dataclasses import fields as dc_fields
            from pybullet_fleet.controller_params import ControllerParams

            fleet_params = ControllerParams.from_dict(self._fleet_controller)
            for f in dc_fields(ControllerParams):
                if getattr(obj.controller_params, f.name) is None:
                    val = getattr(fleet_params, f.name)
                    if val is not None:
                        setattr(obj.controller_params, f.name, val)
        if self._batch_controller is not None:
            if obj._batch_controller is not None:
                logger.warning(
                    "Agent %r is already registered with batch controller %r; " "skipping registration with %r.",
                    obj,
                    obj._batch_controller,
                    self._batch_controller,
                )
            else:
                self._batch_controller.register_agent(obj)

    def remove_object(self, obj: Agent) -> bool:
        result = super().remove_object(obj)
        if (
            result
            and isinstance(obj, Agent)
            and self._batch_controller is not None
            and obj._batch_controller is self._batch_controller
        ):
            self._batch_controller.unregister_agent(obj)
        return result

    # ------------------------------------------------------------------
    # Convenience aliases (delegate to SimObjectManager methods)
    # ------------------------------------------------------------------
    def spawn_agents_grid(
        self,
        num_agents: int,
        grid_params: GridSpawnParams,
        spawn_params: AgentSpawnParams,
        *,
        name_prefix: Optional[str] = None,
    ) -> List[Agent]:
        """Spawn agents in a grid.  Alias for :meth:`spawn_objects_grid`."""
        return self.spawn_objects_grid(
            num_objects=num_agents,
            grid_params=grid_params,
            spawn_params=spawn_params,
            name_prefix=name_prefix,
        )

    def spawn_agents_grid_mixed(
        self,
        num_agents: int,
        grid_params: GridSpawnParams,
        spawn_params_list: List[Tuple[AgentSpawnParams, float]],
        *,
        name_prefix: Optional[str] = None,
    ) -> List[Agent]:
        """Spawn mixed agent types.  Alias for :meth:`spawn_grid_mixed`."""
        return self.spawn_grid_mixed(
            num_objects=num_agents,
            grid_params=grid_params,
            spawn_params_list=spawn_params_list,
            name_prefix=name_prefix,
        )

    def spawn_agent_grid_counts(
        self,
        grid_params: GridSpawnParams,
        spawn_params_count_list: List[Tuple[AgentSpawnParams, int]],
        *,
        name_prefix: Optional[str] = None,
    ) -> List[Agent]:
        """Spawn exact counts per type.  Alias for :meth:`spawn_grid_counts`."""
        return self.spawn_grid_counts(
            grid_params=grid_params,
            spawn_params_count_list=spawn_params_count_list,
            name_prefix=name_prefix,
        )

    def set_goal_pose(self, agent_index: int, goal: Pose) -> bool:
        """
        Set goal pose for a specific agent.

        Args:
            agent_index: Index into :attr:`agents`, not into ``objects`` --
                a manager holding a non-agent would otherwise shift the
                indices of everything after it.
            goal: Target Pose

        Returns:
            Whether the agent took the goal (see :meth:`Agent.set_path`).
        """
        agents = self.agents
        if 0 <= agent_index < len(agents):
            return agents[agent_index].set_goal_pose(goal)
        logger.warning("Invalid agent index %d", agent_index)
        return False

    def set_goal_pose_by_body_id(self, body_id: int, goal: Pose) -> bool:
        """
        Set goal pose for an agent by its PyBullet body ID.

        Note:
            This method uses PyBullet's internal body_id for compatibility
            with direct PyBullet API usage. For manager-level operations,
            prefer using set_goal_pose_by_object_id() instead.

        Args:
            body_id: PyBullet body ID
            goal: Target Pose
        """
        target = self.body_ids.get(body_id)
        if isinstance(target, Agent):
            return target.set_goal_pose(goal)
        logger.warning("Unknown agent body_id %d", body_id)
        return False

    def set_goal_pose_by_object_id(self, object_id: int, goal: Pose) -> bool:
        """
        Set goal pose for an agent by its simulation object ID.

        This is the preferred method for manager-level operations, as it uses
        the simulation's unique object_id rather than PyBullet's internal body_id.

        Args:
            object_id: Simulation object ID (from SimObject.object_id)
            goal: Target Pose
        """
        target = self.object_ids.get(object_id)
        if isinstance(target, Agent):
            return target.set_goal_pose(goal)
        logger.warning("Unknown agent object_id %d", object_id)
        return False

    def stop_all(self):
        """Stop all agents and clear their goals."""
        for agent in self.agents:
            agent.stop()

    def set_goal_pose_all(self, goal_factory: Callable[[Agent], Pose]) -> None:
        """
        Set goal pose for all agents using a factory function.

        This helper method simplifies bulk goal assignment by applying
        a factory function to each agent. The factory can use agent-specific
        data (current pose, user_data, etc.) to create customized goals.

        Args:
            goal_factory: Function that takes an Agent and returns a Pose

        Example::

            # Set all agents to move to their designated home positions
            def get_home_goal(robot):
                return robot.user_data['home_position']

            manager.set_goal_pose_all(get_home_goal)

            # Move all agents 5 meters forward from their current position
            def move_forward(robot):
                current_pos = robot.get_pose().position
                return Pose.from_xyz(current_pos[0] + 5, current_pos[1], current_pos[2])

            manager.set_goal_pose_all(move_forward)

        Note:
            This is a convenience wrapper around individual set_goal_pose calls.
            No performance benefit over explicit loops, but improves code clarity.
        """
        for agent in self.agents:
            goal = goal_factory(agent)
            agent.set_goal_pose(goal)

    def set_joints_targets_all(self, targets_factory: Callable[[Agent], Union[list, dict]], max_force: float = 500.0) -> None:
        """
        Set joint targets for all agents using a factory function.

        This helper method simplifies bulk joint control by applying
        a factory function to each agent. Useful for coordinated multi-robot
        arm movements.

        Args:
            targets_factory: Function that takes an Agent and returns joint targets
                           (list of positions or dict of {joint_name: position})
            max_force: Maximum force for joint control (default: 500.0)

        Example::

            # Set all robot arms to neutral position
            def get_neutral_joints(robot):
                return [0.0, 0.0, 0.0, 0.0]

            manager.set_joints_targets_all(get_neutral_joints)

            # Set custom positions based on robot index
            def get_custom_joints(robot):
                idx = robot.user_data.get('index', 0)
                return [idx * 0.1, 0.0, 0.0, 0.0]

            manager.set_joints_targets_all(get_custom_joints, max_force=1000.0)

        Note:
            Only applicable to agents with controllable joints (robot arms, etc.).
            Mobile robots without arm joints will ignore this command.
        """
        for agent in self.agents:
            targets = targets_factory(agent)
            agent.set_joints_targets(targets, max_force=max_force)

    def get_moving_count(self) -> int:
        """Get number of agents currently moving."""
        return sum(1 for agent in self.agents if agent.is_moving)

    def add_action_sequence_all(self, action_factory: Callable[[Agent], List[Any]]) -> None:
        """
        Apply action sequence to all agents using a factory function.

        This helper method simplifies bulk action assignment by applying
        a factory function to each agent. The factory can use agent-specific
        data (pose, user_data, etc.) to create customized action sequences.

        Args:
            action_factory: Function that takes an Agent and returns a list of Actions

        Example::

            def create_pick_drop_sequence(robot):
                target = robot.user_data['target_object']
                return [
                    PickAction(target_object_id=target.body_id),
                    MoveAction(path=Path.from_positions([[10, 10, 0]])),
                    DropAction(drop_pose=Pose(position=[10, 10, 0]))
                ]

            # Apply to all agents at once
            manager.add_action_sequence_all(create_pick_drop_sequence)

        Note:
            This is a convenience wrapper. Internally it still loops through agents,
            so there's no performance benefit over explicit loops. The main advantage
            is code clarity and reduced boilerplate.
        """
        for agent in self.agents:
            actions = action_factory(agent)
            agent.add_action_sequence(actions)

    def add_action_all(self, action_factory: Callable[[Agent], Any]) -> None:
        """
        Add a single action to all agents using a factory function.

        Similar to add_action_sequence_all, but adds a single action instead of a sequence.

        Args:
            action_factory: Function that takes an Agent and returns a single Action

        Example::

            def create_move_action(robot):
                goal_pos = robot.user_data['next_position']
                return MoveAction(path=Path.from_positions([goal_pos]))

            manager.add_action_all(create_move_action)
        """
        for agent in self.agents:
            action = action_factory(agent)
            agent.add_action(action)

    def register_callback(self, callback: Callable[["AgentManager", float], None], frequency: Optional[float] = None):
        """
        Register a callback for custom agent update logic.

        Multiple callbacks can be registered, each with their own frequency.
        This method automatically registers with sim_core if available.

        The callback provides access to the AgentManager instance, making it easy to:
        - Access all agents via manager.objects
        - Use manager methods (set_goal_pose, query states, etc.)
        - Store manager-level state in agent.user_data

        The callback should have signature:
            callback(manager: AgentManager, dt: float) -> None

        Args:
            callback: Function to call for agent updates
            frequency: Update frequency in Hz. If None, uses the default frequency
                      from __init__ (default: None)

        Example (Goal management)::

            def goal_update_logic(manager, dt):
                for agent in manager.objects:
                    if not agent.is_moving:
                        # Set new goal for stopped agents
                        new_goal = calculate_next_goal(agent)
                        agent.set_goal_pose(new_goal)

            manager.register_callback(goal_update_logic, frequency=4.0)

        Example (State tracking)::

            def state_tracker(manager, dt):
                for agent in manager.objects:
                    # Track agent statistics
                    if 'total_distance' not in agent.user_data:
                        agent.user_data['total_distance'] = 0.0
                    agent.user_data['total_distance'] += np.linalg.norm(agent.velocity) * dt

            manager.register_callback(state_tracker, frequency=10.0)

        Comparison with sim_core.register_callback():

        - AgentManager callback: Provides manager reference and filtered agent list
        - sim_core callback: Provides sim_core reference for all objects
        - Use AgentManager for agent-specific operations
        - Use sim_core for general simulation-wide operations
        """
        # Use default frequency if not specified
        if frequency is None:
            frequency = self._update_frequency

        # Store callback info
        self._callbacks.append({"func": callback, "frequency": frequency})

        # Register with sim_core (let sim_core handle frequency management)
        if self.sim_core is not None:
            # Create wrapper to match sim_core callback signature
            def _callback_wrapper(core, dt):
                callback(self, dt)

            self.sim_core.register_callback(_callback_wrapper, frequency=frequency)
            logger.info(
                "Registered callback at %s Hz (total: %d callback(s))",
                frequency,
                len(self._callbacks),
            )
        else:
            logger.warning(
                "sim_core not set, callback registered but not active. " "Total: %d callback(s)",
                len(self._callbacks),
            )

    def setup_camera(self, camera_config: Optional[Dict] = None) -> None:
        """
        Setup camera to view all managed objects in the simulation.

        This method automatically extracts positions from all managed objects
        and configures the camera to fit them in the view.

        Args:
            camera_config: Dictionary with camera settings (from yaml config).
                          If None, uses default settings.

        Example::

            # Basic usage with default settings
            agent_manager.setup_camera()

            # With custom camera config
            camera_config = {
                'camera_mode': 'auto',
                'camera_view_type': 'top_down',
                'camera_auto_scale': 0.8
            }
            agent_manager.setup_camera(camera_config)

        Note:
            Requires sim_core to be set during AgentManager initialization.
            Only works when GUI is enabled.
        """
        if self.sim_core is None:
            logger.warning("Cannot setup camera - sim_core is not set")
            return

        if camera_config is None:
            camera_config = {}

        # Extract positions from all managed objects
        try:
            poses = self.get_all_poses()
            entity_positions = [pose.position for pose in poses]

            if len(entity_positions) == 0:
                logger.warning("No objects to setup camera for")
                return

            # Call sim_core's setup_camera with extracted positions
            self.sim_core.setup_camera(camera_config=camera_config, entity_positions=entity_positions)

            logger.info("Camera setup complete for %d objects", len(entity_positions))

        except Exception as e:
            logger.error("Error setting up camera: %s", e)

    def __repr__(self):
        return f"AgentManager(total={len(self.objects)}, " f"moving={self.get_moving_count()})"
