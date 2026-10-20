"""Shared base class for vectorized (batched) kinematic controllers.

A BatchKinematicController manages **N agents at once** using NumPy arrays for
per-agent state, replacing the per-agent Python dispatch that
``KinematicController.compute()`` performs in the default path.

Lifecycle
---------
1. User constructs the controller (no agents yet).
2. User registers each agent via ``c.register_agent(agent)``. The first
   registration auto-binds the controller to the agent's ``sim_core`` and
   appends it to ``sim_core._batch_controllers`` so ``step_once()`` will
   drive it. All subsequent agents must belong to the same sim_core.
3. Each sim step, ``sim_core.step_once()`` calls ``c.batch_advance(dt)``
   during Phase 1, which computes new poses for all registered agents and
   writes them via the buffered batch API ``sim_core.set_poses()``.
4. Agents may be unregistered at runtime. When the last agent leaves, the
   controller auto-detaches from ``sim_core._batch_controllers``.

Subclass responsibilities
-------------------------
Concrete subclasses implement ``batch_advance(dt)`` and any controller-specific
state (e.g. trajectory parameters). Subclasses should call
``self._resize_state(n)`` from ``_on_agents_changed`` to keep their own arrays
sized with ``self._agents``.
"""

from __future__ import annotations

import math
import threading
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import numpy as np
import pybullet as p

from pybullet_fleet.controller import Controller
from pybullet_fleet.geometry import quat_angle_between, quat_slerp_batch
from pybullet_fleet.logging_utils import get_lazy_logger

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore

logger = get_lazy_logger(__name__)

# Registry: registry-name string → BatchKinematicController subclass.
# Keys match the ``_registry_name`` attribute (e.g. ``"batch_omni"``,
# ``"batch_differential"``).  Populated automatically via __init_subclass__
# when a concrete subclass sets ``_registry_name``.
BATCH_CONTROLLER_REGISTRY: Dict[str, Type["BatchKinematicController"]] = {}


class BatchKinematicController(Controller):
    """Vectorized base for kinematic controllers that drive many agents at once.

    Inherits from :class:`Controller` directly (not ``KinematicController``):
    the per-agent ``compute(agent, dt) -> bool`` contract does not apply here
    since batch controllers act through :meth:`batch_advance` during Phase 1
    of ``sim_core.step_once()``. Per-agent kinematic parameters (max_vel,
    accel, etc.) are read from each ``Agent`` at ``set_path`` time.

    To register a custom batch controller, set ``_registry_name`` on the
    subclass and import the module once::

        class MyBatchController(BatchKinematicController):
            _registry_name = "my_batch"

            def batch_advance(self, dt): ...

        # Usage:
        #   AgentManager(sim_core=sim, fleet_controller={"type": "my_batch"})
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "_registry_name", None)
        if name:
            BATCH_CONTROLLER_REGISTRY[name] = cls

    def __init__(self) -> None:
        # ROS bridge callbacks may replace paths while the simulation thread is
        # evaluating vectorized trajectory arrays. Keep those mutations atomic.
        self._state_lock = threading.RLock()
        self._sim_core: Optional["MultiRobotSimulationCore"] = None
        self._agents: List["Agent"] = []
        # id(agent) -> row index. Stable for the lifetime of an agent in this
        # controller; on unregister we compact rows by swapping with the last
        # row, so other agents may see their index change.
        self._agent_index: Dict[int, int] = {}

        # Per-agent output buffers (filled by batch_advance, written to sim).
        self._object_ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._pos_buf: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._orn_buf: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        self._moved_mask: np.ndarray = np.zeros((0,), dtype=bool)
        # True for rows that moved on the *previous* flush — lets _apply_phase1
        # zero an agent's reported velocity on the step it stops.
        self._was_moved: np.ndarray = np.zeros((0,), dtype=bool)

    # ------------------------------------------------------------------ #
    # Lifecycle: agent registration (sim_core attachment is implicit)
    # ------------------------------------------------------------------ #

    def register_agent(self, agent: "Agent") -> int:
        """Register *agent* with this batch controller.

        On the first call the controller records ``agent.sim_core`` so that
        :meth:`_apply_phase1` can call ``set_poses``.  The controller is
        *not* attached to ``sim_core._batch_controllers`` directly;
        ``step_once()`` discovers it through the owning
        :class:`~pybullet_fleet.agent_manager.AgentManager` that is
        registered with the sim.

        Returns:
            The agent's row index in this controller's state arrays.

        Raises:
            ValueError: If the agent is already registered, has no sim_core,
                or belongs to a different sim_core than previously-registered
                agents.
        """
        key = id(agent)
        if key in self._agent_index:
            raise ValueError(f"Agent {agent} is already registered with this batch controller.")
        agent_sim = getattr(agent, "sim_core", None)
        if agent_sim is None:
            raise ValueError(f"Agent {agent} has no sim_core; spawn it via Agent.from_params(..., sim_core) first.")
        if self._sim_core is None:
            self._sim_core = agent_sim
        elif agent_sim is not self._sim_core:
            raise ValueError("Agent belongs to a different sim_core than this batch controller's other agents.")
        idx = len(self._agents)
        self._agents.append(agent)
        self._agent_index[key] = idx
        self._resize_base_buffers(len(self._agents))
        self._object_ids[idx] = agent.object_id
        # Initialize buffer with current pose so a no-op step doesn't move the agent.
        pose = agent.get_pose()
        self._pos_buf[idx] = pose.position
        self._orn_buf[idx] = pose.orientation
        # Mark the agent as batch-driven: agent.update() will still run (so
        # the action queue and event hooks are preserved), but ctrl.compute()
        # is skipped — movement is driven exclusively by batch_advance().
        agent._batch_controller = self
        self._on_agent_registered(idx, agent)
        return idx

    def unregister_agent(self, agent: "Agent") -> None:
        """Unregister *agent*. Compacts the state arrays by swapping with last row.

        Raises:
            KeyError: If the agent is not registered.
        """
        key = id(agent)
        if key not in self._agent_index:
            raise KeyError(f"Agent {agent} is not registered with this batch controller.")
        idx = self._agent_index.pop(key)
        agent._batch_controller = None
        last = len(self._agents) - 1
        if idx != last:
            # Swap last row into the freed slot.
            self._agents[idx] = self._agents[last]
            self._agent_index[id(self._agents[idx])] = idx
            self._swap_rows(idx, last)
        self._agents.pop()
        self._resize_base_buffers(len(self._agents))
        self._on_agent_unregistered(idx, agent)

    def reset(self) -> None:
        """Unregister all agents and reset arrays."""
        # Snapshot before mutation
        agents = list(self._agents)
        for a in agents:
            self.unregister_agent(a)

    # ------------------------------------------------------------------ #
    # Buffer management (called by lifecycle methods above)
    # ------------------------------------------------------------------ #

    def _resize_base_buffers(self, n: int) -> None:
        """Resize the (N, ...) buffers owned by this base class."""
        self._object_ids = self._resize_rows(self._object_ids, n)
        self._pos_buf = self._resize_rows(self._pos_buf, n)
        self._orn_buf = self._resize_rows(self._orn_buf, n)
        self._moved_mask = self._resize_rows(self._moved_mask, n)
        self._was_moved = self._resize_rows(self._was_moved, n)

    @staticmethod
    def _resize_rows(arr: np.ndarray, n: int) -> np.ndarray:
        """Resize ``arr`` along axis 0 to length ``n``, preserving existing rows
        and zero-padding new rows.

        Unlike :func:`numpy.resize`, this does **not** wrap around: new rows
        are always zero-initialised. Trailing dims (if any) are preserved.
        """
        if arr.shape[0] == n:
            return arr
        new_shape = (n,) + arr.shape[1:]
        new_arr = np.zeros(new_shape, dtype=arr.dtype)
        copy_n = min(arr.shape[0], n)
        if copy_n > 0:
            new_arr[:copy_n] = arr[:copy_n]
        return new_arr

    def _swap_rows(self, i: int, j: int) -> None:
        """Swap rows ``i`` and ``j`` in all base buffers. Subclasses override
        to also swap their own per-agent arrays.
        """
        self._object_ids[[i, j]] = self._object_ids[[j, i]]
        self._pos_buf[[i, j]] = self._pos_buf[[j, i]]
        self._orn_buf[[i, j]] = self._orn_buf[[j, i]]
        self._moved_mask[[i, j]] = self._moved_mask[[j, i]]
        self._was_moved[[i, j]] = self._was_moved[[j, i]]

    def _on_agent_registered(self, idx: int, agent: "Agent") -> None:
        """Hook for subclasses to initialise their own per-agent state row."""

    def _on_agent_unregistered(self, idx: int, agent: "Agent") -> None:
        """Hook for subclasses to clear their own per-agent state row."""

    def _on_cancel_path(self, idx: int) -> None:
        """Cancel in-progress path for agent at *idx* (called by agent.stop()).

        Subclasses override to reset their phase/path arrays to IDLE so
        batch_advance() stops updating that agent immediately.
        """

    # ------------------------------------------------------------------ #
    # Phase 1 entry point (called by core_simulation.step_once)
    # ------------------------------------------------------------------ #

    def synchronized_set_path(self, agent: "Agent", path, **kwargs) -> bool:
        """Set one path without racing a concurrent simulation step.

        ``_is_moving`` is set here, inside the lock, rather than by the caller
        afterwards. The flag belongs to the same state transition as the
        arrays: a concurrent ``step_once()`` can complete a very short or
        zero-distance path and clear the flag, and a caller setting it after
        the lock released would then leave an idle row reporting
        ``is_moving``. Setting it before ``set_path()`` is equally wrong --
        that is the half-committed state this guard exists to prevent -- so it
        goes after the call and still under the lock.

        Returns ``False`` when the controller refuses the path (see
        ``path_rejection``); the row and ``_is_moving`` are then untouched.
        """
        with self._state_lock:
            accepted = getattr(self, "set_path")(agent, path, **kwargs)
            if accepted is False:
                return False
            agent._is_moving = True
            return True

    def synchronized_cancel_path(self, agent: "Agent") -> None:
        """Cancel one path without racing a concurrent simulation step."""
        with self._state_lock:
            idx = self._agent_index.get(id(agent))
            if idx is not None:
                self._on_cancel_path(idx)

    def synchronized_batch_advance(self, dt: float) -> np.ndarray:
        """Advance vectorized trajectory state without concurrent path writes."""
        with self._state_lock:
            return self.batch_advance(dt)

    @abstractmethod
    def batch_advance(self, dt: float) -> np.ndarray:
        """Compute one timestep for all registered agents.

        Implementations must:

        1. Fill ``self._pos_buf`` and ``self._orn_buf`` with new poses for all
           agents.
        2. Fill ``self._moved_mask`` with True for agents whose pose changed.
        3. Call ``self._apply_phase1()`` to write the moved rows back to sim.

        Returns:
            ``self._moved_mask`` — a (N,) boolean array.
        """

    def _apply_phase1(self, dt: float = 0.0) -> None:
        """Write the rows flagged in ``self._moved_mask`` back to sim via the
        buffered batch API ``sim_core.set_poses``, and refresh each moved agent's
        reported velocity.

        Batch controllers otherwise never set ``_current_velocity`` /
        ``_current_angular_velocity``, so ``RobotHandler`` would publish zero
        odometry/diagnostics velocity for moving robots (unlike the per-agent
        controllers). When ``dt > 0`` we derive linear/yaw velocity from the
        cached pre-update pose → new pose for the moved rows, and zero the
        velocity of agents that stopped this step.

        No-op when no agents moved (and none just stopped) or no sim_core attached.
        """
        if self._sim_core is None or len(self._agents) == 0:
            return

        mask = self._moved_mask
        # Agents that were moving last flush but not this one have stopped — zero
        # their reported velocity (matches the per-agent controllers going IDLE).
        just_stopped = self._was_moved & ~mask
        if just_stopped.any():
            for i in np.flatnonzero(just_stopped):
                ag = self._agents[i]
                ag._current_velocity[:] = 0.0
                ag._current_angular_velocity = 0.0
        self._was_moved = mask.copy()

        if not bool(mask.any()):
            return

        # Per-agent velocity from the cached (pre-update) pose → new pose. The
        # agent's pose cache still holds the old pose because set_poses() runs
        # afterwards. get_pose() is cached for kinematic agents (no PyBullet call).
        if dt > 0.0:
            inv_dt = 1.0 / dt
            for i in np.flatnonzero(mask):
                ag = self._agents[i]
                old = ag.get_pose()
                np.subtract(self._pos_buf[i], old.position, out=ag._current_velocity)
                ag._current_velocity *= inv_dt
                new_yaw = p.getEulerFromQuaternion(tuple(self._orn_buf[i]))[2]
                dyaw = (new_yaw - old.yaw + math.pi) % (2.0 * math.pi) - math.pi
                ag._current_angular_velocity = dyaw * inv_dt

        # set_poses accepts a Sequence[int] for object_ids; passing a NumPy
        # boolean-indexed view of int64 ids is cheap (no Python loop).
        ids = self._object_ids[mask].tolist()
        positions = self._pos_buf[mask]
        orientations = self._orn_buf[mask]
        self._sim_core.set_poses(ids, positions, orientations)

    # ------------------------------------------------------------------ #
    # Controller ABC contract — batch controllers act through batch_advance()
    # during Phase 1 of step_once(), not per-agent compute(). The compute()
    # stub is required because Controller.compute is abstract, and is also a
    # safety net in case Agent.update() is ever called on a registered agent
    # (it should be dormant because register_agent sets _needs_update=False).
    # ------------------------------------------------------------------ #

    def compute(self, agent, dt: float) -> bool:
        return False

    def set_velocity(self, **kwargs) -> None:
        raise NotImplementedError(
            "Velocity mode is not supported in batch controllers. "
            "Use the per-agent OmniController / DifferentialController instead."
        )


def resolve_batch_controller_key(entry: str) -> Type["BatchKinematicController"]:
    """Resolve a batch controller from a registry name or a dotted import path.

    Mirrors the ``type:`` / ``class:`` resolution used by per-agent controllers
    and plugins:

    * **Registry name** (e.g. ``"batch_omni"``) — looked up case-insensitively
      in :data:`BATCH_CONTROLLER_REGISTRY` (populated when a subclass sets
      ``_registry_name``).
    * **Dotted path** (e.g. ``"my_pkg.MyBatchController"``) — imported
      dynamically and validated to be a :class:`BatchKinematicController`
      subclass. Lets custom batch controllers be selected from YAML without
      pre-importing their module.

    Raises:
        ValueError: If *entry* is neither a known registry name nor a dotted
            path resolving to a ``BatchKinematicController`` subclass.
    """
    key = entry.lower()
    if key in BATCH_CONTROLLER_REGISTRY:
        return BATCH_CONTROLLER_REGISTRY[key]

    if "." in entry:
        from pybullet_fleet.config_utils import resolve_class

        cls = resolve_class(entry)
        if not (isinstance(cls, type) and issubclass(cls, BatchKinematicController)):
            raise ValueError(f"Batch controller class {entry!r} must be a " f"BatchKinematicController subclass, got {cls!r}")
        return cls

    available = list(BATCH_CONTROLLER_REGISTRY)
    raise ValueError(
        f"Unknown batch controller {entry!r}. "
        f"Available: {available}. "
        "Use a registry name (after importing its module) or a dotted "
        "'module.ClassName' path."
    )
