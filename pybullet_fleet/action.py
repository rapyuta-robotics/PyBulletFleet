"""
action.py
Base classes and implementations for robot actions.

Provides a unified interface for different types of robot actions:
- MoveAction: Path following
- PickAction: Object picking with approach
- DropAction: Object dropping with approach
- WaitAction: Timed waiting (e.g., charging)

Actions are executed through an action queue system in the Agent class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union, TYPE_CHECKING
import logging
import numpy as np
import pybullet as p

from ._replan import refused_until_stopped
from .geometry import Pose, Path
from .types import ActionStatus, MovementDirection
from . import tools
from .logging_utils import get_named_lazy_logger

if TYPE_CHECKING:
    from .sim_object import SimObject

# Create logger for this module (for module-level use only)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default constants shared across Action classes
# ---------------------------------------------------------------------------
DEFAULT_MOVE_SKIP_THRESHOLD: float = 0.05
"""Default minimum distance (m) for MOVING_TO_PICK / MOVING_TO_DROP.

When the robot is already closer than this to the pick/drop pose,
the MoveAction sub-step is skipped entirely.  Prevents edge-case
hangs with zero-distance paths."""

# Note: ActionStatus and MovementDirection have been moved to types.py

DEFAULT_MAX_FORCE: float = 500.0
"""Default maximum motor force (N) for joint/EE control."""

DEFAULT_JOINT_TOLERANCE: float = 0.01
"""Default joint-space tolerance (rad or m) for convergence checks."""

DEFAULT_EE_TOLERANCE: float = 0.02
"""Default Cartesian end-effector tolerance (m) for convergence checks."""


class PickPhase(Enum):
    """Phases of pick action execution."""

    INIT = "init"
    APPROACHING = "approaching"
    MOVING_TO_PICK = "moving_to_pick"
    PICKING = "picking"
    RETREATING = "retreating"


class DropPhase(Enum):
    """Phases of drop action execution."""

    INIT = "init"
    APPROACHING = "approaching"
    MOVING_TO_DROP = "moving_to_drop"
    DROPPING = "dropping"
    RETREATING = "retreating"


class Action(ABC):
    """
    Base class for all robot actions.

    Actions represent high-level tasks that agents can perform:
    - MoveAction: Navigate along a path
    - PickAction: Pick up an object
    - DropAction: Drop an object
    - WaitAction: Wait for a duration

    Actions are executed through Agent.add_action() or Agent.add_action_sequence().
    Each action manages its own state and returns True when complete.

    Attributes:
        status: Current execution status
        start_time: When action started (simulation time)
        end_time: When action completed (simulation time)
        error_message: Error description if action failed
    """

    def __init__(self):
        self.status = ActionStatus.NOT_STARTED
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error_message: Optional[str] = None
        # Instance-level named logger with action class prefix
        self._log = get_named_lazy_logger(__name__, prefix=f"[{self.__class__.__name__}]")

    @abstractmethod
    def execute(self, agent, dt: float) -> bool:
        """
        Execute action for one timestep.

        Args:
            agent: Agent instance executing this action
            dt: Timestep duration in seconds

        Returns:
            True if action is complete (success or failure), False if still executing
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset action state for re-execution."""
        self.status = ActionStatus.NOT_STARTED
        self.start_time = None
        self.end_time = None
        self.error_message = None

    def cancel(self):
        """Cancel action execution."""
        self.status = ActionStatus.CANCELLED
        if self.start_time is not None and self.end_time is None:
            # Action was in progress, set end time
            self.end_time = self.start_time  # Use start time as placeholder

    def get_duration(self) -> Optional[float]:
        """
        Get action execution duration.

        Returns:
            Duration in seconds, or None if not yet completed
        """
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def is_complete(self) -> bool:
        """Check if action has completed (successfully or with failure)."""
        return self.status in [ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.CANCELLED]

    def _update_log_prefix(self, agent=None):
        """Update log prefix using agent's identification.

        Reuses the agent's log prefix (which includes object_id and name)
        and appends the action class name.

        Typically called once at action start (in :meth:`_log_start`)
        since agent does not change during execution.

        Format: ``[Agent:id:name:ActionClass] `` or ``[ActionClass] `` if no agent.
        """
        action_name = self.__class__.__name__
        if agent is not None and hasattr(agent, "_log"):
            # Reuse agent's prefix: "[Agent:3:robot_A] " -> "Agent:3:robot_A"
            agent_prefix = agent._log.prefix.strip().strip("[]")
            if agent_prefix:
                self._log.set_prefix(f"[{agent_prefix}:{action_name}] ")
                return
        self._log.set_prefix(f"[{action_name}] ")

    def _log_start(self, agent=None):
        """
        Log action start with all dataclass field values.
        Automatically logs all public fields of the action.

        Args:
            agent: Agent instance (optional, for agent ID logging)
        """
        self._update_log_prefix(agent)

        # Get all dataclass fields if available
        if hasattr(self, "__dataclass_fields__"):
            params = {}
            fields = getattr(self, "__dataclass_fields__", {})  # type: ignore
            for field_name, field_info in fields.items():
                # Skip private fields (starting with _) and init=False fields
                if not field_name.startswith("_") and field_info.init:
                    value = getattr(self, field_name)
                    params[field_name] = value

            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self._log.info(f"START - {params_str}")
        else:
            self._log.info("START")

    def _log_failure(self, reason: str, agent=None):
        """
        Log action failure with reason.

        Args:
            reason: Failure reason
            agent: Agent instance (optional, for agent ID logging)
        """
        self.error_message = reason
        self._log.info(f"FAILED - {reason}")
        self._log.error(f"Action failed: {reason}")

    def _log_phase_transition(self, phase, agent=None):
        """
        Log phase transition for multi-phase actions.

        Args:
            phase: New phase (Enum)
            agent: Agent instance (optional, for agent ID logging)
        """
        self._log.info(f"Phase transition -> {phase.value.upper()}")

    def _log_end(self):
        """Log action completion with duration.

        Called when the action finishes successfully.
        Logs at INFO level with the action's total duration.
        """
        duration = self.get_duration()
        if duration is not None:
            self._log.info(f"END - duration={duration:.2f}s")
        else:
            self._log.info("END")


def _pending_path(agent):
    """The path ``agent``'s controller holds to start later, or ``None``."""
    from pybullet_fleet.controller import KinematicController

    if agent._batch_controller is not None:
        return agent._batch_controller.pending_path(agent)
    base = agent._controllers[0] if agent._controllers else None
    return base.pending_path() if isinstance(base, KinematicController) else None


@dataclass
class MoveAction(Action):
    """
    Move along a path.

    Wraps the existing Agent path following functionality into the Action interface.

    The action ends when the agent comes to rest with no path of its own held
    for later: ``COMPLETED`` when it stands at the path's last waypoint,
    ``CANCELLED`` when it stopped elsewhere -- the path was braked
    (:meth:`Agent.brake <pybullet_fleet.agent.Agent.brake>`) or replaced by a
    newer one. A path the agent refuses for good (a fixed base, an empty path)
    ``FAILED``; one it refuses only while it is moving is tried again on the
    next update.

    Args:
        path: Path object with waypoints to follow
        auto_approach: Add approach waypoint if far from start (default: True)
        final_orientation_align: Align to final orientation after reaching end (default: True)
        direction: Movement direction - MovementDirection.FORWARD (default, face movement direction)
                  or MovementDirection.BACKWARD (maintain orientation, move backward).
                  Can also accept string "forward" or "backward" for convenience.

    Example::

        # Move forward (normal)
        path = Path.from_positions([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        action = MoveAction(path=path)
        agent.add_action(action)

        # Move backward
        action = MoveAction(path=path, direction=MovementDirection.BACKWARD)
        agent.add_action(action)
    """

    path: Path
    auto_approach: bool = True
    final_orientation_align: bool = True
    direction: Union[MovementDirection, str] = MovementDirection.FORWARD

    def __post_init__(self):
        super().__init__()
        # Normalize string to enum if needed
        if isinstance(self.direction, str):
            self.direction = MovementDirection(self.direction)

    def execute(self, agent, dt: float) -> bool:
        """Execute move action."""
        if self.status == ActionStatus.NOT_STARTED:
            # Start action: set path on agent
            self.status = ActionStatus.IN_PROGRESS
            self.start_time = agent.sim_core.sim_time if agent.sim_core else 0.0

            self._log_start(agent)

            # Use agent's existing path following system
            # Path visualization is handled by agent based on agent.path_visualize setting
            accepted = agent.set_path(
                self.path,
                auto_approach=self.auto_approach,
                final_orientation_align=self.final_orientation_align,
                direction=self.direction,  # Pass direction parameter (enum or string)
            )
            if accepted is False:
                if refused_until_stopped(agent, self.path):
                    # The agent is still moving and cannot take this path from
                    # where it is (see Agent.set_path); the motion it has runs
                    # on. Start again on the next update -- once that motion
                    # ends, the path is taken.
                    self.status = ActionStatus.NOT_STARTED
                    return False
                # Refused for good (a fixed base, an empty path).
                self.status = ActionStatus.FAILED
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                rejection = agent.path_rejection(list(self.path))
                self._log_failure(rejection.reason if rejection is not None else "path refused", agent)
                return True

        # The move ends when the agent is at rest with no path held for later
        # (goal_replacement brake / wait_complete); it succeeded only if the
        # agent stands at this path's end -- not if the path was braked or
        # replaced. agent.path is not consulted: a batch-owned agent's base
        # controller is idle, so it is always empty there.
        if not agent.is_moving and _pending_path(agent) is None:
            self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            if self._reached_end(agent):
                self.status = ActionStatus.COMPLETED
                self._log_end()
            else:
                self.status = ActionStatus.CANCELLED
                self._log.info("CANCELLED - the agent stopped short of the path's end (braked or replaced)")
            return True

        return False

    #: Distance (m) from the last waypoint that counts as having reached it.
    END_TOLERANCE = 1e-4

    def _reached_end(self, agent) -> bool:
        """Whether the agent stands at this path's last waypoint."""
        waypoints = list(self.path)
        if not waypoints:
            return False
        goal = np.asarray(waypoints[-1].position, dtype=float)
        position = np.asarray(agent.get_pose().position, dtype=float)
        if agent.controller_params.navigation_2d:
            goal, position = goal[:2], position[:2]
        return float(np.linalg.norm(position - goal)) <= self.END_TOLERANCE

    def reset(self):
        """Reset action state."""
        super().reset()


@dataclass
class JointAction(Action):
    """
    Joint positions control as an Action.

    Args:
        target_joint_positions: List of target joint positions
            (radians for revolute, metres for prismatic)
        max_force: Maximum force to apply (default: 500.0)
        tolerance: Position tolerance to consider as reached.
            Accepts a single float (applied to all joints), a list (one per
            joint), or a dict ``{joint_name: tol}``.
            When ``None`` (the default), the tolerance is resolved from
            ``agent.joint_tolerance`` at the first ``execute()`` call and
            written back to this attribute so the resolved value is
            inspectable after the action starts.

            .. note::

                A scalar tolerance applies the **same value** to every joint
                regardless of type.  For mixed prismatic / revolute chains,
                consider specifying per-joint tolerances so that prismatic
                joints (metres) can be tighter than revolute joints (radians).
        wait_time: Optional seconds to wait after reaching target (default: 0.0)

    Example::

        # Scalar tolerance (same for all joints)
        action = JointAction(target_joint_positions=[0.0, 1.0, 0.0, 0.0])
        agent.add_action(action)

        # Per-joint tolerance for mixed prismatic/revolute
        action = JointAction(
            target_joint_positions={"rail_joint": 0.5, "elbow_to_wrist": -0.3},
            tolerance={"rail_joint": 0.005, "elbow_to_wrist": 0.05},  # 5 mm / 0.05 rad
        )
    """

    target_joint_positions: Union[list, dict]
    max_force: float = DEFAULT_MAX_FORCE
    tolerance: Optional[Union[float, list, dict]] = None

    def __post_init__(self):
        super().__init__()

    def execute(self, agent, dt: float) -> bool:
        if self.status == ActionStatus.NOT_STARTED:
            self.status = ActionStatus.IN_PROGRESS
            self.start_time = agent.sim_core.sim_time if agent.sim_core else 0.0

            self._log_start(agent)

            # Resolve tolerance once: Action explicit > agent.joint_tolerance
            if self.tolerance is None:
                tol = agent.joint_tolerance
                # Copy mutable containers to avoid aliasing the agent's value
                if isinstance(tol, (dict, list)):
                    tol = tol.copy()
                self.tolerance = tol

            agent.set_joints_targets(self.target_joint_positions, max_force=self.max_force)

        # Check if all joints are within tolerance using Agent utility
        if agent.are_joints_at_targets(self.target_joint_positions, tolerance=self.tolerance):
            self.status = ActionStatus.COMPLETED
            self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            self._log_end()
            return True
        return False

    def reset(self):
        super().reset()


@dataclass
class PoseAction(Action):
    """
    End-effector pose control via inverse kinematics.

    Calls ``agent.move_end_effector()`` on the first step. When IK returns
    a candidate, the action waits for its joint targets to settle. After
    settling, it completes with ``COMPLETED`` only when both the initial
    reachability check and final EE Cartesian check pass; otherwise it
    completes with ``FAILED``.

    ``tolerance`` is used for both the IK reachability pre-check
    (via ``move_end_effector()``) and the final EE Cartesian distance
    verification (via ``are_ee_at_target()``).  Joint convergence
    uses the default joint tolerance (not this value).

    Args:
        target_position: Target EE position [x, y, z] in world frame.
        target_orientation: Target EE orientation as quaternion [qx, qy, qz, qw] (optional).
        end_effector_link: Link index (int), name (str), or None for auto-detect.
        max_force: Maximum force for motor control (default: 500.0).
        tolerance: EE Cartesian distance tolerance in metres (default: 0.02).

    Example::

        action = PoseAction(target_position=[0.3, 0.0, 0.5])
        agent.add_action(action)

        action = PoseAction(
            target_position=[0.3, 0.0, 0.5],
            target_orientation=[0, 0, 0, 1],
        )
        agent.add_action(action)
    """

    target_position: List[float]
    target_orientation: Optional[tuple] = None
    end_effector_link: Union[int, str, None] = None
    max_force: float = DEFAULT_MAX_FORCE
    tolerance: float = DEFAULT_EE_TOLERANCE

    # Internal state
    _reachable: bool = field(default=False, init=False)

    def __post_init__(self):
        super().__init__()

    def execute(self, agent, dt: float) -> bool:
        if self.status == ActionStatus.NOT_STARTED:
            self.status = ActionStatus.IN_PROGRESS
            self.start_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            self._log_start(agent)

            self._reachable = agent.move_end_effector(
                self.target_position,
                self.target_orientation,
                self.end_effector_link,
                self.max_force,
                self.tolerance,
            )
            if not self._reachable:
                self._log.warning("IK target is not reachable (best-effort movement will proceed)")

        # All joints reached their IK-solved targets (use default joint tolerance)
        if agent.are_joints_at_targets():
            # Also check EE Cartesian position
            ee_at_target = agent.are_ee_at_target(
                self.target_position,
                target_orientation=self.target_orientation,
                end_effector_link=self.end_effector_link,
                tolerance=self.tolerance,
            )
            self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            if self._reachable and ee_at_target:
                self.status = ActionStatus.COMPLETED
                self._log_end()
            else:
                self.status = ActionStatus.FAILED
                self._log_failure("IK target was not reachable")
            return True

        return False

    def reset(self):
        super().reset()
        self._reachable = False


@dataclass
class WaitAction(Action):
    """
    Wait for a specified duration.

    Useful for simulating charging, loading, or other time-based operations.

    Args:
        duration: Time to wait in seconds
        action_type: Type of wait action (default: "idle")
                    Options: "idle", "charge", "loading", "custom"
        show_indicator: Show progress indicator (default: True)
        indicator_color: RGB color for indicator [r, g, b] (default: yellow)

    Example::

        # Wait for 5 seconds (charging)
        action = WaitAction(duration=5.0, action_type="charge")
        agent.add_action(action)
    """

    duration: float
    action_type: str = "idle"

    # Internal state
    _elapsed_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        super().__init__()

    def execute(self, agent, dt: float) -> bool:
        """Execute wait action."""
        if self.status == ActionStatus.NOT_STARTED:
            self.status = ActionStatus.IN_PROGRESS
            self.start_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            self._elapsed_time = 0.0

            self._log_start(agent)

        # Update elapsed time
        self._elapsed_time += dt

        # Check if duration has elapsed
        if self._elapsed_time >= self.duration:
            self.status = ActionStatus.COMPLETED
            self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            self._log_end()
            return True

        return False

    def reset(self):
        """Reset action state."""
        super().reset()
        self._elapsed_time = 0.0


@dataclass
class PickAction(Action):
    """
    Pick an object and attach it to the robot.

    Kinematic implementation:
    1. Move to approach pose (if specified or auto-calculated)
    2. Teleport object to attachment position
    3. Attach object using constraint

    Args:
        target_object_id: Specific object body ID to pick (optional)
        target_position: Pick from position - auto-select nearest pickable object (optional)
        search_radius: Search radius when using target_position (default: 0.5m)
        approach_pose: Explicit approach pose (optional, auto-calculated if None)
        approach_offset: Auto-calculate approach offset distance (default: 1.0m)
        pick_offset: Distance from target where pick is executed (default: 0.0m)
        attach_link: Link to attach to - can be index (int) or name (str). -1 or "base_link" for base link (default: -1)
        attach_relative_pose: Position and orientation offset in link's frame as Pose object

    Note:
        Path visualization is controlled by agent.path_visualize setting.

    Example::

        # Pick specific object with position offset (using link index)
        action = PickAction(
            target_object_id=pallet.body_id,
            approach_offset=1.0,
            pick_offset=0.3,
            attach_link=-1,  # Base link
            attach_relative_pose=Pose.from_euler(0.6, 0, -0.2, roll=np.pi/2, pitch=0, yaw=0)
        )
        agent.add_action(action)

        # Pick using link name
        action = PickAction(
            target_object_id=pallet.body_id,
            attach_link="sensor_mast",  # Attach to named link
            attach_relative_pose=Pose.from_euler(0, 0, 0.5)
        )
        agent.add_action(action)

        # Pick nearest object at location
        action = PickAction(
            target_position=[5, 0, 0.1],
            search_radius=0.5
        )
        agent.add_action(action)
    """

    # Target specification (one of these must be provided)
    target_object_id: Optional[int] = None
    target_position: Optional[List[float]] = None
    search_radius: float = 0.5
    use_2d_search: bool = True  # XY-only distance for find_nearest (items on tables/shelves)

    # Approach configuration
    use_approach: bool = True  # Whether to use approach phase
    approach_pose: Optional[Pose] = None  # Explicit approach pose (if None, auto-calculated)
    approach_offset: float = 1.0  # Distance from target for approach pose (used if approach_pose is None)
    pick_offset: float = 0.0  # Distance from target where pick is executed (0 = at target position)
    move_skip_threshold: float = DEFAULT_MOVE_SKIP_THRESHOLD  # Skip move when closer than this (m)

    # Attachment configuration
    attach_link: Union[int, str] = -1  # Link index or name to attach to (-1 or "base_link" for base link)
    attach_relative_pose: Pose = field(
        default_factory=lambda: Pose(position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0])
    )  # Relative pose of attached object in parent frame

    # Internal state
    _phase: PickPhase = field(default=PickPhase.INIT, init=False)
    _attach_link_index: int = field(default=-1, init=False)  # Resolved link index
    _target_object: Optional["SimObject"] = field(default=None, init=False)
    _approach_action: Optional[MoveAction] = field(default=None, init=False)
    _forward_action: Optional[MoveAction] = field(default=None, init=False)
    _retreat_action: Optional[MoveAction] = field(default=None, init=False)
    _original_pose: Optional[Pose] = field(default=None, init=False)
    _pick_pose: Optional[Pose] = field(default=None, init=False)  # Pose where pick is executed

    # Joint control (optional — mutually exclusive with ee_target_position)
    joint_targets: Optional[Union[list, dict]] = None  # List or dict {joint_name: position}
    joint_tolerance: Optional[Union[float, list, dict]] = None
    joint_max_force: float = DEFAULT_MAX_FORCE
    _joint_action: Optional[JointAction] = field(default=None, init=False)  # Joint action for picking phase

    # EE pose control via IK (alternative to joint_targets)
    ee_target_position: Optional[List[float]] = None  # EE target [x,y,z] in world frame
    ee_target_orientation: Optional[tuple] = None  # quaternion [qx,qy,qz,qw]
    ee_end_effector_link: Union[int, str, None] = None  # For IK resolution
    ee_tolerance: float = DEFAULT_EE_TOLERANCE  # Cartesian EE tolerance (meters) for PoseAction
    continue_on_ik_failure: bool = True  # Continue pick even if IK target unreachable

    # Internal sub-action for EE pose control
    _pose_action: Optional[PoseAction] = field(default=None, init=False)

    def __post_init__(self):
        super().__init__()
        if self.joint_targets is not None and self.ee_target_position is not None:
            raise ValueError("Cannot specify both joint_targets and ee_target_position")
        # Resolve link name to index will be done during execution when agent is available
        self._attach_link_index = -1 if isinstance(self.attach_link, str) else self.attach_link

    def execute(self, agent, dt: float) -> bool:
        """Execute pick action."""
        if self.status == ActionStatus.NOT_STARTED:
            self.status = ActionStatus.IN_PROGRESS
            self.start_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            self._phase = PickPhase.INIT

            # Resolve link name to index using tools function
            self._attach_link_index = tools.resolve_link_index(agent.body_id, self.attach_link)

            self._log_start(agent)

            if isinstance(self.attach_link, str):
                self._log.debug(f"Resolved link '{self.attach_link}' to index {self._attach_link_index}")

        # Phase 1: Initialize - find target object and calculate poses
        if self._phase == PickPhase.INIT:
            if not self._find_target_object(agent):
                self.status = ActionStatus.FAILED
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                self._log_failure("Target object not found or not pickable", agent)
                return True

            self._log_phase_transition(PickPhase.INIT, agent)

            # Save original pose for retreat
            self._original_pose = agent.get_pose()

            # Calculate pick pose (where pick is executed)
            self._pick_pose = self._calculate_pick_pose(agent)

            obj_pos = self._target_object.get_pose().position
            pick_pos = self._pick_pose.position
            self._log.info(f"Object at {obj_pos[:2]}, pick pose at {pick_pos[:2]}")

            if self.use_approach:
                # Calculate or use approach pose
                if self.approach_pose is None:
                    self.approach_pose = self._calculate_approach_pose(agent)

                approach_pos = self.approach_pose.position
                self._log.info(f"Approach pose at {approach_pos[:2]} (offset={self.approach_offset:.3f}m)")

                # Create move action for approach (move forward)
                approach_path = Path([self.approach_pose])
                self._approach_action = MoveAction(path=approach_path, auto_approach=True, final_orientation_align=True)
                self._phase = PickPhase.APPROACHING
                self._log_phase_transition(PickPhase.APPROACHING, agent)
                self._log.info(f"Target object {self._target_object.body_id} found, approaching...")
            else:
                # Skip approach, go directly to moving to pick pose
                self._phase = PickPhase.MOVING_TO_PICK
                self._log_phase_transition(PickPhase.MOVING_TO_PICK, agent)
                self._log.info(f"Target object {self._target_object.body_id} found, moving to pick position...")

        # Phase 2: Approach target (move forward to approach_pose)
        if self._phase == PickPhase.APPROACHING:
            if self._approach_action.execute(agent, dt):
                self._phase = PickPhase.MOVING_TO_PICK
                self._log_phase_transition(PickPhase.MOVING_TO_PICK, agent)
                self._log.info("Reached approach pose, moving to pick position...")

        # Phase 3: Move forward to pick pose
        if self._phase == PickPhase.MOVING_TO_PICK:
            # Fixed-base robots (arms) don't need base movement — skip directly to picking
            if agent.use_fixed_base:
                self._phase = PickPhase.PICKING
                self._log_phase_transition(PickPhase.PICKING, agent)
                self._log.info("Fixed-base robot, skipping base movement, picking...")
            else:
                # Create forward action once to move from current position to pick pose
                if self._forward_action is None:
                    current_pos = agent.get_pose().position
                    pick_pos = self._pick_pose.position
                    distance = np.linalg.norm(np.array(pick_pos[:2]) - np.array(current_pos[:2]))

                    # Skip movement when already at pick pose (e.g. pick_offset == distance to item)
                    if distance < self.move_skip_threshold:
                        self._log.info(
                            f"Already at pick pose (distance={distance:.4f}m < {self.move_skip_threshold}m), skipping movement"
                        )
                        self._phase = PickPhase.PICKING
                        self._log_phase_transition(PickPhase.PICKING, agent)
                    else:
                        forward_path = Path([self._pick_pose])

                        self._forward_action = MoveAction(
                            path=forward_path,
                            auto_approach=True,
                            final_orientation_align=False,
                            direction=MovementDirection.FORWARD,
                        )
                        self._log.info(f"Moving forward {distance:.3f}m to pick pose...")

                if self._forward_action is not None and self._forward_action.execute(agent, dt):
                    self._phase = PickPhase.PICKING
                    self._log_phase_transition(PickPhase.PICKING, agent)
                    self._log.info("Reached pick position, picking...")

        # Phase 4: Pick object (with optional joint control)
        if self._phase == PickPhase.PICKING:
            # EE pose control via PoseAction (once)
            if self.ee_target_position is not None and self._pose_action is None and self._joint_action is None:
                self._pose_action = PoseAction(
                    target_position=self.ee_target_position,
                    target_orientation=self.ee_target_orientation,
                    end_effector_link=self.ee_end_effector_link,
                    max_force=self.joint_max_force,
                    tolerance=self.ee_tolerance,
                )

            # Execute PoseAction sub-action for EE control
            if self._pose_action is not None:
                if not self._pose_action.execute(agent, dt):
                    return False
                # PoseAction finished — check result
                if self._pose_action.status == ActionStatus.FAILED:
                    if not self.continue_on_ik_failure:
                        self.status = ActionStatus.FAILED
                        self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                        self._log_failure("IK target was not reachable for pick", agent)
                        return True
                    self._log.warning("IK target unreachable but continuing pick (continue_on_ik_failure=True)")

            # If joint_targets specified, execute joint action first
            elif self.joint_targets:
                if self._joint_action is None:
                    self._joint_action = JointAction(
                        target_joint_positions=self.joint_targets,
                        max_force=self.joint_max_force,
                        tolerance=self.joint_tolerance,
                    )
                if not self._joint_action.execute(agent, dt):
                    return False

            # Attach object — attach_object() handles the teleport internally,
            # placing the object at the correct world position derived from
            # parent link + attach_relative_pose.
            success = agent.attach_object(
                self._target_object, parent_link_index=self._attach_link_index, relative_pose=self.attach_relative_pose
            )

            if success:
                if self.use_approach:
                    # Create retreat action (move backward to approach pose)
                    current_pos = agent.get_pose().position
                    approach_pos = self.approach_pose.position
                    retreat_distance = np.linalg.norm(np.array(approach_pos[:2]) - np.array(current_pos[:2]))
                    self._log.info(
                        f"Successfully picked object {self._target_object.body_id}, retreating {retreat_distance:.3f}m..."
                    )

                    retreat_path = Path([self.approach_pose])
                    self._retreat_action = MoveAction(
                        path=retreat_path,
                        auto_approach=True,
                        final_orientation_align=False,  # Keep current orientation
                        direction=MovementDirection.BACKWARD,  # Move backward
                    )
                    self._phase = PickPhase.RETREATING
                    self._log_phase_transition(PickPhase.RETREATING, agent)
                else:
                    # No approach, complete immediately
                    self.status = ActionStatus.COMPLETED
                    self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                    self._log_end()
                    return True
            else:
                self.status = ActionStatus.FAILED
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                self._log_failure("Failed to attach object", agent)
                return True

        # Phase 5: Retreat to approach pose (move backward)
        if self._phase == PickPhase.RETREATING:
            if self._retreat_action.execute(agent, dt):
                self.status = ActionStatus.COMPLETED
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                self._log_end()
                return True

        return False

    def _find_target_object(self, agent) -> bool:
        """Find target object based on ID or position."""
        if self.target_object_id is not None:
            # Find by ID
            if agent.sim_core is None:
                return False

            for obj in agent.sim_core.sim_objects:
                if obj.body_id == self.target_object_id and obj.pickable:
                    self._target_object = obj
                    return True
            return False

        elif self.target_position is not None:
            # Find nearest pickable object within search radius of target_position
            if agent.sim_core is None:
                return False

            nearest_obj = tools.find_nearest(
                agent.sim_core.sim_objects,
                position=self.target_position,
                search_radius=self.search_radius,
                predicate=lambda o: o.pickable and not o.is_attached(),
                exclude=agent,
                use_2d=self.use_2d_search,
            )
            if nearest_obj is not None:
                self._target_object = nearest_obj
                return True
            return False

        else:
            return False

    def _calculate_pick_pose(self, agent) -> Pose:
        """Calculate pose where pick is executed (pick_offset away from target)."""
        obj_pose = self._target_object.get_pose()
        return tools.calculate_offset_pose(
            target_position=obj_pose.position,
            current_position=agent.get_pose().position,
            offset=self.pick_offset,
            keep_height=True,
        )

    def _calculate_approach_pose(self, agent) -> Pose:
        """Calculate approach pose based on target object position."""
        obj_pose = self._target_object.get_pose()
        return tools.calculate_offset_pose(
            target_position=obj_pose.position,
            current_position=agent.get_pose().position,
            offset=self.approach_offset,
            keep_height=True,
        )

    def reset(self):
        """Reset action state."""
        super().reset()
        self._phase = PickPhase.INIT
        self._target_object = None
        self._approach_action = None
        self._forward_action = None
        self._retreat_action = None
        self._original_pose = None
        self._pick_pose = None
        self._joint_action = None
        self._pose_action = None


@dataclass
class DropAction(Action):
    """
    Drop an attached object at a specified location.

    Kinematic implementation:
    1. Move to approach pose (if specified or auto-calculated)
    2. Detach object
    3. Teleport object to drop position

    Args:
        drop_pose: Where to drop the object (position and orientation).
                  Orientation defaults to no rotation ``[0, 0, 0, 1]``.
        approach_pose: Explicit approach pose (optional, auto-calculated if None)
        approach_offset: Auto-calculate approach offset distance (default: 0.2m)
        place_gently: Place at exact position vs drop from height (default: True)
        drop_height: Height above drop_pose to release if not gentle (default: 0.1m)
        target_object_id: Specific object to drop (None = drop first attached)

    Example::

        # Drop at specific location
        action = DropAction(
            drop_pose=Pose(position=[10, 5, 0.1]),
            place_gently=True
        )
        agent.add_action(action)

        # Drop with specific orientation
        action = DropAction(
            drop_pose=Pose(position=[10, 5, 0.1], orientation=[0, 0, 0.7, 0.7]),
        )

    Note:
        Path visualization is controlled by agent.path_visualize setting.
    """

    # Drop location (required)
    drop_pose: Pose

    # Approach configuration
    use_approach: bool = True  # Whether to use approach phase
    approach_pose: Optional[Pose] = None  # Explicit approach pose (if None, auto-calculated)
    approach_offset: float = 1.0  # Distance from target for approach pose (used if approach_pose is None)
    drop_offset: float = 0.0  # Distance from drop_pose where drop is executed (0 = at drop_pose)
    move_skip_threshold: float = DEFAULT_MOVE_SKIP_THRESHOLD  # Skip move when closer than this (m)

    # Drop behavior
    place_gently: bool = True
    drop_height: float = 0.1

    # Target object (None = drop first attached object)
    target_object_id: Optional[int] = None

    # Relative drop pose: if set, the object is placed at its current
    # (pre-detach) pose transformed by this offset instead of being
    # teleported to ``drop_pose``.  Useful for EE-attached objects
    # where the absolute world position is hard to predict.
    drop_relative_pose: Optional[Pose] = None

    # Internal state
    _phase: DropPhase = field(default=DropPhase.INIT, init=False)
    _target_object: Optional["SimObject"] = field(default=None, init=False)
    _approach_action: Optional[MoveAction] = field(default=None, init=False)
    _forward_action: Optional[MoveAction] = field(default=None, init=False)
    _retreat_action: Optional[MoveAction] = field(default=None, init=False)
    _original_pose: Optional[Pose] = field(default=None, init=False)
    _drop_pose: Optional[Pose] = field(default=None, init=False)  # Pose where drop is executed

    # Joint control (optional — mutually exclusive with ee_target_position)
    joint_targets: Optional[Union[list, dict]] = None  # List or dict {joint_name: position}
    joint_tolerance: Optional[Union[float, list, dict]] = None
    joint_max_force: float = DEFAULT_MAX_FORCE
    _joint_action: Optional[JointAction] = field(default=None, init=False)  # Joint action for dropping phase

    # EE pose control via IK (alternative to joint_targets)
    ee_target_position: Optional[List[float]] = None  # EE target [x,y,z] in world frame
    ee_target_orientation: Optional[tuple] = None  # quaternion [qx,qy,qz,qw]
    ee_end_effector_link: Union[int, str, None] = None  # For IK resolution
    ee_tolerance: float = DEFAULT_EE_TOLERANCE  # Cartesian EE tolerance (meters) for PoseAction
    continue_on_ik_failure: bool = True  # Continue drop even if IK target unreachable

    # Internal sub-action for EE pose control
    _pose_action: Optional[PoseAction] = field(default=None, init=False)

    def __post_init__(self):
        super().__init__()
        if self.joint_targets is not None and self.ee_target_position is not None:
            raise ValueError("Cannot specify both joint_targets and ee_target_position")

    def execute(self, agent, dt: float) -> bool:
        """Execute drop action."""
        if self.status == ActionStatus.NOT_STARTED:
            self.status = ActionStatus.IN_PROGRESS
            self.start_time = agent.sim_core.sim_time if agent.sim_core else 0.0
            self._phase = DropPhase.INIT

            self._log_start(agent)

        # Phase 1: Initialize - find target object to drop and calculate poses
        if self._phase == DropPhase.INIT:
            if not self._find_attached_object(agent):
                self.status = ActionStatus.FAILED
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                self._log_failure("No attached object found to drop", agent)
                return True

            self._log_phase_transition(DropPhase.INIT, agent)

            # Save original pose for retreat
            self._original_pose = agent.get_pose()

            # Calculate drop pose (where drop is executed)
            self._drop_pose = self._calculate_drop_pose(agent)

            target_pos = self.drop_pose.position
            drop_pos = self._drop_pose.position
            self._log.info(f"Target at {target_pos[:2]}, drop pose at {drop_pos[:2]}")

            if self.use_approach:
                # Calculate or use approach pose
                if self.approach_pose is None:
                    self.approach_pose = self._calculate_approach_pose(agent)

                approach_pos = self.approach_pose.position
                self._log.info(f"Approach pose at {approach_pos[:2]} (offset={self.approach_offset:.3f}m)")

                # Create move action for approach (move forward)
                approach_path = Path([self.approach_pose])
                self._approach_action = MoveAction(path=approach_path, auto_approach=True, final_orientation_align=True)
                self._phase = DropPhase.APPROACHING
                self._log_phase_transition(DropPhase.APPROACHING, agent)
                self._log.info("Approaching drop position...")
            else:
                # Skip approach, go directly to moving to drop pose
                self._phase = DropPhase.MOVING_TO_DROP
                self._log_phase_transition(DropPhase.MOVING_TO_DROP, agent)
                self._log.info("Moving to drop position...")

        # Phase 2: Approach drop position (move forward to approach_pose)
        if self._phase == DropPhase.APPROACHING:
            if self._approach_action.execute(agent, dt):
                self._phase = DropPhase.MOVING_TO_DROP
                self._log_phase_transition(DropPhase.MOVING_TO_DROP, agent)
                self._log.info("Reached approach pose, moving to drop position...")

        # Phase 3: Move forward to drop pose
        if self._phase == DropPhase.MOVING_TO_DROP:
            # Fixed-base robots (arms) don't need base movement — skip directly to dropping
            if agent.use_fixed_base:
                self._phase = DropPhase.DROPPING
                self._log_phase_transition(DropPhase.DROPPING, agent)
                self._log.info("Fixed-base robot, skipping base movement, dropping...")
            else:
                # Create forward action once to move from current position to drop pose
                if self._forward_action is None:
                    current_pos = agent.get_pose().position
                    drop_pos = self._drop_pose.position
                    distance = np.linalg.norm(np.array(drop_pos[:2]) - np.array(current_pos[:2]))

                    # Skip movement when already at drop pose (e.g. drop_offset == distance to ingestor)
                    if distance < self.move_skip_threshold:
                        self._log.info(
                            f"Already at drop pose (distance={distance:.4f}m < {self.move_skip_threshold}m), skipping movement"
                        )
                        self._phase = DropPhase.DROPPING
                        self._log_phase_transition(DropPhase.DROPPING, agent)
                    else:
                        forward_path = Path([self._drop_pose])
                        self._forward_action = MoveAction(
                            path=forward_path,
                            auto_approach=True,
                            final_orientation_align=False,
                            direction=MovementDirection.FORWARD,
                        )
                        self._log.info(f"Moving forward {distance:.3f}m to drop pose...")

                if self._forward_action is not None and self._forward_action.execute(agent, dt):
                    self._phase = DropPhase.DROPPING
                    self._log_phase_transition(DropPhase.DROPPING, agent)
                    self._log.info("Reached drop position, dropping...")

        # Phase 4: Drop object (with optional joint control)
        if self._phase == DropPhase.DROPPING:
            # EE pose control via PoseAction (once)
            if self.ee_target_position is not None and self._pose_action is None and self._joint_action is None:
                self._pose_action = PoseAction(
                    target_position=self.ee_target_position,
                    target_orientation=self.ee_target_orientation,
                    end_effector_link=self.ee_end_effector_link,
                    max_force=self.joint_max_force,
                    tolerance=self.ee_tolerance,
                )

            # Execute PoseAction sub-action for EE control
            if self._pose_action is not None:
                if not self._pose_action.execute(agent, dt):
                    return False
                # PoseAction finished — check result
                if self._pose_action.status == ActionStatus.FAILED:
                    if not self.continue_on_ik_failure:
                        self.status = ActionStatus.FAILED
                        self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                        self._log_failure("IK target was not reachable for drop", agent)
                        return True
                    self._log.warning("IK target unreachable but continuing drop (continue_on_ik_failure=True)")

            # If joint_targets specified, execute joint action first
            elif self.joint_targets:
                if self._joint_action is None:
                    self._joint_action = JointAction(
                        target_joint_positions=self.joint_targets,
                        max_force=self.joint_max_force,
                        tolerance=self.joint_tolerance,
                    )
                if not self._joint_action.execute(agent, dt):
                    return False

            # Detach object and teleport to final position.
            # detach_object() supports both absolute (drop_pose) and relative
            # (drop_relative_pose) modes internally.
            if self.drop_relative_pose is not None:
                success = agent.detach_object(self._target_object, drop_relative_pose=self.drop_relative_pose)
            else:
                # Absolute mode — apply drop_height adjustment if not placing gently
                drop_pos = self.drop_pose.position
                if self.place_gently:
                    final_drop = self.drop_pose
                else:
                    final_drop = Pose(
                        position=[drop_pos[0], drop_pos[1], drop_pos[2] + self.drop_height],
                        orientation=list(self.drop_pose.orientation),
                    )
                success = agent.detach_object(self._target_object, drop_pose=final_drop)

            if not success:
                self.status = ActionStatus.FAILED
                self._log_failure("Failed to detach object", agent)
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                self._log.info(f"Failed: {self.error_message}")
                return True

            # If not placing gently, give object a small downward velocity
            if not self.place_gently:
                p.resetBaseVelocity(self._target_object.body_id, linearVelocity=[0, 0, -0.5], angularVelocity=[0, 0, 0])

            if self.use_approach:
                # Create retreat action (move backward to approach pose)
                current_pos = agent.get_pose().position
                approach_pos = self.approach_pose.position
                retreat_distance = np.linalg.norm(np.array(approach_pos[:2]) - np.array(current_pos[:2]))
                self._log.info(
                    f"Successfully dropped object {self._target_object.body_id}, retreating {retreat_distance:.3f}m..."
                )

                # Use visualization settings from action or agent defaults

                retreat_path = Path([self.approach_pose])
                self._retreat_action = MoveAction(
                    path=retreat_path,
                    auto_approach=True,
                    final_orientation_align=False,  # Keep current orientation
                    direction=MovementDirection.BACKWARD,  # Move backward
                )
                self._phase = DropPhase.RETREATING
                self._log_phase_transition(DropPhase.RETREATING, agent)
            else:
                # No approach, complete immediately
                self.status = ActionStatus.COMPLETED
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                self._log_end()
                return True

        # Phase 5: Retreat to approach pose (move backward)
        if self._phase == DropPhase.RETREATING:
            if self._retreat_action.execute(agent, dt):
                self.status = ActionStatus.COMPLETED
                self.end_time = agent.sim_core.sim_time if agent.sim_core else 0.0
                self._log_end()
                return True

        return False

    def _find_attached_object(self, agent) -> bool:
        """Find attached object to drop."""
        attached = agent.get_attached_objects()

        if len(attached) == 0:
            return False

        if self.target_object_id is not None:
            # Find specific object
            for obj in attached:
                if obj.body_id == self.target_object_id:
                    self._target_object = obj
                    return True
            return False
        else:
            # Drop first attached object
            self._target_object = attached[0]
            return True

    def _calculate_drop_pose(self, agent) -> Pose:
        """Calculate pose where drop is executed (drop_offset away from drop_pose)."""
        return tools.calculate_offset_pose(
            target_position=self.drop_pose.position,
            current_position=agent.get_pose().position,
            offset=self.drop_offset,
            keep_height=True,
        )

    def _calculate_approach_pose(self, agent) -> Pose:
        """Calculate approach pose based on drop pose."""
        return tools.calculate_offset_pose(
            target_position=self.drop_pose.position,
            current_position=agent.get_pose().position,
            offset=self.approach_offset,
            keep_height=True,
        )

    def reset(self):
        """Reset action state."""
        super().reset()
        self._phase = DropPhase.INIT
        self._target_object = None
        self._approach_action = None
        self._forward_action = None
        self._retreat_action = None
        self._original_pose = None
        self._drop_pose = None
        self._joint_action = None
        self._pose_action = None
