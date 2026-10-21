"""
types.py
Common types and enumerations used across the package.

This module contains shared type definitions to avoid circular imports
and provide a single source of truth for common types.
"""

from enum import Enum


class MotionMode(str, Enum):
    """
    Motion mode for robot movement control.

    Attributes:
        OMNIDIRECTIONAL: Robot can move in any direction without rotating first (holonomic)
        DIFFERENTIAL: Robot must rotate to face target, then move forward (non-holonomic)
    """

    OMNIDIRECTIONAL = "omnidirectional"
    DIFFERENTIAL = "differential"


class PosePhase(str, Enum):
    """
    Phase for TPI pose control (shared by all kinematic controllers).

    Attributes:
        ROTATE: Robot is rotating in place (slerp TPI)
        FORWARD: Robot is moving forward in a straight line (distance TPI)
    """

    ROTATE = "rotate"
    FORWARD = "forward"


class MovementDirection(str, Enum):
    """
    Movement direction for robot motion.

    Note: This is only used in differential drive mode (MotionMode.DIFFERENTIAL).
          In omnidirectional mode, the robot always faces the movement direction.

    Attributes:
        FORWARD: Robot X+ axis points towards target, moves forward along path
        BACKWARD: Robot X- axis points towards target (X+ points away), moves along path
                 (robot appears to move "backward" relative to its X+ orientation)
        AUTO: Automatically choose FORWARD or BACKWARD based on the angle
              between the robot's current heading and the movement direction.
              Uses BACKWARD when the goal is more than 90° behind the robot.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    AUTO = "auto"


class ActionStatus(Enum):
    """
    Action execution status.

    Attributes:
        NOT_STARTED: Action has not begun execution
        IN_PROGRESS: Action is currently executing
        COMPLETED: Action completed successfully
        FAILED: Action failed to complete
        CANCELLED: Action was cancelled before completion
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SpatialHashCellSizeMode(Enum):
    """
    Collision detection spatial hash cell size calculation mode.

    Determines how the spatial hash grid cell size is calculated for
    efficient broad-phase collision detection.

    Attributes:
        CONSTANT: Fixed cell size provided by user (fastest, requires manual tuning)
        AUTO_ADAPTIVE: Recalculate on object add/remove (adaptive, some overhead)
        AUTO_INITIAL: Calculate once at simulation start (default, balanced performance)
    """

    CONSTANT = "constant"
    AUTO_ADAPTIVE = "auto_adaptive"
    AUTO_INITIAL = "auto_initial"


class DoorState(str, Enum):
    """Door operational state.

    Attributes:
        CLOSED: Door is fully closed.
        OPENING: Door is transitioning toward the open position.
        OPEN: Door is fully open.
        CLOSING: Door is transitioning toward the closed position.
    """

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"


class CallbackPhase(str, Enum):
    """When in a simulation step a ``register_callback()`` callback runs.

    Attributes:
        STEP: After the objects have been updated, before their new poses are
            written to PyBullet and before the collision check.
        POST_STEP: At the end of the step, right after the ``post_step`` event,
            once the poses are written and the step's collision check has run.
    """

    STEP = "step"
    POST_STEP = "post_step"


class GoalReplacement(str, Enum):
    """What a moving agent does with a new goal or path it cannot reach from its current motion.

    A new path set while the agent is moving continues that motion when the
    straight-line trajectory to its first waypoint exists from the current
    velocity within the configured limits: the goal lies on the line of
    motion, ahead or behind, and the agent can still stop at it. Otherwise --
    a goal inside the braking distance, off the line of motion, or one that
    needs a turn in place -- this setting decides. ``WAIT_COMPLETE`` alone
    also holds a path the agent could continue into: the current path has
    priority. Only the latest held path is kept; dropping one is logged.

    Attributes:
        BRAKE: Accept the path; brake along the current motion at the
            configured deceleration, then start the path from rest.
        REJECT: Refuse the path (``set_path`` returns ``False``); the current
            motion runs on. The refusal is temporary: the same path is taken
            once the agent is at rest.
        WAIT_COMPLETE: Accept every path set while a path is in progress;
            finish the current one (reach its goal and stop), then start the
            new path from rest. With no path in progress (a velocity
            command) it brakes like ``BRAKE``.
        IMMEDIATE: Accept the path and start it from rest at once, dropping the
            current velocity in a single step -- the only mode that does not
            respect the configured deceleration.
    """

    BRAKE = "brake"
    REJECT = "reject"
    WAIT_COMPLETE = "wait_complete"
    IMMEDIATE = "immediate"


class ControllerMode(str, Enum):
    """Internal mode for KinematicController state machine.

    Attributes:
        IDLE: No active command — zero velocity, no pose target.
        VELOCITY: Processing cmd_vel input (body-frame velocity).
        POSE: Processing goal_pose / path via TPI trajectory.
    """

    IDLE = "idle"
    VELOCITY = "velocity"
    POSE = "pose"


class CollisionMode(Enum):
    """
    Collision detection mode for individual objects.

    Determines how an object participates in collision detection:
    - Spatial dimension (2D vs 3D)
    - Movement type (static vs dynamic)
    - Collision detection enablement

    Attributes:
        NORMAL_3D: Normal 3D collision detection (default)
                   - Full 3D AABB overlap check
                   - Checks all 27 spatial hash neighbors (3x3x3)
                   - For dynamic/kinematic objects that move in 3D space

        NORMAL_2D: 2D collision detection (Z-axis ignored)
                   - 2D AABB overlap check (X, Y only)
                   - Checks only 9 spatial hash neighbors (3x3x1, Z=0)
                   - For ground vehicles, 2D robots

        STATIC: Static object mode (never moves, but has collision)
                - Registered in collision system during spawn
                - Never updates AABB/spatial grid (performance optimization)
                - Examples: walls, shelves, structures

        DISABLED: Collision detection completely disabled (ghost object)
                  - No AABB calculation
                  - No spatial grid registration
                  - No collision checking
                  - PyBullet physics collision also disabled
                  - Examples: decorative objects, visual markers, UI elements

    Note: The actual collision filtering (e.g., ignore_static) is done at
          core_simulation level, not in this mode definition.

    Example:
        # Normal 3D robot
        robot = SimObject(..., collision_mode=CollisionMode.NORMAL_3D)

        # 2D ground vehicle
        vehicle = SimObject(..., collision_mode=CollisionMode.NORMAL_2D)

        # Static structure
        wall = SimObject(..., collision_mode=CollisionMode.STATIC)

        # Ghost object (no collision)
        decoration = SimObject(..., collision_mode=CollisionMode.DISABLED)
    """

    NORMAL_3D = "normal_3d"
    NORMAL_2D = "normal_2d"
    STATIC = "static"
    DISABLED = "disabled"


class CollisionDetectionMethod(Enum):
    """
    Collision detection method for narrow-phase collision checking.

    Determines which PyBullet API to use for detecting actual contacts
    between objects after broad-phase filtering (AABB + spatial hashing).

    Design Philosophy (from design document):
    - Physics OFF (kinematics): Use CLOSEST_POINTS for safety margin detection
    - Physics ON (physics): Use CONTACT_POINTS for actual contact logging

    Attributes:
        CLOSEST_POINTS: Use getClosestPoints() - distance-based, kinematics-safe (RECOMMENDED DEFAULT)
                       Best for: kinematics motion, safety clearance, collision avoidance
                       Works with: Physics ON/OFF, stable with resetBasePositionAndOrientation

        CONTACT_POINTS: Use getContactPoints() - physics contact manifold (PHYSICS MODE).
                        Best for physics simulation, actual contact logging, and debugging.
                        Requires an explicit collision-detection pass; stepSimulation() is
                        the normal physics path. Cache/solver-dependent for kinematic pairs.

        HYBRID: Use getContactPoints when either object uses physics; use
                getClosestPoints only for kinematic-kinematic pairs (ADVANCED).
                Best for mixed physics/kinematics with different detection needs.

    Recommended defaults:
        - physics_enabled=False → CLOSEST_POINTS (default)
        - physics_enabled=True  → CONTACT_POINTS or HYBRID
    """

    CLOSEST_POINTS = "closest_points"  # Default for kinematics
    CONTACT_POINTS = "contact_points"  # For physics simulation
    HYBRID = "hybrid"  # Advanced mixed mode
