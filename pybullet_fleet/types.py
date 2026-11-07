"""
types.py
Common types and enumerations used across the package.

This module contains shared type definitions to avoid circular imports
and provide a single source of truth for common types.
"""

import dataclasses
import os
import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import Enum
from types import FrameType
from typing import Iterator, Optional


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
    """When in a simulation step a ``register_callback()`` callback or a plugin's ``on_step()`` runs.

    Each value is the name of the step event it subscribes to on
    ``sim.events`` (:attr:`SimEvents.PRE_STEP <pybullet_fleet.events.SimEvents.PRE_STEP>`,
    :attr:`SimEvents.STEP <pybullet_fleet.events.SimEvents.STEP>`,
    :attr:`SimEvents.POST_STEP <pybullet_fleet.events.SimEvents.POST_STEP>`);
    callbacks run after that event's handlers, and plugins after the callbacks.

    Attributes:
        PRE_STEP: At the start of the step, right after the ``pre_step``
            event's handlers and before the clock advances: ``sim_time`` is
            still the previous state's time, and nothing has moved yet. Use it
            to apply commands that this step should act on.
        STEP: After the objects have been updated, before their new poses are
            written to PyBullet and before the collision check.
        POST_STEP: At the end of the step, right after the ``post_step`` event,
            once the poses are written and the step's collision check has run.
    """

    PRE_STEP = "pre_step"
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
    - Clearance on the z axis (NORMAL_2D vs NORMAL_3D)
    - Movement type (static vs dynamic)
    - Collision detection enablement

    Attributes:
        NORMAL_3D: Normal collision detection (default)
                   - A pair is a candidate when its AABBs are within the collision clearance on every axis
                   - For dynamic/kinematic objects

        NORMAL_2D: Deprecated, use NORMAL_3D. As NORMAL_3D, without the clearance on the z axis (z is not ignored)
                   - A pair involving one needs real z overlap; x and y get the clearance as for NORMAL_3D
                   - Differs from NORMAL_3D only with a non-zero clearance (CLOSEST_POINTS / HYBRID)
                   - Setting it emits a FutureWarning; it still works and will be removed in a future release

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

        # Static structure
        wall = SimObject(..., collision_mode=CollisionMode.STATIC)

        # Ghost object (no collision)
        decoration = SimObject(..., collision_mode=CollisionMode.DISABLED)
    """

    NORMAL_3D = "normal_3d"
    NORMAL_2D = "normal_2d"
    STATIC = "static"
    DISABLED = "disabled"


_NORMAL_2D_DEPRECATION_MESSAGE = (
    "CollisionMode.NORMAL_2D is deprecated and will be removed in a future release. "
    "It only removes the collision clearance (collision_margin) on the z axis: a pair with a NORMAL_2D body "
    "must really overlap in z. Use CollisionMode.NORMAL_3D instead. To keep no clearance on the z axis, set "
    "collision_margin=(m, m, 0) for your margin m; it reports what NORMAL_2D did, for every pair, not only those "
    "with this body. Otherwise NORMAL_3D alone differs only in reporting near misses within the clearance above "
    "or below too (none with CONTACT_POINTS or a zero margin)."
)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
_EXAMPLES_DIR = os.path.join(_PACKAGE_DIR, "examples") + os.sep
_normal_2d_warning_suppressed: ContextVar[bool] = ContextVar("_normal_2d_warning_suppressed", default=False)
# The class whose next NORMAL_2D object construction is not to warn (see
# _spawning_from_warned_params); objects spawned while building it still warn.
_normal_2d_skip_next_object: ContextVar[Optional[type]] = ContextVar("_normal_2d_skip_next_object", default=None)


def _is_internal_frame(frame: FrameType) -> bool:
    filename = frame.f_code.co_filename
    if filename == "<string>":
        # The __init__ that dataclasses generates (dataclasses.replace() calls it) is
        # compiled from a string; other code there (python -c, exec) is the caller's.
        return frame.f_code.co_name == "__init__" and dataclasses.is_dataclass(frame.f_locals.get("self"))
    if filename == dataclasses.__file__:
        return True
    return filename.startswith(_PACKAGE_DIR) and not filename.startswith(_EXAMPLES_DIR)


def _warn_if_normal_2d(mode: object) -> None:
    """Emit the NORMAL_2D ``FutureWarning`` if *mode* is NORMAL_2D.

    The warning is attributed to the first caller outside the package (the
    bundled examples count as outside). The default filter keeps one warning
    per line of that code, so a line spawning thousands of objects warns once;
    a per-process flag would hide other lines and break ``pytest.warns``.
    """
    if mode is not CollisionMode.NORMAL_2D or _normal_2d_warning_suppressed.get():
        return
    frame = sys._getframe(1)
    level = 2
    while frame is not None and _is_internal_frame(frame):
        frame = frame.f_back
        level += 1
    warnings.warn(_NORMAL_2D_DEPRECATION_MESSAGE, FutureWarning, stacklevel=level if frame is not None else 2)


def _warn_if_new_object_is_normal_2d(obj: object, mode: object) -> None:
    """:func:`_warn_if_normal_2d` for an object's ``__init__``.

    Silent for the one object that :func:`_spawning_from_warned_params` is
    building; later objects, including those it spawns, warn as usual.
    """
    if mode is CollisionMode.NORMAL_2D:
        skip = _normal_2d_skip_next_object.get()
        if skip is not None and isinstance(obj, skip):
            _normal_2d_skip_next_object.set(None)
            return
    _warn_if_normal_2d(mode)


class _spawning_from_warned_params:
    """Warn once about NORMAL_2D spawn params, then spawn a *cls* from them.

    Building the params warned already, unless their mode was set to NORMAL_2D
    afterwards; then entering this warns, attributed to the caller. Inside the
    block the first NORMAL_2D object of *cls* -- the one being spawned -- does
    not warn again; anything else spawned there (by an overridden factory, a
    controller or a plugin) does. A class, not a generator: contextlib's frame
    would otherwise take the warning's attribution.
    """

    def __init__(self, cls: type, spawn_params: object) -> None:
        self._cls = cls
        self._spawn_params = spawn_params
        self._token: Optional[Token] = None

    def __enter__(self) -> None:
        mode = getattr(self._spawn_params, "collision_mode", None)
        if mode is not CollisionMode.NORMAL_2D:
            return
        if not getattr(self._spawn_params, "_normal_2d_warned", False):
            _warn_if_normal_2d(mode)
        self._token = _normal_2d_skip_next_object.set(self._cls)

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _normal_2d_skip_next_object.reset(self._token)
            self._token = None


@contextmanager
def _normal_2d_warning_already_given() -> Iterator[None]:
    """Do not warn about NORMAL_2D inside this block.

    For internal steps that pass on a mode the user has already been warned
    about (spawning from spawn params) or that must keep NORMAL_2D themselves.
    """
    token = _normal_2d_warning_suppressed.set(True)
    try:
        yield
    finally:
        _normal_2d_warning_suppressed.reset(token)


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
