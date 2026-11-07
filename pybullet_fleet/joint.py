"""Per-joint state for kinematically driven URDF joints.

:class:`KinematicJoint` holds what :meth:`Agent._update_kinematic_joints`
tracks for one joint. It used to live in five dictionaries keyed by joint
index -- position, last target, motion profile, speed and trajectory -- which
meant five lookups per joint per step, five places to keep in step whenever a
joint was reset or restored, and nowhere to read to find out what one joint
was doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

from pybullet_fleet._tpi import PhaseProfile, extract_profile
from pybullet_fleet._tpi_state import capture_tpi_state, restore_tpi_state


@dataclass
class KinematicJoint:
    """The state of one kinematically driven joint.

    The fields fall into three groups.

    *Command*: :attr:`position` is where the joint is, :attr:`target` where it
    was last told to go. A target persists after arrival, so that
    ``are_joints_at_targets()`` can still be asked about it.

    *Configuration*: :attr:`max_velocity`, :attr:`max_accel` and
    :attr:`max_decel`, set through :meth:`Agent.set_joint_motion_profile`.
    Each is ``None`` until given one, and :attr:`max_velocity` then falls back
    to the URDF's ``<limit velocity>``.

    *Execution*: :attr:`speed`, :attr:`trajectory`, :attr:`trajectory_target`,
    :attr:`elapsed` and :attr:`braking` belong to the update loop. They say how the joint is
    getting to its target right now, and mean nothing to anyone outside a
    step.
    """

    index: int

    position: float = 0.0
    target: Optional[float] = None

    max_velocity: Optional[float] = None
    max_accel: Optional[float] = None
    max_decel: Optional[float] = None

    #: Signed: the sign is the direction of travel. Carried between steps, so
    #: a target replaced mid-travel becomes the next trajectory's ``v0``.
    speed: float = 0.0
    #: The solved ``TwoPointInterpolation``, typed loosely to keep the solver
    #: out of this module's imports.
    trajectory: Optional[Any] = None
    #: The target :attr:`trajectory` was solved for. A different target makes
    #: it stale, which is how the update loop knows to re-plan.
    trajectory_target: Optional[float] = None
    #: Shedding speed with no trajectory, because none could be solved for
    #: this target from this speed. Stays true through the step where the
    #: speed reaches exactly zero short of the target, which is what keeps
    #: that step counted as motion.
    braking: bool = False
    #: Time :attr:`trajectory` has been followed, advanced by ``dt`` each step;
    #: the trajectory starts at ``t0 = 0`` and is evaluated at this time.
    #: Counted per joint rather than read off a clock, so a checkpoint carries
    #: it exactly and a restored joint repeats the same steps.
    elapsed: float = 0.0
    #: :attr:`trajectory` as closed-form phase scalars, and the trajectory
    #: they were read from. Derived, so neither is compared or captured;
    #: :meth:`clear_trajectory` drops both with the trajectory.
    _profile: Optional[PhaseProfile] = field(default=None, repr=False, compare=False)
    _profile_of: Optional[Any] = field(default=None, repr=False, compare=False)

    def capture_execution(self, name: str) -> dict:
        """Represent this joint's command and trajectory at a completed step.

        ``t0_offset`` is when the trajectory started, relative to this step:
        minus :attr:`elapsed`, which a restore reads back exactly.
        """
        trajectory = None
        if self.trajectory is not None:
            trajectory = capture_tpi_state(self.trajectory)
            trajectory["t0_offset"] = trajectory.pop("t0") - self.elapsed
            trajectory["target"] = self.trajectory_target
        return {
            "name": name,
            "position": float(self.position),
            "target": self.target,
            "max_velocity": self.max_velocity,
            "max_accel": self.max_accel,
            "max_decel": self.max_decel,
            "speed": float(self.speed),
            "braking": self.braking,
            "trajectory": trajectory,
        }

    @classmethod
    def from_execution(cls, index: int, state: dict) -> "KinematicJoint":
        """Validate and rebuild one joint without changing the live robot."""

        def finite(value: object) -> bool:
            return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

        position = state.get("position")
        target = state.get("target")
        if not finite(position) or (target is not None and not finite(target)):
            raise ValueError("Joint checkpoint contains a nonfinite or invalid value")
        limits = [state.get(key) for key in ("max_velocity", "max_accel", "max_decel")]
        if any(value is not None and (not finite(value) or value <= 0) for value in limits):
            raise ValueError("Joint checkpoint contains invalid motion limits")
        max_velocity, max_accel, max_decel = limits
        if max_decel is not None and max_accel is None:
            raise ValueError("Joint checkpoint has deceleration without acceleration")
        speed = state.get("speed")
        braking = state.get("braking")
        if not finite(speed) or type(braking) is not bool or (max_accel is None and (speed != 0 or braking)):
            raise ValueError("Joint checkpoint contains invalid execution state")
        trajectory_data = state.get("trajectory")
        trajectory = None
        trajectory_target = None
        elapsed = 0.0
        if trajectory_data is not None:
            required = {"p0", "pe", "v0", "ve", "vmax", "accel", "decel", "t0_offset", "target"}
            if not isinstance(trajectory_data, dict) or set(trajectory_data) != required or braking:
                raise ValueError("Joint checkpoint contains invalid trajectory")
            if not finite(trajectory_data["t0_offset"]) or not finite(trajectory_data["target"]):
                raise ValueError("Joint checkpoint contains nonfinite trajectory metadata")
            if target is None or trajectory_data["target"] != target:
                raise ValueError("Joint checkpoint trajectory target differs from joint target")
            if max_accel is None:
                raise ValueError("Unconfigured joint cannot have a ramp trajectory")
            trajectory = restore_tpi_state(
                {
                    **{key: value for key, value in trajectory_data.items() if key not in {"t0_offset", "target"}},
                    "t0": 0.0,
                }
            )
            trajectory_target = float(trajectory_data["target"])
            elapsed = -float(trajectory_data["t0_offset"])
        return cls(
            index=index,
            position=float(position),
            target=None if target is None else float(target),
            max_velocity=max_velocity,
            max_accel=max_accel,
            max_decel=max_decel,
            speed=float(speed),
            trajectory=trajectory,
            trajectory_target=trajectory_target,
            braking=braking,
            elapsed=elapsed,
        )

    @property
    def is_ramped(self) -> bool:
        """Whether the joint accelerates rather than stepping to full speed."""
        return self.max_accel is not None

    @property
    def is_moving(self) -> bool:
        """Whether the joint is in motion, with a plan or without one.

        A joint carrying speed is moving even with no trajectory: changing a
        motion profile voids the trajectory it was solved against but not the
        speed, and the replacement is not planned until the next step.
        """
        return self.trajectory is not None or self.braking or bool(self.speed)

    def trajectory_profile(self) -> PhaseProfile:
        """:attr:`trajectory` as the scalars :func:`trapezoid_point` evaluates.

        Read once per trajectory, however it was set -- planned by the update
        loop or rebuilt from a checkpoint -- and kept until it changes.
        """
        trajectory = self.trajectory
        if trajectory is None:
            raise RuntimeError(f"Joint {self.index} has no trajectory")
        if self._profile_of is not trajectory:
            self._profile = extract_profile(trajectory)
            self._profile_of = trajectory
        return self._profile

    @property
    def decel(self) -> Optional[float]:
        """The braking rate, which follows the acceleration when unset."""
        return self.max_decel if self.max_decel is not None else self.max_accel

    def clear_motion(self) -> None:
        """Put the joint at rest: no speed, no plan, not braking."""
        self.speed = 0.0
        self.clear_trajectory()
        self.braking = False

    def clear_trajectory(self) -> None:
        """Void the plan but keep the speed.

        What a profile change calls for: the trajectory was solved against the
        old limits, but the speed is where the joint actually is. The update
        loop re-plans from it on the next step. :attr:`braking` is left alone
        for the same reason -- the next step recomputes it, and clearing it
        here would report a braking joint as stopped for one step.
        """
        self.trajectory = None
        self.trajectory_target = None
        self.elapsed = 0.0
        # The cached profile would otherwise keep the dropped solver alive.
        self._profile = None
        self._profile_of = None
