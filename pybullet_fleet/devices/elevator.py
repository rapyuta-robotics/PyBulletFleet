"""Elevator device — Agent subclass with prismatic Z joint and auto-attach.

Objects (agents, sim_objects) on the platform are automatically attached
before movement and detached on arrival, so they physically ride the
elevator without any external teleport logic.
"""

from __future__ import annotations

import pybullet as p

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from pybullet_fleet.action import JointAction
from pybullet_fleet.agent import (  # noqa: F401 - Agent/AgentSpawnParams/IKParams resolve inherited API type hints
    Agent,
    AgentSpawnParams,
    IKParams,
    validate_joint_motion_profile,
)
from pybullet_fleet.controller import Controller  # noqa: F401 - resolves inherited API type hints
from pybullet_fleet.config_utils import config_get_bool
from pybullet_fleet.controller_params import ControllerParams  # noqa: F401 - resolves inherited API type hints
from pybullet_fleet.devices.elevator_state_machine import (
    ElevatorRequestPolicy,
    ElevatorRequestResult,
    ElevatorState,
    ElevatorStateMachine,
)
from pybullet_fleet.logging_utils import get_lazy_logger
from pybullet_fleet.types import ActionStatus, _normal_2d_warning_already_given

if TYPE_CHECKING:
    from pybullet_fleet.sim_object import SimObject

logger = get_lazy_logger(__name__)


@dataclass
class ElevatorParams(AgentSpawnParams):
    """Spawn parameters for an Elevator device.

    Extends AgentSpawnParams with elevator-specific floor/joint config.

    Attributes:
        floors: Floor name → joint position mapping.
            Required — ``None`` raises TypeError in ``__post_init__``.
        initial_floor: Name of the floor the elevator starts at.
        joint_name: Name of the prismatic joint that moves the platform.
        platform_link: Name of the link with the platform collision box.
        request_policy: Handling for floor requests received while moving.
        boarding_height: Metres above the platform's top within which an
            object counts as on board when the cabin starts moving.  Keep it
            below the distance between two floors: an object waiting on the
            floor above the cabin would otherwise ride along, and arrive at
            the height it had above the platform.
        max_speed: Cabin speed in m/s.  ``None`` uses the URDF's
            ``<limit velocity="...">`` on ``joint_name``, which is where the
            speed otherwise has to be set -- so without this, two installations
            that differ only in how fast the cabin travels need two URDFs.
        max_accel: Cabin acceleration in m/s^2.  ``None`` keeps the cabin at
            full speed from the first step, as before.  Giving it makes the
            travel a trapezoid, which is what a real cabin does: 0.6 m at
            0.4 m/s takes 1.5 s flat and 1.9 s with 1.0 m/s^2 ramps.
        max_decel: Cabin deceleration.  Defaults to ``max_accel``.
        report_platform_contacts: Report a body standing on the platform as
            in collision with the elevator.  Off by default: a robot waiting
            on the platform before departure, or left there on arrival, rests
            on it, which is not a collision.  A body counts as standing on the
            platform when it is over the platform's footprint and its lowest
            point is not below the platform's mid-height; a body touching the
            platform from below (the cabin coming down onto it) is still
            reported.  Passengers attached for a ride are not affected: they
            follow ``SimulationParams.ignore_attached_collision``.  The
            rule leaves the body out of contact with the whole elevator, so
            it only applies when the platform link is the elevator's one
            collider: an elevator URDF with collider walls, a ceiling or
            doors reports bodies on the platform as if this were on (and
            logs a warning at spawn), so that driving into a wall is seen.
    """

    floors: Optional[Dict[str, float]] = None
    initial_floor: str = ""
    joint_name: str = "lift"
    platform_link: str = "platform"
    request_policy: ElevatorRequestPolicy | str = ElevatorRequestPolicy.REJECT
    boarding_height: float = 1.0
    max_speed: Optional[float] = None
    max_accel: Optional[float] = None
    max_decel: Optional[float] = None
    report_platform_contacts: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.floors is None:
            raise TypeError("ElevatorParams requires floors (Dict[str, float])")
        if not self.initial_floor:
            self.initial_floor = next(iter(self.floors), "")
        if self.boarding_height < 0:
            raise ValueError(f"ElevatorParams.boarding_height must not be negative, got {self.boarding_height}")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ElevatorParams":
        """Create ElevatorParams from a config dict.

        Delegates base-field parsing to ``AgentSpawnParams.from_dict``
        and adds elevator-specific fields (``floors``, ``initial_floor``,
        ``joint_name``, ``platform_link``, ...).
        """
        with _normal_2d_warning_already_given():  # cls(...) below warns about a deprecated collision mode
            base = AgentSpawnParams.from_dict(config)
        base_kwargs = {f.name: getattr(base, f.name) for f in fields(base)}
        return cls(
            **base_kwargs,
            floors=config.get("floors"),
            initial_floor=config.get("initial_floor", ""),
            joint_name=config.get("joint_name", "lift"),
            max_speed=config.get("max_speed"),
            max_accel=config.get("max_accel"),
            max_decel=config.get("max_decel"),
            platform_link=config.get("platform_link", "platform"),
            request_policy=config.get("request_policy", ElevatorRequestPolicy.REJECT),
            boarding_height=float(config.get("boarding_height", 1.0)),
            report_platform_contacts=config_get_bool(config, "report_platform_contacts", False),
        )


def _warn_if_boarding_reaches_next_floor(boarding_height: float, floors: Dict[str, float]) -> None:
    """Warn when ``boarding_height`` reaches the next floor: an object waiting there would board."""
    heights = sorted(set(floors.values()))
    gaps = [upper - lower for lower, upper in zip(heights, heights[1:])]
    if gaps and boarding_height >= min(gaps):
        logger.warning(
            "Elevator boarding_height %.3f m reaches the closest floor spacing %.3f m: an object on the next "
            "floor up would board the cabin. Set boarding_height below the floor spacing.",
            boarding_height,
            min(gaps),
        )


class Elevator(Agent):
    """Elevator with named floors and automatic passenger attachment.

    Uses JointAction for floor transitions.  Before movement starts,
    all sim_objects whose AABB overlaps the platform collision box are
    attached to the platform link.  On arrival they are detached.

    Config (YAML)::

        type: elevator
        urdf_path: robots/elevator.urdf
        use_fixed_base: true
        floors: {L1: 0.0, L2: 8.0, L3: 16.0}
        initial_floor: L1
        joint_name: lift
        platform_link: platform
        request_policy: queue  # reject (default), replace_next, queue, or redirect
        boarding_height: 1.0  # metres above the platform that count as on board
        report_platform_contacts: false  # bodies standing on the platform are not collisions
    """

    _spawn_params_cls = ElevatorParams
    _entity_type_name = "elevator"

    # Instance attributes populated in from_params (declared for type checking).
    _floors: Dict[str, float]
    _joint_name: str
    _platform_link: str
    # A class default, so an Elevator built without from_params still has one.
    _boarding_height: float = 1.0
    _platform_link_index: Optional[int] = None
    _passengers: List["SimObject"]
    _state_machine: ElevatorStateMachine

    @classmethod
    def from_params(cls, spawn_params: "ElevatorParams", sim_core=None) -> "Elevator":
        """Create an Elevator from ElevatorParams.

        Raises:
            TypeError: If *spawn_params* is not an ``ElevatorParams`` instance
                or ``floors`` is missing.
        """
        if not isinstance(spawn_params, ElevatorParams):
            raise TypeError(f"Elevator.from_params requires ElevatorParams, got {type(spawn_params).__name__}")
        # `is not None`, not truthiness: a zero is invalid, and skipping it
        # here would silently fall back to the URDF instead of rejecting it.
        has_profile = any(v is not None for v in (spawn_params.max_speed, spawn_params.max_accel, spawn_params.max_decel))
        if has_profile:
            # Before the body exists. super().from_params() registers the
            # agent with sim_core and PyBullet, so a profile refused after
            # that would leave a half-configured elevator in the simulation.
            validate_joint_motion_profile(
                max_velocity=spawn_params.max_speed,
                max_accel=spawn_params.max_accel,
                max_decel=spawn_params.max_decel,
            )
        agent = cast("Elevator", super().from_params(spawn_params, sim_core))
        try:
            agent._floors = spawn_params.floors  # type: ignore[assignment]
            agent._joint_name = spawn_params.joint_name
            agent._platform_link = spawn_params.platform_link
            agent._boarding_height = float(spawn_params.boarding_height)
            _warn_if_boarding_reaches_next_floor(agent._boarding_height, spawn_params.floors)
            # Bodies standing on the platform are not collisions unless asked.
            # The rule covers the whole elevator body (it is decided before the
            # narrow phase, which has no link to go by), so it only holds when
            # the platform is the elevator's one collider.
            agent._filters_own_contacts = not spawn_params.report_platform_contacts and not agent._warn_other_colliders()
            agent._passengers = []  # Currently attached passengers
            if has_profile:
                # Only the joint name is left to go wrong here, the values
                # having been checked above.
                agent.set_joint_motion_profile(
                    agent._joint_name,
                    max_velocity=spawn_params.max_speed,
                    max_accel=spawn_params.max_accel,
                    max_decel=spawn_params.max_decel,
                )
            agent._state_machine = ElevatorStateMachine(
                agent._floors,
                spawn_params.initial_floor,
                agent,
                request_policy=spawn_params.request_policy,
            )
        except Exception:
            # The agent is already in the simulation by now -- an unknown
            # joint name or starting floor would otherwise leave a body and a
            # registry entry behind for a constructor that raised.
            if agent.sim_core is not None:
                agent.sim_core.remove_object(agent)
            elif agent.body_id is not None:
                # No core to unregister from, but from_params() still loaded
                # the URDF into PyBullet, so the body is there regardless.
                try:
                    p.removeBody(agent.body_id, physicsClientId=agent._pid)
                except p.error:
                    logger.warning(f"Failed to remove PyBullet body {agent.body_id} for a refused elevator")
            raise
        return agent

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_floor(self) -> str:
        """Floor where the elevator currently is (or departed from while moving).

        While the elevator is in transit this returns the **departure** floor,
        not the destination.  Use :attr:`target_floor` for the destination.
        """
        return self._state_machine.current_floor

    @property
    def target_floor(self) -> str:
        """Destination floor (equals :attr:`current_floor` when idle)."""
        return self._state_machine.target_floor

    @property
    def available_floors(self) -> List[str]:
        """List of available floor names."""
        return list(self._floors.keys())

    @property
    def is_moving(self) -> bool:
        """True while elevator is transitioning between floors."""
        return self._state_machine.is_moving

    @property
    def state(self) -> ElevatorState:
        """Current backend-neutral elevator lifecycle state."""
        return self._state_machine.state

    @property
    def request_policy(self) -> ElevatorRequestPolicy:
        """Policy applied to requests received while the cabin is moving."""
        return self._state_machine.request_policy

    @property
    def pending_floors(self) -> tuple[str, ...]:
        """Requested floors to visit after the current transition."""
        return self._state_machine.pending_floors

    @property
    def passengers(self) -> List["SimObject"]:
        """Currently attached passengers (read-only copy)."""
        return list(self._passengers)

    # ------------------------------------------------------------------
    # Floor request
    # ------------------------------------------------------------------

    def request_floor(self, floor_name: str, instant: bool = False) -> ElevatorRequestResult:
        """Request the elevator to move to the given floor.

        Attaches all objects on the platform, then starts the joint action.

        Args:
            floor_name: Name of the target floor (must exist in ``floors``).
            instant: Put the cabin at the floor at once, passengers aboard,
                instead of riding there (see
                :meth:`ElevatorStateMachine.request_floor`). The next update
                observes the arrival and releases the passengers.

        Returns:
            Whether the request was accepted immediately, queued, replaced, or
            rejected according to :attr:`request_policy`.
        """
        result = self._state_machine.request_floor(floor_name, instant=instant)
        if result is ElevatorRequestResult.REJECTED:
            if floor_name not in self._floors:
                logger.warning("Unknown floor: %s (available: %s)", floor_name, list(self._floors.keys()))
            return result
        if result is not ElevatorRequestResult.ACCEPTED:
            logger.info("Elevator '%s': %s request for '%s'", self.name, result.value, floor_name)
            return result
        logger.info(
            "Elevator '%s': moving '%s' -> '%s' (%d passengers)",
            self.name,
            self.current_floor,
            floor_name,
            len(self._passengers),
        )
        return result

    # ------------------------------------------------------------------
    # Update — detach on arrival
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Per-step update.  Detaches passengers when movement completes."""
        super().update(dt)

        passenger_count = len(self._passengers)
        if self._state_machine.update():
            logger.info(
                "Elevator '%s': arrived at '%s', %d passengers released",
                self.name,
                self.current_floor,
                passenger_count,
            )

    # ------------------------------------------------------------------
    # ElevatorMotionAdapter implementation (PyBullet-specific mechanics)
    # ------------------------------------------------------------------

    def begin_motion(self, target_height: float, instant: bool = False) -> None:
        """Command the PyBullet lift joint to move to ``target_height``.

        With ``instant`` the joint is put there at once, carrying the attached
        passengers (:meth:`Agent.set_joint_position`).
        """
        self.clear_actions()
        if instant:
            self.set_joint_position(self._joint_name, target_height)
            return
        self.add_action(JointAction(target_joint_positions={self._joint_name: target_height}))

    def motion_in_progress(self) -> bool:
        """Observe whether the cabin is still moving.

        The action alone is not enough once the lift joint has a motion
        profile. ``JointAction`` completes as soon as the joint is inside its
        tolerance, but a ramped joint is still braking through the last
        millimetres after that -- measured on a 0.6 m ride at 0.4 m/s with
        1 m/s^2 ramps, the action finished at 1.78 s with the cabin 7.2 mm
        short, and the joint settled at 1.88 s. Taking the action's word for
        it let the state machine mark the cabin arrived and release its
        passengers while it was still moving.
        """
        action = self.get_current_action()
        if action is not None and isinstance(action, JointAction) and action.status == ActionStatus.IN_PROGRESS:
            return True
        return self.is_joint_moving(self._joint_name)

    def attach_platform_passengers(self) -> int:
        """Attach platform occupants using PyBullet constraints."""
        self._attach_platform_objects()
        return len(self._passengers)

    def detach_passengers(self) -> int:
        """Release constrained passengers after a completed lift motion."""
        count = len(self._passengers)
        self._detach_all_passengers()
        return count

    # ------------------------------------------------------------------
    # Platform object detection & attach/detach
    # ------------------------------------------------------------------

    def _get_platform_link_index(self) -> int:
        """Resolve the platform link name to a PyBullet link index (looked up once)."""
        if self._platform_link_index is None:
            self._platform_link_index = self._find_platform_link_index()
        return self._platform_link_index

    def _find_platform_link_index(self) -> int:
        """Look the platform link up by name among the body's links."""
        num_joints = p.getNumJoints(self.body_id, physicsClientId=self._pid)
        for i in range(num_joints):
            info = p.getJointInfo(self.body_id, i, physicsClientId=self._pid)
            link_name = info[12].decode("utf-8")
            if link_name == self._platform_link:
                return i
        logger.warning("Platform link '%s' not found, using link 0", self._platform_link)
        return 0

    def _warn_other_colliders(self) -> bool:
        """Whether the elevator has colliders besides its platform; if so, say so once.

        Bodies standing on the platform are then reported as collisions, so
        that contact with a cabin wall, ceiling or door is not hidden.
        """
        platform = self._get_platform_link_index()
        others = []
        for link in range(-1, p.getNumJoints(self.body_id, physicsClientId=self._pid)):
            if link != platform and p.getCollisionShapeData(self.body_id, link, physicsClientId=self._pid):
                name = (
                    "base" if link == -1 else p.getJointInfo(self.body_id, link, physicsClientId=self._pid)[12].decode("utf-8")
                )
                others.append(name)
        if not others:
            return False
        logger.warning(
            "Elevator '%s' has colliders besides its platform link '%s' (%s): bodies standing on the "
            "platform are reported as collisions with it, as with report_platform_contacts=True",
            self.name,
            self._platform_link,
            ", ".join(others),
        )
        return True

    def _get_platform_aabb(self) -> tuple:
        """Get the AABB of the platform link.

        Returns:
            Tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z)).
        """
        link_idx = self._get_platform_link_index()
        return p.getAABB(self.body_id, link_idx, physicsClientId=self._pid)

    def _ignores_contact_with(self, other: "SimObject") -> bool:
        """A body standing on the platform is not in collision with the elevator.

        Standing on it: its base over the platform's footprint, and no link
        below the platform's mid-height. The footprint is the base link's, as
        for boarding: an arm beside the cabin reaching over the platform is not
        standing on it, and its contact is reported. The height is every
        link's, so a robot whose wheels (child links) sink through the platform
        is reported too. A passenger (attached to the elevator) is left to the
        attached-object rule. Only asked while ``report_platform_contacts`` is
        off and the platform is the elevator's only collider: the rule covers
        every link of the elevator.
        """
        if other._attachment_root() is self:
            return False
        core = self.sim_core
        object_id = other.object_id
        # Only the collision check asks, so the cache holds every link of
        # ``other`` as of this step. Not cached (collision off, not initialised
        # yet): its base link from PyBullet.
        whole = core._cached_aabbs_dict.get(object_id)
        if whole is not None and core._aabb_links.get(object_id) == ():
            base = whole  # no child link with a collider: the cached box is the base's
        else:
            base = p.getAABB(other.body_id, physicsClientId=self._pid)
            if whole is None:
                whole = base
        (b_min_x, b_min_y, _), (b_max_x, b_max_y, _) = base
        lowest = whole[0][2]
        (p_min_x, p_min_y, p_min_z), (p_max_x, p_max_y, p_max_z) = p.getAABB(
            self.body_id, self._get_platform_link_index(), physicsClientId=self._pid
        )
        return (
            lowest >= 0.5 * (p_min_z + p_max_z)
            and b_min_x <= p_max_x
            and b_max_x >= p_min_x
            and b_min_y <= p_max_y
            and b_max_y >= p_min_y
        )

    def _find_objects_on_platform(self) -> List["SimObject"]:
        """Find sim_objects whose AABB overlaps the platform AABB.

        Only considers objects that are not the elevator itself and are
        not already attached to something.
        """
        if self.sim_core is None:
            return []

        platform_min, platform_max = self._get_platform_aabb()

        # Expand platform AABB upward to catch objects standing on it (robot
        # feet may be just above the platform top surface), by boarding_height.
        search_min = (platform_min[0], platform_min[1], platform_min[2])
        search_max = (platform_max[0], platform_max[1], platform_max[2] + self._boarding_height)

        # Only the objects near the platform; a core without query_aabb (a
        # stand-in in tests, say) gets the full scan.
        query = getattr(self.sim_core, "query_aabb", None)
        nearby = query(search_min, search_max) if query is not None else self.sim_core.sim_objects

        candidates = []
        for obj in nearby:
            if obj is self:
                continue
            if obj.is_attached():
                continue
            # The base link's box, not the whole-body box the collision cache
            # holds: boarding asks where the body stands. With every link, an
            # arm beside the cabin reaching over the platform would be boarded.
            try:
                obj_min, obj_max = p.getAABB(obj.body_id, physicsClientId=self._pid)
            except Exception:
                continue

            # AABB overlap test
            if (
                obj_min[0] <= search_max[0]
                and obj_max[0] >= search_min[0]
                and obj_min[1] <= search_max[1]
                and obj_max[1] >= search_min[1]
                and obj_min[2] <= search_max[2]
                and obj_max[2] >= search_min[2]
            ):
                candidates.append(obj)

        return candidates

    def _attach_platform_objects(self) -> None:
        """Attach all objects currently on the platform."""
        objects = self._find_objects_on_platform()
        link_idx = self._get_platform_link_index()

        for obj in objects:
            # Temporarily enable pickable for the attach call
            # (agents have pickable=False by default, but elevator attachment is valid)
            was_pickable = obj.pickable
            obj.pickable = True
            success = self.attach_object(obj, parent_link_index=link_idx, keep_world_pose=True)
            if not was_pickable:
                obj.pickable = was_pickable

            if success:
                self._passengers.append(obj)
                logger.info(
                    "Elevator '%s': attached '%s' (body %d) to platform",
                    self.name,
                    getattr(obj, "name", "?"),
                    obj.body_id,
                )

    def _detach_all_passengers(self) -> None:
        """Detach all currently attached passengers."""
        for obj in list(self._passengers):
            self.detach_object(obj)
            logger.info(
                "Elevator '%s': detached '%s' (body %d)",
                self.name,
                getattr(obj, "name", "?"),
                obj.body_id,
            )
        self._passengers.clear()
