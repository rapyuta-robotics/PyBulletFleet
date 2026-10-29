"""Tests for Elevator."""

import pytest
from pybullet_fleet import Agent, AgentSpawnParams, Pose, SimObject
from pybullet_fleet.sim_object import SimObjectSpawnParams, ShapeParams
from pybullet_fleet.devices.elevator import Elevator, ElevatorParams


@pytest.fixture
def sim_core(pybullet_env):
    from tests.conftest import MockSimCore

    sc = MockSimCore()
    sc._client = pybullet_env
    return sc


def _make_elevator(sim_core, floors=None, initial_floor="L1", **extra):
    """Helper: create an Elevator with standard config."""
    if floors is None:
        floors = {"L1": 0.0, "L2": 8.0}
    params = ElevatorParams(
        urdf_path="robots/elevator.urdf",
        initial_pose=Pose.from_xyz(0, 0, 0),
        use_fixed_base=True,
        floors=floors,
        initial_floor=initial_floor,
        joint_name="lift",
        platform_link="platform",
        **extra,
    )
    return Elevator.from_params(params, sim_core)


def _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2):
    """Helper: create a small pickable box sitting on the elevator platform."""
    params = SimObjectSpawnParams(
        visual_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
        initial_pose=Pose.from_xyz(x, y, z),
        mass=0.0,
        pickable=True,
        name="box_on_platform",
    )
    return SimObject.from_params(params, sim_core)


class TestElevator:
    def test_elevator_initial_floor(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0, "L3": 16.0})
        assert elev.current_floor == "L1"

    def test_elevator_request_floor(self, sim_core):
        elev = _make_elevator(sim_core)
        elev.request_floor("L2")
        sim_core.tick(2000)
        assert elev.current_floor == "L2"

    def test_elevator_available_floors(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0, "L3": 16.0})
        assert set(elev.available_floors) == {"L1", "L2", "L3"}

    def test_elevator_is_agent_subclass(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0})
        assert isinstance(elev, Agent)
        assert isinstance(elev, Elevator)


class TestElevatorAutoAttach:
    """Tests for auto-attach/detach of objects on the platform."""

    def test_find_objects_on_platform(self, sim_core):
        """Objects within the platform AABB are detected."""
        elev = _make_elevator(sim_core)
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        found = elev._find_objects_on_platform()
        assert box in found

    def test_find_objects_ignores_distant(self, sim_core):
        """Objects far from the platform are not detected."""
        elev = _make_elevator(sim_core)
        # Place box far outside elevator (platform is 3x3 centered at origin)
        far_box = _make_box_on_platform(sim_core, x=20.0, y=20.0, z=0.2)
        found = elev._find_objects_on_platform()
        assert far_box not in found

    def test_an_object_above_the_boarding_height_is_not_on_board(self, sim_core):
        """An object waiting on the floor above the cabin does not ride along."""
        elev = _make_elevator(sim_core, boarding_height=0.2)
        on = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        above = _make_box_on_platform(sim_core, x=0.5, y=0.5, z=0.8)
        found = elev._find_objects_on_platform()
        assert on in found
        assert above not in found
        assert above in _make_elevator(sim_core)._find_objects_on_platform(), "the default reaches 1 m up"

    def test_find_objects_ignores_self(self, sim_core):
        """Elevator itself is never in the candidate list."""
        elev = _make_elevator(sim_core)
        found = elev._find_objects_on_platform()
        assert elev not in found

    def test_attach_on_request_floor(self, sim_core):
        """request_floor() auto-attaches objects on the platform."""
        elev = _make_elevator(sim_core)
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        elev.request_floor("L2")
        # Box should now be attached
        assert box.is_attached()
        assert box in elev.passengers
        assert elev.is_moving

    def test_detach_on_arrival(self, sim_core):
        """Passengers are detached when the elevator arrives at destination."""
        elev = _make_elevator(sim_core)
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        elev.request_floor("L2")
        assert box.is_attached()
        # Tick enough for the joint to reach target
        sim_core.tick(2000)
        # After arrival, box should be detached
        assert not box.is_attached()
        assert len(elev.passengers) == 0
        assert not elev.is_moving

    def test_box_rides_elevator_z(self, sim_core):
        """Attached object's Z position changes with the platform."""
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0})
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        initial_z = box.get_pose().position[2]
        elev.request_floor("L2")
        # Tick partway — joint should have moved up some amount
        sim_core.tick(500)
        mid_z = box.get_pose().position[2]
        assert mid_z > initial_z + 0.5, f"Box Z should have risen: {initial_z} -> {mid_z}"

    def test_multiple_objects_attach(self, sim_core):
        """Multiple objects on the platform are all attached."""
        elev = _make_elevator(sim_core)
        box1 = _make_box_on_platform(sim_core, x=0.5, y=0.5, z=0.2)
        box2 = _make_box_on_platform(sim_core, x=-0.5, y=-0.5, z=0.2)
        elev.request_floor("L2")
        assert box1.is_attached()
        assert box2.is_attached()
        assert len(elev.passengers) == 2

    def test_no_attach_when_already_at_floor(self, sim_core):
        """Requesting current floor does nothing (no attach, no movement)."""
        elev = _make_elevator(sim_core)
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        elev.request_floor("L1")  # Already at L1
        assert not box.is_attached()
        assert not elev.is_moving

    def test_request_unknown_floor(self, sim_core):
        """Requesting an unknown floor logs a warning and does nothing."""
        elev = _make_elevator(sim_core)
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        elev.request_floor("NONEXISTENT")
        assert not box.is_attached()
        assert not elev.is_moving

    def test_agent_on_platform_attached(self, sim_core):
        """Agent (pickable=False by default) on platform is still attached.

        ElevatorDevice (now Elevator) temporarily sets pickable=True for the attach call.
        """
        elev = _make_elevator(sim_core)
        agent_params = AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.2),
        )
        robot = Agent.from_params(agent_params, sim_core)
        assert not robot.pickable  # Default for agents
        elev.request_floor("L2")
        assert robot.is_attached()
        # pickable should be restored to False after attach
        assert not robot.pickable

    def test_is_moving_property(self, sim_core):
        """is_moving reflects elevator movement state."""
        elev = _make_elevator(sim_core)
        assert not elev.is_moving
        elev.request_floor("L2")
        assert elev.is_moving
        sim_core.tick(2000)
        assert not elev.is_moving

    def test_duplicate_request_during_movement_is_noop(self, sim_core):
        """Requesting the same target floor while moving is a no-op.

        RMF sends repeated lift requests.  A duplicate request must NOT
        restart the JointAction (clear_actions + add_action) because that
        can reset progress or cause timing-dependent early detach.
        """
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0})
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        elev.request_floor("L2")
        assert elev.is_moving
        assert box.is_attached()

        # Tick partway — elevator is mid-transit
        sim_core.tick(50)
        assert elev.is_moving
        assert box.is_attached()

        # Duplicate request — should NOT restart movement
        action_before = elev.get_current_action()
        elev.request_floor("L2")
        action_after = elev.get_current_action()

        # Same action object — not restarted
        assert action_before is action_after, "Duplicate request_floor restarted the JointAction"
        assert elev.is_moving
        assert box.is_attached()

        # Complete the journey — should arrive normally
        sim_core.tick(2000)
        assert not elev.is_moving
        assert not box.is_attached()
        assert elev.current_floor == "L2"


class TestElevatorFloorProperties:
    """Tests for current_floor vs target_floor semantics."""

    def test_target_floor_initial(self, sim_core):
        """target_floor equals initial_floor when idle."""
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0})
        assert elev.target_floor == "L1"

    def test_target_floor_during_movement(self, sim_core):
        """target_floor returns destination floor while moving."""
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0})
        elev.request_floor("L2")
        assert elev.target_floor == "L2"

    def test_current_floor_stays_departure_during_movement(self, sim_core):
        """current_floor returns departure floor while elevator is in transit."""
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0})
        assert elev.current_floor == "L1"
        elev.request_floor("L2")
        # Tick a few steps — elevator should be mid-transit
        sim_core.tick(5)
        assert elev.is_moving, "Elevator should still be in transit"
        # current_floor must NOT jump to L2 while still moving
        assert elev.current_floor == "L1"

    def test_current_floor_updates_on_arrival(self, sim_core):
        """current_floor updates to destination after arrival."""
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0})
        elev.request_floor("L2")
        sim_core.tick(2000)
        assert not elev.is_moving
        assert elev.current_floor == "L2"
        assert elev.target_floor == "L2"

    def test_target_floor_after_arrival(self, sim_core):
        """target_floor stays at destination after arrival."""
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0})
        elev.request_floor("L2")
        sim_core.tick(2000)
        assert elev.target_floor == "L2"

    def test_current_and_target_differ_mid_transit(self, sim_core):
        """current_floor and target_floor are different while moving."""
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 8.0})
        elev.request_floor("L2")
        sim_core.tick(5)
        assert elev.is_moving
        assert elev.current_floor != elev.target_floor
        assert elev.current_floor == "L1"
        assert elev.target_floor == "L2"


class TestElevatorParams:
    """Tests for ElevatorParams dataclass validation."""

    def test_direct_construction(self):
        """ElevatorParams direct construction with typed fields."""
        params = ElevatorParams(
            urdf_path="robots/elevator.urdf",
            floors={"L1": 0.0, "L2": 8.0},
            initial_floor="L1",
            joint_name="lift",
            platform_link="platform",
        )
        assert params.floors == {"L1": 0.0, "L2": 8.0}
        assert params.initial_floor == "L1"
        assert params.joint_name == "lift"
        assert params.platform_link == "platform"
        assert isinstance(params, AgentSpawnParams)

    def test_minimal_construction(self):
        """ElevatorParams with only floors (defaults for rest)."""
        params = ElevatorParams(
            urdf_path="robots/elevator.urdf",
            floors={"G": 0.0, "F1": 3.0},
        )
        assert params.floors == {"G": 0.0, "F1": 3.0}
        assert params.initial_floor == "G"  # first floor key
        assert params.joint_name == "lift"
        assert params.platform_link == "platform"

    def test_missing_floors_raises(self):
        """ElevatorParams raises TypeError when floors is None."""
        with pytest.raises(TypeError, match="floors"):
            ElevatorParams(urdf_path="robots/elevator.urdf")

    def test_from_dict_top_level(self):
        """ElevatorParams.from_dict reads elevator fields at top level."""
        params = ElevatorParams.from_dict(
            {
                "name": "test_elevator",
                "urdf_path": "robots/elevator.urdf",
                "floors": {"L1": 0.0, "L2": 8.0},
                "initial_floor": "L1",
            }
        )
        assert params.floors == {"L1": 0.0, "L2": 8.0}
        assert params.initial_floor == "L1"

    def test_elevator_from_params_wrong_type(self, sim_core):
        """Elevator.from_params raises TypeError when given AgentSpawnParams."""
        params = AgentSpawnParams(
            urdf_path="robots/elevator.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0),
            use_fixed_base=True,
        )
        with pytest.raises(TypeError, match="ElevatorParams"):
            Elevator.from_params(params, sim_core)


class TestElevatorRedirect:
    def test_a_moving_cabin_turns_to_the_latest_floor_with_its_passengers(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0, "L3": 8.0}, request_policy="redirect")
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        elev.request_floor("L3")
        sim_core.tick(100)
        assert elev.is_moving
        assert elev.request_floor("L2").value == "redirected"
        sim_core.tick(3000)
        assert elev.current_floor == "L2"
        assert not elev.is_moving
        assert box.get_pose().position[2] == pytest.approx(4.2, abs=0.05)

    def test_negative_boarding_height_is_refused(self, sim_core):
        with pytest.raises(ValueError):
            _make_elevator(sim_core, boarding_height=-0.1)

    def test_boarding_height_from_dict(self):
        params = ElevatorParams.from_dict(
            {"name": "e1", "urdf_path": "robots/elevator.urdf", "floors": {"L1": 0.0}, "boarding_height": 0.25}
        )
        assert params.boarding_height == 0.25
