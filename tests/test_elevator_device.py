"""Tests for Elevator."""

import pytest
from pybullet_fleet import Agent, AgentSpawnParams, Pose, SimObject
from pybullet_fleet.sim_object import SimObjectSpawnParams, ShapeParams
from pybullet_fleet.devices.elevator import Elevator, ElevatorParams
from pybullet_fleet.devices.elevator_state_machine import ElevatorRequestPolicy, ElevatorRequestResult


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


class TestElevatorInstantRide:
    """request_floor(instant=True): the cabin is at the floor at once, passengers aboard."""

    def test_cabin_and_passenger_are_there_without_a_step(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0})
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        z = box.get_pose().position[2]

        elev.request_floor("L2", instant=True)

        assert elev.get_joint_state("lift")[0] == pytest.approx(4.0)
        assert box.get_pose().position[2] == pytest.approx(z + 4.0, abs=1e-6)
        assert box.is_attached()

    def test_it_arrives_and_releases_on_the_next_update(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0})
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)

        elev.request_floor("L2", instant=True)
        sim_core.tick(1)

        assert elev.current_floor == "L2"
        assert not elev.is_moving
        assert not box.is_attached()

    def test_a_moving_cabin_is_put_at_the_new_floor(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0, "L3": 8.0})
        elev.request_floor("L2")
        sim_core.tick(10)
        assert elev.is_moving

        assert elev.request_floor("L3", instant=True) is ElevatorRequestResult.REDIRECTED
        sim_core.tick(1)

        assert elev.current_floor == "L3"
        assert elev.get_joint_state("lift")[0] == pytest.approx(8.0)


class TestElevatorCarriedLoad:
    """A robot that carries a load rides with it: the load stays the robot's, not the cabin's."""

    def _robot_with_load(self, sim_core):
        robot = Agent.from_params(
            AgentSpawnParams(urdf_path="robots/simple_cube.urdf", initial_pose=Pose.from_xyz(0, 0, 0.2)),
            sim_core,
        )
        load = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
                initial_pose=Pose.from_xyz(0, 0, 0.6),
                mass=0.0,
                pickable=True,
                name="load",
            ),
            sim_core,
        )
        assert robot.attach_object(load, keep_world_pose=True)
        return robot, load

    @pytest.mark.parametrize("instant", [False, True])
    def test_the_load_rides_with_its_robot_and_stays_on_it(self, sim_core, instant):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0})
        robot, load = self._robot_with_load(sim_core)
        robot_z, load_z = robot.get_pose().position[2], load.get_pose().position[2]

        elev.request_floor("L2", instant=instant)

        # The cabin takes the robot only: an attached load would be left behind on release.
        assert elev.passengers == [robot]

        sim_core.tick(2000)

        assert elev.current_floor == "L2"
        assert not elev.is_moving
        assert robot.get_pose().position[2] == pytest.approx(robot_z + 4.0, abs=1e-3)
        assert load.get_pose().position[2] == pytest.approx(load_z + 4.0, abs=1e-3)
        assert not robot.is_attached()
        assert load in robot.get_attached_objects()


class TestElevatorRedirect:
    def test_a_moving_cabin_turns_to_the_latest_floor_with_its_passengers(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0, "L3": 8.0}, request_policy="redirect")
        box = _make_box_on_platform(sim_core, x=0.0, y=0.0, z=0.2)
        elev.request_floor("L3")
        sim_core.tick(100)
        assert elev.is_moving
        assert elev.request_floor("L2") is ElevatorRequestResult.REDIRECTED
        sim_core.tick(3000)
        assert elev.current_floor == "L2"
        assert not elev.is_moving
        assert box.get_pose().position[2] == pytest.approx(4.2, abs=0.05)

    def test_a_reversal_brakes_and_turns_within_the_profile(self, sim_core):
        """Redirected against its motion, the cabin brakes at max_accel, turns, and stops on the floor."""
        elev = _make_elevator(
            sim_core,
            floors={"L1": 0.0, "L2": 4.0, "L3": 8.0},
            request_policy="redirect",
            max_speed=1.0,
            max_accel=1.0,
        )
        dt = sim_core._dt
        heights = []

        def run(steps):
            for _ in range(steps):
                sim_core.tick(1)
                heights.append(elev.get_joint_state("lift")[0])

        elev.request_floor("L3")
        run(int(3.0 / dt))  # up to full speed, 2.5 m up
        assert elev.request_floor("L1") is ElevatorRequestResult.REDIRECTED
        run(int(12.0 / dt))

        speeds = [(b - a) / dt for a, b in zip(heights, heights[1:])]
        accels = [abs(b - a) / dt for a, b in zip(speeds, speeds[1:])]
        assert max(accels) <= 1.0 + 1e-6  # no step change in speed, turning included
        assert max(heights) == pytest.approx(2.5 + 1.0**2 / (2 * 1.0), abs=1e-3)  # braked to a stop first
        assert min(heights) >= -1e-9  # no overshoot below L1
        assert elev.current_floor == "L1"
        assert heights[-1] == pytest.approx(0.0, abs=1e-6)

    def test_a_cabin_switched_to_redirect_drops_floors_queued_before(self, sim_core):
        elev = _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0, "L3": 8.0}, request_policy="queue")
        elev.request_floor("L3")
        sim_core.tick(10)
        assert elev.request_floor("L2") is ElevatorRequestResult.QUEUED
        elev._state_machine.request_policy = ElevatorRequestPolicy.REDIRECT

        assert elev.request_floor("L3") is ElevatorRequestResult.REPLACED
        assert elev.pending_floors == ()

    def test_boarding_height_reaching_the_next_floor_warns(self, sim_core, caplog):
        with caplog.at_level("WARNING", logger="pybullet_fleet.devices.elevator"):
            _make_elevator(sim_core, floors={"L1": 0.0, "L2": 0.9, "L3": 4.0}, boarding_height=1.0)
        assert "boarding_height" in caplog.text

    def test_boarding_height_below_the_floor_spacing_does_not_warn(self, sim_core, caplog):
        with caplog.at_level("WARNING", logger="pybullet_fleet.devices.elevator"):
            _make_elevator(sim_core, floors={"L1": 0.0, "L2": 4.0}, boarding_height=1.0)
        assert "boarding_height" not in caplog.text

    def test_negative_boarding_height_is_refused(self, sim_core):
        with pytest.raises(ValueError):
            _make_elevator(sim_core, boarding_height=-0.1)

    def test_boarding_height_from_dict(self):
        params = ElevatorParams.from_dict(
            {"name": "e1", "urdf_path": "robots/elevator.urdf", "floors": {"L1": 0.0}, "boarding_height": 0.25}
        )
        assert params.boarding_height == 0.25


class TestElevatorPlatformContacts:
    """A body standing on the platform is not in collision with the elevator unless asked.

    The broadphase AABB covers every link, so the elevator's box includes its
    platform: without this a robot resting on the platform before departure
    or after arrival would be reported as colliding with the cabin.
    """

    @pytest.fixture
    def sim(self):
        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=False, timestep=0.02, monitor=False, enable_floor=False)
        )
        sim.initialize_simulation()
        yield sim
        sim.close()

    @staticmethod
    def _elevator(sim, **extra):
        return Elevator.from_params(
            ElevatorParams(
                urdf_path="robots/elevator.urdf",
                initial_pose=Pose.from_xyz(0, 0, 0),
                use_fixed_base=True,
                floors={"L1": 0.0, "L2": 8.0},
                initial_floor="L1",
                **extra,
            ),
            sim,
        )

    @staticmethod
    def _robot_on_platform(sim):
        # mobile_robot's wheels reach 0.252 m below its origin; the platform top is at 0.1.
        return Agent.from_params(
            AgentSpawnParams(urdf_path="robots/mobile_robot.urdf", initial_pose=Pose.from_xyz(0, 0, 0.35)), sim_core=sim
        )

    @staticmethod
    def _box_under_platform(sim):
        # Touches the platform's underside (bottom at -0.1): a cabin coming down onto it.
        return SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
                initial_pose=Pose.from_xyz(1.0, 1.0, -0.29),
                mass=0.0,
            ),
            sim,
        )

    @staticmethod
    def _pair(a, b):
        return tuple(sorted((a.object_id, b.object_id)))

    def test_a_robot_resting_on_the_platform_is_not_reported_by_default(self, sim):
        elev = self._elevator(sim)
        robot = self._robot_on_platform(sim)
        sim.step_once()
        assert self._pair(elev, robot) not in sim.get_active_collision_pairs()

    def test_report_platform_contacts_reports_a_robot_resting_on_the_platform(self, sim):
        elev = self._elevator(sim, report_platform_contacts=True)
        robot = self._robot_on_platform(sim)
        sim.step_once()
        assert self._pair(elev, robot) in sim.get_active_collision_pairs()

    @pytest.mark.parametrize("report", [False, True])
    def test_a_body_touching_the_platform_from_below_is_reported(self, sim, report):
        elev = self._elevator(sim, report_platform_contacts=report)
        box = self._box_under_platform(sim)
        sim.step_once()
        assert self._pair(elev, box) in sim.get_active_collision_pairs()

    def test_a_robot_leaving_the_platform_area_downward_is_reported_again(self, sim):
        """The rule follows the geometry: a body sunk through the platform is no longer standing on it."""
        elev = self._elevator(sim)
        robot = self._robot_on_platform(sim)
        sim.step_once()
        robot.set_pose(Pose.from_xyz(0, 0, 0.15))  # wheels at -0.10, below the platform's mid-plane
        sim.step_once()
        assert self._pair(elev, robot) in sim.get_active_collision_pairs()

    _REACHING_URDF = """<?xml version="1.0"?>
<robot name="reaching">
  <link name="base">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
    <inertial><mass value="0.0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
  </link>
  <joint name="reach" type="fixed">
    <parent link="base"/><child link="tool"/><origin xyz="-0.3 0 0"/>
  </joint>
  <link name="tool">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
    <inertial><mass value="0.0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
  </link>
</robot>
"""

    def test_a_link_reaching_over_the_platform_from_beside_the_cabin_is_reported(self, sim, tmp_path):
        """Standing is judged by the base's footprint: a body beside the cabin whose
        link reaches over the platform, into its floor, is not standing on it."""
        elev = self._elevator(sim)
        urdf = tmp_path / "reaching.urdf"
        urdf.write_text(self._REACHING_URDF)
        # Base x 1.55..1.75, beside the platform (x -1.5..1.5, top at z 0.1);
        # the tool x 1.25..1.45 over it, its bottom 0.05 into the floor.
        body = Agent.from_params(
            AgentSpawnParams(urdf_path=str(urdf), initial_pose=Pose.from_xyz(1.65, 0, 0.15), use_fixed_base=True),
            sim_core=sim,
        )
        sim.step_once()
        assert self._pair(elev, body) in sim.get_active_collision_pairs()

    def test_a_riding_passenger_keeps_the_attached_object_rule(self):
        """The platform rule leaves passengers alone: with ignore_attached_collision off they are reported."""
        from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

        sim = MultiRobotSimulationCore(
            SimulationParams(
                gui=False,
                physics=False,
                timestep=0.02,
                monitor=False,
                enable_floor=False,
                ignore_attached_collision=False,
            )
        )
        sim.initialize_simulation()
        try:
            elev = self._elevator(sim)
            robot = self._robot_on_platform(sim)
            elev.request_floor("L2")
            assert robot.is_attached()
            sim.step_once()
            assert self._pair(elev, robot) in sim.get_active_collision_pairs()
        finally:
            sim.close()

    _WALLED_URDF = """<?xml version="1.0"?>
<robot name="walled_elevator">
  <link name="shaft"><inertial><mass value="0.0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
  <joint name="lift" type="prismatic">
    <parent link="shaft"/><child link="platform"/>
    <axis xyz="0 0 1"/><limit lower="0" upper="30.0" effort="100" velocity="2.0"/>
  </joint>
  <link name="platform">
    <collision><geometry><box size="3.0 3.0 0.2"/></geometry></collision>
    <inertial><mass value="0.0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
  </link>
  <joint name="wall_mount" type="fixed">
    <parent link="platform"/><child link="wall"/><origin xyz="1.45 0 1.1"/>
  </joint>
  <link name="wall">
    <collision><geometry><box size="0.1 3.0 2.0"/></geometry></collision>
    <inertial><mass value="0.0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
  </link>
</robot>
"""

    def _walled_elevator(self, sim, tmp_path, **extra):
        urdf = tmp_path / "walled_elevator.urdf"
        urdf.write_text(self._WALLED_URDF)
        return Elevator.from_params(
            ElevatorParams(
                urdf_path=str(urdf),
                initial_pose=Pose.from_xyz(0, 0, 0),
                use_fixed_base=True,
                floors={"L1": 0.0, "L2": 8.0},
                initial_floor="L1",
                **extra,
            ),
            sim,
        )

    def test_a_robot_on_the_platform_driving_into_a_cabin_wall_is_reported(self, sim, tmp_path):
        """The platform rule must not hide contact with the cabin's other colliders."""
        elev = self._walled_elevator(sim, tmp_path)
        robot = self._robot_on_platform(sim)
        robot.set_pose(Pose.from_xyz(1.3, 0, 0.35))  # on the platform, against the wall at x = 1.4
        sim.step_once()
        assert self._pair(elev, robot) in sim.get_active_collision_pairs()

    def test_an_elevator_with_other_colliders_warns_once_that_platform_contacts_are_reported(self, sim, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="pybullet_fleet.devices.elevator"):
            self._walled_elevator(sim, tmp_path)
        warnings = [r for r in caplog.records if "report_platform_contacts" in r.getMessage()]
        assert len(warnings) == 1
        assert "wall" in warnings[0].getMessage()

    def test_report_platform_contacts_on_does_not_warn_for_other_colliders(self, sim, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="pybullet_fleet.devices.elevator"):
            self._walled_elevator(sim, tmp_path, report_platform_contacts=True)
        assert not [r for r in caplog.records if "report_platform_contacts" in r.getMessage()]

    def test_the_bundled_elevator_does_not_warn(self, sim, caplog):
        with caplog.at_level("WARNING", logger="pybullet_fleet.devices.elevator"):
            self._elevator(sim)
        assert not [r for r in caplog.records if "report_platform_contacts" in r.getMessage()]

    def test_report_platform_contacts_from_dict(self):
        params = ElevatorParams.from_dict(
            {"name": "e1", "urdf_path": "robots/elevator.urdf", "floors": {"L1": 0.0}, "report_platform_contacts": True}
        )
        assert params.report_platform_contacts is True
        default = ElevatorParams.from_dict({"name": "e1", "urdf_path": "robots/elevator.urdf", "floors": {"L1": 0.0}})
        assert default.report_platform_contacts is False

    @pytest.mark.parametrize("value, expected", [("false", False), ("no", False), ("0", False), ("true", True), (1, True)])
    def test_report_platform_contacts_from_dict_parses_quoted_booleans(self, value, expected):
        config = {"name": "e1", "urdf_path": "robots/elevator.urdf", "floors": {"L1": 0.0}}
        params = ElevatorParams.from_dict({**config, "report_platform_contacts": value})
        assert params.report_platform_contacts is expected

    def test_report_platform_contacts_from_dict_rejects_an_unknown_value(self):
        config = {"name": "e1", "urdf_path": "robots/elevator.urdf", "floors": {"L1": 0.0}}
        with pytest.raises(ValueError, match="report_platform_contacts"):
            ElevatorParams.from_dict({**config, "report_platform_contacts": "maybe"})
