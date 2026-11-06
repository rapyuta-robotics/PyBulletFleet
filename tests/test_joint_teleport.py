"""A joint can be put at a position at once, without moving there.

What a simulator's "set entity state" asks of a fixed-base mechanism such as a
lift cabin: placing the base would carry the whole shaft away, so the joint has
to jump instead. Anything attached to the moved link goes with it, and the body
counts as moved for collision checking, exactly as after a step that moved it.
"""

import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams

URDF = "robots/elevator.urdf"  # ships with pybullet_fleet: prismatic "lift", <limit velocity="2.0">
JOINT = "lift"
DT = 0.02


@pytest.fixture
def sim_core_params():
    return {"timestep": DT}


def _lift(sim_core):
    return Agent.from_params(AgentSpawnParams(urdf_path=URDF, name="lift", use_fixed_base=True), sim_core=sim_core)


def _box(sim_core, z=0.2):
    params = SimObjectSpawnParams(
        visual_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.2, 0.2, 0.2]),
        initial_pose=Pose.from_xyz(0.0, 0.0, z),
        mass=0.0,
        pickable=True,
        name="box",
    )
    return SimObject.from_params(params, sim_core)


def _position(agent):
    return agent.get_joint_state_by_name(JOINT)[0]


class TestSetJointPosition:
    def test_the_joint_is_there_without_a_step(self, sim_core):
        agent = _lift(sim_core)

        assert agent.set_joint_position(JOINT, 3.0)

        assert _position(agent) == pytest.approx(3.0)
        assert not agent.is_joint_moving(JOINT)

    def test_it_stays_there(self, sim_core):
        agent = _lift(sim_core)
        agent.set_joint_position(JOINT, 3.0)

        for _ in range(50):
            sim_core.step_once()

        assert _position(agent) == pytest.approx(3.0)

    def test_a_joint_on_its_way_elsewhere_stops_where_it_is_put(self, sim_core):
        agent = _lift(sim_core)
        agent.set_joint_motion_profile(JOINT, max_velocity=0.5, max_accel=1.0, max_decel=1.0)
        agent.set_joint_target_by_name(JOINT, 4.0)
        for _ in range(20):
            sim_core.step_once()
        assert agent.is_joint_moving(JOINT)

        agent.set_joint_position(JOINT, 1.0)
        for _ in range(100):
            sim_core.step_once()

        # The old target is replaced and the ramp's speed is gone: no braking past 1.0.
        assert _position(agent) == pytest.approx(1.0)
        assert not agent.is_joint_moving(JOINT)
        assert agent.is_joint_at_target_by_name(JOINT, 1.0)

    def test_the_index_works_as_well_as_the_name(self, sim_core):
        agent = _lift(sim_core)
        index = agent._joint_index_by_name(JOINT)

        assert agent.set_joint_position(index, 2.0)

        assert _position(agent) == pytest.approx(2.0)

    def test_an_attached_object_goes_with_the_link(self, sim_core):
        agent = _lift(sim_core)
        box = _box(sim_core)
        link = agent._joint_index_by_name(JOINT)
        assert agent.attach_object(box, parent_link_index=link, keep_world_pose=True)
        z = box.get_pose().position[2]

        agent.set_joint_position(JOINT, 3.0)

        assert box.get_pose().position[2] == pytest.approx(z + 3.0, abs=1e-6)

    def test_the_body_counts_as_moved(self, sim_core):
        agent = _lift(sim_core)
        sim_core._moved_this_step.clear()

        agent.set_joint_position(JOINT, 3.0)

        assert agent.object_id in sim_core._moved_this_step

    def test_an_unknown_joint_is_refused(self, sim_core):
        agent = _lift(sim_core)

        assert not agent.set_joint_position("no_such_joint", 1.0)

        assert _position(agent) == pytest.approx(0.0)


class TestSetJointPositionChecks:
    def test_a_fixed_joint_is_refused_with_a_warning(self, sim_core, caplog):
        agent = _manipulator(sim_core)

        with caplog.at_level("WARNING"):
            assert not agent.set_joint_position("base_to_mount", 0.5)

        assert "fixed" in caplog.text

    def test_a_position_outside_the_limits_is_applied_with_a_warning(self, sim_core, caplog):
        agent = _lift(sim_core)

        with caplog.at_level("WARNING"):
            assert agent.set_joint_position(JOINT, 40.0)  # the URDF's upper limit is 30

        assert _position(agent) == pytest.approx(40.0)
        assert "outside its limits" in caplog.text

    def test_with_physics_the_joint_is_held_with_the_given_force(self, monkeypatch):
        import pybullet as p

        sim = MultiRobotSimulationCore(
            SimulationParams(gui=False, physics=True, timestep=DT, monitor=False, enable_monitor_gui=False)
        )
        sim.initialize_simulation()
        try:
            # mass=None keeps the URDF's masses: a dynamic arm, whose joints are motor-driven.
            agent = Agent.from_params(
                AgentSpawnParams(urdf_path="robots/arm_robot.urdf", name="arm", use_fixed_base=True, mass=None),
                sim_core=sim,
            )
            assert not agent._use_kinematic_joints
            calls = []
            real = p.setJointMotorControl2
            monkeypatch.setattr(p, "setJointMotorControl2", lambda *a, **k: (calls.append(k), real(*a, **k))[1])

            assert agent.set_joint_position("shoulder_to_elbow", 3.0 / 4, max_force=123.0)

            assert calls and calls[-1]["force"] == 123.0
            assert calls[-1]["targetPosition"] == 0.75
        finally:
            p.disconnect(sim.client)


MANIPULATOR = "robots/mobile_manipulator.urdf"  # wheels, a fixed mount, and four arm joints
ARM = ["mount_to_shoulder", "shoulder_to_elbow", "elbow_to_wrist", "wrist_to_end"]


def _manipulator(sim_core):
    return Agent.from_params(
        AgentSpawnParams(urdf_path=MANIPULATOR, name="manipulator", use_fixed_base=True), sim_core=sim_core
    )


class TestSetJointsPositions:
    def test_a_dict_places_the_named_joints(self, sim_core):
        agent = _manipulator(sim_core)
        positions = {"mount_to_shoulder": 0.5, "shoulder_to_elbow": -0.4, "elbow_to_wrist": 0.3}

        assert agent.set_joints_positions(positions)

        for name, position in positions.items():
            assert agent.get_joint_state_by_name(name)[0] == pytest.approx(position)
        assert agent.get_joint_state_by_name("wrist_to_end")[0] == pytest.approx(0.0)

    def test_a_list_places_every_joint_and_ignores_the_fixed_ones(self, sim_core):
        agent = _manipulator(sim_core)
        positions = [0.1 * (i + 1) for i in range(agent.get_num_joints())]

        assert agent.set_joints_positions(positions)

        for name in ARM:
            index = agent._joint_index_by_name(name)
            assert agent.get_joint_state(index)[0] == pytest.approx(positions[index])

    def test_a_list_of_the_wrong_length_sets_nothing(self, sim_core):
        agent = _manipulator(sim_core)

        assert not agent.set_joints_positions([0.5])

        assert agent.get_joint_state_by_name("mount_to_shoulder")[0] == pytest.approx(0.0)

    def test_a_dict_naming_a_fixed_or_unknown_joint_still_sets_the_others(self, sim_core):
        agent = _manipulator(sim_core)

        assert not agent.set_joints_positions({"base_to_mount": 1.0, "no_such_joint": 1.0, "mount_to_shoulder": 0.5})

        assert agent.get_joint_state_by_name("mount_to_shoulder")[0] == pytest.approx(0.5)

    def test_attached_objects_follow_once_for_all_the_joints(self, sim_core, monkeypatch):
        agent = _lift(sim_core)
        box = _box(sim_core)
        assert agent.attach_object(box, parent_link_index=agent._joint_index_by_name(JOINT), keep_world_pose=True)
        z = box.get_pose().position[2]
        follows = []
        real = agent.update_attached_objects_kinematics
        monkeypatch.setattr(agent, "update_attached_objects_kinematics", lambda: (follows.append(1), real())[1])

        assert agent.set_joints_positions({JOINT: 3.0})

        assert follows == [1]
        assert box.get_pose().position[2] == pytest.approx(z + 3.0, abs=1e-6)

    def test_the_body_counts_as_moved(self, sim_core):
        agent = _manipulator(sim_core)
        sim_core._moved_this_step.clear()

        agent.set_joints_positions({"mount_to_shoulder": 0.5})

        assert agent.object_id in sim_core._moved_this_step
