"""Tests for SimObject.from_urdf() URDF loading.

The counterpart of ``tests/test_sdf_loader.py``: SDF already had a
non-agent loader, URDF did not, so a URDF-defined body with no behaviour
had to be created as an ``Agent`` and then visited by the per-step update
loop for the life of the process.
"""

import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams
from pybullet_fleet.types import CollisionMode

URDF = "cube_small.urdf"  # ships with pybullet_data


@pytest.fixture
def sim_core():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_monitor_gui=False))
    sim.initialize_simulation()
    yield sim
    try:
        p.disconnect(sim.client)
    except p.error:
        pass


class TestFromUrdf:
    def test_loads_and_registers(self, sim_core):
        obj = SimObject.from_urdf(URDF, sim_core=sim_core)
        assert isinstance(obj, SimObject)
        assert not isinstance(obj, Agent)
        assert obj.object_id >= 0
        assert obj in sim_core.sim_objects

    def test_honours_the_pose_it_is_given(self, sim_core):
        obj = SimObject.from_urdf(URDF, pose=Pose.from_xyz(1.0, 2.0, 3.0), sim_core=sim_core)
        assert list(obj.get_pose().position) == pytest.approx([1.0, 2.0, 3.0])

    def test_defaults_to_static_kinematic_scenery(self, sim_core):
        """The case this exists for: static structure, walls and fixtures."""
        obj = SimObject.from_urdf(URDF, sim_core=sim_core, collision_mode=CollisionMode.STATIC)
        assert obj.mass == 0.0
        assert obj.is_kinematic is True
        assert obj.pickable is False
        assert obj.is_static is True

    def test_mass_none_keeps_the_urdf_values(self, sim_core):
        obj = SimObject.from_urdf(URDF, sim_core=sim_core, mass=None, use_fixed_base=False)
        assert obj.mass > 0.0

    def test_name_defaults_to_the_urdf_robot_name(self, sim_core):
        assert SimObject.from_urdf(URDF, sim_core=sim_core).name
        assert SimObject.from_urdf(URDF, sim_core=sim_core, name="shelf_a").name == "shelf_a"

    def test_it_stays_out_of_the_per_step_update_loop(self, sim_core):
        """The whole point. An Agent is visited every step to walk an empty
        action queue, check joints it has none of and sweep plugins it does
        not own; a SimObject is skipped by _needs_update."""
        scenery = SimObject.from_urdf(URDF, sim_core=sim_core)
        agent = Agent.from_params(AgentSpawnParams(urdf_path=URDF, name="bot", use_fixed_base=True), sim_core=sim_core)
        assert scenery._needs_update is False
        assert agent._needs_update is True

    def test_a_missing_file_is_reported_as_such(self, sim_core):
        with pytest.raises(FileNotFoundError, match="Failed to load URDF"):
            SimObject.from_urdf("/nonexistent/definitely_not_here.urdf", sim_core=sim_core)


class TestSpawnParamsUrdfPath:
    def test_from_params_routes_a_urdf_to_from_urdf(self, sim_core):
        obj = SimObject.from_params(
            SimObjectSpawnParams(
                urdf_path=URDF,
                name="tile",
                initial_pose=Pose.from_xyz(4.0, 0.0, 0.0),
                collision_mode=CollisionMode.STATIC,
            ),
            sim_core=sim_core,
        )
        assert obj.name == "tile"
        assert obj.collision_mode is CollisionMode.STATIC
        assert list(obj.get_pose().position) == pytest.approx([4.0, 0.0, 0.0])

    def test_from_dict_accepts_urdf_path(self, sim_core):
        obj = SimObject.from_dict({"name": "tile", "urdf_path": URDF, "pose": [1.0, 1.0, 0.0]}, sim_core=sim_core)
        assert list(obj.get_pose().position) == pytest.approx([1.0, 1.0, 0.0])

    def test_the_shape_path_is_unchanged(self, sim_core):
        """No urdf_path means the existing from_mesh route, exactly as before."""
        obj = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
                name="box",
                initial_pose=Pose.from_xyz(2.0, 0.0, 0.0),
            ),
            sim_core=sim_core,
        )
        assert obj.name == "box"
        assert list(obj.get_pose().position) == pytest.approx([2.0, 0.0, 0.0])


def test_field_order_is_unchanged_for_the_subclasses():
    """urdf_path is declared last on SimObjectSpawnParams on purpose.

    AgentSpawnParams and ElevatorParams inherit these fields, so declaring it
    anywhere else would silently change what the third positional argument
    means for all three dataclasses.
    """
    import dataclasses

    from pybullet_fleet.devices.elevator import ElevatorParams

    assert [f.name for f in dataclasses.fields(SimObjectSpawnParams)][-1] == "urdf_path"
    for cls in (AgentSpawnParams, ElevatorParams):
        names = [f.name for f in dataclasses.fields(cls)]
        assert names[:10] == [
            "visual_shape",
            "collision_shape",
            "initial_pose",
            "mass",
            "pickable",
            "name",
            "visual_frame_pose",
            "collision_frame_pose",
            "collision_mode",
            "user_data",
        ]
        assert names[10] == "urdf_path"


class TestReviewFollowUps:
    """Cases raised in review on the first version of this change."""

    def test_mass_none_totals_every_link_not_just_the_base(self, sim_core):
        """use_fixed_base zeroes the base link, so reading the base alone made
        a multi-link URDF come out mass 0 and is_kinematic True while its own
        links were still dynamic. Agent.from_urdf already totals them."""
        # kuka_iiwa: base link 0 kg under use_fixed_base, 17.5 kg across its
        # own links.
        obj = SimObject.from_urdf("kuka_iiwa", sim_core=sim_core, mass=None, use_fixed_base=True)
        assert obj.mass == pytest.approx(17.5, abs=0.1)
        assert obj.is_kinematic is False

    def test_a_urdf_spawn_is_recorded_like_a_shape_spawn(self, sim_core):
        """from_params() took an early return for the URDF branch, which
        skipped the state-recording block the shape branch runs, so a URDF
        object was silently missing from an active recording."""
        recorded = []
        sim_core._record_state_spawn = lambda obj, params: recorded.append((obj, params))

        params = SimObjectSpawnParams(urdf_path=URDF, name="tile")
        obj = SimObject.from_params(params, sim_core=sim_core)

        assert len(recorded) == 1
        spawned, effective = recorded[0]
        assert spawned is obj
        assert effective.urdf_path == obj._checkpoint_spawn_params.urdf_path
        assert effective.name == params.name
        assert effective.mass == obj.mass
        assert effective.pickable == obj.pickable
        assert effective.collision_mode == obj.collision_mode


class TestDisabledCollisionCoversEveryLink:
    """Raised in review: this factory accepts articulated URDFs, and the
    DISABLED filter was applied to link -1 alone -- so every child link kept
    colliding physically while the object was excluded from PyBulletFleet's
    own checks."""

    ARTICULATED = "kuka_iiwa"  # seven revolute joints

    @staticmethod
    def _contacting_links(sim_core, obj, other):
        """Which of obj's links PyBullet reports touching `other`."""
        import pybullet as pb

        pb.performCollisionDetection(physicsClientId=sim_core.client)
        touching = []
        for link in range(-1, pb.getNumJoints(obj.body_id, physicsClientId=sim_core.client)):
            if pb.getContactPoints(bodyA=obj.body_id, linkIndexA=link, bodyB=other.body_id, physicsClientId=sim_core.client):
                touching.append(link)
        return touching

    @pytest.fixture
    def overlapping(self, sim_core):
        """An articulated body and a box sharing the same space."""
        arm = SimObject.from_urdf(self.ARTICULATED, sim_core=sim_core, use_fixed_base=True)
        box = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[1.0, 1.0, 1.0]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[1.0, 1.0, 1.0]),
                name="box",
                initial_pose=Pose.from_xyz(0.0, 0.0, 0.5),
            ),
            sim_core=sim_core,
        )
        return arm, box

    def test_the_fixture_really_does_collide(self, sim_core, overlapping):
        """Otherwise the two tests below would pass without proving anything."""
        arm, box = overlapping
        touching = self._contacting_links(sim_core, arm, box)
        assert len(touching) > 1, "needs more than the base link touching to be meaningful"

    def test_disabling_filters_out_every_link(self, sim_core, overlapping):
        arm, box = overlapping
        arm.set_collision_mode(CollisionMode.DISABLED)
        assert self._contacting_links(sim_core, arm, box) == []

    def test_re_enabling_brings_them_all_back(self, sim_core, overlapping):
        """A superset, not an equality, and deliberately so.

        Re-enabling writes PyBullet's documented defaults (1, -1), which a
        fixed base loaded from URDF does not start with -- it is filtered more
        tightly -- so the base link can come back colliding where it did not
        before. That asymmetry is pre-existing for the base link and is now
        applied consistently to the rest; restoring the true original would
        need a getter PyBullet does not expose.
        """
        arm, box = overlapping
        before = self._contacting_links(sim_core, arm, box)

        arm.set_collision_mode(CollisionMode.DISABLED)
        arm.set_collision_mode(CollisionMode.NORMAL_3D)

        after = self._contacting_links(sim_core, arm, box)
        assert set(before) <= set(after), "every link that collided before collides again"
        assert set(after) - set(before) <= {-1}, "only the fixed base may differ"

    def test_it_also_applies_at_spawn(self, sim_core):
        arm = SimObject.from_urdf(
            self.ARTICULATED,
            sim_core=sim_core,
            use_fixed_base=True,
            collision_mode=CollisionMode.DISABLED,
        )
        box = SimObject.from_params(
            SimObjectSpawnParams(
                visual_shape=ShapeParams(shape_type="box", half_extents=[1.0, 1.0, 1.0]),
                collision_shape=ShapeParams(shape_type="box", half_extents=[1.0, 1.0, 1.0]),
                name="box",
                initial_pose=Pose.from_xyz(0.0, 0.0, 0.5),
            ),
            sim_core=sim_core,
        )
        assert self._contacting_links(sim_core, arm, box) == []


class TestSharedLoader:
    """`Agent.from_urdf` and `SimObject.from_urdf` load through one helper.

    They were near-identical and had already drifted: only one totalled the
    links' mass, and only one turned an unreadable file into a
    `FileNotFoundError`.
    """

    def test_both_factories_resolve_the_same_mass(self, sim_core):
        from pybullet_fleet import Agent, AgentSpawnParams

        obj = SimObject.from_urdf("kuka_iiwa", sim_core=sim_core, mass=None, use_fixed_base=True)
        agent = Agent.from_urdf("kuka_iiwa", sim_core=sim_core, mass=None, use_fixed_base=True)
        assert obj.mass == pytest.approx(agent.mass)
        assert obj.mass > 0.0

    def test_both_report_a_missing_file_the_same_way(self, sim_core):
        from pybullet_fleet import Agent

        with pytest.raises(FileNotFoundError, match="Failed to load URDF"):
            SimObject.from_urdf("/nonexistent/nope.urdf", sim_core=sim_core)
        with pytest.raises(FileNotFoundError, match="Failed to load URDF"):
            Agent.from_urdf("/nonexistent/nope.urdf", sim_core=sim_core)

    def test_both_accept_global_scaling(self, sim_core):
        import pybullet as pb

        from pybullet_fleet import Agent

        def extent(body):
            lo, hi = pb.getAABB(body.body_id, physicsClientId=sim_core.client)
            return hi[0] - lo[0]

        plain = SimObject.from_urdf(URDF, sim_core=sim_core, name="plain")
        scaled = SimObject.from_urdf(URDF, sim_core=sim_core, name="scaled", global_scaling=3.0)
        assert extent(scaled) > extent(plain) * 2

        agent_plain = Agent.from_urdf(URDF, sim_core=sim_core, name="ap")
        agent_scaled = Agent.from_urdf(URDF, sim_core=sim_core, name="as", global_scaling=3.0)
        assert extent(agent_scaled) > extent(agent_plain) * 2

    def test_kinematic_mass_zeroes_every_link_either_way(self, sim_core):
        import pybullet as pb

        from pybullet_fleet import Agent

        for body in (
            SimObject.from_urdf("kuka_iiwa", sim_core=sim_core, mass=0.0, use_fixed_base=True),
            Agent.from_urdf("kuka_iiwa", sim_core=sim_core, mass=0.0, use_fixed_base=True),
        ):
            n = pb.getNumJoints(body.body_id, physicsClientId=sim_core.client)
            for link in range(-1, n):
                assert pb.getDynamicsInfo(body.body_id, link, physicsClientId=sim_core.client)[0] == 0.0


class TestTheModelNameIsResolvedOnce:
    """Raised in review: `Agent.from_urdf()` resolved the name a second time
    after the shared helper had already resolved it and loaded that path.

    For an auto-discovered name that repeats an uncached
    `robot_descriptions` scan, and a second resolution that disagreed would
    leave the body just loaded orphaned.
    """

    def _count_resolutions(self, monkeypatch):
        """Count every resolution, wherever the name was bound.

        `agent.py` used to do `from .robot_models import resolve_model`, so
        patching only `robot_models` misses its call entirely -- the name was
        bound at import time. Both are patched, with `raising=False` because
        the module-level binding is gone once the duplicate resolution is.
        """
        import pybullet_fleet.agent as agent_module
        import pybullet_fleet.robot_models as robot_models

        calls = []
        original = robot_models.resolve_model

        def counting(name, *args, **kwargs):
            calls.append(name)
            return original(name, *args, **kwargs)

        monkeypatch.setattr(robot_models, "resolve_model", counting)
        monkeypatch.setattr(agent_module, "resolve_model", counting, raising=False)
        return calls

    def test_the_agent_factory_resolves_once(self, sim_core, monkeypatch):
        from pybullet_fleet import Agent

        calls = self._count_resolutions(monkeypatch)
        Agent.from_urdf(URDF, sim_core=sim_core, name="bot")
        assert calls == [URDF]

    def test_the_sim_object_factory_resolves_once(self, sim_core, monkeypatch):
        calls = self._count_resolutions(monkeypatch)
        SimObject.from_urdf(URDF, sim_core=sim_core, name="thing")
        assert calls == [URDF]

    def test_the_agent_keeps_the_resolved_path(self, sim_core):
        """What the second resolution was there for."""
        from pybullet_fleet import Agent
        from pybullet_fleet.robot_models import resolve_model

        agent = Agent.from_urdf(URDF, sim_core=sim_core, name="bot")
        assert agent.urdf_path == resolve_model(URDF)
