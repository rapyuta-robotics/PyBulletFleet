"""A body built from shapes lands on its simulation's physics client, not PyBullet's default one.

PyBullet calls without ``physicsClientId`` go to client 0. These tests keep a
second client connected so that the simulation's own client is not 0, which is
what happens with two simulations in one process.
"""

import os

import pybullet as p
import pytest

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject

MESH_PATH = os.path.join(os.path.dirname(__file__), "..", "pybullet_fleet", "mesh", "cube.obj")


@pytest.fixture
def two_clients():
    """A bystander client and a simulation on another one; the simulation's client is never 0."""
    # PyBullet hands out the lowest free client id, so the bystander, connected
    # first, takes 0 if it is free, and the simulation gets another id.
    bystander = p.connect(p.DIRECT)
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False, log_level="warning"))
    assert sim.client != 0
    yield sim
    sim.close()
    p.disconnect(bystander)


def _mesh():
    return ShapeParams(shape_type="mesh", mesh_path=MESH_PATH, mesh_scale=[0.2, 0.2, 0.2])


def _box():
    return ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1])


def _bodies(client):
    return p.getNumBodies(physicsClientId=client)


class TestAgentFromMesh:
    @pytest.mark.parametrize("shape", [_box, _mesh], ids=["box", "mesh"])
    def test_creates_its_body_on_the_simulations_client(self, two_clients, shape):
        sim = two_clients
        default_before = _bodies(0) if p.isConnected(0) else None
        before = _bodies(sim.client)
        agent = Agent.from_mesh(
            visual_shape=shape(),
            collision_shape=shape(),
            pose=Pose.from_xyz(1.0, 2.0, 0.5),
            sim_core=sim,
        )
        assert _bodies(sim.client) == before + 1
        if default_before is not None:
            assert _bodies(0) == default_before
        position, _ = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=sim.client)
        assert position == pytest.approx((1.0, 2.0, 0.5))

    @pytest.mark.parametrize("shape", [_box, _mesh], ids=["box", "mesh"])
    def test_from_params_creates_its_body_on_the_simulations_client(self, two_clients, shape):
        sim = two_clients
        default_before = _bodies(0)
        before = _bodies(sim.client)
        params = AgentSpawnParams(visual_shape=shape(), collision_shape=shape(), initial_pose=Pose.from_xyz(1.0, 2.0, 0.5))
        agent = Agent.from_params(params, sim)
        assert _bodies(sim.client) == before + 1
        assert _bodies(0) == default_before
        position, _ = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=sim.client)
        assert position == pytest.approx((1.0, 2.0, 0.5))


class TestTheSharedShapeCache:
    def test_does_not_hand_one_clients_shapes_to_another(self, two_clients):
        sim = two_clients
        other = p.connect(p.DIRECT)
        try:
            # One extra shape on the simulation's client, so the same mesh gets
            # a different id on each client and a shared id would show.
            p.createVisualShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=sim.client)
            on_other = SimObject.create_shared_shapes(visual_shape=_mesh(), physics_client_id=other)
            on_sim = SimObject.create_shared_shapes(visual_shape=_mesh(), physics_client_id=sim.client)
            assert on_sim != on_other
            # A shape id is only meaningful on the client that made it: the
            # second call must have built its own, which its client can draw.
            assert p.getVisualShapeData(
                p.createMultiBody(baseVisualShapeIndex=on_sim[0], physicsClientId=sim.client), physicsClientId=sim.client
            )
            assert SimObject.create_shared_shapes(visual_shape=_mesh(), physics_client_id=other) == on_other
            assert SimObject.create_shared_shapes(visual_shape=_mesh(), physics_client_id=sim.client) == on_sim
            assert len(SimObject._shared_shapes) == 2
        finally:
            p.disconnect(other)


def _new_sim():
    return MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False, log_level="warning"))


def _assert_builds_real_geometry(client):
    """Shapes from the cache must exist on ``client``: a body built from them has visual and collision data."""
    visual_id, collision_id = SimObject.create_shared_shapes(
        visual_shape=_mesh(), collision_shape=_mesh(), physics_client_id=client
    )
    body = p.createMultiBody(
        baseMass=1.0, baseVisualShapeIndex=visual_id, baseCollisionShapeIndex=collision_id, physicsClientId=client
    )
    assert p.getVisualShapeData(body, physicsClientId=client)
    assert p.getCollisionShapeData(body, -1, physicsClientId=client)


class TestTheSharedShapeCacheAcrossConnections:
    """PyBullet reuses a client id after a disconnect, so a cache keyed by the id
    alone would hand the next simulation shape ids from the old connection."""

    @pytest.mark.parametrize("ending", ["close", "disconnect"])
    def test_a_later_simulation_on_a_reused_client_id_gets_its_own_shapes(self, ending):
        first = _new_sim()
        SimObject.create_shared_shapes(visual_shape=_mesh(), collision_shape=_mesh(), physics_client_id=first.client)
        if ending == "close":
            first.close()
        else:  # a caller disconnecting the client itself, outside close()
            p.disconnect(first.client)
        second = _new_sim()
        try:
            assert second.client == first.client  # the case under test: PyBullet reused the id
            _assert_builds_real_geometry(second.client)
        finally:
            second.close()

    def test_close_drops_only_its_own_clients_entries(self):
        sim = _new_sim()
        other = p.connect(p.DIRECT)
        try:
            kept = SimObject.create_shared_shapes(visual_shape=_mesh(), physics_client_id=other)
            SimObject.create_shared_shapes(visual_shape=_mesh(), physics_client_id=sim.client)
            sim.close()
            assert len(SimObject._shared_shapes) == 1
            assert SimObject.create_shared_shapes(visual_shape=_mesh(), physics_client_id=other) == kept
        finally:
            p.disconnect(other)

    def test_reset_rebuilds_shapes_that_reset_simulation_removed(self):
        sim = _new_sim()
        try:
            SimObject.create_shared_shapes(visual_shape=_mesh(), collision_shape=_mesh(), physics_client_id=sim.client)
            sim.reset()
            _assert_builds_real_geometry(sim.client)
        finally:
            sim.close()


class TestTheClientIsExplicit:
    """Shape and body helpers do not fall back to PyBullet's default client unnoticed."""

    def test_physics_client_of_a_simulation_is_its_client(self, two_clients):
        from pybullet_fleet.sim_object import _physics_client_of

        assert _physics_client_of(two_clients) == two_clients.client

    def test_physics_client_of_no_simulation_is_the_default_client(self):
        from pybullet_fleet.sim_object import _physics_client_of

        assert _physics_client_of(None) == 0

    def test_create_shared_shapes_without_a_client_warns(self, two_clients):
        with pytest.warns(DeprecationWarning, match="physics_client_id"):
            SimObject.create_shared_shapes(visual_shape=_box())

    def test_create_shared_shapes_with_a_client_does_not_warn(self, two_clients, recwarn):
        SimObject.create_shared_shapes(visual_shape=_box(), physics_client_id=two_clients.client)

        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_create_body_from_shapes_requires_a_client(self, two_clients):
        with pytest.raises(TypeError, match="physics_client_id"):
            SimObject._create_body_from_shapes(
                visual_id=-1, collision_id=-1, pose=Pose.from_xyz(0, 0, 0)
            )  # pyright: ignore[reportCallIssue]

    def test_load_urdf_body_requires_a_client(self, two_clients):
        from pybullet_fleet.sim_object import load_urdf_body

        with pytest.raises(TypeError, match="physics_client_id"):
            load_urdf_body("robots/elevator.urdf")  # pyright: ignore[reportCallIssue]
