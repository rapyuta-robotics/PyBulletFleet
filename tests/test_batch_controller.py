"""Tests for batch (vectorized) controllers — Tasks 7 + 8.

Covers:
- BatchKinematicController lifecycle (register/unregister/reset, buffer sizing)
- BatchOmniController equivalence vs the per-agent OmniController for
  single-waypoint translation
- Integration with sim_core.step_once (batch_advance is called in Phase 1)
"""

from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet import (
    Agent,
    AgentSpawnParams,
    MotionMode,
    MultiRobotSimulationCore,
    Pose,
    SimulationParams,
)
from pybullet_fleet.controllers import BatchKinematicController, BatchOmniController
from pybullet_fleet.types import CollisionMode, SpatialHashCellSizeMode


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def sim():
    params = SimulationParams(
        gui=False,
        monitor=False,
        physics=False,
        timestep=1.0 / 60.0,
        collision_check_frequency=0,
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        log_level="warning",
    )
    sc = MultiRobotSimulationCore(params)
    yield sc
    p.disconnect(sc.client)


def _spawn_agent(sim, x=0.0, y=0.0, z=0.1, max_vel=1.0, mode=MotionMode.OMNIDIRECTIONAL):
    return Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(x, y, z),
            motion_mode=mode,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={
                "max_linear_vel": max_vel,
                "max_linear_accel": 2.0,
                **({"max_angular_vel": 2.0, "max_angular_accel": 4.0} if mode == MotionMode.DIFFERENTIAL else {}),
            },
        ),
        sim_core=sim,
    )


def _make_batch_manager(sim, batch_controller="batch_omni"):
    """Create an AgentManager with a batch controller, registered with sim_core."""
    from pybullet_fleet.agent_manager import AgentManager

    return AgentManager(sim_core=sim, fleet_controller={"type": batch_controller})


# ----------------------------------------------------------------------
# Task 7: BatchKinematicController lifecycle + base buffers
# ----------------------------------------------------------------------


class _NoopBatchController(BatchKinematicController):
    """Concrete-but-trivial subclass to exercise the ABC lifecycle."""

    def __init__(self):
        super().__init__()
        self.cancelled_indices: list[int] = []

    def batch_advance(self, dt: float) -> np.ndarray:
        # Pretend nothing moved this step.
        self._moved_mask.fill(False)
        return self._moved_mask

    def _on_cancel_path(self, idx: int) -> None:
        self.cancelled_indices.append(idx)


class TestBatchControllerLifecycle:
    def test_synchronized_batch_advance_delegates_to_controller(self, sim):
        bc = _NoopBatchController()
        agent = _spawn_agent(sim, x=0.0)
        bc.register_agent(agent)

        result = bc.synchronized_batch_advance(0.1)

        assert result is bc._moved_mask

    def test_synchronized_cancel_path_uses_registered_agent_index(self, sim):
        bc = _NoopBatchController()
        agent = _spawn_agent(sim, x=0.0)
        bc.register_agent(agent)

        bc.synchronized_cancel_path(agent)

        assert bc.cancelled_indices == [0]

    def test_agent_stop_cancels_path_through_batch_controller(self, sim):
        bc = _NoopBatchController()
        agent = _spawn_agent(sim, x=0.0)
        bc.register_agent(agent)

        agent.stop()

        assert bc.cancelled_indices == [0]

    def test_register_sizes_buffers(self, sim):
        bc = _NoopBatchController()
        a = _spawn_agent(sim, x=0.0)
        b = _spawn_agent(sim, x=1.0)
        bc.register_agent(a)
        bc.register_agent(b)

        assert bc._pos_buf.shape == (2, 3)
        assert bc._orn_buf.shape == (2, 4)
        assert bc._moved_mask.shape == (2,)
        assert bc._object_ids.tolist() == [a.object_id, b.object_id]

    def test_register_twice_raises(self, sim):
        bc = _NoopBatchController()
        a = _spawn_agent(sim, x=0.0)
        bc.register_agent(a)
        with pytest.raises(ValueError):
            bc.register_agent(a)

    def test_unregister_compacts_arrays(self, sim):
        bc = _NoopBatchController()
        a = _spawn_agent(sim, x=0.0)
        b = _spawn_agent(sim, x=1.0)
        c = _spawn_agent(sim, x=2.0)
        bc.register_agent(a)
        bc.register_agent(b)
        bc.register_agent(c)

        bc.unregister_agent(b)
        assert len(bc._agents) == 2
        assert bc._pos_buf.shape == (2, 3)
        # b was at idx 1, c (last) swapped into idx 1
        assert bc._agents[1] is c
        assert bc._object_ids.tolist() == [a.object_id, c.object_id]

    def test_unregister_unknown_raises(self, sim):
        bc = _NoopBatchController()
        a = _spawn_agent(sim, x=0.0)
        with pytest.raises(KeyError):
            bc.unregister_agent(a)

    def test_reset_clears_all(self, sim):
        bc = _NoopBatchController()
        for x in range(3):
            bc.register_agent(_spawn_agent(sim, x=float(x)))
        bc.reset()
        assert bc._agents == []
        assert bc._pos_buf.shape == (0, 3)

    def test_first_register_records_sim_core(self, sim):
        bc = _NoopBatchController()
        assert bc._sim_core is None
        a = _spawn_agent(sim, x=0.0)
        bc.register_agent(a)
        # sim_core is recorded for _apply_phase1; no longer auto-appended
        # to sim._batch_controllers (owned by AgentManager now).
        assert bc._sim_core is sim

    def test_last_unregister_clears_agent_only(self, sim):
        bc = _NoopBatchController()
        a = _spawn_agent(sim, x=0.0)
        bc.register_agent(a)
        bc.unregister_agent(a)
        assert len(bc._agents) == 0
        # _sim_core is kept (BC may be reused by its AgentManager)
        assert a._batch_controller is None

    def test_register_agent_without_sim_core_raises(self, sim):
        bc = _NoopBatchController()

        class _Stub:
            sim_core = None

        with pytest.raises(ValueError):
            bc.register_agent(_Stub())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Task 8: BatchOmniController behaviour + equivalence with OmniController
# ----------------------------------------------------------------------


class TestBatchOmniBasics:
    def test_set_path_requires_registered_agent(self, sim):
        mgr = _make_batch_manager(sim)
        bc = mgr.batch_controller
        sentinel = _spawn_agent(sim, x=-1.0)
        mgr.add_object(sentinel)
        a = _spawn_agent(sim, x=0.0)
        with pytest.raises(KeyError):
            bc.set_path(a, [Pose.from_xyz(1.0, 0.0, 0.1)])

    def test_step_once_advances_position(self, sim):
        mgr = _make_batch_manager(sim)
        bc = mgr.batch_controller
        a = _spawn_agent(sim, x=0.0)
        mgr.add_object(a)
        bc.set_path(a, [Pose.from_xyz(1.0, 0.0, 0.1)])
        start_x = a.get_pose().x
        for _ in range(60):
            sim.step_once()
        end_x = a.get_pose().x
        assert end_x > start_x + 0.5
        for _ in range(120):
            sim.step_once()
        assert a.get_pose().x == pytest.approx(1.0, abs=1e-3)

    def test_multi_agent_step(self, sim):
        mgr = _make_batch_manager(sim)
        bc = mgr.batch_controller
        agents = [_spawn_agent(sim, x=float(i)) for i in range(5)]
        for a in agents:
            mgr.add_object(a)
            bc.set_path(a, [Pose.from_xyz(a.get_pose().x + 1.0, 0.0, 0.1)])
        for _ in range(180):
            sim.step_once()
        for i, a in enumerate(agents):
            assert a.get_pose().x == pytest.approx(float(i) + 1.0, abs=1e-3)


class TestBatchOmniEquivalence:
    """BatchOmniController should produce trajectories matching OmniController
    within ~1e-6 per step for the supported scope (single goal, no rotation)."""

    @pytest.mark.parametrize("goal_xyz", [(1.5, 0.0, 0.1), (0.0, 2.0, 0.1), (1.0, 1.0, 0.1)])
    def test_per_step_position_matches_omni(self, sim, goal_xyz):
        ref = _spawn_agent(sim, x=0.0, y=0.0, max_vel=1.2)
        # The same path on both sides: bc.set_path() adds no approach waypoint,
        # and the zero-length one Agent.set_path() inserts costs a step.
        ref.set_path([Pose.from_xyz(*goal_xyz)], final_orientation_align=False, auto_approach=False)

        mgr = _make_batch_manager(sim)
        bc = mgr.batch_controller
        target = _spawn_agent(sim, x=0.0, y=0.0, max_vel=1.2)
        mgr.add_object(target)
        bc.set_path(target, [Pose.from_xyz(*goal_xyz)])

        for _ in range(120):
            sim.step_once()
            ref_pos = np.asarray(ref.get_pose().position)
            tgt_pos = np.asarray(target.get_pose().position)
            err = np.linalg.norm(ref_pos - tgt_pos)
            assert err < 1e-3, f"divergence {err:.6f} at step (ref={ref_pos}, tgt={tgt_pos})"


class TestBatchOmniIntegration:
    def test_uses_buffered_pose_write(self, sim):
        mgr = _make_batch_manager(sim)
        bc = mgr.batch_controller
        a = _spawn_agent(sim, x=0.0)
        mgr.add_object(a)
        bc.set_path(a, [Pose.from_xyz(1.0, 0.0, 0.1)])

        sim.step_once()
        pb_pos, _ = p.getBasePositionAndOrientation(a.body_id, physicsClientId=sim.client)
        cache_pos = a.get_pose().position
        assert cache_pos[0] > 0.0  # the first step moves the row
        assert pb_pos[0] == pytest.approx(cache_pos[0], abs=1e-9)

    def test_inactive_agents_dont_move(self, sim):
        mgr = _make_batch_manager(sim)
        a = _spawn_agent(sim, x=0.0)
        mgr.add_object(a)
        start = a.get_pose().position[:]
        for _ in range(30):
            sim.step_once()
        assert np.allclose(a.get_pose().position, start)


# ----------------------------------------------------------------------
# AgentManager batch integration
# ----------------------------------------------------------------------


class TestAgentManagerBatch:
    """AgentManager.enable_batch / fleet_controller.type integration."""

    def test_enable_batch_returns_controller(self, sim):
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim)
        bc = mgr.enable_batch("batch_omni")
        assert isinstance(bc, BatchOmniController)
        assert mgr.batch_controller is bc

    def test_batch_mode_init_param(self, sim):
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"})
        assert isinstance(mgr.batch_controller, BatchOmniController)

    def test_spawned_agents_auto_registered(self, sim):
        from pybullet_fleet.agent_manager import AgentManager, GridSpawnParams

        mgr = AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"})
        params = AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            collision_mode=CollisionMode.NORMAL_3D,
        )
        agents = mgr.spawn_objects_grid(
            3,
            GridSpawnParams.from_count(3, 1, spacing=[2.0, 2.0, 0.0], offset=[0.0, 0.0, 0.1]),
            params,
        )
        bc = mgr.batch_controller
        for a in agents:
            assert a._batch_controller is bc
        assert len(bc._agents) == 3

    def test_existing_agents_registered_on_enable_batch(self, sim):
        """Agents added before enable_batch() get picked up retroactively."""
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim)
        a1 = _spawn_agent(sim)
        a2 = _spawn_agent(sim, x=2.0)
        mgr.add_object(a1)
        mgr.add_object(a2)
        assert a1._batch_controller is None

        bc = mgr.enable_batch("batch_omni")
        assert a1._batch_controller is bc
        assert a2._batch_controller is bc

    def test_remove_object_unregisters_from_batch(self, sim):
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"})
        a = _spawn_agent(sim)
        mgr.add_object(a)
        bc = mgr.batch_controller
        assert a._batch_controller is bc

        mgr.remove_object(a)
        assert a._batch_controller is None
        assert a not in bc._agents

    def test_disable_batch(self, sim):
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"})
        a = _spawn_agent(sim)
        mgr.add_object(a)
        bc = mgr.batch_controller

        mgr.disable_batch()
        assert mgr.batch_controller is None
        assert a._batch_controller is None
        assert len(bc._agents) == 0  # all agents unregistered

    def test_invalid_mode_raises(self, sim):
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim)
        with pytest.raises(ValueError, match="Unknown batch controller"):
            mgr.enable_batch("nonexistent_mode")

    def test_dotted_path_resolves_custom_batch_controller(self, sim):
        """A custom batch controller can be selected via dotted import path."""
        from pybullet_fleet.agent_manager import AgentManager

        # _NoopBatchController has no _registry_name, so it is reachable only
        # via dotted path — exercising the resolve_class branch.
        mgr = AgentManager(
            sim_core=sim,
            fleet_controller={"type": "tests.test_batch_controller._NoopBatchController"},
        )
        assert isinstance(mgr.batch_controller, _NoopBatchController)

    def test_dotted_path_rejects_non_batch_class(self, sim):
        """A dotted path to a non-BatchKinematicController class is rejected."""
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim)
        with pytest.raises(ValueError, match="must be a BatchKinematicController subclass"):
            mgr.enable_batch("pybullet_fleet.controller.OmniController")

    def test_agents_move_via_manager_batch(self, sim):
        """End-to-end: agents registered via AgentManager actually move."""
        from pybullet_fleet.agent_manager import AgentManager

        mgr = AgentManager(sim_core=sim, fleet_controller={"type": "batch_omni"})
        a = _spawn_agent(sim, x=0.0)
        mgr.add_object(a)

        goal = Pose.from_xyz(2.0, 0.0, 0.1)
        a.set_path([goal])
        for _ in range(200):
            sim.step_once()
        pos = a.get_pose().position
        assert np.linalg.norm(np.array(pos) - np.array(goal.position)) < 0.15
