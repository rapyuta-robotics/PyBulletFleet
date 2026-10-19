"""
Tests for the two-phase step refactor (spec: docs/design/two-phase-step/spec.md).

Task 2 scope: set_pose buffers writes when _in_step=True, flushes in Phase 2.
Outside step_once(), set_pose still goes to PyBullet immediately.
"""

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
from pybullet_fleet.events import SimEvents
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionMode, SpatialHashCellSizeMode


@pytest.fixture
def sim():
    """Kinematic sim_core suitable for two-phase tests."""
    params = SimulationParams(
        gui=False,
        monitor=False,
        physics=False,
        timestep=1.0 / 240.0,
        collision_check_frequency=0,
        spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
        spatial_hash_cell_size=2.0,
        log_level="warning",
    )
    sc = MultiRobotSimulationCore(params)
    yield sc
    p.disconnect(sc.client)


def _make_agent(sim, x=0.0, y=0.0, z=0.1):
    return Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/simple_cube.urdf",
            initial_pose=Pose.from_xyz(x, y, z),
            motion_mode=MotionMode.OMNIDIRECTIONAL,
            collision_mode=CollisionMode.NORMAL_3D,
            controller={"max_linear_vel": 1.0},
        ),
        sim_core=sim,
    )


class TestTwoPhaseStepState:
    """Task 1: _in_step flag + _pending_pose_ids buffer (sim_core state lifecycle)."""

    def test_in_step_false_outside_step(self, sim):
        assert sim._in_step is False
        assert sim._pending_pose_ids == set()

    def test_in_step_true_inside_callback(self, sim):
        seen = {}

        def cb(sim_core, dt):
            seen["in_step"] = sim_core._in_step

        sim.register_callback(cb)
        sim.step_once()
        assert seen["in_step"] is True
        # And it's cleared again after step_once returns
        assert sim._in_step is False


class TestImmediatePathOutsideStep:
    """Outside step_once(), set_pose must remain synchronous (legacy contract)."""

    def test_set_pose_writes_to_pybullet_immediately(self, sim):
        agent = _make_agent(sim, 0, 0, 0.1)
        agent.set_pose(Pose.from_xyz(5.0, 6.0, 0.1))
        # PyBullet reflects it right now
        pos, _ = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=sim.client)
        assert pos[0] == pytest.approx(5.0)
        assert pos[1] == pytest.approx(6.0)
        # No leftover buffer state
        assert sim._pending_pose_ids == set()


class TestBufferedPathDuringStep:
    """Inside step_once(), set_pose must defer the PyBullet write to Phase 2."""

    def test_callback_set_pose_is_buffered_then_flushed(self, sim):
        agent = _make_agent(sim, 0, 0, 0.1)
        observed = {}

        def cb(_sim, _dt):
            agent.set_pose(Pose.from_xyz(3.0, 4.0, 0.1))
            # Pose cache is updated eagerly (D2)
            cached = agent.get_pose()
            observed["cached_position"] = (cached.position[0], cached.position[1])
            # Buffer holds the pending write (object id registered with sim_core)
            observed["in_pending_ids"] = agent.object_id in _sim._pending_pose_ids
            # PyBullet has NOT been written yet (still at the spawn pose)
            pos, _ = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=_sim.client)
            observed["pybullet_during_callback"] = (pos[0], pos[1])

        sim.register_callback(cb)
        sim.step_once()

        # Cache updated immediately for in-step observers
        assert observed["cached_position"] == pytest.approx((3.0, 4.0))
        assert observed["in_pending_ids"] is True
        # PyBullet still showed the old pose mid-callback
        assert observed["pybullet_during_callback"] == pytest.approx((0.0, 0.0))

        # After step_once(): buffer drained, PyBullet now matches the new pose
        assert sim._pending_pose_ids == set()
        pos, _ = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=sim.client)
        assert pos[0] == pytest.approx(3.0)
        assert pos[1] == pytest.approx(4.0)

    def test_flush_handles_multiple_objects(self, sim):
        agents = [_make_agent(sim, i * 2.0, 0, 0.1) for i in range(5)]
        targets = [(10.0 + i, 5.0, 0.1) for i in range(5)]

        def cb(_sim, _dt):
            for ag, tgt in zip(agents, targets):
                ag.set_pose(Pose.from_xyz(*tgt))

        sim.register_callback(cb)
        sim.step_once()

        for ag, tgt in zip(agents, targets):
            pos, _ = p.getBasePositionAndOrientation(ag.body_id, physicsClientId=sim.client)
            assert pos[0] == pytest.approx(tgt[0])
            assert pos[1] == pytest.approx(tgt[1])
        assert sim._pending_pose_ids == set()

    def test_last_write_wins_within_step(self, sim):
        agent = _make_agent(sim, 0, 0, 0.1)

        def cb(_sim, _dt):
            agent.set_pose(Pose.from_xyz(1.0, 1.0, 0.1))
            agent.set_pose(Pose.from_xyz(2.0, 2.0, 0.1))
            agent.set_pose(Pose.from_xyz(7.0, 8.0, 0.1))

        sim.register_callback(cb)
        sim.step_once()

        pos, _ = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=sim.client)
        assert pos[0] == pytest.approx(7.0)
        assert pos[1] == pytest.approx(8.0)

    def test_buffer_cleared_even_if_callback_raises(self, sim):
        agent = _make_agent(sim, 0, 0, 0.1)

        def cb(_sim, _dt):
            agent.set_pose(Pose.from_xyz(1.0, 1.0, 0.1))
            raise RuntimeError("boom")

        sim.register_callback(cb)
        with pytest.raises(RuntimeError, match="boom"):
            sim.step_once()
        # _in_step must be cleared so subsequent set_pose calls take the immediate path
        assert sim._in_step is False

    def test_noop_set_pose_skips_buffer(self, sim):
        """set_pose with unchanged position must NOT be registered for flush
        (saves redundant p.resetBasePositionAndOrientation calls for idle agents)."""
        agent = _make_agent(sim, 1.5, 2.5, 0.1)

        def cb(_sim, _dt):
            # Re-write the exact same pose: no movement, no flush needed.
            agent.set_pose(Pose.from_xyz(1.5, 2.5, 0.1))
            # Mid-callback: buffer must NOT contain this object.
            assert agent.object_id not in _sim._pending_pose_ids

        sim.register_callback(cb)
        sim.step_once()
        # And after step_once: still nothing pending and PyBullet pose unchanged.
        assert sim._pending_pose_ids == set()
        pos, _ = p.getBasePositionAndOrientation(agent.body_id, physicsClientId=sim.client)
        assert pos[0] == pytest.approx(1.5)
        assert pos[1] == pytest.approx(2.5)


class TestDeferredAABBAndGrid:
    """Task 3: AABB + spatial grid refresh moved out of _flush_pending_poses
    into the dedicated _flush_aabb_and_grid() Phase 3 step.

    Externally, the contract is identical: after step_once() returns, AABBs
    and spatial-grid cells reflect every buffered pose write, and collision
    detection sees the up-to-date geometry.
    """

    def test_aabb_reflects_buffered_pose_after_step(self, sim):
        agent = _make_agent(sim, 0, 0, 0.1)
        # Force AABB to be computed at spawn pose.
        sim._update_object_aabb(agent.object_id)
        aabb_before = sim._cached_aabbs_dict[agent.object_id]

        def cb(_sim, _dt):
            agent.set_pose(Pose.from_xyz(20.0, 0.0, 0.1))

        sim.register_callback(cb)
        sim.step_once()

        aabb_after = sim._cached_aabbs_dict[agent.object_id]
        # AABB center should have moved with the buffered pose write.
        cx_before = 0.5 * (aabb_before[0][0] + aabb_before[1][0])
        cx_after = 0.5 * (aabb_after[0][0] + aabb_after[1][0])
        assert cx_after - cx_before == pytest.approx(20.0, abs=0.05)
        assert sim._pending_pose_ids == set()

    def test_spatial_grid_reflects_buffered_pose_after_step(self, sim):
        agent = _make_agent(sim, 0, 0, 0.1)
        # Initialize cell size + AABBs + grid (mimics what real check_collisions setup does).
        sim.set_collision_spatial_hash_cell_size_mode()
        sim._update_object_aabb(agent.object_id)
        sim._update_object_spatial_grid(agent.object_id)
        assert sim._cached_cell_size is not None

        def cb(_sim, _dt):
            # Teleport far enough to land in a different 2.0-sized cell.
            agent.set_pose(Pose.from_xyz(20.0, 20.0, 0.1))

        sim.register_callback(cb)
        sim.step_once()

        # The grid must contain the agent under its new cell key.
        found_cells = [cell for cell, ids in sim._cached_spatial_grid.items() if agent.object_id in ids]
        assert len(found_cells) >= 1
        # And the cell key must correspond to the new (20, 20) area, not (0, 0).
        for cell in found_cells:
            assert cell[0] >= 5  # 20.0 / 2.0 cell size = 10; allow some AABB padding
            assert cell[1] >= 5
        assert sim._pending_pose_ids == set()

    def test_collision_detection_sees_buffered_pose(self, sim):
        # Two agents far apart at spawn; callback teleports them on top of each other.
        a = _make_agent(sim, 0.0, 0.0, 0.1)
        b = _make_agent(sim, 50.0, 50.0, 0.1)

        def cb(_sim, _dt):
            a.set_pose(Pose.from_xyz(0.0, 0.0, 0.1))
            b.set_pose(Pose.from_xyz(0.0, 0.0, 0.1))

        sim.register_callback(cb)
        sim.step_once()
        # check_collisions() ran inside step_once with the new buffered poses.
        # The pair (a, b) must have been considered (both share the same grid cell).
        collisions = sim.check_collisions()[0]
        pair = {a.object_id, b.object_id}
        assert any(
            {c[0], c[1]} == pair for c in collisions
        ), f"Expected collision between agents at same position; got {collisions}"


def _make_box(sim, x, y, z, half=0.1):
    """Spawn a small box SimObject attached/movable via set_pose."""
    return SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="box", half_extents=[half, half, half], rgba_color=[0.5, 0.5, 0.5, 1.0]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[half, half, half]),
        pose=Pose.from_xyz(x, y, z),
        mass=0.0,  # kinematic
        sim_core=sim,
    )


class TestAttachedPropagationDuringStep:
    """Task 4: attached-object propagation must work correctly under the buffered
    write path. The recursive _propagate_to_attached() call inside _set_pose_internal
    keeps each child's _cached_pose in sync (D2 source of truth) AND registers each
    child into _pending_pose_ids so Phase 2/3 flushes its PyBullet write + AABB/grid.
    """

    def test_single_level_attached_propagates_in_buffered_path(self, sim):
        parent = _make_agent(sim, 0.0, 0.0, 0.1)
        child = _make_box(sim, 0.0, 0.0, 0.1)
        parent.attach_object(child, relative_pose=Pose.from_xyz(0.5, 0.0, 0.0))

        def cb(_sim, _dt):
            parent.set_pose(Pose.from_xyz(10.0, 0.0, 0.1))
            # Mid-callback (D2): child cache must already reflect the new world pose.
            cp = child.get_pose().position
            assert cp[0] == pytest.approx(10.5, abs=1e-6)
            # And the child must be registered for Phase 2/3 flush.
            assert child.object_id in _sim._pending_pose_ids
            assert parent.object_id in _sim._pending_pose_ids

        sim.register_callback(cb)
        sim.step_once()

        # After step: PyBullet reflects the propagated child pose.
        pos, _ = p.getBasePositionAndOrientation(child.body_id, physicsClientId=sim.client)
        assert pos[0] == pytest.approx(10.5, abs=1e-6)
        assert sim._pending_pose_ids == set()

    def test_multi_level_attached_propagates_in_buffered_path(self, sim):
        # parent → child → grandchild  (chain via base-link attachment)
        parent = _make_agent(sim, 0.0, 0.0, 0.1)
        child = _make_box(sim, 0.0, 0.0, 0.1)
        grandchild = _make_box(sim, 0.0, 0.0, 0.1)
        parent.attach_object(child, relative_pose=Pose.from_xyz(0.5, 0.0, 0.0))
        child.attach_object(grandchild, relative_pose=Pose.from_xyz(0.3, 0.0, 0.0))

        def cb(_sim, _dt):
            parent.set_pose(Pose.from_xyz(7.0, 2.0, 0.1))

        sim.register_callback(cb)
        sim.step_once()

        # Parent + child + grandchild all flushed to PyBullet with correct offsets.
        pp, _ = p.getBasePositionAndOrientation(parent.body_id, physicsClientId=sim.client)
        cp, _ = p.getBasePositionAndOrientation(child.body_id, physicsClientId=sim.client)
        gp, _ = p.getBasePositionAndOrientation(grandchild.body_id, physicsClientId=sim.client)
        assert pp[0] == pytest.approx(7.0, abs=1e-6)
        assert cp[0] == pytest.approx(7.5, abs=1e-6)
        assert gp[0] == pytest.approx(7.8, abs=1e-6)

    def test_attached_noop_skip(self, sim):
        """Parent no-op set_pose must NOT trigger child propagation (saves redundant work)."""
        parent = _make_agent(sim, 1.0, 2.0, 0.1)
        child = _make_box(sim, 0.0, 0.0, 0.1)
        parent.attach_object(child, relative_pose=Pose.from_xyz(0.5, 0.0, 0.0))

        # Warm up so cached_pose / attach is in sync.
        sim.step_once()
        assert sim._pending_pose_ids == set()

        def cb(_sim, _dt):
            # Same pose: no movement → no buffer registration, no child propagation.
            parent.set_pose(Pose.from_xyz(1.0, 2.0, 0.1))
            assert parent.object_id not in _sim._pending_pose_ids
            assert child.object_id not in _sim._pending_pose_ids

        sim.register_callback(cb)
        sim.step_once()

    def test_link_attached_object_uses_current_base_pose_after_flush(self, sim):
        """EE-attached objects must not lag one step behind buffered base motion."""
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/mobile_manipulator.urdf",
                initial_pose=Pose.from_xyz(0.0, 0.0, 0.3),
                motion_mode=MotionMode.DIFFERENTIAL,
                mass=0.0,
                use_fixed_base=False,
            ),
            sim_core=sim,
        )
        ee_link_index = 6
        offset = Pose.from_xyz(0.0, 0.0, 0.06)
        box = _make_box(sim, 0.0, 0.0, 0.1, half=0.05)
        assert agent.attach_object(box, parent_link_index=ee_link_index, relative_pose=offset)

        def cb(_sim, _dt):
            agent.set_pose(Pose.from_xyz(1.0, 0.0, 0.3))

        sim.register_callback(cb)
        sim.step_once()

        link_state = p.getLinkState(agent.body_id, ee_link_index, computeForwardKinematics=1, physicsClientId=sim.client)
        expected_pos, _ = p.multiplyTransforms(
            link_state[0],
            link_state[1],
            offset.position,
            offset.orientation,
        )
        box_pos, _ = p.getBasePositionAndOrientation(box.body_id, physicsClientId=sim.client)
        assert box_pos == pytest.approx(expected_pos, abs=1e-6)


class TestProfilingFields:
    """Task 5: step_once(return_profiling=True) populates the new phase keys."""

    def test_profiling_keys_present(self, sim):
        agent = _make_agent(sim, 0.0, 0.0, 0.1)

        def cb(_sim, _dt):
            agent.set_pose(Pose.from_xyz(1.0, 0.0, 0.1))

        sim.register_callback(cb)
        result = sim.step_once(return_profiling=True)

        # New two-phase keys + legacy per-object-update metric.
        for key in ("phase1_update", "phase2_pose_flush", "phase3_aabb_grid_flush", "agent_update"):
            assert key in result, f"missing profiling key: {key}"
            assert isinstance(result[key], float)
            assert result[key] >= 0.0

    def test_phase2_nonzero_when_buffered_writes(self, sim):
        agent = _make_agent(sim, 0.0, 0.0, 0.1)

        def cb(_sim, _dt):
            # Force a real pose change so the buffer is non-empty.
            agent.set_pose(Pose.from_xyz(5.0, 5.0, 0.1))

        sim.register_callback(cb)
        result = sim.step_once(return_profiling=True)
        # With at least one buffered write, Phase 2 + 3 must execute real work.
        assert result["phase2_pose_flush"] > 0.0
        assert result["phase3_aabb_grid_flush"] > 0.0


class TestSetPosesGetPosesAPI:
    """Task 6.5: public batch set_poses()/get_poses() on MultiRobotSimulationCore.

    D8: same buffered/immediate split as SimObject.set_pose() — routed through
    the _in_step flag — exposed as a single batch call.
    """

    @staticmethod
    def _three_agents(sim):
        a = _make_agent(sim, 0.0, 0.0, 0.1)
        b = _make_agent(sim, 1.0, 0.0, 0.1)
        c = _make_agent(sim, 2.0, 0.0, 0.1)
        return a, b, c

    def test_get_poses_returns_cached_arrays(self, sim):
        a, b, c = self._three_agents(sim)
        positions, orientations = sim.get_poses([a.object_id, b.object_id, c.object_id])
        assert positions.shape == (3, 3)
        assert orientations.shape == (3, 4)
        assert positions[0, 0] == pytest.approx(0.0)
        assert positions[1, 0] == pytest.approx(1.0)
        assert positions[2, 0] == pytest.approx(2.0)

    def test_get_poses_empty(self, sim):
        positions, orientations = sim.get_poses([])
        assert positions.shape == (0, 3)
        assert orientations.shape == (0, 4)

    def test_get_poses_unknown_id_raises(self, sim):
        a, _, _ = self._three_agents(sim)
        with pytest.raises(KeyError):
            sim.get_poses([a.object_id, 99999])

    def test_set_poses_immediate_outside_step(self, sim):
        import numpy as np

        a, b, _ = self._three_agents(sim)
        new_pos = np.array([[5.0, 5.0, 0.1], [6.0, 6.0, 0.1]])
        new_orn = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        sim.set_poses([a.object_id, b.object_id], new_pos, new_orn)

        # PyBullet sees the new poses immediately (outside step).
        pb_pos_a, _ = p.getBasePositionAndOrientation(a.body_id, physicsClientId=sim.client)
        pb_pos_b, _ = p.getBasePositionAndOrientation(b.body_id, physicsClientId=sim.client)
        assert pb_pos_a == pytest.approx((5.0, 5.0, 0.1))
        assert pb_pos_b == pytest.approx((6.0, 6.0, 0.1))

        # And get_poses reflects them.
        positions, _ = sim.get_poses([a.object_id, b.object_id])
        assert positions[0] == pytest.approx([5.0, 5.0, 0.1])
        assert positions[1] == pytest.approx([6.0, 6.0, 0.1])

    def test_set_poses_buffered_inside_step(self, sim):
        import numpy as np

        a, b, _ = self._three_agents(sim)
        new_pos = np.array([[5.0, 5.0, 0.1], [6.0, 6.0, 0.1]])
        new_orn = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

        captured = {}

        def cb(_sim, _dt):
            sim.set_poses([a.object_id, b.object_id], new_pos, new_orn)
            # During the step PyBullet is still at the old pose…
            pb_pos_a, _ = p.getBasePositionAndOrientation(a.body_id, physicsClientId=sim.client)
            captured["pb_pos_a_midstep"] = pb_pos_a
            # …but the cache (get_poses) is already up to date.
            positions, _ = sim.get_poses([a.object_id, b.object_id])
            captured["cached_positions"] = positions.copy()
            # IDs registered for Phase 2 flush.
            captured["pending"] = {a.object_id, b.object_id}.issubset(sim._pending_pose_ids)

        sim.register_callback(cb)
        sim.step_once()

        assert captured["pb_pos_a_midstep"] == pytest.approx((0.0, 0.0, 0.1))
        assert captured["cached_positions"][0] == pytest.approx([5.0, 5.0, 0.1])
        assert captured["pending"] is True

        # After step_once flushes, PyBullet matches.
        pb_pos_a, _ = p.getBasePositionAndOrientation(a.body_id, physicsClientId=sim.client)
        pb_pos_b, _ = p.getBasePositionAndOrientation(b.body_id, physicsClientId=sim.client)
        assert pb_pos_a == pytest.approx((5.0, 5.0, 0.1))
        assert pb_pos_b == pytest.approx((6.0, 6.0, 0.1))

    def test_set_poses_empty_is_noop(self, sim):
        import numpy as np

        sim.set_poses([], np.zeros((0, 3)), np.zeros((0, 4)))  # must not raise

    def test_set_poses_shape_mismatch_raises(self, sim):
        import numpy as np

        a, b, _ = self._three_agents(sim)
        with pytest.raises(ValueError):
            sim.set_poses([a.object_id, b.object_id], np.zeros((3, 3)), np.zeros((2, 4)))
        with pytest.raises(ValueError):
            sim.set_poses([a.object_id], np.zeros((1, 2)), np.zeros((1, 4)))

    def test_set_poses_unknown_id_raises(self, sim):
        import numpy as np

        a, _, _ = self._three_agents(sim)
        with pytest.raises(KeyError):
            sim.set_poses(
                [a.object_id, 99999],
                np.zeros((2, 3)),
                np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
            )


class TestPendingPoseFromPreStep:
    """A pose written from ``pre_step`` is pending until the step flushes it,
    while the clock has already advanced."""

    def test_a_pose_written_in_pre_step_stays_visible_until_flushed(self, sim):
        """A dynamic (mass > 0) body moved from pre_step keeps its pending pose
        after the clock advances: get_pose() must not reread the unflushed
        PyBullet state, and the pending pose is what gets written."""
        box = SimObject.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.1, 0.1, 0.1]),
            pose=Pose.from_xyz(0.0, 0.0, 0.5),
            mass=1.0,
            sim_core=sim,
        )
        assert not box.is_kinematic
        seen = []
        sim.events.on(SimEvents.PRE_STEP, lambda **_: box.set_pose(Pose.from_xyz(1.0, 0.0, 0.5)))
        sim.register_callback(lambda sc, dt: seen.append(box.get_pose().position[0]))
        sim.step_once()
        assert seen == [pytest.approx(1.0)]
        position, _ = p.getBasePositionAndOrientation(box.body_id, physicsClientId=sim.client)
        assert position[0] == pytest.approx(1.0)
