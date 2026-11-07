"""Asymmetric angular accel/decel: ``ControllerParams.max_angular_decel``.

The linear axis gained ``max_linear_decel`` first; rotation stayed a
symmetric trapezoid because every rotation TPI was built without a ``decel``
and the batched controllers discarded the braking scalar. These tests pin the
angular counterpart in the per-agent and batched controllers, and that an
unset ``max_angular_decel`` reproduces the symmetric profile exactly.
"""

import math

import numpy as np
import pybullet as p
import pytest

from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet.types import MotionMode

DT = 0.02
ANG_VEL = 1.0
ANG_ACCEL = 2.0


def _new_core():
    """A second simulation, for a test that restores into a fresh one (``sim_core`` is the first)."""
    from pybullet_fleet import MultiRobotSimulationCore, SimulationParams

    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, timestep=DT, monitor=False, enable_monitor_gui=False)
    )
    sim.initialize_simulation()
    return sim


def _close(sim):
    try:
        p.disconnect(sim.client)
    except p.error:
        pass


@pytest.fixture
def sim_core_params():
    """The shared ``sim_core`` fixture (tests/conftest.py), at this module's timestep."""
    return {"timestep": DT}


def _agent(sim_core, manager, name, mode, decel, x=0.0, y=0.0):
    from pybullet_fleet import Agent, AgentSpawnParams, Pose

    controller = {
        "max_linear_vel": 2.0,
        "max_linear_accel": 2.0,
        "max_angular_vel": ANG_VEL,
        "max_angular_accel": ANG_ACCEL,
        "navigation_2d": True,
    }
    if decel is not None:
        controller["max_angular_decel"] = decel
    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="cube_small.urdf",
            name=name,
            initial_pose=Pose.from_yaw(x, y, 0.1, 0.0),
            motion_mode=mode,
            controller=controller,
        ),
        sim_core=sim_core,
    )
    manager.add_object(agent)
    return agent


def _goal(mode, x=0.0, y=0.0):
    """A goal whose motion is mostly rotation.

    Omni: a short hop then a quarter turn at the goal (final rotation).
    Differential: a turn to face +y, the drive, and a turn back to yaw 0
    (rotation and final rotation).
    """
    from pybullet_fleet import Pose

    if mode == MotionMode.OMNIDIRECTIONAL:
        return Pose.from_yaw(x + 0.5, y, 0.1, math.pi / 2)
    return Pose.from_yaw(x, y + 1.0, 0.1, 0.0)


def _drive(sim_core, agents, max_steps=3000):
    """Step until every agent stops; return each agent's orientation per step."""
    traces = [[] for _ in agents]
    for _ in range(max_steps):
        sim_core.step_once()
        for trace, agent in zip(traces, agents):
            trace.append(np.asarray(agent.get_pose().orientation, dtype=np.float64))
        if not any(agent.is_moving for agent in agents):
            break
    return traces


def _worst_quat_gap(a, b):
    n = min(len(a), len(b))
    # q and -q are the same orientation.
    return max(min(np.max(np.abs(qa - qb)), np.max(np.abs(qa + qb))) for qa, qb in zip(a[:n], b[:n]))


class TestControllerParamsAngularDecel:
    def test_unset_by_default(self):
        assert ControllerParams().max_angular_decel is None

    def test_unset_mirrors_effective_angular_accel(self):
        p_default = ControllerParams()
        assert p_default._eff_angular_decel() == p_default._eff_angular_accel()
        p_set = ControllerParams(max_angular_accel=3.0)
        assert p_set._eff_angular_decel() == 3.0
        assert p_set.scalar_max_angular_decel() == pytest.approx(3.0)

    def test_explicit_value_is_used(self):
        params = ControllerParams(max_angular_accel=3.0, max_angular_decel=0.5)
        assert params.scalar_max_angular_decel() == pytest.approx(0.5)
        assert params.scalar_max_angular_accel() == pytest.approx(3.0)

    def test_per_axis_takes_yaw_component(self):
        params = ControllerParams(max_angular_accel=[1.0, 1.0, 4.0], max_angular_decel=[2.0, 2.0, 8.0])
        assert params.scalar_max_angular_decel() == pytest.approx(8.0)

    def test_from_dict_and_as_config(self):
        params = ControllerParams.from_dict({"max_angular_accel": 2.0, "max_angular_decel": 0.5})
        assert params.max_angular_decel == 0.5
        assert params.as_config()["max_angular_decel"] == 0.5

    def test_agent_accessor(self, sim_core):
        from pybullet_fleet import AgentManager

        manager = AgentManager(sim_core=sim_core)
        soft = _agent(sim_core, manager, "soft", MotionMode.OMNIDIRECTIONAL, 0.5)
        unset = _agent(sim_core, manager, "unset", MotionMode.OMNIDIRECTIONAL, None)
        np.testing.assert_allclose(soft.max_angular_decel, [0.5, 0.5, 0.5])
        np.testing.assert_allclose(unset.max_angular_decel, unset.max_angular_accel)

    def test_replay_validation_accepts_and_rejects(self):
        from pybullet_fleet.replay.kinematic_profile import _validate_controller_parameters

        _validate_controller_parameters(ControllerParams(max_angular_decel=0.5).as_config())
        # ControllerParams refuses -1.0 itself, so write it into the record directly.
        record = ControllerParams().as_config()
        record["max_angular_decel"] = -1.0
        with pytest.raises(ValueError, match="max_angular_decel"):
            _validate_controller_parameters(record)

    @pytest.mark.parametrize(
        "value",
        [0.0, -1.0, math.inf, math.nan, True, [1.0, 1.0, 0.0], [-1.0, 1.0, 2.0], [1.0, 2.0], "fast"],
        ids=["zero", "negative", "inf", "nan", "bool", "zero_yaw", "negative_x", "two_axes", "string"],
    )
    def test_invalid_value_is_refused_when_set(self, value):
        with pytest.raises(ValueError, match="max_angular_decel"):
            ControllerParams(max_angular_decel=value)
        params = ControllerParams(max_angular_decel=0.5)
        with pytest.raises(ValueError, match="max_angular_decel"):
            params.max_angular_decel = value
        assert params.max_angular_decel == 0.5

    @pytest.mark.parametrize("value", [None, 2, 0.5, np.float32(0.5), [0.0, 0.0, 2.0], (1.0, 1.0, 4.0)])
    def test_valid_value_is_accepted(self, value):
        params = ControllerParams.from_dict({"max_angular_decel": value})
        params.max_angular_decel = value
        assert params.max_angular_decel is value


class TestPerAgentRotationBrakesAtDecel:
    def test_rotation_tpi_carries_the_decel(self):
        from pybullet_fleet.controller import OmniController

        controller = OmniController(
            ControllerParams(max_angular_vel=ANG_VEL, max_angular_accel=ANG_ACCEL, max_angular_decel=0.5)
        )
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        quarter = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
        assert controller._init_rotation_tpi(None, identity, quarter, 0.0)
        assert controller._tpi_rotation_angle.amax_accel == pytest.approx(ANG_ACCEL)
        assert controller._tpi_rotation_angle.amax_decel == pytest.approx(0.5)

    @pytest.mark.parametrize("decel", [0.5, 8.0])
    def test_turn_duration_matches_the_closed_form(self, sim_core, decel):
        """A quarter turn in place takes w/a + w/d + cruise, not the symmetric time."""
        from pybullet_fleet import AgentManager, Pose

        manager = AgentManager(sim_core=sim_core)
        agent = _agent(sim_core, manager, "turn", MotionMode.OMNIDIRECTIONAL, decel)
        agent.set_path([Pose.from_yaw(0.0, 0.0, 0.1, math.pi / 2)], auto_approach=False)
        steps = len(_drive(sim_core, [agent])[0])

        theta = math.pi / 2
        w, a, d = ANG_VEL, ANG_ACCEL, decel
        expected = w / a + w / d + (theta - w * w / (2 * a) - w * w / (2 * d)) / w
        assert steps * DT == pytest.approx(expected, abs=3 * DT)


@pytest.mark.parametrize(
    "mode,batch",
    [(MotionMode.OMNIDIRECTIONAL, "batch_omni"), (MotionMode.DIFFERENTIAL, "batch_differential")],
)
class TestBatchMatchesPerAgent:
    @pytest.mark.parametrize("decel", [0.5, 8.0])
    def test_orientation_agrees_step_for_step(self, sim_core, mode, batch, decel):
        """Arrival time alone is no proof: it comes from t_total either way.
        Where the body is during the braking phase is what differs."""
        from pybullet_fleet import AgentManager

        plain = AgentManager(sim_core=sim_core)
        batched = AgentManager(sim_core=sim_core)
        batched.enable_batch(batch)
        single = _agent(sim_core, plain, "single", mode, decel)
        many = _agent(sim_core, batched, "batched", mode, decel)
        for agent in (single, many):
            agent.set_path([_goal(mode)], auto_approach=False)

        a, b = _drive(sim_core, [single, many])
        assert _worst_quat_gap(a, b) < 1e-9
        assert abs(len(a) - len(b)) <= 1

    @pytest.mark.parametrize(
        "limits",
        [
            {"max_angular_accel": [1.0, 1.0, 4.0]},
            {"max_angular_decel": [1.0, 1.0, 4.0]},
            {"max_angular_vel": [0.5, 0.5, 2.0]},
            # A zero x component is valid: only the yaw component drives a turn.
            {"max_angular_vel": [0.0, 0.0, 2.0], "max_angular_accel": [0.0, 0.0, 4.0]},
        ],
        ids=["accel", "decel", "vel", "zero_x"],
    )
    def test_per_axis_limits_use_the_yaw_component(self, sim_core, mode, batch, limits):
        """With per-axis ``[a, a, b]`` angular limits both paths turn at the yaw (``z``) limit."""
        from pybullet_fleet import AgentManager

        plain = AgentManager(sim_core=sim_core)
        batched = AgentManager(sim_core=sim_core)
        batched.enable_batch(batch)
        single = _agent(sim_core, plain, "single", mode, None)
        many = _agent(sim_core, batched, "batched", mode, None)
        for agent in (single, many):
            for key, value in limits.items():
                setattr(agent.controller_params, key, value)
            agent.set_path([_goal(mode)], auto_approach=False)

        a, b = _drive(sim_core, [single, many])
        assert _worst_quat_gap(a, b) < 1e-9
        assert abs(len(a) - len(b)) <= 1

    def test_reported_yaw_rate_agrees_step_for_step(self, sim_core, mode, batch):
        """The batched yaw rate is the derivative of the same profile, braking included."""
        from pybullet_fleet import AgentManager

        plain = AgentManager(sim_core=sim_core)
        batched = AgentManager(sim_core=sim_core)
        batched.enable_batch(batch)
        single = _agent(sim_core, plain, "single", mode, 0.5)
        many = _agent(sim_core, batched, "batched", mode, 0.5)
        for agent in (single, many):
            agent.set_path([_goal(mode)], auto_approach=False)

        worst = 0.0
        braking_seen = False
        for _ in range(3000):
            sim_core.step_once()
            worst = max(worst, abs(single.angular_velocity - many.angular_velocity))
            braking_seen |= 0.0 < abs(single.angular_velocity) < ANG_VEL - 1e-6
            if not single.is_moving and not many.is_moving:
                break
        assert braking_seen
        assert worst < 1e-9

    def test_batch_rows_hold_the_decel(self, sim_core, mode, batch):
        from pybullet_fleet import AgentManager

        manager = AgentManager(sim_core=sim_core)
        controller = manager.enable_batch(batch)
        agent = _agent(sim_core, manager, "a", mode, 0.5)
        agent.set_path([_goal(mode)], auto_approach=False)
        for _ in range(200):
            sim_core.step_once()
            idx = controller._agent_index[id(agent)]
            if controller._rot_t_total[idx] > 0.0:
                break
        assert controller._rot_decel[idx] == pytest.approx(0.5)
        assert controller._rot_accel[idx] == pytest.approx(ANG_ACCEL)


@pytest.mark.parametrize("mode", [MotionMode.OMNIDIRECTIONAL, MotionMode.DIFFERENTIAL])
@pytest.mark.parametrize("batch", [False, True])
def test_unset_decel_reproduces_the_symmetric_profile_exactly(mode, batch):
    """None and an explicit decel equal to the accel give bit-identical motion."""
    from pybullet_fleet import AgentManager

    batch_name = {MotionMode.OMNIDIRECTIONAL: "batch_omni", MotionMode.DIFFERENTIAL: "batch_differential"}[mode]
    traces = []
    for decel in (None, ANG_ACCEL):
        sim = _new_core()
        manager = AgentManager(sim_core=sim)
        if batch:
            manager.enable_batch(batch_name)
        agent = _agent(sim, manager, "a", mode, decel)
        agent.set_path([_goal(mode)], auto_approach=False)
        traces.append(_drive(sim, [agent])[0])
        _close(sim)
    unset, explicit = traces
    assert len(unset) == len(explicit)
    for qa, qb in zip(unset, explicit):
        assert np.array_equal(qa, qb)


class TestRowCompactionCarriesTheAngularBrakingScalar:
    """``unregister_agent()`` moves the last row into the removed one; the
    rotation braking array has to move with the rest of the trajectory."""

    @pytest.mark.parametrize(
        "mode,batch",
        [(MotionMode.OMNIDIRECTIONAL, "batch_omni"), (MotionMode.DIFFERENTIAL, "batch_differential")],
    )
    def test_removing_a_non_last_agent_keeps_each_survivor_scalar(self, sim_core, mode, batch):
        from pybullet_fleet import AgentManager

        manager = AgentManager(sim_core=sim_core)
        controller = manager.enable_batch(batch)
        doomed = _agent(sim_core, manager, "doomed", mode, 0.5)
        survivor = _agent(sim_core, manager, "survivor", mode, 8.0, x=3.0)

        rows = controller._rot_decel
        rows[controller._agent_index[id(doomed)]] = 0.5
        rows[controller._agent_index[id(survivor)]] = 8.0
        manager.remove_object(doomed)

        assert controller._rot_decel[controller._agent_index[id(survivor)]] == pytest.approx(8.0)

    @staticmethod
    def _survivor_trace(remove_the_other):
        """Its own core per run: two runs sharing one would start from
        different clocks with the earlier run's agents still in the scene."""
        from pybullet_fleet import AgentManager

        sim = _new_core()
        manager = AgentManager(sim_core=sim)
        manager.enable_batch("batch_differential")
        mode = MotionMode.DIFFERENTIAL
        doomed = _agent(sim, manager, "doomed", mode, 0.5)
        survivor = _agent(sim, manager, "survivor", mode, 8.0, x=3.0)
        doomed.set_path([_goal(mode)], auto_approach=False)
        survivor.set_path([_goal(mode, x=3.0)], auto_approach=False)
        sim.step_once()
        if remove_the_other:
            manager.remove_object(doomed)
        trace = _drive(sim, [survivor])[0]
        _close(sim)
        return trace

    def test_a_survivor_mid_rotation_turns_the_same_either_way(self):
        undisturbed = self._survivor_trace(remove_the_other=False)
        after_removal = self._survivor_trace(remove_the_other=True)
        assert _worst_quat_gap(undisturbed, after_removal) < 1e-9
