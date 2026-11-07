"""A ramped joint restored mid-travel takes the same path as the run it came from.

A ramped joint carries execution state -- its signed speed, whether it is
braking, and the trajectory it is following -- and that state only means
something under the motion profile (``max_velocity``, ``max_accel``,
``max_decel``) that produced it. A checkpoint therefore carries both, and a
restore applies the profile before the execution state, so an agent that never
had the profile configured resumes exactly where the original left off.

Each case drives one joint into a state, checkpoints it, and compares every
later step of the uninterrupted run against a restored one: into a freshly
constructed agent (which has no profile of its own), and back into the same
agent after it has moved on.
"""

import pytest

from pybullet_fleet import Agent, AgentSpawnParams

URDF = "robots/elevator.urdf"  # prismatic "lift", <limit velocity="2.0">
JOINT = "lift"
DT = 0.02
PROFILE = {"max_velocity": 0.4, "max_accel": 1.0, "max_decel": 0.5}
AFTER = 250  # steps compared after the checkpoint; every case is at rest by then


@pytest.fixture
def sim_core_params():
    return {"timestep": DT}


def _spawn(sim_core, name):
    return Agent.from_params(AgentSpawnParams(urdf_path=URDF, name=name, use_fixed_base=True), sim_core=sim_core)


def _state(agent):
    joint = agent._joints[agent._joint_index_by_name(JOINT)]
    return joint.position, joint.speed, joint.braking


def _cruise(sim_core, agent, steps):
    """Head for a target far enough to cruise, for ``steps`` steps."""
    agent.set_joint_target(JOINT, 1.0)
    for _ in range(steps):
        sim_core.step_once()


def _moving(sim_core, agent):
    """Accelerating towards a target far enough to cruise."""
    _cruise(sim_core, agent, 10)


def _braking(sim_core, agent):
    """Shedding speed with no trajectory: the new target is inside the stopping distance."""
    _cruise(sim_core, agent, 30)
    position, speed, _ = _state(agent)
    agent.set_joint_target(JOINT, position + 0.25 * speed * speed / (2 * PROFILE["max_decel"]))
    for _ in range(3):
        sim_core.step_once()
    assert _state(agent)[2] is True, "the joint is braking"


def _reversing(sim_core, agent):
    """Turned round: braked to rest away from a target behind it and planned back."""
    _cruise(sim_core, agent, 30)
    agent.set_joint_target(JOINT, -0.5)
    for _ in range(60):
        sim_core.step_once()
    assert _state(agent)[1] < 0.0, "the joint is travelling back"


def _heading_away(sim_core, agent):
    """Still travelling away from a target behind it, braking before it turns."""
    _cruise(sim_core, agent, 30)
    agent.set_joint_target(JOINT, -0.5)
    for _ in range(5):
        sim_core.step_once()
    _, speed, braking = _state(agent)
    assert speed > 0.0 and braking is True, "the joint is still heading away"


SCENARIOS = {"moving": _moving, "braking": _braking, "reversing": _reversing, "heading_away": _heading_away}


def _drive(sim_core, scenario):
    agent = _spawn(sim_core, "original")
    agent.set_joint_motion_profile(JOINT, **PROFILE)
    SCENARIOS[scenario](sim_core, agent)
    assert agent.is_joint_moving(JOINT), "the checkpoint is taken mid-motion"
    return agent


def _path(sim_core, agent, steps=AFTER):
    path = []
    for _ in range(steps):
        sim_core.step_once()
        path.append(_state(agent))
    return path


def _assert_same_path(restored, original):
    assert len(restored) == len(original)
    for step, (got, want) in enumerate(zip(restored, original)):
        assert got[0] == pytest.approx(want[0], abs=1e-9), f"position diverged at step {step}"
        assert got[1] == pytest.approx(want[1], abs=1e-9), f"speed diverged at step {step}"
        assert got[2] == want[2], f"braking diverged at step {step}"
    assert original[-1][1] == 0.0, "the compared window runs to rest"


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
class TestRestoredRampFollowsTheOriginalPath:
    def test_into_a_freshly_constructed_agent(self, sim_core, scenario):
        original = _drive(sim_core, scenario)
        captured = original.capture_kinematic_joint_execution()

        fresh = _spawn(sim_core, "fresh")
        fresh.restore_kinematic_joint_execution(captured)

        index = fresh._joint_index_by_name(JOINT)
        assert (fresh._joints[index].max_velocity, fresh._joints[index].max_accel, fresh._joints[index].max_decel) == (
            PROFILE["max_velocity"],
            PROFILE["max_accel"],
            PROFILE["max_decel"],
        ), "the profile came with the checkpoint"
        restored = []
        expected = []
        for _ in range(AFTER):
            sim_core.step_once()
            expected.append(_state(original))
            restored.append(_state(fresh))
        _assert_same_path(restored, expected)

    def test_back_into_the_same_agent(self, sim_core, scenario):
        agent = _drive(sim_core, scenario)
        captured = agent.capture_kinematic_joint_execution()
        expected = _path(sim_core, agent)

        agent.restore_kinematic_joint_execution(captured)

        _assert_same_path(_path(sim_core, agent), expected)


def _timeline(sim_core, agents, steps=AFTER):
    """Every agent's exact joint state and motion flag, step by step."""
    timelines = [[] for _ in agents]
    for _ in range(steps):
        sim_core.step_once()
        for timeline, agent in zip(timelines, agents):
            timeline.append((*_state(agent), agent.is_joint_moving(JOINT)))
    return timelines


def _finish_step(timeline):
    return next(step for step, entry in enumerate(timeline) if not entry[3])


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
class TestRestoredRampIsStepExact:
    """A restored joint repeats the original's arithmetic, not an approximation of it.

    The joint counts the time it has followed its trajectory and the checkpoint
    carries that count, so no clock reading is subtracted on either side: the
    restored joint is bit-identical every step, and its motion ends on the same
    step -- also after the original has run long enough for a clock to have
    lost digits.
    """

    def test_into_a_fresh_agent_after_a_long_run(self, sim_core, scenario):
        # Idle for a long while first, so an absolute time base would be far
        # from the fresh agent's.
        original = _spawn(sim_core, "original")
        original.set_joint_motion_profile(JOINT, **PROFILE)
        for _ in range(2000):
            sim_core.step_once()
        SCENARIOS[scenario](sim_core, original)
        captured = original.capture_kinematic_joint_execution()
        fresh = _spawn(sim_core, "fresh")
        fresh.restore_kinematic_joint_execution(captured)

        expected, restored = _timeline(sim_core, [original, fresh])
        assert restored == expected
        assert _finish_step(restored) == _finish_step(expected)

    def test_back_into_the_same_agent(self, sim_core, scenario):
        agent = _drive(sim_core, scenario)
        captured = agent.capture_kinematic_joint_execution()
        (expected,) = _timeline(sim_core, [agent])
        agent.restore_kinematic_joint_execution(captured)
        (restored,) = _timeline(sim_core, [agent])
        assert restored == expected
        assert _finish_step(restored) == _finish_step(expected)
