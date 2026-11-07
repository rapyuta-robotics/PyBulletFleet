"""The closed form reproduces a TwoPointInterpolation that starts or ends moving.

``trapezoid_distance()`` is what the batched controllers evaluate each step
instead of ``TwoPointInterpolation.get_point()``. It has to express everything
the solver can build for a path leg, or whatever it cannot express is
silently substituted: a start speed ``v0`` -- including one pointing away from
the target, which brakes through zero and reverses --, a start speed above
``vmax`` (the first phase then brakes down to it), and a trajectory running
toward a lower position than it starts from. Path legs end at rest; the scalar
form ``trapezoid_point()``, which joints use, also covers an end speed ``ve``.

Arrival time alone proves nothing here: it comes from ``t_total``, which is
right either way. What differs is where the body is during each phase, so
every case compares positions and speeds against ``get_point()`` on a grid of
times.
"""

import itertools
import warnings

import numpy as np
import pytest

from pybullet_fleet._tpi import (
    build_tpi,
    extract_profile,
    trapezoid_distance,
    trapezoid_point,
    trapezoid_speed,
    try_build_tpi,
)

VMAX = 2.0
V0S = (0.0, 0.5, 1.9, -0.5, -1.9, 2.6, -2.6)  # +-2.6: above vmax
VES = (0.0, 0.4)
DISTANCES = (0.05, 0.3, 1.0, 4.0, 12.0)
LIMITS = ((1.0, 1.0), (1.0, 3.0), (3.0, 1.0), (4.0, 0.5))
STARTS = ((0.0, 1.0), (2.5, -1.0))  # (p0, direction of travel)


def _trajectories():
    """Every solvable (v0, ve, distance, limits, direction) combination."""
    for v0, ve, distance, (accel, decel), (p0, direction) in itertools.product(V0S, VES, DISTANCES, LIMITS, STARTS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # v0 above vmax warns
            tpi = try_build_tpi(
                p0=p0,
                pe=p0 + direction * distance,
                vmax=VMAX,
                accel=accel,
                t0=1.5,
                v0=direction * v0,
                ve=direction * ve,
                decel=decel,
            )
        if tpi is None:
            continue  # inside the braking distance
        if ve > 0.0 and abs(tpi.get_point(tpi.get_end_time() - 1e-12)[1] - tpi.ve) > 1e-6:
            continue  # the solver's own trajectory does not end at ve
        yield tpi


def _grid(tpi):
    return np.concatenate(
        [np.linspace(tpi.t0 - 0.2, tpi.get_end_time() + 0.2, 301), np.array([tpi.t0]), tpi.t0 + np.cumsum(tpi.dt)]
    )


def _cases():
    cases = list(_trajectories())
    assert len(cases) > 300, "the grid should be broad"
    return cases


CASES = _cases()
#: Trajectories ending at rest: what the batched closed form evaluates.
AT_REST = [tpi for tpi in CASES if tpi.ve == 0.0]


def _ids(cases):
    return [
        f"v0={tpi.v0:+.2g},ve={tpi.ve:+.2g},d={tpi.pe - tpi.p0:+.3g},a={tpi.amax_accel:g}/{tpi.amax_decel:g}" for tpi in cases
    ]


def test_the_grid_covers_the_cases_that_matter():
    assert any(tpi.v0 * (tpi.pe - tpi.p0) < 0 for tpi in CASES), "reversal"
    assert any(abs(tpi.v0) > VMAX for tpi in CASES), "start above vmax"
    assert any(tpi.ve != 0.0 for tpi in CASES), "end moving"
    assert any(tpi.pe < tpi.p0 for tpi in CASES), "travel toward lower positions"
    assert any(tpi.case == 0 for tpi in CASES) and any(tpi.case == 1 for tpi in CASES)


@pytest.mark.parametrize("tpi", CASES, ids=_ids(CASES))
def test_the_scalar_closed_form_matches_get_point(tpi):
    profile = extract_profile(tpi)
    for t in _grid(tpi):
        position, velocity = trapezoid_point(profile, float(t))
        expected_position, expected_velocity, _ = tpi.get_point(float(t))
        assert position == pytest.approx(expected_position, abs=1e-9)
        assert velocity == pytest.approx(expected_velocity, abs=1e-9)


@pytest.mark.parametrize("tpi", AT_REST, ids=_ids(AT_REST))
def test_the_batched_closed_form_matches_get_point(tpi):
    profile = extract_profile(tpi)
    t = _grid(tpi)
    tau = t - profile.t0

    def col(value):
        return np.full_like(tau, value)

    args = (col(profile.t_accel), col(profile.t_const), col(profile.t_total), col(profile.accel))
    distance = trapezoid_distance(tau, *args, col(profile.distance), decel=col(profile.decel), v0=col(profile.v0))
    speed = trapezoid_speed(tau, *args, decel=col(profile.decel), v0=col(profile.v0))
    expected = np.array([tpi.get_point(float(x))[:2] for x in t])
    assert profile.p0 + profile.direction * distance == pytest.approx(expected[:, 0], abs=1e-9)
    assert profile.direction * speed == pytest.approx(expected[:, 1], abs=1e-9)


def test_the_batched_and_scalar_forms_agree_over_a_whole_fleet():
    """One call over every trajectory at once: the rows do not interfere."""
    profiles = [extract_profile(tpi) for tpi in AT_REST]
    fields = {name: np.array([getattr(pr, name) for pr in profiles]) for name in profiles[0]._fields}
    for frac in np.linspace(-0.05, 1.05, 23):
        tau = frac * fields["t_total"]
        distance = trapezoid_distance(
            tau,
            fields["t_accel"],
            fields["t_const"],
            fields["t_total"],
            fields["accel"],
            fields["distance"],
            decel=fields["decel"],
            v0=fields["v0"],
        )
        expected = [trapezoid_point(pr, pr.t0 + x)[0] for pr, x in zip(profiles, tau)]
        assert fields["p0"] + fields["direction"] * distance == pytest.approx(expected, abs=1e-12)


def test_a_start_above_vmax_brakes_down_to_it_in_the_first_phase():
    """The solver slows such a start at its decel rate. The phase scalars carry
    that braking as the first phase's acceleration, so the batched controllers
    see the same trajectory instead of one accelerating further."""
    with pytest.warns(UserWarning, match="exceeds vmax"):
        tpi = try_build_tpi(p0=0.0, pe=10.0, vmax=VMAX, accel=1.0, t0=0.0, v0=1.5 * VMAX, decel=3.0)
    profile = extract_profile(tpi)
    assert profile.decel == 3.0
    assert profile.accel == -3.0


class TestFromRestUnchanged:
    """``v0 = 0`` is what every batched trajectory was before: identical."""

    @pytest.mark.parametrize("accel,decel", LIMITS)
    @pytest.mark.parametrize("distance", DISTANCES)
    def test_the_phase_params_are_the_amax_scalars(self, distance, accel, decel):
        tpi = build_tpi(p0=0.0, pe=distance, vmax=VMAX, accel=accel, t0=0.0, decel=decel)
        profile = extract_profile(tpi)
        assert profile.accel == accel and profile.decel == decel
        assert profile.v0 == 0.0 and profile.ve == 0.0

    def test_zero_v0_is_bit_identical_to_omitting_it(self):
        tpi = build_tpi(p0=0.0, pe=3.0, vmax=VMAX, accel=1.0, t0=0.0, decel=2.5)
        profile = extract_profile(tpi)
        t_accel, t_const, t_total, accel, decel = (
            profile.t_accel,
            profile.t_const,
            profile.t_total,
            profile.accel,
            profile.decel,
        )
        tau = np.linspace(-0.1, t_total + 0.1, 997)
        ta, tc, tt, a = (np.full_like(tau, x) for x in (t_accel, t_const, t_total, accel))
        zeros = np.zeros_like(tau)
        dist = np.full_like(tau, 3.0)
        d = np.full_like(tau, decel)
        assert np.array_equal(
            trapezoid_distance(tau, ta, tc, tt, a, dist, decel=d),
            trapezoid_distance(tau, ta, tc, tt, a, dist, decel=d, v0=zeros),
        )
        assert np.array_equal(
            trapezoid_speed(tau, ta, tc, tt, a, decel=d), trapezoid_speed(tau, ta, tc, tt, a, decel=d, v0=zeros)
        )


def test_a_degenerate_trajectory_stays_where_it_starts():
    tpi = build_tpi(p0=0.7, pe=0.7, vmax=VMAX, accel=1.0, t0=2.0)
    profile = extract_profile(tpi)
    for t in (1.0, 2.0, 3.0):
        assert trapezoid_point(profile, t) == (0.7, 0.0)


def test_a_finished_trajectory_is_exactly_at_its_end():
    """``get_point()`` returns ``pe`` itself once the trajectory is over; so
    does the closed form, rather than ``p0`` plus the distance, which can miss
    it in the last bit (2.5 - |0.92 - 2.5| is not 0.92)."""
    starts = [-3.0, -0.1, 0.3, 2.5, 3.0]
    ends = [-2.6, -0.21, 0.92, 1.4, 2.73]
    for p0, pe in itertools.product(starts, ends):
        tpi = build_tpi(p0=p0, pe=pe, vmax=0.5, accel=1.0, t0=0.0)
        profile = extract_profile(tpi)
        for t in (tpi.get_end_time(), tpi.get_end_time() + 1.0):
            assert trapezoid_point(profile, t)[0] == tpi.get_point(t)[0] == pe


def test_batch_and_per_agent_agree_when_a_goal_is_replaced_above_a_lowered_vmax():
    """The batched controller evaluates the closed form, the per-agent one the
    TPI. Lowering vmax while cruising and then replacing the goal starts the
    new trajectory above vmax: both brake down to it at the decel rate."""
    import pybullet as p

    from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
    from pybullet_fleet.agent_manager import AgentManager

    dt = 0.05
    tracks = []
    for batch in (False, True):
        sc = MultiRobotSimulationCore(
            SimulationParams(
                gui=False,
                monitor=False,
                physics=False,
                timestep=dt,
                target_rtf=0,
                collision_check_frequency=0,
                log_level="error",
            )
        )
        sc.initialize_simulation()
        try:
            agent = Agent.from_params(
                AgentSpawnParams(
                    urdf_path="robots/simple_cube.urdf",
                    initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
                    motion_mode=MotionMode.OMNIDIRECTIONAL,
                    controller={"max_linear_vel": 2.0, "max_linear_accel": 4.0, "max_linear_decel": 1.0},
                ),
                sim_core=sc,
            )
            if batch:
                AgentManager(sim_core=sc, fleet_controller={"type": "batch_omni"}).add_object(agent)
            agent.set_goal_pose(Pose.from_xyz(100.0, 0.0, 0.1))
            for _ in range(20):
                sc.step_once()
            agent.controller_params.max_linear_vel = 1.0
            x0 = agent.get_pose().position[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # the solver warns about v0 > vmax
                assert agent.set_goal_pose(Pose.from_xyz(x0 + 20.0, 0.0, 0.1))
                track = []
                for _ in range(40):
                    sc.step_once()
                    track.append(agent.get_pose().position[0] - x0)
            tracks.append(track)
        finally:
            p.disconnect(sc.client)
    assert tracks[1] == pytest.approx(tracks[0], abs=1e-6)
