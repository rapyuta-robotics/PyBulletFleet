"""The collision pipeline reports what it reported on the reference revision.

Each scene of ``tests/collision_parity/scenes.py`` is recorded afresh and
compared with its golden trace under ``tests/collision_parity/golden/``: the
``COLLISION_STARTED`` / ``COLLISION_ENDED`` events of every step, the active
pairs of the collision observation and their ``signed_distance``. A change
that knowingly alters what a scene reports is written down in
``DOCUMENTED_DIFFERENCES``, pair by pair; the test fails on a difference that
is not written down, and on a written-down difference that no longer occurs.
Record the goldens again with ``python -m tests.collision_parity.record``
and say in the commit which revision they come from.
"""

from typing import Dict, Set

import pytest

from tests.collision_parity.scenes import SCENES
from tests.collision_parity.trace import (
    CollisionTrace,
    Difference,
    PairSample,
    StepRecord,
    compare_traces,
    golden_path,
    record_scene,
)

#: Known differences from the golden traces, by scene. Each entry is the
#: ``Difference`` (step, kind, pair) the comparison reports; write next to it
#: why the new behaviour is right.
DOCUMENTED_DIFFERENCES: Dict[str, Set[Difference]] = {}


def _ordered(differences):
    return sorted(differences, key=lambda d: (d.step, d.kind, d.pair))


@pytest.mark.parametrize("name", list(SCENES))
def test_scene_reports_what_the_golden_trace_recorded(name):
    path = golden_path(name)
    assert path.exists(), f"no golden trace for {name}: record it with python -m tests.collision_parity.record {name}"
    golden = CollisionTrace.load(path)
    actual = record_scene(SCENES[name])
    differences = set(compare_traces(golden, actual))
    documented = DOCUMENTED_DIFFERENCES.get(name, set())
    unexpected = _ordered(differences - documented)
    stale = _ordered(documented - differences)
    assert not unexpected, (
        f"{name} differs from the golden trace recorded on {golden.reference} "
        f"({len(unexpected)} undocumented difference(s)):\n  " + "\n  ".join(str(d) for d in unexpected)
    )
    assert not stale, f"{name}: documented differences that no longer occur, remove them:\n  " + "\n  ".join(
        str(d) for d in stale
    )


@pytest.mark.parametrize("name", ["crossing_lanes", "physics_contact"])
def test_recording_a_scene_twice_gives_the_same_trace(name):
    first = record_scene(SCENES[name])
    second = record_scene(SCENES[name])
    assert compare_traces(first, second) == []


@pytest.mark.parametrize("name", list(SCENES))
def test_each_golden_trace_reports_a_collision_that_starts_and_ends(name):
    """A scene whose trace holds no event would pass any implementation."""
    trace = CollisionTrace.load(golden_path(name))
    kinds = {event[0] for step in trace.steps for event in step.events}
    assert kinds == {"started", "ended"}, f"{name} records {kinds or 'no events'}"
    assert trace.scene == name


def _trace(*steps: StepRecord) -> CollisionTrace:
    return CollisionTrace("synthetic", None, list(steps))


def test_compare_reports_each_kind_of_difference():
    golden = _trace(
        StepRecord(1, [("started", "a", "b")], 1, [PairSample(("a", "b"), 0.01, 1)]),
        StepRecord(2, [], 2, [PairSample(("a", "b"), 0.01, 1), PairSample(("c", "d"), -0.02, 2)]),
        StepRecord(3, [], 3, []),
    )
    actual = _trace(
        StepRecord(1, [], 1, [PairSample(("a", "b"), 0.01, 1)]),
        StepRecord(2, [], 1, [PairSample(("a", "b"), 0.03, 2), PairSample(("e", "f"), None, 2)]),
    )
    differences = compare_traces(golden, actual)
    assert [(d.step, d.kind, d.pair) for d in differences] == [
        (-1, "length", ("", "")),
        (1, "events", ("", "")),
        (2, "observation_step", ("", "")),
        (2, "distance", ("a", "b")),
        (2, "sample_step", ("a", "b")),
        (2, "pairs", ("c", "d")),
        (2, "pairs", ("e", "f")),
    ]
    assert Difference(2, "distance", ("a", "b")) in set(differences)  # identity ignores the detail


def test_compare_tolerates_a_rounding_difference_in_the_distance():
    golden = _trace(StepRecord(1, [], 1, [PairSample(("a", "b"), 0.0100000000, 1)]))
    actual = _trace(StepRecord(1, [], 1, [PairSample(("a", "b"), 0.0100000004, 1)]))
    assert compare_traces(golden, actual) == []
    missing = _trace(StepRecord(1, [], 1, [PairSample(("a", "b"), None, 1)]))
    assert [d.kind for d in compare_traces(golden, missing)] == ["distance"]


def test_a_trace_survives_a_round_trip_through_json(tmp_path):
    trace = record_scene(SCENES["collision_modes"], reference="test")
    path = tmp_path / "trace.json"
    trace.save(path)
    loaded = CollisionTrace.load(path)
    assert loaded.reference == "test"
    assert compare_traces(trace, loaded) == []
