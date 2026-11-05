"""Record a collision trace over a scripted scene, and compare two traces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pybullet as p

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents

GOLDEN_DIR = Path(__file__).parent / "golden"

#: Two signed distances closer than this are the same measurement.
DISTANCE_TOLERANCE = 1e-6

Pair = Tuple[str, str]
Event = Tuple[str, str, str]  # kind ("started" / "ended"), name a, name b with a < b

SceneScript = Callable[[MultiRobotSimulationCore], Iterator[None]]


@dataclass(frozen=True)
class Scene:
    """A scripted scene: simulation parameters and a generator that drives it.

    The generator receives the simulation, builds the objects, and yields once
    per step: the harness steps the simulation after each ``yield`` and records
    the result. Code between two yields runs between those two steps.
    """

    name: str
    params: Dict[str, Any]
    script: SceneScript
    doc: str = ""

    def make_sim(self) -> MultiRobotSimulationCore:
        params = dict(gui=False, monitor=False, log_level="warning")
        params.update(self.params)
        return MultiRobotSimulationCore(SimulationParams(**params))


@dataclass
class PairSample:
    pair: Pair
    signed_distance: Optional[float]
    sample_step: int

    def to_json(self) -> List[Any]:
        return [list(self.pair), self.signed_distance, self.sample_step]

    @classmethod
    def from_json(cls, data: List[Any]) -> "PairSample":
        return cls((data[0][0], data[0][1]), data[1], data[2])


@dataclass
class StepRecord:
    """What the collision system reported by the end of one step."""

    step: int
    #: Events emitted since the previous record, sorted (their order within a
    #: step is not part of the contract).
    events: List[Event] = field(default_factory=list)
    #: ``CollisionObservation.step``; None before the first completed check.
    observation_step: Optional[int] = None
    #: Active pairs of the observation, sorted by name pair.
    pairs: List[PairSample] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "events": [list(e) for e in self.events],
            "observation_step": self.observation_step,
            "pairs": [s.to_json() for s in self.pairs],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "StepRecord":
        return cls(
            step=data["step"],
            events=[(e[0], e[1], e[2]) for e in data["events"]],
            observation_step=data["observation_step"],
            pairs=[PairSample.from_json(s) for s in data["pairs"]],
        )


@dataclass
class CollisionTrace:
    scene: str
    #: The revision the trace was recorded on, for the reader of a mismatch.
    reference: Optional[str]
    steps: List[StepRecord]

    def to_json(self) -> Dict[str, Any]:
        return {"scene": self.scene, "reference": self.reference, "steps": [s.to_json() for s in self.steps]}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CollisionTrace":
        return cls(data["scene"], data.get("reference"), [StepRecord.from_json(s) for s in data["steps"]])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "CollisionTrace":
        return cls.from_json(json.loads(path.read_text()))


def golden_path(scene_name: str) -> Path:
    return GOLDEN_DIR / f"{scene_name}.json"


def _names(a: Any, b: Any) -> Pair:
    names = sorted((a.name, b.name))
    return (names[0], names[1])


def record_scene(scene: Scene, reference: Optional[str] = None) -> CollisionTrace:
    """Run ``scene`` to its end and return everything the collision system reported.

    Every object of the scene must carry a unique ``name``: the trace refers
    to objects by name so that it does not depend on how ids are assigned.
    """
    sim = scene.make_sim()
    try:
        events: List[Event] = []

        def started(obj_a=None, obj_b=None, **_):
            events.append(("started",) + _names(obj_a, obj_b))

        def ended(obj_a=None, obj_b=None, **_):
            events.append(("ended",) + _names(obj_a, obj_b))

        sim.events.on(SimEvents.COLLISION_STARTED, started)
        sim.events.on(SimEvents.COLLISION_ENDED, ended)

        steps: List[StepRecord] = []
        script = scene.script(sim)
        for _ in script:  # one iteration per yield: the step that follows it
            sim.step_once()
            steps.append(_snapshot(sim, events))
            events.clear()
        return CollisionTrace(scene.name, reference, steps)
    finally:
        p.disconnect(sim.client)


def _snapshot(sim: MultiRobotSimulationCore, events: List[Event]) -> StepRecord:
    by_id = {}
    for obj in sim.sim_objects:
        if obj.name is None:
            raise ValueError(f"object {obj.object_id} has no name; parity scenes name every object")
        if obj.name in by_id.values():
            raise ValueError(f"two objects are named {obj.name!r}; parity scenes name every object uniquely")
        by_id[obj.object_id] = obj.name
    record = StepRecord(step=sim.step_count, events=sorted(events))
    observation = sim.get_collision_observation()
    if observation is not None:
        record.observation_step = observation.step
        samples = []
        for sample in observation.pairs:
            a, b = (by_id[i] for i in sample.object_ids)
            pair = (a, b) if a < b else (b, a)
            samples.append(PairSample(pair, sample.signed_distance, sample.sample_step))
        record.pairs = sorted(samples, key=lambda s: s.pair)
    return record


@dataclass(frozen=True)
class Difference:
    """One way a recorded trace departs from the golden one.

    ``kind`` is one of ``length`` (a different number of steps), ``events``
    (the events of a step differ), ``observation_step``, ``pairs`` (a pair is
    active in one trace only), ``distance`` and ``sample_step`` (the same
    active pair, measured differently). ``pair`` names the pair for the last
    three kinds and is empty otherwise. ``detail`` is for the reader and is
    not part of the identity.
    """

    step: int
    kind: str
    pair: Pair = ("", "")
    detail: str = field(default="", compare=False, hash=False)

    def __str__(self) -> str:
        where = f"step {self.step}: {self.kind}"
        if self.pair != ("", ""):
            where += f" {self.pair[0]}/{self.pair[1]}"
        return f"{where}: {self.detail}" if self.detail else where


def _same_distance(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=DISTANCE_TOLERANCE)


def compare_traces(golden: CollisionTrace, actual: CollisionTrace) -> List[Difference]:
    """Every difference between two traces of the same scene, in step order."""
    differences: List[Difference] = []
    if len(golden.steps) != len(actual.steps):
        differences.append(
            Difference(-1, "length", detail=f"{len(golden.steps)} steps in the golden trace, {len(actual.steps)} recorded")
        )
    for g, a in zip(golden.steps, actual.steps):
        if g.step != a.step:
            differences.append(Difference(g.step, "length", detail=f"golden step {g.step} is recorded step {a.step}"))
            break
        if g.events != a.events:
            differences.append(Difference(g.step, "events", detail=f"golden {g.events}, recorded {a.events}"))
        if g.observation_step != a.observation_step:
            differences.append(
                Difference(
                    g.step,
                    "observation_step",
                    detail=f"golden {g.observation_step}, recorded {a.observation_step}",
                )
            )
        g_pairs = {s.pair: s for s in g.pairs}
        a_pairs = {s.pair: s for s in a.pairs}
        for pair in sorted(set(g_pairs) | set(a_pairs)):
            if pair not in a_pairs:
                differences.append(Difference(g.step, "pairs", pair, "active in the golden trace only"))
            elif pair not in g_pairs:
                differences.append(Difference(g.step, "pairs", pair, "active in the recorded trace only"))
            else:
                gs, as_ = g_pairs[pair], a_pairs[pair]
                if not _same_distance(gs.signed_distance, as_.signed_distance):
                    differences.append(
                        Difference(g.step, "distance", pair, f"golden {gs.signed_distance}, recorded {as_.signed_distance}")
                    )
                if gs.sample_step != as_.sample_step:
                    differences.append(
                        Difference(g.step, "sample_step", pair, f"golden {gs.sample_step}, recorded {as_.sample_step}")
                    )
    return differences
