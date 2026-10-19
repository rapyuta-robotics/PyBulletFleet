"""Completed-step recording and playback for declared execution profiles.

Profiles own their construction and state schema. This module owns the durable
record envelope, completed-step capture schedule, files, and selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

import pybullet as p

from pybullet_fleet.core_simulation import MultiRobotSimulationCore
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import SimObject, SimObjectSpawnParams


def _pose_record(pose: Pose) -> dict[str, list[float]]:
    return {"position": list(pose.position), "orientation": list(pose.orientation)}


def _pose_from_record(value: dict) -> Pose:
    return Pose(position=list(value["position"]), orientation=list(value["orientation"]))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or not _finite_tree(value):
        raise ValueError(f"Invalid recording document: {path}")
    return value


def _keys(value: Any, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has missing or unsupported fields")
    return value


def _vector(value: Any, length: int, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{label} must be a finite {length}-vector")
    return value


@dataclass(frozen=True)
class CompletedStepContext:
    """Input supplied to a registered data provider at a completed boundary."""

    sim: MultiRobotSimulationCore
    step: int
    elapsed_time: float


@dataclass(frozen=True)
class DataRecord:
    """Named, versioned user data captured after every completed step."""

    name: str
    version: int
    capture: Callable[[CompletedStepContext], Any]
    required_for_restore: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or type(self.version) is not int
            or self.version < 1
            or type(self.required_for_restore) is not bool
            or not callable(self.capture)
        ):
            raise ValueError(
                "Data record requires a nonempty name, positive version, callable capture, and boolean restore flag"
            )


class RecordingProfile(Protocol):
    """Supported-state contract implemented by each concrete profile."""

    profile_id: str
    version: int
    coverage: dict[str, str]

    def construction(self, sim: MultiRobotSimulationCore) -> dict: ...
    def capture(self, sim: MultiRobotSimulationCore) -> dict: ...
    def validate_checkpoint(self, manifest: dict, state: dict) -> None: ...
    def restore(self, manifest: dict, state: dict, *, gui: bool, target_rtf: float) -> MultiRobotSimulationCore: ...


# Only code installed and explicitly registered in this process may interpret
# artifact IDs. Never import a module path supplied by an artifact.
_PROFILE_FACTORIES: dict[tuple[str, int], Callable[[dict], RecordingProfile]] = {}
_DATA_RESTORERS: dict[tuple[str, int], Callable[[MultiRobotSimulationCore, Any], None]] = {}


def register_recording_profile(profile_id: str, version: int, factory: Callable[[dict], RecordingProfile]) -> None:
    """Register trusted local code for a versioned recording profile."""
    if not profile_id or type(version) is not int or version < 1 or not callable(factory):
        raise ValueError("Invalid recording profile registration")
    key = (profile_id, version)
    if key in _PROFILE_FACTORIES and _PROFILE_FACTORIES[key] is not factory:
        raise ValueError(f"Recording profile already registered: {key}")
    _PROFILE_FACTORIES[key] = factory


def register_data_restorer(name: str, version: int, restore: Callable[[MultiRobotSimulationCore, Any], None]) -> None:
    """Register installed application code for a required DataRecord."""
    if not name or type(version) is not int or version < 1 or not callable(restore):
        raise ValueError("Invalid data restorer registration")
    key = (name, version)
    if key in _DATA_RESTORERS and _DATA_RESTORERS[key] is not restore:
        raise ValueError(f"Data restorer already registered: {key}")
    _DATA_RESTORERS[key] = restore


def _validate_envelope(manifest: dict, state: dict, profile: RecordingProfile) -> None:
    """Validate shared artifact structure before profile-specific state."""
    _keys(
        manifest,
        {
            "profile",
            "version",
            "complete",
            "construction",
            "records",
            "checkpoint_every_steps",
            "last_completed_step",
            "coverage",
            "checkpoint_sha256",
            "frames_sha256",
            "inputs_sha256",
        },
        "manifest",
    )
    if manifest["profile"] != profile.profile_id or manifest["version"] != profile.version or manifest["complete"] is not True:
        raise ValueError("Unsupported or incomplete recording")
    _keys(state, {"sim", "agents", "objects", "records"}, "checkpoint")
    sim_state = _keys(state["sim"], {"type", "version", "step", "elapsed_time"}, "simulation state")
    if (
        not isinstance(sim_state["type"], str)
        or not sim_state["type"]
        or type(sim_state["version"]) is not int
        or sim_state["version"] < 1
    ):
        raise ValueError("Invalid simulation state type or version")
    if not isinstance(manifest["construction"], dict):
        raise ValueError("Invalid construction data")
    dt = manifest["construction"].get("timestep")
    step = sim_state["step"]
    if (
        type(step) is not int
        or type(manifest["last_completed_step"]) is not int
        or not 1 <= step <= manifest["last_completed_step"]
        or isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
        or isinstance(sim_state["elapsed_time"], bool)
        or not isinstance(sim_state["elapsed_time"], (int, float))
        or not math.isclose(sim_state["elapsed_time"], step * dt, rel_tol=0, abs_tol=dt * 1e-9)
    ):
        raise ValueError("Checkpoint step/time mismatch")
    if not isinstance(state["agents"], dict) or not isinstance(state["objects"], dict):
        raise ValueError("Invalid entity state sections")
    for section in ("agents", "objects"):
        for key, entry in state[section].items():
            if not isinstance(key, str) or not key:
                raise ValueError("Invalid entity state key")
            wrapped = _keys(entry, {"type", "version", "state"}, "entity state")
            if (
                not isinstance(wrapped["type"], str)
                or not wrapped["type"]
                or type(wrapped["version"]) is not int
                or wrapped["version"] < 1
            ):
                raise ValueError("Invalid entity state type or version")
    declared = manifest["records"]
    if not isinstance(declared, dict) or not isinstance(state["records"], dict):
        raise ValueError("Checkpoint data records disagree with manifest")
    if set(state["records"]) - set(declared) or any(
        info.get("required_for_restore") and name not in state["records"]
        for name, info in declared.items()
        if isinstance(info, dict)
    ):
        raise ValueError("Checkpoint data records disagree with manifest")
    for name, info in declared.items():
        if not isinstance(name, str) or not name or not isinstance(info, dict):
            raise ValueError("Invalid data record declaration")
        _keys(info, {"version", "required_for_restore"}, "record declaration")
        if type(info["version"]) is not int or info["version"] < 1 or type(info["required_for_restore"]) is not bool:
            raise ValueError("Invalid data record version or requirement")
        if name not in state["records"]:
            continue
        record = _keys(state["records"][name], {"version", "value"}, "data record")
        if record["version"] != info["version"] or not _finite_tree(record["value"]):
            raise ValueError("Invalid data record value or version")
        if info["required_for_restore"] and record["value"] is None:
            raise ValueError("Required data record is empty")
    if not _finite_tree(state):
        raise ValueError("Checkpoint contains non-serializable or non-finite data")


class StateRecorder:
    """PBF-owned durable writer attached to one ordinary simulation run."""

    def __init__(
        self,
        sim: MultiRobotSimulationCore,
        output: str | None,
        profile: RecordingProfile,
        records: tuple[DataRecord, ...] = (),
        checkpoint_every_steps: int = 1,
    ) -> None:
        if type(checkpoint_every_steps) is not int or checkpoint_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be a positive integer")
        if (
            not isinstance(profile.profile_id, str)
            or not profile.profile_id
            or type(profile.version) is not int
            or profile.version < 1
        ):
            raise ValueError("Profile ID must be nonempty and version positive")
        construction = profile.construction(sim)
        self.sim = sim
        self.profile = profile
        if len({record.name for record in records}) != len(records):
            raise ValueError("Data record names must be unique")
        self.records = records
        self.path = (
            Path(output)
            if output
            else Path.cwd() / "recordings" / f"state-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "checkpoints").mkdir()
        self._frames = (self.path / "frames.jsonl").open("w", encoding="utf-8")
        self._inputs = (self.path / "inputs.jsonl").open("w", encoding="utf-8")
        self._pending_inputs: list[dict] = []
        self._closed = False
        self._last_step = 0
        self._cadence = checkpoint_every_steps
        self._manifest = {
            "profile": profile.profile_id,
            "version": profile.version,
            "records": {
                record.name: {"version": record.version, "required_for_restore": record.required_for_restore}
                for record in records
            },
            "complete": False,
            "construction": construction,
            "checkpoint_every_steps": checkpoint_every_steps,
            "last_completed_step": 0,
            "checkpoint_sha256": {},
            "frames_sha256": None,
            "inputs_sha256": None,
            "coverage": profile.coverage,
        }
        _write_json_atomic(self.path / "manifest.json", self._manifest)

    def record_input(self, operation: str, details: dict) -> None:
        """Buffer an accepted supported operation with its simulation phase."""
        if self._closed:
            raise RuntimeError("State recorder is closed")
        if not _finite_tree(details):
            raise ValueError("Unsupported effective input payload")
        phase = "POST_STEP" if self.sim._in_post_step else "PRE_STEP" if self.sim._in_step else "outside_step"
        # The step an input belongs to: the one under way inside a step (whose
        # counter has already advanced, except while PRE_STEP is emitted), and
        # the completed one outside.
        step = self.sim.step_count + (1 if self.sim._in_pre_step else 0)
        self._pending_inputs.append(
            {
                "operation": operation,
                "details": details,
                "step": step,
                "phase": phase,
                "order": len(self._pending_inputs),
            }
        )

    def record_spawn(self, obj: SimObject, params: SimObjectSpawnParams) -> None:
        """Allow the profile to encode a supported entity creation."""
        encode = getattr(self.profile, "spawn_record", None)
        if encode is None:
            raise ValueError("Profile does not support runtime entity creation")
        details = encode(obj, params)
        if details is not None:
            self.record_input("spawn_object", details)

    def _flush_pending_inputs(self) -> None:
        for item in self._pending_inputs:
            self._inputs.write(json.dumps(item, allow_nan=False, sort_keys=True) + "\n")
        self._inputs.flush()
        self._pending_inputs.clear()

    def on_completed_step(self) -> None:
        if self._closed:
            raise RuntimeError("State recorder is closed")
        state = dict(self.profile.capture(self.sim))
        _keys(state, {"sim", "agents", "objects"}, "profile state")
        sim_state = state["sim"]
        step = sim_state["step"]
        state["records"] = {}
        context = CompletedStepContext(self.sim, step, sim_state["elapsed_time"])
        for record in self.records:
            value = record.capture(context)
            if not _finite_tree(value):
                raise ValueError(f"Data record {record.name!r} is not finite JSON data")
            state["records"][record.name] = {"version": record.version, "value": value}
        if step != self._last_step + 1:
            raise RuntimeError("State recorder observed a missing or duplicate completed step")
        complete_manifest = {**self._manifest, "complete": True, "last_completed_step": step}
        _validate_envelope(complete_manifest, state, self.profile)
        self.profile.validate_checkpoint(complete_manifest, state)
        self._flush_pending_inputs()
        # The result frame is a read-only state sample, not an execution checkpoint.
        frame = {
            "step": step,
            "elapsed_time": sim_state["elapsed_time"],
            "agents": state["agents"],
            "objects": state["objects"],
            "records": state["records"],
        }
        self._frames.write(json.dumps(frame, allow_nan=False, sort_keys=True) + "\n")
        self._frames.flush()
        if step % self._cadence == 0:
            checkpoint_path = self.path / "checkpoints" / f"{step:09d}.json"
            _write_json_atomic(checkpoint_path, state)
            self._manifest["checkpoint_sha256"][str(step)] = _sha256(checkpoint_path)
        self._last_step = step
        self._manifest["last_completed_step"] = step

    def close(self) -> None:
        if self._closed:
            return
        self._flush_pending_inputs()
        self._frames.close()
        self._inputs.close()
        self._manifest["frames_sha256"] = _sha256(self.path / "frames.jsonl")
        self._manifest["inputs_sha256"] = _sha256(self.path / "inputs.jsonl")
        self._manifest["complete"] = True
        _write_json_atomic(self.path / "manifest.json", self._manifest)
        self._closed = True
        if self.sim._state_recorder is self:
            self.sim._state_recorder = None

    def abort(self, error: Exception) -> None:
        """Release resources and keep a failed artifact explicitly incomplete."""
        if self._closed:
            return
        self._closed = True
        for stream in (self._frames, self._inputs):
            try:
                stream.close()
            except Exception:
                pass
        self._manifest["complete"] = False
        self._manifest["recording_error"] = f"{type(error).__name__}: {error}"
        try:
            _write_json_atomic(self.path / "manifest.json", self._manifest)
        except Exception:
            pass


def load_recording_manifest(directory: str | Path) -> dict:
    """Read a complete artifact's profile declaration before choosing a loader."""
    manifest = _read_json(Path(directory) / "manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError("Incomplete recording")
    return manifest


def _load_recording_streams(path: Path, manifest: dict, *, retain: bool = False) -> tuple[list[dict], list[dict]]:
    """Verify the complete result/input streams shared by restore and playback."""
    frames_path = path / "frames.jsonl"
    inputs_path = path / "inputs.jsonl"
    try:
        if _sha256(frames_path) != manifest.get("frames_sha256") or _sha256(inputs_path) != manifest.get("inputs_sha256"):
            raise ValueError("Recording stream integrity check failed")
    except OSError as exc:
        raise ValueError("Recording stream is missing or unreadable") from exc
    frames: list[dict] = []
    inputs: list[dict] = []
    try:
        with frames_path.open(encoding="utf-8") as stream:
            frame_count = 0
            for frame_count, line in enumerate(stream, 1):
                frame = json.loads(line)
                if not isinstance(frame, dict) or type(frame.get("step")) is not int or frame["step"] != frame_count:
                    raise ValueError("Recording frame sequence is incomplete")
                if retain:
                    frames.append(frame)
        with inputs_path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Invalid recording input")
                if retain:
                    inputs.append(item)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("Invalid recording stream") from exc
    if type(manifest.get("last_completed_step")) is not int or frame_count != manifest["last_completed_step"]:
        raise ValueError("Recording stream sequence is incomplete")
    return frames, inputs


def load_supported_checkpoint(directory: str | Path, *, at_or_before: float, profile: RecordingProfile) -> tuple[dict, dict]:
    """Load the latest completed checkpoint no later than the requested time."""
    path = Path(directory)
    manifest = _read_json(path / "manifest.json")
    if (
        manifest.get("profile") != profile.profile_id
        or manifest.get("version") != profile.version
        or not manifest.get("complete")
    ):
        raise ValueError("Unsupported or incomplete recording")
    _load_recording_streams(path, manifest)
    construction = manifest["construction"]
    dt = construction["timestep"]
    if not math.isfinite(at_or_before) or at_or_before < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid checkpoint selection time")
    # Decimal-from-string avoids binary 0.3/0.1 rounding down to step 2,
    # without ever choosing a checkpoint strictly after the requested time.
    selected = int((Decimal(str(at_or_before)) / Decimal(str(dt))).to_integral_value(rounding=ROUND_FLOOR))
    selected = min(selected, manifest["last_completed_step"])
    while selected > 0:
        checkpoint_path = path / "checkpoints" / f"{selected:09d}.json"
        if checkpoint_path.is_file():
            if _sha256(checkpoint_path) != manifest["checkpoint_sha256"].get(str(selected)):
                raise ValueError("Checkpoint integrity check failed")
            state = _read_json(checkpoint_path)
            _validate_envelope(manifest, state, profile)
            profile.validate_checkpoint(manifest, state)
            if state["sim"]["step"] != selected:
                raise ValueError("Checkpoint filename and step disagree")
            return manifest, state
        selected -= 1
    raise ValueError("No checkpoint at or before the requested time")


def restore_supported_simulation(
    manifest: dict, state: dict, *, profile: RecordingProfile, gui: bool = False, target_rtf: float = 0.0
) -> MultiRobotSimulationCore:
    """Validate a declared profile and restore its supported world."""
    _validate_envelope(manifest, state, profile)
    profile.validate_checkpoint(manifest, state)
    return profile.restore(manifest, state, gui=gui, target_rtf=target_rtf)


def load_supported_simulation(
    directory: str | Path, *, at_or_before: float, gui: bool = False, target_rtf: float = 0.0
) -> MultiRobotSimulationCore:
    """Construct a declared supported simulation from a complete artifact.

    The first ordinary ``run_simulation()`` on the returned core continues
    from the restored completed-step boundary. Required application data is
    handed to explicitly registered local restorers before this returns.
    """
    manifest = load_recording_manifest(directory)
    # Importing this PBF module registers only PBF-owned built-in profiles.
    from pybullet_fleet.replay import kinematic_profile  # noqa: F401

    key = (manifest.get("profile"), manifest.get("version"))
    factory = _PROFILE_FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"No trusted recording profile registered for {key}")
    profile = factory(manifest)
    manifest, state = load_supported_checkpoint(directory, at_or_before=at_or_before, profile=profile)
    restorers = []
    for name, declaration in manifest["records"].items():
        if declaration["required_for_restore"]:
            restorer = _DATA_RESTORERS.get((name, declaration["version"]))
            if restorer is None:
                raise ValueError(f"No trusted data restorer registered for {name!r}")
            restorers.append((name, restorer))
    sim = restore_supported_simulation(manifest, state, profile=profile, gui=gui, target_rtf=target_rtf)
    try:
        for name, restorer in restorers:
            restorer(sim, state["records"][name]["value"])
        return sim
    except BaseException:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)
        raise


class ResultPlayback:
    """Read recorded results without executing PBF simulation steps."""

    def __init__(self, directory: str | Path, *, profile: RecordingProfile | None = None) -> None:
        path = Path(directory)
        self._path = path
        self.profile = profile
        self.manifest = _read_json(path / "manifest.json")
        wrong_profile = profile is not None and (
            self.manifest.get("profile") != profile.profile_id or self.manifest.get("version") != profile.version
        )
        if not self.manifest.get("complete") or wrong_profile:
            raise ValueError("Unsupported or incomplete playback artifact")
        self.frames, self.inputs = _load_recording_streams(path, self.manifest, retain=True)
        self.index = 0

    @property
    def path(self) -> Path:
        """Artifact directory for profile-specific renderers."""
        return self._path

    @property
    def state(self) -> dict:
        return self.frames[self.index]

    @property
    def events(self) -> list[dict]:
        return [item for item in self.inputs if item["step"] == self.state["step"]]

    def seek_step(self, step: int) -> dict:
        if type(step) is not int or not 1 <= step <= len(self.frames):
            raise ValueError("Playback step is outside the recorded range")
        self.index = step - 1
        return self.state

    def step(self) -> dict:
        return self.seek_step(self.state["step"] + 1)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.frames)

    def __enter__(self) -> "ResultPlayback":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def play_gui(self, *, rtf: float = 1.0, hold: bool = False, camera_config: dict | None = None) -> None:
        """Ask the declared profile to render the recorded frames."""
        if self.profile is None:
            raise ValueError("GUI playback requires the recording profile")
        if not math.isfinite(rtf) or rtf <= 0:
            raise ValueError("Playback RTF must be finite and positive")
        render = getattr(self.profile, "play_gui", None)
        if render is None:
            raise NotImplementedError("This recording profile has no GUI renderer")
        if camera_config is None:
            render(self, rtf=rtf, hold=hold)
        else:
            render(self, rtf=rtf, hold=hold, camera_config=camera_config)
