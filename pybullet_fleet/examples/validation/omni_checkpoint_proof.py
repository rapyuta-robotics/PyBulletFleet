"""Fresh-process proof for one physics-off omni navigation checkpoint.

This is an external coordinator for a deliberately fixed profile, not a
general checkpoint or replay framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import pybullet as p

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.controller import OmniController
from pybullet_fleet.controller_params import ControllerParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.geometry import Pose

# Fixed inputs and assertions for this one verification scenario. These are
# not limits of a future checkpoint API: an ordinary recorder must read the
# effective runtime configuration and received commands from PBF's boundary.
PROFILE = "pbf.single_omni_straight_checkpoint"
VERSION = 3
DT = 0.1
CHECKPOINT_STEP = 3
END_TIME = 1.3
ROBOT_NAME = "checkpoint-robot"
INITIAL = Pose.from_xyz(0.0, 0.0, 0.1)
GOAL = Pose.from_xyz(0.4, 0.0, 0.1)
CONTROLLER = {"type": "omni", "navigation_2d": True, "max_linear_vel": 0.8, "max_linear_accel": 1.0}
ASSET = Path(__file__).resolve().parents[2] / "robots" / "simple_cube.urdf"


def _conditions() -> dict:
    # The bundled asset hash is this proof's compatibility policy, not a
    # requirement to hash every user-provided URDF in a general artifact.
    return {
        "asset": "package:pybullet_fleet/robots/simple_cube.urdf",
        "asset_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "robot": ROBOT_NAME,
        "initial_position": list(INITIAL.position),
        "goal_position": list(GOAL.position),
        "controller": dict(CONTROLLER),
        "physics": False,
        "timestep": DT,
    }


def _make_sim(*, gui: bool = False, view_rtf: float = 1.0) -> tuple[MultiRobotSimulationCore, Agent]:
    if gui and (not math.isfinite(view_rtf) or view_rtf <= 0):
        raise ValueError("GUI viewing RTF must be finite and positive")
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=gui,
            monitor=False,
            enable_monitor_gui=False,
            physics=False,
            enable_floor=False,
            timestep=DT,
            target_rtf=view_rtf if gui else 0,
            collision_check_frequency=0,
            log_level="warning",
        )
    )
    try:
        robot = Agent.from_params(
            AgentSpawnParams(
                name=ROBOT_NAME,
                urdf_path=str(ASSET),
                initial_pose=INITIAL,
                mass=0.0,
                controller=OmniController(ControllerParams.from_dict(CONTROLLER)),
            ),
            sim,
        )
        sim.initialize_simulation()
        return sim, robot
    except BaseException:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)
        raise


def _hold_final_gui(sim: MultiRobotSimulationCore) -> None:
    print("Simulation finished. Close the GUI window or press Ctrl+C to exit.")
    try:
        while p.isConnected(sim.client):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def _sample(sim: MultiRobotSimulationCore, robot: Agent, *, completed_step: int | None = None) -> dict:
    step = sim.step_count if completed_step is None else completed_step
    pose = robot.get_pose()
    return {
        "step": step,
        "elapsed_time": round(step * DT, 10),
        "position": list(pose.position),
        "orientation": list(pose.orientation),
        "velocity": robot.get_velocity().tolist(),
        "angular_velocity": float(robot.angular_velocity),
        "moving": robot.is_moving,
    }


def _controller(robot: Agent) -> OmniController:
    controller = robot.controller
    if type(controller) is not OmniController or len(robot.controllers) != 1 or not robot.is_action_queue_empty():
        raise ValueError("Checkpoint profile requires one built-in per-agent omni controller and no Actions")
    return controller


def _issue_goal(robot: Agent) -> None:
    # This profile deliberately excludes auto-approach and final alignment.
    # The driver chooses GOAL; the controller retains the effective active goal.
    # A future input journal must also record when the accepted command arrived.
    robot.set_path([GOAL], auto_approach=False, final_orientation_align=False)
    controller = _controller(robot)
    if len(controller.path) != 1:
        raise RuntimeError("Checkpoint proof requires exactly one effective waypoint")


def _capture(sim: MultiRobotSimulationCore, robot: Agent) -> dict:
    step, elapsed_time = sim.get_completed_step_boundary()
    # Proof assertion: compare an in-motion checkpoint at exactly S_3. A
    # general capture operation must not require this step or moving=True.
    if step != CHECKPOINT_STEP or not robot.is_moving:
        raise ValueError("Capture must occur at the supported in-motion completed step")
    controller = _controller(robot).capture_navigation_state()
    if not 0 < elapsed_time < END_TIME:
        raise ValueError("Checkpoint time is outside the supported run")
    return {
        "profile": PROFILE,
        "version": VERSION,
        "conditions": _conditions(),
        "state": {**_sample(sim, robot), "controller": controller},
    }


def _number(value: object, label: str, *, positive: bool = False) -> float:
    """Local JSON scalar validation; a future artifact layer may reuse this rule."""
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _vector(value: object, length: int, label: str) -> list[float]:
    """Local JSON shape validation, independent of the omni trajectory."""
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must have {length} elements")
    return [_number(item, label) for item in value]


def _keys(value: object, expected: set[str], label: str) -> dict:
    """Reject absent/unknown JSON fields before applying any simulation state."""
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has missing or unsupported fields")
    return value


def validate_checkpoint(checkpoint: object) -> dict:
    """Validate JSON structure and this proof's fixed profile before core mutation.

    Structural checks may move to shared artifact tooling. S_3, the fixed
    goal/origin/limits and the exact trajectory are scenario assertions, not
    generic requirements for a future checkpoint loader.
    """
    # General artifact concern: identify a supported schema/profile and reject
    # incomplete or malformed fields before changing the fresh simulation.
    record = _keys(checkpoint, {"profile", "version", "conditions", "state"}, "checkpoint")
    if record["profile"] != PROFILE or type(record["version"]) is not int or record["version"] != VERSION:
        raise ValueError("Unsupported checkpoint profile or version")
    if record["conditions"] != _conditions():
        raise ValueError("Checkpoint construction or execution conditions differ")
    # The following exact step, target and motion checks narrow this record to
    # the one deterministic example rather than defining a universal schema.
    state = _keys(
        record["state"],
        {"step", "elapsed_time", "position", "orientation", "velocity", "angular_velocity", "moving", "controller"},
        "state",
    )
    if type(state["step"]) is not int or state["step"] != CHECKPOINT_STEP:
        raise ValueError("Unsupported checkpoint step")
    elapsed = _number(state["elapsed_time"], "elapsed_time")
    if not math.isclose(elapsed, CHECKPOINT_STEP * DT, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Checkpoint clock is inconsistent")
    position = _vector(state["position"], 3, "position")
    orientation = _vector(state["orientation"], 4, "orientation")
    velocity = _vector(state["velocity"], 3, "velocity")
    angular_velocity = _number(state["angular_velocity"], "angular_velocity")
    if state["moving"] is not True or not math.isclose(sum(q * q for q in orientation), 1, abs_tol=1e-9):
        raise ValueError("Checkpoint must describe a moving robot with a unit quaternion")
    if any(abs(a - b) > 1e-9 for a, b in zip(orientation, INITIAL.orientation)) or abs(angular_velocity) > 1e-9:
        raise ValueError("Checkpoint is outside the straight, fixed-orientation profile")
    controller = _keys(
        state["controller"],
        {
            "path",
            "waypoint_index",
            "goal",
            "align_final_orientation",
            "final_target_orientation",
            "final_alignment_in_progress",
            "phase",
            "forward",
            "rotation",
            "replacement",
        },
        "controller",
    )
    # The profile drives one path: nothing is held for goal_replacement.
    if _keys(controller["replacement"], {"pending", "braking"}, "replacement") != {"pending": None, "braking": False}:
        raise ValueError("Checkpoint holds a replacement path outside the supported profile")
    goal_pose = _keys(controller["goal"], {"position", "orientation"}, "goal")
    goal = _vector(goal_pose["position"], 3, "goal_position")
    goal_orientation = _vector(goal_pose["orientation"], 4, "goal_orientation")
    if controller["path"] != [goal_pose] or controller["waypoint_index"] != 0:
        raise ValueError("Checkpoint path differs from the supported straight profile")
    forward = _keys(
        controller["forward"],
        {"origin", "t0", "vmax", "accel", "decel", "v0", "orientation_after_rotation"},
        "forward trajectory",
    )
    origin = _vector(forward["origin"], 3, "origin")
    t0 = _number(forward["t0"], "t0")
    vmax = _number(forward["vmax"], "vmax", positive=True)
    accel = _number(forward["accel"], "accel", positive=True)
    decel = _number(forward["decel"], "decel", positive=True)
    if (
        goal != list(GOAL.position)
        or goal_orientation != list(GOAL.orientation)
        or origin != list(INITIAL.position)
        or not math.isclose(t0, 0.0, abs_tol=1e-10)
        or not math.isclose(vmax, CONTROLLER["max_linear_vel"], abs_tol=1e-10)
        or not math.isclose(accel, CONTROLLER["max_linear_accel"], abs_tol=1e-10)
        or not math.isclose(decel, accel, abs_tol=1e-10)
        or controller["phase"] != "forward"
        or controller["rotation"] is not None
        or controller["align_final_orientation"] is not False
        or controller["final_target_orientation"] is not None
        or controller["final_alignment_in_progress"] is not False
        or forward["orientation_after_rotation"] is not None
        or forward["v0"] != 0.0
    ):
        raise ValueError("Checkpoint trajectory differs from the supported profile")
    # A completed step's pose is its trajectory at the step's end, k*dt: the
    # checkpoint's elapsed time (sim_time outside a step).
    from pybullet_fleet._tpi import build_tpi

    tpi = build_tpi(0, math.dist(origin, goal), vmax, accel, t0, decel=decel)
    traveled, expected_speed, _ = tpi.get_point(CHECKPOINT_STEP * DT)
    direction = [(g - o) / math.dist(origin, goal) for o, g in zip(origin, goal)]
    expected_position = [o + d * traveled for o, d in zip(origin, direction)]
    expected_velocity = [d * expected_speed for d in direction]
    if any(abs(a - b) > 1e-8 for a, b in zip(position, expected_position)) or any(
        abs(a - b) > 1e-8 for a, b in zip(velocity, expected_velocity)
    ):
        raise ValueError("Checkpoint pose or velocity does not match its original trajectory")
    return record


def write_checkpoint(path: Path, checkpoint: dict) -> None:
    """Validate and atomically publish this profile's JSON artifact."""
    validate_checkpoint(checkpoint)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(checkpoint, allow_nan=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path: Path) -> dict:
    """Parse JSON strictly, then apply profile-specific validation.

    Duplicate-key rejection and JSON parsing may become shared artifact tools;
    this function does not imply that other profiles use this exact schema.
    """

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate checkpoint field {key}")
            result[key] = value
        return result

    record = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=lambda value: _number(value, value))
    return validate_checkpoint(record)


def restore_checkpoint(sim: MultiRobotSimulationCore, robot: Agent, checkpoint: dict) -> None:
    """Apply a validated profile record to a fresh instance without reissuing its goal."""
    record = validate_checkpoint(checkpoint)
    if len(sim.agents) != 1 or sim.agents[0] is not robot or robot.name != ROBOT_NAME:
        raise ValueError("Restore requires the fixed one-robot world")
    if sim.get_completed_step_boundary() != (0, 0.0):
        raise ValueError("Restore requires a fresh initialized core")
    controller = _controller(robot)
    state = record["state"]
    pose = Pose(position=list(state["position"]), orientation=list(state["orientation"]))
    robot.restore_motion_state(pose, state["velocity"], state["angular_velocity"], moving=True)
    controller.restore_navigation_state(state["controller"], pose.orientation)
    sim.restore_completed_step_boundary(state["step"], state["elapsed_time"])


def run_reference(*, gui: bool = False, view_rtf: float = 1.0, hold_gui: bool = False) -> list[dict]:
    sim, robot = _make_sim(gui=gui, view_rtf=view_rtf)
    try:
        _issue_goal(robot)
        samples = [_sample(sim, robot)]

        def after_step(**_: object) -> None:
            # POST_STEP observes the new pose and the counter of the step it completes.
            completed_step = sim.step_count
            samples.append(_sample(sim, robot, completed_step=completed_step))
            if hold_gui and gui and completed_step * DT >= END_TIME:
                _hold_final_gui(sim)

        sim.events.on(SimEvents.POST_STEP, after_step)
        sim.run_simulation(duration=END_TIME)
        return samples
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def run_source(path: Path, *, gui: bool = False, view_rtf: float = 1.0, hold_gui: bool = False) -> dict:
    sim, robot = _make_sim(gui=gui, view_rtf=view_rtf)
    try:
        _issue_goal(robot)
        for _ in range(CHECKPOINT_STEP):
            sim.step_once()
            if gui:
                time.sleep(DT / view_rtf)
        checkpoint = _capture(sim, robot)
        write_checkpoint(path, checkpoint)
        if hold_gui and gui:
            _hold_final_gui(sim)
        return checkpoint
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def run_restored(path: Path, *, gui: bool = False, view_rtf: float = 1.0, hold_gui: bool = False) -> list[dict]:
    checkpoint = read_checkpoint(path)
    sim, robot = _make_sim(gui=gui, view_rtf=view_rtf)
    try:
        restore_checkpoint(sim, robot, checkpoint)
        samples = [_sample(sim, robot)]

        def after_step(**_: object) -> None:
            completed_step = sim.step_count
            samples.append(_sample(sim, robot, completed_step=completed_step))
            if hold_gui and gui and completed_step * DT >= END_TIME:
                _hold_final_gui(sim)

        sim.events.on(SimEvents.POST_STEP, after_step)
        sim.run_simulation(duration=END_TIME, resume=True)
        return samples
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "source", "restore"))
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint file; source can create one automatically")
    parser.add_argument("--output", type=Path, help="JSON results file; omitted creates a unique temporary directory")
    parser.add_argument("--gui", action="store_true", help="Show this mode in the PyBullet GUI")
    parser.add_argument("--rtf", type=float, default=1.0, help="GUI viewing real-time factor (default: 1)")
    args = parser.parse_args()
    if args.mode == "restore" and args.checkpoint is None:
        parser.error("--checkpoint is required for restore")
    if args.gui and (not math.isfinite(args.rtf) or args.rtf <= 0):
        parser.error("--rtf must be finite and positive with --gui")
    generated_dir = (
        Path(tempfile.mkdtemp(prefix=f"pbf-omni-{args.mode}-"))
        if args.output is None or (args.mode == "source" and args.checkpoint is None)
        else None
    )
    if args.output is None:
        if generated_dir is None:
            raise RuntimeError("Missing generated result directory")
        output = generated_dir / "result.json"
    else:
        output = args.output
    if args.mode == "source" and args.checkpoint is None:
        if generated_dir is None:
            raise RuntimeError("Missing generated checkpoint directory")
        checkpoint_path = generated_dir / "checkpoint.json"
    else:
        checkpoint_path = args.checkpoint
    if args.mode == "reference":
        result = run_reference(gui=args.gui, view_rtf=args.rtf, hold_gui=args.gui)
    elif args.mode == "source":
        if checkpoint_path is None:
            raise RuntimeError("Missing source checkpoint path")
        result = run_source(checkpoint_path, gui=args.gui, view_rtf=args.rtf, hold_gui=args.gui)
    else:
        if checkpoint_path is None:
            raise RuntimeError("Missing restore checkpoint path")
        result = run_restored(checkpoint_path, gui=args.gui, view_rtf=args.rtf, hold_gui=args.gui)
    output.write_text(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
    print(f"Saved result to {output}")
    if args.mode == "source":
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
