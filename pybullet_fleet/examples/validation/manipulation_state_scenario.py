"""Inspect, record, restore and play back a supported mobile-manipulator run.

The default observation run remains separate from the opt-in omni recording
profile. Scenario stages are application state, not a general PBF task model.
"""

from __future__ import annotations

import argparse
import math
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import pybullet as p

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.commands import RobotAttachCommand, RobotGoalCommand2D, RobotNamedJointPositionsCommand
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.fleet_api import FleetCommandDispatcher, FleetStateProvider
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams
from pybullet_fleet.types import CollisionMode
from pybullet_fleet.replay.kinematic_profile import KinematicProfile
from pybullet_fleet.replay import DataRecord, load_supported_simulation, register_data_restorer

_ROBOT_NAME = "mobile-arm"
_BOX_ENTITY_ID = "box-001"  # Driver-owned identity; runtime object_id is not durable.
_JOINT = "shoulder_to_elbow"
_PICK_TARGET = 0.8
_CARRY_TARGET = -0.6
_BOX_OFFSET = Pose.from_xyz(0, 0, 0.07)
_CAMERA_CONFIG = {
    "camera_mode": "manual",
    "camera_distance": 3.0,
    "camera_yaw": 45,
    "camera_pitch": -35,
    "camera_target": [0.5, 0, 0.7],
}


# Scenario-owned state. The PBF profile records supported simulation state;
# only selected fields here are saved through the DataRecord below.
@dataclass
class _ScenarioRun:
    """External driver state; PBF owns only the restored simulation."""

    stage: str
    observations: dict[str, dict]
    lifecycle: list[dict]
    identity: dict[str, int] = field(default_factory=dict)
    trajectory: list[dict] = field(default_factory=list)
    callback_error: list[Exception] = field(default_factory=list)
    recording_path: str | None = None


# The values contain only driver data, not the simulation or live objects.
_RESTORED_RUNS: weakref.WeakKeyDictionary[MultiRobotSimulationCore, _ScenarioRun] = weakref.WeakKeyDictionary()


def run_scenario(
    *,
    gui: bool = False,
    view_rtf: float = 1.0,
    record_output: str | None = None,
    omni_profile: bool = False,
) -> dict:
    """Run explicit stages using existing PBF APIs and return in-memory facts."""
    if gui and (not math.isfinite(view_rtf) or view_rtf <= 0):
        raise ValueError("view_rtf must be finite and positive in GUI mode")

    dt = 0.1
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=gui,
            monitor=False,
            enable_monitor_gui=False,
            physics=False,
            enable_floor=False,
            timestep=dt,
            target_rtf=view_rtf if gui else 0,
            collision_check_frequency=0,
            log_level="warning",
        )
    )
    controller = (
        {"type": "omni", "navigation_2d": True, "max_linear_vel": 1.0, "max_linear_accel": 2.0}
        if record_output is not None or omni_profile
        else {
            "type": "differential",
            "max_linear_vel": 1.0,
            "max_linear_accel": 2.0,
            "max_angular_vel": 1.5,
            "max_angular_accel": 3.0,
        }
    )
    try:
        robot = Agent.from_params(
            AgentSpawnParams(
                name=_ROBOT_NAME,
                urdf_path=str(Path(__file__).resolve().parents[2] / "robots" / "mobile_manipulator.urdf"),
                initial_pose=Pose.from_xyz(0, 0, 0.3),
                mass=0.0,
                use_fixed_base=False,
                controller=controller,
            ),
            sim,
        )
        run = _configure_scenario(
            sim,
            robot=robot,
            box=None,
            application_state={},
            record_output=record_output,
            record_initial_observation=True,
            omni_profile=omni_profile,
        )
        sim.run_simulation(duration=5.0 if record_output is not None or omni_profile else 3.0)
        result = _result(run)
        return result
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def _configure_scenario(
    sim: MultiRobotSimulationCore,
    *,
    robot: Agent,
    box: SimObject | None,
    application_state: dict,
    record_initial_observation: bool,
    omni_profile: bool,
    record_output: str | None = None,
) -> _ScenarioRun:
    """Install scenario behavior on a fresh or PBF-restored core."""
    if sim.params.gui:
        sim.setup_camera(camera_config=_CAMERA_CONFIG)

    dt = sim.params.timestep
    dispatcher = FleetCommandDispatcher(sim)
    provider = FleetStateProvider(sim)
    run = _ScenarioRun(
        stage=application_state.get("stage", "spawn"),
        observations=dict(application_state.get("observations", {})),
        lifecycle=list(application_state.get("lifecycle", [])),
        identity={_BOX_ENTITY_ID: box.object_id} if box is not None else {},
    )
    callback_phase = "outside_step"

    def lifecycle_record(operation: str, obj: SimObject) -> dict:
        # POST_STEP sees the counter of the step it completes; PRE_STEP and
        # outside a step see the last completed one.
        completed_step = sim.step_count
        return {
            "operation": operation,
            "phase": callback_phase,
            "counter_during_callback": sim.step_count,
            "completed_step": completed_step,
            "sim_time": round(completed_step * dt, 10),
            "object_id": obj.object_id,
        }

    def observe(label: str, step: int) -> None:
        present = box is not None and box in sim.sim_objects
        state = provider.get_states_3d([_ROBOT_NAME])[0]
        robot_pose = robot.get_pose()
        box_pose = box.get_pose() if present and box is not None else None
        run.observations[label] = {
            "step": step,
            "sim_time": round(step * dt, 10),
            "robot_object_id": robot.object_id,
            "base_position": list(robot_pose.position),
            "base_is_moving": state.is_moving,
            "joint_position": robot.get_joint_state_by_name(_JOINT)[0],
            "joint_reported_velocity": robot.get_joint_state_by_name(_JOINT)[1],
            "box_present": present,
            "box_runtime_object_id": box.object_id if present and box is not None else None,
            "box_position": list(box_pose.position) if box_pose is not None else None,
            "box_attached": box.is_attached() if present and box is not None else False,
            "attached_children": [child.object_id for child in robot.get_attached_objects()],
        }

    def on_spawn(*, obj: SimObject) -> None:
        if obj.name == _BOX_ENTITY_ID:
            run.lifecycle.append(lifecycle_record("spawn", obj))

    def on_remove(*, obj: SimObject) -> None:
        if obj.name == _BOX_ENTITY_ID:
            run.lifecycle.append(lifecycle_record("delete", obj))

    def require_ack(ack, action: str) -> None:
        if _ROBOT_NAME not in ack.accepted_names:
            raise RuntimeError(f"{action} rejected: {dict(ack.rejected)}")

    def before_step(**_: object) -> None:
        nonlocal box, callback_phase
        if run.callback_error or sim.step_count != 0:
            return
        try:
            callback_phase = "PRE_STEP"
            shape = ShapeParams(shape_type="box", half_extents=[0.05, 0.05, 0.05])
            box = SimObject.from_params(
                SimObjectSpawnParams(
                    name=_BOX_ENTITY_ID,
                    initial_pose=Pose.from_xyz(1.0, 0, 0.7),
                    mass=0.0,
                    pickable=True,
                    collision_mode=CollisionMode.DISABLED,
                    visual_shape=shape,
                    collision_shape=shape,
                ),
                sim,
            )
            run.identity[_BOX_ENTITY_ID] = box.object_id
            run.stage = "base_motion"
        except Exception as exc:
            run.callback_error.append(exc)

    def after_step(**_: object) -> None:
        nonlocal box, callback_phase
        if run.callback_error:
            return
        try:
            callback_phase = "POST_STEP"
            # POST_STEP sees the new physical state and the counter of the step
            # that produced it.
            step = sim.step_count
            if run.stage == "base_motion":
                if step == 1:
                    require_ack(
                        dispatcher.navigate(
                            [
                                RobotGoalCommand2D(
                                    name=_ROBOT_NAME,
                                    position=(0.6, 0),
                                    yaw=0.4 if record_output is not None or omni_profile else 0.0,
                                    z=0.3,
                                )
                            ],
                            command_id="base-to-box",
                        ),
                        "base navigate",
                    )
                elif step == 3:
                    observe("CP1", step)
                elif step > 3 and not robot.is_moving:
                    require_ack(
                        dispatcher.joint_command(
                            [RobotNamedJointPositionsCommand(_ROBOT_NAME, {_JOINT: _PICK_TARGET})],
                            command_id="arm-to-pick",
                        ),
                        "arm target",
                    )
                    run.stage = "arm_motion"
            elif run.stage == "arm_motion":
                if "CP2" not in run.observations:
                    observe("CP2", step)
                elif robot.are_joints_at_targets_by_name({_JOINT: _PICK_TARGET}, tolerance=0.01):
                    require_ack(
                        dispatcher.attach(
                            [
                                RobotAttachCommand(
                                    _ROBOT_NAME,
                                    attach=True,
                                    object_name=_BOX_ENTITY_ID,
                                    parent_link="end_effector",
                                    offset=_BOX_OFFSET,
                                )
                            ],
                            command_id="attach-box",
                        ),
                        "attach",
                    )
                    require_ack(
                        dispatcher.joint_command(
                            [RobotNamedJointPositionsCommand(_ROBOT_NAME, {_JOINT: _CARRY_TARGET})],
                            command_id="arm-carry",
                        ),
                        "carry target",
                    )
                    run.stage = "attached_motion"
            elif run.stage == "attached_motion":
                if "CP3" not in run.observations:
                    observe("CP3", step)
                elif robot.are_joints_at_targets_by_name({_JOINT: _CARRY_TARGET}, tolerance=0.01):
                    require_ack(
                        dispatcher.attach(
                            [RobotAttachCommand(_ROBOT_NAME, attach=False, object_name=_BOX_ENTITY_ID)],
                            command_id="detach-box",
                        ),
                        "detach",
                    )
                    observe("CP4", step)
                    run.stage = "delete"
            elif run.stage == "delete":
                if box is None:
                    raise RuntimeError("box missing before deletion")
                sim.remove_object(box)
                if box in sim.sim_objects or box.is_attached() or box in robot.get_attached_objects():
                    raise RuntimeError("removed box remains live or attached")
                box = None
                observe("after_delete", step)
                run.stage = "done"
            current_box = box if box is not None and box in sim.sim_objects else None
            run.trajectory.append(
                {
                    "step": step,
                    "sim_time": round(step * dt, 10),
                    "base_position": list(robot.get_pose().position),
                    "base_orientation": list(robot.get_pose().orientation),
                    "base_is_moving": robot.is_moving,
                    "joint_position": robot.get_joint_state_by_name(_JOINT)[0],
                    "box_position": list(current_box.get_pose().position) if current_box is not None else None,
                    "box_attached": current_box.is_attached() if current_box is not None else False,
                    "stage": run.stage,
                }
            )
        except Exception as exc:
            run.callback_error.append(exc)

    sim.events.on(SimEvents.OBJECT_SPAWNED, on_spawn)
    sim.events.on(SimEvents.OBJECT_REMOVED, on_remove)
    sim.events.on(SimEvents.PRE_STEP, before_step)
    sim.events.on(SimEvents.POST_STEP, after_step)
    if record_initial_observation:
        observe("before_spawn", 0)
    if record_output is not None:
        # Recording extension: persist the driver's stage and observations;
        # KinematicProfile separately captures PBF-owned execution state.
        def capture_application_state(_context) -> dict:
            # EventBus logs handler exceptions. A durable recording must fail
            # visibly instead of marking that incomplete execution as complete.
            if run.callback_error:
                raise RuntimeError("Scenario callback failed before checkpoint") from run.callback_error[0]
            return {"stage": run.stage, "observations": run.observations, "lifecycle": run.lifecycle}

        recorder = sim.configure_state_recording(
            output=record_output,
            profile=KinematicProfile(
                allowed_event_handlers=(on_spawn, on_remove, before_step, after_step),
            ),
            records=(DataRecord("scenario", 1, capture_application_state, required_for_restore=True),),
        )
        run.recording_path = str(recorder.path)
    return run


def _result(run: _ScenarioRun) -> dict:
    if run.callback_error:
        raise RuntimeError("scenario callback failed") from run.callback_error[0]
    if run.stage != "done":
        raise RuntimeError(f"scenario did not finish: {run.stage}")
    return {
        "observations": run.observations,
        "lifecycle": run.lifecycle,
        "identity": run.identity,
        "trajectory": run.trajectory,
        "final_stage": run.stage,
        "recording_path": run.recording_path,
    }


def _restore_scenario_state(sim: MultiRobotSimulationCore, value: dict) -> None:
    """Reinstall scenario behavior after PBF restores its owned state."""
    if (
        not isinstance(value, dict)
        or set(value) != {"stage", "observations", "lifecycle"}
        or not isinstance(value["stage"], str)
        or value["stage"] not in {"spawn", "base_motion", "arm_motion", "attached_motion", "delete", "done"}
        or not isinstance(value["observations"], dict)
        or not isinstance(value["lifecycle"], list)
    ):
        raise ValueError("Invalid manipulation scenario state")
    robot = sim.get_unique_agent_by_name(_ROBOT_NAME)
    boxes = sim.find_objects_by_name(_BOX_ENTITY_ID)
    if len(boxes) > 1:
        raise LookupError(f"Expected at most one object named {_BOX_ENTITY_ID!r}; found {len(boxes)}")
    _RESTORED_RUNS[sim] = _configure_scenario(
        sim,
        robot=robot,
        box=boxes[0] if boxes else None,
        application_state=value,
        record_initial_observation=False,
        omni_profile=True,
    )


def resume_scenario(directory: str, *, at_or_before: float, gui: bool = False, view_rtf: float = 1.0) -> dict:
    """Load the supported world and continue the registered external driver."""
    # Register the application restorer before loading its required DataRecord.
    register_data_restorer("scenario", 1, _restore_scenario_state)
    sim = load_supported_simulation(directory, at_or_before=at_or_before, gui=gui, target_rtf=view_rtf if gui else 0.0)
    try:
        run = _RESTORED_RUNS.pop(sim)
        sim.run_simulation(duration=5.0)
        result = _result(run)
        return result
    finally:
        if p.isConnected(sim.client):
            p.disconnect(sim.client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI")
    parser.add_argument("--rtf", type=float, default=1.0, help="GUI viewing real-time factor")
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument(
        "--record", nargs="?", const="", metavar="DIR", help="Record a supported run (omit DIR for an automatic name)"
    )
    operations.add_argument("--restore", metavar="DIR", help="Resume a supported checkpoint in this recording")
    operations.add_argument("--playback", metavar="DIR", help="Read recorded results without executing the simulation")
    parser.add_argument("--time", type=float, help="Select the latest checkpoint at or before this simulated time")
    args = parser.parse_args()
    if args.playback is not None:
        from pybullet_fleet.replay.state_recording import ResultPlayback, load_recording_manifest

        profile = KinematicProfile.from_manifest(load_recording_manifest(args.playback))
        playback = ResultPlayback(args.playback, profile=profile)
        if args.gui:
            playback.play_gui(rtf=args.rtf, camera_config=_CAMERA_CONFIG)
        else:
            for frame in playback:
                print(
                    f"step={frame['step']} time={frame['elapsed_time']:.1f}s "
                    f"box_present={_BOX_ENTITY_ID in frame['objects']}"
                )
        return
    if args.restore is not None:
        if args.time is None:
            raise ValueError("--restore requires --time in this supported profile")
        result = resume_scenario(args.restore, at_or_before=args.time, gui=args.gui, view_rtf=args.rtf)
    else:
        output = args.record
        if output is None:
            result = run_scenario(gui=args.gui, view_rtf=args.rtf)
        else:
            result = run_scenario(gui=args.gui, view_rtf=args.rtf, record_output=output)
            print(f"State recording: {result['recording_path']}")
    for label, observation in result["observations"].items():
        print(
            f"{label}: step={observation['step']} time={observation['sim_time']:.1f}s "
            f"box_present={observation['box_present']} attached={observation['box_attached']}"
        )


if __name__ == "__main__":
    main()
