#!/usr/bin/env python3
"""Validate supported recording and fresh-process restore with a fixed arm.

This adapts the low-level pick/drop callback for checkpoint tests. The ordinary
``examples/arm/pick_drop_arm_demo.py`` remains the simpler teaching example.
"""

from __future__ import annotations

import argparse
import os
import sys
import pybullet as p

try:
    import pybullet_fleet  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    import pybullet_fleet  # noqa: F401

from pybullet_fleet.agent import Agent
from pybullet_fleet.config_utils import load_yaml_config, merge_configs
from pybullet_fleet.core_simulation import MultiRobotSimulationCore
from pybullet_fleet.geometry import Pose
from pybullet_fleet.replay import DataRecord, load_supported_simulation, register_data_restorer
from pybullet_fleet.replay.kinematic_profile import KinematicProfile
from pybullet_fleet.robot_models import auto_detect_profile, resolve_model
from pybullet_fleet.sim_object import ShapeParams, SimObject

ARM_NAME = "pick-drop-arm"
BOX_NAME = "pick-drop-box"
RECORD_NAME = "pbf.example.pick_drop_arm.stage"
JOINT_TOL = 0.05
OFFSET_POSE = Pose.from_xyz(0, 0, 0.14)

# The ordinary example keeps its original model-specific joint and box poses.
JOINT_PRESETS = {
    "arm_robot": {
        "pick": [1.5, 1.5, 1.5, 0.0],
        "place": [-1.5, 1.5, 1.5, 0.0],
        "init": [0.0, 0.0, 0.0, 0.0],
        "box_pick": [0.3, 0.0, 0.1],
        "box_place": [-0.3, 0.0, 0.1],
    },
    "panda": {
        "pick": [0.0, 0.4, 0.0, -1.5, 0.0, 1.9, 0.8, 0.0, 0.0, 0.04, 0.04, 0.0],
        "place": [3.14, 0.4, 0.0, -1.5, 0.0, 1.9, 0.8, 0.0, 0.0, 0.04, 0.04, 0.0],
        "init": [0.0, -0.8, 0.0, -2.3, 0.0, 1.6, 0.8, 0.0, 0.0, 0.04, 0.04, 0.0],
        "box_pick": [0.65, 0.0, 0.25],
        "box_place": [-0.65, 0.0, 0.25],
    },
    "kuka_iiwa": {
        "pick": [0.0, 0.5, 0.0, -1.4, 0.0, 1.2, 0.0],
        "place": [3.14, 0.5, 0.0, -1.4, 0.0, 1.2, 0.0],
        "init": [0.0, -0.5, 0.0, -1.5, 0.0, 0.7, 0.0],
        "box_pick": [0.65, 0.0, 0.25],
        "box_place": [-0.65, 0.0, 0.25],
    },
}


# Scenario-owned control state. The PBF profile saves the arm, joints and box;
# capture_stage() below saves these variables separately for continuation.
picked = False
# States: move_to_pick -> pick -> move_to_place -> drop -> move_to_init
#       -> move_to_place2 -> pick2 -> move_to_pick2 -> drop2 -> move_to_init2 (loop)
step_state = 0
_targets_set = False


def capture_stage(_context: object) -> dict:
    """Save only scenario control state as a custom DataRecord."""
    return {
        "model": _model,
        "step_state": step_state,
        "targets_set": _targets_set,
        "picked": picked,
        "pick_link_name": _pick_link_name,
    }


def configure_demo(
    sim: MultiRobotSimulationCore, model: str, *, pick_link_name: str | None = None, recording: bool = False
) -> None:
    """Bind scenario behavior to either a fresh or a PBF-restored simulation."""
    global sim_core, arm_agent, box_sim, _model, _pick_link_name
    global PICK_JOINTS, PLACE_JOINTS, JOINT_INIT, BOX_PICK_POSE, BOX_PLACE_POSE, PICK_LINK_INDEX
    sim_core = sim
    _model = model
    arm_agent = sim.get_unique_agent_by_name(ARM_NAME)
    box_sim = sim.get_unique_object_by_name(BOX_NAME)
    preset = JOINT_PRESETS[model]
    PICK_JOINTS, PLACE_JOINTS, JOINT_INIT = preset["pick"], preset["place"], preset["init"]
    BOX_PICK_POSE = Pose.from_xyz(*preset["box_pick"])
    BOX_PLACE_POSE = Pose.from_xyz(*preset["box_place"])
    if pick_link_name is None:
        # Profile detection probes joints. Do this only before a fresh run.
        profile = auto_detect_profile(arm_agent.body_id, sim.client)
        PICK_LINK_INDEX = profile.ee_link_index if profile.ee_link_index >= 0 else arm_agent.get_num_joints() - 1
        print(f"Using {model}: EE={profile.ee_link_name}, joints={len(profile.movable_joint_names)}")
        print(f"  PICK={[round(x, 2) for x in PICK_JOINTS]}  PLACE={[round(x, 2) for x in PLACE_JOINTS]}")
    else:
        links = [
            i
            for i in range(arm_agent.get_num_joints())
            if p.getJointInfo(arm_agent.body_id, i, physicsClientId=sim.client)[12].decode("utf-8") == pick_link_name
        ]
        if len(links) != 1:
            raise ValueError("Saved pick link is absent or ambiguous")
        PICK_LINK_INDEX = links[0]
    _pick_link_name = p.getJointInfo(arm_agent.body_id, PICK_LINK_INDEX, physicsClientId=sim.client)[12].decode("utf-8")
    if pick_link_name is None:
        print(f"  EE link index: {PICK_LINK_INDEX} ({_pick_link_name})")
    # The 10 Hz callback's scheduler phase is not checkpointed. At a 0.1 s
    # timestep, every-step recording has the same nominal cadence and resumes
    # without needing that hidden phase. Ordinary runs retain the original 10 Hz.
    sim.register_callback(pick_drop_callback, frequency=None if recording else 10)


def restore_pick_drop_stage(sim: MultiRobotSimulationCore, value: dict) -> None:
    """Restore scenario control state and callback after PBF restores entities."""
    global picked, step_state, _targets_set
    if set(value) != {"model", "step_state", "targets_set", "picked", "pick_link_name"}:
        raise ValueError("Invalid pick/drop stage record")
    if type(value["step_state"]) is not int or type(value["targets_set"]) is not bool or type(value["picked"]) is not bool:
        raise ValueError("Invalid pick/drop stage value")
    if value["model"] not in JOINT_PRESETS or not 0 <= value["step_state"] <= 7:
        raise ValueError("Unsupported pick/drop stage")
    if not isinstance(value["pick_link_name"], str) or not value["pick_link_name"]:
        raise ValueError("Saved pick link name is missing")
    configure_demo(sim, value["model"], pick_link_name=value["pick_link_name"], recording=True)
    picked, step_state, _targets_set = value["picked"], value["step_state"], value["targets_set"]


# Register on import so direct load_supported_simulation() callers can restore
# this application record without going through this example's main().
register_data_restorer(RECORD_NAME, 1, restore_pick_drop_stage)


def pick_drop_callback(sim_core, dt):
    """Drive the pick/drop scenario; recording does not execute this on playback."""
    global picked
    step = sim_core.step_count

    def _set_targets_once(targets):
        """Set joint targets only once per state transition (avoids redundant calls)."""
        global _targets_set
        if not _targets_set:
            arm_agent.set_all_joints_targets(targets)
            _targets_set = True

    def _advance(next_state, msg):
        """Advance to next state and reset target flag."""
        global step_state, _targets_set
        step_state = next_state
        _targets_set = False
        print(f"[STEP {step}] {msg}")

    if step_state == 0:
        # Move to initial pose, place box at pick position
        _set_targets_once(JOINT_INIT)
        box_sim.set_pose(BOX_PICK_POSE)
        if arm_agent.are_all_joints_at_targets(JOINT_INIT, tolerance=JOINT_TOL):
            _advance(1, "At init -> moving to pick")

    elif step_state == 1:
        # Move arm to pick position
        _set_targets_once(PICK_JOINTS)
        if arm_agent.are_all_joints_at_targets(PICK_JOINTS, tolerance=JOINT_TOL):
            arm_agent.attach_object(box_sim, parent_link_index=PICK_LINK_INDEX, relative_pose=OFFSET_POSE)
            picked = True
            _advance(2, "Picked box (attached)")

    elif step_state == 2:
        # Move arm to place position (box follows via attach)
        _set_targets_once(PLACE_JOINTS)
        if arm_agent.are_all_joints_at_targets(PLACE_JOINTS, tolerance=JOINT_TOL):
            arm_agent.detach_object(box_sim)
            picked = False
            _advance(3, "Dropped box (detached)")

    elif step_state == 3:
        # Return to initial pose
        _set_targets_once(JOINT_INIT)
        if arm_agent.are_all_joints_at_targets(JOINT_INIT, tolerance=JOINT_TOL):
            _advance(4, "At init -> starting reverse")

    elif step_state == 4:
        # Place box at place position, move to initial
        _set_targets_once(JOINT_INIT)
        box_sim.set_pose(BOX_PLACE_POSE)
        if arm_agent.are_all_joints_at_targets(JOINT_INIT, tolerance=JOINT_TOL):
            _advance(5, "Moving to place (reverse)")

    elif step_state == 5:
        # Move arm to place position to pick box
        _set_targets_once(PLACE_JOINTS)
        if arm_agent.are_all_joints_at_targets(PLACE_JOINTS, tolerance=JOINT_TOL):
            arm_agent.attach_object(box_sim, parent_link_index=PICK_LINK_INDEX, relative_pose=OFFSET_POSE)
            picked = True
            _advance(6, "Picked box (attached, reverse)")

    elif step_state == 6:
        # Move arm to pick position (carrying box back)
        _set_targets_once(PICK_JOINTS)
        if arm_agent.are_all_joints_at_targets(PICK_JOINTS, tolerance=JOINT_TOL):
            arm_agent.detach_object(box_sim)
            picked = False
            _advance(7, "Dropped box (detached, reverse)")

    elif step_state == 7:
        # Return to initial pose, then loop
        _set_targets_once(JOINT_INIT)
        if arm_agent.are_all_joints_at_targets(JOINT_INIT, tolerance=JOINT_TOL):
            _advance(0, "At init (loop)")


def build_example(model: str, *, gui: bool | None = None, recording: bool = False) -> MultiRobotSimulationCore:
    """Construct the ordinary example using its original robot and box."""
    if model not in JOINT_PRESETS:
        raise ValueError(f"No preset for {model!r}; available: {', '.join(JOINT_PRESETS)}")
    overrides: dict = {"simulation": {}}
    if gui is not None:
        overrides["simulation"]["gui"] = gui
    if recording:
        # Collision state is not yet part of this supported checkpoint profile.
        overrides["simulation"].update({"physics": False, "collision_check_frequency": 0})
    sim = MultiRobotSimulationCore.from_dict(merge_configs(load_yaml_config("config/config.yaml"), overrides))
    arm_asset = resolve_model(model)
    Agent.from_urdf(
        urdf_path=arm_asset,
        pose=Pose.from_xyz(0, 0, 0),
        use_fixed_base=True,
        sim_core=sim,
        name=ARM_NAME,
    )
    box_start = Pose.from_xyz(*JOINT_PRESETS[model]["box_pick"])
    SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="box", half_extents=[0.05] * 3, rgba_color=[1, 0, 0, 1]),
        collision_shape=ShapeParams(shape_type="box", half_extents=[0.05] * 3),
        pose=box_start,
        mass=0.0,
        sim_core=sim,
        name=BOX_NAME,
    )
    if gui is not False:
        sim.setup_camera(
            camera_config={
                "camera_mode": "manual",
                "camera_distance": 2.0,
                "camera_yaw": 45,
                "camera_pitch": -30,
                "camera_target": [0, 0, 0.2],
            }
        )
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-arm pick/drop demo")
    parser.add_argument("--robot", default="panda", choices=tuple(JOINT_PRESETS))
    parser.add_argument("--rtf", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="DIR")
    parser.add_argument("--restore", metavar="DIR")
    parser.add_argument("--time", type=float, default=None, help="latest checkpoint at or before this simulated time")
    args = parser.parse_args()
    if args.restore:
        if args.record is not None or args.time is None:
            parser.error("--restore requires --time and cannot be combined with --record")
        # PBF rebuilds the supported world, then invokes the registered
        # application restorer above before returning the runnable core.
        sim = load_supported_simulation(
            args.restore, at_or_before=args.time, gui=not args.headless, target_rtf=args.rtf or 0.0
        )
    else:
        sim = build_example(args.robot, gui=False if args.headless else None, recording=args.record is not None)
        if args.rtf is not None:
            sim.params.target_rtf = args.rtf
        global picked, step_state, _targets_set
        picked, step_state, _targets_set = False, 0, False
        configure_demo(sim, args.robot, recording=args.record is not None)
        if args.record is not None:
            # PBF's profile records execution state; this DataRecord adds the
            # scenario variables needed to make later callback decisions.
            recorder = sim.configure_state_recording(
                profile=KinematicProfile(allowed_callbacks=(pick_drop_callback,)),
                output=args.record or None,
                records=(DataRecord(RECORD_NAME, 1, capture_stage, required_for_restore=True),),
            )
            print(f"Recording: {recorder.path}")
    # Both paths now run the same scenario callback through the ordinary loop.
    sim.run_simulation(duration=args.duration)


if __name__ == "__main__":
    main()
