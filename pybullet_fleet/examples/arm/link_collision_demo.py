#!/usr/bin/env python3
"""
link_collision_demo.py
Collision of an articulated body's links, not only its base.

A body's broadphase box is the union of every link that has a collision shape,
so a link reaching out past the base is found and checked. This demo puts the
contact where only a link can make it:

1. A fixed arm sweeps its base joint. A box stands within the arm's reach, but
   well outside its base: only the upper arm passes through it.
2. A mobile manipulator drives along a line with its arm stretched out to the
   side, past a pillar the base clears by 0.25 m: only the arm hits it.
3. A joint moved directly through PyBullet (``p.resetJointState``) is seen
   after ``notify_geometry_changed``: the arm is swung onto a second box by
   hand between two steps.

Every contact turns both bodies the collision colour and prints a
COLLISION_STARTED / COLLISION_ENDED line. At the end, the demo checks that
each obstacle was hit by a link and never by a base.

Usage:
    python examples/arm/link_collision_demo.py
    python examples/arm/link_collision_demo.py --no-gui --duration 20

    Press 'w' to toggle collision shape visualization (wireframe)
    Press SPACE to pause/resume simulation
"""

import argparse
import math
import os
import sys

import pybullet as p

# Run from a source checkout without installing: fall back to the repo root
# so `import pybullet_fleet` resolves. Installed/editable users never hit this.
try:
    import pybullet_fleet  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    import pybullet_fleet  # noqa: F401

from pybullet_fleet.action import JointAction, MoveAction, WaitAction
from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.geometry import Path, Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject

ARM_JOINTS = ("base_to_shoulder", "shoulder_to_elbow", "elbow_to_wrist", "wrist_to_end")
MM_ARM_JOINTS = ("mount_to_shoulder", "shoulder_to_elbow", "elbow_to_wrist", "wrist_to_end")

#: With its elbow joint at 90 degrees the bundled arm's upper arm lies level,
#: pointing along -y at a shoulder yaw of 0, +x at +90 degrees, +y at 180 and
#: -x at -90; its end effector is then 0.34-0.46 m out, 0.29-0.41 m up.
LEVEL = math.pi / 2
#: Scene 1: the fixed arm, and a box 0.45 m out along +x, at end-effector height.
#: The arm's base is 0.15 m wide each way.
ARM_AT = (0.0, -2.0, 0.0)
SWEEP_BOX_AT = (ARM_AT[0] + 0.45, ARM_AT[1], 0.35)
#: Scene 2: the mobile manipulator drives along y = 0 from x = -2 to x = 2 with
#: its arm pointing to +y (shoulder yaw 180 degrees). Its base and wheels end
#: 0.33 m to the side; the pillar starts 0.37 m to the side, inside the arm's reach.
PILLAR_AT = (0.0, 0.42, 0.4)
#: Scene 3: a box along -x, which the sweep in scene 1 never reaches.
HAND_BOX_AT = (ARM_AT[0] - 0.45, ARM_AT[1], 0.35)


def _box(sim_core, name, at, half_extents, rgba):
    return SimObject.from_mesh(
        visual_shape=ShapeParams(shape_type="box", half_extents=half_extents, rgba_color=rgba),
        collision_shape=ShapeParams(shape_type="box", half_extents=half_extents),
        pose=Pose.from_xyz(*at),
        mass=0.0,
        sim_core=sim_core,
        name=name,
    )


def _joint_index(agent, name):
    for index in range(p.getNumJoints(agent.body_id, physicsClientId=agent._pid)):
        if p.getJointInfo(agent.body_id, index, physicsClientId=agent._pid)[1].decode() == name:
            return index
    raise KeyError(name)


def _link_name(body, link):
    if link < 0:
        return "base"
    return p.getJointInfo(body.body_id, link, physicsClientId=body._pid)[12].decode()


def _touching_links(sim_core, a, b):
    """Which links of ``a`` touch ``b`` right now (closest points within the margin)."""
    margin = sim_core.params.collision_margin
    margin = max(margin) if isinstance(margin, (list, tuple)) else margin
    points = p.getClosestPoints(a.body_id, b.body_id, margin, physicsClientId=sim_core.client)
    return sorted({_link_name(a, point[3]) for point in points})


def main():
    parser = argparse.ArgumentParser(description="Link collision demo")
    parser.add_argument("--duration", type=float, default=24.0, help="Simulation duration (seconds)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Target real-time factor (0: as fast as possible)")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    args = parser.parse_args()

    sim_core = MultiRobotSimulationCore(
        SimulationParams(
            gui=not args.no_gui,
            physics=False,
            target_rtf=args.rtf,
            timestep=0.02,
            monitor=False,
            enable_monitor_gui=False,
            enable_collision_color_change=True,
        )
    )
    sim_core.initialize_simulation()
    sim_core.params.camera_config.update({"camera_distance": 5.0, "camera_target": [0.0, -0.8, 0.3]})

    # Scene 1 and 3: a fixed arm with its upper arm held level, and two boxes.
    arm = Agent.from_params(
        AgentSpawnParams(
            name="arm", urdf_path="arm_robot", initial_pose=Pose.from_xyz(*ARM_AT), use_fixed_base=True, mass=0.0
        ),
        sim_core=sim_core,
    )
    sweep_box = _box(sim_core, "sweep_box", SWEEP_BOX_AT, [0.05, 0.05, 0.1], [0.95, 0.67, 0.21, 1.0])
    hand_box = _box(sim_core, "hand_box", HAND_BOX_AT, [0.05, 0.05, 0.1], [0.67, 0.65, 0.2, 1.0])

    # Scene 2: a mobile manipulator with its arm stretched out to the left, and a pillar.
    robot = Agent.from_params(
        AgentSpawnParams(
            name="mobile_manipulator",
            urdf_path="mobile_manipulator",
            initial_pose=Pose.from_xyz(-2.0, 0.0, 0.1),
            controller={"type": "omni", "max_linear_vel": 0.6, "max_linear_accel": 1.0},
            mass=0.0,
        ),
        sim_core=sim_core,
    )
    pillar = _box(sim_core, "pillar", PILLAR_AT, [0.05, 0.05, 0.4], [0.64, 0.26, 0.1, 1.0])

    hits = {}  # obstacle name -> set of link names that touched it

    def on_started(obj_a, obj_b, **_):
        names = {obj_a.name, obj_b.name}
        for agent in (arm, robot):
            for obstacle in (sweep_box, hand_box, pillar):
                if names == {agent.name, obstacle.name}:
                    links = _touching_links(sim_core, agent, obstacle)
                    hits.setdefault(obstacle.name, set()).update(links)
                    print(f"[{sim_core.sim_time:6.2f}s] COLLISION_STARTED {agent.name} / {obstacle.name}: links {links}")

    def on_ended(obj_a, obj_b, **_):
        print(f"[{sim_core.sim_time:6.2f}s] COLLISION_ENDED   {obj_a.name} / {obj_b.name}")

    sim_core.events.on(SimEvents.COLLISION_STARTED, on_started)
    sim_core.events.on(SimEvents.COLLISION_ENDED, on_ended)

    # Scene 1: upper arm level, pointing along -y, then sweep the shoulder from
    # 0 to 180 degrees and back twice: through the box at +x (90 degrees) each way.
    for yaw in (0.0, math.pi, 0.0, math.pi, 0.0):
        arm.add_action(JointAction(target_joint_positions=dict(zip(ARM_JOINTS, (yaw, LEVEL, 0.0, 0.0)))))
    arm.add_action(WaitAction(duration=1.0))

    # Scene 2: arm turned to +y and laid level, then drive past the pillar.
    stretched = (math.pi, LEVEL, 0.0, 0.0)
    robot.add_action(JointAction(target_joint_positions=dict(zip(MM_ARM_JOINTS, stretched))))
    robot.add_action(MoveAction(path=Path.from_positions([[2.0, 0.0, 0.1]]), final_orientation_align=False))

    # Scene 3: once scene 1 is done, swing the arm onto hand_box by hand.
    shoulder = _joint_index(arm, "base_to_shoulder")
    state = {"moved_by_hand": False}

    def move_by_hand(sim, dt):
        del dt
        if state["moved_by_hand"] or not arm.is_action_queue_empty():
            return
        p.resetJointState(arm.body_id, shoulder, -math.pi / 2, physicsClientId=sim.client)
        sim.notify_geometry_changed(arm)  # without this, the moved link is not seen until the arm moves again
        state["moved_by_hand"] = True
        print(f"[{sim.sim_time:6.2f}s] base_to_shoulder reset to -90 degrees with p.resetJointState")

    sim_core.register_callback(move_by_hand, frequency=10.0)

    sim_core.run_simulation(duration=args.duration)

    print("\nLinks that touched each obstacle:")
    ok = True
    for obstacle in (sweep_box, pillar, hand_box):
        links = sorted(hits.get(obstacle.name, ()))
        print(f"  {obstacle.name}: {links or 'none'}")
        ok = ok and bool(links) and "base" not in links
    print("OK: every obstacle was hit by a link, none by a base" if ok else "UNEXPECTED: see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
