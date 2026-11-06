#!/usr/bin/env python3
"""EventBus demo — subscribe to simulation lifecycle and per-entity events.

Demonstrates:
- Global events: PRE_STEP, POST_STEP, OBJECT_SPAWNED, AGENT_SPAWNED,
  ACTION_STARTED, ACTION_COMPLETED, COLLISION_STARTED, COLLISION_ENDED
- Per-entity events: PRE_UPDATE, POST_UPDATE, ACTION_COMPLETED
- Custom events: any string key works

Usage:
    python examples/basics/event_bus_demo.py
"""

import os
import sys


# Run from a source checkout without installing: fall back to the repo root
# so `import pybullet_fleet` resolves. Installed/editable users never hit this.
try:
    import pybullet_fleet  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    import pybullet_fleet  # noqa: F401

from pybullet_fleet import (
    MultiRobotSimulationCore,
    Agent,
    AgentSpawnParams,
    Pose,
)
from pybullet_fleet.action import MoveAction
from pybullet_fleet.config_utils import load_yaml_config, merge_configs
from pybullet_fleet.events import SimEvents
from pybullet_fleet.geometry import Path
from pybullet_fleet.sim_object import SimObject, SimObjectSpawnParams, ShapeParams
from pybullet_fleet.types import CollisionMode


_BASE_CONFIG = "config/config.yaml"
_OVERRIDES = {"simulation": {"gui": False, "monitor": False, "duration": 3.0, "enable_floor": False}}


def main():
    sim = MultiRobotSimulationCore.from_dict(merge_configs(load_yaml_config(_BASE_CONFIG), _OVERRIDES))
    sim.initialize_simulation()

    # ==========================================================
    # 1. Global events — sim.events.on(event, handler)
    #    Handlers receive **kwargs matching the event signature
    #    documented in SimEvents docstrings.
    # ==========================================================
    step_count = {"pre": 0, "post": 0}

    def on_pre_step(dt, sim_time, **_):
        step_count["pre"] += 1
        if step_count["pre"] <= 3:
            print(f"  [PRE_STEP]   step={step_count['pre']}  sim_time={sim_time:.3f}  dt={dt:.4f}")

    def on_post_step(dt, sim_time, **_):
        step_count["post"] += 1
        if step_count["post"] <= 3:
            print(f"  [POST_STEP]  step={step_count['post']}  sim_time={sim_time:.3f}  dt={dt:.4f}")

    def on_object_spawned(obj, **_):
        print(f"  [OBJECT_SPAWNED]  {obj.name} (id={obj.object_id})")

    def on_agent_spawned(agent, **_):
        print(f"  [AGENT_SPAWNED]   {agent.name} (id={agent.body_id})")

    def on_action_started(agent, action, **_):
        print(f"  [ACTION_STARTED]  {agent.name} → {type(action).__name__}")

    def on_action_completed(agent, action, status, **_):
        print(f"  [ACTION_COMPLETED] {agent.name} → {type(action).__name__} ({status.name})")

    def on_collision_started(obj_a, obj_b, **_):
        print(f"  [COLLISION_STARTED] {obj_a.name} ↔ {obj_b.name}")

    def on_collision_ended(obj_a, obj_b, **_):
        print(f"  [COLLISION_ENDED]   {obj_a.name} ↔ {obj_b.name}")

    sim.events.on(SimEvents.PRE_STEP, on_pre_step)
    sim.events.on(SimEvents.POST_STEP, on_post_step)
    sim.events.on(SimEvents.OBJECT_SPAWNED, on_object_spawned)
    sim.events.on(SimEvents.AGENT_SPAWNED, on_agent_spawned)
    sim.events.on(SimEvents.ACTION_STARTED, on_action_started)
    sim.events.on(SimEvents.ACTION_COMPLETED, on_action_completed)
    sim.events.on(SimEvents.COLLISION_STARTED, on_collision_started)
    sim.events.on(SimEvents.COLLISION_ENDED, on_collision_ended)

    # ==========================================================
    # 2. Spawn entities — triggers OBJECT_SPAWNED / AGENT_SPAWNED
    # ==========================================================
    print("\n--- Spawning entities ---")
    box = SimObject.from_params(
        SimObjectSpawnParams(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.15, 0.15, 0.15]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.15, 0.15, 0.15]),
            initial_pose=Pose.from_xyz(1.0, 1.0, 0.15),  # on the robot's (0,0)->(2,2) path → collision
            name="obstacle_box",
            collision_mode=CollisionMode.NORMAL_3D,
        ),
        sim_core=sim,
    )

    agent = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(0, 0, 0.1),
            controller={"type": "omni", "max_linear_vel": 2.0},
            collision_mode=CollisionMode.NORMAL_3D,
            name="robot_1",
        ),
        sim_core=sim,
    )

    # ==========================================================
    # 3. Per-entity events — agent.events.on(event, handler)
    #    These fire only for this specific agent instance.
    # ==========================================================
    def on_pre_update(dt, **_):
        """Called before each agent.update() — per-entity only."""
        pass  # hot path: keep lightweight

    def on_post_update(dt, moved, **_):
        """Called after each agent.update() with movement result."""
        if moved:
            pos = agent.get_pose().position
            # Print occasionally (every ~1s) to avoid flooding
            if int(sim.sim_time * 10) % 10 == 0:
                print(f"  [POST_UPDATE]  {agent.name} moved to ({pos[0]:.1f}, {pos[1]:.1f})")

    def on_entity_action_completed(action, status, **_):
        """Per-entity version — no 'agent' kwarg (it's implied)."""
        print(f"  [per-entity ACTION_COMPLETED] {type(action).__name__} → {status.name}")

    agent.events.on(SimEvents.PRE_UPDATE, on_pre_update)
    agent.events.on(SimEvents.POST_UPDATE, on_post_update)
    agent.events.on(SimEvents.ACTION_COMPLETED, on_entity_action_completed)

    # ==========================================================
    # 4. Custom events — any string works as event name
    # ==========================================================
    def on_custom(message, **_):
        print(f"  [CUSTOM]  {message}")

    sim.events.on("my_plugin_event", on_custom)
    sim.events.emit("my_plugin_event", message="Hello from custom event!")

    # ==========================================================
    # 5. Give agent a move action → triggers ACTION_STARTED/COMPLETED
    # ==========================================================
    print("\n--- Running simulation ---")
    action = MoveAction(path=Path.from_positions([[2.0, 2.0, 0.1]]))
    agent.add_action(action)

    sim.run_simulation()

    # ==========================================================
    # Summary
    # ==========================================================
    print("\n--- Summary ---")
    print(f"Total steps:  pre={step_count['pre']}, post={step_count['post']}")
    print(f"sim_objects:  {len(sim.sim_objects)}")
    print("Done.")


if __name__ == "__main__":
    main()
