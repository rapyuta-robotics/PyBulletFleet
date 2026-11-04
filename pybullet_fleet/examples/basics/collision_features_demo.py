#!/usr/bin/env python3
"""
collision_features_demo.py
Comprehensive demonstration of collision detection features.

This demo showcases:
1. Multi-cell registration: Large objects span multiple spatial grid cells
2. Collision modes: NORMAL_3D, NORMAL_2D, DISABLED
3. Collision visualization: Color changes when objects collide
4. Real-time collision statistics

Features demonstrated:
- NORMAL_3D mode (Green, Orange, Cyan): full 3D collision detection
- NORMAL_2D mode (Blue, Magenta): XY collision detection; z must actually overlap
- DISABLED mode (White): No collision detection at all
- Multi-cell registration (Red center object): every object is registered in each grid cell it reaches
- Collision color change: Objects change color when actively colliding
- Real-time performance metrics: Monitor collision detection performance

Layout:
- Left side (x < 0): 3 robots with NORMAL_3D, NORMAL_2D, NORMAL_3D modes
- Right side (x > 0): 3 robots with NORMAL_3D, NORMAL_2D, DISABLED modes
- Center: Large multi-cell object for collision demonstration

Usage:
    python examples/basics/collision_features_demo.py
    python examples/basics/collision_features_demo.py --duration 30 --rtf 1.5

    Press 'w' to toggle collision shape visualization (wireframe)
    Press SPACE to pause/resume simulation
"""

import argparse
import os
import sys


import numpy as np

# Run from a source checkout without installing: fall back to the repo root
# so `import pybullet_fleet` resolves. Installed/editable users never hit this.
try:
    import pybullet_fleet  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    import pybullet_fleet  # noqa: F401

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.sim_object import SimObject, ShapeParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore
from pybullet_fleet.geometry import Pose
from pybullet_fleet.types import CollisionMode

# ========================================
# Configuration
# ========================================

# Bundled mesh resolved from the installed package (works installed or from checkout).

MESH_PATH = os.path.join(os.path.dirname(pybullet_fleet.__file__), "mesh", "cube.obj")

# Simulation area: 20m x 20m
AREA_SIZE = 20.0

# Small cell size to demonstrate multi-cell registration
CELL_SIZE = 2.0  # 2m cells

# Collision detection margin (safety buffer)
COLLISION_MARGIN = 0.1  # 10cm

# ========================================
# Demo Configuration
# ========================================


def create_demo_objects(sim_core: MultiRobotSimulationCore):
    """
    Create demonstration objects showcasing different collision features.

    Layout:
    - Left side (x < 0): 3 robots with NORMAL_3D, NORMAL_2D, NORMAL_3D modes
    - Right side (x > 0): 3 robots with NORMAL_3D, NORMAL_2D, DISABLED modes
    - Center: Large object demonstrating multi-cell registration
    """
    objects = []

    print("\n" + "=" * 70)
    print("Creating Demo Objects")
    print("=" * 70)

    # ========================================
    # Left Side: 3D, 2D, STATIC modes
    # ========================================
    print("\n--- Left Side Robots (NORMAL_3D, NORMAL_2D, NORMAL_3D) ---")

    # Robot 1: NORMAL_3D (Green)
    print("  [1] NORMAL_3D mode (Green)")
    spawn_params = AgentSpawnParams(
        initial_pose=Pose.from_xyz(-5.0, -4.0, 0.5),
        visual_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
            rgba_color=[0.2, 0.8, 0.2, 1.0],  # Green
        ),
        collision_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
        ),
        mass=0.0,
        collision_mode=CollisionMode.NORMAL_3D,
        name="Left_3D_Robot",
    )
    robot = Agent.from_params(spawn_params, sim_core=sim_core)
    objects.append(robot)
    print(f"      Created at position {spawn_params.initial_pose.position}")

    # Robot 2: NORMAL_2D (Blue)
    print("  [2] NORMAL_2D mode (Blue) (XY only)")
    spawn_params = AgentSpawnParams(
        initial_pose=Pose.from_xyz(-5.0, 0.0, 0.5),
        visual_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
            rgba_color=[0.2, 0.2, 0.8, 1.0],  # Blue
        ),
        collision_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
        ),
        mass=0.0,
        collision_mode=CollisionMode.NORMAL_2D,
        name="Left_2D_Robot",
    )
    robot = Agent.from_params(spawn_params, sim_core=sim_core)
    objects.append(robot)
    print(f"      Created at position {spawn_params.initial_pose.position}")

    # Robot 3: Another NORMAL_3D with different color (Orange)
    print("  [3] NORMAL_3D mode (Orange)")
    spawn_params = AgentSpawnParams(
        initial_pose=Pose.from_xyz(-5.0, 4.0, 0.5),
        visual_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
            rgba_color=[1.0, 0.5, 0.0, 1.0],  # Orange
        ),
        collision_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
        ),
        mass=0.0,
        collision_mode=CollisionMode.NORMAL_3D,
        name="Left_3D_Robot_2",
    )
    robot = Agent.from_params(spawn_params, sim_core=sim_core)
    objects.append(robot)
    print(f"      Created at position {spawn_params.initial_pose.position}")

    # ========================================
    # Right Side: 3D, 2D, DISABLED modes
    # ========================================
    print("\n--- Right Side Robots (NORMAL_3D, NORMAL_2D, DISABLED) ---")

    # Robot 4: NORMAL_3D (Cyan)
    print("  [4] NORMAL_3D mode (Cyan)")
    spawn_params = AgentSpawnParams(
        initial_pose=Pose.from_xyz(5.0, -4.0, 0.5),
        visual_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
            rgba_color=[0.2, 0.8, 0.8, 1.0],  # Cyan
        ),
        collision_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
        ),
        mass=0.0,
        collision_mode=CollisionMode.NORMAL_3D,
        name="Right_3D_Robot",
    )
    robot = Agent.from_params(spawn_params, sim_core=sim_core)
    objects.append(robot)
    print(f"      Created at position {spawn_params.initial_pose.position}")

    # Robot 5: NORMAL_2D (Magenta)
    print("  [5] NORMAL_2D mode (Magenta) (XY only)")
    spawn_params = AgentSpawnParams(
        initial_pose=Pose.from_xyz(5.0, 0.0, 0.5),
        visual_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
            rgba_color=[0.8, 0.2, 0.8, 1.0],  # Magenta
        ),
        collision_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
        ),
        mass=0.0,
        collision_mode=CollisionMode.NORMAL_2D,
        name="Right_2D_Robot",
    )
    robot = Agent.from_params(spawn_params, sim_core=sim_core)
    objects.append(robot)
    print(f"      Created at position {spawn_params.initial_pose.position}")

    # Robot 6: DISABLED (White/Gray)
    print("  [6] DISABLED mode (White) - No collision detection")
    spawn_params = AgentSpawnParams(
        initial_pose=Pose.from_xyz(5.0, 4.0, 0.5),
        visual_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
            rgba_color=[0.9, 0.9, 0.9, 1.0],  # White
        ),
        collision_shape=ShapeParams(
            shape_type="mesh",
            mesh_path=MESH_PATH,
            mesh_scale=[0.5, 0.5, 0.5],
        ),
        mass=0.0,
        collision_mode=CollisionMode.DISABLED,
        name="Right_DISABLED_Robot",
    )
    robot = Agent.from_params(spawn_params, sim_core=sim_core)
    objects.append(robot)
    print(f"      Created at position {spawn_params.initial_pose.position}")

    # ========================================
    # Center: Large Multi-Cell Object
    # ========================================
    print("\n--- Center: Large Multi-Cell Object ---")
    print(f"  - Cell size: {CELL_SIZE}m")
    print("  - Every object is registered in each cell it reaches (its AABB plus half the margin)")

    # Large cube (4m x 4m x 2m) - will span multiple cells
    large_obj = SimObject.from_mesh(
        visual_shape=ShapeParams(
            shape_type="box",
            half_extents=[2.0, 2.0, 1.0],  # 4m x 4m x 2m
            rgba_color=[0.8, 0.2, 0.2, 1.0],  # Red
        ),
        collision_shape=ShapeParams(
            shape_type="box",
            half_extents=[2.0, 2.0, 1.0],
        ),
        pose=Pose.from_xyz(0.0, 0.0, 1.0),
        mass=0.0,
        sim_core=sim_core,
        collision_mode=CollisionMode.NORMAL_3D,
    )
    objects.append(large_obj)

    # Check multi-cell registration
    cells = sim_core._cached_object_to_cell.get(large_obj.object_id, [])
    print(f"  Large object (4×4×2m) registered to {len(cells)} cells: {cells[:5]}{'...' if len(cells) > 5 else ''}")

    print(f"\nTotal objects created: {len(objects)}")
    print("\nCollision Mode Summary:")
    print("  - NORMAL_3D (Green, Orange, Cyan): Full 3D collision detection")
    print("  - NORMAL_2D (Blue, Magenta): XY-plane only collision")
    print("  - DISABLED (White): No collision at all")
    print("  - Large object (Red): Multi-cell registration demo")

    return objects


def setup_camera(sim_core: MultiRobotSimulationCore):
    """Setup camera for optimal viewing."""
    # Top-down view to see all objects
    sim_core.setup_camera(
        camera_config={
            "camera_mode": "manual",
            "camera_distance": 25.0,
            "camera_yaw": 0,
            "camera_pitch": -45,
            "camera_target": [0, 0, 0],
        }
    )


def create_movement_pattern(t: float, robot_id: int, side: str = "left") -> Pose:
    """
    Create circular movement pattern for robots.

    Args:
        t: Current time
        robot_id: Robot identifier (0, 1, 2) within each side
        side: "left" or "right" side of the field

    Returns:
        New pose for the robot
    """
    # Phase offset for each robot (0, 1, 2 on each side)
    phase = robot_id * (2 * np.pi / 3)

    # Circular motion parameters
    radius = 3.0
    angular_speed = 0.5  # rad/s

    # Center of circle
    center_x = -5.0 if side == "left" else 5.0
    center_y = 0.0

    # Calculate position
    angle = angular_speed * t + phase
    x = center_x + radius * np.cos(angle)
    y = center_y + radius * np.sin(angle)
    z = 0.5

    # Calculate orientation (face movement direction)
    yaw = angle + np.pi / 2

    return Pose.from_yaw(x, y, z, yaw)


def main():
    parser = argparse.ArgumentParser(description="Collision Features Demo")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration (seconds)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Target real-time factor override")
    args = parser.parse_args()

    print("=" * 70)
    print("Collision Features Demo")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("  1. Multi-cell registration for large objects")
    print("  2. All collision modes: NORMAL_3D, NORMAL_2D, STATIC, DISABLED")
    print("  3. Collision visualization with color changes")
    print("  4. Real-time performance monitoring")
    print("\nRobot Layout:")
    print("  - Left side (3 robots):")
    print("      * Green: NORMAL_3D mode")
    print("      * Blue: NORMAL_2D mode")
    print("      * Orange: NORMAL_3D mode")
    print("  - Right side (3 robots):")
    print("      * Cyan: NORMAL_3D mode")
    print("      * Magenta: NORMAL_2D mode")
    print("      * White: DISABLED mode (no collision detection)")
    print("  - Center: Large red object (multi-cell registration)")
    print("\nControls:")
    print("  - Press 'w' to toggle collision shape visualization")
    print("  - Press SPACE to pause/resume simulation")
    print("=" * 70)

    # ========================================
    # Create Simulation
    # ========================================
    sim_core = MultiRobotSimulationCore.from_dict(
        {
            "simulation": {
                "gui": True,
                "physics": False,
                "monitor": True,
                "enable_monitor_gui": True,
                "collision_margin": COLLISION_MARGIN,
                "spatial_hash_cell_size_mode": "constant",
                "spatial_hash_cell_size": CELL_SIZE,
                "enable_collision_color_change": True,
                "enable_collision_shapes": False,
                "enable_time_profiling": True,
                "duration": args.duration,
                "timestep": 0.01,
                "target_rtf": args.rtf,
                "log_level": "info",
            }
        }
    )
    sim_core.set_collision_spatial_hash_cell_size_mode()

    # Enable collision visualization controls
    sim_core.configure_visualizer(
        enable_structure_transparency=False,
        enable_shadows=True,
    )

    # Create demo objects
    objects = create_demo_objects(sim_core)

    # Filter robots for movement - all except large central object (which has no name attribute)
    # Get robots by name to identify their movement group
    left_robots = [obj for obj in objects if getattr(obj, "name", None) and obj.name.startswith("Left_")]
    right_robots = [obj for obj in objects if getattr(obj, "name", None) and obj.name.startswith("Right_")]

    # Note: STATIC robot will still be moved to show it maintains STATIC mode behavior

    print("\n" + "=" * 70)
    print("Simulation Configuration")
    print("=" * 70)
    print(f"Cell size: {sim_core._cached_cell_size}m")
    print(f"Collision margin: {COLLISION_MARGIN}m")
    print(f"Total cells in grid: {len(sim_core._cached_spatial_grid)}")
    print(f"Total objects: {len(objects)}")
    print(f"  - Left side robots: {len(left_robots)} ({', '.join([r.name for r in left_robots])})")
    print(f"  - Right side robots: {len(right_robots)} ({', '.join([r.name for r in right_robots])})")
    print(f"  - Static objects: {len(objects) - len(left_robots) - len(right_robots)}")

    # Setup camera
    setup_camera(sim_core)

    # Enable rendering
    sim_core.enable_rendering()

    print("\n" + "=" * 70)
    print("Starting Simulation")
    print("=" * 70)

    # ========================================
    # Simulation Loop
    # ========================================
    def update_callback(sim_core: MultiRobotSimulationCore, dt: float):
        """Update robot positions in circular patterns."""
        t = sim_core.sim_time

        # Move left side robots (all 3 robots move, but STATIC robot won't trigger updates)
        for i, robot in enumerate(left_robots):
            new_pose = create_movement_pattern(t, i, side="left")
            robot.set_pose(new_pose)

        # Move right side robots
        for i, robot in enumerate(right_robots):
            new_pose = create_movement_pattern(t, i, side="right")
            robot.set_pose(new_pose)

        # Log collision statistics every 5 seconds
        if int(t) % 5 == 0 and abs(t - int(t)) < dt:
            active_collisions = sim_core.get_active_collision_pairs()
            print(f"\n[t={t:.1f}s] Active collisions: {len(active_collisions)}")
            if active_collisions:
                print(f"  Collision pairs: {active_collisions[:3]}{'...' if len(active_collisions) > 3 else ''}")

    # Register update callback
    sim_core.register_callback(update_callback, frequency=None)  # Run every step

    # Run simulation
    try:
        sim_core.run_simulation(duration=args.duration)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    print("\n" + "=" * 70)
    print("Simulation Complete")
    print("=" * 70)
    print(f"Total simulation time: {sim_core.sim_time:.2f}s")
    print(f"Total collisions detected: {sim_core.collision_count}")
    print(f"Final active collisions: {len(sim_core.get_active_collision_pairs())}")


if __name__ == "__main__":
    main()
