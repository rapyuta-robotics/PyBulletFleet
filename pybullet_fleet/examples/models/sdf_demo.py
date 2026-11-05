#!/usr/bin/env python3
"""SDF & mesh loading demo — bulk-loading patterns.

Demonstrates:
- SimObject.from_sdf() with a multi-model SDF (one file → 5 SimObjects)
- SimObject.from_sdf() with global_scaling and name_prefix
- load_mesh_directory() — bulk-load a folder of OBJ meshes

Usage:
    python examples/models/sdf_demo.py
"""

import os
import sys


import pybullet_data

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
    load_mesh_directory,
)
from pybullet_fleet.config_utils import load_yaml_config, merge_configs
from pybullet_fleet.sim_object import SimObject
from pybullet_fleet.types import CollisionMode


_BASE_CONFIG = "config/config.yaml"  # bundled config, resolved against the package by load_yaml_config
_OVERRIDES = {"simulation": {"duration": 15.0, "enable_floor": False}}


def main():
    sim = MultiRobotSimulationCore.from_dict(merge_configs(load_yaml_config(_BASE_CONFIG), _OVERRIDES))

    # ==========================================================
    # 1. Multi-model SDF — one file → multiple bodies
    #    warehouse_layout.sdf contains 5 <model> tags:
    #      ground_plane + shelf_A1/A2/B1/B2
    #    loadSDF() returns all 5 as separate SimObjects.
    # ==========================================================
    warehouse = SimObject.from_sdf("mesh/warehouse_layout.sdf", sim_core=sim)
    print(f"Loaded {len(warehouse)} bodies from warehouse_layout.sdf:")
    for obj in warehouse:
        print(f"  {obj.name} (id={obj.object_id})")

    # ==========================================================
    # 2. Single-model SDF with global_scaling / name_prefix
    # ==========================================================
    shelves = SimObject.from_sdf("kiva_shelf/model.sdf", sim_core=sim)
    for shelf in shelves:
        shelf.set_pose(Pose.from_xyz(5.0, 0.0, 0.0))
    print(f"kiva_shelf (original) → {len(shelves)} body")

    shelves_half = SimObject.from_sdf(
        "kiva_shelf/model.sdf",
        sim_core=sim,
        global_scaling=0.5,
        name_prefix="small_shelf",
    )
    for shelf in shelves_half:
        shelf.set_pose(Pose.from_xyz(-5.0, 0.0, 0.0))
    print(f"kiva_shelf (0.5x) → {len(shelves_half)} body")

    # ==========================================================
    # 3. load_mesh_directory() — bulk-load a folder of OBJ meshes
    #    collision_mode=DISABLED — visual-only, no collision overhead.
    # ==========================================================
    mesh_dir = pybullet_data.getDataPath() + "/franka_panda/meshes/collision"
    meshes = load_mesh_directory(mesh_dir, sim_core=sim, pattern="link[0-2].obj", collision_mode=CollisionMode.DISABLED)
    print(f"load_mesh_directory → {len(meshes)} objects from {mesh_dir}")

    # ==========================================================
    # Spawn a mobile robot to navigate the warehouse
    # ==========================================================
    Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_xyz(3, 0, 0.1),
            controller={"type": "omni", "max_linear_vel": 2.0},
            collision_mode=CollisionMode.NORMAL_3D,
        ),
        sim_core=sim,
    )

    sim.run_simulation()


if __name__ == "__main__":
    main()
