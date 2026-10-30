"""Warehouse-shaped steady state: N moving robots in aisles between tens of thousands of
static boxes (floor tiles plus bins on the odd rows). The statics never overlap a robot's
AABB in z or y, so every collision pair is broad-phase work only, as in a real aisle.

Usage: static_bench.py <n_movers> <field_width_m> [coll_hz]
  200 robots on a 2 m grid, driving +x for 200 m; tiles on a 1 m grid over
  x in [-2, 230] and y in [-2, field_width]; bins on 3 of 4 tiles of every odd row.
"""

import os
import statistics
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, os.path.join(_ROOT, "benchmark"))
sys.path.insert(0, _ROOT)
import batch_perf as bp
from pybullet_fleet import Pose
from pybullet_fleet.agent_manager import SimObjectManager
from pybullet_fleet.commands import RobotGoalCommand2D
from pybullet_fleet.fleet_api import FleetCommandDispatcher
from pybullet_fleet.sim_object import CollisionMode, ShapeParams, SimObjectSpawnParams

n = int(sys.argv[1])
width = int(sys.argv[2])
coll = int(sys.argv[3]) if len(sys.argv) > 3 else 60

TILE = ShapeParams(shape_type="box", half_extents=[0.5, 0.5, 0.05])
BIN = ShapeParams(shape_type="box", half_extents=[0.22, 0.33, 0.17])


def static_params():
    out = []
    for i in range(-2, 231):
        for j in range(-2, width + 1):
            out.append(
                SimObjectSpawnParams(
                    collision_shape=TILE,
                    visual_shape=TILE,
                    mass=0.0,
                    collision_mode=CollisionMode.STATIC,
                    initial_pose=Pose.from_xyz(float(i), float(j), -0.3),
                )
            )
            if j % 2 == 1 and i % 4 < 3:
                out.append(
                    SimObjectSpawnParams(
                        collision_shape=BIN,
                        visual_shape=BIN,
                        mass=0.0,
                        collision_mode=CollisionMode.STATIC,
                        initial_pose=Pose.from_xyz(float(i), float(j), 0.17),
                    )
                )
    return out


sim, mgr, agents = bp._make_manager_and_agents(n, coll, "diff", "batch")
t0 = time.perf_counter()
params = static_params()
with sim.batch_spawn():
    statics = SimObjectManager(sim_core=sim).spawn_objects_batch(params)
t_spawn = time.perf_counter() - t0
print(f"movers={n} statics={len(statics)} spawn={t_spawn:.1f}s")

cmds = []
for a in agents:
    pz = a.get_pose()
    cmds.append(RobotGoalCommand2D(name=a.name, position=(pz.x + 200.0, pz.y), yaw=0.0, z=pz.z))
FleetCommandDispatcher(sim).navigate(cmds, source="b", command_id="b")
for _ in range(20):
    sim.step_once()
keys = ["phase1_update", "phase2_pose_flush", "phase3_aabb_grid_flush", "collision_check", "total"]
acc = {k: [] for k in keys}
for _ in range(150):
    prof = sim.step_once(return_profiling=True)
    for k in keys:
        acc[k].append(prof.get(k, 0.0))  # already in ms
print(f"N={n} statics={len(statics)} mode=diff coll={coll} dt=0.1 active_pairs={len(sim._active_collision_pairs)}")
for k in keys:
    print(f"  {k:24s} mean={statistics.mean(acc[k]):7.3f}ms p50={statistics.median(acc[k]):7.3f}")
t = statistics.median(acc["total"])
print(f"RTF={100/t:.1f}x (median)")
