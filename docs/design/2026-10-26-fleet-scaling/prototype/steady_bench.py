"""Steady-state: all N agents keep moving (long straight goals), measure per-phase."""

import os
import statistics
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, os.path.join(_ROOT, "benchmark"))
sys.path.insert(0, _ROOT)
import batch_perf as bp
from pybullet_fleet.commands import RobotGoalCommand2D
from pybullet_fleet.fleet_api import FleetCommandDispatcher

n = int(sys.argv[1])
mode = sys.argv[2]
coll = int(sys.argv[3])
sim, mgr, agents = bp._make_manager_and_agents(n, coll, mode, "batch")
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
        acc[k].append(prof.get(k, 0.0))
moving = sum(1 for a in agents if a.is_moving)
print(f"N={n} mode={mode} coll={coll} moving={moving}/{n} dt=0.1")
for k in keys:
    m = statistics.mean(acc[k])
    print(f"  {k:24s} mean={m:7.3f}ms p50={statistics.median(acc[k]):7.3f}")
t = statistics.mean(acc["total"])
print(f"  RTF={100.0/t:.1f}x  (50x needs <=2.0ms)")
