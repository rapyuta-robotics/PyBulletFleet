"""examples/arm/link_collision_demo.py runs headless and every obstacle is hit by a link, never a base.

Each obstacle stands where only a link can reach it, so the demo fails (exit 1)
as soon as the broadphase box stops covering an articulated body's links.
"""

import os
import subprocess
import sys

DEMO = os.path.join(os.path.dirname(__file__), "..", "pybullet_fleet", "examples", "arm", "link_collision_demo.py")


def test_every_obstacle_is_hit_by_a_link_and_none_by_a_base():
    result = subprocess.run(
        [sys.executable, DEMO, "--no-gui", "--rtf", "0", "--duration", "10"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "OK: every obstacle was hit by a link, none by a base" in result.stdout
    for obstacle in ("sweep_box", "pillar", "hand_box"):
        line = next(line for line in result.stdout.splitlines() if line.strip().startswith(obstacle + ":"))
        assert "end_effector" in line and "base" not in line, line
