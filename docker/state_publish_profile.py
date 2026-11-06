#!/usr/bin/env python3
"""Where the time goes when the state publisher publishes one 1000-robot FleetState.

Splits the conversion from a state-plane snapshot, the serialization, and the
publish call (with one same-process subscriber), inside the bridge image::

    docker compose -f docker/docker-compose.yaml run --rm --no-deps \
      -v "$(pwd)/docker:/docker:ro" bridge python3 /docker/state_publish_profile.py --robots 1000
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import rclpy
from rclpy.serialization import serialize_message

from pybullet_fleet.state_plane import Snapshot
from pybullet_fleet_msgs.msg import FleetState
from pybullet_fleet_ros.conversions import sim_time_to_ros_time
from pybullet_fleet_ros.fleet_ros_interface import snapshot_to_fleet_state_msg


def _snapshot(n: int) -> Snapshot:
    return Snapshot(
        seq=1,
        generation=1,
        sim_time=12.3,
        step=123,
        names=tuple(f"robot_{i:04d}" for i in range(n)),
        object_id=np.arange(n),
        pos=np.random.rand(n, 3),
        orn=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        lin_vel=np.random.rand(n, 3),
        ang_vel=np.random.rand(n, 3),
        is_moving=np.ones(n, dtype=bool),
        battery_soc=np.full(n, 0.5, dtype=np.float32),
        is_charging=np.zeros(n, dtype=bool),
    )


def _timed(fn, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("state_publish_profile")
    pub = node.create_publisher(FleetState, "/profile/states", 10)
    received = []
    node.create_subscription(FleetState, "/profile/states", received.append, 10)
    snapshot = _snapshot(args.robots)
    stamp = sim_time_to_ros_time(snapshot.sim_time)
    for _ in range(5):
        snapshot_to_fleet_state_msg(snapshot, stamp=stamp)
    convert_ms = _timed(lambda: snapshot_to_fleet_state_msg(snapshot, stamp=stamp), args.iterations)
    msg = snapshot_to_fleet_state_msg(snapshot, stamp=stamp)
    serialize_ms = _timed(lambda: serialize_message(msg), args.iterations)
    publish_ms = _timed(lambda: pub.publish(msg), args.iterations)
    raw = serialize_message(msg)
    publish_raw_ms = _timed(lambda: pub.publish(raw), args.iterations)
    rclpy.spin_once(node, timeout_sec=0.5)
    print(
        f"robots={args.robots}: convert {convert_ms:.2f} ms, serialize {serialize_ms:.2f} ms, "
        f"publish(msg) {publish_ms:.2f} ms, publish(serialized bytes) {publish_raw_ms:.2f} ms, "
        f"payload {len(raw)} bytes, received {len(received)}"
    )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
