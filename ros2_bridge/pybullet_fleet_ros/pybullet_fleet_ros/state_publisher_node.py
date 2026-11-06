"""Publish /fleet/states and /clock from the shared-memory state plane, in a process of its own.

The bridge process steps the simulation; with ``state_plane.enabled`` in its
config it writes every step's fleet state to a shared-memory plane
(:mod:`pybullet_fleet.state_plane`) instead of building ROS messages on the
stepping thread. This node attaches to that plane, reads each snapshot and
publishes it, so message construction, serialization and DDS run here and
never contend with the simulation for its interpreter lock.

Run it next to the bridge::

    ros2 run pybullet_fleet_ros state_publisher --ros-args -p plane_name:=pbf_state -p mode:=every_step

or with ``ros2 launch pybullet_fleet_ros bridge.launch.py ... state_publisher:=true``.

``/clock`` is published from the snapshot's ``sim_time``, so a state and the
clock that accompanies it always name the same step. The bridge stops
publishing both when the plane is enabled.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

from pybullet_fleet.state_plane import MODE_EVERY_STEP, MODE_LATEST, MODES, SharedStateReader, Snapshot

from .conversions import sim_time_to_ros_time
from .fleet_endpoints import FLEET_API_NAMESPACE, fleet_endpoint, normalize_fleet_namespace
from .fleet_ros_interface import _fleet_state_qos, _require_fleet_msgs, snapshot_to_fleet_state_msg
from .fleet_state_cdr import FleetStateEncoder
from .interface_config import FleetApiConfig
from .param_utils import get_bool_param, get_float_param

try:
    from pybullet_fleet_msgs.msg import FleetState
except ImportError:  # pragma: no cover - the node needs the generated messages
    FleetState = None


class StatePublisherLoop:
    """The read-and-publish loop, separated from the ROS node so a test can drive it with a fake reader.

    Args:
        reader: Anything with ``read_next(timeout)`` / ``read_latest(timeout)`` returning a
            :class:`~pybullet_fleet.state_plane.Snapshot` or ``None``.
        publish: Called with each snapshot to publish.
        mode: ``every_step`` follows the stream in order, ``latest`` takes the newest.
        publish_rate: Snapshots per simulated second to publish (``None`` publishes all).
        read_timeout: Seconds a read waits before the loop checks whether to stop.
    """

    def __init__(
        self,
        reader: Any,
        publish,
        *,
        mode: str = MODE_EVERY_STEP,
        publish_rate: Optional[float] = None,
        read_timeout: float = 1.0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.reader = reader
        self._publish = publish
        self._mode = mode
        self._min_interval = None if not publish_rate else 1.0 / float(publish_rate)
        self._read_timeout = read_timeout
        self._last_published_sim_time = -math.inf
        self.snapshots_read = 0
        self.snapshots_published = 0
        self.publish_seconds = 0.0

    def once(self) -> bool:
        """Read one snapshot and publish it if it is due. Returns False on a read timeout."""
        if self._mode == MODE_LATEST:
            snapshot = self.reader.read_latest(timeout=self._read_timeout)
        else:
            snapshot = self.reader.read_next(timeout=self._read_timeout)
        if snapshot is None:
            return False
        self.snapshots_read += 1
        if self._min_interval is not None and snapshot.sim_time - self._last_published_sim_time < self._min_interval - 1e-9:
            return True
        self._last_published_sim_time = snapshot.sim_time
        started = time.perf_counter()
        self._publish(snapshot)
        self.publish_seconds += time.perf_counter() - started
        self.snapshots_published += 1
        return True

    def run(self, should_stop, on_idle=None) -> None:
        """Loop until *should_stop()* is true; *on_idle()* runs after each read timeout."""
        while not should_stop():
            if not self.once() and on_idle is not None:
                on_idle()


class StatePublisherNode(Node):
    """ROS node around :class:`StatePublisherLoop` and a :class:`SharedStateReader`."""

    def __init__(self) -> None:
        super().__init__("pybullet_fleet_state_publisher")
        from rcl_interfaces.msg import ParameterDescriptor

        # Numbers may arrive as integers from the command line (publish_rate:=10).
        _dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("plane_name", "pbf_state")
        self.declare_parameter("mode", MODE_EVERY_STEP)
        self.declare_parameter("publish_rate", 0.0, _dyn)
        self.declare_parameter("publish_clock", True, _dyn)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("fleet_namespace", FLEET_API_NAMESPACE)
        self.declare_parameter("rmf_frame_offset", [0.0, 0.0], _dyn)
        self.declare_parameter("attach_timeout", 60.0, _dyn)
        self.declare_parameter("read_timeout", 1.0, _dyn)
        # Serialize straight from the arrays (FleetStateEncoder) instead of building
        # the message: 0.05 ms instead of about 11 ms for 1000 robots.
        self.declare_parameter("raw_cdr", True, _dyn)
        _require_fleet_msgs()
        self._plane_name = str(self.get_parameter("plane_name").value)
        self._mode = str(self.get_parameter("mode").value)
        publish_rate = get_float_param(self, "publish_rate", 0.0)
        self._publish_clock = get_bool_param(self, "publish_clock", True)
        self._frame_id = str(self.get_parameter("frame_id").value)
        offset = list(self.get_parameter("rmf_frame_offset").value or [0.0, 0.0])
        self._xy_offset = (float(offset[0]), float(offset[1]))
        namespace = normalize_fleet_namespace(str(self.get_parameter("fleet_namespace").value))
        self._encoder = (
            FleetStateEncoder(frame_id=self._frame_id, xy_offset=self._xy_offset)
            if get_bool_param(self, "raw_cdr", True)
            else None
        )
        self._state_pub = self.create_publisher(
            FleetState, fleet_endpoint(namespace, "states"), _fleet_state_qos(FleetApiConfig())
        )
        self._clock_pub = self.create_publisher(Clock, "/clock", 10) if self._publish_clock else None
        self._reader = self._attach(get_float_param(self, "attach_timeout", 60.0))
        self._loop = StatePublisherLoop(
            self._reader,
            self._publish_snapshot,
            mode=self._mode,
            publish_rate=publish_rate or None,
            read_timeout=get_float_param(self, "read_timeout", 1.0),
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop.run, args=(self._stop.is_set, self._reattach_if_stale), daemon=True, name="state_plane"
        )
        self._thread.start()
        self._report_timer = self.create_timer(10.0, self._report)
        self.get_logger().info(
            f"StatePublisher started: plane={self._plane_name}, mode={self._mode}, "
            f"publish_rate={publish_rate or 'every snapshot'}, clock={self._publish_clock}, "
            f"encoding={'raw CDR' if self._encoder is not None else 'message'}"
        )

    def _attach(self, timeout: float) -> SharedStateReader:
        """Attach to the plane, waiting for the bridge to create it."""
        deadline = time.monotonic() + timeout
        announced = False
        while True:
            try:
                return SharedStateReader(self._plane_name)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"state plane {self._plane_name!r} did not appear within {timeout:.0f} s; "
                        "is the bridge running with state_plane.enabled?"
                    )
                if not announced:
                    self.get_logger().info(f"waiting for state plane {self._plane_name!r}")
                    announced = True
                time.sleep(0.25)

    def _reattach_if_stale(self) -> None:
        """After a quiet second: if the bridge restarted and replaced the plane, follow it."""
        if self._stop.is_set() or not self._reader.is_stale():
            return
        self.get_logger().warning(f"state plane {self._plane_name!r} was replaced; re-attaching")
        self._reader.close()
        self._reader = self._attach(get_float_param(self, "attach_timeout", 60.0))
        self._loop.reader = self._reader

    def _publish_snapshot(self, snapshot: Snapshot) -> None:
        if self._stop.is_set() or not rclpy.ok():
            return
        stamp = sim_time_to_ros_time(snapshot.sim_time)
        try:
            if self._clock_pub is not None:
                clock = Clock()
                clock.clock = stamp
                self._clock_pub.publish(clock)
            if self._state_pub.get_subscription_count() == 0:
                return
            if self._encoder is not None:
                self._state_pub.publish(
                    self._encoder.encode_snapshot(snapshot, stamp_sec=stamp.sec, stamp_nanosec=stamp.nanosec)
                )
            else:
                self._state_pub.publish(
                    snapshot_to_fleet_state_msg(snapshot, stamp=stamp, frame_id=self._frame_id, xy_offset=self._xy_offset)
                )
        except Exception:
            if self._stop.is_set() or not rclpy.ok():
                return  # shutting down: the publishers are already gone
            raise

    def _report(self) -> None:
        loop = self._loop
        if loop.snapshots_published:
            per_publish_ms = 1000.0 * loop.publish_seconds / loop.snapshots_published
        else:
            per_publish_ms = 0.0
        self.get_logger().info(
            f"state plane: read={loop.snapshots_read} published={loop.snapshots_published} "
            f"({per_publish_ms:.2f} ms per publish), reader tail={self._reader.tail} head={self._reader.head}"
        )

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._reader.close()
        return super().destroy_node()


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = StatePublisherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
