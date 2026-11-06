"""Serialize a FleetState straight from state-plane arrays, without building the ROS message.

Building a 1000-robot ``FleetState`` in rclpy costs about 7 ms and serializing
it another 4 ms, while handing the publisher an already serialized message
costs 10 µs (``Publisher.publish`` accepts ``bytes``). The layout of a
``FleetState`` on the wire (CDR, little-endian, as every ROS 2 RMW encodes a
plain message) depends on the robot names only through their lengths, so for
a given set of robots the bytes that change from step to step sit at fixed
offsets: the stamp, the poses, the velocities, the flags and the battery
levels. :class:`FleetStateEncoder` keeps a byte template per generation of
the fleet and writes those fields into a copy of it with numpy.

Wire layout (every offset relative to the start of the body, which follows
the 4-byte encapsulation header; ``align(k)`` pads to a multiple of k)::

    header.stamp.sec      int32
    header.stamp.nanosec  uint32
    header.frame_id       uint32 length (bytes + NUL), bytes, NUL
    robots                uint32 count, then per robot:
        name              align(4), uint32 length, bytes, NUL
        object_id         align(4), int32
        pose, twist       align(8), 13 x float64 (position, orientation xyzw, linear, angular)
        is_moving, has_battery_soc            2 x bool
        battery_soc       align(8), float64
        has_is_charging, is_charging          2 x bool

The encoding is checked byte for byte against ``serialize_message`` of the
message built the ordinary way (see the tests), so an RMW whose CDR differed
would fail that test rather than publish garbage.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

CDR_LE_HEADER = b"\x00\x01\x00\x00"
_FLOATS_PER_ROBOT = 13  # position 3, orientation 4, linear 3, angular 3


def _align(offset: int, k: int) -> int:
    return (offset + k - 1) // k * k


class FleetStateEncoder:
    """Serialize snapshots of one fleet to FleetState CDR bytes.

    Args:
        frame_id: ``header.frame_id`` of every message.
        xy_offset: Added to every position's x and y (the RMF frame offset).
    """

    def __init__(self, frame_id: str = "odom", xy_offset: tuple[float, float] = (0.0, 0.0)) -> None:
        self._frame_id = frame_id.encode("utf-8")
        self._xy_offset = np.array([xy_offset[0], xy_offset[1], 0.0])
        self._generation: Optional[int] = None
        self._names: Optional[tuple[str, ...]] = None
        self._object_id: Optional[np.ndarray] = None
        self._template = b""
        self._float_index = np.zeros(0, dtype=np.int64)  # float64 slots of pose + twist, row major
        self._battery_index = np.zeros(0, dtype=np.int64)  # float64 slot of battery_soc per robot
        self._flag_index = np.zeros((0, 4), dtype=np.int64)  # byte offsets of the four bools per robot

    # -- template --------------------------------------------------------------

    def _rebuild(self, names: Sequence[str], object_id: np.ndarray) -> None:
        n = len(names)
        body = bytearray()
        body += b"\x00" * 8  # stamp, written per message
        frame = self._frame_id + b"\x00"
        body += np.uint32(len(frame)).tobytes() + frame
        body += b"\x00" * (_align(len(body), 4) - len(body))
        body += np.uint32(n).tobytes()
        float_index = np.empty((n, _FLOATS_PER_ROBOT), dtype=np.int64)
        battery_index = np.empty(n, dtype=np.int64)
        flag_index = np.empty((n, 4), dtype=np.int64)
        ids = object_id.astype(np.int32)
        for i, name in enumerate(names):
            encoded = name.encode("utf-8") + b"\x00"
            body += b"\x00" * (_align(len(body), 4) - len(body))
            body += np.uint32(len(encoded)).tobytes() + encoded
            body += b"\x00" * (_align(len(body), 4) - len(body))
            body += ids[i].tobytes()
            body += b"\x00" * (_align(len(body), 8) - len(body))
            start = len(body) // 8
            float_index[i] = np.arange(start, start + _FLOATS_PER_ROBOT)
            body += b"\x00" * (8 * _FLOATS_PER_ROBOT)
            flag_index[i, 0] = len(body)
            flag_index[i, 1] = len(body) + 1
            body += b"\x00\x00"
            body += b"\x00" * (_align(len(body), 8) - len(body))
            battery_index[i] = len(body) // 8
            body += b"\x00" * 8
            flag_index[i, 2] = len(body)
            flag_index[i, 3] = len(body) + 1
            body += b"\x00\x00"
        # Keep the body a multiple of 8 so a float64 view covers it; the padding is
        # cut off again when the message is assembled.
        self._body_len = len(body)
        body += b"\x00" * (_align(len(body), 8) - len(body))
        self._template = bytes(body)
        self._float_index = float_index.ravel()
        self._battery_index = battery_index
        self._flag_index = flag_index
        self._names = tuple(names)
        self._object_id = object_id.copy()

    def _template_for(self, generation: int, names: Sequence[str], object_id: np.ndarray) -> None:
        if (
            self._generation == generation
            and self._names is not None
            and len(self._names) == len(names)
            and self._object_id is not None
            and self._object_id.shape == object_id.shape
            and np.array_equal(self._object_id, object_id)
        ):
            return
        self._rebuild(names, object_id)
        self._generation = generation

    # -- encoding --------------------------------------------------------------

    def encode(
        self,
        *,
        stamp_sec: int,
        stamp_nanosec: int,
        generation: int,
        names: Sequence[str],
        object_id: np.ndarray,
        pos: np.ndarray,
        orn: np.ndarray,
        lin_vel: np.ndarray,
        ang_vel: np.ndarray,
        is_moving: np.ndarray,
        battery_soc: np.ndarray,
        is_charging: np.ndarray,
    ) -> bytes:
        """The serialized FleetState for these rows (one per robot, in order)."""
        n = len(names)
        self._template_for(generation, names, object_id)
        body = bytearray(self._template)
        body[0:4] = np.int32(stamp_sec).tobytes()
        body[4:8] = np.uint32(stamp_nanosec).tobytes()
        if n:
            floats = np.frombuffer(body, dtype="<f8")
            values = np.empty((n, _FLOATS_PER_ROBOT))
            values[:, 0:3] = pos + self._xy_offset
            values[:, 3:7] = orn
            values[:, 7:10] = lin_vel
            values[:, 10:13] = ang_vel
            floats[self._float_index] = values.ravel()
            battery = np.asarray(battery_soc, dtype=np.float64)
            has_battery = ~np.isnan(battery)
            floats[self._battery_index] = np.where(has_battery, battery, 0.0)
            bytes_view = np.frombuffer(body, dtype=np.uint8)
            bytes_view[self._flag_index[:, 0]] = np.asarray(is_moving, dtype=np.uint8)
            bytes_view[self._flag_index[:, 1]] = has_battery.astype(np.uint8)
            bytes_view[self._flag_index[:, 2]] = has_battery.astype(np.uint8)
            bytes_view[self._flag_index[:, 3]] = np.asarray(is_charging, dtype=np.uint8)
        return CDR_LE_HEADER + bytes(body[: self._body_len])

    def encode_snapshot(self, snapshot, *, stamp_sec: int, stamp_nanosec: int) -> bytes:
        """``encode`` for a :class:`~pybullet_fleet.state_plane.Snapshot`."""
        return self.encode(
            stamp_sec=stamp_sec,
            stamp_nanosec=stamp_nanosec,
            generation=snapshot.generation,
            names=snapshot.names,
            object_id=snapshot.object_id,
            pos=snapshot.pos,
            orn=snapshot.orn,
            lin_vel=snapshot.lin_vel,
            ang_vel=snapshot.ang_vel,
            is_moving=snapshot.is_moving,
            battery_soc=snapshot.battery_soc,
            is_charging=snapshot.is_charging,
        )
