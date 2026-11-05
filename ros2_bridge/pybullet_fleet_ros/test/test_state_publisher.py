"""State publisher: the read-and-publish loop, the snapshot conversion and the state_plane config section."""

import numpy as np
import pytest

from pybullet_fleet.state_plane import Snapshot
from pybullet_fleet_ros.interface_config import StatePlaneConfig, resolve_state_plane_config
from pybullet_fleet_ros.state_publisher_node import StatePublisherLoop


def _snapshot(seq: int, sim_time: float, n: int = 2) -> Snapshot:
    return Snapshot(
        seq=seq,
        generation=1,
        sim_time=sim_time,
        step=seq,
        names=tuple(f"robot{i}" for i in range(n)),
        object_id=np.arange(n),
        pos=np.array([[float(i), 2.0, 0.1] for i in range(n)]),
        orn=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        lin_vel=np.full((n, 3), 0.5),
        ang_vel=np.array([[0.0, 0.0, 0.25]] * n),
        is_moving=np.array([True] + [False] * (n - 1)),
        battery_soc=np.array([0.75] + [np.nan] * (n - 1), dtype=np.float32),
        is_charging=np.array([True] + [False] * (n - 1)),
    )


class _FakeReader:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self.next_calls = 0
        self.latest_calls = 0

    def read_next(self, timeout=None):
        self.next_calls += 1
        return self._snapshots.pop(0) if self._snapshots else None

    def read_latest(self, timeout=None):
        self.latest_calls += 1
        if not self._snapshots:
            return None
        newest = self._snapshots[-1]
        self._snapshots.clear()
        return newest


def test_loop_publishes_every_snapshot_in_order():
    reader = _FakeReader([_snapshot(0, 0.0), _snapshot(1, 0.1), _snapshot(2, 0.2)])
    published = []
    loop = StatePublisherLoop(reader, published.append, mode="every_step")
    assert loop.once() and loop.once() and loop.once()
    assert not loop.once()  # nothing left: a read timeout
    assert [s.seq for s in published] == [0, 1, 2]
    assert loop.snapshots_read == 3 and loop.snapshots_published == 3


def test_loop_in_latest_mode_takes_the_newest():
    reader = _FakeReader([_snapshot(0, 0.0), _snapshot(1, 0.1), _snapshot(2, 0.2)])
    published = []
    loop = StatePublisherLoop(reader, published.append, mode="latest")
    assert loop.once()
    assert [s.seq for s in published] == [2]
    assert reader.latest_calls == 1 and reader.next_calls == 0


def test_loop_throttles_by_simulated_time():
    reader = _FakeReader([_snapshot(i, i * 0.1) for i in range(10)])
    published = []
    loop = StatePublisherLoop(reader, published.append, publish_rate=5.0)
    while loop.once():
        pass
    assert [round(s.sim_time, 1) for s in published] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert loop.snapshots_read == 10 and loop.snapshots_published == 5


def test_loop_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        StatePublisherLoop(_FakeReader([]), lambda s: None, mode="sometimes")


def test_snapshot_converts_to_fleet_state_message():
    pytest.importorskip("pybullet_fleet_msgs.msg", reason="pybullet_fleet_msgs not generated")
    from pybullet_fleet_ros.conversions import sim_time_to_ros_time
    from pybullet_fleet_ros.fleet_ros_interface import snapshot_to_fleet_state_msg

    snapshot = _snapshot(7, 1.5, n=3)
    msg = snapshot_to_fleet_state_msg(snapshot, stamp=sim_time_to_ros_time(1.5), xy_offset=(10.0, -1.0))

    assert msg.header.frame_id == "odom"
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 500_000_000
    assert [r.name for r in msg.robots] == ["robot0", "robot1", "robot2"]
    assert [r.object_id for r in msg.robots] == [0, 1, 2]
    assert msg.robots[1].pose.position.x == pytest.approx(11.0)
    assert msg.robots[1].pose.position.y == pytest.approx(1.0)
    assert msg.robots[1].pose.orientation.w == pytest.approx(1.0)
    assert msg.robots[0].twist.linear.x == pytest.approx(0.5)
    assert msg.robots[0].twist.angular.z == pytest.approx(0.25)
    assert msg.robots[0].is_moving and not msg.robots[1].is_moving
    assert msg.robots[0].has_battery_soc and msg.robots[0].battery_soc == pytest.approx(0.75)
    assert msg.robots[0].has_is_charging and msg.robots[0].is_charging
    assert not msg.robots[1].has_battery_soc and msg.robots[1].battery_soc == 0.0
    assert not msg.robots[1].has_is_charging and not msg.robots[1].is_charging


def test_snapshot_conversion_matches_the_state_based_conversion():
    pytest.importorskip("pybullet_fleet_msgs.msg", reason="pybullet_fleet_msgs not generated")
    from rclpy.serialization import serialize_message

    from pybullet_fleet.states import RobotState3D
    from pybullet_fleet_ros.conversions import sim_time_to_ros_time
    from pybullet_fleet_ros.fleet_ros_interface import fleet_state_to_msg, snapshot_to_fleet_state_msg

    snapshot = _snapshot(3, 0.3, n=4)
    states = [
        RobotState3D(
            name=snapshot.names[i],
            object_id=int(snapshot.object_id[i]),
            position=tuple(snapshot.pos[i]),
            orientation=tuple(snapshot.orn[i]),
            linear_velocity=tuple(snapshot.lin_vel[i]),
            angular_velocity=tuple(snapshot.ang_vel[i]),
            is_moving=bool(snapshot.is_moving[i]),
            battery_soc=None if np.isnan(snapshot.battery_soc[i]) else float(snapshot.battery_soc[i]),
            is_charging=None if np.isnan(snapshot.battery_soc[i]) else bool(snapshot.is_charging[i]),
        )
        for i in range(4)
    ]
    stamp = sim_time_to_ros_time(0.3)
    assert serialize_message(snapshot_to_fleet_state_msg(snapshot, stamp=stamp)) == serialize_message(
        fleet_state_to_msg(states, stamp=stamp)
    )


def test_state_plane_config_defaults_and_validation():
    assert resolve_state_plane_config({}) == StatePlaneConfig()
    config = resolve_state_plane_config(
        {"state_plane": {"enabled": True, "name": "plane_a", "mode": "latest", "slots": 4, "wait_timeout": 0.5}}
    )
    assert config == StatePlaneConfig(enabled=True, name="plane_a", mode="latest", slots=4, wait_timeout=0.5)
    with pytest.raises(ValueError, match="mode"):
        resolve_state_plane_config({"state_plane": {"mode": "sometimes"}})
    with pytest.raises(ValueError, match="slots"):
        resolve_state_plane_config({"state_plane": {"slots": 1}})
    with pytest.raises(ValueError, match="wait_timeout"):
        resolve_state_plane_config({"state_plane": {"wait_timeout": 0}})
    with pytest.raises(ValueError, match="capacity"):
        resolve_state_plane_config({"state_plane": {"capacity": 0}})


def _random_snapshot(n: int, names=None, seed: int = 1) -> Snapshot:
    rng = np.random.default_rng(seed)
    return Snapshot(
        seq=1,
        generation=3,
        sim_time=12.345678901,
        step=5,
        names=tuple(names if names is not None else [f"robot_{i}" for i in range(n)]),
        object_id=np.arange(n) * 7,
        pos=rng.random((n, 3)) * 100,
        orn=rng.random((n, 4)),
        lin_vel=rng.random((n, 3)),
        ang_vel=rng.random((n, 3)),
        is_moving=rng.random(n) > 0.5,
        battery_soc=np.where(rng.random(n) > 0.5, rng.random(n), np.nan).astype(np.float32),
        is_charging=rng.random(n) > 0.5,
    )


@pytest.mark.parametrize(
    "n, names",
    [
        (0, []),
        (1, ["a"]),
        (3, ["", "x", "robot_long_name_12345"]),
        (7, ["日本語", "r", "rr", "rrr", "rrrr", "rrrrr", "rrrrrr"]),
        (1000, None),
    ],
)
def test_cdr_encoder_matches_rclpy_serialization_byte_for_byte(n, names):
    pytest.importorskip("pybullet_fleet_msgs.msg", reason="pybullet_fleet_msgs not generated")
    from rclpy.serialization import serialize_message

    from pybullet_fleet_ros.conversions import sim_time_to_ros_time
    from pybullet_fleet_ros.fleet_ros_interface import snapshot_to_fleet_state_msg
    from pybullet_fleet_ros.fleet_state_cdr import FleetStateEncoder

    snapshot = _random_snapshot(n, names)
    stamp = sim_time_to_ros_time(snapshot.sim_time)
    expected = serialize_message(snapshot_to_fleet_state_msg(snapshot, stamp=stamp, xy_offset=(1.5, -2.0)))
    encoder = FleetStateEncoder(xy_offset=(1.5, -2.0))
    assert encoder.encode_snapshot(snapshot, stamp_sec=stamp.sec, stamp_nanosec=stamp.nanosec) == expected
    # The template is reused for the same generation; the second encoding must be as exact.
    second = _random_snapshot(n, names, seed=2)
    expected2 = serialize_message(snapshot_to_fleet_state_msg(second, stamp=stamp, xy_offset=(1.5, -2.0)))
    assert encoder.encode_snapshot(second, stamp_sec=stamp.sec, stamp_nanosec=stamp.nanosec) == expected2


def test_cdr_encoder_rebuilds_its_template_when_the_fleet_changes():
    pytest.importorskip("pybullet_fleet_msgs.msg", reason="pybullet_fleet_msgs not generated")
    from rclpy.serialization import deserialize_message

    from pybullet_fleet_msgs.msg import FleetState
    from pybullet_fleet_ros.fleet_state_cdr import FleetStateEncoder

    encoder = FleetStateEncoder()
    first = _random_snapshot(2, ["a", "b"])
    encoder.encode_snapshot(first, stamp_sec=0, stamp_nanosec=0)
    second = _random_snapshot(3, ["a", "b", "c"])
    raw = encoder.encode_snapshot(second, stamp_sec=1, stamp_nanosec=5)
    msg = deserialize_message(raw, FleetState)
    assert [r.name for r in msg.robots] == ["a", "b", "c"]
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 5
    assert msg.robots[2].pose.position.x == pytest.approx(second.pos[2, 0])
