"""Shared-memory state plane: layout, seqlock integrity, delivery modes, readers in other processes."""

import multiprocessing
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
import pytest

from pybullet_fleet.state_plane import (
    FLAG_HAS_BATTERY,
    FleetFrame,
    SharedStateReader,
    SharedStateWriter,
    StateLayout,
    StatePlanePublisher,
    collect_fleet_frame,
)


def _frame(n: int, t: float = 0.0, step: int = 0, first_id: int = 0) -> FleetFrame:
    return FleetFrame(
        sim_time=t,
        step=step,
        names=tuple(f"robot{i}" for i in range(first_id, first_id + n)),
        object_id=np.arange(first_id, first_id + n),
        pos=np.full((n, 3), t),
        orn=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        lin_vel=np.full((n, 3), 0.5),
        ang_vel=np.zeros((n, 3)),
        is_moving=np.array([i % 2 == 0 for i in range(n)]),
        battery_soc=np.array([np.nan if i % 3 else 0.5 for i in range(n)], dtype=np.float32),
        is_charging=np.array([i % 3 == 0 for i in range(n)]),
    )


@pytest.fixture
def plane_name():
    return "pbf_test_" + uuid.uuid4().hex[:10]


@pytest.fixture
def writer(plane_name):
    w = SharedStateWriter(plane_name, capacity=16, slots=4, wait_timeout=0.2)
    yield w
    w.close()


# ---------------------------------------------------------------------------
# Layout and round trip
# ---------------------------------------------------------------------------


def test_layout_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        StateLayout(slots=1, capacity=1, name_width=8, reader_slots=1)
    with pytest.raises(ValueError):
        StateLayout(slots=2, capacity=0, name_width=8, reader_slots=1)
    with pytest.raises(ValueError):
        StateLayout(slots=2, capacity=1, name_width=8, reader_slots=10_000)


def test_writer_rejects_unknown_mode(plane_name):
    with pytest.raises(ValueError, match="mode"):
        SharedStateWriter(plane_name, capacity=4, mode="sometimes")


def test_frame_round_trips_every_column(writer, plane_name):
    frame = _frame(5, t=1.5, step=15)
    with SharedStateReader(plane_name) as reader:
        seq = writer.write(frame)
        snapshot = reader.read_next(timeout=1.0)
    assert snapshot is not None
    assert snapshot.seq == seq == 0
    assert snapshot.sim_time == 1.5
    assert snapshot.step == 15
    assert snapshot.generation == 1
    assert snapshot.names == frame.names
    np.testing.assert_array_equal(snapshot.object_id, frame.object_id)
    np.testing.assert_array_equal(snapshot.pos, frame.pos)
    np.testing.assert_array_equal(snapshot.orn, frame.orn)
    np.testing.assert_array_equal(snapshot.lin_vel, frame.lin_vel)
    np.testing.assert_array_equal(snapshot.is_moving, frame.is_moving)
    np.testing.assert_array_equal(snapshot.is_charging, frame.is_charging)
    np.testing.assert_array_equal(np.isnan(snapshot.battery_soc), np.isnan(frame.battery_soc))
    assert snapshot.battery_soc[0] == pytest.approx(0.5)


def test_frame_larger_than_capacity_is_refused(writer):
    with pytest.raises(ValueError, match="rows"):
        writer.write(_frame(17))


def test_reader_rejects_a_foreign_segment(plane_name):
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=plane_name, create=True, size=8192)
    try:
        with pytest.raises(ValueError, match="not a PyBulletFleet state plane"):
            SharedStateReader(plane_name)
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# Generation and names
# ---------------------------------------------------------------------------


def test_generation_bumps_only_when_the_row_set_changes(writer, plane_name):
    with SharedStateReader(plane_name) as reader:
        writer.write(_frame(3))
        writer.write(_frame(3, t=0.1))
        assert reader.read_next(timeout=1.0).generation == 1
        assert reader.read_next(timeout=1.0).generation == 1
        writer.write(_frame(4, t=0.2))
        snapshot = reader.read_next(timeout=1.0)
        assert snapshot.generation == 2
        assert snapshot.names == ("robot0", "robot1", "robot2", "robot3")
        writer.write(_frame(2, t=0.3, first_id=10))
        snapshot = reader.read_next(timeout=1.0)
        assert snapshot.generation == 3
        assert snapshot.names == ("robot10", "robot11")
        np.testing.assert_array_equal(snapshot.object_id, [10, 11])


# ---------------------------------------------------------------------------
# Delivery modes and backpressure
# ---------------------------------------------------------------------------


def test_without_a_reader_the_writer_never_waits(writer):
    started = time.monotonic()
    for i in range(20):
        writer.write(_frame(2, t=i * 0.1, step=i))
    assert time.monotonic() - started < 0.1
    assert writer.seq == 19
    assert writer.overruns == 0


def test_every_step_delivers_in_order_without_gaps(writer, plane_name):
    reader = SharedStateReader(plane_name)
    received = []

    def consume():
        while len(received) < 30:
            snapshot = reader.read_next(timeout=2.0)
            if snapshot is None:
                break
            received.append(snapshot.seq)

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        for i in range(30):
            writer.write(_frame(2, t=i * 0.1, step=i))
    finally:
        thread.join()
        reader.close()
    assert received == list(range(30))
    assert writer.overruns == 0


def test_every_step_waits_for_a_slow_reader_then_overwrites_after_the_timeout(writer, plane_name):
    reader = SharedStateReader(plane_name)
    try:
        for i in range(4):
            writer.write(_frame(1, t=i))  # fills the 4 slots; the reader has taken none
        started = time.monotonic()
        writer.write(_frame(1, t=4.0))
        waited = time.monotonic() - started
        assert waited >= 0.2  # wait_timeout
        assert writer.overruns == 1
        # In the overrun state further writes do not wait again.
        started = time.monotonic()
        writer.write(_frame(1, t=5.0))
        assert time.monotonic() - started < 0.05
        # The reader now sees the oldest surviving snapshot, not the ones overwritten.
        snapshot = reader.read_next(timeout=1.0)
        assert snapshot.seq == writer.seq - writer.layout.slots + 1
        # Once it catches up, every_step delivery resumes (a write waits again if needed).
        while reader.tail < writer.seq:
            reader.read_next(timeout=1.0)
        writer.write(_frame(1, t=6.0))
        assert reader.read_next(timeout=1.0).seq == writer.seq
    finally:
        reader.close()


def test_latest_mode_never_waits_and_the_reader_skips_to_the_newest(plane_name):
    writer = SharedStateWriter(plane_name, capacity=4, slots=4, mode="latest", wait_timeout=0.2)
    reader = SharedStateReader(plane_name)
    try:
        started = time.monotonic()
        for i in range(50):
            writer.write(_frame(1, t=i * 0.1, step=i))
        assert time.monotonic() - started < 0.2
        snapshot = reader.read_latest(timeout=1.0)
        assert snapshot.seq == 49
        assert reader.read_latest(timeout=0.05) is None  # nothing newer yet
        writer.write(_frame(1, t=5.0, step=50))
        assert reader.read_latest(timeout=1.0).seq == 50
    finally:
        reader.close()
        writer.close()


def test_read_next_times_out_when_nothing_is_written(writer, plane_name):
    with SharedStateReader(plane_name) as reader:
        started = time.monotonic()
        assert reader.read_next(timeout=0.1) is None
        assert 0.1 <= time.monotonic() - started < 1.0


def test_closed_reader_is_forgotten_by_the_writer(writer, plane_name):
    reader = SharedStateReader(plane_name)
    assert writer.active_readers() == [0]
    reader.close()
    assert writer.active_readers() == []
    for i in range(10):
        writer.write(_frame(1, t=i))  # no waiting on a gone reader
    assert writer.overruns == 0


def test_stale_reader_is_dropped_and_re_registers_on_its_next_read(writer, plane_name):
    reader = SharedStateReader(plane_name, heartbeat=False)
    try:
        reader._readers[reader.index]["heartbeat"] = time.time() - 10.0
        assert writer.active_readers() == []
        reader._heartbeat = True
        writer.write(_frame(1))
        assert reader.read_next(timeout=1.0) is not None
        assert writer.active_readers() == [reader.index]
    finally:
        reader.close()


def test_reader_slots_are_limited(plane_name):
    writer = SharedStateWriter(plane_name, capacity=2, reader_slots=2)
    readers = [SharedStateReader(plane_name), SharedStateReader(plane_name)]
    try:
        with pytest.raises(RuntimeError, match="no free reader slot"):
            SharedStateReader(plane_name)
    finally:
        for reader in readers:
            reader.close()
        writer.close()


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_a_slot_being_written_is_not_returned(writer, plane_name):
    with SharedStateReader(plane_name) as reader:
        writer.write(_frame(2, t=1.0))
        # Mark slot 0 as mid-write, as the writer does before storing the rows.
        writer._slot_meta[0]["seq_end"] = -1
        assert reader.read_next(timeout=0.1) is None
        writer._slot_meta[0]["seq_end"] = 0
        assert reader.read_next(timeout=1.0).seq == 0


def test_a_reader_starts_with_the_newest_snapshot_present_when_it_attached(writer, plane_name):
    writer.write(_frame(1, t=0.0))
    writer.write(_frame(1, t=0.1))
    with SharedStateReader(plane_name) as reader:
        assert reader.read_next(timeout=1.0).seq == 1
        assert reader.read_next(timeout=0.05) is None
        writer.write(_frame(1, t=0.2))
        assert reader.read_next(timeout=1.0).seq == 2


def test_a_slot_overwritten_during_the_copy_is_read_again(writer, plane_name):
    writer.write(_frame(2, t=1.0))
    with SharedStateReader(plane_name) as reader:
        original = reader._read_slot

        def torn_then_fine(seq):
            snapshot = original(seq)
            if not calls:
                calls.append(seq)
                writer._slot_meta[0]["seq_begin"] = 99  # the writer started a new frame here
                writer._slot_meta[0]["seq_begin"] = 0
                return None
            return snapshot

        calls = []
        reader._read_slot = torn_then_fine
        assert reader.read_next(timeout=1.0).seq == 0
        assert calls == [0]


# ---------------------------------------------------------------------------
# Another process
# ---------------------------------------------------------------------------


def _reader_process(name, count, queue):
    reader = SharedStateReader(name)
    seqs = []
    try:
        while len(seqs) < count:
            snapshot = reader.read_next(timeout=5.0)
            if snapshot is None:
                break
            seqs.append(snapshot.seq)
    finally:
        reader.close()
    queue.put(seqs)


def test_reader_in_another_process_gets_every_snapshot(plane_name):
    writer = SharedStateWriter(plane_name, capacity=8, slots=4, wait_timeout=2.0)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_reader_process, args=(plane_name, 40, queue))
    process.start()
    try:
        deadline = time.monotonic() + 10.0
        while not writer.active_readers() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert writer.active_readers() == [0]
        for i in range(40):
            writer.write(_frame(3, t=i * 0.1, step=i))
        assert queue.get(timeout=10.0) == list(range(40))
        process.join(timeout=10.0)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        writer.close()


def test_the_segment_survives_a_reader_process_exit(plane_name):
    writer = SharedStateWriter(plane_name, capacity=8, slots=4)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_reader_process, args=(plane_name, 1, queue))
    process.start()
    try:
        deadline = time.monotonic() + 10.0
        while not writer.active_readers() and time.monotonic() < deadline:
            time.sleep(0.01)
        writer.write(_frame(1))
        assert queue.get(timeout=10.0) == [0]
        process.join(timeout=10.0)
        # Python 3.10-3.12 would have unlinked the segment here through the reader's resource tracker.
        writer.write(_frame(1, t=1.0))
        with SharedStateReader(plane_name) as reader:
            assert reader.read_latest(timeout=1.0).seq == 1
    finally:
        if process.is_alive():
            process.terminate()
        writer.close()


# ---------------------------------------------------------------------------
# Frames from a simulation
# ---------------------------------------------------------------------------


@dataclass
class _Pose:
    position: tuple
    orientation: tuple


class _Agent:
    def __init__(self, name, object_id, soc=None):
        self.name = name
        self.object_id = object_id
        self.velocity = np.array([1.0, 0.0, 0.0])
        self.angular_velocity = 0.25
        self.is_moving = True
        self.battery_soc = soc
        self.is_charging = soc is not None

    def get_pose(self):
        return _Pose((float(self.object_id), 2.0, 0.1), (0.0, 0.0, 0.0, 1.0))


class _Sim:
    def __init__(self, agents):
        self.agents = agents
        self.sim_time = 3.5
        self.step_count = 35


def test_collect_fleet_frame_reads_agents():
    frame = collect_fleet_frame(_Sim([_Agent("a", 1, soc=0.8), _Agent("", 2)]))
    assert frame.sim_time == 3.5 and frame.step == 35
    assert frame.names == ("a", "agent_2")  # an unnamed agent is named as FleetStateProvider names it
    np.testing.assert_array_equal(frame.object_id, [1, 2])
    np.testing.assert_array_equal(frame.pos[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(frame.lin_vel[:, 0], [1.0, 1.0])
    np.testing.assert_array_equal(frame.ang_vel[:, 2], [0.25, 0.25])
    assert frame.battery_soc[0] == pytest.approx(0.8) and np.isnan(frame.battery_soc[1])
    np.testing.assert_array_equal(frame.is_charging, [True, False])


def test_publisher_writes_on_post_step_and_honours_publish_rate(plane_name):
    from pybullet_fleet.events import EventBus, SimEvents

    class _EventSim(_Sim):
        def __init__(self, agents):
            super().__init__(agents)
            self.events = EventBus()

    sim = _EventSim([_Agent("a", 1)])
    publisher = StatePlanePublisher(sim, name=plane_name, capacity=4, mode="latest", publish_rate=5.0).install()
    try:
        with SharedStateReader(plane_name) as reader:
            for i in range(10):
                sim.sim_time = i * 0.1
                sim.events.emit(SimEvents.POST_STEP, dt=0.1, sim_time=sim.sim_time)
            # 5 Hz over 1 s of sim time: steps at 0.0, 0.2, 0.4, 0.6, 0.8.
            assert publisher.frames_written == 5
            snapshot = reader.read_latest(timeout=1.0)
            assert snapshot.sim_time == pytest.approx(0.8)
            assert snapshot.names == ("a",)
    finally:
        publisher.close()
    sim.events.emit(SimEvents.POST_STEP, dt=0.1, sim_time=2.0)
    assert publisher.frames_written == 5  # uninstalled


def test_has_battery_flag_marks_rows_with_a_battery(writer, plane_name):
    writer.write(_frame(3))
    flags = writer._slot_rows[0]["flags"][:3]
    assert list(flags & FLAG_HAS_BATTERY) == [FLAG_HAS_BATTERY, 0, 0]


def test_writer_replaces_a_segment_left_behind_by_a_dead_writer(plane_name):
    from multiprocessing import shared_memory

    stale = shared_memory.SharedMemory(name=plane_name, create=True, size=4096)
    stale.close()  # not unlinked: as after a SIGKILL
    writer = SharedStateWriter(plane_name, capacity=2)
    try:
        assert writer.layout.total_bytes > 4096
        writer.write(_frame(1))
        with SharedStateReader(plane_name) as reader:
            assert reader.read_latest(timeout=1.0).seq == 0
    finally:
        writer.close()


def test_reader_notices_when_the_writer_replaced_the_segment(plane_name):
    first = SharedStateWriter(plane_name, capacity=2)
    reader = SharedStateReader(plane_name)
    try:
        assert not reader.is_stale()
        first.close()
        assert reader.is_stale()  # gone
        second = SharedStateWriter(plane_name, capacity=2)
        try:
            assert reader.is_stale()  # a different segment under the same name
            with SharedStateReader(plane_name) as fresh:
                assert not fresh.is_stale()
        finally:
            second.close()
    finally:
        reader.close()
